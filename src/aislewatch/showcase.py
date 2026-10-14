"""Showcase site: teacher labelling -> base Qwen3.5-4B -> AisleWatch, then model comparison charts.

    uv run aw-showcase build    # assemble data/showcase/ (frames, GIFs, teacher captions, data.json)
    uv run aw-showcase serve    # http://127.0.0.1:8790

Everything shown comes from results already on disk (data/results/), except the teacher captions for the
training windows in section 1, which are fetched once from Gemini 3.8 Flash (~$0.01 per window) and cached.
Costs are list prices: measured tokens x price for API models (Opus: estimated, it ran through the Claude
subscription), GPU-hour price x measured throughput for self-hosted models.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http.server
import io
import json
import shutil
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from PIL import Image

from aislewatch import evaluate as ev
from aislewatch import preprocess as pp

OUT = pp.REPO_ROOT / "data" / "showcase"
RESULTS = pp.REPO_ROOT / "data" / "results"
DATASET = "v2"  # its test split is v1's, byte for byte
WINDOWS_PER_CAMERA_MONTH = 30 * 24 * 3600 / pp.WINDOW_S  # continuous: one 5 s window after another
TEACHER = "gemini-3.8-flash"

# Section 2/3 sample, chosen from the per-window results (see build log): five where the base model is wrong and
# AisleWatch right (one per error type), two where both are right, one AisleWatch miss.
SAMPLE = ["B-seq-prop-box-leave-t2-w1", "B-seq-drift-ajar-t3-w1", "B-prop-cart-s0", "B-seq-prop-box-leave-t0-w0",
          "B-seq-close-up-t1-w1", "B-seq-pass-in-t1-w0", "cam-B_closed-s0", "B-ajar-phone-s0"]

# Section 1: training windows from six different rooms/cameras, one per (camera, state).
TEACHER_PICKS = [("cam-A", "propped", "sequence"), ("cam-D", "passing", "sequence"), ("cam-F", "ajar", "still"),
                 ("cam-G", "propped", "still"), ("cam-H", "closed", "sequence"), ("cam-I", "passing", "sequence")]

# Rows for the comparison charts: (results file, display name, group). group "ours" is highlighted.
ROWS = [("v2/test/qwen4b-v2", "AisleWatch (Qwen3.5-4B + LoRA)", "ours"),
        ("v1/test/gemini-3.8-flash", "Gemini 3.8 Flash", "api"),
        ("v1/test/gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "api"),
        ("v1/test/gpt-6.1-sol", "GPT-6.1 Sol", "api"),
        ("v2/test/claude-opus-5.5", "Claude Opus 5.5", "api"),
        ("v2/test/qwen4b-base", "Qwen3.5-4B (base)", "open"),
        ("v1/test/qwen9b-base", "Qwen3.5-9B (base)", "open")]

# Opus ran through the Claude Code subscription, so its wall-clock time includes CLI overhead. Its latency is
# estimated from OpenRouter's served stats for the Anthropic endpoint (checked 2026-10-14):
# time to first token + output tokens / throughput.
OPUS_OR = {"ttft_p50_s": 3.35, "tps_p50": 75, "ttft_p90_s": 6.03, "tps_p90": 38, "checked": "2026-10-14"}


def _frames_from_test(window_id: str) -> list[bytes]:
    d = pp.REPO_ROOT / "data" / "datasets" / DATASET / "test" / window_id
    return [(d / f"f{i:02d}.jpg").read_bytes() for i in range(pp.FRAMES_PER_WINDOW)]


def _frames_from_messages(messages: list[dict]) -> list[bytes]:
    urls = [p["image_url"]["url"] for p in messages[1]["content"] if p.get("image_url")]
    return [base64.b64decode(u.split(",", 1)[1]) for u in urls]


def _write_window(window_id: str, frames: list[bytes]) -> None:
    d = OUT / "frames" / window_id
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        (d / f"f{i:02d}.jpg").write_bytes(f)
    ims = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    ims[0].save(OUT / "gifs" / f"{window_id}.gif", save_all=True, append_images=ims[1:],
                duration=int(pp.FRAME_SPACING_S * 1000), loop=0)


def _results(path: str) -> dict[str, dict]:
    return {json.loads(l)["window_id"]: json.loads(l) for l in (RESULTS / f"{path}-structured.jsonl").open()}


def _summary(path: str) -> dict:
    return json.loads((RESULTS / f"{path}-structured.summary.json").read_text())


def _cost_per_window(name: str, s: dict) -> tuple[float, bool]:
    """(usd per window, estimated?)"""
    if "est_usd_per_window" in s:
        return s["est_usd_per_window"], True
    price = ev.models_config()[name]["price"]
    if "usd_per_m_input" in price:
        return (s["prompt_tokens_mean"] * price["usd_per_m_input"] + s["completion_tokens_mean"] * price["usd_per_m_output"]) / 1e6, False
    return price["usd_per_gpu_hour"] * s["wall_s"] / 3600 / s["n"], False


def _latency(name: str, s: dict, rows: dict) -> tuple[float, float, bool]:
    """(p50 s, p95 s, estimated?). Opus: from OpenRouter's served stats, not our CLI wall clock."""
    if not s.get("latency_comparable", True):
        out_tok = sum(len(r["text"] or "") / 4 for r in rows.values()) / len(rows)
        return (round(OPUS_OR["ttft_p50_s"] + out_tok / OPUS_OR["tps_p50"], 2),
                round(OPUS_OR["ttft_p90_s"] + out_tok / OPUS_OR["tps_p90"], 2), True)
    return s["latency_p50_s"], s["latency_p95_s"], False


async def _teacher_captions(windows: list[tuple[str, str, list[bytes]]]) -> dict[str, dict]:
    cache = OUT / "teacher.json"
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [w for w in windows if w[0] not in have]
    if todo:
        cfg = ev.models_config()[TEACHER]
        async with httpx.AsyncClient(timeout=180) as client:
            for wid, cam, frames in todo:
                text, latency, usage, err = await ev.call(client, cfg, ev.build_request(cfg, frames, cam, "structured"))
                have[wid] = {"pred": ev.parse(text), "error": err, "latency_s": round(latency, 2), "usage": usage}
                print(f"teacher {wid}: {(have[wid]['pred'] or {}).get('door_state')} ({err or 'ok'})")
        cache.write_text(json.dumps(have, indent=2))
    return have


def build() -> None:
    for sub in ("frames", "gifs"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    fields = pp.caption_field_order()

    # Section 1: teacher on training windows
    train = pq.read_table(pp.REPO_ROOT / "data" / "datasets" / DATASET / "train.parquet").to_pylist()
    picks = []
    for cam, state, source in TEACHER_PICKS:
        match = [r for r in train if r["camera_id"] == cam and r["door_state"] == state and r["source"] == source]
        if not match:
            raise SystemExit(f"no training window for {cam} {state} {source}; change TEACHER_PICKS")
        r = next((r for r in match if r["window_id"].endswith(("-s0", "-w0", "-w1"))), match[0])
        frames = _frames_from_messages(r["messages"])
        _write_window(r["window_id"], frames)
        label = json.loads(r["messages"][-1]["content"][0]["text"]) if isinstance(r["messages"][-1]["content"], list) \
            else json.loads(r["messages"][-1]["content"])
        picks.append((r["window_id"], cam, frames, label))
    teacher = asyncio.run(_teacher_captions([(w, c, f) for w, c, f, _ in picks]))
    teacher_section = [{"window_id": w, "camera_id": c, "label": lab, "teacher": teacher[w]["pred"]} for w, c, _, lab in picks]

    # Sections 2 and 3: the same camera-B test windows, base vs AisleWatch
    base, ours = _results(f"{DATASET}/test/qwen4b-base"), _results(f"{DATASET}/test/qwen4b-v2")
    sample = []
    for wid in SAMPLE:
        _write_window(wid, _frames_from_test(wid))
        sample.append({"window_id": wid, "camera_id": ours[wid]["camera_id"], "label": ours[wid]["label"],
                       "base": base[wid]["pred"], "ours": ours[wid]["pred"]})

    # Section 4: comparison rows
    models = []
    for path, display, group in ROWS:
        f = RESULTS / f"{path}-structured.summary.json"
        if not f.exists():
            print(f"skip {display}: no results at {f}")
            continue
        s, rows = _summary(path), _results(path)
        name = path.split("/")[-1]
        usd, cost_est = _cost_per_window(name, s)
        p50, p95, lat_est = _latency(name, s, rows)
        models.append({"name": display, "group": group, "macro_f1": s["macro_f1"], "accuracy": s["accuracy"],
                       "alert_recall": s["alert_recall"], "false_alarm_rate": s["false_alarm_rate"],
                       "valid_json_rate": s["valid_json_rate"], "latency_p50_s": p50, "latency_p95_s": p95,
                       "latency_estimated": lat_est, "usd_per_window": usd, "cost_estimated": cost_est,
                       "usd_per_camera_month": usd * WINDOWS_PER_CAMERA_MONTH, "via": s.get("via", "")})

    stats = {k: {m: _summary(f"{DATASET}/test/{k}")[m] for m in ("macro_f1", "accuracy", "alert_recall", "false_alarm_rate", "n")}
             for k in ("qwen4b-base", "qwen4b-v2")}
    data = {"fields": fields, "frame_spacing_s": pp.FRAME_SPACING_S, "frames_per_window": pp.FRAMES_PER_WINDOW,
            "teacher_model": "Gemini 3.8 Flash", "teacher": teacher_section, "sample": sample, "stats": stats,
            "models": models, "windows_per_camera_month": WINDOWS_PER_CAMERA_MONTH, "opus_latency_source": OPUS_OR}
    (OUT / "data.json").write_text(json.dumps(data, indent=2))
    shutil.copy(Path(__file__).with_name("showcase.html"), OUT / "index.html")
    print(f"built {OUT}: {len(teacher_section)} teacher windows, {len(sample)} sample windows, {len(models)} models")


def serve(port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
    print(f"AisleWatch showcase: http://127.0.0.1:{port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    build() if a.cmd == "build" else serve(a.port)


if __name__ == "__main__":
    main()
