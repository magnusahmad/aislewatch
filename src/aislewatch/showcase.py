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

# Section 2/3 sample: sequence windows with the most visible motion (2-3 keyframe changes in 5 s; stills and
# held keyframes look frozen), chosen from the per-window results: four where the base model is wrong and
# AisleWatch right, one where both are right, one AisleWatch miss.
SAMPLE = ["B-seq-cart-through-t0-w1", "B-seq-walk-past-propped-t2-w0", "B-seq-prop-box-leave-t2-w0",
          "B-seq-close-up-t3-w0", "B-seq-two-pass-t0-w1", "B-seq-walk-past-ajar-t3-w0"]

# Section 1: one training sequence per camera, the one with the most visible motion.
TEACHER_CAMERAS = ["cam-A", "cam-D", "cam-F", "cam-G", "cam-H", "cam-I"]

# Rows for the comparison charts: (results file, display name, group). group "ours" is highlighted.
ROWS = [("v2/test/qwen4b-v2", "AisleWatch (Qwen3.5-4B + LoRA)", "ours"),
        ("v1/test/gemini-3.8-flash", "Gemini 3.8 Flash", "api"),
        ("v1/test/gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "api"),
        ("v1/test/gpt-6.1-sol", "GPT-6.1 Sol", "api"),
        ("v2/test/claude-opus-5.5", "Claude Opus 5.5", "api"),
        ("v2/test/qwen4b-base", "Qwen3.5-4B (base)", "open"),
        ("v1/test/qwen9b-base", "Qwen3.5-9B (base)", "open")]

# Section 5, live demo: a camera no model trained on. The clip plays the keyframe sequence (technician props the
# door with a box and leaves), then holds on the empty room; the models get the measured window of that end state
# (base Qwen said closed, AisleWatch and Gemini said propped on exactly these frames).
LIVE_WINDOW = "B-seq-prop-box-leave-t2-w3"
LIVE_SEQUENCE = pp.REPO_ROOT / "data" / "gen" / "seq" / "cam-B" / "B-seq-prop-box-leave"
LIVE_MODELS = [("qwen4b-base", "Qwen3.5-4B (base)", "vllm"), ("gemini-3.8-flash", "Gemini 3.8 Flash", "api"),
               ("qwen4b-v2", "AisleWatch", "vllm")]
LIVE_RUNS = OUT / "live_runs"
BOOTSTRAP = 2000

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


def _motion(frames: list[bytes]) -> int:
    """Keyframe changes visible in a window (mean grey-level step > 4 on a 128x72 thumbnail)."""
    import numpy as np
    g = [np.asarray(Image.open(io.BytesIO(f)).convert("L").resize((128, 72)), dtype=float) for f in frames]
    return sum(float(np.abs(g[i] - g[i - 1]).mean()) > 4 for i in range(1, len(g)))


def _write_window(window_id: str, frames: list[bytes]) -> None:
    d = OUT / "frames" / window_id
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        (d / f"f{i:02d}.jpg").write_bytes(f)
    ims = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    ims[0].save(OUT / "gifs" / f"{window_id}.gif", save_all=True, append_images=ims[1:],
                duration=int(pp.FRAME_SPACING_S * 1000), loop=0)


def _f1_interval(rows: dict[str, dict], groups: dict[str, str]) -> list[float]:
    """95% interval on macro F1, resampling whole scenes (windows from one scene succeed or fail together)."""
    import random
    by_group: dict[str, list[dict]] = {}
    for wid, r in rows.items():
        by_group.setdefault(groups.get(wid, wid), []).append(r)
    keys, rng, vals = sorted(by_group), random.Random(0), []
    for _ in range(BOOTSTRAP):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in by_group[k]]
        vals.append(ev.metrics(sample)["macro_f1"])
    vals.sort()
    return [round(vals[int(0.025 * BOOTSTRAP)], 3), round(vals[int(0.975 * BOOTSTRAP) - 1], 3)]


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
    for cam in TEACHER_CAMERAS:
        seqs = [r for r in train if r["camera_id"] == cam and r["source"] == "sequence"]
        r = max(seqs, key=lambda r: (_motion(_frames_from_messages(r["messages"])), r["window_id"]))
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
    groups = {json.loads(l)["window_id"]: json.loads(l)["group"]
              for l in (pp.REPO_ROOT / "data" / "datasets" / DATASET / "test" / "test.jsonl").read_text().splitlines()}
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
        models.append({"name": display, "group": group, "macro_f1": s["macro_f1"], "macro_f1_ci": _f1_interval(rows, groups),
                       "accuracy": s["accuracy"],
                       "alert_recall": s["alert_recall"], "false_alarm_rate": s["false_alarm_rate"],
                       "valid_json_rate": s["valid_json_rate"], "latency_p50_s": p50, "latency_p95_s": p95,
                       "latency_estimated": lat_est, "usd_per_window": usd, "cost_estimated": cost_est,
                       "usd_per_camera_month": usd * WINDOWS_PER_CAMERA_MONTH, "via": s.get("via", "")})

    # Section 5: live demo clip (full-resolution keyframes for display) and the window the models receive
    live_dir = OUT / "live"
    shutil.rmtree(live_dir, ignore_errors=True); live_dir.mkdir(parents=True)
    keys = sorted(LIVE_SEQUENCE.glob("k*.png"), key=lambda p: int(p.stem[1:]))
    for i, k in enumerate(keys):
        Image.open(k).convert("RGB").save(live_dir / f"k{i + 1}.jpg", quality=88)
    _write_window(LIVE_WINDOW, _frames_from_test(LIVE_WINDOW))
    live_label = ours[LIVE_WINDOW]["label"] if LIVE_WINDOW in ours else _results(f"{DATASET}/test/qwen4b-v2")[LIVE_WINDOW]["label"]
    live = {"window_id": LIVE_WINDOW, "camera_id": "cam-B", "keyframes": len(keys), "label": live_label,
            "models": [{"id": i, "name": n} for i, n, _ in LIVE_MODELS]}

    stats = {k: {m: _summary(f"{DATASET}/test/{k}")[m] for m in ("macro_f1", "accuracy", "alert_recall", "false_alarm_rate", "n")}
             for k in ("qwen4b-base", "qwen4b-v2")}
    data = {"fields": fields, "frame_spacing_s": pp.FRAME_SPACING_S, "frames_per_window": pp.FRAMES_PER_WINDOW,
            "teacher_model": "Gemini 3.8 Flash", "teacher": teacher_section, "sample": sample, "stats": stats,
            "models": models, "windows_per_camera_month": WINDOWS_PER_CAMERA_MONTH, "opus_latency_source": OPUS_OR,
            "live": live, "bootstrap": BOOTSTRAP}
    (OUT / "data.json").write_text(json.dumps(data, indent=2))
    shutil.copy(Path(__file__).with_name("showcase.html"), OUT / "index.html")
    shutil.copytree(Path(__file__).with_name("showcase_assets"), OUT / "assets", dirs_exist_ok=True)
    print(f"built {OUT}: {len(teacher_section)} teacher windows, {len(sample)} sample windows, {len(models)} models")


def _stream_model(model_id: str, kind: str, vllm_url: str | None, frames: list[bytes], t0: float, emit) -> None:
    """Stream one model's answer, emitting {model, type: start|delta|done|error|offline, text, t} events."""
    import os
    import time
    cfg = dict(ev.models_config()[model_id])
    if kind == "vllm":
        if not vllm_url:
            return emit({"model": model_id, "type": "offline", "text": "GPU not connected", "t": 0})
        cfg["base_url"] = vllm_url
    body = ev.build_request(cfg, frames, "cam-B", "structured")
    body["stream"] = True
    headers = {"Authorization": f"Bearer {os.environ[cfg['api_key_env']]}"} if cfg.get("api_key_env") else {}
    if kind == "vllm" and os.environ.get("AW_VLLM_KEY"):  # the Modal deployment requires its API key
        headers = {"Authorization": f"Bearer {os.environ['AW_VLLM_KEY']}"}
    emit({"model": model_id, "type": "start", "t": round(time.perf_counter() - t0, 3)})
    text = ""
    try:
        with httpx.Client(timeout=httpx.Timeout(120, connect=5)) as client:
            with client.stream("POST", cfg["base_url"].rstrip("/") + "/chat/completions", json=body, headers=headers) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.read()[:200]!r}")
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue  # SSE comments (OpenRouter keep-alives) and blank lines
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                    if delta:
                        text += delta
                        emit({"model": model_id, "type": "delta", "text": delta, "t": round(time.perf_counter() - t0, 3)})
        emit({"model": model_id, "type": "done", "pred": ev.parse(text), "t": round(time.perf_counter() - t0, 3)})
    except Exception as e:  # noqa: BLE001 - shown in the panel
        offline = kind == "vllm" and isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout))
        emit({"model": model_id, "type": "offline" if offline else "error",
              "text": "GPU not connected" if offline else f"{type(e).__name__}: {str(e)[:160]}",
              "t": round(time.perf_counter() - t0, 3)})


class _Handler(http.server.SimpleHTTPRequestHandler):
    vllm_url: str | None = None

    def end_headers(self) -> None:  # always serve the latest build (a cached page once showed stale grading)
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

    def _send(self, ev_: dict) -> None:
        self.wfile.write(f"data: {json.dumps(ev_)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/live":
            return self.send_error(404)
        import queue
        import threading
        import time
        ev.load_env()
        frames = _frames_from_test(LIVE_WINDOW)
        q: queue.Queue = queue.Queue()
        t0 = time.perf_counter()
        threads = [threading.Thread(target=_stream_model, args=(m, kind, self.vllm_url, frames, t0, q.put), daemon=True)
                   for m, _, kind in LIVE_MODELS]
        for th in threads:
            th.start()
        self._sse_start()
        events, finished = [], 0
        while finished < len(threads):
            e = q.get()
            events.append(e)
            finished += e["type"] in ("done", "error", "offline")
            try:
                self._send(e)
            except (BrokenPipeError, ConnectionResetError):
                break
        # Record complete live runs (all three answered) so the demo can replay one if the venue network fails.
        if all(any(e["model"] == m and e["type"] == "done" for e in events) for m, _, _ in LIVE_MODELS):
            LIVE_RUNS.mkdir(parents=True, exist_ok=True)
            (LIVE_RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(events))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/live/replay":
            import time
            runs = sorted(LIVE_RUNS.glob("*.json")) if LIVE_RUNS.exists() else []
            if not runs:
                return self.send_error(404, "no recorded live run yet")
            self._sse_start()
            self._send({"type": "replay", "recorded": runs[-1].stem})
            start = time.perf_counter()
            for e in json.loads(runs[-1].read_text()):
                time.sleep(max(0.0, e.get("t", 0) - (time.perf_counter() - start)))
                self._send(e)
            return None
        if self.path == "/api/live/status":
            body = json.dumps({"vllm": bool(self.vllm_url), "replay_available": LIVE_RUNS.exists() and any(LIVE_RUNS.glob("*.json"))}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)
            return None
        return super().do_GET()


def serve(port: int, vllm_url: str | None) -> None:
    _Handler.vllm_url = vllm_url
    handler = functools.partial(_Handler, directory=str(OUT))
    print(f"AisleWatch showcase: http://127.0.0.1:{port}  (Qwen live via {vllm_url or 'not connected'})")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8790)
    s.add_argument("--vllm-url", help="vLLM serving qwen4b-base and qwen4b-v2, e.g. http://localhost:8000/v1 through an SSH tunnel")
    a = ap.parse_args()
    build() if a.cmd == "build" else serve(a.port, a.vllm_url)


if __name__ == "__main__":
    main()
