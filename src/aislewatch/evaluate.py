"""Evaluate captioning models on a dataset split and write per-window predictions plus a summary.

Works with any OpenAI-compatible endpoint: vLLM (base and LoRA adapters), Gemini's OpenAI-compatible
API, OpenAI. Endpoints and prices come from config/models.json.

Usage:
  uv run aw-eval --dataset v1 --split test --model qwen-base [--model qwen-v1 ...] [--prompt structured|generic]
                 [--shuffle-frames] [--limit N] [--concurrency 6]
  uv run aw-scoreboard --dataset v1
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from aislewatch import preprocess as pp

STATES = ["closed", "passing", "propped", "ajar"]
ALERT_STATES = {"propped", "ajar"}
GENERIC_PROMPT = ("These are 10 consecutive frames from one fixed CCTV camera ({camera_id}) in a data centre, 0.5 seconds "
                  "apart, covering 5 seconds. Describe what happens in one or two sentences.")
DOOR_WORDS = re.compile(r"\b(propped|held open|left open|ajar|not (fully )?closed|wedged|open(ed)?|closed|shut)\b", re.I)


# ---------- data -------------------------------------------------------------------------------

def load_split(dataset: str, split: str) -> list[dict]:
    root = pp.REPO_ROOT / "data" / "datasets" / dataset
    items = []
    if split == "test":
        for line in (root / "test" / "test.jsonl").read_text().splitlines():
            r = json.loads(line)
            d = root / r["frames_dir"]
            frames = [(d / f"f{i:02d}.jpg").read_bytes() for i in range(pp.FRAMES_PER_WINDOW)]
            items.append({**r, "frames": frames})
    else:
        for r in pq.read_table(root / f"{split}.parquet").to_pylist():
            imgs = [c["image_url"]["url"] for c in r["messages"][1]["content"] if c.get("image_url")]
            frames = [base64.b64decode(u.split(",", 1)[1]) for u in imgs]
            label = json.loads(r["messages"][2]["content"][0]["text"])
            items.append({"window_id": r["window_id"], "camera_id": r["camera_id"], "source": r["source"],
                          "group": r["group"], "label": label, "frames": frames})
    return items


def models_config() -> dict:
    return json.loads((pp.REPO_ROOT / "config" / "models.json").read_text())


# ---------- requests ---------------------------------------------------------------------------

def build_request(model_cfg: dict, frames: list[bytes], camera_id: str, prompt: str) -> dict:
    messages = pp.build_messages(frames, camera_id)
    if prompt == "generic":
        images = [p for p in messages[1]["content"] if p["type"] == "image_url"]
        content = []
        for i, img in enumerate(images):
            content += [{"type": "text", "text": f"Frame {i + 1} (t={i * pp.FRAME_SPACING_S:.1f}s): "}, img]
        content.append({"type": "text", "text": "\n" + GENERIC_PROMPT.replace("{camera_id}", camera_id)})
        messages = [{"role": "user", "content": content}]
    body = {"model": model_cfg["model"], "messages": messages, "temperature": 0, "max_tokens": 400}
    if prompt == "structured":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "caption", "schema": pp.caption_schema(), "strict": True}}
    body.update(model_cfg.get("extra_body", {}))
    return body


async def call(client: httpx.AsyncClient, model_cfg: dict, body: dict) -> tuple[str | None, float, dict, str | None]:
    headers = {}
    if model_cfg.get("api_key_env"):
        headers["Authorization"] = f"Bearer {os.environ[model_cfg['api_key_env']]}"
    t = time.perf_counter()
    for attempt in range(4):
        try:
            r = await client.post(model_cfg["base_url"].rstrip("/") + "/chat/completions", json=body, headers=headers)
            if r.status_code in (429, 500, 502, 503) and attempt < 3:
                await asyncio.sleep(2 ** attempt * 3)
                continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"].get("content"), time.perf_counter() - t, data.get("usage", {}), None
        except Exception as e:  # noqa: BLE001 - recorded per window
            err = f"{type(e).__name__}: {e}"
            if attempt == 3:
                return None, time.perf_counter() - t, {}, err
            await asyncio.sleep(2 ** attempt * 3)
    return None, time.perf_counter() - t, {}, "exhausted retries"


def parse(text: str | None) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1].strip()
    t = re.sub(r"^```(json)?|```$", "", t, flags=re.M).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) and obj.get("door_state") in STATES else None


# ---------- metrics ----------------------------------------------------------------------------

def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    valid = [r for r in rows if r["pred"] is not None]
    pred = lambda r: r["pred"]["door_state"] if r["pred"] else "invalid"  # noqa: E731
    per_class = {}
    for s in STATES:
        tp = sum(1 for r in rows if r["label"]["door_state"] == s and pred(r) == s)
        fp = sum(1 for r in rows if r["label"]["door_state"] != s and pred(r) == s)
        fn = sum(1 for r in rows if r["label"]["door_state"] == s and pred(r) != s)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        support = tp + fn
        per_class[s] = {"precision": round(p, 3), "recall": round(rc, 3),
                        "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0, "support": support}
    present = [s for s in STATES if per_class[s]["support"]]
    alert_truth = [r for r in rows if r["label"]["door_state"] in ALERT_STATES]
    no_alert_truth = [r for r in rows if r["label"]["door_state"] not in ALERT_STATES]
    fields = [f for f in pp.caption_field_order() if f not in ("door_state", "caption")]
    field_acc = {f: round(sum(1 for r in valid if r["pred"].get(f) == r["label"].get(f)) / len(valid), 3) if valid else 0.0
                 for f in fields}
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "n": n,
        "valid_json_rate": round(len(valid) / n, 3) if n else 0.0,
        "accuracy": round(sum(1 for r in rows if pred(r) == r["label"]["door_state"]) / n, 3) if n else 0.0,
        "macro_f1": round(sum(per_class[s]["f1"] for s in present) / len(present), 3) if present else 0.0,
        "alert_recall": round(sum(1 for r in alert_truth if pred(r) in ALERT_STATES) / len(alert_truth), 3) if alert_truth else None,
        "false_alarm_rate": round(sum(1 for r in no_alert_truth if pred(r) in ALERT_STATES) / len(no_alert_truth), 3) if no_alert_truth else None,
        "per_class": per_class,
        "confusion": dict(Counter(f"{r['label']['door_state']}->{pred(r)}" for r in rows)),
        "field_accuracy": field_acc,
        "latency_p50_s": round(lat[len(lat) // 2], 2) if lat else None,
        "latency_p95_s": round(lat[max(0, int(len(lat) * 0.95) - 1)], 2) if lat else None,
        "prompt_tokens_mean": round(sum(r["usage"].get("prompt_tokens", 0) for r in rows) / n) if n else None,
        "completion_tokens_mean": round(sum(r["usage"].get("completion_tokens", 0) for r in rows) / n) if n else None,
    }


def generic_metrics(rows: list[dict]) -> dict:
    mentions = [r for r in rows if r["text"] and DOOR_WORDS.search(r["text"])]
    return {"n": len(rows), "mentions_door_state_rate": round(len(mentions) / len(rows), 3) if rows else 0.0}


# ---------- run --------------------------------------------------------------------------------

async def run_model(name: str, cfg: dict, items: list[dict], prompt: str, shuffle: bool, concurrency: int,
                    out_dir: Path, seed: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    rng = random.Random(seed)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=180) as client:
        async def one(item):
            frames = list(item["frames"])
            if shuffle:
                rng.shuffle(frames)
            async with sem:
                text, latency, usage, err = await call(client, cfg, build_request(cfg, frames, item["camera_id"], prompt))
            return {"window_id": item["window_id"], "camera_id": item["camera_id"], "source": item["source"],
                    "label": item["label"], "text": text, "pred": parse(text) if prompt == "structured" else None,
                    "latency_s": round(latency, 3), "usage": usage, "error": err}
        rows = await asyncio.gather(*(one(i) for i in items))
    wall_s = time.perf_counter() - started
    tag = f"{name}-{prompt}" + ("-shuffled" if shuffle else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{tag}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = metrics(rows) if prompt == "structured" else generic_metrics(rows)
    summary.update(model=name, prompt=prompt, shuffled=shuffle, errors=sum(1 for r in rows if r["error"]),
                   wall_s=round(wall_s, 1), concurrency=concurrency)
    (out_dir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", required=True, help="name from config/models.json")
    ap.add_argument("--prompt", default="structured", choices=["structured", "generic"])
    ap.add_argument("--shuffle-frames", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", help="override base_url for vLLM models (e.g. through a tunnel)")
    args = ap.parse_args()
    cfgs = models_config()
    items = load_split(args.dataset, args.split)[: args.limit]
    out_dir = pp.REPO_ROOT / "data" / "results" / args.dataset / args.split
    for name in args.model:
        cfg = dict(cfgs[name])
        if args.base_url and cfg.get("kind") == "vllm":
            cfg["base_url"] = args.base_url
        s = asyncio.run(run_model(name, cfg, items, args.prompt, args.shuffle_frames, args.concurrency, out_dir, args.seed))
        print(json.dumps({k: s[k] for k in s if k not in ("per_class", "confusion", "field_accuracy")}))


def scoreboard() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    d = pp.REPO_ROOT / "data" / "results" / args.dataset / args.split
    cfgs = models_config()
    lines = ["| Model | Prompt | n | Macro F1 | Accuracy | Propped/ajar recall | False alarms | Valid JSON | p95 latency | $/10k windows |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(d.glob("*.summary.json")):
        s = json.loads(p.read_text())
        if s["prompt"] != "structured":
            lines.append(f"| {s['model']} | generic | {s['n']} | mentions door state: {s['mentions_door_state_rate']:.0%} | | | | | | |")
            continue
        cfg = cfgs.get(s["model"], {})
        cost = cost_per_10k(cfg, s)
        lines.append(f"| {s['model']}{' (shuffled)' if s['shuffled'] else ''} | structured | {s['n']} | {s['macro_f1']:.3f} | "
                     f"{s['accuracy']:.3f} | {s['alert_recall']} | {s['false_alarm_rate']} | {s['valid_json_rate']:.0%} | "
                     f"{s['latency_p95_s']}s | {cost} |")
    out = d / "SCOREBOARD.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cost_per_10k(cfg: dict, s: dict) -> str:
    price = cfg.get("price")
    if not price:
        return "n/a"
    if "usd_per_m_input" in price:
        c = (s["prompt_tokens_mean"] * price["usd_per_m_input"] + s["completion_tokens_mean"] * price["usd_per_m_output"]) / 1e6
        return f"${c * 10_000:.2f}"
    if "usd_per_call" in price:
        return f"${price['usd_per_call'] * 10_000:.2f}"
    if "usd_per_gpu_hour" in price and s.get("wall_s"):
        # Throughput-based: GPU cost of the measured run spread over its windows (only meaningful when the
        # GPU served this model alone at the stated concurrency).
        return f"${price['usd_per_gpu_hour'] * s['wall_s'] / 3600 / s['n'] * 10_000:.2f}*"
    return "n/a"


if __name__ == "__main__":
    main()
