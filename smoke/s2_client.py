"""S2 client, run on the laptop through the SSH tunnel (localhost:8000 -> pod 127.0.0.1:8000).

Checks: base and S1 adapter both return schema-valid JSON with thinking off; runtime
adapter loading works; p50/p95 latency per window at 3 cameras x 2 models.

Usage: uv run python smoke/s2_client.py [--base-url http://localhost:8000] [--adapter-path /root/runs/s1/adapter]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from aislewatch import preprocess as pp

REQ_DIR = pp.REPO_ROOT / "data" / "smoke" / "requests"


def payload(model: str, messages: list[dict]) -> dict:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 400,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "caption", "schema": pp.caption_schema(), "strict": True}},
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def caption(client: httpx.AsyncClient, model: str, messages: list[dict]) -> tuple[dict | None, float, dict]:
    t = time.perf_counter()
    r = await client.post("/v1/chat/completions", json=payload(model, messages))
    elapsed = time.perf_counter() - t
    r.raise_for_status()
    body = r.json()
    text = body["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if "<think>" in (text or ""):
        print("  WARNING: thinking tokens in output")
    return parsed, elapsed, body.get("usage", {})


async def main(base_url: str, adapter_path: str) -> None:
    requests = [json.loads(p.read_text()) for p in sorted(REQ_DIR.glob("w*.json"))]
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        models = [m["id"] for m in (await client.get("/v1/models")).json()["data"]]
        print("Served models:", models)

        print("\n== Runtime adapter load ==")
        r = await client.post("/v1/load_lora_adapter", json={"lora_name": "smoke-rt", "lora_path": adapter_path})
        print(r.status_code, r.text[:200])

        print("\n== Correctness (thinking off, structured output) ==")
        for model in ["qwen-base", "smoke-s1", "smoke-rt"]:
            for req in requests:
                parsed, elapsed, usage = await caption(client, model, req["messages"])
                truth = req["label"]["door_state"]
                got = parsed.get("door_state") if parsed else "INVALID"
                print(f"  {model:10s} {req['window_id']}: door_state={got} (truth {truth}) "
                      f"{elapsed:.2f}s prompt_tokens={usage.get('prompt_tokens')}")

        print("\n== Load: 3 cameras x 2 models, 5 rounds ==")
        latencies = []
        for _ in range(5):
            jobs = [caption(client, m, requests[i % len(requests)]["messages"])
                    for i in range(3) for m in ("qwen-base", "smoke-s1")]
            for parsed, elapsed, _usage in await asyncio.gather(*jobs):
                latencies.append(elapsed)
        latencies.sort()
        p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
        print(f"  n={len(latencies)} p50={statistics.median(latencies):.2f}s p95={p95:.2f}s (window cadence 5 s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--adapter-path", default="/home/ubuntu/runs/s1/adapter")
    args = ap.parse_args()
    asyncio.run(main(args.base_url, args.adapter_path))
