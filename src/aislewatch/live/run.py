"""Run the live system: scenario feeds -> captioner -> tracker + temperature simulator.

Usage: uv run aw-live --scenario demo [--speed 1.0] [--base-url http://localhost:8000/v1] [--fresh]
Set the live adapter with: uv run aw-live --set-current qwen-v1   (or qwen-base)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time

from aislewatch.live import feeds, workers
from aislewatch.live.store import LIVE_DIR
from aislewatch.preprocess import REPO_ROOT


async def main_async(args) -> None:
    stop = asyncio.Event()
    clock = LIVE_DIR / "clock.json"
    if not clock.exists():
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        clock.write_text(json.dumps({"t0": round(time.time())}))
    t0 = workers.site_clock_start()
    tasks = [asyncio.create_task(feeds.run_feeds(args.scenario, args.speed, t0, stop))]
    if not args.no_caption:
        tasks.append(asyncio.create_task(workers.run_captioner(stop, args.base_url)))
    tasks += [asyncio.create_task(workers.run_tracker(stop)), asyncio.create_task(workers.run_sensors(stop))]
    try:
        await asyncio.gather(*tasks)
    finally:
        stop.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="demo")
    ap.add_argument("--speed", type=float, default=1.0, help="site seconds per wall second")
    ap.add_argument("--base-url", help="vLLM endpoint, e.g. http://localhost:8000/v1 through the tunnel")
    ap.add_argument("--fresh", action="store_true", help="wipe data/live and the Hermes site-rules skill before starting")
    ap.add_argument("--no-caption", action="store_true", help="feeds, sensors and tracker only (no model calls)")
    ap.add_argument("--set-current", help="set the live adapter model name and exit")
    args = ap.parse_args()
    if args.set_current:
        p = REPO_ROOT / "models" / "current.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model": args.set_current}))
        print(f"current model -> {args.set_current}")
        return
    if args.fresh:
        if LIVE_DIR.exists():
            shutil.rmtree(LIVE_DIR)
        from aislewatch.live.mcp_server import SITE_RULES
        if SITE_RULES.exists():  # the demo creates this live from the manager's correction
            shutil.rmtree(SITE_RULES.parent)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
