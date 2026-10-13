"""Scenario feeds: play labelled stills and keyframe sequences as 2 fps camera streams.

A scenario (config/scenarios/<name>.json) lists segments per camera. Each segment is a still held for
some seconds, or a keyframe sequence with a hold time per keyframe. The player cuts the stream into
5 s windows (10 frames), runs them through the same preprocessing as training, writes the frames
to disk and a row to the `windows` table with the ground-truth state at the window's end.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from aislewatch import degrade
from aislewatch import preprocess as pp
from aislewatch.live.store import LIVE_DIR, connect

GEN = pp.REPO_ROOT / "data" / "gen"


@dataclass
class Frame:
    image: Path
    state: str
    source: str


def _state(meta_path: Path) -> str:
    m = json.loads(meta_path.read_text())
    return m.get("human_state") or m.get("frame_state") or m.get("intended_state")


def expand_segment(seg: dict) -> list[Frame]:
    """One segment -> frames at 2 fps."""
    per_s = int(1 / pp.FRAME_SPACING_S)
    if "still" in seg:
        img = GEN / f"{seg['still']}.png"
        return [Frame(img, _state(img.with_suffix(".json")), seg["still"])] * int(seg["seconds"] * per_s)
    seq_dir = GEN / "seq" / seg["sequence"]
    keys = sorted(seq_dir.glob("k*.png"), key=lambda p: int(p.stem[1:]))
    holds = seg.get("hold_s") or [seg.get("each_s", 2)] * len(keys)
    out: list[Frame] = []
    for k, hold in zip(keys, holds):
        out += [Frame(k, _state(k.with_suffix(".json")), seg["sequence"])] * int(hold * per_s)
    return out


def load_scenario(name: str) -> dict:
    return json.loads((pp.REPO_ROOT / "config" / "scenarios" / f"{name}.json").read_text())


class CameraFeed:
    def __init__(self, camera_id: str, aisle: str, segments: list[dict], seed: int):
        self.camera_id, self.aisle = camera_id, aisle
        self.frames = [f for seg in segments for f in expand_segment(seg)]
        self.rng = random.Random(seed)
        self.cond = degrade.Conditions.sample(self.rng, mild=True)  # one camera, one look
        self.pos = 0
        self._cache: dict[Path, Image.Image] = {}

    def _img(self, p: Path) -> Image.Image:
        if p not in self._cache:
            self._cache[p] = pp.prepare_frame(Image.open(p))
        return self._cache[p]

    def next_window(self) -> tuple[list[bytes], Frame]:
        idx = [(self.pos + i) % len(self.frames) for i in range(pp.FRAMES_PER_WINDOW)]
        self.pos = (self.pos + pp.FRAMES_PER_WINDOW) % len(self.frames)
        frames = [pp.encode_jpeg(degrade.apply(self._img(self.frames[i].image), self.cond, self.rng)) for i in idx]
        return frames, self.frames[idx[-1]]


async def run_feeds(scenario: str, speed: float, site_t0: float, stop: asyncio.Event) -> None:
    sc = load_scenario(scenario)
    con = connect()
    feeds = [CameraFeed(c["camera_id"], c["aisle"], c["segments"], seed=i) for i, c in enumerate(sc["cameras"])]
    wall_t0 = time.time()
    n = 0
    while not stop.is_set():
        t_start = site_t0 + n * pp.WINDOW_S
        for feed in feeds:
            frames, last = await asyncio.to_thread(feed.next_window)  # CPU work off the event loop
            window_id = f"{feed.camera_id}-{int(t_start)}"
            d = LIVE_DIR / "frames" / feed.camera_id / window_id
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                (d / f"f{i:02d}.jpg").write_bytes(f)
            con.execute("INSERT OR REPLACE INTO windows VALUES (?,?,?,?,?,?,?,?)",
                        (window_id, feed.camera_id, feed.aisle, t_start, t_start + pp.WINDOW_S,
                         str(d.relative_to(LIVE_DIR)), last.state, last.source))
        n += 1
        # Pace to real time / speed: window n becomes available once its 5 s of site time has passed.
        delay = wall_t0 + n * pp.WINDOW_S / speed - time.time()
        await asyncio.sleep(0)  # yield even when behind schedule, so the other workers run
        if delay > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
