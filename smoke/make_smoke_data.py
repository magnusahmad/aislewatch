"""Build the S1/S2 smoke-test data from the baseline clip.

Two 5 s windows (0-5 s passing, 5-10 s propped), each with 4 light augmentations,
gives 8 training rows: enough to check the format, the loss mask and that the
model can overfit. Also writes the two clean windows as S2 request files.

Usage: uv run python smoke/make_smoke_data.py <clip.mp4>
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageEnhance

from aislewatch import preprocess as pp

OUT = pp.REPO_ROOT / "data" / "smoke"
CAMERA_ID = "cam-A"

# Ground truth for the baseline clip (Oct 10 test): technician opens the door,
# props it with a box and walks back into the aisle.
WINDOWS = [
    (0.0, {
        "door_state": "passing", "door_opening": "partial", "state_changed_in_window": True,
        "people_visible": 1, "person_in_doorway_at_end": True, "object_in_doorway": "none",
        "caption": "A technician carrying a cardboard box slides the left containment door open and walks into the doorway.",
    }),
    (5.0, {
        "door_state": "propped", "door_opening": "partial", "state_changed_in_window": True,
        "people_visible": 1, "person_in_doorway_at_end": False, "object_in_doorway": "box",
        "caption": "The technician leaves a cardboard box in the door track holding the door open and walks back into the aisle.",
    }),
]


def augment(frames: list[Image.Image], seed: int) -> list[Image.Image]:
    """Same random crop and brightness/contrast for all frames of a window."""
    rng = random.Random(seed)
    w, h = frames[0].size
    scale = rng.uniform(0.9, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    b, c = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
    out = []
    for f in frames:
        g = f.crop((x0, y0, x0 + cw, y0 + ch))
        g = ImageEnhance.Brightness(g).enhance(b)
        out.append(ImageEnhance.Contrast(g).enhance(c))
    return out


def main(clip: Path) -> None:
    (OUT / "sft").mkdir(parents=True, exist_ok=True)
    (OUT / "requests").mkdir(parents=True, exist_ok=True)
    rows = []
    for w_idx, (t0, label) in enumerate(WINDOWS):
        frames = pp.extract_window(clip, t0)
        frame_dir = OUT / "frames" / f"w{w_idx}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        clean = [pp.encode_jpeg(pp.resize(f)) for f in frames]
        for i, jpeg in enumerate(clean):
            (frame_dir / f"f{i:02d}.jpg").write_bytes(jpeg)
        request = {"window_id": f"smoke-w{w_idx}", "label": label,
                   "messages": pp.build_messages(clean, CAMERA_ID)}
        (OUT / "requests" / f"w{w_idx}.json").write_text(json.dumps(request))
        variants = [clean] + [
            [pp.encode_jpeg(pp.resize(f)) for f in augment(frames, seed=w_idx * 100 + k)] for k in range(1, 4)
        ]
        for k, jpegs in enumerate(variants):
            rows.append({
                "messages": pp.build_messages(jpegs, CAMERA_ID) + [pp.assistant_message(label)],
                "window_id": f"smoke-w{w_idx}-aug{k}",
            })
    pq.write_table(pa.Table.from_pylist(rows), OUT / "sft" / "train.parquet")
    print(f"Wrote {len(rows)} rows to {OUT / 'sft' / 'train.parquet'}")
    print(f"Wrote S2 requests to {OUT / 'requests'}; frames to {OUT / 'frames'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
