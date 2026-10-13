"""Contact strips for generated keyframe sequences: one row per sequence, keyframes left to right,
each tile captioned with its intended door state. For quick visual QA of person/scene drift.

Usage: uv run aw-review-sheets data/gen/seq/cam-A [--out data/gen/review/seq-cam-A.jpg]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from aislewatch.preprocess import REPO_ROOT

TILE_W, TILE_H = 480, 270


def strip(seq_dir: Path) -> Image.Image:
    frames = sorted(seq_dir.glob("k*.png"), key=lambda p: int(p.stem[1:]))
    row = Image.new("RGB", (TILE_W * max(len(frames), 1), TILE_H + 22), "black")
    d = ImageDraw.Draw(row)
    d.text((6, 4), seq_dir.name, fill="white")
    for i, f in enumerate(frames):
        im = Image.open(f).convert("RGB").resize((TILE_W, TILE_H))
        meta_path = f.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        label = f"{f.stem}: {meta.get('frame_state', '?')}"
        if meta.get("label") not in (None, "unverified"):
            label += f" [{meta['label']}]"
        dd = ImageDraw.Draw(im)
        dd.rectangle([0, TILE_H - 22, 10 + 7 * len(label), TILE_H], fill="black")
        dd.text((5, TILE_H - 18), label, fill="yellow")
        row.paste(im, (i * TILE_W, 22))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cam_dir")
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cam_dir = REPO_ROOT / args.cam_dir
    seqs = sorted(p for p in cam_dir.iterdir() if p.is_dir() and (not args.only or p.name in args.only))
    rows = [strip(s) for s in seqs if any(s.glob("k*.png"))]
    width = max(r.width for r in rows)
    sheet = Image.new("RGB", (width, sum(r.height for r in rows)), "black")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    out = REPO_ROOT / (args.out or f"data/gen/review/seq-{cam_dir.name}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    print(out)


if __name__ == "__main__":
    main()
