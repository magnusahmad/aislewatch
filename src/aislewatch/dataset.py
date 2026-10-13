"""Build training, validation and test windows from labelled stills and keyframe sequences.

- Stills -> static 10-frame windows (same scene, per-frame sensor noise).
- Keyframe sequences -> windows over a timeline where each keyframe is held for a random 1-3 s,
  so the same sequence yields several windows with different timing.
- Camera B is the held-out test camera and never enters train/val.
- Labels come from the human review (label == "accepted", human_state); --include-unverified
  falls back to intended states for dry runs only.

Outputs data/datasets/<name>/: train.parquet and val.parquet (prime-rl SFT `messages`), test/ (frames
plus test.jsonl for evaluation), manifest.json (counts per split and state, provenance).

Usage: uv run aw-build-dataset --name v1 [--include-unverified]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from aislewatch import degrade
from aislewatch import preprocess as pp

GEN = pp.REPO_ROOT / "data" / "gen"
TEST_CAMERAS = {"cam-B"}
OBJ_WORDS = {"box": "cardboard box", "cart": "equipment cart", "other": "object"}


# ---------- labels -------------------------------------------------------------------------------

def _state(meta: dict, include_unverified: bool) -> str | None:
    if meta.get("label") == "accepted" and meta.get("human_state"):
        return meta["human_state"]
    if meta.get("label") == "rejected":
        return None
    return meta.get("intended_state") if include_unverified else None


def _has_person(meta: dict) -> bool:
    people = meta.get("prompt", "").split("People:", 1)[-1].split("\n", 1)[0].strip().lower()
    return bool(people) and not people.startswith(("none", "nobody"))


def _object(sid: str, state: str) -> str:
    if state != "propped":
        return "none"
    if "cart" in sid:
        return "cart"
    if "box" in sid:
        return "box"
    return "other"  # extinguisher, wedge


def _caption(state: str, opening: str, obj: str, person: bool, in_doorway: bool) -> str:
    busy = ", while a technician works nearby without moving through the doorway" if person and not in_doorway else ""
    if state == "closed":
        return "The containment doors are closed" + (", and a technician is working nearby." if person else ".")
    if state == "passing":
        return "A technician is moving through the open containment doorway."
    if state == "propped":
        return f"The containment door is held open by a {OBJ_WORDS[obj]} in the doorway and nobody is passing through{busy}."
    what = "Both containment doors are fully open" if opening == "full" else "The containment door is left partly open"
    return f"{what} with nothing holding it and nobody in the doorway{busy}."


def still_caption(meta: dict, state: str) -> dict:
    sid = meta["id"]
    opening = "none" if state == "closed" else ("full" if "open-full" in sid or "open-full" in meta.get("base_image", "") else "partial")
    obj = _object(sid, state)
    person = _has_person(meta)
    return {"door_state": state, "door_opening": opening, "state_changed_in_window": False,
            "people_visible": int(person), "person_in_doorway_at_end": False, "object_in_doorway": obj,
            "caption": _caption(state, opening, obj, person, False)}


def keyframe_fields(seq: str, meta: dict, state: str) -> dict:
    person = _has_person(meta)
    in_doorway = state == "passing"
    if state == "closed":
        opening = "none"
    elif "cart-through" in seq:
        opening = "full"
    else:
        opening = "partial"
    obj = _object(seq, state)
    if state == "passing" and "cart-through" in seq and meta.get("order") == 2:
        obj = "cart"
    return {"door_state": state, "door_opening": opening, "people_visible": int(person),
            "person_in_doorway_at_end": in_doorway, "object_in_doorway": obj,
            "caption": _caption(state, opening, obj, person, in_doorway)}


# ---------- windows ------------------------------------------------------------------------------

def _frames(images: list[Image.Image], cond: degrade.Conditions, rng: random.Random) -> list[bytes]:
    return [pp.encode_jpeg(degrade.apply(pp.prepare_frame(im), cond, rng)) for im in images]


def _hash(group: str) -> int:
    return int(hashlib.sha1(group.encode()).hexdigest(), 16)


def choose_val_groups(groups_by_state: dict[str, list[str]], val_frac: float) -> set[str]:
    """Per state, the val_frac of groups with the lowest hash (at least one), so val covers every state."""
    chosen: set[str] = set()
    for state, groups in groups_by_state.items():
        ordered = sorted(set(groups), key=_hash)
        chosen.update(ordered[:max(1, round(len(ordered) * val_frac))])
    return chosen


def _split(camera: str, group: str, val_groups: set[str]) -> str:
    if camera in TEST_CAMERAS:
        return "test"
    return "val" if group in val_groups else "train"


def collect_stills(include_unverified: bool):
    for meta_path in sorted(list((GEN / "stills").glob("cam-*/*.json")) + list((GEN / "bases").glob("*.json"))):
        meta = json.loads(meta_path.read_text())
        img_path = meta_path.with_suffix(".png")
        if not img_path.exists():
            continue
        if meta_path.parent.name == "bases":
            meta.setdefault("id", meta_path.stem)
        state = _state(meta, include_unverified)
        if state:
            yield meta, img_path, state


def collect_sequences(include_unverified: bool):
    for seq_dir in sorted((GEN / "seq").glob("cam-*/*")):
        keys = sorted(seq_dir.glob("k*.json"), key=lambda p: int(p.stem[1:]))
        metas = [json.loads(k.read_text()) for k in keys]
        states = [_state(m, include_unverified) for m in metas]
        if not keys or any(s is None for s in states):
            continue  # a sequence is usable only if every keyframe is labelled
        yield seq_dir, metas, [k.with_suffix(".png") for k in keys], states


def build(name: str, include_unverified: bool, seed: int, still_variants: dict, seq_variants: dict,
          val_frac: float) -> Path:
    rng = random.Random(seed)
    out = pp.REPO_ROOT / "data" / "datasets" / name
    (out / "test").mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": [], "test": []}
    counts = {s: Counter() for s in rows}

    def add(split, window_id, camera, frames, caption, source, group):
        if split == "test":
            d = out / "test" / window_id
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                (d / f"f{i:02d}.jpg").write_bytes(f)
            rows["test"].append({"window_id": window_id, "camera_id": camera, "source": source, "group": group,
                                 "frames_dir": str(d.relative_to(out)), "label": caption})
        else:
            rows[split].append({"messages": pp.build_messages(frames, camera) + [pp.assistant_message(caption)],
                                "window_id": window_id, "camera_id": camera, "source": source, "group": group,
                                "door_state": caption["door_state"]})
        counts[split][caption["door_state"]] += 1

    stills = list(collect_stills(include_unverified))
    seqs = list(collect_sequences(include_unverified))
    by_state: dict[str, list[str]] = {}
    for meta, img_path, state in stills:
        if (meta.get("camera_id") or img_path.stem.split("_")[0]) not in TEST_CAMERAS:
            by_state.setdefault(state, []).append(f"still:{meta['id']}")
    for seq_dir, metas, _, states in seqs:
        if metas[0]["camera_id"] not in TEST_CAMERAS:
            by_state.setdefault("seq:" + states[-1], []).append(f"seq:{seq_dir.name}")
    val_groups = choose_val_groups(by_state, val_frac)

    for meta, img_path, state in stills:
        camera = meta.get("camera_id") or img_path.stem.split("_")[0]
        group = f"still:{meta['id']}"
        split = _split(camera, group, val_groups)
        img = Image.open(img_path).convert("RGB")
        caption = still_caption(meta, state)
        for v in range(still_variants[split]):
            cond = degrade.Conditions.sample(rng, mild=(v == 0))
            add(split, f"{meta['id']}-s{v}", camera, _frames([img] * pp.FRAMES_PER_WINDOW, cond, rng),
                caption, "still", group)

    for seq_dir, metas, imgs, states in seqs:
        camera = metas[0]["camera_id"]
        seq = seq_dir.name
        split = _split(camera, f"seq:{seq}", val_groups)
        pil = [Image.open(p).convert("RGB") for p in imgs]
        for v in range(seq_variants[split]):
            timeline = [i for i in range(len(pil)) for _ in range(rng.randint(2, 6))]  # 1-3 s per keyframe
            cond = degrade.Conditions.sample(rng, mild=(v == 0))
            for w, start in enumerate(range(0, max(1, len(timeline) - pp.FRAMES_PER_WINDOW + 1), 4)):
                idx = timeline[start:start + pp.FRAMES_PER_WINDOW]
                if len(idx) < pp.FRAMES_PER_WINDOW:
                    break
                last, first = idx[-1], idx[0]
                caption = {**keyframe_fields(seq, metas[last], states[last]),
                           "state_changed_in_window": states[first] != states[last]}
                caption = {k: caption[k] for k in pp.caption_field_order()}
                add(split, f"{seq}-t{v}-w{w}", camera, _frames([pil[i] for i in idx], cond, rng),
                    caption, "sequence", f"seq:{seq}")

    for split in ("train", "val"):
        rng.shuffle(rows[split])
        if rows[split]:
            pq.write_table(pa.Table.from_pylist(rows[split]), out / f"{split}.parquet")
    with (out / "test" / "test.jsonl").open("w") as f:
        for r in rows["test"]:
            f.write(json.dumps(r) + "\n")
    manifest = {"val_groups": sorted(val_groups), "name": name, "seed": seed, "include_unverified": include_unverified,
                "long_side_px": pp.LONG_SIDE_PX, "test_cameras": sorted(TEST_CAMERAS),
                "counts": {s: dict(c) for s, c in counts.items()},
                "totals": {s: sum(c.values()) for s, c in counts.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=1), manifest["totals"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    build(args.name, args.include_unverified, args.seed,
          still_variants={"train": 6, "val": 2, "test": 2}, seq_variants={"train": 4, "val": 2, "test": 2},
          val_frac=args.val_frac)


if __name__ == "__main__":
    main()
