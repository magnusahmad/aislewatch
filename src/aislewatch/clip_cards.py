"""Make clip cards (first frame, last frame, prompt) for manual frames-to-video generation in Flow.

Writes data/gen/clip-cards/<camera>/<clip-id>/{first.png,last.png,prompt.txt} plus CARDS.md, a
checklist to work through. Downloaded clips go in data/gen/clips/inbox/<clip-id>[-take2].mp4.

Usage: uv run aw-clip-cards config/gen/clips-cam-A.json
"""

from __future__ import annotations

import json
import shutil
import sys

from aislewatch.preprocess import REPO_ROOT


def main() -> None:
    spec = json.loads((REPO_ROOT / sys.argv[1]).read_text())
    cam = spec["camera_id"]
    out = REPO_ROOT / "data" / "gen" / "clip-cards" / cam
    inbox = REPO_ROOT / "data" / "gen" / "clips" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    def frame(name: str):
        return REPO_ROOT / (spec["base_closed"] if name == "base" else f"{spec['stills_dir']}/{name}.png")

    lines = [f"# Clip cards: {cam} ({spec['split']})", "",
             "For each card: in Flow, Frames to video, Veo 3.1, 16:9, 8 s. Upload `first.png` as the start frame "
             "and `last.png` as the end frame, paste `prompt.txt`. Make 2 takes; save the better one (or both) as "
             f"`{inbox.relative_to(REPO_ROOT)}/<clip-id>.mp4` (second keeper: `<clip-id>-take2.mp4`).",
             "Reject a take if the camera moves, the doors swing on hinges instead of sliding, people pop in or out, "
             "or you can't tell the door state at the end.", ""]
    for clip in spec["clips"]:
        d = out / clip["id"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frame(clip["first"]), d / "first.png")
        shutil.copyfile(frame(clip["last"]), d / "last.png")
        prompt = spec["template"].format(action=clip["action"])
        (d / "prompt.txt").write_text(prompt + "\n")
        lines += [f"## [ ] {clip['id']}", f"- Expected: {clip['timeline_hint']}",
                  f"- Frames: `{clip['first']}` → `{clip['last']}`", f"- Folder: `{d.relative_to(REPO_ROOT)}`",
                  "", "```", prompt, "```", ""]
    (out / "CARDS.md").write_text("\n".join(lines))
    print(f"Wrote {len(spec['clips'])} cards to {out}")


if __name__ == "__main__":
    main()
