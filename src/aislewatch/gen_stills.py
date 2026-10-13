"""Generate door-state stills by editing a base image with Codex's built-in image_gen (subscription, no API).

One `codex exec` run per job, so a failure doesn't take down the batch. Each output gets a
sidecar JSON with the intended state; the label stays "unverified" until a person checks it.

Usage: uv run aw-gen-stills config/gen/cam-A-test.json [--only ID ...] [--force]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from aislewatch.preprocess import REPO_ROOT


def run_job(spec: dict, job: dict, force: bool) -> dict:
    out_dir = REPO_ROOT / job.get("out_dir", spec.get("out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job['id']}.png"
    meta_path = out_dir / f"{job['id']}.json"
    if out_path.exists() and not force:
        return {"id": job["id"], "status": "skipped (exists)"}
    template = spec["templates"][job.get("template", "edit")] if "templates" in spec else spec["template"]
    prompt = template.format_map({**spec.get("fragments", {}), **job, "out_path": out_path.relative_to(REPO_ROOT)})
    base_image = job.get("base_image", spec.get("base_image"))
    log_path = out_dir / f"{job['id']}.codex.log"
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            # Prompt before -i: -i takes multiple files and would swallow it.
            ["codex", "exec", "-C", str(REPO_ROOT), "-s", "workspace-write", prompt,
             "-i", str(REPO_ROOT / base_image)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=900,
        )
    seconds = round(time.time() - started, 1)
    ok = proc.returncode == 0 and out_path.exists()
    meta = {
        "id": job["id"], "camera_id": job.get("camera_id", spec.get("camera_id")), "base_image": base_image,
        "intended_state": job["intended_state"], "note": job.get("note", ""),
        "prompt": prompt, "generator": "codex image_gen (built-in, subscription)",
        "seconds": seconds, "exit_code": proc.returncode,
        "label": "unverified", "human_state": None,
    }
    if ok:
        meta_path.write_text(json.dumps(meta, indent=2))
    return {"id": job["id"], "status": "ok" if ok else f"FAILED (exit {proc.returncode}, see {log_path.name})",
            "seconds": seconds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    spec = json.loads((REPO_ROOT / args.spec).read_text())
    for job in spec["jobs"]:
        if args.only and job["id"] not in args.only:
            continue
        print(json.dumps(run_job(spec, job, args.force)), flush=True)


if __name__ == "__main__":
    main()
