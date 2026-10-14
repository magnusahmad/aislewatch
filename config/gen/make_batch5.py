"""Write batch5 edit specs: clone proven cam-A still and sequence jobs onto new cameras D-I.

Each new camera gets the same mix as the original cameras (8 stills, 5 five-keyframe sequences),
edited from its approved base in data/gen/bases/cam-<X>_closed.png. Uses the seq-cam-A edit template
(sliding doors stay flat and parallel; generic "door sign" wording that fits every hall).

Usage: python3 config/gen/make_batch5.py D E F G H I   -> config/gen/batch5-cam-<X>.json
"""

import copy
import json
import sys
from pathlib import Path

GEN = Path(__file__).parent
STILLS = ["A-closed-busy", "A-closed-dim", "A-ajar-10", "A-ajar-30", "A-open-full",
          "A-prop-box", "A-prop-cart", "A-prop-wedge"]
SEQS = ["A-seq-pass-in", "A-seq-walk-past-closed", "A-seq-drift-ajar", "A-seq-prop-box-leave", "A-seq-cart-through"]


def cam_a_jobs() -> tuple[dict, dict]:
    stills, seqs = {}, {}
    for f in sorted(GEN.glob("*.json")):
        d = json.loads(f.read_text())
        for j in d.get("jobs", []):
            if j.get("camera_id", d.get("camera_id")) != "cam-A":
                continue
            if "sequence" in j:
                seqs.setdefault(j["sequence"], []).append(j)
            elif j.get("template", "edit") == "edit":
                stills[j["id"]] = j
    return stills, seqs


def main(cams: list[str]) -> None:
    template = json.loads((GEN / "seq-cam-A.json").read_text())["templates"]["edit"]
    stills, seqs = cam_a_jobs()
    for c in cams:
        cam, base = f"cam-{c}", f"data/gen/bases/cam-{c}_closed.png"
        jobs = []
        for sid in STILLS:
            j = {k: v for k, v in copy.deepcopy(stills[sid]).items() if k in ("id", "intended_state", "note", "change", "people")}
            j.update(id=sid.replace("A-", f"{c}-", 1), camera_id=cam, base_image=base, out_dir=f"data/gen/stills/{cam}")
            jobs.append(j)
        for name in SEQS:
            new = name.replace("A-", f"{c}-", 1)
            for j in sorted(seqs[name], key=lambda j: j["order"]):
                j = copy.deepcopy(j)
                j.update(camera_id=cam, sequence=new, out_dir=f"data/gen/seq/{cam}/{new}",
                         base_image=base if j["order"] == 1 else f"data/gen/seq/{cam}/{new}/k{j['order'] - 1}.png")
                j.pop("split", None)  # the dataset builder assigns splits
                jobs.append(j)
        out = GEN / f"batch5-cam-{c}.json"
        out.write_text(json.dumps({"templates": {"edit": template}, "jobs": jobs}, indent=2))
        print(out.name, len(jobs), "jobs")


if __name__ == "__main__":
    main(sys.argv[1:])
