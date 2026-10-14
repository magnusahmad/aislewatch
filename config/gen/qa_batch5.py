"""Record Claude's QA for batch5 images: suggested_state + qa_note in each sidecar, redo list in data/gen/batch5-redo.json.

Usage: python3 config/gen/qa_batch5.py '<json: {"<sidecar path>": ["<suggested_state>", "<note>"] | ["REDO", "<reason>"]}>'
"""
import json, sys
from pathlib import Path

REDO = Path("data/gen/batch5-redo.json")
redo = json.loads(REDO.read_text()) if REDO.exists() else {}
for path, (state, note) in json.loads(sys.argv[1]).items():
    m = json.loads(Path(path).read_text())
    if state == "REDO":
        redo[path] = note
        m["qa_note"] = f"REDO queued: {note}"
        m["suggested_state"] = m["intended_state"]
    else:
        redo.pop(path, None)
        m["suggested_state"], m["qa_note"] = state, note
    Path(path).write_text(json.dumps(m, indent=2))
REDO.write_text(json.dumps(redo, indent=2))
print(f"recorded {len(json.loads(sys.argv[1]))}; redo queue: {len(redo)}")
