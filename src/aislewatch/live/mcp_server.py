"""MCP server for Hermes: read the live store and send footage to the frontier VLM (SPEC §6.2).

Times are site-local and accept "14:05", "14:05:30" or ISO strings; every result carries window IDs.
Register with Hermes:  hermes mcp add aislewatch --command uv --args run --project <repo> aw-mcp
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer

from aislewatch import evaluate as ev
from aislewatch import preprocess as pp
from aislewatch.live.store import LIVE_DIR, connect, current_model, rows
from aislewatch.live.workers import site_clock_start

mcp = MCPServer("aislewatch")
INSPECT_MODEL = "gemini-3.8-flash"
MAX_INSPECT_FRAMES = 30


def _t(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        pass
    if "T" in value or "-" in value:
        return datetime.fromisoformat(value).timestamp()
    parts = [int(p) for p in value.split(":")] + [0, 0]
    lt = time.localtime(site_clock_start())  # times like "14:05" are on the site's current day
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, parts[0], parts[1], parts[2], 0, 0, -1))


def _fmt(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


@mcp.tool()
def get_alerts(since: str = "", include_shadow: bool = False) -> list[dict]:
    """Alerts since a time: containment breaches (from the live captioning model) and temperature excursions (inlet sensors).
    include_shadow=True also returns breaches another model would have raised."""
    con = connect()
    out = []
    live = current_model()
    for a in rows(con, "SELECT * FROM alerts WHERE detected_at >= ? ORDER BY detected_at", _t(since, 0)):
        if a["model_id"] not in (None, live) and not include_shadow:
            continue  # breach alerts from a model that is not the live captioner
        out.append({"alert_id": a["alert_id"], "type": a["type"], "camera_id": a["camera_id"], "aisle": a["aisle"],
                    "started_at": _fmt(a["started_at"]), "detected_at": _fmt(a["detected_at"]), "detail": a["detail"],
                    "model": a["model_id"], "window_ids": json.loads(a["window_ids"])[-6:]})
    return out


@mcp.tool()
def get_timeline(camera: str, t0: str = "", t1: str = "") -> list[dict]:
    """Door-state segments for a camera from the live captioning model, merged across consecutive windows."""
    con = connect()
    model = current_model()
    ws = rows(con, """SELECT w.window_id, w.t_start, w.t_end, c.door_state, c.caption FROM windows w
                      LEFT JOIN captions c ON c.window_id = w.window_id AND c.model_id = ?
                      WHERE w.camera_id = ? AND w.t_end >= ? AND w.t_start <= ? ORDER BY w.t_start""",
              model, camera, _t(t0, 0), _t(t1, 1e12))
    segs: list[dict] = []
    for w in ws:
        state = w["door_state"] or "uncaptioned"
        if segs and segs[-1]["door_state"] == state:
            segs[-1].update(to=_fmt(w["t_end"]), last_window=w["window_id"])
            segs[-1]["windows"] += 1
        else:
            segs.append({"door_state": state, "from": _fmt(w["t_start"]), "to": _fmt(w["t_end"]), "windows": 1,
                         "first_window": w["window_id"], "last_window": w["window_id"], "example_caption": w["caption"]})
    return segs


@mcp.tool()
def search_captions(query: str, camera: str = "", t0: str = "", t1: str = "", limit: int = 20) -> list[dict]:
    """Search caption text and door states (simple substring match, any word)."""
    con = connect()
    words = [w.lower() for w in query.split() if len(w) > 2] or [query.lower()]
    cond = " OR ".join(["lower(c.caption) LIKE ? OR c.door_state = ?"] * len(words))
    args = [x for w in words for x in (f"%{w}%", w)]
    sql = f"""SELECT w.window_id, w.camera_id, w.t_start, c.door_state, c.caption FROM windows w
              JOIN captions c ON c.window_id = w.window_id AND c.model_id = ?
              WHERE ({cond}) AND w.t_end >= ? AND w.t_start <= ? {"AND w.camera_id = ?" if camera else ""}
              ORDER BY w.t_start DESC LIMIT ?"""
    params = [current_model(), *args, _t(t0, 0), _t(t1, 1e12)] + ([camera] if camera else []) + [limit]
    return [{**r, "t_start": _fmt(r["t_start"])} for r in rows(con, sql, *params)]


@mcp.tool()
def get_sensor(aisle: str, t0: str = "", t1: str = "") -> dict:
    """Inlet temperature for an aisle: summary, peak and recent readings."""
    con = connect()
    rs = rows(con, "SELECT t, temp_c FROM sensor WHERE aisle = ? AND t >= ? AND t <= ? ORDER BY t", aisle, _t(t0, 0), _t(t1, 1e12))
    if not rs:
        return {"aisle": aisle, "readings": 0}
    peak = max(rs, key=lambda r: r["temp_c"])
    return {"aisle": aisle, "readings": len(rs), "first": f"{rs[0]['temp_c']} C at {_fmt(rs[0]['t'])}",
            "latest": f"{rs[-1]['temp_c']} C at {_fmt(rs[-1]['t'])}", "peak": f"{peak['temp_c']} C at {_fmt(peak['t'])}",
            "recent": [f"{_fmt(r['t'])} {r['temp_c']} C" for r in rs[-12:]]}


@mcp.tool()
async def inspect_clip(camera: str, t0: str, t1: str, question: str) -> dict:
    """Send the original frames for a camera and time span (max ~60 s) to the frontier VLM with a question."""
    con = connect()
    ws = rows(con, "SELECT window_id, t_start, frames_dir FROM windows WHERE camera_id = ? AND t_end >= ? AND t_start <= ? ORDER BY t_start",
              camera, _t(t0, 0), _t(t1, 1e12))
    frames = [(w["t_start"] + i * pp.FRAME_SPACING_S, LIVE_DIR / w["frames_dir"] / f"f{i:02d}.jpg", w["window_id"])
              for w in ws for i in range(pp.FRAMES_PER_WINDOW)]
    if not frames:
        return {"error": "no footage in that span"}
    step = max(1, len(frames) // MAX_INSPECT_FRAMES)
    frames = frames[::step][:MAX_INSPECT_FRAMES]
    content = []
    for t, path, _ in frames:
        content += [{"type": "text", "text": f"[{_fmt(t)}]"},
                    {"type": "image_url", "image_url": {"url": pp.to_data_url(path.read_bytes())}}]
    content.append({"type": "text", "text": (f"These are CCTV frames from camera {camera} in a data centre, in time order, "
                                             f"each labelled with its time. The camera views a cold-aisle containment "
                                             f"entrance with sliding doors. Question: {question}\nAnswer factually, cite "
                                             f"the frame times you rely on, and say if something cannot be seen.")})
    cfg = ev.models_config()[INSPECT_MODEL]
    body = {"model": cfg["model"], "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 800}
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            text, latency, usage, err = await ev.call(client, cfg, body)
    except KeyError as e:
        return {"error": f"missing API key {e}; set it in .env"}
    return {"camera": camera, "from": _fmt(frames[0][0]), "to": _fmt(frames[-1][0]), "frames_sent": len(frames),
            "model": INSPECT_MODEL, "answer": text, "error": err, "window_ids": sorted({w for _, _, w in frames})}


SITE_RULES = Path.home() / ".hermes" / "skills" / "aislewatch" / "site-rules" / "SKILL.md"


def save_site_rule(rule: str) -> bool:
    """Append a rule to the Hermes site-rules skill (created on first use). Returns False if already present."""
    rule = " ".join(rule.split())
    lines = SITE_RULES.read_text().splitlines() if SITE_RULES.exists() else [
        "---", "name: site-rules",
        "description: Site rules stated by the manager. Check these before concluding whether anything is a violation or needs an alert.",
        "---", "", "# Site rules", "", "Stated by the site manager; apply them in every investigation.", "", "## Rules", ""]
    if any(rule.lower() == ln[2:].lower() for ln in lines if ln.startswith("- ")):
        return False
    SITE_RULES.parent.mkdir(parents=True, exist_ok=True)
    SITE_RULES.write_text("\n".join(lines + [f"- {rule}"]) + "\n")
    return True


@mcp.tool()
def record_correction(manager_text: str, corrected_fact: str, hermes_claim: str = "", alert_id: int | None = None,
                      camera: str = "", t0: str = "", t1: str = "", rule: str = "") -> dict:
    """Record a manager correction for the training loop (processed by the Codex correction worker).
    If the manager states a site rule, pass it as `rule` (one plain sentence with any threshold); it is saved to
    the site-rules skill and applies to investigations from now on."""
    con = connect()
    cid = f"corr-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    refs = {k: v for k, v in {"alert_id": alert_id, "camera_id": camera, "t0": t0, "t1": t1}.items() if v}
    con.execute("INSERT INTO corrections VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cid, datetime.now().isoformat(timespec="seconds"), "manager", json.dumps(refs), manager_text,
                 hermes_claim, corrected_fact, int(bool(rule)), "new", None))
    result = {"correction_id": cid, "status": "new", "refs": refs}
    if rule:
        added = save_site_rule(rule)
        result["site_rule"] = {"rule": rule, "saved_to": "site-rules skill", "new": added,
                               "applies": "to your investigations from now on; the captioning model learns from the "
                                          "correction at its next training run"}
    return result


@mcp.tool()
def loop_status() -> dict:
    """Training-loop status: current adapter, corrections and their state."""
    con = connect()
    counts = {r["status"]: r["n"] for r in rows(con, "SELECT status, count(*) n FROM corrections GROUP BY status")}
    return {"current_model": current_model(), "corrections": counts,
            "latest": rows(con, "SELECT correction_id, status, corrected_fact FROM corrections ORDER BY created_at DESC LIMIT 5")}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
