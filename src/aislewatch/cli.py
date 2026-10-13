"""`aw`: the command set the Codex correction worker uses (workers/correction_worker.md).

Every command prints JSON. Writes go only to the live store's labels/teacher_labels/corrections tables
and data/reports/. Camera B windows are test material and can never be labelled.

  aw correction show <id>                     aw correction set-status <id> <status> [--report PATH]
  aw correction next                          (oldest correction with status new)
  aw alert show <alert_id>
  aw windows list (--correction ID [--pad 60] | --camera CAM --from T --to T)
  aw captions show --windows ID [ID ...] [--models qwen-current,qwen-base]
  aw teacher label --windows ID [ID ...] [--hint TEXT]
  aw search similar --window ID [--limit 40]
  aw search disagreements [--since-min 60] [--state STATE] [--limit 40]
  aw label accept --window ID --from-teacher [--correction ID]
  aw review enqueue --window ID --reason TEXT [--correction ID]
  aw worker run [--once]                      (polls corrections, runs Codex per correction)

Set AW_FAKE_TEACHER=1 to test the plumbing without API keys (teacher = ground truth, labels marked fake).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import httpx

from aislewatch import evaluate as ev
from aislewatch import preprocess as pp
from aislewatch.live.store import LIVE_DIR, connect, current_model, rows

TEST_CAMERAS = {"cam-B"}
TEACHERS = ["gemini-3.8-flash", "gpt-6.1-sol"]


def out(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def _fmt(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def _time_arg(value: str) -> float:
    from aislewatch.live.mcp_server import _t
    return _t(value, 0)


# ---------- corrections / alerts / windows ---------------------------------------------------

def correction_show(con, cid: str) -> dict:
    c = rows(con, "SELECT * FROM corrections WHERE correction_id = ?", cid)
    if not c:
        sys.exit(f"no correction {cid}")
    c = c[0]
    refs = json.loads(c["refs"])
    span = None
    if "alert_id" in refs:
        a = rows(con, "SELECT * FROM alerts WHERE alert_id = ?", refs["alert_id"])
        if a:
            span = {"camera_id": a[0]["camera_id"], "t0": a[0]["started_at"], "t1": a[0]["detected_at"]}
    if refs.get("camera_id"):
        span = {"camera_id": refs["camera_id"], "t0": _time_arg(refs.get("t0", "")) or None,
                "t1": _time_arg(refs.get("t1", "")) or None}
    return {**c, "refs": refs, "resolved_span": span and {**span, "t0_hms": span["t0"] and _fmt(span["t0"]),
                                                          "t1_hms": span["t1"] and _fmt(span["t1"])}}


def windows_list(con, camera: str, t0: float, t1: float) -> list[dict]:
    return [{**w, "t_start_hms": _fmt(w["t_start"]), "test_camera": w["camera_id"] in TEST_CAMERAS}
            for w in rows(con, "SELECT window_id, camera_id, t_start, t_end FROM windows WHERE camera_id=? AND t_end>=? AND t_start<=? ORDER BY t_start",
                          camera, t0, t1)]


def captions_show(con, ids: list[str], models: list[str]) -> list[dict]:
    live = current_model()
    names = {m: (f"qwen-current ({live})" if m == "qwen-current" else m) for m in models}
    res = []
    for wid in ids:
        caps = {r["model_id"]: r["door_state"] for r in rows(con, "SELECT model_id, door_state FROM captions WHERE window_id=?", wid)}
        res.append({"window_id": wid, **{names[m]: caps.get(live if m == "qwen-current" else m) for m in models}})
    return res


# ---------- teachers -------------------------------------------------------------------------

async def _teach(ids: list[str], hint: str | None) -> list[dict]:
    con = connect()
    fake = os.environ.get("AW_FAKE_TEACHER") == "1"
    cfgs = ev.models_config()
    results = []
    async with httpx.AsyncClient(timeout=180) as client:
        async def one(wid: str, teacher: str):
            w = rows(con, "SELECT * FROM windows WHERE window_id=?", wid)[0]
            if fake:
                pred, err = {"door_state": w["gt_state"]}, None
            else:
                d = LIVE_DIR / w["frames_dir"]
                frames = [(d / f"f{i:02d}.jpg").read_bytes() for i in range(pp.FRAMES_PER_WINDOW)]
                cfg = cfgs[teacher]
                body = ev.build_request(cfg, frames, w["camera_id"], "structured")
                if hint:
                    body["messages"][-1]["content"].append(
                        {"type": "text", "text": f"\nManager note (may help, verify against the frames): {hint}"})
                text, _, _, err = await ev.call(client, cfg, body)
                pred = ev.parse(text)
            con.execute("INSERT OR REPLACE INTO teacher_labels VALUES (?,?,?,?,?,?,?)",
                        (wid, teacher + ("-fake" if fake else ""), pred and pred["door_state"], json.dumps(pred),
                         hint or "", err, time.time()))
            return {"window_id": wid, "teacher": teacher, "door_state": pred and pred["door_state"], "error": err}
        results = await asyncio.gather(*(one(w, t) for w in ids for t in TEACHERS))
    by_win: dict[str, dict] = {}
    for r in results:
        by_win.setdefault(r["window_id"], {})[r["teacher"]] = r["door_state"]
    return [{"window_id": w, "votes": v, "agree": len(set(v.values())) == 1 and None not in v.values()} for w, v in by_win.items()]


def _votes(con, wid: str) -> dict:
    return {r["model_id"]: r["door_state"] for r in rows(con, "SELECT model_id, door_state FROM teacher_labels WHERE window_id=? ORDER BY created_at", wid)}


# ---------- search ---------------------------------------------------------------------------

def search_similar(con, wid: str, limit: int) -> list[dict]:
    """A random sample (across the whole history) of unlabelled windows on non-test cameras where the live
    model said the same state as for this window. If that state was wrong here, some of these are likely
    wrong too, and teacher labels on them become the training examples."""
    model = current_model()
    st = con.execute("SELECT door_state FROM captions WHERE window_id=? AND model_id=?", (wid, model)).fetchone()
    if not st:
        return []
    return rows(con, f"""SELECT w.window_id, w.camera_id, c.door_state FROM windows w JOIN captions c
                        ON c.window_id=w.window_id AND c.model_id=? WHERE c.door_state=? AND w.window_id != ?
                        AND w.camera_id NOT IN ({",".join("?" * len(TEST_CAMERAS))})
                        AND NOT EXISTS (SELECT 1 FROM labels l WHERE l.window_id=w.window_id)
                        ORDER BY random() LIMIT ?""", model, st[0], wid, *TEST_CAMERAS, limit)


def search_disagreements(con, since_min: float, state: str | None, limit: int) -> list[dict]:
    """Windows where the live model and the base model disagree (cheap proxy for likely errors)."""
    cur = current_model()
    t_max = con.execute("SELECT max(t_end) FROM windows").fetchone()[0] or 0
    q = f"""SELECT w.window_id, w.camera_id, a.door_state AS live, b.door_state AS base FROM windows w
            JOIN captions a ON a.window_id=w.window_id AND a.model_id=?
            JOIN captions b ON b.window_id=w.window_id AND b.model_id='qwen-base'
            WHERE a.door_state != b.door_state AND w.t_end >= ?
            AND w.camera_id NOT IN ({",".join("?" * len(TEST_CAMERAS))}) {"AND a.door_state = ?" if state else ""}
            ORDER BY w.t_end DESC LIMIT ?"""
    return rows(con, q, cur, t_max - since_min * 60, *TEST_CAMERAS, *([state] if state else []), limit)


# ---------- labels ---------------------------------------------------------------------------

def _guard(con, wid: str) -> dict:
    w = rows(con, "SELECT * FROM windows WHERE window_id=?", wid)
    if not w:
        sys.exit(json.dumps({"error": f"no window {wid}"}))
    if w[0]["camera_id"] in TEST_CAMERAS:
        sys.exit(json.dumps({"error": f"{wid} is on a test camera; it can never be labelled"}))
    return w[0]


def label_accept(con, wid: str, correction: str | None) -> dict:
    _guard(con, wid)
    votes = _votes(con, wid)
    states = set(votes.values())
    if len(states) != 1 or None in states:
        sys.exit(json.dumps({"error": "teachers do not agree; use aw review enqueue", "votes": votes}))
    state = states.pop()
    rec = json.loads(rows(con, "SELECT json FROM teacher_labels WHERE window_id=? ORDER BY created_at DESC LIMIT 1", wid)[0]["json"])
    fake = any(k.endswith("-fake") for k in votes)
    con.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?,?)",
                (wid, state, json.dumps(rec), "accepted", "teacher_agreed" + ("_fake" if fake else ""), correction,
                 None, json.dumps(votes), time.time()))
    return {"window_id": wid, "door_state": state, "status": "accepted", "votes": votes}


def review_enqueue(con, wid: str, reason: str, correction: str | None) -> dict:
    _guard(con, wid)
    con.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?,?)",
                (wid, None, None, "review", "queued", correction, reason, json.dumps(_votes(con, wid)), time.time()))
    return {"window_id": wid, "status": "review", "reason": reason}


# ---------- worker ---------------------------------------------------------------------------

def worker_run(once: bool) -> None:
    con = connect()
    while True:
        nxt = rows(con, "SELECT correction_id FROM corrections WHERE status='new' ORDER BY created_at LIMIT 1")
        if nxt:
            cid = nxt[0]["correction_id"]
            log = pp.REPO_ROOT / "data" / "reports" / f"{cid}.codex.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"processing {cid}", flush=True)
            with log.open("w") as f:
                subprocess.run(["codex", "exec", "-C", str(pp.REPO_ROOT), "-s", "workspace-write",
                                f"Process correction {cid}. Follow workers/correction_worker.md exactly. "
                                f"Run aw commands as: uv run aw <command>."],
                               stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
            status = rows(con, "SELECT status FROM corrections WHERE correction_id=?", cid)[0]["status"]
            if status in ("new", "processing"):
                con.execute("UPDATE corrections SET status='failed', worker_report=? WHERE correction_id=?",
                            (f"worker exited without finishing; see {log}", cid))
            print(f"{cid}: {rows(con, 'SELECT status FROM corrections WHERE correction_id=?', cid)[0]['status']}", flush=True)
        elif once:
            return
        else:
            time.sleep(30)


# ---------- entry ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(prog="aw")
    sub = ap.add_subparsers(dest="group", required=True)
    c = sub.add_parser("correction").add_subparsers(dest="cmd", required=True)
    s = c.add_parser("show"); s.add_argument("id")
    s = c.add_parser("set-status"); s.add_argument("id"); s.add_argument("status", choices=["new", "processing", "done", "failed"]); s.add_argument("--report")
    c.add_parser("next")
    a = sub.add_parser("alert").add_subparsers(dest="cmd", required=True)
    s = a.add_parser("show"); s.add_argument("id", type=int)
    w = sub.add_parser("windows").add_subparsers(dest="cmd", required=True)
    s = w.add_parser("list"); s.add_argument("--correction"); s.add_argument("--pad", type=float, default=60)
    s.add_argument("--camera"); s.add_argument("--from", dest="t0"); s.add_argument("--to", dest="t1")
    cp = sub.add_parser("captions").add_subparsers(dest="cmd", required=True)
    s = cp.add_parser("show"); s.add_argument("--windows", nargs="+", required=True); s.add_argument("--models", default="qwen-current,qwen-base")
    t = sub.add_parser("teacher").add_subparsers(dest="cmd", required=True)
    s = t.add_parser("label"); s.add_argument("--windows", nargs="+", required=True); s.add_argument("--hint")
    se = sub.add_parser("search").add_subparsers(dest="cmd", required=True)
    s = se.add_parser("similar"); s.add_argument("--window", required=True); s.add_argument("--limit", type=int, default=40)
    s = se.add_parser("disagreements"); s.add_argument("--since-min", type=float, default=60); s.add_argument("--state"); s.add_argument("--limit", type=int, default=40)
    lb = sub.add_parser("label").add_subparsers(dest="cmd", required=True)
    s = lb.add_parser("accept"); s.add_argument("--window", required=True); s.add_argument("--from-teacher", action="store_true", required=True); s.add_argument("--correction")
    rv = sub.add_parser("review").add_subparsers(dest="cmd", required=True)
    s = rv.add_parser("enqueue"); s.add_argument("--window", required=True); s.add_argument("--reason", required=True); s.add_argument("--correction")
    wk = sub.add_parser("worker").add_subparsers(dest="cmd", required=True)
    s = wk.add_parser("run"); s.add_argument("--once", action="store_true")
    args = ap.parse_args()
    con = connect()
    g, cmd = args.group, args.cmd
    if g == "correction" and cmd == "show":
        out(correction_show(con, args.id))
    elif g == "correction" and cmd == "set-status":
        con.execute("UPDATE corrections SET status=?, worker_report=coalesce(?, worker_report) WHERE correction_id=?", (args.status, args.report, args.id))
        out({"correction_id": args.id, "status": args.status})
    elif g == "correction" and cmd == "next":
        out(rows(con, "SELECT correction_id FROM corrections WHERE status='new' ORDER BY created_at LIMIT 1"))
    elif g == "alert":
        out(rows(con, "SELECT * FROM alerts WHERE alert_id=?", args.id))
    elif g == "windows":
        if args.correction:
            span = correction_show(con, args.correction)["resolved_span"]
            if not span or not span["camera_id"]:
                sys.exit(json.dumps({"error": "correction has no camera/time span; pass --camera/--from/--to"}))
            t0 = (span["t0"] or 0) - args.pad
            t1 = (span["t1"] or span["t0"] or time.time()) + args.pad
            out(windows_list(con, span["camera_id"], t0, t1))
        else:
            out(windows_list(con, args.camera, _time_arg(args.t0), _time_arg(args.t1)))
    elif g == "captions":
        out(captions_show(con, args.windows, args.models.split(",")))
    elif g == "teacher":  # teachers may look at test-camera windows (evidence); labels never come from them
        out(asyncio.run(_teach(args.windows, args.hint)))
    elif g == "search" and cmd == "similar":
        out(search_similar(con, args.window, args.limit))
    elif g == "search":
        out(search_disagreements(con, args.since_min, args.state, args.limit))
    elif g == "label":
        out(label_accept(con, args.window, args.correction))
    elif g == "review":
        out(review_enqueue(con, args.window, args.reason, args.correction))
    elif g == "worker":
        worker_run(args.once)


if __name__ == "__main__":
    main()
