"""Live workers: captioner (small VLM on every window), door-state tracker with alerts, and the
temperature simulator. Plain code decides alerts; the model only reports what it sees.
"""

from __future__ import annotations

import asyncio
import json

import random
import time

import httpx

from aislewatch import evaluate as ev
from aislewatch import preprocess as pp
from aislewatch.live.store import LIVE_DIR, connect, current_model, rows

OPEN_STATES = {"propped", "ajar"}


def _rules() -> dict:
    return json.loads((pp.REPO_ROOT / "config" / "rules.json").read_text())


# ---------- captioner ------------------------------------------------------------------------

async def run_captioner(stop: asyncio.Event, base_url: str | None, concurrency: int = 6) -> None:
    con = connect()
    cfgs = ev.models_config()
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120) as client:
        async def caption(win: dict, model_id: str) -> None:
            cfg = dict(cfgs.get(model_id) or {**cfgs["qwen-base"], "model": model_id})
            if base_url and cfg.get("kind") == "vllm":
                cfg["base_url"] = base_url
            d = LIVE_DIR / win["frames_dir"]
            frames = [(d / f"f{i:02d}.jpg").read_bytes() for i in range(pp.FRAMES_PER_WINDOW)]
            async with sem:
                text, latency, usage, err = await ev.call(client, cfg, ev.build_request(cfg, frames, win["camera_id"], "structured"))
            pred = ev.parse(text)
            con.execute("INSERT OR REPLACE INTO captions VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (win["window_id"], model_id, pred["door_state"] if pred else None, json.dumps(pred) if pred else text,
                         (pred or {}).get("caption"), int(latency * 1000), usage.get("prompt_tokens"),
                         usage.get("completion_tokens"), err or (None if pred else "invalid json"), time.time()))

        while not stop.is_set():
            models = ["qwen-base"] + ([current_model()] if current_model() != "qwen-base" else [])
            todo = []
            for m in models:
                todo += [(w, m) for w in rows(con, """SELECT * FROM windows w WHERE NOT EXISTS
                         (SELECT 1 FROM captions c WHERE c.window_id = w.window_id AND c.model_id = ?)
                         ORDER BY t_end DESC LIMIT 24""", m)]
            if todo:
                await asyncio.gather(*(caption(w, m) for w, m in todo))
            else:
                await asyncio.sleep(1)


# ---------- tracker --------------------------------------------------------------------------

def track_once(con, model_id: str, threshold_s: float, shadow: bool) -> None:
    """Recompute open runs per camera from this model's captions; raise one alert per run over threshold."""
    for cam in [r["camera_id"] for r in rows(con, "SELECT DISTINCT camera_id FROM windows")]:
        seq = rows(con, """SELECT w.window_id, w.aisle, w.t_start, w.t_end, c.door_state FROM windows w
                           JOIN captions c ON c.window_id = w.window_id AND c.model_id = ?
                           WHERE w.camera_id = ? ORDER BY w.t_end""", model_id, cam)
        run: list[dict] = []
        for w in seq:
            if w["door_state"] in OPEN_STATES:
                run.append(w)
            else:
                run = []
                continue
            open_s = run[-1]["t_end"] - run[0]["t_start"]
            if open_s >= threshold_s:
                exists = con.execute("SELECT 1 FROM alerts WHERE type='containment_breach' AND camera_id=? AND model_id=? AND started_at=?",
                                     (cam, model_id, run[0]["t_start"])).fetchone()
                if not exists:
                    con.execute("INSERT INTO alerts (type, camera_id, aisle, model_id, started_at, detected_at, window_ids, detail, shadow) VALUES (?,?,?,?,?,?,?,?,?)",
                                ("containment_breach", cam, w["aisle"], model_id, run[0]["t_start"], w["t_end"],
                                 json.dumps([r["window_id"] for r in run]),
                                 f"Containment door {w['door_state']} for {int(open_s)} s", int(shadow)))


async def run_tracker(stop: asyncio.Event) -> None:
    con = connect()
    while not stop.is_set():
        rules = _rules()
        cur = current_model()
        track_once(con, cur, rules["containment_breach_after_s"], shadow=False)
        if cur != "qwen-base":
            track_once(con, "qwen-base", rules["containment_breach_after_s"], shadow=True)
        await asyncio.sleep(2)


# ---------- temperature simulator ------------------------------------------------------------

async def run_sensors(stop: asyncio.Event) -> None:
    """Inlet temperature per aisle, driven by GROUND TRUTH door state (never by model output).
    Recomputed over the whole history each tick (one reading per window), so the curve is reproducible."""
    con = connect()
    while not stop.is_set():
        rules = _rules()
        for aisle in [r["aisle"] for r in rows(con, "SELECT DISTINCT aisle FROM windows")]:
            ws = rows(con, "SELECT t_start, t_end, gt_state FROM windows WHERE aisle=? ORDER BY t_end", aisle)
            temp, open_since, t_prev, above = 0.0, None, ws[0]["t_start"], False
            readings = []
            for w in ws:
                dt_min = (w["t_end"] - t_prev) / 60
                t_prev = w["t_end"]
                if w["gt_state"] in OPEN_STATES:
                    open_since = open_since if open_since is not None else w["t_start"]
                    if w["t_end"] - open_since > rules["temp_delay_s"]:
                        temp = min(rules["temp_max_rise_c"], temp + rules["temp_rise_c_per_min"] * dt_min)
                else:
                    open_since = None
                    temp = max(0.0, temp - rules["temp_decay_c_per_min"] * dt_min)
                noise = random.Random(int(w["t_end"]) ^ hash(aisle) & 0xFFFF).gauss(0, 0.05)
                readings.append((aisle, w["t_end"], round(rules["temp_baseline_c"] + temp + noise, 2)))
                crossed = temp >= rules["temp_alert_rise_c"]
                if crossed and not above:
                    exists = con.execute("SELECT 1 FROM alerts WHERE type='temperature_excursion' AND aisle=? AND started_at=?",
                                         (aisle, w["t_end"])).fetchone()
                    if not exists:
                        con.execute("INSERT INTO alerts (type, camera_id, aisle, model_id, started_at, detected_at, window_ids, detail) VALUES (?,?,?,?,?,?,?,?)",
                                    ("temperature_excursion", None, aisle, None, w["t_end"], w["t_end"], "[]",
                                     f"Inlet temperature +{temp:.1f} C above baseline"))
                above = crossed
            con.executemany("INSERT OR REPLACE INTO sensor VALUES (?,?,?)", readings)
        await asyncio.sleep(2)


def site_clock_start() -> float:
    """Site time starts at the wall-clock time the live run started (written by aw-live), so timestamps
    match the real clock at speed 1. Falls back to now."""
    p = LIVE_DIR / "clock.json"
    return json.loads(p.read_text())["t0"] if p.exists() else time.time()
