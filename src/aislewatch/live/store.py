"""SQLite store for the live system: windows, captions, alerts, sensor readings, corrections."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from aislewatch.preprocess import REPO_ROOT

LIVE_DIR = REPO_ROOT / "data" / "live"
DB_PATH = LIVE_DIR / "aislewatch.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
  window_id TEXT PRIMARY KEY, camera_id TEXT, aisle TEXT, t_start REAL, t_end REAL,
  frames_dir TEXT, gt_state TEXT, source TEXT);
CREATE INDEX IF NOT EXISTS windows_cam_t ON windows(camera_id, t_end);
CREATE TABLE IF NOT EXISTS captions (
  window_id TEXT, model_id TEXT, door_state TEXT, json TEXT, caption TEXT, latency_ms INTEGER,
  prompt_tokens INTEGER, completion_tokens INTEGER, error TEXT, created_at REAL,
  PRIMARY KEY (window_id, model_id));
CREATE TABLE IF NOT EXISTS alerts (
  alert_id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, camera_id TEXT, aisle TEXT, model_id TEXT,
  started_at REAL, detected_at REAL, window_ids TEXT, detail TEXT, shadow INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS sensor (aisle TEXT, t REAL, temp_c REAL, PRIMARY KEY (aisle, t));
CREATE TABLE IF NOT EXISTS teacher_labels (
  window_id TEXT, model_id TEXT, door_state TEXT, json TEXT, hint TEXT, error TEXT, created_at REAL,
  PRIMARY KEY (window_id, model_id, hint));
CREATE TABLE IF NOT EXISTS labels (
  window_id TEXT PRIMARY KEY, door_state TEXT, json TEXT, status TEXT, source TEXT, correction_id TEXT,
  reason TEXT, teacher_votes TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS corrections (
  correction_id TEXT PRIMARY KEY, created_at TEXT, source TEXT, refs TEXT, manager_text TEXT,
  hermes_claim TEXT, corrected_fact TEXT, is_rule INTEGER, status TEXT, worker_report TEXT);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def rows(con: sqlite3.Connection, sql: str, *args) -> list[dict]:
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def current_model() -> str:
    p = REPO_ROOT / "models" / "current.json"
    return json.loads(p.read_text())["model"] if p.exists() else "qwen-base"
