"""SQLite store for videos, jobs and their event log. No Redis, no Celery: one file.

A job's progress is written as EVENTS (an append-only log with increasing ids). The live stream replays them from
any point, so a browser that reconnects loses nothing, and the same log is the history shown under
"How Gauge checked its work". JSON columns hold plans, results and needs; they are written through `jsonable` so
numpy values and NaN can never break a write.
"""
import json
import math
import sqlite3
import threading
import time
import uuid
from pathlib import Path

TERMINAL = ("done", "failed", "needs_input")  # a job in one of these states will not change by itself
ACTIVE = ("queued", "planning", "tracking", "measuring", "explaining")


def new_id():
    return uuid.uuid4().hex


def jsonable(obj):
    """Make anything the tools return safe for JSON: numpy -> python, NaN/inf -> None, tuples -> lists."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if np is not None and isinstance(obj, np.generic):
        obj = obj.item()
    if np is not None and isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return jsonable(vars(obj))
    return str(obj)


def _dump(v):
    return None if v is None else json.dumps(jsonable(v), separators=(",", ":"))


def _load(v):
    return None if v is None else json.loads(v)


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY, created REAL NOT NULL, filename TEXT, original TEXT, prepared TEXT, key TEXT,
    file_id TEXT, info TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, video_id TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, question TEXT NOT NULL,
    known_values TEXT, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, message TEXT, plan TEXT, result TEXT,
    needs TEXT, error TEXT, usage TEXT, parent_id TEXT);
CREATE INDEX IF NOT EXISTS jobs_video ON jobs (video_id, created);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, ts REAL NOT NULL, kind TEXT NOT NULL,
    message TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS events_job ON events (job_id, id);
"""

JOB_JSON = ("known_values", "plan", "result", "needs", "usage")
JOB_FIELDS = ("status", "progress", "message", "plan", "result", "needs", "error", "usage")


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True) if self.path != ":memory:" else None
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)

    def close(self):
        with self._lock:
            self._db.close()

    # ---- videos
    def add_video(self, video_id, filename, original, prepared, key, info):
        with self._lock:
            self._db.execute("INSERT INTO videos (id, created, filename, original, prepared, key, info) VALUES (?,?,?,?,?,?,?)",
                             (video_id, time.time(), filename, str(original), str(prepared), key, _dump(info)))

    def get_video(self, video_id):
        with self._lock:
            row = self._db.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["info"] = _load(d["info"])
        return d

    def set_file_id(self, video_id, file_id):
        with self._lock:
            self._db.execute("UPDATE videos SET file_id=? WHERE id=?", (file_id, video_id))

    # ---- jobs
    def create_job(self, video_id, question, known_values=None, parent_id=None):
        job_id, now = new_id(), time.time()
        with self._lock:
            self._db.execute("INSERT INTO jobs (id, video_id, created, updated, question, known_values, status, progress, "
                             "message, parent_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (job_id, video_id, now, now, question, _dump(known_values or {}), "queued", 0.0,
                              "Waiting to start", parent_id))
        return job_id

    def _job(self, row):
        if row is None:
            return None
        d = dict(row)
        for k in JOB_JSON:
            d[k] = _load(d[k])
        return d

    def get_job(self, job_id):
        with self._lock:
            return self._job(self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def update_job(self, job_id, **fields):
        bad = set(fields) - set(JOB_FIELDS)
        if bad:
            raise ValueError(f"cannot update {sorted(bad)}")
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields) + ", updated=?"
        vals = [_dump(v) if k in JOB_JSON else v for k, v in fields.items()] + [time.time(), job_id]
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {cols} WHERE id=?", vals)

    def jobs_for_video(self, video_id):
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs WHERE video_id=? ORDER BY created, rowid", (video_id,)).fetchall()
        return [self._job(r) for r in rows]

    def latest_plan(self, video_id):
        """The plan of the most recent question about this video that had one (for follow-up questions)."""
        for job in reversed(self.jobs_for_video(video_id)):
            if job["plan"] and job["plan"].get("tool"):
                return job["plan"]
        return None

    def count_active(self):
        with self._lock:
            return self._db.execute(f"SELECT COUNT(*) FROM jobs WHERE status IN ({','.join('?' * len(ACTIVE))})", ACTIVE).fetchone()[0]

    def usage_since(self, since):
        """(jobs, SAM frames billed) for jobs created since `since` (a unix time): the basis of the daily cap."""
        with self._lock:
            rows = self._db.execute("SELECT usage FROM jobs WHERE created>=?", (since,)).fetchall()
        usages = [_load(r[0]) or {} for r in rows]
        return len(rows), sum(int(u.get("sam_frames_billed") or 0) for u in usages)

    def mark_stale_failed(self):
        """Jobs left in progress by a restart can never finish: fail them so nobody waits forever. Returns how many."""
        with self._lock:
            rows = self._db.execute(f"SELECT id FROM jobs WHERE status IN ({','.join('?' * len(ACTIVE))})", ACTIVE).fetchall()
        for (job_id,) in rows:
            msg = "Gauge was restarted while this question was being worked on. Please ask it again."
            self.add_event(job_id, "error", msg)
            self.update_job(job_id, status="failed", message=msg, error="restarted")
        return len(rows)

    # ---- events
    def add_event(self, job_id, kind, message="", data=None):
        with self._lock:
            cur = self._db.execute("INSERT INTO events (job_id, ts, kind, message, data) VALUES (?,?,?,?,?)",
                                   (job_id, time.time(), kind, message, _dump(data)))
            return cur.lastrowid

    def events_after(self, job_id, after_id=0, limit=500):
        with self._lock:
            rows = self._db.execute("SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT ?",
                                    (job_id, after_id, limit)).fetchall()
        return [{"id": r["id"], "ts": r["ts"], "kind": r["kind"], "message": r["message"], "data": _load(r["data"])}
                for r in rows]
