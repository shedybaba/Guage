"""Per-visitor limits: one person cannot use up the shared budget.

A visitor is identified by the address of the machine that connected. Behind Railway's proxy that address is in
the X-Forwarded-For header; the proxy ADDS the real client address at the end, so the last entry is the one
to trust (earlier entries can be written by the visitor). Counts live in memory and reset on a restart, which
is fine for a cost guard: the global limits in jobs.py still apply to everyone together.
"""
import time

from fastapi import HTTPException, Request

from . import config

WINDOW_S = 3600


def visitor_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    last = forwarded.split(",")[-1].strip()
    if last:
        return last
    return request.client.host if request.client else "unknown"


class VisitorLimiter:
    """Counts what each visitor did in the last hour, per kind of action ('questions', 'uploads', 'voice')."""

    def __init__(self):
        self._times = {}

    def check(self, request: Request, kind: str, limit: int, message: str):
        """Raise 429 if this visitor already did `kind` `limit` times in the last hour; otherwise count it."""
        if limit <= 0:  # 0 switches the limit off
            return
        now = time.time()
        key = (kind, visitor_id(request))
        recent = [t for t in self._times.get(key, []) if now - t < WINDOW_S]
        if len(recent) >= limit:
            self._times[key] = recent
            raise HTTPException(429, message)
        recent.append(now)
        self._times[key] = recent
        if len(self._times) > 5000:  # forget visitors who have been quiet for an hour
            self._times = {k: v for k, v in self._times.items() if v and now - v[-1] < WINDOW_S}

