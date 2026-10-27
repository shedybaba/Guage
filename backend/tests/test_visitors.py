"""Per-visitor limits: one visitor is stopped, another is not; the proxy header decides who is who."""
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.visitors import VisitorLimiter, visitor_id


def make_request(forwarded=None, host="10.0.0.1"):
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "headers": headers, "client": (host, 1234), "method": "GET", "path": "/"})


def test_visitor_id_trusts_the_last_forwarded_entry():
    assert visitor_id(make_request("6.6.6.6, 203.0.113.9")) == "203.0.113.9"  # the first entry could be written by the visitor
    assert visitor_id(make_request(None, host="10.0.0.1")) == "10.0.0.1"


def test_limit_stops_the_same_visitor_but_not_another():
    lim = VisitorLimiter()
    a, b = make_request("203.0.113.1"), make_request("203.0.113.2")
    lim.check(a, "questions", 2, "slow down")
    lim.check(a, "questions", 2, "slow down")
    with pytest.raises(HTTPException) as e:
        lim.check(a, "questions", 2, "slow down")
    assert e.value.status_code == 429 and e.value.detail == "slow down"
    lim.check(b, "questions", 2, "slow down")  # someone else is unaffected
    lim.check(a, "uploads", 2, "slow down")  # other kinds of action are counted separately


def test_limit_zero_switches_it_off():
    lim = VisitorLimiter()
    for _ in range(50):
        lim.check(make_request("203.0.113.1"), "questions", 0, "never")


def test_old_actions_are_forgotten_after_an_hour(monkeypatch):
    import app.visitors as v
    lim, now = VisitorLimiter(), [1000.0]
    monkeypatch.setattr(v.time, "time", lambda: now[0])
    r = make_request("203.0.113.1")
    lim.check(r, "questions", 1, "slow down")
    with pytest.raises(HTTPException):
        lim.check(r, "questions", 1, "slow down")
    now[0] += 3601
    lim.check(r, "questions", 1, "slow down")
