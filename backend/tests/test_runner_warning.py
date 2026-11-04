"""A tool that measures one object must say so when several were tracked."""
from types import SimpleNamespace

import numpy as np

from app import runner
from app.sam import TrackPoint


def _track(oid, n=40):
    return [TrackPoint(frame=i, object_id=oid, x1=0, y1=0, x2=10 + i, y2=10 + i, confidence=0.9, cx=5, cy=5, area=100 + 10 * i)
            for i in range(n)]


def test_single_object_tool_warns_when_several_objects_were_tracked(monkeypatch):
    monkeypatch.setattr(runner, "_run_tool", lambda *a: {"value": 1, "warnings": []})
    plan = SimpleNamespace(tool="growth_over_time")
    checked = SimpleNamespace(tracks={0: _track(0), 1: _track(1, 20), 2: _track(2, 10)})
    out = runner.run_tool(plan, checked, None, None)
    assert any("measured only one" in w and "object 0" in w for w in out["warnings"])


def test_no_warning_for_one_object_or_a_multi_object_tool(monkeypatch):
    monkeypatch.setattr(runner, "_run_tool", lambda *a: {"value": 1, "warnings": []})
    one = SimpleNamespace(tracks={0: _track(0)})
    many = SimpleNamespace(tracks={0: _track(0), 1: _track(1)})
    assert runner.run_tool(SimpleNamespace(tool="growth_over_time"), one, None, None)["warnings"] == []
    assert runner.run_tool(SimpleNamespace(tool="count_objects"), many, None, None)["warnings"] == []
