"""Tests for the quality checks and the retry loop, using a scripted fake tracker.

No network and no money: the fake tracker returns whatever each scenario needs,
so every retry path (other phrase, box re-acquire, merge, give up) is exercised.
"""
import asyncio

import numpy as np

from app.quality import find_issues, merge_fragments, track_with_checks
from app.sam import TrackPoint, TrackResult

N = 200


def pt(frame, obj="0", conf=0.95, x=None):
    x = 300 + 100 * np.sin(frame / 8.0) if x is None else x
    return TrackPoint(frame, obj, x - 30, 400, x + 30, 460, conf, x, 430.0, 2800.0)


def good(frames=range(N), obj="0", conf=0.95):
    return [pt(f, obj, conf) for f in frames]


def result(points, phrase="x", billed=None):
    return TrackResult(phrase, points, billed if billed is not None else len(points), [])


class FakeTracker:
    """phrase_results: {phrase: points}. box_fn(frame, box, direction) -> points."""

    def __init__(self, phrase_results, box_fn=None):
        self.phrase_results = phrase_results
        self.box_fn = box_fn
        self.phrase_calls, self.box_calls = [], []

    async def track_phrase(self, phrase):
        self.phrase_calls.append(phrase)
        return result(self.phrase_results.get(phrase, []), phrase)

    async def track_box(self, frame, box, direction="forward"):
        self.box_calls.append((frame, box, direction))
        pts = self.box_fn(frame, box, direction) if self.box_fn else []
        return result(pts, "box")


def run(tracker, phrases, total=N):
    return asyncio.run(track_with_checks(tracker, phrases, total))


# ---------- find_issues ----------

def test_clean_track_has_no_issues():
    assert find_issues(good(), N) == []


def test_small_gaps_are_tolerated():
    pts = [p for p in good() if p.frame not in (50, 51)]  # two frames lost
    assert find_issues(pts, N) == []


def test_finds_a_gap_with_its_range():
    pts = [p for p in good() if not 80 <= p.frame <= 99]
    issues = find_issues(pts, N)
    assert [(i["kind"], i["start"], i["end"]) for i in issues] == [("gap", 80, 99)]


def test_finds_a_leading_and_trailing_gap():
    pts = [p for p in good() if 10 <= p.frame < 180]
    kinds = {(i["kind"], i["start"], i["end"]) for i in find_issues(pts, N)}
    assert kinds == {("gap", 0, 9), ("gap", 180, 199)}


def test_finds_low_confidence_stretch():
    pts = [pt(f, conf=0.2 if 60 <= f < 70 else 0.95) for f in range(N)]
    issues = find_issues(pts, N)
    assert [(i["kind"], i["start"], i["end"]) for i in issues] == [("low_confidence", 60, 69)]


def test_finds_a_jump_to_another_object():
    pts = [pt(f, x=(900 if f == 100 else None)) for f in range(N)]
    issues = find_issues(pts, N)
    assert any(i["kind"] == "jump" and i["start"] in (99, 100) for i in issues)


def test_normal_fast_motion_is_not_a_jump():
    assert not [i for i in find_issues(good(), N) if i["kind"] == "jump"]


# ---------- merge_fragments (identity switches) ----------

def test_merges_a_continuous_identity_switch():
    pts = good(range(0, 100), "0") + good(range(100, 200), "1")  # same object, new id
    merged = merge_fragments(pts)
    assert {p.object_id for p in merged} == {"0"}
    assert len(merged) == N


def test_does_not_merge_a_far_away_second_object():
    other = [pt(f, "1", x=1000.0) for f in range(100, 150)]
    merged = merge_fragments(good(range(0, 100), "0") + other)
    assert {p.object_id for p in merged} == {"0", "1"}


def test_does_not_merge_objects_present_at_the_same_time():
    two = good(range(N), "0") + [pt(f, "1", x=320.0) for f in range(50, 100)]
    assert {p.object_id for p in merge_fragments(two)} == {"0", "1"}


# ---------- the retry loop ----------

def test_clean_track_needs_no_retry():
    t = FakeTracker({"red ball": good()})
    r = run(t, ["red ball", "ball"])
    assert r.status == "ok"
    assert t.phrase_calls == ["red ball"] and t.box_calls == []
    assert any("no gaps" in e.message.lower() or "passed" in e.message.lower() for e in r.log)


def test_zero_matches_tries_the_alternative_phrase():
    t = FakeTracker({"pendulum bob": [], "glass marble": good()})
    r = run(t, ["pendulum bob", "glass marble"])
    assert r.status == "ok"
    assert r.phrase == "glass marble"
    assert t.phrase_calls == ["pendulum bob", "glass marble"]
    assert any("no match" in e.message.lower() for e in r.log)


def test_picks_the_better_phrase_when_the_first_is_poor():
    poor = [pt(f, conf=0.2 if 40 <= f < 120 else 0.95) for f in range(N)]
    t = FakeTracker({"bob": poor, "metal ball": good()})
    r = run(t, ["bob", "metal ball"])
    assert r.phrase == "metal ball" and r.status == "ok"


def test_box_reacquire_fills_a_gap_from_the_last_good_frame():
    pts = [p for p in good() if not 80 <= p.frame <= 119]

    def box_fn(frame, box, direction):
        assert direction == "forward" and frame == 79
        return [pt(f, "0") for f in range(79, N)]

    t = FakeTracker({"ball": pts}, box_fn)
    r = run(t, ["ball"])
    assert r.status == "ok"
    assert len(r.points) == N
    assert t.box_calls[0][0] == 79
    assert any("re-acquir" in e.message.lower() for e in r.log)


def test_box_reacquire_never_overwrites_good_frames():
    pts = [p for p in good() if not 80 <= p.frame <= 119]
    marker = {p.frame: p.cx for p in pts}

    def box_fn(frame, box, direction):
        return [pt(f, "0", x=300 + 100 * np.sin(f / 8.0) + 5.0) for f in range(79, N)]  # shifted by 5 px

    r = run(FakeTracker({"ball": pts}, box_fn), ["ball"])
    for p in r.points:
        if p.frame in marker:
            assert p.cx == marker[p.frame]


def test_box_reacquire_rejects_a_result_that_jumps_away():
    pts = [p for p in good() if not 80 <= p.frame <= 119]
    t = FakeTracker({"ball": pts}, lambda f, b, d: [pt(x, "0", x=950.0) for x in range(79, N)])
    r = run(t, ["ball"])
    assert r.status == "warnings"
    assert any(i["kind"] == "gap" for i in r.issues)
    assert any("rejected" in e.message.lower() for e in r.log)


def test_leading_gap_is_reacquired_backwards():
    pts = [p for p in good() if p.frame >= 30]

    def box_fn(frame, box, direction):
        assert direction == "backward" and frame == 30
        return [pt(f, "0") for f in range(0, 31)]

    r = run(FakeTracker({"ball": pts}, box_fn), ["ball"])
    assert r.status == "ok" and len(r.points) == N


def test_gives_up_honestly_when_nothing_is_found():
    t = FakeTracker({})
    r = run(t, ["a", "b", "c"])
    assert r.status == "failed"
    assert r.points == []
    assert r.advice and any("not found" in e.message.lower() or "no match" in e.message.lower() for e in r.log)


def test_gives_up_gracefully_with_advice_when_gap_cannot_be_fixed():
    pts = [p for p in good() if not 80 <= p.frame <= 119]
    r = run(FakeTracker({"ball": pts}, lambda *a: []), ["ball"])
    assert r.status == "warnings"
    assert r.advice
    assert any(e.step == "give_up" for e in r.log)


def test_retry_budget_is_bounded():
    pts = [p for p in good() if p.frame % 40 not in range(10, 25)]  # many gaps
    t = FakeTracker({"ball": pts}, lambda *a: [])
    run(t, ["ball"])
    assert len(t.box_calls) <= 3


def test_billed_frames_are_summed_across_retries():
    pts = [p for p in good() if not 80 <= p.frame <= 119]
    t = FakeTracker({"ball": pts}, lambda f, b, d: [pt(x, "0") for x in range(79, N)])
    r = run(t, ["ball"])
    assert r.frames_billed == len(pts) + (N - 79)
