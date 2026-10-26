"""Tests for tracking several objects at once (counting, collisions) and the new checks.

Scripted fake tracker again: no network, no cost.
"""
import asyncio

import numpy as np

from app.quality import (find_camera_motion, find_issues, find_object_issues, merge_fragments,
                         track_objects_with_checks)
from app.sam import TrackPoint, TrackResult

N = 200
SIZE = (720, 1280)


def opt(frame, obj, base, conf=0.9, wobble=40.0, y=600.0, shape=None):
    k = int(obj if shape is None else shape)  # `shape` picks the motion pattern; ids can differ from it
    x = base + wobble * np.sin(frame / (6.0 + 3 * k) + k)  # each object moves differently
    return TrackPoint(frame, str(obj), x - 30, y - 30, x + 30, y + 30, conf, x, y, 2800.0)


def obj_track(obj, base, frames=range(N), **kw):
    return [opt(f, obj, base, **kw) for f in frames]  # (pass shape= to give a new id the same motion)


def three(frames=range(N)):
    return obj_track(0, 150, frames) + obj_track(1, 360, frames) + obj_track(2, 570, frames)


def res(points, label="x", billed=None):
    return TrackResult(label, points, len(points) if billed is None else billed, [])


class MultiFake:
    def __init__(self, phrase_results, box_fn=None):
        self.phrase_results, self.box_fn = phrase_results, box_fn
        self.phrase_calls, self.box_calls = [], []

    async def track_phrase(self, phrase):
        self.phrase_calls.append(phrase)
        return res(self.phrase_results.get(phrase, []), phrase)

    async def track_boxes(self, prompts, direction="forward"):
        self.box_calls.append((prompts, direction))
        return res(self.box_fn(prompts, direction) if self.box_fn else [], "boxes")


def run(tracker, phrases, **kw):
    return asyncio.run(track_objects_with_checks(tracker, phrases, N, SIZE, **kw))


def truth_boxes(bases):
    """A box_fn that returns each prompted object's true path from its anchor frame on."""
    def fn(prompts, direction):
        out = []
        for idx, (frame, box) in enumerate(prompts):
            cx = (box[0] + box[2]) / 2
            obj = min(range(len(bases)), key=lambda k: abs(bases[k] - cx))
            out += [replace_id(p, idx) for p in obj_track(obj, bases[obj], range(frame, N))]
        return out
    return fn


def replace_id(p, idx):
    from dataclasses import replace
    return replace(p, object_id=str(idx))


BASES = [150, 360, 570]


# ---------- the jump rule: collisions are not jumps, teleports are ----------

def test_a_collision_bounce_is_not_a_jump():
    # a ball sitting still gets hit: fast out, abrupt reversal, fast back (big steps on both sides)
    xs = [300] * 40 + [300 + 30 * k for k in range(1, 6)] + [450 - 30 * k for k in range(1, 6)] + [300] * 40
    pts = [TrackPoint(f, "0", x - 30, 570, x + 30, 630, 0.9, float(x), 600.0, 2800.0) for f, x in enumerate(xs)]
    assert [i for i in find_issues(pts, len(xs)) if i["kind"] == "jump"] == []


def test_an_isolated_teleport_is_a_jump():
    xs = [300] * 40 + [700] * 40  # jumps to another object and stays there
    pts = [TrackPoint(f, "0", x - 30, 570, x + 30, 630, 0.9, float(x), 600.0, 2800.0) for f, x in enumerate(xs)]
    assert [i["start"] for i in find_issues(pts, len(xs)) if i["kind"] == "jump"] == [39]


# ---------- out of frame vs lost ----------

def test_leaving_and_entering_the_picture_is_not_an_issue():
    leaves = [opt(f, 0, 600 + 2.5 * f, wobble=0) for f in range(0, 60)]  # drifts off the right edge
    assert find_object_issues(leaves, N, SIZE) == []


def test_a_hole_next_to_the_picture_edge_is_out_of_frame():
    left = [opt(f, 0, 30, wobble=0) for f in range(0, 80)] + [opt(f, 0, 30, wobble=0) for f in range(120, N)]
    assert find_object_issues(left, N, SIZE) == []


def test_vanishing_in_the_middle_of_the_picture_is_a_loss():
    gone = [opt(f, 0, 360, wobble=0) for f in range(0, 120)]
    issues = find_object_issues(gone, N, SIZE)
    assert [(i["kind"], i["start"], i["end"]) for i in issues] == [("gap", 120, 199)]


def test_an_interior_hole_in_the_middle_of_the_picture_is_a_loss():
    pts = [opt(f, 0, 360) for f in range(N) if not 80 <= f < 100]
    assert [(i["kind"], i["start"]) for i in find_object_issues(pts, N, SIZE)] == [("gap", 80)]


# ---------- merging with several objects ----------

def test_merges_one_broken_object_among_three():
    pts = (obj_track(0, 150) + obj_track(1, 360, range(0, 100)) + obj_track(7, 360, range(100, N), shape=1)
           + obj_track(2, 570))
    merged = merge_fragments(pts)
    assert len({p.object_id for p in merged}) == 3 and len(merged) == len(pts)


def test_merges_a_chain_of_fragments():
    pts = (obj_track(0, 360, range(0, 60), shape=0) + obj_track(1, 360, range(60, 130), shape=0)
           + obj_track(2, 360, range(130, N), shape=0))
    assert len({p.object_id for p in merge_fragments(pts)}) == 1


def test_does_not_merge_two_objects_that_never_overlap_but_are_far_apart():
    pts = obj_track(0, 150, range(0, 80)) + obj_track(1, 570, range(90, N))  # two parts, one after the other
    assert len({p.object_id for p in merge_fragments(pts)}) == 2


# ---------- the loop ----------

def test_clean_scene_finds_all_objects_without_retries():
    t = MultiFake({"ball": three()})
    r = run(t, ["ball"])
    assert r.status == "ok" and r.count == 3
    assert t.box_calls == [] and t.phrase_calls == ["ball"]


def test_edge_gaps_do_not_trigger_retries():
    # object 2 only appears at frame 90 (entering from the right edge), object 0 exits through the left edge at frame 120
    pts = ([opt(f, 0, 150 - 1.2 * f, wobble=0) for f in range(0, 120)] + obj_track(1, 360)
           + [opt(f, 2, 680 - 2 * (f - 90), wobble=0) for f in range(90, N)])
    t = MultiFake({"ball": pts})
    r = run(t, ["ball"], check_camera=False)
    assert t.box_calls == [] and r.count == 3


def test_two_lost_objects_are_re_acquired_in_one_request():
    pts = [p for p in three() if not (p.object_id == "0" and 60 <= p.frame < 90)
           and not (p.object_id == "2" and 100 <= p.frame < 130)]
    t = MultiFake({"ball": pts}, truth_boxes(BASES))
    r = run(t, ["ball"])
    assert len(t.box_calls) == 1  # batched: cost does not grow with the number of objects
    assert len(t.box_calls[0][0]) == 2
    assert r.status == "ok" and all(len(v) == N for v in r.tracks.values())


def test_only_the_broken_objects_are_prompted():
    pts = [p for p in three() if not (p.object_id == "1" and 60 <= p.frame < 90)]
    t = MultiFake({"ball": pts}, truth_boxes(BASES))
    run(t, ["ball"])
    assert len(t.box_calls[0][0]) == 1 and t.box_calls[0][0][0][0] == 59  # anchored on frame 59


def test_a_re_acquired_object_that_lands_on_another_object_is_rejected():
    pts = [p for p in three() if not (p.object_id == "0" and 60 <= p.frame < 90)]
    # the box request follows the neighbour (object 1) instead of object 0
    t = MultiFake({"ball": pts}, lambda prompts, d: [
        replace_id(p, 0) for p in obj_track(1, 360, range(59, N))])
    r = run(t, ["ball"])
    assert r.status == "warnings"
    assert len(r.tracks["0"]) < N
    assert any("another object" in e.message for e in r.log)


def test_gap_only_does_not_try_another_name():
    pts = [p for p in three() if not (p.object_id == "0" and 60 <= p.frame < 90)]
    t = MultiFake({"ball": pts, "metal ball": three()}, truth_boxes(BASES))
    r = run(t, ["ball", "metal ball"])
    assert t.phrase_calls == ["ball"] and r.status == "ok"


def test_zero_matches_tries_the_other_name():
    t = MultiFake({"part": [], "bolt": three()})
    r = run(t, ["part", "bolt"])
    assert t.phrase_calls == ["part", "bolt"] and r.phrase == "bolt" and r.count == 3


def test_low_confidence_on_one_object_prefers_a_better_name():
    poor = obj_track(0, 150, conf=0.2) + obj_track(1, 360) + obj_track(2, 570)
    t = MultiFake({"ball": poor, "red ball": three()})
    r = run(t, ["ball", "red ball"])
    assert r.phrase == "red ball" and r.status == "ok"


def test_nothing_found_fails_honestly():
    r = run(MultiFake({}), ["a", "b"])
    assert r.status == "failed" and r.count == 0 and r.advice


def test_box_budget_is_bounded():
    pts = [p for p in three() if p.frame % 40 not in range(10, 25)]
    t = MultiFake({"ball": pts}, lambda *a: [])
    run(t, ["ball"])
    assert len(t.box_calls) <= 3


def test_billed_frames_add_up():
    pts = [p for p in three() if not (p.object_id == "0" and 60 <= p.frame < 90)]
    t = MultiFake({"ball": pts}, truth_boxes(BASES))
    r = run(t, ["ball"])
    assert r.frames_billed == len(pts) + (N - 59)


# ---------- camera motion ----------

def shaky(shift):
    """Three objects all moving by `shift` px per frame, together."""
    pts = []
    for f in range(N):
        for i, base in enumerate(BASES):
            x = base + shift * f % 300
            pts.append(TrackPoint(f, str(i), x - 30, 570, x + 30, 630, 0.9, float(x), 600.0, 2800.0))
    return pts


def test_detects_a_moving_camera():
    tracks = {}
    for p in shaky(6):
        tracks.setdefault(p.object_id, []).append(p)
    assert [i["kind"] for i in find_camera_motion(tracks, N)] == ["camera_motion"]


def test_a_still_camera_with_one_moving_object_is_fine():
    tracks = {}
    for p in three():
        tracks.setdefault(p.object_id, []).append(p)
    tracks["0"] = obj_track(0, 150, wobble=120)  # one object swings a lot, the rest stay put
    tracks["1"] = obj_track(1, 360, wobble=0)
    tracks["2"] = obj_track(2, 570, wobble=0)
    assert find_camera_motion(tracks, N) == []


def test_objects_swinging_independently_are_not_a_camera_move():
    tracks = {str(i): obj_track(i, BASES[i], wobble=120) for i in range(3)}
    assert find_camera_motion(tracks, N) == []


def test_camera_check_can_be_switched_off_for_a_conveyor():
    t = MultiFake({"part": shaky(6)})
    on = run(t, ["part"])
    off = run(MultiFake({"part": shaky(6)}), ["part"], check_camera=False)
    assert any(i["kind"] == "camera_motion" for i in on.scene_issues)
    assert off.scene_issues == []


def many(n=7, poor=()):
    """n well-separated objects; those in `poor` have low confidence throughout."""
    pts = []
    for i in range(n):
        pts += obj_track(i, 80 + 90 * i, conf=0.2 if i in poor else 0.9)
    return pts


def test_one_poor_object_among_seven_is_not_a_reason_to_try_another_name():
    t = MultiFake({"ball": many(7, poor={3}), "metal ball": many(7)})
    r = run(t, ["ball", "metal ball"])
    assert t.phrase_calls == ["ball"] and r.count == 7  # one half-hidden ball is a local problem, not a wrong name


def test_a_quarter_of_the_objects_in_trouble_does_justify_another_name():
    t = MultiFake({"ball": many(7, poor={2, 5}), "metal ball": many(7)})
    r = run(t, ["ball", "metal ball"])
    assert t.phrase_calls == ["ball", "metal ball"] and r.phrase == "metal ball"


def test_a_name_that_only_adds_a_poor_extra_object_does_not_win():
    first = many(3, poor={0, 1})  # two of three objects poor: another name is tried
    second = many(3, poor={0, 1}) + obj_track(9, 600, conf=0.2)  # same trouble plus one more poor object: no better
    t = MultiFake({"ball": first, "shiny ball": second})
    r = run(t, ["ball", "shiny ball"])
    assert r.phrase == "ball" and r.count == 3  # the extra object did not buy the win


def test_a_clearly_better_name_still_wins_even_with_the_same_number_of_objects():
    t = MultiFake({"ball": many(4, poor={0, 1}), "red ball": many(4)})
    assert run(t, ["ball", "red ball"]).phrase == "red ball"


def test_the_log_says_what_actually_happened_when_a_name_is_kept():
    t = MultiFake({"ball": many(7, poor={3}), "metal ball": many(7)})
    r = run(t, ["ball", "metal ball"])
    msgs = " ".join(e.message for e in r.log)
    assert "Only 1 of 7 objects have a tracking problem" in msgs and "Only frames are missing" not in msgs
