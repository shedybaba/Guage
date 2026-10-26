"""Regression test on a REAL clip (Newton's cradle, handheld camera).

Runs only on a machine that has the clip in samples/ and the cached SAM tracking in data/
(neither is committed: videos are never pushed). Everything it checks was measured on that clip:
the impacts alternate between the two end balls, the camera lurches in the last third of the clip,
and the answer must stay consistent with momentum being conserved.
"""
from pathlib import Path

import pytest

from app import sam, signals, video
from app.quality import _group_by_object, merge_fragments
from app.tools.collision import collision_momentum

CLIP = Path(__file__).resolve().parents[2] / "samples" / "newtons_cradle.mp4"


@pytest.fixture(scope="module")
def cradle():
    if not CLIP.exists():
        pytest.skip("cradle clip not on this machine")
    cached = sam.load_cached(sam.video_key(CLIP), "steel ball")
    if cached is None:
        pytest.skip("no cached SAM tracking for the cradle clip (run scripts/multi_demo.py once)")
    info = video.probe(CLIP)
    tracks = _group_by_object(merge_fragments(cached.points))
    t, pos, sizes = signals.positions_from_tracks(tracks, info.timestamps)
    return collision_momentum(t, pos, sizes=sizes)


def test_finds_a_consistent_answer(cradle):
    assert cradle["value"] is not None
    assert abs(cradle["value"]) < 15 and 2 < cradle["uncertainty"] < 15  # conserved within an honest error


def test_uses_the_clean_impacts_and_rejects_the_camera_lurch(cradle):
    ev = cradle["extra"]["events"]
    assert cradle["extra"]["n_valid"] >= 6
    late_valid = [e for e in ev if e["frame"] > 315 and e["valid"]]
    assert late_valid == []  # the zoom/rotation stretch must not leak into the answer
    assert any("unstable scene" in r for e in ev for r in e["reasons"])


def test_alternating_end_balls_carry_the_momentum(cradle):
    movers = {obj for e in cradle["extra"]["events"] if e["valid"] for obj, _ in e["before_objects"][:1]}
    assert movers == {"1", "4"}  # the two end balls swap the impact back and forth


def test_says_what_it_assumed(cradle):
    text = " ".join(cradle["warnings"]).lower()
    assert "equal masses" in text and "camera moved" in text and "slow motion" in text
