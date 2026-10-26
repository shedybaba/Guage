"""Tests for counting objects and counting line crossings, on synthetic belts with known answers."""
import numpy as np
import pytest

from app.sam import TrackPoint
from app.tools.counting import count_objects, line_crossings

FPS = 30.0
SIZE = (720, 1280)  # width, height of the picture


def times(n_frames):
    return [f / FPS for f in range(n_frames)]


def tp(frame, oid, x, y=600.0, conf=0.9, size=60.0):
    return TrackPoint(frame, str(oid), x - size / 2, y - size / 2, x + size / 2, y + size / 2, conf, float(x), float(y),
                      size * size)


def part(oid, start_s, speed=200.0, x0=-30.0, n_frames=300, y=600.0, conf=0.9, seconds=None, noise=0.0, seed=0):
    """A part riding a belt: enters at x0 at start_s and moves at `speed` px/s until it leaves the picture."""
    rng = np.random.default_rng(seed)
    pts = []
    for f in range(n_frames):
        t = f / FPS - start_s
        if t < 0:
            continue
        x = x0 + speed * t
        if not -45 <= x <= SIZE[0] + 45 or (seconds is not None and t > seconds):
            if x > SIZE[0] + 45:
                break
            continue
        pts.append(tp(f, oid, x + rng.normal(0, noise), y + rng.normal(0, noise), conf))
    return pts


def belt(starts, **kw):
    pts = []
    for i, s in enumerate(starts):
        pts += part(i, s, **kw)
    return pts


def group(points):
    tracks = {}
    for p in points:
        tracks.setdefault(p.object_id, []).append(p)
    return tracks


# ---------------- count_objects ----------------

def test_result_shape():
    r = count_objects(group(belt([0.0, 2.0])), times(300), SIZE)
    for key in ("value", "uncertainty", "units", "method", "fit_quality", "series_for_chart", "warnings", "extra"):
        assert key in r
    assert r["units"] == "objects"


def test_counts_parts_on_a_belt():
    r = count_objects(group(belt([0.0, 2.0, 4.0, 6.0, 8.0])), times(450), SIZE)
    assert r["value"] == 5 and r["uncertainty"] == 0
    assert r["extra"]["max_simultaneous"] == 2


def test_static_objects_are_all_counted():
    pts = [tp(f, i, 100 + 90 * i, 500) for i in range(7) for f in range(300)]
    r = count_objects(group(pts), times(300), SIZE)
    assert r["value"] == 7 and r["extra"]["max_simultaneous"] == 7


def test_a_ghost_detection_is_ignored_but_reported():
    pts = belt([0.0, 2.0, 4.0]) + [tp(f, "ghost", 300, 200) for f in range(100, 103)]  # 3 frames only
    r = count_objects(group(pts), times(400), SIZE)
    assert r["value"] == 3
    assert r["extra"]["ignored_short_tracks"] == ["ghost"]


def test_a_borderline_short_track_widens_the_range():
    pts = belt([0.0, 3.0]) + [tp(f, "blip", 300, 200) for f in range(120, 132)]  # 0.4 s: counted, but unsure
    r = count_objects(group(pts), times(400), SIZE)
    assert r["value"] == 3 and r["uncertainty"] > 0
    assert r["extra"]["range"] == [2, 3]


def test_a_low_confidence_track_widens_the_range():
    pts = belt([0.0, 3.0]) + [tp(f, "weak", 300, 200, conf=0.3) for f in range(100, 160)]
    r = count_objects(group(pts), times(400), SIZE)
    assert r["value"] == 3 and r["extra"]["range"] == [2, 3]


def test_an_id_change_does_not_double_count():
    whole = part(0, 0.0)
    split = [p if p.frame < 60 else TrackPoint(**{**p.__dict__, "object_id": "9"}) for p in whole]
    r = count_objects(group(split + part(1, 3.0)), times(450), SIZE)
    assert r["value"] == 2


def test_the_count_is_never_below_the_most_objects_seen_together():
    pts = belt([0.0, 0.5, 1.0, 1.5])  # four parts overlapping on the belt
    r = count_objects(group(pts), times(300), SIZE)
    assert r["value"] >= r["extra"]["max_simultaneous"] == 4


def test_leaving_and_coming_back_in_the_same_place_may_be_one_object():
    a = [tp(f, "a", 300, 600) for f in range(0, 60)]
    b = [tp(f, "b", 305, 600) for f in range(105, 200)]  # a new id 1.5 s later, right where "a" stopped
    r = count_objects(group(a + b), times(300), SIZE)
    assert r["value"] == 2 and r["extra"]["range"] == [1, 2]
    assert any("same place" in w for w in r["warnings"])


def test_nothing_tracked_gives_zero_with_a_warning():
    r = count_objects({}, times(100), SIZE)
    assert r["value"] == 0 and r["warnings"]


def test_the_warning_says_the_count_depends_on_the_tracking():
    r = count_objects(group(belt([0.0])), times(300), SIZE)
    assert any("tracking" in w.lower() for w in r["warnings"])


# ---------------- line_crossings ----------------

def cross_time(start_s, speed=200.0, x0=-30.0, line=360.0):
    return start_s + (line - x0) / speed


def test_counts_each_part_that_crosses_the_centre_line():
    r = line_crossings(group(belt([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], n_frames=600)), times(600), SIZE)
    assert r["value"] == 6
    assert r["extra"]["by_direction"] == {"left_to_right": 6, "right_to_left": 0}
    assert r["extra"]["objects_crossed"] == 6
    assert r["units"] == "crossings"


def test_crossing_times_are_interpolated_between_frames():
    r = line_crossings(group(belt([0.0, 2.0, 4.0])), times(450), SIZE)
    got = [c["t"] for c in r["extra"]["crossings"]]
    for g, s in zip(got, (0.0, 2.0, 4.0)):
        assert g == pytest.approx(cross_time(s), abs=0.01)


def test_rate_per_minute_from_the_spacing():
    r = line_crossings(group(belt([i * 2.0 for i in range(8)], n_frames=600)), times(600), SIZE)  # one part every 2 s
    rate = r["extra"]["rate_per_minute"]
    assert rate["value"] == pytest.approx(30.0, abs=1.0)
    assert 0 < rate["uncertainty"] < 3
    assert rate["units"] == "per minute"


def test_few_crossings_use_the_whole_clip_for_the_rate():
    r = line_crossings(group(belt([0.0, 2.0])), times(300), SIZE)  # 10 s clip, 2 crossings
    rate = r["extra"]["rate_per_minute"]
    assert rate["value"] == pytest.approx(12.0, abs=0.5) and "whole clip" in rate["basis"]
    assert rate["uncertainty"] > 5  # two events say little about a rate


def test_irregular_spacing_gives_a_wider_rate_uncertainty():
    steady = line_crossings(group(belt([i * 2.0 for i in range(8)], n_frames=600)), times(600), SIZE)
    irregular = line_crossings(group(belt([0.0, 0.7, 3.5, 4.0, 8.0, 9.5, 12.0, 14.0], n_frames=600)), times(600), SIZE)
    assert irregular["extra"]["rate_per_minute"]["uncertainty"] > steady["extra"]["rate_per_minute"]["uncertainty"]


def test_directions_are_counted_separately():
    pts = (belt([0.0, 3.0, 6.0], n_frames=600) + part(10, 1.0, speed=-200.0, x0=SIZE[0] + 30, n_frames=600)
           + part(11, 4.0, speed=-200.0, x0=SIZE[0] + 30, n_frames=600))
    r = line_crossings(group(pts), times(600), SIZE)
    assert r["value"] == 5
    assert r["extra"]["by_direction"] == {"left_to_right": 3, "right_to_left": 2}
    assert r["extra"]["net"] == 1


def test_jitter_at_the_line_is_not_counted_as_crossings():
    rng = np.random.default_rng(3)
    hover = [tp(f, 0, 360 + rng.normal(0, 4), 600) for f in range(0, 60)]  # sits on the line, jittering
    rest = [tp(f, 0, 360 + 4 * (f - 59) * 3, 600) for f in range(60, 120)]  # then really leaves to the right
    r = line_crossings(group(hover + rest), times(150), SIZE)
    assert r["value"] <= 1
    assert r["extra"]["jitter_sensitive"] >= 1  # the raw sign changes were many; the dead band ignored them
    assert r["uncertainty"] > 0


def test_a_part_lost_while_crossing_is_still_counted_once():
    pts = [p for p in part(0, 0.0) if not 40 <= p.frame <= 55]  # hidden for 16 frames around the line
    r = line_crossings(group(pts), times(300), SIZE)
    assert r["value"] == 1


def test_horizontal_line():
    pts = [tp(f, 0, 300, 100 + 12 * f) for f in range(80)]  # falls down the picture
    r = line_crossings(group(pts), times(100), SIZE, line={"orientation": "horizontal", "position": 600})
    assert r["value"] == 1 and r["extra"]["by_direction"]["top_to_bottom"] == 1


def test_a_line_through_two_points():
    pts = [tp(f, 0, 100 + 8 * f, 100 + 8 * f) for f in range(60)]  # moves along the diagonal
    r = line_crossings(group(pts), times(100), SIZE, line={"points": [(0, 600), (720, 0)]})  # crosses the path
    assert r["value"] == 1


def test_a_swinging_object_crosses_twice_per_period():
    T = 1.0
    pts = [tp(f, 0, 360 + 150 * np.sin(2 * np.pi * (f / FPS) / T)) for f in range(300)]  # 10 s
    r = line_crossings(group(pts), times(300), SIZE)
    assert r["value"] in (19, 20)  # 2 per period; the one at t=0 sits on the line
    assert r["extra"]["objects_crossed"] == 1  # one object, many crossings
    assert r["extra"]["rate_per_minute"]["value"] == pytest.approx(120.0, rel=0.03)
    dirs = [c["direction"] for c in r["extra"]["crossings"]]
    assert all(a != b for a, b in zip(dirs, dirs[1:]))  # alternates left-to-right, right-to-left


def test_default_line_is_the_vertical_centre_of_the_picture():
    pts = [tp(f, 0, 100 + 10 * f) for f in range(60)]  # crosses x=360 at about frame 26
    assert line_crossings(group(pts), times(100), SIZE)["value"] == 1
    assert line_crossings(group(pts), times(100), (1000, 1280))["value"] == 1  # centre 500: still crossed
    assert line_crossings(group(pts), times(100), (1600, 1280))["value"] == 0  # centre 800: never reached


def test_no_crossing_warns_when_the_line_is_outside_the_path():
    pts = [tp(f, 0, 100 + 2 * f) for f in range(100)]  # stays at x 100..300
    r = line_crossings(group(pts), times(100), SIZE)
    assert r["value"] == 0
    assert any("outside" in w.lower() for w in r["warnings"])


def test_no_objects_gives_zero_and_a_warning():
    r = line_crossings({}, times(100), SIZE)
    assert r["value"] == 0 and r["warnings"]


def test_a_moving_camera_dragging_still_objects_across_the_line_is_flagged():
    # seven objects that never move, but the whole picture pans right by 8 px per frame
    pts = [tp(f, i, 40 + 50 * i + 8 * f, 500) for i in range(7) for f in range(60)]
    r = line_crossings(group(pts), times(100), SIZE)
    assert r["value"] > 0  # the pan drags them over the line ...
    assert r["extra"]["camera_motion_suspected"] is True  # ... and the tool says so
    assert any("camera" in w.lower() for w in r["warnings"])
    quiet = line_crossings(group(pts), times(100), SIZE, check_camera=False)
    assert quiet["extra"]["camera_motion_suspected"] is False
