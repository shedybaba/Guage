"""Tests for the rotation tool on simulated markers going round with known speeds."""
import numpy as np
import pytest

from app.tools.rotation import rotation_rate


def marker(rpm, seconds=6.0, fps=30.0, radius=80.0, centre=(320.0, 240.0), noise=0.8, phase=0.7, ccw=True, seed=0,
           speed=None):
    """Positions of a marker going round. `speed(t)` may replace the constant speed: it gives the angle in turns."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    turns = (rpm / 60.0) * t if speed is None else speed(t)
    ang = 2 * np.pi * turns * (1 if ccw else -1) + phase
    x = centre[0] + radius * np.cos(ang) + rng.normal(0, noise, t.size)
    y = centre[1] - radius * np.sin(ang) + rng.normal(0, noise, t.size)  # the image y axis points down
    return t, x, y


def test_recovers_a_known_speed():
    r = rotation_rate(*marker(120.0))
    assert r["units"] == "rpm" and r["value"] == pytest.approx(120.0, rel=0.005)
    assert abs(r["value"] - 120.0) <= 3 * r["uncertainty"] + 0.05
    x = r["extra"]
    assert x["direction"] == "counter-clockwise" and x["frequency"]["value"] == pytest.approx(2.0, rel=0.005)
    assert x["period"]["value"] == pytest.approx(0.5, rel=0.005) and x["revolutions"]["value"] == pytest.approx(12.0, abs=0.2)
    assert x["centre_px"] == pytest.approx([320, 240], abs=1.0) and x["radius_px"] == pytest.approx(80, abs=1.0)
    assert r["warnings"] == []


def test_direction_is_found():
    r = rotation_rate(*marker(75.0, ccw=False, seed=1))
    assert r["extra"]["direction"] == "clockwise" and r["value"] == pytest.approx(75.0, rel=0.01)


@pytest.mark.parametrize("rpm", [6.0, 30.0, 300.0, 540.0])
def test_a_wide_range_of_speeds(rpm):
    seconds = max(6.0, 4 * 60.0 / rpm)
    r = rotation_rate(*marker(rpm, seconds=seconds, seed=2))
    assert r["value"] == pytest.approx(rpm, rel=0.01)


def test_uses_real_timestamps_not_frame_numbers():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 8, 240))
    ang = 2 * np.pi * (90 / 60.0) * t
    r = rotation_rate(t, 200 + 60 * np.cos(ang) + rng.normal(0, 0.5, t.size), 150 - 60 * np.sin(ang) + rng.normal(0, 0.5, t.size))
    assert r["value"] == pytest.approx(90.0, rel=0.01)


def test_frames_lost_do_not_spoil_it():
    t, x, y = marker(150.0, seconds=8.0, seed=4)
    keep = np.random.default_rng(4).random(t.size) > 0.2  # a fifth of the frames missing
    r = rotation_rate(t[keep], x[keep], y[keep])
    assert r["value"] == pytest.approx(150.0, rel=0.01)


def test_the_resting_part_before_the_fan_starts_is_left_out_and_said():
    t = np.arange(0, 10.0, 1 / 30)
    turns = np.where(t < 3.0, 0.0, (t - 3.0) * 2.0)  # at rest for 3 s, then 120 rpm
    r = rotation_rate(*marker(0, seconds=10.0, speed=lambda t: np.where(t < 3.0, 0.0, (t - 3.0) * 2.0), seed=5))
    assert r["value"] == pytest.approx(120.0, rel=0.02)
    assert r["fit_quality"]["segment"]["trimmed"] and r["fit_quality"]["segment"]["start_s"] > 2.5
    assert any("at rest" in w for w in r["warnings"])


def test_a_speed_that_changes_is_reported_as_an_average():
    r = rotation_rate(*marker(0, seconds=10.0, speed=lambda t: 0.5 * t * 2 + 0.1 * t ** 2, seed=6))  # speeds up from 60 rpm
    assert any("changes during the clip" in w for w in r["warnings"])
    x = r["extra"]
    assert x["rpm_second_half"] > x["rpm_first_half"] * 1.3


def test_too_fast_for_the_camera_is_refused_not_guessed():
    # 900 rpm at 30 fps: half a turn between frames, indistinguishable from slower or backward rotation
    r = rotation_rate(*marker(900.0, seconds=4.0, seed=7))
    assert r["value"] is None and "wagon-wheel" in r["warnings"][-1]


def test_the_wagon_wheel_limit_is_real_and_cannot_be_detected():
    """25 turns a second filmed at 30 frames a second looks EXACTLY like 5 turns a second the other way round: the data
    carries no trace of the difference, so no tool can refuse it. This is why the answer states its assumption (less than
    half a turn between frames) and why the README says to film fast things in slow motion."""
    r = rotation_rate(*marker(1500.0, seconds=4.0, seed=8))
    assert r["value"] == pytest.approx(300.0, rel=0.01) and r["extra"]["direction"] == "clockwise"


def test_close_to_the_limit_warns():
    r = rotation_rate(*marker(540.0, seconds=4.0, seed=9))  # about 3.3 frames per turn
    assert r["value"] == pytest.approx(540.0, rel=0.01) and any("frames per revolution" in w for w in r["warnings"])


def test_a_still_marker_is_not_a_rotation():
    rng = np.random.default_rng(10)
    t = np.arange(0, 6.0, 1 / 30)
    r = rotation_rate(t, 100 + rng.normal(0, 0.3, t.size), 100 + rng.normal(0, 0.3, t.size))
    assert r["value"] is None and "hardly moves" in r["warnings"][-1]


def test_a_straight_path_is_not_a_rotation():
    t = np.arange(0, 6.0, 1 / 30)
    r = rotation_rate(t, 50 + 60 * t, 100 + 5 * t + np.random.default_rng(11).normal(0, 0.3, t.size))
    assert r["value"] is None


def test_a_marker_too_close_to_the_axis_is_refused():
    r = rotation_rate(*marker(100.0, radius=1.5, noise=1.0, seed=12))
    assert r["value"] is None


def test_less_than_half_a_turn_is_refused_and_a_couple_of_turns_warns():
    assert rotation_rate(*marker(30.0, seconds=0.9, seed=13))["value"] is None
    r = rotation_rate(*marker(60.0, seconds=1.6, seed=14))
    assert r["value"] == pytest.approx(60.0, rel=0.03) and any("revolutions were seen" in w for w in r["warnings"])


def test_a_path_that_is_not_a_circle_is_refused():
    t = np.arange(0, 6.0, 1 / 30)  # a rolling wheel: circle plus steady drift
    ang = 2 * np.pi * 1.0 * t
    r = rotation_rate(t, 100 + 40 * t + 30 * np.cos(ang), 200 - 30 * np.sin(ang))
    assert r["value"] is None or any("circle" in w for w in r["warnings"])


def test_the_quoted_uncertainty_covers_the_truth():
    hits, runs = 0, 40
    for seed in range(runs):
        rng = np.random.default_rng(300 + seed)
        rpm = rng.uniform(40, 400)
        t, x, y = marker(rpm, seconds=6.0, noise=1.5, phase=rng.uniform(0, 6), radius=rng.uniform(30, 120), seed=seed)
        r = rotation_rate(t, x, y)
        hits += r["value"] is not None and abs(r["value"] - rpm) <= 2 * r["uncertainty"]
    assert hits / runs >= 0.9


def test_chart_series_line_up():
    r = rotation_rate(*marker(120.0, seed=15))
    c = r["series_for_chart"]
    assert len(c["t"]) == len(c["turns"]) == len(c["fitted"]) and c["turns"][-1] == pytest.approx(12.0, abs=0.3)
    assert c["fitted"][-1] == pytest.approx(c["turns"][-1], abs=0.2)


def test_too_few_frames_are_refused():
    t = np.arange(10) / 30.0
    assert rotation_rate(t, np.cos(t), np.sin(t))["value"] is None
