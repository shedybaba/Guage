"""Regression tests on REAL clips: a swinging pendulum has a known relation between amplitude, period and speed.

The peak speed from local derivative fits must equal amplitude x omega, with omega from the pendulum tool's period:
two independent routes. Skipped when the clips or the cached tracking are not on this machine.
"""
import numpy as np
import pytest

from app import signals
from app.tools.kinematics import velocity_acceleration
from app.tools.pendulum import pendulum_period
from tests.real_clips import load

CLIPS = [("pendulum.mp4", "glass marble"), ("pendulum2.mp4", "brass ball")]


def swing(clip, phrase):
    tracks, info = load(clip, phrase)
    points = [p for pts in tracks.values() for p in pts]
    s = signals.series_for_object(points, signals.pick_main_object(points), info.timestamps)
    p = pendulum_period(s["t"], s["x"], s["y"])
    fq = p["fit_quality"]
    t0 = s["t"] - s["t"][0]
    inside = (t0 >= fq["segment"]["start_s"]) & (t0 <= fq["segment"]["end_s"])
    axis = s[fq["axis"]][inside]
    return s, inside, (axis.max() - axis.min()) / 2, 2 * np.pi / p["value"]


@pytest.mark.parametrize("clip,phrase", CLIPS)
def test_peak_speed_equals_amplitude_times_omega(clip, phrase):
    s, inside, amp, omega = swing(clip, phrase)
    r = velocity_acceleration(s["t"][inside], s["x"][inside], s["y"][inside], quantity="speed")
    assert abs(r["value"] - amp * omega) <= 2.5 * r["uncertainty"]


@pytest.mark.parametrize("clip,phrase", CLIPS)
def test_peak_acceleration_equals_amplitude_times_omega_squared(clip, phrase):
    s, inside, amp, omega = swing(clip, phrase)
    r = velocity_acceleration(s["t"][inside], s["x"][inside], s["y"][inside])
    pk = r["extra"]["peak_acceleration"]
    assert abs(pk["value"] - amp * omega ** 2) <= 2.5 * pk["uncertainty"]
