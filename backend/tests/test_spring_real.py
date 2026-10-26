"""Regression tests on REAL clips: two swinging pendulums are real damped oscillations.

They check the damping measurement against itself: the decay rate from the fit and the decay
rate from the peak heights are independent measurements and must agree within their uncertainties.
Skipped when the clips or the cached tracking are not on this machine.
"""
import numpy as np
import pytest

from app import signals
from app.tools.pendulum import pendulum_period
from app.tools.spring import oscillation_frequency
from tests.real_clips import load

CLIPS = [("pendulum.mp4", "glass marble"), ("pendulum2.mp4", "brass ball")]


def analyse(clip, phrase):
    tracks, info = load(clip, phrase)
    points = [p for pts in tracks.values() for p in pts]
    s = signals.series_for_object(points, signals.pick_main_object(points), info.timestamps)
    return s, oscillation_frequency(s["t"], s["x"], s["y"])


@pytest.mark.parametrize("clip,phrase", CLIPS)
def test_frequency_matches_the_pendulum_tools_period(clip, phrase):
    s, m = analyse(clip, phrase)
    p = pendulum_period(s["t"], s["x"], s["y"])
    assert 1 / m["value"] == pytest.approx(p["value"], rel=1e-6)


@pytest.mark.parametrize("clip,phrase", CLIPS)
def test_the_two_damping_measurements_agree(clip, phrase):
    _, m = analyse(clip, phrase)
    z = m["extra"]["damping_ratio"]
    env = m["extra"]["damping_envelope"]
    assert z["measurable"] and env["decay_rate"] is not None
    b, sb = z["decay_rate"]["value"], z["decay_rate"]["uncertainty"]
    gap = abs(b - env["decay_rate"])
    assert gap <= 3 * np.hypot(sb, env["decay_rate_uncertainty"])  # independent estimates, one answer
    assert 0 < z["value"] < 0.02  # a lightly damped pendulum, as seen


@pytest.mark.parametrize("clip,phrase", CLIPS)
def test_a_pendulum_is_not_called_friction_damped(clip, phrase):
    _, m = analyse(clip, phrase)
    assert not m["extra"]["damping_envelope"]["type"].startswith("linear")
