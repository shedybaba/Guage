"""Tests for the pendulum period fit, using synthetic trajectories.

The maths must be proven correct without any AI or real video: we generate a
damped sinusoid with a known period, add noise, and check we recover it.
"""
import numpy as np
import pytest

from app.tools.pendulum import pendulum_period, G_STANDARD


def synthetic(T=1.5, fps=30.0, seconds=12.0, amp=80.0, damping=0.03, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    x = 320 + amp * np.exp(-damping * t) * np.cos(2 * np.pi * t / T + 0.4)
    x = x + rng.normal(0, noise, t.size)
    y = 100 + rng.normal(0, 0.3, t.size)  # almost no vertical motion
    return t, x, y


def test_recovers_known_period():
    t, x, y = synthetic(T=1.5)
    r = pendulum_period(t, x, y)
    assert r["units"] == "s"
    assert r["value"] == pytest.approx(1.5, abs=0.02)
    assert 0 < r["uncertainty"] < 0.05
    assert r["fit_quality"]["r2"] > 0.99
    assert r["warnings"] == []


def test_uses_the_axis_that_moves():
    t, x, y = synthetic(T=1.2)
    r = pendulum_period(t, y, x)  # swap: the swing is now in y
    assert r["value"] == pytest.approx(1.2, abs=0.02)


def test_computes_g_from_string_length():
    T = 1.4192  # a 0.50 m pendulum
    t, x, y = synthetic(T=T, seconds=15.0)
    r = pendulum_period(t, x, y, length_m=0.50)
    g = r["extra"]["g"]
    assert g["value"] == pytest.approx(4 * np.pi**2 * 0.50 / r["value"] ** 2)
    assert g["value"] == pytest.approx(G_STANDARD, rel=0.03)
    assert g["uncertainty"] > 0
    assert g["percent_error_vs_standard"] < 3


def test_warns_when_fewer_than_two_cycles():
    t, x, y = synthetic(T=1.5, seconds=2.5)
    r = pendulum_period(t, x, y)
    assert any("cycles" in w for w in r["warnings"])


def test_handles_uneven_timestamps():
    # phones record variable frame rate: timestamps must be used, not frame/fps
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 12, 360))
    x = 320 + 80 * np.cos(2 * np.pi * t / 1.5)
    y = np.full_like(t, 100.0)
    r = pendulum_period(t, x, y)
    assert r["value"] == pytest.approx(1.5, abs=0.02)


def test_rejects_flat_signal():
    t = np.arange(0, 10, 1 / 30)
    r = pendulum_period(t, np.full_like(t, 5.0), np.full_like(t, 7.0))
    assert r["value"] is None
    assert r["warnings"]
