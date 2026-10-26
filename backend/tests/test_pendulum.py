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


def with_setup_motion(T=0.86, setup=4.0, swing=8.0, fps=30.0, amp=140.0, seed=3):
    """A random wander (camera moving, hand in shot) followed by a clean swing."""
    rng = np.random.default_rng(seed)
    n_setup = int(setup * fps)
    wander = 400 + np.cumsum(rng.normal(0, 9, n_setup))
    t = np.arange(0, setup + swing, 1 / fps)
    ts = t[n_setup:] - t[n_setup]
    swing_x = wander[-1] + amp * np.exp(-0.03 * ts) * np.sin(2 * np.pi * ts / T)
    x = np.concatenate([wander, swing_x]) + rng.normal(0, 1.5, t.size)
    y = 700 + rng.normal(0, 2.0, t.size)
    return t, x, y


def test_finds_the_clean_swing_inside_setup_motion():
    t, x, y = with_setup_motion(T=0.86, setup=4.0)
    r = pendulum_period(t, x, y)
    assert r["value"] == pytest.approx(0.86, abs=0.01)
    seg = r["fit_quality"]["segment"]
    assert seg["trimmed"] is True
    assert 3.6 <= seg["start_s"] <= 5.0  # the real swing starts at 4.0 s
    assert r["fit_quality"]["r2"] > 0.97
    assert any("set-up" in w or "setup" in w for w in r["warnings"])


def test_clean_signal_is_not_trimmed():
    t, x, y = synthetic(T=1.5, seconds=12.0)
    r = pendulum_period(t, x, y)
    seg = r["fit_quality"]["segment"]
    assert seg["trimmed"] is False
    assert seg["start_s"] == pytest.approx(0.0, abs=0.01)


def test_auto_segment_can_be_turned_off():
    t, x, y = with_setup_motion()
    r = pendulum_period(t, x, y, auto_segment=False)
    assert r["fit_quality"]["segment"]["trimmed"] is False


def correlated_noise(n, rho, sigma, rng):
    """AR(1) noise: neighbouring frames share errors, like real tracking jitter."""
    e = rng.normal(0, sigma * np.sqrt(1 - rho**2), n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


def test_uncertainty_covers_the_true_error_with_correlated_noise():
    """With realistic noise the quoted +/- must cover the truth most of the time."""
    T_true, hits, runs = 1.2, 0, 24
    for seed in range(runs):
        rng = np.random.default_rng(100 + seed)
        t = np.arange(0, 10, 1 / 30)
        x = 320 + 90 * np.exp(-0.03 * t) * np.cos(2 * np.pi * t / T_true + rng.uniform(0, 6))
        x = x + correlated_noise(t.size, 0.9, 6.0, rng)
        y = np.full_like(t, 100.0)
        r = pendulum_period(t, x, y)
        if abs(r["value"] - T_true) <= 2 * r["uncertainty"]:
            hits += 1
    assert hits / runs >= 0.85


def test_uncertainty_never_below_the_frame_timing_floor():
    t, x, y = synthetic(T=1.5, noise=0.0)
    r = pendulum_period(t, x, y)
    comps = r["fit_quality"]["uncertainty_components"]
    assert comps["timing_floor"] > 0
    assert r["uncertainty"] >= comps["timing_floor"]
    assert r["uncertainty"] == pytest.approx(max(comps["fit"], comps["cycle_scatter"] or 0, comps["timing_floor"]))


def test_rejects_flat_signal():
    t = np.arange(0, 10, 1 / 30)
    r = pendulum_period(t, np.full_like(t, 5.0), np.full_like(t, 7.0))
    assert r["value"] is None
    assert r["warnings"]
