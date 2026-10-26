"""Tests for free fall and velocity/acceleration, on simulated motion with known physics.

Image y points DOWN, so a falling object has positive acceleration in y. Distances in the simulation are in
metres and converted to pixels with a known scale, so every answer has a true value.
"""
import numpy as np
import pytest

from app.tools.kinematics import free_fall, velocity_acceleration

G = 9.81
SCALE = 0.002  # metres per pixel


def correlated(n, rho, sigma, rng):
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


def drop(g=G, scale=SCALE, fps=120.0, rest=1.0, height=1.0, after=0.6, vx=0.0, v0=0.0, tilt_deg=0.0,
         noise=0.5, rho=0.0, seed=0, drag=0.0):
    """An object held still, released, falling `height` metres and then lying still on the floor."""
    rng = np.random.default_rng(seed)
    # fall time from  height = v0 tf + g tf^2 / 2
    tf = (-v0 + np.sqrt(v0 ** 2 + 2 * g * height)) / g
    t = np.arange(0, rest + tf + after, 1 / fps)
    tau = np.clip(t - rest, 0, tf)
    if drag:  # simple linear drag: terminal velocity g / drag
        y_m = v0 * (1 - np.exp(-drag * tau)) / drag + g / drag * (tau - (1 - np.exp(-drag * tau)) / drag)
        y_m = np.where(t - rest < 0, 0.0, y_m)
        y_m = np.minimum(y_m, height * 1.5)
    else:
        y_m = v0 * tau + 0.5 * g * tau ** 2
    x_m = vx * tau
    x_px = 300.0 + x_m / scale
    y_px = 100.0 + y_m / scale
    if tilt_deg:
        a = np.radians(tilt_deg)
        dx, dy = x_px - 300.0, y_px - 100.0
        x_px, y_px = 300.0 + dx * np.cos(a) + dy * np.sin(a), 100.0 - dx * np.sin(a) + dy * np.cos(a)
    nx = correlated(len(t), rho, noise, rng) if rho else rng.normal(0, noise, len(t))
    ny = correlated(len(t), rho, noise, rng) if rho else rng.normal(0, noise, len(t))
    return t, x_px + nx, y_px + ny, {"rest": rest, "fall": tf, "release_frame": int(rest * fps)}


def ok(value, sigma, truth, k=2.5, allowance=0.0):
    assert abs(value - truth) <= k * sigma + allowance, (value, sigma, truth)


# ---------------------------------------------------------------- free fall

def test_result_shape():
    t, x, y, _ = drop()
    r = free_fall(t, x, y, scale_m_per_px=SCALE)
    for key in ("value", "uncertainty", "units", "method", "fit_quality", "series_for_chart", "warnings", "extra"):
        assert key in r
    assert r["units"] == "m/s^2"


def test_recovers_g_from_a_drop():
    t, x, y, _ = drop()
    r = free_fall(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    ok(r["value"], r["uncertainty"], G)
    assert 0.05 * G >= r["uncertainty"] >= 0.01 * G * 0.9  # the scale's 1% is in the budget, and nothing silly beyond


def test_recovers_a_different_acceleration():
    t, x, y, _ = drop(g=3.7, seed=2)  # e.g. a dropped object on Mars
    r = free_fall(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    ok(r["value"], r["uncertainty"], 3.7)


def test_picks_the_fall_out_of_rest_and_floor():
    t, x, y, info = drop()
    r = free_fall(t, x, y, scale_m_per_px=SCALE)
    seg = r["fit_quality"]["segment"]
    assert abs(seg["start_s"] - info["rest"]) < 0.08  # release
    assert abs(seg["end_s"] - (info["rest"] + info["fall"])) < 0.08  # landing
    assert r["fit_quality"]["trimmed"] is True


def test_fits_only_the_first_flight_when_the_object_bounces():
    t, x, y, info = drop(after=0.0)
    fps = 120.0
    # a rebound at half the impact speed, then a second shorter fall
    vi = G * info["fall"]
    tb = np.arange(0, 2 * 0.5 * vi / G, 1 / fps)
    rebound = (100.0 + 1.0 / SCALE) - (0.5 * vi * tb - 0.5 * G * tb ** 2) / SCALE
    t2 = np.concatenate([t, t[-1] + 1 / fps + tb])
    y2 = np.concatenate([y, rebound + np.random.default_rng(1).normal(0, 0.5, len(tb))])
    x2 = np.concatenate([x, np.full(len(tb), 300.0)])
    r = free_fall(t2, x2, y2, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    ok(r["value"], r["uncertainty"], G)
    assert r["fit_quality"]["segment"]["end_s"] < t[-1] + 0.1  # did not run on into the rebound


def test_a_tilted_phone_does_not_change_g_and_the_tilt_is_reported():
    t, x, y, _ = drop(tilt_deg=12.0, seed=3)
    r = free_fall(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    ok(r["value"], r["uncertainty"], G)
    assert abs(abs(r["extra"]["tilt_deg"]) - 12.0) < 1.0
    assert any("tilt" in w.lower() for w in r["warnings"])


def test_a_thrown_object_still_gives_g():
    t, x, y, _ = drop(vx=1.5, v0=-2.0, height=0.5, seed=4)  # thrown up and sideways, lands 0.5 m below the start
    r = free_fall(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    ok(r["value"], r["uncertainty"], G)


def test_without_a_scale_it_reports_pixels_and_the_implied_scale():
    t, x, y, _ = drop()
    r = free_fall(t, x, y)
    assert r["units"] == "px/s^2"
    assert any("scale" in w.lower() for w in r["warnings"])
    implied = r["extra"]["scale_if_g_is_9.81"]
    ok(implied["value"], implied["uncertainty"], SCALE, allowance=SCALE * 0.01)


def test_the_assumed_scale_uncertainty_is_stated_when_none_is_given():
    t, x, y, _ = drop()
    r = free_fall(t, x, y, scale_m_per_px=SCALE)
    assert r["uncertainty"] >= 0.02 * G * 0.95  # assumes 2%
    assert any("2%" in w for w in r["warnings"])
    given = free_fall(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.005)
    assert given["uncertainty"] < r["uncertainty"]
    assert not any("assumed" in w.lower() for w in given["warnings"])


def test_compares_with_standard_gravity():
    t, x, y, _ = drop(g=8.0, seed=5)
    r = free_fall(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    assert r["extra"]["consistent_with_standard_g"] is False
    assert r["extra"]["percent_difference_from_standard"] == pytest.approx(100 * (8.0 - G) / G, abs=3)
    good = free_fall(*drop(seed=6)[:3], scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    assert good["extra"]["consistent_with_standard_g"] is True


def test_air_drag_is_flagged():
    t, x, y, _ = drop(height=3.0, drag=1.2, noise=0.2, seed=7)  # a light object: noticeable drag
    r = free_fall(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    assert any("changes during" in w.lower() or "drag" in w.lower() for w in r["warnings"])


def test_a_very_short_fall_warns_to_use_slow_motion():
    t, x, y, _ = drop(fps=30.0, height=0.4)  # about 8 frames
    r = free_fall(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    assert r["value"] is not None
    assert any("slow motion" in w.lower() for w in r["warnings"])


def test_faster_filming_gives_a_tighter_answer():
    slow = free_fall(*drop(fps=30.0, height=1.0, seed=8)[:3], scale_m_per_px=SCALE, scale_uncertainty_fraction=0.001)
    fast = free_fall(*drop(fps=240.0, height=1.0, seed=8)[:3], scale_m_per_px=SCALE, scale_uncertainty_fraction=0.001)
    assert fast["uncertainty"] < slow["uncertainty"]


def test_an_object_that_does_not_fall_gives_no_value():
    t = np.arange(0, 3, 1 / 60.0)
    r = free_fall(t, np.full_like(t, 300.0), np.full_like(t, 100.0), scale_m_per_px=SCALE)
    assert r["value"] is None and r["warnings"]


def test_constant_velocity_is_not_a_fall():
    t = np.arange(0, 3, 1 / 60.0)
    r = free_fall(t, 300.0 + 50 * t, 100.0 + 200 * t, scale_m_per_px=SCALE)
    assert r["value"] is None and r["warnings"]


def test_the_quoted_uncertainty_covers_g_most_of_the_time():
    """Over many noisy drops (correlated noise, random heights, frame rates and tilts) |g - truth| <= 2 sigma at
    least 85% of the time. The scale error is set to almost nothing, so this tests the statistical uncertainty:
    a generous scale term must not be able to hide an over-confident fit."""
    hits, runs = 0, 80
    for seed in range(runs):
        rng = np.random.default_rng(8000 + seed)
        g = 9.0 + 1.6 * rng.random()
        t, x, y, _ = drop(g=g, fps=float(rng.choice([30, 60, 120, 240])), height=0.4 + 1.6 * rng.random(),
                          noise=0.3 + 0.9 * rng.random(), rho=float(rng.choice([0.0, 0.5, 0.8])), seed=8000 + seed,
                          tilt_deg=float(rng.uniform(-10, 10)))
        r = free_fall(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.0001)
        if r["value"] is not None and abs(r["value"] - g) <= 2 * r["uncertainty"]:
            hits += 1
    assert hits / runs >= 0.85


# ------------------------------------------------------- velocity / acceleration

def ramp(theta_deg=20.0, factor=1.0, scale=SCALE, fps=60.0, rest=0.8, length=1.5, noise=0.4, seed=0, g=G):
    """A ball starts still at the top of a ramp, accelerates down it at factor * g sin(theta), then is caught."""
    rng = np.random.default_rng(seed)
    th = np.radians(theta_deg)
    a = factor * g * np.sin(th)
    t_run = np.sqrt(2 * length / a)
    t = np.arange(0, rest + t_run + 0.8, 1 / fps)
    tau = np.clip(t - rest, 0, t_run)
    s = 0.5 * a * tau ** 2 / scale  # pixels along the ramp
    x = 100.0 + s * np.cos(th) + rng.normal(0, noise, len(t))
    y = 100.0 + s * np.sin(th) + rng.normal(0, noise, len(t))
    return t, x, y, {"a": a, "t_run": t_run, "rest": rest, "theta": theta_deg}


def test_va_result_shape():
    t, x, y, _ = ramp()
    r = velocity_acceleration(t, x, y, scale_m_per_px=SCALE)
    for key in ("value", "uncertainty", "units", "method", "fit_quality", "series_for_chart", "warnings", "extra"):
        assert key in r
    assert r["units"] == "m/s^2"


def test_measures_the_acceleration_down_a_ramp():
    t, x, y, info = ramp(theta_deg=20.0, factor=1.0)
    r = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    ok(r["value"], r["uncertainty"], info["a"], allowance=0.02)
    seg = r["fit_quality"]["segment"]
    assert abs(seg["start_s"] - info["rest"]) < 0.15


def test_compares_with_g_sin_theta_for_sliding_and_rolling():
    t, x, y, info = ramp(theta_deg=20.0, factor=5 / 7)  # a solid ball rolling without slipping
    r = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01, ramp_angle_deg=20.0)
    ramp_info = r["extra"]["ramp"]
    assert ramp_info["expected"]["sliding_without_friction"] == pytest.approx(G * np.sin(np.radians(20)), rel=1e-6)
    assert ramp_info["expected"]["rolling_solid_sphere"] == pytest.approx(5 / 7 * G * np.sin(np.radians(20)), rel=1e-6)
    assert ramp_info["closest"] == "rolling_solid_sphere"
    assert ramp_info["ratio_to_g_sin_theta"]["value"] == pytest.approx(5 / 7, abs=0.05)


def test_a_sliding_block_is_matched_to_g_sin_theta():
    t, x, y, _ = ramp(theta_deg=15.0, factor=1.0, seed=2)
    r = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01, ramp_angle_deg=15.0)
    assert r["extra"]["ramp"]["closest"] == "sliding_without_friction"


def test_warns_when_the_filmed_slope_disagrees_with_the_stated_angle():
    t, x, y, _ = ramp(theta_deg=30.0, seed=3)
    r = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, ramp_angle_deg=15.0)  # told 15, the video shows 30
    assert any("angle" in w.lower() for w in r["warnings"])
    assert r["extra"]["path_angle_deg"] == pytest.approx(30.0, abs=1.5)


def test_speed_at_a_given_time_and_peak_speed_on_a_ramp():
    t, x, y, info = ramp(theta_deg=20.0, factor=1.0, seed=4)
    a, rest, t_run = info["a"], info["rest"], info["t_run"]
    r = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01, quantity="speed",
                              at_time=rest + 0.3)
    ok(r["value"], r["uncertainty"], a * 0.3, allowance=0.03)
    assert r["units"] == "m/s"
    peak = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01, quantity="speed")
    assert 0.8 * a * t_run <= peak["value"] <= a * t_run + 3 * peak["uncertainty"]  # the abrupt stop smears the last frames


def test_constant_velocity_has_no_acceleration_and_the_right_speed():
    t = np.arange(0, 4, 1 / 60.0)
    rng = np.random.default_rng(1)
    x = 100 + 200.0 * t + rng.normal(0, 0.4, len(t))  # 200 px/s = 0.4 m/s
    y = 300 + rng.normal(0, 0.4, len(t))
    sp = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01, quantity="average_speed")
    ok(sp["value"], sp["uncertainty"], 0.4, allowance=0.005)
    acc = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    assert abs(acc["value"]) < 0.05
    assert any("no acceleration" in w.lower() or "constant speed" in w.lower() for w in acc["warnings"])


def test_peak_speed_of_a_swinging_object_matches_amplitude_times_omega():
    t = np.arange(0, 10, 1 / 60.0)
    A, T = 150.0, 1.4  # px, s
    rng = np.random.default_rng(2)
    x = 300 + A * np.sin(2 * np.pi * t / T) + rng.normal(0, 0.4, len(t))
    y = 200 + rng.normal(0, 0.4, len(t))
    r = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01, quantity="speed")
    ok(r["value"], r["uncertainty"], A * SCALE * 2 * np.pi / T, allowance=0.01)
    extra = r["extra"]
    pk = extra["peak_acceleration"]
    ok(pk["value"], pk["uncertainty"], A * SCALE * (2 * np.pi / T) ** 2)


def test_average_speed_is_the_path_length_over_the_time():
    t, x, y, info = ramp(theta_deg=20.0, factor=1.0, seed=5)
    r = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01, quantity="average_speed")
    assert r["value"] == pytest.approx(1.5 / info["t_run"], rel=0.08)  # 1.5 m in t_run, from rest to caught


def test_without_a_scale_everything_is_in_pixels():
    t, x, y, info = ramp()
    r = velocity_acceleration(t, x, y)
    assert r["units"] == "px/s^2"
    assert any("scale" in w.lower() for w in r["warnings"])
    assert r["value"] == pytest.approx(info["a"] / SCALE, rel=0.05)


def test_series_for_the_chart_has_speed_and_acceleration():
    t, x, y, _ = ramp()
    s = velocity_acceleration(t, x, y, scale_m_per_px=SCALE)["series_for_chart"]
    for key in ("t", "speed", "tangential_acceleration", "fitted_speed"):
        assert key in s and len(s[key]) == len(t)


def test_acceleration_coverage():
    """The quoted uncertainty of a ramp acceleration covers the truth at 2 sigma in at least 85% of noisy runs (scale error removed)."""
    hits, runs = 0, 80
    for seed in range(runs):
        rng = np.random.default_rng(8000 + seed)
        th = float(rng.uniform(8, 35))
        t, x, y, info = ramp(theta_deg=th, factor=float(rng.uniform(0.6, 1.0)), fps=float(rng.choice([30, 60, 120])),
                             noise=0.3 + 0.7 * rng.random(), seed=8000 + seed)
        r = velocity_acceleration(t, x, y, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.0001)
        if r["value"] is not None and abs(r["value"] - info["a"]) <= 2 * r["uncertainty"]:
            hits += 1
    assert hits / runs >= 0.85
