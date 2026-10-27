"""Tests for the growth tool on simulated time-lapses with known answers."""
import numpy as np
import pytest

from app.tools.growth import growth_over_time


def ar1(n, rho, sigma, rng):
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


def clip(fn, seconds=60.0, fps=10.0, noise=1.0, rho=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    return t, fn(t) + ar1(t.size, rho, noise, rng)


def test_steady_growth_is_measured_with_its_rate():
    t, h = clip(lambda t: 100 + t)  # 100 px to 160 px: +60%
    r = growth_over_time(t, h, "height", speedup=3600.0)  # 1 s of video = 1 hour: the clip covers 60 hours
    x = r["extra"]
    assert r["units"] == "%" and r["value"] == pytest.approx(59.4, abs=2.0)  # 100 -> 159.9
    assert abs(r["value"] - 59.9) <= 2.5 * r["uncertainty"] + 0.5
    assert x["growth_rate"]["units"] == "px/h" and x["growth_rate"]["value"] == pytest.approx(1.0, rel=0.05)
    assert x["model"] == "linear" and x["grew_measurably"]
    assert any("steady" in s for s in x["statements"])


def test_the_whole_change_is_measured_not_just_the_part_between_window_centres():
    """A mean over the first and last 10% would lose several percent of a steady change; the edge fits must not."""
    t = np.arange(0, 100.0, 0.5)
    r = growth_over_time(t, 200 + t, "height", speedup=60.0)  # exactly 200 -> 299.5, no noise
    assert r["value"] == pytest.approx(49.75, abs=0.3)


def test_exponential_growth_gives_the_doubling_time():
    # the area doubles every 10 h; the clip covers 30 h (one second of video is 30 minutes)
    t, a = clip(lambda t: 4000 * 2 ** (t / 20.0), seconds=60.0, noise=15.0, seed=3)
    r = growth_over_time(t, a, "area", speedup=1800.0)
    x = r["extra"]
    assert r["value"] == pytest.approx(700.0, rel=0.05)  # 8x -> +700%
    assert x["model"] == "exponential"
    dt = x["doubling_time"]
    assert dt["units"] == "h" and dt["value"] == pytest.approx(10.0, rel=0.08)
    assert abs(dt["value"] - 10.0) <= 3 * dt["uncertainty"]
    assert any("speeds up" in s for s in x["statements"])


def test_no_growth_is_reported_as_not_measurable():
    t, h = clip(lambda t: 120 + 0 * t, noise=1.5, seed=4)
    r = growth_over_time(t, h, "height", speedup=60.0)
    assert r["extra"]["grew_measurably"] is False
    assert any("No measurable growth" in w for w in r["warnings"])
    assert "doubling_time" not in r["extra"]


def test_shrinking_is_negative_and_said():
    t, h = clip(lambda t: 200 - t, seed=5)
    r = growth_over_time(t, h, "height", speedup=60.0)
    assert r["value"] < -20 and any("smaller" in w for w in r["warnings"])


def test_glitches_of_the_outline_are_left_out_and_reported():
    t, h = clip(lambda t: 100 + t, seed=6)
    clean = growth_over_time(t, h, "height", speedup=3600.0)
    h2 = h.copy()
    h2[[30, 31, 200, 450]] *= 0.2  # the object lost for a moment
    r = growth_over_time(t, h2, "height", speedup=3600.0)
    assert r["extra"]["outliers_removed"] == 4 and any("glitched" in w for w in r["warnings"])
    assert r["value"] == pytest.approx(clean["value"], abs=1.5)


def test_without_real_time_there_is_no_rate_and_it_says_so():
    t, h = clip(lambda t: 100 + t, seed=7)
    r = growth_over_time(t, h, "height")
    assert "growth_rate" not in r["extra"] and "doubling_time" not in r["extra"]
    assert any("No real time" in w for w in r["warnings"])
    assert r["value"] == pytest.approx(59.4, abs=2.5)  # the percentage needs no real time


def test_a_known_length_gives_real_units_and_the_ruler_error_only_touches_those():
    t, h = clip(lambda t: 100 + t, seed=8)
    a = growth_over_time(t, h, "height", speedup=3600.0, scale_m_per_px=0.0005)  # 1 px = 0.5 mm
    b = growth_over_time(t, h, "height", speedup=3600.0, scale_m_per_px=0.0005, scale_uncertainty_fraction=0.05)
    assert a["extra"]["absolute_change"]["units"] == "cm"  # a plant of about 6 cm is shown in cm
    assert a["extra"]["absolute_change"]["value"] == pytest.approx(3.0, abs=0.15)  # 60 px * 0.5 mm
    assert a["extra"]["growth_rate"]["units"] == "cm/h"
    assert b["extra"]["absolute_change"]["uncertainty"] > 1.4 * a["extra"]["absolute_change"]["uncertainty"]  # the ruler adds to it
    assert b["uncertainty"] == pytest.approx(a["uncertainty"])  # the percentage does not depend on the ruler
    assert not any("pixels" in w for w in b["warnings"])
    assert any("pixels" in w for w in growth_over_time(t, h, "height", speedup=3600.0)["warnings"])


def test_area_uses_squared_units():
    t, a = clip(lambda t: 40000 + 200 * t, noise=50.0, seed=9)
    r = growth_over_time(t, a, "area", speedup=60.0, scale_m_per_px=0.001)  # 40000 px^2 = 0.04 m^2 = 400 cm^2
    assert r["extra"]["start_size"]["units"] == "cm²" and r["extra"]["start_size"]["value"] == pytest.approx(400.0, rel=0.02)


def test_quoted_uncertainty_covers_the_truth_with_correlated_noise():
    hits, runs = 0, 30
    for seed in range(runs):
        t, h = clip(lambda t: 100 + 0.8 * t + 0.01 * t ** 2, noise=2.0, rho=0.9, seed=100 + seed)
        truth = ((100 + 0.8 * t[-1] + 0.01 * t[-1] ** 2) / (100 + 0.8 * t[0] + 0.01 * t[0] ** 2) - 1) * 100
        r = growth_over_time(t, h, "height", speedup=3600.0)
        hits += abs(r["value"] - truth) <= 2 * r["uncertainty"]
    assert hits / runs >= 0.85


def test_growth_that_then_reverses_is_flagged():
    t = np.arange(0, 60.0, 0.2)
    h = 100 + 60 * np.sin(np.pi * t / 80.0) ** 2 * (t < 40) + np.where(t >= 40, 60 * np.sin(np.pi * 40 / 80.0) ** 2 - (t - 40) * 2.5, 0)
    r = growth_over_time(t, h + np.random.default_rng(1).normal(0, 0.5, t.size), "height", speedup=60.0)
    assert any("peaked" in w for w in r["warnings"])


def test_chart_series_and_time_units():
    t, h = clip(lambda t: 100 + t, seed=10)
    c = growth_over_time(t, h, "height", speedup=3600.0)["series_for_chart"]  # 60 h
    assert c["t_unit"] == "h" and len(c["t"]) == len(c["measured"]) == len(c["fitted"]) and c["t"][-1] == pytest.approx(60, abs=1)
    for seconds_real, unit in ((90, "s"), (3 * 3600, "h"), (10 * 86400, "day"), (3000, "min")):
        r = growth_over_time(t, h, "height", speedup=seconds_real / 60.0)
        assert r["series_for_chart"]["t_unit"] == unit


def test_too_few_frames_and_junk_are_refused():
    assert growth_over_time(np.arange(10.0), np.arange(10.0) + 5, "height")["value"] is None
    t = np.arange(0, 6.0, 0.1)
    assert growth_over_time(t, np.full_like(t, np.nan), "height")["value"] is None
    assert growth_over_time(t, np.full_like(t, -3.0), "height")["value"] is None


def test_uneven_timestamps_are_used_not_frame_numbers():
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 60, 400))
    r = growth_over_time(t, 100 + t + rng.normal(0, 1, t.size), "height", speedup=3600.0)
    assert r["extra"]["growth_rate"]["value"] == pytest.approx(1.0, rel=0.06)


def test_a_plateaued_mask_area_is_not_mistaken_for_glitches():
    """SAM's mask area changes in small steps (a few frames at one value, then the next): that is not noise to remove."""
    rng = np.random.default_rng(12)
    t = np.arange(0, 60.0, 0.1)
    true = 5000 + 9000 * t / 60.0
    a = np.round(true / 150) * 150 + np.repeat(rng.normal(0, 20, 60), 10)  # stair steps, a little noise per step
    r = growth_over_time(t, a, "area", speedup=3600.0)
    assert r["extra"]["outliers_removed"] == 0 and not any("glitched" in w for w in r["warnings"])
    assert r["value"] == pytest.approx((14000 / 5000 - 1) * 100, abs=15.0)
