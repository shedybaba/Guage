"""Tests for the spring-and-mass tool, using synthetic springs with known mass, stiffness and damping.

Physics: omega_n = sqrt(k/m);  b = zeta * omega_n;  omega_d = omega_n * sqrt(1 - zeta^2);
x(t) = c + A exp(-b t) cos(omega_d t + phi).  What the camera sees oscillates at omega_d, but the
spring constant belongs to the undamped omega_n:  k = m omega_n^2.
"""
import numpy as np
import pytest

from app.tools.spring import oscillation_frequency

FPS = 30.0


def spring(m=0.2, k=20.0, zeta=0.05, A=80.0, seconds=12.0, c=400.0, phi=0.7, noise=0.4, seed=0,
           vertical=True, envelope="exp", fps=FPS):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    wn = np.sqrt(k / m)
    b = zeta * wn
    wd = wn * np.sqrt(1 - zeta ** 2)
    if envelope == "exp":
        amp = A * np.exp(-b * t)
    else:  # sliding friction: the amplitude falls in a straight line
        amp = A * (1 - t / (1.3 * seconds))
    s = c + amp * np.cos(wd * t + phi) + rng.normal(0, noise, t.size)
    other = 300.0 + rng.normal(0, 0.3, t.size)
    x, y = (other, s) if vertical else (s, other)
    truth = {"fd": wd / (2 * np.pi), "fn": wn / (2 * np.pi), "zeta": zeta, "b": b, "k": k, "m": m}
    return t, x, y, truth


def consistent(value, sigma, truth, k=2.5, allowance=0.0):
    assert abs(value - truth) <= k * sigma + allowance, (value, sigma, truth)


def test_result_shape():
    t, x, y, _ = spring()
    r = oscillation_frequency(t, x, y)
    for key in ("value", "uncertainty", "units", "method", "fit_quality", "series_for_chart", "warnings", "extra"):
        assert key in r
    assert r["units"] == "Hz"


def test_recovers_frequency_damping_and_spring_constant():
    t, x, y, tr = spring()
    r = oscillation_frequency(t, x, y, mass_kg=tr["m"])
    consistent(r["value"], r["uncertainty"], tr["fd"], allowance=0.002)
    assert r["uncertainty"] < 0.02
    z = r["extra"]["damping_ratio"]
    consistent(z["value"], z["uncertainty"], tr["zeta"], allowance=0.002)
    k = r["extra"]["spring_constant"]
    consistent(k["value"], k["uncertainty"], tr["k"], allowance=0.1)
    assert k["units"] == "N/m"


def test_spring_constant_uses_the_undamped_frequency():
    # heavy damping: the visible frequency is ~5% below the natural one, and k must not inherit that error
    t, x, y, tr = spring(zeta=0.3, A=120.0, seconds=8.0, noise=0.3)
    r = oscillation_frequency(t, x, y, mass_kg=tr["m"])
    assert r["extra"]["natural_frequency"]["value"] > r["value"]
    consistent(r["extra"]["natural_frequency"]["value"], r["extra"]["natural_frequency"]["uncertainty"], tr["fn"],
               allowance=0.01)
    k = r["extra"]["spring_constant"]
    consistent(k["value"], k["uncertainty"], tr["k"], allowance=0.3)
    naive = tr["m"] * (2 * np.pi * r["value"]) ** 2
    assert abs(naive - tr["k"]) > abs(k["value"] - tr["k"])  # the naive m*(2*pi*f)^2 would be worse


def test_mass_uncertainty_flows_into_k():
    t, x, y, tr = spring()
    sure = oscillation_frequency(t, x, y, mass_kg=0.2)["extra"]["spring_constant"]
    unsure = oscillation_frequency(t, x, y, mass_kg=0.2, mass_uncertainty_kg=0.02)["extra"]["spring_constant"]
    assert unsure["uncertainty"] > sure["uncertainty"]
    assert unsure["uncertainty"] >= 0.1 * unsure["value"] * 0.9  # a 10% mass error is a 10% k error at least


def test_the_springs_own_mass_is_added_as_a_third():
    t, x, y, tr = spring(m=0.2 + 0.06 / 3, k=20.0)  # effective mass = hanging mass + spring mass / 3
    plain = oscillation_frequency(t, x, y, mass_kg=0.2)["extra"]["spring_constant"]
    corrected = oscillation_frequency(t, x, y, mass_kg=0.2, spring_mass_kg=0.06)["extra"]["spring_constant"]
    assert abs(corrected["value"] - 20.0) < abs(plain["value"] - 20.0)
    consistent(corrected["value"], corrected["uncertainty"], 20.0, allowance=0.2)


def test_no_mass_means_no_spring_constant_and_says_so():
    t, x, y, _ = spring()
    r = oscillation_frequency(t, x, y)
    assert "spring_constant" not in r["extra"]
    assert any("mass" in w.lower() for w in r["warnings"])


def test_the_mass_note_mentions_the_spring_when_its_mass_is_not_given():
    t, x, y, tr = spring()
    r = oscillation_frequency(t, x, y, mass_kg=tr["m"])
    assert any("spring's own mass" in w for w in r["warnings"])
    r2 = oscillation_frequency(t, x, y, mass_kg=tr["m"], spring_mass_kg=0.01)
    assert not any("spring's own mass" in w for w in r2["warnings"])


def test_undamped_spring_is_reported_as_not_measurably_damped():
    t, x, y, tr = spring(zeta=0.0, seconds=8.0)
    z = oscillation_frequency(t, x, y)["extra"]["damping_ratio"]
    assert z["measurable"] is False
    assert z["upper_bound"] > 0 and z["upper_bound"] < 0.1


def test_works_for_horizontal_motion_too():
    t, x, y, tr = spring(vertical=False, seed=2)
    r = oscillation_frequency(t, x, y)
    assert r["fit_quality"]["axis"] == "x"
    consistent(r["value"], r["uncertainty"], tr["fd"], allowance=0.002)


def test_cuts_off_the_release_and_fits_the_clean_oscillation():
    t, x, y, tr = spring(seconds=14.0, seed=5)
    rng = np.random.default_rng(1)
    n = int(3 * FPS)  # three seconds of a hand lifting the mass before letting go
    wander = 400 + np.cumsum(rng.normal(0, 6, n))
    y = np.concatenate([wander, y[: len(y) - n]])  # release at t = 3 s
    r = oscillation_frequency(t, x, y, mass_kg=tr["m"])
    assert r["fit_quality"]["segment"]["trimmed"] is True
    assert r["fit_quality"]["segment"]["start_s"] >= 2.5
    consistent(r["value"], r["uncertainty"], tr["fd"], allowance=0.003)


def test_straight_line_decay_is_recognised_as_friction_not_viscous():
    t, x, y, _ = spring(zeta=0.04, envelope="linear", seconds=14.0, noise=0.2)
    r = oscillation_frequency(t, x, y)
    assert r["extra"]["damping_envelope"]["type"].startswith("linear")
    assert any("friction" in w.lower() for w in r["warnings"])


def test_exponential_decay_is_recognised_as_viscous():
    t, x, y, _ = spring(zeta=0.02, seconds=14.0, noise=0.2)
    r = oscillation_frequency(t, x, y)
    assert r["extra"]["damping_envelope"]["type"].startswith("exponential")
    assert not any("friction" in w.lower() for w in r["warnings"])


def test_a_fast_spring_filmed_at_30_fps_is_flagged():
    t, x, y, tr = spring(m=0.05, k=300.0, seconds=5.0)  # about 12 Hz: under 3 frames per cycle
    r = oscillation_frequency(t, x, y)
    assert any("frames per cycle" in w for w in r["warnings"])


def test_scale_gives_amplitude_and_top_speed_in_si_units():
    t, x, y, tr = spring(A=100.0, zeta=0.02)
    r = oscillation_frequency(t, x, y, scale_m_per_px=0.001)
    amp = r["extra"]["amplitude"]
    assert amp["units"] == "m" and amp["value"] == pytest.approx(0.1, rel=0.1)
    v = r["extra"]["max_speed"]
    assert v["units"] == "m/s"
    assert v["value"] == pytest.approx(amp["value"] * 2 * np.pi * r["value"], rel=0.01)


def test_few_cycles_warn_and_widen_the_damping_uncertainty():
    t, x, y, tr = spring(seconds=3.0)  # under 5 cycles
    r = oscillation_frequency(t, x, y)
    long = oscillation_frequency(*spring(seconds=12.0)[:3])
    assert any("cycles" in w for w in r["warnings"])
    assert r["extra"]["damping_ratio"]["uncertainty"] > long["extra"]["damping_ratio"]["uncertainty"]


def test_flat_signal_returns_no_value():
    t = np.arange(0, 10, 1 / FPS)
    r = oscillation_frequency(t, np.full_like(t, 5.0), np.full_like(t, 7.0))
    assert r["value"] is None and r["warnings"]


def test_the_quoted_uncertainties_cover_the_truth_most_of_the_time():
    """Over many noisy springs, frequency, damping ratio and k are each within 2 sigma at least 85% of the time."""
    hits = {"f": 0, "z": 0, "k": 0}
    runs = 30
    for seed in range(runs):
        rng = np.random.default_rng(900 + seed)
        m, k, zeta = 0.15 + 0.1 * rng.random(), 15 + 15 * rng.random(), 0.02 + 0.08 * rng.random()
        t, x, y, tr = spring(m=m, k=k, zeta=zeta, noise=0.8, seed=900 + seed, seconds=10.0)
        # tracking noise in real clips is correlated from frame to frame
        e = np.zeros(len(t))
        for i in range(1, len(t)):
            e[i] = 0.8 * e[i - 1] + rng.normal(0, 0.5)
        y = y + e
        r = oscillation_frequency(t, x, y, mass_kg=m)
        hits["f"] += abs(r["value"] - tr["fd"]) <= 2 * r["uncertainty"]
        z = r["extra"]["damping_ratio"]
        hits["z"] += z["measurable"] and abs(z["value"] - zeta) <= 2 * z["uncertainty"] or \
            (not z["measurable"] and zeta <= z["upper_bound"])
        kk = r["extra"]["spring_constant"]
        hits["k"] += abs(kk["value"] - k) <= 2 * kk["uncertainty"]
    for name, h in hits.items():
        assert h / runs >= 0.85, (name, h)


def test_noise_is_not_mistaken_for_an_oscillation():
    rng = np.random.default_rng(0)
    t = np.arange(0, 12, 1 / FPS)
    r = oscillation_frequency(t, 100 + rng.normal(0, 0.3, t.size), 50 + rng.normal(0, 0.3, t.size))
    assert r["value"] is None and "No clear oscillation" in r["warnings"][-1]
