"""Tests for the collision tool, using synthetic collisions with known physics.

Physics used: for masses m1 (moving at v1) and m2 (at rest) and restitution e,
  v1' = (m1 - e*m2) v1 / (m1 + m2),   v2' = (1 + e) m1 v1 / (m1 + m2)
Total momentum m1 v1 = m1 v1' + m2 v2' is conserved whatever e is.
"""
import numpy as np
import pytest

from app.tools.collision import collision_momentum

FPS = 30.0


def consistent(r, expected=0.0, k=2.5):
    """The answer must agree with the truth within its OWN quoted uncertainty (plus a 1% allowance)."""
    assert r["value"] is not None
    assert abs(r["value"] - expected) <= k * r["uncertainty"] + 1.0, (r["value"], r["uncertainty"])
    assert r["uncertainty"] < 10  # and the uncertainty must be useful, not a shrug


def collision_scene(m1=1.0, m2=1.0, v1=900.0, e=1.0, tc=1.0123, seconds=3.0, noise=0.3, seed=0,
                    n_static=4, shake=0.0, after_factor=1.0, second=None, friction=None):
    """Ball A hits ball B at time tc (between frames). Returns (t, positions, info)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / FPS)
    v1p = (m1 - e * m2) * v1 / (m1 + m2)
    v2p = (1 + e) * m1 * v1 / (m1 + m2) * after_factor

    def pair(tc_, speed_scale=1.0, x0=300.0):
        a = np.where(t < tc_, x0 + v1 * speed_scale * (t - tc_), x0 + v1p * speed_scale * (t - tc_))
        b = np.where(t < tc_, x0 + 60, x0 + 60 + v2p * speed_scale * (t - tc_) * (t >= tc_))
        return a, b

    pos = {}
    a, b = pair(tc)
    if friction:  # after the impact A and B slow to a stop (time constant `friction` seconds)
        k = np.maximum(t - tc, 0.0)
        a = np.where(t < tc, a, 300.0 + v1p * friction * (1 - np.exp(-k / friction)))
        b = np.where(t < tc, b, 360.0 + v2p * friction * (1 - np.exp(-k / friction)))
    pos["A"], pos["B"] = a, b
    if second:
        tc2, scale = second
        c, d = pair(tc2, scale, x0=900.0)
        pos["C"], pos["D"] = c, d
    for k in range(n_static):
        pos[f"S{k}"] = np.full_like(t, 100.0 + 90 * k) + 0.0
    shift = np.cumsum(rng.normal(0, shake, t.size)) if shake else np.zeros_like(t)
    out = {}
    for name, x in pos.items():
        out[name] = (x + shift + rng.normal(0, noise, t.size),
                     np.full_like(t, 400.0) + 0.3 * shift + rng.normal(0, noise, t.size))
    return t, out, {"v1p": v1p, "v2p": v2p}


def test_returns_the_standard_result_shape():
    t, pos, _ = collision_scene()
    r = collision_momentum(t, pos)
    for key in ("value", "uncertainty", "units", "method", "fit_quality", "series_for_chart", "warnings", "extra"):
        assert key in r
    assert r["units"] == "%"


def test_equal_masses_elastic_conserves_momentum_and_energy():
    t, pos, _ = collision_scene(e=1.0)
    r = collision_momentum(t, pos, masses={k: 1.0 for k in pos})
    consistent(r)
    assert r["extra"]["kinetic_energy_ratio"] == pytest.approx(1.0, abs=0.12)
    ev = [e for e in r["extra"]["events"] if e["valid"]]
    assert len(ev) == 1
    assert ev[0]["momentum_before"] == pytest.approx(900.0, rel=0.08)
    assert abs(ev[0]["t"] - 1.0123) < 0.05


def test_unequal_masses_and_inelastic_still_conserve_momentum():
    t, pos, info = collision_scene(m1=2.0, m2=1.0, e=0.5)
    masses = {k: 1.0 for k in pos}
    masses["A"] = 2.0
    r = collision_momentum(t, pos, masses=masses)
    consistent(r)
    expected_ke = (2.0 * info["v1p"] ** 2 + 1.0 * info["v2p"] ** 2) / (2.0 * 900.0 ** 2)
    assert r["extra"]["kinetic_energy_ratio"] == pytest.approx(expected_ke, rel=0.12)
    assert r["extra"]["kinetic_energy_ratio"] < 0.9  # inelastic: energy is lost, momentum is not


def test_reports_momentum_that_was_really_lost():
    t, pos, _ = collision_scene(after_factor=0.8)  # the target leaves 20% too slowly
    r = collision_momentum(t, pos, masses={k: 1.0 for k in pos})
    consistent(r, expected=-20.0)
    assert r["value"] < -10


@pytest.mark.parametrize("tc", [1.0, 1.0123, 1.02, 1.0333, 1.0499])
def test_does_not_depend_on_where_between_frames_the_impact_falls(tc):
    t, pos, _ = collision_scene(tc=tc, seed=3)
    consistent(collision_momentum(t, pos, masses={k: 1.0 for k in pos}))


def test_camera_shake_is_removed_using_the_still_objects():
    t, pos, _ = collision_scene(shake=4.0, seed=5)  # whole picture jitters ~4 px per frame
    r = collision_momentum(t, pos, masses={k: 1.0 for k in pos})
    assert r["fit_quality"]["stabilised"] is True
    consistent(r)


def test_without_enough_still_objects_it_does_not_stabilise_and_says_so():
    t, pos, _ = collision_scene(n_static=1)  # only 3 objects: a median cannot separate the camera
    r = collision_momentum(t, pos, masses={k: 1.0 for k in pos})
    assert r["fit_quality"]["stabilised"] is False
    assert any("camera" in w.lower() for w in r["warnings"])
    consistent(r)  # still fine when the camera really is still


def test_two_impacts_are_both_found_and_averaged():
    t, pos, _ = collision_scene(second=(2.0, 1.0), seconds=3.5, seed=2)
    r = collision_momentum(t, pos, masses={k: 1.0 for k in pos})
    valid = [e for e in r["extra"]["events"] if e["valid"]]
    assert len(valid) == 2
    consistent(r)


def test_a_weak_impact_is_rejected_with_a_reason_but_still_listed():
    t, pos, _ = collision_scene(second=(2.0, 0.25), seconds=3.5, seed=2, friction=0.15)  # A and B stop; second impact is 4x weaker
    r = collision_momentum(t, pos, masses={k: 1.0 for k in pos})
    events = r["extra"]["events"]
    weak = [e for e in events if not e["valid"]]
    assert len(weak) == 1 and weak[0]["reasons"]
    assert r["extra"]["n_valid"] == 1 and r["extra"]["n_events"] == 2
    assert any("rejected" in w.lower() for w in r["warnings"])


def test_missing_masses_assume_equal_and_say_so():
    t, pos, _ = collision_scene()
    r = collision_momentum(t, pos)
    assert any("equal masses" in w.lower() for w in r["warnings"])
    assert "relative" in r["extra"]["momentum_units"]
    ok = collision_momentum(t, pos, masses={k: 1.0 for k in pos})
    assert not any("equal masses" in w.lower() for w in ok["warnings"])


def test_masses_and_scale_give_si_units():
    t, pos, _ = collision_scene(v1=900.0)
    masses = {k: 1.0 for k in pos}
    masses["A"] = masses["B"] = 2.0
    r = collision_momentum(t, pos, masses=masses, scale_m_per_px=0.001)
    ev = [e for e in r["extra"]["events"] if e["valid"]][0]
    assert r["extra"]["momentum_units"] == "kg*m/s"
    assert ev["momentum_before"] == pytest.approx(2.0 * 0.9, rel=0.06)


def test_no_collision_gives_no_value():
    t = np.arange(0, 3, 1 / FPS)
    pos = {f"S{k}": (np.full_like(t, 100.0 + 90 * k), np.full_like(t, 400.0)) for k in range(6)}
    pos["S0"] = (100.0 + 50 * t, np.full_like(t, 400.0))  # one object glides smoothly, nothing hits anything
    r = collision_momentum(t, pos)
    assert r["value"] is None and r["warnings"]


def test_the_quoted_uncertainty_covers_the_truth_most_of_the_time():
    """Over many noisy clips (noise like real tracking), |error| <= 2 sigma at least 85% of the time."""
    hits, runs = 0, 30
    for seed in range(runs):
        t, pos, _ = collision_scene(noise=0.8, seed=100 + seed, tc=1.0 + 0.001 * seed, e=0.7, m1=1.5, m2=1.0)
        masses = {k: 1.0 for k in pos}
        masses["A"] = 1.5
        r = collision_momentum(t, pos, masses=masses)
        if r["value"] is not None and abs(r["value"]) <= 2 * r["uncertainty"]:
            hits += 1
    assert hits / runs >= 0.85


def test_camera_zoom_makes_the_impact_unreliable_and_it_is_rejected():
    t, pos, _ = collision_scene(seed=4)
    zoom = 1 + 0.5 * t  # the picture zooms in around x = 500: still objects drift apart by different amounts
    zoomed = {k: (500 + (x - 500) * zoom, y) for k, (x, y) in pos.items()}
    r = collision_momentum(t, zoomed, masses={k: 1.0 for k in pos})
    events = r["extra"]["events"]
    assert events and not any(e["valid"] for e in events)
    assert any("unstable scene" in reason for e in events for reason in e["reasons"])
    assert r["value"] is None


def test_systematic_camera_error_is_part_of_the_quoted_uncertainty():
    t, pos, _ = collision_scene(seed=4, noise=0.3)
    mild = 1 + 0.03 * t  # a small zoom: not enough to reject, enough to matter
    zoomed = {k: (500 + (x - 500) * mild, y) for k, (x, y) in pos.items()}
    plain = collision_momentum(t, pos, masses={k: 1.0 for k in pos})
    r = collision_momentum(t, zoomed, masses={k: 1.0 for k in pos})
    if r["value"] is not None:  # accepted: its error budget must show the camera term
        ev = [e for e in r["extra"]["events"] if e["valid"]][0]
        assert ev["systematic_camera_error"] > 0
        assert r["uncertainty"] >= plain["uncertainty"]


def test_positions_from_tracks_builds_arrays_with_gaps():
    from app.sam import TrackPoint
    from app.signals import positions_from_tracks
    pts = [TrackPoint(f, "a", 0, 0, 10, 10, 0.9, 5.0 + f, 5.0, 100.0) for f in (0, 1, 3)]
    t, pos, sizes = positions_from_tracks({"a": pts}, [0.0, 0.1, 0.2, 0.3])
    assert list(t) == [0.0, 0.1, 0.2, 0.3]
    x, y = pos["a"]
    assert x[0] == 5.0 and x[3] == 8.0 and np.isnan(x[2])
    assert sizes["a"] == pytest.approx(np.hypot(10, 10))
