"""Collision: momentum before and after an impact, from tracked positions.

For every impact we estimate each object's velocity just before and just after
the moment of contact (a short fit on each side, evaluated at the contact
time; a straight line unless the curvature is clearly real), then compare the total momentum  p = sum(m_i * v_i)  before and after.
The headline value is the percent change of |p|; for an isolated system it is 0.

Real clips make this hard, so the tool is careful:
- A handheld camera shifts every object. With 5 or more objects the shift is
  estimated as the median movement of all of them and removed. With fewer it cannot
  be, and the tool says so.
- Impacts are found automatically (a sudden change of velocities). Each is checked
  with tests on the INPUT only (is it strong, is the scene stable, do the fits
  match), never on whether the answer looks right. Rejected impacts stay in the list
  with their reasons.
- The answer is the mean over the valid impacts, with the standard error.
Pure maths: no AI is involved in any number produced here.
"""
import numpy as np
from scipy.signal import find_peaks

MIN_OBJECTS_TO_STABILISE = 5
WINDOW = 3  # frames on each side used to find the impact
PRE_FRAMES, POST_FRAMES = 7, 7  # frames on each side used to estimate velocities
DETECT_FRACTION = 0.15  # candidate impacts: a velocity change of at least 15% of the biggest one ...
STRENGTH_FRACTION = 0.4  # ... but only impacts carrying at least 40% of the strongest impact's momentum count
MAX_EVENTS = 40
MAX_SYSTEMATIC_FRACTION = 0.05  # camera distortion may put at most 5% systematic error into the momentum
MAX_RESIDUAL_FRACTION = 0.015  # the velocity fits may miss by at most 1.5% of the distance travelled


def _fill_holes(a):
    """Fill missing positions (NaN) by straight lines; hold the ends."""
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    if ok.sum() < 2:
        return None
    return np.interp(np.arange(len(a)), np.nonzero(ok)[0], a[ok])


def _velocity_at(t, y, tau):
    """Velocity at time tau: (v, sigma_v, rms residual, curvature).

    A straight line is used unless the curvature is clearly real (more than 3 sigma): the
    window usually starts a little after the contact, and a needless quadratic would
    amplify the noise when extrapolating back to it.
    """
    x = t - tau
    c2, cov2 = np.polyfit(x, y, 2, cov=True)
    if abs(c2[0]) > 3 * np.sqrt(cov2[0, 0]):
        resid = float(np.sqrt(np.mean((y - np.polyval(c2, x)) ** 2)))
        return float(c2[1]), float(np.sqrt(cov2[1, 1])), resid, float(c2[0])
    c1, cov1 = np.polyfit(x, y, 1, cov=True)
    resid = float(np.sqrt(np.mean((y - np.polyval(c1, x)) ** 2)))
    return float(c1[0]), float(np.sqrt(cov1[0, 0])), resid, 0.0


def _reference_objects(raw_v):
    """Indices of the objects that stay still the whole clip (they reveal the camera's movement).

    An object is still if, after removing the median movement of all objects, what is left
    over is about the size of tracking noise: within 2.5x the noise level, taken as the median
    of the quietest half of the objects. Fewer than 3, or fewer than half of all objects, means
    the camera cannot be separated from real motion.
    """
    resid = raw_v - np.median(raw_v, axis=0)[None]
    rms = np.sqrt(np.mean(np.sum(resid ** 2, axis=-1), axis=1))
    noise_level = float(np.median(np.sort(rms)[:max(1, (len(rms) + 1) // 2)]))
    ref = np.nonzero(rms <= 2.5 * noise_level + 1e-9)[0]
    return ref if len(ref) >= 3 and len(ref) * 2 >= len(rms) else None


def _scene_spread(P, t, f):
    """How much the objects disagree about the camera movement around frame f, in px/s.

    Each object's average velocity over the whole window is compared with the median of all
    objects; the typical (median) leftover is the spread. Tracking noise averages away over
    the window, while camera zoom or rotation (objects shifting by different amounts) does not.
    """
    lo, hi = max(0, f - PRE_FRAMES + 1), min(len(t) - 1, f + POST_FRAMES)
    v = (P[:, hi] - P[:, lo]) / (t[hi] - t[lo])
    return float(np.hypot(*np.median(np.abs(v - np.median(v, axis=0)), axis=0)))


def _empty(warning, warnings=None):
    return {"value": None, "uncertainty": None, "units": "%",
            "method": "momentum before vs after each impact", "fit_quality": {},
            "series_for_chart": {}, "warnings": (warnings or []) + [warning], "extra": {}}


def collision_momentum(t, positions, masses=None, scale_m_per_px=None, sizes=None, min_separation_s=0.2):
    """Momentum change across impact(s) in a clip.

    t: time of every frame (s). positions: {object id: (x, y)} arrays (NaN = not found).
    masses: {object id: kg}; missing = all equal (only valid if they really are equal).
    scale_m_per_px: metres per pixel, for momentum in kg*m/s (the percent change needs no scale).
    sizes: {object id: pixels}; used to warn when objects are at very different distances.
    """
    t = np.asarray(t, float)
    ids = [k for k in positions if _fill_holes(positions[k][0]) is not None]
    if len(ids) < 2 or len(t) < 20:
        return _empty("Need at least two tracked objects and about a second of video.")
    P = np.stack([np.stack([_fill_holes(positions[k][0]), _fill_holes(positions[k][1])], axis=-1) for k in ids])
    K, N = len(ids), len(t)
    dt = float(np.median(np.diff(t)))
    fps = 1.0 / dt
    warnings = []

    # ---- camera: remove the movement all objects share ----
    raw_v = np.diff(P, axis=1) / np.diff(t)[None, :, None]  # (K, N-1, 2) px/s
    ref = _reference_objects(raw_v) if K >= MIN_OBJECTS_TO_STABILISE else None
    stabilised = ref is not None
    if stabilised:
        # camera movement = the plain MEAN of the still objects (a median of a few values is slightly
        # biased when one object moves fast, and that bias would be added up over all objects)
        cam_v = raw_v[ref].mean(axis=0)
        shift = np.vstack([[0.0, 0.0], np.cumsum(cam_v * np.diff(t)[:, None], axis=0)])
        S = P - shift[None]
        cam_speed = float(np.mean(np.hypot(*cam_v.T)))
    else:
        S, spread, cam_speed = P, np.zeros(N - 1), 0.0
        warnings.append(f"The camera's movement could not be separated from the objects' own movement ({K} objects "
                        f"tracked; at least {MIN_OBJECTS_TO_STABILISE} are needed, most of them staying still), so this "
                        "is only valid if the camera was completely still.")

    # ---- find impacts: a sudden change of the velocities of all objects together ----
    V = np.gradient(S, t, axis=1)

    def impact_score(idx):
        """Sum over objects of how much their velocity changes across each frame (px/s)."""
        sc = np.zeros(N)
        for f in range(WINDOW + 1, N - WINDOW - 1):
            after = V[idx, f + 1:f + 1 + WINDOW].mean(axis=1)
            before = V[idx, f - WINDOW:f].mean(axis=1)
            sc[f] = float(np.sum(np.hypot(*(after - before).T)))
        return sc

    score = impact_score(np.arange(K))
    if stabilised:
        # the still objects only change velocity through tracking noise: their typical score sets the
        # floor (a robust level, so that occasional camera jerks do not push it above the real impacts)
        sref = impact_score(ref)[WINDOW + 1:N - WINDOW - 1]
        noise = (K / len(ref)) * float(np.median(sref) + 6 * 1.4826 * np.median(np.abs(sref - np.median(sref))))
    else:
        noise = float(np.median(score) + 8 * 1.4826 * np.median(np.abs(score - np.median(score))))
    noise = max(noise, 0.25 * fps)  # a velocity change of under a quarter pixel per frame is not an impact
    if score.max() <= 0:
        return _empty("No impact was found in this clip.", warnings)
    height = max(DETECT_FRACTION * score.max(), noise)
    peaks, props = find_peaks(score, height=height, prominence=0.5 * height,
                              distance=max(2, int(min_separation_s * fps)))
    if len(peaks) > MAX_EVENTS:  # keep the strongest candidates, in time order
        peaks = np.sort(peaks[np.argsort(-props["peak_heights"])[:MAX_EVENTS]])
    if len(peaks) == 0:
        return _empty("No impact was found in this clip (no sudden change in how the objects move).", warnings)

    # ---- masses and units ----
    if masses:
        m = np.array([float(masses.get(k, 1.0)) for k in ids])
        if any(k not in masses for k in ids):
            warnings.append("Some objects had no mass given; they were treated as 1.")
    else:
        m = np.ones(K)
        warnings.append("No masses were given, so all objects are assumed to have EQUAL masses; the percent change "
                        "is only valid if that is true.")
    if scale_m_per_px:
        unit_scale, units = float(scale_m_per_px), "kg*m/s" if masses else "relative (equal masses, m/s)"
    else:
        unit_scale, units = 1.0, "kg*px/s" if masses else "relative (equal masses, px/s)"

    # ---- one estimate per impact ----
    movers = np.ones(K)
    if stabilised:
        movers[ref] = 0.0  # still objects only carry the camera estimate's error
    mw = m * movers

    def measure(g):
        """Everything about an impact between frames g and g+1: momenta, uncertainties, fit quality."""
        tc = float((t[g] + t[g + 1]) / 2)
        vel = {}
        for side, fr in (("before", np.arange(g - PRE_FRAMES + 1, g + 1)), ("after", np.arange(g + 1, g + 1 + POST_FRAMES))):
            v, sv, res, curv = np.zeros((K, 2)), np.zeros((K, 2)), np.zeros(K), np.zeros((K, 2))
            for i in range(K):
                for ax in (0, 1):
                    v[i, ax], sv[i, ax], r, curv[i, ax] = _velocity_at(t[fr], S[i, fr, ax], tc)
                    res[i] = max(res[i], r)
            # the contact moment falls somewhere inside one frame interval: add that uncertainty
            vel[side] = (v, np.sqrt(sv ** 2 + (np.abs(curv) * dt / np.sqrt(3)) ** 2), res)
        (vb, sb, rb), (va, sa, ra) = vel["before"], vel["after"]
        # Objects that stay still all clip have zero momentum; including them would only add the
        # error of the camera estimate (the same for all objects, so it would add up coherently).
        pb, pa = (mw[:, None] * vb).sum(axis=0), (mw[:, None] * va).sum(axis=0)
        spb = float(np.sqrt(np.sum((mw[:, None] * sb) ** 2)))
        spa = float(np.sqrt(np.sum((mw[:, None] * sa) ** 2)))
        if stabilised:
            # the camera correction is the same for every moving object, so its error adds up
            # coherently: total moving mass x the error of the still objects' average velocity
            spb = float(np.hypot(spb, mw.sum() * np.sqrt(np.sum(np.mean(sb[ref] ** 2, axis=0) / len(ref)))))
            spa = float(np.hypot(spa, mw.sum() * np.sqrt(np.sum(np.mean(sa[ref] ** 2, axis=0) / len(ref)))))
        nb, na = float(np.hypot(*pb)), float(np.hypot(*pa))
        sp_b, sp_a = np.hypot(*vb.T), np.hypot(*va.T)
        peak = float(max(sp_b.max(), sp_a.max()))
        travel = peak * PRE_FRAMES * dt  # distance the fastest object covers in a fit window
        big = np.concatenate([rb[sp_b > 0.3 * peak], ra[sp_a > 0.3 * peak]]) if peak > 0 else np.zeros(1)
        out = {"t": tc, "nb": nb, "na": na, "spb": spb, "spa": spa, "peak": peak,
               "jump": float(np.sum(np.hypot(*(va - vb).T))),
               "resid": float(big.max() / travel) if peak > 0 else 0.0,
               "ke": float(np.sum(mw * sp_a ** 2) / max(np.sum(mw * sp_b ** 2), 1e-12)),
               "before_objects": [(ids[i], float(sp_b[i] * unit_scale)) for i in np.argsort(-sp_b)[:2]],
               "after_objects": [(ids[i], float(sp_a[i] * unit_scale)) for i in np.argsort(-sp_a)[:2]]}
        if nb > 3 * spb:
            ratio = na / nb
            out["percent_change"] = float(100 * (ratio - 1))
            out["sigma"] = float(100 * ratio * np.sqrt((spb / nb) ** 2 + (spa / max(na, 1e-9)) ** 2))
        return out

    events = []
    for f in peaks:
        f = int(f)
        ev = {"frame": f, "t": float((t[f] + t[f + 1]) / 2), "valid": True, "reasons": []}
        near = [g for g in range(f - 2, f + 3) if g - PRE_FRAMES + 1 >= 0 and g + 1 + POST_FRAMES <= N]
        if f not in near:
            ev.update(valid=False, reasons=["too close to the start or end of the clip to measure both sides"])
            events.append(ev)
            continue
        # Which frame holds the contact? At the right one the velocity jump between the two fit windows is
        # largest (a wrong frame makes a window straddle the impact and blends the two velocities). The
        # data may not settle it exactly, so every frame whose jump is within 5% of the best is also
        # measured, and the spread of their answers is added to the uncertainty.
        meas = {g: measure(g) for g in near}
        best = max(meas, key=lambda g: meas[g]["jump"])
        plausible = {g: x for g, x in meas.items() if x["jump"] >= 0.95 * meas[best]["jump"]}
        mb = meas[best]
        ev.update(frame=best, t=mb["t"], momentum_before=mb["nb"] * unit_scale, momentum_after=mb["na"] * unit_scale,
                  kinetic_energy_ratio=mb["ke"], before_objects=mb["before_objects"], after_objects=mb["after_objects"],
                  _sys=(_scene_spread(P, t, best) * float(mw.sum()) / mb["nb"]) if stabilised and mb["nb"] > 0 else 0.0,
                  _peak=mb["peak"], _resid=mb["resid"])
        if "percent_change" in mb:
            alt = [x["percent_change"] for x in plausible.values() if "percent_change" in x]
            frame_sigma = float(np.std(alt)) if len(alt) >= 2 else 0.0
            # camera zoom/rotation shifts objects by different amounts: the leftover disagreement of the
            # still objects, times the moving mass, estimates the systematic error in the momentum
            sys_sigma = 100.0 * (mb["na"] / mb["nb"]) * ev["_sys"]
            ev.update(percent_change=mb["percent_change"], contact_frame_ambiguity=frame_sigma,
                      systematic_camera_error=sys_sigma,
                      uncertainty=float(np.sqrt(mb["sigma"] ** 2 + frame_sigma ** 2 + sys_sigma ** 2)))
        else:
            ev.update(valid=False, reasons=["no measurable momentum before the impact (the total is within "
                                            "its own uncertainty of zero)"])
        events.append(ev)

    # ---- gate each impact on its input quality ----
    measured = [e for e in events if "momentum_before" in e]
    strongest = max((e["momentum_before"] for e in measured), default=0.0)
    for e in measured:
        if len(measured) >= 2 and e["momentum_before"] < STRENGTH_FRACTION * strongest:
            e["valid"] = False
            e["reasons"].append(f"weak: carries only {100 * e['momentum_before'] / strongest:.0f}% of the strongest "
                                "impact's momentum, so it is probably not a real impact")
        if stabilised and e["_sys"] > MAX_SYSTEMATIC_FRACTION:
            e["valid"] = False
            e["reasons"].append(f"unstable scene: camera zoom or rotation makes the still objects disagree, which could "
                                f"put about {100 * e['_sys']:.0f}% systematic error into the momentum")
        if e["_resid"] > MAX_RESIDUAL_FRACTION:
            e["valid"] = False
            e["reasons"].append(f"poor velocity fit (misses by {100 * e['_resid']:.0f}% of the distance travelled)")
    for e in events:
        for k in ("_sys", "_peak", "_resid"):
            e.pop(k, None)

    valid = [e for e in events if e["valid"]]
    n_events = len(events)
    extra = {"events": events, "n_events": n_events, "n_valid": len(valid), "momentum_units": units,
             "masses_given": bool(masses)}
    if not valid:
        out = _empty("Every impact found was rejected (see the reasons); no reliable momentum change can be given.", warnings)
        out["extra"] = extra
        out["fit_quality"] = {"stabilised": stabilised, "n_objects": K}
        return out

    pcs = np.array([e["percent_change"] for e in valid])
    sig = np.array([e["uncertainty"] for e in valid])
    n = len(valid)
    unc = float(max(np.std(pcs, ddof=1) / np.sqrt(n) if n >= 2 else 0.0, np.sqrt(np.mean(sig ** 2) / n)))
    extra.update(median_percent_change=float(np.median(pcs)),
                 kinetic_energy_ratio=float(np.mean([e["kinetic_energy_ratio"] for e in valid])),
                 momentum_before=float(np.mean([e["momentum_before"] for e in valid])),
                 momentum_after=float(np.mean([e["momentum_after"] for e in valid])),
                 spread_between_impacts=float(np.std(pcs, ddof=1)) if n >= 2 else None)
    if n_events > n:
        warnings.append(f"{n_events - n} of {n_events} impacts were rejected for poor input quality "
                        "(weak, unstable scene or poor fit); only the {} valid ones are used.".format(n))
    if stabilised and cam_speed > 0:
        warnings.append(f"The camera moved (about {cam_speed:.0f} px/s on average); its movement was removed "
                        "using the objects that stayed still, which handles shifting but not zoom or rotation.")
    if sizes and len(sizes) >= 2 and max(sizes.values()) / max(min(sizes.values()), 1e-9) > 1.2:
        warnings.append("The objects look different in size on screen (different distances from the camera), so "
                        "their pixel speeds are not on the same scale; momentum can be off by roughly that ratio.")
    peak_move = max(max(np.hypot(*(np.diff(S[i], axis=0).T)).max() for i in range(K)), 0.0)
    if sizes and peak_move > 0.3 * float(np.median(list(sizes.values()))):
        warnings.append("The objects move more than a third of their size between frames, so each impact happens "
                        "between two frames; film in slow motion for sharper numbers.")

    return {"value": float(pcs.mean()), "uncertainty": unc, "units": "%",
            "method": "total momentum after vs before each impact, from velocity fits at the contact time; "
                      "mean over valid impacts with standard error",
            "fit_quality": {"stabilised": stabilised, "n_objects": K, "camera_speed_px_per_s": cam_speed,
                            "impacts_valid": n, "impacts_found": n_events, "frame_rate": fps},
            "series_for_chart": {"t": (t - t[0]).tolist(),
                                 "speeds": {str(ids[i]): np.hypot(*V[i].T).tolist() for i in range(K)},
                                 "event_t": [e["t"] - float(t[0]) for e in events],
                                 "event_valid": [e["valid"] for e in events],
                                 "p_before": [e.get("momentum_before") for e in events],
                                 "p_after": [e.get("momentum_after") for e in events]},
            "warnings": warnings, "extra": extra}
