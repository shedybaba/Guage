"""Spring and mass: oscillation frequency, damping ratio, and the spring constant k.

Fit  x(t) = A exp(-b t) cos(omega_d t + phi) + c  to the tracked position (same machinery as the
pendulum: clean-swing detection and an honest period uncertainty). From the fit:

  omega_d = 2 pi / T                      what the camera sees (the damped frequency)
  omega_n = sqrt(omega_d^2 + b^2)         the spring's own frequency, without damping
  zeta    = b / omega_n                   damping ratio (0 = never stops, 1 = no oscillation at all)
  k       = m_eff * omega_n^2             spring constant; m_eff = m + (spring mass) / 3

k belongs to omega_n, not to the visible frequency, so heavy damping would otherwise make k too small.
Pure maths: no AI is involved in any number produced here.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .oscillation import (MIN_CYCLES, MIN_R2, _fit, _honest_uncertainty, _model, find_swing_segment,
                          no_clear_oscillation)

MAX_DAMPING = 30.0  # fastest decay (1/s) the fit may find: springs die out faster than pendulums
FEW_CYCLES = 5  # the damping needs several cycles to be pinned down
MIN_FRAMES_PER_CYCLE = 6
MIN_PEAKS_FOR_ENVELOPE = 5


def _empty(warning):
    return {"value": None, "uncertainty": None, "units": "Hz", "method": "damped sinusoid fit",
            "fit_quality": {}, "series_for_chart": {}, "warnings": [warning], "extra": {}}


def _sub(value, uncertainty, units, method, **more):
    return {"value": float(value), "uncertainty": float(uncertainty), "units": units, "method": method, **more}


def _envelope(t0, s, centre, T, noise):
    """Look at the peaks themselves: how fast does their height fall, and in what shape?

    An independent check of the damping from the fit: exponential decay (viscous damping) is a
    straight line in ln(height); sliding friction makes the height itself fall in a straight line.
    """
    d = s - centre
    dt = float(np.median(np.diff(t0)))
    dist = max(1, int(0.35 * T / dt))
    peaks = []
    for sign in (1, -1):
        idx, _ = find_peaks(sign * d, distance=dist)
        for i in idx:
            if 0 < i < len(d) - 1:  # parabola through three points gives the true peak height between frames
                y0, y1, y2 = sign * d[i - 1], sign * d[i], sign * d[i + 1]
                denom = y0 - 2 * y1 + y2
                peak = y1 - (y2 - y0) ** 2 / (8 * denom) if denom != 0 else y1
                peaks.append((float(t0[i]), float(peak)))
    peaks = sorted(p for p in peaks if p[1] > 4 * noise)  # lose peaks that have sunk into the noise
    out = {"type": "undetermined", "n_peaks": len(peaks), "decay_rate": None, "decay_rate_uncertainty": None,
           "r2_exponential": None, "r2_linear": None}
    if len(peaks) < MIN_PEAKS_FOR_ENVELOPE:
        return out
    tp, hp = np.array([p[0] for p in peaks]), np.array([p[1] for p in peaks])
    n = len(tp)

    def regress(y):
        A = np.stack([tp, np.ones(n)], axis=1)
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        se = np.sqrt(sse / (n - 2) / np.sum((tp - tp.mean()) ** 2)) if n > 2 else float("nan")
        return coef, (1 - sse / sst if sst > 0 else 0.0), float(se)

    ce, r2e, se_e = regress(np.log(hp))
    cl, r2l, _ = regress(hp)
    # Compare the two shapes where the noise lives (the heights themselves), each with two parameters:
    # a shape is called only if it fits at least twice as well as the other.
    sse_lin = float(np.sum((hp - (cl[0] * tp + cl[1])) ** 2))
    try:
        (a_e, b_e), _ = curve_fit(lambda tt, a_, b_: a_ * np.exp(-b_ * tt), tp, hp, p0=[np.exp(ce[1]), -ce[0]], maxfev=5000)
        sse_exp = float(np.sum((hp - a_e * np.exp(-b_e * tp)) ** 2))
    except (RuntimeError, ValueError):
        sse_exp = float("inf")
    out.update(decay_rate=float(-ce[0]), decay_rate_uncertainty=se_e, r2_exponential=r2e, r2_linear=r2l,
               sse_exponential=sse_exp, sse_linear=sse_lin)
    drop = (hp[0] - hp[-1]) / hp[0]
    if drop < 0.2:
        out["type"] = "none measurable (the height barely falls)"
    elif sse_lin < 0.5 * sse_exp:
        out["type"] = "linear (friction-like)"
    elif sse_exp < 0.5 * sse_lin:
        out["type"] = "exponential (viscous)"
    else:
        out["type"] = "undetermined (a line and an exponential fit about equally well)"
    return out


def oscillation_frequency(t, x, y, mass_kg=None, mass_uncertainty_kg=None, spring_mass_kg=None,
                          scale_m_per_px=None, auto_segment=True):
    """Frequency, damping ratio and (given the mass) spring constant of a spring-mass oscillation.

    t: timestamps (s). x, y: tracked position (pixels). The motion is along one axis (usually
    vertical), so the axis that moves most is fitted. mass_kg: the hanging/sliding mass.
    spring_mass_kg: the spring's own mass (a third of it moves with the mass). scale_m_per_px: metres per
    pixel, for amplitude and top speed.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    if len(t) < 10:
        return _empty("Not enough tracked frames to fit an oscillation.")
    axis = "x" if np.std(x) >= np.std(y) else "y"
    s = x if axis == "x" else y
    if np.std(s) < 1e-6:
        return _empty("The tracked object does not move, so no oscillation can be measured.")

    seg = find_swing_segment(t, s) if auto_segment else None
    i0, i1 = seg if seg else (0, len(t) - 1)
    ts, ss = t[i0:i1 + 1], s[i0:i1 + 1]
    trimmed = bool(seg) and (i0 > 0 or i1 < len(t) - 1)
    t0 = ts - ts[0]

    best = _fit(t0, ss, max_damping=MAX_DAMPING)
    if best is None:
        return _empty("The fit did not converge; the motion may not be a clean oscillation.")
    sse, popt, pcov = best
    A, b, T, phi, c = (float(v) for v in popt)
    sigma_T, comps = _honest_uncertainty(t0, ss, popt, pcov, _model(t0, *popt))
    ss_tot = float(np.sum((ss - ss.mean()) ** 2))
    r2 = 1 - sse / ss_tot if ss_tot > 0 else 0.0
    cycles = float(t0[-1] / T)
    unclear = no_clear_oscillation(popt, sse, len(ss), r2)
    if unclear:
        return _empty(unclear)
    dt = float(np.median(np.diff(t0)))
    fps = 1.0 / dt
    noise = float(np.sqrt(sse / len(ss)))

    # ---- decay rate b: the fit's covariance (widened for correlated noise) and, independently, the peaks ----
    infl = comps["noise_inflation"]
    sb_fit = float(np.sqrt(pcov[1, 1])) * infl if np.isfinite(pcov[1, 1]) else float("inf")
    env = _envelope(t0, ss, c, T, noise)
    sb_env = env["decay_rate_uncertainty"]
    sigma_b = max(sb_fit, sb_env if sb_env is not None and np.isfinite(sb_env) else 0.0)
    if not np.isfinite(sigma_b):
        sigma_b = abs(b) + 1.0
    corr = pcov[1, 2] / np.sqrt(pcov[1, 1] * pcov[2, 2]) if pcov[1, 1] > 0 and pcov[2, 2] > 0 else 0.0
    corr = float(np.clip(corr, -1, 1))

    # ---- frequencies, damping ratio (error propagation through the covariance of b and T) ----
    wd = 2 * np.pi / T
    wn = float(np.hypot(wd, b))
    s_wd = 2 * np.pi * sigma_T / T ** 2
    cov = np.array([[sigma_b ** 2, -corr * sigma_b * s_wd], [-corr * sigma_b * s_wd, s_wd ** 2]])  # (b, omega_d)
    j_wn = np.array([b / wn, wd / wn])
    j_z = np.array([wd ** 2 / wn ** 3, -b * wd / wn ** 3])
    s_wn = float(np.sqrt(j_wn @ cov @ j_wn))
    zeta = b / wn
    s_zeta = float(np.sqrt(j_z @ cov @ j_z))
    f_d, s_fd = 1.0 / T, sigma_T / T ** 2
    measurable = b > 2 * sigma_b

    warnings = []
    if trimmed:
        warnings.append(f"Used only {ts[0] - t[0]:.1f} s to {ts[-1] - t[0]:.1f} s of the clip, where the spring "
                        "oscillates cleanly; the rest looks like set-up motion (a hand, a moving camera) and was left out.")
    if cycles < MIN_CYCLES:
        warnings.append(f"Only {cycles:.1f} cycles were recorded; film at least {MIN_CYCLES} full cycles for a reliable frequency.")
    elif cycles < FEW_CYCLES:
        warnings.append(f"Only {cycles:.1f} cycles were recorded; the damping ratio needs about {FEW_CYCLES} or more to be pinned down.")
    if r2 < MIN_R2:
        warnings.append(f"The fit quality is low (R2 = {r2:.2f}); the tracking may be noisy or the motion is not a clean oscillation.")
    if fps / f_d < MIN_FRAMES_PER_CYCLE:
        warnings.append(f"Only {fps / f_d:.1f} frames per cycle: too few to time the oscillation precisely, and a faster "
                        "motion could be mistaken for this one (aliasing). Film in slow motion.")
    if env["type"].startswith("linear"):
        warnings.append("The height of the oscillation falls in a straight line, like sliding friction, not exponentially. "
                        "The damping ratio assumes viscous damping, so treat it as approximate.")
    if env["decay_rate"] is not None and np.isfinite(env["decay_rate_uncertainty"] or np.nan):
        gap = abs(env["decay_rate"] - b)
        if gap > 3 * np.hypot(sb_fit, env["decay_rate_uncertainty"]) and gap > 0.05 * max(b, 0.1):
            warnings.append(f"The two ways of measuring the damping disagree (fit {b:.3f}/s, peak heights "
                            f"{env['decay_rate']:.3f}/s); the damping ratio is less certain than it looks.")

    chart_fit = [None] * len(t)
    chart_fit[i0:i1 + 1] = _model(t0, *popt).tolist()
    upper, lower = [None] * len(t), [None] * len(t)
    upper[i0:i1 + 1] = (c + A * np.exp(-b * t0)).tolist()
    lower[i0:i1 + 1] = (c - A * np.exp(-b * t0)).tolist()

    damping = _sub(zeta, s_zeta, "(no units)", "b / sqrt(omega_d^2 + b^2) from the damped-sinusoid fit",
                   measurable=bool(measurable), decay_rate=_sub(b, sigma_b, "1/s", "fit"),
                   time_constant=(_sub(1 / b, sigma_b / b ** 2, "s", "1 / b: time for the amplitude to fall to 1/e")
                                  if measurable else None),
                   quality_factor=(_sub(1 / (2 * zeta), s_zeta / (2 * zeta ** 2), "(no units)", "1 / (2 zeta)")
                                   if measurable and zeta > 0 else None))
    if not measurable:
        damping["upper_bound"] = float((max(b, 0.0) + 2 * sigma_b) / wn)
        warnings.append("No damping could be measured (the amplitude barely changed); only an upper limit is given.")

    result = {
        "value": f_d, "uncertainty": s_fd, "units": "Hz",
        "method": "damped sinusoid fit x(t) = A e^(-bt) cos(omega_d t + phi) + c; f = 1/T",
        "fit_quality": {"r2": r2, "cycles": cycles, "axis": axis, "n_points": int(len(ts)),
                        "frames_per_cycle": fps / f_d,
                        "segment": {"start_s": float(ts[0] - t[0]), "end_s": float(ts[-1] - t[0]), "trimmed": trimmed},
                        "uncertainty_components": comps,
                        "uncertainty_method": "frequency: from the period, whose uncertainty is the largest of fit covariance "
                                              "(widened for correlated noise), cycle scatter and frame timing; "
                                              "damping: the larger of fit covariance (widened) and the peak-height regression"},
        "series_for_chart": {"t": (t - t[0]).tolist(), "measured": s.tolist(), "fitted": chart_fit,
                             "envelope_upper": upper, "envelope_lower": lower},
        "warnings": warnings,
        "extra": {
            "natural_frequency": _sub(wn / (2 * np.pi), s_wn / (2 * np.pi), "Hz", "sqrt(f_d^2 + (b / 2 pi)^2): without damping"),
            "damping_ratio": damping,
            "equilibrium": float(c),
            "damping_envelope": env,
        },
    }

    if scale_m_per_px:
        a_m = A * scale_m_per_px
        s_a = float(np.sqrt(pcov[0, 0])) * infl * scale_m_per_px if np.isfinite(pcov[0, 0]) else 0.0
        result["extra"]["amplitude"] = _sub(a_m, s_a, "m", "A of the fit, at the start of the fitted stretch")
        result["extra"]["max_speed"] = _sub(a_m * wd, float(np.hypot(s_a * wd, a_m * s_wd)), "m/s", "A * omega_d")

    if mass_kg:
        extra_mass = (spring_mass_kg or 0.0) / 3.0
        m_eff = mass_kg + extra_mass
        s_m = float(np.hypot(mass_uncertainty_kg or 0.0, 0.0))
        k = m_eff * wn ** 2
        rel = np.hypot(s_m / m_eff, 2 * s_wn / wn)
        result["extra"]["spring_constant"] = _sub(
            k, k * rel, "N/m", "k = (m + spring mass / 3) * omega_n^2, with propagated uncertainty",
            effective_mass_kg=float(m_eff))
        if not spring_mass_kg:
            warnings.append("The spring's own mass was not given, so only the hanging mass is used; a spring's own mass "
                            "adds about a third of itself to the moving mass, which makes k slightly larger.")
    else:
        warnings.append("No mass was given, so the spring constant k cannot be calculated: tell Gauge the mass.")
    return result
