"""Pendulum period: fit a damped sinusoid to a tracked position signal.

Model: x(t) = A * exp(-b t) * cos(2 pi t / T + phi) + c
The period T comes from a least-squares fit. If the string length L is known,
g = 4 pi^2 L / T^2. Pure maths: no AI is involved in any number produced here.

Two things make the answer trustworthy on real video:
1. Swing detection: set-up motion (a hand pushing the bob, a moving camera) is
   cut off, and only the part where the object swings cleanly is fitted.
2. An honest uncertainty: the largest of three independent estimates (fit
   covariance corrected for correlated noise, cycle-to-cycle scatter, and the
   frame-timing floor), so the "+/-" is never optimistic.
"""
import numpy as np

from .oscillation import MIN_CYCLES, MIN_R2, _fit, _honest_uncertainty, _model, find_swing_segment

G_STANDARD = 9.81


def _empty(warning):
    return {"value": None, "uncertainty": None, "units": "s",
            "method": "damped sinusoid fit", "fit_quality": {},
            "series_for_chart": {}, "warnings": [warning], "extra": {}}


def pendulum_period(t, x, y, length_m=None, length_uncertainty_m=None, auto_segment=True):
    """Return the swing period (and g if the string length is given).

    t: timestamps in seconds. x, y: tracked position (pixels or metres).
    The swing happens mostly along one axis, so we fit the axis that moves most.
    With auto_segment, only the part of the clip where the object swings
    cleanly is used; the chart series still contains the whole clip.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    if len(t) < 10:
        return _empty("Not enough tracked frames to fit a period.")

    axis = "x" if np.std(x) >= np.std(y) else "y"
    s = x if axis == "x" else y
    if np.std(s) < 1e-6:
        return _empty("The tracked object does not move, so no period can be measured.")

    seg = find_swing_segment(t, s) if auto_segment else None
    i0, i1 = seg if seg else (0, len(t) - 1)
    ts, ss = t[i0:i1 + 1], s[i0:i1 + 1]
    trimmed = bool(seg) and (i0 > 0 or i1 < len(t) - 1)
    t0 = ts - ts[0]

    best = _fit(t0, ss)
    if best is None:
        return _empty("The fit did not converge; the motion may not be a clean oscillation.")
    sse, popt, pcov = best
    T = float(popt[2])
    sigma_T, comps = _honest_uncertainty(t0, ss, popt, pcov, _model(t0, *popt))
    ss_tot = float(np.sum((ss - ss.mean()) ** 2))
    r2 = 1 - sse / ss_tot if ss_tot > 0 else 0.0
    cycles = float(t0[-1] / T)

    warnings = []
    if trimmed:
        warnings.append(
            f"Used only {ts[0] - t[0]:.1f} s to {ts[-1] - t[0]:.1f} s of the clip, where the object swings cleanly; "
            "the rest looks like set-up motion (a push, a moving camera) and was left out.")
    if cycles < MIN_CYCLES:
        warnings.append(f"Only {cycles:.1f} cycles were recorded; film at least {MIN_CYCLES} full cycles for a reliable period.")
    if r2 < MIN_R2:
        warnings.append(f"The fit quality is low (R2 = {r2:.2f}); the tracking may be noisy or the motion is not a clean swing.")

    chart_fit = [None] * len(t)
    chart_fit[i0:i1 + 1] = _model(t0, *popt).tolist()
    result = {
        "value": T,
        "uncertainty": sigma_T,
        "units": "s",
        "method": "damped sinusoid fit x(t) = A e^(-bt) cos(2 pi t / T + phi) + c",
        "fit_quality": {"r2": r2, "cycles": cycles, "axis": axis,
                        "damping_rate_per_s": float(popt[1]), "n_points": int(len(ts)),
                        "equilibrium": float(popt[4]),
                        "segment": {"start_s": float(ts[0] - t[0]), "end_s": float(ts[-1] - t[0]),
                                    "trimmed": trimmed},
                        "uncertainty_components": comps,
                        "uncertainty_method": "largest of: fit covariance corrected for correlated noise, "
                                              "cycle-to-cycle scatter, frame-timing floor"},
        "series_for_chart": {"t": (t - t[0]).tolist(), "measured": s.tolist(), "fitted": chart_fit},
        "warnings": warnings,
        "extra": {},
    }

    if length_m:
        g = 4 * np.pi**2 * length_m / T**2
        rel = (2 * sigma_T / T) ** 2
        if length_uncertainty_m:
            rel += (length_uncertainty_m / length_m) ** 2
        result["extra"]["g"] = {
            "value": float(g),
            "uncertainty": float(g * np.sqrt(rel)),
            "units": "m/s^2",
            "method": "g = 4 pi^2 L / T^2 with propagated uncertainty",
            "standard": G_STANDARD,
            "percent_error_vs_standard": float(abs(g - G_STANDARD) / G_STANDARD * 100),
        }
        if not 9.0 <= g <= 10.6:
            result["warnings"].append(
                f"g = {g:.2f} m/s^2 is outside the plausible range (9.0 to 10.6); check the string length (to the bob's centre) and the tracking.")
    return result
