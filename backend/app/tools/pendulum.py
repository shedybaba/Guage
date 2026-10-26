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
from scipy.optimize import curve_fit

G_STANDARD = 9.81
MIN_CYCLES = 2
MIN_R2 = 0.9
WINDOW_CYCLES = 3  # swing detection looks at windows 3 periods long
CLEAN_RESIDUAL = 0.2  # window is a clean swing if residual < 20% of its amplitude
MIN_AMPLITUDE_FRACTION = 0.25  # ... and its amplitude is at least 25% of the biggest swing


def _model(t, A, b, T, phi, c):
    return A * np.exp(-b * t) * np.cos(2 * np.pi * t / T + phi) + c


def _initial_period(t, s, min_cycles=0):
    """Guess the period from the FFT peak of the signal on a uniform grid."""
    grid = np.linspace(t[0], t[-1], len(t))
    u = np.interp(grid, t, s - s.mean())
    spectrum = np.abs(np.fft.rfft(u * np.hanning(len(u))))
    freqs = np.fft.rfftfreq(len(u), d=(grid[-1] - grid[0]) / (len(u) - 1))
    spectrum[0] = 0
    if min_cycles:
        spectrum[freqs < min_cycles / (grid[-1] - grid[0])] = 0
    f = freqs[int(np.argmax(spectrum))]
    return 1.0 / f if f > 0 else None


def _empty(warning):
    return {"value": None, "uncertainty": None, "units": "s",
            "method": "damped sinusoid fit", "fit_quality": {},
            "series_for_chart": {}, "warnings": [warning], "extra": {}}


def find_swing_segment(t, s):
    """Find where the signal swings cleanly. Returns (start_index, end_index) or None.

    Slides a window 3 periods long along the signal. In each window we fit a
    plain sinusoid (plus offset and a straight-line drift) by linear least
    squares, trying frequencies near the expected one. A window is "clean" when
    the leftover error is small compared with the swing size and the swing is
    big. The longest run of clean windows is the segment.
    """
    t = np.asarray(t, float)
    s = np.asarray(s, float)
    duration = t[-1] - t[0]
    T0 = _initial_period(t, s, min_cycles=2)
    if T0 is None or duration < (WINDOW_CYCLES + 1) * T0:
        return None  # too short to look for a segment: use everything
    win, step = WINDOW_CYCLES * T0, T0 / 4
    freqs = np.linspace(0.88, 1.12, 13) / T0

    starts, amps, resid = [], [], []
    a = t[0]
    while a + win <= t[-1] + 1e-9:
        m = (t >= a) & (t < a + win)
        if m.sum() >= 12:
            tt, ss = t[m] - a - win / 2, s[m]
            best = None
            for f in freqs:
                design = np.column_stack([np.cos(2 * np.pi * f * tt), np.sin(2 * np.pi * f * tt),
                                          np.ones_like(tt), tt])
                coef, *_ = np.linalg.lstsq(design, ss, rcond=None)
                rms = float(np.sqrt(np.mean((ss - design @ coef) ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, float(np.hypot(coef[0], coef[1])))
            starts.append(a)
            resid.append(best[0])
            amps.append(best[1])
        a += step
    if not starts:
        return None
    amps, resid, starts = np.array(amps), np.array(resid), np.array(starts)
    clean = (resid < CLEAN_RESIDUAL * amps) & (amps >= MIN_AMPLITUDE_FRACTION * amps.max())
    if not clean.any():
        return None

    best_run, run_start = (0, 0), None
    for i, ok in enumerate(list(clean) + [False]):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            if i - run_start > best_run[1] - best_run[0]:
                best_run = (run_start, i)
            run_start = None
    first, last = best_run
    t_start = t[0] if first == 0 else starts[first]  # a run that reaches an edge window keeps the edge
    t_end = t[-1] if last == len(clean) else starts[last - 1] + win
    idx = np.nonzero((t >= t_start) & (t <= t_end))[0]
    if len(idx) < 10 or (t[idx[-1]] - t[idx[0]]) < MIN_CYCLES * T0:
        return None
    return int(idx[0]), int(idx[-1])


def _fit(t0, s):
    """Best damped-sinusoid fit over a few phase starts, or None."""
    T0 = _initial_period(t0, s)
    if T0 is None:
        return None
    p0 = [(np.max(s) - np.min(s)) / 2, 0.01, T0, 0.0, float(np.mean(s))]
    lower = [0, 0, T0 * 0.5, -2 * np.pi, -np.inf]
    upper = [np.inf, 5.0, T0 * 1.5, 2 * np.pi, np.inf]
    best = None
    for phi0 in (0.0, np.pi / 2, np.pi, -np.pi / 2):  # phase guess matters
        p0[3] = phi0
        try:
            popt, pcov = curve_fit(_model, t0, s, p0=p0, bounds=(lower, upper), maxfev=4000)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((s - _model(t0, *popt)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)
    return best


def _cycle_periods(t, s, centre, T):
    """Period of each individual cycle, from upward crossings of the centre line."""
    d = s - centre
    up = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
    if len(up) < 2:
        return np.array([])
    # linear interpolation gives the crossing time between two frames
    times = t[up] - d[up] * (t[up + 1] - t[up]) / (d[up + 1] - d[up])
    periods = np.diff(times)
    return periods[np.abs(periods - T) < 0.3 * T]  # drop crossings caused by noise


def _honest_uncertainty(t0, s, popt, pcov, fitted):
    """The largest of three independent uncertainty estimates.

    fit: from the fit covariance, widened when neighbouring residuals are
         correlated (they are, in real tracking), since the covariance
         assumes independent noise.
    cycle_scatter: how much single-cycle periods differ, divided by sqrt(n).
    timing_floor: frames are only sampled every dt, so each crossing time is
         uncertain by about dt/sqrt(12); over N cycles that gives sqrt(2)*that/N.
    """
    T = float(popt[2])
    resid = s - fitted
    r = resid - resid.mean()
    rho = float(np.sum(r[1:] * r[:-1]) / np.sum(r * r)) if np.sum(r * r) > 0 else 0.0
    inflation = float(np.clip(np.sqrt((1 + rho) / (1 - rho)), 1.0, 10.0)) if rho > 0 else 1.0
    raw = float(np.sqrt(pcov[2, 2])) if np.isfinite(pcov[2, 2]) else float("nan")
    fit_sigma = raw * inflation if np.isfinite(raw) else float("nan")

    periods = _cycle_periods(t0, s, popt[4], T)
    scatter = float(np.std(periods, ddof=1) / np.sqrt(len(periods))) if len(periods) >= 3 else None

    dt = float(np.median(np.diff(t0)))
    cycles = (t0[-1] - t0[0]) / T
    floor = float(np.sqrt(2) * (dt / np.sqrt(12)) / max(cycles, 1.0))

    parts = [p for p in (fit_sigma, scatter, floor) if p is not None and np.isfinite(p)]
    comps = {"fit": fit_sigma, "cycle_scatter": scatter, "timing_floor": floor,
             "residual_autocorrelation": rho}
    return max(parts), comps


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
