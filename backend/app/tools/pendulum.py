"""Pendulum period: fit a damped sinusoid to a tracked position signal.

Model: x(t) = A * exp(-b t) * cos(2 pi t / T + phi) + c
The period T and its uncertainty come from the least-squares fit (the
covariance matrix). If the string length L is known, g = 4 pi^2 L / T^2.
Pure maths: no AI is involved in any number produced here.
"""
import numpy as np
from scipy.optimize import curve_fit

G_STANDARD = 9.81
MIN_CYCLES = 2
MIN_R2 = 0.9


def _model(t, A, b, T, phi, c):
    return A * np.exp(-b * t) * np.cos(2 * np.pi * t / T + phi) + c


def _initial_period(t, s):
    """Guess the period from the FFT peak of the signal on a uniform grid."""
    grid = np.linspace(t[0], t[-1], len(t))
    u = np.interp(grid, t, s - s.mean())
    spectrum = np.abs(np.fft.rfft(u * np.hanning(len(u))))
    freqs = np.fft.rfftfreq(len(u), d=(grid[-1] - grid[0]) / (len(u) - 1))
    spectrum[0] = 0
    f = freqs[int(np.argmax(spectrum))]
    return 1.0 / f if f > 0 else None


def _empty(warning):
    return {"value": None, "uncertainty": None, "units": "s",
            "method": "damped sinusoid fit", "fit_quality": {},
            "series_for_chart": {}, "warnings": [warning], "extra": {}}


def pendulum_period(t, x, y, length_m=None, length_uncertainty_m=None):
    """Return the swing period (and g if the string length is given).

    t: timestamps in seconds. x, y: tracked position (pixels or metres).
    The swing happens mostly along one axis, so we fit the axis that moves most.
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
    t0 = t - t[0]
    if np.std(s) < 1e-6:
        return _empty("The tracked object does not move, so no period can be measured.")

    T0 = _initial_period(t0, s)
    if T0 is None:
        return _empty("Could not find an oscillation in the tracked signal.")

    amp = (np.max(s) - np.min(s)) / 2
    p0 = [amp, 0.01, T0, 0.0, float(np.mean(s))]
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
    if best is None:
        return _empty("The fit did not converge; the motion may not be a clean oscillation.")

    sse, popt, pcov = best
    T = float(popt[2])
    sigma_T = float(np.sqrt(pcov[2, 2])) if np.isfinite(pcov[2, 2]) else float("nan")
    ss_tot = float(np.sum((s - s.mean()) ** 2))
    r2 = 1 - sse / ss_tot if ss_tot > 0 else 0.0
    duration = float(t0[-1] - t0[0])
    cycles = duration / T

    warnings = []
    if cycles < MIN_CYCLES:
        warnings.append(f"Only {cycles:.1f} cycles were recorded; film at least {MIN_CYCLES} full cycles for a reliable period.")
    if r2 < MIN_R2:
        warnings.append(f"The fit quality is low (R2 = {r2:.2f}); the tracking may be noisy or the motion is not a clean swing.")
    if not np.isfinite(sigma_T):
        warnings.append("The uncertainty of the period could not be estimated.")

    fitted = _model(t0, *popt)
    result = {
        "value": T,
        "uncertainty": sigma_T,
        "units": "s",
        "method": "damped sinusoid fit x(t) = A e^(-bt) cos(2 pi t / T + phi) + c",
        "fit_quality": {"r2": r2, "cycles": cycles, "axis": axis,
                        "damping_rate_per_s": float(popt[1]), "n_points": int(len(t))},
        "series_for_chart": {"t": t0.tolist(), "measured": s.tolist(), "fitted": fitted.tolist()},
        "warnings": warnings,
        "extra": {},
    }

    if length_m:
        g = 4 * np.pi**2 * length_m / T**2
        rel = (2 * sigma_T / T) ** 2 if np.isfinite(sigma_T) else 0.0
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
