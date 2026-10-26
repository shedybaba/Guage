"""Growth over time: how much something grew over a time-lapse, how fast, and how long it takes to double.

The size of the tracked object is read in every frame (its height, its width, or its area). The headline numbers do not
depend on any growth model: the size at the very START is compared with the size at the very END. Each is read from a short
straight line fitted to the first (or last) 15% of the frames and evaluated at the first (or last) frame, which uses many
frames instead of trusting one, and does not under-state the change the way an average over the window would.

  total change   = (end - start) / start               a percentage; needs no ruler and no real time
  growth rate    = (end - start) / real time           needs the real time the video covers (a time-lapse is sped up)
  doubling time  = ln 2 / (ln(end / start) / time)     only if it grew by more than its uncertainty

A straight line and an exponential curve are also fitted. They are used to draw the chart and to say in words how the
growth looks (steady, or speeding up); they never change the headline numbers.

The uncertainty comes from the scatter of the sizes around the fitted trend, widened because neighbouring frames share
errors (a leaf in shadow stays small for several frames). A growth that is smaller than that uncertainty is reported as
not measurable instead of as a number. Pure maths: no AI is involved in any number produced here.
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

MIN_SAMPLES = 20
END_FRACTION = 0.15  # the start and the end size come from lines fitted to the first and last 15% of the frames
MIN_END_SAMPLES = 8
OUTLIER_SIGMAS = 6.0  # a size this far from its neighbours' median is a glitch of the outline, not growth
MAX_RHO = 0.97
SHAPE_MARGIN = 0.8  # a model is called better only if its squared error is under 80% of the other's

LENGTH_UNITS = {"height": "height", "width": "width"}
SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0}


def _empty(warning, units="%"):
    return {"value": None, "uncertainty": None, "units": units, "method": "size at the start compared with the size at the end",
            "fit_quality": {}, "series_for_chart": {}, "warnings": [warning], "extra": {}}


def _time_unit(span_s):
    """The unit a rate is quoted in, chosen from how much real time the clip covers."""
    return "s" if span_s < 300 else "min" if span_s < 7200 else "h" if span_s < 3 * 86400 else "day"


def _size_unit(quantity, typical_si, has_scale):
    """(factor from the tool's units to the displayed unit, displayed unit)."""
    if not has_scale:
        return 1.0, "px²" if quantity == "area" else "px"
    if quantity == "area":
        return (1e4, "cm²") if typical_si < 1.0 else (1.0, "m²")
    return (1e3, "mm") if typical_si < 0.05 else (1e2, "cm") if typical_si < 2.0 else (1.0, "m")


def _rho(resid):
    if len(resid) < 8 or np.std(resid) == 0:
        return 0.0
    r = resid - resid.mean()
    return float(np.clip(np.sum(r[1:] * r[:-1]) / np.sum(r * r), 0.0, MAX_RHO))


def _edge_value(tau, s, at, floor, rho_global=0.0):
    """The size at one end of the clip from a straight line fitted to a short stretch next to it: (value, its standard error, rho).

    The line's value at the very end is less certain than at its middle, and the error is taken from the line's own
    scatter, widened because neighbouring frames share errors.
    """
    A = np.stack([tau - at, np.ones_like(tau)], axis=1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    resid = s - A @ coef
    dof = max(len(s) - 2, 1)
    sigma = max(float(np.sqrt(np.sum(resid ** 2) / dof)), floor)
    # a short line soaks up slow noise, so its own residuals understate the correlation: use the whole clip's as well
    rho = max(_rho(resid), rho_global)
    var_at_end = float(np.linalg.inv(A.T @ A)[1, 1])
    return float(coef[1]), sigma * float(np.sqrt((1 + rho) / (1 - rho))) * float(np.sqrt(var_at_end)), rho


def _fits(tau, s):
    """Straight line and exponential, each with 2 parameters. Returns {name: (predicted, sse)} for the ones that worked."""
    out = {}
    A = np.stack([tau, np.ones_like(tau)], axis=1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    pred = A @ coef
    out["linear"] = (pred, float(np.sum((s - pred) ** 2)), coef)
    if np.all(s > 0):
        c = np.polyfit(tau, np.log(s), 1)
        try:
            (a, r), _ = curve_fit(lambda x, a_, r_: a_ * np.exp(r_ * x), tau, s, p0=[np.exp(c[1]), c[0]], maxfev=5000)
            pred_e = a * np.exp(r * tau)
            if np.all(np.isfinite(pred_e)):
                out["exponential"] = (pred_e, float(np.sum((s - pred_e) ** 2)), np.array([a, r]))
        except (RuntimeError, ValueError, OverflowError):
            pass
    return out


def growth_over_time(t, size, quantity="area", scale_m_per_px=None, scale_uncertainty_fraction=None, speedup=None,
                     coverage=None):
    """How much, how fast, and how long to double.

    t: time of each tracked frame in the video (s). size: height, width (px) or area (px^2) of the object in each frame.
    quantity: "height", "width" or "area". scale_m_per_px: metres per pixel (optional; without it sizes stay in pixels).
    speedup: real seconds that pass per second of video (30 for a time-lapse where 1 s of video is 30 s of real life);
             without it no rate can be given. coverage: share of the video's frames in which the object was found.
    """
    method = "size at the start compared with the size at the end (each from a straight line fitted to the first or last 15% of the frames)"
    t, size = np.asarray(t, float), np.asarray(size, float)
    ok = np.isfinite(t) & np.isfinite(size) & (size > 0)
    t, size = t[ok], size[ok]
    if len(t) < MIN_SAMPLES:
        return _empty(f"Only {len(t)} frames with a usable size: at least {MIN_SAMPLES} are needed to measure growth.")
    order = np.argsort(t)
    t, size = t[order], size[order]
    warnings, notes = [], []

    # glitches of the outline (the object lost for a frame, a shadow) are removed before anything is measured
    w = max(5, (len(size) // 20) | 1)
    local = median_filter(size, size=w, mode="nearest")
    dev = size - local
    sigma_local = 1.4826 * float(np.median(np.abs(dev - np.median(dev))))
    keep = np.abs(dev) <= OUTLIER_SIGMAS * max(sigma_local, 1e-12 * float(np.median(size)))
    removed = int((~keep).sum())
    if removed:
        pct = 100 * removed / len(size)
        warnings.append(f"{removed} frame(s) where the outline of the object glitched ({pct:.0f}% of them) were left out.")
        if pct > 20:
            warnings.append("Many frames were unreliable: the object may be partly hidden or the lighting may change a lot.")
    t, size, local = t[keep], size[keep], local[keep]
    if len(t) < MIN_SAMPLES:
        return _empty("Too few frames were left after removing outline glitches to measure growth.")

    span_video = float(t[-1] - t[0])
    if span_video <= 0:
        return _empty("The frames have no time spread, so growth cannot be measured.")
    k = float(speedup) if speedup else None
    tau = (t - t[0]) * (k if k else 1.0)  # real seconds if the speed-up is known, else video seconds
    real_span = float(tau[-1])

    # the trend: used for the noise level, the chart and the words, never for the headline
    fits = _fits(tau, size)
    sse_l = fits["linear"][1]
    sse_e = fits["exponential"][1] if "exponential" in fits else np.inf
    if sse_e < SHAPE_MARGIN * sse_l:
        model = "exponential"
    elif sse_l < SHAPE_MARGIN * sse_e:
        model = "linear"
    else:
        model = "undetermined"
    best = min(fits, key=lambda n: fits[n][1])
    pred, _, coef = fits[best]
    resid = size - pred
    n = len(size)
    sigma = float(np.sqrt(np.sum(resid ** 2) / max(n - 2, 1)))
    rho = _rho(resid)

    # start and end size: a short line fitted to each end, evaluated at the first / last frame
    m = min(max(MIN_END_SAMPLES, int(round(END_FRACTION * n))), n // 3)
    floor = 0.29 if quantity != "area" else 1.2 * float(np.sqrt(np.median(size)))  # outline rounding to whole pixels
    s0, se0, rho0 = _edge_value(tau[:m], size[:m], tau[0], floor, rho)
    s1, se1, rho1 = _edge_value(tau[-m:], size[-m:], tau[-1], floor, rho)
    rho = max(rho, rho0, rho1)
    if s0 <= 0 or s1 <= 0:
        return _empty("The size at the start or the end could not be read reliably from the frames.")
    delta, d_unc = s1 - s0, float(np.hypot(se0, se1))
    R = s1 / s0
    R_unc = R * float(np.sqrt((se0 / s0) ** 2 + (se1 / s1) ** 2))
    pct, pct_unc = (R - 1.0) * 100.0, 100.0 * R_unc
    grew = abs(delta) > 2 * d_unc

    # sizes in real units
    has_scale = scale_m_per_px is not None
    f = scale_m_per_px ** (2 if quantity == "area" else 1) if has_scale else 1.0
    factor, unit = _size_unit(quantity, float(np.median(size)) * f, has_scale)
    conv = f * factor
    # the ruler's own uncertainty only touches sizes in real units, not the percentage
    rel_scale = (scale_uncertainty_fraction or 0.0) * (2 if quantity == "area" else 1)
    abs_unc = lambda v, u: float(np.hypot(u, abs(v) * rel_scale))  # noqa: E731

    extra = {
        "quantity": quantity, "model": model,
        "total_change_percent": {"value": pct, "uncertainty": pct_unc, "units": "%"},
        "start_size": {"value": s0 * conv, "units": unit}, "end_size": {"value": s1 * conv, "units": unit},
        "absolute_change": {"value": delta * conv, "uncertainty": abs_unc(delta * conv, d_unc * conv), "units": unit},
        "size_ratio": {"value": R, "uncertainty": R_unc, "units": ""},
        "grew_measurably": bool(grew), "has_scale": has_scale, "real_time_known": bool(k),
        "outliers_removed": removed,
    }

    tu = _time_unit(real_span) if k else "s"
    per = SECONDS[tu]
    if k:
        extra["real_duration"] = {"value": real_span / per, "units": tu}
        rate = delta * conv / (real_span / per)
        extra["growth_rate"] = {"value": rate, "uncertainty": abs_unc(rate, d_unc * conv / (real_span / per)),
                                "units": f"{unit}/{tu}"}
        r = np.log(R) / (real_span / per) if R > 0 else float("nan")
        r_unc = (R_unc / R) / (real_span / per)
        extra["relative_growth_rate"] = {"value": 100 * r, "uncertainty": 100 * r_unc, "units": f"%/{tu}"}
        if r > 2 * r_unc:
            dt = np.log(2) / r
            extra["doubling_time"] = {"value": dt, "uncertainty": dt * r_unc / r, "units": tu}
        # the two fitted trends as a cross-check of the average rate
        a, b = fits["linear"][2]
        extra["linear_fit_rate"] = {"value": a * conv * per, "units": f"{unit}/{tu}"}

    # ---- warnings
    if not grew:
        warnings.append("No measurable growth: the size at the end is not different from the size at the start by more "
                        "than the noise in the measurement.")
    if delta < 0 and grew:
        warnings.append("The object got smaller over the clip.")
    smooth = median_filter(size, size=w, mode="nearest")
    peak_i = int(np.argmax(smooth))
    if grew and delta > 0 and peak_i < 0.8 * n and smooth[peak_i] - s1 > 3 * d_unc:
        warnings.append("The size peaked before the end of the clip and then fell, so the total change compares the end "
                        "with the start, not with the largest size.")
    if k is None:
        warnings.append("No real time was given, so no growth rate or doubling time can be stated: only the change in size.")
    if not has_scale:
        warnings.append("No known length was given, so sizes are in pixels. The percentage change is not affected.")
    if quantity in LENGTH_UNITS:
        warnings.append(f"The {quantity} is read from the outline's {'vertical' if quantity == 'height' else 'horizontal'} "
                        "extent in the picture, so the camera must stay still and the growth must be along that direction.")
    else:
        warnings.append("The area is the size of the outline seen by the camera; moving closer, tilting or hiding part of "
                        "the object changes it as well as growth does.")
    if len(size) < 40:
        warnings.append("Few frames were used; film a longer or denser time-lapse for a steadier result.")
    if coverage is not None and coverage < 0.8:
        warnings.append(f"The object was found in only {coverage * 100:.0f}% of the frames.")
    if rho > 0.7:
        warnings.append("Sizes in neighbouring frames share their errors strongly, so the uncertainty was widened.")

    statements = []
    if grew and model == "exponential":
        statements.append("The growth speeds up over time (it fits an exponential curve better than a straight line).")
    elif grew and model == "linear":
        statements.append("The growth is steady (it fits a straight line better than an exponential curve).")

    # ---- the chart
    xs = tau / per if k else tau
    series = {"t": [float(v) for v in xs], "t_unit": tu if k else "s", "y_label": f"{quantity} ({unit})",
              "measured": [float(v * conv) for v in size], "smooth": [float(v * conv) for v in smooth],
              "fitted": [float(v * conv) for v in pred], "fit_name": best}
    return {"value": pct, "uncertainty": pct_unc, "units": "%", "method": method,
            "fit_quality": {"n_frames": int(n), "noise": float(sigma), "residual_correlation": rho, "end_window_frames": m,
                            "model": model, "sse_linear": sse_l, "sse_exponential": None if np.isinf(sse_e) else sse_e,
                            "real_span_s": real_span if k else None, "video_span_s": span_video},
            "series_for_chart": series, "warnings": warnings, "extra": {**extra, "statements": statements}}
