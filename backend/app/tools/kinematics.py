"""Free fall and velocity/acceleration from a tracked path.

Both rest on one idea: over a short stretch of motion with constant acceleration the position is a
parabola, x(t) = x0 + v0 t + a t^2 / 2, in each of the two image directions together. So the acceleration
is a VECTOR, found from a quadratic fit in x and y, and its direction tells us about the camera for free:

  free fall            |a| should be g, and it should point straight down; if it does not, the phone is tilted
  ramp / rolling ball  a should be g sin(theta), or 5/7 of that for a solid ball that rolls without slipping
  swinging / driving   speed and acceleration along the path come from local fits at every frame

Real clips contain rest before the motion, and a landing or a catch after it, so the stretch of constant
acceleration is found first (windows whose acceleration agree) and grown frame by frame to its true ends.
The uncertainty is statistical (fit, widened for correlated noise) plus the SCALE uncertainty, which is
usually the biggest error of all: a hand-measured reference length is rarely better than 1 to 3%.
Pure maths: no AI is involved in any number produced here.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

G_STANDARD = 9.81
ASSUMED_SCALE_UNCERTAINTY = 0.02  # used (and said so) when a scale is given without its uncertainty
ASSUMED_ANGLE_UNCERTAINTY_DEG = 1.0
MIN_SEGMENT_FRAMES = 6
SHORT_SEGMENT_FRAMES = 12
CONSISTENCY = 0.3  # windows agree on the acceleration within 30% (or 3 sigma)
GROW_SIGMAS = 3.0
ANGLE_MISMATCH_DEG = 4.0

# acceleration of a ball/cylinder rolling down a slope without slipping, as a fraction of g sin(theta)
SLOPE_FACTORS = {
    "sliding_without_friction": 1.0,
    "rolling_solid_sphere": 5 / 7,
    "rolling_solid_cylinder": 2 / 3,
    "rolling_hollow_sphere": 3 / 5,
    "rolling_hoop": 1 / 2,
}


def _empty(units, method, warning, extra_warnings=None):
    return {"value": None, "uncertainty": None, "units": units, "method": method, "fit_quality": {},
            "series_for_chart": {}, "warnings": (extra_warnings or []) + [warning], "extra": {}}


def _clean(t, x, y):
    t, x, y = (np.asarray(v, float) for v in (t, x, y))
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    order = np.argsort(t)
    return t[order], x[order], y[order]


def _odd_window(t, window_s, lo=5, hi=25):
    fps = (len(t) - 1) / (t[-1] - t[0])
    w = int(round(window_s * fps))
    w = max(lo, min(hi, w))
    return w + 1 - w % 2  # odd


# ------------------------------------------------------------ local fits

def local_fits(t, x, y, window):
    """At every frame, fit a parabola to the `window` frames around it (both axes).

    Returns v (N,2) and a (N,2) in px/s and px/s^2 (NaN near the ends), the noise sigma of the
    positions (px), and the standard errors of v and a (scalars: the same for every full window
    when the frame rate is steady).
    """
    n, h = len(t), window // 2
    v, a = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    rss, sv_f, sa_f = [], [], []
    for i in range(h, n - h):
        tau = t[i - h:i + h + 1] - t[i]
        A = np.stack([np.ones_like(tau), tau, tau ** 2], axis=1)
        inv = np.linalg.inv(A.T @ A)
        sv_f.append(np.sqrt(inv[1, 1]))
        sa_f.append(2 * np.sqrt(inv[2, 2]))
        r = 0.0
        for k, s in enumerate((x, y)):
            c = inv @ (A.T @ s[i - h:i + h + 1])
            v[i, k], a[i, k] = c[1], 2 * c[2]
            r += float(np.sum((s[i - h:i + h + 1] - A @ c) ** 2))
        rss.append(r / 2.0)  # per axis
    dof = window - 3
    # median of chi-square / dof is below 1: correct for it so the noise is not underestimated
    sigma = float(np.sqrt(np.median(rss) / (dof * (1 - 2 / (9 * dof)) ** 3))) if rss else float("nan")
    return v, a, sigma, float(np.median(sv_f)) * sigma, float(np.median(sa_f)) * sigma


def find_constant_acceleration(a, sigma_a, window):
    """The longest run of windows whose accelerations agree and are clearly not zero.

    The run must last longer than one window: the abrupt stop of a caught ball makes a few neighbouring
    windows agree on a big deceleration, but only a real stretch of steady acceleration lasts longer than
    the window that measures it. Returns (first centre index, last centre index, reference acceleration
    vector) or None.
    """
    ok = np.isfinite(a).all(axis=1)
    mag = np.hypot(a[:, 0], a[:, 1])
    cand = np.nonzero(ok & (mag > 4 * sigma_a))[0]
    if len(cand) < 3:
        return None
    tol = lambda ref: max(4.2 * sigma_a, CONSISTENCY * np.hypot(*ref))  # noqa: E731
    best_i, best_n = None, 0
    for i in cand:
        n = int(np.sum(np.hypot(*(a[cand] - a[i]).T) <= tol(a[i])))
        if n > best_n:
            best_i, best_n = i, n
    near = cand[np.hypot(*(a[cand] - a[best_i]).T) <= tol(a[best_i])]
    ref = np.median(a[near], axis=0)
    member = ok & (np.hypot(a[:, 0] - ref[0], a[:, 1] - ref[1]) <= tol(ref))
    # allow a single stray window inside a run
    idx = np.nonzero(member)[0]
    runs, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i - prev <= 2:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    if start is not None:
        runs.append((start, prev))
    first, last = max(runs, key=lambda r: r[1] - r[0])
    if last - first + 1 < max(4, window):
        return None
    return int(first), int(last), ref


# ------------------------------------------------------------ the fit over the stretch

def _fit_axis(tau, s, deg=2, sigma_floor=0.0):
    """Polynomial fit; returns coefficients (low order first), covariance, residuals.

    The covariance uses the fit's own residual variance, but never less than sigma_floor squared: a fit
    over few frames has so few degrees of freedom that its residuals can come out far too small.
    """
    A = np.stack([tau ** k for k in range(deg + 1)], axis=1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    resid = s - A @ coef
    dof = max(len(s) - deg - 1, 1)
    var = max(float(np.sum(resid ** 2) / dof), sigma_floor ** 2)
    cov = var * np.linalg.inv(A.T @ A)
    return coef, cov, resid


def _grow(t, x, y, i0, i1, sigma):
    """Extend the stretch outward while the next frame still sits on the fitted parabola (within 3 sigma)."""
    n = len(t)
    for _ in range(n):
        moved = False
        for side in (-1, 1):
            j = i0 - 1 if side < 0 else i1 + 1
            if j < 0 or j >= n or i1 - i0 + 1 < 4:
                continue
            tau = t[i0:i1 + 1] - t[i0]
            tj = t[j] - t[i0]
            fits = []
            for s in (x, y):
                c, _, _ = _fit_axis(tau, s[i0:i1 + 1])
                fits.append(abs(s[j] - (c[0] + c[1] * tj + c[2] * tj ** 2)))
            if max(fits) <= GROW_SIGMAS * sigma:
                if side < 0:
                    i0 = j
                else:
                    i1 = j
                moved = True
        if not moved:
            break
    return i0, i1


def fit_constant_acceleration(t, x, y, i0, i1, sigma_floor=0.0):
    """Fit x(t) and y(t) as parabolas over frames i0..i1: acceleration vector, its uncertainty, and fit facts."""
    tau = t[i0:i1 + 1] - t[i0]
    cx, covx, rx = _fit_axis(tau, x[i0:i1 + 1], sigma_floor=sigma_floor)
    cy, covy, ry = _fit_axis(tau, y[i0:i1 + 1], sigma_floor=sigma_floor)
    a = np.array([2 * cx[2], 2 * cy[2]])
    sa = np.array([2 * np.sqrt(covx[2, 2]), 2 * np.sqrt(covy[2, 2])])
    r = np.concatenate([rx - rx.mean(), ry - ry.mean()])
    r1, r2 = rx - rx.mean(), ry - ry.mean()
    num = float(np.sum(r1[1:] * r1[:-1]) + np.sum(r2[1:] * r2[:-1]))
    den = float(np.sum(r1 ** 2) + np.sum(r2 ** 2))
    rho = num / den if den > 0 else 0.0
    inflation = float(np.clip(np.sqrt((1 + rho) / (1 - rho)), 1.0, 10.0)) if rho > 0 else 1.0
    sa = sa * inflation
    # the time of every frame is known only to about dt/sqrt(12); over the fit that is a tiny extra error
    v0 = np.array([cx[1], cy[1]])
    rms = float(np.sqrt(np.mean(r ** 2)))
    mag = float(np.hypot(*a))
    s_mag = float(np.sqrt((a[0] * sa[0]) ** 2 + (a[1] * sa[1]) ** 2) / mag) if mag > 0 else float(np.hypot(*sa))
    return {"a": a, "sa": sa, "mag": mag, "s_mag": s_mag, "v0": v0, "x0": np.array([cx[0], cy[0]]), "tau": tau,
            "rms": rms, "rho": rho, "inflation": inflation, "n": i1 - i0 + 1}


REST_BUFFER = 6  # frames next to a boundary that are not trusted to be "rest" yet
REST_LENGTH = 40  # how far back/forward to look for the object lying still
MAX_PLAUSIBLE_DELTA = 4.0  # boundaries whose squared error is within 4 sigma^2 of the best are "plausible"


def _rest_level(arr, lo, hi):
    """Mean of arr[lo:hi]; None if fewer than 4 frames."""
    return float(np.mean(arr[lo:hi])) if hi - lo >= 4 else None


def _is_rest(x, y, lo, hi, sigma):
    """Do frames lo..hi-1 sit still (scatter no bigger than tracking noise)?"""
    if hi - lo < 8:
        return False
    return bool(np.std(x[lo:hi]) <= 2.0 * sigma and np.std(y[lo:hi]) <= 2.0 * sigma)


def _total_sse(t, x, y, i0, i1, rest_before, rest_after):
    """Squared error of: [rest level] + [parabola over i0..i1] + [rest level], summed over both axes."""
    tau = t[i0:i1 + 1] - t[i0]
    total = 0.0
    for s in (x, y):
        _, _, r = _fit_axis(tau, s[i0:i1 + 1])
        total += float(np.sum(r ** 2))
        if rest_before:
            seg = s[rest_before[0]:i0]
            total += float(np.sum((seg - seg.mean()) ** 2)) if len(seg) else 0.0
        if rest_after:
            seg = s[i1 + 1:rest_after[1]]
            total += float(np.sum((seg - seg.mean()) ** 2)) if len(seg) else 0.0
    return total


def refine_boundaries(t, x, y, i0, i1, sigma):
    """Place the ends of the stretch where the data say, and say how sure that is.

    If the object lay still before (or after) the stretch, every frame near the boundary belongs either to
    the rest or to the parabola, and the boundary is the split with the smallest total squared error. Frames
    next to the flat bottom of the parabola fit both about equally, so every boundary within 4 sigma^2 of the
    best is "plausible", and the spread of the acceleration over them is returned as an extra uncertainty.
    Without any rest the ends are only grown, and the uncertainty is the change from trimming 2 frames.
    Returns (i0, i1, boundary_sigma_px_per_s2, rest_before, rest_after).
    """
    n = len(t)
    before = None
    lo = max(0, i0 - REST_LENGTH)
    if _is_rest(x, y, lo, max(lo, i0 - REST_BUFFER), sigma):
        before = (lo, i0)
    after = None
    hi = min(n, i1 + 1 + REST_LENGTH)
    if _is_rest(x, y, min(hi, i1 + 1 + REST_BUFFER), hi, sigma):
        after = (i1 + 1, hi)

    def mag(a, b):
        if b - a + 1 < MIN_SEGMENT_FRAMES:
            return None
        return fit_constant_acceleration(t, x, y, a, b, sigma)["mag"]

    plausible_mags = []
    for _ in range(4):  # coordinate descent over the two boundaries
        changed = False
        if before:
            cands = [k for k in range(max(lo + 4, i0 - REST_BUFFER), min(i1 - MIN_SEGMENT_FRAMES, i0 + REST_BUFFER) + 1)]
            sse = {k: _total_sse(t, x, y, k, i1, (lo, k), after) for k in cands}
            best = min(sse, key=sse.get)
            changed |= best != i0
            i0, before = best, (lo, best)
        if after:
            cands = [k for k in range(max(i0 + MIN_SEGMENT_FRAMES, i1 - REST_BUFFER), min(hi - 4, i1 + REST_BUFFER) + 1)]
            sse = {k: _total_sse(t, x, y, i0, k, before, (k + 1, hi)) for k in cands}
            best = min(sse, key=sse.get)
            changed |= best != i1
            i1, after = best, (best + 1, hi)
        if not changed:
            break

    sigma_b = 0.0
    if before or after:
        vals = [mag(i0, i1)]
        if before:
            base = _total_sse(t, x, y, i0, i1, before, after)
            for k in range(max(lo + 4, i0 - REST_BUFFER), min(i1 - MIN_SEGMENT_FRAMES, i0 + REST_BUFFER) + 1):
                if _total_sse(t, x, y, k, i1, (lo, k), after) <= base + MAX_PLAUSIBLE_DELTA * sigma ** 2:
                    vals.append(mag(k, i1))
        if after:
            base = _total_sse(t, x, y, i0, i1, before, after)
            for k in range(max(i0 + MIN_SEGMENT_FRAMES, i1 - REST_BUFFER), min(hi - 4, i1 + REST_BUFFER) + 1):
                if _total_sse(t, x, y, i0, k, before, (k + 1, hi)) <= base + MAX_PLAUSIBLE_DELTA * sigma ** 2:
                    vals.append(mag(i0, k))
        vals = [v for v in vals if v is not None]
        sigma_b = float(np.std(vals)) if len(vals) >= 2 else 0.0
    else:
        base = mag(i0, i1)
        vals = [base] + [m for m in (mag(i0 + 2, i1), mag(i0, i1 - 2), mag(i0 + 2, i1 - 2)) if m is not None]
        sigma_b = float(np.std(vals)) if len(vals) >= 2 else 0.0
    return i0, i1, sigma_b, before, after


def _rest_noise(x, y, ranges):
    """(lag-1 correlation, noise sigma in px) of the tracking noise, measured on frames where the object lies still.

    A fit over the moving stretch soaks up the slow part of correlated noise, so its residuals look whiter than
    the noise really is; frames at rest have no signal to soak it up, so they show the true correlation.
    """
    num = den = 0.0
    devs = []
    for lo, hi in ranges:
        for s in (x, y):
            d = s[lo:hi] - np.mean(s[lo:hi])
            if len(d) >= 8:
                num += float(np.sum(d[1:] * d[:-1]))
                den += float(np.sum(d * d))
                devs.append(d)
    if not devs or den <= 0:
        return None, None
    allv = np.concatenate(devs)
    return num / den, float(np.sqrt(np.mean(allv ** 2) * len(allv) / max(len(allv) - len(devs), 1)))


def fit_stretch(t, x, y, i0, i1, sigma):
    """Refine the ends of the stretch, fit it, and add the boundary uncertainty to the acceleration's.

    If the object lay still before or after the stretch, the tracking noise and its correlation are measured
    there (more reliable than from the fit's own residuals) and the uncertainty widened accordingly.
    """
    i0, i1, sigma_b, rest_before, rest_after = refine_boundaries(t, x, y, i0, i1, sigma)
    rests = [r for r in (rest_before, rest_after) if r]
    rho_rest, sigma_rest = _rest_noise(x, y, [(lo, hi) for lo, hi in rests]) if rests else (None, None)
    fit = fit_constant_acceleration(t, x, y, i0, i1, max(sigma, sigma_rest or 0.0))
    if rho_rest is not None and rho_rest > 0:
        infl_rest = float(np.clip(np.sqrt((1 + rho_rest) / (1 - rho_rest)), 1.0, 10.0))
        if infl_rest > fit["inflation"]:
            fit["sa"] = fit["sa"] * infl_rest / fit["inflation"]
            fit["s_mag"] = fit["s_mag"] * infl_rest / fit["inflation"]
            fit["inflation"], fit["rho"] = infl_rest, rho_rest
    fit["noise_from_rest"] = rho_rest is not None
    fit["i0"], fit["i1"] = i0, i1
    fit["boundary_sigma"] = sigma_b
    fit["rest_before"], fit["rest_after"] = bool(rest_before), bool(rest_after)
    fit["s_mag_statistical"] = fit["s_mag"]
    fit["s_mag"] = float(np.hypot(fit["s_mag"], sigma_b))
    # the boundary uncertainty is along |a|; spread it over the components in proportion to the acceleration itself
    if fit["mag"] > 0:
        fit["sa"] = np.hypot(fit["sa"], sigma_b * np.abs(fit["a"]) / fit["mag"])
    return fit


def _jerk_significant(t, x, y, i0, i1, inflation):
    """Is the acceleration changing over the stretch (air drag, a bend in the ramp)? Cubic term vs its error."""
    if i1 - i0 + 1 < 10:
        return False
    tau = t[i0:i1 + 1] - t[i0]
    best = 0.0
    for s in (x, y):
        c, cov, _ = _fit_axis(tau, s[i0:i1 + 1], deg=3)
        se = np.sqrt(cov[3, 3]) * inflation
        if se > 0:
            best = max(best, abs(c[3]) / se)
    return best > 4.0


def _scale_fraction(scale, given):
    """(relative scale uncertainty, warning or None)."""
    if not scale:
        return 0.0, None
    if given is not None:
        return float(given), None
    return ASSUMED_SCALE_UNCERTAINTY, (f"No uncertainty was given for the scale, so {100 * ASSUMED_SCALE_UNCERTAINTY:.0f}% "
                                       "is assumed (a hand-measured reference length is rarely better). Give the real one "
                                       "to tighten or widen the result.")


def _find_stretch(t, x, y, window, auto_segment):
    """(i0, i1, local-fit results at the base window) for the stretch of constant acceleration.

    A short window is sharp but noisy: a gentle acceleration can hide inside its noise. So longer windows
    are tried (2x, 4x, 8x) until the acceleration shows. The ends are then grown, and later refined.
    """
    base = local_fits(t, x, y, window)
    if not np.isfinite(base[2]):
        return None, None, base
    if not auto_segment:
        return 0, len(t) - 1, base
    n = len(t)
    for mult in (1, 2, 4, 8):
        w = window * mult
        w = w + 1 - w % 2
        if w > n // 2:
            break
        v, a, sigma, sv, sa = base if mult == 1 else local_fits(t, x, y, w)
        found = find_constant_acceleration(a, sa, w)
        if found is not None:
            c0, c1, _ = found
            i0, i1 = _grow(t, x, y, c0, c1, base[2])
            return i0, i1, base
    return None, None, base


# ------------------------------------------------------------ free fall

def free_fall(t, x, y, scale_m_per_px=None, scale_uncertainty_fraction=None, auto_segment=True):
    """Acceleration of a falling object (g) from its tracked path.

    t: time (s). x, y: tracked position (pixels; y points down). scale_m_per_px: metres per pixel in the
    plane of the fall (without it the answer is in px/s^2 and an implied scale is given instead).
    Only the stretch of free fall is used: rest before release and the landing after are cut off.
    """
    units = "m/s^2" if scale_m_per_px else "px/s^2"
    method = "constant-acceleration (parabola) fit to the free-fall stretch; |a| of the fitted acceleration vector"
    t, x, y = _clean(t, x, y)
    if len(t) < 12:
        return _empty(units, method, "Not enough tracked frames to measure a fall.")
    window = _odd_window(t, 0.12)
    i0, i1, (v, a, sigma, sv, sa) = _find_stretch(t, x, y, window, auto_segment)
    if i0 is None:
        return _empty(units, method, "The object does not accelerate steadily anywhere in the clip, so no fall "
                                     "was found (is it moving at constant speed, or lying still?).")
    n = i1 - i0 + 1
    if n < MIN_SEGMENT_FRAMES:
        return _empty(units, method, f"Only {n} frames of steady acceleration were found; that is too few. "
                                     "Film the fall in slow motion.")
    fit = fit_stretch(t, x, y, i0, i1, sigma)
    i0, i1 = fit["i0"], fit["i1"]
    n = i1 - i0 + 1
    if fit["mag"] < 4 * fit["s_mag"]:
        return _empty(units, method, "No significant acceleration was found, so this is not a fall.")

    scale_rel, scale_warning = _scale_fraction(scale_m_per_px, scale_uncertainty_fraction)
    to_si = scale_m_per_px or 1.0
    g = fit["mag"] * to_si
    s_g = float(np.hypot(fit["s_mag"] * to_si, g * scale_rel))
    ax, ay = fit["a"]
    tilt = float(np.degrees(np.arctan2(ax, ay)))  # angle between the acceleration and "straight down in the picture"

    warnings = []
    if scale_warning:
        warnings.append(scale_warning)
    if not scale_m_per_px:
        warnings.append("No scale (metres per pixel) was given, so the acceleration is in pixels/s^2. Give a known length in "
                        "the plane of the fall to get g in m/s^2.")
    if ay <= 0 or abs(tilt) > 45:
        warnings.append("The acceleration does not point down in the picture: the video may be rotated or mirrored, "
                        "or this is not a free fall.")
    elif abs(tilt) > 3:
        warnings.append(f"The fall is tilted by {abs(tilt):.0f} degrees in the picture (the phone was not upright); "
                        "this is handled, because the acceleration is measured along the path, not along the picture.")
    if n < SHORT_SEGMENT_FRAMES:
        warnings.append(f"Only {n} frames of fall: the result is imprecise. Film in slow motion (120 or 240 fps).")
    if _jerk_significant(t, x, y, i0, i1, fit["inflation"]):
        warnings.append("The acceleration changes during the fall (air drag on a light object, or the object was pushed): "
                        "the average over the fall is reported, which is below g if there is drag.")

    dur = float(t[i1] - t[i0])
    tau = fit["tau"]
    v_end = fit["v0"] + fit["a"] * tau[-1]
    p_end = fit["x0"] + fit["v0"] * tau[-1] + 0.5 * fit["a"] * tau[-1] ** 2
    drop = float(np.hypot(*(p_end - fit["x0"])) * to_si)
    implied = G_STANDARD / fit["mag"]
    consistent = bool(abs(g - G_STANDARD) <= 2 * s_g) if scale_m_per_px else None

    chart_fit = [None] * len(t)
    for k in range(i0, i1 + 1):
        chart_fit[k] = float(fit["x0"][1] + fit["v0"][1] * (t[k] - t[i0]) + 0.5 * fit["a"][1] * (t[k] - t[i0]) ** 2)
    out = {
        "value": float(g), "uncertainty": s_g, "units": units, "method": method,
        "fit_quality": {"n_frames": n, "duration_s": dur, "frames_per_second": float((len(t) - 1) / (t[-1] - t[0])),
                        "noise_px": sigma, "fit_rms_px": fit["rms"], "residual_autocorrelation": fit["rho"],
                        "noise_inflation": fit["inflation"], "scale_uncertainty_fraction": scale_rel,
                        "statistical_uncertainty": float(fit["s_mag_statistical"] * to_si),
                        "boundary_uncertainty": float(fit["boundary_sigma"] * to_si),
                        "rest_before": fit["rest_before"], "rest_after": fit["rest_after"],
                        "trimmed": bool(i0 > 0 or i1 < len(t) - 1),
                        "segment": {"start_s": float(t[i0] - t[0]), "end_s": float(t[i1] - t[0])}},
        "series_for_chart": {"t": (t - t[0]).tolist(), "y_px": y.tolist(), "x_px": x.tolist(), "fitted_y_px": chart_fit},
        "warnings": warnings,
        "extra": {"tilt_deg": tilt, "initial_speed": float(np.hypot(*fit["v0"]) * to_si),
                  "impact_speed": float(np.hypot(*v_end) * to_si), "drop_distance": drop,
                  "scale_if_g_is_9.81": {"value": float(implied), "uncertainty": float(implied * fit["s_mag"] / fit["mag"]),
                                         "units": "m/px", "method": "9.81 / measured acceleration in px/s^2, "
                                                                    "valid if this really is free fall"}},
    }
    if scale_m_per_px:
        out["extra"]["percent_difference_from_standard"] = float(100 * (g - G_STANDARD) / G_STANDARD)
        out["extra"]["consistent_with_standard_g"] = consistent
        out["extra"]["standard_g"] = G_STANDARD
    return out


# ------------------------------------------------------------ velocity and acceleration

def _runs(mask):
    runs, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def velocity_acceleration(t, x, y, scale_m_per_px=None, scale_uncertainty_fraction=None, quantity="acceleration",
                          at_time=None, ramp_angle_deg=None, ramp_angle_uncertainty_deg=None, auto_segment=True,
                          window_s=0.1):
    """Speed and acceleration along the path.

    quantity: "acceleration" (along the path, from the stretch of steady acceleration), "speed" (at `at_time`
    if given, else the peak) or "average_speed" (path length / time while moving). Everything else is also
    returned in `extra`. ramp_angle_deg: slope of a ramp, to compare the acceleration with g sin(theta).
    window_s: time span of the local fits (shorter = sharper but noisier).
    """
    if quantity not in ("acceleration", "speed", "average_speed"):
        raise ValueError("quantity must be 'acceleration', 'speed' or 'average_speed'")
    pix = not scale_m_per_px
    to_si = scale_m_per_px or 1.0
    sp_units = "px/s" if pix else "m/s"
    acc_units = "px/s^2" if pix else "m/s^2"
    units = {"acceleration": acc_units, "speed": sp_units, "average_speed": sp_units}[quantity]
    method = "local parabola fits at every frame (speed, acceleration) and a constant-acceleration fit over the steady stretch"
    t, x, y = _clean(t, x, y)
    if len(t) < 12:
        return _empty(units, method, "Not enough tracked frames to measure speed.")
    window = _odd_window(t, window_s)
    fps = (len(t) - 1) / (t[-1] - t[0])
    v, a, sigma, sv, sa = local_fits(t, x, y, window)
    if not np.isfinite(sigma):
        return _empty(units, method, "The clip is too short for the smoothing window.")
    scale_rel, scale_warning = _scale_fraction(scale_m_per_px, scale_uncertainty_fraction)

    speed = np.hypot(v[:, 0], v[:, 1])
    ok = np.isfinite(speed)
    moving = ok & (speed > 4 * sv)
    warnings = []
    if scale_warning:
        warnings.append(scale_warning)
    if pix:
        warnings.append("No scale (metres per pixel) was given, so speed and acceleration are in pixels. Give a known "
                        "length in the plane of the motion to get m/s and m/s^2.")
    if not moving.any():
        return _empty(units, method, "The object does not move measurably in this clip.", warnings)

    runs = _runs(moving)
    m0, m1 = max(runs, key=lambda r: r[1] - r[0])  # the longest stretch of real movement
    dur = float(t[m1] - t[m0])
    sp_si = speed * to_si
    path = float(np.trapezoid(sp_si[m0:m1 + 1], t[m0:m1 + 1]))
    avg = path / dur if dur > 0 else float("nan")
    window_time = window / fps
    s_avg = float(np.hypot(sv * to_si / np.sqrt(max(1.0, dur / window_time)), avg * scale_rel))
    ipk = int(np.nanargmax(np.where(ok, speed, -1)))
    # a windowed fit flattens a sharp peak by a few percent ((omega h)^2 / 10 for a swing), so that is allowed for
    s_pk = float(np.sqrt((sv * to_si) ** 2 + (sp_si[ipk] * scale_rel) ** 2 + (0.03 * sp_si[ipk]) ** 2))

    # acceleration along the path from the stretch of steady acceleration (or the moving stretch if there is none)
    i0, i1, _ = _find_stretch(t, x, y, window, auto_segment)
    steady = i0 is not None and i1 - i0 + 1 >= MIN_SEGMENT_FRAMES
    if not steady:
        i0, i1 = m0, m1
    fit = fit_stretch(t, x, y, i0, i1, sigma)
    i0, i1 = fit["i0"], fit["i1"]
    v_mid = fit["v0"] + fit["a"] * fit["tau"][len(fit["tau"]) // 2]
    direction = v_mid / np.hypot(*v_mid) if np.hypot(*v_mid) > 0 else np.array([1.0, 0.0])
    a_t = float(fit["a"] @ direction)
    s_at = float(np.sqrt((direction[0] * fit["sa"][0]) ** 2 + (direction[1] * fit["sa"][1]) ** 2))
    a_si = a_t * to_si
    s_a_si = float(np.hypot(s_at * to_si, abs(a_si) * scale_rel))
    n_fit = fit["n"]
    if not steady:
        warnings.append("No stretch of steady acceleration was found; the motion is at constant speed or its acceleration "
                        "keeps changing. The acceleration given is the average over the whole movement.")
    if abs(a_t) < 3 * s_at:
        warnings.append("No acceleration could be measured (the object moves at constant speed, within the noise).")
    if n_fit < SHORT_SEGMENT_FRAMES:
        warnings.append(f"The steady stretch has only {n_fit} frames; the acceleration is imprecise. Film in slow motion.")

    path_angle = float(np.degrees(np.arctan2(abs(direction[1]), abs(direction[0]))))  # from the picture's horizontal
    extra = {
        "peak_speed": {"value": float(sp_si[ipk]), "uncertainty": s_pk, "units": sp_units, "time_s": float(t[ipk] - t[0]),
                       "basis": "highest speed of the local fits, with a 3% allowance for the smoothing of a sharp peak"},
        "average_speed": {"value": float(avg), "uncertainty": s_avg, "units": sp_units,
                          "basis": f"path length / time over the {dur:.2f} s while the object moves"},
        "acceleration": {"value": float(a_si), "uncertainty": s_a_si, "units": acc_units,
                         "basis": f"constant-acceleration fit over {n_fit} frames from {t[i0] - t[0]:.2f} s to {t[i1] - t[0]:.2f} s"},
        "path_angle_deg": path_angle,
        "path_length": {"value": path, "units": "px" if pix else "m"},
        "moving_from_s": float(t[m0] - t[0]), "moving_to_s": float(t[m1] - t[0]),
    }
    # The peak of a noisy series is biased upward (the maximum picks noise spikes), so the acceleration is first
    # averaged over one window; that averaging flattens a sharp peak by a few percent. Both are in the uncertainty.
    a_clean = np.where(np.isfinite(a), a, 0.0)
    mag_a = np.hypot(uniform_filter1d(a_clean[:, 0], window), uniform_filter1d(a_clean[:, 1], window))
    mag_a[:window] = 0.0
    mag_a[-window:] = 0.0
    if mag_a.max() > 0:
        ia = int(np.argmax(mag_a))
        peak_a = float(mag_a[ia] * to_si)
        extra["peak_acceleration"] = {
            "value": peak_a, "units": acc_units, "time_s": float(t[ia] - t[0]),
            "uncertainty": float(np.sqrt((sa * to_si / np.sqrt(window)) ** 2 + (0.05 * peak_a) ** 2 + (peak_a * scale_rel) ** 2)),
            "basis": "peak of the acceleration averaged over one window, with a 5% allowance for the averaging and "
                     "the bias of picking a maximum"}

    if quantity == "speed" and at_time is not None:
        tt = float(at_time) + float(t[0]) if at_time <= t[-1] - t[0] + 1e-9 else float(at_time)
        if not t[0] <= tt <= t[-1] or not np.isfinite(np.interp(tt, t[ok], speed[ok])):
            return _empty(units, method, f"No speed could be measured at t = {at_time:.2f} s (outside the clip or its ends).",
                          warnings)
        val = float(np.interp(tt, t[ok], speed[ok]) * to_si)
        value, unc = val, float(np.hypot(sv * to_si, val * scale_rel))
    elif quantity == "speed":
        value, unc = float(sp_si[ipk]), s_pk
    elif quantity == "average_speed":
        value, unc = float(avg), s_avg
    else:
        value, unc = float(a_si), s_a_si

    if ramp_angle_deg is not None:
        th = float(np.radians(ramp_angle_deg))
        sth = float(np.radians(ramp_angle_uncertainty_deg if ramp_angle_uncertainty_deg is not None
                               else ASSUMED_ANGLE_UNCERTAINTY_DEG))
        if pix:
            warnings.append("The ramp comparison needs a scale: acceleration is in px/s^2, so it cannot be compared with g.")
        else:
            expected, s_exp = {}, {}
            for name, f in SLOPE_FACTORS.items():
                expected[name] = f * G_STANDARD * np.sin(th)
                s_exp[name] = f * G_STANDARD * np.cos(th) * sth
            closest = min(expected, key=lambda k: abs(a_si - expected[k]))
            gs = G_STANDARD * np.sin(th)
            ratio = a_si / gs
            s_ratio = float(np.hypot(s_a_si / gs, ratio * np.cos(th) / np.sin(th) * sth))
            within = abs(a_si - expected[closest]) <= 2.5 * float(np.hypot(s_a_si, s_exp[closest]))
            extra["ramp"] = {"angle_deg": float(ramp_angle_deg), "expected": {k: float(vv) for k, vv in expected.items()},
                             "closest": closest if within else "none (the acceleration matches none of them within its error)",
                             "ratio_to_g_sin_theta": {"value": float(ratio), "uncertainty": s_ratio},
                             "note": "a solid ball rolling without slipping accelerates at 5/7 of g sin(theta), a block "
                                     "sliding without friction at g sin(theta); friction or slipping gives values between."}
        if abs(path_angle - abs(ramp_angle_deg)) > ANGLE_MISMATCH_DEG:
            warnings.append(f"The path in the video is {path_angle:.0f} degrees from horizontal but the ramp angle given is "
                            f"{abs(ramp_angle_deg):.0f} degrees: the angle may be misread, or the phone was tilted or looking "
                            "at the ramp from an angle.")

    chart = {"t": (t - t[0]).tolist(),
             "speed": [None if not np.isfinite(s) else float(s * to_si) for s in speed],
             "tangential_acceleration": [None] * len(t), "fitted_speed": [None] * len(t)}
    for k in range(len(t)):
        if np.isfinite(speed[k]) and speed[k] > 4 * sv and np.isfinite(a[k]).all():
            vh = v[k] / speed[k]
            chart["tangential_acceleration"][k] = float(a[k] @ vh * to_si)
    for k in range(i0, i1 + 1):
        tau_k = t[k] - t[i0]
        chart["fitted_speed"][k] = float(np.hypot(*(fit["v0"] + fit["a"] * tau_k)) * to_si)
    return {"value": value, "uncertainty": unc, "units": units, "method": method,
            "fit_quality": {"window_frames": window, "window_s": window_time, "frames_per_second": fps,
                            "noise_px": sigma, "speed_noise": float(sv * to_si), "n_frames": n_fit,
                            "noise_inflation": fit["inflation"], "scale_uncertainty_fraction": scale_rel,
                            "segment": {"start_s": float(t[i0] - t[0]), "end_s": float(t[i1] - t[0]), "steady": bool(steady)}},
            "series_for_chart": chart, "warnings": warnings, "extra": extra}
