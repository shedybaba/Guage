"""Rotation speed (rpm) of a fan, wheel or shaft from a marker that goes round with it.

The marker (a sticker, a painted dot, a blade tip) is tracked, and its centre goes round the axis in a circle.

  1. a circle is fitted to the positions: its centre is the axis, its radius is the marker's distance from it
  2. the angle of the marker round that centre is worked out in every frame and unwrapped (counted in whole turns)
  3. the angle grows in a straight line when the speed is steady; the slope is the angular speed, rpm = slope x 30 / pi
  4. an independent check: the frequency of the marker's position (as a complex number) by Fourier transform

A camera takes a picture every 1/30 s, so a fast marker can look slower or even backwards (the wagon-wheel effect). When
the marker turns by a large part of a revolution between frames the speed cannot be told from a slower or backward one,
so the tool refuses instead of guessing. The stretch where the marker is really turning is found automatically (a fan being
switched on is at rest first), and a speed that changes during the stretch is reported as such.

Positive angles are counter-clockwise as seen in the video. Pure maths: no AI is involved in any number produced here.
"""
import numpy as np
from scipy.optimize import least_squares

MIN_SAMPLES = 20
WARN_STEP_RAD = 1.5  # turning more than ~86 degrees per frame: fewer than ~4 frames per turn
REFUSE_STEP_RAD = 2.5  # more than ~143 degrees per frame: cannot be told from a slower or backward rotation
MAX_RHO = 0.97
MIN_TURNS_REFUSE = 0.5
TIMESTAMP_RESOLUTION_S = 1e-3
CLOCK_ERROR = 1e-4  # relative: video timestamps are assumed true to 0.01%
MIN_TURNS_WARN = 2.0
MOVING_FRACTION = 0.25  # the marker is "turning" where its local speed is above 25% of its fast speed
JUMP_REFUSE = 0.9  # median step as a share of the orbit's width: about 128 degrees per frame or more
CIRCLE_WARN, CIRCLE_REFUSE = 0.12, 0.30  # radial scatter relative to the radius


_TOO_FAST = ("The marker turns more than about 140 degrees between frames, so a fast rotation cannot be told from a "
             "slower or a backward one (the wagon-wheel effect). Film at a higher frame rate (slow motion), or slow the "
             "rotation down.")


def _empty(warning):
    return {"value": None, "uncertainty": None, "units": "rpm", "method": "angle of the marker round its fitted circle",
            "fit_quality": {}, "series_for_chart": {}, "warnings": [warning], "extra": {}}


def _fit_circle(x, y):
    """Centre and radius of the circle through the points (algebraic fit, then a robust geometric refinement)."""
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    r0 = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9)))
    scale = max(1.0, 0.01 * r0)
    res = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2], [cx, cy, r0], loss="soft_l1", f_scale=scale)
    cx, cy, r = res.x
    return float(cx), float(cy), float(abs(r))


def _rho(resid):
    if len(resid) < 8 or np.std(resid) == 0:
        return 0.0
    r = resid - resid.mean()
    return float(np.clip(np.sum(r[1:] * r[:-1]) / np.sum(r * r), 0.0, MAX_RHO))


def _slope(t, y):
    """Straight-line slope of y over t with its standard error, widened for correlated residuals, and the residuals."""
    tm = t - t.mean()
    sxx = float(np.sum(tm ** 2))
    slope = float(np.sum(tm * (y - y.mean())) / sxx)
    resid = y - (y.mean() + slope * tm)
    sigma = float(np.sqrt(np.sum(resid ** 2) / max(len(t) - 2, 1)))
    rho = _rho(resid)
    return slope, sigma / np.sqrt(sxx) * np.sqrt((1 + rho) / (1 - rho)), resid, rho


def _local_speed(t, theta, window_s, n_eval=300):
    """Angular speed (rad/s) around the clip, from short straight-line fits: (centres in time, speeds)."""
    n = len(t)
    centres = np.unique(np.linspace(0, n - 1, min(n, n_eval)).astype(int))
    out_t, out_w = [], []
    for c in centres:
        lo = np.searchsorted(t, t[c] - window_s / 2)
        hi = np.searchsorted(t, t[c] + window_s / 2, side="right")
        if hi - lo >= 5:
            out_t.append(t[c])
            out_w.append(_slope(t[lo:hi], theta[lo:hi])[0])
    return np.array(out_t), np.array(out_w)


def _longest_run(mask, min_len):
    best, start = (0, 0), None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best if best[1] - best[0] >= min_len else None


def rotation_rate(t, x, y):
    """Revolutions per minute of a marker going round an axis.

    t: time of each frame (s). x, y: the marker's centre in pixels (the image y axis points down).
    """
    t, x, y = (np.asarray(a, float) for a in (t, x, y))
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    if len(t) < MIN_SAMPLES:
        return _empty(f"Only {len(t)} frames with a position: at least {MIN_SAMPLES} are needed to measure rotation.")
    extent = float(max(np.ptp(x), np.ptp(y)))
    if extent < 6.0:
        return _empty("The marker hardly moves in the picture, so there is no rotation to measure. Make sure the tracked "
                      "object is a marker that goes round with the wheel or fan.")
    warnings = []
    # Too fast to read? A marker that jumps almost across its whole orbit between frames is turning about half a revolution
    # per frame or more. It then sits at only a few spots, so no circle can be fitted, and the speed cannot be told from a
    # slower or backward one (the wagon-wheel effect). This is checked first, because it needs no circle.
    jump = float(np.median(np.hypot(np.diff(x), np.diff(y)))) / extent
    if jump >= JUMP_REFUSE:
        return _empty(_TOO_FAST)
    cx, cy, R = _fit_circle(x, y)
    if R > 2.0 * extent:
        return _empty("The marker moves in a nearly straight line rather than round a centre, so no rotation speed can be "
                      "measured. Track a marker fixed to the turning part.")
    radial = np.hypot(x - cx, y - cy) - R
    sigma_r = 1.4826 * float(np.median(np.abs(radial - np.median(radial))))
    circle_scatter = sigma_r / R
    if circle_scatter > CIRCLE_REFUSE:
        return _empty("The marker's path is not a circle, so its rotation speed cannot be measured. The turning part may be "
                      "moving across the picture (a rolling wheel), or the tracked object is not fixed to it.")
    if circle_scatter > CIRCLE_WARN:
        warnings.append("The marker's path is not a very clean circle (the axis may drift or the marker slides), so the "
                        "speed is less certain.")
    angle_noise = float(np.hypot(sigma_r, 0.3) / R)  # rad
    if angle_noise > 0.5:
        return _empty("The marker is too close to the centre of rotation, compared with the tracking noise, to read its "
                      "angle. Put the marker nearer the edge of the turning part.")

    theta = np.arctan2(cy - y, x - cx)  # counter-clockwise on screen is positive
    step = np.angle(np.exp(1j * np.diff(theta)))  # each frame-to-frame change, wrapped to (-pi, pi]
    dts = np.diff(t)
    dt = float(np.median(dts))
    med_step = float(np.median(np.abs(step)))
    if med_step >= REFUSE_STEP_RAD:
        return _empty(_TOO_FAST)
    # a gap in the tracking where the marker could turn a large part of a revolution cannot be followed: use the longest piece
    w0 = float(np.median(np.abs(step) / dts))
    breaks = np.where(w0 * dts > 2.0)[0] + 1
    pieces = np.split(np.arange(len(t)), breaks)
    piece = max(pieces, key=len)
    if len(piece) < MIN_SAMPLES:
        return _empty("The marker was lost for too long between sightings to follow its rotation. Try again with the "
                      "marker in view throughout.")
    if len(pieces) > 1:
        warnings.append("The marker was lost for a while, and the rotation cannot be followed across such a gap, so only the "
                        "longest unbroken part of the clip was used.")
    t, theta, x, y = t[piece], theta[piece], x[piece], y[piece]
    step = np.angle(np.exp(1j * np.diff(theta)))
    unwrapped = theta[0] + np.concatenate([[0.0], np.cumsum(step)])

    # where is it really turning? (a fan switched on is at rest first, and may be switched off at the end)
    win = max(0.5, 7 * dt)
    lt, lw = _local_speed(t, unwrapped, win)
    if len(lw) < 3:
        return _empty("Too few frames to find the rotation speed.")
    fast = float(np.percentile(np.abs(lw), 90))
    if fast <= 0:
        return _empty("The marker does not turn.")
    moving = np.abs(lw) > MOVING_FRACTION * fast
    run = _longest_run(moving, max(3, int(0.5 / max(np.median(np.diff(lt)), 1e-9))))
    if run is None:
        return _empty("The marker does not turn steadily for long enough to measure its speed.")
    t_from = float(lt[run[0]] - 0.5 * np.median(np.diff(lt)))
    t_to = float(lt[run[1] - 1] + 0.5 * np.median(np.diff(lt)))
    sel = (t >= t_from) & (t <= t_to)
    if sel.sum() < MIN_SAMPLES // 2:
        return _empty("The marker does not turn steadily for long enough to measure its speed.")
    ts, us = t[sel], unwrapped[sel]
    trimmed = bool(ts[0] - t[0] > 0.5 or t[-1] - ts[-1] > 0.5)
    if trimmed:
        warnings.append(f"The marker was turning only from {ts[0]:.1f} s to {ts[-1]:.1f} s of the clip (at rest "
                        "before or after), so only that part was used.")

    omega, se, resid, rho = _slope(ts, us)
    span = float(ts[-1] - ts[0])
    turns = abs(omega) * span / (2 * np.pi)
    if turns < MIN_TURNS_REFUSE:
        return _empty("The marker turns less than half a revolution in the clip, which is too little to measure a speed.")
    if turns < MIN_TURNS_WARN:
        warnings.append(f"Only about {turns:.1f} revolutions were seen; film for longer for a steadier result.")
    # Timestamps are exact to about a millisecond and average out over the frames of the fit; a phone's clock may be off a
    # little. (Not the frame period: the marker is sampled at the frame's own time, so a frame is not 'rounded' to a time.)
    floor = abs(omega) * float(np.hypot(TIMESTAMP_RESOLUTION_S / (span * np.sqrt(len(ts))), CLOCK_ERROR))
    unc_omega = max(se, floor)
    rpm, rpm_unc = abs(omega) * 30 / np.pi, unc_omega * 30 / np.pi
    freq, freq_unc = abs(omega) / (2 * np.pi), unc_omega / (2 * np.pi)
    frames_per_rev = 2 * np.pi / max(abs(omega) * dt, 1e-12)
    if float(np.median(np.abs(step))) >= WARN_STEP_RAD:
        warnings.append(f"The marker makes only about {frames_per_rev:.1f} frames per revolution, so the frame rate is close "
                        "to the limit for this speed; film in slow motion to be sure the speed is right.")

    # the independent check: Fourier transform of the marker's position as a complex number
    fft_rpm = None
    if span > 0 and len(ts) >= 16:
        grid = np.arange(ts[0], ts[-1], dt)
        z = np.interp(grid, ts, x[sel] - cx) + 1j * np.interp(grid, ts, cy - y[sel])
        spec = np.fft.fft((z - z.mean()) * np.hanning(len(z)), 16 * len(z))
        k = int(np.argmax(np.abs(spec)))
        mag = np.abs(spec)
        if 0 < k < len(spec) - 1:
            a, b, c = mag[k - 1], mag[k], mag[k + 1]
            k = k + 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else k
        n_bins = len(spec)
        f_peak = (k if k <= n_bins / 2 else k - n_bins) / (n_bins * dt)
        fft_rpm = abs(f_peak) * 60.0
        tol = max(3 * freq_unc, 0.8 / span)
        if abs(abs(f_peak) - freq) > tol:
            warnings.append("A second method (a Fourier analysis of the marker's position) gives a different speed "
                            f"({fft_rpm:.1f} rpm), so treat this result with caution: the tracking may have jumped or the "
                            "marker may have been mistaken for something else.")

    # does the speed change within the stretch?
    mid = len(ts) // 2
    rpm_first = rpm_second = None
    if mid >= 8 and len(ts) - mid >= 8:
        w1, s1 = _slope(ts[:mid], us[:mid])[:2]
        w2, s2 = _slope(ts[mid:], us[mid:])[:2]
        rpm_first, rpm_second = abs(w1) * 30 / np.pi, abs(w2) * 30 / np.pi
        if abs(w1 - w2) > max(3 * np.hypot(s1, s2), 0.05 * abs(omega)):
            warnings.append(f"The rotation speed changes during the clip (about {rpm_first:.0f} rpm at first, about "
                            f"{rpm_second:.0f} rpm at the end), so the result is the average over that stretch.")

    direction = "counter-clockwise" if omega > 0 else "clockwise"
    fitted = us.mean() + omega * (t - ts.mean())  # the straight line through the turning stretch, drawn over the whole clip
    series = {"t": [float(v) for v in t], "turns": [float((u - us[0]) / (2 * np.pi)) for u in unwrapped],
              "fitted": [float((f_ - us[0]) / (2 * np.pi)) for f_ in fitted]}
    return {"value": float(rpm), "uncertainty": float(rpm_unc), "units": "rpm",
            "method": "angle of the marker round the axis, counted in whole turns, with a straight line fitted to it",
            "fit_quality": {"r2": float(1 - np.sum(resid ** 2) / np.sum((us - us.mean()) ** 2)) if np.ptp(us) > 0 else 0.0,
                            "n_frames": int(len(ts)), "revolutions": float(turns), "residual_correlation": rho,
                            "frames_per_revolution": float(frames_per_rev),
                            "segment": {"start_s": float(ts[0]), "end_s": float(ts[-1]), "trimmed": trimmed},
                            "circle_scatter": float(circle_scatter), "timing_floor_rpm": float(floor * 30 / np.pi)},
            "series_for_chart": series, "warnings": warnings,
            "extra": {"frequency": {"value": float(freq), "uncertainty": float(freq_unc), "units": "Hz"},
                      "period": {"value": float(1 / freq), "uncertainty": float(freq_unc / freq ** 2), "units": "s"},
                      "angular_speed": {"value": float(abs(omega)), "uncertainty": float(unc_omega), "units": "rad/s"},
                      "revolutions": {"value": float(turns), "uncertainty": float(rpm_unc * span / 60), "units": ""},
                      "direction": direction, "centre_px": [cx, cy], "radius_px": R,
                      "rpm_first_half": rpm_first, "rpm_second_half": rpm_second, "fft_rpm": fft_rpm}}
