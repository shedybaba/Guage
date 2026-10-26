"""Counting: how many objects there are, and how many cross a line (with a rate per minute).

Both work on the tracks produced by the multi-object quality loop (one track per object).
They are pure maths on tracked positions: no AI is involved in any number produced here.

count_objects: number of distinct objects seen. Tracks that are too short to be real
(flicker) are ignored, ids that changed mid-clip are joined, and anything doubtful widens
the range instead of being hidden: short-but-plausible tracks, low confidence, and an
object that vanished and reappeared in the same place under a new id.

line_crossings: how many times objects cross a line, in each direction, when, and the rate
per minute. A dead band around the line stops jitter from counting as crossings, and the
crossing moment is interpolated between frames.
"""
import numpy as np

from ..quality import find_camera_motion, merge_fragments

MIN_TRACK_SECONDS = 0.3  # shorter tracks are ignored as flicker, unless they touch an edge of the picture/clip
MIN_TRACK_FRAMES = 4
BORDERLINE_FACTOR = 2.0  # tracks up to twice the minimum are counted but make the count uncertain
LOW_CONFIDENCE = 0.5
REENTRY_WINDOW_S = 2.0  # a "new" object this soon, in the same place, may be the old one back
BAND_FRACTION = 0.15  # the dead band around a line: 15% of the object's size


def _empty(units, method, warning):
    return {"value": 0, "uncertainty": 0.0, "units": units, "method": method, "fit_quality": {},
            "series_for_chart": {}, "warnings": [warning], "extra": {}}


def _near_edge(p, frame_size):
    w, h = frame_size
    m = max(10.0, 0.5 * float(np.hypot(p.x2 - p.x1, p.y2 - p.y1)))
    return p.x1 <= m or p.y1 <= m or p.x2 >= w - m or p.y2 >= h - m


def _group(points):
    tracks = {}
    for p in points:
        tracks.setdefault(p.object_id, []).append(p)
    return {k: sorted(v, key=lambda p: p.frame) for k, v in tracks.items()}


# ---------------- how many objects ----------------

def count_objects(tracks, timestamps, frame_size, min_track_seconds=MIN_TRACK_SECONDS,
                  reentry_window_s=REENTRY_WINDOW_S):
    """Count the distinct objects in a clip.

    tracks: {object id: [TrackPoint]} (for example CheckedObjects.tracks).
    timestamps: time of every frame (s). frame_size: (width, height) of the picture.
    """
    method = "distinct tracked objects after joining id changes, ignoring flicker"
    t = np.asarray(timestamps, float)
    flat = [p for pts in tracks.values() for p in pts]
    if not flat:
        return _empty("objects", method, "No objects were tracked, so the count is 0.")
    n_frames = len(t)
    dt = float(np.median(np.diff(t)))
    raw_ids = len(tracks)
    merged = _group(merge_fragments(flat))

    kept, borderline, ignored = {}, [], []
    for oid, pts in merged.items():
        dur = float(t[pts[-1].frame] - t[pts[0].frame]) + dt
        edge = (pts[0].frame == 0 or pts[-1].frame == n_frames - 1
                or _near_edge(pts[0], frame_size) or _near_edge(pts[-1], frame_size))
        if (dur < min_track_seconds or len(pts) < MIN_TRACK_FRAMES) and not edge:
            ignored.append(oid)
            continue
        kept[oid] = pts
        conf = [p.confidence for p in pts if p.confidence is not None]
        if (not edge and dur < BORDERLINE_FACTOR * min_track_seconds) or (conf and np.mean(conf) < LOW_CONFIDENCE):
            borderline.append(oid)

    # an object that vanished mid-picture and a "new" one that appeared in the same place soon after
    reentries = []
    ordered = sorted(kept, key=lambda k: kept[k][0].frame)
    used = set()
    for b in ordered:
        first = kept[b][0]
        if _near_edge(first, frame_size):
            continue
        for a in ordered:
            last = kept[a][-1]
            gap = float(t[first.frame] - t[last.frame])
            diag = float(np.hypot(last.x2 - last.x1, last.y2 - last.y1))
            if (a != b and a not in used and 0 < gap <= reentry_window_s and not _near_edge(last, frame_size)
                    and np.hypot(first.cx - last.cx, first.cy - last.cy) <= diag):
                reentries.append((a, b))
                used.add(a)
                break

    value = len(kept)
    doubtful = set(borderline) | {b for _, b in reentries}
    low = value - len(doubtful)

    present = {}
    for pts in kept.values():
        for p in pts:
            present[p.frame] = present.get(p.frame, 0) + 1
    max_sim = max(present.values()) if present else 0

    warnings = []
    if reentries:
        warnings.append(f"{len(reentries)} object(s) reappeared in the same place within {reentry_window_s:.0f} s "
                        "under a new id (the same place, probably the same object?); they are counted separately here.")
    if borderline:
        warnings.append(f"{len(borderline)} track(s) were very short or low-confidence; they are counted but may be errors.")
    if ignored:
        warnings.append(f"{len(ignored)} very short track(s) were ignored as flicker.")
    warnings.append("The count is only as good as the tracking: an object that was never found, or two touching "
                    "objects tracked as one, cannot be seen from here.")
    starts = sorted(float(t[pts[0].frame]) for pts in kept.values())
    return {"value": value, "uncertainty": (value - low) / 2.0, "units": "objects", "method": method,
            "fit_quality": {"tracks_found": raw_ids, "ids_joined": raw_ids - len(merged), "ignored": len(ignored),
                            "borderline": len(borderline), "possible_reappearances": len(reentries)},
            "series_for_chart": {"t": [s - float(t[0]) for s in starts], "cumulative": list(range(1, value + 1)),
                                 "tracks": [{"id": oid, "start_s": float(t[pts[0].frame] - t[0]),
                                             "end_s": float(t[pts[-1].frame] - t[0])} for oid, pts in kept.items()]},
            "warnings": warnings,
            "extra": {"range": [low, value], "max_simultaneous": max_sim, "ignored_short_tracks": ignored,
                      "borderline_tracks": borderline, "possible_reappearances": [list(p) for p in reentries]}}


# ---------------- line crossings ----------------

def _line_function(line, frame_size):
    """(signed distance function, label for negative->positive, label for positive->negative, description)."""
    w, h = frame_size
    if line is None:
        line = {"orientation": "vertical", "position": w / 2.0}
    if "points" in line:
        (x1, y1), (x2, y2) = line["points"]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            raise ValueError("The two points of the line are the same point.")
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        return (lambda x, y: (x - x1) * nx + (y - y1) * ny, "to_positive_side", "to_negative_side",
                f"line through ({x1:.0f}, {y1:.0f}) and ({x2:.0f}, {y2:.0f})")
    pos = float(line["position"])
    if line.get("orientation") == "horizontal":
        return (lambda x, y: y - pos), "top_to_bottom", "bottom_to_top", f"horizontal line at y = {pos:.0f}"
    if line.get("orientation") == "vertical":
        return (lambda x, y: x - pos), "left_to_right", "right_to_left", f"vertical line at x = {pos:.0f}"
    raise ValueError("line must have orientation 'vertical' or 'horizontal' with a position, or 'points'.")


def _crossings_of_track(frames, d, t, band):
    """Banded crossings of one track: list of (time, +1 or -1), and the number of raw sign changes."""
    sgn = np.sign(d)
    nz = sgn[sgn != 0]
    raw = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    out, state, last_same = [], 0, None
    for i, di in enumerate(d):
        if state == 0:
            if abs(di) > band:
                state, last_same = int(np.sign(di)), i
            continue
        if di * state > 0:
            last_same = i
        if di * state < -band:
            k = last_same  # last point still on the old side; the next one is on or past the line
            d0, d1 = d[k], d[k + 1]
            tc = float(t[k] + d0 * (t[k + 1] - t[k]) / (d0 - d1))
            out.append((tc, -state))
            state, last_same = -state, i
    return out, raw


def line_crossings(tracks, timestamps, frame_size, line=None, band=None, check_camera=True):
    """Count how often objects cross a line, by direction, with times and a rate per minute.

    line: None (vertical line through the middle of the picture), {"orientation": "vertical" or
    "horizontal", "position": pixels}, or {"points": [(x1, y1), (x2, y2)]}.
    band: dead band around the line in pixels (default 15% of the object size): an object must
    get this far past the line before the crossing counts, so jitter on the line is not counted.
    check_camera: warn when all objects shift together (a moving camera drags still objects across a
    fixed line); on a conveyor that is normal, so callers switch it off there.
    """
    method = "sign changes of the object's centre relative to the line, with a dead band"
    t = np.asarray(timestamps, float)
    flat = [p for pts in tracks.values() for p in pts]
    if not flat:
        return _empty("crossings", method, "No objects were tracked, so no crossings could be counted.")
    dist, label_pos, label_neg, desc = _line_function(line, frame_size)
    dt = float(np.median(np.diff(t)))
    duration = float(t[-1] - t[0]) + dt
    size = float(np.median([np.hypot(p.x2 - p.x1, p.y2 - p.y1) for p in flat]))
    band = float(band) if band is not None else max(2.0, BAND_FRACTION * size)

    events, raw_total, straddles = [], 0, False
    for oid, pts in _group(flat).items():
        pts = [p for p in pts if p.frame < len(t)]
        if len(pts) < 2:
            continue
        frames = np.array([p.frame for p in pts])
        d = np.array([dist(p.cx, p.cy) for p in pts])
        straddles |= bool(d.min() < 0 < d.max())
        found, raw = _crossings_of_track(frames, d, t[frames], band)
        raw_total += raw
        events += [{"object": oid, "t": tc - float(t[0]), "direction": label_pos if sign > 0 else label_neg}
                   for tc, sign in found]
    events.sort(key=lambda e: e["t"])
    n = len(events)
    by_dir = {label_pos: sum(e["direction"] == label_pos for e in events),
              label_neg: sum(e["direction"] == label_neg for e in events)}
    jitter = max(0, raw_total - n)

    warnings = []
    if n == 0 and not straddles:
        warnings.append(f"No object ever got to the {desc}: it lies outside the area where the objects moved. "
                        "Check where the line should be.")
    camera = find_camera_motion(tracks, len(t)) if check_camera else []
    if camera:
        warnings.append(f"All objects shifted together ({camera[0]['detail']}). If that is the camera moving, still "
                        "objects are being dragged across the line and the crossings are not real; if it is a "
                        "conveyor carrying everything, this is normal.")
    if jitter:
        warnings.append(f"{jitter} possible crossing(s) were ignored because the object only wobbled around the line "
                        f"(within {band:.0f} px of it).")
    return {"value": n, "uncertainty": jitter / 2.0, "units": "crossings", "method": method,
            "fit_quality": {"line": desc, "dead_band_px": band, "clip_seconds": duration},
            "series_for_chart": {"t": [e["t"] for e in events], "cumulative": list(range(1, n + 1)),
                                 "direction": [e["direction"] for e in events]},
            "warnings": warnings,
            "extra": {"crossings": events, "by_direction": by_dir, "net": by_dir[label_pos] - by_dir[label_neg],
                      "objects_crossed": len({e["object"] for e in events}), "jitter_sensitive": jitter,
                      "camera_motion_suspected": bool(camera),
                      "range": [n, n + jitter], "rate_per_minute": _rate([e["t"] for e in events], duration, dt)}}


def _rate(times, duration, dt):
    """Crossings per minute: from the spacing between crossings when there are enough, else from the whole clip."""
    n = len(times)
    out = {"units": "per minute", "method": "60 / mean time between crossings"}
    if n >= 3:
        gaps = np.diff(times)
        mean = (times[-1] - times[0]) / (n - 1)
        # the spacing varies (a real belt is not perfectly regular): the mean's standard error says how
        # well one clip pins the steady rate; frame timing sets a floor
        sigma_mean = max(float(np.std(gaps, ddof=1) / np.sqrt(n - 1)), np.sqrt(2) * dt / np.sqrt(12) / (n - 1))
        out.update(value=60.0 / mean, uncertainty=60.0 * sigma_mean / mean ** 2,
                   basis=f"between the first and last of {n} crossings")
    else:
        out.update(value=60.0 * n / duration, uncertainty=60.0 * np.sqrt(max(n, 1)) / duration,
                   basis="whole clip (fewer than 3 crossings, so the spacing says little)")
    return out
