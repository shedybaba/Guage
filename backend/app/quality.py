"""Track quality checks and the self-correction loop.

After SAM tracks an object we check the result before measuring anything:
missing frames (gaps), low confidence, jumps, and identity switches. If the
track is not good we pick the fix that matches the problem:

  zero matches, low confidence, jumps -> (a) try another noun phrase,
  missing frames (a gap)              -> (b) re-acquire the object with a box
                                         prompt from the last good frame,
  anything left over                  -> (c) give up gracefully and say what
                                         went wrong and how to film it better.

A gap means the object was hidden or blurred, so another name cannot see
through it, while a box prompt keeps following the very same object (same
mask, same centroid). Mixing names inside one track could shift the centroid.

Every step is written to a plain-language log that the UI shows as
"How Gauge checked its work". Nothing here changes a measurement: it only
decides which tracked points are good enough to measure from.
"""
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

LOW_CONFIDENCE = 0.5
MIN_GAP_FRAMES = 3  # shorter holes are tolerated (the signal is smoothed over them)
MIN_LOW_CONF_RUN = 3
JUMP_FACTOR = 4.0  # a step more than 4x the median step ...
JUMP_MIN_DIAGONALS = 0.5  # ... and more than half the object's size is a jump
MAX_MERGE_GAP_FRAMES = 30
NAME_FIXABLE = ("low_confidence", "jump")  # problems another noun phrase can fix; gaps need a box prompt


@dataclass
class LogEntry:
    step: str  # track | check | retry_phrase | retry_box | merge | give_up
    message: str
    ok: bool | None = None


@dataclass
class CheckedTrack:
    status: str  # ok | warnings | failed
    phrase: str | None
    object_id: str | None
    points: list
    issues: list
    log: list
    advice: list
    frames_billed: int
    total_frames: int

    @property
    def coverage(self) -> float:
        return len(self.points) / self.total_frames if self.total_frames else 0.0


# ---------- checks ----------

def _by_frame(points):
    return {p.frame: p for p in points}


def _median_step_and_diag(points):
    pts = sorted(points, key=lambda p: p.frame)
    steps = [float(np.hypot(b.cx - a.cx, b.cy - a.cy))
             for a, b in zip(pts, pts[1:]) if b.frame - a.frame == 1]
    diag = float(np.median([np.hypot(p.x2 - p.x1, p.y2 - p.y1) for p in pts])) if pts else 0.0
    return (float(np.median(steps)) if steps else 0.0), diag


def _runs(frames):
    """Group sorted frame numbers into (start, end) runs of consecutive frames."""
    runs, start, prev = [], None, None
    for f in frames:
        if start is None:
            start = prev = f
        elif f == prev + 1:
            prev = f
        else:
            runs.append((start, prev))
            start = prev = f
    if start is not None:
        runs.append((start, prev))
    return runs


def jump_threshold(points) -> float:
    step, diag = _median_step_and_diag(points)
    return max(JUMP_FACTOR * step, JUMP_MIN_DIAGONALS * diag)


def _step(ordered, i):
    """Distance moved between ordered[i] and ordered[i+1] if they are consecutive frames, else None."""
    a, b = ordered[i], ordered[i + 1]
    return float(np.hypot(b.cx - a.cx, b.cy - a.cy)) if b.frame - a.frame == 1 else None


def _jump_issues(points):
    """Teleports, not fast motion. A big step is a jump only if it is an isolated spike.

    A collision or a swing that starts suddenly has big steps on both sides
    (speed before and after are similar), so it is NOT a jump. A jump to
    another object either stays there (the steps around it are small) or goes
    out and straight back (the object is nearly where it was two frames ago).
    """
    if len(points) < 3:
        return []
    ordered = sorted(points, key=lambda p: p.frame)
    thr = jump_threshold(points)
    issues = []
    for i in range(len(ordered) - 1):
        s = _step(ordered, i)
        if s is None or s <= thr:
            continue
        neighbours = [v for v in (_step(ordered, i - 1) if i > 0 else None,
                                  _step(ordered, i + 1) if i + 2 < len(ordered) else None) if v is not None]
        isolated = not neighbours or s > 3 * max(neighbours)
        out_and_back = (i + 2 < len(ordered) and ordered[i + 2].frame - ordered[i].frame == 2
                        and np.hypot(ordered[i + 2].cx - ordered[i].cx,
                                     ordered[i + 2].cy - ordered[i].cy) < 0.3 * s)
        if isolated or out_and_back:
            a, b = ordered[i], ordered[i + 1]
            issues.append({"kind": "jump", "start": a.frame, "end": b.frame,
                           "detail": f"position jumped between frames {a.frame} and {b.frame}"})
    return issues


def find_issues(points, total_frames, edge_gaps=True):
    """List the problems in the track of ONE object. Empty list = good track.

    edge_gaps=False ignores missing frames at the very start or end of the clip
    (the object may simply not be in view yet, or have left).
    """
    issues = []
    present = {p.frame for p in points}
    for a, b in _runs([f for f in range(total_frames) if f not in present]):
        if b - a + 1 >= MIN_GAP_FRAMES and (edge_gaps or (a > 0 and b < total_frames - 1)):
            issues.append({"kind": "gap", "start": a, "end": b,
                           "detail": f"object not found in frames {a} to {b}"})
    low = sorted(p.frame for p in points if p.confidence is not None and p.confidence < LOW_CONFIDENCE)
    for a, b in _runs(low):
        if b - a + 1 >= MIN_LOW_CONF_RUN:
            issues.append({"kind": "low_confidence", "start": a, "end": b,
                           "detail": f"low confidence in frames {a} to {b}"})
    issues += _jump_issues(points)
    return sorted(issues, key=lambda i: (i["start"], i["kind"]))


def _near_edge(p, frame_size):
    """True if the object's box touches (or nearly touches) the edge of the picture."""
    w, h = frame_size
    m = max(10.0, 0.5 * float(np.hypot(p.x2 - p.x1, p.y2 - p.y1)))
    return p.x1 <= m or p.y1 <= m or p.x2 >= w - m or p.y2 >= h - m


def find_object_issues(points, total_frames, frame_size):
    """Problems for one of SEVERAL objects, where leaving and entering the picture is normal.

    - Missing frames at the start or end of the clip are ignored (not in view yet / left).
    - A hole in the middle is ignored when the object was at the picture's edge just before
      or after it: it left the frame and came back.
    - An object that vanishes in the middle of the picture and never returns is a real loss.
    """
    issues = []
    for i in find_issues(points, total_frames, edge_gaps=False):
        if i["kind"] == "gap":
            by = _by_frame(points)
            before, after = by.get(i["start"] - 1), by.get(i["end"] + 1)
            if (before is not None and _near_edge(before, frame_size)) or \
               (after is not None and _near_edge(after, frame_size)):
                continue
        issues.append(i)
    if points:
        last = max(points, key=lambda p: p.frame)
        if total_frames - 1 - last.frame >= MIN_GAP_FRAMES and not _near_edge(last, frame_size):
            issues.append({"kind": "gap", "start": last.frame + 1, "end": total_frames - 1,
                           "detail": f"object disappeared in the middle of the picture after frame {last.frame}"})
    return sorted(issues, key=lambda i: (i["start"], i["kind"]))


def merge_fragments(points, max_gap_frames=MAX_MERGE_GAP_FRAMES):
    """Join object ids that are really one object whose id changed mid-clip.

    Two tracks are joined when they are never visible in the same frame, one
    ends shortly before the other starts, and it ends and starts at (nearly)
    the same place. Repeats until nothing more joins, so chains of fragments
    become one track. Objects visible together are never joined. The earlier
    track keeps its id.
    """
    tracks = {}
    for p in points:
        tracks.setdefault(p.object_id, []).append(p)
    if len(tracks) < 2:
        return list(points)
    for k in tracks:
        tracks[k].sort(key=lambda p: p.frame)
    steps = [s for t in tracks.values() for s in
             (float(np.hypot(b.cx - a.cx, b.cy - a.cy)) for a, b in zip(t, t[1:]) if b.frame - a.frame == 1)]
    step = float(np.median(steps)) if steps else 0.0
    diag = float(np.median([np.hypot(p.x2 - p.x1, p.y2 - p.y1) for p in points]))

    while True:
        best = None
        for a, ta in tracks.items():
            for b, tb in tracks.items():
                gap = tb[0].frame - ta[-1].frame
                if a == b or not 0 < gap <= max_gap_frames:
                    continue
                d = float(np.hypot(ta[-1].cx - tb[0].cx, ta[-1].cy - tb[0].cy))
                if d <= max(JUMP_MIN_DIAGONALS * diag, JUMP_FACTOR * step * gap) and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        tracks[a] += [replace(p, object_id=a) for p in tracks.pop(b)]
    return sorted((p for t in tracks.values() for p in t), key=lambda p: p.frame)


# ---------- the retry loop ----------

def _score(points, total_frames):
    """Higher is better: coverage, minus penalties for low confidence and jumps."""
    if not points or not total_frames:
        return 0.0
    low = sum(1 for p in points if p.confidence is not None and p.confidence < LOW_CONFIDENCE)
    jumps = sum(1 for i in find_issues(points, total_frames) if i["kind"] == "jump")
    return len(points) / total_frames - 0.5 * low / total_frames - 0.02 * min(jumps, 10)


def _main_object(points):
    counts = Counter(p.object_id for p in points)
    return counts.most_common(1)[0][0]


def _advice(issues, no_match: bool):
    if no_match:
        return ["Gauge could not find the object. Make sure it is fully visible and well lit, "
                "and name it by what it looks like (for example 'red ball', 'metal spring')."]
    tips = {
        "gap": "The object was lost for part of the clip. Keep it in view, avoid things passing in front of it, "
               "and use a plain background.",
        "low_confidence": "Tracking was unsure in places. Use brighter, even lighting and a background that contrasts with the object.",
        "jump": "The tracked position jumped. Keep the phone completely still and keep similar objects out of view.",
        "camera_motion": "The camera seems to have moved (all objects shifted together), so distances and speeds in "
                         "pixels are unreliable. Prop the phone on something solid or use a tripod.",
    }
    return [tips[k] for k in dict.fromkeys(i["kind"] for i in issues)]


def _fill_from_box(points, new_points, object_id, direction, from_frame, total_frames, others=None):
    """Add frames from a box-prompt run into the gaps of `points`. Never overwrites good frames.

    A run of new frames is accepted only if it starts right next to existing
    points at (nearly) the same place; otherwise it is rejected as a different object.
    `others` (frame -> points of the other tracked objects) lets us drop frames where the
    re-acquired object sits on top of another object, which happens with identical-looking
    objects (the box followed the neighbour). Returns (all points, frames added, runs rejected).
    """
    have = _by_frame(points)
    thr = jump_threshold(points)
    in_range = (lambda f: f > from_frame) if direction == "forward" else (lambda f: f < from_frame)
    fresh = {p.frame: p for p in new_points
             if p.frame not in have and in_range(p.frame) and 0 <= p.frame < total_frames
             and (p.confidence is None or p.confidence >= LOW_CONFIDENCE)}
    added, rejected = [], 0
    if others:
        clash = {f for f, p in fresh.items() if any(
            np.hypot(p.cx - o.cx, p.cy - o.cy) < 0.3 * np.hypot(p.x2 - p.x1, p.y2 - p.y1)
            for o in others.get(f, []))}
        rejected += 1 if clash else 0
        fresh = {f: p for f, p in fresh.items() if f not in clash}
    for a, b in _runs(sorted(fresh)):
        neighbours = [(have.get(a - 1), fresh[a]), (have.get(b + 1), fresh[b])]
        close = [np.hypot(n.cx - f.cx, n.cy - f.cy) <= thr for n, f in neighbours if n is not None]
        if close and all(close):
            added += [replace(fresh[f], object_id=object_id) for f in range(a, b + 1)]
        else:
            rejected += 1
    return sorted(points + added, key=lambda p: p.frame), len(added), rejected


async def track_with_checks(tracker, phrases, total_frames, *, max_phrases=3, max_box_retries=3):
    """Track one object, check the result, and retry in order (phrase, box) until it is good.

    `tracker` needs two async methods: track_phrase(phrase) and
    track_box(frame, (left, top, right, bottom), direction), each returning a TrackResult.
    """
    log, billed = [], 0
    best = None  # (score, phrase, points, issues)

    for n, phrase in enumerate(phrases[:max_phrases]):
        if n:
            log.append(LogEntry("retry_phrase", f"Trying another name for the object: '{phrase}'."))
        res = await tracker.track_phrase(phrase)
        billed += 0 if res.cached else (res.frames_billed or 0)
        if not res.points:
            log.append(LogEntry("track", f"No match found for '{phrase}'.", ok=False))
            continue
        merged = merge_fragments(res.points)
        if len({p.object_id for p in res.points}) > len({p.object_id for p in merged}):
            log.append(LogEntry("merge", "The object's id changed during the clip; joined the pieces into one track.", ok=True))
        main_id = _main_object(merged)
        points = [p for p in merged if p.object_id == main_id]
        issues = find_issues(points, total_frames)
        conf = [p.confidence for p in points if p.confidence is not None]
        log.append(LogEntry("track", f"Tracked '{phrase}' in {len(points)} of {total_frames} frames"
                            + (f", average confidence {np.mean(conf):.2f}." if conf else "."), ok=not issues))
        for i in issues:
            log.append(LogEntry("check", f"Problem: {i['detail']}.", ok=False))
        score = _score(points, total_frames)
        if best is None or score > best[0]:
            best = (score, phrase, points, issues)
        if not any(i["kind"] in NAME_FIXABLE for i in issues):
            if issues and n + 1 < len(phrases[:max_phrases]):
                log.append(LogEntry("check", "Only frames are missing (the object was hidden), so another "
                                    "name would not help; going straight to re-acquiring it.", ok=None))
            break

    if best is None:
        log.append(LogEntry("give_up", "The object was not found under any of the names tried.", ok=False))
        return CheckedTrack("failed", None, None, [], [], log, _advice([], True), billed, total_frames)

    _, phrase, points, issues = best
    if best[3] and len(phrases[:max_phrases]) > 1 and phrase != phrases[0]:
        log.append(LogEntry("retry_phrase", f"Keeping '{phrase}', the best of the names tried.", ok=True))
    object_id = points[0].object_id

    attempted, calls = set(), 0
    while calls < max_box_retries:
        gaps = [i for i in issues if i["kind"] == "gap" and (i["start"], i["end"]) not in attempted]
        if not gaps:
            break
        gap = gaps[0]
        attempted.add((gap["start"], gap["end"]))
        by_frame = _by_frame(points)
        if gap["start"] == 0:
            direction, from_frame = "backward", gap["end"] + 1
        else:
            direction, from_frame = "forward", gap["start"] - 1
        anchor = by_frame.get(from_frame)
        if anchor is None:
            continue
        log.append(LogEntry("retry_box", f"Re-acquiring the object from frame {from_frame} "
                            f"({direction}) to fill frames {gap['start']} to {gap['end']}."))
        res = await tracker.track_box(from_frame, (anchor.x1, anchor.y1, anchor.x2, anchor.y2), direction)
        calls += 1
        billed += 0 if res.cached else (res.frames_billed or 0)
        points, added, rejected = _fill_from_box(points, res.points, object_id, direction, from_frame, total_frames)
        if added:
            log.append(LogEntry("retry_box", f"Re-acquired the object in {added} more frames.", ok=True))
        if rejected:
            log.append(LogEntry("retry_box", f"Rejected {rejected} re-acquired stretch(es) that jumped to a "
                                "different place (probably another object).", ok=False))
        if not added and not rejected:
            log.append(LogEntry("retry_box", "The object was not found again in those frames.", ok=False))
        issues = find_issues(points, total_frames)

    if issues:
        log.append(LogEntry("give_up", "Some problems could not be fixed. The measurement will carry a warning.", ok=False))
        status = "warnings"
    else:
        log.append(LogEntry("check", "Checks passed: no gaps, no low-confidence stretches, no jumps.", ok=True))
        status = "ok"
    return CheckedTrack(status, phrase, object_id, points, issues, log, _advice(issues, False), billed, total_frames)


# ---------- several objects at once (counting, collisions) ----------

CAMERA_STEP_FRACTION = 0.02  # all objects moving more than 2% of their size per frame, together ...
CAMERA_MIN_FRAME_SHARE = 0.10  # ... in at least 10% of the frames, suggests the camera moved
MAX_OBJECTS_PER_REQUEST = 16  # SAM tracks at most 16 objects per request


@dataclass
class CheckedObjects:
    status: str  # ok | warnings | failed
    phrase: str | None
    tracks: dict  # object id -> list of points, one entry per object
    issues: dict  # object id -> remaining problems
    scene_issues: list  # problems about the whole scene (camera motion)
    log: list
    advice: list
    frames_billed: int
    total_frames: int

    @property
    def count(self) -> int:
        return len(self.tracks)


def find_camera_motion(tracks, total_frames):
    """Detect a moving camera: three or more objects shifting the same way, frame after frame.

    The median displacement over the objects ignores the one that is really moving (a struck ball)
    and cancels objects that move in different directions.
    Not valid when everything legitimately moves together (parts on a conveyor), so
    callers can switch it off.
    """
    by = {oid: _by_frame(pts) for oid, pts in tracks.items()}
    sizes = [np.hypot(p.x2 - p.x1, p.y2 - p.y1) for pts in tracks.values() for p in pts]
    if len(by) < 3 or not sizes:
        return []
    limit = CAMERA_STEP_FRACTION * float(np.median(sizes))
    moving, counted = [], 0
    for f in range(1, total_frames):
        d = [(b[f].cx - b[f - 1].cx, b[f].cy - b[f - 1].cy) for b in by.values() if f in b and f - 1 in b]
        if len(d) < 3:
            continue
        counted += 1
        # a camera shifts (nearly) every object the SAME way: the typical displacement is large AND
        # most objects agree with it. A struck ball among still ones, or objects swinging
        # independently, do not agree.
        mx, my = float(np.median([v[0] for v in d])), float(np.median([v[1] for v in d]))
        if np.hypot(mx, my) <= limit:
            continue
        agree = sum(1 for v in d if np.hypot(v[0] - mx, v[1] - my) <= max(0.35 * np.hypot(mx, my), 0.5 * limit))
        if agree >= max(3, int(np.ceil(0.75 * len(d)))):
            moving.append(f)
    if counted and len(moving) >= 5 and len(moving) / counted >= CAMERA_MIN_FRAME_SHARE:
        return [{"kind": "camera_motion", "start": moving[0], "end": moving[-1],
                 "detail": f"all objects shifted together in {100 * len(moving) / counted:.0f}% of the frames "
                           "(the camera probably moved)"}]
    return []


def _group_by_object(points):
    tracks = {}
    for p in points:
        tracks.setdefault(p.object_id, []).append(p)
    return {k: sorted(v, key=lambda p: p.frame) for k, v in sorted(tracks.items())}


async def track_objects_with_checks(tracker, phrases, total_frames, frame_size, *, max_phrases=3,
                                    max_box_retries=3, check_camera=True):
    """Like track_with_checks, but for every object a phrase matches (parts, balls, ...).

    - Another name is tried only for zero matches, low confidence or jumps; the group of
      objects that scores best is kept.
    - Objects that left the picture and came back, or that were not in view yet, are normal.
    - Objects lost in the middle of the picture are re-acquired with box prompts, ALL of
      them in one request (up to 16 objects), so the cost does not multiply with the count.
    - A re-acquired object that lands on top of another object is rejected.
    `tracker` needs track_phrase(phrase) and track_boxes([(frame, box), ...], direction); in the
    result of track_boxes the object ids are "0", "1", ... in the order the boxes were given.
    """
    log, billed, best = [], 0, None
    names = phrases[:max_phrases]

    for n, phrase in enumerate(names):
        if n:
            log.append(LogEntry("retry_phrase", f"Trying another name for the objects: '{phrase}'."))
        res = await tracker.track_phrase(phrase)
        billed += 0 if res.cached else (res.frames_billed or 0)
        if not res.points:
            log.append(LogEntry("track", f"No match found for '{phrase}'.", ok=False))
            continue
        merged = merge_fragments(res.points)
        before, after = len({p.object_id for p in res.points}), len({p.object_id for p in merged})
        if after < before:
            log.append(LogEntry("merge", f"Joined {before - after} object id(s) that were the same object "
                                "after an id change.", ok=True))
        tracks = _group_by_object(merged)
        issues = {oid: find_object_issues(pts, total_frames, frame_size) for oid, pts in tracks.items()}
        log.append(LogEntry("track", f"Tracked '{phrase}': found {len(tracks)} object(s).",
                            ok=not any(issues.values())))
        for oid, iss in issues.items():
            for i in iss:
                log.append(LogEntry("check", f"Object {oid}: {i['detail']}.", ok=False))
        score = sum(_score(pts, total_frames) for pts in tracks.values())
        if best is None or score > best[0]:
            best = (score, phrase, tracks, issues)
        if not any(i["kind"] in NAME_FIXABLE for iss in issues.values() for i in iss):
            if any(issues.values()) and n + 1 < len(names):
                log.append(LogEntry("check", "Only frames are missing (objects were hidden), so another name "
                                    "would not help; going straight to re-acquiring them.", ok=None))
            break

    if best is None:
        log.append(LogEntry("give_up", "No objects were found under any of the names tried.", ok=False))
        return CheckedObjects("failed", None, {}, {}, [], log, _advice([], True), billed, total_frames)

    _, phrase, tracks, issues = best
    if phrase != names[0]:
        log.append(LogEntry("retry_phrase", f"Keeping '{phrase}', the best of the names tried.", ok=True))

    attempted, calls = set(), 0
    while calls < max_box_retries:
        chosen = {}
        for oid, iss in issues.items():
            for g in iss:
                if g["kind"] == "gap" and (oid, g["start"], g["end"]) not in attempted:
                    chosen.setdefault(oid, g)
        batch, prompts = [], []
        for oid, g in list(chosen.items())[:MAX_OBJECTS_PER_REQUEST]:
            attempted.add((oid, g["start"], g["end"]))
            anchor = _by_frame(tracks[oid]).get(g["start"] - 1)
            if anchor is not None:
                batch.append((oid, g, anchor))
                prompts.append((anchor.frame, (anchor.x1, anchor.y1, anchor.x2, anchor.y2)))
        if not prompts:
            if not chosen:
                break
            continue
        log.append(LogEntry("retry_box", "Re-acquiring object(s) " + ", ".join(str(o) for o, _, _ in batch)
                            + f" from their last good frame in one request."))
        res = await tracker.track_boxes(prompts, "forward")
        calls += 1
        billed += 0 if res.cached else (res.frames_billed or 0)
        returned = {}
        for p in res.points:
            returned.setdefault(p.object_id, []).append(p)
        for idx, (oid, g, anchor) in enumerate(batch):
            others = {}
            for k, pts in tracks.items():
                if k != oid:
                    for p in pts:
                        others.setdefault(p.frame, []).append(p)
            tracks[oid], added, rejected = _fill_from_box(
                tracks[oid], returned.get(str(idx), []), oid, "forward", anchor.frame, total_frames, others)
            if added:
                log.append(LogEntry("retry_box", f"Object {oid}: re-acquired in {added} more frames.", ok=True))
            if rejected:
                log.append(LogEntry("retry_box", f"Object {oid}: rejected {rejected} re-acquired stretch(es) "
                                    "that jumped elsewhere or landed on another object.", ok=False))
            if not added and not rejected:
                log.append(LogEntry("retry_box", f"Object {oid}: not found again in those frames.", ok=False))
            issues[oid] = find_object_issues(tracks[oid], total_frames, frame_size)

    scene = find_camera_motion(tracks, total_frames) if check_camera else []
    for i in scene:
        log.append(LogEntry("check", f"Scene: {i['detail']}.", ok=False))
    remaining = [i for iss in issues.values() for i in iss] + scene
    if remaining:
        log.append(LogEntry("give_up", "Some problems could not be fixed. Measurements will carry a warning.", ok=False))
    else:
        log.append(LogEntry("check", f"Checks passed for all {len(tracks)} objects.", ok=True))
    return CheckedObjects("warnings" if remaining else "ok", phrase, tracks, issues, scene, log,
                          _advice(remaining, False), billed, total_frames)
