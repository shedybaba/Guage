"""Track quality checks and the self-correction loop.

After SAM tracks an object we check the result before measuring anything:
missing frames (gaps), low confidence, jumps, and identity switches. If the
track is not good we retry in a fixed order:

  (a) try another noun phrase for the same object,
  (b) re-acquire the object with a box prompt from the last good frame,
  (c) give up gracefully and say what went wrong and how to film it better.

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


def find_issues(points, total_frames):
    """List the problems in the track of ONE object. Empty list = good track."""
    issues = []
    present = {p.frame for p in points}
    for a, b in _runs([f for f in range(total_frames) if f not in present]):
        if b - a + 1 >= MIN_GAP_FRAMES:
            issues.append({"kind": "gap", "start": a, "end": b,
                           "detail": f"object not found in frames {a} to {b}"})
    low = sorted(p.frame for p in points if p.confidence is not None and p.confidence < LOW_CONFIDENCE)
    for a, b in _runs(low):
        if b - a + 1 >= MIN_LOW_CONF_RUN:
            issues.append({"kind": "low_confidence", "start": a, "end": b,
                           "detail": f"low confidence in frames {a} to {b}"})
    if len(points) >= 3:
        thr = jump_threshold(points)
        ordered = sorted(points, key=lambda p: p.frame)
        for a, b in zip(ordered, ordered[1:]):
            if b.frame - a.frame == 1 and np.hypot(b.cx - a.cx, b.cy - a.cy) > thr:
                issues.append({"kind": "jump", "start": a.frame, "end": b.frame,
                               "detail": f"position jumped between frames {a.frame} and {b.frame}"})
    return sorted(issues, key=lambda i: (i["start"], i["kind"]))


def merge_fragments(points, max_gap_frames=MAX_MERGE_GAP_FRAMES):
    """Join object ids that are really one object whose id changed mid-clip.

    A fragment is merged into the main object only if they are never visible in
    the same frame and the fragment starts or ends right next to the main track
    at (nearly) the same place. Objects visible together are left alone.
    """
    groups = {}
    for p in points:
        groups.setdefault(p.object_id, []).append(p)
    if len(groups) < 2:
        return list(points)
    main_id = max(groups, key=lambda k: len(groups[k]))
    merged = {p.frame: p for p in groups[main_id]}
    step, diag = _median_step_and_diag(groups[main_id])
    others = sorted((k for k in groups if k != main_id), key=lambda k: min(p.frame for p in groups[k]))
    result = list(groups[main_id])
    for k in others:
        frag = sorted(groups[k], key=lambda p: p.frame)
        if any(p.frame in merged for p in frag):
            result += frag  # visible at the same time: a different object, keep it separate
            continue
        first, last = frag[0], frag[-1]
        before = max((f for f in merged if f < first.frame), default=None)
        after = min((f for f in merged if f > last.frame), default=None)

        def close(a, b, gap):
            return gap <= max_gap_frames and np.hypot(a.cx - b.cx, a.cy - b.cy) <= max(
                JUMP_MIN_DIAGONALS * diag, JUMP_FACTOR * step * gap)

        ok = (before is not None and close(merged[before], first, first.frame - before)) or \
             (after is not None and close(last, merged[after], after - last.frame))
        if ok:
            for p in frag:
                q = replace(p, object_id=main_id)
                merged[q.frame] = q
                result.append(q)
        else:
            result += frag  # not continuous with the main track: keep it as its own object
    return sorted(result, key=lambda p: p.frame)


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
    }
    return [tips[k] for k in dict.fromkeys(i["kind"] for i in issues)]


def _fill_from_box(points, new_points, object_id, direction, from_frame, total_frames):
    """Add frames from a box-prompt run into the gaps of `points`. Never overwrites good frames.

    A run of new frames is accepted only if it starts right next to existing
    points at (nearly) the same place; otherwise it is rejected as a different object.
    Returns (all points, frames added, runs rejected).
    """
    have = _by_frame(points)
    thr = jump_threshold(points)
    in_range = (lambda f: f > from_frame) if direction == "forward" else (lambda f: f < from_frame)
    fresh = {p.frame: p for p in new_points
             if p.frame not in have and in_range(p.frame) and 0 <= p.frame < total_frames
             and (p.confidence is None or p.confidence >= LOW_CONFIDENCE)}
    added, rejected = [], 0
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
        if not issues:
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
