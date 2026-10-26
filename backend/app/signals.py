"""Turn tracked points into clean time series and basic quality numbers.

Note the image y axis points DOWN. Time comes from the real frame timestamps.
"""
from collections import defaultdict

import numpy as np

LOW_CONFIDENCE = 0.5
JUMP_FACTOR = 4.0  # a step more than 4x the median step counts as a jump


def pick_main_object(points):
    """The object id that appears in the most frames (ties: higher mean confidence)."""
    by_id = defaultdict(list)
    for p in points:
        by_id[p.object_id].append(p)
    if not by_id:
        return None
    def score(item):
        pts = item[1]
        conf = [p.confidence for p in pts if p.confidence is not None]
        return (len(pts), float(np.mean(conf)) if conf else 0.0)
    return max(by_id.items(), key=score)[0]


def series_for_object(points, object_id, timestamps):
    """Arrays t, cx, cy, area, confidence for one object, ordered by frame."""
    pts = sorted((p for p in points if p.object_id == object_id and p.frame < len(timestamps)),
                 key=lambda p: p.frame)
    t = np.array([timestamps[p.frame] for p in pts])
    return {
        "frame": np.array([p.frame for p in pts]),
        "t": t,
        "x": np.array([p.cx for p in pts]),
        "y": np.array([p.cy for p in pts]),
        "area": np.array([p.area for p in pts]),
        "confidence": np.array([np.nan if p.confidence is None else p.confidence for p in pts]),
    }


def quality_report(series, total_frames):
    """Gaps, low confidence and jumps: the inputs to the retry loop later."""
    frames = series["frame"]
    if len(frames) < 3:
        return {"tracked_frames": int(len(frames)), "total_frames": total_frames, "coverage": 0.0,
                "gap_count": 0, "missing_frames": total_frames - int(len(frames)),
                "low_confidence_frames": 0, "jump_count": 0,
                "mean_confidence": None, "min_confidence": None}
    gaps = int(np.sum(np.diff(frames) > 1))
    conf = series["confidence"]
    valid = conf[np.isfinite(conf)]
    steps = np.hypot(np.diff(series["x"]), np.diff(series["y"]))
    med = float(np.median(steps)) or 1e-9
    return {
        "tracked_frames": int(len(frames)),
        "total_frames": total_frames,
        "coverage": len(frames) / total_frames,
        "gap_count": gaps,
        "missing_frames": total_frames - int(len(frames)),
        "low_confidence_frames": int(np.sum(valid < LOW_CONFIDENCE)),
        "jump_count": int(np.sum(steps > JUMP_FACTOR * med)),
        "mean_confidence": float(valid.mean()) if len(valid) else None,
        "min_confidence": float(valid.min()) if len(valid) else None,
    }
