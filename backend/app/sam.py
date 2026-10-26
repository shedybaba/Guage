"""SAM 3.1 tracking: one streamed request per noun phrase, parsed into per-frame points.

A phrase is a short noun phrase naming ONE object ("pendulum bob"). Zero matches
is a valid result (empty list), not an error. Results are cached per
(file_id, phrase) on disk so follow-up questions and re-runs are not billed again.
"""
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from meta_sam_parser import (SegmentationBoxRecord, SegmentationMaskRecord,
                             decode_mask_to_raster, parse_responses_stream,
                             video_segmentation_format)

from .config import DATA_DIR, SAM_MODEL


@dataclass
class TrackPoint:
    frame: int
    object_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None
    cx: float  # centroid of the mask (more stable than the box centre)
    cy: float
    area: float  # mask area in source pixels


@dataclass
class TrackResult:
    phrase: str
    points: list  # list[TrackPoint]
    frames_billed: int | None
    diagnostics: list


def _mask_stats(rec: SegmentationMaskRecord):
    """Centroid and area of a box-local mask, in source-video pixels."""
    raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8)
    w, h = rec.mask.width, rec.mask.height
    ys, xs = np.nonzero(raster.reshape(h, w))
    if len(xs) == 0:
        return None
    b = rec.bounds
    sx, sy = (b.right - b.left) / w, (b.bottom - b.top) / h  # mask may be scaled to the box
    cx = b.left + (xs.mean() + 0.5) * sx
    cy = b.top + (ys.mean() + 0.5) * sy
    return cx, cy, float(len(xs) * sx * sy)


def _build_points(records) -> list:
    boxes, masks = {}, {}
    for r in records:
        if r.frame is None:
            continue
        key = (r.frame.frame_index, r.object_id)
        if isinstance(r, SegmentationBoxRecord):
            boxes[key] = r
        elif isinstance(r, SegmentationMaskRecord):
            masks[key] = r
    points = []
    for key, b in sorted(boxes.items()):
        stats = _mask_stats(masks[key]) if key in masks else None
        cx, cy, area = stats or ((b.left + b.right) / 2, (b.top + b.bottom) / 2,
                                 (b.right - b.left) * (b.bottom - b.top))
        points.append(TrackPoint(key[0], key[1], b.left, b.top, b.right, b.bottom,
                                 b.confidence, cx, cy, area))
    return points


async def _events(stream, usage_out: dict):
    """Pass events to the parser while remembering billed frames from the final event."""
    async for event in stream:
        if event.type == "response.completed":
            usage = getattr(event.response, "usage", None)
            usage_out["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
        yield event


def video_key(path) -> str:
    """Content hash of a video file: the cache key that survives re-uploads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cache_file(key: str, phrase: str):
    h = hashlib.sha256(f"{key}|{phrase}".encode()).hexdigest()[:20]
    return DATA_DIR / "sam_cache" / f"{h}.json"


def load_cached(key: str, phrase: str):
    p = _cache_file(key, phrase)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return TrackResult(d["phrase"], [TrackPoint(**x) for x in d["points"]], d["frames_billed"], d["diagnostics"])


async def track(client, file_id: str, phrase: str, key: str | None = None,
                use_cache: bool = True) -> TrackResult:
    """Track `phrase` through the whole video. `client` is an AsyncOpenAI for Meta.

    `key` identifies the video for the cache (see video_key); it defaults to file_id.
    """
    key = key or file_id
    if use_cache and (cached := load_cached(key, phrase)):
        return cached
    usage_out: dict = {}
    async with client.responses.stream(
        model=SAM_MODEL,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": phrase},
            {"type": "input_video", "file_id": file_id}]}],
        metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
    ) as stream:
        parsed = parse_responses_stream(_events(stream, usage_out), video_segmentation_format())
        final = await parsed.final_result()

    usage = usage_out.get("usage") or {}
    billed = usage.get("video_frames_processed") if isinstance(usage, dict) else None
    result = TrackResult(phrase, _build_points(final.records), billed,
                         [d.message for d in final.diagnostics])
    p = _cache_file(key, phrase)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"phrase": phrase, "points": [asdict(x) for x in result.points],
                             "frames_billed": billed, "diagnostics": result.diagnostics}))
    return result
