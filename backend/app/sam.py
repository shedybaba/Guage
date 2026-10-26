"""SAM 3.1 tracking: one streamed request per noun phrase, parsed into per-frame points.

A phrase is a short noun phrase naming ONE object ("pendulum bob"). Zero matches
is a valid result (empty list), not an error. Results are cached per
(file_id, phrase) on disk so follow-up questions and re-runs are not billed again.
"""
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from meta_sam_parser import (FramePrompt, PromptBox, PromptObject, SegmentationBoxRecord,
                             SegmentationMaskRecord, build_object_prompt, decode_mask_to_raster,
                             parse_responses_stream, video_segmentation_format)

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
    long_side: float | None = None  # length of the mask along its main axis in pixels (a ruler's length even if tilted)


@dataclass
class TrackResult:
    phrase: str
    points: list  # list[TrackPoint]
    frames_billed: int | None
    diagnostics: list
    cached: bool = False  # True when served from the local cache (nothing was billed)


def _mask_stats(rec: SegmentationMaskRecord):
    """Centroid, area and length along the main axis of a box-local mask, in source-video pixels."""
    raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8)
    w, h = rec.mask.width, rec.mask.height
    ys, xs = np.nonzero(raster.reshape(h, w))
    if len(xs) == 0:
        return None
    b = rec.bounds
    sx, sy = (b.right - b.left) / w, (b.bottom - b.top) / h  # mask may be scaled to the box
    px, py = (xs + 0.5) * sx, (ys + 0.5) * sy
    cx, cy = b.left + px.mean(), b.top + py.mean()
    if len(xs) >= 3:
        cov = np.cov(np.stack([px, py]))
        axis = np.linalg.eigh(cov)[1][:, -1]  # direction of greatest spread
        along = px * axis[0] + py * axis[1]
        long_side = float(along.max() - along.min() + max(sx, sy))
    else:
        long_side = float(max(sx, sy))
    return cx, cy, float(len(xs) * sx * sy), long_side


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
        cx, cy, area, long_side = stats or ((b.left + b.right) / 2, (b.top + b.bottom) / 2,
                                            (b.right - b.left) * (b.bottom - b.top),
                                            max(b.right - b.left, b.bottom - b.top))
        points.append(TrackPoint(key[0], key[1], b.left, b.top, b.right, b.bottom,
                                 b.confidence, cx, cy, area, long_side))
    return points


async def _events(stream, usage_out: dict, progress=None):
    """Pass events to the parser while remembering billed frames from the final event.

    Each frame arrives as one text delta, so counting them gives live progress (progress(frames_so_far)).
    """
    done = 0
    async for event in stream:
        if event.type == "response.output_text.delta":
            done += 1
            if progress is not None and done % 10 == 0:
                progress(done)
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


async def _run(client, file_id: str, text: str, metadata: dict, key: str, label: str,
               use_cache: bool = True, progress=None) -> TrackResult:
    """One streamed SAM request (phrase or box prompt) parsed into points, cached by `label`."""
    if use_cache and (cached := load_cached(key, label)):
        cached.cached = True
        return cached
    if callable(file_id):  # uploaded only now that a live request is really needed
        file_id = await file_id()
    usage_out: dict = {}
    async with client.responses.stream(
        model=SAM_MODEL,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_video", "file_id": file_id}]}],
        metadata={"include_confidence": "true", "mask_encoding": "one_bit", **metadata},
    ) as stream:
        parsed = parse_responses_stream(_events(stream, usage_out, progress), video_segmentation_format())
        final = await parsed.final_result()

    usage = usage_out.get("usage") or {}
    billed = usage.get("video_frames_processed") if isinstance(usage, dict) else None
    result = TrackResult(label, _build_points(final.records), billed,
                         [d.message for d in final.diagnostics])
    p = _cache_file(key, label)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"phrase": label, "points": [asdict(x) for x in result.points],
                             "frames_billed": billed, "diagnostics": result.diagnostics}))
    if final.raw_output:  # SAM's own text, with the masks, for the overlay player
        with gzip.open(p.with_suffix(".raw.gz"), "wt") as f:
            f.write(final.raw_output)
    return result


def load_raw(key: str, label: str):
    """SAM's raw output text for a cached request (it carries the masks), or None."""
    p = _cache_file(key, label).with_suffix(".raw.gz")
    if not p.exists():
        return None
    with gzip.open(p, "rt") as f:
        return f.read()


async def track(client, file_id: str, phrase: str, key: str | None = None,
                use_cache: bool = True, progress=None) -> TrackResult:
    """Track `phrase` through the whole video. `client` is an AsyncOpenAI for Meta.

    `key` identifies the video for the cache (see video_key); it defaults to file_id.
    """
    return await _run(client, file_id, phrase, {}, key or file_id, phrase, use_cache, progress)


def _prompt_object(object_id: int, frame: int, box, size) -> PromptObject:
    """One object's box on one frame, clipped to the video's displayed size (width, height)."""
    w, h = size
    left, top = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    right, bottom = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    return PromptObject(id=object_id, prompts=[FramePrompt(frame=frame, box=PromptBox(left, top, right, bottom))])


def box_prompt_text(frame: int, box, size) -> str:
    """Prompt text for one box on one frame."""
    return build_object_prompt(size=tuple(size), objects=[_prompt_object(0, frame, box, size)])


def boxes_prompt_text(prompts, size) -> str:
    """Prompt text for several objects at once: prompts is [(frame, box), ...]; object ids are 0, 1, ..."""
    return build_object_prompt(size=tuple(size), objects=[
        _prompt_object(i, frame, box, size) for i, (frame, box) in enumerate(prompts)])


async def track_box(client, file_id: str, frame: int, box, size, direction: str = "forward",
                    key: str | None = None, use_cache: bool = True, progress=None) -> TrackResult:
    """Re-acquire an object from a box on `frame`, tracking `direction` ("forward" or "backward")."""
    text = box_prompt_text(frame, box, size)
    label = f"box|{direction}|{text}"
    return await _run(client, file_id, text, {"propagation_direction": direction},
                      key or file_id, label, use_cache, progress)


async def track_boxes(client, file_id: str, prompts, size, direction: str = "forward",
                      key: str | None = None, use_cache: bool = True, progress=None) -> TrackResult:
    """Re-acquire SEVERAL objects in one request (up to 16). Result ids are "0", "1", ... in prompt order."""
    text = boxes_prompt_text(prompts, size)
    label = f"boxes|{direction}|{text}"
    return await _run(client, file_id, text, {"propagation_direction": direction},
                      key or file_id, label, use_cache, progress)


class SamTracker:
    """The tracker the retry loop uses: phrase and box requests on one uploaded video.

    on_progress(what, frames_done) is called every 10 frames of a live request (not for cached results).
    """

    def __init__(self, client, file_id: str, size, key: str | None = None, use_cache: bool = True, on_progress=None):
        self.client, self.file_id, self.size, self.key, self.use_cache = client, file_id, size, key, use_cache
        self.on_progress = on_progress

    def _progress(self, what):
        return (lambda n: self.on_progress(what, n)) if self.on_progress else None

    async def track_phrase(self, phrase: str) -> TrackResult:
        return await track(self.client, self.file_id, phrase, self.key, self.use_cache, self._progress(f"'{phrase}'"))

    async def track_box(self, frame: int, box, direction: str = "forward") -> TrackResult:
        return await track_box(self.client, self.file_id, frame, box, self.size, direction,
                               self.key, self.use_cache, self._progress("a lost object"))

    async def track_boxes(self, prompts, direction: str = "forward") -> TrackResult:
        return await track_boxes(self.client, self.file_id, prompts, self.size, direction,
                                 self.key, self.use_cache, self._progress("lost objects"))
