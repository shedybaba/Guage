"""SAM 3.1 tracking: one streamed request per noun phrase, parsed into per-frame points.

A phrase is a short noun phrase naming ONE object ("pendulum bob"). Zero matches
is a valid result (empty list), not an error. Results are cached per
(file_id, phrase) on disk so follow-up questions and re-runs are not billed again.
"""
import asyncio
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass

import meta_sam_parser._segmentation as _parser_segmentation
import numpy as np
import openai
from meta_sam_parser import (FramePrompt, PromptBox, PromptObject, SegmentationBoxRecord,
                             SegmentationMaskRecord, build_object_prompt,
                             parse_responses_stream, video_segmentation_format)

from . import maskstats
from .config import DATA_DIR, SAM_MODEL


def _light_mask_check(mask):
    """Stand-in for the parser's own check of every mask while the stream is read.

    The parser fully decodes and re-encodes each mask just to validate it, in pure Python, on the
    server's main thread: on a clip with large masks that froze the whole server for minutes. Only
    the cheap structural checks stay here; the one real decode happens later in `maskstats`, in
    worker processes, and a mask that cannot be read there falls back to its box.
    """
    if not isinstance(mask.payload, str) or not mask.payload[:1] in ("!", "~"):
        raise _parser_segmentation.InvalidSegmentationMaskError("Mask payload must start with ! or ~.")
    return b""


_parser_segmentation.decode_mask_to_raster = _light_mask_check


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


def _mask_job(rec: SegmentationMaskRecord):
    b, m = rec.bounds, rec.mask
    return (m.encoding, m.payload, m.width, m.height, b.left, b.top, b.right, b.bottom)


def _mask_stats(rec: SegmentationMaskRecord):
    """Centroid, area and length along the main axis of a box-local mask, in source-video pixels."""
    return maskstats.stats(_mask_job(rec))


def _pair_records(records):
    boxes, masks = {}, {}
    for r in records:
        if r.frame is None:
            continue
        key = (r.frame.frame_index, r.object_id)
        if isinstance(r, SegmentationBoxRecord):
            boxes[key] = r
        elif isinstance(r, SegmentationMaskRecord):
            masks[key] = r
    return boxes, masks


def _points(boxes, masks, stats_by_key) -> list:
    points = []
    for key, b in sorted(boxes.items()):
        stats = stats_by_key.get(key)
        cx, cy, area, long_side = stats or ((b.left + b.right) / 2, (b.top + b.bottom) / 2,
                                            (b.right - b.left) * (b.bottom - b.top),
                                            max(b.right - b.left, b.bottom - b.top))
        points.append(TrackPoint(key[0], key[1], b.left, b.top, b.right, b.bottom,
                                 b.confidence, cx, cy, area, long_side))
    return points


def _build_points(records) -> list:
    """Per-frame points, decoding every mask in this process (small results and tests)."""
    boxes, masks = _pair_records(records)
    return _points(boxes, masks, {k: maskstats.stats(_mask_job(m)) for k, m in masks.items() if k in boxes})


async def _build_points_async(records) -> list:
    """Per-frame points without blocking the server: big results are decoded in worker processes."""
    boxes, masks = _pair_records(records)
    keys = [k for k in masks if k in boxes]
    jobs = [_mask_job(masks[k]) for k in keys]
    pixels = sum(j[2] * j[3] for j in jobs)
    if pixels < maskstats.PARALLEL_FROM_PIXELS:
        results = await asyncio.to_thread(maskstats.stats_many, jobs)
    else:
        loop = asyncio.get_running_loop()
        parts = maskstats.chunks(jobs, maskstats.workers())
        try:
            done = await asyncio.gather(*[loop.run_in_executor(maskstats.pool(), maskstats.stats_many, p)
                                          for p in parts])
            results = maskstats.merge(done, len(jobs))
        except Exception:  # worker processes unavailable: same work in a thread (and a fresh pool next time)
            maskstats.reset_pool()
            results = await asyncio.to_thread(maskstats.stats_many, jobs)
    return _points(boxes, masks, dict(zip(keys, results)))


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
    provider = file_id if callable(file_id) else None
    if provider:  # uploaded only now that a live request is really needed
        file_id = await provider()
    try:
        final, usage_out = await _stream(client, file_id, text, metadata, progress)
    except openai.NotFoundError as e:
        if provider is None or getattr(e, "code", None) != "file_not_found":
            raise
        # Meta no longer has the uploaded video (it expires after 30 days, or was deleted): upload it again, once
        final, usage_out = await _stream(client, await provider(stale=file_id), text, metadata, progress)

    usage = usage_out.get("usage") or {}
    billed = usage.get("video_frames_processed") if isinstance(usage, dict) else None
    result = TrackResult(label, await _build_points_async(final.records), billed,
                         [d.message for d in final.diagnostics])
    p = _cache_file(key, label)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"phrase": label, "points": [asdict(x) for x in result.points],
                             "frames_billed": billed, "diagnostics": result.diagnostics}))
    if final.raw_output:  # SAM's own text, with the masks, for the overlay player
        with gzip.open(p.with_suffix(".raw.gz"), "wt") as f:
            f.write(final.raw_output)
    return result


async def _stream(client, file_id: str, text: str, metadata: dict, progress):
    """The streamed request itself: returns the parsed result and the usage block."""
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
    return final, usage_out


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
