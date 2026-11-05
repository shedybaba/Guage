"""Mask statistics (centroid, area, long side) computed from SAM's compressed masks, fast and in parallel.

SAM's one_bit masks are range-coded, and the decoder in meta_sam_parser is pure Python. Its public
`decode_mask_to_raster` decodes AND re-encodes every mask to prove it is canonical, and the parser
also calls it on every mask while reading the stream, so each mask used to be processed four times,
on the server's main thread. On a clip with large masks (about 100,000 pixels each) that froze the
whole server for minutes. Here each mask is decoded once, in worker processes spread over the CPUs,
so the server keeps answering while a clip is being read.

This module imports only numpy and the parser's codec, so worker processes start quickly.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster

try:  # the decoder without the canonical re-encode check (private, so fall back if it ever moves)
    from meta_sam_parser._mask_codec import _decode_raster
except ImportError:  # pragma: no cover
    _decode_raster = None

# (encoding, payload, width, height, left, top, right, bottom)
MaskJob = tuple


def raster(encoding: str, payload: str, width: int, height: int) -> bytes:
    """The mask as one byte (0/1) per pixel, decoded once."""
    if encoding == "one_bit" and _decode_raster is not None and payload.startswith("!"):
        out = _decode_raster(payload, width, height)
        if len(out) != width * height:
            raise ValueError("mask size mismatch")
        return out
    return decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=width, height=height))


def stats(job: MaskJob):
    """Centroid, area and length along the main axis of a box-local mask, in source-video pixels.

    Returns None for an empty or unreadable mask (the caller then uses the box).
    """
    encoding, payload, w, h, left, top, right, bottom = job
    try:
        r = np.frombuffer(raster(encoding, payload, w, h), dtype=np.uint8)
    except Exception:
        return None
    ys, xs = np.nonzero(r.reshape(h, w))
    if len(xs) == 0:
        return None
    sx, sy = (right - left) / w, (bottom - top) / h  # mask may be scaled to the box
    px, py = (xs + 0.5) * sx, (ys + 0.5) * sy
    cx, cy = left + px.mean(), top + py.mean()
    if len(xs) >= 3:
        cov = np.cov(np.stack([px, py]))
        axis = np.linalg.eigh(cov)[1][:, -1]  # direction of greatest spread
        along = px * axis[0] + py * axis[1]
        long_side = float(along.max() - along.min() + max(sx, sy))
    else:
        long_side = float(max(sx, sy))
    return float(cx), float(cy), float(len(xs) * sx * sy), long_side


def stats_many(jobs: list) -> list:
    return [stats(j) for j in jobs]


# Small jobs (tests, image requests) are done in place: starting worker processes costs about a second.
PARALLEL_FROM_PIXELS = 2_000_000
_pool: ProcessPoolExecutor | None = None


def workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:  # "spawn": a clean process, safe next to the server's threads
        _pool = ProcessPoolExecutor(max_workers=workers(), mp_context=multiprocessing.get_context("spawn"))
    return _pool


def reset_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
    _pool = None


def chunks(jobs: list, n: int) -> list:
    """Split into n interleaved parts, so each worker gets a similar share of big and small masks."""
    return [jobs[i::n] for i in range(n) if jobs[i::n]]


def merge(parts: list, n_jobs: int) -> list:
    n = len(parts)
    out = [None] * n_jobs
    for i, part in enumerate(parts):
        out[i::n] = part
    return out
