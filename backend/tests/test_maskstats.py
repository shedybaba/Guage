"""Mask statistics: the fast single decode matches the parser's strict decoder, in place and in worker processes."""
import asyncio

import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from meta_sam_parser._mask_codec import _encode_raster
from types import SimpleNamespace as NS

from app import maskstats, sam


def _disc(w, h, cx, cy, r):
    ys, xs = np.mgrid[0:h, 0:w]
    return bytes(((xs - cx) ** 2 + (ys - cy) ** 2 <= r * r).astype(np.uint8).ravel())


def test_fast_decode_matches_the_strict_decoder():
    raster = _disc(40, 30, 18, 14, 9)
    payload = _encode_raster(raster, 40, 30)
    strict = decode_mask_to_raster(SegmentationMask(encoding="one_bit", payload=payload, width=40, height=30))
    assert maskstats.raster("one_bit", payload, 40, 30) == strict == raster


def test_unreadable_mask_gives_none_so_the_box_is_used():
    assert maskstats.stats(("one_bit", "!garbage", 10, 10, 0, 0, 10, 10)) is None


def _records(n):
    recs = []
    for f in range(n):
        raster = _disc(40, 30, 10 + f % 20, 14, 8)
        mask = SegmentationMask(encoding="one_bit", payload=_encode_raster(raster, 40, 30), width=40, height=30)
        frame = NS(frame_index=f)
        recs.append(sam.SegmentationBoxRecord(order=2 * f, object_id="0", frame=frame, left=100, top=50,
                                              right=140, bottom=80, confidence=0.9))
        recs.append(sam.SegmentationMaskRecord(order=2 * f + 1, object_id="0", frame=frame, identity=None, revision=1,
                                               mask=mask, bounds=NS(left=100, top=50, right=140, bottom=80),
                                               confidence=0.9))
    return recs


def test_parallel_points_equal_in_place_points(monkeypatch):
    recs = _records(12)
    expected = sam._build_points(recs)
    monkeypatch.setattr(maskstats, "PARALLEL_FROM_PIXELS", 0)
    monkeypatch.setattr(maskstats, "workers", lambda: 2)
    try:
        got = asyncio.run(sam._build_points_async(recs))
    finally:
        maskstats.reset_pool()
    assert got == expected
    assert abs(expected[3].cx - (100 + 13.5)) < 0.01 and abs(expected[3].cy - 64.5) < 0.01
