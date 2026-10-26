"""Helpers for tests that run on REAL clips (they skip themselves when the clip or its cached tracking is missing).

Videos are never committed: they live in samples/ on the user's machine, and SAM results in data/.
Run the matching script in scripts/ once to create the cache.
"""
import asyncio
from pathlib import Path

import pytest

from app import sam, video
from app.quality import track_objects_with_checks

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


class CacheTracker:
    """Plays back cached SAM results, so the real quality loop runs with no network and no cost."""

    def __init__(self, key, size):
        self.key, self.size = key, size

    async def track_phrase(self, phrase):
        res = sam.load_cached(self.key, phrase)
        if res is None:
            raise LookupError(phrase)
        res.cached = True
        return res

    async def track_boxes(self, prompts, direction="forward"):
        res = sam.load_cached(self.key, f"boxes|{direction}|{sam.boxes_prompt_text(prompts, self.size)}")
        if res is None:
            raise LookupError("box request")
        res.cached = True
        return res


def load(clip, phrase):
    """Tracks of a real clip as the quality loop produces them (from the cache), plus the video info."""
    path = SAMPLES / clip
    if not path.exists():
        pytest.skip(f"{clip} is not on this machine")
    info = video.probe(path)
    size = (info.width, info.height)
    try:
        r = asyncio.run(track_objects_with_checks(CacheTracker(sam.video_key(path), size), [phrase],
                                                  info.frame_count, size, check_camera=False))
    except LookupError:
        pytest.skip(f"no cached SAM tracking for {clip} (run scripts/counting_demo.py once)")
    return r.tracks, info
