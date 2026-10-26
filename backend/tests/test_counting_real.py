"""Regression tests on REAL clips: the cradle (count) and a swinging marble (crossings vs the pendulum tool).

They run only on a machine that has the clips in samples/ and the cached SAM tracking in data/
(neither is committed: videos are never pushed). Run scripts/counting_demo.py once to create the cache.
"""
import asyncio
from pathlib import Path

import pytest

from app import sam, signals, video
from app.quality import track_objects_with_checks
from app.tools.counting import count_objects, line_crossings
from app.tools.pendulum import pendulum_period

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


def test_counts_the_seven_balls_of_the_cradle():
    tracks, info = load("newtons_cradle.mp4", "steel ball")
    r = count_objects(tracks, info.timestamps, (info.width, info.height))
    assert r["value"] == 7 and r["extra"]["max_simultaneous"] == 7


def test_a_handheld_camera_is_flagged_on_the_cradle():
    tracks, info = load("newtons_cradle.mp4", "steel ball")
    r = line_crossings(tracks, info.timestamps, (info.width, info.height))
    assert r["extra"]["camera_motion_suspected"] is True


@pytest.mark.parametrize("clip,phrase", [("pendulum.mp4", "glass marble"), ("pendulum2.mp4", "brass ball")])
def test_crossings_of_the_rest_position_match_the_pendulum_period(clip, phrase):
    tracks, info = load(clip, phrase)
    points = [p for pts in tracks.values() for p in pts]
    oid = signals.pick_main_object(points)
    s = signals.series_for_object(points, oid, info.timestamps)
    m = pendulum_period(s["t"], s["x"], s["y"])
    fq = m["fit_quality"]
    line = {"orientation": "vertical", "position": fq["equilibrium"]}
    x = line_crossings(tracks, info.timestamps, (info.width, info.height), line=line, check_camera=False)
    inside = [e["t"] for e in x["extra"]["crossings"] if fq["segment"]["start_s"] <= e["t"] <= fq["segment"]["end_s"]]
    assert abs(len(inside) - 2 * fq["cycles"]) <= 2  # two crossings per period
    spacing = (inside[-1] - inside[0]) / (len(inside) - 1)
    assert spacing == pytest.approx(m["value"] / 2, rel=0.01)  # one half period apart
