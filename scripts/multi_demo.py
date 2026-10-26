"""Live test of the multi-object checks and retry loop (counting, collisions).

    backend/.venv/bin/python scripts/multi_demo.py samples/newtons_cradle.mp4 \\
        --phrases "steel ball" [--simulate-gap OBJECT START END ...] [--no-camera-check]

--simulate-gap deletes frames of one object from the first track, pretending it was
hidden (repeat the option for several objects). The box re-acquire that follows is a
real SAM request; the filled frames are compared with the ones that were deleted.
This is a test tool only.
"""
import argparse
import asyncio
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

from app import sam, video  # noqa: E402
from app.meta_client import async_client, sync_client  # noqa: E402
from app.quality import track_objects_with_checks  # noqa: E402


class GappedTracker:
    """Wraps a real tracker but hides frames of chosen objects from phrase results."""

    def __init__(self, inner, gaps):
        self.inner, self.gaps = inner, gaps  # gaps: [(object_id, start, end)]
        self.original = {}

    async def track_phrase(self, phrase):
        res = await self.inner.track_phrase(phrase)
        kept = []
        for p in res.points:
            hidden = any(p.object_id == o and s <= p.frame <= e for o, s, e in self.gaps)
            if hidden:
                self.original[(p.object_id, p.frame)] = p
            else:
                kept.append(p)
        return replace(res, points=kept)

    async def track_boxes(self, prompts, direction="forward"):
        return await self.inner.track_boxes(prompts, direction)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--phrases", nargs="+", required=True)
    ap.add_argument("--simulate-gap", nargs=3, action="append", metavar=("OBJECT", "START", "END"), default=[])
    ap.add_argument("--no-camera-check", action="store_true", help="for conveyors, where everything moves together")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    info = video.probe(args.clip)
    key = sam.video_key(args.clip)
    with tempfile.TemporaryDirectory() as tmp:
        ready = video.prepare(args.clip, Path(tmp) / "ready.mp4")
        file_id = video.upload(sync_client(), ready)
    tracker = sam.SamTracker(async_client(), file_id, (info.width, info.height), key=key,
                             use_cache=not args.no_cache)
    gaps = [(o, int(s), int(e)) for o, s, e in args.simulate_gap]
    if gaps:
        tracker = GappedTracker(tracker, gaps)

    r = await track_objects_with_checks(tracker, args.phrases, info.frame_count, (info.width, info.height),
                                        check_camera=not args.no_camera_check)

    print(f"\nStatus: {r.status}   phrase: {r.phrase}   objects: {r.count}   frames billed this run: {r.frames_billed}")
    print("\nHow Gauge checked its work:")
    for e in r.log:
        mark = {True: "[ok]  ", False: "[!]   ", None: "      "}[e.ok]
        print(f"  {mark}{e.message}")
    for a in r.advice:
        print("  advice:", a)
    print("\nObjects (left to right at the first frame they appear):")
    for oid, pts in sorted(r.tracks.items(), key=lambda kv: kv[1][0].cx):
        print(f"  object {oid}: {len(pts)} frames ({pts[0].frame}-{pts[-1].frame}), "
              f"remaining problems: {[i['kind'] for i in r.issues[oid]] or 'none'}")

    if gaps and getattr(tracker, "original", None):
        for o, s, e in gaps:
            got = {p.frame: p for p in r.tracks.get(o, [])}
            errs = [np.hypot(got[f].cx - q.cx, got[f].cy - q.cy)
                    for (oo, f), q in tracker.original.items() if oo == o and f in got]
            total = sum(1 for (oo, _f) in tracker.original if oo == o)
            if errs:
                print(f"\nObject {o}: re-acquired {len(errs)} of {total} hidden frames; distance from SAM's original track "
                      f"mean {np.mean(errs):.1f} px, max {np.max(errs):.1f} px.")
            else:
                print(f"\nObject {o}: none of the {total} hidden frames were re-acquired.")


asyncio.run(main())
