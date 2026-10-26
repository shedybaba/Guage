"""Live test of the quality checks and retry loop on a real clip.

    backend/.venv/bin/python scripts/quality_demo.py samples/pendulum.mp4 \\
        --phrases "pendulum bob" "glass marble" [--simulate-gap 150 199]

--simulate-gap deletes those frames from the first track, pretending the object
was hidden, so you can watch the box re-acquire step fill them in. It is a test
tool only: the filled frames come from a real SAM box-prompt request, and are
compared with the frames that were deleted.
"""
import argparse
import asyncio
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

from app import sam, signals, video  # noqa: E402
from app.tools.pendulum import pendulum_period  # noqa: E402
from app.meta_client import async_client, sync_client  # noqa: E402
from app.quality import track_with_checks  # noqa: E402


class GappedTracker:
    """Wraps a real tracker but hides some frames from phrase results (simulated occlusion)."""

    def __init__(self, inner, start, end):
        self.inner, self.start, self.end = inner, start, end
        self.original = {}

    async def track_phrase(self, phrase):
        res = await self.inner.track_phrase(phrase)
        for p in res.points:
            if self.start <= p.frame <= self.end:
                self.original[p.frame] = p
        kept = [p for p in res.points if not self.start <= p.frame <= self.end]
        return replace(res, points=kept)

    async def track_box(self, frame, box, direction="forward"):
        return await self.inner.track_box(frame, box, direction)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--phrases", nargs="+", required=True)
    ap.add_argument("--simulate-gap", nargs=2, type=int, metavar=("START", "END"))
    ap.add_argument("--fit-pendulum", action="store_true", help="after tracking, fit the swing period")
    ap.add_argument("--length-m", type=float, default=None, help="string length to the bob's centre")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    info = video.probe(args.clip)
    key = sam.video_key(args.clip)
    with tempfile.TemporaryDirectory() as tmp:
        ready = video.prepare(args.clip, Path(tmp) / "ready.mp4")
        file_id = video.upload(sync_client(), ready)
    tracker = sam.SamTracker(async_client(), file_id, (info.width, info.height), key=key,
                             use_cache=not args.no_cache)
    if args.simulate_gap:
        tracker = GappedTracker(tracker, *args.simulate_gap)

    r = await track_with_checks(tracker, args.phrases, info.frame_count)

    print(f"\nStatus: {r.status}   phrase: {r.phrase}   frames tracked: {len(r.points)}/{r.total_frames}"
          f"   frames billed this run: {r.frames_billed}")
    print("\nHow Gauge checked its work:")
    for e in r.log:
        mark = {True: "[ok]  ", False: "[!]   ", None: "      "}[e.ok]
        print(f"  {mark}{e.message}")
    for a in r.advice:
        print("  advice:", a)

    if args.fit_pendulum and r.points:
        series = signals.series_for_object(r.points, r.object_id, info.timestamps)
        m = pendulum_period(series["t"], series["x"], series["y"], length_m=args.length_m)
        if m["value"] is None:
            print("\nNo period:", m["warnings"])
        else:
            fq = m["fit_quality"]
            print(f"\nPeriod T = {m['value']:.4f} +/- {m['uncertainty']:.4f} s  (R2 {fq['r2']:.4f}, "
                  f"{fq['cycles']:.1f} cycles, axis {fq['axis']}, segment {fq['segment']['start_s']:.1f}-{fq['segment']['end_s']:.1f} s)")
            if "g" in m["extra"]:
                g = m["extra"]["g"]
                print(f"g = {g['value']:.2f} +/- {g['uncertainty']:.2f} m/s^2 ({g['percent_error_vs_standard']:.1f}% from 9.81)")
            for w in m["warnings"]:
                print("WARNING:", w)

    if args.simulate_gap and tracker.original:
        got = {p.frame: p for p in r.points}
        errs = [np.hypot(got[f].cx - o.cx, got[f].cy - o.cy)
                for f, o in tracker.original.items() if f in got]
        if errs:
            print(f"\nRe-acquired {len(errs)} of {len(tracker.original)} hidden frames. "
                  f"Distance from SAM's original track: mean {np.mean(errs):.1f} px, max {np.max(errs):.1f} px "
                  f"(the object is about {np.median([p.x2 - p.x1 for p in r.points]):.0f} px wide).")
        else:
            print("\nNone of the hidden frames were re-acquired.")


asyncio.run(main())
