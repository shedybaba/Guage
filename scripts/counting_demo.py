"""Counting on a real clip: how many objects, and how many cross a line.

    backend/.venv/bin/python scripts/counting_demo.py samples/newtons_cradle.mp4 --phrases "steel ball"
    backend/.venv/bin/python scripts/counting_demo.py samples/pendulum.mp4 --phrases "glass marble" --at-rest-position

--line-x / --line-y put the line at a pixel position (default: the vertical centre line).
--at-rest-position fits the pendulum first and puts the line at its rest position, then checks the
crossings against the period (2 per period, spaced T/2 apart): a cross-check between two tools.
"""
import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

from app import sam, signals, video  # noqa: E402
from app.meta_client import async_client, sync_client  # noqa: E402
from app.quality import track_objects_with_checks  # noqa: E402
from app.tools.counting import count_objects, line_crossings  # noqa: E402
from app.tools.pendulum import pendulum_period  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--phrases", nargs="+", required=True)
    ap.add_argument("--line-x", type=float)
    ap.add_argument("--line-y", type=float)
    ap.add_argument("--at-rest-position", action="store_true")
    ap.add_argument("--no-camera-check", action="store_true", help="for conveyors")
    args = ap.parse_args()

    info = video.probe(args.clip)
    size = (info.width, info.height)
    key = sam.video_key(args.clip)
    with tempfile.TemporaryDirectory() as tmp:
        file_id = video.upload(sync_client(), video.prepare(args.clip, Path(tmp) / "ready.mp4"))
    tracker = sam.SamTracker(async_client(), file_id, size, key=key)
    r = await track_objects_with_checks(tracker, args.phrases, info.frame_count, size,
                                        check_camera=not args.no_camera_check)
    print(f"Tracking: {r.status}, {r.count} objects, {r.frames_billed} frames billed this run")

    c = count_objects(r.tracks, info.timestamps, size)
    print(f"\nObjects: {c['value']} (range {c['extra']['range'][0]} to {c['extra']['range'][1]}), "
          f"at most {c['extra']['max_simultaneous']} in view together; ignored: {c['extra']['ignored_short_tracks']}")
    for w in c["warnings"]:
        print("  note:", w)

    line, seg = None, None
    if args.at_rest_position:
        oid = signals.pick_main_object([p for pts in r.tracks.values() for p in pts])
        s = signals.series_for_object([p for pts in r.tracks.values() for p in pts], oid, info.timestamps)
        m = pendulum_period(s["t"], s["x"], s["y"])
        fq = m["fit_quality"]
        line = {"orientation": "vertical" if fq["axis"] == "x" else "horizontal", "position": fq["equilibrium"]}
        seg = (fq["segment"]["start_s"], fq["segment"]["end_s"])
        print(f"\nPendulum tool: T = {m['value']:.4f} +/- {m['uncertainty']:.4f} s, rest position {fq['equilibrium']:.1f} px, "
              f"clean swing {seg[0]:.1f} to {seg[1]:.1f} s ({fq['cycles']:.1f} cycles)")
    elif args.line_x is not None:
        line = {"orientation": "vertical", "position": args.line_x}
    elif args.line_y is not None:
        line = {"orientation": "horizontal", "position": args.line_y}

    x = line_crossings(r.tracks, info.timestamps, size, line=line)
    print(f"\nLine: {x['fit_quality']['line']}")
    print(f"Crossings: {x['value']}  by direction {x['extra']['by_direction']}, net {x['extra']['net']}, "
          f"objects that crossed: {x['extra']['objects_crossed']}")
    rate = x["extra"]["rate_per_minute"]
    print(f"Rate: {rate['value']:.1f} +/- {rate['uncertainty']:.1f} per minute ({rate['basis']})")
    for w in x["warnings"]:
        print("  note:", w)

    if seg:
        ev = [e for e in x["extra"]["crossings"] if seg[0] <= e["t"] <= seg[1]]
        times = np.array([e["t"] for e in ev])
        if len(times) >= 3:
            half = np.diff(times)
            print(f"\nCross-check inside the clean swing: {len(ev)} crossings expected about {2 * fq['cycles']:.1f} "
                  f"(2 per period); spacing {half.mean():.4f} s, expected T/2 = {m['value'] / 2:.4f} s; "
                  f"rate {60 / half.mean():.1f}/min, expected {120 / m['value']:.1f}/min")


asyncio.run(main())
