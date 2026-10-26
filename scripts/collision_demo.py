"""Collision test on a real clip: track every object, check the tracks, measure the momentum change.

    backend/.venv/bin/python scripts/collision_demo.py samples/newtons_cradle.mp4 --phrases "steel ball" \\
        [--mass 0.03] [--scale-m-per-px 0.0005]

Without --mass all objects are assumed to have equal masses (stated in the warnings).
"""
import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import sam, signals, video  # noqa: E402
from app.meta_client import async_client, sync_client  # noqa: E402
from app.quality import track_objects_with_checks  # noqa: E402
from app.tools.collision import collision_momentum  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--phrases", nargs="+", required=True)
    ap.add_argument("--mass", type=float, default=None, help="mass of EACH object in kg (all equal)")
    ap.add_argument("--scale-m-per-px", type=float, default=None)
    ap.add_argument("--show-events", action="store_true")
    args = ap.parse_args()

    info = video.probe(args.clip)
    key = sam.video_key(args.clip)
    with tempfile.TemporaryDirectory() as tmp:
        file_id = video.upload(sync_client(), video.prepare(args.clip, Path(tmp) / "ready.mp4"))
    tracker = sam.SamTracker(async_client(), file_id, (info.width, info.height), key=key)
    r = await track_objects_with_checks(tracker, args.phrases, info.frame_count, (info.width, info.height))
    print(f"Tracking: {r.status}, {r.count} objects, {r.frames_billed} frames billed this run")
    for e in r.log:
        print("  ", e.message)

    t, positions, sizes = signals.positions_from_tracks(r.tracks, info.timestamps)
    masses = {k: args.mass for k in positions} if args.mass else None
    m = collision_momentum(t, positions, masses=masses, scale_m_per_px=args.scale_m_per_px, sizes=sizes)

    print()
    if m["value"] is None:
        print("No result:", m["warnings"][-1])
    else:
        x = m["extra"]
        print(f"Momentum change across impacts: {m['value']:+.1f}% +/- {m['uncertainty']:.1f}%  "
              f"({x['n_valid']} valid of {x['n_events']} impacts found)")
        spread = x["spread_between_impacts"]
        print(f"  median {x['median_percent_change']:+.1f}%, spread between impacts "
              f"{'n/a (one impact)' if spread is None else f'{spread:.1f}%'}, "
              f"kinetic energy after/before = {x['kinetic_energy_ratio']:.2f}")
        print(f"  typical momentum {x['momentum_before']:.3g} -> {x['momentum_after']:.3g} ({x['momentum_units']})")
    for w in m["warnings"]:
        print("WARNING:", w)
    if args.show_events:
        for e in m["extra"].get("events", []):
            pc = f"{e['percent_change']:+6.1f}%" if "percent_change" in e else "   n/a "
            print(f"  frame {e['frame']:3d} t={e['t']:5.2f}s {'valid  ' if e['valid'] else 'REJECTED'} {pc}  "
                  f"{'; '.join(e['reasons'])[:110]}")


asyncio.run(main())
