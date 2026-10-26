"""Spring-and-mass analysis on a real clip: frequency, damping ratio and (given a mass) the spring constant.

    backend/.venv/bin/python scripts/spring_demo.py samples/spring.mp4 --phrases "metal spring" --mass 0.2 \\
        [--spring-mass 0.05] [--scale-m-per-px 0.001]

Any damped oscillation works, so a swinging pendulum is a fair test of the damping measurement
(no spring constant is meaningful there, so leave --mass out).
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
from app.tools.pendulum import pendulum_period  # noqa: E402
from app.tools.spring import oscillation_frequency  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--phrases", nargs="+", required=True)
    ap.add_argument("--mass", type=float)
    ap.add_argument("--mass-uncertainty", type=float)
    ap.add_argument("--spring-mass", type=float)
    ap.add_argument("--scale-m-per-px", type=float)
    ap.add_argument("--compare-pendulum", action="store_true", help="also run the pendulum tool and compare periods")
    args = ap.parse_args()

    info = video.probe(args.clip)
    size = (info.width, info.height)
    key = sam.video_key(args.clip)
    with tempfile.TemporaryDirectory() as tmp:
        file_id = video.upload(sync_client(), video.prepare(args.clip, Path(tmp) / "ready.mp4"))
    tracker = sam.SamTracker(async_client(), file_id, size, key=key)
    r = await track_objects_with_checks(tracker, args.phrases, info.frame_count, size, check_camera=False)
    print(f"Tracking: {r.status}, {r.count} object(s), {r.frames_billed} frames billed this run")
    for e in r.log:
        if not e.ok:
            print("  ", e.message)

    points = [p for pts in r.tracks.values() for p in pts]
    oid = signals.pick_main_object(points)
    s = signals.series_for_object(points, oid, info.timestamps)
    m = oscillation_frequency(s["t"], s["x"], s["y"], mass_kg=args.mass, mass_uncertainty_kg=args.mass_uncertainty,
                              spring_mass_kg=args.spring_mass, scale_m_per_px=args.scale_m_per_px)
    if m["value"] is None:
        print("No result:", m["warnings"])
        return
    fq, x = m["fit_quality"], m["extra"]
    print(f"\nFrequency (damped): {m['value']:.4f} +/- {m['uncertainty']:.4f} Hz  (period {1 / m['value']:.4f} s, "
          f"{fq['cycles']:.1f} cycles, {fq['frames_per_cycle']:.0f} frames per cycle, axis {fq['axis']}, "
          f"segment {fq['segment']['start_s']:.1f}-{fq['segment']['end_s']:.1f} s)")
    nf = x["natural_frequency"]
    print(f"Natural frequency:  {nf['value']:.4f} +/- {nf['uncertainty']:.4f} Hz")
    z = x["damping_ratio"]
    if z["measurable"]:
        print(f"Damping ratio zeta: {z['value']:.5f} +/- {z['uncertainty']:.5f}   decay rate b = {z['decay_rate']['value']:.4f} "
              f"+/- {z['decay_rate']['uncertainty']:.4f} /s   amplitude time constant {z['time_constant']['value']:.1f} s")
    else:
        print(f"Damping: not measurable; zeta below {z['upper_bound']:.4f}")
    env = x["damping_envelope"]
    if env["decay_rate"] is not None:
        print(f"Peak-height check:  {env['n_peaks']} peaks, decay {env['decay_rate']:.4f} +/- {env['decay_rate_uncertainty']:.4f} /s, "
              f"shape: {env['type']}")
    if "spring_constant" in x:
        k = x["spring_constant"]
        print(f"Spring constant k = {k['value']:.3f} +/- {k['uncertainty']:.3f} N/m (effective mass {k['effective_mass_kg']:.4f} kg)")
    if "amplitude" in x:
        print(f"Amplitude {x['amplitude']['value']:.4f} m, top speed {x['max_speed']['value']:.3f} m/s")
    for w in m["warnings"]:
        print("WARNING:", w)
    if args.compare_pendulum:
        p = pendulum_period(s["t"], s["x"], s["y"])
        print(f"\nPendulum tool: T = {p['value']:.4f} +/- {p['uncertainty']:.4f} s  vs spring tool 1/f = {1 / m['value']:.4f} s")


asyncio.run(main())
