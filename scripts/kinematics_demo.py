"""Speed, acceleration and free fall on a real clip.

    backend/.venv/bin/python scripts/kinematics_demo.py samples/drop.mp4 --phrases "ball" --free-fall --scale-m-per-px 0.002
    backend/.venv/bin/python scripts/kinematics_demo.py samples/ramp.mp4 --phrases "ball" --ramp-angle 20 --scale-m-per-px 0.002
    backend/.venv/bin/python scripts/kinematics_demo.py samples/pendulum.mp4 --phrases "glass marble" --check-pendulum

--check-pendulum compares the peak speed with amplitude x omega from the pendulum tool: a cross-check
between two independent routes (local derivative fits vs the fitted period).
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
from app.tools.kinematics import free_fall, velocity_acceleration  # noqa: E402
from app.tools.pendulum import pendulum_period  # noqa: E402


def show(label, d, fmt="{:.4g}"):
    print(f"  {label}: " + fmt.format(d["value"]) + " +/- " + fmt.format(d["uncertainty"]) + f" {d['units']}"
          + (f"  ({d['basis']})" if d.get("basis") else ""))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--phrases", nargs="+", required=True)
    ap.add_argument("--scale-m-per-px", type=float)
    ap.add_argument("--scale-uncertainty", type=float, help="relative, e.g. 0.01 for 1%%")
    ap.add_argument("--free-fall", action="store_true")
    ap.add_argument("--ramp-angle", type=float)
    ap.add_argument("--check-pendulum", action="store_true")
    ap.add_argument("--window-s", type=float, default=0.1)
    args = ap.parse_args()

    info = video.probe(args.clip)
    size = (info.width, info.height)
    key = sam.video_key(args.clip)
    with tempfile.TemporaryDirectory() as tmp:
        file_id = video.upload(sync_client(), video.prepare(args.clip, Path(tmp) / "ready.mp4"))
    tracker = sam.SamTracker(async_client(), file_id, size, key=key)
    r = await track_objects_with_checks(tracker, args.phrases, info.frame_count, size, check_camera=False)
    print(f"Tracking: {r.status}, {r.count} object(s), {r.frames_billed} frames billed this run, {info.fps:.0f} fps")
    points = [p for pts in r.tracks.values() for p in pts]
    s = signals.series_for_object(points, signals.pick_main_object(points), info.timestamps)
    kw = dict(scale_m_per_px=args.scale_m_per_px, scale_uncertainty_fraction=args.scale_uncertainty)

    if args.free_fall:
        m = free_fall(s["t"], s["x"], s["y"], **kw)
        if m["value"] is None:
            print("Free fall: no result:", m["warnings"][-1])
        else:
            fq, x = m["fit_quality"], m["extra"]
            print(f"\nFree fall: g = {m['value']:.4g} +/- {m['uncertainty']:.2g} {m['units']}   "
                  f"({fq['n_frames']} frames, {fq['segment']['start_s']:.2f}-{fq['segment']['end_s']:.2f} s, tilt {x['tilt_deg']:+.1f} deg)")
            print(f"  statistical +/- {fq['statistical_uncertainty']:.2g}, boundary +/- {fq['boundary_uncertainty']:.2g}, "
                  f"scale uncertainty {100 * fq['scale_uncertainty_fraction']:.1f}%")
            if "consistent_with_standard_g" in x:
                print(f"  vs 9.81: {x['percent_difference_from_standard']:+.1f}%  consistent: {x['consistent_with_standard_g']}")
            sc = x["scale_if_g_is_9.81"]
            print(f"  implied scale if this is free fall: {sc['value']:.5f} +/- {sc['uncertainty']:.5f} m/px")
            print(f"  drop {x['drop_distance']:.3g}, impact speed {x['impact_speed']:.3g}")
            for w in m["warnings"]:
                print("  WARNING:", w)

    v = velocity_acceleration(s["t"], s["x"], s["y"], ramp_angle_deg=args.ramp_angle, window_s=args.window_s, **kw)
    if v["value"] is None:
        print("Velocity/acceleration: no result:", v["warnings"][-1])
        return
    x = v["extra"]
    print(f"\nVelocity and acceleration (window {v['fit_quality']['window_s']:.2f} s, moving {x['moving_from_s']:.2f}-{x['moving_to_s']:.2f} s):")
    show("peak speed", {**x["peak_speed"], "basis": f"at {x['peak_speed']['time_s']:.2f} s"})
    show("average speed", x["average_speed"])
    show("acceleration along the path", x["acceleration"])
    if "peak_acceleration" in x:
        show("peak acceleration", {**x["peak_acceleration"], "basis": f"at {x['peak_acceleration']['time_s']:.2f} s"})
    if "ramp" in x:
        rp = x["ramp"]
        print(f"  ramp {rp['angle_deg']:.1f} deg: expected {{{', '.join(f'{k}: {val:.3g}' for k, val in rp['expected'].items())}}}")
        print(f"  closest: {rp['closest']}; a / (g sin theta) = {rp['ratio_to_g_sin_theta']['value']:.3f} +/- {rp['ratio_to_g_sin_theta']['uncertainty']:.3f}")
    for w in v["warnings"]:
        print("  WARNING:", w)

    if args.check_pendulum:
        p = pendulum_period(s["t"], s["x"], s["y"])
        fq = p["fit_quality"]
        seg = (s["t"] - s["t"][0] >= fq["segment"]["start_s"]) & (s["t"] - s["t"][0] <= fq["segment"]["end_s"])
        axis = s[fq["axis"]][seg]
        amp = (axis.max() - axis.min()) / 2
        omega = 2 * np.pi / p["value"]
        v2 = velocity_acceleration(s["t"][seg], s["x"][seg], s["y"][seg], quantity="speed", window_s=args.window_s)
        print(f"\nPendulum cross-check inside the clean swing ({fq['cycles']:.1f} cycles, T = {p['value']:.4f} s):")
        print(f"  peak speed from local fits        : {v2['value']:.1f} +/- {v2['uncertainty']:.1f} px/s")
        print(f"  amplitude x omega (pendulum tool): {amp * omega:.1f} px/s  (amplitude {amp:.1f} px, omega {omega:.4f} rad/s)")
        a_pk = velocity_acceleration(s["t"][seg], s["x"][seg], s["y"][seg], window_s=args.window_s)["extra"]["peak_acceleration"]
        print(f"  peak acceleration from local fits : {a_pk['value']:.0f} +/- {a_pk['uncertainty']:.0f} px/s^2   vs amplitude x omega^2 = {amp * omega ** 2:.0f}")


asyncio.run(main())
