"""Day 1 gate test: video -> SAM 3.1 -> centroids -> fitted pendulum period.

Usage (from the repo root):
    backend/.venv/bin/python scripts/gate_test.py samples/pendulum.mp4 --length-m 0.50

Prints a plain report. If tracking is clean and the period is plausible, the
approach works. API keys are read from .env and never printed.
"""
import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import sam, signals, video  # noqa: E402
from app.meta_client import async_client, sync_client  # noqa: E402
from app.tools.pendulum import pendulum_period  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--phrase", default="pendulum bob")
    ap.add_argument("--length-m", type=float, default=None, help="string length to the bob's centre")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    info = video.probe(args.clip)
    print(f"Video: {info.width}x{info.height}, {info.frame_count} frames, {info.fps:.2f} fps, "
          f"{info.duration:.1f} s, rotation {info.rotation}, codec {info.codec}")

    key = sam.video_key(args.clip)
    cached = None if args.no_cache else sam.load_cached(key, args.phrase)
    file_id = None
    if cached:
        print("Using cached SAM result (no upload, nothing billed).")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            ready = video.prepare(args.clip, Path(tmp) / "ready.mp4")
            file_id = video.upload(sync_client(), ready)
        print("Uploaded to Meta Files API.")

    result = cached or await sam.track(async_client(), file_id, args.phrase, key=key)
    print(f"SAM phrase '{args.phrase}': {len(result.points)} points, frames billed: {result.frames_billed}")
    for d in result.diagnostics[:5]:
        print("  parser note:", d)
    if not result.points:
        print("NOT FOUND: SAM returned zero matches. Try another phrase.")
        return

    obj = signals.pick_main_object(result.points)
    series = signals.series_for_object(result.points, obj, info.timestamps)
    q = signals.quality_report(series, info.frame_count)
    print(f"Main object id {obj}. Quality: {q}")

    r = pendulum_period(series["t"], series["x"], series["y"], length_m=args.length_m)
    if r["value"] is None:
        print("No period:", r["warnings"])
        return
    print(f"Period T = {r['value']:.3f} s +/- {r['uncertainty']:.3f} s  "
          f"(R2 = {r['fit_quality']['r2']:.4f}, {r['fit_quality']['cycles']:.1f} cycles, axis {r['fit_quality']['axis']})")
    seg = r["fit_quality"]["segment"]
    print(f"Fitted segment: {seg['start_s']:.2f} s to {seg['end_s']:.2f} s "
          f"({'trimmed set-up motion' if seg['trimmed'] else 'whole clip'})")
    comps = r["fit_quality"]["uncertainty_components"]
    print("Uncertainty parts (s): " + ", ".join(
        f"{k} = {v:.5f}" for k, v in comps.items() if k != "residual_autocorrelation" and v is not None)
        + " -> the largest is used")
    g = r["extra"].get("g")
    if g:
        print(f"g = {g['value']:.2f} +/- {g['uncertainty']:.2f} m/s^2 vs 9.81 "
              f"({g['percent_error_vs_standard']:.1f}% error)")
    for w in r["warnings"]:
        print("WARNING:", w)


asyncio.run(main())
