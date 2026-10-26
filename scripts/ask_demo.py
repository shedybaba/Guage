"""Ask Gauge a question about a real clip: planner -> tracking -> tool -> facts -> explanation (live Muse Spark).

    backend/.venv/bin/python scripts/ask_demo.py samples/pendulum.mp4 "What is the swing period?"
    backend/.venv/bin/python scripts/ask_demo.py samples/pendulum.mp4 "What value of g does that give? The string is half a metre."
    backend/.venv/bin/python scripts/ask_demo.py samples/newtons_cradle.mp4 "Is momentum conserved?" --plan-only

--plan-only stops after the plan. --set string_length_m=0.5 hands a value over like the API will.
SAM results come from the cache when the same objects were tracked before (nothing billed); Muse Spark calls cost
a fraction of a cent. This script is also a preview of the job runner that comes next.
"""
import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import sam, signals, video  # noqa: E402
from app.explain import build_facts, explain  # noqa: E402
from app.llm import SparkLLM  # noqa: E402
from app.meta_client import async_client, sync_client  # noqa: E402
from app.planner import make_plan  # noqa: E402
from app.quality import track_objects_with_checks  # noqa: E402
from app.tools.collision import collision_momentum  # noqa: E402
from app.tools.counting import count_objects, line_crossings  # noqa: E402
from app.tools.kinematics import free_fall, velocity_acceleration  # noqa: E402
from app.tools.pendulum import pendulum_period  # noqa: E402
from app.tools.spring import oscillation_frequency  # noqa: E402


def run_tool(plan, checked, info, scale):
    """Run the planned tool on the tracked objects. (Scale from a reference object is not built yet: pass --scale.)"""
    v = {k: e.value for k, e in plan.values.items()}
    points = [p for pts in checked.tracks.values() for p in pts]
    main = signals.pick_main_object(points)
    s = signals.series_for_object(points, main, info.timestamps)
    size = (info.width, info.height)
    if plan.tool == "pendulum_period":
        return pendulum_period(s["t"], s["x"], s["y"], length_m=v.get("string_length"))
    if plan.tool == "oscillation_frequency":
        return oscillation_frequency(s["t"], s["x"], s["y"], mass_kg=v.get("mass"), spring_mass_kg=v.get("spring_mass"),
                                     scale_m_per_px=scale)
    if plan.tool == "free_fall":
        return free_fall(s["t"], s["x"], s["y"], scale_m_per_px=scale)
    if plan.tool == "velocity_acceleration":
        return velocity_acceleration(s["t"], s["x"], s["y"], scale_m_per_px=scale, quantity=plan.quantity,
                                     at_time=v.get("time"), ramp_angle_deg=v.get("ramp_angle"))
    if plan.tool == "collision_momentum":
        t, pos, sizes = signals.positions_from_tracks(checked.tracks, info.timestamps)
        masses = {k: v["mass"] for k in pos} if "mass" in v else None
        return collision_momentum(t, pos, masses=masses, scale_m_per_px=scale, sizes=sizes)
    if plan.tool == "count_objects":
        return count_objects(checked.tracks, info.timestamps, size)
    if plan.tool == "line_crossings":
        line = {"orientation": plan.line["orientation"], "position": plan.line["position_px"]}
        return line_crossings(checked.tracks, info.timestamps, size, line=line, check_camera=plan.check_camera)
    raise SystemExit(f"no runner for {plan.tool}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("question")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="e.g. string_length_m=0.5")
    ap.add_argument("--scale", type=float, help="metres per pixel (until scale-from-a-reference is built)")
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()

    info = video.probe(args.clip)
    llm = SparkLLM(async_client())
    frames = video.keyframes(args.clip, args.frames)
    known = {k: float(v) for k, v in (kv.split("=") for kv in args.set)}
    vid = {"width": info.width, "height": info.height, "fps": info.fps, "duration": info.duration}

    plan = await make_plan(llm, args.question, video=vid, keyframes=frames, known_values=known)
    print(f"Q: {args.question}")
    print(f"PLAN: {plan.interpretation}")
    print(f"  tool={plan.tool}  outputs={plan.outputs}  quantity={plan.quantity}  objects={[(t.role, t.phrases) for t in plan.targets]}")
    print(f"  values={ {k: (v.said, 'verified' if v.verified else 'UNVERIFIED', v.source) for k, v in plan.values.items()} }")
    for m in plan.missing:
        print(f"  MISSING {m.name}: {m.question}")
    for n in plan.notes:
        print("  note:", n)
    if plan.unsupported:
        print("  UNSUPPORTED:", plan.unsupported["reason"])
    print(f"  planner cost: {plan.usage.input_tokens} in / {plan.usage.output_tokens} out tokens "
          f"({plan.usage.reasoning_tokens} reasoning) = ${plan.usage.cost_usd:.5f}")
    if args.plan_only or not plan.ready:
        return

    key = sam.video_key(args.clip)
    with tempfile.TemporaryDirectory() as tmp:
        file_id = video.upload(sync_client(), video.prepare(args.clip, Path(tmp) / "ready.mp4"))
    tracker = sam.SamTracker(async_client(), file_id, (info.width, info.height), key=key)
    checked = await track_objects_with_checks(tracker, plan.main_phrases, info.frame_count, (info.width, info.height),
                                              check_camera=plan.check_camera)
    print(f"\nTRACKING: {checked.status}, {checked.count} object(s), {checked.frames_billed} frames billed this run")
    for e in checked.log:
        print("  ", e.message)
    if not checked.tracks:
        print("  nothing found; stopping.")
        return

    result = run_tool(plan, checked, info, args.scale)
    if result.get("value") is None:
        print("TOOL: no result:", result["warnings"][-1])
        return
    facts = build_facts(plan.tool, result, plan, checks=[e.message for e in checked.log])
    ex = await explain(llm, facts)
    print(f"\nANSWER: {facts.headline.label}: {facts.headline.display}")
    print(f"EXPLANATION ({ex.source}, {ex.attempts} model call(s)"
          + (f", rejected numbers {ex.rejected_numbers}" if ex.rejected_numbers else "") + f"):")
    print("  " + ex.summary)
    for s in ex.how_measured:
        print("   -", s)
    for c in ex.caveats:
        print("   !", c)
    if ex.reason:
        print("  (fallback because:", ex.reason + ")")
    print(f"  explainer cost: {ex.usage.input_tokens} in / {ex.usage.output_tokens} out = ${ex.usage.cost_usd or 0:.5f}")


asyncio.run(main())
