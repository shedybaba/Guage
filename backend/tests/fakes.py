"""Fakes for testing the job runner and the API end to end: a model that answers like a good Muse Spark, and a
SAM that returns simulated worlds. Everything inside Gauge (planner checks, quality loop, tools, number check,
store, events) runs for real; only the two paid services are replaced.
"""
import json
import subprocess

import numpy as np

from app.jobs import JobRunner, Limits
from app.llm import JsonResult, Usage
from app.sam import TrackPoint, TrackResult

FPS = 30.0
SECONDS = 12
W, H = 320, 240


def make_clip(path, seconds=SECONDS, fps=int(FPS), size=f"{W}x{H}"):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}", "-t", str(seconds),
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def draft(**over):
    base = {"interpretation": "Measure the swing period of the glass marble.", "tool": "pendulum_period",
            "outputs": ["period"], "targets": [{"role": "main", "phrases": ["glass marble"]}], "quantity": None,
            "values": [], "line": None, "everything_moves_together": False, "compare_with_ramp": False,
            "reference_object": "none", "unsupported_reason": None}
    base.update(over)
    return base


def value(name, v, unit, quote):
    return {"name": name, "value": v, "unit": unit, "quote": quote}


ROUTES = {
    "swing period": draft(),
    "what value of g does that give? the string is half a metre": draft(
        outputs=["g"], values=[value("string_length", 0.5, "metre", "half a metre")]),
    "what value of g does that give?": draft(outputs=["g"]),
    "forearm": draft(outputs=["g"], values=[value("string_length", 0.3, "m", "about a forearm")]),
    "how high can she jump": draft(tool="unsupported", unsupported_reason="Gauge does not measure sports.", targets=[]),
    "drop with a stick": draft(tool="free_fall", outputs=["g"], interpretation="Measure g from the falling ball.",
                               targets=[{"role": "main", "phrases": ["ball"]}, {"role": "reference", "phrases": ["metre stick"]}],
                               values=[value("reference_length", 1, "m", "1 m")]),
    "drop without a name": draft(tool="free_fall", outputs=["g"], interpretation="Measure g from the falling ball.",
                                 targets=[{"role": "main", "phrases": ["ball"]}], values=[value("reference_length", 1, "m", "1 m")]),
    "find the ghost": draft(targets=[{"role": "main", "phrases": ["unicorn"]}]),
    "still object": draft(targets=[{"role": "main", "phrases": ["rock"]}]),
    "how many rpm": draft(tool="rotation_rate", outputs=["rpm"], interpretation="Measure the rotation speed of the fan.",
                          targets=[{"role": "main", "phrases": ["red sticker"]}]),
    "how much did the plant grow": draft(tool="growth_over_time", outputs=["total_change"], quantity="height",
                                         interpretation="Measure how much the plant grew.",
                                         targets=[{"role": "main", "phrases": ["green plant"]}]),
    "how fast does the plant grow": draft(tool="growth_over_time", outputs=["growth_rate"], quantity="height",
                                          interpretation="Measure how fast the plant grows.",
                                          targets=[{"role": "main", "phrases": ["green plant"]}]),
    "plant grow over three days": draft(tool="growth_over_time", outputs=["growth_rate"], quantity="height",
                                        interpretation="Measure how fast the plant grows.",
                                        targets=[{"role": "main", "phrases": ["green plant"]}],
                                        values=[value("real_duration", 3, "days", "three days")]),
    "count them": draft(tool="count_objects", outputs=["count"], interpretation="Count the parts.",
                        targets=[{"role": "main", "phrases": ["part"]}]),
}


class RoutedLLM:
    """Answers the planner by looking the question up in ROUTES, and the explainer by quoting the headline."""

    def __init__(self, routes=None, fail_plan=None, fail_explain=False):
        self.routes, self.fail_plan, self.fail_explain, self.calls = routes or ROUTES, fail_plan, fail_explain, []

    async def json(self, instructions, parts, schema, name, effort=None):
        self.calls.append(name)
        usage = Usage(1000, 300, 100, 1, 0.0003)
        if name == "gauge_plan":
            if self.fail_plan:
                raise self.fail_plan
            text = parts[0]["text"].lower()
            q = text.split("question:", 1)[1].split("\n")[0].strip()
            best = max((k for k in self.routes if k in q), key=len, default=None)
            if best is None:
                raise AssertionError(f"no scripted plan for question: {q}")
            return JsonResult(self.routes[best], usage)
        facts = json.loads(parts[0]["text"].split("FACTS:\n", 1)[1].split("\n\n")[0])
        head = facts["headline"]["display"]
        if self.fail_explain:
            return JsonResult({"summary": "It is 123.456 of something.", "how_measured": [], "caveats": []}, usage)
        return JsonResult({"summary": f"The result is {head}.", "how_measured": ["The object was followed in every frame."],
                           "caveats": []}, usage)


def pt(frame, oid, x, y, size=20.0, conf=0.95, long_side=None):
    return TrackPoint(frame, str(oid), x - size / 2, y - size / 2, x + size / 2, y + size / 2, conf, float(x), float(y),
                      size * size, long_side)


def world(phrase, n):
    """The simulated SAM answer for a phrase: a list of TrackPoints for n frames."""
    rng = np.random.default_rng(sum(map(ord, phrase)))  # deterministic: the same phrase gives the same noise every run
    t = np.arange(n) / FPS
    if phrase in ("glass marble", "marble"):  # a pendulum with a 1.2 s period, swinging after a 1.5 s set-up
        x = 160 + 100 * np.exp(-0.01 * np.clip(t - 1.5, 0, None)) * np.sin(2 * np.pi * np.clip(t - 1.5, 0, None) / 1.2)
        return [pt(f, 0, x[f] + rng.normal(0, 0.2), 120 + rng.normal(0, 0.2)) for f in range(n)]
    if phrase == "ball":  # held still for 2 s, then a 2 m drop; the metre stick is 250 px long (0.004 m per pixel)
        tf = np.sqrt(2 * 2.0 / 9.81)
        tau = np.clip(t - 2.0, 0, tf)
        y = 20 + 0.5 * 9.81 * tau ** 2 / 0.004
        return [pt(f, 0, 160 + rng.normal(0, 0.15), y[f] + rng.normal(0, 0.15)) for f in range(n)]
    if phrase == "metre stick":
        return [pt(f, 0, 40, 120 + rng.normal(0, 0.1), size=10.0, long_side=250.0) for f in range(n)]
    if phrase == "red sticker":  # a marker going round a fan hub at 90 rpm (1.5 turns a second)
        ang = 2 * np.pi * 1.5 * t
        return [pt(f, 0, 160 + 60 * np.cos(ang[f]) + rng.normal(0, 0.3), 120 - 60 * np.sin(ang[f]) + rng.normal(0, 0.3), size=10.0)
                for f in range(n)]
    if phrase == "green plant":  # a time-lapse: the plant's outline grows from 60 to 140 pixels tall, 133%
        return [pt(f, 0, 160, 200 - (60 + 80 * f / n) / 2 + rng.normal(0, 0.2), size=60 + 80 * f / n + rng.normal(0, 0.3))
                for f in range(n)]
    if phrase == "rock":
        return [pt(f, 0, 100 + rng.normal(0, 0.1), 100 + rng.normal(0, 0.1)) for f in range(n)]
    if phrase == "part":  # three parts crossing the picture left to right
        out = []
        for i, start in enumerate((0.0, 3.0, 6.0)):
            for f in range(n):
                x = -10 + 80.0 * (f / FPS - start)
                if -10 <= x <= W + 10:
                    out.append(pt(f, i, x, 60 + 30 * i))
        return out
    return []  # not found


class FakeTracker:
    def __init__(self, n_frames, on_progress=None, delay=0.0, boom=None):
        self.n, self.on_progress, self.delay, self.boom = n_frames, on_progress, delay, boom
        self.phrases = []

    async def track_phrase(self, phrase):
        import asyncio
        self.phrases.append(phrase)
        if self.boom:
            raise self.boom
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.on_progress:
            self.on_progress(f"'{phrase}'", self.n // 2)
            self.on_progress(f"'{phrase}'", self.n)
        return TrackResult(phrase, world(phrase, self.n), self.n, [])

    async def track_boxes(self, prompts, direction="forward"):
        return TrackResult("boxes", [], 0, [])

    async def track_box(self, frame, box, direction="forward"):
        return TrackResult("box", [], 0, [])


class Harness:
    """A JobRunner wired to the fakes, plus a record of what was called."""

    def __init__(self, store, data_dir, llm=None, limits=None, tracker_kwargs=None, camera_fn=None):
        self.llm = llm or RoutedLLM()
        self.tracker_calls = 0
        self.trackers = []
        kw = tracker_kwargs or {}

        async def factory(vrow, on_progress):
            self.tracker_calls += 1
            tr = FakeTracker(vrow["info"]["frame_count"], on_progress, **kw)
            self.trackers.append(tr)
            return tr

        self.runner = JobRunner(store, data_dir, self.llm, factory, keyframes_fn=lambda v: ["data:image/jpeg;base64,AAAA"],
                                limits=limits or Limits(), camera_fn=camera_fn)
