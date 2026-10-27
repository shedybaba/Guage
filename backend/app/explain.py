"""The explainer: results from the tools become a short plain-language answer, and no number is invented on the way.

1. Code builds a FACTS sheet from the tool's result: every number is formatted by code ("1.419 ± 0.002 s"),
   and every sentence about a comparison ("consistent with 9.81 m/s²") is written by code too.
2. Muse Spark turns the facts into a short explanation, under the rule: use only numbers from the facts.
3. Code CHECKS the explanation: every number in it must trace back to the facts (formatting.py). If one does
   not, the model is told which and tries once more; if it fails again, a template written entirely in code is
   used instead. A wrong number therefore never reaches the user, whatever the model does.
Limits, said plainly: the check covers digits. It cannot judge whether a sentence is a fair reading of the numbers
(the facts include code-written verdicts such as "consistent with the standard value" for the model to follow), and
it does not police numbers written as words, which the prompt forbids.
"""
import json
import re
from dataclasses import dataclass, field

import numpy as np

from .formatting import AllowedNumbers, find_unsupported_numbers, format_measurement, format_number
from .llm import LLMError, Usage
from .tools.registry import OUTPUT_LABELS

G_STANDARD = 9.81


@dataclass
class Fact:
    id: str
    label: str
    display: str
    value: float | None = None
    uncertainty: float | None = None
    units: str = ""


@dataclass
class Facts:
    question: str
    tool: str
    headline: Fact
    numbers: list = field(default_factory=list)  # list[Fact]
    inputs: list = field(default_factory=list)  # what the user told Gauge: [{"label","display"}]
    statements: list = field(default_factory=list)  # sentences written by code (verdicts, comparisons)
    checks: list = field(default_factory=list)  # how Gauge checked its work (plain messages)
    warnings: list = field(default_factory=list)
    method: list = field(default_factory=list)  # how it was measured, in fixed words
    formulas: list = field(default_factory=list)

    def to_prompt(self):
        f = lambda x: {"label": x.label, "display": x.display}  # noqa: E731
        return {"question": self.question, "headline": f(self.headline), "numbers": [f(n) for n in self.numbers],
                "you_were_told": self.inputs, "verdicts_written_by_code": self.statements,
                "how_gauge_checked_its_work": self.checks, "warnings": self.warnings,
                "method_in_fixed_words": self.method, "formulas": self.formulas}

    def allowed(self):
        """Every number the model may write: those in any text the code wrote, and the raw values behind them."""
        a = AllowedNumbers().add_text(self.question)
        a.add_any([vars(self.headline)] + [vars(n) for n in self.numbers])
        a.add_any([self.inputs, self.statements, self.checks, self.warnings, self.method, self.formulas])
        return a


# ---------------------------------------------------------------- building the facts

def _fact(fid, label, value, unc=None, units="", sig=4):
    return Fact(fid, label, format_measurement(value, unc, units, sig), value, unc, units)


def _sub(fid, label, d):
    return _fact(fid, label, d["value"], d.get("uncertainty"), d.get("units", ""))


def _inputs(plan):
    out = []
    if plan is None:
        return out
    for name, v in plan.values.items():
        tag = {"question": "you said", "earlier": "you said earlier", "given": "you set", "table": "assumed from a known size"}[v.source]
        out.append({"label": f"{name.replace('_', ' ')} ({tag})", "display": v.said})
    return out


def _pendulum(r, plan):
    fq, x = r["fit_quality"], r["extra"]
    T, s = r["value"], r["uncertainty"]
    nums = [_fact("period", "Swing period", T, s, "s"), _fact("frequency", "Frequency", 1 / T, s / T ** 2, "Hz"),
            _fact("cycles", "Full swings measured", fq["cycles"], None, "", sig=3)]
    statements, formulas, head = [], [], nums[0]
    if "g" in x:
        g = x["g"]
        nums.append(_fact("g", "Gravitational acceleration g", g["value"], g["uncertainty"], g["units"]))
        nums.append(_fact("g_diff", "Difference from the standard value", g["percent_error_vs_standard"], None, "%", sig=2))
        within = abs(g["value"] - G_STANDARD) <= 2 * g["uncertainty"]
        statements.append(f"The standard value is {G_STANDARD} m/s²; the measured g is "
                          f"{'consistent with it within its uncertainty' if within else 'NOT consistent with it within its uncertainty'}.")
        formulas.append("g = 4π²L/T²")
        head = nums[-2] if "g" in (plan.outputs if plan else []) else nums[0]
    seg = fq["segment"]
    if seg["trimmed"]:
        nums.append(_fact("used_from", "Used from", seg["start_s"], None, "s"))
        nums.append(_fact("used_to", "Used until", seg["end_s"], None, "s"))
    method = ["The object was found and followed in every frame, and the centre of its outline was its position.",
              "A damped sine wave was fitted to its position over time; the period of that wave is the swing period."]
    if "g" in x:
        method.append("g follows from the pendulum formula, using the string length you gave and the measured period.")
    return head, nums, statements, method, formulas


def _spring(r, plan):
    x, fq = r["extra"], r["fit_quality"]
    f = _fact("frequency", "Oscillation frequency", r["value"], r["uncertainty"], "Hz")
    nums = [f, _sub("natural_frequency", "Natural frequency without damping", x["natural_frequency"]),
            _fact("cycles", "Full oscillations measured", fq["cycles"], None, "", sig=3)]
    statements, formulas, head = [], [], f
    z = x["damping_ratio"]
    if z["measurable"]:
        damping = _fact("damping", "Damping ratio", z["value"], z["uncertainty"], "")
        nums.append(damping)
        nums.append(_sub("decay_rate", "Decay rate of the amplitude", z["decay_rate"]))
        if z.get("time_constant"):
            nums.append(_sub("time_constant", "Time for the amplitude to fall to about a third", z["time_constant"]))
        if plan and "damping" in plan.outputs:
            head = damping
    else:
        nums.append(_fact("damping_limit", "Damping ratio is below", z["upper_bound"], None, ""))
        statements.append("No damping could be measured: the oscillation barely died away in the clip.")
    if "spring_constant" in x:
        k = x["spring_constant"]
        nums.append(_sub("k", "Spring constant k", k))
        formulas.append("k = m ω² (m: moving mass, ω: natural angular frequency)")
        if plan and "spring_constant" in plan.outputs:
            head = nums[-1]
    env = x["damping_envelope"]["type"]
    if env.startswith(("linear", "exponential")):
        statements.append(f"The amplitude fell {'in a straight line (like sliding friction)' if env.startswith('linear') else 'exponentially (like viscous damping)'}.")
    method = ["The object was found and followed in every frame.",
              "A damped sine wave was fitted to its position over time to get the frequency and how fast the swings shrink."]
    if "spring_constant" in x:
        method.append("The spring constant uses the natural frequency (what the frequency would be without damping) and the mass.")
    return head, nums, statements, method, formulas


def _free_fall(r, plan):
    x, fq = r["extra"], r["fit_quality"]
    g = _fact("g", "Acceleration of the fall" if r["units"] != "m/s^2" else "Acceleration of the fall (g)", r["value"],
              r["uncertainty"], r["units"])
    nums = [g, _fact("frames", "Frames of free fall used", fq["n_frames"], None, ""),
            _fact("duration", "Duration of the fall", fq["duration_s"], None, "s")]
    statements = []
    if "consistent_with_standard_g" in x:
        nums.append(_fact("diff", "Difference from the standard value", abs(x["percent_difference_from_standard"]), None, "%", sig=2))
        statements.append(f"The standard value is {G_STANDARD} m/s²; the measured value is "
                          f"{'consistent with it within its uncertainty' if x['consistent_with_standard_g'] else 'NOT consistent with it within its uncertainty'}.")
        nums.append(_fact("drop", "Distance fallen", x["drop_distance"], None, "m"))
        nums.append(_fact("impact", "Speed at the end of the fall", x["impact_speed"], None, "m/s"))
    if abs(x["tilt_deg"]) > 3:
        nums.append(_fact("tilt", "Tilt of the camera", abs(x["tilt_deg"]), None, "degrees"))
    method = ["The object was found and followed in every frame.",
              "The stretch where it was falling freely was picked out, and a parabola (constant acceleration) was fitted to it.",
              "The acceleration is the size of that fitted acceleration, converted to metres with the known length in the picture."]
    return g, nums, statements, method, []


def _kinematics(r, plan):
    x = r["extra"]
    nums, statements = [], []
    sp = x["peak_speed"]
    nums.append(_sub("peak_speed", "Top speed", sp))
    nums.append(_sub("average_speed", "Average speed while moving", x["average_speed"]))
    nums.append(_sub("acceleration", "Acceleration along the path", x["acceleration"]))
    q = plan.quantity if plan else "acceleration"
    head = {"speed": nums[0], "average_speed": nums[1], "acceleration": nums[2]}[q]
    if q == "speed" and plan and "time" in plan.values:
        head = _fact("speed_at", f"Speed at {plan.values['time'].said}", r["value"], r["uncertainty"], r["units"])
        nums.insert(0, head)
    if "ramp" in x:
        rp = x["ramp"]
        nums.append(_fact("ratio", "Measured acceleration divided by g sin(angle)", rp["ratio_to_g_sin_theta"]["value"],
                          rp["ratio_to_g_sin_theta"]["uncertainty"], ""))
        for name, val in rp["expected"].items():
            nums.append(_fact(f"expected_{name}", "Expected for " + name.replace("_", " "), val, None, "m/s^2"))
        statements.append(f"The closest match is: {rp['closest'].replace('_', ' ')}.")
    method = ["The object was found and followed in every frame.",
              "Speed and acceleration come from short parabola fits at every frame, which smooth out the tracking noise.",
              "The acceleration comes from fitting a constant-acceleration parabola to the stretch where it speeds up steadily."]
    return head, nums, statements, method, []


def _collision(r, plan):
    x = r["extra"]
    head = _fact("momentum_change", "Change of momentum across an impact", r["value"], r["uncertainty"], "%")
    nums = [head, _fact("valid", "Impacts used", x["n_valid"], None, ""), _fact("found", "Impacts found", x["n_events"], None, ""),
            _fact("ke", "Kinetic energy after divided by before", x["kinetic_energy_ratio"], None, "", sig=2)]
    statements = []
    if abs(r["value"]) <= 2 * r["uncertainty"]:
        statements.append("Momentum is conserved within the measurement's uncertainty.")
    else:
        statements.append("The change of momentum is larger than the measurement's uncertainty.")
    if not x.get("masses_given"):
        statements.append("All objects were assumed to have the same mass.")
    method = ["Every object was found and followed, and the camera's own movement was removed using the objects that stayed still.",
              "Each impact was found as a sudden change of velocities; velocities just before and just after were fitted.",
              "Momentum is mass times velocity added up over the moving objects; the result is the average over the valid impacts."]
    return head, nums, statements, method, []


def _count(r, plan):
    x = r["extra"]
    head = _fact("count", "Number of objects", r["value"], r["uncertainty"] or None, "")
    nums = [head, _fact("together", "Most objects in view at once", x["max_simultaneous"], None, "")]
    if x["range"][0] != x["range"][1]:
        nums.append(_fact("low", "At least", x["range"][0], None, ""))
        nums.append(_fact("high", "At most", x["range"][1], None, ""))
    method = ["Every object was found and followed through the video, and pieces of the same object were joined.",
              "Very short detections were ignored as flicker; doubtful ones widen the range instead of being hidden."]
    return head, nums, [], method, []


def _crossings(r, plan):
    x = r["extra"]
    rate = x["rate_per_minute"]
    total = _fact("crossings", "Crossings of the line", r["value"], r["uncertainty"] or None, "")
    nums = [total, _sub("rate", "Rate", rate), _fact("objects", "Different objects that crossed", x["objects_crossed"], None, "")]
    for name, n in x["by_direction"].items():
        nums.append(_fact(name, "Crossings " + name.replace("_", " "), n, None, ""))
    head = nums[1] if plan and "rate_per_minute" in plan.outputs else total
    statements = [f"Basis of the rate: {rate['basis']}."]
    method = ["Every object was found and followed; a crossing is counted when the object's centre gets clearly past the line.",
              "The moment of each crossing is placed between frames, and the rate comes from the spacing of the crossings."]
    return head, nums, statements, method, []


def _growth(r, plan):
    x = r["extra"]
    outs = plan.outputs if plan else ["total_change"]
    q = x["quantity"]
    what = {"height": "height", "width": "width", "area": "area"}[q]
    change = _sub("total_change", f"Total change in {what}", x["total_change_percent"])
    nums = [change, _sub("start_size", f"{what.capitalize()} at the start", x["start_size"]),
            _sub("end_size", f"{what.capitalize()} at the end", x["end_size"]),
            _sub("absolute_change", f"Change in {what}", x["absolute_change"])]
    statements = list(x.get("statements", []))
    head = change
    if "growth_rate" in x:
        rate = _sub("growth_rate", "Average growth rate", x["growth_rate"])
        rel = _sub("relative_growth_rate", "Average relative growth rate", x["relative_growth_rate"])
        nums += [rate, rel, _sub("real_duration", "Real time covered by the video", x["real_duration"])]
        if "growth_rate" in outs:
            head = rate
    if "doubling_time" in x:
        dbl = _sub("doubling_time", "Doubling time at the average growth rate", x["doubling_time"])
        nums.append(dbl)
        if "doubling_time" in outs:
            head = dbl
    elif "doubling_time" in outs:
        statements.append("No doubling time can be stated: the object did not grow by more than the measurement's uncertainty."
                          if "growth_rate" in x else "No doubling time can be stated without the real time the video covers.")
    if not x["grew_measurably"]:
        statements.append("The size at the end is not different from the size at the start by more than the measurement's "
                          "uncertainty.")
    method = ["The object was found and followed in every frame, and its " + ("outline's height" if q == "height" else
              "outline's width" if q == "width" else "outline area") + " was read in each one.",
              "The size at the start and at the end came from short straight lines fitted to the first and last frames, "
              "so a single odd frame does not decide the answer.",
              "The uncertainty comes from how much the sizes scatter around the trend, widened because neighbouring frames "
              "share errors."]
    formulas = ["change = (end size - start size) / start size"]
    if "growth_rate" in x:
        formulas += ["growth rate = (end size - start size) / real time", "doubling time = ln 2 / relative growth rate"]
        if x.get("real_time_known"):
            method.append("Real time comes from what you said about how long the time-lapse covers.")
    return head, nums, statements, method, formulas


def _rotation(r, plan):
    x, fq = r["extra"], r["fit_quality"]
    rpm = _fact("rpm", "Rotation speed", r["value"], r["uncertainty"], "rpm")
    turns = _sub("revolutions", "Revolutions in the part used", x["revolutions"])
    nums = [rpm, _sub("frequency", "Rotation frequency", x["frequency"]), _sub("period", "Time for one revolution", x["period"]),
            turns, _sub("angular_speed", "Angular speed", x["angular_speed"]),
            _fact("frames_per_rev", "Frames per revolution", fq["frames_per_revolution"], None, "", sig=3)]
    seg = fq["segment"]
    if seg["trimmed"]:
        nums.append(_fact("used_from", "Used from", seg["start_s"], None, "s"))
        nums.append(_fact("used_to", "Used until", seg["end_s"], None, "s"))
    if x.get("fft_rpm") is not None:
        nums.append(_fact("fft_rpm", "Check by Fourier analysis", x["fft_rpm"], None, "rpm"))
    head = turns if plan and "revolutions" in plan.outputs and "rpm" not in plan.outputs else rpm
    statements = [f"As seen in the video, the marker goes round {x['direction']}."]
    method = ["The marker was found and followed in every frame.",
              "A circle was fitted to its path to find the axis, the marker's angle round that axis was counted in whole turns, "
              "and a straight line was fitted to the angle over time: its slope is the rotation speed.",
              "This assumes the marker turns less than half a turn between two frames; a much faster rotation would look "
              "slower, or even backward, on camera."]
    return head, nums, statements, method, ["rpm = angular speed x 30 / pi"]


BUILDERS = {"pendulum_period": _pendulum, "oscillation_frequency": _spring, "free_fall": _free_fall,
            "velocity_acceleration": _kinematics, "collision_momentum": _collision, "count_objects": _count,
            "line_crossings": _crossings, "growth_over_time": _growth, "rotation_rate": _rotation}


def _relevant_warnings(tool, result, plan):
    """The tool's warnings that matter for what was asked (a missing mass is only a problem if k was requested)."""
    out = []
    for w in result.get("warnings", []):
        if tool == "oscillation_frequency" and "spring constant k cannot be calculated" in w \
                and plan and "spring_constant" not in plan.outputs:
            continue
        out.append(w)
    return out


def build_facts(tool, result, plan=None, checks=None, question=None):
    """The facts sheet for one tool result. `checks`: plain messages from the quality loop's log."""
    head, nums, statements, method, formulas = BUILDERS[tool](result, plan)
    return Facts(question or (plan.question if plan else ""), tool, head, nums, _inputs(plan), statements,
                 list(checks or []), _relevant_warnings(tool, result, plan), method, formulas)


# ---------------------------------------------------------------- the explanation

EXPLANATION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["summary", "how_measured", "caveats"],
    "properties": {"summary": {"type": "string"}, "how_measured": {"type": "array", "items": {"type": "string"}},
                   "caveats": {"type": "array", "items": {"type": "string"}}},
}

INSTRUCTIONS = """You write the plain-language explanation of a measurement for a student, a teacher or an engineer. \
Code has already measured everything; you receive its FACTS as JSON. You only put words around them.

Rules about numbers (they are checked by code, and an answer that breaks them is thrown away):
- Use ONLY numbers that appear in the facts, copied exactly as written (for example 1.419 ± 0.002 s). Never put quotation marks around numbers or units. Never calculate, \
round differently, convert units, work out a difference, a ratio or a percentage, or add any other number.
- If you want to compare two numbers, use words only ("close to", "a little below"), never a new number.
- Do not write quantities as number words ("two", "half"). Do not number list items. Do not mention software or model names.

How to write:
- summary: answer the question first, in one to three plain sentences, with the headline result and its uncertainty. Follow \
the verdicts written by code exactly; do not contradict them.
- how_measured: three to five short plain sentences based on "method_in_fixed_words" and "how_gauge_checked_its_work" (mention \
a retry or a fix if one happened).
- caveats: at most three short sentences, turning the warnings that matter for trusting this result into plain advice \
(how to film better, what to check). Leave it empty if there is nothing important.
Never claim more certainty than the facts give."""


@dataclass
class Explanation:
    summary: str
    how_measured: list
    caveats: list
    source: str  # "model" or "fallback"
    attempts: int = 0
    rejected_numbers: list = field(default_factory=list)  # numbers the model wrote that the code did not supply
    reason: str = ""  # why the fallback was used
    usage: Usage = field(default_factory=Usage)

    def text(self):
        return "\n".join([self.summary] + self.how_measured + self.caveats)

    def to_dict(self):
        return {"summary": self.summary, "how_measured": self.how_measured, "caveats": self.caveats,
                "source": self.source, "attempts": self.attempts, "rejected_numbers": self.rejected_numbers,
                "reason": self.reason}


def fallback_explanation(facts):
    """The answer written entirely by code: used when the model fails or breaks the number rule."""
    h = facts.headline
    summary = f"{h.label}: {h.display}."
    if facts.statements:
        summary += " " + facts.statements[0]
    return Explanation(summary, list(facts.method) + [c for c in facts.checks[:2]], list(facts.warnings[:3]), "fallback")


def strip_number_quotes(text):
    """Remove quotation marks wrapped around a figure ("0.86 s" -> 0.86 s): the model copies them from the facts sheet."""
    return re.sub(r'["“”]([-+−]?\d[^"“”\n]*)["“”]', r"\1", text)


async def explain(llm, facts, max_attempts=2):
    """Ask Muse Spark for the explanation and check every number in it; fall back to code-written text if needed."""
    allowed = facts.allowed()
    prompt = json.dumps(facts.to_prompt(), ensure_ascii=False)
    usage, bad, reason, made = Usage(), [], "", 0
    for attempt in range(1, max_attempts + 1):
        text = f"FACTS:\n{prompt}"
        if bad:
            text += (f"\n\nYour previous answer contained numbers that are not in the facts: {', '.join(bad)}. "
                     "Rewrite it using only numbers from the facts, exactly as written.")
        try:
            res = await llm.json(INSTRUCTIONS, [{"type": "input_text", "text": text}], EXPLANATION_SCHEMA, "gauge_explanation")
        except LLMError as e:
            reason = str(e)
            break
        made = attempt
        usage.add(res.usage)
        d = res.data
        parts = [str(d.get("summary", ""))] + [str(s) for s in d.get("how_measured", [])] + [str(s) for s in d.get("caveats", [])]
        bad = find_unsupported_numbers("\n".join(parts), allowed)
        if not bad and d.get("summary", "").strip():
            return Explanation(strip_number_quotes(d["summary"].strip()), [strip_number_quotes(s.strip()) for s in d["how_measured"]],
                               [strip_number_quotes(s.strip()) for s in d["caveats"]], "model", attempt, [], "", usage)
        reason = f"the explanation contained numbers that were not measured: {', '.join(bad)}" if bad else "the summary was empty"
    fb = fallback_explanation(facts)
    fb.attempts, fb.rejected_numbers, fb.reason, fb.usage = made, bad, reason, usage
    return fb
