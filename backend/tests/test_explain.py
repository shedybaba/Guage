"""Tests for the facts sheet and the explainer's number check, with a scripted fake model.

Facts are built from the REAL tool functions on simulated data, so the field names and numbers have to line up.
"""
import asyncio
import json

import numpy as np
import pytest

from app.explain import (BUILDERS, EXPLANATION_SCHEMA, build_facts, explain, fallback_explanation)
from app.formatting import find_unsupported_numbers, format_measurement
from app.llm import JsonResult, LLMError, Usage
from app.planner import Plan, ValueEntry
from app.quality import track_objects_with_checks  # noqa: F401  (imported to keep the module graph honest)
from app.tools.collision import collision_momentum
from app.tools.counting import count_objects, line_crossings
from app.tools.kinematics import free_fall, velocity_acceleration
from app.tools.pendulum import pendulum_period
from app.tools.spring import oscillation_frequency
from tests.test_collision import collision_scene
from tests.test_counting import belt, group, times
from tests.test_kinematics import SCALE, drop, ramp
from tests.test_pendulum import synthetic
from tests.test_spring import spring


def plan_obj(tool, outputs, values=None, quantity=None, question="q"):
    p = Plan(question=question, tool=tool, outputs=outputs, quantity=quantity)
    for name, (v, said) in (values or {}).items():
        p.values[name] = ValueEntry(name, v, "m", said, said, "question", True, False)
    return p


class ScriptedLLM:
    """Returns one scripted answer per call (a dict, or an LLMError to raise)."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    async def json(self, instructions, parts, schema, name, effort=None):
        self.calls.append(parts[0]["text"])
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return JsonResult(a, Usage(500, 200, 100, 1, 0.0001))


def good(text):
    return {"summary": text, "how_measured": ["The object was followed in every frame."], "caveats": []}


# ------------------------------------------------------------ every tool builds a sheet from its real result

def all_results():
    t, x, y = synthetic(T=1.4192, seconds=15.0)
    pend = pendulum_period(t, x, y, length_m=0.5)
    ts, xs, ys, tr = spring(m=0.2, k=20.0)
    spr = oscillation_frequency(ts, xs, ys, mass_kg=0.2)
    tf, xf, yf, _ = drop()
    ff = free_fall(tf, xf, yf, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01)
    tr_, xr, yr, _ = ramp()
    va = velocity_acceleration(tr_, xr, yr, scale_m_per_px=SCALE, scale_uncertainty_fraction=0.01, ramp_angle_deg=20.0)
    tc, pos, _ = collision_scene()
    col = collision_momentum(tc, pos, masses={k: 1.0 for k in pos})
    from app.sam import TrackPoint
    cnt_tracks = group(belt([0.0, 2.0, 4.0], n_frames=450))
    cnt = count_objects(cnt_tracks, times(450), (720, 1280))
    crs = line_crossings(group(belt([i * 2.0 for i in range(6)], n_frames=450)), times(450), (720, 1280))
    return {"pendulum_period": (pend, plan_obj("pendulum_period", ["g"], {"string_length": (0.5, "0.5 m")}, question="g?")),
            "oscillation_frequency": (spr, plan_obj("oscillation_frequency", ["spring_constant"], {"mass": (0.2, "200 g")})),
            "free_fall": (ff, plan_obj("free_fall", ["g"])),
            "velocity_acceleration": (va, plan_obj("velocity_acceleration", ["acceleration"], quantity="acceleration")),
            "collision_momentum": (col, plan_obj("collision_momentum", ["momentum_change"])),
            "count_objects": (cnt, plan_obj("count_objects", ["count"])),
            "line_crossings": (crs, plan_obj("line_crossings", ["rate_per_minute"]))}


@pytest.fixture(scope="module")
def results():
    return all_results()


def test_every_tool_has_a_builder(results):
    assert set(results) == set(BUILDERS)


@pytest.mark.parametrize("tool", sorted(BUILDERS))
def test_a_facts_sheet_is_built_for_every_tool(results, tool):
    result, plan = results[tool]
    f = build_facts(tool, result, plan, checks=["Tracked 'ball' in 362 of 362 frames, average confidence 0.93."])
    assert f.headline.display and f.headline.display != "n/a"
    assert f.numbers and f.method
    json.dumps(f.to_prompt())  # must be plain JSON


def test_the_headline_follows_what_was_asked(results):
    r, p = results["pendulum_period"]
    assert build_facts("pendulum_period", r, p).headline.id == "g"
    p2 = plan_obj("pendulum_period", ["period"])
    assert build_facts("pendulum_period", r, p2).headline.id == "period"
    r3, p3 = results["line_crossings"]
    assert build_facts("line_crossings", r3, p3).headline.id == "rate"
    assert build_facts("line_crossings", r3, plan_obj("line_crossings", ["crossings"])).headline.id == "crossings"


def test_numbers_in_the_sheet_are_formatted_by_code(results):
    r, p = results["pendulum_period"]
    f = build_facts("pendulum_period", r, p)
    period = next(n for n in f.numbers if n.id == "period")
    assert period.display == format_measurement(r["value"], r["uncertainty"], "s")
    g = next(n for n in f.numbers if n.id == "g")
    assert g.display == format_measurement(r["extra"]["g"]["value"], r["extra"]["g"]["uncertainty"], "m/s^2")
    assert "±" in g.display and "m/s²" in g.display


def test_what_the_user_told_gauge_is_listed(results):
    r, p = results["pendulum_period"]
    f = build_facts("pendulum_period", r, p)
    assert any("0.5 m" == i["display"] for i in f.inputs)


def test_verdicts_are_written_by_code(results):
    r, p = results["pendulum_period"]
    f = build_facts("pendulum_period", r, p)
    assert any("consistent with it" in s or "NOT consistent" in s for s in f.statements)


def test_the_raw_floats_are_not_shown_to_the_model_but_are_allowed_when_rounded(results):
    r, p = results["pendulum_period"]
    f = build_facts("pendulum_period", r, p)
    sent = json.dumps(f.to_prompt())
    assert str(r["value"]) not in sent  # only display strings are sent
    a = f.allowed()
    two_digits = f"{r['value']:.2f}"
    assert find_unsupported_numbers(f"about {two_digits} s", a) == []


# ------------------------------------------------------------ the explainer

def facts_for_pendulum(results):
    r, p = results["pendulum_period"]
    return build_facts("pendulum_period", r, p, checks=["Tracked 'glass marble' in 362 of 362 frames, average confidence 0.93."],
                       question="What value of g does that give? The string is 0.5 m.")


def test_a_clean_explanation_is_accepted(results):
    f = facts_for_pendulum(results)
    text = f"The pendulum gives g = {f.headline.display}, using the 0.5 m string."
    llm = ScriptedLLM(good(text))
    e = asyncio.run(explain(llm, f))
    assert e.source == "model" and e.attempts == 1 and e.rejected_numbers == []
    assert f.headline.display in e.summary
    assert e.usage.calls == 1 and e.usage.cost_usd > 0


def test_a_rounded_version_of_a_measured_number_is_accepted(results):
    f = facts_for_pendulum(results)
    period = next(n for n in f.numbers if n.id == "period")
    llm = ScriptedLLM(good(f"The period is about {period.value:.2f} seconds."))
    assert asyncio.run(explain(llm, f)).source == "model"


def test_an_invented_number_is_rejected_and_the_model_is_told_which(results):
    f = facts_for_pendulum(results)
    bad = good("The pendulum gives g of 9.9 m/s², close to 9.81 m/s².")
    better = good(f"The pendulum gives g = {f.headline.display}.")
    llm = ScriptedLLM(bad, better)
    e = asyncio.run(explain(llm, f))
    assert e.source == "model" and e.attempts == 2
    assert "9.9" in llm.calls[1] and "not in the facts" in llm.calls[1]  # the retry names the offending number


def test_two_bad_answers_fall_back_to_text_written_by_code(results):
    f = facts_for_pendulum(results)
    llm = ScriptedLLM(good("g is 9.9 m/s²."), good("g is 9.95 m/s² and the period 1.5 s."))
    e = asyncio.run(explain(llm, f))
    assert e.source == "fallback" and e.attempts == 2
    assert "9.95" in e.rejected_numbers and "1.5" in e.rejected_numbers
    assert "numbers that were not measured" in e.reason
    assert f.headline.display in e.summary and "9.95" not in e.text()


def test_a_model_failure_falls_back_too(results):
    f = facts_for_pendulum(results)
    e = asyncio.run(explain(ScriptedLLM(LLMError("the model request failed")), f))
    assert e.source == "fallback" and e.attempts == 0 and "failed" in e.reason


def test_an_empty_summary_is_not_accepted(results):
    f = facts_for_pendulum(results)
    e = asyncio.run(explain(ScriptedLLM(good(""), good("")), f))
    assert e.source == "fallback"


@pytest.mark.parametrize("tool", sorted(BUILDERS))
def test_the_fallback_text_passes_the_number_check_itself(results, tool):
    result, plan = results[tool]
    f = build_facts(tool, result, plan, checks=["Tracked 'ball' in 362 of 362 frames, average confidence 0.93."])
    fb = fallback_explanation(f)
    assert fb.source == "fallback" and fb.summary.startswith(f.headline.label)
    assert find_unsupported_numbers(fb.text(), f.allowed()) == []  # code-written text can never trip the rule


def test_the_request_carries_the_rules_and_only_display_strings(results):
    f = facts_for_pendulum(results)
    llm = ScriptedLLM(good(f"g = {f.headline.display}"))
    asyncio.run(explain(llm, f))
    sent = llm.calls[0]
    assert sent.startswith("FACTS:") and f.headline.display in sent
    assert "unsupported" not in sent


def test_the_schema_is_strict_compatible():
    assert EXPLANATION_SCHEMA["additionalProperties"] is False
    assert set(EXPLANATION_SCHEMA["required"]) == set(EXPLANATION_SCHEMA["properties"])


def test_a_number_from_a_warning_may_be_used(results):
    r, p = results["pendulum_period"]
    r = {**r, "warnings": ["Used only 4.5 s to 12.0 s of the clip, where the object swings cleanly."]}
    f = build_facts("pendulum_period", r, p)
    llm = ScriptedLLM(good(f"g = {f.headline.display}."), )
    ok = good(f"g = {f.headline.display}.")
    ok["caveats"] = ["Only the part from 4.5 s to 12.0 s of the clip was clean enough to use."]
    e = asyncio.run(explain(ScriptedLLM(ok), f))
    assert e.source == "model"


def test_a_missing_mass_is_only_a_caveat_when_the_spring_constant_was_asked_for(results):
    r, _ = results["oscillation_frequency"]
    noisy = {**r, "warnings": ["No mass was given, so the spring constant k cannot be calculated: tell Gauge the mass."]}
    asked = build_facts("oscillation_frequency", noisy, plan_obj("oscillation_frequency", ["spring_constant"]))
    only_damping = build_facts("oscillation_frequency", noisy, plan_obj("oscillation_frequency", ["damping"]))
    assert any("spring constant" in w for w in asked.warnings)
    assert only_damping.warnings == []


def test_the_spring_constant_is_only_in_the_method_when_it_was_computed(results):
    r, p = results["oscillation_frequency"]
    with_k = build_facts("oscillation_frequency", r, p)
    without = build_facts("oscillation_frequency", {**r, "extra": {k: v for k, v in r["extra"].items() if k != "spring_constant"}},
                          plan_obj("oscillation_frequency", ["damping"]))
    assert any("spring constant" in m for m in with_k.method)
    assert not any("spring constant" in m for m in without.method)
