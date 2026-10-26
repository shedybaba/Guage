"""Tests for the planner, with a scripted fake model: no network, no cost.

The model's draft is untrusted. These tests feed it good and bad drafts and check that code catches the bad ones.
"""
import asyncio

import pytest

from app.llm import JsonResult, LLMError, Usage
from app.planner import (PlannerError, clean_phrase, make_plan, normalise_plan, plan_schema, build_instructions,
                         previous_summary)

VIDEO = {"width": 720, "height": 1280, "fps": 30.0, "duration": 12.0}


def draft(**over):
    base = {"interpretation": "Measure the swing period of the glass marble.", "tool": "pendulum_period",
            "outputs": ["period"], "targets": [{"role": "main", "phrases": ["glass marble", "marble"]}],
            "quantity": None, "values": [], "line": None, "everything_moves_together": False,
            "compare_with_ramp": False, "reference_object": "none", "unsupported_reason": None}
    base.update(over)
    return base


def value(name, v, unit, quote):
    return {"name": name, "value": v, "unit": unit, "quote": quote}


class FakeLLM:
    def __init__(self, data=None, error=None):
        self.data, self.error, self.calls = data, error, []

    async def json(self, instructions, parts, schema, name, effort=None):
        self.calls.append({"instructions": instructions, "parts": parts, "schema": schema, "name": name})
        if self.error:
            raise self.error
        return JsonResult(self.data, Usage(100, 50, 30, 1, 0.00002))


def plan_for(question, raw, **kw):
    return normalise_plan(raw, question, video_size=(720, 1280), **kw)


# ------------------------------------------------------------ the happy path

def test_a_simple_question_gives_a_ready_plan():
    p = plan_for("What is the swing period?", draft())
    assert p.tool == "pendulum_period" and p.outputs == ["period"]
    assert p.main_phrases == ["glass marble", "marble"]
    assert p.ready and p.missing == [] and p.unsupported is None
    assert p.interpretation == "Measure the swing period of the glass marble."


def test_g_without_a_string_length_asks_for_it():
    p = plan_for("What value of g does that give?", draft(outputs=["g"]))
    assert not p.ready
    assert [m.name for m in p.missing] == ["string_length"]
    assert "string" in p.missing[0].question.lower()


def test_a_stated_length_is_read_converted_and_verified():
    q = "What value of g does that give? The string is half a metre."
    p = plan_for(q, draft(outputs=["g"], values=[value("string_length", 0.5, "metre", "half a metre")]))
    v = p.values["string_length"]
    assert v.value == 0.5 and v.unit == "m" and v.verified and not v.needs_confirmation
    assert p.ready


def test_units_are_converted_by_code_not_by_the_model():
    q = "g from this pendulum? the string is 50 cm"
    p = plan_for(q, draft(outputs=["g"], values=[value("string_length", 50, "cm", "50 cm")]))
    assert p.values["string_length"].value == pytest.approx(0.5)
    assert p.values["string_length"].said == "50 cm"


# ------------------------------------------------------------ the model's numbers are not trusted

def test_a_number_whose_quote_is_not_in_the_question_is_dropped_and_asked_again():
    q = "What value of g does that give?"
    p = plan_for(q, draft(outputs=["g"], values=[value("string_length", 0.5, "m", "the string is 0.5 m")]))
    assert "string_length" not in p.values
    assert [m.name for m in p.missing] == ["string_length"]
    assert any("not in your question" in n for n in p.notes)


def test_a_wrong_number_is_kept_but_flagged_for_confirmation():
    q = "The string is 0.15 m long. What is g?"
    p = plan_for(q, draft(outputs=["g"], values=[value("string_length", 0.5, "m", "0.15 m")]))  # model got it wrong
    v = p.values["string_length"]
    assert not v.verified and v.needs_confirmation
    assert any("check it" in n for n in p.notes)


def test_a_spoken_number_that_cannot_be_re_read_needs_confirmation():
    q = "The string is about a forearm long. What is g?"
    p = plan_for(q, draft(outputs=["g"], values=[value("string_length", 0.3, "m", "about a forearm")]))
    assert p.values["string_length"].needs_confirmation


def test_the_wrong_kind_of_unit_is_ignored():
    q = "What is g? The string is 5 kg"
    p = plan_for(q, draft(outputs=["g"], values=[value("string_length", 5, "kg", "5 kg")]))
    assert "string_length" not in p.values and any("not a length unit" in n for n in p.notes)
    assert [m.name for m in p.missing] == ["string_length"]


def test_an_implausible_value_is_flagged():
    q = "What is g? The string is 500 m"
    p = plan_for(q, draft(outputs=["g"], values=[value("string_length", 500, "m", "500 m")]))
    assert p.values["string_length"].needs_confirmation


def test_values_given_by_the_caller_override_and_are_trusted():
    q = "What is g? The string is 40 cm"
    p = plan_for(q, draft(outputs=["g"], values=[value("string_length", 40, "cm", "40 cm")]),
                 known_values={"string_length_m": 0.55})
    v = p.values["string_length"]
    assert v.value == 0.55 and v.source == "given" and not v.needs_confirmation
    assert any("setting" in n for n in p.notes)


def test_numbers_from_an_earlier_question_still_apply():
    first = plan_for("The string is 0.5 m. What is the period?", draft(values=[value("string_length", 0.5, "m", "0.5 m")]))
    follow = plan_for("What value of g does that give?", draft(outputs=["g"]), previous=first.values)
    assert follow.values["string_length"].value == 0.5 and follow.values["string_length"].source == "earlier"
    assert follow.ready


def test_an_interpretation_with_an_invented_number_is_replaced():
    q = "What is the swing period?"
    p = plan_for(q, draft(interpretation="Measure the period, which is about 1.4 seconds."))
    assert "1.4" not in p.interpretation
    assert p.interpretation == "Measure the swing period of the glass marble."
    assert any("replaced" in n for n in p.notes)


def test_an_interpretation_may_repeat_a_number_the_user_gave():
    q = "What is g? The string is 50 cm"
    p = plan_for(q, draft(outputs=["g"], interpretation="Find g for the pendulum with the 50 cm string.",
                          values=[value("string_length", 50, "cm", "50 cm")]))
    assert "50 cm" in p.interpretation


# ------------------------------------------------------------ tools and outputs

def test_an_unsupported_question_says_so_and_lists_what_gauge_can_do():
    p = plan_for("How high did she jump?", draft(tool="unsupported", unsupported_reason="Gauge does not measure sports."))
    assert p.unsupported and p.tool is None and not p.ready
    assert p.unsupported["reason"] == "Gauge does not measure sports."
    assert len(p.unsupported["can_measure"]) >= 5


def test_an_invented_tool_name_is_treated_as_unsupported():
    p = plan_for("q", draft(tool="quantum_flux_meter"))
    assert p.unsupported is not None


def test_a_planned_but_unbuilt_tool_is_unsupported_for_now():
    p = plan_for("How fast does the plant grow?", draft(tool="growth_over_time"))
    assert p.unsupported is not None


def test_outputs_the_tool_cannot_give_are_dropped_with_a_note():
    p = plan_for("q", draft(outputs=["period", "spring_constant"]))
    assert p.outputs == ["period"] and any("spring constant" in n for n in p.notes)


def test_no_outputs_at_all_falls_back_to_the_tools_main_one():
    assert plan_for("q", draft(outputs=[])).outputs == ["period"]


def test_outputs_that_belong_to_another_tool_switch_to_it():
    p = plan_for("How many are there?", draft(outputs=["count"], targets=[{"role": "main", "phrases": ["ball"]}]))
    assert p.tool == "count_objects" and p.outputs == ["count"]


def test_a_spring_constant_needs_a_mass():
    raw = draft(tool="oscillation_frequency", outputs=["spring_constant"], targets=[{"role": "main", "phrases": ["spring"]}])
    assert [m.name for m in plan_for("What is k?", raw).missing] == ["mass"]
    q = "What is k? The mass is 200 g"
    ok = plan_for(q, {**raw, "values": [value("mass", 200, "g", "200 g")]})
    assert ok.values["mass"].value == pytest.approx(0.2) and ok.ready


def test_free_fall_needs_a_scale():
    raw = draft(tool="free_fall", outputs=["g"], targets=[{"role": "main", "phrases": ["ball"]}])
    assert [m.name for m in plan_for("What is g?", raw).missing] == ["reference_length"]


def test_a_named_reference_object_supplies_the_scale_from_a_table_and_asks_to_confirm():
    raw = draft(tool="free_fall", outputs=["g"], targets=[{"role": "main", "phrases": ["ball"]}],
                reference_object="a4_paper")
    p = plan_for("What is g? The A4 sheet is in the picture.", raw)
    v = p.values["reference_length"]
    assert v.value == pytest.approx(0.297) and v.source == "table" and v.needs_confirmation
    assert p.ready  # nothing missing, but the value is to be confirmed
    assert any("confirm" in n.lower() for n in p.notes)


def test_a_ramp_comparison_needs_the_angle():
    raw = draft(tool="velocity_acceleration", outputs=["acceleration"], compare_with_ramp=True,
                targets=[{"role": "main", "phrases": ["ball"]}], values=[value("reference_length", 1, "m", "1 m")])
    p = plan_for("Compare with theory for the ramp. The metre stick is 1 m.", raw)
    assert [m.name for m in p.missing] == ["ramp_angle"]


def test_quantity_defaults_follow_the_outputs():
    base = dict(tool="velocity_acceleration", targets=[{"role": "main", "phrases": ["ball"]}],
                values=[value("reference_length", 1, "m", "1 m")])
    q = "How fast is the ball? The ruler is 1 m."
    assert plan_for(q, draft(outputs=["speed"], **base)).quantity == "speed"
    assert plan_for(q, draft(outputs=["acceleration"], **base)).quantity == "acceleration"
    assert plan_for(q, draft(outputs=["average_speed"], **base)).quantity == "average_speed"
    assert plan_for(q, draft(outputs=["speed"], quantity="average_speed", **base)).quantity == "average_speed"


def test_collision_percent_change_needs_nothing_but_momentum_needs_mass_and_scale():
    raw = draft(tool="collision_momentum", outputs=["momentum_change"], targets=[{"role": "main", "phrases": ["steel ball"]}])
    assert plan_for("Is momentum conserved?", raw).ready
    raw2 = {**raw, "outputs": ["momentum"]}
    assert {m.name for m in plan_for("What is the momentum?", raw2).missing} == {"mass", "reference_length"}


def test_asking_for_both_momentum_and_its_change_does_not_block_on_mass():
    raw = draft(tool="collision_momentum", outputs=["momentum", "momentum_change"],
                targets=[{"role": "main", "phrases": ["steel ball"]}])
    assert plan_for("Is momentum conserved when the balls collide?", raw).ready  # the percent change needs no mass


def test_a_question_the_chosen_tool_cannot_answer_switches_to_one_that_can():
    raw = draft(tool="pendulum_period", outputs=["damping"])  # damping is not a pendulum_period output
    p = plan_for("How quickly does the swing die away?", raw)
    assert p.tool == "oscillation_frequency" and p.outputs == ["damping"] and p.ready
    assert any("Using oscillation frequency" in n for n in p.notes)


def test_it_never_answers_a_different_question_than_the_one_asked():
    raw = draft(tool="pendulum_period", outputs=["growth"])  # nothing offers this
    p = plan_for("How fast does it grow?", raw)
    assert p.tool is None and p.unsupported is not None


# ------------------------------------------------------------ the line and the belt

def test_the_default_line_is_the_vertical_middle_and_the_user_is_told():
    raw = draft(tool="line_crossings", outputs=["crossings"], targets=[{"role": "main", "phrases": ["part"]}])
    p = plan_for("How many parts crossed?", raw)
    assert p.line["orientation"] == "vertical" and p.line["position_px"] == pytest.approx(360.0) and p.line["defaulted"]
    assert any("middle" in n for n in p.notes)


def test_a_line_position_fraction_becomes_pixels_by_code():
    raw = draft(tool="line_crossings", outputs=["crossings"], targets=[{"role": "main", "phrases": ["part"]}],
                line={"orientation": "horizontal", "position_fraction": 0.25})
    p = plan_for("How many crossed the line a quarter of the way down?", raw)
    assert p.line["position_px"] == pytest.approx(320.0) and not p.line["defaulted"]  # 0.25 * 1280


def test_a_line_fraction_outside_the_picture_is_ignored():
    raw = draft(tool="line_crossings", outputs=["crossings"], targets=[{"role": "main", "phrases": ["part"]}],
                line={"orientation": "vertical", "position_fraction": 1.7})
    assert plan_for("q", raw).line["defaulted"]


def test_a_conveyor_switches_the_camera_check_off():
    raw = draft(tool="line_crossings", outputs=["crossings"], targets=[{"role": "main", "phrases": ["part"]}],
                everything_moves_together=True)
    assert plan_for("parts on a belt", raw).check_camera is False
    assert plan_for("parts", {**raw, "everything_moves_together": False}).check_camera is True


# ------------------------------------------------------------ object names obey SAM's rules

@pytest.mark.parametrize("raw,expected", [
    ("glass marble", "glass marble"), ("The Red Ball.", "red ball"), ("segment the red ball", "red ball"),
    ("a metal spring", "metal spring"), ("steel-ball", "steel-ball"), ("A4 paper", "a4 paper"),
    ("ball, cart", None), ("ball and cart", None), ("ball or cart", None), ("not the red ball", None),
    ("one two three four five", None), ("", None), ("ball + cart", None), ("what is a ball?", None),
])
def test_phrase_rules(raw, expected):
    assert clean_phrase(raw) == expected


def test_bad_phrases_are_dropped_and_good_ones_kept_in_order():
    raw = draft(targets=[{"role": "main", "phrases": ["ball, cart", "red ball", "the red ball", "round red object"]}])
    p = plan_for("q", raw)
    assert p.main_phrases == ["red ball", "round red object"]  # duplicates merge, lists are dropped
    assert any("short noun phrase" in n for n in p.notes)


def test_no_usable_object_name_asks_the_user():
    raw = draft(targets=[{"role": "main", "phrases": ["ball and cart", "the thing that swings and moves"]}])
    p = plan_for("q", raw)
    assert [m.name for m in p.missing] == ["object"] and not p.ready


def test_at_most_three_phrases_per_object():
    raw = draft(targets=[{"role": "main", "phrases": ["a b", "c d", "e f", "g h", "i j"]}])
    assert len(plan_for("q", raw).main_phrases) == 3


# ------------------------------------------------------------ the call to the model

def test_the_request_carries_the_question_the_frames_and_the_rules():
    llm = FakeLLM(draft())
    p = asyncio.run(make_plan(llm, "What is the swing period?", video=VIDEO,
                              keyframes=["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"]))
    call = llm.calls[0]
    texts = [x["text"] for x in call["parts"] if x["type"] == "input_text"]
    images = [x for x in call["parts"] if x["type"] == "input_image"]
    assert "What is the swing period?" in texts[0] and "720x1280" in texts[0]
    assert [i["image_url"] for i in images] == ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"]
    assert "NEVER calculate" in call["instructions"] and "pendulum_period" in call["instructions"]
    assert call["name"] == "gauge_plan"
    assert p.usage.calls == 1 and p.usage.cost_usd > 0


def test_the_schema_is_strict_compatible():
    s = plan_schema()
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    assert "unsupported" in s["properties"]["tool"]["enum"] and "growth_over_time" not in s["properties"]["tool"]["enum"]


def test_a_follow_up_gets_the_earlier_context():
    first = plan_for("The string is 0.5 m. What is the period?", draft(values=[value("string_length", 0.5, "m", "0.5 m")]))
    llm = FakeLLM(draft(outputs=["g"]))
    follow = asyncio.run(make_plan(llm, "What value of g does that give?", video=VIDEO, previous_plan=first))
    assert "Earlier in this conversation" in llm.calls[0]["parts"][0]["text"]
    assert "glass marble" in llm.calls[0]["parts"][0]["text"] and "0.5 m" in llm.calls[0]["parts"][0]["text"]
    assert follow.ready and follow.values["string_length"].value == 0.5


def test_a_model_failure_becomes_a_planner_error():
    with pytest.raises(PlannerError):
        asyncio.run(make_plan(FakeLLM(error=LLMError("down")), "What is the period?"))


def test_an_empty_question_is_refused_without_calling_the_model():
    llm = FakeLLM(draft())
    with pytest.raises(PlannerError):
        asyncio.run(make_plan(llm, "   "))
    assert llm.calls == []


def test_the_prompt_lists_every_available_tool_and_no_unavailable_one():
    text = build_instructions()
    for name in ("pendulum_period", "oscillation_frequency", "free_fall", "velocity_acceleration", "collision_momentum",
                 "count_objects", "line_crossings"):
        assert name in text
    assert "growth_over_time" not in text and "rotation_rate" not in text


def test_to_dict_is_json_ready():
    import json
    p = plan_for("What is g?", draft(outputs=["g"]))
    d = json.loads(json.dumps(p.to_dict()))
    assert d["tool"] == "pendulum_period" and d["ready"] is False and d["missing"][0]["name"] == "string_length"
