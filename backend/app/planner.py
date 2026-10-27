"""The planner: a question in plain words becomes a checked plan for the measurement.

Muse Spark reads the question (and looks at a few frames of the video) and fills in a plan: which tool, which
objects (named the way they LOOK, so that SAM can find them), and the numbers the user stated, each with the exact
words they came from. It never measures and never calculates.

Everything the model says is then checked in code:
- the tool must exist and be available, and the outputs must be ones that tool can give,
- every object name must obey SAM's rules (one short noun phrase, no lists),
- every number must come with a quote that really is in the question; units are converted here, not by the model;
  a number that cannot be re-read from the quote, or looks implausible, is flagged for the user to confirm,
- what is MISSING (a string length for g, a mass for k, a scale) is decided by the registry, not by the model.
"""
import re
from dataclasses import dataclass, field

from .formatting import AllowedNumbers, find_unsupported_numbers
from .llm import LLMError, Usage
from .tools.registry import (ASKS, AVAILABLE, KNOWN_VALUE_KEYS, OUTPUT_LABELS, REFERENCES, TOOLS, VALUE_NAMES,
                             MissingInput, _ask, missing_inputs, supported_summary)
from .units import QUANTITIES, SI_NAME, UnitError, convert, plausible, verify_quote

MAX_PHRASE_WORDS = 4
MAX_PHRASES = 3
_LEADING = {"the", "a", "an", "some", "this", "that", "segment", "find", "track", "detect", "show", "please", "of"}
_BAD_PHRASE = re.compile(r"[,;+/&:()\[\]]|\b(and|or|not|without|except|but)\b")


class PlannerError(RuntimeError):
    """The question could not be turned into a plan (the model failed)."""


@dataclass
class Target:
    role: str  # "main": what is measured; "reference": an object of known size used as a ruler
    phrases: list  # the best name first, then alternative names for the SAME object


@dataclass
class ValueEntry:
    name: str
    value: float  # in SI units (m, kg, degrees, s)
    unit: str  # the SI unit's name
    said: str  # as stated by the user, e.g. "50 cm"
    quote: str
    source: str  # "question", "given" (by the caller), "earlier" (a previous question) or "table"
    verified: bool  # the number was re-read from the user's own words and matches
    needs_confirmation: bool
    note: str = ""


@dataclass
class Plan:
    question: str
    tool: str | None
    outputs: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    quantity: str | None = None
    line: dict | None = None
    check_camera: bool = True
    compare_with_ramp: bool = False
    values: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    interpretation: str = ""
    unsupported: dict | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def main_phrases(self):
        return next((t.phrases for t in self.targets if t.role == "main"), [])

    @property
    def reference_phrases(self):
        return next((t.phrases for t in self.targets if t.role == "reference"), [])

    @property
    def ready(self):
        return self.unsupported is None and not self.missing

    @classmethod
    def from_dict(cls, d):
        """Rebuild a plan stored as JSON (the inverse of to_dict), for follow-up questions."""
        return cls(question=d.get("question", ""), tool=d.get("tool"), outputs=list(d.get("outputs", [])),
                   targets=[Target(**t) for t in d.get("targets", [])], quantity=d.get("quantity"), line=d.get("line"),
                   check_camera=d.get("check_camera", True), compare_with_ramp=d.get("compare_with_ramp", False),
                   values={k: ValueEntry(**v) for k, v in d.get("values", {}).items()},
                   missing=[MissingInput(**m) for m in d.get("missing", [])], notes=list(d.get("notes", [])),
                   interpretation=d.get("interpretation", ""), unsupported=d.get("unsupported"))

    def to_dict(self):
        return {"question": self.question, "compare_with_ramp": self.compare_with_ramp,
                "tool": self.tool, "outputs": self.outputs, "interpretation": self.interpretation,
                "targets": [{"role": t.role, "phrases": t.phrases} for t in self.targets],
                "quantity": self.quantity, "line": self.line, "check_camera": self.check_camera,
                "values": {k: vars(v) for k, v in self.values.items()},
                "missing": [vars(m) for m in self.missing], "notes": self.notes, "unsupported": self.unsupported,
                "ready": self.ready}


# ---------------------------------------------------------------- the schema and the prompt

def plan_schema():
    tools = list(AVAILABLE) + ["unsupported"]
    outputs = sorted(OUTPUT_LABELS)
    return {
        "type": "object", "additionalProperties": False,
        "required": ["interpretation", "tool", "outputs", "targets", "quantity", "values", "line",
                     "everything_moves_together", "compare_with_ramp", "reference_object", "unsupported_reason"],
        "properties": {
            "interpretation": {"type": "string"},
            "tool": {"type": "string", "enum": tools},
            "outputs": {"type": "array", "items": {"type": "string", "enum": outputs}},
            "targets": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["role", "phrases"],
                "properties": {"role": {"type": "string", "enum": ["main", "reference"]},
                               "phrases": {"type": "array", "items": {"type": "string"}}}}},
            "quantity": {"anyOf": [{"type": "string", "enum": ["acceleration", "speed", "average_speed", "height", "width", "area"]},
                                   {"type": "null"}]},
            "values": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["name", "value", "unit", "quote"],
                "properties": {"name": {"type": "string", "enum": list(VALUE_NAMES)}, "value": {"type": "number"},
                               "unit": {"type": "string"}, "quote": {"type": "string"}}}},
            "line": {"anyOf": [{"type": "object", "additionalProperties": False,
                                "required": ["orientation", "position_fraction"],
                                "properties": {"orientation": {"type": "string", "enum": ["vertical", "horizontal"]},
                                               "position_fraction": {"anyOf": [{"type": "number"}, {"type": "null"}]}}},
                               {"type": "null"}]},
            "everything_moves_together": {"type": "boolean"},
            "compare_with_ramp": {"type": "boolean"},
            "reference_object": {"type": "string", "enum": list(REFERENCES) + ["none"]},
            "unsupported_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }


def build_instructions():
    tool_lines = "\n".join(
        f"- {t.name}: {t.summary} Objects tracked: {t.objects}. The user may state: {t.inputs or 'nothing'}. "
        f"Example questions: {' | '.join(t.examples)}" for t in AVAILABLE.values())
    out_lines = ", ".join(f"{k} ({v})" for k, v in sorted(OUTPUT_LABELS.items()))
    refs = ", ".join(REFERENCES)
    return f"""You are the planning step of Gauge, an instrument that measures things in a video of a physical experiment \
or a machine. You do NOT measure anything and you NEVER calculate, convert or estimate a number: all numbers are \
produced later by deterministic code. Your only job is to read the user's question (and look at the video frames you \
are shown) and fill in a plan.

Tools Gauge has:
{tool_lines}

Quantities you may list in "outputs": {out_lines}.

Rules:
1. tool: the single best tool. If no tool fits, use "unsupported" and explain in unsupported_reason, in plain words \
(Gauge cannot measure sports performance, medical things, or anything that is not a physical motion or count).
2. outputs: only what the user asked for. Use "momentum" only if the user asks for the momentum VALUE itself; a question \
about whether momentum is conserved or how much is lost is "momentum_change" alone. Questions about how quickly an \
oscillation dies away, damping or decay are "damping".
3. targets: the objects an object detector must find. Name each as it LOOKS in the frames you are shown ("glass marble", \
"red ball", "metal spring", "steel ball"), not by its role or what the user called it if that is not what it looks like. \
Each phrase is ONE short noun phrase of one to three words. Put the best phrase first, then up to two other ways to name \
the SAME object. Never use commas, "and", "or", lists, verbs or negations. role "main" is what is measured; role \
"reference" is an object of known size used as a ruler (only if the user mentions one). For rotation_rate the main target is the small marker that goes round (a "red sticker", a "white dot", a "blade tip"), \
not the whole fan or wheel. Give one target per kind of object: \
for several identical objects, one phrase finds them all.
4. values: ONLY numbers the user wrote or said in the question. Copy the number and the unit exactly as stated ("50", "cm"), \
and put the exact words they came from in quote. If the user gave no number, return an empty list. Never convert units, \
never compute, never guess a number. Allowed names: {', '.join(VALUE_NAMES)}. For a time-lapse, "real_duration" is the real time the WHOLE video covers \
("the video covers 3 days") and "frame_interval" is the real time between two frames ("a photo every 10 minutes").
5. quantity: for velocity_acceleration: "acceleration", "speed" (a speed at a given time, or the top speed) or \
"average_speed"; if the user names a time for a speed, give it as the value "time". For growth_over_time: "height" if the \
user asks how tall or high it got, "width" for how wide or spread, otherwise "area". For every other tool null.
6. line (line_crossings only): the orientation and where the line is as a fraction of the picture width (vertical) or height \
(horizontal), 0 to 1 (0.5 is the middle). Use null for the position if the user did not say. For other tools use null.
7. everything_moves_together: true only if the user says the objects are on a conveyor or belt that carries them all.
8. compare_with_ramp: true if the user wants the acceleration compared with what theory predicts for a ramp.
9. reference_object: one of {refs} if the user says that object is visible as a ruler; otherwise "none".
10. interpretation: ONE plain sentence telling the user what you understood, without any numbers you were not given.
This may be a follow-up question about the same video; if earlier context is given, keep the same objects unless the \
question is about something else, and numbers the user stated earlier still apply.
"""


def build_parts(question, video=None, keyframes=None, previous=None):
    lines = [f"Question: {question}"]
    if video:
        lines.append(f"Video: {video.get('width')}x{video.get('height')} pixels, {video.get('fps', 0):.0f} frames per "
                     f"second, {video.get('duration', 0):.1f} seconds long.")
    if previous:
        lines.append("Earlier in this conversation about the same video: " + previous)
    if keyframes:
        lines.append(f"{len(keyframes)} frames from the video follow, in time order.")
    parts = [{"type": "input_text", "text": "\n".join(lines)}]
    parts += [{"type": "input_image", "image_url": url} for url in (keyframes or [])]
    return parts


def previous_summary(plan):
    """A short text of an earlier plan, for follow-up questions."""
    if plan is None or plan.tool is None:
        return None
    vals = ", ".join(f"{k} = {v.said}" for k, v in plan.values.items())
    objs = "; ".join(f"{t.role}: {', '.join(t.phrases)}" for t in plan.targets)
    return f"asked: \"{plan.question}\"; tool {plan.tool}; objects: {objs or 'none'}; numbers stated: {vals or 'none'}."


# ---------------------------------------------------------------- checking what the model said

_QUESTION_WORDS = {"what", "which", "where", "how", "who", "why", "is", "are", "does", "do"}


def clean_phrase(p):
    """One SAM-ready noun phrase, or None if it cannot be made to obey SAM's rules."""
    if "?" in (p or ""):
        return None  # a question is not an object name
    p = re.sub(r"[\"'.!]", "", (p or "").lower()).strip()
    words = p.split()
    if words and words[0] in _QUESTION_WORDS:
        return None
    while words and words[0] in _LEADING:
        words.pop(0)
    p = " ".join(words)
    if not p or _BAD_PHRASE.search(p) or len(words) > MAX_PHRASE_WORDS or not re.fullmatch(r"[a-z0-9][a-z0-9 \-]*", p):
        return None
    return p


def _targets(raw_targets, notes):
    out = []
    for t in raw_targets or []:
        seen, phrases = set(), []
        for p in t.get("phrases", []):
            c = clean_phrase(p)
            if c is None:
                notes.append(f"Ignored the object name '{p}': an object name must be one short noun phrase.")
            elif c not in seen:
                seen.add(c)
                phrases.append(c)
        if phrases and t.get("role") in ("main", "reference") and all(x.role != t["role"] for x in out):
            out.append(Target(t["role"], phrases[:MAX_PHRASES]))
    return out


def _values(raw_values, question, notes):
    out = {}
    for v in raw_values or []:
        name = v.get("name")
        if name not in QUANTITIES or name in out:
            continue
        found, verified = verify_quote(question, v.get("quote", ""), v.get("value", 0))
        if not found:
            notes.append(f"Ignored a {name.replace('_', ' ')} of {v.get('value')} {v.get('unit')}: the words "
                         f"'{v.get('quote', '')}' are not in your question.")
            continue
        try:
            si, si_unit = convert(name, v["value"], v["unit"])
        except UnitError as e:
            notes.append(f"Ignored the {name.replace('_', ' ')}: {e}.")
            continue
        sane = plausible(name, si)
        note = ""
        if not verified:
            note = "Gauge could not re-read this number from your words; please check it."
        elif not sane:
            note = "This looks unusual for what it is; please check the units."
        out[name] = ValueEntry(name, si, si_unit, f"{v['value']:g} {v['unit']}".strip(), v["quote"], "question",
                               verified, (not verified) or (not sane), note)
    return out


def normalise_plan(raw, question, *, video_size=None, known_values=None, previous=None):
    """Turn the model's draft into a checked Plan. Pure code: no model, so it is fully testable."""
    notes = []
    plan = Plan(question=question, tool=None)
    tool = raw.get("tool")
    if tool not in TOOLS or tool == "unsupported" or not TOOLS[tool].available:
        reason = raw.get("unsupported_reason") or "That is not something Gauge measures."
        if tool in TOOLS and not TOOLS[tool].available:
            reason = f"Gauge cannot do that yet ({TOOLS[tool].summary.rstrip('.')} is planned but not built)."
        plan.unsupported = {"reason": reason, "can_measure": supported_summary()}
        plan.interpretation = "Gauge can't answer this one."
        return plan
    spec = TOOLS[tool]
    plan.tool = tool

    wanted = list(dict.fromkeys(raw.get("outputs") or []))
    outputs = [o for o in wanted if o in spec.outputs]
    if wanted and not outputs:
        # nothing the user asked for can come from this tool: use one that can, or say so; never answer a different question
        other = next((t for t in AVAILABLE.values() if any(o in t.outputs for o in wanted)), None)
        if other is None:
            labels = ", ".join(OUTPUT_LABELS.get(o, o) for o in wanted)
            plan.tool = None
            plan.unsupported = {"reason": f"Gauge cannot measure the {labels}.", "can_measure": supported_summary()}
            plan.interpretation = "Gauge can't answer this one."
            return plan
        spec = other
        plan.tool = tool = other.name
        outputs = [o for o in wanted if o in spec.outputs]
        notes.append(f"Using {tool.replace('_', ' ')} because the chosen measurement cannot give what you asked for.")
    dropped = [o for o in wanted if o not in spec.outputs]
    if dropped:
        notes.append(f"{', '.join(OUTPUT_LABELS.get(o, o) for o in dropped)} cannot be given by this measurement and was left out.")
    plan.outputs = outputs or [spec.outputs[0]]

    plan.targets = _targets(raw.get("targets"), notes)
    given_ref = clean_phrase((known_values or {}).get("reference_object")) if isinstance((known_values or {}).get("reference_object"), str) else None
    if given_ref:  # the caller named the object that has the known length: it replaces whatever the model guessed
        plan.targets = [t for t in plan.targets if t.role != "reference"] + [Target("reference", [given_ref])]
    plan.values = _values(raw.get("values"), question, notes)

    # numbers carried from an earlier question, then numbers handed over by the caller (these win)
    for name, v in (previous or {}).items():
        if name not in plan.values:
            plan.values[name] = ValueEntry(**{**vars(v), "source": "earlier"})
    for key, val in (known_values or {}).items():
        name = KNOWN_VALUE_KEYS.get(key)
        if name and val is not None:
            if name in plan.values and plan.values[name].source == "question" and abs(plan.values[name].value - val) > 1e-9:
                notes.append(f"You gave a {name.replace('_', ' ')} in the question and as a setting; the setting "
                             f"({val:g}) is used.")
            unit = SI_NAME[QUANTITIES[name][0]]
            plan.values[name] = ValueEntry(name, float(val), unit, f"{val:g} {unit}", "", "given", True,
                                           not plausible(name, float(val)))

    ref = raw.get("reference_object")
    if ref in REFERENCES and "reference_length" not in plan.values:
        r = REFERENCES[ref]
        plan.values["reference_length"] = ValueEntry(
            "reference_length", r["length_m"], "m", f"{r['length_m']:g} m", "", "table", True, True,
            f"Assumed from the size of {r['label']}; the object must lie along the direction of motion's plane.")
        notes.append(f"Using the known size of {r['label']} ({r['length_m']:g} m) as the ruler; please confirm.")

    if spec.objects and not plan.main_phrases:
        plan.missing.append(_ask("object"))

    allowed_q = {"velocity_acceleration": ("acceleration", "speed", "average_speed"), "growth_over_time": ("height", "width", "area")}
    plan.quantity = raw.get("quantity") if raw.get("quantity") in allowed_q.get(tool, ()) else None
    if tool == "growth_over_time" and plan.quantity is None:
        plan.quantity = "area"
    if tool == "velocity_acceleration":
        if plan.quantity is None:
            plan.quantity = ("speed" if "speed" in plan.outputs else "average_speed" if "average_speed" in plan.outputs
                             else "acceleration")
        plan.compare_with_ramp = bool(raw.get("compare_with_ramp"))
    plan.check_camera = not bool(raw.get("everything_moves_together")) if spec.objects == "several" else True

    if tool == "line_crossings":
        line = raw.get("line") or {}
        orient = line.get("orientation") if line.get("orientation") in ("vertical", "horizontal") else "vertical"
        frac = line.get("position_fraction")
        defaulted = not (isinstance(frac, (int, float)) and 0.0 <= frac <= 1.0)
        frac = 0.5 if defaulted else float(frac)
        size = video_size[0] if orient == "vertical" else video_size[1] if video_size else None
        plan.line = {"orientation": orient, "position_fraction": frac, "position_px": frac * size if size else None,
                     "defaulted": defaulted}
        if defaulted:
            notes.append(f"No line position was given, so the {orient} line through the middle of the picture is used.")

    plan.missing += [m for m in missing_inputs(tool, plan.outputs, plan.values, plan.compare_with_ramp)
                     if all(m.name != x.name for x in plan.missing)]
    plan.notes = notes
    for v in plan.values.values():
        if v.needs_confirmation and v.note:
            plan.notes.append(f"{v.name.replace('_', ' ')} ({v.said}): {v.note}")

    # the model's one-sentence restatement may only contain numbers the user gave; otherwise use a code template
    allowed = AllowedNumbers().add_text(question)
    for v in plan.values.values():
        allowed.add_text(v.said)
    text = (raw.get("interpretation") or "").strip()
    if not text or find_unsupported_numbers(text, allowed):
        what = ", ".join(OUTPUT_LABELS[o] for o in plan.outputs)
        obj = plan.main_phrases[0] if plan.main_phrases else "the object"
        text = f"Measure the {what} of the {obj}." if tool not in ("count_objects", "line_crossings") else \
            f"Measure the {what} ({obj})."
        if raw.get("interpretation"):
            plan.notes.append("The model's one-line summary was replaced because it contained a number you did not give.")
    plan.interpretation = text
    return plan


async def make_plan(llm, question, *, video=None, keyframes=None, known_values=None, previous_plan=None):
    """Ask Muse Spark for a plan and check it. `video`: {"width","height","fps","duration"}. `keyframes`: image data URLs."""
    if not question or not question.strip():
        raise PlannerError("The question is empty.")
    try:
        res = await llm.json(build_instructions(), build_parts(question, video, keyframes, previous_summary(previous_plan)),
                             plan_schema(), "gauge_plan")
    except LLMError as e:
        raise PlannerError(str(e)) from e
    size = (video["width"], video["height"]) if video and video.get("width") else None
    plan = normalise_plan(res.data, question, video_size=size, known_values=known_values,
                          previous=previous_plan.values if previous_plan else None)
    plan.usage = res.usage
    return plan
