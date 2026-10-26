"""What Gauge can measure: one description of every tool, used to brief the planner and to decide what is missing.

The language model only ever picks a tool and the object names; WHICH inputs a measurement needs is decided here, in
code, so a forgetful or over-eager model cannot make Gauge compute g without a string length.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    available: bool
    summary: str
    outputs: tuple  # quantities this tool can report (see OUTPUT_LABELS)
    objects: str  # "one" or "several" objects are tracked
    inputs: str  # what the user may state, in plain words, for the prompt
    examples: tuple


OUTPUT_LABELS = {
    "period": "swing period", "frequency": "frequency", "g": "gravitational acceleration g",
    "damping": "damping ratio", "spring_constant": "spring constant k", "acceleration": "acceleration",
    "speed": "speed", "average_speed": "average speed", "momentum_change": "change of momentum across a collision",
    "momentum": "momentum before and after a collision", "count": "number of objects",
    "crossings": "number of crossings of a line", "rate_per_minute": "rate of crossings per minute",
}

TOOLS = {t.name: t for t in (
    ToolSpec("pendulum_period", True,
             "Period (and frequency) of a swinging pendulum; also g if the string length is stated.",
             ("period", "frequency", "g"), "one",
             "string length (to the centre of the bob)",
             ("What is the swing period?", "What value of g does that give? The string is 0.5 m.")),
    ToolSpec("oscillation_frequency", True,
             "Frequency, damping ratio and how fast the swings die away for ANY oscillation (a mass on a spring, but also a "
             "swinging pendulum); the spring constant k if the mass is stated. Use this for damping and decay questions.",
             ("frequency", "damping", "spring_constant"), "one",
             "hanging mass, the spring's own mass, a known length for amplitude/speed",
             ("What is the spring's frequency?", "What is the spring constant? The mass is 200 g.")),
    ToolSpec("free_fall", True,
             "Acceleration of a falling or thrown object (g) from a video of the fall.",
             ("g", "acceleration"), "one",
             "a known length visible in the picture in the plane of the fall (for the scale)",
             ("What is the acceleration of the dropped ball?", "Measure g from this drop. The metre stick is 1 m.")),
    ToolSpec("velocity_acceleration", True,
             "Speed, average speed and acceleration of one moving object (a rolling ball, a cart); compares with "
             "g sin(theta) for a ramp.",
             ("speed", "average_speed", "acceleration"), "one",
             "a known length for the scale; the ramp angle; a time for 'speed at ...'",
             ("How fast is the ball going?", "What is the acceleration down the 20 degree ramp?")),
    ToolSpec("collision_momentum", True,
             "Momentum before and after one or more collisions between several objects (e.g. Newton's cradle, carts).",
             ("momentum_change", "momentum"), "several",
             "the mass of the objects (if equal) and a known length for the scale",
             ("Is momentum conserved in the collision?", "How much momentum is lost in each impact?")),
    ToolSpec("count_objects", True,
             "Count the distinct objects in the clip (parts on a tray, balls, ...).", ("count",), "several",
             "nothing", ("How many balls are there?", "How many parts are in the video?")),
    ToolSpec("line_crossings", True,
             "Count objects crossing a line (by direction) and the rate per minute, e.g. parts on a belt.",
             ("crossings", "rate_per_minute"), "several",
             "where the line is (default: the middle of the picture) and whether everything moves together (a belt)",
             ("How many parts crossed the line?", "How many parts per minute pass the middle?")),
    ToolSpec("growth_over_time", False, "Growth of something over a time-lapse (a plant, dough).", ("growth",),
             "one", "", ()),
    ToolSpec("rotation_rate", False, "Rotation speed (rpm) of a fan or wheel with a marker.", ("rpm",), "one", "", ()),
)}

AVAILABLE = {k: v for k, v in TOOLS.items() if v.available}

# Real-world sizes the user can name instead of measuring: numbers come from THIS table, never from the model.
REFERENCES = {
    "a4_paper": {"label": "A4 paper (long side)", "length_m": 0.297},
    "us_letter_paper": {"label": "US letter paper (long side)", "length_m": 0.2794},
    "credit_card": {"label": "a credit or bank card (long side)", "length_m": 0.0856},
    "metre_stick": {"label": "a metre stick", "length_m": 1.0},
}

# names of values the user may state, and the question asked when one is missing
VALUE_NAMES = ("string_length", "mass", "spring_mass", "reference_length", "ramp_angle", "time")
# the names a caller of the API may use for values, mapped to ours
KNOWN_VALUE_KEYS = {"string_length_m": "string_length", "mass_kg": "mass", "spring_mass_kg": "spring_mass",
                    "reference_length_m": "reference_length", "ramp_angle_deg": "ramp_angle", "time_s": "time"}
ASKS = {
    "string_length": ("What is the length of the string, from the pivot to the centre of the bob? "
                      "(for example 0.5 m or 50 cm)", "length"),
    "mass": ("What is the mass of the object? (for example 200 g or 0.2 kg)", "mass"),
    "reference_length": ("How long is something you can see in the picture, in the same plane as the motion "
                         "(a ruler, a metre stick, a sheet of A4 paper)? Gauge uses it to turn pixels into metres.",
                         "length"),
    "ramp_angle": ("What is the angle of the ramp? (for example 20 degrees)", "angle"),
    "object": ("What should Gauge measure? Describe the object in a few words (for example 'red ball').", None),
}


@dataclass(frozen=True)
class MissingInput:
    name: str
    question: str
    dimension: str | None


def _ask(name):
    q, dim = ASKS[name]
    return MissingInput(name, q, dim)


def missing_inputs(tool, outputs, values, compare_with_ramp=False):
    """The inputs this measurement cannot do without that the user has not given. `values` has the names given so far."""
    out = []
    if tool == "pendulum_period" and "g" in outputs and "string_length" not in values:
        out.append(_ask("string_length"))
    elif tool == "oscillation_frequency" and "spring_constant" in outputs and "mass" not in values:
        out.append(_ask("mass"))
    elif tool == "free_fall" and "reference_length" not in values:
        out.append(_ask("reference_length"))
    elif tool == "velocity_acceleration":
        if "reference_length" not in values:
            out.append(_ask("reference_length"))
        if compare_with_ramp and "ramp_angle" not in values:
            out.append(_ask("ramp_angle"))
    elif tool == "collision_momentum" and "momentum" in outputs and "momentum_change" not in outputs:
        if "mass" not in values:
            out.append(_ask("mass"))
        if "reference_length" not in values:
            out.append(_ask("reference_length"))
    return out


def supported_summary():
    """A plain list of what Gauge can measure, for answering an unsupported question honestly."""
    return [f"{t.summary}" for t in AVAILABLE.values()]
