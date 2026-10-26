"""Units and number words: the model reports what the user SAID, code does the arithmetic.

A language model must not convert units or compute anything we then treat as a measurement. So the planner
returns the value exactly as stated ("50", "cm") plus the words it came from, and this module converts it, checks
the units fit the quantity, checks the value is sane, and (where it can) re-reads the number from the user's own
words to confirm the model got it right.
"""
import math
import re

# metres, kilograms, degrees, seconds
LENGTH = {"m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0, "cm": 0.01, "centimetre": 0.01,
          "centimetres": 0.01, "centimeter": 0.01, "centimeters": 0.01, "mm": 0.001, "millimetre": 0.001,
          "millimetres": 0.001, "millimeter": 0.001, "millimeters": 0.001, "km": 1000.0, "in": 0.0254, "inch": 0.0254,
          "inches": 0.0254, "ft": 0.3048, "foot": 0.3048, "feet": 0.3048}
MASS = {"kg": 1.0, "kilogram": 1.0, "kilograms": 1.0, "kilo": 1.0, "kilos": 1.0, "g": 0.001, "gram": 0.001,
        "grams": 0.001, "mg": 1e-6, "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
        "oz": 0.028349523, "ounce": 0.028349523, "ounces": 0.028349523}
ANGLE = {"deg": 1.0, "degree": 1.0, "degrees": 1.0, "°": 1.0, "rad": 180 / math.pi, "radian": 180 / math.pi,
         "radians": 180 / math.pi}
TIME = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0, "ms": 0.001, "min": 60.0, "minute": 60.0,
        "minutes": 60.0}
UNIT_TABLES = {"length": LENGTH, "mass": MASS, "angle": ANGLE, "time": TIME}
SI_NAME = {"length": "m", "mass": "kg", "angle": "degrees", "time": "s"}

# what each named value is, and a sane range in SI units (outside it we ask the user to confirm)
QUANTITIES = {
    "string_length": ("length", 0.01, 30.0),
    "mass": ("mass", 1e-4, 1e4),
    "spring_mass": ("mass", 1e-4, 1e3),
    "reference_length": ("length", 0.005, 100.0),
    "ramp_angle": ("angle", 0.5, 89.0),
    "time": ("time", 0.0, 3600.0),
}


class UnitError(ValueError):
    pass


def convert(name, value, unit):
    """(SI value, SI unit name) for a named quantity; raises UnitError for an unknown or wrong-kind unit."""
    if name not in QUANTITIES:
        raise UnitError(f"unknown quantity '{name}'")
    kind = QUANTITIES[name][0]
    key = (unit or "").strip().lower().rstrip(".")
    if key in ("", "unitless") and kind == "angle":
        key = "deg"
    table = UNIT_TABLES[kind]
    if key not in table:
        other = next((k for k, t in UNIT_TABLES.items() if key in t), None)
        hint = f" (that is a unit of {other}, not {kind})" if other else ""
        raise UnitError(f"'{unit}' is not a {kind} unit{hint}")
    return float(value) * table[key], SI_NAME[kind]


def plausible(name, si_value):
    """True if the SI value is in a sane range for this quantity."""
    _, lo, hi = QUANTITIES[name]
    return lo <= si_value <= hi


# ---------------------------------------------------------------- number words

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_FRACTIONS = {"half": 0.5, "halves": 0.5, "quarter": 0.25, "quarters": 0.25, "third": 1 / 3, "thirds": 1 / 3,
              "tenth": 0.1, "tenths": 0.1}


def _integer_words(tokens, i):
    """Parse 'twenty five', 'one hundred and twenty' ... starting at tokens[i]; returns (value, next index) or None."""
    total, current, used, j = 0, 0, False, i
    while j < len(tokens):
        w = tokens[j]
        if w in _ONES:
            current += _ONES[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred" and (used or current):
            current = (current or 1) * 100
        elif w == "and" and used and j + 1 < len(tokens) and (tokens[j + 1] in _ONES or tokens[j + 1] in _TENS):
            j += 1
            continue
        else:
            break
        used = True
        j += 1
    if not used:
        return None
    return total + current, j


def words_to_number(text):
    """The first number written in `text`, in digits or words ('0.5', 'half a metre', 'two and a half', 'point five').

    Returns a float or None. Used only to cross-check what the model extracted, never to invent a value.
    """
    tokens = re.findall(r"\d+(?:\.\d+)?|\.\d+|[a-z]+", text.lower().replace("-", " "))
    for i, w in enumerate(tokens):
        if re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", w):
            return float(w)
        if w == "point" and i + 1 < len(tokens):  # 'point five', 'one point two five'
            digits = ""
            for d in tokens[i + 1:]:
                if d in _ONES and _ONES[d] < 10:
                    digits += str(_ONES[d])
                else:
                    break
            if digits:
                return float("0." + digits)
        if w in _FRACTIONS or (w == "a" and i + 1 < len(tokens) and tokens[i + 1] in _FRACTIONS):
            return _FRACTIONS[w if w in _FRACTIONS else tokens[i + 1]]
        got = _integer_words(tokens, i)
        if got:
            value, j = got
            rest = tokens[j:j + 4]
            if rest[:1] == ["point"]:
                digits = ""
                for d in rest[1:]:
                    if d in _ONES and _ONES[d] < 10:
                        digits += str(_ONES[d])
                    else:
                        break
                if digits:
                    return float(f"{value}.{digits}")
            if rest[:3] in (["and", "a", "half"], ["and", "a", "quarter"]):
                return value + _FRACTIONS[rest[2]]
            if rest[:2] == ["and", "half"]:
                return value + 0.5
            return float(value)
    return None


def verify_quote(question, quote, value):
    """Is `quote` really in the question, and does its number equal `value`? -> (found, verified).

    found: the quoted words appear in the question. verified: the number read from those words equals the value
    the model gave. found-but-not-verified values are shown to the user to confirm.
    """
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()  # noqa: E731
    found = bool(quote) and norm(quote) in norm(question)
    if not found:
        return False, False
    read = words_to_number(quote)
    return True, read is not None and abs(read - float(value)) <= 1e-9 * max(1.0, abs(read))
