"""Number formatting and the number check.

The rule of the whole project: numbers come from code, never from a language model. Two parts make that real:

1. format_measurement: code decides how a value and its uncertainty are written ("1.419 +/- 0.002 s"),
   so the answer card and the fallback text never depend on a model.
2. find_unsupported_numbers: when a model writes an explanation, every number in it must trace back to
   a number the code supplied. Anything else is reported and the text is rejected.
"""
import math
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

_UNIT_EXPONENT = re.compile(r"(?<=[A-Za-zµ°/·])\s?(?:\^\s?-?\d+|[⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"(?<![\w.])[−\-+]?\d+(?:\.\d+)?(?:[eE][+\-−]?\d+)?")
_NAMES = re.compile(r"SAM\s*3\.1|Muse\s*Spark(?:\s*1\.\d)?|Savitzky|2\s*sigma|3\s*sigma", re.IGNORECASE)
_LIST_MARKER = re.compile(r"(?m)^\s*\(?\d{1,2}[.)]\s+")


def significant_decimals(x, sig=2):
    """Number of decimals that shows x with `sig` significant digits (can be negative: round to tens)."""
    if x == 0 or not math.isfinite(x):
        return 0
    return sig - 1 - int(math.floor(math.log10(abs(x))))


def _round_to(x, decimals):
    q = Decimal(1).scaleb(-decimals)
    if decimals >= 0:
        return Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP).quantize(Decimal(1))


def format_number(x, decimals=None, sig=4):
    """A plain-notation string for x. By default `sig` significant digits."""
    if x is None or not math.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    d = significant_decimals(x, sig) if decimals is None else decimals
    s = format(_round_to(x, d), "f")
    if decimals is None and "." in s:
        s = s.rstrip("0").rstrip(".")  # default precision: no padding zeros (7, not 7.000)
    if float(s) == 0:
        s = s.lstrip("-")  # a value that rounds to zero is zero, not "-0.0"
    return "0" if decimals is None and float(s) == 0 else s


def _unit_text(units):
    return (units or "").replace("^2", "²").replace("^3", "³").replace("^-1", "⁻¹")


def format_measurement(value, uncertainty=None, units="", sig=4):
    """'1.419 ± 0.002 s': the uncertainty to one significant digit (two if it starts with 1), the value to match.

    Without an uncertainty the value is shown with `sig` significant digits (4 by default).
    """
    u = _unit_text(units)
    if value is None or not math.isfinite(value):
        return "n/a"
    suffix = f" {u}" if u and not u.startswith("(") else ""
    if uncertainty is None or not math.isfinite(uncertainty) or uncertainty <= 0:
        return f"{format_number(value, sig=sig)}{suffix}"
    first = int(f"{abs(uncertainty):e}"[0])
    d = significant_decimals(uncertainty, 2 if first == 1 else 1)
    return f"{format_number(value, d)} ± {format_number(uncertainty, d)}{suffix}"


# ---------------------------------------------------------------- the number check

def number_tokens(text):
    """Every number written in `text`, as (string, float, decimals, significant digits). Units' exponents are ignored."""
    t = _NAMES.sub(" ", text)
    t = _UNIT_EXPONENT.sub("", t)
    t = _LIST_MARKER.sub("", t)
    t = _THOUSANDS.sub("", t)
    out = []
    for m in _NUMBER.finditer(t):
        raw = m.group(0)
        norm = raw.replace("−", "-").lstrip("+")
        try:
            d = Decimal(norm.lstrip("-"))
        except InvalidOperation:
            continue
        digits = norm.lstrip("-").split("e")[0].split("E")[0]
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        sig = len(digits.replace(".", "").lstrip("0")) or 1
        out.append((raw, float(d), decimals, sig))
    return out


class AllowedNumbers:
    """The numbers a model may write: every number the code supplied, as text or as a raw value."""

    def __init__(self):
        self.values = []  # floats appearing in text we wrote (compared as values: '0.50' equals '0.5')
        self.raws = []  # raw floats from results (compared after rounding to what the model wrote)

    def add_text(self, text):
        for _, v, _, _ in number_tokens(str(text)):
            self.values.append(v)
        return self

    def add_value(self, x):
        if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
            self.raws.append(abs(float(x)))
        return self

    def add_any(self, obj):
        """Add every number and every string inside a nested structure."""
        if isinstance(obj, dict):
            for v in obj.values():
                self.add_any(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self.add_any(v)
        elif isinstance(obj, str):
            self.add_text(obj)
        else:
            self.add_value(obj)
        return self

    def supports(self, value, decimals, sig):
        v = abs(value)
        if any(abs(v - a) <= 1e-12 * max(1.0, v) for a in self.values + self.raws):
            return True  # exactly a number the code supplied (as text or as a raw value, such as a count of 7)
        if sig >= 2 and v != 0:  # two or more digits are specific enough to compare with a rounded raw number
            for a in self.raws:
                if a != 0 and abs(round(a, decimals) - v) <= 1e-9 * max(1.0, v):
                    return True
        if v == 0:
            return any(a == 0 for a in self.raws) or any(x == 0 for x in self.values)
        return False


def find_unsupported_numbers(text, allowed):
    """Numbers in `text` that the code never supplied. Empty list means the text is clean."""
    bad = []
    for raw, value, decimals, sig in number_tokens(text):
        if not allowed.supports(value, decimals, sig):
            bad.append(raw)
    return bad
