"""Tests for unit conversion and number words (no model involved)."""
import pytest

from app.units import UnitError, convert, plausible, verify_quote, words_to_number


@pytest.mark.parametrize("name,value,unit,expected", [
    ("string_length", 50, "cm", 0.5), ("string_length", 0.5, "m", 0.5), ("string_length", 500, "mm", 0.5),
    ("string_length", 20, "inches", 0.508), ("string_length", 2, "feet", 0.6096), ("string_length", 1, "metre", 1.0),
    ("mass", 200, "g", 0.2), ("mass", 0.2, "kilograms", 0.2), ("mass", 1, "lb", 0.45359237),
    ("ramp_angle", 20, "degrees", 20.0), ("ramp_angle", 20, "°", 20.0), ("ramp_angle", 0.5, "rad", 28.6478897565),
    ("time", 2500, "ms", 2.5), ("time", 1, "minute", 60.0),
])
def test_conversions(name, value, unit, expected):
    got, _ = convert(name, value, unit)
    assert got == pytest.approx(expected, rel=1e-6)


def test_the_wrong_kind_of_unit_is_refused_with_a_hint():
    with pytest.raises(UnitError, match="mass"):
        convert("string_length", 5, "kg")
    with pytest.raises(UnitError, match="not a length unit"):
        convert("string_length", 5, "furlongs")


def test_a_bare_angle_means_degrees():
    assert convert("ramp_angle", 30, "")[0] == 30.0


def test_sane_ranges():
    assert plausible("string_length", 0.5)
    assert not plausible("string_length", 500.0)  # a 500 m string: probably a unit slip, ask the user
    assert not plausible("ramp_angle", 120.0)
    assert plausible("mass", 0.2)


@pytest.mark.parametrize("text,expected", [
    ("0.5 m", 0.5), ("50 centimetres", 50.0), ("half a metre", 0.5), ("a quarter of a metre", 0.25),
    ("point five metres", 0.5), ("one point two five", 1.25), ("two and a half", 2.5), ("two hundred grams", 200.0),
    ("twenty five centimetres", 25.0), ("one hundred and twenty", 120.0), ("thirty degrees", 30.0),
    ("zero point three", 0.3), ("the string is 1.5 m long", 1.5), ("fifty", 50.0),
])
def test_numbers_in_words_and_digits(text, expected):
    assert words_to_number(text) == pytest.approx(expected)


def test_no_number_means_none():
    assert words_to_number("the string is long") is None


def test_a_quote_must_appear_in_the_question():
    q = "What value of g does that give? The string is half a metre."
    assert verify_quote(q, "half a metre", 0.5) == (True, True)
    assert verify_quote(q, "Half  a   Metre", 0.5) == (True, True)  # case and spacing do not matter
    assert verify_quote(q, "a metre and a half", 1.5) == (False, False)  # never said
    assert verify_quote(q, "", 0.5) == (False, False)


def test_a_wrong_number_is_found_but_not_verified():
    q = "The string is 0.15 m long"
    assert verify_quote(q, "0.15 m", 0.15) == (True, True)
    assert verify_quote(q, "0.15 m", 0.50) == (True, False)  # the model misheard / mis-extracted: the user must confirm


def test_a_phrase_without_a_readable_number_is_found_but_not_verified():
    assert verify_quote("the string is about a forearm long", "about a forearm", 0.3) == (True, False)
