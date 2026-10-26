"""Tests for number formatting and the number check (no model involved)."""
import pytest

from app.formatting import (AllowedNumbers, find_unsupported_numbers, format_measurement, format_number,
                            number_tokens, significant_decimals)


def test_measurement_uses_one_significant_digit_of_the_uncertainty():
    assert format_measurement(1.4192, 0.0025, "s") == "1.419 ± 0.003 s"
    assert format_measurement(9.6643, 0.0432, "m/s^2") == "9.66 ± 0.04 m/s²"
    assert format_measurement(1058.3, 12.4, "px/s") == "1058 ± 12 px/s"


def test_an_uncertainty_starting_with_1_keeps_two_digits():
    assert format_measurement(0.8582, 0.0016, "s") == "0.8582 ± 0.0016 s"
    assert format_measurement(1.4192, 0.0016, "s") == "1.4192 ± 0.0016 s"
    assert format_measurement(120.04, 0.12, "per minute") == "120.04 ± 0.12 per minute"


def test_large_uncertainty_rounds_the_value_to_match():
    assert format_measurement(4.7, 5.3, "%") == "5 ± 5 %"
    assert format_measurement(1234.5, 80, "px/s^2") == "1230 ± 80 px/s²"


def test_no_uncertainty_shows_four_significant_digits():
    assert format_measurement(1.41921, None, "s") == "1.419 s"
    assert format_measurement(0.0, 0.0, "s") == "0 s"


def test_missing_values_are_not_made_up():
    assert format_measurement(None, 0.1, "s") == "n/a"
    assert format_measurement(float("nan"), None, "s") == "n/a"


def test_format_number_and_decimals():
    assert significant_decimals(0.0016, 1) == 3
    assert significant_decimals(1234, 2) == -2
    assert format_number(3.14159, 2) == "3.14"
    assert format_number(7) == "7"


def test_tokens_ignore_unit_exponents_and_list_markers_and_model_names():
    toks = [t[0] for t in number_tokens("g = 9.66 m/s² and 3 px/s^2, tracked with SAM 3.1\n1. First step")]
    assert toks == ["9.66", "3"]


def test_tokens_handle_signs_thousands_and_percent():
    toks = [t[1] for t in number_tokens("a change of −4.7% and 1,234.5 px, or +2")]
    assert toks == [4.7, 1234.5, 2.0]


def allowed(*texts, raws=()):
    a = AllowedNumbers()
    for t in texts:
        a.add_text(t)
    for r in raws:
        a.add_value(r)
    return a


def test_a_number_the_code_wrote_is_allowed_whatever_its_trailing_zeros():
    a = allowed("T = 1.419 ± 0.002 s", "you said 0.50 m")
    assert find_unsupported_numbers("The period is 1.419 s, give or take 0.002 s, and L is 0.5 m.", a) == []


def test_an_invented_number_is_caught():
    a = allowed("T = 1.419 ± 0.002 s")
    assert find_unsupported_numbers("The period is 1.419 s but g is 9.8 m/s².", a) == ["9.8"]


def test_a_rounded_raw_number_with_two_digits_is_allowed():
    a = allowed(raws=[1.41921, 0.0016])
    assert find_unsupported_numbers("About 1.42 s, with an error of 0.0016 s.", a) == []
    assert find_unsupported_numbers("About 1.43 s.", a) == ["1.43"]


def test_a_single_digit_is_not_matched_against_raw_numbers():
    a = allowed(raws=[0.6, 8.7])  # '1' and '9' could be rounded from these, but one digit is too loose to trust
    assert find_unsupported_numbers("About 9 cycles", a) == ["9"]
    b = allowed("8.7 cycles")
    assert find_unsupported_numbers("8.7 cycles", b) == []


def test_numbers_inside_a_nested_result_are_collected():
    a = AllowedNumbers().add_any({"value": 1.419, "extra": {"n": 7, "text": "tracked 362 of 362 frames"}, "ok": True})
    assert find_unsupported_numbers("1.419 s from 7 objects in 362 frames", a) == []
    assert find_unsupported_numbers("in 363 frames", a) == ["363"]


def test_the_sign_is_ignored_when_comparing():
    a = allowed("momentum changed by -4.7%")
    assert find_unsupported_numbers("a 4.7% drop", a) == []


def test_zero_needs_support_too():
    assert find_unsupported_numbers("0 crossings", allowed("0 crossings")) == []
    assert find_unsupported_numbers("0 crossings", allowed("12 crossings")) == ["0"]


def test_a_value_that_rounds_to_zero_has_no_minus_sign():
    assert format_measurement(-0.003, 0.8, "%") == "0.0 ± 0.8 %"
    assert format_measurement(-0.0004, 0.2, "s") == "0.0 ± 0.2 s"
    assert format_number(-0.001, 1) == "0.0"
    assert format_measurement(-4.7, 5.3, "%") == "-5 ± 5 %"  # a real negative keeps its sign


def test_derived_numbers_without_an_uncertainty_can_use_fewer_digits():
    assert format_measurement(0.10523, None, "%", sig=2) == "0.11 %"
    assert format_measurement(10.5517, None, "", sig=3) == "10.6"
