# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for format_legend_value (sigima.objects.scalar.common).
"""

from __future__ import annotations

import pytest

from sigima.objects.scalar.common import format_legend_value

# ===================================================================
# 1. Plain display: values whose string representation is ≤ 6 chars
# ===================================================================


class TestPlainDisplay:
    """Values that fit within 6 characters are returned as-is."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, "0"),
            (42, "42"),
            (-1, "-1"),
            (123456, "123456"),  # exactly 6 chars
            (-12345, "-12345"),  # exactly 6 chars (sign counts)
        ],
    )
    def test_integers_within_limit(self, value, expected) -> None:
        """Integers that fit within 6 chars should be returned as-is."""
        assert format_legend_value(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.5, "0.5"),
            (1.5, "1.5"),
            (-1.5, "-1.5"),
            (3.14, "3.14"),
            (0.001, "0.001"),  # 5 chars
        ],
    )
    def test_floats_within_limit(self, value, expected) -> None:
        """Floats that fit within 6 chars should be returned as-is."""
        assert format_legend_value(value) == expected

    def test_float_with_integer_value(self) -> None:
        """A float like 42.0 should display as '42' (integer form)."""
        assert format_legend_value(42.0) == "42"

    def test_zero_float(self) -> None:
        """A float like 0.0 should display as '0'."""
        assert format_legend_value(0.0) == "0"


# ===================================================================
# 2. Scientific display: plain > 6 chars, scientific ≤ 12 chars
# ===================================================================


class TestExactScientificDisplay:
    """Values that exceed 6 chars in plain form but fit within 12 chars
    in exact scientific notation."""

    def test_large_integer(self) -> None:
        """1234567 → plain '1234567' (7 chars > 6) → scientific."""
        result = format_legend_value(1234567)
        assert "e" in result or "E" in result
        assert len(result) <= 12

    def test_small_float(self) -> None:
        """0.000001 → plain '1e-06' (5 chars ≤ 6) → stays plain.
        But 0.0000012 → plain '1.2e-06' via repr which is 7 chars > 6."""
        result = format_legend_value(1.2e-06)
        assert "e" in result
        assert len(result) <= 12

    def test_negative_scientific(self) -> None:
        """Test that negative values also switch to scientific if plain is > 6 chars."""
        result = format_legend_value(-1234567)
        assert "e" in result
        assert len(result) <= 12

    @pytest.mark.parametrize(
        "value",
        [
            1.5e-6,
            1.0e10,
            -9.99e8,
            1.23456e6,
        ],
    )
    def test_various_scientific_values(self, value) -> None:
        """
        Test a variety of values that should switch to scientific but fit within
        12 chars.
        """
        result = format_legend_value(value)
        assert len(result) <= 12


# ===================================================================
# 3. Rounded scientific: scientific > 12 chars
# ===================================================================


class TestRoundedScientificDisplay:
    """Values whose exact scientific notation exceeds 12 chars get rounded
    to fit within 12 chars."""

    def test_many_significant_digits(self) -> None:
        """A value with many significant digits should be rounded."""
        # 1.23456789012345e-10 has a very long scientific notation
        result = format_legend_value(1.23456789012345e-10)
        assert "e" in result
        assert len(result) <= 12

    def test_negative_many_digits(self) -> None:
        """Negative value with many significant digits should also be rounded."""
        result = format_legend_value(-1.23456789012345e20)
        assert "e" in result
        assert len(result) <= 12

    def test_rounded_value_fits_limit(self) -> None:
        """Rounded scientific value should fit within 12 chars."""
        val = 1.234567890123e-100
        result = format_legend_value(val)
        assert "e" in result
        assert len(result) <= 12


# ===================================================================
# 4. Special values
# ===================================================================


class TestSpecialValues:
    """NaN, inf, -inf are handled gracefully."""

    def test_nan(self) -> None:
        """NaN should be displayed as 'nan'."""
        result = format_legend_value(float("nan"))
        assert result == "nan"

    def test_inf(self) -> None:
        """Positive infinity should be displayed as 'inf'."""
        result = format_legend_value(float("inf"))
        assert result == "inf"

    def test_negative_inf(self) -> None:
        """Negative infinity should be displayed as '-inf'."""
        result = format_legend_value(float("-inf"))
        assert result == "-inf"


# ===================================================================
# 5. Boundary conditions
# ===================================================================


class TestBoundaryConditions:
    """Test exact boundary conditions for the 6-char and 12-char limits."""

    def test_exactly_6_chars_plain(self) -> None:
        """A value with exactly 6 chars plain representation stays plain."""
        # 123456 as int → "123456" → 6 chars
        assert format_legend_value(123456) == "123456"
        assert len(format_legend_value(123456)) == 6

    def test_7_chars_plain_switches_to_scientific(self) -> None:
        """A value with 7 chars plain switches to scientific."""
        result = format_legend_value(1234567)
        assert len("1234567") == 7  # plain would be 7 chars
        assert "e" in result  # switched to scientific

    def test_exactly_12_chars_scientific(self) -> None:
        """A scientific string with exactly 12 chars is displayed as-is."""
        # 1.234567e+07 → exact sci repr is "1.234567e+07" → 12 chars
        val = 1.234567e07
        result = format_legend_value(val)
        assert len(result) <= 12

    def test_scientific_exceeding_12_gets_rounded(self) -> None:
        """A scientific string > 12 chars gets rounded to fit."""
        # Value with many significant digits
        val = 1.23456789012e07
        result = format_legend_value(val)
        assert len(result) <= 12
