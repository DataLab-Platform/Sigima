# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
CSV Labels & Units Parsing Unit Test
=====================================

Unit tests for ``get_labels_units_from_dataframe``, ``normalize_units``,
``_normalize_whitespace`` and ``_split_label_unit``, focusing on:

- Trimming leading and trailing whitespace in CSV column titles
- Edge cases for unit extraction from parenthesized suffixes
- Nested parentheses in units
- Unit normalization (middle dots, spaces between units, operators)
- Exotic whitespace (tabs, non-breaking spaces)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import pandas as pd

from sigima.io.signal.funcs import (
    _normalize_whitespace,
    _split_label_unit,
    get_labels_units_from_dataframe,
    normalize_units,
    read_csv,
)
from sigima.tests.env import execenv
from sigima.tests.helpers import get_test_fnames

# ---------------------------------------------------------------------------
#  _normalize_whitespace  –  isolated tests
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    """Tests for the _normalize_whitespace helper."""

    def test_leading_trailing_spaces(self) -> None:
        """Leading and trailing spaces are removed."""
        assert _normalize_whitespace("  hello  ") == "hello"

    def test_tabs(self) -> None:
        """Tabs are removed."""
        assert _normalize_whitespace("\tWavelength (nm)\t") == "Wavelength (nm)"

    def test_non_breaking_space(self) -> None:
        """Non-breaking spaces are removed."""
        assert _normalize_whitespace("\xa0Signal\xa0") == "Signal"

    def test_mixed_whitespace(self) -> None:
        """Mixed whitespace characters are all removed."""
        assert _normalize_whitespace(" \t\xa0Label (V)\xa0\t ") == "Label (V)"

    def test_empty_string(self) -> None:
        """Empty string remains empty."""
        assert _normalize_whitespace("") == ""

    def test_only_spaces(self) -> None:
        """String with only spaces becomes empty."""
        assert _normalize_whitespace("   ") == ""

    def test_internal_spaces_preserved(self) -> None:
        """Spaces inside the label should not be collapsed."""
        assert _normalize_whitespace("  My Label (nm)  ") == "My Label (nm)"


# ---------------------------------------------------------------------------
#  _split_label_unit  –  isolated tests
# ---------------------------------------------------------------------------


class TestSplitLabelUnit:
    """Tests for the _split_label_unit helper."""

    def test_simple(self) -> None:
        """Basic case: label and unit separated by ' ('."""
        assert _split_label_unit("Wavelength (nm)") == ("Wavelength", "nm")

    def test_no_parentheses(self) -> None:
        """No parentheses → no unit extracted."""
        assert _split_label_unit("Voltage") == ("Voltage", "")

    def test_nested_parentheses(self) -> None:
        """First ' (' is used: nested parens stay inside the unit."""
        assert _split_label_unit("NestedParen (a.u. (norm))") == (
            "NestedParen",
            "a.u. (norm)",
        )

    def test_empty_parentheses(self) -> None:
        """Empty parentheses produce an empty unit string."""
        assert _split_label_unit("Signal ()") == ("Signal", "")

    def test_no_space_before_paren(self) -> None:
        """No space before '(' → no unit extracted."""
        assert _split_label_unit("Label(nm)") == ("Label(nm)", "")

    def test_only_paren_no_label(self) -> None:
        """Edge case: ' (unit)' with no label before the space."""
        label, unit = _split_label_unit(" (nm)")
        # After strip the input passed here would just be "(nm)" — no space before (
        # But _normalize_whitespace is called *before* _split_label_unit, so this
        # would actually arrive as "(nm)".
        # Direct call: " (nm)" has a space at position 0 → label="", unit="nm"
        assert label == ""
        assert unit == "nm"

    def test_multiple_paren_groups(self) -> None:
        """Multiple ' (' groups: the first one wins."""
        assert _split_label_unit("CH1 (filtered) (V)") == (
            "CH1",
            "filtered) (V",
        )

    def test_trailing_paren_missing(self) -> None:
        """No closing ')' at end → no unit."""
        assert _split_label_unit("Label (nm") == ("Label (nm", "")

    def test_unit_with_special_chars(self) -> None:
        """Units with special characters like degree symbol."""
        assert _split_label_unit("Temperature (°C)") == ("Temperature", "°C")

    def test_unit_with_percent(self) -> None:
        """Units with percent symbol."""
        assert _split_label_unit("Reflectance (%)") == ("Reflectance", "%")

    def test_unit_with_number(self) -> None:
        """Units that contain numbers, like 'cm-1'."""
        assert _split_label_unit("Intensity (cm-1)") == ("Intensity", "cm-1")


# ---------------------------------------------------------------------------
#  normalize_units  –  isolated tests
# ---------------------------------------------------------------------------


class TestNormalizeUnits:
    """Tests for the normalize_units function."""

    def test_middle_dot(self) -> None:
        """Middle dot (U+00B7) replaced by '*'."""
        assert normalize_units("kg·m") == "kg*m"

    def test_unicode_dot(self) -> None:
        """Unicode dot operator (U+22C5) replaced by '*'."""
        assert normalize_units("kg⋅m") == "kg*m"

    def test_spaces_between_units(self) -> None:
        """Spaces between alphabetic tokens become '*'."""
        assert normalize_units("kg m") == "kg*m"

    def test_spaces_around_slash(self) -> None:
        """Spaces around '/' are removed."""
        assert normalize_units("kg / s") == "kg/s"

    def test_spaces_around_caret(self) -> None:
        """Spaces around '^' are removed."""
        assert normalize_units("m ^ 2") == "m^2"

    def test_combined_complex(self) -> None:
        """Complex unit string with middle dot, spaces and slash."""
        assert normalize_units("kg·m / s²") == "kg*m/s²"

    def test_leading_trailing_spaces(self) -> None:
        """Leading/trailing spaces are stripped."""
        assert normalize_units("  nm  ") == "nm"

    def test_multiple_spaces(self) -> None:
        """Multiple spaces between units become a single '*'."""
        assert normalize_units("kg   m") == "kg*m"

    def test_already_normalized(self) -> None:
        """Already normalized string is returned unchanged."""
        assert normalize_units("kg*m/s^2") == "kg*m/s^2"

    def test_empty_string(self) -> None:
        """Empty input returns empty output."""
        assert normalize_units("") == ""

    def test_single_unit(self) -> None:
        """Single unit without operators."""
        assert normalize_units("nm") == "nm"

    def test_spaces_around_star(self) -> None:
        """Spaces around '*' are removed."""
        assert normalize_units("kg * m") == "kg*m"


# ---------------------------------------------------------------------------
#  get_labels_units_from_dataframe  –  whitespace trimming
# ---------------------------------------------------------------------------


class TestGetLabelsUnitsWhitespace:
    """Whitespace trimming in column headers."""

    def test_leading_trailing_spaces_with_units(self) -> None:
        """Spaces around 'Label (unit)' are trimmed from both label and unit."""
        df = pd.DataFrame({"  Wavelength (nm)  ": [1.0], "  Intensity (a.u.)  ": [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "Wavelength"
        assert xunit == "nm"
        assert ylabels == ["Intensity"]
        assert yunits == ["a.u."]

    def test_leading_trailing_spaces_without_units(self) -> None:
        """Plain labels with surrounding spaces are trimmed."""
        df = pd.DataFrame({"  Time  ": [1.0], "  Signal  ": [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "Time"
        assert ylabels == ["Signal"]
        assert xunit == ""
        assert yunits == [""]

    def test_leading_space_only(self) -> None:
        """Leading-only whitespace is trimmed."""
        df = pd.DataFrame({"  Frequency (Hz)": [1.0], "  Power (W)": [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "Frequency"
        assert xunit == "Hz"
        assert ylabels == ["Power"]
        assert yunits == ["W"]

    def test_trailing_space_only(self) -> None:
        """Trailing-only whitespace is trimmed."""
        df = pd.DataFrame({"Frequency (Hz)  ": [1.0], "Power (W)  ": [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "Frequency"
        assert xunit == "Hz"
        assert ylabels == ["Power"]
        assert yunits == ["W"]

    def test_tab_whitespace(self) -> None:
        """Tab characters are also trimmed."""
        df = pd.DataFrame({"\tTime (s)\t": [1.0], "\tVoltage (V)\t": [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "Time"
        assert xunit == "s"
        assert ylabels == ["Voltage"]
        assert yunits == ["V"]

    def test_non_breaking_space(self) -> None:
        """Non-breaking space (U+00A0) is treated as whitespace."""
        df = pd.DataFrame({"\xa0Time (s)\xa0": [1.0], "\xa0Voltage (V)\xa0": [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "Time"
        assert xunit == "s"
        assert ylabels == ["Voltage"]
        assert yunits == ["V"]

    def test_no_whitespace(self) -> None:
        """Clean headers without extra whitespace (baseline)."""
        df = pd.DataFrame({"Wavelength (nm)": [1.0], "Intensity (a.u.)": [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "Wavelength"
        assert xunit == "nm"
        assert ylabels == ["Intensity"]
        assert yunits == ["a.u."]

    def test_multiple_y_columns_with_spaces(self) -> None:
        """All Y columns get their whitespace trimmed."""
        df = pd.DataFrame(
            {
                "  X (m)  ": [1.0],
                "  Y1 (V)  ": [2.0],
                "  Y2 (A)  ": [3.0],
                "  Y3  ": [4.0],
            }
        )
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "X"
        assert xunit == "m"
        assert ylabels == ["Y1", "Y2", "Y3"]
        assert yunits == ["V", "A", ""]

    def test_only_spaces_label(self) -> None:
        """Column header with only spaces becomes empty string."""
        df = pd.DataFrame({"   ": [1.0], "Y": [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == ""
        assert xunit == ""
        assert ylabels == ["Y"]
        assert yunits == [""]


# ---------------------------------------------------------------------------
#  get_labels_units_from_dataframe  –  unit parsing edge cases
# ---------------------------------------------------------------------------


class TestGetLabelsUnitsEdgeCases:
    """Edge cases for parenthesized unit extraction."""

    def test_complex_units_with_middle_dot(self) -> None:
        """Units containing middle dot are normalized."""
        df = pd.DataFrame({"Force (kg·m/s²)": [1.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "Force"
        assert xunit == "kg*m/s²"

    def test_units_with_spaces(self) -> None:
        """Spaces inside unit parentheses are normalized."""
        df = pd.DataFrame({"Pressure (kg / m^2)": [1.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "Pressure"
        assert xunit == "kg/m^2"

    def test_units_with_caret(self) -> None:
        """Exponent (^) in units."""
        df = pd.DataFrame({"Area (m^2)": [1.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "Area"
        assert xunit == "m^2"

    def test_nested_parentheses(self) -> None:
        """Nested parentheses: first ' (' delimits label from unit."""
        df = pd.DataFrame({"NestedParen (a.u. (norm))": [1.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "NestedParen"
        assert xunit == "a.u. (norm)"

    def test_empty_parentheses(self) -> None:
        """Empty parentheses '()' – no unit extracted."""
        df = pd.DataFrame({"Signal ()": [1.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "Signal"
        assert xunit == ""

    def test_no_parentheses(self) -> None:
        """No parentheses at all – no unit."""
        df = pd.DataFrame({"Voltage": [1.0], "Current": [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "Voltage"
        assert xunit == ""
        assert ylabels == ["Current"]
        assert yunits == [""]

    def test_parentheses_in_label_name(self) -> None:
        """Parenthesized text at end is treated as unit."""
        df = pd.DataFrame({"Signal (raw)": [1.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "Signal"
        assert xunit == "raw"

    def test_multiple_parenthesized_groups(self) -> None:
        """Multiple ' (' groups: first ' (' is the split point."""
        df = pd.DataFrame({"CH1 (filtered) (V)": [1.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "CH1"
        # Everything between first ' (' and last ')' is captured
        assert xunit == "filtered) (V"

    def test_unit_with_special_chars(self) -> None:
        """Units with degree symbol and superscript."""
        df = pd.DataFrame({"Temperature (°C)": [25.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "Temperature"
        assert xunit == "°C"

    def test_unit_with_percent(self) -> None:
        """Percent sign as unit."""
        df = pd.DataFrame({"Reflectance (%)": [50.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "Reflectance"
        assert xunit == "%"

    def test_unit_with_number(self) -> None:
        """Unit containing numeric characters."""
        df = pd.DataFrame({"Intensity (cm-1)": [1.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "Intensity"
        assert xunit == "cm-1"

    def test_no_space_before_paren(self) -> None:
        """No space before '(' → no unit extracted."""
        df = pd.DataFrame({"Label(nm)": [1.0]})
        xlabel, _, xunit, _ = get_labels_units_from_dataframe(df)
        assert xlabel == "Label(nm)"
        assert xunit == ""

    def test_single_column_dataframe(self) -> None:
        """DataFrame with only one column (X only, no Y)."""
        df = pd.DataFrame({"  Time (s)  ": [1.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "Time"
        assert xunit == "s"
        assert ylabels == []
        assert yunits == []

    def test_numeric_column_names(self) -> None:
        """Integer column names (no header in CSV)."""
        df = pd.DataFrame({0: [1.0], 1: [2.0]})
        xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
        assert xlabel == "0"
        assert ylabels == ["1"]
        assert xunit == ""
        assert yunits == [""]


# ---------------------------------------------------------------------------
#  Integration: read the whitespace_units.csv test file
# ---------------------------------------------------------------------------


def test_read_whitespace_units_csv() -> None:
    """End-to-end: read CSV file with whitespace in headers and verify parsing."""
    execenv.print("Testing whitespace/units CSV file end-to-end ...")

    filenames = get_test_fnames("whitespace_units.csv", in_folder="curve_formats")
    assert len(filenames) > 0, "whitespace_units.csv test file not found"

    csv_data = read_csv(filenames[0])

    # X column: "  Wavelength (nm)  "
    assert csv_data.xlabel is not None
    execenv.print(f"  xlabel={csv_data.xlabel!r}, xunit={csv_data.xunit!r}")
    assert csv_data.xlabel == "Wavelength"
    assert csv_data.xunit == "nm"

    # Y columns should all be trimmed
    assert csv_data.ylabels is not None
    assert csv_data.yunits is not None
    execenv.print(f"  ylabels={csv_data.ylabels}")
    execenv.print(f"  yunits={csv_data.yunits}")

    # Check data shape: 11 rows x 11 columns
    assert csv_data.xydata.shape == (11, 11)

    # Check all Y labels and units are properly trimmed
    for label in csv_data.ylabels:
        assert label == label.strip(), f"Y label not trimmed: {label!r}"
    for unit in csv_data.yunits:
        assert unit == unit.strip(), f"Y unit not trimmed: {unit!r}"

    # Verify specific columns
    # "  Padded NoUnit  " → label="Padded NoUnit", unit=""
    padded_idx = csv_data.ylabels.index("Padded NoUnit")
    assert csv_data.yunits[padded_idx] == ""

    # " NestedParen (a.u. (norm)) " → label="NestedParen", unit="a.u. (norm)"
    nested_idx = csv_data.ylabels.index("NestedParen")
    assert csv_data.yunits[nested_idx] == "a.u. (norm)"
