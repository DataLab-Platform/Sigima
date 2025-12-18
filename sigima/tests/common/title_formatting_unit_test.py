# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Title formatting system unit tests

Testing the configurable title formatting system that allows different applications
(Sigima standalone vs DataLab integration) to use different title formatting strategies.

This test verifies:
  - SimpleTitleFormatter: Human-readable titles for standalone Sigima usage
  - PlaceholderTitleFormatter: DataLab-compatible placeholder titles
  - Configuration system: Setting and getting default formatters
  - Integration: Testing with computation functions from sigima.proc.base
  - Compatibility: Ensuring DataLab-style placeholder resolution works
"""

from __future__ import annotations

import pytest

from sigima import create_signal
from sigima.proc.base import dst_1_to_1, dst_2_to_1, dst_n_to_1, new_signal_result
from sigima.proc.title_formatting import (
    PlaceholderTitleFormatter,
    SimpleTitleFormatter,
    get_default_title_formatter,
    set_default_title_formatter,
)


class TestSimpleTitleFormatter:
    """Test suite for SimpleTitleFormatter class."""

    def test_1_to_1_operations(self):
        """Test SimpleTitleFormatter for 1-to-1 operations."""
        formatter = SimpleTitleFormatter()

        # Test basic function name formatting
        assert (
            formatter.format_1_to_1_title("gaussian_filter") == "Gaussian Filter Result"
        )

        # Test with suffix
        result = formatter.format_1_to_1_title("gaussian_filter", "sigma=1.5")
        assert result == "Gaussian Filter Result (sigma=1.5)"

    def test_n_to_1_operations(self):
        """Test SimpleTitleFormatter for n-to-1 operations."""
        formatter = SimpleTitleFormatter()

        # Test basic n-to-1 formatting
        result = formatter.format_n_to_1_title("add_signals", 3)
        assert result == "Add Signals of 3 Objects"

        # Test with suffix
        result = formatter.format_n_to_1_title("mean", 5, "weighted")
        assert result == "Mean of 5 Objects (weighted)"

    def test_2_to_1_operations(self):
        """Test SimpleTitleFormatter for 2-to-1 operations."""
        formatter = SimpleTitleFormatter()

        # Test basic 2-to-1 formatting
        assert formatter.format_2_to_1_title("subtract") == "Subtract Result"

        # Test with operator
        assert formatter.format_2_to_1_title("-") == "Binary Operation -"

        # Test with suffix
        result = formatter.format_2_to_1_title("divide", "method=euclidean")
        assert result == "Divide Result (method=euclidean)"

    def test_humanize_function_name(self):
        """Test the function name humanization logic."""
        formatter = SimpleTitleFormatter()

        # Test the internal logic through the public interface
        # The formatter converts snake_case to Title Case internally
        result = formatter.format_1_to_1_title("gaussian_filter")
        assert "Gaussian Filter" in result

        result = formatter.format_1_to_1_title("moving_average")
        assert "Moving Average" in result

        # Test single words
        result = formatter.format_1_to_1_title("normalize")
        assert "Normalize" in result

    def test_edge_cases(self):
        """Test edge cases for SimpleTitleFormatter."""
        formatter = SimpleTitleFormatter()

        # Test empty function name
        assert formatter.format_1_to_1_title("") == " Result"

        # Test None suffix (should be handled gracefully)
        result = formatter.format_1_to_1_title("test", None)
        assert result == "Test Result"

        # Test empty suffix
        result = formatter.format_1_to_1_title("test", "")
        assert result == "Test Result"


class TestPlaceholderTitleFormatter:
    """Test suite for PlaceholderTitleFormatter class."""

    def test_1_to_1_operations(self):
        """Test PlaceholderTitleFormatter for 1-to-1 operations."""
        formatter = PlaceholderTitleFormatter()

        # Test basic function name formatting
        assert formatter.format_1_to_1_title("wiener") == "wiener({0})"

        # Test with suffix
        result = formatter.format_1_to_1_title("gaussian_filter", "sigma=1.5")
        assert result == "gaussian_filter({0})|sigma=1.5"

    def test_n_to_1_operations(self):
        """Test PlaceholderTitleFormatter for n-to-1 operations."""
        formatter = PlaceholderTitleFormatter()

        # Test basic n-to-1 formatting
        result = formatter.format_n_to_1_title("sum", 3)
        assert result == "sum({0}, {1}, {2})"

        # Test with suffix
        result = formatter.format_n_to_1_title("mean", 5, "weighted=True")
        assert result == "mean({0}, {1}, {2}, {3}, {4})|weighted=True"

    def test_2_to_1_operations(self):
        """Test PlaceholderTitleFormatter for 2-to-1 operations."""
        formatter = PlaceholderTitleFormatter()

        # Test basic 2-to-1 formatting
        assert formatter.format_2_to_1_title("subtract") == "subtract({0}, {1})"

        # Test with operator
        assert formatter.format_2_to_1_title("-") == "{0}-{1}"

        # Test with suffix
        result = formatter.format_2_to_1_title("divide", "method=euclidean")
        assert result == "divide({0}, {1})|method=euclidean"

    def test_edge_cases(self):
        """Test edge cases for PlaceholderTitleFormatter."""
        formatter = PlaceholderTitleFormatter()

        # Test empty function name
        assert formatter.format_1_to_1_title("") == "({0})"

        # Test None suffix (should be handled gracefully)
        result = formatter.format_1_to_1_title("test", None)
        assert result == "test({0})"

        # Test empty suffix
        result = formatter.format_1_to_1_title("test", "")
        assert result == "test({0})"

    def test_datalab_compatibility(self):
        """Test DataLab-style placeholder patterns."""
        formatter = PlaceholderTitleFormatter()

        # Test typical DataLab placeholder patterns (DataLab handles resolution itself)
        test_cases = [
            ("wiener", None, "wiener({0})"),
            ("gaussian", "sigma=2.0", "gaussian({0})|sigma=2.0"),
            ("add", "3 objects", "add({0})|3 objects"),
        ]

        for func_name, suffix, expected in test_cases:
            result = formatter.format_1_to_1_title(func_name, suffix)
            assert result == expected


class TestFormatterConfiguration:
    """Test suite for the global formatter configuration system."""

    def test_configuration_system(self):
        """Test the global formatter configuration system."""
        # Store original formatter to restore later
        original_formatter = get_default_title_formatter()

        try:
            # Test setting SimpleTitleFormatter
            simple_formatter = SimpleTitleFormatter()
            set_default_title_formatter(simple_formatter)
            current_formatter = get_default_title_formatter()
            assert isinstance(current_formatter, SimpleTitleFormatter)

            # Test setting PlaceholderTitleFormatter
            placeholder_formatter = PlaceholderTitleFormatter()
            set_default_title_formatter(placeholder_formatter)
            current_formatter = get_default_title_formatter()
            assert isinstance(current_formatter, PlaceholderTitleFormatter)

        finally:
            # Restore original formatter
            set_default_title_formatter(original_formatter)


class TestIntegrationWithComputationFunctions:
    """Test suite for integration with computation functions from sigima.proc.base."""

    def test_integration_with_dst_1_to_1(self):
        """Test integration with dst_1_to_1 function."""
        original_formatter = get_default_title_formatter()

        try:
            # Test with SimpleTitleFormatter
            set_default_title_formatter(SimpleTitleFormatter())
            src = create_signal("Test Signal", x=[1, 2, 3], y=[4, 5, 6])
            result = dst_1_to_1(src, "gaussian_filter", "sigma=1.0")
            assert "Gaussian Filter Result" in result.title
            assert "sigma=1.0" in result.title

            # Test with PlaceholderTitleFormatter
            set_default_title_formatter(PlaceholderTitleFormatter())
            result2 = dst_1_to_1(src, "gaussian_filter", "sigma=1.0")
            assert result2.title == "gaussian_filter({0})|sigma=1.0"

        finally:
            set_default_title_formatter(original_formatter)

    def test_integration_with_dst_2_to_1(self):
        """Test integration with dst_2_to_1 function."""
        original_formatter = get_default_title_formatter()

        try:
            # Test with SimpleTitleFormatter
            set_default_title_formatter(SimpleTitleFormatter())
            src1 = create_signal("Signal1", x=[1, 2, 3], y=[1, 2, 3])
            src2 = create_signal("Signal2", x=[1, 2, 3], y=[4, 5, 6])
            result = dst_2_to_1(src1, src2, "subtract")
            assert result.title == "Subtract Result"

            # Test with PlaceholderTitleFormatter
            set_default_title_formatter(PlaceholderTitleFormatter())
            result2 = dst_2_to_1(src1, src2, "subtract")
            assert result2.title == "subtract({0}, {1})"

        finally:
            set_default_title_formatter(original_formatter)

    def test_integration_with_dst_n_to_1(self):
        """Test integration with dst_n_to_1 function."""
        original_formatter = get_default_title_formatter()

        try:
            # Test with SimpleTitleFormatter
            set_default_title_formatter(SimpleTitleFormatter())
            signals = [
                create_signal(f"Signal{i}", x=[1, 2, 3], y=[i, i + 1, i + 2])
                for i in range(1, 4)
            ]
            result = dst_n_to_1(signals, "sum")
            assert result.title == "Sum of 3 Objects"

            # Test with PlaceholderTitleFormatter
            set_default_title_formatter(PlaceholderTitleFormatter())
            result2 = dst_n_to_1(signals, "sum")
            assert result2.title == "sum({0}, {1}, {2})"

        finally:
            set_default_title_formatter(original_formatter)


class TestFormatterConsistency:
    """Test suite for formatter consistency and compatibility."""

    def test_formatter_consistency(self):
        """Test that formatters produce consistent results."""
        simple = SimpleTitleFormatter()
        placeholder = PlaceholderTitleFormatter()

        # Test same function name produces consistent patterns
        func_name = "gaussian_filter"
        suffix = "sigma=1.0"

        simple_result = simple.format_1_to_1_title(func_name, suffix)
        placeholder_result = placeholder.format_1_to_1_title(func_name, suffix)

        # Both should include the suffix information
        assert suffix in simple_result
        assert suffix in placeholder_result

        # Simple should be human-readable
        assert "Gaussian Filter" in simple_result

        # Placeholder should use function name as-is
        assert func_name in placeholder_result


class TestNewSignalResult:
    """Test suite for new_signal_result function title formatting."""

    def test_suffix_not_duplicated(self):
        """Test that suffix is not duplicated in new_signal_result output.

        Regression test for bug where suffix was added twice to the title:
        once by the formatter and once explicitly in new_signal_result.
        """
        src = create_signal(title="test_signal")
        suffix = "center=(10.0, 20.0)"

        # Test with SimpleTitleFormatter
        original_formatter = get_default_title_formatter()
        try:
            set_default_title_formatter(SimpleTitleFormatter())
            result = new_signal_result(src, "radial_profile", suffix)
            # Count occurrences of suffix in title - should be exactly 1
            assert result.title.count(suffix) == 1
            assert result.title == "Radial Profile Result (center=(10.0, 20.0))"

            # Test with PlaceholderTitleFormatter
            set_default_title_formatter(PlaceholderTitleFormatter())
            result = new_signal_result(src, "radial_profile", suffix)
            # Count occurrences of suffix in title - should be exactly 1
            assert result.title.count(suffix) == 1
            assert result.title == "radial_profile({0})|center=(10.0, 20.0)"

        finally:
            set_default_title_formatter(original_formatter)


if __name__ == "__main__":
    pytest.main([__file__])
