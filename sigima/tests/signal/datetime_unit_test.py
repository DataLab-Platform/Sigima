# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DateTime support unit tests
===========================

Unit tests for datetime functionality in SignalObj.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from sigima.objects import create_signal
from sigima.objects.signal.constants import VALID_TIME_UNITS
from sigima.tests.env import execenv


def test_signal_datetime_methods() -> None:
    """Test SignalObj datetime methods."""
    execenv.print("Testing SignalObj datetime methods...")

    # Create datetime data
    base_time = datetime(2025, 10, 6, 10, 0, 0)
    timestamps = [base_time + timedelta(seconds=i) for i in range(10)]
    values = np.sin(np.arange(10) * 0.5)

    format_str = "%Y-%m-%d %H:%M:%S"

    # Test different units
    for unit in VALID_TIME_UNITS:
        # Create signal with initial data
        signal = create_signal(
            "Test Signal", x=np.arange(10, dtype=float), y=values.copy()
        )

        # Initially should not be datetime
        assert not signal.is_x_datetime()

        # Set x from datetime
        signal.set_x_from_datetime(timestamps, unit=unit, format_str=format_str)

        # Check datetime flag
        assert signal.is_x_datetime()
        assert signal.metadata["x_datetime"] is True
        assert signal.xunit == unit
        assert signal.metadata["x_datetime_format"] == format_str

        # Check x data is float
        assert isinstance(signal.x, np.ndarray)
        assert signal.x.dtype in (np.float32, np.float64)

        # Get x as datetime
        dt_values = signal.get_x_as_datetime()
        assert isinstance(dt_values, np.ndarray)
        # Accept any datetime64 resolution (ns, us, ms, s) for compatibility with
        # different NumPy/pandas versions
        assert np.issubdtype(dt_values.dtype, np.datetime64)

        # Verify y values are unchanged
        assert np.allclose(signal.y, values)

    execenv.print("  ✓ SignalObj datetime methods test passed")


def test_datetime_with_string_input() -> None:
    """Test datetime conversion from string input."""
    execenv.print("Testing datetime conversion from strings...")

    signal = create_signal("String DateTime Test")

    # Create datetime strings
    date_strings = [
        "2025-10-06 10:00:00",
        "2025-10-06 10:00:01",
        "2025-10-06 10:00:02",
    ]
    values = [1.0, 2.0, 3.0]

    # Set from strings
    signal.set_x_from_datetime(date_strings, unit="s")
    signal.y = values

    # Verify it worked
    assert signal.is_x_datetime()
    assert len(signal.x) == len(date_strings)

    # Get back as datetime
    dt_values = signal.get_x_as_datetime()
    assert len(dt_values) == len(date_strings)

    execenv.print("  ✓ String datetime conversion test passed")


def test_datetime_copy() -> None:
    """Test that datetime metadata is preserved when copying signal."""
    execenv.print("Testing datetime metadata preservation in copy...")

    signal = create_signal("Original")
    timestamps = [datetime(2025, 10, 6, 10, 0, i) for i in range(5)]
    signal.set_x_from_datetime(timestamps, unit="ms")
    signal.y = np.arange(5, dtype=float)

    # Copy signal
    signal_copy = signal.copy()

    # Verify datetime metadata is preserved
    assert signal_copy.is_x_datetime()
    assert signal_copy.xunit == "ms"
    assert np.array_equal(signal.x, signal_copy.x)

    execenv.print("  ✓ Datetime metadata preservation test passed")


def test_datetime_non_datetime_signal() -> None:
    """Test that non-datetime signals work correctly."""
    execenv.print("Testing non-datetime signal behavior...")

    signal = create_signal("Regular Signal", x=np.arange(10), y=np.sin(np.arange(10)))

    # Should not be datetime
    assert not signal.is_x_datetime()

    # get_x_as_datetime should return regular x
    x_data = signal.get_x_as_datetime()
    assert np.array_equal(x_data, signal.x)

    execenv.print("  ✓ Non-datetime signal test passed")


def test_datetime_invalid_unit() -> None:
    """Test that invalid units raise appropriate errors."""
    execenv.print("Testing invalid unit handling...")

    timestamps = [datetime(2025, 10, 6, 10, 0, 0)]

    # Test SignalObj.set_x_from_datetime with invalid unit
    signal = create_signal("Test")
    with pytest.raises(ValueError, match="Invalid unit"):
        signal.set_x_from_datetime(timestamps, unit="invalid")

    execenv.print("  ✓ Invalid unit handling test passed")


def test_datetime_arithmetic_operations() -> None:
    """Test that datetime signals work with arithmetic operations."""
    execenv.print("Testing datetime signal arithmetic...")

    # Create two signals with datetime x
    base_time = datetime(2025, 10, 6, 10, 0, 0)
    timestamps = [base_time + timedelta(seconds=i) for i in range(10)]

    signal1 = create_signal("Signal 1")
    signal1.set_x_from_datetime(timestamps, unit="s")
    signal1.y = np.arange(10, dtype=float)

    signal2 = create_signal("Signal 2")
    signal2.set_x_from_datetime(timestamps, unit="s")
    signal2.y = np.arange(10, dtype=float) * 2

    # The x data should be identical floats
    assert np.array_equal(signal1.x, signal2.x)

    # Verify we can do arithmetic on y
    result_y = signal1.y + signal2.y
    expected_y = np.arange(10, dtype=float) * 3
    assert np.allclose(result_y, expected_y)

    execenv.print("  ✓ Datetime signal arithmetic test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
