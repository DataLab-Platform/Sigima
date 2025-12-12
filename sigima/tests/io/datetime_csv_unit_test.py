# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DateTime CSV I/O Unit Test
==========================

Unit tests for reading CSV files with datetime columns.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os
import os.path as osp
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sigima.io import read_signal, read_signals, write_signal
from sigima.io.signal.formats import CSVSignalFormat
from sigima.objects import SignalObj, create_signal
from sigima.tests.env import execenv
from sigima.tests.helpers import get_test_fnames


def test_datetime_csv_io() -> None:
    """Test reading CSV file with datetime X column."""
    execenv.print("Testing datetime CSV I/O...")

    # Get path to the datetime test file
    filenames = get_test_fnames("datetime.txt", in_folder="curve_formats")
    assert len(filenames) > 0, "datetime.txt test file not found"
    filename = filenames[0]
    assert osp.exists(filename), f"Test file not found: {filename}"

    # Read signals from file (read_signals returns a list, read_signal returns first)
    signals = read_signals(filename)

    # Should create multiple signals (one per Y column)
    assert len(signals) > 0, "No signals were read from datetime.txt"
    execenv.print(f"  Read {len(signals)} signals from file")

    # Test first signal (Temperature)
    signal = signals[0]
    execenv.print(f"  First signal: {signal.title}")

    # Check that datetime metadata was detected
    assert signal.metadata.get("x_datetime", False), (
        "DateTime metadata not detected in X column"
    )
    assert signal.xunit == "s", "DateTime unit should be 's' (seconds)"

    # Check X data is float (timestamps)
    assert isinstance(signal.x, np.ndarray), "X data should be numpy array"
    assert signal.x.dtype in (np.float32, np.float64), "X data should be float"
    execenv.print(f"  X data type: {signal.x.dtype}")
    execenv.print(f"  X data shape: {signal.x.shape}")
    execenv.print(f"  First 5 X values: {signal.x[:5]}")

    # Remove NaN values before checking monotonicity
    x_clean = signal.x[~np.isnan(signal.x)]
    execenv.print(f"  Clean X shape (no NaNs): {x_clean.shape}")
    execenv.print(f"  First clean X value (timestamp): {x_clean[0]:.2f}")

    # Check X values are monotonically increasing
    assert np.all(np.diff(x_clean) >= 0), "X values should be monotonic"

    # Check Y data exists and has correct length
    assert isinstance(signal.y, np.ndarray), "Y data should be numpy array"
    assert len(signal.x) == len(signal.y), "X and Y should have same length"
    execenv.print(f"  Y data shape: {signal.y.shape}")
    execenv.print(f"  First Y value: {signal.y[0]}")

    # Test datetime conversion back
    dt_values = signal.get_x_as_datetime()
    assert dt_values is not None, "Should be able to get datetime values"
    assert len(dt_values) == len(signal.x), (
        "Datetime array should have same length as X"
    )
    execenv.print(f"  First datetime value: {dt_values[0]}")

    # Check that the datetime is reasonable (should be 2025-06-19 10:00:00)
    # Convert to string to check
    dt_str = pd.to_datetime(dt_values[0]).strftime("%Y-%m-%d %H:%M:%S")
    expected_start = "2025-06-19 10:00:00"
    assert dt_str == expected_start, (
        f"Expected first datetime to be {expected_start}, got {dt_str}"
    )

    # Check labels were extracted correctly
    assert signal.xlabel, "X label should be set"
    assert signal.ylabel, "Y label should be set"
    execenv.print(f"  X label: {signal.xlabel}")
    execenv.print(f"  Y label: {signal.ylabel}")

    # Check we have multiple signals (Temperature, Humidity, Dew Point)
    assert len(signals) >= 3, "Should have at least 3 signals"
    signal_titles = [s.ylabel for s in signals]
    execenv.print(f"  Signal Y labels: {signal_titles}")

    # All signals should have the same datetime metadata
    for sig in signals:
        assert sig.is_x_datetime(), "All signals should have datetime X"
        assert sig.xunit == "s"

    # Check that all signals have the same X data (timestamps)
    for sig in signals[1:]:
        assert np.array_equal(sig.x, signals[0].x), (
            "All signals should share the same X data"
        )

    execenv.print("  ✓ DateTime CSV I/O test passed")


def test_datetime_csv_write_with_datetime(tmp_path: Path):
    """Test writing CSV file with datetime X values"""
    # Create signal with datetime X
    timestamps = [
        datetime(2025, 1, 1, 10, 0, 0),
        datetime(2025, 1, 1, 10, 0, 1),
        datetime(2025, 1, 1, 10, 0, 2),
    ]
    signal = SignalObj()
    signal.set_x_from_datetime(timestamps, unit="s")
    signal.y = np.array([1.0, 2.0, 3.0])
    signal.ylabel = "Temperature"
    signal.xlabel = "Time"

    # Write to CSV
    csv_file = tmp_path / "datetime_signal.csv"
    fmt = CSVSignalFormat()
    fmt.write(str(csv_file), signal)

    # Read file back and check contents
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Should have header + 3 data lines
    assert len(lines) == 4
    # Header should be "Time,Temperature"
    assert "Time" in lines[0]
    assert "Temperature" in lines[0]
    # First data line should contain datetime string
    assert "2025-01-01" in lines[1]
    assert "10:00:00" in lines[1]


def test_datetime_csv_roundtrip() -> None:
    """Test that datetime signals can be written and read back."""
    execenv.print("Testing datetime CSV roundtrip...")

    # Create a signal with datetime X
    base_time = datetime(2025, 10, 6, 10, 0, 0)
    timestamps = [base_time + timedelta(minutes=i * 5) for i in range(20)]
    values = 20 + np.random.randn(20) * 2

    signal = create_signal("Test Temperature")
    signal.set_x_from_datetime(timestamps, unit="s")
    signal.y = values
    signal.ylabel = "Temperature"
    signal.yunit = "°C"

    # Write to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as tmp:
        tmp_path = tmp.name

    try:
        write_signal(tmp_path, signal)
        execenv.print(f"  Wrote signal to: {tmp_path}")

        # Read it back
        signal_read = read_signal(tmp_path)
        execenv.print("  Signal read back successfully")

        # Check datetime metadata is preserved
        assert signal_read.is_x_datetime(), "DateTime metadata should be preserved"

        # Check Y values match (X will be timestamps now, not exact datetime strings)
        assert len(signal_read.y) == len(values), "Y length should match"
        assert np.allclose(signal_read.y, values, atol=0.01), "Y values should match"

        execenv.print("  ✓ DateTime CSV roundtrip test passed")

    finally:
        # Clean up temporary file
        if osp.exists(tmp_path):
            os.unlink(tmp_path)


def test_numeric_csv_not_interpreted_as_datetime() -> None:
    """Test that purely numeric CSV columns are not misinterpreted as datetime.

    Regression test: When X column contains large numeric values like frequencies
    in Hz (e.g., 4.884e+06), pd.to_datetime() would interpret them as nanoseconds
    since Unix epoch, causing the data to be corrupted. Numeric columns should be
    skipped by the datetime detection logic.
    """
    execenv.print("Testing numeric CSV not interpreted as datetime...")

    # Create a signal with large numeric X values (frequencies in Hz)
    # Values range from 0 to 20 GHz - these should NOT be interpreted as datetime
    x = np.linspace(0, 2e10, 100)  # 0 to 20 GHz
    y = np.sin(x / 1e9)  # Some arbitrary Y data

    signal = create_signal("Frequency Response")
    signal.set_xydata(x, y)
    signal.xlabel = "Frequency"
    signal.xunit = "Hz"
    signal.ylabel = "Amplitude"
    signal.yunit = "dB"

    # Write to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as tmp:
        tmp_path = tmp.name

    try:
        write_signal(tmp_path, signal)
        execenv.print(f"  Wrote signal to: {tmp_path}")

        # Read it back
        signal_read = read_signal(tmp_path)
        execenv.print("  Signal read back successfully")

        # CRITICAL: Should NOT be interpreted as datetime
        assert not signal_read.is_x_datetime(), (
            "Numeric X data should NOT be interpreted as datetime"
        )

        # Check X values are preserved (not corrupted by datetime conversion)
        assert np.allclose(signal_read.x, x), (
            "X values should be preserved (not converted to timestamps)"
        )
        assert signal_read.x.min() == 0, "X min should be 0"
        assert np.isclose(signal_read.x.max(), 2e10), "X max should be 2e10"

        # Check Y values match
        assert np.allclose(signal_read.y, y, atol=0.01), "Y values should match"

        execenv.print("  ✓ Numeric CSV not interpreted as datetime test passed")

    finally:
        # Clean up temporary file
        if osp.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
