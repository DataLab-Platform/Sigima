# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for :py:mod:`sigima.tools.datatypes`.

These tests focus on the resolution-handling of :py:func:`datetime64_to_seconds`,
ensuring that all common ``numpy.datetime64`` resolutions (``ns``, ``us``, ``ms``,
``s``, ``D``...) are converted to Unix timestamps with the correct scale factor.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from sigima.tools.datatypes import datetime64_to_seconds

# Reference Unix timestamp for 2025-01-01 00:00:00 UTC
REF_DATE = "2025-01-01"
REF_TS = float(np.datetime64(REF_DATE, "s").astype("int64"))


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_datetime64_to_seconds_resolutions(unit: str) -> None:
    """``datetime64_to_seconds`` must return correct Unix seconds whatever the
    resolution of the input array (``ns``, ``us``, ``ms``, ``s``)."""
    arr = np.array([REF_DATE], dtype=f"datetime64[{unit}]")
    seconds = datetime64_to_seconds(arr)
    assert seconds.shape == (1,)
    assert np.isclose(seconds[0], REF_TS, rtol=0, atol=1e-3)


def test_datetime64_to_seconds_day_resolution() -> None:
    """Day-level resolution must also be normalized correctly to seconds."""
    arr = np.array([REF_DATE], dtype="datetime64[D]")
    seconds = datetime64_to_seconds(arr)
    assert np.isclose(seconds[0], REF_TS, rtol=0, atol=1e-3)


def test_datetime64_to_seconds_multiple_values() -> None:
    """Multi-element arrays should yield the expected per-element timestamps."""
    arr = np.array(
        ["1970-01-01T00:00:00", "2000-01-01T00:00:00", "2025-01-01T00:00:00"],
        dtype="datetime64[s]",
    )
    seconds = datetime64_to_seconds(arr)
    expected = np.array(
        [
            0.0,
            float(np.datetime64("2000-01-01", "s").astype("int64")),
            float(np.datetime64("2025-01-01", "s").astype("int64")),
        ]
    )
    assert np.allclose(seconds, expected)


def test_datetime64_to_seconds_pandas_compatibility() -> None:
    """Pandas ``DatetimeIndex.values`` (typically ``ns``) must convert
    consistently with :py:meth:`datetime.datetime.timestamp`."""
    dt = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    arr = pd.to_datetime([dt]).values
    seconds = datetime64_to_seconds(arr)
    assert np.isclose(seconds[0], dt.timestamp(), rtol=0, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
