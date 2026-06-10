# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Data Type Conversion Algorithms (see parent package :mod:`sigima.algorithms`)
"""

from __future__ import annotations

import numpy as np


def datetime64_to_seconds(dt_values: np.ndarray) -> np.ndarray:
    """Convert numpy datetime64 array to float seconds (Unix timestamps).

    This function handles different datetime resolutions (ns, us, ms, s) correctly,
    providing compatibility with both older pandas versions (using nanoseconds) and
    pandas 3.0+ (using microseconds by default).

    Args:
        dt_values: NumPy array of datetime64 values (from pandas Series.values or
         DatetimeIndex.values)

    Returns:
        NumPy array of float64 Unix timestamps (seconds since 1970-01-01)

    Example:
        >>> import pandas as pd
        >>> dt_series = pd.to_datetime(['2025-01-01 10:00:00'])
        >>> seconds = datetime64_to_seconds(dt_series.values)
        >>> seconds[0] > 1.7e9  # Year 2025
        True
    """
    # Normalize all datetime64 resolutions (ns, us, ms, s, m, h, D...) to
    # nanoseconds before converting to float seconds. This avoids the previous
    # substring-matching heuristic which silently produced wrong results for
    # coarser resolutions such as datetime64[s] or datetime64[D].
    return dt_values.astype("datetime64[ns]").view(np.int64) / 1e9


def clip_astype(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert array to a new data type, after having clipped values to the new
    data type's range if it is an integer type.
    If data type is not integer, this is equivalent to ``data.astype(dtype)``.

    Args:
        data: Array to convert
        dtype: Data type to convert to

    Returns:
        Array converted to new data type
    """
    if np.issubdtype(dtype, np.integer):
        return np.clip(data, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return data.astype(dtype)
