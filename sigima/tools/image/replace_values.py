# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Replace special values in 2D images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Low-level NumPy/OpenCV algorithms for replacing NaN, +Inf and -Inf values in
2D arrays.  These functions operate on raw arrays and are called by the
higher-level :mod:`sigima.proc.image.exposure` functions.
"""

from __future__ import annotations

import warnings

import numpy as np

# Stat function registry and fallback chains.
# When a statistic produces NaN (e.g. nanmean on [+inf, -inf]), we fall back
# to the next statistic in the chain before resorting to 0.0.
_STAT_FUNCS = {
    "min": np.nanmin,
    "max": np.nanmax,
    "mean": np.nanmean,
    "median": np.nanmedian,
}

_STAT_FALLBACKS: dict[str, list[str]] = {
    "mean": ["median", "min"],
    "median": ["mean", "min"],
    "min": ["max"],
    "max": ["min"],
}


def _compute_stat_with_fallback(valid: np.ndarray, stat: str) -> float:
    """Compute *stat* on *valid* data with a fallback chain.

    If the primary statistic yields NaN (can happen when *valid* contains
    mixed infinities), the function tries the fallback statistics defined in
    :data:`_STAT_FALLBACKS` before returning ``0.0``.
    """
    chain = [stat] + _STAT_FALLBACKS.get(stat, [])
    for name in chain:
        with np.errstate(invalid="ignore"):
            value = float(_STAT_FUNCS[name](valid))
        if not np.isnan(value):
            if name != stat:
                warnings.warn(
                    f"Statistic '{stat}' produced NaN; falling back to '{name}'.",
                    stacklevel=3,
                )
            return value
    warnings.warn(
        "All statistics produced NaN; filling with 0.",
        stacklevel=3,
    )
    return 0.0


def replace_with_fixed_2d(
    data: np.ndarray, mask: np.ndarray, value: float
) -> np.ndarray:
    """Replace masked positions with a fixed *value*.

    Args:
        data: 2-D data array (modified in place).
        mask: boolean mask (``True`` where replacement is needed).
        value: replacement value.

    Returns:
        *data* (modified in place).
    """
    data[mask] = value
    return data


def replace_with_stat_2d(
    data: np.ndarray,
    mask: np.ndarray,
    stat: str,
) -> np.ndarray:
    """Replace masked positions with a statistic computed on valid data.

    Only finite values (excluding NaN, +Inf, -Inf) are used to compute the
    statistic, so that special values still present for other targets do not
    bias the result.

    Args:
        data: 2-D data array (modified in place).
        mask: boolean mask.
        stat: one of ``"min"``, ``"max"``, ``"mean"``, ``"median"``.

    Returns:
        *data* (modified in place).
    """
    valid = data[~mask & np.isfinite(data)]
    if valid.size == 0:
        warnings.warn(
            "No valid data to compute statistic; filling with 0.", stacklevel=2
        )
        data[mask] = 0.0
        return data
    data[mask] = _compute_stat_with_fallback(valid, stat)
    return data


def neighbor_replace_2d(
    data: np.ndarray,
    mask: np.ndarray,
    n: int,
    stat: str,
) -> np.ndarray:
    """Replace masked positions using statistics of their neighborhood.

    When no valid neighbor is found within the initial radius *n*, the search
    radius is progressively doubled until valid data is found or the full
    image has been scanned.  If still no valid neighbor exists, the
    corresponding global statistic is used as a fallback.  As a last resort
    the value is set to ``0.0``.

    Args:
        data: 2-D data array (modified in place).
        mask: boolean mask.
        n: neighborhood radius (the window is ``(2*n+1) × (2*n+1)``).
        stat: ``"min"``, ``"max"``, ``"mean"`` or ``"median"``.

    Returns:
        *data* (modified in place).
    """
    funcs = {
        "min": np.nanmin,
        "max": np.nanmax,
        "mean": np.nanmean,
        "median": np.nanmedian,
    }
    func = funcs[stat]
    rows, cols = data.shape
    max_dim = max(rows, cols)
    data_orig = data.copy()
    data_orig[mask] = np.nan

    # Pre-compute global fallback (all valid finite values)
    all_valid = data_orig[np.isfinite(data_orig)]
    if all_valid.size > 0:
        global_fallback = float(func(all_valid))
    else:
        global_fallback = 0.0

    for r, c in zip(*np.where(mask)):
        value = _neighbor_search_2d(data_orig, r, c, n, rows, cols, max_dim, func)
        if value is not None:
            data[r, c] = value
        else:
            if global_fallback != 0.0:
                warnings.warn(
                    f"No valid neighbor found at ({r}, {c}); "
                    f"using global {stat} as fallback.",
                    stacklevel=2,
                )
            data[r, c] = global_fallback
    return data


def _neighbor_search_2d(
    data_orig: np.ndarray,
    r: int,
    c: int,
    n: int,
    rows: int,
    cols: int,
    max_dim: int,
    func,
) -> float | None:
    """Search for valid neighbors with progressive radius expansion.

    Returns the computed statistic or ``None`` if no valid neighbor exists.
    """
    radius = n
    while radius < max_dim:
        r_lo, r_hi = max(0, r - radius), min(rows, r + radius + 1)
        c_lo, c_hi = max(0, c - radius), min(cols, c + radius + 1)
        patch = data_orig[r_lo:r_hi, c_lo:c_hi]
        valid = patch[np.isfinite(patch)]
        if valid.size > 0:
            with np.errstate(invalid="ignore"):
                result = float(func(valid))
            if not np.isnan(result):
                return result
        # Double the radius for the next attempt
        radius *= 2
    return None


def count_special_values_2d(
    data: np.ndarray,
) -> dict[str, int]:
    """Count special values in a 2-D array.

    Args:
        data: 2-D data array.

    Returns:
        Dictionary with keys ``"nan"``, ``"posinf"``, ``"neginf"``
        and integer counts.
    """
    return {
        "nan": int(np.count_nonzero(np.isnan(data))),
        "posinf": int(np.count_nonzero(np.isposinf(data))),
        "neginf": int(np.count_nonzero(np.isneginf(data))),
    }
