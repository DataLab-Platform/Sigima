# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Replace special values in 1D signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Low-level NumPy algorithms for replacing NaN, +Inf and -Inf values in 1D
arrays.  These functions operate on raw arrays and are called by the
higher-level :mod:`sigima.proc.signal.processing` functions.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.interpolate

from sigima.enums import Interpolation1DMethod

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


def check_uniform_sampling(x: np.ndarray, rtol: float = 1e-6) -> bool:
    """Check whether *x* is uniformly sampled.

    Args:
        x: 1-D array of abscissa values (must be sorted).
        rtol: relative tolerance for the spacing comparison.

    Returns:
        ``True`` if the spacing between consecutive points is constant
        (within *rtol*).
    """
    if x.size < 2:
        return True
    dx = np.diff(x)
    return bool(np.allclose(dx, dx[0], rtol=rtol))


def replace_with_fixed(y: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
    """Replace masked positions with a fixed *value*.

    Args:
        y: data array (modified in place).
        mask: boolean mask (``True`` where replacement is needed).
        value: replacement value.

    Returns:
        *y* (modified in place).
    """
    y[mask] = value
    return y


def replace_with_stat(
    y: np.ndarray,
    mask: np.ndarray,
    stat: str,
) -> np.ndarray:
    """Replace masked positions with a statistic computed on valid data.

    Only finite values (excluding NaN, +Inf, -Inf) are used to compute the
    statistic, so that special values still present for other targets do not
    bias the result.

    Args:
        y: data array (modified in place).
        mask: boolean mask.
        stat: one of ``"min"``, ``"max"``, ``"mean"``, ``"median"``.

    Returns:
        *y* (modified in place).
    """
    valid = y[~mask & np.isfinite(y)]
    if valid.size == 0:
        warnings.warn(
            "No valid data to compute statistic; filling with 0.", stacklevel=2
        )
        y[mask] = 0.0
        return y
    y[mask] = _compute_stat_with_fallback(valid, stat)
    return y


def delete_masked_points(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Delete points where *mask* is ``True``.

    Args:
        x: abscissa array.
        y: ordinate array.
        mask: boolean mask of points to remove.

    Returns:
        Tuple ``(x_new, y_new)`` with masked points removed.
    """
    keep = ~mask
    return x[keep], y[keep]


def forward_fill(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked positions with the previous valid value (forward fill).

    If the first element(s) are masked, they are filled with the first valid value
    encountered.

    Args:
        y: data array (modified in place).
        mask: boolean mask.

    Returns:
        *y* (modified in place).
    """
    indices = np.where(~mask, np.arange(len(y)), 0)
    np.maximum.accumulate(indices, out=indices)
    # Handle leading masked values: fill with first valid
    first_valid = np.argmax(~mask)
    indices[:first_valid] = first_valid
    y[:] = y[indices]
    return y


def backward_fill(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked positions with the next valid value (backward fill).

    If the last element(s) are masked, they are filled with the last valid value
    encountered.

    Args:
        y: data array (modified in place).
        mask: boolean mask.

    Returns:
        *y* (modified in place).
    """
    n = len(y)
    indices = np.where(~mask, np.arange(n), n - 1)
    # Reverse accumulate minimum
    indices = np.minimum.accumulate(indices[::-1])[::-1]
    # Handle trailing masked values
    last_valid = n - 1 - np.argmax(~mask[::-1])
    indices[indices > last_valid] = last_valid
    y[:] = y[indices]
    return y


_INTERP_METHOD_MAP = {
    "interp_linear": Interpolation1DMethod.LINEAR,
    "interp_spline": Interpolation1DMethod.SPLINE,
    "interp_quadratic": Interpolation1DMethod.QUADRATIC,
    "interp_cubic": Interpolation1DMethod.CUBIC,
    "interp_pchip": Interpolation1DMethod.PCHIP,
}


def interpolate_masked(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    method: str,
) -> np.ndarray:
    """Interpolate values at masked positions using valid neighbors.

    Args:
        x: abscissa array.
        y: data array (modified in place).
        mask: boolean mask.
        method: strategy value string (e.g. ``"interp_linear"``).

    Returns:
        *y* (modified in place).
    """
    valid = ~mask
    if valid.sum() < 2:
        warnings.warn(
            "Not enough valid points for interpolation; filling with 0.", stacklevel=2
        )
        y[mask] = 0.0
        return y

    interp_method = _INTERP_METHOD_MAP[method]
    x_valid, y_valid = x[valid], y[valid]
    x_masked = x[mask]

    if interp_method == Interpolation1DMethod.LINEAR:
        y[mask] = np.interp(x_masked, x_valid, y_valid)
    elif interp_method == Interpolation1DMethod.SPLINE:
        knots, coeffs, degree = scipy.interpolate.splrep(x_valid, y_valid, s=0)
        y[mask] = scipy.interpolate.splev(x_masked, (knots, coeffs, degree), der=0)
    elif interp_method == Interpolation1DMethod.QUADRATIC:
        coeffs = np.polyfit(x_valid, y_valid, min(2, len(x_valid) - 1))
        y[mask] = np.polyval(coeffs, x_masked)
    elif interp_method == Interpolation1DMethod.CUBIC:
        interp = scipy.interpolate.Akima1DInterpolator(x_valid, y_valid)
        y[mask] = interp(x_masked)
    elif interp_method == Interpolation1DMethod.PCHIP:
        interp = scipy.interpolate.PchipInterpolator(x_valid, y_valid)
        y[mask] = interp(x_masked)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    return y


def neighbor_replace(
    y: np.ndarray,
    mask: np.ndarray,
    n: int,
    stat: str,
) -> np.ndarray:
    """Replace masked positions using statistics of their *n* nearest valid neighbors.

    When no valid neighbor is found within the initial radius *n*, the search
    radius is progressively doubled until valid data is found or the full array
    has been scanned.  If still no valid neighbor exists, the corresponding
    global statistic is used as a fallback.  As a last resort the value is set
    to ``0.0``.

    Args:
        y: data array (modified in place).
        mask: boolean mask.
        n: number of neighbors to consider on each side.
        stat: ``"min"``, ``"max"``, ``"mean"`` or ``"median"``.

    Returns:
        *y* (modified in place).
    """
    funcs = {
        "min": np.nanmin,
        "max": np.nanmax,
        "mean": np.nanmean,
        "median": np.nanmedian,
    }
    func = funcs[stat]
    size = len(y)
    # Work on a copy to avoid using already-replaced values
    y_orig = y.copy()
    y_orig[mask] = np.nan

    # Pre-compute global fallback (all valid, i.e. non-masked, finite values)
    all_valid = y_orig[np.isfinite(y_orig)]
    if all_valid.size > 0:
        global_fallback = float(func(all_valid))
    else:
        global_fallback = 0.0

    for idx in np.where(mask)[0]:
        value = _neighbor_search_1d(y_orig, idx, n, size, func)
        if value is not None:
            y[idx] = value
        else:
            if global_fallback != 0.0:
                warnings.warn(
                    f"No valid neighbor found at index {idx}; "
                    f"using global {stat} as fallback.",
                    stacklevel=2,
                )
            y[idx] = global_fallback
    return y


def _neighbor_search_1d(
    y_orig: np.ndarray, idx: int, n: int, size: int, func
) -> float | None:
    """Search for valid neighbors with progressive radius expansion.

    Returns the computed statistic or ``None`` if no valid neighbor exists.
    """
    radius = n
    while radius < size:
        lo = max(0, idx - radius)
        hi = min(size, idx + radius + 1)
        neighbors = y_orig[lo:hi]
        valid = neighbors[np.isfinite(neighbors)]
        if valid.size > 0:
            with np.errstate(invalid="ignore"):
                result = float(func(valid))
            if not np.isnan(result):
                return result
        # Double the radius for the next attempt
        radius *= 2
    return None


def count_special_values(
    y: np.ndarray,
) -> dict[str, int]:
    """Count special values in a 1-D array.

    Args:
        y: data array.

    Returns:
        Dictionary with keys ``"nan"``, ``"posinf"``, ``"neginf"``
        and integer counts.
    """
    return {
        "nan": int(np.count_nonzero(np.isnan(y))),
        "posinf": int(np.count_nonzero(np.isposinf(y))),
        "neginf": int(np.count_nonzero(np.isneginf(y))),
    }
