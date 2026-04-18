# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Features (see parent package :mod:`sigima.algorithms.signal`)
"""

from __future__ import annotations

import numpy as np

from sigima.tools.checks import check_1d_array, check_1d_arrays


@check_1d_array(min_size=2, finite_only=True)
def find_zero_crossings(y: np.ndarray) -> np.ndarray:
    """Find the left indices of the zero-crossing intervals in the given array.

    A zero crossing is detected when consecutive non-zero samples have opposite
    signs. Samples exactly equal to zero are skipped: a contiguous run of zero
    samples that bridges samples of opposite sign counts as a single crossing,
    located at the index of the last non-zero sample before the run; a run of
    zeros that does not change the sign (e.g. ``[1, 0, 1]``) is not reported as
    a crossing.

    Args:
        y: Input array.

    Returns:
        An array of indices where zero-crossings occur.
    """
    sgn = np.sign(np.asarray(y))
    nonzero_idx = np.flatnonzero(sgn)
    if nonzero_idx.size < 2:
        return np.array([], dtype=int)
    # Detect sign changes between consecutive non-zero samples (skipping zeros).
    changes = np.flatnonzero(np.diff(sgn[nonzero_idx]) != 0)
    return nonzero_idx[changes]


@check_1d_arrays(x_sorted=True)
def find_x_axis_crossings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find the :math:`x_n` values where :math:`y = f(x)` intercepts the x-axis.

    This function uses zero-crossing detection and interpolation to find the x values
    where :math:`y = 0`.

    Args:
        x: X data.
        y: Y data.

    Returns:
        Array of x-intercepts. The array is empty if no intercept is found.
    """
    # Find zero crossings.
    xi_before = find_zero_crossings(y)
    if len(xi_before) == 0:
        return np.array([])
    # Interpolate to find x values at zero crossings.
    xi_after = xi_before + 1
    slope = (y[xi_after] - y[xi_before]) / (x[xi_after] - x[xi_before])
    with np.errstate(divide="ignore"):
        x0 = -y[xi_before] / slope + x[xi_before]
        x0 = np.where(np.isfinite(x0), x0, (x[xi_before] + x[xi_after]) / 2)
        # mask = ~np.isfinite(x0)
        # x0[mask] = xi_before[mask]
    return x0


@check_1d_arrays(x_min_size=2, x_finite_only=True, x_sorted=True)
def find_y_at_x_value(x: np.ndarray, y: np.ndarray, x_target: float) -> float:
    """Return the y value at a specified x value using linear interpolation.

    Args:
        x: X data.
        y: Y data.
        x_target: Input x value.

    Returns:
        Interpolated y value at x_target, or `nan` if input value is not within the
        interpolation range.
    """
    if np.isnan(x_target):
        return np.nan
    return float(np.interp(x_target, x, y, left=np.nan, right=np.nan))


@check_1d_arrays
def find_x_values_at_y(x: np.ndarray, y: np.ndarray, y_target: float) -> np.ndarray:
    """Find all x values where :math:`y = f(x)` equals the value :math:`y_target`.

    Args:
        x: X data.
        y: Y data.
        y_target: Target value.

    Returns:
        Array of x values where :math:`y = f(x)` equals :math:`y_target`.
    """
    return find_x_axis_crossings(x, y - y_target)


@check_1d_arrays(x_evenly_spaced=True)
def find_bandwidth_coordinates(
    x: np.ndarray, y: np.ndarray, threshold: float = -3.0
) -> tuple[float, float, float, float] | None:
    """Compute the bandwidth of the signal at a given threshold relative to the maximum.

    Args:
        x: X data.
        y: Y data.
        threshold: Threshold in decibel (relative to the maximum) at which the bandwidth
         is computed. Defaults to -3.0 dB.

    Returns:
        Segment coordinates of the bandwidth of the signal at the given threshold.
        Returns None if the bandwidth cannot be determined.
    """
    level: float = np.max(y) + threshold
    crossings = find_x_values_at_y(x, y, level)
    if len(crossings) == 1:
        # One crossing: 1) baseband bandwidth if max is above crossing
        #               2) passband bandwidth if max is below crossing
        if x[np.argmax(y)] < crossings[0]:  # Baseband bandwidth
            coords = (0.0, level, crossings[0], level)
        else:
            coords = (crossings[0], level, x[-1], level)
    elif len(crossings) == 2:  # Passband bandwidth
        # Two crossings: 1) passband bandwidth if max is above both crossings
        #                2) no bandwidth if max is below both crossings
        #                3) baseband bandwidth if max is between crossings
        coords = (crossings[0], level, crossings[1], level)
    else:
        # No crossing or more than two crossings: cannot determine bandwidth
        return None
    return coords


def contrast(y: np.ndarray) -> float:
    """Compute contrast.

    The contrast is defined as ``(max - min) / (|max| + |min|)``. For
    non-negative signals this matches the standard Michelson contrast
    ``(max - min) / (max + min)``. For signals containing negative samples,
    the absolute values in the denominator keep the result well-defined and
    bounded in ``[0, 1]``. If both ``max`` and ``min`` are zero, ``nan`` is
    returned.

    Args:
        y: Input array

    Returns:
        Contrast value in ``[0, 1]``, or ``nan`` if undefined.
    """
    max_, min_ = np.max(y), np.min(y)
    denom = np.abs(max_) + np.abs(min_)
    if denom == 0:
        return float("nan")
    return float((max_ - min_) / denom)
