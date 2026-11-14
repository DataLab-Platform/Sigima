# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometric analysis module
-------------------------

This module provides functions for geometric analysis of images, including
centroid detection, shape fitting, and spatial measurements.

Features include:

- Various centroid detection algorithms (Fourier-based, projected profile,
  automatic selection)
- Enclosing circle calculation for thresholded regions
- Radial profile extraction around specified centers
- Absolute level calculation from relative thresholds

These tools support precise geometric measurements and shape analysis
for scientific and technical image analysis applications.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy import ma
from skimage import measure

from sigima.tools.checks import check_2d_array
from sigima.tools.image.preprocessing import get_absolute_level


@check_2d_array
def get_centroid_fourier(data: np.ndarray) -> tuple[float, float]:
    """Return image centroid using Fourier algorithm

    Args:
        data: Input data

    Returns:
        Centroid coordinates (row, col)
    """
    # Fourier transform method as discussed by Weisshaar et al.
    # (http://www.mnd-umwelttechnik.fh-wiesbaden.de/pig/weisshaar_u5.pdf)
    rows, cols = data.shape
    if rows == 1 or cols == 1:
        return 0, 0

    i = np.arange(0, rows).reshape(1, rows)
    sin_a = np.sin((i - 1) * 2 * np.pi / (rows - 1)).T
    cos_a = np.cos((i - 1) * 2 * np.pi / (rows - 1)).T

    j = np.arange(0, cols).reshape(cols, 1)
    sin_b = np.sin((j - 1) * 2 * np.pi / (cols - 1)).T
    cos_b = np.cos((j - 1) * 2 * np.pi / (cols - 1)).T

    a = np.nansum((cos_a * data))
    b = np.nansum((sin_a * data))
    c = np.nansum((data * cos_b))
    d = np.nansum((data * sin_b))

    rphi = (0 if b > 0 else 2 * np.pi) if a > 0 else np.pi
    cphi = (0 if d > 0 else 2 * np.pi) if c > 0 else np.pi

    if a * c == 0.0:
        return 0, 0

    row = (np.arctan(b / a) + rphi) * (rows - 1) / (2 * np.pi) + 1
    col = (np.arctan(d / c) + cphi) * (cols - 1) / (2 * np.pi) + 1

    row = np.nan if row is np.ma.masked else row
    col = np.nan if col is np.ma.masked else col

    return row, col


@check_2d_array
def get_projected_profile_centroid(
    data: np.ndarray, smooth_ratio: float = 1 / 40, method: str = "median"
) -> tuple[float, float]:
    """
    Estimate centroid from smoothed 1D projections.

    Args:
        data: 2D image array
        smooth_ratio: Ratio of smoothing window size (default: 1/40)
        method: 'median' (default) or 'barycenter'

    Returns:
        (y, x) coordinates
    """
    x_profile = data.sum(axis=0)
    y_profile = data.sum(axis=1)
    window_size = max(1, int(min(data.shape) * smooth_ratio))
    kernel = np.ones(window_size) / window_size
    x_profile = np.convolve(x_profile, kernel, mode="same")
    y_profile = np.convolve(y_profile, kernel, mode="same")
    x_profile -= np.min(x_profile)
    y_profile -= np.min(y_profile)

    if method == "median":
        x_integral = np.cumsum(x_profile)
        y_integral = np.cumsum(y_profile)
        x_center = np.interp(
            0.5 * x_integral[-1], x_integral, np.arange(len(x_integral))
        )
        y_center = np.interp(
            0.5 * y_integral[-1], y_integral, np.arange(len(y_integral))
        )
    elif method == "barycenter":  # pragma: no cover
        #  (ignored for coverage because median gives better results)
        x_center = np.sum(np.arange(len(x_profile)) * x_profile) / np.sum(x_profile)
        y_center = np.sum(np.arange(len(y_profile)) * y_profile) / np.sum(y_profile)
    else:
        raise ValueError("Unknown method: choose 'median' or 'barycenter'")

    return float(y_center), float(x_center)


@check_2d_array
def get_centroid_auto(
    data: np.ndarray,
    return_method: bool = False,
) -> tuple[float, float] | tuple[float, float, Literal["fourier", "skimage"]]:
    """
    Automatically select the most reliable centroid estimation method:
    - Prefer Fourier if it is more consistent with the projected median.
    - Fallback to scikit-image centroid if Fourier is less coherent.

    Args:
        data: 2D image array.
        return_method: If True, also return the name of the selected method.

    Returns:
        (row, col): Estimated centroid coordinates (float).
        Optionally, the selected method as string: "fourier" or "skimage".
    """
    try:
        row_f, col_f = get_centroid_fourier(data)
    except Exception:  # pylint: disable=broad-except
        row_f, col_f = float("nan"), float("nan")

    row_m, col_m = get_projected_profile_centroid(data, method="median")
    # Convert data (ndarray) to a simple array to compute centroid with the new
    # einsum optimisation introduce in numpy 2.4.0 and scikit-image 0.26.0
    img = np.array(data)
    row_s, col_s = measure.centroid(img)

    dist_f = np.hypot(row_f - row_m, col_f - col_m)
    dist_s = np.hypot(row_s - row_m, col_s - col_m)

    if not (np.isnan(row_f) or np.isnan(col_f)) and dist_f < dist_s:
        result = (row_f, col_f)
        method = "fourier"
    else:
        result = (row_s, col_s)
        method = "skimage"

    return result + (method,) if return_method else result


def _diameter_circle(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    center = (a + b) / 2
    return float(center[0]), float(center[1]), float(np.linalg.norm(a - center))


def _contains(circle: tuple[float, float, float], point: np.ndarray) -> bool:
    x, y, radius = circle
    return math.hypot(point[0] - x, point[1] - y) <= radius + 1e-14


def _circle_with_two_boundary_points(
    points: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[float, float, float]:
    """Find the smallest circle containing points with a and b on its boundary."""
    diameter = _diameter_circle(a, b)
    left = right = None
    left_cross, right_cross = -math.inf, math.inf
    ab = b - a
    for point in points:
        if _contains(diameter, point):
            continue
        ap = point - a
        cross = ab[0] * ap[1] - ab[1] * ap[0]
        if cross == 0:
            raise ValueError(
                "Cannot determine an enclosing circle for collinear points"
            )
        ab2 = float(np.dot(ab, ab))
        ap2 = float(np.dot(ap, ap))
        center = a + np.array(
            [ap[1] * ab2 - ab[1] * ap2, ab[0] * ap2 - ap[0] * ab2]
        ) / (2 * cross)
        candidate = (
            float(center[0]),
            float(center[1]),
            float(np.linalg.norm(center - a)),
        )
        center_cross = ab[0] * (center[1] - a[1]) - ab[1] * (center[0] - a[0])
        if cross > 0 and center_cross > left_cross:
            left, left_cross = candidate, center_cross
        elif cross < 0 and center_cross < right_cross:
            right, right_cross = candidate, center_cross
    if left is None:
        return diameter if right is None else right
    if right is None:
        return left
    return min(left, right, key=lambda circle: circle[2])


def _minimum_enclosing_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Compute a deterministic minimum enclosing circle of physical points."""
    minimum = points.min(axis=0)
    extent = points.max(axis=0) - minimum
    scale = float(extent.max())
    if scale == 0:
        return float(minimum[0]), float(minimum[1]), 0.0
    origin = minimum + extent / 2
    normalized = (points - origin) / scale
    normalized = normalized[np.random.default_rng(0).permutation(len(normalized))]
    circle = None
    for i, point in enumerate(normalized):
        if circle is not None and _contains(circle, point):
            continue
        circle = (float(point[0]), float(point[1]), 0.0)
        for j, other in enumerate(normalized[:i]):
            if not _contains(circle, other):
                circle = _circle_with_two_boundary_points(normalized[:j], point, other)
    assert circle is not None
    x, y = origin + scale * np.array(circle[:2])
    radius = max(
        scale * circle[2],
        float(np.max(np.hypot(points[:, 0] - x, points[:, 1] - y))),
    )
    return float(x), float(y), radius


@check_2d_array
def get_enclosing_circle(
    data: np.ndarray,
    level: float = 0.5,
    *,
    xcoords: np.ndarray | None = None,
    ycoords: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return the minimum circle enclosing centers of pixels above the threshold.

    Args:
        data: Input data
        level: Relative level within the data range (default: 0.5).
        xcoords: Physical pixel-center coordinates along columns, or pixel indices.
        ycoords: Physical pixel-center coordinates along rows, or pixel indices.

    Returns:
        A tuple (x, y, radius)

    Raises:
        ValueError: No pixel is above the threshold, or coordinates are invalid.
    """
    arr = ma.masked_where(np.isnan(data), ma.asarray(data))
    if arr.count() == 0:
        raise ValueError("No valid pixel for enclosing circle")
    if not np.all(np.isfinite(arr.compressed())):
        raise ValueError("Enclosing circle data contains infinite values")
    threshold = get_absolute_level(arr, level)
    selected = ma.filled(arr > threshold, False)
    if not np.any(selected):
        raise ValueError("No pixel above the enclosing circle threshold")
    x = (
        np.arange(data.shape[1], dtype=float)
        if xcoords is None
        else np.asarray(xcoords)
    )
    y = (
        np.arange(data.shape[0], dtype=float)
        if ycoords is None
        else np.asarray(ycoords)
    )
    if (
        x.shape != (data.shape[1],)
        or y.shape != (data.shape[0],)
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("Invalid pixel-center coordinates")
    rows = np.flatnonzero(np.any(selected, axis=1))
    xmin = np.min(np.where(selected[rows], x, np.inf), axis=1)
    xmax = np.max(np.where(selected[rows], x, -np.inf), axis=1)
    points = np.column_stack((np.concatenate((xmin, xmax)), np.tile(y[rows], 2)))
    return _minimum_enclosing_circle(points)
