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

from typing import Literal

import numpy as np
from skimage import __version__, measure

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
    row_s, col_s = measure.centroid(data)

    dist_f = np.hypot(row_f - row_m, col_f - col_m)
    dist_s = np.hypot(row_s - row_m, col_s - col_m)

    if not (np.isnan(row_f) or np.isnan(col_f)) and dist_f < dist_s:
        result = (row_f, col_f)
        method = "fourier"
    else:
        result = (row_s, col_s)
        method = "skimage"

    return result + (method,) if return_method else result


@check_2d_array(non_constant=True)
def get_enclosing_circle(
    data: np.ndarray, level: float = 0.5
) -> tuple[int, int, float]:
    """Return (x, y, radius) for the circle contour enclosing image
    values above threshold relative level (.5 means FWHM)

    Args:
        data: Input data
        level: Relative level (default: 0.5)

    Returns:
        A tuple (x, y, radius)

    Raises:
        ValueError: No contour was found
    """
    # New shape model parameters introduce in scikit-image v0.26.0
    # (see: https://github.com/scikit-image/scikit-image/commit/6d6e7924cca105320690f716a14c5bd11055bf43):
    # In order to keep Python 3.9 compatibility, need to check  skimage.__version__
    # In skimage v0.26.0, for CircleModel and EllipseModel, params length change:
    # - CircleModel : xc and yc are combine in an array center
    # - EllipseModel : xc and yc are combine in a center array,
    # a and b are combine in a axis_lengths array
    # TODO scikit-image v0.26.0 shape model parameters : check future updates and Python
    # 3.9 compatibility
    skimage_version_minor = int(str(__version__).split(".")[1])
    data_th = data.copy()
    data_th[data <= get_absolute_level(data, level)] = 0.0
    contours = measure.find_contours(data_th)
    result = None
    max_radius = 1.0
    for contour in contours:
        if skimage_version_minor < 26:
            model = measure.CircleModel()
            if model.estimate(contour):
                yc, xc, radius = model.params
                if radius > max_radius:
                    result = (int(xc), int(yc), radius)
                    max_radius = radius
        else:
            model = measure.CircleModel.from_estimate(contour)
            if model:
                yc = model.center[1]
                xc = model.center[0]
                radius = model.radius
                if radius > max_radius:
                    result = (int(xc), int(yc), radius)
                    max_radius = radius
    if result is None:
        raise ValueError("No contour was found")
    return result
