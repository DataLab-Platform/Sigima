# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Filtering computation module.
--------------------------------

This module provides spatial and frequency-based filtering operations for images.
Filtering functions are essential for enhancing image quality and removing noise.

Main features include:
    * Gaussian, median, moving average and Wiener filters
    * Butterworth and frequency domain Gaussian filters.

Filtering functions are essential for enhancing image quality
and removing noise prior to further analysis.
"""

# pylint: disable=invalid-name  # Allows short names like x, y...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported in the
#   `sigima.params` module.
# - All functions decorated with `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

import guidata.dataset as gds  # type: ignore[import]
import scipy.ndimage as spi  # type: ignore[import]
import scipy.signal as sps  # type: ignore[import]
from skimage import filters  # type: ignore[import]

import sigima.tools.image
from sigima.config import _
from sigima.objects.image import ImageObj
from sigima.proc.base import (
    GaussianParam,
    MovingAverageParam,
    MovingMedianParam,
)
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import Wrap1to1Func, dst_1_to_1, restore_data_outside_roi

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "ButterworthParam",
    "GaussianFreqFilterParam",
    "butterworth",
    "gaussian_filter",
    "gaussian_freq_filter",
    "moving_average",
    "moving_median",
    "wiener",
]

# MARK: Noise reduction filters


@computation_function()
def gaussian_filter(src: ImageObj, p: GaussianParam) -> ImageObj:
    """Compute gaussian filter with :py:func:`scipy.ndimage.gaussian_filter`.

    Args:
        src: Input image object.
        p: Parameters.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(spi.gaussian_filter, sigma=p.sigma)(src)


@computation_function()
def moving_average(src: ImageObj, p: MovingAverageParam) -> ImageObj:
    """Compute moving average with :py:func:`scipy.ndimage.uniform_filter`.

    Args:
        src: Input image object.
        p: Parameters.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(
        spi.uniform_filter, size=p.n, mode=p.mode, func_name="moving_average"
    )(src)


@computation_function()
def moving_median(src: ImageObj, p: MovingMedianParam) -> ImageObj:
    """Compute moving median with :py:func:`scipy.ndimage.median_filter`.

    Args:
        src: Input image object.
        p: Parameters.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(
        spi.median_filter, size=p.n, mode=p.mode, func_name="moving_median"
    )(src)


@computation_function()
def wiener(src: ImageObj) -> ImageObj:
    """Compute Wiener filter with :py:func:`scipy.signal.wiener`

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(sps.wiener)(src)


class ButterworthParam(gds.DataSet):
    """Butterworth filter parameters."""

    cut_off = gds.FloatItem(
        _("Cut-off frequency ratio"),
        default=0.005,
        min=0.0,
        max=0.5,
        help=_("Cut-off frequency ratio"),
    )
    high_pass = gds.BoolItem(
        _("High-pass filter"),
        default=False,
        help=_("If True, apply high-pass filter instead of low-pass"),
    )
    order = gds.IntItem(
        _("Order"),
        default=2,
        min=1,
        help=_("Order of the Butterworth filter"),
    )


# MARK: Frequency filters


@computation_function()
def butterworth(src: ImageObj, p: ButterworthParam) -> ImageObj:
    """Compute Butterworth filter with :py:func:`skimage.filters.butterworth`.

    Args:
        src: Input image object.
        p: Parameters.

    Returns:
        Output image object.
    """
    dst = dst_1_to_1(
        src,
        "butterworth",
        f"cut_off={p.cut_off:.3f}, order={p.order}, high_pass={p.high_pass}",
    )
    dst.data = filters.butterworth(src.data, p.cut_off, p.high_pass, p.order)
    restore_data_outside_roi(dst, src)
    return dst


class GaussianFreqFilterParam(GaussianParam):
    """Parameters for Gaussian filter applied in the frequency domain."""

    sigma = gds.FloatItem(
        "σ",
        default=1.0,
        unit="pixel⁻¹",
        min=0.0,
        help=_("Standard deviation of the Gaussian filter"),
    )
    f0 = gds.FloatItem(
        _("Center frequency"),
        default=1.0,
        unit="pixel⁻¹",
        min=0.0,
        help=_("Center frequency of the Gaussian filter"),
    )
    sigma = gds.FloatItem(
        "σ",
        default=0.5,
        unit="pixels⁻¹",
        min=0.0,
        help=_("Standard deviation of the Gaussian filter"),
    )
    ifft_result_type = gds.ChoiceItem(
        _("Inverse FFT result"),
        (("real", _("Real part")), ("abs", _("Absolute value"))),
        default="real",
        help=_("How to return the inverse FFT result"),
    )


@computation_function()
def gaussian_freq_filter(src: ImageObj, p: GaussianFreqFilterParam) -> ImageObj:
    """Apply a Gaussian filter in the frequency domain.

    Args:
        src: Source image object.
        p: Parameters.

    Returns:
        Output image object.
    """
    dst = dst_1_to_1(
        src,
        "frequency_domain_gaussian_filter",
        f"sigma={p.sigma:.3f}, f0={p.f0:.3f}",
    )
    dst.data = sigima.tools.image.gaussian_freq_filter(src.data, p.f0, p.sigma)
    restore_data_outside_roi(dst, src)
    return dst
