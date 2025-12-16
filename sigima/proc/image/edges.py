# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Edge detection computation module.
----------------------------------

This module implements edge detection computation functions, enabling the identification
of boundaries in an image. Edge detection may be used for image segmentation, shape
analysis, and feature extraction. Methods rely on detection of significant transitions
in pixel intensity.

Available filters (in alphabetical order):
    * canny: Canny edge detector
    * farid: Farid filter
    * farid_h: Horizontal Farid filter
    * farid_v: Vertical Farid filter
    * laplace: Laplace filter
    * prewitt: Prewitt filter
    * prewitt_h: Horizontal Prewitt filter
    * prewitt_v: Vertical Prewitt filter
    * roberts: Roberts filter
    * scharr: Scharr filter
    * scharr_h: Horizontal Scharr filter
    * scharr_v: Vertical Scharr filter
    * sobel: Sobel filter
    * sobel_h: Horizontal Sobel filter
    * sobel_v: Vertical Sobel filter

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
from skimage import feature, filters  # type: ignore[import]
from skimage.util import img_as_ubyte  # type: ignore[import]

from sigima.config import _
from sigima.enums import FilterMode
from sigima.objects.image import ImageObj
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import Wrap1to1Func, dst_1_to_1, restore_data_outside_roi

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "CannyParam",
    "canny",
    "farid",
    "farid_h",
    "farid_v",
    "laplace",
    "prewitt",
    "prewitt_h",
    "prewitt_v",
    "roberts",
    "scharr",
    "scharr_h",
    "scharr_v",
    "sobel",
    "sobel_h",
    "sobel_v",
]


class CannyParam(gds.DataSet):
    """Canny filter parameters."""

    sigma = gds.FloatItem(
        "Sigma",
        default=1.0,
        unit="pixels",
        min=0.0,
        nonzero=True,
        help=_("Standard deviation of the Gaussian filter."),
    )
    low_threshold = gds.FloatItem(
        _("Low threshold"),
        default=0.1,
        min=0.0,
        help=_("Lower bound for hysteresis thresholding (linking edges)."),
    )
    high_threshold = gds.FloatItem(
        _("High threshold"),
        default=0.9,
        min=0.0,
        help=_("Upper bound for hysteresis thresholding (linking edges)."),
    )
    use_quantiles = gds.BoolItem(
        _("Use quantiles"),
        default=True,
        help=_(
            "If True then treat low_threshold and high_threshold as quantiles "
            "of the edge magnitude image, rather than absolute edge magnitude "
            "values. If True then the thresholds must be in the range [0, 1]."
        ),
    )
    mode = gds.ChoiceItem(_("Mode"), FilterMode, default=FilterMode.CONSTANT)
    cval = gds.FloatItem(
        "cval",
        default=0.0,
        help=_("Value to fill past edges of input if mode is constant."),
    )


@computation_function()
def canny(src: ImageObj, p: CannyParam) -> ImageObj:
    """Compute Canny filter with :py:func:`skimage.feature.canny`.

    Args:
        src: Input image object.
        p: Parameters.

    Returns:
        Output image object.
    """
    dst = dst_1_to_1(
        src,
        "canny",
        f"sigma={p.sigma}, low_threshold={p.low_threshold}, "
        f"high_threshold={p.high_threshold}, use_quantiles={p.use_quantiles}, "
        f"mode={p.mode}, cval={p.cval}",
    )
    dst.data = img_as_ubyte(
        feature.canny(
            src.data,
            sigma=p.sigma,
            low_threshold=p.low_threshold,
            high_threshold=p.high_threshold,
            use_quantiles=p.use_quantiles,
            mode=p.mode,
            cval=p.cval,
        )
    )
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def farid(src: ImageObj) -> ImageObj:
    """Compute Farid filter with :py:func:`skimage.filters.farid`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.farid)(src)


@computation_function()
def farid_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Farid filter with :py:func:`skimage.filters.farid_h`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.farid_h)(src)


@computation_function()
def farid_v(src: ImageObj) -> ImageObj:
    """Compute vertical Farid filter with :py:func:`skimage.filters.farid_v`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.farid_v)(src)


@computation_function()
def laplace(src: ImageObj) -> ImageObj:
    """Compute Laplace filter with :py:func:`skimage.filters.laplace`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.laplace)(src)


@computation_function()
def prewitt(src: ImageObj) -> ImageObj:
    """Compute Prewitt filter with :py:func:`skimage.filters.prewitt`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.prewitt)(src)


@computation_function()
def prewitt_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Prewitt filter with :py:func:`skimage.filters.prewitt_h`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.prewitt_h)(src)


@computation_function()
def prewitt_v(src: ImageObj) -> ImageObj:
    """Compute vertical Prewitt filter with :py:func:`skimage.filters.prewitt_v`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.prewitt_v)(src)


@computation_function()
def roberts(src: ImageObj) -> ImageObj:
    """Compute Roberts filter with :py:func:`skimage.filters.roberts`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.roberts)(src)


@computation_function()
def scharr(src: ImageObj) -> ImageObj:
    """Compute Scharr filter with :py:func:`skimage.filters.scharr`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.scharr)(src)


@computation_function()
def scharr_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Scharr filter with :py:func:`skimage.filters.scharr_h`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.scharr_h)(src)


@computation_function()
def sobel(src: ImageObj) -> ImageObj:
    """Compute Sobel filter with :py:func:`skimage.filters.sobel`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.sobel)(src)


@computation_function()
def sobel_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Sobel filter with :py:func:`skimage.filters.sobel_h`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.sobel_h)(src)


@computation_function()
def sobel_v(src: ImageObj) -> ImageObj:
    """Compute vertical Sobel filter with :py:func:`skimage.filters.sobel_v`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.sobel_v)(src)


@computation_function()
def scharr_v(src: ImageObj) -> ImageObj:
    """Compute vertical Scharr filter with :py:func:`skimage.filters.scharr_v`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    return Wrap1to1Func(filters.scharr_v)(src)
