# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Exposure computation module
---------------------------

This module provides tools for adjusting and analyzing image exposure and contrast.

Main features include:

- Histogram computation and equalization
- Contrast adjustment and normalization
- Logarithmic and gamma correction

Exposure processing improves the visual quality and interpretability of images,
especially under variable lighting conditions.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

import warnings

import guidata.dataset as gds
import numpy as np
from skimage import exposure

import sigima.enums
import sigima.tools.image
from sigima.config import _
from sigima.enums import ReplacementStrategyImage
from sigima.objects.image import ImageObj, ROI2DParam
from sigima.objects.signal import SignalObj
from sigima.proc.base import (
    ClipParam,
    HistogramParam,
    NormalizeParam,
    ReplaceSpecialValuesImageParam,
    new_signal_result,
)
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import (
    Wrap1to1Func,
    dst_1_to_1,
    dst_2_to_1,
    restore_data_outside_roi,
)
from sigima.tools.image import replace_values as rv2d

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "AdjustGammaParam",
    "AdjustLogParam",
    "AdjustSigmoidParam",
    "EqualizeAdaptHistParam",
    "EqualizeHistParam",
    "FlatFieldParam",
    "RescaleIntensityParam",
    "adjust_gamma",
    "adjust_log",
    "adjust_sigmoid",
    "clip",
    "equalize_adapthist",
    "equalize_hist",
    "flatfield",
    "histogram",
    "normalize",
    "offset_correction",
    "replace_special_values",
    "rescale_intensity",
]


class AdjustGammaParam(gds.DataSet):
    """Gamma adjustment parameters"""

    gamma = gds.FloatItem(
        _("Gamma"),
        default=1.0,
        min=0.0,
        help=_("Gamma correction factor (higher values give more contrast)."),
    )
    gain = gds.FloatItem(
        _("Gain"),
        default=1.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )


@computation_function()
def adjust_gamma(src: ImageObj, p: AdjustGammaParam) -> ImageObj:
    """Gamma correction with :py:func:`skimage.exposure.adjust_gamma`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "adjust_gamma", f"gamma={p.gamma}, gain={p.gain}")
    dst.data = exposure.adjust_gamma(src.data, gamma=p.gamma, gain=p.gain)
    restore_data_outside_roi(dst, src)
    return dst


class AdjustLogParam(gds.DataSet):
    """Logarithmic adjustment parameters"""

    gain = gds.FloatItem(
        _("Gain"),
        default=1.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )
    inv = gds.BoolItem(
        _("Inverse"),
        default=False,
        help=_("If True, apply inverse logarithmic transformation."),
    )


@computation_function()
def adjust_log(src: ImageObj, p: AdjustLogParam) -> ImageObj:
    """Compute log correction with :py:func:`skimage.exposure.adjust_log`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "adjust_log", f"gain={p.gain}, inv={p.inv}")
    dst.data = exposure.adjust_log(src.data, gain=p.gain, inv=p.inv)
    restore_data_outside_roi(dst, src)
    return dst


class AdjustSigmoidParam(gds.DataSet):
    """Sigmoid adjustment parameters"""

    cutoff = gds.FloatItem(
        _("Cutoff"),
        default=0.5,
        min=0.0,
        max=1.0,
        help=_("Cutoff value (higher values give more contrast)."),
    )
    gain = gds.FloatItem(
        _("Gain"),
        default=10.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )
    inv = gds.BoolItem(
        _("Inverse"),
        default=False,
        help=_("If True, apply inverse sigmoid transformation."),
    )


@computation_function()
def adjust_sigmoid(src: ImageObj, p: AdjustSigmoidParam) -> ImageObj:
    """Compute sigmoid correction with :py:func:`skimage.exposure.adjust_sigmoid`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(
        src, "adjust_sigmoid", f"cutoff={p.cutoff}, gain={p.gain}, inv={p.inv}"
    )
    dst.data = exposure.adjust_sigmoid(
        src.data, cutoff=p.cutoff, gain=p.gain, inv=p.inv
    )
    restore_data_outside_roi(dst, src)
    return dst


class RescaleIntensityParam(gds.DataSet):
    """Intensity rescaling parameters"""

    _dtype_list = ["image", "dtype"] + ImageObj.get_valid_dtypenames()
    in_range = gds.ChoiceItem(
        _("Input range"),
        list(zip(_dtype_list, _dtype_list)),
        default="image",
        help=_(
            "Min and max intensity values of input image ('image' refers to input "
            "image min/max levels, 'dtype' refers to input image data type range)."
        ),
    )
    out_range = gds.ChoiceItem(
        _("Output range"),
        list(zip(_dtype_list, _dtype_list)),
        default="dtype",
        help=_(
            "Min and max intensity values of output image  ('image' refers to input "
            "image min/max levels, 'dtype' refers to input image data type range).."
        ),
    )


@computation_function()
def rescale_intensity(src: ImageObj, p: RescaleIntensityParam) -> ImageObj:
    """Rescale image intensity levels
    with :py:func:`skimage.exposure.rescale_intensity`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(
        src,
        "rescale_intensity",
        f"in_range={p.in_range}, out_range={p.out_range}",
    )
    dst.data = exposure.rescale_intensity(
        src.data, in_range=p.in_range, out_range=p.out_range
    )
    restore_data_outside_roi(dst, src)
    return dst


class EqualizeHistParam(gds.DataSet):
    """Histogram equalization parameters"""

    nbins = gds.IntItem(
        _("Number of bins"),
        min=1,
        default=256,
        help=_("Number of bins for image histogram."),
    )


@computation_function()
def equalize_hist(src: ImageObj, p: EqualizeHistParam) -> ImageObj:
    """Histogram equalization with :py:func:`skimage.exposure.equalize_hist`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "equalize_hist", f"nbins={p.nbins}")
    dst.data = exposure.equalize_hist(src.data, nbins=p.nbins)
    restore_data_outside_roi(dst, src)
    return dst


class EqualizeAdaptHistParam(EqualizeHistParam):
    """Adaptive histogram equalization parameters"""

    clip_limit = gds.FloatItem(
        _("Clipping limit"),
        default=0.01,
        min=0.0,
        max=1.0,
        help=_("Clipping limit (higher values give more contrast)."),
    )


@computation_function()
def equalize_adapthist(src: ImageObj, p: EqualizeAdaptHistParam) -> ImageObj:
    """Adaptive histogram equalization
    with :py:func:`skimage.exposure.equalize_adapthist`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(
        src, "equalize_adapthist", f"nbins={p.nbins}, clip_limit={p.clip_limit}"
    )
    dst.data = exposure.equalize_adapthist(
        src.data, clip_limit=p.clip_limit, nbins=p.nbins
    )
    restore_data_outside_roi(dst, src)
    return dst


class FlatFieldParam(gds.DataSet):
    """Flat-field parameters"""

    threshold = gds.FloatItem(_("Threshold"), default=0.0)


@computation_function()
def flatfield(src1: ImageObj, src2: ImageObj, p: FlatFieldParam) -> ImageObj:
    """Compute flat field correction with :py:func:`sigima.tools.image.flatfield`

    Args:
        src1: raw data image object
        src2: flat field image object
        p: flat field parameters

    Returns:
        Output image object
    """
    dst = dst_2_to_1(src1, src2, "flatfield", f"threshold={p.threshold}")
    dst.data = sigima.tools.image.flatfield(src1.data, src2.data, p.threshold)
    restore_data_outside_roi(dst, src1)
    return dst


# MARK: compute_1_to_1 functions -------------------------------------------------------
# Functions with 1 input image and 1 output image
# --------------------------------------------------------------------------------------


@computation_function()
def normalize(src: ImageObj, p: NormalizeParam) -> ImageObj:
    """
    Normalize image data depending on its maximum,
    with :py:func:`sigima.tools.image.normalize`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    method: sigima.enums.NormalizationMethod = p.method
    dst = dst_1_to_1(src, "normalize", suffix=f"ref={method.value}")
    dst.data = sigima.tools.image.normalize(src.data, method)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def histogram(src: ImageObj, p: HistogramParam) -> SignalObj:
    """Compute histogram of the image data, with :py:func:`numpy.histogram`

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the histogram
    """
    data = src.get_masked_view().compressed()
    suffix = p.get_suffix(data)  # Also updates p.lower and p.upper
    y, bin_edges = np.histogram(data, bins=p.bins, range=(p.lower, p.upper))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    dst = new_signal_result(
        src,
        "histogram",
        suffix=suffix,
        units=(src.zunit, ""),
        labels=(src.zlabel, _("Counts")),
    )
    dst.set_xydata(x, y)
    dst.set_metadata_option("shade", 0.5)
    dst.set_metadata_option("curvestyle", "Steps")
    return dst


@computation_function()
def clip(src: ImageObj, p: ClipParam) -> ImageObj:
    """Apply clipping with :py:func:`numpy.clip`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.clip, a_min=p.lower, a_max=p.upper)(src)


@computation_function()
def offset_correction(src: ImageObj, p: ROI2DParam) -> ImageObj:
    """Apply offset correction

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "offset_correction", p.get_suffix())
    dst.data = src.data - np.nanmean(p.get_data(src))
    restore_data_outside_roi(dst, src)
    return dst


def _apply_image_strategy(
    data: np.ndarray,
    mask: np.ndarray,
    strategy: ReplacementStrategyImage,
    neighbor_size: int,
    constant_value: float,
) -> np.ndarray:
    """Apply a single replacement strategy to masked positions in an image.

    Args:
        data: 2-D data array (may be modified in place).
        mask: boolean mask of positions to replace.
        strategy: replacement strategy to apply.
        neighbor_size: neighborhood radius for neighbor-based strategies.
        constant_value: value used for the CONSTANT strategy.

    Returns:
        Data array with replacements applied.
    """
    s = strategy
    if not np.any(mask) or s == ReplacementStrategyImage.NONE:
        return data

    if s == ReplacementStrategyImage.ZERO:
        rv2d.replace_with_fixed_2d(data, mask, 0.0)
    elif s == ReplacementStrategyImage.CONSTANT:
        rv2d.replace_with_fixed_2d(data, mask, constant_value)
    elif s == ReplacementStrategyImage.MIN:
        rv2d.replace_with_stat_2d(data, mask, "min")
    elif s == ReplacementStrategyImage.MAX:
        rv2d.replace_with_stat_2d(data, mask, "max")
    elif s == ReplacementStrategyImage.MEAN:
        rv2d.replace_with_stat_2d(data, mask, "mean")
    elif s == ReplacementStrategyImage.MEDIAN:
        rv2d.replace_with_stat_2d(data, mask, "median")
    elif s == ReplacementStrategyImage.NEIGHBOR_MIN:
        rv2d.neighbor_replace_2d(data, mask, neighbor_size, "min")
    elif s == ReplacementStrategyImage.NEIGHBOR_MAX:
        rv2d.neighbor_replace_2d(data, mask, neighbor_size, "max")
    elif s == ReplacementStrategyImage.NEIGHBOR_MEAN:
        rv2d.neighbor_replace_2d(data, mask, neighbor_size, "mean")
    elif s == ReplacementStrategyImage.NEIGHBOR_MEDIAN:
        rv2d.neighbor_replace_2d(data, mask, neighbor_size, "median")
    else:
        raise ValueError(f"Unsupported image replacement strategy: {s}")
    return data


@computation_function()
def replace_special_values(
    src: ImageObj, p: ReplaceSpecialValuesImageParam
) -> ImageObj:
    """Replace NaN, +Inf and -Inf values in an image.

    Each target (NaN, +Inf, -Inf) is treated independently with its own strategy.

    Args:
        src: input image object.
        p: parameters specifying the strategy for each target.

    Returns:
        Output image object with special values replaced.
    """
    strategies = []
    if p.nan_strategy != ReplacementStrategyImage.NONE:
        strategies.append(f"NaN→{p.nan_strategy.value}")
    if p.posinf_strategy != ReplacementStrategyImage.NONE:
        strategies.append(f"+Inf→{p.posinf_strategy.value}")
    if p.neginf_strategy != ReplacementStrategyImage.NONE:
        strategies.append(f"-Inf→{p.neginf_strategy.value}")
    suffix = ", ".join(strategies) if strategies else "none"

    dst = dst_1_to_1(src, "replace_special_values", suffix)

    if np.issubdtype(src.data.dtype, np.integer):
        warnings.warn(
            _(
                "Replace special values is not applicable to integer images "
                "because they cannot contain NaN or infinite values."
            ),
            stacklevel=2,
        )
        return dst

    data = dst.data.copy()

    for target, strategy, const_val, neigh_size in (
        (np.isnan, p.nan_strategy, p.nan_constant_value, p.nan_neighbor_size),
        (
            np.isposinf,
            p.posinf_strategy,
            p.posinf_constant_value,
            p.posinf_neighbor_size,
        ),
        (
            np.isneginf,
            p.neginf_strategy,
            p.neginf_constant_value,
            p.neginf_neighbor_size,
        ),
    ):
        mask = target(data)
        data = _apply_image_strategy(data, mask, strategy, neigh_size, const_val)

    dst.data = data
    restore_data_outside_roi(dst, src)
    return dst
