# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Restoration computation module
------------------------------

This module provides image restoration techniques, such as
denoising, inpainting, and deblurring. These methods aim to recover
the original quality of images by removing artifacts, noise, or
distortions.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

from typing import TYPE_CHECKING

import guidata.dataset as gds
import numpy as np
import pywt
from skimage import morphology, restoration

from sigima.config import _
from sigima.enums import ShrinkageMethod, ThresholdMethod, WaveletMode
from sigima.objects.image import ImageObj, ROI2DParam
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import Wrap1to1Func, dst_1_to_1, restore_data_outside_roi

if TYPE_CHECKING:
    import sigima.params


# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "DenoiseBilateralParam",
    "DenoiseTVParam",
    "DenoiseWaveletParam",
    "denoise_bilateral",
    "denoise_tophat",
    "denoise_tv",
    "denoise_wavelet",
    "erase",
]


class DenoiseTVParam(gds.DataSet):
    """Total Variation denoising parameters"""

    weight = gds.FloatItem(
        _("Denoising weight"),
        default=0.1,
        min=0,
        nonzero=True,
        help=_(
            "The greater weight, the more denoising "
            "(at the expense of fidelity to input)."
        ),
    )
    eps = gds.FloatItem(
        "Epsilon",
        default=0.0002,
        min=0,
        nonzero=True,
        help=_(
            "Relative difference of the value of the cost function that "
            "determines the stop criterion. The algorithm stops when: "
            "(E_(n-1) - E_n) < eps * E_0"
        ),
    )
    max_num_iter = gds.IntItem(
        _("Max. iterations"),
        default=200,
        min=0,
        nonzero=True,
        help=_("Maximal number of iterations used for the optimization"),
    )


@computation_function()
def denoise_tv(src: ImageObj, p: DenoiseTVParam) -> ImageObj:
    """Compute Total Variation denoising
    with :py:func:`skimage.restoration.denoise_tv_chambolle`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap1to1Func(
        restoration.denoise_tv_chambolle,
        weight=p.weight,
        eps=p.eps,
        max_num_iter=p.max_num_iter,
        func_name="denoise_tv",
    )(src)


class DenoiseBilateralParam(gds.DataSet):
    """Bilateral filter denoising parameters"""

    sigma_spatial = gds.FloatItem(
        "σ<sub>spatial</sub>",
        default=1.0,
        min=0,
        nonzero=True,
        unit="pixels",
        help=_(
            "Standard deviation for range distance. "
            "A larger value results in averaging of pixels "
            "with larger spatial differences."
        ),
    )
    mode = gds.ChoiceItem(_("Mode"), WaveletMode, default=WaveletMode.CONSTANT)
    cval = gds.FloatItem(
        "cval",
        default=0.0,
        help=_(
            "Used in conjunction with mode 'constant', "
            "the value outside the image boundaries."
        ),
    )


@computation_function()
def denoise_bilateral(src: ImageObj, p: DenoiseBilateralParam) -> ImageObj:
    """Compute bilateral filter denoising
    with :py:func:`skimage.restoration.denoise_bilateral`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap1to1Func(
        restoration.denoise_bilateral,
        sigma_spatial=p.sigma_spatial,
        mode=p.mode,
        cval=p.cval,
    )(src)


class DenoiseWaveletParam(gds.DataSet):
    """Wavelet denoising parameters"""

    wavelets = pywt.wavelist()
    wavelet = gds.ChoiceItem(
        _("Wavelet"), list(zip(wavelets, wavelets)), default="sym9"
    )
    mode = gds.ChoiceItem(_("Mode"), ThresholdMethod, default=ThresholdMethod.SOFT)
    method = gds.ChoiceItem(
        _("Method"), ShrinkageMethod, default=ShrinkageMethod.VISU_SHRINK
    )


@computation_function()
def denoise_wavelet(src: ImageObj, p: DenoiseWaveletParam) -> ImageObj:
    """Compute Wavelet denoising
    with :py:func:`skimage.restoration.denoise_wavelet`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap1to1Func(
        restoration.denoise_wavelet, wavelet=p.wavelet, mode=p.mode, method=p.method
    )(src)


@computation_function()
def denoise_tophat(src: ImageObj, p: sigima.params.MorphologyParam) -> ImageObj:
    """Denoise using White Top-Hat
    with :py:func:`skimage.morphology.white_tophat`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "denoise_tophat", f"radius={p.radius}")
    dst.data = src.data - morphology.white_tophat(src.data, morphology.disk(p.radius))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def erase(src: ImageObj, p: ROI2DParam | list[ROI2DParam]) -> ImageObj:
    """Erase an area of the image using the mean value of the image.

    .. note::

        The erased area is defined by a region of interest (ROI) parameter set.
        This ROI must not be mistaken with the ROI of the image object. If the
        image object has a ROI, it is not used in this processing, except to
        restore the data outside the ROI (as in all other processing).

    Args:
        src: input image object
        p: parameters defining the area to erase (region of interest)

    Returns:
        Output image object
    """
    params = [p] if isinstance(p, ROI2DParam) else p
    suffix = None
    if len(params) == 1:
        suffix = params[0].get_suffix()
    dst = dst_1_to_1(src, "erase", suffix)
    for param in params:
        value = np.nanmean(param.get_data(src))
        erase_roi = param.to_single_roi(src)
        mask = erase_roi.to_mask(src)
        dst.data[~mask] = value
    restore_data_outside_roi(dst, src)
    return dst
