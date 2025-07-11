# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Restoration computation module
------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

import guidata.dataset as gds
import pywt
from skimage import morphology, restoration

from sigima.config import _
from sigima.objects.image import ImageObj
from sigima.proc import computation_function
from sigima.proc.base import dst_1_to_1
from sigima.proc.image.base import Wrap1to1Func, restore_data_outside_roi
from sigima.proc.image.morphology import MorphologyParam


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
    modes = ("constant", "edge", "symmetric", "reflect", "wrap")
    mode = gds.ChoiceItem(_("Mode"), list(zip(modes, modes)), default="constant")
    cval = gds.FloatItem(
        "cval",
        default=0,
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
    modes = ("soft", "hard")
    mode = gds.ChoiceItem(_("Mode"), list(zip(modes, modes)), default="soft")
    methods = ("BayesShrink", "VisuShrink")
    method = gds.ChoiceItem(
        _("Method"), list(zip(methods, methods)), default="VisuShrink"
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
def denoise_tophat(src: ImageObj, p: MorphologyParam) -> ImageObj:
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
