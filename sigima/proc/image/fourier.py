# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Fourier computation module
--------------------------

This module implements Fourier transform operations and related spectral analysis tools
for images.

Main features include:

- Forward and inverse Fast Fourier Transform (FFT)
- Magnitude and phase spectrum calculation
- Power spectral density (PSD) computation

Fourier analysis is commonly used for frequency-domain filtering and periodicity
analysis in images.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

import sigima.tools.image
from sigima.config import _
from sigima.objects.image import ImageObj
from sigima.proc.base import FFTParam, SpectrumParam
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import Wrap1to1Func, dst_1_to_1

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "fft",
    "ifft",
    "magnitude_spectrum",
    "phase_spectrum",
    "psd",
]


@computation_function()
def fft(src: ImageObj, p: FFTParam | None = None) -> ImageObj:
    """Compute FFT with :py:func:`sigima.tools.image.fft2d`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    if p is None:
        p = FFTParam()
    dst = dst_1_to_1(src, "fft")
    dst.data = sigima.tools.image.fft2d(src.data, shift=p.shift)
    dst.save_attr_to_metadata("xunit", "")
    dst.save_attr_to_metadata("yunit", "")
    dst.save_attr_to_metadata("zunit", "")
    dst.save_attr_to_metadata("xlabel", _("Frequency"))
    dst.save_attr_to_metadata("ylabel", _("Frequency"))
    return dst


@computation_function()
def ifft(src: ImageObj, p: FFTParam | None = None) -> ImageObj:
    """Compute inverse FFT with :py:func:`sigima.tools.image.ifft2d`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    if p is None:
        p = FFTParam()
    dst = dst_1_to_1(src, "ifft")
    dst.data = sigima.tools.image.ifft2d(src.data, shift=p.shift)
    dst.restore_attr_from_metadata("xunit", "")
    dst.restore_attr_from_metadata("yunit", "")
    dst.restore_attr_from_metadata("zunit", "")
    dst.restore_attr_from_metadata("xlabel", "")
    dst.restore_attr_from_metadata("ylabel", "")
    return dst


@computation_function()
def magnitude_spectrum(src: ImageObj, p: SpectrumParam | None = None) -> ImageObj:
    """Compute magnitude spectrum
    with :py:func:`sigima.tools.image.magnitude_spectrum`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    decibel = p is not None and p.decibel
    dst = dst_1_to_1(src, "magnitude_spectrum", f"dB={decibel}")
    dst.data = sigima.tools.image.magnitude_spectrum(src.data, log_scale=decibel)
    dst.xunit = dst.yunit = dst.zunit = ""
    dst.xlabel = dst.ylabel = _("Frequency")
    return dst


@computation_function()
def phase_spectrum(src: ImageObj) -> ImageObj:
    """Compute phase spectrum
    with :py:func:`sigima.tools.image.phase_spectrum`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = Wrap1to1Func(sigima.tools.image.phase_spectrum)(src)
    dst.xunit = dst.yunit = dst.zunit = ""
    dst.xlabel = dst.ylabel = _("Frequency")
    return dst


@computation_function()
def psd(src: ImageObj, p: SpectrumParam | None = None) -> ImageObj:
    """Compute power spectral density
    with :py:func:`sigima.tools.image.psd`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    decibel = p is not None and p.decibel
    dst = dst_1_to_1(src, "psd", f"dB={decibel}")
    dst.data = sigima.tools.image.psd(src.data, log_scale=decibel)
    dst.xunit = dst.yunit = dst.zunit = ""
    dst.xlabel = dst.ylabel = _("Frequency")
    return dst
