# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Mathematical Operations Module
------------------------------

This module implements mathematical operations on images, such as inversion,
absolute value, real/imaginary part extraction, type casting, and exponentiation.

Main features include:

- Pixel-wise mathematical transformations (e.g., log, exp, abs, real, imag, log10)
- Type casting and other value-level operations

These functions enable flexible manipulation of image data at the value level.
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

import sigima.tools.image
from sigima.config import _
from sigima.config import options as sigima_options
from sigima.enums import AngleUnit
from sigima.objects.image import ImageObj
from sigima.proc.base import AngleUnitParam, PhaseParam
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import (
    Wrap1to1Func,
    dst_1_to_1,
    dst_2_to_1,
    restore_data_outside_roi,
)
from sigima.tools import coordinates
from sigima.tools.datatypes import clip_astype

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "DataTypeIParam",
    "Log10ZPlusNParam",
    "absolute",
    "absolute",
    "astype",
    "complex_from_magnitude_phase",
    "complex_from_real_imag",
    "convolution",
    "deconvolution",
    "exp",
    "exp",
    "imag",
    "imag",
    "inverse",
    "inverse",
    "log10",
    "log10",
    "log10_z_plus_n",
    "phase",
    "real",
    "real",
]


@computation_function()
def inverse(src: ImageObj) -> ImageObj:
    """Compute the inverse of an image and return the new result image object

    Args:
        src: input image object

    Returns:
        Result image object 1 / **src** (new object)
    """
    dst = dst_1_to_1(src, "inverse")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dst.data = np.reciprocal(src.data, dtype=float)
        dst.data[np.isinf(dst.data)] = np.nan
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def absolute(src: ImageObj) -> ImageObj:
    """Compute absolute value with :py:data:`numpy.absolute`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.absolute)(src)


@computation_function()
def real(src: ImageObj) -> ImageObj:
    """Compute real part with :py:func:`numpy.real`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.real)(src)


@computation_function()
def imag(src: ImageObj) -> ImageObj:
    """Compute imaginary part with :py:func:`numpy.imag`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.imag)(src)


@computation_function()
def phase(src: ImageObj, p: PhaseParam) -> ImageObj:
    """Compute the phase (argument) of a complex image.

    The function uses :py:func:`numpy.angle` to compute the argument and
    :py:func:`numpy.unwrap` to unwrap it.

    Args:
        src: Input image object.
        p: Phase parameters.

    Returns:
        Image object containing the phase, optionally unwrapped.
    """
    suffix = "unwrap" if p.unwrap else ""
    dst = dst_1_to_1(src, "phase", suffix)
    data = src.get_data()
    argument = np.angle(data)
    if p.unwrap:
        argument = np.unwrap(argument)
    if p.unit == AngleUnit.DEGREE:
        argument = np.rad2deg(argument)
    dst.data = argument
    dst.zunit = p.unit
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def complex_from_magnitude_phase(
    src1: ImageObj, src2: ImageObj, p: AngleUnitParam
) -> ImageObj:
    """Combine magnitude and phase images into a complex image.

    .. warning::

        This function assumes that the input images have the same dimensions.

    Args:
        src1: Magnitude (module) image.
        src2: Phase (argument) image.
        p: Parameters (provides unit for the phase).

    Returns:
        Image object with complex-valued z.
    """
    dst = dst_2_to_1(src1, src2, "mag_phase")
    assert p.unit is not None
    dst.data = coordinates.polar_to_complex(src1.data, src2.data, unit=p.unit)
    return dst


@computation_function()
def complex_from_real_imag(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Combine two real images into a complex image using real + i * imag.

    .. warning::

        This function assumes that the input images have the same dimensions and are
        properly aligned.

    Args:
        src1: Real part image.
        src2: Imaginary part image.

    Returns:
        Image object with complex-valued z.

    Raises:
        ValueError: If the x or y coordinates of the two images are not the same.
    """
    dst = dst_2_to_1(src1, src2, "real_imag")
    assert src1.data is not None
    assert src2.data is not None
    dst.data = src1.data + 1j * src2.data
    return dst


@computation_function()
def convolution(src: ImageObj, kernel: ImageObj) -> ImageObj:
    """Convolve an image with a kernel.

    The kernel should ideally be smaller than the input image and centered.

    Args:
        src: Input image object.
        kernel: Kernel image object.

    Returns:
        Output image object.

    Notes:
        The behavior of kernel normalization is controlled by the global configuration
        option ``sigima.config.options.auto_normalize_kernel``.
    """
    # Get configuration option for kernel normalization
    normalize_kernel = sigima_options.auto_normalize_kernel.get()

    dst = dst_2_to_1(src, kernel, "⊛")
    dst.data = sigima.tools.image.convolve(
        src.data,
        kernel.data,
        normalize_kernel_flag=normalize_kernel,
    )
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def deconvolution(src: ImageObj, kernel: ImageObj) -> ImageObj:
    """Deconvolve a kernel from an image using Fast Fourier Transform (FFT).

    Args:
        src: Input image object.
        kernel: Kernel image object.

    Returns:
        Output image object.

    Notes:
        The behavior of kernel normalization is controlled by the global configuration
        option ``sigima.config.options.auto_normalize_kernel``.
    """
    # Get configuration option for kernel normalization
    normalize_kernel = sigima_options.auto_normalize_kernel.get()

    dst = dst_2_to_1(src, kernel, "⊛⁻¹")
    dst.data = sigima.tools.image.deconvolve(
        src.data,
        kernel.data,
        normalize_kernel_flag=normalize_kernel,
    )
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def log10(src: ImageObj) -> ImageObj:
    """Compute log10 with :py:data:`numpy.log10`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.log10)(src)


@computation_function()
def exp(src: ImageObj) -> ImageObj:
    """Compute exponential with :py:data:`numpy.exp`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.exp)(src)


class Log10ZPlusNParam(gds.DataSet):
    """Log10(z+n) parameters"""

    n = gds.FloatItem("n")


@computation_function()
def log10_z_plus_n(src: ImageObj, p: Log10ZPlusNParam) -> ImageObj:
    """Compute log10(z+n) with :py:data:`numpy.log10`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "log10_z_plus_n", f"n={p.n}")
    dst.data = np.log10(src.data + p.n)
    restore_data_outside_roi(dst, src)
    return dst


class DataTypeIParam(gds.DataSet):
    """Convert image data type parameters"""

    dtype_str = gds.ChoiceItem(
        _("Destination data type"),
        list(zip(ImageObj.get_valid_dtypenames(), ImageObj.get_valid_dtypenames())),
        help=_("Output image data type."),
    )


@computation_function()
def astype(src: ImageObj, p: DataTypeIParam) -> ImageObj:
    """Convert image data type with :py:func:`sigima.tools.datatypes.clip_astype`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "clip_astype", p.dtype_str)
    dst.data = clip_astype(src.data, p.dtype_str)
    return dst
