# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Arithmetic computation module
-----------------------------

This module provides arithmetic operations for images, such as pixel-wise addition,
subtraction, multiplication, division, as well as operations with constants
and combined arithmetic formulas.

Main features include:

- Pixel-wise addition, subtraction, multiplication, and division between images
- Application of arithmetic operations with constants to images
- Support for quadratic difference and general arithmetic expressions

These functions are typically used for basic algebraic processing and normalization
of image data.
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

import numpy as np

from sigima.enums import MathOperator
from sigima.objects.image import ImageObj
from sigima.proc.base import (
    ArithmeticParam,
    ConstantParam,
)
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import (
    dst_1_to_1,
    dst_2_to_1,
    dst_n_to_1,
    restore_data_outside_roi,
)
from sigima.tools.datatypes import clip_astype

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "addition",
    "addition_constant",
    "arithmetic",
    "average",
    "difference",
    "difference_constant",
    "division",
    "division",
    "division_constant",
    "product",
    "product_constant",
    "quadratic_difference",
    "standard_deviation",
]

# MARK: compute_n_to_1 functions -------------------------------------------------------
# Functions with N input images and 1 output image
# --------------------------------------------------------------------------------------
# Those functions are perfoming a computation on N input images and return a single
# output image. If we were only executing these functions locally, we would not need
# to define them here, but since we are using the multiprocessing module, we need to
# define them here so that they can be pickled and sent to the worker processes.
# Also, we need to systematically return the output image object, even if it is already
# modified in place, because the multiprocessing module will not be able to retrieve
# the modified object from the worker processes.


@computation_function()
def addition(src_list: list[ImageObj]) -> ImageObj:
    """Add images in the list and return the result image object

    Args:
        src_list: list of input image objects

    Returns:
        Output image object (modified in place)
    """
    dst = dst_n_to_1(src_list, "Σ")  # `dst` data is initialized to `src_list[0]` data
    for src in src_list[1:]:
        dst.data = np.add(dst.data, src.data, dtype=float)
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def average(src_list: list[ImageObj]) -> ImageObj:
    """Compute the average of images in the list and return the result image object

    Args:
        src_list: list of input image objects

    Returns:
        Output image object (modified in place)
    """
    dst = dst_n_to_1(src_list, "µ")  # `dst` data is initialized to `src_list[0]` data
    for src in src_list[1:]:
        dst.data = np.add(dst.data, src.data, dtype=float)
    dst.data /= len(src_list)
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def standard_deviation(src_list: list[ImageObj]) -> ImageObj:
    """Compute the element-wise standard deviation of multiple images.

    The first image in the list defines the "base" image. All other images are
    used to compute the element-wise standard deviation with the base image.

    .. note::

        If all images share the same region of interest (ROI), the standard deviation
        is computed only within the ROI.

    .. warning::

        It is assumed that all images have the same size and x-coordinates.

    Args:
        src_list: List of source images.

    Returns:
        Image object representing the standard deviation of the source images.
    """
    dst = dst_n_to_1(src_list, "𝜎")  # `dst` data is initialized to `src_list[0]` data
    assert dst.data is not None
    y_array = np.array([src.data for src in src_list], dtype=dst.data.dtype)
    dst.data = np.std(y_array, axis=0, ddof=0)
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def product(src_list: list[ImageObj]) -> ImageObj:
    """Multiply images in the list and return the result image object

    Args:
        src_list: list of input image objects

    Returns:
        Output image object (modified in place)
    """
    dst = dst_n_to_1(src_list, "Π")  # `dst` data is initialized to `src_list[0]` data
    for src in src_list[1:]:
        dst.data = np.multiply(dst.data, src.data, dtype=float)
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def addition_constant(src: ImageObj, p: ConstantParam) -> ImageObj:
    """Add **dst** and a constant value and return the new result image object

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** + **p.value** (new object)
    """
    # For the addition of a constant value, we convert the constant value to the same
    # data type as the input image, for consistency.
    value = np.array(p.value).astype(dtype=src.data.dtype)
    dst = dst_1_to_1(src, "+", str(value))
    dst.data = np.add(src.data, value, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def difference_constant(src: ImageObj, p: ConstantParam) -> ImageObj:
    """Subtract a constant value from an image and return the new result image object

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** - **p.value** (new object)
    """
    # For the subtraction of a constant value, we convert the constant value to the same
    # data type as the input image, for consistency.
    value = np.array(p.value).astype(dtype=src.data.dtype)
    dst = dst_1_to_1(src, "-", str(value))
    dst.data = np.subtract(src.data, value, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def product_constant(src: ImageObj, p: ConstantParam) -> ImageObj:
    """Multiply **dst** by a constant value and return the new result image object

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** * **p.value** (new object)
    """
    # For the multiplication by a constant value, we do not convert the constant value
    # to the same data type as the input image, because we want to allow the user to
    # multiply an image by a constant value of a different data type. The final data
    # type conversion ensures that the output image has the same data type as the input
    # image.
    dst = dst_1_to_1(src, "×", str(p.value))
    dst.data = np.multiply(src.data, p.value, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def division_constant(src: ImageObj, p: ConstantParam) -> ImageObj:
    """Divide an image by a constant value and return the new result image object

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** / **p.value** (new object)
    """
    # For the division by a constant value, we do not convert the constant value to the
    # same data type as the input image, because we want to allow the user to divide an
    # image by a constant value of a different data type. The final data type conversion
    # ensures that the output image has the same data type as the input image.
    dst = dst_1_to_1(src, "/", str(p.value))
    dst.data = np.divide(src.data, p.value, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


# MARK: compute_2_to_1 functions -------------------------------------------------------
# Functions with N input images + 1 input image and N output images
# --------------------------------------------------------------------------------------


@computation_function()
def arithmetic(src1: ImageObj, src2: ImageObj, p: ArithmeticParam) -> ImageObj:
    """Compute arithmetic operation on two images

    Args:
        src1: input image object
        src2: input image object
        p: arithmetic parameters

    Returns:
        Result image object
    """
    initial_dtype = src1.data.dtype
    title = p.operation.replace("obj1", "{0}").replace("obj2", "{1}")
    dst = src1.copy(title=title)
    o, a, b = p.operator, p.factor, p.constant
    # Apply operator
    if o in (MathOperator.MULTIPLY, MathOperator.DIVIDE) and a == 0.0:
        dst.data = np.ones_like(src1.data) * b
    elif o == MathOperator.ADD:
        dst.data = np.add(src1.data, src2.data, dtype=float) * a + b
    elif o == MathOperator.SUBTRACT:
        dst.data = np.subtract(src1.data, src2.data, dtype=float) * a + b
    elif o == MathOperator.MULTIPLY:
        dst.data = np.multiply(src1.data, src2.data, dtype=float) * a + b
    elif o == MathOperator.DIVIDE:
        dst.data = np.divide(src1.data, src2.data, dtype=float) * a + b
    # Eventually convert to initial data type
    if p.restore_dtype:
        dst.data = clip_astype(dst.data, initial_dtype)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def difference(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute difference between two images

    Args:
        src1: input image object
        src2: input image object

    Returns:
        Result image object **src1** - **src2** (new object)
    """
    dst = dst_2_to_1(src1, src2, "-")
    dst.data = np.subtract(src1.data, src2.data, dtype=float)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def quadratic_difference(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute quadratic difference between two images

    Args:
        src1: input image object
        src2: input image object

    Returns:
        Result image object (**src1** - **src2**) / sqrt(2.0) (new object)
    """
    dst = dst_2_to_1(src1, src2, "quadratic_difference")
    dst.data = np.subtract(src1.data, src2.data, dtype=float) / np.sqrt(2.0)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def division(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute division between two images

    Args:
        src1: input image object
        src2: input image object

    Returns:
        Result image object **src1** / **src2** (new object)
    """
    dst = dst_2_to_1(src1, src2, "/")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dst.data = np.divide(src1.data, src2.data, dtype=float)
    restore_data_outside_roi(dst, src1)
    return dst
