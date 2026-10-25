# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Arithmetic operations on signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import warnings

import numpy as np

from sigima.enums import MathOperator, SignalsToImageOrientation
from sigima.objects import ImageObj, SignalObj, create_image
from sigima.proc.base import (
    ArithmeticParam,
    ConstantParam,
    SignalsToImageParam,
    dst_1_to_1,
    dst_2_to_1,
    dst_n_to_1,
)
from sigima.proc.contracts import (
    EXACT,
    SMALLEST_GRID_LINEAR_HOLD,
    SOURCE_GRID_LINEAR,
    SOURCE_GRID_LINEAR_HOLD,
    ReplayQualification,
)
from sigima.proc.decorator import computation_function
from sigima.proc.signal.base import (
    is_uncertainty_data_available,
    restore_data_outside_roi,
)
from sigima.proc.signal.mathops import inverse
from sigima.tools.signal import scaling


def __signals_y_to_array(signals: list[SignalObj]) -> np.ndarray:
    """Create an array from a list of signals, extracting the `y` attribute.

    Args:
        signals: List of signal objects.

    Returns:
        A NumPy array stacking the `y` attribute from all signals, with a data type
        holding all of them (e.g. complex if any signal is complex), whatever their
        order.
    """
    arrays = [sig.y for sig in signals]
    return np.array(arrays, dtype=np.result_type(*arrays))


def __signals_dy_to_array(signals: list[SignalObj]) -> np.ndarray:
    """Create an array from a list of signals, extracting the `dy` attribute.

    Args:
        signals: List of signal objects.

    Returns:
        A NumPy array stacking the `dy` attribute from all signals, with a data type
        holding all of them, whatever their order.
    """
    arrays = [sig.dy for sig in signals]
    return np.array(arrays, dtype=np.result_type(*arrays))


def __product_uncertainty(y_array: np.ndarray, dy_array: np.ndarray) -> np.ndarray:
    """Return the first-order uncertainty of the product of independent factors.

    ``σ² = Σ_i σ_i² Π_{j≠i} |y_j|²`` is computed without dividing by the factors,
    so that it is defined where a factor is zero. Each term is computed from the
    mantissas and exponents of the factors (:func:`numpy.frexp`), so that no
    intermediate product overflows or underflows whatever the order and the
    magnitudes of the factors, and the terms are scaled by the largest one before
    being squared.

    Args:
        y_array: Factors (one row per signal), real or complex.
        dy_array: Uncertainties of the factors (one row per signal).

    Returns:
        Uncertainty of the product (NaN where it overflows).
    """
    abs_y = np.abs(y_array)
    abs_dy = np.abs(dy_array)
    ftype = np.result_type(abs_y.dtype, abs_dy.dtype, np.float32)
    y_mant, y_exp = np.frexp(abs_y.astype(ftype, copy=False))
    dy_mant, dy_exp = np.frexp(abs_dy.astype(ftype, copy=False))
    ones = np.ones_like(y_mant[:1])
    # Products of the mantissas before (prefix) and after (suffix) each factor:
    # mantissas lie in [0.5, 1) (or are zero), so these products cannot overflow
    prefix = np.cumprod(np.concatenate((ones, y_mant[:-1])), axis=0)
    suffix = np.cumprod(np.concatenate((ones, y_mant[:0:-1])), axis=0)[::-1]
    # Term i is mant[i] * 2**exp[i]
    mant = dy_mant * prefix * suffix
    exp = dy_exp + (np.sum(y_exp, axis=0) - y_exp)
    positive = mant > 0.0
    exp_max = np.max(np.where(positive, exp, np.iinfo(exp.dtype).min), axis=0)
    exp_max = np.where(np.any(positive, axis=0), exp_max, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ratios = np.ldexp(mant, exp - exp_max)
        scale = np.max(ratios, axis=0)
        ratios /= np.where(scale > 0.0, scale, 1.0)
        uncertainty = np.ldexp(scale * np.sqrt(np.sum(ratios**2, axis=0)), exp_max)
    uncertainty[np.isinf(uncertainty)] = np.nan
    return uncertainty


@computation_function(
    operation_id="sigima.signal.addition",
    contract_version=1,
    x_alignment=SMALLEST_GRID_LINEAR_HOLD,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def addition(src_list: list[SignalObj]) -> SignalObj:
    """Compute the element-wise sum of multiple signals.

    The first signal in the list defines the "base" signal. All other signals are added
    element-wise to the base signal.

    .. note::

        If all signals share the same region of interest (ROI), the sum is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that all signals have the same size and x-coordinates.

    Args:
        src_list: List of source signals.

    Returns:
        Signal object representing the sum of the source signals.
    """
    dst = dst_n_to_1(src_list, "Σ")  # `dst` data is initialized to `src_list[0]` data.
    dst.y = np.sum(__signals_y_to_array(src_list), axis=0)
    if is_uncertainty_data_available(src_list):
        dst.dy = np.sqrt(np.sum(__signals_dy_to_array(src_list) ** 2, axis=0))
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function(
    operation_id="sigima.signal.average",
    contract_version=1,
    x_alignment=SMALLEST_GRID_LINEAR_HOLD,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def average(src_list: list[SignalObj]) -> SignalObj:
    """Compute the element-wise average of multiple signals.

    The first signal in the list defines the "base" signal. All other signals are
    averaged element-wise with the base signal.

    .. note::

        If all signals share the same region of interest (ROI), the average is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that all signals have the same size and x-coordinates.

    Args:
        src_list: List of source signals.

    Returns:
        Signal object representing the average of the source signals.
    """
    dst = dst_n_to_1(src_list, "µ")  # `dst` data is initialized to `src_list[0]` data.
    dst.y = np.mean(__signals_y_to_array(src_list), axis=0)
    if is_uncertainty_data_available(src_list):
        dy_array = __signals_dy_to_array(src_list)
        dst.dy = np.sqrt(np.sum(dy_array**2, axis=0)) / len(src_list)
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function(
    operation_id="sigima.signal.standard_deviation",
    contract_version=1,
    x_alignment=SMALLEST_GRID_LINEAR_HOLD,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def standard_deviation(src_list: list[SignalObj]) -> SignalObj:
    """Compute the element-wise standard deviation of multiple signals.

    The first signal in the list defines the "base" signal. All other signals are
    used to compute the element-wise standard deviation with the base signal.

    .. note::

        If all signals share the same region of interest (ROI), the standard deviation
        is computed only within the ROI.

    .. warning::

        It is assumed that all signals have the same size and x-coordinates.

    Args:
        src_list: List of source signals.

    Returns:
        Signal object representing the standard deviation of the source signals.
    """
    dst = dst_n_to_1(src_list, "𝜎")  # `dst` data is initialized to `src_list[0]` data
    dst.y = np.std(__signals_y_to_array(src_list), axis=0, ddof=0)
    if is_uncertainty_data_available(src_list):
        dst.dy = dst.y / np.sqrt(2 * (len(src_list) - 1))
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function(
    operation_id="sigima.signal.product",
    contract_version=1,
    x_alignment=SMALLEST_GRID_LINEAR_HOLD,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def product(src_list: list[SignalObj]) -> SignalObj:
    """Compute the element-wise product of multiple signals.

    The first signal in the list defines the "base" signal. All other signals are
    multiplied element-wise with the base signal.

    .. note::

        If all signals share the same region of interest (ROI), the product is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated to first order for independent factors, also
        where a factor is zero (the moduli of complex factors are used).

    .. warning::

        It is assumed that all signals have the same size and x-coordinates.

    Args:
        src_list: List of source signals.

    Returns:
        Signal object representing the product of the source signals.
    """
    dst = dst_n_to_1(src_list, "Π")  # `dst` data is initialized to `src_list[0]` data.
    y_array = __signals_y_to_array(src_list)
    dst.y = np.prod(y_array, axis=0)
    if is_uncertainty_data_available(src_list):
        dst.dy = __product_uncertainty(y_array, __signals_dy_to_array(src_list))
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function(
    operation_id="sigima.signal.addition_constant",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def addition_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Compute the sum of a signal and a constant value.

    The function adds a constant value to each element of the input signal.

    .. note::

        If the signal has a region of interest (ROI), the addition is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.
        p: Constant value.

    Returns:
        Result signal object representing the sum of the input signal and the constant.
    """
    # Uncertainty propagation: dst_1_to_1() copies all data including uncertainties.
    # For addition with constant: σ(y + c) = σ(y), so no modification needed.
    dst = dst_1_to_1(src, "+", str(p.value))
    dst.y += p.value
    restore_data_outside_roi(dst, src)
    return dst


@computation_function(
    operation_id="sigima.signal.difference_constant",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def difference_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Compute the difference between a signal and a constant value.

    The function subtracts a constant value from each element of the input signal.

    .. note::

        If the signal has a region of interest (ROI), the subtraction is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.
        p: Constant value.

    Returns:
        Result signal object representing the difference between the input signal and
        the constant.
    """
    # Uncertainty propagation: dst_1_to_1() copies all data including uncertainties.
    # For subtraction with constant: σ(y - c) = σ(y), so no modification needed.
    dst = dst_1_to_1(src, "-", str(p.value))
    dst.y -= p.value
    restore_data_outside_roi(dst, src)
    return dst


@computation_function(
    operation_id="sigima.signal.product_constant",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def product_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Compute the product of a signal and a constant value.

    The function multiplies each element of the input signal by a constant value.

    .. note::

        If the signal has a region of interest (ROI), the multiplication is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.
        p: Constant value.

    Returns:
        Result signal object representing the product of the input signal and the
        constant.
    """
    assert p.value is not None
    # Uncertainty propagation: dst_1_to_1() copies all data including uncertainties.
    # For multiplication with constant: σ(c*y) = |c| * σ(y), so modification needed.
    dst = dst_1_to_1(src, "×", str(p.value))
    dst.y *= p.value
    if is_uncertainty_data_available(src):
        dst.dy *= np.abs(p.value)  # Modify in-place since dy already copied from src
    restore_data_outside_roi(dst, src)
    return dst


@computation_function(
    operation_id="sigima.signal.division_constant",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def division_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Compute the division of a signal by a constant value.

    The function divides each element of the input signal by a constant value.

    .. note::

        If the signal has a region of interest (ROI), the division is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.
        p: Constant value.

    Returns:
        Result signal object representing the division of the input signal by the
        constant.
    """
    assert p.value is not None
    # Uncertainty propagation: dst_1_to_1() copies all data including uncertainties.
    # For division with constant: σ(y/c) = σ(y) / |c|, so modification needed.
    dst = dst_1_to_1(src, "/", str(p.value))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dst.y /= p.value
        dst.y[np.isinf(dst.y)] = np.nan
        if is_uncertainty_data_available(src):
            dst.dy /= np.abs(p.value)  # Modify in-place since dy already copied
            dst.dy[np.isinf(dst.dy)] = np.nan
    restore_data_outside_roi(dst, src)
    return dst


@computation_function(
    operation_id="sigima.signal.arithmetic",
    contract_version=1,
    x_alignment=SOURCE_GRID_LINEAR_HOLD,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def arithmetic(src1: SignalObj, src2: SignalObj, p: ArithmeticParam) -> SignalObj:
    """Perform an arithmetic operation on two signals.

    The function applies the specified arithmetic operation to each element of the input
    signals.

    .. note::

        The operation is performed only within the region of interest of `src1`.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that both signals have the same size and x-coordinates.

    Args:
        src1: First input signal.
        src2: Second input signal.
        p: Arithmetic operation parameters.

    Returns:
        Result signal object representing the arithmetic operation on the input signals.
    """
    initial_dtype = src1.xydata.dtype
    title = p.operation.replace("obj1", "{0}").replace("obj2", "{1}")
    dst = src1.copy(title=title)
    a = ConstantParam.create(value=p.factor)
    b = ConstantParam.create(value=p.constant)
    if p.operator == MathOperator.ADD:
        dst = addition_constant(product_constant(addition([src1, src2]), a), b)
    elif p.operator == MathOperator.SUBTRACT:
        dst = addition_constant(product_constant(difference(src1, src2), a), b)
    elif p.operator == MathOperator.MULTIPLY:
        dst = addition_constant(product_constant(product([src1, src2]), a), b)
    elif p.operator == MathOperator.DIVIDE:
        dst = addition_constant(product_constant(product([src1, inverse(src2)]), a), b)
    # Eventually convert to initial data type
    if p.restore_dtype:
        dst.xydata = dst.xydata.astype(initial_dtype)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function(
    operation_id="sigima.signal.difference",
    contract_version=1,
    x_alignment=SOURCE_GRID_LINEAR,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute the element-wise difference between two signals.

    The function subtracts each element of the second signal from the corresponding
    element of the first signal.

    .. note::

        If both signals share the same region of interest (ROI), the difference is
        performed only within the ROI.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that both signals have the same size and x-coordinates.

    Args:
        src1: First input signal.
        src2: Second input signal.

    Returns:
        Result signal object representing the difference between the input signals.
    """
    dst = dst_2_to_1(src1, src2, "-")
    dst.y = src1.y - src2.y
    if is_uncertainty_data_available([src1, src2]):
        dy_array = __signals_dy_to_array([src1, src2])
        dst.dy = np.sqrt(np.sum(dy_array**2, axis=0))
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function(
    operation_id="sigima.signal.quadratic_difference",
    contract_version=1,
    x_alignment=SOURCE_GRID_LINEAR_HOLD,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def quadratic_difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute the normalized difference between two signals.

    The function computes the element-wise difference between the two signals and
    divides the result by sqrt(2.0).

    .. note::

        If both signals share the same region of interest (ROI), the operation is
        performed only within the ROI.

    .. note::

        Uncertainties are propagated. For two input signals with identical standard
        deviations, the standard deviation of the output signal equals the standard
        deviation of each of the input signals.

    .. warning::

        It is assumed that both signals have the same size and x-coordinates.

    Args:
        src1: First input signal.
        src2: Second input signal.

    Returns:
        Result signal object representing the quadratic difference between the input
        signals.
    """
    norm = ConstantParam.create(value=1.0 / np.sqrt(2.0))
    return product_constant(difference(src1, src2), norm)


@computation_function(
    operation_id="sigima.signal.division",
    contract_version=1,
    x_alignment=SOURCE_GRID_LINEAR_HOLD,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def division(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute the element-wise division between two signals.

    The function divides each element of the first signal by the corresponding element
    of the second signal.

    .. note::

        If both signals share the same region of interest (ROI), the division is
        performed only within the ROI.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that both signals have the same size and x-coordinates.

    Args:
        src1: First input signal.
        src2: Second input signal.

    Returns:
        Result signal object representing the division of the input signals.
    """
    dst = product([src1, inverse(src2)])
    return dst


@computation_function(
    operation_id="sigima.signal.signals_to_image",
    contract_version=1,
    x_alignment=SMALLEST_GRID_LINEAR_HOLD,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def signals_to_image(src_list: list[SignalObj], p: SignalsToImageParam) -> ImageObj:
    """Combine multiple signals into an image.

    The function takes a list of signals and combines them into a 2D image,
    arranging them either as rows or columns based on the specified orientation.
    Optionally, each signal can be normalized before combining.

    .. note::

        All signals must have the same size (number of data points).

    .. note::

        If normalization is enabled, each signal is normalized independently
        using the specified normalization method before being added to the image.

    Args:
        src_list: List of source signals to combine.
        p: Parameters specifying orientation and normalization options.

    Returns:
        Image object representing the combined signals.

    Raises:
        ValueError: If the signal list is empty or signals have different sizes.
    """
    if not src_list:
        raise ValueError("The signal list is empty.")

    # Check that all signals have the same size
    sizes = [len(sig.y) for sig in src_list]
    if len(set(sizes)) > 1:
        raise ValueError(
            f"All signals must have the same size. Found sizes: {set(sizes)}"
        )

    # Prepare data array
    y_array = __signals_y_to_array(src_list)

    # Normalize if requested
    if p.normalize:
        for i in range(len(src_list)):
            y_array[i] = scaling.normalize(y_array[i], p.normalize_method)

    # Arrange as rows or columns
    if p.orientation == SignalsToImageOrientation.COLUMNS:
        data = y_array.T
    else:  # ROWS
        data = y_array

    # Create the result image
    suffix_parts = [f"n={len(src_list)}", f"orientation={p.orientation}"]
    if p.normalize:
        suffix_parts.append(f"norm={p.normalize_method}")
    suffix = ", ".join(suffix_parts)
    title = f"combined_signals|{suffix}"

    dst = create_image(title, data)
    if p.orientation == SignalsToImageOrientation.ROWS:
        dst.xlabel = src_list[0].ylabel
    else:
        dst.ylabel = src_list[0].ylabel
    dst.zlabel = src_list[0].ylabel
    dst.zunit = src_list[0].yunit

    return dst
