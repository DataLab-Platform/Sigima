# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Mathematical operations on signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import warnings

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.enums import AngleUnit
from sigima.objects import SignalObj
from sigima.proc.alignment import SOURCE_GRID_LINEAR_HOLD
from sigima.proc.base import AngleUnitParam, PhaseParam, dst_1_to_1, dst_2_to_1
from sigima.proc.decorator import computation_function
from sigima.proc.signal.base import (
    Wrap1to1Func,
    is_uncertainty_data_available,
    restore_data_outside_roi,
)
from sigima.tools import coordinates


@computation_function(operation_id="sigima.signal.transpose", contract_version=1)
def transpose(src: SignalObj) -> SignalObj:
    """Transpose signal (swap X and Y axes).

    Args:
        src: Source signal.

    Returns:
        Result signal object.
    """
    dst = dst_1_to_1(src, "transpose")
    x, y = src.get_data()
    dst.set_xydata(y, x, src.dy, src.dx)
    dst.xlabel = src.ylabel
    dst.ylabel = src.xlabel
    dst.xunit = src.yunit
    dst.yunit = src.xunit
    return dst


@computation_function(operation_id="sigima.signal.inverse", contract_version=1)
def inverse(src: SignalObj) -> SignalObj:
    """Compute the element-wise inverse of a signal.

    The function computes the reciprocal (1/y) of each element of the input signal.

    .. note::

        If the signal has a region of interest (ROI), the inverse is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.

    Returns:
        Result signal object representing the inverse of the input signal.
    """
    dst = dst_1_to_1(src, "invert")
    x, y = src.get_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dst.set_xydata(x, np.reciprocal(y))
        dst.y[np.isinf(dst.y)] = np.nan
        if is_uncertainty_data_available(src):
            err = np.abs(dst.y) * (src.dy / np.abs(src.y))
            err[np.isinf(err)] = np.nan
            dst.dy = err
    restore_data_outside_roi(dst, src)
    return dst


@computation_function(operation_id="sigima.signal.absolute", contract_version=1)
def absolute(src: SignalObj) -> SignalObj:
    """Compute absolute value with :py:data:`numpy.absolute`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.absolute)(src)


@computation_function(operation_id="sigima.signal.real", contract_version=1)
def real(src: SignalObj) -> SignalObj:
    """Compute real part with :py:func:`numpy.real`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.real)(src)


@computation_function(operation_id="sigima.signal.imag", contract_version=1)
def imag(src: SignalObj) -> SignalObj:
    """Compute imaginary part with :py:func:`numpy.imag`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.imag)(src)


@computation_function(operation_id="sigima.signal.phase", contract_version=1)
def phase(src: SignalObj, p: PhaseParam) -> SignalObj:
    """Compute the phase (argument) of a complex signal.

    The function uses :py:func:`numpy.angle` to compute the argument and
    :py:func:`numpy.unwrap` to unwrap it.

    Args:
        src: Input signal object.
        p: Phase parameters.

    Returns:
        Signal object containing the phase, optionally unwrapped.
    """
    suffix = "unwrap" if p.unwrap else ""
    dst = dst_1_to_1(src, "phase", suffix)
    x, y = src.get_data()
    argument = np.angle(y)
    if p.unwrap:
        argument = np.unwrap(argument)
    if p.unit == AngleUnit.DEGREE:
        argument = np.rad2deg(argument)
    dst.set_xydata(x, argument, src.dx, None)
    dst.yunit = p.unit
    restore_data_outside_roi(dst, src)
    return dst


@computation_function(
    operation_id="sigima.signal.complex_from_real_imag",
    contract_version=1,
    x_alignment=SOURCE_GRID_LINEAR_HOLD,
)
def complex_from_real_imag(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Combine two real signals into a complex signal using real + i * imag.

    .. warning::

        The x coordinates of the two signals must be the same.

    Args:
        src1: Real part signal.
        src2: Imaginary part signal.

    Returns:
        Result signal object with complex-valued y.
    """
    if not np.array_equal(src1.x, src2.x):
        warnings.warn(
            "The x coordinates of the two signals are not the same. "
            "Results may be incorrect."
        )
    dst = dst_2_to_1(src1, src2, "real_imag")
    y = src1.y + 1j * src2.y
    dst.set_xydata(src1.x, y, src1.dx, None)
    return dst


@computation_function(
    operation_id="sigima.signal.complex_from_magnitude_phase",
    contract_version=1,
    x_alignment=SOURCE_GRID_LINEAR_HOLD,
)
def complex_from_magnitude_phase(
    src1: SignalObj, src2: SignalObj, p: AngleUnitParam
) -> SignalObj:
    """Combine magnitude and phase signals into a complex signal.

    .. warning::

        The x coordinates of the two signals must be the same.

    .. warning::

        Negative values are not allowed for the radius and will be clipped to 0.

    Args:
        src1: Magnitude (module) signal.
        src2: Phase (argument) signal.
        p: Parameters (must provide unit for the phase).

    Returns:
        Result signal object with complex-valued y.
    """
    if not np.array_equal(src1.x, src2.x):
        warnings.warn(
            "The x coordinates of the two signals are not the same. "
            "Results may be incorrect."
        )
    if np.any(src1.y < 0.0):
        warnings.warn("Negative radius values are not allowed. They will be set to 0.")
        src1.y = np.maximum(src1.y, 0.0)
    dst = dst_2_to_1(src1, src2, "mag_phase")
    assert p.unit is not None
    y = coordinates.polar_to_complex(src1.y, src2.y, unit=p.unit)
    dst.set_xydata(src1.x, y, src1.x, None)
    return dst


class DataTypeSParam(gds.DataSet, title=_("Convert data type")):
    """Convert signal data type parameters"""

    dtype_str = gds.ChoiceItem(
        _("Destination data type"),
        list(zip(SignalObj.get_valid_dtypenames(), SignalObj.get_valid_dtypenames())),
        help=_("Output image data type."),
    )


@computation_function(operation_id="sigima.signal.astype", contract_version=1)
def astype(src: SignalObj, p: DataTypeSParam) -> SignalObj:
    """Convert data type with :py:func:`numpy.astype`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "astype", f"dtype={p.dtype_str}")
    dst.xydata = src.xydata.astype(p.dtype_str)
    return dst


@computation_function(operation_id="sigima.signal.log10", contract_version=1)
def log10(src: SignalObj) -> SignalObj:
    """Compute Log10 with :py:data:`numpy.log10`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "log10")
    x, y = src.get_data()

    # Compute result
    result_y = np.log10(y)
    dst.set_xydata(x, result_y, src.dx, src.dy)

    # Uncertainty propagation: σ(log₁₀(y)) = σ(y) / (y * ln(10))
    if is_uncertainty_data_available(src):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dst.dy = src.dy / (y * np.log(10))
            dst.dy[np.isinf(dst.dy) | np.isnan(dst.dy)] = np.nan

    restore_data_outside_roi(dst, src)
    return dst


@computation_function(operation_id="sigima.signal.exp", contract_version=1)
def exp(src: SignalObj) -> SignalObj:
    """Compute exponential with :py:data:`numpy.exp`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "exp")
    x, y = src.get_data()

    # Compute result
    result_y = np.exp(y)
    dst.set_xydata(x, result_y, src.dx, src.dy)

    # Uncertainty propagation: σ(eʸ) = eʸ * σ(y)
    if is_uncertainty_data_available(src):
        dst.dy = np.abs(result_y) * src.dy

    restore_data_outside_roi(dst, src)
    return dst


@computation_function(operation_id="sigima.signal.sqrt", contract_version=1)
def sqrt(src: SignalObj) -> SignalObj:
    """Compute square root with :py:data:`numpy.sqrt`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "sqrt")
    x, y = src.get_data()

    # Compute result
    result_y = np.sqrt(y)
    dst.set_xydata(x, result_y, src.dx, src.dy)

    # Uncertainty propagation: σ(√y) = σ(y) / (2√y)
    if is_uncertainty_data_available(src):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dst.dy = src.dy / (2 * np.sqrt(y))
            dst.dy[np.isinf(dst.dy) | np.isnan(dst.dy)] = np.nan

    restore_data_outside_roi(dst, src)
    return dst


class PowerParam(gds.DataSet, title=_("Power")):
    """Power parameters"""

    power = gds.FloatItem(_("Power"), default=2.0)


@computation_function(operation_id="sigima.signal.power", contract_version=1)
def power(src: SignalObj, p: PowerParam) -> SignalObj:
    """Compute power with :py:data:`numpy.power`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "^", str(p.power))
    dst.y = np.power(src.y, p.power)

    # Uncertainty propagation: σ(y^n) = |n * y^(n-1)| * σ(y)
    if is_uncertainty_data_available(src):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dst.dy *= np.abs(p.power * np.power(src.y, p.power - 1))
            dst.dy[np.isinf(dst.dy) | np.isnan(dst.dy)] = np.nan

    restore_data_outside_roi(dst, src)
    return dst


@computation_function(operation_id="sigima.signal.to_polar", contract_version=1)
def to_polar(src: SignalObj, p: AngleUnitParam) -> SignalObj:
    """Convert Cartesian coordinates to polar coordinates.

    This function converts the x and y coordinates of a signal to polar coordinates
    using :py:func:`sigima.tools.coordinates.to_polar`.

    .. warning::

        X and y must share the same units for the computation to make sense.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.

    Raises:
        ValueError: If the x and y units are not the same.
    """
    assert p.unit is not None
    if src.xunit != src.yunit:
        warnings.warn(
            f"X and y units are not the same: {src.xunit} != {src.yunit}. "
            "Results will be incorrect."
        )
    dst = dst_1_to_1(src, "Polar coordinates", f"unit={p.unit}")
    x, y = src.get_data()
    r, theta = coordinates.to_polar(x, y, p.unit)
    dst.set_xydata(r, theta)
    dst.xlabel = _("Radius")
    dst.ylabel = _("Angle")
    dst.yunit = p.unit
    return dst


@computation_function(operation_id="sigima.signal.to_cartesian", contract_version=1)
def to_cartesian(src: SignalObj, p: AngleUnitParam) -> SignalObj:
    """Convert polar coordinates to Cartesian coordinates.

    This function converts the r and theta coordinates of a signal to Cartesian
    coordinates using :py:func:`sigima.tools.coordinates.to_cartesian`.

    .. note::

        It is assumed that the x-axis represents the radius and the y-axis the angle.

    .. warning::

        Negative values are not allowed for the radius and will be clipped to 0.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    dst = dst_1_to_1(src, "Cartesian coordinates", f"unit={p.unit}")
    r, theta = src.get_data()
    if np.any(r < 0.0):
        warnings.warn("Negative radius values are not allowed. They will be set to 0.")
        r = np.maximum(r, 0.0)
    x, y = coordinates.to_cartesian(r, theta, p.unit)
    dst.set_xydata(x, y)
    dst.xlabel = _("x")
    dst.ylabel = _("y")
    dst.yunit = src.xunit
    return dst
