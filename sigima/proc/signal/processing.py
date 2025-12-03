# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see sigima/LICENSE for details)

"""
Signal processing operations
============================

This module provides signal processing operations:

- Zero padding
- Interpolation and resampling
- Convolution and deconvolution
- Signal manipulation functions

.. note::

    Most operations use functions from :mod:`sigima.tools.signal` for actual
    computations.
"""

from __future__ import annotations

import warnings

import guidata.dataset as gds
import numpy as np
import scipy.integrate as spt
import scipy.signal as sps
from guidata.dataset import FuncProp, GetAttrProp

from sigima.config import _
from sigima.config import options as sigima_options
from sigima.enums import Interpolation1DMethod, NormalizationMethod, WindowingMethod
from sigima.objects import ROI1DParam, SignalObj
from sigima.proc.base import ClipParam, NormalizeParam, dst_2_to_1
from sigima.proc.decorator import computation_function
from sigima.tools.signal import fourier, interpolation, scaling, windowing

from .base import dst_1_to_1, is_uncertainty_data_available, restore_data_outside_roi


class InterpolationParam(gds.DataSet, title=_("Interpolation")):
    """Interpolation parameters"""

    method = gds.ChoiceItem(
        _("Interpolation method"),
        [
            (Interpolation1DMethod.LINEAR, "Linear"),
            (Interpolation1DMethod.SPLINE, "Spline"),
            (Interpolation1DMethod.QUADRATIC, "Quadratic"),
            (Interpolation1DMethod.CUBIC, "Cubic"),
            (Interpolation1DMethod.BARYCENTRIC, "Barycentric"),
            (Interpolation1DMethod.PCHIP, "PCHIP"),
        ],
        default=Interpolation1DMethod.LINEAR,
    )
    fill_value = gds.FloatItem(
        _("Fill value"),
        default=None,
        help=_(
            "Value to use for points outside the interpolation domain "
            "(used only with linear, cubic and pchip methods)."
        ),
        check=False,
    )


@computation_function()
def interpolate(src1: SignalObj, src2: SignalObj, p: InterpolationParam) -> SignalObj:
    """Interpolate data with :py:func:`sigima.tools.signal.interpolation.interpolate`.

    Args:
        src1: Source signal to interpolate.
        src2: Signal with new x-axis.
        p: Parameters.

    Returns:
        Result signal object.
    """
    suffix = f"method={p.method}"
    if p.fill_value is not None and p.method in (
        Interpolation1DMethod.LINEAR,
        Interpolation1DMethod.CUBIC,
        Interpolation1DMethod.PCHIP,
    ):
        suffix += f", fill_value={p.fill_value}"
    dst = dst_2_to_1(src1, src2, "interpolate", suffix)
    x1, y1 = src1.get_data()
    xnew, _y2 = src2.get_data()
    ynew = interpolation.interpolate(x1, y1, xnew, p.method, p.fill_value)
    dst.set_xydata(xnew, ynew)
    return dst


class Resampling1DParam(InterpolationParam):
    """Resample parameters for 1D signals"""

    xmin = gds.FloatItem(_("X<sub>min</sub>"), allow_none=True)
    xmax = gds.FloatItem(_("X<sub>max</sub>"), allow_none=True)
    _prop = GetAttrProp("dx_or_nbpts")
    _modes = (("dx", "ΔX"), ("nbpts", _("Number of points")))
    mode = gds.ChoiceItem(_("Mode"), _modes, default="nbpts", radio=True).set_prop(
        "display", store=_prop
    )
    dx = gds.FloatItem("ΔX", allow_none=True).set_prop(
        "display", active=FuncProp(_prop, lambda x: x == "dx")
    )
    nbpts = gds.IntItem(_("Number of points"), allow_none=True).set_prop(
        "display", active=FuncProp(_prop, lambda x: x == "nbpts")
    )

    def update_from_obj(self, obj: SignalObj) -> None:
        """Update parameters from a signal object."""
        if self.xmin is None:
            self.xmin = obj.x[0]
        if self.xmax is None:
            self.xmax = obj.x[-1]
        if self.dx is None:
            self.dx = obj.x[1] - obj.x[0]
        if self.nbpts is None:
            self.nbpts = len(obj.x)


@computation_function()
def resampling(src: SignalObj, p: Resampling1DParam) -> SignalObj:
    """Resample data with :py:func:`sigima.tools.signal.interpolation.interpolate`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    # Create new x-axis based on parameters
    if p.mode == "dx":
        assert p.dx is not None
        xnew = np.arange(p.xmin, p.xmax + p.dx / 2, p.dx)
    else:
        assert p.nbpts is not None
        xnew = np.linspace(p.xmin, p.xmax, p.nbpts)

    method: Interpolation1DMethod = p.method
    suffix = f"method={method.value}"
    if p.fill_value is not None and method in (
        Interpolation1DMethod.LINEAR,
        Interpolation1DMethod.CUBIC,
        Interpolation1DMethod.PCHIP,
    ):
        suffix += f", fill_value={p.fill_value}"

    dst = dst_1_to_1(src, "resampling", suffix)
    x, y = src.get_data()
    ynew = interpolation.interpolate(x, y, xnew, method, p.fill_value)
    dst.set_xydata(xnew, ynew)
    return dst


def check_same_sample_rate(src1: SignalObj, src2: SignalObj) -> None:
    """Check if two signals have a constant step size *and* the same sample rate.

    Args:
        src1: First signal.
        src2: Second signal.

    Raises:
        ValueError: If the signals do not have a constant step size
         or the same sample rate.
    """
    for sig in (src1, src2):
        if not np.allclose(np.diff(sig.x), sig.x[1] - sig.x[0]):
            raise ValueError(f"Signal {sig.title} must have a constant step size (dx).")
    dx1 = src1.x[1] - src1.x[0]
    dx2 = src2.x[1] - src2.x[0]
    if not np.isclose(dx1, dx2):
        raise ValueError(f"Signals must have the same sample rate (dx): {dx1} != {dx2}")


@computation_function()
def deconvolution(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute deconvolution.

    The function computes the deconvolution of a signal using
    :py:func:`sigima_.algorithms.signal.fourier.deconvolve`.

    Args:
        src1: Source signal.
        src2: Filter signal.

    Returns:
        Result signal.

    Notes:
        The kernel normalization behavior can be configured globally using
        ``sigima.config.options.auto_normalize_kernel``.
    """
    check_same_sample_rate(src1, src2)
    dst = dst_2_to_1(src1, src2, "⊛⁻¹", f"filter={src2.title}")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()

    # Get kernel normalization option from configuration
    normalize_kernel = sigima_options.auto_normalize_kernel.get()

    result_y = fourier.deconvolve(
        x1,
        y1,
        y2,
        normalize_kernel_flag=normalize_kernel,
        reg=2.0,
        gain_max=None,
        auto_scale=True,
    )
    dst.set_xydata(x1, result_y, None, None)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def normalize(src: SignalObj, p: NormalizeParam) -> SignalObj:
    """Normalize data with :py:func:`sigima.tools.signal.level.normalize`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    method: NormalizationMethod = p.method
    dst = dst_1_to_1(src, "normalize", f"ref={method.value}")
    x, y = src.get_data()
    normalized_y = scaling.normalize(y, method)
    dst.set_xydata(x, normalized_y)

    # Uncertainty propagation for normalization
    # σ(y/norm_factor) = σ(y) / norm_factor
    if is_uncertainty_data_available(src):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Calculate normalization factor
            if method == NormalizationMethod.MAXIMUM:
                norm_factor = np.nanmax(y)
            elif method == NormalizationMethod.AMPLITUDE:
                norm_factor = np.nanmax(y) - np.nanmin(y)
            elif method == NormalizationMethod.AREA:
                norm_factor = np.nansum(y)
            elif method == NormalizationMethod.ENERGY:
                norm_factor = np.sqrt(np.nansum(np.abs(y) ** 2))
            elif method == NormalizationMethod.RMS:
                norm_factor = np.sqrt(np.nanmean(np.abs(y) ** 2))
            else:
                raise RuntimeError(f"Unsupported normalization method: {method}")

            if norm_factor != 0:
                dst.dy = src.dy / np.abs(norm_factor)
            else:
                dst.dy[:] = np.nan

    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def derivative(src: SignalObj) -> SignalObj:
    """Compute derivative with :py:func:`numpy.gradient`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "derivative")
    x, y = src.get_data()
    dst.set_xydata(x, np.gradient(y, x))

    # Uncertainty propagation for numerical derivative
    # For gradient using finite differences: σ(dy/dx) ≈ σ(y) / Δx
    if is_uncertainty_data_available(src):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Use the same gradient approach as numpy.gradient for uncertainty
            dst.dy = np.gradient(src.dy, x)
            dst.dy[np.isinf(dst.dy) | np.isnan(dst.dy)] = np.nan

    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def integral(src: SignalObj) -> SignalObj:
    """Compute integral with :py:func:`scipy.integrate.cumulative_trapezoid`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "integral")
    x, y = src.get_data()
    dst.set_xydata(x, spt.cumulative_trapezoid(y, x, initial=0.0))

    # Uncertainty propagation for numerical integration
    # For cumulative trapezoidal integration, uncertainties accumulate
    if is_uncertainty_data_available(src):
        # Propagate uncertainties through cumulative trapezoidal rule
        # σ(∫y dx) ≈ √(Σ(σ(y_i) * Δx_i)²) for independent measurements
        dx = np.diff(x)
        dy_squared = src.dy[:-1] ** 2 + src.dy[1:] ** 2  # Trapezoidal rule uncertainty
        # Propagated variance for trapezoidal integration
        dst.dy = np.zeros_like(dst.y)  # Initialize uncertainty array
        dst.dy[0] = 0.0  # Initial value has no uncertainty
        dst.dy[1:] = np.sqrt(np.cumsum(dy_squared * (dx**2) / 4))

    restore_data_outside_roi(dst, src)
    return dst


class XYCalibrateParam(gds.DataSet, title=_("Calibration")):
    """Signal calibration parameters"""

    axes = (("x", _("X-axis")), ("y", _("Y-axis")))
    axis = gds.ChoiceItem(_("Calibrate"), axes, default="y")
    a = gds.FloatItem("a", default=1.0)
    b = gds.FloatItem("b", default=0.0)


@computation_function()
def calibration(src: SignalObj, p: XYCalibrateParam) -> SignalObj:
    """Compute linear calibration

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "calibration", f"{p.axis}={p.a}*{p.axis}+{p.b}")
    x, y = src.get_data()
    if p.axis == "x":
        dst.set_xydata(p.a * x + p.b, y, src.dx, src.dy)
        # For X-axis calibration: uncertainties in x are scaled, y unchanged
        if is_uncertainty_data_available(src):
            dst.dx = np.abs(p.a) * src.dx if src.dx is not None else None
            # Y uncertainties remain the same
    else:
        dst.set_xydata(x, p.a * y + p.b, src.dx, src.dy)
        # For Y-axis calibration: σ(a*y + b) = |a| * σ(y)
        if is_uncertainty_data_available(src):
            if dst.dy is not None:
                dst.dy *= np.abs(p.a)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def clip(src: SignalObj, p: ClipParam) -> SignalObj:
    """Compute maximum data clipping with :py:func:`numpy.clip`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "clip", f"[{p.lower}, {p.upper}]")
    x, y = src.get_data()

    # Compute result
    result_y = np.clip(y, p.lower, p.upper)
    dst.set_xydata(x, result_y, src.dx, src.dy)

    # Uncertainty propagation: σ(clip(y)) = σ(y) where not clipped, 0 where clipped
    if is_uncertainty_data_available(src):
        dst.dy = src.dy.copy()
        if p.lower is not None:
            dst.dy[y <= p.lower] = 0
        if p.upper is not None:
            dst.dy[y >= p.upper] = 0

    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def offset_correction(src: SignalObj, p: ROI1DParam) -> SignalObj:
    """Correct offset: subtract the mean value of the signal in the specified range
    (baseline correction)

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "offset_correction", f"{p.xmin:.3g}≤x≤{p.xmax:.3g}")
    _roi_x, roi_y = p.get_data(src)
    dst.y -= np.mean(roi_y)
    restore_data_outside_roi(dst, src)
    return dst


class DetrendingParam(gds.DataSet, title=_("Detrending")):
    """Detrending parameters"""

    methods = (("linear", _("Linear")), ("constant", _("Constant")))
    method = gds.ChoiceItem(_("Detrending method"), methods, default="linear")


@computation_function()
def detrending(src: SignalObj, p: DetrendingParam) -> SignalObj:
    """Detrend data with :py:func:`scipy.signal.detrend`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "detrending", f"method={p.method}")
    x, y = src.get_data()
    dst.set_xydata(x, sps.detrend(y, type=p.method))
    restore_data_outside_roi(dst, src)
    return dst


class WindowingParam(gds.DataSet, title=_("Windowing")):
    """Windowing parameters."""

    _meth_prop = gds.GetAttrProp("method")
    method = gds.ChoiceItem(
        _("Method"), WindowingMethod, default=WindowingMethod.HAMMING
    ).set_prop("display", store=_meth_prop)
    alpha = gds.FloatItem(
        "Alpha",
        default=0.5,
        help=_("Shape parameter of the Tukey windowing function"),
    ).set_prop(
        "display", active=gds.FuncProp(_meth_prop, lambda x: x == WindowingMethod.TUKEY)
    )
    beta = gds.FloatItem(
        "Beta",
        default=14.0,
        help=_("Shape parameter of the Kaiser windowing function"),
    ).set_prop(
        "display",
        active=gds.FuncProp(_meth_prop, lambda x: x == WindowingMethod.KAISER),
    )
    sigma = gds.FloatItem(
        "Sigma",
        default=0.5,
        help=_("Shape parameter of the Gaussian windowing function"),
    ).set_prop(
        "display",
        active=gds.FuncProp(_meth_prop, lambda x: x == WindowingMethod.GAUSSIAN),
    )


@computation_function()
def apply_window(src: SignalObj, p: WindowingParam) -> SignalObj:
    """Compute windowing with :py:func:`sigima.tools.signal.windowing.apply_window`.

    Args:
        src: Source signal.
        p: Parameters for windowing.

    Returns:
        Result signal object.
    """
    method: WindowingMethod = p.method
    suffix = f"method={method.value}"
    if method == WindowingMethod.GAUSSIAN:
        suffix += f", sigma={p.sigma:.3f}"
    elif method == WindowingMethod.KAISER:
        suffix += f", beta={p.beta:.3f}"
    elif method == WindowingMethod.TUKEY:
        suffix += f", alpha={p.alpha:.3f}"
    dst = dst_1_to_1(src, "apply_window", suffix)
    assert p.alpha is not None
    dst.y = windowing.apply_window(dst.y, method, p.alpha)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def reverse_x(src: SignalObj) -> SignalObj:
    """Reverse x-axis

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "reverse_x")
    dst.y = dst.y[::-1]
    return dst


@computation_function()
def convolution(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute convolution of two signals with :py:func:`scipy.signal.convolve`.

    Args:
        src1: Source signal 1.
        src2: Source signal 2.

    Returns:
        Result signal.

    Notes:
        The behavior of kernel normalization is controlled by the global configuration
        option ``sigima.config.options.auto_normalize_kernel``.
    """
    check_same_sample_rate(src1, src2)
    dst = dst_2_to_1(src1, src2, "⊛")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()

    # Get configuration option for kernel normalization
    normalize_kernel = sigima_options.auto_normalize_kernel.get()

    ynew = fourier.convolve(
        x1,
        y1,
        y2,
        normalize_kernel_flag=normalize_kernel,
    )
    dst.set_xydata(x1, ynew, None, None)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def xy_mode(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Simulate the X-Y mode of an oscilloscope.

    Use the first signal as the X-axis and the second signal as the Y-axis.

    Args:
        src1: First input signal (X-axis).
        src2: Second input signal (Y-axis).

    Returns:
        A signal object representing the X-Y mode.
    """
    dst = dst_2_to_1(src1, src2, "", "X-Y Mode")
    p = Resampling1DParam()
    p.xmin = max(src1.x[0], src2.x[0])
    p.xmax = min(src1.x[-1], src2.x[-1])
    assert p.xmin < p.xmax, "X-Y mode: No overlap between signals."
    p.mode = "nbpts"
    p.nbpts = min(src1.x.size, src2.x.size)
    _, y1 = resampling(src1, p).get_data()
    _, y2 = resampling(src2, p).get_data()
    dst.set_xydata(y1, y2)
    dst.title = "{1} = f({0})"
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def replace_x_by_other_y(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Create a new signal using Y from src1 and Y from src2 as X coordinates.

    This is useful for calibration scenarios where one signal contains calibration
    data (e.g., wavelengths) in its Y values, and you want to plot another signal's
    Y values against these calibration points.

    The two signals must have the same number of points.

    Args:
        src1: First signal (provides Y data for output).
        src2: Second signal (provides Y data to be used as X coordinates for output).

    Returns:
        A new signal with X from src2.y and Y from src1.y.

    Raises:
        ValueError: If signals don't have the same number of points.
    """
    if src1.y.size != src2.y.size:
        raise ValueError(
            f"Signals must have the same number of points: "
            f"{src1.y.size} != {src2.y.size}"
        )
    dst = dst_2_to_1(src1, src2, "replace_x_by_other_y")
    dst.set_xydata(src2.y, src1.y)
    dst.xlabel = src2.ylabel if src2.ylabel else "X"
    dst.xunit = src2.yunit if src2.yunit else ""
    # Y label and unit remain from src1
    restore_data_outside_roi(dst, src1)
    return dst
