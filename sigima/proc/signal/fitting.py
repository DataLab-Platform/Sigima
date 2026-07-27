# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see sigima/LICENSE for details)

"""
Curve fitting operations
========================

This module provides curve fitting operations for signal objects:

- Linear and polynomial fits
- Gaussian, Lorentzian, and Voigt fits
- Exponential and CDF fits

.. note::

    Most operations use functions from :mod:`sigima.tools.signal.fitting` for
    actual computations.
"""

from __future__ import annotations

from typing import Callable

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.objects import SignalObj
from sigima.proc.base import dst_2_to_1
from sigima.proc.decorator import computation_function
from sigima.tools.signal import fitting

from .base import dst_1_to_1


def __generic_fit(
    src: SignalObj,
    fitfunc: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, dict[str, float]]],
) -> SignalObj:
    """Generic fitting function.

    Args:
        src: source signal
        fitfunc: fitting function

    Returns:
        Fitting result signal object
    """
    dst = dst_1_to_1(src, fitfunc.__name__)

    # Fit only on ROI if available
    x_roi = src.x[~src.get_masked_view().mask]
    y_roi = src.get_masked_view().compressed()
    _fitted_y_roi, fit_params = fitfunc(x_roi, y_roi)

    # Evaluate fit on full x range
    fitted_y = fitting.evaluate_fit(src.x, **fit_params)
    dst.set_xydata(src.x, fitted_y)

    # Store fit parameters in metadata
    dst.metadata["fit_params"] = fit_params
    return dst


@computation_function()
def linear_fit(src: SignalObj) -> SignalObj:
    """Compute linear fit with :py:func:`numpy.polyfit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.linear_fit)


class PolynomialFitParam(gds.DataSet, title=_("Polynomial fit")):
    """Polynomial fitting parameters"""

    degree = gds.IntItem(_("Degree"), 3, min=1, max=10, slider=True)


@computation_function()
def polynomial_fit(src: SignalObj, p: PolynomialFitParam) -> SignalObj:
    """Compute polynomial fit with :py:func:`numpy.polyfit`

    Args:
        src: source signal
        p: polynomial fitting parameters

    Returns:
        Result signal object
    """
    # Note: no need to check degree here as gds.IntItem already enforces min=1
    return __generic_fit(src, lambda x, y: fitting.polynomial_fit(x, y, p.degree))


@computation_function()
def gaussian_fit(src: SignalObj) -> SignalObj:
    """Compute Gaussian fit with :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.gaussian_fit)


@computation_function()
def lorentzian_fit(src: SignalObj) -> SignalObj:
    """Compute Lorentzian fit with :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.lorentzian_fit)


@computation_function()
def voigt_fit(src: SignalObj) -> SignalObj:
    """Compute Voigt fit with :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.voigt_fit)


@computation_function()
def exponential_fit(src: SignalObj) -> SignalObj:
    """Compute exponential fit with :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.exponential_fit)


@computation_function()
def cdf_fit(src: SignalObj) -> SignalObj:
    """Compute CDF fit with :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.cdf_fit)


@computation_function()
def planckian_fit(src: SignalObj) -> SignalObj:
    """Compute Planckian fit with :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.planckian_fit)


@computation_function()
def twohalfgaussian_fit(src: SignalObj) -> SignalObj:
    """Compute two-half-Gaussian fit with :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.twohalfgaussian_fit)


@computation_function()
def sigmoid_fit(src: SignalObj) -> SignalObj:
    """Compute sigmoid fit with :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.sigmoid_fit)


@computation_function()
def piecewiseexponential_fit(src: SignalObj) -> SignalObj:
    """Compute piecewise exponential fit (raise-decay) with
    :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.piecewiseexponential_fit)


@computation_function()
def sinusoidal_fit(src: SignalObj) -> SignalObj:
    """Compute sinusoidal fit with :py:func:`scipy.optimize.curve_fit`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return __generic_fit(src, fitting.sinusoidal_fit)


def extract_fit_params(signal: SignalObj) -> fitting.FitParams:
    """Extract fit parameters from a fitted signal.

    Args:
        signal: Signal object containing fit metadata

    Returns:
        Fit parameters
    """
    if "fit_params" not in signal.metadata:
        raise ValueError("Signal does not contain fit parameters")
    fit_params_dict = signal.metadata["fit_params"]
    if not isinstance(fit_params_dict, dict):
        raise ValueError("Signal fit parameters must be a dictionary")
    fitting.validate_fit_params(fit_params_dict)
    return fit_params_dict


@computation_function()
def evaluate_fit(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Evaluate fit function from src1 on the x-axis of src2.

    This function extracts fit parameters from `src1` (which must contain fit metadata
    from a previous fitting operation) and evaluates the fit function on the x-axis
    of `src2`.

    Args:
        src1: Signal object containing fit parameters in metadata (from a fit operation)
        src2: Signal object whose x-axis will be used for evaluation

    Returns:
        New signal with the fit evaluated on src2's x-axis
    """
    fit_params = extract_fit_params(src1)
    dst = dst_2_to_1(src1, src2, "evaluate_fit")

    # Evaluate fit on src2's x-axis
    x = src2.x
    y = fitting.evaluate_fit(x, **fit_params)

    dst.set_xydata(x, y)
    dst.title = f"Fitted {fit_params['fit_type']}"

    # Copy fit parameters to destination metadata
    dst.metadata["fit_params"] = fit_params
    return dst
