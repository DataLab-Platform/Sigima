# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Curve Fitting Algorithms
=========================

This module provides curve fitting functions without GUI dependencies.
The functions take x,y data and return fitted curves and parameters.

These functions are designed to be used programmatically and in tests,
providing the core fitting algorithms without PlotPy GUI components.
"""

from __future__ import annotations

import string
import warnings
from collections.abc import Mapping
from numbers import Real
from typing import Type, Union

import numpy as np
import scipy.optimize
import scipy.special

from sigima.tools.signal import pulse

FIT_PARAMS_VERSION = 2
PEAK_PARAMETERIZATION = pulse.PEAK_PARAMETERIZATION
PEAK_FIT_TYPES = frozenset(
    {"gaussian", "lorentzian", "voigt", "multigaussian", "multilorentzian"}
)
FIT_PARAMS_METADATA_KEYS = frozenset(
    {
        "fit_type",
        "residual_rms",
        "fit_params_version",
        "peak_parameterization",
        "interactive",
    }
)
FitParamValue = Union[float, int, str, bool]
FitParams = dict[str, FitParamValue]


def _is_finite_real(value: object) -> bool:
    """Return whether a value is a finite, non-boolean real number."""
    return (
        isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)
    )


def _is_legacy_amplitude_key(name: str) -> bool:
    """Return whether *name* is an old area-based peak parameter."""
    return name == "amp" or (
        name.startswith("amp_") and name.removeprefix("amp_").isdigit()
    )


def create_fit_params(
    fit_type: str,
    values: Mapping[str, Real],
    *,
    residual_rms: Real | None = None,
    interactive: bool = False,
) -> FitParams:
    """Create a canonical fit-parameter dictionary.

    Peak fits carry both a structural version and an explicit scientific
    parameterization marker. Other fit types retain their historical shape.

    Args:
        fit_type: Fit type identifier used by :func:`evaluate_fit`.
        values: Numeric model parameter values.
        residual_rms: Optional root-mean-square residual.
        interactive: Whether the fit was committed from an interactive editor.

    Returns:
        Canonical fit-parameter dictionary.
    """
    reserved = FIT_PARAMS_METADATA_KEYS.intersection(values)
    if reserved:
        raise ValueError(f"Reserved fit parameter keys: {sorted(reserved)}")
    invalid_values = [
        name for name, value in values.items() if not _is_finite_real(value)
    ]
    if invalid_values:
        raise ValueError(f"Non-numeric fit parameters: {sorted(invalid_values)}")
    if residual_rms is not None and not _is_finite_real(residual_rms):
        raise ValueError("Residual RMS must be a finite real number")
    params: FitParams = {name: float(value) for name, value in values.items()}
    params["fit_type"] = fit_type
    if residual_rms is not None:
        params["residual_rms"] = float(residual_rms)
    if interactive:
        params["interactive"] = True
    if fit_type in PEAK_FIT_TYPES:
        params["fit_params_version"] = FIT_PARAMS_VERSION
        params["peak_parameterization"] = PEAK_PARAMETERIZATION
    validate_fit_params(params)
    return params


def convert_legacy_peak_fit_params(
    fit_params: Mapping[str, FitParamValue],
) -> FitParams:
    """Convert an area-based peak fit dictionary to the height-based schema.

    The input mapping is never mutated.

    Args:
        fit_params: Historical, unversioned peak fit parameters.

    Returns:
        Converted copy using ``amplitude`` parameter names and schema version 2.

    Raises:
        ValueError: If the mapping is incomplete, already versioned or not a
         supported peak fit.
    """
    fit_type = fit_params.get("fit_type")
    if not isinstance(fit_type, str) or fit_type not in PEAK_FIT_TYPES:
        raise ValueError(f"Unsupported legacy peak fit type: {fit_type!r}")
    if "fit_params_version" in fit_params or "peak_parameterization" in fit_params:
        raise ValueError("Fit parameters are already versioned")
    current_names = [
        name
        for name in fit_params
        if name == "amplitude" or name.startswith("amplitude_")
    ]
    if current_names:
        raise ValueError(
            f"Legacy peak fit parameters mix old and current keys: {current_names}"
        )

    model_by_type: dict[str, Type[pulse.PulseFitModel]] = {
        "gaussian": pulse.GaussianModel,
        "lorentzian": pulse.LorentzianModel,
        "voigt": pulse.VoigtModel,
        "multigaussian": pulse.GaussianModel,
        "multilorentzian": pulse.LorentzianModel,
    }
    model = model_by_type[fit_type]
    converted = dict(fit_params)
    legacy_names = sorted(name for name in converted if _is_legacy_amplitude_key(name))
    if not legacy_names:
        raise ValueError("No legacy area parameter found")
    for legacy_name in legacy_names:
        suffix = legacy_name.removeprefix("amp")
        sigma_name = f"sigma{suffix}"
        if sigma_name not in converted:
            raise ValueError(f"Missing parameter required for conversion: {sigma_name}")
        area = converted.pop(legacy_name)
        sigma = converted[sigma_name]
        if not _is_finite_real(area) or not _is_finite_real(sigma):
            raise ValueError(f"Non-numeric legacy peak parameters: {legacy_name}")
        converted[f"amplitude{suffix}"] = float(model.amplitude_from_area(area, sigma))
    converted["fit_params_version"] = FIT_PARAMS_VERSION
    converted["peak_parameterization"] = PEAK_PARAMETERIZATION
    validate_fit_params(converted)
    return converted


class FitComputer:
    """Base class for fit computers"""

    PARAMS_NAMES: tuple[str] = ()  # To be defined by subclasses

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def get_params_names(self) -> tuple[str]:
        """Return the names of the parameters used in this fit."""
        return self.PARAMS_NAMES

    def check_params(self, **params) -> None:
        """Check that all required parameters are provided."""
        missing = [p for p in self.get_params_names() if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

    @classmethod
    def args_kwargs_to_list(cls, *args, **kwargs) -> list[float]:
        """Convert args and kwargs to a parameter list."""
        if kwargs and args:
            raise ValueError("Cannot mix positional and keyword arguments")
        if cls.PARAMS_NAMES:
            param_names = cls.PARAMS_NAMES
        else:
            if not kwargs:
                raise ValueError("No parameter names available and no kwargs provided")
            param_names = cls.infer_param_names_from_kwargs(kwargs)
        if len(args) > len(param_names):
            raise ValueError("Too many positional arguments")
        if args:
            params = list(args)
        else:
            params = []
            for name in param_names:
                if name in kwargs:
                    params.append(kwargs[name])
                else:
                    raise ValueError(f"Missing required parameter: {name}")
        return params

    @classmethod
    def infer_param_names_from_kwargs(cls, kwargs: dict) -> tuple[str, ...]:
        """Infer parameter names from kwargs. Override in subclasses if needed."""
        return tuple(kwargs.keys())

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate the fit function at given x values."""
        raise NotImplementedError("Subclasses must implement evaluate method")

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for fitting. To be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement compute_initial_params method"
        )

    # pylint: disable=unused-argument
    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for fitting."""
        return None

    def create_params(self, y_fitted: np.ndarray, **params) -> FitParams:
        """Create a fit parameters dictionary from given parameters."""
        self.check_params(**params)
        fit_type = self.__class__.__name__.replace("FitComputer", "").lower()
        residual_rms = np.sqrt(np.mean((self.y - y_fitted) ** 2))
        return create_fit_params(fit_type, params, residual_rms=residual_rms)

    def fit(self) -> tuple[np.ndarray, dict[str, float]]:
        """Fit the model to the data."""
        # Default implementation uses scipy curve_fit
        return self.optimize_fit_with_scipy()

    def optimize_fit_with_scipy(self) -> tuple[np.ndarray, np.ndarray]:
        """Generic fitting function using `scipy.optimize.curve_fit`

        Returns:
            tuple: (fitted_y_values, fitted_parameters)
        """
        initial_params = self.compute_initial_params()
        bounds = self.compute_bounds(**initial_params)  # pylint: disable=E1128
        if bounds is not None:
            # Convert bounds to scipy format
            lower_bounds = [b[0] for b in bounds]
            upper_bounds = [b[1] for b in bounds]
            bounds_scipy = (lower_bounds, upper_bounds)
        else:
            bounds_scipy = (-np.inf, np.inf)

        # Create a wrapper function that unpacks parameters correctly
        def objective_func(x, *params):
            """Wrapper function for scipy curve_fit."""
            param_dict = dict(zip(self.get_params_names(), params))
            try:
                # Try as classmethod first
                return self.__class__.evaluate(x, **param_dict)
            except TypeError:
                # Fall back to instance method
                return self.evaluate(x, **param_dict)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=scipy.optimize.OptimizeWarning
                )
                popt, _ = scipy.optimize.curve_fit(
                    objective_func,
                    self.x,
                    self.y,
                    p0=list(initial_params.values()),
                    bounds=bounds_scipy,
                    maxfev=5000,
                )
        except (RuntimeError, ValueError, TypeError) as err:
            # Fallback to initial parameters if optimization fails
            warnings.warn(f"Optimization failed: {err}. Using initial parameters.")
            try:
                # Try as classmethod first
                fitted_y = self.__class__.evaluate(self.x, **initial_params)
            except TypeError:
                # Fall back to instance method
                fitted_y = self.evaluate(self.x, **initial_params)
            result_params = self.create_params(fitted_y, **initial_params)
            return fitted_y, result_params

        names = self.get_params_names()
        assert len(popt) == len(names), "Unexpected number of parameters"
        param_dict = dict(zip(names, popt))
        try:
            # Try as classmethod first
            fitted_y = self.__class__.evaluate(self.x, **param_dict)
        except TypeError:
            # Fall back to instance method
            fitted_y = self.evaluate(self.x, **param_dict)
        params = self.create_params(fitted_y, **param_dict)
        return fitted_y, params


class LinearFitComputer(FitComputer):
    """Linear fit computer"""

    PARAMS_NAMES = ("a", "b")  # slope and intercept

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate linear function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        a, b = cls.args_kwargs_to_list(*args, **kwargs)
        return a * x + b

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for linear fitting using numpy polyfit."""
        coeffs = np.polyfit(self.x, self.y, 1)
        a, b = coeffs
        return {"a": a, "b": b}


class PolynomialFitComputer(FitComputer):
    """Polynomial fit computer of given degree"""

    def __init__(self, x: np.ndarray, y: np.ndarray, degree: int = 2) -> None:
        super().__init__(x, y)
        if degree < 1:
            raise ValueError("Degree must be at least 1")
        self.degree = degree

    def get_params_names(self) -> tuple[str]:
        """Return the names of the parameters used in this fit."""
        return tuple(string.ascii_lowercase[: self.degree + 1])

    @classmethod
    def infer_param_names_from_kwargs(cls, kwargs: dict) -> tuple[str, ...]:
        """Infer parameter names for polynomial from kwargs."""
        # Parameters are named 'a', 'b', 'c', ... in order
        param_keys = [k for k in kwargs.keys() if k in string.ascii_lowercase]
        if not param_keys:
            raise ValueError("No valid polynomial parameters found")
        # Sort to ensure correct order (a, b, c, ...)
        return tuple(sorted(param_keys))

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate polynomial function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        coeffs = cls.args_kwargs_to_list(*args, **kwargs)
        return np.polyval(coeffs, x)

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for polynomial fitting using numpy polyfit."""
        coeffs = np.polyfit(self.x, self.y, self.degree)
        param_names = self.get_params_names()

        # Map numpy polyfit coefficients (highest to lowest degree) to parameter names
        return dict(zip(param_names, coeffs))


def _estimate_peak_baseline(y: np.ndarray) -> float:
    """Estimate a peak signal's baseline from edge samples."""
    edge_size = max(1, len(y) // 10)
    return float(np.median(np.concatenate((y[:edge_size], y[-edge_size:]))))


def _estimate_signed_peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Estimate signed peak height, center and baseline from edge samples."""
    baseline = _estimate_peak_baseline(y)
    deviations = y - baseline
    positive_index = int(np.argmax(deviations))
    negative_index = int(np.argmin(deviations))
    peak_index = (
        positive_index
        if deviations[positive_index] >= abs(deviations[negative_index])
        else negative_index
    )
    return float(deviations[peak_index]), float(x[peak_index]), baseline


class GaussianFitComputer(FitComputer):
    """Gaussian fit computer"""

    PARAMS_NAMES = ("amplitude", "sigma", "x0", "y0")

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate Gaussian function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        amplitude, sigma, x0, y0 = cls.args_kwargs_to_list(*args, **kwargs)
        return pulse.GaussianModel.evaluate(x, amplitude, sigma, x0, y0)

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for Gaussian fitting."""
        dx = np.max(self.x) - np.min(self.x)
        sigma = dx * 0.1
        amplitude, x0, y0 = _estimate_signed_peak(self.x, self.y)

        return {"amplitude": amplitude, "sigma": sigma, "x0": x0, "y0": y0}

    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for Gaussian fitting."""
        dy = np.max(self.y) - np.min(self.y)
        y_min = np.min(self.y)
        y_max = np.max(self.y)

        return [
            (-2.0 * dy, 2.0 * dy),  # amplitude
            (initial_params["sigma"] * 0.1, initial_params["sigma"] * 10),  # sigma
            (np.min(self.x), np.max(self.x)),  # x0
            (y_min - 0.2 * dy, y_max + 0.2 * dy),  # y0
        ]


class LorentzianFitComputer(FitComputer):
    """Lorentzian fit computer"""

    PARAMS_NAMES = ("amplitude", "sigma", "x0", "y0")

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate Lorentzian function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        amplitude, sigma, x0, y0 = cls.args_kwargs_to_list(*args, **kwargs)
        return pulse.LorentzianModel.evaluate(x, amplitude, sigma, x0, y0)

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for Lorentzian fitting."""
        dx = np.max(self.x) - np.min(self.x)
        sigma = dx * 0.1
        amplitude, x0, y0 = _estimate_signed_peak(self.x, self.y)

        return {"amplitude": amplitude, "sigma": sigma, "x0": x0, "y0": y0}

    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for Lorentzian fitting."""
        dy = np.max(self.y) - np.min(self.y)
        y_min = np.min(self.y)
        y_max = np.max(self.y)

        return [
            (-2.0 * dy, 2.0 * dy),  # amplitude
            (initial_params["sigma"] * 0.1, initial_params["sigma"] * 10),  # sigma
            (np.min(self.x), np.max(self.x)),  # x0
            (y_min - 0.2 * dy, y_max + 0.2 * dy),  # y0
        ]


class VoigtFitComputer(FitComputer):
    """Voigt fit computer"""

    PARAMS_NAMES = ("amplitude", "sigma", "x0", "y0")

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate Voigt function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        amplitude, sigma, x0, y0 = cls.args_kwargs_to_list(*args, **kwargs)
        return pulse.VoigtModel.evaluate(x, amplitude, sigma, x0, y0)

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for Voigt fitting."""
        dx = np.max(self.x) - np.min(self.x)
        sigma = dx * 0.1
        amplitude, x0, y0 = _estimate_signed_peak(self.x, self.y)

        return {"amplitude": amplitude, "sigma": sigma, "x0": x0, "y0": y0}

    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for Voigt fitting."""
        sigma = initial_params["sigma"]
        dy = np.max(self.y) - np.min(self.y)
        return [
            (-10.0 * dy, 10.0 * dy),  # amplitude
            (sigma * 0.01, sigma * 10),  # sigma
            (np.min(self.x), np.max(self.x)),  # x0
            (
                np.min(self.y) - dy,
                np.max(self.y) + dy,
            ),  # y0
        ]


class ExponentialFitComputer(FitComputer):
    """Exponential fit computer: y = a * exp(b * x) + y0"""

    PARAMS_NAMES = ("a", "b", "y0")

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate exponential function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        a, b, y0 = cls.args_kwargs_to_list(*args, **kwargs)
        # Clip b to prevent overflow
        b_clipped = np.clip(b, -50, 50)
        return a * np.exp(b_clipped * x) + y0

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for exponential fitting."""
        y_range = np.max(self.y) - np.min(self.y)
        y_min = np.min(self.y)

        # Estimate from data
        if len(self.y) > 1:
            # Try to determine if it's growth or decay
            if self.y[0] > self.y[-1]:
                # Decay
                a = y_range
                b = -1.0 / (np.max(self.x) - np.min(self.x))
            else:
                # Growth
                a = y_range * 0.1
                b = 1.0 / (np.max(self.x) - np.min(self.x))
        else:
            a = y_range
            b = -1.0

        y0 = y_min

        return {"a": a, "b": b, "y0": y0}

    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for exponential fitting."""
        y_range = np.max(self.y) - np.min(self.y)
        y_min = np.min(self.y)

        return [
            (-y_range * 1000, y_range * 1000),  # a
            (-10, 10),  # b (reasonable range to prevent overflow)
            (y_min - 0.5 * y_range, y_min + 0.5 * y_range),  # y0
        ]


class PlanckianFitComputer(FitComputer):
    """Planckian fit computer"""

    PARAMS_NAMES = ("amp", "x0", "sigma", "y0")

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Return Planckian fitting function

        Args:
            x: wavelength values (in nm or other units)
            amp: amplitude scaling factor
            x0: peak wavelength (Wien's displacement law)
            sigma: width parameter (larger sigma = wider peak)
            y0: baseline offset
        """
        # pylint: disable=unbalanced-tuple-unpacking
        amp, x0, sigma, y0 = cls.args_kwargs_to_list(*args, **kwargs)

        # Planck-like function with Wien's displacement law behavior
        # The function peaks at approximately x0 when properly parameterized

        x = np.asarray(x, dtype=float)
        y = np.full_like(x, y0, dtype=float)

        # Only compute for positive wavelengths
        valid_mask = x > 0
        if not np.any(valid_mask):
            return y

        x_valid = x[valid_mask]

        try:
            # Wien's displacement law: λ_max * T = constant
            # For a proper Planckian curve, we need:
            # d/dx [x^(-5) / (exp(c/x) - 1)] = 0 at x = x0
            # This gives us c = 5*x0 for the peak condition

            # The exponential argument that produces peak at x0
            wien_constant = 5.0

            # Use sigma to control the effective temperature/width
            # sigma=1.0 gives the canonical Planck curve
            # sigma>1.0 gives broader curves (cooler)
            # sigma<1.0 gives sharper curves (hotter)
            temperature_factor = sigma

            exp_argument = wien_constant * x0 / (x_valid * temperature_factor)

            # Clip to prevent overflow
            exp_argument = np.clip(exp_argument, 0, 50)

            # Planck function components:
            # 1. The wavelength dependence: x^(-5)
            wavelength_factor = (x_valid / x0) ** (-5)

            # 2. The exponential term: 1/(exp(arg) - 1)
            exp_denominator = np.expm1(exp_argument)  # exp(x) - 1

            # Avoid division by very small numbers
            exp_denominator = np.where(
                np.abs(exp_denominator) < 1e-12, 1e-12, exp_denominator
            )

            # Combine the Planckian terms
            planck_curve = wavelength_factor / exp_denominator

            # Apply amplitude and add to baseline
            y[valid_mask] += amp * planck_curve

        except (OverflowError, ZeroDivisionError, RuntimeWarning):
            # If computation fails, return baseline only
            pass

        return y

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for Planckian fitting."""
        dy = np.max(self.y) - np.min(self.y)
        x_peak = self.x[np.argmax(self.y)]
        y_min = np.min(self.y)
        return {"amp": dy, "x0": x_peak, "sigma": 1.0, "y0": y_min}

    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for Planckian fitting."""
        return [
            (initial_params["amp"] * 0.01, initial_params["amp"] * 100),  # amp
            (np.min(self.x), np.max(self.x)),  # x0
            (0.1, 5.0),  # sigma
            (
                initial_params["y0"] - 0.2 * initial_params["amp"],
                initial_params["y0"] + 0.2 * initial_params["amp"],
            ),  # y0
        ]


class TwoHalfGaussianFitComputer(FitComputer):
    """Two Half-Gaussian fit computer"""

    PARAMS_NAMES = (
        "amp_left",
        "amp_right",
        "sigma_left",
        "sigma_right",
        "x0",
        "y0_left",
        "y0_right",
    )

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Return two half-Gaussian with separate left/right amplitudes

        Args:
            x: x values
            amp_left: amplitude for left side (x < x0)
            amp_right: amplitude for right side (x >= x0)
            sigma_left: standard deviation for x < x0
            sigma_right: standard deviation for x > x0
            x0: center position
            y0_left: baseline offset for x < x0
            y0_right: baseline offset for x >= x0
        """
        # pylint: disable=unbalanced-tuple-unpacking
        amp_left, amp_right, sigma_left, sigma_right, x0, y0_left, y0_right = (
            cls.args_kwargs_to_list(*args, **kwargs)
        )

        y = np.zeros_like(x)

        # Left side (x < x0): use amp_left, sigma_left and y0_left
        left_mask = x < x0
        if np.any(left_mask):
            exp_left = np.exp(-0.5 * ((x[left_mask] - x0) / sigma_left) ** 2)
            y[left_mask] = y0_left + amp_left * exp_left

        # Right side (x >= x0): use amp_right, sigma_right and y0_right
        right_mask = x >= x0
        if np.any(right_mask):
            exp_right = np.exp(-0.5 * ((x[right_mask] - x0) / sigma_right) ** 2)
            y[right_mask] = y0_right + amp_right * exp_right

        return y

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for Two Half-Gaussian fitting."""
        # Parameter estimation with separate baseline analysis
        dx = np.max(self.x) - np.min(self.x)
        dy = np.max(self.y) - np.min(self.y)
        x_peak = self.x[np.argmax(self.y)]

        # Estimate separate baselines for left and right sides
        left_mask = self.x < x_peak
        right_mask = self.x >= x_peak

        # Use the lower quartile of each side for robust baseline estimation
        if np.any(left_mask):
            y0_left = np.percentile(self.y[left_mask], 25)
        else:
            y0_left = np.min(self.y)
        if np.any(right_mask):
            y0_right = np.percentile(self.y[right_mask], 25)
        else:
            y0_right = np.min(self.y)

        # Peak amplitude estimation (above average baseline)
        avg_baseline = (y0_left + y0_right) / 2
        amp_guess = np.max(self.y) - avg_baseline
        half_max = avg_baseline + amp_guess * 0.5

        # Find points at half maximum
        left_points = np.where((self.x < x_peak) & (self.y >= half_max))[0]
        right_points = np.where((self.x > x_peak) & (self.y >= half_max))[0]

        # Estimate sigma values from half-width measurements
        if len(left_points) > 0:
            left_hw = x_peak - self.x[left_points[0]]
            sigma_left = left_hw / np.sqrt(2 * np.log(2))
        else:
            sigma_left = dx * 0.05

        if len(right_points) > 0:
            right_hw = self.x[right_points[-1]] - x_peak
            sigma_right = right_hw / np.sqrt(2 * np.log(2))
        else:
            sigma_right = dx * 0.05

        x0 = x_peak

        if np.any(left_mask):
            left_peak_val = np.max(self.y[left_mask])
            amp_left = left_peak_val - y0_left
        else:
            amp_left = dy * 0.5

        if np.any(right_mask):
            right_peak_val = np.max(self.y[right_mask])
            amp_right = right_peak_val - y0_right
        else:
            amp_right = dy * 0.5

        return {
            "amp_left": amp_left,
            "amp_right": amp_right,
            "sigma_left": sigma_left,
            "sigma_right": sigma_right,
            "x0": x0,
            "y0_left": y0_left,
            "y0_right": y0_right,
        }


class DoubleExponentialFitComputer(FitComputer):
    """Piecewise exponential (raise-decay) fit computer."""

    PARAMS_NAMES = ("x_center", "a_left", "b_left", "a_right", "b_right", "y0")

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Return piecewise exponential (raise-decay) fitting function

        Args:
            x: time values
            x_center: center position (boundary between left and right components)
            a_left: left component amplitude coefficient
            b_left: left component time constant coefficient
            a_right: right component amplitude coefficient
            b_right: right component time constant coefficient
            y0: baseline offset
        """
        # pylint: disable=unbalanced-tuple-unpacking
        x_center, a_left, b_left, a_right, b_right, y0 = cls.args_kwargs_to_list(
            *args, **kwargs
        )
        y = np.zeros_like(x)
        y[x < x_center] = a_left * np.exp(b_left * x[x < x_center]) + y0
        y[x >= x_center] = a_right * np.exp(b_right * x[x >= x_center]) + y0
        return y

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for piecewise exponential (raise-decay)
        fitting."""
        y_range = np.max(self.y) - np.min(self.y)
        x_range = np.max(self.x) - np.min(self.x)
        y_max = np.max(self.y)

        # Baseline is rarely different from zero:
        y0 = 0.0

        # Analyze signal characteristics for better initial guesses
        peak_idx = np.argmax(self.y)

        # Estimate x_center as the peak position
        x_center = self.x[peak_idx]

        # Estimate parameters (a_left, b_left, a_right, b_right) by decomposing
        # the signal into growth and decay components based on peak position, and
        # fitting each curve with exponential functions using exponential_fit().
        # X center estimation is very rough here, so we need to remove say 10% of
        # the x range on each side to avoid fitting artifacts.
        x_range = np.max(self.x) - np.min(self.x)
        x_left_mask = self.x < (x_center - 0.1 * x_range)
        x_right_mask = self.x >= (x_center + 0.1 * x_range)

        x_left, y_left = self.x[x_left_mask], self.y[x_left_mask]
        x_right, y_right = self.x[x_right_mask], self.y[x_right_mask]

        left_params = {"a": 0.0, "b": 0.1, "y0": 0.0}
        right_params = {"a": 0.0, "b": 0.1, "y0": 0.0}
        if np.any(x_left_mask):
            _y_fitted, left_params = ExponentialFitComputer(x_left, y_left).fit()
        if np.any(x_right_mask):
            _y_fitted, right_params = ExponentialFitComputer(x_right, y_right).fit()

        a_left = left_params["a"]
        b_left = left_params["b"]
        a_right = right_params["a"]
        b_right = right_params["b"]
        y0 = (left_params["y0"] + right_params["y0"]) / 2

        # Set bounds for parameters - b can be positive or negative
        amp_bound = max(abs(y_max - y0), y_range) * 2
        rate_bound = 5.0 / max(x_range, 1e-6)  # Avoid division by zero

        # Ensure initial parameters are within bounds
        b_left = np.clip(b_left, -rate_bound, rate_bound)
        b_right = np.clip(b_right, -rate_bound, rate_bound)
        a_left = np.clip(a_left, -amp_bound, amp_bound)
        a_right = np.clip(a_right, -amp_bound, amp_bound)

        return {
            "x_center": x_center,
            "a_left": a_left,
            "b_left": b_left,
            "a_right": a_right,
            "b_right": b_right,
            "y0": y0,
        }


class BaseMultiPeakFitComputer(FitComputer):
    """Base class for multi-peak fit computers"""

    PULSE_MODEL: Type[pulse.PulseFitModel]  # To be defined by subclasses

    def __init__(
        self, x: np.ndarray, y: np.ndarray, peak_indices: list[int] | None = None
    ) -> None:
        super().__init__(x, y)
        self.peak_indices = peak_indices

    def get_params_names(self) -> tuple[str]:
        """Return the names of the parameters used in this fit."""
        n_peaks = len(self.peak_indices)
        names = []
        for i in range(n_peaks):
            names.extend([f"amplitude_{i + 1}", f"sigma_{i + 1}", f"x0_{i + 1}"])
        names.append("y0")
        return tuple(names)

    @classmethod
    def infer_param_names_from_kwargs(cls, kwargs: dict) -> tuple[str, ...]:
        """Infer parameter names for multi-gaussian from kwargs."""
        # Find all amplitude_X parameters to count peaks
        amplitude_params = [k for k in kwargs.keys() if k.startswith("amplitude_")]
        n_peaks = len(amplitude_params)
        if n_peaks == 0:
            raise ValueError("No amplitude parameters found")

        names = []
        for i in range(1, n_peaks + 1):
            names.extend([f"amplitude_{i}", f"sigma_{i}", f"x0_{i}"])
        names.append("y0")
        return tuple(names)

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate the fit function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        paramlist = cls.args_kwargs_to_list(*args, **kwargs)
        # Determine number of peaks from parameter count
        n_peaks = (
            len(paramlist) - 1
        ) // 3  # -1 for y0, then divide by 3 params per peak
        y_result = np.zeros_like(x) + paramlist[-1]
        for i in range(n_peaks):
            amplitude, sigma, x0 = paramlist[3 * i : 3 * i + 3]
            y_result += cls.PULSE_MODEL.evaluate(x, amplitude, sigma, x0, 0.0)
        return y_result

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for Multi Gaussian fitting."""
        params = {}
        baseline = _estimate_peak_baseline(self.y)
        for i, peak_idx in enumerate(self.peak_indices):
            if i > 0:
                istart = (self.peak_indices[i - 1] + peak_idx) // 2
            else:
                istart = 0
            if i < len(self.peak_indices) - 1:
                iend = (self.peak_indices[i + 1] + peak_idx) // 2
            else:
                iend = len(self.x) - 1
            local_dx = 0.5 * (self.x[iend] - self.x[istart])
            amplitude = self.y[peak_idx] - baseline
            sigma = local_dx * 0.1
            x0 = self.x[peak_idx]

            params[f"amplitude_{i + 1}"] = amplitude
            params[f"sigma_{i + 1}"] = sigma
            params[f"x0_{i + 1}"] = x0

        params["y0"] = baseline
        return params

    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for Multi Lorentzian fitting."""
        bounds = []
        for i, peak_idx in enumerate(self.peak_indices):
            if i > 0:
                istart = (self.peak_indices[i - 1] + peak_idx) // 2
            else:
                istart = 0
            if i < len(self.peak_indices) - 1:
                iend = (self.peak_indices[i + 1] + peak_idx) // 2
            else:
                iend = len(self.x) - 1
            local_dx = 0.5 * (self.x[iend] - self.x[istart])
            local_dy = np.max(self.y[istart:iend]) - np.min(self.y[istart:iend])
            amplitude_range = 10.0 * max(
                local_dy, abs(initial_params[f"amplitude_{i + 1}"])
            )
            bounds.extend(
                [
                    (-amplitude_range, amplitude_range),  # amplitude
                    (local_dx * 0.001, local_dx * 10.0),  # sigma
                    (self.x[istart], self.x[iend]),  # x0
                ]
            )
        y0 = initial_params["y0"]
        dy = np.max(self.y) - np.min(self.y)
        bounds.append((y0 - dy, y0 + dy))
        return bounds


class MultiGaussianFitComputer(BaseMultiPeakFitComputer):
    """Multi Gaussian fit computer"""

    PULSE_MODEL = pulse.GaussianModel


class MultiLorentzianFitComputer(BaseMultiPeakFitComputer):
    """Multi Lorentzian fit computer"""

    PULSE_MODEL = pulse.LorentzianModel


class SinusoidalFitComputer(FitComputer):
    """Sinusoidal fit computer."""

    PARAMS_NAMES = ("amplitude", "frequency", "phase", "offset")

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate sinusoidal function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        amplitude, frequency, phase, offset = cls.args_kwargs_to_list(*args, **kwargs)
        return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for sinusoidal fitting."""
        # Parameter estimation using FFT for frequency
        dy = np.max(self.y) - np.min(self.y)
        amplitude = dy / 2
        offset = np.mean(self.y)
        phase = 0.0

        # Estimate frequency using FFT
        if len(self.x) > 2:
            dt = self.x[1] - self.x[0]  # Assuming evenly spaced
            fft_y = np.fft.fft(self.y - offset)
            freqs = np.fft.fftfreq(len(self.y), dt)
            # Find dominant frequency (excluding DC component)
            dominant_idx = np.argmax(np.abs(fft_y[1 : len(fft_y) // 2])) + 1
            frequency = np.abs(freqs[dominant_idx])
        else:
            frequency = 1.0 / (np.max(self.x) - np.min(self.x))

        return {
            "amplitude": amplitude,
            "frequency": frequency,
            "phase": phase,
            "offset": offset,
        }

    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for sinusoidal fitting."""
        dy = initial_params["amplitude"] * 2
        y0 = initial_params["offset"]
        return [
            (0.0, dy),  # amplitude
            (0.0, 2.0 * initial_params["frequency"]),  # frequency
            (-2 * np.pi, 2 * np.pi),  # phase
            (y0 - dy, y0 + dy),  # offset
        ]


class CDFFitComputer(FitComputer):
    """Cumulative Distribution Function (CDF) fit computer"""

    PARAMS_NAMES = ("amplitude", "mu", "sigma", "baseline")

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate CDF function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        amplitude, mu, sigma, baseline = cls.args_kwargs_to_list(*args, **kwargs)
        erf = scipy.special.erf  # pylint: disable=no-member
        return amplitude * erf((x - mu) / (sigma * np.sqrt(2))) + baseline

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for CDF fitting."""
        # Parameter estimation
        y_min, y_max = np.min(self.y), np.max(self.y)
        dy = y_max - y_min
        x_min, x_max = np.min(self.x), np.max(self.x)
        dx = x_max - x_min
        return {
            # `amplitude * erf(...)` spans `2 * amplitude` peak-to-peak:
            "amplitude": dy / 2,
            "mu": (x_min + x_max) / 2,
            "sigma": dx / 10,
            # `baseline` is a level, not a range:
            "baseline": y_min + dy / 2,
        }

    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for CDF fitting."""
        y_min, y_max = np.min(self.y), np.max(self.y)
        dy = initial_params["amplitude"]
        x_min, x_max = np.min(self.x), np.max(self.x)
        dx = x_max - x_min
        return [
            (0.0, dy * 2),  # amplitude
            (x_min, x_max),  # mu
            (dx * 0.001, dx),  # sigma
            (y_min - dy, y_max + dy),  # baseline
        ]


class SigmoidFitComputer(FitComputer):
    """Sigmoid fit computer."""

    PARAMS_NAMES = ("amplitude", "k", "x0", "offset")

    @classmethod
    def evaluate(cls, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Evaluate Sigmoid function at given x values."""
        # pylint: disable=unbalanced-tuple-unpacking
        amplitude, k, x0, offset = cls.args_kwargs_to_list(*args, **kwargs)
        return amplitude / (1 + np.exp(-k * (x - x0))) + offset

    def compute_initial_params(self) -> dict[str, float]:
        """Compute initial parameters for Sigmoid fitting."""
        y_min, y_max = np.min(self.y), np.max(self.y)
        dy = y_max - y_min
        x_min, x_max = np.min(self.x), np.max(self.x)
        dx = x_max - x_min
        return {
            "amplitude": dy,
            "k": 4.0 / dx,
            "x0": (x_min + x_max) / 2,
            "offset": y_min,
        }

    def compute_bounds(self, **initial_params) -> list[tuple[float, float]] | None:
        """Compute parameter bounds for Sigmoid fitting."""
        y_min, y_max = np.min(self.y), np.max(self.y)
        dy = initial_params["amplitude"]
        x_min, x_max = np.min(self.x), np.max(self.x)
        dx = x_max - x_min
        return [
            (0.0, 10 * dy),  # amplitude
            (0.1 / dx, 100.0 / dx),  # k
            (x_min, x_max),  # x0
            (y_min - dy, y_max + dy),  # offset
        ]


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute linear fit: y = a*x + b.

    Args:
        x: x data array
        y: y data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return LinearFitComputer(x, y).fit()


def polynomial_fit(
    x: np.ndarray, y: np.ndarray, degree: int = 2
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute polynomial fit.

    Args:
        x: x data array
        y: y data array
        degree: polynomial degree

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return PolynomialFitComputer(x, y, degree).fit()


def gaussian_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute Gaussian fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return GaussianFitComputer(x, y).fit()


def lorentzian_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compute Lorentzian fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return LorentzianFitComputer(x, y).fit()


def exponential_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute exponential fit: y = a * exp(b * x) + y0.

    Args:
        x: x data array
        y: y data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return ExponentialFitComputer(x, y).fit()


def planckian_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute Planckian (blackbody radiation) fit.

    Args:
        x: wavelength data array
        y: intensity data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return PlanckianFitComputer(x, y).fit()


def twohalfgaussian_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute two half-Gaussian fit for asymmetric peaks with separate baselines.

    Now supports separate amplitudes for even better asymmetric peak fitting.

    Args:
        x: x data array
        y: y data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return TwoHalfGaussianFitComputer(x, y).fit()


def piecewiseexponential_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute piecewise exponential fit (raise-decay).

    Args:
        x: time data array
        y: intensity data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return DoubleExponentialFitComputer(x, y).fit()


def multilorentzian_fit(
    x: np.ndarray, y: np.ndarray, peak_indices: list[int]
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute multi-Lorentzian fit for multiple peaks.

    Args:
        x: x data array
        y: y data array
        peak_indices: list of peak indices

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return MultiLorentzianFitComputer(x, y, peak_indices).fit()


def multigaussian_fit(
    x: np.ndarray, y: np.ndarray, peak_indices: list[int]
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute multi-Gaussian fit for multiple peaks.

    Args:
        x: x data array
        y: y data array
        peak_indices: list of peak indices

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return MultiGaussianFitComputer(x, y, peak_indices).fit()


def sinusoidal_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute sinusoidal fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return SinusoidalFitComputer(x, y).fit()


def voigt_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute Voigt fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return VoigtFitComputer(x, y).fit()


def cdf_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute Cumulative Distribution Function (CDF) fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return CDFFitComputer(x, y).fit()


def sigmoid_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute Sigmoid (Logistic) fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        A tuple containing the fitted y values and a dictionary of fit parameters.
    """
    return SigmoidFitComputer(x, y).fit()


FIT_TYPE_MAPPING = {
    "linear": LinearFitComputer,
    "polynomial": PolynomialFitComputer,
    "gaussian": GaussianFitComputer,
    "lorentzian": LorentzianFitComputer,
    "exponential": ExponentialFitComputer,
    "planckian": PlanckianFitComputer,
    "twohalfgaussian": TwoHalfGaussianFitComputer,
    "doubleexponential": DoubleExponentialFitComputer,
    "multilorentzian": MultiLorentzianFitComputer,
    "multigaussian": MultiGaussianFitComputer,
    "sinusoidal": SinusoidalFitComputer,
    "voigt": VoigtFitComputer,
    "cdf": CDFFitComputer,
    "sigmoid": SigmoidFitComputer,
}


def validate_fit_params(fit_params: Mapping[str, FitParamValue]) -> None:
    """Validate a fit-parameter dictionary before numerical evaluation.

    Args:
        fit_params: Fit parameters and metadata.

    Raises:
        ValueError: If the fit type, schema or parameter names are invalid.
        pulse.LegacyPeakParameterizationError: If an unversioned area-based peak
         dictionary is detected.
    """
    fit_type = fit_params.get("fit_type")
    if not isinstance(fit_type, str) or fit_type not in FIT_TYPE_MAPPING:
        raise ValueError(f"Unsupported fit type: {fit_type!r}")

    if fit_type in PEAK_FIT_TYPES:
        version = fit_params.get("fit_params_version")
        parameterization = fit_params.get("peak_parameterization")
        legacy_keys = [name for name in fit_params if _is_legacy_amplitude_key(name)]
        if version is None and legacy_keys:
            raise pulse.LegacyPeakParameterizationError(
                "Unversioned area-based peak fit parameters cannot be evaluated. "
                "Convert them explicitly with convert_legacy_peak_fit_params()."
            )
        if version != FIT_PARAMS_VERSION:
            raise ValueError(
                f"Unsupported fit_params_version for {fit_type}: {version!r}"
            )
        if parameterization != PEAK_PARAMETERIZATION:
            raise ValueError(
                f"Unsupported peak parameterization for {fit_type}: "
                f"{parameterization!r}"
            )
        if legacy_keys:
            raise ValueError(f"Legacy peak parameter keys in version 2: {legacy_keys}")

    residual_rms = fit_params.get("residual_rms")
    if residual_rms is not None and not _is_finite_real(residual_rms):
        raise ValueError("Residual RMS must be a finite real number")
    interactive = fit_params.get("interactive")
    if interactive is not None and not isinstance(interactive, bool):
        raise ValueError("Interactive fit marker must be a boolean")

    values = {
        name: value
        for name, value in fit_params.items()
        if name not in FIT_PARAMS_METADATA_KEYS
    }
    fit_class = FIT_TYPE_MAPPING[fit_type]
    expected_names = (
        fit_class.PARAMS_NAMES
        if fit_class.PARAMS_NAMES
        else fit_class.infer_param_names_from_kwargs(values)
    )
    missing = set(expected_names).difference(values)
    unexpected = set(values).difference(expected_names)
    if missing or unexpected:
        raise ValueError(
            f"Invalid parameters for {fit_type}: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    non_numeric = [name for name, value in values.items() if not _is_finite_real(value)]
    if non_numeric:
        raise ValueError(f"Non-numeric fit parameters: {sorted(non_numeric)}")
    if fit_type in PEAK_FIT_TYPES:
        invalid_widths = [
            name
            for name, value in values.items()
            if (name == "sigma" or name.startswith("sigma_")) and value <= 0
        ]
        if invalid_widths:
            raise ValueError(f"Non-positive fit widths: {sorted(invalid_widths)}")


def evaluate_fit(x: np.ndarray, **fit_params) -> np.ndarray:
    """Evaluate fit function with given parameters at x values.

    Args:
        x: X values to evaluate at
        **fit_params: Fit parameters (any of the ``*Params`` dataclasses)

    Returns:
        Y values computed from the fit function
    """
    validate_fit_params(fit_params)
    params = {
        name: value
        for name, value in fit_params.items()
        if name not in FIT_PARAMS_METADATA_KEYS
    }
    fcclass: Type[FitComputer] = FIT_TYPE_MAPPING[fit_params["fit_type"]]
    return fcclass.evaluate(x, **params)
