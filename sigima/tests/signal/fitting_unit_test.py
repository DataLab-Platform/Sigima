# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Curve fitting unit tests
========================

This module contains comprehensive tests for the curve fitting functions
in sigima.tools.signal.fitting, validating mathematical accuracy and robustness.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import scipy.special

import sigima.objects
import sigima.proc.signal
from sigima.tests import guiutils
from sigima.tests.data import get_test_signal
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result, check_scalar_result
from sigima.tools.signal import fitting, peakdetection, pulse

EXPECTED_FIT_PARAMS = {
    "gaussian_fit": {
        "amp": 151.5516963005346,
        "sigma": 10.093908516282582,
        "x0": 49.98522207721181,
        "y0": 0.14038830988167578,
        "fit_type": "gaussian",
    },
    "exponential_fit": {
        "a": 23299.374597935774,
        "b": -1.012051879085819,
        "y0": 0.3018450161545937,
        "fit_type": "exponential",
    },
    "twohalfgaussian_fit": {
        "amp_left": 2.989346344212517,
        "amp_right": 2.508788078396881,
        "sigma_left": 0.9821153800588559,
        "sigma_right": 4.737040821453857,
        "x0": 0.9751190925078642,
        "y0_left": 1.9970402083155143,
        "y0_right": 2.4917164605006117,
        "fit_type": "twohalfgaussian",
    },
    "piecewiseexponential_fit": {
        "x_center": 4.985324084088387,
        "a_left": 0.9784183389713168,
        "b_left": 1.0050512118447683,
        "a_right": 22480.004610557487,
        "b_right": -1.00498734825442,
        "y0": 0.05215861106687306,
        "fit_type": "doubleexponential",
    },
}


def __check_tools_proc_interface(
    toolsfunc: Callable[..., np.ndarray],
    procfunc: Callable[[sigima.objects.SignalObj], sigima.objects.SignalObj],
    x: np.ndarray,
    y: np.ndarray,
):
    """Helper to check interface between `sigima.tools` and `sigima.proc`."""
    fitted_y, params = toolsfunc(x, y)
    src = sigima.objects.create_signal("Test data", x, y)
    dst = procfunc(src)
    check_array_result(
        f"{toolsfunc.__name__}-proc interface", dst.y, fitted_y, rtol=1e-10
    )
    guiutils.view_curves_if_gui([src, dst], title=f"Test {toolsfunc.__name__}")

    # Also try to fit real experimental data if available
    try:
        experiment_signal = get_test_signal(f"{toolsfunc.__name__}.txt")
        fitted_signal = procfunc(experiment_signal)
        guiutils.view_curves_if_gui(
            [experiment_signal, fitted_signal], title=f"Test {toolsfunc.__name__}"
        )
        fit_params = fitted_signal.metadata["fit_params"]
        exp_params = EXPECTED_FIT_PARAMS.get(toolsfunc.__name__)
        if exp_params is None:
            for key, value in fit_params.items():
                if isinstance(value, np.floating):
                    fit_params[key] = float(value)
            raise ValueError(f"Unable to find expected params for: {repr(fit_params)}")
        for key, exp_value in exp_params.items():
            assert key in fit_params, f"Missing fit parameter: {key}"
            act_value = fit_params[key]
            if isinstance(exp_value, (int, float, np.floating)):
                # Increased absolute tolerance (atol=1e-6) to handle minor
                # floating-point precision differences observed with newer dependencies
                check_scalar_result(
                    f"Parameter {key}", act_value, exp_value, rtol=1e-5, atol=1e-6
                )
            else:
                assert act_value == exp_value, (
                    f"Parameter {key} differs: {act_value} != {exp_value}"
                )
    except FileNotFoundError:
        pass

    return fitted_y, params


@pytest.mark.validation
def test_signal_linear_fit() -> None:
    """Linear fitting validation test."""
    execenv.print("Testing linear fitting with perfect synthetic data...")

    # Generate perfect linear data
    x = np.linspace(0, 10, 100)
    a_true, b_true = 2.5, 1.3
    y = a_true * x + b_true

    fitted_y, params = fitting.linear_fit(x, y)
    check_scalar_result("Linear fit slope", params["a"], a_true, rtol=1e-10)
    check_scalar_result("Linear fit intercept", params["b"], b_true, rtol=1e-10)
    check_array_result("Linear fit y-values", fitted_y, y, rtol=1e-10)

    execenv.print("Testing linear fitting with noisy synthetic data...")

    # Set random seed for reproducible tests
    np.random.seed(42)

    x = np.linspace(0, 10, 100)
    a_true, b_true = 2.5, 1.3
    y_clean = a_true * x + b_true
    noise = np.random.normal(0, 0.1, len(x))
    y = y_clean + noise

    fitted_y, params = __check_tools_proc_interface(
        fitting.linear_fit, sigima.proc.signal.linear_fit, x, y
    )
    # With noise, we expect reasonable accuracy
    assert np.abs(params["a"] - a_true) < 0.05, "Slope should be accurate within 5%"
    assert np.abs(params["b"] - b_true) < 0.1, "Intercept should be accurate within 0.1"


@pytest.mark.validation
def test_polynomial_fit() -> None:
    """Polynomial fitting validation test."""
    execenv.print("Testing polynomial fitting with perfect synthetic data...")

    # Generate perfect quadratic data
    x = np.linspace(-5, 5, 100)
    a_true, b_true, c_true = 1.0, -2.0, 1.0
    y = a_true * x**2 + b_true * x + c_true

    fitted_y, params = fitting.polynomial_fit(x, y, degree=2)
    check_scalar_result("Polynomial fit a", params["a"], a_true, rtol=1e-10)
    check_scalar_result("Polynomial fit b", params["b"], b_true, rtol=1e-10)
    check_scalar_result("Polynomial fit c", params["c"], c_true, rtol=1e-10)
    check_array_result("Polynomial fit y-values", fitted_y, y, rtol=1e-10)

    execenv.print("Testing polynomial fitting with noisy synthetic data...")

    # Set random seed for reproducible tests
    np.random.seed(123)

    x = np.linspace(-5, 5, 100)
    a_true, b_true, c_true = 1.0, -2.0, 1.0
    y_clean = a_true * x**2 + b_true * x + c_true
    noise = np.random.normal(0, 2.0, len(x))
    y = y_clean + noise

    # Test tools interface
    fitted_y_tools, _params_tools = fitting.polynomial_fit(x, y, degree=2)

    # Test proc interface (needs PolynomialFitParam)
    src = sigima.objects.create_signal("Test data", x, y)
    poly_param = sigima.proc.signal.PolynomialFitParam()
    poly_param.degree = 2
    dst = sigima.proc.signal.polynomial_fit(src, poly_param)
    fitted_y = dst.y
    params = dst.metadata["fit_params"]

    # Check that both interfaces give similar results
    check_array_result(
        "polynomial_fit-proc interface", fitted_y, fitted_y_tools, rtol=1e-10
    )
    guiutils.view_curves_if_gui([src, dst], title="Test polynomial_fit")
    # With noise, we expect reasonable accuracy
    assert np.abs(params["a"] - a_true) < 0.1, "Coefficient a should be accurate"
    assert np.abs(params["b"] - b_true) < 0.2, "Coefficient b should be accurate"
    assert np.abs(params["c"] - c_true) < 0.5, "Coefficient c should be accurate"


@pytest.mark.validation
def test_signal_gaussian_fit() -> None:
    """Gaussian fitting validation test."""
    execenv.print("Testing Gaussian fitting with perfect synthetic data...")

    x = np.linspace(-5, 5, 200)
    peak_amplitude_true, sigma_true, x0_true, y0_true = 3.0, 1.5, 0.5, 0.2
    # Generate data using the peak amplitude form
    y = peak_amplitude_true * np.exp(-0.5 * ((x - x0_true) / sigma_true) ** 2) + y0_true

    _fitted_y, params = fitting.gaussian_fit(x, y)

    # Convert fitted amp to peak amplitude for comparison
    fitted_peak_amplitude = pulse.GaussianModel.amplitude(
        params["amp"], params["sigma"]
    )
    check_scalar_result(
        "Gaussian peak amplitude", fitted_peak_amplitude, peak_amplitude_true, rtol=1e-3
    )
    check_scalar_result("Gaussian sigma", params["sigma"], sigma_true, rtol=1e-3)
    check_scalar_result("Gaussian center", params["x0"], x0_true, rtol=1e-3)
    check_scalar_result("Gaussian offset", params["y0"], y0_true, rtol=1e-3)

    execenv.print("Testing Gaussian fitting with noisy synthetic data...")

    # Set random seed for reproducible tests
    np.random.seed(123)

    x = np.linspace(-5, 5, 200)
    amp_true, sigma_true, x0_true, y0_true = 3.0, 1.5, 0.5, 0.2
    y_clean = amp_true * np.exp(-0.5 * ((x - x0_true) / sigma_true) ** 2) + y0_true
    noise = np.random.normal(0, 0.05, len(x))
    y = y_clean + noise

    _fitted_y, params = __check_tools_proc_interface(
        fitting.gaussian_fit, sigima.proc.signal.gaussian_fit, x, y
    )
    # With noise, expect reasonable accuracy
    assert np.abs(params["x0"] - x0_true) < 0.2, "Gaussian center should be accurate"
    assert np.abs(params["y0"] - y0_true) < 0.2, "Gaussian offset should be accurate"


@pytest.mark.validation
def test_signal_lorentzian_fit() -> None:
    """Lorentzian fitting validation test."""
    execenv.print("Testing Lorentzian fitting with perfect synthetic data...")

    x = np.linspace(-10, 10, 200)
    peak_amplitude_true, sigma_true, x0_true, y0_true = 4.0, 1.5, -0.5, 0.2

    # Lorentzian function using peak amplitude: peak_amp / (1 + ((x - x0) / sigma)^2)
    y = peak_amplitude_true / (1 + ((x - x0_true) / sigma_true) ** 2) + y0_true
    y += np.random.normal(0, 0.1, len(x))

    _fitted_y, params = __check_tools_proc_interface(
        fitting.lorentzian_fit, sigima.proc.signal.lorentzian_fit, x, y
    )

    # Convert fitted amp to peak amplitude for comparison
    fitted_peak_amplitude = pulse.LorentzianModel.amplitude(
        params["amp"], params["sigma"]
    )
    check_scalar_result(
        "Lorentzian peak amplitude",
        fitted_peak_amplitude,
        peak_amplitude_true,
        rtol=1e-2,
    )
    assert np.abs(params["sigma"] - sigma_true) / sigma_true < 0.1, (
        "Sigma should be accurate"
    )
    assert np.abs(params["x0"] - x0_true) < 0.1, "Center should be accurate"
    assert np.abs(params["y0"] - y0_true) < 0.1, "Offset should be accurate"


@pytest.mark.validation
def test_signal_voigt_fit() -> None:
    """Voigt fitting validation test."""
    execenv.print("Testing Voigt fitting with synthetic data...")

    # Generate synthetic Voigt-like data (approximate using Gaussian for simplicity)
    x = np.linspace(-10, 10, 200)
    amplitude_true = 2.0
    sigma_true = 1.5
    x0_true = 2.0
    y0_true = 0.5

    # Use Gaussian as approximation for test data
    y = amplitude_true * np.exp(-0.5 * ((x - x0_true) / sigma_true) ** 2) + y0_true
    y += np.random.normal(0, 0.02, x.size)  # Add small amount of noise

    fitted_y, params = __check_tools_proc_interface(
        fitting.voigt_fit, sigima.proc.signal.voigt_fit, x, y
    )

    # Check that fitted curve is reasonable
    assert isinstance(fitted_y, np.ndarray), "Fitted y should be numpy array"
    assert fitted_y.shape == y.shape, "Fitted y should have same shape as input"

    # Check parameter structure
    assert "amp" in params, "Should have amplitude parameter"
    assert "sigma" in params, "Should have sigma parameter"
    assert "x0" in params, "Should have x0 parameter"
    assert "y0" in params, "Should have y0 parameter"

    # Parameters should be reasonable (within factor of 5 of true values)
    assert 0.1 * amplitude_true < params["amp"] < 5 * amplitude_true
    assert 0.1 * sigma_true < params["sigma"] < 5 * sigma_true
    assert x0_true - 2 * sigma_true < params["x0"] < x0_true + 2 * sigma_true


@pytest.mark.validation
def test_signal_exponential_fit() -> None:
    """Exponential decay fitting validation test."""
    execenv.print("Testing exponential decay fitting...")

    x = np.linspace(0, 5, 100)
    a_true, b_true, y0_true = 10.0, -2.0, 1.0
    y = a_true * np.exp(b_true * x) + y0_true
    y += np.random.normal(0, 0.2, len(x))  # Add some noise

    _fitted_y, params = __check_tools_proc_interface(
        fitting.exponential_fit, sigima.proc.signal.exponential_fit, x, y
    )
    # Check parameter accuracy
    assert np.abs(params["a"] - a_true) / a_true < 0.1, "Amplitude should be accurate"
    assert np.abs(params["b"] - b_true) / abs(b_true) < 0.1, (
        "Decay rate should be accurate"
    )
    assert np.abs(params["y0"] - y0_true) < 0.2, "Offset should be accurate"

    execenv.print("Testing exponential growth fitting...")

    x = np.linspace(0, 3, 50)
    a_true, b_true, y0_true = 2.0, 1.5, 0.5
    y = a_true * np.exp(b_true * x) + y0_true

    _fitted_y, params = fitting.exponential_fit(x, y)
    # Growth fitting is more challenging due to rapid increase
    assert np.abs(params["b"] - b_true) / b_true < 0.2, (
        "Growth rate should be reasonably accurate"
    )


@pytest.mark.validation
def test_signal_piecewiseexponential_fit() -> None:
    """Piecewise exponential (raise-decay) fitting validation test."""
    execenv.print("Testing piecewise exponential (raise-decay) fitting...")

    # Set random seed for reproducible test results
    np.random.seed(42)

    x = np.linspace(0, 10, 100)
    amp1_true, amp2_true = 8.0, 3.0
    tau1_true, tau2_true = 0.5, 3.0
    y0_true = 1.0

    y = (
        amp1_true * np.exp(-x / tau1_true)
        + amp2_true * np.exp(-x / tau2_true)
        + y0_true
        + np.random.normal(0, 0.2, len(x))
    )

    fitted_y, _params = __check_tools_proc_interface(
        fitting.piecewiseexponential_fit,
        sigima.proc.signal.piecewiseexponential_fit,
        x,
        y,
    )

    # Verify the fit quality is good (R² > 0.95)
    r2 = 1 - np.sum((y - fitted_y) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert r2 > 0.95, f"Fit quality R² = {r2:.3f} should be > 0.95"

    # Verify the overall model makes sense: check that fitted curve is reasonable
    rms_error = np.sqrt(np.mean((y - fitted_y) ** 2))
    assert rms_error < 1.0, f"RMS error = {rms_error:.3f} should be < 1.0"


@pytest.mark.validation
def test_signal_planckian_fit() -> None:
    """Planckian fitting validation test.

    This test uses realistic parameters that produce a characteristic
    blackbody radiation curve with a prominent peak, as would be
    observed in thermal radiation measurements.
    """
    execenv.print("Testing Planckian fitting with synthetic blackbody data...")

    # Wavelength range in micrometers (typical range for thermal radiation)
    x = np.linspace(0.5, 5.0, 150)

    # True parameters for realistic blackbody curve with more prominent peak
    amp_true = 50.0  # Higher amplitude for more prominent peak
    x0_true = 1.0  # Peak wavelength (Wien's displacement)
    sigma_true = 0.8  # Temperature factor (sharper curve)
    y0_true = 0.5  # Baseline offset

    # Generate true Planckian data using the actual model
    y = fitting.PlanckianFitComputer.evaluate(
        x, amp=amp_true, x0=x0_true, sigma=sigma_true, y0=y0_true
    )

    # Add realistic noise (proportional to signal strength)
    noise_level = 0.02 * (np.max(y) - np.min(y))
    y += np.random.normal(0, noise_level, x.size)

    fitted_y, params = __check_tools_proc_interface(
        fitting.planckian_fit, sigima.proc.signal.planckian_fit, x, y
    )

    # Check that fitted curve is reasonable
    assert isinstance(fitted_y, np.ndarray), "Fitted y should be numpy array"
    assert fitted_y.shape == y.shape, "Fitted y should have same shape as input"

    # Check parameter structure
    assert "amp" in params, "Should have amp parameter"
    assert "x0" in params, "Should have x0 parameter"
    assert "sigma" in params, "Should have sigma parameter"
    assert "y0" in params, "Should have y0 parameter"

    # Check that the fit produces a realistic peak location
    # The peak should be close to the Wien displacement law prediction
    peak_x_fitted = x[np.argmax(fitted_y)]
    peak_x_true = x[np.argmax(y)]

    execenv.print(f"True peak at: {peak_x_true:.3f} μm")
    execenv.print(f"Fitted peak at: {peak_x_fitted:.3f} μm")
    execenv.print(
        f"Fitted parameters: amp={params['amp']:.2f}, "
        f"x0={params['x0']:.2f}, σ={params['sigma']:.2f}"
    )
    # Peak location should be reasonably accurate (within 20% of wavelength range)
    # Planckian fitting can be challenging due to the complex function form
    wavelength_tolerance = 0.20 * (x[-1] - x[0])
    assert np.abs(peak_x_fitted - peak_x_true) < wavelength_tolerance, (
        f"Peak location accuracy: fitted={peak_x_fitted:.3f}, true={peak_x_true:.3f}"
    )

    # Check that the curve has a reasonable dynamic range
    # Use a more relaxed criterion since Planckian curves can be quite flat
    dynamic_range = np.max(fitted_y) - np.min(fitted_y)
    mean_level = np.mean(fitted_y)
    assert dynamic_range > 0.05 * mean_level, (
        "Fitted curve should have reasonable dynamic range"
    )

    # Check that there is a discernible peak (not completely flat)
    peak_value = np.max(fitted_y)
    edge_values = [fitted_y[0], fitted_y[-1]]
    max_edge = np.max(edge_values)
    assert peak_value > max_edge, "Peak should be higher than edge values"

    # Parameters should be in reasonable ranges for physical systems
    assert params["amp"] > 0, "Amplitude should be positive"
    assert params["x0"] > 0, "Peak wavelength should be positive"
    assert params["sigma"] > 0, "Sigma should be positive"


@pytest.mark.validation
def test_signal_cdf_fit() -> None:
    """CDF fitting validation test."""
    execenv.print("Testing CDF fitting with synthetic data...")

    # Generate synthetic CDF data
    x = np.linspace(-5, 5, 200)
    amplitude_true = 2.0
    mu_true = 0.5
    sigma_true = 1.0
    baseline_true = 1.0

    # Generate CDF data using error function
    erf = scipy.special.erf  # pylint: disable=no-member
    y = amplitude_true * erf((x - mu_true) / (sigma_true * np.sqrt(2))) + baseline_true
    y += np.random.normal(0, 0.05, x.size)  # Add small amount of noise

    fitted_y, params = __check_tools_proc_interface(
        fitting.cdf_fit, sigima.proc.signal.cdf_fit, x, y
    )

    # Check that fitted curve is reasonable
    assert isinstance(fitted_y, np.ndarray), "Fitted y should be numpy array"
    assert fitted_y.shape == y.shape, "Fitted y should have same shape as input"

    # Check parameter structure
    assert "amplitude" in params, "Should have amplitude parameter"
    assert "mu" in params, "Should have mu parameter"
    assert "sigma" in params, "Should have sigma parameter"
    assert "baseline" in params, "Should have baseline parameter"

    # Parameters should be reasonable (within factor of 3 of true values)
    assert 0.3 * amplitude_true < params["amplitude"] < 3 * amplitude_true
    assert 0.3 * sigma_true < params["sigma"] < 3 * sigma_true
    assert mu_true - 2 * sigma_true < params["mu"] < mu_true + 2 * sigma_true


@pytest.mark.validation
def test_signal_sigmoid_fit() -> None:
    """Sigmoid fitting validation test."""
    execenv.print("Testing Sigmoid fitting with synthetic data...")

    # Generate synthetic sigmoid data
    x = np.linspace(-5, 5, 200)
    amplitude_true = 3.0
    k_true = 1.0
    x0_true = 1.0
    offset_true = 0.5

    # Generate sigmoid data
    y = offset_true + amplitude_true / (1.0 + np.exp(-k_true * (x - x0_true)))
    y += np.random.normal(0, 0.05, x.size)  # Add small amount of noise

    fitted_y, params = __check_tools_proc_interface(
        fitting.sigmoid_fit, sigima.proc.signal.sigmoid_fit, x, y
    )

    # Check that fitted curve is reasonable
    assert isinstance(fitted_y, np.ndarray), "Fitted y should be numpy array"
    assert fitted_y.shape == y.shape, "Fitted y should have same shape as input"

    # Check parameter structure
    assert "amplitude" in params, "Should have amplitude parameter"
    assert "k" in params, "Should have k parameter"
    assert "x0" in params, "Should have x0 parameter"
    assert "offset" in params, "Should have offset parameter"

    # Parameters should be reasonable (within factor of 3 of true values)
    assert 0.3 * amplitude_true < params["amplitude"] < 3 * amplitude_true
    assert 0.3 * k_true < params["k"] < 3 * k_true
    assert x0_true - 2 < params["x0"] < x0_true + 2


@pytest.mark.validation
def test_signal_twohalfgaussian_fit() -> None:
    """Two half-Gaussian fitting validation test."""
    execenv.print("Testing two half-Gaussian fitting...")

    x = np.linspace(-5, 10, 200)
    (
        amp_true,
        x0_true,
        sigma_left_true,
        sigma_right_true,
        y0_left_true,
        y0_right_true,
    ) = (5.0, 2.0, 1.0, 2.5, 0.3, 0.5)

    # Create asymmetric Gaussian with separate baselines (enhanced test)
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < x0_true:
            y[i] = y0_left_true + amp_true * np.exp(
                -0.5 * ((xi - x0_true) / sigma_left_true) ** 2
            )
        else:
            y[i] = y0_right_true + amp_true * np.exp(
                -0.5 * ((xi - x0_true) / sigma_right_true) ** 2
            )

    # Add noise
    y += np.random.normal(0, 0.1, len(x))

    # Test the tools layer directly for now
    _fitted_y, params = __check_tools_proc_interface(
        fitting.twohalfgaussian_fit, sigima.proc.signal.twohalfgaussian_fit, x, y
    )

    # Check that center position is reasonable
    assert np.abs(params["x0"] - x0_true) < 0.5, "Center should be accurate"

    # Check that baseline offsets are reasonable
    assert np.abs(params["y0_left"] - y0_left_true) < 0.3, (
        "Left baseline should be accurate"
    )
    assert np.abs(params["y0_right"] - y0_right_true) < 0.3, (
        "Right baseline should be accurate"
    )

    # Check that the fitted parameters make sense
    assert params["sigma_left"] > 0, "Left sigma should be positive"
    assert params["sigma_right"] > 0, "Right sigma should be positive"
    assert "amp_left" in params, "Should have amp_left parameter"
    assert "amp_right" in params, "Should have amp_right parameter"
    assert params["amp_left"] > 0, "Left amplitude should be positive"
    assert params["amp_right"] > 0, "Right amplitude should be positive"


# This is not a validation test as there is no computation function for multi
# gaussian fitting in sigima.proc.signal
def test_multigaussian_single_peak() -> None:
    """Multi-Gaussian fitting validation test with single peak."""
    execenv.print("Testing multi-Gaussian fitting with single peak...")

    x = np.linspace(-10, 10, 200)
    amp_true, sigma_true, x0_true, y0_true = 4.0, 1.5, -0.5, 0.2

    # Single Gaussian
    y = amp_true * np.exp(-0.5 * ((x - x0_true) / sigma_true) ** 2) + y0_true
    y += np.random.normal(0, 0.02, len(x))

    # Find peak indices for the function
    peaks = peakdetection.peak_indices(
        y, thres=0.2
    )  # Use higher threshold for better detection
    execenv.print(f"Detected peaks at indices: {peaks}")

    # If no peaks detected, use manual peak
    if len(peaks) == 0:
        peak_idx = np.argmax(y)
        peaks = np.array([peak_idx])

    _y, params = fitting.multigaussian_fit(x, y, peak_indices=peaks.tolist())

    # Check results - expect at least one peak
    assert len(peaks) >= 1, "Should detect at least one peak"
    assert "amp_1" in params, "Should have amplitude for first peak"
    assert "sigma_1" in params, "Should have sigma for first peak"
    assert "x0_1" in params, "Should have x0 for first peak"
    assert "y0" in params, "Should have y0 baseline parameter"


# This is not a validation test as there is no computation function for multi
# gaussian fitting in sigima.proc.signal
def test_multigaussian_double_peak() -> None:
    """Multi-Gaussian fitting validation test with double peaks."""
    execenv.print("Testing multi-Gaussian fitting with double peaks...")

    x = np.linspace(-10, 10, 300)
    # Two well-separated peaks
    amp1, sigma1, x01 = 3.0, 1.0, -3.0
    amp2, sigma2, x02 = 2.0, 1.5, 4.0
    y0_true = 0.1

    y = (
        amp1 * np.exp(-0.5 * ((x - x01) / sigma1) ** 2)
        + amp2 * np.exp(-0.5 * ((x - x02) / sigma2) ** 2)
        + y0_true
    )
    y += np.random.normal(0, 0.02, len(x))

    # Find peak indices
    peaks = peakdetection.peak_indices(y, thres=0.3, min_dist=20)
    execenv.print(f"Detected peaks at indices: {peaks}")

    # If insufficient peaks detected, use manual peaks
    if len(peaks) < 2:
        peaks = np.array([np.argmax(y[:150]), 150 + np.argmax(y[150:])])

    try:
        yf, params = fitting.multigaussian_fit(x, y, peak_indices=peaks.tolist())
        guiutils.view_curves_if_gui([[x, y], [x, yf]], title="Test multigaussian_fit")

        # Check that we detected two peaks and got results
        assert len(peaks) >= 2, "Should detect at least two peaks"
        assert "amp_1" in params, "Should have amplitude for first peak"
        assert "amp_2" in params, "Should have amplitude for second peak"
        assert "sigma_1" in params, "Should have sigma for first peak"
        assert "sigma_2" in params, "Should have sigma for second peak"
        assert "x0_1" in params, "Should have x0 for first peak"
        assert "x0_2" in params, "Should have x0 for second peak"
        assert "y0" in params, "Should have y0 baseline parameter"
    except ValueError as e:
        if "infeasible" in str(e):
            execenv.print(
                "Multi-Gaussian fit failed due to optimization bounds "
                "(expected for complex fitting)"
            )
        else:
            raise


# This is not a validation test as there is no computation function for multi
# lorentzian fitting in sigima.proc.signal
def test_multilorentzian_single_peak() -> None:
    """Multi-Lorentzian fitting validation test with single peak."""
    execenv.print("Testing multi-Lorentzian fitting with single peak...")

    x = np.linspace(-10, 10, 200)
    amp_true, sigma_true, x0_true, y0_true = 4.0, 1.5, -0.5, 0.2

    # Single Lorentzian
    y = amp_true / (1 + ((x - x0_true) / sigma_true) ** 2) + y0_true
    y += np.random.normal(0, 0.02, len(x))

    # Find peak indices for the function
    peaks = peakdetection.peak_indices(
        y, thres=0.2
    )  # Use higher threshold for better detection
    execenv.print(f"Detected peaks at indices: {peaks}")

    # If no peaks detected, use manual peak
    if len(peaks) == 0:
        peak_idx = np.argmax(y)
        peaks = np.array([peak_idx])

    _y, params = fitting.multilorentzian_fit(x, y, peak_indices=peaks.tolist())

    # Check results - expect at least one peak
    assert len(peaks) >= 1, "Should detect at least one peak"
    assert "amp_1" in params, "Should have amplitude for first peak"
    assert "sigma_1" in params, "Should have sigma for first peak"
    assert "x0_1" in params, "Should have x0 for first peak"
    assert "y0" in params, "Should have y0 baseline parameter"


# This is not a validation test as there is no computation function for multi
# lorentzian fitting in sigima.proc.signal
def test_multilorentzian_double_peak() -> None:
    """Multi-Lorentzian fitting validation test with double peaks."""
    execenv.print("Testing multi-Lorentzian fitting with double peaks...")

    x = np.linspace(-10, 10, 300)
    # Two well-separated peaks
    amp1, sigma1, x01 = 3.0, 1.0, -3.0
    amp2, sigma2, x02 = 2.0, 1.5, 4.0
    y0_true = 0.1

    y = (
        amp1 / (1 + ((x - x01) / sigma1) ** 2)
        + amp2 / (1 + ((x - x02) / sigma2) ** 2)
        + y0_true
    )
    y += np.random.normal(0, 0.02, len(x))

    # Find peak indices
    peaks = peakdetection.peak_indices(y, thres=0.3, min_dist=20)
    execenv.print(f"Detected peaks at indices: {peaks}")

    # If insufficient peaks detected, use manual peaks
    if len(peaks) < 2:
        peaks = np.array([np.argmax(y[:150]), 150 + np.argmax(y[150:])])

    try:
        yf, params = fitting.multilorentzian_fit(x, y, peak_indices=peaks.tolist())
        guiutils.view_curves_if_gui([[x, y], [x, yf]], title="Test multilorentzian_fit")

        # Check that we detected two peaks and got results
        assert len(peaks) >= 2, "Should detect at least two peaks"
        assert "amp_1" in params, "Should have amplitude for first peak"
        assert "amp_2" in params, "Should have amplitude for second peak"
        assert "sigma_1" in params, "Should have sigma for first peak"
        assert "sigma_2" in params, "Should have sigma for second peak"
        assert "x0_1" in params, "Should have x0 for first peak"
        assert "x0_2" in params, "Should have x0 for second peak"
        assert "y0" in params, "Should have y0 baseline parameter"
    except ValueError as e:
        if "infeasible" in str(e):
            execenv.print(
                "Multi-Lorentzian fit failed due to optimization bounds "
                "(expected for complex fitting)"
            )
        else:
            raise


@pytest.mark.validation
def test_sinusoidal_fit() -> None:
    """Sinusoidal fitting validation test."""
    execenv.print("Testing sinusoidal fitting with synthetic data...")

    # Generate synthetic sinusoidal data
    x = np.linspace(0, 10, 200)
    amplitude_true = 2.0
    frequency_true = 1.5  # cycles per unit x
    phase_true = 0.5  # radians
    offset_true = 1.0

    # Generate sinusoidal data
    y = offset_true + amplitude_true * np.sin(
        2 * np.pi * frequency_true * x + phase_true
    )
    y += np.random.normal(0, 0.1, x.size)  # Add small amount of noise

    fitted_y, params = __check_tools_proc_interface(
        fitting.sinusoidal_fit, sigima.proc.signal.sinusoidal_fit, x, y
    )

    # Check that fitted curve is reasonable
    assert isinstance(fitted_y, np.ndarray), "Fitted y should be numpy array"
    assert fitted_y.shape == y.shape, "Fitted y should have same shape as input"

    # Check parameter structure
    assert "amplitude" in params, "Should have amplitude parameter"
    assert "frequency" in params, "Should have frequency parameter"
    assert "phase" in params, "Should have phase parameter"
    assert "offset" in params, "Should have offset parameter"

    # Parameters should be reasonable (within factor of 2 of true values)
    assert 0.5 * amplitude_true < params["amplitude"] < 2 * amplitude_true
    assert 0.5 * frequency_true < params["frequency"] < 2 * frequency_true
    assert -np.pi < params["phase"] < np.pi  # Phase should be within -π to π
    assert 0.5 * offset_true < params["offset"] < 2 * offset_true


def test_fitting_error_handling() -> None:
    """Test error handling in fitting functions."""
    execenv.print("Testing fitting error handling...")

    # Test with insufficient data points
    x_short = np.array([1, 2])
    y_short = np.array([1, 2])

    fitted_y, params = fitting.linear_fit(x_short, y_short)
    # Should either succeed (linear fit needs only 2 points) or fail gracefully
    assert isinstance(fitted_y, np.ndarray), "Result should be a numpy array"
    assert "a" in params and "b" in params, "Params should have a and b attributes"

    # Test with mismatched array sizes - this should raise an exception
    x_mismatch = np.array([1, 2, 3])
    y_mismatch = np.array([1, 2])

    try:
        fitted_y, params = fitting.linear_fit(x_mismatch, y_mismatch)
        # If no exception is raised, we just check that we got some result
        assert fitted_y is not None, (
            "Should get some result even with mismatched arrays"
        )
    except (TypeError, ValueError):
        # This is expected behavior - the function raises an exception
        pass


def test_fitting_functions_available() -> None:
    """Test that expected fitting functions are available."""
    execenv.print("Testing availability of fitting functions...")

    # Check that expected functions exist and are callable
    expected_functions = [
        "linear_fit",
        "gaussian_fit",
        "lorentzian_fit",
        "voigt_fit",
        "exponential_fit",
        "piecewiseexponential_fit",
        "planckian_fit",
        "cdf_fit",
        "sigmoid_fit",
        "twohalfgaussian_fit",
        "multilorentzian_fit",
        "sinusoidal_fit",
    ]

    for func_name in expected_functions:
        assert hasattr(fitting, func_name), f"Function {func_name} should exist"
        func = getattr(fitting, func_name)
        assert callable(func), f"Function {func_name} should be callable"


@pytest.mark.validation
def test_signal_evaluate_fit() -> None:
    """Test evaluate_fit as a computation function (2-to-1)."""
    execenv.print("Testing evaluate_fit computation function...")

    # Create a signal with linear data
    x1 = np.linspace(0, 10, 100)
    y1 = 3.0 * x1 + 1.0 + np.random.normal(0, 0.5, len(x1))
    src1 = sigima.objects.create_signal("Test data 1", x1, y1)

    # Perform a fit to get fit parameters
    fitted_signal = sigima.proc.signal.linear_fit(src1)
    assert "fit_params" in fitted_signal.metadata, "Fit should produce metadata"

    # Create a second signal with a different x-axis
    x2 = np.linspace(-5, 15, 50)
    y2 = np.zeros_like(x2)  # Data doesn't matter, only x-axis is used
    src2 = sigima.objects.create_signal("Test data 2", x2, y2)

    # Evaluate fit from src1 on x-axis of src2 (2-to-1 operation)
    result = sigima.proc.signal.evaluate_fit(fitted_signal, src2)

    # Verify the result
    assert len(result.x) == len(x2), "Result should have same length as src2"
    check_array_result("X-axis", result.x, x2, rtol=1e-10)

    # The y values should be the fit evaluated on x2
    fit_params = sigima.proc.signal.extract_fit_params(fitted_signal)
    expected_y = fitting.evaluate_fit(x2, **fit_params)
    check_array_result("Evaluated fit", result.y, expected_y, rtol=1e-10)

    # Check that fit parameters are preserved
    assert "fit_params" in result.metadata, "Result should contain fit_params"
    result_params = sigima.proc.signal.extract_fit_params(result)
    assert result_params["a"] == fit_params["a"], "Fitted a should match"
    assert result_params["b"] == fit_params["b"], "Fitted b should match"

    # Remove "fit_params" and verify that the extract_fit_params raises a ValueError:
    del result.metadata["fit_params"]
    with pytest.raises(ValueError, match="Signal does not contain fit parameters"):
        sigima.proc.signal.extract_fit_params(result)


def test_fitting_user_experience() -> None:
    """Test user experience aspects of fitting functions."""
    execenv.print("Testing user experience of fitting functions...")

    # Test that fitting functions return results in expected formats
    x = np.linspace(0, 10, 100)
    y = 3.0 * x + 1.0 + np.random.normal(0, 0.5, len(x))

    fitted_y, params = fitting.linear_fit(x, y)
    assert isinstance(fitted_y, np.ndarray), "Fitted y should be a numpy array"
    assert "a" in params and "b" in params, "Params should have a and b attributes"
    fitted_y2 = fitting.evaluate_fit(x, **params)
    check_array_result("Evaluate fit", fitted_y2, fitted_y, rtol=1e-10)

    # Test that metadata is correctly attached to SignalObj when using proc functions
    src = sigima.objects.create_signal("Test data", x, y)
    dst = sigima.proc.signal.linear_fit(src)
    assert "fit_params" in dst.metadata, "Metadata should contain fit_params"
    fit_params = sigima.proc.signal.extract_fit_params(dst)
    assert "a" in fit_params and "b" in fit_params, "fit_params should contain a and b"
    assert fit_params["a"] == params["a"], "Fitted a should match"
    assert fit_params["b"] == params["b"], "Fitted b should match"
    # Use the new 2-to-1 signature: evaluate fit from dst on x-axis of src
    dst2 = sigima.proc.signal.evaluate_fit(dst, src)
    check_array_result("Evaluate fit on SignalObj", dst2.y, dst.y, rtol=1e-10)


if __name__ == "__main__":
    guiutils.enable_gui()
    # test_signal_linear_fit()
    test_polynomial_fit()
    test_signal_gaussian_fit()
    test_signal_lorentzian_fit()
    test_signal_voigt_fit()
    test_signal_exponential_fit()
    test_signal_piecewiseexponential_fit()
    test_signal_planckian_fit()
    test_signal_cdf_fit()
    test_signal_sigmoid_fit()
    test_signal_twohalfgaussian_fit()
    test_multigaussian_single_peak()
    test_multigaussian_double_peak()
    test_multilorentzian_single_peak()
    test_multilorentzian_double_peak()
    test_sinusoidal_fit()
    test_fitting_error_handling()
    test_fitting_functions_available()
    test_signal_evaluate_fit()
    test_fitting_user_experience()
    execenv.print("All fitting unit tests passed!")
