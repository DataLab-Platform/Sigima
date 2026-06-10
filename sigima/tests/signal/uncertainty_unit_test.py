# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for uncertainty propagation in signal operations

Features from signal processing functions that include uncertainty propagation.
This test covers the mathematical functions (sqrt, log10, exp, clip, absolute,
real, imag) and arithmetic operations (addition, average, product, difference,
constant operations).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.signal
import sigima.tests.data
from sigima.tests.helpers import check_array_result


def __create_signal_with_uncertainty() -> sigima.objects.SignalObj:
    """Create a signal with uncertainty data for testing."""
    obj = sigima.tests.data.create_periodic_signal(sigima.objects.SignalTypes.COSINE)
    obj.dy = 0.1 * np.abs(obj.y) + 0.01  # 10% relative + 0.01 absolute
    return obj


def __create_signal_without_uncertainty() -> sigima.objects.SignalObj:
    """Create a signal without uncertainty data for testing."""
    obj = sigima.tests.data.create_periodic_signal(sigima.objects.SignalTypes.COSINE)
    obj.dy = None
    return obj


def __verify_uncertainty_propagation(
    func: Callable[[sigima.objects.SignalObj], sigima.objects.SignalObj],
    param: sigima.params.GaussianParam
    | sigima.params.MovingAverageParam
    | sigima.params.MovingMedianParam
    | None = None,
) -> None:
    """Test uncertainty propagation for a given signal processing function."""
    src = __create_signal_with_uncertainty()
    if param is None:
        result = func(src)
    else:
        result = func(src, param)

    # Check that uncertainties are propagated (should be unchanged for filters)
    assert result.dy is not None, "Uncertainty should be propagated"
    check_array_result("Uncertainty propagation", result.dy, src.dy)

    # Test without uncertainty
    src = __create_signal_without_uncertainty()
    result_no_unc = func(src)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_exp_uncertainty_propagation() -> None:
    """Test uncertainty propagation for exponential function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    result = sigima.proc.signal.exp(src)

    # Check result values
    check_array_result("Exponential values", result.y, np.exp(src.y))

    # Check uncertainty propagation: σ(eʸ) = eʸ * σ(y) = dst.y * σ(y)
    expected_dy = np.abs(result.y) * src.dy
    check_array_result("Exponential uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.exp(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_sqrt_uncertainty_propagation() -> None:
    """Test uncertainty propagation for square root function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()

    # Suppress warnings for sqrt of negative values in test data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = sigima.proc.signal.sqrt(src)

    # Check result values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        check_array_result("Square root values", result.y, np.sqrt(src.y))

    # Check uncertainty propagation: σ(√y) = σ(y) / (2√y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_dy = src.dy / (2 * np.sqrt(src.y))
        expected_dy[np.isinf(expected_dy) | np.isnan(expected_dy)] = np.nan

    check_array_result("Square root uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result_no_unc = sigima.proc.signal.sqrt(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_power_uncertainty_propagation() -> None:
    """Test uncertainty propagation for power function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    p = 3.0
    param = sigima.params.PowerParam.create(power=p)
    result = sigima.proc.signal.power(src, param)

    # Check result values
    check_array_result("Power values", result.y, src.y**p)

    # Check uncertainty propagation: σ(yᵖ) = |p * y^(p-1)| * σ(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_dy = np.abs(p * src.y ** (p - 1)) * src.dy
        expected_dy[np.isinf(expected_dy) | np.isnan(expected_dy)] = np.nan

    check_array_result("Power uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.power(src_no_unc, param)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_log10_uncertainty_propagation() -> None:
    """Test uncertainty propagation for log10 function."""
    # Test with uncertainty - use positive values to avoid log domain issues
    src = __create_signal_with_uncertainty()
    src.y = np.abs(src.y) + 1.0  # Ensure positive values
    result = sigima.proc.signal.log10(src)

    # Check result values
    check_array_result("Log10 values", result.y, np.log10(src.y))

    # Check uncertainty propagation: σ(log₁₀(y)) = σ(y) / (y * ln(10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_dy = src.dy / (src.y * np.log(10))
        expected_dy[np.isinf(expected_dy) | np.isnan(expected_dy)] = np.nan

    check_array_result("Log10 uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    src_no_unc.y = np.abs(src_no_unc.y) + 1.0  # Ensure positive values
    result_no_unc = sigima.proc.signal.log10(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_clip_uncertainty_propagation() -> None:
    """Test uncertainty propagation for clipping function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()

    # Test clipping with both limits
    param = sigima.params.ClipParam.create(lower=-0.5, upper=0.5)
    result = sigima.proc.signal.clip(src, param)

    # Check result values
    expected_y = np.clip(src.y, param.lower, param.upper)
    check_array_result("Clip values", result.y, expected_y)

    # Check uncertainty propagation: σ(clip(y)) = σ(y) where not clipped,
    # 0 where clipped
    expected_dy = src.dy.copy()
    expected_dy[src.y <= param.lower] = 0
    expected_dy[src.y >= param.upper] = 0
    check_array_result("Clip uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.clip(src_no_unc, param)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_normalize_uncertainty_propagation() -> None:
    """Test uncertainty propagation for normalization function."""
    # Test different normalization methods
    for method in sigima.enums.NormalizationMethod:
        # Test with uncertainty
        src = __create_signal_with_uncertainty()
        param = sigima.params.NormalizeParam()
        param.method = method
        result = sigima.proc.signal.normalize(src, param)

        # Check that uncertainties are propagated appropriately for each method
        assert result.dy is not None, f"Uncertainty should be propagated for {method}"

        # For most methods, uncertainty should be non-zero where input uncertainty
        # exists
        if method != sigima.enums.NormalizationMethod.AMPLITUDE:
            # For non-amplitude methods, check that uncertainties exist and are finite
            assert np.any(np.isfinite(result.dy)), (
                f"Some uncertainties should be finite for {method}"
            )
        else:
            # For amplitude normalization, check the specific uncertainty formula
            # σ(amplitude_norm(y)) = σ(y) / (max(y) - min(y))
            denom = np.max(src.y) - np.min(src.y)
            if denom != 0:
                expected_dy = src.dy / denom
                check_array_result(
                    "Amplitude normalize uncertainty propagation",
                    result.dy,
                    expected_dy,
                )

        # Test without uncertainty
        src_no_unc = __create_signal_without_uncertainty()
        result_no_unc = sigima.proc.signal.normalize(src_no_unc, param)
        assert result_no_unc.dy is None, (
            f"Uncertainty should be None when input has no uncertainty for {method}"
        )


def test_derivative_uncertainty_propagation() -> None:
    """Test uncertainty propagation for derivative function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    result = sigima.proc.signal.derivative(src)

    # Check that uncertainties are propagated
    assert result.dy is not None, "Uncertainty should be propagated"

    # For numerical derivatives, the uncertainty depends on the finite difference scheme
    # numpy.gradient uses central differences for interior points:
    # dy/dx ≈ (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    # So σ(dy/dx) ≈ sqrt(σ(y[i+1])² + σ(y[i-1])²) / (x[i+1] - x[i-1])

    # For a more general test, we verify that:
    # 1. Uncertainties exist and are finite where input uncertainties exist
    # 2. The uncertainty scaling is reasonable compared to input uncertainties
    assert np.any(np.isfinite(result.dy)), "Some uncertainties should be finite"

    # The derivative uncertainty should generally be larger than input uncertainty
    # due to the division by dx (assuming dx < 1 for typical signals)
    x = src.x
    typical_dx = np.median(np.diff(x))
    if typical_dx > 0:
        # Expected rough scaling: derivative uncertainty ~ input uncertainty / dx
        expected_scale = 1.0 / typical_dx
        # Allow for significant variation due to the complexity of gradient calculation
        max_ratio = np.nanmax(result.dy / src.dy)
        assert max_ratio > 0.1 * expected_scale, (
            f"Derivative uncertainty scaling seems too small: {max_ratio} vs "
            f"expected ~{expected_scale}"
        )

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.derivative(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_integral_uncertainty_propagation() -> None:
    """Test uncertainty propagation for integral function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    result = sigima.proc.signal.integral(src)

    # Check that uncertainties are propagated
    assert result.dy is not None, "Uncertainty should be propagated"

    # For cumulative integration, uncertainties should accumulate
    # The first point should have zero uncertainty (initial value)
    assert result.dy[0] == 0.0, "Initial integral value should have zero uncertainty"

    # For cumulative trapezoidal integration, uncertainties should generally increase
    # as we accumulate more measurements
    # Check that uncertainties are non-decreasing (allowing for numerical precision)
    diff_dy = np.diff(result.dy)
    # Allow small negative differences due to numerical precision
    assert np.all(diff_dy >= -1e-10), "Integral uncertainties should generally increase"

    # The integral uncertainties should be finite and non-negative
    assert np.all(np.isfinite(result.dy)), "All integral uncertainties should be finite"
    assert np.all(result.dy >= 0), "All integral uncertainties should be non-negative"

    # Integral uncertainty should be positive (assuming we have some integration range)
    max_integral_uncertainty = np.max(result.dy)
    assert max_integral_uncertainty > 0, (
        "Maximum integral uncertainty should be positive"
    )

    # Validate the uncertainty propagation formula implementation
    # The integral function uses: σ(∫y dx) ≈ √(Σ(σ(y_i) * Δx_i)²) for trapezoidal rule
    # Specifically: dy_squared = src.dy[:-1]² + src.dy[1:]²
    # and dst.dy[1:] = √(cumsum(dy_squared * dx² / 4))
    dx = np.diff(src.x)
    dy_squared = src.dy[:-1] ** 2 + src.dy[1:] ** 2
    expected_uncertainties = np.zeros_like(result.dy)
    expected_uncertainties[0] = 0.0  # Initial value
    expected_uncertainties[1:] = np.sqrt(np.cumsum(dy_squared * (dx**2) / 4))

    # The computed uncertainties should match the expected formula
    check_array_result(
        "Integral uncertainty propagation", result.dy, expected_uncertainties
    )

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.integral(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_calibration_uncertainty_propagation() -> None:
    """Test uncertainty propagation for calibration function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    # Add X uncertainty for testing X-axis calibration
    src.dx = 0.05 * np.abs(src.x) + 0.001  # 5% relative + 0.001 absolute

    # Test Y-axis calibration: y' = a*y + b
    a, b = 2.5, 0.3
    param = sigima.params.XYCalibrateParam.create(axis="y", a=a, b=b)
    result = sigima.proc.signal.calibration(src, param)

    # Check uncertainty propagation: σ(a*y + b) = |a| * σ(y)
    expected_dy = np.abs(a) * src.dy
    check_array_result(
        "Y-axis calibration uncertainty propagation", result.dy, expected_dy
    )

    # Test X-axis calibration: x' = a*x + b
    param = sigima.params.XYCalibrateParam.create(axis="x", a=a, b=b)
    result = sigima.proc.signal.calibration(src, param)

    # Check X uncertainty propagation: σ(a*x + b) = |a| * σ(x)
    if src.dx is not None:
        expected_dx = np.abs(a) * src.dx
        check_array_result(
            "X-axis calibration uncertainty propagation", result.dx, expected_dx
        )

    # Y uncertainties should remain the same for x-axis calibration
    check_array_result("X-axis calibration dy unchanged", result.dy, src.dy)

    # Test with negative scaling factor to check absolute value
    a_neg = -1.5
    param_neg = sigima.params.XYCalibrateParam.create(axis="y", a=a_neg, b=b)
    result_neg = sigima.proc.signal.calibration(src, param_neg)
    expected_dy_neg = np.abs(a_neg) * src.dy
    check_array_result(
        "Y-axis calibration negative scaling uncertainty",
        result_neg.dy,
        expected_dy_neg,
    )

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.calibration(src_no_unc, param)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_absolute_uncertainty_propagation() -> None:
    """Test uncertainty propagation for absolute value function."""
    __verify_uncertainty_propagation(sigima.proc.signal.absolute)


def test_real_uncertainty_propagation() -> None:
    """Test uncertainty propagation for real part function."""
    __verify_uncertainty_propagation(sigima.proc.signal.real)


def test_imag_uncertainty_propagation() -> None:
    """Test uncertainty propagation for imaginary part function."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    result = sigima.proc.signal.imag(src)

    # Check result values
    check_array_result("Imaginary part values", result.y, np.imag(src.y))

    # Check uncertainty propagation: uncertainties unchanged for imaginary part
    check_array_result("Imaginary part uncertainty propagation", result.dy, src.dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = sigima.proc.signal.imag(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_is_uncertainty_data_available() -> None:
    """Test the is_uncertainty_data_available helper function."""
    # Single signal with uncertainty
    src_with = __create_signal_with_uncertainty()
    assert sigima.proc.signal.is_uncertainty_data_available(src_with), (
        "Should return True for signal with uncertainty"
    )

    # Single signal without uncertainty
    src_without = __create_signal_without_uncertainty()
    assert not sigima.proc.signal.is_uncertainty_data_available(src_without), (
        "Should return False for signal without uncertainty"
    )

    # List of signals - all with uncertainty
    src_list_with = [__create_signal_with_uncertainty() for _ in range(3)]
    assert sigima.proc.signal.is_uncertainty_data_available(src_list_with), (
        "Should return True for list where all signals have uncertainty"
    )

    # List of signals - mixed
    src_list_mixed = [
        __create_signal_with_uncertainty(),
        __create_signal_without_uncertainty(),
    ]
    assert not sigima.proc.signal.is_uncertainty_data_available(src_list_mixed), (
        "Should return False for list with mixed uncertainty availability"
    )

    # List of signals - all without uncertainty
    src_list_without = [__create_signal_without_uncertainty() for _ in range(3)]
    assert not sigima.proc.signal.is_uncertainty_data_available(src_list_without), (
        "Should return False for list where no signals have uncertainty"
    )


def test_inverse_uncertainty_propagation() -> None:
    """Test uncertainty propagation for signal inversion."""
    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    # Ensure values are not too close to zero to avoid division issues
    src.y = src.y + 2.0  # Shift away from zero
    result = sigima.proc.signal.inverse(src)

    # Check result values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_y = 1.0 / src.y
        expected_y[np.isinf(expected_y)] = np.nan
    check_array_result("Inverse values", result.y, expected_y)

    # Check uncertainty propagation: σ(1/y) = |1/y| * σ(y) / |y| = σ(y) / |y|²
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_dy = np.abs(result.y) * src.dy / np.abs(src.y)
        expected_dy[np.isinf(expected_dy)] = np.nan
    check_array_result("Inverse uncertainty propagation", result.dy, expected_dy)

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    src_no_unc.y = src_no_unc.y + 2.0  # Shift away from zero
    result_no_unc = sigima.proc.signal.inverse(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_gaussian_filter_uncertainty_propagation() -> None:
    """Test uncertainty propagation for Gaussian filter."""
    param = sigima.params.GaussianParam.create(sigma=2.0)
    __verify_uncertainty_propagation(sigima.proc.signal.gaussian_filter, param)


def test_wiener_filter_uncertainty_propagation() -> None:
    """Test uncertainty propagation for Wiener filter."""
    __verify_uncertainty_propagation(sigima.proc.signal.wiener)


def test_moving_average_uncertainty_propagation() -> None:
    """Test uncertainty propagation for moving average filter."""
    param = sigima.params.MovingAverageParam.create(n=5)
    __verify_uncertainty_propagation(sigima.proc.signal.moving_average, param)


def test_moving_median_uncertainty_propagation() -> None:
    """Test uncertainty propagation for moving median filter."""
    param = sigima.params.MovingMedianParam.create(n=5)
    __verify_uncertainty_propagation(sigima.proc.signal.moving_median, param)


def test_wrap1to1func_basic_behavior() -> None:
    """Test basic Wrap1to1Func behavior with uncertainty propagation.

    Wrap1to1Func should preserve uncertainty unchanged for any wrapped function.
    """
    # Test with a mathematical function (np.sin)
    # Note: This tests the wrapper behavior, not the direct sin function
    compute_sin_wrapped = sigima.proc.signal.Wrap1to1Func(np.sin)

    # Test with uncertainty
    src = __create_signal_with_uncertainty()
    result = compute_sin_wrapped(src)

    # Check result values
    check_array_result("Wrapped sin values", result.y, np.sin(src.y))

    # Check uncertainty propagation (should be unchanged when using Wrap1to1Func)
    check_array_result("Wrapped sin uncertainty propagation", result.dy, src.dy)

    # Test with a custom function
    def custom_multiply(y):
        """Custom function: multiply by 3."""
        return 3 * y

    compute_custom = sigima.proc.signal.Wrap1to1Func(custom_multiply)

    result_custom = compute_custom(src)

    # Check result values
    check_array_result("Custom multiply values", result_custom.y, 3 * src.y)

    # Check uncertainty propagation (should be unchanged for any wrapped function)
    check_array_result(
        "Custom multiply uncertainty propagation", result_custom.dy, src.dy
    )

    # Test without uncertainty
    src_no_unc = __create_signal_without_uncertainty()
    result_no_unc = compute_custom(src_no_unc)
    assert result_no_unc.dy is None, (
        "Uncertainty should be None when input has no uncertainty"
    )


def test_wrap1to1func_with_args_kwargs() -> None:
    """Test Wrap1to1Func with additional args and kwargs."""

    def power_func(y, power=2):
        """Raise y to a power."""
        return y**power

    # Test with power=3 using kwargs
    compute_power = sigima.proc.signal.Wrap1to1Func(power_func, power=3)

    src = __create_signal_with_uncertainty()
    result = compute_power(src)

    # Check result values
    check_array_result("Power 3 values", result.y, src.y**3)

    # Check uncertainty propagation (should be unchanged when using Wrap1to1Func)
    # Note: This is different from the mathematical uncertainty propagation
    # which would be σ(y³) = 3 * y² * σ(y)
    check_array_result("Power 3 uncertainty propagation", result.dy, src.dy)

    # Test with positional arguments
    def multiply_add(y, multiplier, addend):
        """Custom function: y * multiplier + addend."""
        return y * multiplier + addend

    compute_multiply_add = sigima.proc.signal.Wrap1to1Func(multiply_add, 2, addend=5)

    result_multiply_add = compute_multiply_add(src)

    # Check result values
    expected_y = src.y * 2 + 5
    check_array_result("Multiply-add values", result_multiply_add.y, expected_y)

    # Check uncertainty propagation (preserved unchanged)
    check_array_result(
        "Multiply-add uncertainty propagation", result_multiply_add.dy, src.dy
    )


def test_extract_roi_uncertainty_propagation() -> None:
    """extract_roi must preserve dx/dy uncertainty data from the source signal.

    Regression test: ``extract_roi`` used to drop ``dx`` and ``dy``.
    """
    x = np.linspace(0.0, 10.0, 11)
    y = np.sin(x)
    obj = sigima.objects.create_signal("s", x, y)
    obj.dx = 0.01 * np.ones_like(x)
    obj.dy = 0.1 * np.ones_like(y)

    p = sigima.objects.ROI1DParam.create(xmin=2.0, xmax=7.0)
    dst = sigima.proc.signal.extract_roi(obj, p)

    assert dst.dx is not None, "extract_roi dropped dx uncertainty data"
    assert dst.dy is not None, "extract_roi dropped dy uncertainty data"
    assert dst.dx.shape == dst.x.shape
    assert dst.dy.shape == dst.y.shape
    check_array_result("extract_roi dx", dst.dx, np.full_like(dst.x, 0.01))
    check_array_result("extract_roi dy", dst.dy, np.full_like(dst.y, 0.1))


def test_extract_rois_uncertainty_propagation() -> None:
    """extract_rois must preserve dx/dy uncertainty data from the source signal."""
    x = np.linspace(0.0, 10.0, 21)
    y = np.cos(x)
    obj = sigima.objects.create_signal("s", x, y)
    obj.dx = 0.005 * np.ones_like(x)
    obj.dy = 0.05 * np.ones_like(y)

    p1 = sigima.objects.ROI1DParam.create(xmin=1.0, xmax=3.0)
    p2 = sigima.objects.ROI1DParam.create(xmin=6.0, xmax=8.0)
    dst = sigima.proc.signal.extract_rois(obj, [p1, p2])

    assert dst.dx is not None, "extract_rois dropped dx uncertainty data"
    assert dst.dy is not None, "extract_rois dropped dy uncertainty data"
    assert dst.dx.shape == dst.x.shape
    assert dst.dy.shape == dst.y.shape
    check_array_result("extract_rois dx", dst.dx, np.full_like(dst.x, 0.005))
    check_array_result("extract_rois dy", dst.dy, np.full_like(dst.y, 0.05))


if __name__ == "__main__":
    test_sqrt_uncertainty_propagation()
    test_log10_uncertainty_propagation()
    test_exp_uncertainty_propagation()
    test_clip_uncertainty_propagation()
    test_derivative_uncertainty_propagation()
    test_integral_uncertainty_propagation()
    test_absolute_uncertainty_propagation()
    test_real_uncertainty_propagation()
    test_imag_uncertainty_propagation()
    test_is_uncertainty_data_available()
    test_inverse_uncertainty_propagation()
    test_gaussian_filter_uncertainty_propagation()
    test_wiener_filter_uncertainty_propagation()
    test_moving_average_uncertainty_propagation()
    test_moving_median_uncertainty_propagation()
    test_wrap1to1func_basic_behavior()
    test_wrap1to1func_with_args_kwargs()
    test_extract_roi_uncertainty_propagation()
    test_extract_rois_uncertainty_propagation()
