# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for signal operations
--------------------------------

Features from the "Operations" menu are covered by this test.
The "Operations" menu contains basic operations on signals, such as
addition, multiplication, division, and more.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
import sigima.tests.data
from sigima.enums import (
    AngleUnit,
    MathOperator,
    NormalizationMethod,
    SignalsToImageOrientation,
)
from sigima.objects.signal import SignalObj
from sigima.proc.base import AngleUnitParam
from sigima.proc.signal import complex_from_magnitude_phase, complex_from_real_imag
from sigima.tests.helpers import check_array_result
from sigima.tools.coordinates import polar_to_complex


def __create_two_signals() -> tuple[sigima.objects.SignalObj, sigima.objects.SignalObj]:
    """Create two signals for testing."""
    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINE, freq=50.0, size=100
    )
    s1.dy = 0.05 * np.ones_like(s1.y)
    s2 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.SINE, freq=25.0, size=100
    )
    s2.dy = 0.8 * np.ones_like(s2.y)
    return s1, s2


def __create_n_signals(n: int = 100) -> list[sigima.objects.SignalObj]:
    """Create a list of `n` different signals for testing."""
    signals = []
    for i in range(n):
        s = sigima.tests.data.create_periodic_signal(
            sigima.objects.SignalTypes.COSINE,
            freq=50.0 + i,
            size=100,
            a=(i + 1) * 0.1,
        )
        s.dy = 0.5 * np.ones_like(s.y)
        signals.append(s)
    return signals


def __create_one_signal_and_constant() -> tuple[
    sigima.objects.SignalObj, sigima.params.ConstantParam
]:
    """Create one signal and a constant for testing."""
    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINE, freq=50.0, size=100
    )
    s1.dy = 0.5 * np.ones_like(s1.y)
    param = sigima.params.ConstantParam.create(value=-np.pi)
    return s1, param


@pytest.mark.validation
def test_signal_addition() -> None:
    """Signal addition test."""
    slist = __create_n_signals()
    n = len(slist)
    s1 = sigima.proc.signal.addition(slist)
    exp_y = np.zeros_like(s1.y)
    for s in slist:
        exp_y += s.y
    check_array_result(f"Addition of {n} signals", s1.y, exp_y)
    expected_dy = np.sqrt(sum(sig.dy**2 for sig in slist))
    check_array_result("Addition error propagation", s1.dy, expected_dy)


@pytest.mark.validation
def test_signal_average() -> None:
    """Signal average test."""
    slist = __create_n_signals()
    n = len(slist)
    s1 = sigima.proc.signal.average(slist)
    exp_y = np.zeros_like(s1.y)
    for s in slist:
        exp_y += s.y
    exp_y /= n
    check_array_result(f"Average of {n} signals", s1.y, exp_y)
    expected_dy = np.sqrt(sum(s.dy**2 for s in slist)) / n
    check_array_result("Average error propagation", s1.dy, expected_dy)


@pytest.mark.validation
def test_signal_standard_deviation() -> None:
    """Signal standard deviation test."""
    slist = __create_n_signals()
    n = len(slist)
    s1 = sigima.proc.signal.standard_deviation(slist)
    exp = np.zeros_like(s1.y)
    average = np.mean([s.y for s in slist], axis=0)
    for s in slist:
        exp += (s.y - average) ** 2
    exp = np.sqrt(exp / n)
    check_array_result(f"Standard Deviation of {n} signals", s1.y, exp)
    # Add uncertainty to source signals:
    for sig in slist:
        sig.dy = np.abs(0.1 * sig.y) + 0.1
    s2 = sigima.proc.signal.standard_deviation(slist)
    expected_dy = exp / np.sqrt(2 * (n - 1))
    check_array_result("Standard Deviation error propagation", s2.dy, expected_dy)


@pytest.mark.validation
def test_signal_product() -> None:
    """Signal multiplication test."""
    slist = __create_n_signals()
    n = len(slist)
    s1 = sigima.proc.signal.product(slist)
    exp_y = np.ones_like(s1.y)
    for s in slist:
        exp_y *= s.y
    check_array_result(f"Product of {n} signals", s1.y, exp_y)
    expected_dy = np.abs(exp_y) * np.sqrt(sum((s.dy / s.y) ** 2 for s in slist))
    check_array_result("Product error propagation", s1.dy, expected_dy)


@pytest.mark.validation
def test_signal_difference() -> None:
    """Signal difference test."""
    s1, s2 = __create_two_signals()
    s3 = sigima.proc.signal.difference(s1, s2)
    check_array_result("Signal difference", s3.y, s1.y - s2.y)
    expected_dy = np.sqrt(s1.dy**2 + s2.dy**2)
    check_array_result("Difference error propagation", s3.dy, expected_dy)
    x = np.array([0.0, 1.0, 2.0, 3.0])
    a = sigima.objects.create_signal("a", x, np.array([1.0, 4.0, 9.0, 16.0]))
    b = sigima.objects.create_signal("b", x, np.array([1.0, 1.0, 2.0, 3.0]))
    c = sigima.proc.signal.difference(a, b)
    assert np.array_equal(c.x, x)
    assert np.array_equal(c.y, [0.0, 3.0, 7.0, 13.0])


@pytest.mark.validation
def test_signal_quadratic_difference() -> None:
    """Signal quadratic difference validation test."""
    s1, s2 = __create_two_signals()
    s3 = sigima.proc.signal.quadratic_difference(s1, s2)
    check_array_result("Signal quadratic difference", s3.y, (s1.y - s2.y) / np.sqrt(2))


@pytest.mark.validation
def test_signal_division() -> None:
    """Signal division test."""
    s1, s2 = __create_two_signals()
    s3 = sigima.proc.signal.division(s1, s2)
    check_array_result("Signal division", s3.y, s1.y / s2.y)
    expected_dy = np.abs(s1.y / s2.y) * np.sqrt(
        (s1.dy / s1.y) ** 2 + (s2.dy / s2.y) ** 2
    )
    check_array_result("Division error propagation", s3.dy, expected_dy)


@pytest.mark.validation
def test_signal_addition_constant() -> None:
    """Signal addition with constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima.proc.signal.addition_constant(s1, param)
    check_array_result("Signal addition with constant", s2.y, s1.y + param.value)
    # Error should be unchanged after addition of a constant
    check_array_result("Addition constant error propagation", s2.dy, s1.dy)


@pytest.mark.validation
def test_signal_product_constant() -> None:
    """Signal multiplication by constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima.proc.signal.product_constant(s1, param)
    check_array_result("Signal multiplication by constant", s2.y, s1.y * param.value)
    # Error is scaled by the absolute value of the constant
    assert param.value is not None
    expected_dy = np.abs(param.value) * s1.dy
    check_array_result("Product constant error propagation", s2.dy, expected_dy)


@pytest.mark.validation
def test_signal_difference_constant() -> None:
    """Signal difference with constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima.proc.signal.difference_constant(s1, param)
    check_array_result("Signal difference with constant", s2.y, s1.y - param.value)
    # Error is unchanged after subtraction of a constant
    check_array_result("Difference constant error propagation", s2.dy, s1.dy)


@pytest.mark.validation
def test_signal_division_constant() -> None:
    """Signal division by constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima.proc.signal.division_constant(s1, param)
    check_array_result("Signal division by constant", s2.y, s1.y / param.value)
    assert param.value is not None
    expected_dy = s1.dy / np.abs(param.value)
    check_array_result("Division constant error propagation", s2.dy, expected_dy)


@pytest.mark.validation
def test_signal_inverse() -> None:
    """Signal inversion validation test."""
    s1 = __create_two_signals()[0]
    inv_signal = sigima.proc.signal.inverse(s1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        exp_y = 1.0 / s1.y
        exp_y[np.isinf(exp_y)] = np.nan
        expected_dy = np.abs(exp_y) * s1.dy / np.abs(s1.y)
        expected_dy[np.isinf(expected_dy)] = np.nan
    check_array_result("Signal inverse", inv_signal.y, exp_y)
    check_array_result("Inverse error propagation", inv_signal.dy, expected_dy)


@pytest.mark.validation
def test_signal_absolute() -> None:
    """Absolute value validation test."""
    s1 = __create_two_signals()[0]
    abs_signal = sigima.proc.signal.absolute(s1)
    check_array_result("Absolute value", abs_signal.y, np.abs(s1.y))


@pytest.mark.validation
def test_signal_real() -> None:
    """Real part validation test."""
    s1 = __create_two_signals()[0]
    re_signal = sigima.proc.signal.real(s1)
    check_array_result("Real part", re_signal.y, np.real(s1.y))


@pytest.mark.validation
def test_signal_imag() -> None:
    """Imaginary part validation test."""
    s1 = __create_two_signals()[0]
    im_signal = sigima.proc.signal.imag(s1)
    check_array_result("Imaginary part", im_signal.y, np.imag(s1.y))


@pytest.mark.validation
def test_signal_complex_from_real_imag() -> None:
    """Test :py:func:`sigima.proc.signal.complex_from_real_imag`."""
    x = np.linspace(0.0, 1.0, 5)
    real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    imag = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # Create SignalObj instances for real and imaginary parts
    s_real = SignalObj("real")
    s_real.set_xydata(x, real)
    s_imag = SignalObj("imag")
    s_imag.set_xydata(x, imag)
    # Create complex signal from real and imaginary parts
    result = complex_from_real_imag(s_real, s_imag)
    check_array_result(
        "complex_from_real_imag",
        result.y,
        real + 1j * imag,
    )


@pytest.mark.validation
def test_signal_phase() -> None:
    """Phase angle validation test."""
    # Create a base signal and make it complex for testing
    base_signal = __create_two_signals()[0]
    y_complex = base_signal.y + 1j * base_signal.y[::-1]
    complex_signal = sigima.objects.create_signal("complex", base_signal.x, y_complex)

    # Test phase extraction in radians without unwrapping
    param_rad = sigima.params.PhaseParam.create(unit=AngleUnit.RADIAN, unwrap=False)
    result_rad = sigima.proc.signal.phase(complex_signal, param_rad)
    check_array_result("Phase in radians", result_rad.y, np.angle(y_complex))

    # Test phase extraction in degrees without unwrapping
    param_deg = sigima.params.PhaseParam.create(unit=AngleUnit.DEGREE, unwrap=False)
    result_deg = sigima.proc.signal.phase(complex_signal, param_deg)
    check_array_result("Phase in degrees", result_deg.y, np.angle(y_complex, deg=True))

    # Test phase extraction in radians with unwrapping
    param_rad_unwrap = sigima.params.PhaseParam.create(
        unit=AngleUnit.RADIAN, unwrap=True
    )
    result_rad_unwrap = sigima.proc.signal.phase(complex_signal, param_rad_unwrap)
    check_array_result(
        "Phase in radians with unwrapping",
        result_rad_unwrap.y,
        np.unwrap(np.angle(y_complex)),
    )

    # Test phase extraction in degrees with unwrapping
    param_deg_unwrap = sigima.params.PhaseParam.create(
        unit=AngleUnit.DEGREE, unwrap=True
    )
    result_deg_unwrap = sigima.proc.signal.phase(complex_signal, param_deg_unwrap)
    check_array_result(
        "Phase in degrees with unwrapping",
        result_deg_unwrap.y,
        np.unwrap(np.angle(y_complex, deg=True), period=360.0),
    )


MAGNITUDE_PHASE_TEST_CASES = [
    (np.array([0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2.0, 0.0]), AngleUnit.RADIAN),
    (np.array([0.0, 90.0, 180.0, 270.0, 0.0]), AngleUnit.DEGREE),
]


@pytest.mark.parametrize("phase, unit", MAGNITUDE_PHASE_TEST_CASES)
@pytest.mark.validation
def test_signal_complex_from_magnitude_phase(
    phase: np.ndarray, unit: AngleUnit
) -> None:
    """Test :py:func:`sigima.proc.signal.complex_from_magnitude_phase`.

    Args:
        phase (np.ndarray): Angles in radians or degrees.
        unit (AngleUnit): Unit of the angles, either radian or degree.
    """
    x = np.linspace(0.0, 1.0, 5)
    magnitude = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    # Create signal instances for magnitude and phase
    s_mag = SignalObj("magnitude")
    s_mag.set_xydata(x, magnitude)
    s_phase = SignalObj("phase")
    s_phase.set_xydata(x, phase)
    # Create complex signal from magnitude and phase
    p = AngleUnitParam.create(unit=unit)
    result = complex_from_magnitude_phase(s_mag, s_phase, p)
    unit_str = "rad" if unit == AngleUnit.RADIAN else "°"
    check_array_result(
        f"complex_from_magnitude_phase_{unit_str}",
        result.y,
        polar_to_complex(magnitude, phase, unit=unit_str),
    )


def __test_all_complex_from_magnitude_phase() -> None:
    """Test all combinations of magnitude and phase."""
    for phase, unit in MAGNITUDE_PHASE_TEST_CASES:
        test_signal_complex_from_magnitude_phase(phase, unit)


@pytest.mark.validation
def test_signal_astype() -> None:
    """Data type conversion validation test."""
    s1 = __create_two_signals()[0]
    for dtype_str in sigima.objects.SignalObj.get_valid_dtypenames():
        p = sigima.params.DataTypeSParam.create(dtype_str=dtype_str)
        astype_signal = sigima.proc.signal.astype(s1, p)
        assert astype_signal.y.dtype == np.dtype(dtype_str)


@pytest.mark.validation
def test_signal_exp() -> None:
    """Exponential validation test."""
    s1 = __create_two_signals()[0]
    exp_signal = sigima.proc.signal.exp(s1)
    check_array_result("Exponential", exp_signal.y, np.exp(s1.y))


@pytest.mark.validation
def test_signal_log10() -> None:
    """Logarithm base 10 validation test."""
    s1 = __create_two_signals()[0]
    log10_signal = sigima.proc.signal.log10(sigima.proc.signal.exp(s1))
    check_array_result("Logarithm base 10", log10_signal.y, np.log10(np.exp(s1.y)))


@pytest.mark.validation
def test_signal_sqrt() -> None:
    """Square root validation test."""
    s1 = sigima.tests.data.get_test_signal("paracetamol.txt")
    sqrt_signal = sigima.proc.signal.sqrt(s1)
    check_array_result("Square root", sqrt_signal.y, np.sqrt(s1.y))


@pytest.mark.validation
def test_signal_power() -> None:
    """Power validation test."""
    s1 = sigima.tests.data.get_test_signal("paracetamol.txt")
    p = sigima.params.PowerParam.create(power=2.0)
    power_signal = sigima.proc.signal.power(s1, p)
    check_array_result("Power", power_signal.y, s1.y**p.power)


@pytest.mark.validation
def test_signal_arithmetic() -> None:
    """Arithmetic operations validation test."""
    s1, s2 = __create_two_signals()
    p = sigima.params.ArithmeticParam.create()
    for operator in MathOperator:
        p.operator = operator
        for factor in (0.0, 1.0, 2.0):
            p.factor = factor
            for constant in (0.0, 1.0, 2.0):
                p.constant = constant
                s3 = sigima.proc.signal.arithmetic(s1, s2, p)
                if operator == MathOperator.ADD:
                    exp = s1.y + s2.y
                elif operator == MathOperator.MULTIPLY:
                    exp = s1.y * s2.y
                elif operator == MathOperator.SUBTRACT:
                    exp = s1.y - s2.y
                elif operator == MathOperator.DIVIDE:
                    exp = s1.y / s2.y
                else:
                    raise ValueError(f"Unknown operator {operator}")
                exp = exp * factor + constant
                check_array_result(f"Arithmetic [{p.get_operation()}]", s3.y, exp)


@pytest.mark.validation
def test_signal_signals_to_image() -> None:
    """Signals to image conversion test."""
    # Create test signals
    slist = __create_n_signals(n=5)
    n = len(slist)
    size = len(slist[0].y)

    # Test without normalization, as rows
    p = sigima.params.SignalsToImageParam()
    p.orientation = SignalsToImageOrientation.ROWS
    p.normalize = False
    img = sigima.proc.signal.signals_to_image(slist, p)
    assert img.data.shape == (n, size), (
        f"Expected shape ({n}, {size}), got {img.data.shape}"
    )
    for i, sig in enumerate(slist):
        title = f"Signals to image (rows) - signal {i}"
        check_array_result(title, img.data[i], sig.y)

    # Test without normalization, as columns
    p.orientation = SignalsToImageOrientation.COLUMNS
    img = sigima.proc.signal.signals_to_image(slist, p)
    assert img.data.shape == (size, n), (
        f"Expected shape ({size}, {n}), got {img.data.shape}"
    )
    for i, sig in enumerate(slist):
        title = f"Signals to image (columns) - signal {i}"
        check_array_result(title, img.data[:, i], sig.y)

    # Test with normalization
    p.normalize = True
    p.normalize_method = NormalizationMethod.MAXIMUM
    p.orientation = SignalsToImageOrientation.ROWS
    img = sigima.proc.signal.signals_to_image(slist, p)
    for i, sig in enumerate(slist):
        expected = sig.y / np.max(np.abs(sig.y))
        title = f"Signals to image (normalized rows) - signal {i}"
        check_array_result(title, img.data[i], expected)


if __name__ == "__main__":
    test_signal_addition()
    test_signal_average()
    test_signal_product()
    test_signal_difference()
    test_signal_quadratic_difference()
    test_signal_division()
    test_signal_addition_constant()
    test_signal_product_constant()
    test_signal_difference_constant()
    test_signal_division_constant()
    test_signal_inverse()
    test_signal_absolute()
    test_signal_real()
    test_signal_imag()
    test_signal_complex_from_real_imag()
    test_signal_phase()
    __test_all_complex_from_magnitude_phase()
    test_signal_astype()
    test_signal_exp()
    test_signal_log10()
    test_signal_sqrt()
    test_signal_power()
    test_signal_arithmetic()
    test_signal_signals_to_image()
