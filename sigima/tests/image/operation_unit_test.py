# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for image operations."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import math
import warnings
from typing import Generator

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.enums import AngleUnit, MathOperator
from sigima.objects.image import ImageObj
from sigima.proc.base import AngleUnitParam
from sigima.proc.image import complex_from_magnitude_phase, complex_from_real_imag
from sigima.tests import guiutils
from sigima.tests.data import (
    create_noisy_gaussian_image,
    iterate_noisy_image_couples,
    iterate_noisy_images,
)
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result
from sigima.tools.coordinates import polar_to_complex


def __create_n_images(n: int = 100) -> list[sigima.objects.ImageObj]:
    """Create a list of N different images for testing."""
    images = []
    for i in range(n):
        param = sigima.objects.NewImageParam.create(
            dtype=sigima.objects.ImageDatatypes.FLOAT32,
            height=128,
            width=128,
        )
        img = create_noisy_gaussian_image(param, level=(i + 1) * 0.1)
        images.append(img)
    return images


@pytest.mark.validation
def test_image_addition() -> None:
    """Image addition test."""
    execenv.print("*** Testing image addition:")
    for ima1, ima2 in iterate_noisy_image_couples(size=128):
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  {dtype1} += {dtype2}: ", end="")
        exp = ima1.data.astype(float) + ima2.data.astype(float)
        ima3 = sigima.proc.image.addition([ima1, ima2])
        check_array_result("Image addition", ima3.data, exp)
    imalist = __create_n_images()
    n = len(imalist)
    ima3 = sigima.proc.image.addition(imalist)
    res = ima3.data
    exp = np.zeros_like(ima3.data)
    for ima in imalist:
        exp += ima.data
    check_array_result(f"  Addition of {n} images", res, exp)


@pytest.mark.validation
def test_image_average() -> None:
    """Image average test."""
    execenv.print("*** Testing image average:")
    for ima1, ima2 in iterate_noisy_image_couples(size=128):
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  µ({dtype1},{dtype2}): ", end="")
        exp = (ima1.data.astype(float) + ima2.data.astype(float)) / 2.0
        ima3 = sigima.proc.image.average([ima1, ima2])
        check_array_result("Image average", ima3.data, exp)
    imalist = __create_n_images()
    n = len(imalist)
    ima3 = sigima.proc.image.average(imalist)
    res = ima3.data
    exp = np.zeros_like(ima3.data)
    for ima in imalist:
        exp += ima.data
    exp /= n
    check_array_result(f"  Average of {n} images", res, exp)


@pytest.mark.validation
def test_image_standard_deviation() -> None:
    """Image standard deviation test."""
    imalist = __create_n_images()
    n = len(imalist)
    s1 = sigima.proc.image.standard_deviation(imalist)
    assert s1.data is not None
    exp = np.zeros_like(s1.data)
    average = np.mean([ima.data for ima in imalist if ima.data is not None], axis=0)
    for ima in imalist:
        exp += (ima.data - average) ** 2
    exp = np.sqrt(exp / n)
    check_array_result(f"Standard Deviation of {n} images", s1.data, exp)


@pytest.mark.validation
def test_image_difference() -> None:
    """Image difference test."""
    execenv.print("*** Testing image difference:")
    for ima1, ima2 in iterate_noisy_image_couples(size=128):
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  {dtype1} -= {dtype2}: ", end="")
        exp = ima1.data.astype(float) - ima2.data.astype(float)
        ima3 = sigima.proc.image.difference(ima1, ima2)
        check_array_result("Image difference", ima3.data, exp)


@pytest.mark.validation
def test_image_quadratic_difference() -> None:
    """Quadratic difference test."""
    execenv.print("*** Testing quadratic difference:")
    for ima1, ima2 in iterate_noisy_image_couples(size=128):
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  ({dtype1} - {dtype2})/√2: ", end="")
        exp = (ima1.data.astype(float) - ima2.data.astype(float)) / np.sqrt(2)
        ima3 = sigima.proc.image.quadratic_difference(ima1, ima2)
        check_array_result("Image quadratic difference", ima3.data, exp)


@pytest.mark.validation
def test_image_product() -> None:
    """Image multiplication test."""
    execenv.print("*** Testing image multiplication:")
    for ima1, ima2 in iterate_noisy_image_couples(size=128):
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  {dtype1} *= {dtype2}: ", end="")
        exp = ima1.data.astype(float) * ima2.data.astype(float)
        ima3 = sigima.proc.image.product([ima1, ima2])
        check_array_result("Image multiplication", ima3.data, exp)
    imalist = __create_n_images()
    n = len(imalist)
    ima3 = sigima.proc.image.product(imalist)
    res = ima3.data
    exp = np.ones_like(ima3.data)
    for ima in imalist:
        exp *= ima.data
    check_array_result(f"  Multiplication of {n} images", res, exp)


@pytest.mark.validation
def test_image_division() -> None:
    """Image division test."""
    execenv.print("*** Testing image division:")
    for ima1, ima2 in iterate_noisy_image_couples(size=128):
        ima2.data = np.ones_like(ima2.data)
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  {dtype1} /= {dtype2}: ", end="")
        exp = ima1.data.astype(float) / ima2.data.astype(float)
        ima3 = sigima.proc.image.division(ima1, ima2)
        if not np.allclose(ima3.data, exp):
            guiutils.view_images_side_by_side_if_gui(
                [ima1.data, ima2.data, ima3.data], ["ima1", "ima2", "ima3"]
            )
        check_array_result("Image division", ima3.data, exp)


def __constparam(value: float) -> sigima.params.ConstantParam:
    """Create a constant parameter."""
    return sigima.params.ConstantParam.create(value=value)


def __iterate_image_with_constant() -> Generator[
    tuple[sigima.objects.ImageObj, sigima.params.ConstantParam], None, None
]:
    """Iterate over all possible image and constant couples for testing."""
    size = 128
    for dtype in sigima.objects.ImageDatatypes:
        param = sigima.objects.NewImageParam.create(
            dtype=dtype, height=size, width=size
        )
        ima = create_noisy_gaussian_image(param, level=0.0)
        for value in (-1.0, 3.14, 5.0):
            p = __constparam(value)
            yield ima, p


def __expected_constant(value: float, dtype: np.dtype) -> float:
    """Expected constant: truncated toward zero for integer images (no wrap)."""
    if np.issubdtype(dtype, np.integer):
        return float(math.trunc(value))
    return float(np.array(value, dtype=dtype))


@pytest.mark.validation
def test_image_addition_constant() -> None:
    """Image addition with constant test."""
    execenv.print("*** Testing image addition with constant:")
    for ima1, p in __iterate_image_with_constant():
        dtype1 = ima1.data.dtype
        execenv.print(f"  {dtype1} += constant ({p.value}): ", end="")
        exp = ima1.data.astype(float) + __expected_constant(p.value, dtype1)
        ima2 = sigima.proc.image.addition_constant(ima1, p)
        check_array_result(f"Image + constant ({p.value})", ima2.data, exp)
    # Constants out of the range of an integer type are never wrapped.
    ima = sigima.objects.create_image("u8", np.full((2, 2), 10, dtype=np.uint8))
    for value, expected in ((-1.0, 9.0), (3.14, 13.0), (300.0, 310.0)):
        p = sigima.params.ConstantParam.create(value=value)
        result = sigima.proc.image.addition_constant(ima, p)
        assert np.array_equal(result.data, np.full((2, 2), expected)), value


@pytest.mark.validation
def test_image_difference_constant() -> None:
    """Image difference with constant test."""
    execenv.print("*** Testing image difference with constant:")
    for ima1, p in __iterate_image_with_constant():
        dtype1 = ima1.data.dtype
        execenv.print(f"  {dtype1} -= constant ({p.value}): ", end="")
        exp = ima1.data.astype(float) - __expected_constant(p.value, dtype1)
        ima2 = sigima.proc.image.difference_constant(ima1, p)
        check_array_result(f"Image - constant ({p.value})", ima2.data, exp)
    ima = sigima.objects.create_image("u8", np.full((2, 2), 10, dtype=np.uint8))
    for value, expected in ((-1.0, 11.0), (3.14, 7.0), (300.0, -290.0)):
        p = sigima.params.ConstantParam.create(value=value)
        result = sigima.proc.image.difference_constant(ima, p)
        assert np.array_equal(result.data, np.full((2, 2), expected)), value


@pytest.mark.validation
def test_image_product_constant() -> None:
    """Image multiplication by constant test."""
    execenv.print("*** Testing image multiplication by constant:")
    for ima1, p in __iterate_image_with_constant():
        dtype1 = ima1.data.dtype
        execenv.print(f"  {dtype1} *= constant ({p.value}): ", end="")
        exp = ima1.data.astype(float) * p.value
        ima2 = sigima.proc.image.product_constant(ima1, p)
        check_array_result(f"Image x constant ({p.value})", ima2.data, exp)


@pytest.mark.validation
def test_image_division_constant() -> None:
    """Image division by constant test."""
    execenv.print("*** Testing image division by constant:")
    for ima1, p in __iterate_image_with_constant():
        dtype1 = ima1.data.dtype
        execenv.print(f"  {dtype1} /= constant ({p.value}): ", end="")
        exp = ima1.data.astype(float) / p.value
        ima2 = sigima.proc.image.division_constant(ima1, p)
        check_array_result(f"Image / constant ({p.value})", ima2.data, exp)


@pytest.mark.validation
def test_image_arithmetic() -> None:
    """Image arithmetic test."""
    execenv.print("*** Testing image arithmetic:")
    # pylint: disable=too-many-nested-blocks
    for ima1, ima2 in iterate_noisy_image_couples(size=128):
        dtype1 = ima1.data.dtype
        p = sigima.params.ArithmeticParam.create()
        for o in MathOperator:
            p.operator = o
            for a in (0.0, 1.0, 2.0):
                p.factor = a
                for b in (0.0, 1.0, 2.0):
                    p.constant = b
                    ima2.data = np.clip(ima2.data, 1, None)  # Avoid division by zero
                    ima3 = sigima.proc.image.arithmetic(ima1, ima2, p)
                    if o in (MathOperator.MULTIPLY, MathOperator.DIVIDE) and a == 0.0:
                        exp = np.ones_like(ima1.data) * b
                    elif o == MathOperator.ADD:
                        exp = np.add(ima1.data, ima2.data, dtype=float) * a + b
                    elif o == MathOperator.MULTIPLY:
                        exp = np.multiply(ima1.data, ima2.data, dtype=float) * a + b
                    elif o == MathOperator.SUBTRACT:
                        exp = np.subtract(ima1.data, ima2.data, dtype=float) * a + b
                    elif o == MathOperator.DIVIDE:
                        exp = np.divide(ima1.data, ima2.data, dtype=float) * a + b
                    if p.restore_dtype:
                        if np.issubdtype(dtype1, np.integer):
                            iinfo1 = np.iinfo(dtype1)
                            exp = np.clip(exp, iinfo1.min, iinfo1.max)
                        exp = exp.astype(dtype1)
                    check_array_result(
                        f"Arithmetic [{p.get_operation()}]", ima3.data, exp
                    )


@pytest.mark.validation
def test_image_inverse() -> None:
    """Image inverse test."""
    execenv.print("*** Testing image inverse:")
    for ima1 in iterate_noisy_images(size=128):
        execenv.print(f"  1/({ima1.data.dtype}): ", end="")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            exp = np.reciprocal(ima1.data, dtype=float)
            exp[np.isinf(exp)] = np.nan
        ima2 = sigima.proc.image.inverse(ima1)
        check_array_result("Image inverse", ima2.data, exp)


@pytest.mark.validation
def test_image_absolute() -> None:
    """Image absolute value test."""
    execenv.print("*** Testing image absolute value:")
    for ima1 in iterate_noisy_images(size=128):
        execenv.print(f"  abs({ima1.data.dtype}): ", end="")
        exp = np.abs(ima1.data)
        ima2 = sigima.proc.image.absolute(ima1)
        check_array_result("Absolute value", ima2.data, exp)


@pytest.mark.validation
def test_image_real() -> None:
    """Image real part test."""
    execenv.print("*** Testing image real part:")
    for ima1 in iterate_noisy_images(size=128):
        execenv.print(f"  re({ima1.data.dtype}): ", end="")
        exp = np.real(ima1.data)
        ima2 = sigima.proc.image.real(ima1)
        check_array_result("Real part", ima2.data, exp)


@pytest.mark.validation
def test_image_imag() -> None:
    """Image imaginary part test."""
    execenv.print("*** Testing image imaginary part:")
    for ima1 in iterate_noisy_images(size=128):
        execenv.print(f"  im({ima1.data.dtype}): ", end="")
        exp = np.imag(ima1.data)
        ima2 = sigima.proc.image.imag(ima1)
        check_array_result("Imaginary part", ima2.data, exp)


@pytest.mark.validation
def test_image_complex_from_real_imag() -> None:
    """Test :py:func:`sigima.proc.image.complex_from_real_imag`."""
    real = np.ones((4, 4))
    ima = np.arange(16).reshape(4, 4)
    ima_real = ImageObj("real")
    ima_real.data = real
    ima_imag = ImageObj("imag")
    ima_imag.data = ima
    result = complex_from_real_imag(ima_real, ima_imag)
    check_array_result(
        "complex_from_real_imag",
        result.data,
        real + 1j * ima,
    )


@pytest.mark.validation
def test_image_phase() -> None:
    """Image phase test."""
    execenv.print("*** Testing image phase:")
    for base_image in iterate_noisy_images():
        # Create a complex image for testing
        assert base_image.data is not None, "Input image data is None."
        complex_data = base_image.data.astype(np.complex128)
        complex_data += 1j * (0.5 * base_image.data + 1.0)
        complex_image = base_image.copy()
        complex_image.data = complex_data

        # Test phase extraction in radians without unwrapping
        param_rad = sigima.params.PhaseParam.create(unit=AngleUnit.RADIAN, unwrap=False)
        result_rad = sigima.proc.image.phase(complex_image, param_rad)
        assert result_rad.data is not None, "Phase in radians data is None."
        expected_rad = np.angle(complex_image.data, deg=False)
        check_array_result("Phase in radians", result_rad.data, expected_rad)

        # Test phase extraction in degrees without unwrapping
        param_deg = sigima.params.PhaseParam.create(unit=AngleUnit.DEGREE, unwrap=False)
        result_deg = sigima.proc.image.phase(complex_image, param_deg)
        assert result_deg.data is not None, "Phase in degrees data is None."
        expected_deg = np.angle(complex_image.data, deg=True)
        check_array_result("Phase in degrees", result_deg.data, expected_deg)

        # Test phase extraction in radians with unwrapping
        param_rad_unwrap = sigima.params.PhaseParam.create(
            unit=AngleUnit.RADIAN, unwrap=True
        )
        result_rad_unwrap = sigima.proc.image.phase(complex_image, param_rad_unwrap)
        expected_rad_unwrap = np.unwrap(np.angle(complex_image.data, deg=False))
        assert result_rad_unwrap.data is not None, (
            "Phase in radians with unwrapping data is None."
        )
        check_array_result(
            "Phase in radians with unwrapping",
            result_rad_unwrap.data,
            expected_rad_unwrap,
        )

        # Test phase extraction in degrees with unwrapping
        param_deg_unwrap = sigima.params.PhaseParam.create(
            unit=AngleUnit.DEGREE, unwrap=True
        )
        result_deg_unwrap = sigima.proc.image.phase(complex_image, param_deg_unwrap)
        expected_deg_unwrap = np.unwrap(
            np.angle(complex_image.data, deg=True), period=360.0
        )
        assert result_deg_unwrap.data is not None, (
            "Phase in degrees with unwrapping data is None."
        )
        check_array_result(
            "Phase in degrees with unwrapping",
            result_deg_unwrap.data,
            expected_deg_unwrap,
        )


MAGNITUDE_PHASE_TEST_CASES = [
    (np.linspace(0, np.pi, 16).reshape(4, 4), AngleUnit.RADIAN),
    (np.linspace(0, 360, 16).reshape(4, 4), AngleUnit.DEGREE),
]


@pytest.mark.parametrize("phase, unit", MAGNITUDE_PHASE_TEST_CASES)
@pytest.mark.validation
def test_image_complex_from_magnitude_phase(phase: np.ndarray, unit: AngleUnit) -> None:
    """Test :py:func:`sigima.proc.image.complex_from_magnitude_phase`.

    Args:
    phase (np.ndarray): Angles in radians or degrees.
    unit (AngleUnit): Unit of the angles, either radian or degree.
    """
    magnitude = np.full((4, 4), 2.0)
    # Create image instances for magnitude and phase
    ima_mag = ImageObj("magnitude")
    ima_mag.data = magnitude
    ima_phase = ImageObj("phase")
    ima_phase.data = phase
    # Create complex signal from magnitude and phase
    p = AngleUnitParam.create(unit=unit)
    result = complex_from_magnitude_phase(ima_mag, ima_phase, p)
    unit_str = "rad" if p.unit == AngleUnit.RADIAN else "°"
    check_array_result(
        "complex_from_magnitude_phase",
        result.data,
        polar_to_complex(magnitude, phase, unit=unit_str),
    )


def __test_all_complex_from_magnitude_phase() -> None:
    """Test all combinations of magnitude and phase."""
    for phase, unit in MAGNITUDE_PHASE_TEST_CASES:
        test_image_complex_from_magnitude_phase(phase, unit)


def __get_numpy_info(dtype: np.dtype) -> np.generic:
    """Get numpy info for a given data type."""
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype)
    return np.finfo(dtype)


@pytest.mark.validation
def test_image_astype() -> None:
    """Image type conversion test."""
    execenv.print("*** Testing image type conversion:")
    for ima1 in iterate_noisy_images(size=128):
        for dtype_str in sigima.objects.ImageObj.get_valid_dtypenames():
            dtype1_str = str(ima1.data.dtype)
            execenv.print(f"  {dtype1_str} -> {dtype_str}: ", end="")
            dtype_exp = np.dtype(dtype_str)
            info_exp = __get_numpy_info(dtype_exp)
            info_ima1 = __get_numpy_info(ima1.data.dtype)
            if info_exp.min < info_ima1.min or info_exp.max > info_ima1.max:
                continue
            exp = np.clip(ima1.data, info_exp.min, info_exp.max).astype(dtype_exp)
            p = sigima.params.DataTypeIParam.create(dtype_str=dtype_str)
            ima2 = sigima.proc.image.astype(ima1, p)
            check_array_result(
                f"Image astype({dtype1_str}->{dtype_str})", ima2.data, exp
            )


@pytest.mark.validation
def test_image_exp() -> None:
    """Image exponential test."""
    execenv.print("*** Testing image exponential:")
    with np.errstate(over="ignore"):
        for ima1 in iterate_noisy_images(size=128):
            execenv.print(f"  exp({ima1.data.dtype}): ", end="")
            exp = np.exp(ima1.data)
            ima2 = sigima.proc.image.exp(ima1)
            check_array_result("Image exp", ima2.data, exp)


@pytest.mark.validation
def test_image_log10() -> None:
    """Image base-10 logarithm test."""
    execenv.print("*** Testing image base-10 logarithm:")
    with np.errstate(over="ignore"):
        for ima1 in iterate_noisy_images(size=128):
            execenv.print(f"  log10({ima1.data.dtype}): ", end="")
            exp = np.log10(np.exp(ima1.data))
            ima2 = sigima.proc.image.log10(sigima.proc.image.exp(ima1))
            check_array_result("Image log10", ima2.data, exp)


@pytest.mark.validation
def test_image_log10_z_plus_n() -> None:
    """Image log(1+n) test."""
    execenv.print("*** Testing image log(1+n):")
    with np.errstate(over="ignore"):
        for ima1 in iterate_noisy_images(size=128):
            execenv.print(f"  log1p({ima1.data.dtype}): ", end="")
            p = sigima.params.Log10ZPlusNParam.create(n=2.0)
            exp = np.log10(ima1.data + p.n)
            ima2 = sigima.proc.image.log10_z_plus_n(ima1, p)
            check_array_result("Image log1p", ima2.data, exp)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_image_addition()
    test_image_average()
    test_image_product()
    test_image_division()
    test_image_difference()
    test_image_quadratic_difference()
    test_image_addition_constant()
    test_image_product_constant()
    test_image_difference_constant()
    test_image_division_constant()
    test_image_arithmetic()
    test_image_inverse()
    test_image_absolute()
    test_image_real()
    test_image_imag()
    test_image_phase()
    __test_all_complex_from_magnitude_phase()
    test_image_astype()
    test_image_exp()
    test_image_log10()
    test_image_log10_z_plus_n()
