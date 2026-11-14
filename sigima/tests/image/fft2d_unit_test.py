# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image FFT unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image
import sigima.tests.data
import sigima.tools.image
from sigima.tests import guiutils
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.gui
def test_image_fft_interactive():
    """2D FFT interactive test."""
    with guiutils.lazy_qt_app_context(force=True):
        from sigima import viz  # pylint: disable=import-outside-toplevel

        # Create a 2D ring image
        execenv.print("Generating 2D ring image...", end=" ")
        data = sigima.tests.data.create_ring_image().data
        execenv.print("OK")

        # FFT
        execenv.print("Computing FFT of image...", end=" ")
        f = sigima.tools.image.fft2d(data)
        data2 = sigima.tools.image.ifft2d(f)
        execenv.print("OK")
        execenv.print("Comparing original and FFT/iFFT images...", end=" ")
        check_array_result(
            "Image FFT/iFFT", np.array(data2.real, dtype=data.dtype), data, rtol=1e-3
        )
        execenv.print("OK")

        images = [data, f.real, f.imag, np.abs(f), data2.real, data2.imag]
        titles = ["Original", "Re(FFT)", "Im(FFT)", "Abs(FFT)", "Re(iFFT)", "Im(iFFT)"]
        viz.view_images_side_by_side(images, titles, rows=2, title="2D FFT/iFFT")


@pytest.mark.validation
def test_image_zero_padding() -> None:
    """2D FFT zero padding validation test."""
    ima1 = sigima.tests.data.create_checkerboard()
    rows, cols = 2, 2
    param = sigima.params.ZeroPadding2DParam.create(rows=rows, cols=cols)
    assert param.strategy == "custom", (
        f"Wrong default strategy: {param.strategy} (expected 'custom')"
    )

    # Validate the zero padding with bottom-right position
    param.position = "bottom-right"
    ima2 = sigima.proc.image.zero_padding(ima1, param)
    sh1, sh2 = ima1.data.shape, ima2.data.shape
    exp_sh2 = (sh1[0] + rows, sh1[1] + cols)
    execenv.print("Validating zero padding for bottom-right position...", end=" ")
    assert sh2 == exp_sh2, f"Wrong shape: {sh2} (expected {exp_sh2})"
    assert np.all(ima2.data[0 : sh1[0], 0 : sh1[1]] == ima1.data), (
        "Altered data in original image area"
    )
    assert np.all(ima2.data[sh1[0] : sh2[0], sh1[1] : sh2[1]] == 0), (
        "Altered data in padded area"
    )
    execenv.print("OK")

    # Validate the zero padding with center position
    param.position = "around"
    ima3 = sigima.proc.image.zero_padding(ima1, param)
    sh3 = ima3.data.shape
    exp_sh3 = (sh1[0] + rows, sh1[1] + cols)
    execenv.print("Validating zero padding for around position...", end=" ")
    assert sh3 == exp_sh3, f"Wrong shape: {sh3} (expected {exp_sh3})"
    assert np.all(
        ima3.data[rows // 2 : sh1[0] + rows // 2, cols // 2 : sh1[1] + cols // 2]
        == ima1.data
    ), "Altered data in original image area"
    assert np.all(ima3.data[0 : rows // 2, :] == 0), "Altered data in padded area (top)"
    assert np.all(ima3.data[sh1[0] + rows // 2 :, :] == 0), (
        "Altered data in padded area (bottom)"
    )
    assert np.all(ima3.data[:, 0 : cols // 2] == 0), (
        "Altered data in padded area (left)"
    )
    assert np.all(ima3.data[:, sh1[1] + cols // 2 :] == 0), (
        "Altered data in padded area (right)"
    )
    execenv.print("OK")

    # Validate zero padding with strategies other than custom size
    # Image size is (200, 300) and the next power of 2 is (256, 512)
    # The multiple of 64 is (256, 320)
    ima4 = sigima.objects.create_image("", np.zeros((200, 300)))
    for strategy, (exp_rows, exp_cols) in (
        ("next_pow2", (56, 212)),
        ("multiple_of_64", (56, 20)),
    ):
        param = sigima.params.ZeroPadding2DParam.create(strategy=strategy)
        param.update_from_obj(ima4)
        assert param.rows == exp_rows, (
            f"Wrong row number for '{param.strategy}' strategy: {param.rows}"
            f" (expected {exp_rows})"
        )
        assert param.cols == exp_cols, (
            f"Wrong column number for '{param.strategy}' strategy: {param.cols}"
            f" (expected {exp_cols})"
        )


@pytest.mark.validation
def test_image_fft() -> None:
    """2D FFT validation test."""
    ima1 = sigima.tests.data.create_checkerboard()
    fft = sigima.proc.image.fft(ima1)
    ifft = sigima.proc.image.ifft(fft)

    # Check that the inverse FFT reconstructs the original image
    check_array_result("Checkerboard image FFT/iFFT", ifft.data.real, ima1.data)

    # Parseval's Theorem Validation
    original_energy = np.sum(np.abs(ima1.data) ** 2)
    transformed_energy = np.sum(np.abs(fft.data) ** 2) / (ima1.data.size)
    check_scalar_result("Parseval's Theorem", transformed_energy, original_energy)


@pytest.mark.validation
def test_image_ifft() -> None:
    """2D iFFT validation test: the inverse FFT reconstructs the image."""
    ima1 = sigima.tests.data.create_checkerboard()
    ifft = sigima.proc.image.ifft(sigima.proc.image.fft(ima1))
    check_array_result("Checkerboard image iFFT(FFT)", ifft.data.real, ima1.data)


@pytest.mark.validation
def test_image_magnitude_spectrum() -> None:
    """2D magnitude spectrum validation test."""
    ima1 = sigima.tests.data.create_checkerboard()
    exp = np.abs(sigima.proc.image.fft(ima1).data)
    exp_db = 20 * np.log10(np.maximum(exp, exp.max() * np.finfo(float).eps))
    param = sigima.params.SpectrumParam()
    for decibel, expected in ((False, exp), (True, exp_db)):
        param.decibel = decibel
        mag = sigima.proc.image.magnitude_spectrum(ima1, param)
        check_array_result(
            f"Checkerboard image FFT magnitude spectrum (dB={decibel})",
            mag.data,
            expected,
        )


@pytest.mark.validation
def test_image_phase_spectrum() -> None:
    """2D phase spectrum validation test."""
    ima1 = sigima.tests.data.create_checkerboard()
    fft = sigima.proc.image.fft(ima1)
    phase = sigima.proc.image.phase_spectrum(ima1)

    # Check that the phase spectrum is correct
    exp = np.rad2deg(np.angle(fft.data))
    check_array_result("Checkerboard image FFT phase spectrum", phase.data, exp)


@pytest.mark.validation
def test_image_psd() -> None:
    """2D Power Spectral Density validation test."""
    ima1 = sigima.tests.data.create_checkerboard()
    exp = np.abs(sigima.proc.image.fft(ima1).data) ** 2
    exp_db = 10 * np.log10(np.maximum(exp, exp.max() * np.finfo(float).eps))
    param = sigima.params.SpectrumParam()
    for decibel, expected in ((False, exp), (True, exp_db)):
        param.decibel = decibel
        psd = sigima.proc.image.psd(ima1, param)
        check_array_result(f"Checkerboard image PSD (dB={decibel})", psd.data, expected)


SPECTRUM_FUNCS_DB = (
    # Function, decibel factor
    (sigima.tools.image.magnitude_spectrum, 20.0),
    (sigima.tools.image.psd, 10.0),
)


@pytest.mark.parametrize("func", [func for func, _factor in SPECTRUM_FUNCS_DB])
def test_image_spectrum_db_unit_independence(func) -> None:
    """Scaling an image only shifts its spectrum in decibel (regression test).

    The decibel floor was absolute (1e-10), so that the dynamic range of the
    spectrum of a low-amplitude image collapsed onto the floor.
    """
    rng = np.random.default_rng(0)
    data = rng.random((64, 64))
    ref = func(data, log_scale=True)
    for scale in (1e3, 1e-12):
        shifted = func(data * scale, log_scale=True)
        # Amplitude scales as `scale`, power as `scale**2`: same shift in decibel
        check_array_result(
            f"{func.__name__} in dB, image scaled by {scale}",
            shifted - ref,
            np.full(ref.shape, 20.0 * np.log10(scale)),
            rtol=0.0,
            atol=1e-6,
        )


@pytest.mark.parametrize("func, factor", SPECTRUM_FUNCS_DB)
def test_image_spectrum_db_zero_lines(func, factor) -> None:
    """Exactly zero spectral lines are floored relative to the peak, finite."""
    data = np.full((32, 32), 5.0)  # Constant image: only the DC line is not zero
    result = func(data, log_scale=True)
    assert np.all(np.isfinite(result)), "Non-finite values in the spectrum"
    peak = result.max()
    floor = np.delete(result.ravel(), result.argmax())
    check_array_result(
        f"{func.__name__} in dB, zero lines",
        floor,
        np.full(floor.shape, peak + factor * np.log10(np.finfo(float).eps)),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("func", [func for func, _factor in SPECTRUM_FUNCS_DB])
def test_image_spectrum_db_zero_image(func) -> None:
    """The spectrum of an all-zero image cannot be expressed in decibel."""
    data = np.zeros((16, 16))
    with pytest.raises(ValueError, match="all-zero image"):
        func(data, log_scale=True)
    assert np.all(func(data, log_scale=False) == 0.0)


if __name__ == "__main__":
    test_image_fft_interactive()
    test_image_zero_padding()
    test_image_fft()
    test_image_magnitude_spectrum()
    test_image_phase_spectrum()
    test_image_psd()
