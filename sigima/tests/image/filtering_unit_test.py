# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for image filtering functions.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as spi
import scipy.signal as sps
from skimage import filters

import sigima.enums
import sigima.params
import sigima.proc.image
import sigima.tools.image
from sigima.objects import ImageObj
from sigima.objects.image import create_image
from sigima.tests import guiutils
from sigima.tests.data import get_test_image
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_image_gaussian_filter() -> None:
    """Validation test for the image Gaussian filter processing."""
    src = get_test_image("flower.npy")
    for sigma in (10.0, 50.0):
        p = sigima.params.GaussianParam.create(sigma=sigma)
        dst = sigima.proc.image.gaussian_filter(src, p)
        exp = spi.gaussian_filter(src.data, sigma=sigma)
        check_array_result(f"GaussianFilter[sigma={sigma}]", dst.data, exp)


@pytest.mark.validation
def test_image_moving_average() -> None:
    """Validation test for the image moving average processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.MovingAverageParam.create(n=30)
    for mode in sigima.enums.FilterMode:
        p.mode = mode
        dst = sigima.proc.image.moving_average(src, p)
        exp = spi.uniform_filter(src.data, size=p.n, mode=mode.value)
        check_array_result(f"MovingAvg[n={p.n},mode={p.mode}]", dst.data, exp)


@pytest.mark.validation
def test_image_moving_median() -> None:
    """Validation test for the image moving median processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.MovingMedianParam.create(n=5)
    for mode in sigima.enums.FilterMode:
        p.mode = mode
        dst = sigima.proc.image.moving_median(src, p)
        exp = spi.median_filter(src.data, size=p.n, mode=mode.value)
        check_array_result(f"MovingMed[n={p.n},mode={p.mode}]", dst.data, exp)


@pytest.mark.validation
def test_image_wiener() -> None:
    """Validation test for the image Wiener filter processing."""
    src = get_test_image("flower.npy")
    dst = sigima.proc.image.wiener(src)
    exp = sps.wiener(src.data)
    check_array_result("Wiener", dst.data, exp)


@pytest.mark.validation
def test_butterworth() -> None:
    """Validation test for the image Butterworth filter processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.ButterworthParam.create(order=2, cut_off=0.5, high_pass=False)
    dst = sigima.proc.image.butterworth(src, p)
    exp = filters.butterworth(src.data, p.cut_off, p.high_pass, p.order)
    check_array_result(
        f"Butterworth[order={p.order},cut_off={p.cut_off},high_pass={p.high_pass}]",
        dst.data,
        exp,
    )


def build_clean_noisy_images(
    shape: tuple[int, int] = (64, 64), noise_level: float = 0.4, freq: float = 0.05
) -> tuple[ImageObj, ImageObj]:
    """Generate a test image with a low-frequency signal and high-frequency noise.

    Args:
        shape: Shape of the image (height, width).
        noise_level: Standard deviation of the Gaussian noise.
        freq: Frequency of the low-frequency signal.

    Returns:
        Tuple of (clean, noisy) where:
        - clean: The low-frequency component of the image.
        - noisy: The noisy image with added Gaussian noise.
    """
    # Low frequency signal: sinusoid with frequency in px^-1
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    x_matrice, y_matrice = np.meshgrid(x, y)
    low_freq = 0.5 * (np.sin(2 * np.pi * freq * x_matrice)) + 0.5 * (
        np.sin(2 * np.pi * freq * y_matrice)
    )

    # High frequency noise
    rng = np.random.default_rng(seed=0)
    high_freq_noise = noise_level * rng.standard_normal(size=shape)

    # Sum of the two components
    img = low_freq + high_freq_noise
    return create_image("clean", low_freq), create_image("noisy", img)


@pytest.mark.validation
@pytest.mark.parametrize("ifft_result_type", ["real", "abs"])
def test_gaussian_freq_filter(ifft_result_type: str) -> None:
    """Validation test for :py:func:`sigima.tools.image.gaussian_freq_filter`."""
    clean, noisy = build_clean_noisy_images(freq=0.05)
    param = sigima.proc.image.GaussianFreqFilterParam.create(
        f0=0.05, sigma=0.05, ifft_result_type=ifft_result_type
    )
    filt = sigima.proc.image.gaussian_freq_filter(noisy, param)
    clean_area = clean.data[10:-10, 10:-10]
    if ifft_result_type == "abs":
        clean_area = np.abs(clean_area)
    guiutils.view_images_side_by_side_if_gui(
        [clean, noisy, filt],
        titles=["Clean", "Noisy", f"Filtered ({ifft_result_type})"],
    )
    mean_noise = float(np.mean(np.abs(clean_area - filt.data[10:-10, 10:-10])))
    check_scalar_result(
        f"gaussian_freq_filter noise reduction ({ifft_result_type})",
        mean_noise,
        0.0,
        atol=0.1,
    )


def _inverse_fft_of_gaussian_bandpass(
    data: np.ndarray, f0: float, sigma: float
) -> np.ndarray:
    """Return the complex inverse FFT of an image filtered by a Gaussian bandpass,
    computed on the unshifted frequency grid (independently of Sigima)."""
    fy = np.fft.fftfreq(data.shape[0])[:, np.newaxis]
    fx = np.fft.fftfreq(data.shape[1])[np.newaxis, :]
    gain = np.exp(-0.5 * ((np.hypot(fx, fy) - f0) / sigma) ** 2)
    return np.fft.ifft2(np.fft.fft2(data) * gain)


@pytest.mark.parametrize("is_complex", [False, True], ids=["real_image", "complex"])
def test_gaussian_freq_filter_ifft_result_type(is_complex: bool) -> None:
    """Regression test: the "Inverse FFT result" parameter is applied
    (Sigima issue #123)."""
    rng = np.random.default_rng(seed=0)
    data = rng.normal(size=(64, 48))
    if is_complex:
        data = data + 1j * rng.normal(size=data.shape)
    src = create_image("noise", data)
    zout = _inverse_fft_of_gaussian_bandpass(data, f0=0.1, sigma=0.05)
    titles = []
    for ifft_result_type, expected in (("real", zout.real), ("abs", np.abs(zout))):
        param = sigima.proc.image.GaussianFreqFilterParam.create(
            f0=0.1, sigma=0.05, ifft_result_type=ifft_result_type
        )
        dst = sigima.proc.image.gaussian_freq_filter(src, param)
        check_array_result(
            f"gaussian_freq_filter[{ifft_result_type}]",
            dst.data,
            expected,
            rtol=1e-10,
            atol=1e-12,
        )
        titles.append(dst.title)
    assert "ifft=real" in titles[0] and "ifft=abs" in titles[1], titles


def test_gaussian_freq_filter_invalid_ifft_result_type() -> None:
    """An unknown inverse FFT result type is refused."""
    with pytest.raises(ValueError, match="ifft_result_type"):
        sigima.tools.image.gaussian_freq_filter(
            np.ones((8, 8)), ifft_result_type="imag"
        )


def test_gaussian_freq_filter_constant_image() -> None:
    """Edge case: filtering a constant image must preserve the constant value
    (DC component)."""
    img_const = np.full((64, 64), fill_value=7.42)
    zout = sigima.tools.image.gaussian_freq_filter(img_const, f0=0.0, sigma=0.05)
    # Ignore borders
    center = zout[10:-10, 10:-10]
    # Assert that all values are (almost) equal to the original constant
    assert np.allclose(center, 7.42, atol=1e-10), "Filtering constant image failed"


def test_gaussian_freq_filter_symmetry() -> None:
    """Test: filtering a symmetric image yields a symmetric result."""
    # Create a symmetric image (e.g., a centered 2D Gaussian)
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    xv, yv = np.meshgrid(x, y)
    img = np.exp(-(xv**2 + yv**2) / 0.1)

    zout = sigima.tools.image.gaussian_freq_filter(img, f0=0.05, sigma=0.02)
    # Symmetry check: image must be (almost) symmetric along both axes
    assert np.allclose(zout, zout[::-1, :], atol=1e-10), "Vertical symmetry lost"
    assert np.allclose(zout, zout[:, ::-1], atol=1e-10), "Horizontal symmetry lost"


if __name__ == "__main__":
    guiutils.enable_gui()
    test_image_gaussian_filter()
    test_image_moving_average()
    test_image_moving_median()
    test_image_wiener()
    test_butterworth()
    for result_type in ("real", "abs"):
        test_gaussian_freq_filter(result_type)
    test_gaussian_freq_filter_ifft_result_type(False)
    test_gaussian_freq_filter_ifft_result_type(True)
    test_gaussian_freq_filter_invalid_ifft_result_type()
    test_gaussian_freq_filter_constant_image()
    test_gaussian_freq_filter_symmetry()
