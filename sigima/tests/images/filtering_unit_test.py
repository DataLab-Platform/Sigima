# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for image filtering functions.
"""

from __future__ import annotations

import numpy as np
import pytest

import sigima.proc.image as sigima_image
from sigima.objects import ImageObj
from sigima.objects.image import create_image
from sigima.tests.helpers import check_scalar_result
from sigima.tools.image import freq_fft_filter


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


@pytest.mark.gui
def test_freq_fft_filter_bandpass_interactive() -> None:
    """Test freq_fft_filter with dtype argument."""
    from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

    clean, noisy = build_clean_noisy_images(freq=0.05)
    for result_type in ("real", "abs"):
        zout_filt = freq_fft_filter(
            zin=noisy.data, f0=0.05, sigma=0.05, ifft_result_type=result_type
        )
        clean_area = clean.data[10:-10, 10:-10]
        if result_type == "abs":
            clean_area = np.abs(clean_area)
        mean_noise = float(np.mean(np.abs(clean_area - zout_filt[10:-10, 10:-10])))
        check_scalar_result(
            f"fft filter noise reduction ({result_type})", mean_noise, 0, atol=0.1
        )
        vistools.view_images_side_by_side(
            [clean, noisy, zout_filt],
            titles=["Start image", "Noisy Image", f"Filtered ({result_type})"],
        )


@pytest.mark.validation
def test_computation_freq_fft() -> None:
    """Validation test for freq_fft computation function."""
    clean, noisy = build_clean_noisy_images(freq=0.05)
    param = sigima_image.FreqFFTParam.create(f0=0.05, sigma=0.05)
    for result_type in ("real", "abs"):
        param.ifft_result_type = result_type
        filt = sigima_image.freq_fft(noisy, param)
        clean_area = clean.data[10:-10, 10:-10]
        if result_type == "abs":
            clean_area = np.abs(clean_area)
        mean_noise = float(np.mean(np.abs(clean_area - filt.data[10:-10, 10:-10])))
        check_scalar_result(
            f"fft filter noise reduction ({result_type})", mean_noise, 0, atol=0.1
        )


def test_freq_fft_filter_constant_image() -> None:
    """Edge case: filtering a constant image must preserve the constant value
    (DC component)."""
    img_const = np.full((64, 64), fill_value=7.42)
    zout = freq_fft_filter(img_const, f0=0.0, sigma=0.05)
    # Ignore borders
    center = zout[10:-10, 10:-10]
    # Assert that all values are (almost) equal to the original constant
    assert np.allclose(center, 7.42, atol=1e-10), "Filtering constant image failed"


def test_freq_fft_filter_symmetry() -> None:
    """Test: filtering a symmetric image yields a symmetric result."""
    # Create a symmetric image (e.g., a centered 2D Gaussian)
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    xv, yv = np.meshgrid(x, y)
    img = np.exp(-(xv**2 + yv**2) / 0.1)

    zout = freq_fft_filter(img, f0=0.05, sigma=0.02)
    # Symmetry check: image must be (almost) symmetric along both axes
    assert np.allclose(zout, zout[::-1, :], atol=1e-10), "Vertical symmetry lost"
    assert np.allclose(zout, zout[:, ::-1], atol=1e-10), "Horizontal symmetry lost"


if __name__ == "__main__":
    test_freq_fft_filter_bandpass_interactive()
    test_computation_freq_fft()
    test_freq_fft_filter_constant_image()
    test_freq_fft_filter_symmetry()
