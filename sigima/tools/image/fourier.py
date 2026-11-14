# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Fourier analysis module
-----------------------

This module provides 2D Fourier transform utilities and frequency domain operations
for image processing.

Features include:

- 2D FFT/IFFT functions with optional shifting
- Spectral analysis (magnitude spectrum, phase spectrum, power spectral density)
- Frequency domain filtering and deconvolution
- Zero padding utilities for FFT operations

These tools support various frequency domain image processing operations
including filtering, spectral analysis, and deconvolution.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import scipy.signal as sps

from sigima.tools._spectral import to_decibel
from sigima.tools.checks import check_2d_array, normalize_kernel

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...


def _check_nonzero_spectrum(spectrum: np.ndarray) -> None:
    """Check that a spectrum can be expressed in decibel.

    Args:
        spectrum: Non-negative spectral values.

    Raises:
        ValueError: If the spectrum is identically zero (all-zero image).
    """
    if not np.any(spectrum):
        raise ValueError(
            "The spectrum of an all-zero image cannot be expressed in decibel"
        )


@check_2d_array
def fft2d(z: np.ndarray, shift: bool = True) -> np.ndarray:
    """Compute FFT of complex array `z`

    Args:
        z: Input data
        shift: Shift zero frequency to center (default: True)

    Returns:
        FFT of input data
    """
    z1 = np.fft.fft2(z)
    if shift:
        z1 = np.fft.fftshift(z1)
    return z1


@check_2d_array
def ifft2d(z: np.ndarray, shift: bool = True) -> np.ndarray:
    """Compute inverse FFT of complex array `z`

    Args:
        z: Input data
        shift: Shift zero frequency to center (default: True)

    Returns:
        Inverse FFT of input data
    """
    if shift:
        z = np.fft.ifftshift(z)
    z1 = np.fft.ifft2(z)
    return z1


@check_2d_array
def magnitude_spectrum(z: np.ndarray, log_scale: bool = False) -> np.ndarray:
    """Compute magnitude spectrum of complex array `z`

    In decibel, the spectrum is floored at the machine epsilon of its peak value
    (about 313 dB below the peak), so that the result does not depend on the unit
    of the data.

    Args:
        z: Input data
        log_scale: Use log scale (default: False)

    Returns:
        Magnitude spectrum of input data

    Raises:
        ValueError: If `log_scale` is True and the input data is all zero.
    """
    z1 = np.abs(fft2d(z))
    if log_scale:
        _check_nonzero_spectrum(z1)
        z1 = to_decibel(z1, 20.0)
    return z1


@check_2d_array
def phase_spectrum(z: np.ndarray) -> np.ndarray:
    """Compute phase spectrum of complex array `z`

    Args:
        z: Input data

    Returns:
        Phase spectrum of input data (in degrees)
    """
    return np.rad2deg(np.angle(fft2d(z)))


@check_2d_array
def psd(z: np.ndarray, log_scale: bool = False) -> np.ndarray:
    """Compute power spectral density of complex array `z`

    In decibel, the density is floored at the machine epsilon of its peak value
    (about 156.5 dB below the peak), so that the result does not depend on the
    unit of the data.

    Args:
        z: Input data
        log_scale: Use log scale (default: False)

    Returns:
        Power spectral density of input data

    Raises:
        ValueError: If `log_scale` is True and the input data is all zero.
    """
    z1 = np.abs(fft2d(z)) ** 2
    if log_scale:
        _check_nonzero_spectrum(z1)
        z1 = to_decibel(z1, 10.0)
    return z1


@check_2d_array
def gaussian_freq_filter(
    data: np.ndarray,
    f0: float = 0.1,
    sigma: float = 0.05,
    ifft_result_type: Literal["real", "abs"] = "real",
) -> np.ndarray:
    """
    Apply a 2D Gaussian bandpass filter in the frequency domain to an image.

    This function performs a 2D Fast Fourier Transform (FFT) on the input image,
    applies a Gaussian filter centered at frequency `f0` with standard deviation `sigma`
    (both expressed in cycles per pixel), and then transforms the result back to the
    spatial domain.

    Args:
        data: Input image data.
        f0: Center frequency of the Gaussian filter (cycles/pixel).
        sigma: Standard deviation of the Gaussian filter (cycles/pixel).
        ifft_result_type: Part of the inverse FFT to return: ``"real"`` for the real
         part, ``"abs"`` for the modulus. As the filter is real and symmetric, the
         inverse FFT of a real image is real (up to rounding errors): the modulus
         is then the absolute value of the real part. Both differ for complex
         images.

    Returns:
        The filtered image.

    Raises:
        ValueError: If `ifft_result_type` is neither ``"real"`` nor ``"abs"``.
    """
    if ifft_result_type not in ("real", "abs"):
        raise ValueError(
            f"Invalid ifft_result_type: {ifft_result_type!r} (must be 'real' or 'abs')"
        )
    n, m = data.shape
    fx = np.fft.fftshift(np.fft.fftfreq(m, d=1))
    fy = np.fft.fftshift(np.fft.fftfreq(n, d=1))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    freq_radius = np.hypot(fx_grid, fy_grid)

    # Create the 2D Gaussian bandpass filter
    gaussian_filter = np.exp(-0.5 * ((freq_radius - f0) / sigma) ** 2)

    # Apply FFT, filter in frequency domain, and inverse FFT
    fft_data = fft2d(data, shift=True)
    filtered_fft = fft_data * gaussian_filter
    zout = ifft2d(filtered_fft, shift=True)
    if ifft_result_type == "abs":
        return np.abs(zout)
    return zout.real


@check_2d_array(non_constant=True)
def convolve(
    data: np.ndarray,
    kernel: np.ndarray,
    normalize_kernel_flag: bool = True,
) -> np.ndarray:
    """
    Perform 2D convolution with a kernel using scipy.signal.convolve.

    This function adds optional kernel normalization to the standard scipy convolution.

    Args:
        data: Input image (2D array).
        kernel: Convolution kernel.
        normalize_kernel_flag: If True, normalize kernel so that ``kernel.sum() == 1``
            to preserve image brightness.

    Returns:
        Convolved image (same shape as input).

    Raises:
        ValueError: If kernel is empty or null.
    """
    if kernel.size == 0 or not np.any(kernel):
        raise ValueError("Convolution kernel cannot be null.")

    # Optionally normalize the kernel
    if normalize_kernel_flag:
        kernel = normalize_kernel(kernel)

    # Use scipy.signal.convolve with 'same' mode to preserve image size
    return sps.convolve(data, kernel, mode="same", method="auto")


@check_2d_array(non_constant=True)
def deconvolve(
    data: np.ndarray,
    kernel: np.ndarray,
    reg: float = 0.0,
    boundary: str = "edge",
    normalize_kernel_flag: bool = True,
) -> np.ndarray:
    """
    Perform 2D FFT deconvolution with correct 'same' geometry (no shift).

    The kernel (PSF) must be centered (impulse at center for identity kernel).
    Odd kernel sizes are recommended.

    Args:
        data: Input image (2D array).
        kernel: Point Spread Function (PSF), centered.
        reg: Regularization parameter (if >0, Wiener/Tikhonov inverse:
         ``H* / (|H|^2 + reg))``.
        boundary: Padding mode ('edge' for constant plateau,
         'reflect' for symmetric mirror).
        normalize_kernel_flag: If True, normalize kernel so that ``kernel.sum() == 1``
            to preserve image brightness.

    Returns:
        Deconvolved image (same shape as input).

    Raises:
        ValueError: If kernel is empty or null.
    """
    if kernel.size == 0 or not np.any(kernel):
        raise ValueError("Deconvolution kernel cannot be null.")

    # Optionally normalize the kernel
    if normalize_kernel_flag:
        kernel = normalize_kernel(kernel)

    H, W = data.shape
    kh, kw = kernel.shape

    if kh % 2 == 0 or kw % 2 == 0:
        # Warning for even-sized kernels (off-by-one in centered FFT)
        warnings.warn(
            f"Deconvolution kernel has even dimension(s) ({kh}×{kw}); "
            f"odd dimensions recommended for centered FFT."
        )

    # Symmetric padding for centered 'same' convolution
    top = kh // 2
    bottom = kh - 1 - top
    left = kw // 2
    right = kw - 1 - left
    data_pad = np.pad(data, ((top, bottom), (left, right)), mode=boundary)
    Hp, Wp = data_pad.shape  # = H+kh-1, W+kw-1

    # Centered PSF to OTF conversion (avoid off-by-one for even sizes)
    kernel_pad = np.zeros_like(data_pad, dtype=float)
    r0 = Hp // 2 - kh // 2
    c0 = Wp // 2 - kw // 2
    kernel_pad[r0 : r0 + kh, c0 : c0 + kw] = kernel
    H_otf = np.fft.fft2(np.fft.ifftshift(kernel_pad))  # center → (0,0)

    # FFT of padded image (no shift)
    Z = np.fft.fft2(data_pad)

    # Frequency domain inversion
    if reg > 0.0:
        Hc = np.conj(H_otf)
        X = Z * Hc / (np.abs(H_otf) ** 2 + float(reg))
    else:
        eps = 1e-12
        X = Z / (H_otf + eps)

    data_true_pad = np.fft.ifft2(X).real

    # Central crop to restore original geometry
    out = data_true_pad[top : top + H, left : left + W]
    return out
