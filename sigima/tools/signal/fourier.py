# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Fourier Analysis (see parent package :mod:`sigima.tools.signal`).

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.signal  # type: ignore[import]

from sigima.tools.checks import check_1d_arrays, normalize_kernel
from sigima.tools.signal.dynamic import sampling_rate


def _to_decibel(values: np.ndarray, factor: float) -> np.ndarray:
    """Convert non-negative spectral values to a decibel scale.

    Spectral values are exactly zero wherever the signal has no content at a given
    frequency -- a synthetic signal, a zero-padded segment or a notch filter all
    produce them. Taking the logarithm as-is yields ``-inf`` samples, which propagate
    to autoscaling, statistics and file export, and emit a divide-by-zero warning.
    Values are therefore floored at the machine epsilon of the peak value, i.e. about
    313 dB below the peak for the amplitude convention: far below any physically
    meaningful level, yet finite.

    Args:
        values: Non-negative spectral values.
        factor: 20 for amplitude quantities, 10 for power quantities.

    Returns:
        Values expressed in decibel.
    """
    peak = np.max(values)
    if peak <= 0.0:
        # Identically zero spectrum: there is no dynamic range to represent
        return np.full(values.shape, -np.inf)
    return factor * np.log10(np.maximum(values, peak * np.finfo(float).eps))


@check_1d_arrays(x_evenly_spaced=True)
def zero_padding(
    x: np.ndarray, y: np.ndarray, n_prepend: int = 0, n_append: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Prepend and append zeros.

    This function pads the input signal with zeros at the beginning and end.

    Args:
        x: X data.
        y: Y data.
        n_prepend: Number of zeros to prepend.
        n_append: Number of zeros to append.

    Returns:
        Tuple (xnew, ynew): Padded x and y.
    """
    if n_prepend < 0:
        raise ValueError("Number of zeros to prepend must be non-negative.")
    if n_append < 0:
        raise ValueError("Number of zeros to append must be non-negative.")

    dx = np.mean(np.diff(x))
    xnew = np.linspace(
        x[0] - n_prepend * dx,
        x[-1] + n_append * dx,
        y.size + n_prepend + n_append,
    )
    ynew = np.pad(y, (n_prepend, n_append), mode="constant")
    return xnew, ynew


@check_1d_arrays(x_evenly_spaced=True)
def fft1d(
    x: np.ndarray, y: np.ndarray, shift: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Fast Fourier Transform (FFT) of a 1D real signal.

    The transform is unnormalized, following NumPy's ``backward`` convention: the
    returned values scale with the number of samples, and the round trip through
    :func:`ifft1d` restores the original amplitudes.

    Args:
        x: Time domain axis (evenly spaced).
        y: Signal values.
        shift: If True, shift zero frequency and its corresponding FFT component to the
        center.

    Returns:
        Tuple (f, sp): Frequency axis and corresponding FFT values.
    """
    dt = np.mean(np.diff(x))
    f = np.fft.fftfreq(x.size, d=dt)  # Frequency axis
    sp = np.fft.fft(y)  # Spectrum values
    if shift:
        f = np.fft.fftshift(f)
        sp = np.fft.fftshift(sp)
    return f, sp


@check_1d_arrays(x_evenly_spaced=False, x_sorted=False, y_dtype=np.complexfloating)
def ifft1d(
    f: np.ndarray, sp: np.ndarray, initial: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the inverse Fast Fourier Transform (FFT) of a 1D complex spectrum.

    Args:
        f: Frequency axis (evenly spaced).
        sp: FFT values.
        initial: Starting value for the time axis.

    Returns:
        Tuple (x, y): Time axis and real signal.

    Raises:
        ValueError: If frequency array is not evenly spaced or has fewer than 2 points.
    """
    if f.size < 2:
        raise ValueError("Frequency array must have at least two elements.")

    if np.all(np.diff(f) >= 0.0):
        # If frequencies are sorted, assume input is shifted.
        # The spectrum needs to be unshifted.
        sp = np.fft.ifftshift(sp)
    else:
        # Otherwise assume input is not shifted.
        # The frequencies need to be shifted.
        f = np.fft.fftshift(f)

    diff_f = np.diff(f)
    df = np.mean(diff_f)
    if not np.allclose(diff_f, df):
        raise ValueError("Frequency array must be evenly spaced.")

    y = np.fft.ifft(sp)
    dt = 1.0 / (f.size * df)
    x = np.linspace(initial, initial + (y.size - 1) * dt, y.size)

    return x, y.real


@check_1d_arrays(x_evenly_spaced=True)
def magnitude_spectrum(
    x: np.ndarray, y: np.ndarray, decibel: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude spectrum.

    The magnitude is the modulus of the unnormalized FFT (see :func:`fft1d`), so it
    scales with the number of samples. Only ratios and relative levels are meaningful.

    Args:
        x: X data.
        y: Y data.
        decibel: Compute the magnitude spectrum root-power level in decibel (dB).

    Returns:
        Tuple (f, mag_spectrum): Frequency values and magnitude spectrum.
    """
    f, spectrum = fft1d(x, y)
    mag_spectrum = np.abs(spectrum)
    if decibel:
        mag_spectrum = _to_decibel(mag_spectrum, 20.0)
    return f, mag_spectrum


@check_1d_arrays(x_evenly_spaced=True)
def phase_spectrum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute phase spectrum.

    Args:
        x: X data.
        y: Y data.

    Returns:
        Tuple (f, phase): Frequency values and phase spectrum in degrees.
    """
    f, spectrum = fft1d(x, y)
    phase = np.rad2deg(np.angle(spectrum))
    return f, phase


@check_1d_arrays(x_evenly_spaced=True)
def psd(
    x: np.ndarray, y: np.ndarray, decibel: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the Power Spectral Density (PSD) using Welch's method.

    The estimate is a power *density*, expressed in squared signal units per hertz,
    computed by :func:`scipy.signal.welch` with its default Hann window and default
    segment length. Signals shorter than that segment length are processed as a single
    segment, which trades frequency resolution for variance.

    Args:
        x: X data.
        y: Y data.
        decibel: Compute the power spectral density power level in decibel (dB).

    Returns:
        Tuple (f, welch_psd): Frequency values and PSD.
    """
    f, welch_psd = scipy.signal.welch(y, fs=sampling_rate(x))
    if decibel:
        welch_psd = _to_decibel(welch_psd, 10.0)
    return f, welch_psd


@check_1d_arrays(x_evenly_spaced=True)
def convolve(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    boundary: Literal["reflect", "symmetric", "edge", "wrap"] = "reflect",
    normalize_kernel_flag: bool = True,
    method: Literal["auto", "direct", "fft"] = "auto",
    correct_group_delay: bool = True,
) -> np.ndarray:
    """Convolve a 1D signal with a kernel, avoiding border artifacts and x-shift.

    The input signal is padded before convolution, then a 'valid' extraction is
    used to return exactly len(y) samples. Non-zero padding (e.g. "reflect")
    prevents the typical edge attenuation caused by implicit zero-padding.
    If the kernel is asymmetric, an optional group-delay correction recenters the
    output on the same x-grid (no shift), using sub-sample interpolation.

    Args:
        x: 1D monotonically increasing and uniformly spaced axis (same length as y).
        y: 1D input signal.
        h: 1D convolution kernel (impulse response).
        boundary: Padding mode passed to ``np.pad`` ("reflect" recommended).
        normalize_kernel_flag: If True, normalize kernel so that ``h.sum() == 1`` to
         preserve DC level.
        method: Convolution method for ``scipy.signal.convolve``.
        correct_group_delay: If True, compensate the kernel center-of-mass shift
         (group delay) to avoid any x-shift in the output.

    Returns:
        Convolved signal with the same length as ``y``, aligned on ``x``.

    Raises:
        ValueError: If inputs are not 1D, empty, or shapes are inconsistent.

    Notes:
        Precondition: ``x`` is strictly increasing with constant spacing. This is
        required for standard discrete convolution to represent a physical LTI
        filtering on the given grid.
    """
    if h.size != y.size:
        raise ValueError("X data and Y data of the filter must have the same size.")

    # ---- Optional DC preservation
    if normalize_kernel_flag:
        h = normalize_kernel(h)

    M = int(h.size)
    if M == 1:
        # With normalization, h == [1]; otherwise scale by h[0]
        return y.copy() if normalize_kernel_flag else y * h[0]

    # ---- Compute asymmetric pad widths so that 'valid' returns exactly len(y)
    w_left = M // 2
    w_right = (M - 1) - w_left

    # ---- Pad the signal to mitigate border artifacts during convolution
    y_pad = np.pad(y, (w_left, w_right), mode=boundary)

    # ---- Linear convolution with 'valid' to get back exactly N samples
    y_conv = scipy.signal.convolve(y_pad, h, mode="valid", method=method)

    if correct_group_delay:
        # Center-of-mass of the kernel in sample units relative to w_left.
        # n runs from -w_left ... +w_right (integer sample offsets).
        n = np.arange(M, dtype=float) - w_left
        denom = h.sum() if h.sum() != 0.0 else 1.0
        mu_samples = float(np.dot(n, h) / denom)  # may be fractional

        if np.isfinite(mu_samples) and mu_samples != 0.0:
            # Sub-sample compensation on the *x-axis* to keep alignment.
            # Positive mu_samples means the effective kernel center is to the right
            # (additional delay); compensate by advancing the output.
            dx = float(x[1] - x[0])  # uniform spacing guaranteed by your decorator
            x_shifted = x + mu_samples * dx
            # Interpolate with edge holding to maintain length and alignment
            y_conv = np.interp(x, x_shifted, y_conv, left=y_conv[0], right=y_conv[-1])

    return y_conv


def _psf_to_otf_1d(h: np.ndarray, L: int) -> np.ndarray:
    """Convert a centered 1D PSF h to an OTF (RFFT length L).

    The PSF center (index floor((M-1)/2)) is shifted to index 0 before FFT so that
    the convolution geometry matches 'same' with a centered kernel.

    Args:
        h: 1D convolution kernel (PSF).
        L: Length of the output OTF (RFFT length, power of two recommended).

    Returns:
        OTF as a 1D complex array of length L//2 + 1 (RFFT output).
    """
    M = h.size
    w_left = M // 2
    h0 = np.roll(h, -w_left)  # center -> index 0
    h_z = np.zeros(L, dtype=float)
    h_z[:M] = h0
    return np.fft.rfft(h_z)


@check_1d_arrays(x_evenly_spaced=True)
def deconvolve(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    *,
    boundary: Literal["reflect", "symmetric", "edge", "wrap"] = "reflect",
    normalize_kernel_flag: bool = True,
    # regularized inverse with derivative prior (recommended):
    method: Literal["wiener", "fft"] = "wiener",
    reg: float = 5e-2,  # increase to reduce ringing (e.g. 5e-2, 1e-1)
    gain_max: float | None = 10.0,  # clamp max |gain| in frequency (None to disable)
    dc_lock: bool = True,  # force exact DC gain (preserve plateau)
    auto_scale: bool = True,  # auto-correct amplitude scaling after deconvolution
) -> np.ndarray:
    """Deconvolve a 1D signal with frequency-dependent regularization and DC lock.

    Strategy:
      1) Pad y with the same geometry as the ``convolve`` step (x-uniform grid).
      2) Build OTF ``H(f)`` from the centered PSF ``h``.
      3) Compute inverse filter:
           - ``wiener`` (recommended): ``H*(f) / (|H|^2 + reg * |D|^2)``, with
             ``|D|^2 = (2 sin(ω/2))^2`` (1st-derivative prior).
           - ``fft``: bare inverse ``1/H(f)`` (unstable; only for noise-free data).
           - Optionally clamp ``|G(f)| ≤ gain_max`` and lock DC gain.
      4) IFFT, then extract the central unpadded segment (``len == len(y)``).
      5) Optionally auto-scale the result to correct amplitude bias from regularization.

    Args:
        x: Strictly increasing, uniformly spaced axis (same length as y).
        y: Observed signal (result of ``y_true ⊛ h``, plus noise).
        h: Centered convolution kernel (PSF).
        boundary: Padding mode (should match your convolution).
        normalize_kernel_flag: If True, normalize ``h`` to preserve DC.
        method: ``"wiener"`` (regularized inverse) or ``"fft"`` (bare inverse).
        reg: Regularization strength for the derivative prior.
        gain_max: Optional clamp on ``|G(f)|`` to avoid wild amplification.
        dc_lock: If True, enforce exact DC gain (preserve mean/plateau).
        auto_scale: If True, auto-correct amplitude scaling after deconvolution.

    Returns:
        Deconvolved signal with the same length as y, x-aligned.
    """
    if x.ndim != 1 or y.ndim != 1 or h.ndim != 1:
        raise ValueError("`x`, `y`, and `h` must be 1D arrays.")
    if y.size == 0 or h.size == 0 or x.size != y.size:
        raise ValueError("Non-empty arrays required and `x` length must match `y`.")
    if y.size != h.size:
        raise ValueError("X data and Y data of the filter must have the same size.")
    if np.all(h == 0.0):
        raise ValueError("Filter is all zeros, cannot be used to deconvolve.")

    y = np.asarray(y, dtype=float)
    h = np.asarray(h, dtype=float)

    # Check if kernel normalization is requested
    if normalize_kernel_flag:
        h = normalize_kernel(h)

    M = int(h.size)
    if M == 1:
        return y.copy()  # normalized h == [1]

    # Padding identical to your convolve() geometry
    w_left = M // 2
    w_right = (M - 1) - w_left
    y_pad = np.pad(y, (w_left, w_right), mode=boundary)

    N = y.size
    Npad = y_pad.size  # N + (M - 1)

    # FFT size for linear convolution equivalence
    L_needed = Npad + M - 1
    L = 1 << int(np.ceil(np.log2(L_needed)))

    # Build spectra
    y_z = np.zeros(L, dtype=float)
    y_z[:Npad] = y_pad
    Y = np.fft.rfft(y_z)

    H = _psf_to_otf_1d(h, L)

    if method == "wiener":
        # Derivative prior: |D(ω)|^2 = (2 sin(ω/2))^2
        k = np.arange(H.size, dtype=float)
        omega = 2.0 * np.pi * k / L
        D2 = (2.0 * np.sin(omega / 2.0)) ** 2

        Hc = np.conjugate(H)
        H2 = (H * Hc).real
        denom = H2 + float(reg) * D2
        # Lock exact DC gain (avoid plateau bias)
        if dc_lock:
            denom[0] = H2[0]  # since D2[0] = 0, this already holds; keep explicit

        G = Hc / denom
    elif method == "fft":
        eps = 1e-12
        G = 1.0 / (H + eps)
    else:
        raise ValueError("Unknown method. Use 'wiener' or 'fft'.")

    # Clamp frequency gain (safety net against spikes)
    if gain_max is not None and gain_max > 0:
        mag = np.abs(G)
        too_big = mag > gain_max
        if np.any(too_big):
            G[too_big] *= gain_max / mag[too_big]

    X = Y * G
    y_true_pad = np.fft.irfft(X, n=L)[:Npad]

    # Extract central segment (same slicing as convolve)
    y_deconv = y_true_pad[w_left : w_left + N]

    # Auto-scale to correct amplitude bias from regularization
    if auto_scale and method == "wiener" and reg > 0:
        # Use energy conservation principle for scaling correction
        # The idea: compare input energy to output energy and adjust

        # Calculate RMS (root mean square) of input and output
        y_rms = np.sqrt(np.mean(y**2)) if len(y) > 0 else 0.0
        y_deconv_rms = np.sqrt(np.mean(y_deconv**2)) if len(y_deconv) > 0 else 0.0

        if y_rms > 1e-12 and y_deconv_rms > 1e-12:
            # Calculate the energy-based scaling factor
            energy_ratio = y_rms / y_deconv_rms

            # Apply scaling if the ratio is reasonable
            # (regularization typically reduces energy)
            if 0.5 < energy_ratio < 5.0:  # Conservative bounds
                y_deconv *= energy_ratio

    return y_deconv


@check_1d_arrays(x_evenly_spaced=True)
def brickwall_filter(
    x: np.ndarray,
    y: np.ndarray,
    mode: Literal["lowpass", "highpass", "bandpass", "bandstop"],
    cut0: float,
    cut1: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply an ideal frequency filter ("brick wall" filter) to a signal.

    Args:
        x: Time domain axis (evenly spaced).
        y: Signal values (same length as x).
        mode: Type of filter to apply.
        cut0: First cutoff frequency.
        cut1: Second cutoff frequency, required for band-pass and band-stop filters.

    Returns:
        Tuple (x, y_filtered), where y_filtered is the filtered signal.

    Raises:
        ValueError: If mode is unknown.
        ValueError: If cut0 is not positive.
        ValueError: If cut1 is missing for band-pass and band-stop filters.
        ValueError: If cut0 > cut1 for band-pass and band-stop filters.
    """
    if mode not in ("lowpass", "highpass", "bandpass", "bandstop"):
        raise ValueError(f"Unknown filter mode: {mode!r}")

    if cut0 <= 0.0:
        raise ValueError("Cutoff frequency must be positive.")

    if mode in ("bandpass", "bandstop"):
        if cut1 is None:
            raise ValueError(f"cut1 must be specified for mode '{mode}'")
        if cut0 > cut1:
            raise ValueError("cut0 must be less than or equal to cut1.")

    freqs, ffty = fft1d(x, y, shift=False)

    if mode == "lowpass":
        frequency_mask = np.abs(freqs) <= cut0
    elif mode == "highpass":
        frequency_mask = np.abs(freqs) >= cut0
    elif mode == "bandpass":
        frequency_mask = (np.abs(freqs) >= cut0) & (np.abs(freqs) <= cut1)
    else:  # bandstop
        frequency_mask = (np.abs(freqs) <= cut0) | (np.abs(freqs) >= cut1)

    ffty_filtered = ffty * frequency_mask
    _, y_filtered = ifft1d(freqs, ffty_filtered)
    y_filtered = y_filtered.real
    return x.copy(), y_filtered
