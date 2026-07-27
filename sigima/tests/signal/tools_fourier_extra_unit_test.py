# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :mod:`sigima.tools.signal.fourier`.

Covers input validation paths for ``zero_padding``, ``ifft1d`` and
``brickwall_filter``, the decibel branches of ``magnitude_spectrum`` and
``psd``, the unsorted-frequency branch of ``ifft1d``, and validation /
options of ``deconvolve`` (Wiener with ``dc_lock`` / ``gain_max`` /
``auto_scale``, and the FFT method).
"""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

from sigima.tools.signal import fourier as f_mod

# ===========================================================================
# zero_padding / magnitude_spectrum / psd / ifft1d / brickwall_filter
# ===========================================================================


def test_zero_padding_validation_errors() -> None:
    """Negative ``n_prepend`` / ``n_append`` arguments to ``zero_padding``
    are rejected with ``ValueError``."""
    x = np.linspace(0.0, 1.0, 8)
    y = np.sin(x)
    with pytest.raises(ValueError):
        f_mod.zero_padding(x, y, n_prepend=-1)
    with pytest.raises(ValueError):
        f_mod.zero_padding(x, y, n_append=-1)


def test_magnitude_spectrum_decibel_branch() -> None:
    """With ``decibel=True`` the magnitude spectrum is expressed in dB
    (so it must contain at least one negative value for any non-flat
    input)."""
    x = np.linspace(0.0, 1.0, 64)
    y = np.sin(2.0 * np.pi * 5.0 * x)
    f, mag_db = f_mod.magnitude_spectrum(x, y, decibel=True)
    assert f.shape == mag_db.shape
    assert np.any(mag_db < 0)


def test_psd_decibel_branch() -> None:
    """With ``decibel=True`` the PSD output array has the same shape as
    the frequency axis (basic dB-branch coverage)."""
    x = np.linspace(0.0, 1.0, 256)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(256)
    f, p_db = f_mod.psd(x, y, decibel=True)
    assert f.shape == p_db.shape


def test_decibel_conversion_is_finite_on_null_spectral_lines() -> None:
    """A spectrum with exactly zero lines must stay finite in dB.

    A constant signal has zero content at every non-zero frequency, so the naive
    ``log10`` yields ``-inf`` samples that propagate to autoscaling and export."""
    x = np.linspace(0.0, 1.0, 64)
    y = np.ones_like(x)
    _f, mag_db = f_mod.magnitude_spectrum(x, y, decibel=True)
    assert np.all(np.isfinite(mag_db))
    # The floor must stay far below the peak, so it cannot be mistaken for signal:
    assert np.min(mag_db) < np.max(mag_db) - 300.0


def test_decibel_conversion_of_identically_zero_spectrum() -> None:
    """An identically zero signal has no dynamic range: -inf is the honest answer."""
    x = np.linspace(0.0, 1.0, 64)
    y = np.zeros_like(x)
    _f, mag_db = f_mod.magnitude_spectrum(x, y, decibel=True)
    assert np.all(np.isneginf(mag_db))


def test_ifft1d_validation_errors() -> None:
    """``ifft1d`` rejects single-sample inputs and all-zero spectra with
    ``ValueError`` (degenerate cases)."""
    with pytest.raises(ValueError):
        f_mod.ifft1d(np.array([0.0]), np.array([1.0 + 0j]))
    f_arr = np.array([0.0, 1.0, 3.0, 4.0])
    sp = np.zeros_like(f_arr, dtype=np.complex128)
    with pytest.raises(ValueError):
        f_mod.ifft1d(f_arr, sp)


def test_ifft1d_unsorted_frequencies_branch() -> None:
    """Unsorted (already-shifted) frequency input goes through fftshift branch."""
    n = 32
    x = np.linspace(0.0, 1.0, n)
    y = np.cos(2.0 * np.pi * 4.0 * x)
    f_unshift = np.fft.fftfreq(n, d=x[1] - x[0])
    sp_unshift = np.fft.fft(y)
    x_back, y_back = f_mod.ifft1d(f_unshift, sp_unshift)
    assert x_back.shape == y_back.shape == (n,)


def test_brickwall_filter_errors() -> None:
    """``brickwall_filter`` rejects unknown modes, zero cut-off frequencies
    and inverted bandpass bounds."""
    x = np.linspace(0.0, 1.0, 32)
    y = np.zeros_like(x)
    with pytest.raises(ValueError):
        f_mod.brickwall_filter(x, y, mode="weird", cut0=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        f_mod.brickwall_filter(x, y, mode="lowpass", cut0=0.0)
    with pytest.raises(ValueError):
        f_mod.brickwall_filter(x, y, mode="bandpass", cut0=1.0, cut1=None)
    with pytest.raises(ValueError):
        f_mod.brickwall_filter(x, y, mode="bandpass", cut0=2.0, cut1=1.0)


# ===========================================================================
# deconvolve - validation + advanced options
# ===========================================================================


def test_deconvolve_validation_errors() -> None:
    """``deconvolve`` rejects 2D inputs, mismatched signal/kernel sizes
    and an all-zero kernel."""
    x = np.linspace(0.0, 1.0, 16)
    y = np.zeros_like(x)
    h = np.ones(16)
    with pytest.raises(ValueError):
        f_mod.deconvolve(x.reshape(-1, 1), y, h)
    with pytest.raises(ValueError):
        f_mod.deconvolve(x, y, np.ones(8))
    with pytest.raises(ValueError):
        f_mod.deconvolve(x, y, np.zeros(16))


def test_deconvolve_empty_arrays_raise() -> None:
    """Empty arrays cannot be deconvolved: ``ValueError`` is raised."""
    with pytest.raises(ValueError):
        f_mod.deconvolve(np.array([]), np.array([]), np.array([]))


def test_deconvolve_unknown_method_raises() -> None:
    """Unknown ``method`` strings are rejected (only ``"wiener"`` and
    ``"fft"`` are supported)."""
    n = 32
    x = np.linspace(0.0, 1.0, n)
    y = np.ones(n)
    h = np.zeros(n)
    h[n // 2] = 1.0
    with pytest.raises(ValueError):
        f_mod.deconvolve(x, y, h, method="bogus")  # type: ignore[arg-type]


def _padded_kernel(n_total: int, sigma: float) -> np.ndarray:
    """Build a centered Gaussian kernel of total length n_total."""
    k = np.arange(n_total) - (n_total - 1) / 2.0
    g = np.exp(-(k**2) / (2.0 * sigma**2))
    return g / g.sum()


def test_deconvolve_wiener_with_options() -> None:
    """Wiener deconvolution with all advanced options enabled (``reg``,
    ``gain_max``, ``dc_lock``, ``auto_scale``) returns a finite array of
    the input shape."""
    n = 64
    x = np.linspace(0.0, 1.0, n)
    base = np.exp(-((np.arange(n) - n / 2) ** 2) / 50.0)
    h = _padded_kernel(n, 1.5)
    y = np.convolve(base, h, mode="same")
    out = f_mod.deconvolve(
        x,
        y,
        h,
        method="wiener",
        reg=1e-3,
        gain_max=10.0,
        dc_lock=True,
        auto_scale=True,
    )
    assert out.shape == y.shape
    assert np.all(np.isfinite(out))


def test_deconvolve_fft_method() -> None:
    """Pure FFT-method deconvolution returns an array of the input shape
    (smoke test for the ``method='fft'`` branch)."""
    n = 64
    x = np.linspace(0.0, 1.0, n)
    base = np.exp(-((np.arange(n) - n / 2) ** 2) / 50.0)
    h = _padded_kernel(n, 1.0)
    y = np.convolve(base, h, mode="same")
    out = f_mod.deconvolve(x, y, h, method="fft")
    assert out.shape == y.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
