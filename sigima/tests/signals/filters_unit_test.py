# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Frequency filters unit tests.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

import sigima.proc.signal as sigima_signal
from sigima.objects.signal import SignalObj, create_signal
from sigima.tests import guiutils
from sigima.tests.helpers import check_array_result, check_scalar_result
from sigima.tools.signal.fourier import brickwall_filter


def get_test_function(
    length: int = 2**15,
    freq: int | float | np.ndarray = 1,
    noise_level: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a test 1D signal + high-freq noise.

    Args:
        length: Length of the signal.
        freq: Frequency of the sine wave, can be a single value or an array of
         frequencies
        noise_level: Standard deviation of the Gaussian noise to be added.

    Returns:
        Tuple of (x, y_clean, y_noisy) where:
        - x: The time or spatial domain (1D array).
        - y_clean: The clean sine wave signal.
    """
    x = np.linspace(0, 1, length)
    if np.isscalar(freq):
        y_clean = np.sin(2 * np.pi * freq * x)
    else:
        freq = np.asarray(freq)
        y_clean = np.sum([np.sin(2 * np.pi * f * x) for f in freq], axis=0)
    rng = np.random.default_rng(seed=0)
    y_noisy = y_clean + noise_level * rng.standard_normal(size=length)
    return x, y_clean, y_noisy


def get_test_signals(
    length: int = 2**15,
    freq: int | float | np.ndarray = 1,
    noise_level: float = 0.2,
) -> tuple[SignalObj, SignalObj]:
    """Create a test SignalObj with a sine wave and high-frequency noise.

    Args:
        length: Length of the signal.
        freq: Frequency of the sine wave, can be a single value or an array of
         frequencies.
        noise_level: Standard deviation of the Gaussian noise to be added.

    Returns:
        Tuple of (clean_signal, noisy_signal) where:
        - clean_signal: The clean sine wave signal.
        - noisy_signal: The noisy signal with added Gaussian noise.
    """
    x, y_clean, y_noisy = get_test_function(
        length=length,
        freq=freq,
        noise_level=noise_level,
    )

    noisy = create_signal("noisy signal", x, y_noisy)
    clean = create_signal("clean signal", x, y_clean)
    return clean, noisy


def test_brickwall_filter_lowpass(request: pytest.FixtureRequest | None = None) -> None:
    """Test brickwall_filter as a lowpass filter."""
    x, y_clean, y_noisy = get_test_function()

    # Lowpass: should keep the sine, remove most noise
    x_filt, y_filt = brickwall_filter(x, y_noisy, 2.0, mode="lowpass")

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        test_signal = create_signal("test signal", x, y_clean)
        filt_signal = create_signal("filtered signal", x_filt, y_filt)
        with qt_app_context():
            # Show original and filtered signals
            view_curves([test_signal, filt_signal])

    # Compare filtered signal to clean signal (ignore edges)
    check_array_result(
        "brickwall lowpass noise reduction",
        y_filt[10 : len(y_clean) - 10],
        y_clean[10 : len(y_clean) - 10],
        atol=0.15,
    )


@pytest.mark.validation
def test_signal_lowpass(request: pytest.FixtureRequest | None = None) -> None:
    """Validation test for frequency filtering."""
    clean, noisy = get_test_signals()

    param = sigima_signal.LowPassFilterParam.create(
        method="brickwall",
        cut0=2.0,
        zero_padding=False,
    )

    # Lowpass: should keep the sine, remove most noise
    filt = sigima_signal.lowpass(noisy, param)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            # Show original and filtered signals
            view_curves([clean, filt])

    # Compare filtered signal to clean signal (ignore edges)
    check_array_result(
        "brickwall lowpass noise reduction",
        filt.y[10 : len(clean.y) - 10],
        clean.y[10 : len(clean.y) - 10],
        atol=0.15,
    )


def test_brickwall_filter_highpass(
    request: pytest.FixtureRequest | None = None,
) -> None:
    """Test brickwall_filter as a highpass filter."""
    noise_level = 0.2  # Set noise level for the test
    x, y_clean, y_noisy = get_test_function(noise_level=noise_level)
    test_signal = create_signal("test signal", x, y_clean)
    # Highpass: should remove the sine, keep noise
    x_filt, y_filt = brickwall_filter(x, y_noisy, cut0=2.0, mode="highpass")
    # Only noise should remain, mean close to 0
    filt_signal = create_signal("filtered signal", x_filt, y_filt)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            # Show original and filtered signals
            view_curves([test_signal, filt_signal])

    mean_variance = np.sqrt(noise_level / len(x))
    expected_err = 3 * mean_variance

    check_scalar_result(
        "brickwall highpass removes low freq",
        float(np.mean(y_filt)),
        0,
        atol=expected_err,
    )


@pytest.mark.validation
def test_signal_highpass(request: pytest.FixtureRequest | None = None) -> None:
    """Validation test for highpass frequency filtering."""
    clean, noisy = get_test_signals()
    param = sigima_signal.HighPassFilterParam.create(
        method="brickwall",
        cut0=2.0,
        zero_padding=False,
    )
    filt = sigima_signal.highpass(noisy, param)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            view_curves([clean, filt])

    # The mean of the filtered signal should be close to zero (since only noise remains)
    mean_variance = np.sqrt(0.2 / len(clean.x))
    expected_err = 3 * mean_variance
    check_scalar_result(
        "brickwall highpass removes low freq",
        float(np.mean(filt.y)),
        0,
        atol=expected_err,
    )


def test_brickwall_filter_stopband(request: pytest.FixtureRequest | None = None):
    """Test brickwall_filter as a stopband filter."""
    x, y_clean, _ = get_test_function(freq=np.array([1, 3, 5]), noise_level=0)
    test_signal = create_signal("test signal", x, y_clean)
    # Stopband: remove frequencies between 0.04 and 0.06
    x_filt, y_filt = brickwall_filter(x, y_clean, cut0=4, cut1=2, mode="bandstop")

    filt_signal = create_signal("filtered signal", x_filt, y_filt)

    x_exp, y_exp, _ = get_test_function(freq=np.array([1, 5]), noise_level=0)

    exp_signal = create_signal("expected signal", x_exp, y_exp)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            # Show original and filtered signals
            view_curves([test_signal, filt_signal, exp_signal])

    # The sine should be removed, so result should be close to noise only
    check_array_result(
        "brickwall stopband",
        y_filt[10 : len(y_filt) - 10],
        y_exp[10 : len(y_exp) - 10],
        atol=1e-3,
    )


@pytest.mark.validation
def test_signal_stopband(request: pytest.FixtureRequest | None = None) -> None:
    """Validation test for stopband frequency filtering."""
    x, y_clean, _ = get_test_function(freq=np.array([1, 3, 5]), noise_level=0)
    noisy = create_signal("noisy signal", x, y_clean)
    clean_x, clean_y, _ = get_test_function(freq=np.array([1, 5]), noise_level=0)
    clean = create_signal("clean signal", clean_x, clean_y)

    param = sigima_signal.BandStopFilterParam.create(
        method="brickwall",
        cut0=2.0,
        cut1=4.0,
        zero_padding=False,
    )
    filt = sigima_signal.bandstop(noisy, param)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            view_curves([clean, filt])

    check_array_result(
        "brickwall stopband",
        filt.y[10 : len(clean.y) - 10],
        clean.y[10 : len(clean.y) - 10],
        atol=1e-3,
    )


def test_brickwall_filter_bandpass(request: pytest.FixtureRequest | None = None):
    """Test brickwall_filter as a stopband filter."""
    x, y_clean, _ = get_test_function(freq=np.array([1, 3, 5]), noise_level=0)
    test_signal = create_signal("test signal", x, y_clean)
    # Stopband: remove frequencies between 0.04 and 0.06
    x_filt, y_filt = brickwall_filter(x, y_clean, cut0=2, cut1=4, mode="bandpass")

    filt_signal = create_signal("filtered signal", x_filt, y_filt)

    x_exp, y_exp, _ = get_test_function(freq=np.array([3]), noise_level=0)

    exp_signal = create_signal("expected signal", x_exp, y_exp)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            # Show original and filtered signals
            view_curves([test_signal, filt_signal, exp_signal])

    # The sine should be removed, so result should be close to noise only
    check_array_result(
        "brickwall bandpass ",
        y_filt[10 : len(y_filt) - 10],
        y_exp[10 : len(y_exp) - 10],
        atol=1e-3,
    )


@pytest.mark.validation
def test_signal_bandpass(request: pytest.FixtureRequest | None = None) -> None:
    """Validation test for bandpass frequency filtering."""
    x, y_clean, _ = get_test_function(freq=np.array([1, 3, 5]), noise_level=0)
    noisy = create_signal("noisy signal", x, y_clean)
    clean_x, clean_y, _ = get_test_function(freq=np.array([3]), noise_level=0)
    clean = create_signal("clean signal", clean_x, clean_y)
    param = sigima_signal.BandPassFilterParam.create(
        method="brickwall",
        cut0=2.0,
        cut1=4.0,
        zero_padding=False,
    )
    filt = sigima_signal.bandpass(noisy, param)

    if guiutils.is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            view_curves([clean, filt])

    check_array_result(
        "brickwall bandpass",
        filt.y[10 : len(clean.y) - 10],
        clean.y[10 : len(clean.y) - 10],
        atol=1e-3,
    )


def test_brickwall_filter_invalid_x():
    """Test brickwall_filter raises on non-uniform x."""
    x, _, y_noisy = get_test_function()
    x_bad = x.copy()
    x_bad[5] += 0.01  # break uniformity
    with pytest.raises(ValueError, match="uniformly spaced"):
        brickwall_filter(x_bad, y_noisy, cut0=0.1)
        raise AssertionError("Expected ValueError for non-uniform x")


if __name__ == "__main__":
    test_brickwall_filter_lowpass()
    test_signal_lowpass()
    test_brickwall_filter_highpass()
    test_signal_highpass()
    test_brickwall_filter_stopband()
    test_signal_stopband()
    test_brickwall_filter_bandpass()
    test_signal_bandpass()
    test_brickwall_filter_invalid_x()
