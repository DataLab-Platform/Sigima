# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal FFT unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps

import sigima.objects
import sigima.params
import sigima.proc.signal as sigima_signal
import sigima.tests.data as ctd
from sigima.tests import guiutils
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result, check_scalar_result
from sigima.tools.signal import fourier


@pytest.mark.validation
def test_signal_zero_padding() -> None:
    """1D FFT zero padding validation test."""
    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=50.0, size=1000
    )

    # Validate zero padding with custom length
    param = sigima.params.ZeroPadding1DParam.create(n=250)
    assert param.strategy == "custom", (
        f"Wrong default strategy: {param.strategy} (expected 'custom')"
    )
    s2 = sigima_signal.zero_padding(s1, param)
    len1 = len(s1.y)
    exp_len2 = len1 + param.n
    execenv.print("Validating zero padding with custom length...", end=" ")
    assert len(s2.y) == exp_len2, f"Wrong length: {len(s2.y)} (expected {exp_len2})"
    assert np.all(s2.x[:len1] == s1.x[:len1]), "Altered X data in original signal area"
    assert np.all(s2.y[:len1] == s1.y[:len1]), "Altered Y data in original signal area"
    assert np.all(s2.y[len1:] == 0), "Non-zero data in zero-padded area"
    execenv.print("OK")
    step1 = s1.x[1] - s1.x[0]
    check_array_result(
        "Zero padding X data",
        s2.x[len1:],
        np.arange(
            s1.x[len1 - 1] + step1, s1.x[len1 - 1] + step1 * (param.n + 1), step1
        ),
    )

    # Validate zero padding with strategies other than custom length
    for strategy, expected_length in (
        ("next_pow2", 24),
        ("double", 1000),
        ("triple", 2000),
    ):
        param = sigima.params.ZeroPadding1DParam.create(strategy=strategy)
        param.update_from_obj(s1)
        assert param.n == expected_length, (
            f"Wrong length for '{param.strategy}' strategy: {param.n}"
            f" (expected {expected_length})"
        )


@pytest.mark.validation
def test_signal_fft() -> None:
    """1D FFT validation test."""
    freq = 50.0
    size = 10000

    # See note in the interactive test above
    xmin = 0.0

    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=freq, size=size, xmin=xmin
    )
    fft = sigima_signal.fft(s1)
    ifft = sigima_signal.ifft(fft)

    # Check that the inverse FFT reconstructs the original signal
    check_array_result("Cosine signal FFT/iFFT X reconstruction", s1.y, ifft.y.real)
    check_array_result("Cosine signal FFT/iFFT Y reconstruction", s1.x, ifft.x.real)

    # Check FFT properties
    mag = np.abs(fft.y)

    # Find the peak in the FFT
    ipk1, ipk2 = np.argmax(mag[: size // 2]), np.argmax(mag[size // 2 :]) + size // 2
    fpk1, fpk2 = fft.x[ipk1], fft.x[ipk2]

    # Verify the peak frequencies are correct
    check_scalar_result("Cosine negative frequency", fpk1, -freq, rtol=0.001)
    check_scalar_result("Cosine positive frequency", fpk2, freq, rtol=0.001)

    # Verify the magnitude at the peak
    exp_mag = size / 2
    check_scalar_result("Cosine peak magnitude", mag[ipk1], exp_mag, rtol=0.05)

    # Verify the symmetry of the FFT
    check_array_result(
        "FFT symmetry",
        mag[1 : size // 2],
        mag[1 + size // 2 :][::-1],
    )


@pytest.mark.validation
def test_signal_ifft(request: pytest.FixtureRequest | None = None) -> None:
    """1D iFFT validation test."""
    # To validate the iFFT, we check that the original signal and the
    # reconstructed signal are equal. We need to set the request
    # to enable the GUI if needed, so that the test can be run interactively.

    guiutils.set_current_request(request)

    newparam = sigima.objects.NewSignalParam.create(
        stype=sigima.objects.SignalTypes.COSINUS, size=500
    )

    # *** Note ***
    #
    # We set xmin to 0.0 to be able to compare the X data of the original and
    # reconstructed signals, because the FFT do not preserve the X data (phase is
    # lost, sampling rate is assumed to be constant), so that comparing the X data
    # is not meaningful if xmin is different.
    newparam.xmin = 0.0

    extra_param = sigima.objects.PeriodicParam()
    s1 = sigima.objects.create_signal_from_param(newparam, extra_param=extra_param)
    t1, y1 = s1.xydata
    for shift in (True, False):
        f1, sp1 = fourier.fft1d(t1, y1, shift=shift)
        t2, y2 = fourier.ifft1d(f1, sp1)

        execenv.print(
            f"Comparing original and recovered signals for `shift={shift}`...",
            end=" ",
        )
        check_array_result("Original and recovered x data", t2, t1, verbose=False)
        check_array_result("Original and recovered y data", y2, y1, verbose=False)
        execenv.print("OK")

        if guiutils.is_gui_enabled():
            # pylint: disable=import-outside-toplevel
            from guidata.qthelpers import qt_app_context

            from sigima.tests.vistools import view_curves

            with qt_app_context():
                view_curves(
                    [
                        sigima.objects.create_signal("Original", t1, y1),
                        sigima.objects.create_signal("Recovered", t2, y2),
                    ]
                )


@pytest.mark.validation
def test_signal_magnitude_spectrum() -> None:
    """1D magnitude spectrum validation test."""
    freq = 50.0
    size = 10000

    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=freq, size=size
    )
    fft = sigima_signal.fft(s1)
    mag = sigima_signal.magnitude_spectrum(s1)
    fpk1 = fft.x[np.argmax(mag.y[: size // 2])]
    check_scalar_result("Cosine negative frequency", fpk1, -freq, rtol=0.001)

    # Check that the magnitude spectrum is correct
    check_array_result("Cosine signal magnitude spectrum X", mag.x, fft.x.real)
    check_array_result("Cosine signal magnitude spectrum Y", mag.y, np.abs(fft.y))


@pytest.mark.validation
def test_signal_phase_spectrum() -> None:
    """1D phase spectrum validation test."""
    freq = 50.0
    size = 10000

    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=freq, size=size
    )
    fft = sigima_signal.fft(s1)
    phase = sigima_signal.phase_spectrum(s1)
    fpk1 = fft.x[np.argmax(phase.y[: size // 2])]
    check_scalar_result("Cosine negative frequency", fpk1, -freq, rtol=0.001)

    # Check that the phase spectrum is correct
    check_array_result("Cosine signal phase spectrum X", phase.x, fft.x.real)
    exp_phase = np.rad2deg(np.angle(fft.y))
    check_array_result("Cosine signal phase spectrum Y", phase.y, exp_phase)


@pytest.mark.validation
def test_signal_psd() -> None:
    """1D Power Spectral Density validation test."""
    freq = 50.0
    size = 10000

    s1 = ctd.create_periodic_signal(
        sigima.objects.SignalTypes.COSINUS, freq=freq, size=size
    )
    param = sigima.params.SpectrumParam()
    for log_scale in (False, True):
        param.log = log_scale
        psd = sigima_signal.psd(s1, param)

        # Check that the PSD is correct (Welch's method is used by default)
        exp_x, exp_y = sps.welch(s1.y, fs=1.0 / (s1.x[1] - s1.x[0]))

        if log_scale:
            exp_y = 10 * np.log10(exp_y)

        check_array_result(f"Cosine signal PSD X (log={log_scale})", psd.x, exp_x)
        check_array_result(f"Cosine signal PSD Y (log={log_scale})", psd.y, exp_y)


if __name__ == "__main__":
    test_signal_zero_padding()
    test_signal_fft()
    test_signal_ifft(request=guiutils.DummyRequest(gui=True))
    test_signal_magnitude_spectrum()
    test_signal_phase_spectrum()
    test_signal_psd()
