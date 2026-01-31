# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal FFT unit test."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps

import sigima.objects
import sigima.params
import sigima.proc.signal
import sigima.tests.data
from sigima.enums import PadLocation1D
from sigima.tests import guiutils
from sigima.tests.data import get_test_signal
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result, check_scalar_result
from sigima.tools.signal import fourier


@pytest.mark.validation
def test_signal_zero_padding() -> None:
    """1D FFT zero padding validation test."""
    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINE, freq=50.0, size=1000
    )

    # Dumb check to ensure that ZeroPadding1DParam won't raise an exception if the
    # `strategy_callback` is used before calling `update_from_obj` (this could happen
    # if the parameter is used in a GUI before being applied to a signal):
    param = sigima.params.ZeroPadding1DParam.create(strategy="next_pow2")
    param.n = 1  # Already the default value, but just to be explicit here
    param.strategy_callback(None, "")  # Nothing should happen here
    assert param.n == 1, "Padding length should remain unchanged"

    # Validate padding length computation
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

    # Validate zero padding with custom strategy
    param = sigima.params.ZeroPadding1DParam.create(strategy="custom", n=250)
    assert param.n is not None
    for location in PadLocation1D:
        execenv.print(f"Validating zero padding with location = {location.value}...")
        param.location = location
        param.update_from_obj(s1)
        s2 = sigima.proc.signal.zero_padding(s1, param)
        len1 = s1.y.size
        n = param.n
        exp_len2 = len1 + n
        assert s2.y.size == exp_len2, f"Wrong length: {len(s2.y)} (expected {exp_len2})"
        if location == PadLocation1D.APPEND:
            dx = s1.x[1] - s1.x[0]
            expected_x = np.pad(
                s1.x,
                (0, n),
                mode="linear_ramp",
                end_values=(s1.x[-1] + dx * n,),
            )
            check_array_result(f"{location.value}: Check x-data", s2.x, expected_x)
            check_array_result(
                f"{location.value}: Check original y-data", s2.y[:len1], s1.y
            )
            check_array_result(
                f"{location.value}: Check padded y-data", s2.y[len1:], np.zeros(n)
            )
        elif location == PadLocation1D.PREPEND:
            dx = s1.x[1] - s1.x[0]
            expected_x = np.pad(
                s1.x,
                (n, 0),
                mode="linear_ramp",
                end_values=(s1.x[0] - dx * n,),
            )
            check_array_result(f"{location.value}: Check x-data", s2.x, expected_x)
            check_array_result(
                f"{location.value}: Check original y-data", s2.y[-len1:], s1.y
            )
            check_array_result(
                f"{location.value}: Check padded y-data", s2.y[:n], np.zeros(n)
            )
        elif location == PadLocation1D.BOTH:
            dx = s1.x[1] - s1.x[0]
            expected_x = np.pad(
                s1.x,
                (n // 2, n - n // 2),
                mode="linear_ramp",
                end_values=(
                    s1.x[0] - dx * (n // 2),
                    s1.x[-1] + dx * (n - n // 2),
                ),
            )
            check_array_result(f"{location.value}: Check x-data", s2.x, expected_x)
            check_array_result(
                f"{location.value}: Check original y-data",
                s2.y[n // 2 : n // 2 + len1],
                s1.y,
            )
            check_array_result(
                f"{location.value}: Check padded y-data (before)",
                s2.y[: n // 2],
                np.zeros(n // 2),
            )
            check_array_result(
                f"{location.value}: Check padded y-data (after)",
                s2.y[-(n - n // 2) :],
                np.zeros(n - n // 2),
            )
        execenv.print("OK")

    # Validate zero padding with other strategies
    for strategy in sigima.params.ZeroPadding1DParam.strategies:
        if strategy == "custom":
            continue  # Already tested above
        param = sigima.params.ZeroPadding1DParam.create(strategy=strategy)
        param.update_from_obj(s1)
        s2 = sigima.proc.signal.zero_padding(s1, param)
        len1 = s1.y.size
        n = param.n
        exp_len2 = len1 + n
        assert s2.y.size == exp_len2, f"Wrong length: {len(s2.y)} (expected {exp_len2})"


@pytest.mark.validation
def test_signal_fft() -> None:
    """1D FFT validation test."""
    freq = 50.0
    size = 10000

    # See note in function `test_signal_ifft` below.
    xmin = 0.0

    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINE, freq=freq, size=size, xmin=xmin
    )
    fft = sigima.proc.signal.fft(s1)
    ifft = sigima.proc.signal.ifft(fft)

    # Check that the inverse FFT reconstructs the original signal.
    check_array_result("Original and recovered x data", s1.y, ifft.y.real)
    check_array_result("Original and recovered y data", s1.x, ifft.x.real)


@pytest.mark.skip(reason="Already covered by the `test_signal_fft` test.")
@pytest.mark.validation
def test_signal_ifft() -> None:
    """1D iFFT validation test."""
    # This is just a way of marking the iFFT test as a validation test because it is
    # already covered by the FFT test above (there is no need to repeat the same test).
    # The tested function is :py:func:`sigima.proc.signal.ifft`.


def test_fft_ifft_tools() -> None:
    """Fourier 1D FFT/iFFT tools unit test."""
    param = sigima.objects.CosineParam.create(size=500)

    # *** Note ***
    #
    # We set xmin to 0.0 to be able to compare the X data of the original and
    # reconstructed signals, because the FFT do not preserve the X data (phase is
    # lost, sampling rate is assumed to be constant), so that comparing the X data
    # is not meaningful if xmin is different.
    param.xmin = 0.0

    s1 = sigima.objects.create_signal_from_param(param)
    assert s1.xydata is not None
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

        guiutils.view_curves_if_gui(
            [
                s1,
                sigima.objects.create_signal("Recovered", t2, y2),
                sigima.objects.create_signal("Difference", t1, np.abs(y2 - y1)),
            ]
        )


@pytest.mark.validation
def test_signal_magnitude_spectrum() -> None:
    """1D magnitude spectrum validation test."""
    freq = 50.0
    size = 10000

    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINE, freq=freq, size=size
    )
    fft = sigima.proc.signal.fft(s1)
    mag = sigima.proc.signal.magnitude_spectrum(s1)

    # Check that the peak frequencies are correct.
    ipk1 = np.argmax(mag.y[: size // 2])
    ipk2 = np.argmax(mag.y[size // 2 :]) + size // 2
    fpk1 = fft.x[ipk1]
    fpk2 = fft.x[ipk2]
    check_scalar_result("Frequency of the first peak", fpk1, -freq, rtol=1e-4)
    check_scalar_result("Frequency of the second peak", fpk2, freq, rtol=1e-4)

    # Check that magnitude spectrum is symmetric.
    check_array_result("Symmetry of magnitude spectrum", mag.y[1::], mag.y[-1:0:-1])

    # Check the magnitude of the peaks.
    exp_mag = size / 2
    check_scalar_result("Magnitude of the first peak", mag.y[ipk1], exp_mag, rtol=0.05)
    check_scalar_result("Magnitude of the second peak", mag.y[ipk2], exp_mag, rtol=0.05)

    # Check that the magnitude spectrum is correct.
    check_array_result("Cosine signal magnitude spectrum X", mag.x, fft.x.real)
    check_array_result("Cosine signal magnitude spectrum Y", mag.y, np.abs(fft.y))

    guiutils.view_curves_if_gui(
        [
            sigima.objects.create_signal("FFT-real", fft.x.real, fft.x.real),
            sigima.objects.create_signal("FFT-imag", fft.x.real, fft.y.imag),
            sigima.objects.create_signal("FFT-magnitude", mag.x.real, mag.y),
        ]
    )


@pytest.mark.validation
def test_signal_phase_spectrum() -> None:
    """1D phase spectrum validation test."""
    freq = 50.0
    size = 10000

    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINE, freq=freq, size=size
    )
    fft = sigima.proc.signal.fft(s1)
    phase = sigima.proc.signal.phase_spectrum(s1)

    # Check that the phase spectrum is correct.
    check_array_result("Cosine signal phase spectrum X", phase.x, fft.x.real)
    exp_phase = np.rad2deg(np.angle(fft.y))
    check_array_result("Cosine signal phase spectrum Y", phase.y, exp_phase)

    guiutils.view_curves_if_gui(
        [
            sigima.objects.create_signal("FFT-real", fft.x.real, fft.x.real),
            sigima.objects.create_signal("FFT-imag", fft.x.real, fft.y.imag),
            sigima.objects.create_signal("Phase", phase.x.real, phase.y),
        ]
    )


@pytest.mark.validation
def test_signal_psd() -> None:
    """1D Power Spectral Density validation test."""
    freq = 50.0
    size = 10000

    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINE, freq=freq, size=size
    )
    param = sigima.params.SpectrumParam()
    for decibel in (False, True):
        param.decibel = decibel
        psd = sigima.proc.signal.psd(s1, param)

        # Check that the PSD is correct.
        exp_x, exp_y = sps.welch(s1.y, fs=1.0 / (s1.x[1] - s1.x[0]))
        if decibel:
            exp_y = 10 * np.log10(exp_y)

        fpk1 = psd.x[np.argmax(psd.y)]
        check_scalar_result("Frequency of the maximum", fpk1, freq, rtol=2e-2)

        check_array_result(f"Cosine signal PSD X (dB={decibel})", psd.x, exp_x)
        check_array_result(f"Cosine signal PSD Y (dB={decibel})", psd.y, exp_y)

        guiutils.view_curves_if_gui(
            [
                sigima.objects.create_signal("PSD", psd.x, psd.y),
            ]
        )


@pytest.mark.gui
def test_signal_spectrum() -> None:
    """Test several FFT-related functions on `dynamic_parameters.txt`."""
    with guiutils.lazy_qt_app_context(force=True):
        # pylint: disable=import-outside-toplevel
        from sigima.viz import view_curves

        sig = get_test_signal("dynamic_parameters.txt")
        view_curves([sig])
        p = sigima.params.SpectrumParam.create(decibel=True)
        ms = sigima.proc.signal.magnitude_spectrum(sig, p)
        view_curves([ms], title="Magnitude spectrum")
        ps = sigima.proc.signal.phase_spectrum(sig)
        view_curves([ps], title="Phase spectrum")
        psd = sigima.proc.signal.psd(sig, p)
        view_curves([psd], title="Power spectral density")


if __name__ == "__main__":
    guiutils.enable_gui()
    test_signal_zero_padding()
    test_signal_fft()
    test_fft_ifft_tools()
    test_signal_magnitude_spectrum()
    test_signal_phase_spectrum()
    test_signal_psd()
    test_signal_spectrum()
