# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for full width computing features
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
import sigima.tests.data
import sigima.tests.helpers
from sigima.tests import guiutils
from sigima.tests.env import execenv
from sigima.tools.signal import pulse


def __test_fwhm_interactive(obj: sigima.objects.SignalObj, method: str) -> None:
    """Interactive test for the full width at half maximum computation."""
    # pylint: disable=import-outside-toplevel
    from sigima import viz

    param = sigima.params.FWHMParam.create(method=method)
    geometry = sigima.proc.signal.fwhm(obj, param)
    x0, y0, x1, y1 = geometry.coords[0]
    x, y = obj.xydata
    viz.view_curve_items(
        [
            viz.create_curve(x.real, y.real, title=obj.title),
            viz.create_segment(x0, y0, x1, y1, "FWHM"),
        ],
        title=f"FWHM [{method}]",
    )


@pytest.mark.gui
def test_signal_fwhm_interactive() -> None:
    """FWHM interactive test."""
    with guiutils.lazy_qt_app_context(force=True):
        execenv.print("Computing FWHM of a multi-peak signal:")
        obj1 = sigima.tests.data.create_paracetamol_signal()
        p = sigima.objects.NormalDistribution1DParam.create(sigma=0.05)
        obj2 = sigima.tests.data.create_noisy_signal(p)
        for method, _mname in sigima.params.FWHMParam.methods:
            execenv.print(f"  Method: {method}")
            for obj in (obj1, obj2):
                if method == "zero-crossing":
                    # Check that a warning is raised when using the zero-crossing method
                    with pytest.warns(UserWarning):
                        __test_fwhm_interactive(obj, method)
                else:
                    __test_fwhm_interactive(obj, method)


@pytest.mark.validation
def test_signal_fwhm() -> None:
    """Validation test for the full width at half maximum computation.

    Tests FWHM computation on:
    1. Real signal data (fwhm.txt) - validates against manual measurement
    2. Synthetic Gaussian signals - validates against theoretical values
    3. Multi-peak signal - validates warning behavior
    4. User-reported signal (fwhm.csv) - non-regression guard against past regressions
    """
    # Test 1: Real signal data (original validation test)
    obj = sigima.tests.data.get_test_signal("fwhm.txt")
    real_fwhm = 2.675  # Manual validation
    for method, exp in (
        ("gauss", 2.40323),
        ("lorentz", 2.78072),
        ("voigt", 2.56591),
        ("zero-crossing", real_fwhm),
    ):
        param = sigima.params.FWHMParam.create(method=method)
        geometry = sigima.proc.signal.fwhm(obj, param)
        length = geometry.segments_lengths()[0]
        sigima.tests.helpers.check_scalar_result(
            f"FWHM[{method}]", length, exp, rtol=0.05
        )

    # Test 2: Synthetic Gaussian signals - systematic offset investigation
    execenv.print("\n  FWHM Gaussian validation (theoretical comparison):")
    sigma_values = [1.0, 2.0]  # Test two sigma values

    for sigma in sigma_values:
        # Create Gaussian signal with known parameters
        gauss_param = sigima.objects.GaussParam()
        gauss_param.size = 1000
        gauss_param.xmin = -10.0
        gauss_param.xmax = 10.0
        gauss_param.sigma = sigma
        gauss_param.mu = 0.0
        gauss_param.amplitude = 1.0
        gauss_param.y0 = 0.0

        sig = sigima.objects.create_signal_from_param(gauss_param)

        # Theoretical FWHM for Gaussian: FWHM = 2 * sigma * sqrt(2 * ln(2))
        theoretical_fwhm = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))

        # Test Gaussian fit method (should be most accurate)
        fwhm_param = sigima.params.FWHMParam.create(method="gauss")
        geometry = sigima.proc.signal.fwhm(sig, fwhm_param)
        computed_fwhm = geometry.segments_lengths()[0]

        execenv.print(
            f"    σ={sigma}: Theoretical={theoretical_fwhm:.6f}, "
            f"Computed={computed_fwhm:.6f}, "
            f"Offset={(computed_fwhm - theoretical_fwhm):.6f}"
        )

        # Gaussian fit should match theoretical value very closely
        sigima.tests.helpers.check_scalar_result(
            f"FWHM[gauss, σ={sigma}]",
            computed_fwhm,
            theoretical_fwhm,
            rtol=0.01,  # 1% tolerance
        )

    # Test 3: Multi-peak signal warning
    obj = sigima.tests.data.create_paracetamol_signal()
    with pytest.warns(UserWarning):
        sigima.proc.signal.fwhm(
            obj, sigima.params.FWHMParam.create(method="zero-crossing")
        )

    # Test 4: Non-regression guard on a user-reported signal.
    # DataLab v0.20.1 returned an aberrant (far too small) width on this signal, while
    # v0.18 and v1.2.1 returned the correct one. See DataLab-Platform/DataLab#356.
    # The expected value below is the ground truth: it must not drift.
    obj = sigima.tests.data.get_test_signal("fwhm.csv")
    param = sigima.params.FWHMParam.create(method="zero-crossing")
    # This signal has several crossing points, hence the expected warnings:
    with pytest.warns(UserWarning):
        geometry = sigima.proc.signal.fwhm(obj, param)
    length = geometry.segments_lengths()[0]
    sigima.tests.helpers.check_scalar_result(
        "FWHM[zero-crossing, fwhm.csv]", length, 1.75636, rtol=1e-4
    )


@pytest.mark.validation
def test_signal_fw1e2() -> None:
    """Validation test for the full width at 1/e^2 maximum computation."""
    obj = sigima.tests.data.get_test_signal("fw1e2.txt")
    exp = 4.06  # Manual validation
    geometry = sigima.proc.signal.fw1e2(obj)
    length = geometry.segments_lengths()[0]
    sigima.tests.helpers.check_scalar_result("FW1E2", length, exp, rtol=0.005)


def __square_pulse(baseline: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a 4-unit wide square pulse of signed `height` above `baseline`."""
    x = np.linspace(0.0, 10.0, 1001)
    y = np.full_like(x, baseline)
    y[(x >= 3.0) & (x <= 7.0)] = baseline + height
    return x, y


@pytest.mark.parametrize("baseline", [0.0, 5.0])
@pytest.mark.parametrize("height", [2.0, -2.0])
def test_full_width_at_ratio_polarity_and_baseline(
    baseline: float, height: float
) -> None:
    """Validation test for `full_width_at_ratio` on shifted and inverted pulses.

    The half-maximum width of a square pulse depends neither on its polarity nor
    on its baseline, and the returned level is always `baseline + height / 2`.
    """
    x, y = __square_pulse(baseline, height)
    x1, level1, x2, level2 = pulse.full_width_at_ratio(x, y, 0.5)
    assert level1 == level2
    tag = f"[baseline={baseline:g},height={height:g}]"
    sigima.tests.helpers.check_scalar_result(
        f"level{tag}", level1, baseline + height / 2, rtol=0.01
    )
    sigima.tests.helpers.check_scalar_result(f"width{tag}", x2 - x1, 4.0, rtol=0.01)


@pytest.mark.validation
def test_signal_full_width_at_y() -> None:
    """Validation test for the full width at y computation."""
    obj = sigima.tests.data.get_test_signal("fwhm.txt")
    real_fwhm = 2.675  # Manual validation
    param = sigima.params.OrdinateParam.create(y=0.5)
    geometry = sigima.proc.signal.full_width_at_y(obj, param)
    length = geometry.segments_lengths()[0]
    sigima.tests.helpers.check_scalar_result("∆X", length, real_fwhm, rtol=0.05)


if __name__ == "__main__":
    test_signal_fwhm_interactive()
    test_signal_fwhm()
    test_signal_fw1e2()
    for __baseline in (0.0, 5.0):
        for __height in (2.0, -2.0):
            test_full_width_at_ratio_polarity_and_baseline(__baseline, __height)
    test_signal_full_width_at_y()
