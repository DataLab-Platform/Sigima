# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Replace X by other signal's Y operation unit test."""

import numpy as np
import pytest

import sigima.proc.signal
from sigima.objects import create_signal
from sigima.tests.helpers import check_array_result


def test_replace_x_by_other_y_size_mismatch():
    """Test that Replace X by other signal's Y raises error
    when signals have different sizes."""
    x1 = np.arange(5)
    y1 = np.array([10, 20, 30, 40, 50])
    sig1 = create_signal("Signal 1", x1, y1)

    x2 = np.arange(3)
    y2 = np.array([400, 450, 500])
    sig2 = create_signal("Signal 2", x2, y2)

    with pytest.raises(ValueError, match="same number of points"):
        sigima.proc.signal.replace_x_by_other_y(sig1, sig2)


@pytest.mark.validation
def test_replace_x_by_other_y():
    """Test realistic wavelength calibration scenario."""
    # Use case (spectroscopy): in many spectroscopic instruments, each acquired signal
    # comes with its own X axis, which may represent internal, non-physical sampling
    # coordinates (e.g., pixel index transformed by optics, scan time,
    # or module-specific sampling grids).
    # Only one array – usually provided as the first signal – contains the calibrated
    # wavelength axis. Although all signals share the same number of samples,
    # their X axes are not physically meaningful for analysis.
    # The correct operation is therefore to discard the X axes of signals 2..N
    # and re-express each signal as a function of the wavelength array from signal 1,
    # without interpolation.

    # Simulated measurement indices
    n_points = 100
    indices = np.arange(n_points)

    # Wavelength calibration (e.g., from spectrometer calibration file)
    # Linear wavelength scale: 400nm to 800nm
    wavelengths_x = indices * 0.2  # Arbitrary physical units for X axis
    wavelengths_y = np.linspace(400, 800, n_points)
    wavelength_signal = create_signal("λ calibration", wavelengths_x, wavelengths_y)
    wavelength_signal.ylabel = "λ"
    wavelength_signal.yunit = "nm"

    # Measurement data (e.g., spectral intensity)
    # Gaussian peak centered around index 50
    intensity_x = indices * 0.3  # Arbitrary physical units for X axis
    intensity_y = 100 * np.exp(-((indices - 50) ** 2) / (2 * 10**2))
    intensity_signal = create_signal("Intensity", intensity_x, intensity_y)
    intensity_signal.ylabel = "I"
    intensity_signal.yunit = "counts"

    # Create calibrated spectrum: intensity vs wavelength
    calibrated = sigima.proc.signal.replace_x_by_other_y(
        intensity_signal, wavelength_signal
    )

    # Verify
    check_array_result("Calibrated X values", calibrated.x, wavelengths_y)
    check_array_result("Calibrated Y values", calibrated.y, intensity_y)
    assert calibrated.xlabel == "λ"
    assert calibrated.xunit == "nm"
    assert calibrated.ylabel == "I"
    assert calibrated.yunit == "counts"


if __name__ == "__main__":
    test_replace_x_by_other_y_size_mismatch()
    test_replace_x_by_other_y()
