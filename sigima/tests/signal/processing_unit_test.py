# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for signal processing functions
------------------------------------------

Features from the "Processing" menu are covered by this test.
The "Processing" menu contains functions to process signals, such as
calibration, smoothing, and baseline correction.

Some of the functions are tested here, such as the signal calibration.
Other functions may be tested in different files, depending on the
complexity of the function.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest
import scipy
import scipy.ndimage as spi
import scipy.signal as sps
from packaging.version import Version

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.signal
import sigima.tests.data
import sigima.tools.coordinates
from sigima.tests.data import get_test_signal
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_signal_calibration() -> None:
    """Validation test for the signal calibration processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima.params.XYCalibrateParam()

    # Test with a = 1 and b = 0: should do nothing
    p.a, p.b = 1.0, 0.0
    for axis, _taxis in p.axes:
        p.axis = axis
        dst = sigima.proc.signal.calibration(src, p)
        exp = src.xydata
        check_array_result("Calibration[identity]", dst.xydata, exp)

    # Testing with random values of a and b
    p.a, p.b = 0.5, 0.1
    for axis, _taxis in p.axes:
        p.axis = axis
        exp_x1, exp_y1 = src.xydata.copy()
        if axis == "x":
            exp_x1 = p.a * exp_x1 + p.b
        else:
            exp_y1 = p.a * exp_y1 + p.b
        dst = sigima.proc.signal.calibration(src, p)
        res_x1, res_y1 = dst.xydata
        title = f"Calibration[{axis},a={p.a},b={p.b}]"
        check_array_result(f"{title}.x", res_x1, exp_x1)
        check_array_result(f"{title}.y", res_y1, exp_y1)


@pytest.mark.validation
def test_signal_transpose() -> None:
    """Validation test for the signal transpose processing."""
    src = get_test_signal("paracetamol.txt")
    dst = sigima.proc.signal.transpose(src)
    exp_y, exp_x = src.xydata
    check_array_result("Transpose|x", dst.x, exp_x)
    check_array_result("Transpose|y", dst.y, exp_y)


@pytest.mark.validation
def test_signal_reverse_x() -> None:
    """Validation test for the signal reverse x processing."""
    src = get_test_signal("paracetamol.txt")
    dst = sigima.proc.signal.reverse_x(src)
    exp = src.data[::-1]
    check_array_result("ReverseX", dst.data, exp)


def test_to_polar() -> None:
    """Unit test for the Cartesian to polar conversion."""
    title = "Cartesian2Polar"
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    r, theta = sigima.tools.coordinates.to_polar(x, y, "rad")
    exp_r = np.array([0.0, np.sqrt(2.0), np.sqrt(8.0), np.sqrt(18.0), np.sqrt(32.0)])
    exp_theta = np.array([0.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0])
    check_array_result(f"{title}|r", r, exp_r)
    check_array_result(f"{title}|theta", theta, exp_theta)

    r, theta = sigima.tools.coordinates.to_polar(x, y, unit="°")
    exp_theta = np.array([0.0, 45.0, 45.0, 45.0, 45.0])
    check_array_result(f"{title}|r", r, exp_r)
    check_array_result(f"{title}|theta", theta, exp_theta)


def test_to_cartesian() -> None:
    """Unit test for the polar to Cartesian conversion."""
    title = "Polar2Cartesian"
    r = np.array([0.0, np.sqrt(2.0), np.sqrt(8.0), np.sqrt(18.0), np.sqrt(32.0)])
    theta = np.array([0.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0])

    x, y = sigima.tools.coordinates.to_cartesian(r, theta, "rad")
    exp_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    exp_y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    check_array_result(f"{title}|x", x, exp_x)
    check_array_result(f"{title}|y", y, exp_y)

    theta = np.array([0.0, 45.0, 45.0, 45.0, 45.0])
    x, y = sigima.tools.coordinates.to_cartesian(r, theta, unit="°")
    check_array_result(f"{title}|x", x, exp_x)
    check_array_result(f"{title}|y", y, exp_y)


@pytest.mark.validation
def test_signal_to_polar() -> None:
    """Validation test for the signal Cartesian to polar processing."""
    title = "Cartesian2Polar"
    p = sigima.params.AngleUnitParam()
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    src = sigima.objects.create_signal("test", x, y)

    for p.unit in sigima.enums.AngleUnit:
        dst1 = sigima.proc.signal.to_polar(src, p)
        dst2 = sigima.proc.signal.to_cartesian(dst1, p)
        check_array_result(f"{title}|x", dst2.x, x)
        check_array_result(f"{title}|y", dst2.y, y)


@pytest.mark.validation
def test_signal_to_cartesian() -> None:
    """Validation test for the signal polar to Cartesian processing."""
    title = "Polar2Cartesian"
    p = sigima.params.AngleUnitParam()
    r = np.array([0.0, np.sqrt(2.0), np.sqrt(8.0), np.sqrt(18.0), np.sqrt(32.0)])

    a_deg = np.array([0.0, 45.0, 45.0, 45.0, 45.0])
    a_rad = np.array([0.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0])
    for p.unit in sigima.enums.AngleUnit:
        theta = a_rad if p.unit == sigima.enums.AngleUnit.RADIAN else a_deg
        src = sigima.objects.create_signal("test", r, theta)
        dst1 = sigima.proc.signal.to_cartesian(src, p)
        dst2 = sigima.proc.signal.to_polar(dst1, p)
        check_array_result(f"{title}|x", dst2.x, r)
        check_array_result(f"{title}|y", dst2.y, theta)


@pytest.mark.validation
def test_signal_normalize() -> None:
    """Validation test for the signal normalization processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima.params.NormalizeParam()
    src.y[10:15] = np.nan  # Adding some NaN values to the signal

    # Given the fact that the normalization methods implementations are
    # straightforward, we do not need to compare arrays with each other,
    # we simply need to check if some properties are satisfied.
    for method in sigima.enums.NormalizationMethod:
        p.method = method
        dst = sigima.proc.signal.normalize(src, p)
        title = f"Normalize[method='{p.method}']"
        exp_min, exp_max = None, None
        if p.method == sigima.enums.NormalizationMethod.MAXIMUM:
            exp_min, exp_max = np.nanmin(src.data) / np.nanmax(src.data), 1.0
        elif p.method == sigima.enums.NormalizationMethod.AMPLITUDE:
            exp_min, exp_max = 0.0, 1.0
        elif p.method == sigima.enums.NormalizationMethod.AREA:
            area = np.nansum(src.data)
            exp_min, exp_max = np.nanmin(src.data) / area, np.nanmax(src.data) / area
        elif p.method == sigima.enums.NormalizationMethod.ENERGY:
            energy = np.sqrt(np.nansum(np.abs(src.data) ** 2))
            exp_min, exp_max = (
                np.nanmin(src.data) / energy,
                np.nanmax(src.data) / energy,
            )
        elif p.method == sigima.enums.NormalizationMethod.RMS:
            rms = np.sqrt(np.nanmean(np.abs(src.data) ** 2))
            exp_min, exp_max = np.nanmin(src.data) / rms, np.nanmax(src.data) / rms
        check_scalar_result(f"{title}|min", np.nanmin(dst.data), exp_min)
        check_scalar_result(f"{title}|max", np.nanmax(dst.data), exp_max)

    # Exact analytical oracles (literals, never derived from Sigima)
    x = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
    y = np.array([-2.0, 0.0, 1.0, 4.0], dtype=np.float64)
    src = sigima.objects.create_signal("oracle", x, y, units=("s", ""))
    src_xydata = src.xydata.copy()
    oracles = {
        sigima.enums.NormalizationMethod.MAXIMUM: [-0.5, 0.0, 0.25, 1.0],
        sigima.enums.NormalizationMethod.AMPLITUDE: [0.0, 1 / 3, 0.5, 1.0],
    }
    for method, expected_y in oracles.items():
        p = sigima.params.NormalizeParam.create(method=method)
        dst = sigima.proc.signal.normalize(src, p)
        assert dst.x.dtype == np.float64 and dst.y.dtype == np.float64
        assert dst.x.shape == x.shape and dst.y.shape == x.shape
        assert np.array_equal(dst.x, x)
        assert np.array_equal(dst.y, np.array(expected_y, dtype=np.float64))
        assert (dst.xunit, dst.yunit) == ("s", "")
        assert np.array_equal(src.xydata, src_xydata)


@pytest.mark.validation
def test_signal_clip() -> None:
    """Validation test for the signal clipping processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima.params.ClipParam()

    for lower, upper in ((float("-inf"), float("inf")), (250.0, 500.0)):
        p.lower, p.upper = lower, upper
        dst = sigima.proc.signal.clip(src, p)
        exp = np.clip(src.data, p.lower, p.upper)
        check_array_result(f"Clip[{lower},{upper}]", dst.data, exp)


@pytest.mark.validation
def test_signal_derivative() -> None:
    """Validation test for the signal derivative processing."""
    src = get_test_signal("paracetamol.txt")
    dst = sigima.proc.signal.derivative(src)
    x, y = src.xydata

    # Compute the derivative using a simple finite difference:
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dydx = dy / dx
    exp = np.zeros_like(y)
    exp[0] = dydx[0]
    exp[1:-1] = (dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1]) / (dx[1:] + dx[:-1])
    exp[-1] = dydx[-1]

    check_array_result("Derivative", dst.y, exp)


@pytest.mark.validation
def test_signal_integral() -> None:
    """Validation test for the signal integral processing."""
    src = get_test_signal("paracetamol.txt")
    src.data /= np.max(src.data)

    # Check the integral of the derivative:
    dst = sigima.proc.signal.integral(sigima.proc.signal.derivative(src))
    # The integral of the derivative should be the original signal, up to a constant:
    dst.y += src.y[0]

    check_array_result("Integral[Derivative]", dst.y, src.y, atol=0.05)

    dst = sigima.proc.signal.integral(src)
    x, y = src.xydata

    # Compute the integral using a simple trapezoidal rule:
    exp = np.zeros_like(y)
    exp[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    exp[0] = exp[1]

    check_array_result("Integral", dst.y, exp, atol=0.05)


@pytest.mark.validation
def test_signal_detrending() -> None:
    """Validation test for the signal detrending processing."""
    src = get_test_signal("paracetamol.txt")
    for method_value, _method_name in sigima.params.DetrendingParam.methods:
        p = sigima.params.DetrendingParam.create(method=method_value)
        dst = sigima.proc.signal.detrending(src, p)
        exp = sps.detrend(src.data, type=p.method)
        check_array_result(f"Detrending [method={p.method}]", dst.data, exp)


@pytest.mark.validation
def test_signal_offset_correction() -> None:
    """Validation test for the signal offset correction processing."""
    src = get_test_signal("paracetamol.txt")
    # Defining the ROI that will be used to estimate the offset
    imin, imax = 0, 20
    p = sigima.objects.ROI1DParam.create(xmin=src.x[imin], xmax=src.x[imax])
    dst = sigima.proc.signal.offset_correction(src, p)
    exp = src.data - np.mean(src.data[imin:imax])
    check_array_result("OffsetCorrection", dst.data, exp)


@pytest.mark.validation
def test_signal_gaussian_filter() -> None:
    """Validation test for the signal Gaussian filter processing."""
    src = get_test_signal("paracetamol.txt")
    for sigma in (10.0, 50.0):
        p = sigima.params.GaussianParam.create(sigma=sigma)
        dst = sigima.proc.signal.gaussian_filter(src, p)
        exp = spi.gaussian_filter(src.data, sigma=sigma)
        check_array_result(f"GaussianFilter[sigma={sigma}]", dst.data, exp)


@pytest.mark.validation
def test_signal_moving_average() -> None:
    """Validation test for the signal moving average processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima.params.MovingAverageParam.create(n=30)
    for mode in sigima.enums.FilterMode:
        p.mode = mode
        dst = sigima.proc.signal.moving_average(src, p)
        exp = spi.uniform_filter(src.data, size=p.n, mode=mode.value)

        # Implementation note:
        # --------------------
        #
        # The SciPy's `uniform_filter` handles the edges more accurately than
        # a method based on a simple convolution with a kernel of ones like this:
        # (the following function was the original implementation of the moving average
        # in Sigima before it was replaced by the SciPy's `uniform_filter` function)
        #
        # def moving_average(y: np.ndarray, n: int) -> np.ndarray:
        #     y_padded = np.pad(y, (n // 2, n - 1 - n // 2), mode="edge")
        #     return np.convolve(y_padded, np.ones((n,)) / n, mode="valid")

        check_array_result(f"MovingAvg[n={p.n},mode={p.mode}]", dst.data, exp, rtol=0.1)


@pytest.mark.validation
@pytest.mark.skipif(
    Version("1.15.0") <= Version(scipy.__version__) <= Version("1.15.2"),
    reason="Skipping test: scipy median_filter is broken in 1.15.0-1.15.2",
)
def test_signal_moving_median() -> None:
    """Validation test for the signal moving median processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima.params.MovingMedianParam.create(n=15)
    for mode in sigima.enums.FilterMode:
        p.mode = mode
        dst = sigima.proc.signal.moving_median(src, p)
        exp = spi.median_filter(src.data, size=p.n, mode=mode.value)
        check_array_result(f"MovingMed[n={p.n},mode={p.mode}]", dst.data, exp, rtol=0.1)


@pytest.mark.validation
def test_signal_wiener() -> None:
    """Validation test for the signal Wiener filter processing."""
    src = get_test_signal("paracetamol.txt")
    dst = sigima.proc.signal.wiener(src)
    exp = sps.wiener(src.data)
    check_array_result("Wiener", dst.data, exp)


@pytest.mark.validation
def test_signal_resampling() -> None:
    """Validation test for the signal resampling processing."""
    src1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.SINE, freq=50.0, size=5
    )
    x1, y1 = src1.xydata
    p1 = sigima.params.Resampling1DParam.create(
        xmin=src1.x[0], xmax=src1.x[-1], nbpts=src1.x.size
    )
    p1.update_from_obj(src1)  # Just to test this method
    dst1 = sigima.proc.signal.resampling(src1, p1)
    dst1x, dst1y = dst1.xydata
    check_array_result("x1new", dst1x, x1)
    check_array_result("y1new", dst1y, y1)

    src2 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.SINE, freq=50.0, size=9
    )
    p2 = sigima.params.Resampling1DParam.create(
        xmin=src1.x[0], xmax=src1.x[-1], nbpts=src1.x.size
    )
    dst2 = sigima.proc.signal.resampling(src2, p2)
    dst2x, dst2y = dst2.xydata
    check_array_result("x2new", dst2x, x1)
    check_array_result("y2new", dst2y, y1)


@pytest.mark.validation
def test_signal_xy_mode() -> None:
    """Validation test for the signal X-Y mode processing."""
    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINE, freq=50.0, size=5
    )
    s2 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.SINE, freq=50.0, size=5
    )
    dst = sigima.proc.signal.xy_mode(s1, s2)
    x, y = dst.xydata
    check_array_result("XYMode", x, s1.y)
    check_array_result("XYMode", y, s2.y)
    check_array_result("XYMode", x**2 + y**2, np.ones_like(x))

    s1 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.COSINE, freq=50.0, size=9
    )
    s2 = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.SINE, freq=50.0, size=5
    )
    dst = sigima.proc.signal.xy_mode(s1, s2)
    x, y = dst.xydata
    check_array_result("XYMode2", x, s1.y[::2])
    check_array_result("XYMode2", y, s2.y)
    check_array_result("XYMode2", x**2 + y**2, np.ones_like(x))


@pytest.mark.validation
def test_signal_histogram() -> None:
    """Validation test for the signal histogram processing."""
    # Create a test signal with known data for histogram analysis
    src = sigima.tests.data.create_periodic_signal(
        sigima.objects.SignalTypes.SINE, freq=50.0, size=1000, a=2.0
    )

    # Test with default parameters
    p = sigima.params.HistogramParam()
    p.bins = 50
    p.lower = None
    p.upper = None
    dst = sigima.proc.signal.histogram(src, p)

    # Validate result properties
    x, y = dst.xydata

    # Check that we got the expected number of bins
    check_scalar_result("Histogram|bins", len(x), 50)
    check_scalar_result("Histogram|bins", len(y), 50)

    # Check that histogram sums to the total number of data points
    check_scalar_result("Histogram|total_counts", np.sum(y), len(src.y))

    # Check that all counts are non-negative
    assert np.all(y >= 0), "Histogram counts should be non-negative"

    # Check that x values are within the data range
    data_min, data_max = np.min(src.y), np.max(src.y)
    assert np.min(x) >= data_min, "Histogram x values should be >= data minimum"
    assert np.max(x) <= data_max, "Histogram x values should be <= data maximum"

    # Test with explicit range
    p.lower = -1.0
    p.upper = 1.0
    p.bins = 20
    dst2 = sigima.proc.signal.histogram(src, p)
    x2, y2 = dst2.xydata

    # Check that we got the expected number of bins
    check_scalar_result("Histogram|explicit_range_bins", len(x2), 20)

    # Check that x values are within the specified range
    assert np.min(x2) >= -1.0, "Histogram x values should be >= lower limit"
    assert np.max(x2) <= 1.0, "Histogram x values should be <= upper limit"

    # Check that counts sum to the number of data points within the range
    data_in_range = src.y[(src.y >= -1.0) & (src.y <= 1.0)]
    check_scalar_result("Histogram|range_counts", np.sum(y2), len(data_in_range))

    # Test with a simple known dataset by creating a signal manually
    # Create a signal with known uniform distribution data
    simple_sig = sigima.objects.create_signal(
        "Test Signal",
        np.linspace(0, 1, 100),  # x values
        np.linspace(0, 1, 100),  # y values: uniform distribution from 0 to 1
    )

    p_uniform = sigima.params.HistogramParam()
    p_uniform.bins = 10
    p_uniform.lower = 0.0
    p_uniform.upper = 1.0
    dst_uniform = sigima.proc.signal.histogram(simple_sig, p_uniform)
    _x_uniform, y_uniform = dst_uniform.xydata

    # For the uniform data (0 to 1), each bin should have exactly 10 values
    expected_count = 10
    check_array_result(
        "Histogram|uniform_counts", y_uniform, np.full(10, expected_count, dtype=float)
    )

    # Test edge case: single bin
    p_single = sigima.params.HistogramParam()
    p_single.bins = 1
    dst_single = sigima.proc.signal.histogram(src, p_single)
    x_single, y_single = dst_single.xydata
    check_scalar_result("Histogram|single_bin_x_count", len(x_single), 1)
    check_scalar_result("Histogram|single_bin_y_count", len(y_single), 1)
    check_scalar_result("Histogram|single_bin_total", y_single[0], len(src.y))

    # Test with binary data
    binary_sig = sigima.objects.create_signal(
        "Binary Signal",
        np.arange(20),  # x values
        np.array([0] * 10 + [1] * 10),  # 10 zeros, 10 ones
    )
    p_binary = sigima.params.HistogramParam()
    p_binary.bins = 2
    p_binary.lower = 0.0
    p_binary.upper = 1.0
    dst_binary = sigima.proc.signal.histogram(binary_sig, p_binary)
    _x_binary, y_binary = dst_binary.xydata

    # Should have 10 counts in each bin for our binary data
    check_array_result("Histogram|binary_counts", y_binary, np.array([10.0, 10.0]))


@pytest.mark.validation
def test_signal_interpolate() -> None:
    """Validation test for the signal interpolation processing."""
    # Create test signals
    x1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y1 = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # Quadratic function: y = x²
    src1 = sigima.objects.create_signal("src1", x1, y1)

    # Create target x-axis for interpolation (denser sampling)
    x2 = np.array([0.5, 1.5, 2.5, 3.5])
    y2 = np.zeros_like(x2)  # Y values don't matter for interpolation target
    src2 = sigima.objects.create_signal("src2", x2, y2)

    p = sigima.params.InterpolationParam()

    # Test linear interpolation
    p.method = sigima.enums.Interpolation1DMethod.LINEAR
    dst = sigima.proc.signal.interpolate(src1, src2, p)
    expected_linear = np.array([0.5, 2.5, 6.5, 12.5])  # Linear interpolation of x²
    check_array_result("Interpolate[LINEAR]", dst.y, expected_linear)
    check_array_result("Interpolate[LINEAR]|x", dst.x, x2)

    # Test spline interpolation (should be exact for polynomial functions)
    p.method = sigima.enums.Interpolation1DMethod.SPLINE
    dst = sigima.proc.signal.interpolate(src1, src2, p)
    expected_spline = np.array([0.25, 2.25, 6.25, 12.25])  # Exact values for x²
    check_array_result("Interpolate[SPLINE]", dst.y, expected_spline, atol=1e-10)
    check_array_result("Interpolate[SPLINE]|x", dst.x, x2)

    # Test quadratic interpolation (should be exact for polynomial functions)
    p.method = sigima.enums.Interpolation1DMethod.QUADRATIC
    dst = sigima.proc.signal.interpolate(src1, src2, p)
    expected_quadratic = np.array([0.25, 2.25, 6.25, 12.25])  # Exact values for x²
    check_array_result("Interpolate[QUADRATIC]", dst.y, expected_quadratic, atol=1e-10)
    check_array_result("Interpolate[QUADRATIC]|x", dst.x, x2)

    # Test cubic interpolation
    p.method = sigima.enums.Interpolation1DMethod.CUBIC
    dst = sigima.proc.signal.interpolate(src1, src2, p)
    expected_cubic = np.array([0.25, 2.25, 6.25, 12.25])  # Should be exact for x²
    check_array_result("Interpolate[CUBIC]", dst.y, expected_cubic, atol=1e-10)
    check_array_result("Interpolate[CUBIC]|x", dst.x, x2)

    # Test PCHIP interpolation
    p.method = sigima.enums.Interpolation1DMethod.PCHIP
    dst = sigima.proc.signal.interpolate(src1, src2, p)
    expected_pchip = np.array([0.3125, 2.21875, 6.23958333, 12.22916667])
    check_array_result("Interpolate[PCHIP]", dst.y, expected_pchip, atol=1e-10)
    check_array_result("Interpolate[PCHIP]|x", dst.x, x2)

    # Test barycentric interpolation
    p.method = sigima.enums.Interpolation1DMethod.BARYCENTRIC
    dst = sigima.proc.signal.interpolate(src1, src2, p)
    expected_barycentric = np.array([0.25, 2.25, 6.25, 12.25])  # Should be exact
    check_array_result(
        "Interpolate[BARYCENTRIC]", dst.y, expected_barycentric, atol=1e-10
    )
    check_array_result("Interpolate[BARYCENTRIC]|x", dst.x, x2)

    # Test fill_value parameter with extrapolation
    x2_extrap = np.array([-1.0, 0.5, 1.5, 5.0])  # Include points outside range
    y2_extrap = np.zeros_like(x2_extrap)
    src2_extrap = sigima.objects.create_signal("src2_extrap", x2_extrap, y2_extrap)
    # First, we test the linear method:
    p.method = sigima.enums.Interpolation1DMethod.LINEAR
    p.fill_value = -999.0  # Custom fill value for extrapolation
    dst = sigima.proc.signal.interpolate(src1, src2_extrap, p)
    expected_with_fill = np.array([-999.0, 0.5, 2.5, -999.0])
    check_array_result("Interpolate[LINEAR+fill_value]", dst.y, expected_with_fill)
    check_array_result("Interpolate[LINEAR+fill_value]|x", dst.x, x2_extrap)
    # Then, we test the pchip method:
    p.method = sigima.enums.Interpolation1DMethod.PCHIP
    dst = sigima.proc.signal.interpolate(src1, src2_extrap, p)
    expected_with_fill_pchip = np.array([-999.0, 0.3125, 2.21875, -999.0])
    check_array_result("Interpolate[PCHIP+fill_value]", dst.y, expected_with_fill_pchip)
    check_array_result("Interpolate[PCHIP+fill_value]|x", dst.x, x2_extrap)


@pytest.mark.validation
def test_signal_apply_window() -> None:
    """Validation test for the signal windowing processing."""
    # Create a test signal with known data
    x = np.linspace(0, 10, 100)
    y = np.ones_like(x)  # Constant signal to make windowing effects visible
    src = sigima.objects.create_signal("test_signal", x, y)

    p = sigima.params.WindowingParam()

    # Test HAMMING window (default)
    p.method = sigima.enums.WindowingMethod.HAMMING
    dst = sigima.proc.signal.apply_window(src, p)
    expected_hamming = y * np.hamming(len(y))
    check_array_result("ApplyWindow[HAMMING]", dst.y, expected_hamming)
    check_array_result("ApplyWindow[HAMMING]|x", dst.x, x)

    # Test BLACKMAN window
    p.method = sigima.enums.WindowingMethod.BLACKMAN
    dst = sigima.proc.signal.apply_window(src, p)
    expected_blackman = y * np.blackman(len(y))
    check_array_result("ApplyWindow[BLACKMAN]", dst.y, expected_blackman)
    check_array_result("ApplyWindow[BLACKMAN]|x", dst.x, x)

    # Test HANN window
    p.method = sigima.enums.WindowingMethod.HANN
    dst = sigima.proc.signal.apply_window(src, p)
    expected_hann = y * np.hanning(len(y))
    check_array_result("ApplyWindow[HANN]", dst.y, expected_hann)
    check_array_result("ApplyWindow[HANN]|x", dst.x, x)

    # Test GAUSSIAN window with custom sigma
    p.method = sigima.enums.WindowingMethod.GAUSSIAN
    p.sigma = 7.0
    dst = sigima.proc.signal.apply_window(src, p)
    expected_gaussian = y * scipy.signal.windows.gaussian(len(y), p.sigma)
    check_array_result("ApplyWindow[GAUSSIAN]", dst.y, expected_gaussian)
    check_array_result("ApplyWindow[GAUSSIAN]|x", dst.x, x)

    # Test KAISER window with custom beta
    p.method = sigima.enums.WindowingMethod.KAISER
    p.beta = 14.0
    dst = sigima.proc.signal.apply_window(src, p)
    expected_kaiser = y * np.kaiser(len(y), p.beta)
    check_array_result("ApplyWindow[KAISER]", dst.y, expected_kaiser)
    check_array_result("ApplyWindow[KAISER]|x", dst.x, x)

    # Test TUKEY window with custom alpha
    p.method = sigima.enums.WindowingMethod.TUKEY
    p.alpha = 0.5
    dst = sigima.proc.signal.apply_window(src, p)
    expected_tukey = y * scipy.signal.windows.tukey(len(y), p.alpha)
    check_array_result("ApplyWindow[TUKEY]", dst.y, expected_tukey)
    check_array_result("ApplyWindow[TUKEY]|x", dst.x, x)

    # Test BARTLETT window
    p.method = sigima.enums.WindowingMethod.BARTLETT
    dst = sigima.proc.signal.apply_window(src, p)
    expected_bartlett = y * np.bartlett(len(y))
    check_array_result("ApplyWindow[BARTLETT]", dst.y, expected_bartlett)
    check_array_result("ApplyWindow[BARTLETT]|x", dst.x, x)

    # Verify windowing preserves edge values for certain windows
    # Most windows should have zero or near-zero values at the edges
    p.method = sigima.enums.WindowingMethod.HAMMING
    dst = sigima.proc.signal.apply_window(src, p)
    assert dst.y[0] < 0.1, "Hamming window should have small edge values"
    assert dst.y[-1] < 0.1, "Hamming window should have small edge values"

    # Verify windowing preserves the original x-axis
    assert np.array_equal(dst.x, src.x), "X-axis should be preserved after windowing"

    # Verify the signal object metadata is properly set
    assert "Apply Window" in dst.title, (
        "Result title should indicate windowing operation"
    )


if __name__ == "__main__":
    test_signal_calibration()
    test_signal_transpose()
    test_to_polar()
    test_to_cartesian()
    test_signal_to_polar()
    test_signal_to_cartesian()
    test_signal_reverse_x()
    test_signal_normalize()
    test_signal_clip()
    test_signal_derivative()
    test_signal_integral()
    test_signal_offset_correction()
    test_signal_gaussian_filter()
    test_signal_moving_average()
    test_signal_moving_median()
    test_signal_wiener()
    test_signal_resampling()
    test_signal_xy_mode()
    test_signal_histogram()
    test_signal_interpolate()
    test_signal_apply_window()
