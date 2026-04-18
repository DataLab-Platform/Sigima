# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for signal analysis features
---------------------------------------

Features from the "Analysis" menu are covered by this test.
The "Analysis" menu contains functions to compute signal properties like
bandwidth, ENOB, etc.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
from sigima.tests import guiutils
from sigima.tests.data import get_test_signal
from sigima.tests.helpers import check_scalar_result
from sigima.tools.signal.features import (
    contrast,
    find_x_axis_crossings,
    find_zero_crossings,
)


@pytest.mark.validation
def test_signal_bandwidth_3db() -> None:
    """Validation test for the bandwidth computation."""
    obj = get_test_signal("bandwidth.txt")
    geometry = sigima.proc.signal.bandwidth_3db(obj)
    assert geometry is not None, "Bandwidth computation failed."
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from sigima import viz

            x0, y0, x1, y1 = geometry.coords[0]
            x, y = obj.xydata
            viz.view_curve_items(
                [
                    viz.create_curve(x.real, y.real, title=obj.title),
                    viz.create_segment(x0, y0, x1, y1, "Bandwidth@-3dB"),
                ],
                title="Bandwidth@-3dB",
            )
    length = geometry.segments_lengths()[0]
    check_scalar_result("Bandwidth@-3dB", length, 38.99301975103714)
    p1 = sigima.params.AbscissaParam.create(x=length)
    geometry_result = sigima.proc.signal.y_at_x(obj, p1)
    assert geometry_result is not None, "Y at X computation failed"
    _, y_val = geometry_result.value  # Get (x, y) tuple from .value property
    check_scalar_result("Value@cutoff", y_val, np.max(obj.y) - 3.0)


@pytest.mark.validation
def test_dynamic_parameters() -> None:
    """Validation test for dynamic parameters computation."""
    obj = get_test_signal("dynamic_parameters.txt")
    param = sigima.params.DynamicParam.create(full_scale=1.0)
    table = sigima.proc.signal.dynamic_parameters(obj, param)
    assert table is not None, "Dynamic parameters computation failed"
    tdict = table.as_dict()
    check_scalar_result("ENOB", tdict["enob"], 5.1, rtol=0.001)
    check_scalar_result("SINAD", tdict["sinad"], 32.49, rtol=0.001)
    check_scalar_result("THD", tdict["thd"], -30.18, rtol=0.001)
    check_scalar_result("SFDR", tdict["sfdr"], 34.03, rtol=0.001)
    check_scalar_result("Freq", tdict["freq"], 49998377.464, rtol=0.001)
    check_scalar_result("SNR", tdict["snr"], 101.52, rtol=0.001)


@pytest.mark.validation
def test_signal_sampling_rate_period() -> None:
    """Validation test for the sampling rate and period computation."""
    obj = get_test_signal("dynamic_parameters.txt")
    table = sigima.proc.signal.sampling_rate_period(obj)
    assert table is not None, "Sampling rate and period computation failed"
    check_scalar_result("Sampling rate", table["fs"][0], 1.0e10, rtol=0.001)
    check_scalar_result("Period", table["T"][0], 1.0e-10, rtol=0.001)


@pytest.mark.validation
def test_signal_contrast() -> None:
    """Validation test for the contrast computation."""
    obj = get_test_signal("fw1e2.txt")
    table = sigima.proc.signal.contrast(obj)
    assert table is not None, "Contrast computation failed"
    check_scalar_result("Contrast", table["contrast"][0], 0.825, rtol=0.001)

    # Regression checks on the underlying ``contrast`` tool: the original
    # Michelson formula ``(max-min)/(max+min)`` returned ``inf`` or ``nan``
    # for signals with negative samples, and was unbounded outside [0, 1].
    value = contrast(np.array([-1.0, 1.0]))
    assert np.isfinite(value), f"contrast must be finite, got {value!r}"
    value = contrast(np.array([-2.0, 1.0]))
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0, f"contrast should be in [0, 1], got {value!r}"
    # For non-negative signals, contrast keeps the standard Michelson value:
    np.testing.assert_allclose(contrast(np.array([1.0, 5.0])), 4.0 / 6.0)
    # A zero signal has an undefined contrast: must return NaN, not raise:
    assert np.isnan(contrast(np.zeros(5)))


@pytest.mark.parametrize(
    "y,expected_count",
    [
        (np.array([-1.0, 0.0, 1.0]), 1),
        (np.array([-1.0, 0.0, 0.0, 1.0]), 1),
        (np.array([1.0, 0.0, -1.0, 0.0, 1.0]), 2),
        (np.array([1.0, 0.0, 1.0]), 0),  # touches zero but does not cross
        (np.array([1.0, -1.0, 1.0]), 2),
        (np.array([1.0, 1.0, 1.0]), 0),
    ],
)
def test_find_zero_crossings_counts(y: np.ndarray, expected_count: int) -> None:
    """find_zero_crossings must report each crossing exactly once.

    Regression test: the original implementation double-counted exact-zero
    samples and reported false crossings on touch-and-bounce patterns.
    """
    crossings = find_zero_crossings(y)
    assert len(crossings) == expected_count, (
        f"Wrong crossing count for {y.tolist()}: got {crossings.tolist()}, "
        f"expected {expected_count}"
    )


def test_find_x_axis_crossings_through_exact_zero() -> None:
    """find_x_axis_crossings must report exactly one crossing for an exact zero."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([-1.0, 0.0, 1.0])
    xc = find_x_axis_crossings(x, y)
    assert len(xc) == 1, f"Expected 1 crossing, got {xc.tolist()}"
    np.testing.assert_allclose(xc, [1.0])


@pytest.mark.validation
def test_signal_x_at_minmax() -> None:
    """Validation test for the x value at min/max computation."""
    obj = get_test_signal("fw1e2.txt")
    table = sigima.proc.signal.x_at_minmax(obj)
    assert table is not None, "X at min/max computation failed"
    check_scalar_result("X@Ymin", table["X@Ymin"][0], 0.803, rtol=0.001)
    check_scalar_result("X@Ymax", table["X@Ymax"][0], 5.184, rtol=0.001)


@pytest.mark.validation
def test_signal_x_at_y() -> None:
    """Validation test for the abscissa finding computation."""
    obj = sigima.objects.create_signal_from_param(sigima.objects.StepParam.create())
    if obj is None:
        raise ValueError("Failed to create test signal")
    param = sigima.proc.signal.OrdinateParam.create(y=0.5)
    geometry = sigima.proc.signal.x_at_y(obj, param)
    assert geometry is not None, "X at Y computation failed"
    x_val, _ = geometry.value  # Get (x, y) tuple from .value property
    check_scalar_result("x|y=0.5", x_val, 0.0)


@pytest.mark.validation
def test_signal_y_at_x() -> None:
    """Validation test for the ordinate finding computation."""
    param = sigima.objects.TriangleParam.create(xmin=0.0, xmax=10.0, size=101)
    obj = sigima.objects.create_signal_from_param(param)
    if obj is None:
        raise ValueError("Failed to create test signal")
    param = sigima.proc.signal.AbscissaParam.create(x=2.5)
    geometry = sigima.proc.signal.y_at_x(obj, param)
    assert geometry is not None, "Y at X computation failed"
    _, y_val = geometry.value  # Get (x, y) tuple from .value property
    check_scalar_result("y|x=2.5", y_val, 1.0)


def test_x_at_y_geometry_result() -> None:
    """Test that x_at_y returns a GeometryResult with MARKER kind."""
    # Create a simple sine wave signal
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    obj = sigima.objects.SignalObj()
    obj.set_xydata(x, y)

    # Find x where y = 0.5
    param = sigima.params.OrdinateParam.create(y=0.5)
    result = sigima.proc.signal.x_at_y(obj, param)

    # Check it's a GeometryResult with MARKER kind
    assert result is not None, "x_at_y returned None"
    assert hasattr(result, "kind"), "Result should have 'kind' attribute"
    assert result.kind == sigima.objects.KindShape.MARKER, (
        f"Expected MARKER kind, got {result.kind}"
    )

    # Check coords shape (should be N×2 for MARKER)
    assert result.coords.shape[1] == 2, (
        f"MARKER coords should be N×2, got {result.coords.shape}"
    )

    # Check .value property returns (x, y) tuple
    x_val, y_val = result.value
    assert isinstance(x_val, float), "x value should be a float"
    assert isinstance(y_val, float), "y value should be a float"
    assert abs(y_val - 0.5) < 1e-10, f"Y value should be 0.5, got {y_val}"

    # Verify the x coordinate is reasonable (between 0 and pi/2 for sin(x) = 0.5)
    assert 0 < x_val < np.pi / 2, f"X value {x_val} out of expected range"


def test_y_at_x_geometry_result() -> None:
    """Test that y_at_x returns a GeometryResult with MARKER kind."""
    # Create a simple quadratic signal
    x = np.linspace(-5, 5, 101)
    y = x**2
    obj = sigima.objects.SignalObj()
    obj.set_xydata(x, y)

    # Find y where x = 3.0
    param = sigima.params.AbscissaParam.create(x=3.0)
    result = sigima.proc.signal.y_at_x(obj, param)

    # Check it's a GeometryResult with MARKER kind
    assert result is not None, "y_at_x returned None"
    assert hasattr(result, "kind"), "Result should have 'kind' attribute"
    assert result.kind == sigima.objects.KindShape.MARKER, (
        f"Expected MARKER kind, got {result.kind}"
    )

    # Check coords shape (should be N×2 for MARKER)
    assert result.coords.shape[1] == 2, (
        f"MARKER coords should be N×2, got {result.coords.shape}"
    )

    # Check .value property returns (x, y) tuple
    x_val, y_val = result.value
    assert isinstance(x_val, float), "x value should be a float"
    assert isinstance(y_val, float), "y value should be a float"
    assert abs(x_val - 3.0) < 1e-10, f"X value should be 3.0, got {x_val}"

    # Verify the y coordinate is correct (should be 9.0 for x^2 at x=3)
    assert abs(y_val - 9.0) < 0.1, f"Y value should be ~9.0, got {y_val}"


def test_geometry_result_value_property() -> None:
    """Test the .value property for POINT, MARKER, and SEGMENT shapes."""
    # Test POINT
    point_result = sigima.objects.GeometryResult.from_coords(
        title="Test Point",
        kind=sigima.objects.KindShape.POINT,
        coords=np.array([[1.5, 2.5]]),
    )
    x, y = point_result.value
    assert abs(x - 1.5) < 1e-10, f"POINT x should be 1.5, got {x}"
    assert abs(y - 2.5) < 1e-10, f"POINT y should be 2.5, got {y}"

    # Test MARKER
    marker_result = sigima.objects.GeometryResult.from_coords(
        title="Test Marker",
        kind=sigima.objects.KindShape.MARKER,
        coords=np.array([[3.0, 4.0]]),
    )
    x, y = marker_result.value
    assert abs(x - 3.0) < 1e-10, f"MARKER x should be 3.0, got {x}"
    assert abs(y - 4.0) < 1e-10, f"MARKER y should be 4.0, got {y}"

    # Test SEGMENT (3-4-5 triangle)
    segment_result = sigima.objects.GeometryResult.from_coords(
        title="Test Segment",
        kind=sigima.objects.KindShape.SEGMENT,
        coords=np.array([[0.0, 0.0, 3.0, 4.0]]),
    )
    length = segment_result.value
    assert isinstance(length, float), "SEGMENT value should be a float"
    assert abs(length - 5.0) < 1e-10, f"SEGMENT length should be 5.0, got {length}"

    # Test error for unsupported shape (CIRCLE)
    circle_result = sigima.objects.GeometryResult.from_coords(
        title="Test Circle",
        kind=sigima.objects.KindShape.CIRCLE,
        coords=np.array([[0.0, 0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="value property only valid for"):
        _ = circle_result.value

    # Test error for multiple rows
    multi_result = sigima.objects.GeometryResult.from_coords(
        title="Multiple Points",
        kind=sigima.objects.KindShape.POINT,
        coords=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    with pytest.raises(ValueError, match="single-row results"):
        _ = multi_result.value


def test_x_at_y_cross_marker_coordinates() -> None:
    """Test that x_at_y returns both x and y coordinates for cross marker display."""
    # Create a test signal
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1  # Linear signal: y = 2x + 1
    obj = sigima.objects.SignalObj()
    obj.set_xydata(x, y)

    # Find x where y = 5.0 (should be x = 2.0)
    param = sigima.params.OrdinateParam.create(y=5.0)
    result = sigima.proc.signal.x_at_y(obj, param)

    assert result is not None
    # Check that we have both coordinates (not NaN)
    coords = result.coords[0]
    x_coord, y_coord = coords
    assert not np.isnan(x_coord), "X coordinate should not be NaN"
    assert not np.isnan(y_coord), "Y coordinate should not be NaN"

    # Verify values are correct
    check_scalar_result("x_at_y x-coordinate", x_coord, 2.0, rtol=0.01)
    check_scalar_result("x_at_y y-coordinate", y_coord, 5.0, rtol=0.01)


def test_y_at_x_cross_marker_coordinates() -> None:
    """Test that y_at_x returns both x and y coordinates for cross marker display."""
    # Create a test signal
    x = np.linspace(0, 10, 100)
    y = x**2  # Quadratic signal: y = x^2
    obj = sigima.objects.SignalObj()
    obj.set_xydata(x, y)

    # Find y where x = 3.0 (should be y = 9.0)
    param = sigima.params.AbscissaParam.create(x=3.0)
    result = sigima.proc.signal.y_at_x(obj, param)

    assert result is not None
    # Check that we have both coordinates (not NaN)
    coords = result.coords[0]
    x_coord, y_coord = coords
    assert not np.isnan(x_coord), "X coordinate should not be NaN"
    assert not np.isnan(y_coord), "Y coordinate should not be NaN"

    # Verify values are correct
    check_scalar_result("y_at_x x-coordinate", x_coord, 3.0, rtol=0.01)
    check_scalar_result("y_at_x y-coordinate", y_coord, 9.0, rtol=0.01)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_signal_bandwidth_3db()
    test_dynamic_parameters()
    test_signal_sampling_rate_period()
    test_signal_contrast()
    test_find_x_axis_crossings_through_exact_zero()
    test_signal_x_at_minmax()
    test_signal_x_at_y()
    test_signal_y_at_x()
    test_x_at_y_geometry_result()
    test_y_at_x_geometry_result()
    test_geometry_result_value_property()
    test_x_at_y_cross_marker_coordinates()
    test_y_at_x_cross_marker_coordinates()
