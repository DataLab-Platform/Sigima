# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for shape module
"""

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import shape


def test_point_inplace_transform_and_copy() -> None:
    """
    Test inplace transformations and copy for PointCoordinates.

    Verifies that translation and rotation work inplace, and that copy creates
    an independent object.
    """
    pt = shape.PointCoordinates([1, 2])
    pt2 = pt.copy()
    assert np.allclose(pt.data, pt2.data)
    pt.translate(3, 4)
    assert np.allclose(pt.data, [4, 6])
    pt2.rotate(np.pi, center=(1, 2))
    assert np.allclose(pt2.data, [1, 2])  # Rotating at own center: no change


def test_segment_inplace_transform() -> None:
    """Test inplace transformations for SegmentCoordinates."""
    seg = shape.SegmentCoordinates([0, 0, 1, 1])
    seg2 = seg.copy()
    seg.translate(1, 1)
    assert np.allclose(seg.data, [1, 1, 2, 2])
    seg2.rotate(np.pi / 2, center=(0, 0))
    assert np.allclose(seg2.data, [0, 0, -1, 1])


def test_rectangle_inplace_transform() -> None:
    """
    Test inplace horizontal and vertical flipping for RectangleCoordinates.

    Verifies that flipping is performed inplace and independently on copies.
    """
    rect = shape.RectangleCoordinates([0.0, 0.0, 2.0, 3.0])
    rect2 = rect.copy()
    rect.fliph(cx=1.5)
    assert np.allclose(rect.data, [1.0, 0.0, 2.0, 3.0])
    rect2.flipv(cy=2.5)
    assert np.allclose(rect2.data, [0.0, 2.0, 2.0, 3.0])


def test_circle_inplace_transform() -> None:
    """
    Test inplace translation and scaling for CircleCoordinates.

    Verifies that only the center is transformed and the radius remains
    unchanged.
    """
    circ = shape.CircleCoordinates([1, 2, 5])
    circ2 = circ.copy()
    circ.translate(2, -1)
    assert np.allclose(circ.data, [3, 1, 5])
    circ2.scale(2, 2)
    assert np.allclose(circ2.data, [2, 4, 5])  # Only center is scaled


def test_polygon_inplace_transform() -> None:
    """
    Test inplace transpose and rotation for PolygonCoordinates.

    Verifies that transpose and rotation are performed inplace and
    independently on copies.
    """
    poly = shape.PolygonCoordinates([0, 0, 1, 0, 1, 1, 0, 1])
    poly2 = poly.copy()
    poly.transpose()
    assert np.allclose(poly.data, [0, 0, 0, 1, 1, 1, 1, 0])
    poly2.rotate(np.pi / 2, center=(0, 0))
    assert np.allclose(poly2.data, [0, 0, 0, 1, -1, 1, -1, 0])


def test_inplace_vs_copy() -> None:
    """
    Test that inplace transformation does not affect a copy.

    Verifies that after translating the original, the copy remains unchanged.
    """
    pt = shape.PointCoordinates([1, 2])
    pt2 = pt.copy()
    pt.translate(1, 1)
    assert not np.allclose(pt.data, pt2.data)


def test_rotate_arbitrary_center() -> None:
    """
    Test rotation around an arbitrary center for all coordinate types.
    """
    pt = shape.PointCoordinates([2, 3])
    pt.rotate(np.pi / 2, center=(1, 1))
    # (2,3) rotated 90° CCW around (1,1) -> ( -1,2 )
    assert np.allclose(pt.data, [-1, 2])

    rect = shape.RectangleCoordinates([2, 3, 4, 5])
    rect.rotate(np.pi / 2, center=(1, 1))
    expected = [-6, 2, 5, 4]
    assert np.allclose(rect.data, expected)

    circ = shape.CircleCoordinates([2, 3, 5])
    circ.rotate(np.pi / 2, center=(1, 1))
    assert np.allclose(circ.data, [-1, 2, 5])

    poly = shape.PolygonCoordinates([2, 3, 4, 5, 6, 7])
    poly.rotate(np.pi / 2, center=(1, 1))
    expected_poly = [-1, 2, -3, 4, -5, 6]
    assert np.allclose(poly.data, expected_poly)


def test_fliph_flipv_arbitrary_axis() -> None:
    """
    Test horizontal and vertical flip with respect to arbitrary axis
    for all coordinate types.
    """
    pt = shape.PointCoordinates([2, 3])
    pt.fliph(cx=1.5)
    assert np.allclose(pt.data, [1.0, 3])
    pt2 = shape.PointCoordinates([2, 3])
    pt2.flipv(cy=2.5)
    assert np.allclose(pt2.data, [2, 2.0])

    rect = shape.RectangleCoordinates([2, 3, 4, 5])
    rect.fliph(cx=3)
    assert np.allclose(rect.data, [0, 3, 4, 5])
    rect2 = shape.RectangleCoordinates([2, 3, 4, 5])
    rect2.flipv(cy=4)
    assert np.allclose(rect2.data, [2, 0, 4, 5])

    circ = shape.CircleCoordinates([2, 3, 5])
    circ.fliph(cx=1)
    assert np.allclose(circ.data, [0, 3, 5])
    circ2 = shape.CircleCoordinates([2, 3, 5])
    circ2.flipv(cy=2)
    assert np.allclose(circ2.data, [2, 1, 5])

    poly = shape.PolygonCoordinates([2, 3, 4, 5])
    poly.fliph(cx=3)
    assert np.allclose(poly.data, [4, 3, 2, 5])
    poly2 = shape.PolygonCoordinates([2, 3, 4, 5])
    poly2.flipv(cy=4)
    assert np.allclose(poly2.data, [2, 5, 4, 3])


def test_transpose_all_types() -> None:
    """
    Test transposition (swap x and y) for all coordinate types.
    """
    pt = shape.PointCoordinates([2, 3])
    pt.transpose()
    assert np.allclose(pt.data, [3, 2])

    rect = shape.RectangleCoordinates([2, 3, 4, 5])
    rect.transpose()
    assert np.allclose(rect.data, [3, 2, 5, 4])

    circ = shape.CircleCoordinates([2, 3, 5])
    circ.transpose()
    assert np.allclose(circ.data, [3, 2, 5])

    poly = shape.PolygonCoordinates([2, 3, 4, 5, 6, 7])
    poly.transpose()
    assert np.allclose(poly.data, [3, 2, 5, 4, 7, 6])


def test_validation_invalid_ndim() -> None:
    """Test that validation rejects non-1D arrays."""
    # PointCoordinates expects 1D array
    with pytest.raises(ValueError, match="Invalid.*coordinates ndim.*expected 1"):
        shape.PointCoordinates([[1, 2]])  # 2D array

    # SegmentCoordinates expects 1D array
    with pytest.raises(ValueError, match="Invalid.*coordinates ndim.*expected 1"):
        shape.SegmentCoordinates([[1, 2, 3, 4]])  # 2D array

    # RectangleCoordinates expects 1D array
    with pytest.raises(ValueError, match="Invalid.*coordinates ndim.*expected 1"):
        shape.RectangleCoordinates(np.array([[0, 0, 2, 3]]))  # 2D array


def test_validation_invalid_shape() -> None:
    """Test that validation rejects arrays with wrong shape."""
    # PointCoordinates expects exactly 2 values
    with pytest.raises(ValueError, match="Invalid.*coordinates shape.*expected.*2"):
        shape.PointCoordinates([1, 2, 3])  # 3 values instead of 2

    # SegmentCoordinates expects exactly 4 values
    with pytest.raises(ValueError, match="Invalid.*coordinates shape.*expected.*4"):
        shape.SegmentCoordinates([1, 2, 3])  # 3 values instead of 4

    # RectangleCoordinates expects exactly 4 values
    with pytest.raises(ValueError, match="Invalid.*coordinates shape.*expected.*4"):
        shape.RectangleCoordinates([1, 2, 3, 4, 5])  # 5 values instead of 4

    # CircleCoordinates expects exactly 3 values
    with pytest.raises(ValueError, match="Invalid.*coordinates shape.*expected.*3"):
        shape.CircleCoordinates([1, 2])  # 2 values instead of 3

    # EllipseCoordinates expects exactly 5 values (x, y, a, b, theta), like the
    # ellipse geometry results
    with pytest.raises(ValueError, match="Invalid.*coordinates shape.*expected.*5"):
        shape.EllipseCoordinates([1, 2, 3, 4])  # 4 values instead of 5


def test_validation_odd_number_of_values() -> None:
    """Test that PolygonCoordinates rejects odd number of values."""
    # PolygonCoordinates requires even number of values (x,y pairs)
    with pytest.raises(
        ValueError, match="Invalid.*coordinates.*even number of values expected"
    ):
        shape.PolygonCoordinates([1, 2, 3])  # 3 values (odd)

    with pytest.raises(
        ValueError, match="Invalid.*coordinates.*even number of values expected"
    ):
        shape.PolygonCoordinates([1, 2, 3, 4, 5])  # 5 values (odd)

    # Valid cases with even number of values should work
    poly = shape.PolygonCoordinates([1, 2, 3, 4])  # 4 values (even) - OK
    assert poly.data.size == 4

    poly = shape.PolygonCoordinates([1, 2, 3, 4, 5, 6])  # 6 values (even) - OK
    assert poly.data.size == 6


def test_validation_edge_cases() -> None:
    """Test validation with edge cases."""
    # Empty arrays should fail for shapes with VALID_SHAPE
    with pytest.raises(ValueError, match="Invalid.*coordinates shape"):
        shape.PointCoordinates([])

    # Test with single value (wrong for all shapes)
    with pytest.raises(ValueError, match="Invalid.*coordinates"):
        shape.PointCoordinates([1])

    # PolygonCoordinates with empty array (0 is even, but probably not useful)
    poly = shape.PolygonCoordinates([])
    assert poly.data.size == 0  # Technically valid


if __name__ == "__main__":
    test_point_inplace_transform_and_copy()
    test_rectangle_inplace_transform()
    test_circle_inplace_transform()
    test_polygon_inplace_transform()
    test_inplace_vs_copy()
    test_rotate_arbitrary_center()
    test_fliph_flipv_arbitrary_axis()
    test_transpose_all_types()
    test_validation_invalid_ndim()
    test_validation_invalid_shape()
    test_validation_odd_number_of_values()
    test_validation_edge_cases()
