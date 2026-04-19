# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the geometric and rotation helpers of
:py:mod:`sigima.tools.coordinates`.

These tests cover code paths not exercised by the existing
``tools_coordinates_unit_test.py`` (which focuses on
:py:func:`polar_to_complex`) and ``signal/processing_unit_test.py``
(which only covers basic ``to_polar`` / ``to_cartesian`` calls):

- Circle ↔ diameter conversions (scalar and array versions)
- Ellipse ↔ diameters/center-axes-angle conversions (scalar and array versions)
- Rotation matrix and 2D vector rotation
- Error paths for ``to_polar``, ``to_cartesian`` and ``polar_to_complex``
"""

from __future__ import annotations

import numpy as np
import pytest

from sigima.tests.helpers import check_array_result
from sigima.tools.coordinates import (
    array_circle_to_center_radius,
    array_circle_to_diameter,
    array_ellipse_to_center_axes_angle,
    array_ellipse_to_diameters,
    circle_to_center_radius,
    circle_to_diameter,
    colvector,
    ellipse_to_center_axes_angle,
    ellipse_to_diameters,
    polar_to_complex,
    rotate,
    to_cartesian,
    to_polar,
    vector_rotation,
)

# ----------------------------------------------------------------------------
# Circle helpers
# ----------------------------------------------------------------------------


def test_circle_roundtrip_scalar() -> None:
    """``circle_to_diameter`` and ``circle_to_center_radius`` are inverses."""
    xc, yc, r = 2.0, -3.0, 5.0
    x0, y0, x1, y1 = circle_to_diameter(xc, yc, r)
    assert (x0, y0, x1, y1) == (xc - r, yc, xc + r, yc)
    xc2, yc2, r2 = circle_to_center_radius(x0, y0, x1, y1)
    assert pytest.approx(xc2) == xc
    assert pytest.approx(yc2) == yc
    assert pytest.approx(r2) == r


def test_array_circle_to_diameter() -> None:
    """Array circle-to-diameter conversion returns a (N, 4) float array."""
    data = np.array([[0.0, 0.0, 1.0], [2.0, -3.0, 5.0], [-1.5, 4.0, 0.5]])
    result = array_circle_to_diameter(data)
    expected = np.array(
        [
            [-1.0, 0.0, 1.0, 0.0],
            [-3.0, -3.0, 7.0, -3.0],
            [-2.0, 4.0, -1.0, 4.0],
        ]
    )
    assert result.dtype == np.float64
    check_array_result("array_circle_to_diameter", result, expected)


def test_array_circle_roundtrip() -> None:
    """Array circle ↔ diameter round-trip is the identity."""
    data = np.array([[0.0, 0.0, 1.0], [2.0, -3.0, 5.0], [-1.5, 4.0, 0.5]])
    result = array_circle_to_center_radius(array_circle_to_diameter(data))
    check_array_result("array_circle_roundtrip", result, data)


# ----------------------------------------------------------------------------
# Ellipse helpers
# ----------------------------------------------------------------------------


def test_ellipse_roundtrip_scalar() -> None:
    """Ellipse scalar conversions round-trip on center, axes and angle."""
    xc, yc, a, b, theta = 1.0, -2.0, 5.0, 2.0, np.pi / 6
    coords = ellipse_to_diameters(xc, yc, a, b, theta)
    xc2, yc2, a2, b2, theta2 = ellipse_to_center_axes_angle(*coords)
    assert pytest.approx(xc2) == xc
    assert pytest.approx(yc2) == yc
    assert pytest.approx(a2) == a
    assert pytest.approx(b2) == b
    assert pytest.approx(theta2) == theta


def test_array_ellipse_roundtrip() -> None:
    """Array ellipse ↔ diameters round-trip is the identity."""
    data = np.array(
        [
            [0.0, 0.0, 3.0, 1.0, 0.0],
            [2.0, -1.0, 4.0, 2.0, np.pi / 4],
            [-1.0, 1.0, 5.0, 3.0, np.pi / 3],
        ]
    )
    coords = array_ellipse_to_diameters(data)
    assert coords.shape == (3, 8)
    assert coords.dtype == np.float64
    result = array_ellipse_to_center_axes_angle(coords)
    check_array_result("array_ellipse_roundtrip", result, data)


def test_array_ellipse_to_diameters_axis_aligned() -> None:
    """An axis-aligned ellipse must produce predictable diameter coordinates."""
    # Single row: center (0,0), semi-major a=3, semi-minor b=1, angle=0
    data = np.array([[0.0, 0.0, 3.0, 1.0, 0.0]])
    coords = array_ellipse_to_diameters(data)
    expected = np.array([[-3.0, 0.0, 3.0, 0.0, 0.0, -1.0, 0.0, 1.0]])
    check_array_result("array_ellipse_axis_aligned", coords, expected)


# ----------------------------------------------------------------------------
# Rotation helpers
# ----------------------------------------------------------------------------


def test_rotate_matrix_properties() -> None:
    """``rotate`` returns a 3×3 rotation matrix in homogeneous coordinates."""
    mat = rotate(np.pi / 2)
    assert mat.shape == (3, 3)
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    check_array_result("rotate_pi_over_2", mat, expected, atol=1e-12)


def test_colvector() -> None:
    """``colvector`` returns a 3-element homogeneous-coordinate vector."""
    vec = colvector(2.0, 3.0)
    assert vec.shape == (3,)
    assert vec[0] == 2.0 and vec[1] == 3.0 and vec[2] == 1.0


@pytest.mark.parametrize(
    "theta,dx,dy,expected",
    [
        (0.0, 1.0, 0.0, (1.0, 0.0)),
        (np.pi / 2, 1.0, 0.0, (0.0, 1.0)),
        (np.pi, 1.0, 0.0, (-1.0, 0.0)),
        (-np.pi / 2, 0.0, 1.0, (1.0, 0.0)),
    ],
)
def test_vector_rotation(
    theta: float, dx: float, dy: float, expected: tuple[float, float]
) -> None:
    """``vector_rotation`` returns the input vector rotated by ``theta``."""
    rx, ry = vector_rotation(theta, dx, dy)
    assert pytest.approx(rx, abs=1e-12) == expected[0]
    assert pytest.approx(ry, abs=1e-12) == expected[1]


# ----------------------------------------------------------------------------
# Error paths for to_polar / to_cartesian / polar_to_complex
# ----------------------------------------------------------------------------


def test_to_polar_invalid_unit() -> None:
    """``to_polar`` raises ``ValueError`` for unsupported unit."""
    with pytest.raises(ValueError):
        to_polar(np.array([1.0]), np.array([0.0]), unit="deg")  # type: ignore[arg-type]


def test_to_cartesian_invalid_unit() -> None:
    """``to_cartesian`` raises ``ValueError`` for unsupported unit."""
    with pytest.raises(ValueError):
        to_cartesian(np.array([1.0]), np.array([0.0]), unit="deg")  # type: ignore[arg-type]


def test_to_cartesian_negative_radius() -> None:
    """``to_cartesian`` rejects negative radius values."""
    with pytest.raises(ValueError):
        to_cartesian(np.array([1.0, -0.5]), np.array([0.0, 0.0]))


def test_polar_to_complex_negative_radius() -> None:
    """``polar_to_complex`` rejects negative radius values."""
    with pytest.raises(ValueError):
        polar_to_complex(np.array([1.0, -0.5]), np.array([0.0, 0.0]))


if __name__ == "__main__":
    pytest.main([__file__])
