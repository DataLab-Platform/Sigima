# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Non-regression test for ellipse contour -> polygon ROI conversion.

Background
----------
``contour_shape`` detects ellipses and, when ROI creation is requested, converts
each fitted ellipse ``(xc, yc, a, b, theta)`` into a polygon ROI through
:func:`sigima.proc.image.detection._ellipse_to_polygon`.

A previous bug made the polygon ROI appear sheared/rotated with respect to the
actual ellipse contour: for rotated contours the polygon could drift by more
than 10 pixels from the data. The root cause was that the polygon sampling did
not account for the scikit-image ``(row, col)`` fitting space (where the fitted
angle is measured relative to the ``y`` axis with swapped semi-axes), producing
non-orthogonal (sheared) axes.

This test pins the correct behaviour: the polygon ROI must
1. have orthogonal semi-axes (a true, non-sheared ellipse), and
2. follow the actual ellipse contour extracted from a rasterized ellipse.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest
from skimage import measure
from skimage.draw import polygon as sk_polygon  # pylint: disable=no-name-in-module

from sigima.proc.image.detection import _ellipse_to_polygon
from sigima.tools.image.preprocessing import fit_ellipse_model


def _make_ellipse_contour(
    xc: float, yc: float, a: float, b: float, theta: float, shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize a true rotated ellipse and return its contour in (row, col)."""
    t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    img = np.zeros(shape)
    rr, cc = sk_polygon(y, x, shape=shape)
    img[rr, cc] = 1.0
    return measure.find_contours(img, 0.5)[0]  # (row, col) = (y, x)


def _min_distance_max(poly_xy: np.ndarray, contour_xy: np.ndarray) -> float:
    """Max over polygon vertices of the nearest distance to the contour."""
    dist = np.sqrt(((poly_xy[:, None, :] - contour_xy[None, :, :]) ** 2).sum(axis=-1))
    return float(dist.min(axis=1).max())


@pytest.mark.parametrize("theta_deg", [0, 30, 45, 60, 90, 120, 150])
def test_ellipse_polygon_axes_are_orthogonal(theta_deg: int) -> None:
    """The sampled polygon must be a true (non-sheared) ellipse.

    Pins the fix for the sheared-ROI bug: the two semi-axis vectors used to
    sample the polygon must be orthogonal for any orientation.
    """
    xc, yc, a, b = 100.0, 50.0, 40.0, 15.0
    theta = np.deg2rad(theta_deg)
    vertices = _ellipse_to_polygon(xc, yc, a, b, theta, n_points=64).reshape(-1, 2)
    center = np.array([xc, yc])

    # cos-term semi-axis vector is vertices[0] - center (t = 0);
    # sin-term semi-axis vector is vertices[16] - center (t = pi/2, n=64).
    u = vertices[0] - center
    v = vertices[16] - center
    assert abs(np.dot(u, v)) < 1e-9, "Ellipse polygon axes are not orthogonal"


@pytest.mark.parametrize("theta_deg", [0, 30, 45, 60, 120, 150])
def test_ellipse_polygon_follows_contour(theta_deg: int) -> None:
    """The polygon ROI must overlay the actual ellipse contour.

    Full pipeline: true ellipse -> raster -> find_contours -> fit_ellipse_model
    -> _ellipse_to_polygon. Each polygon vertex must lie close to the contour
    (within rasterization noise), which would fail for the previous sheared
    formula (drift > 7 px for rotated contours).
    """
    xc0, yc0, A, B = 100.0, 80.0, 40.0, 15.0
    shape = (200, 220)
    theta = np.deg2rad(theta_deg)

    contour_rc = _make_ellipse_contour(xc0, yc0, A, B, theta, shape)
    contour_xy = contour_rc[:, ::-1]  # (x, y)

    fit = fit_ellipse_model(contour_rc)
    assert fit is not None
    xc, yc, a, b, theta_fit = fit

    poly = _ellipse_to_polygon(xc, yc, a, b, theta_fit, n_points=200).reshape(-1, 2)
    max_drift = _min_distance_max(poly, contour_xy)
    assert max_drift < 1.5, (
        f"Polygon ROI drifts {max_drift:.2f} px from the ellipse contour "
        f"(theta={theta_deg} deg)"
    )
