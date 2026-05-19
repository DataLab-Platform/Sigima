# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :class:`sigima.objects.GeometryResult`.

Covers ``bounding_boxes``, ``centers``, ``headers`` for every geometry kind
(segment, rectangle, circle, ellipse, point, marker, polygon),
``concat_geometries`` validation, and the constructor validation paths
for unknown kinds, wrong column counts and ROI index mismatches.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import GeometryResult
from sigima.objects.scalar.geometry import KindShape, concat_geometries

# ===========================================================================
# bounding_boxes / centers / headers
# ===========================================================================


def test_geometry_bounding_boxes_segment() -> None:
    """For ``segment`` shapes the bounding box is just the (x0, y0, x1, y1)
    tuple itself, and the centre is the segment midpoint."""
    coords = np.array([[0.0, 0.0, 3.0, 4.0]])
    geom = GeometryResult(title="t", kind="segment", coords=coords, func_name="t")
    bbox = geom.bounding_boxes()
    assert bbox.shape == (1, 4)
    assert np.allclose(bbox[0], [0.0, 0.0, 3.0, 4.0])
    centers = geom.centers()
    assert np.allclose(centers[0], [1.5, 2.0])


def test_geometry_bounding_boxes_rectangle() -> None:
    """``rectangle`` coords (x, y, width, height) are converted to the
    canonical (x0, y0, x1, y1) bounding box."""
    coords = np.array([[1.0, 2.0, 4.0, 6.0]])
    geom = GeometryResult(title="t", kind="rectangle", coords=coords, func_name="t")
    bbox = geom.bounding_boxes()
    assert np.allclose(bbox[0], [1.0, 2.0, 5.0, 8.0])


def test_geometry_bounding_boxes_circle() -> None:
    """For ``circle`` shapes (cx, cy, r) the bounding box is the inscribed
    square ``[cx-r, cy-r, cx+r, cy+r]``."""
    coords = np.array([[0.0, 0.0, 2.0]])
    geom = GeometryResult(title="t", kind="circle", coords=coords, func_name="t")
    bbox = geom.bounding_boxes()
    assert np.allclose(bbox[0], [-2.0, -2.0, 2.0, 2.0])


def test_geometry_bounding_boxes_ellipse() -> None:
    """For an axis-aligned ``ellipse`` (angle=0) the bounding box matches
    the semi-axes."""
    coords = np.array([[0.0, 0.0, 3.0, 1.0, 0.0]])
    geom = GeometryResult(title="t", kind="ellipse", coords=coords, func_name="t")
    bbox = geom.bounding_boxes()
    assert np.allclose(bbox[0], [-3.0, -1.0, 3.0, 1.0])


def test_geometry_bounding_boxes_point_marker() -> None:
    """For zero-area shapes (``point``/``marker``) the bounding box
    collapses to a single (x, y, x, y) point."""
    coords = np.array([[1.0, 2.0]])
    for kind in ("point", "marker"):
        geom = GeometryResult(title="t", kind=kind, coords=coords, func_name="t")
        bbox = geom.bounding_boxes()
        assert np.allclose(bbox[0], [1.0, 2.0, 1.0, 2.0])


def test_geometry_bounding_boxes_polygon_and_headers() -> None:
    """For a ``polygon`` the bounding box is the min/max envelope of all
    vertices, and the dynamic ``headers`` list one ``xN``/``yN`` pair per
    vertex."""
    coords = np.array([[0.0, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0]])
    geom = GeometryResult(title="t", kind="polygon", coords=coords, func_name="t")
    bbox = geom.bounding_boxes()
    assert np.allclose(bbox[0], [0.0, 0.0, 4.0, 3.0])
    headers = geom.headers
    assert headers == ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"]


def test_geometry_concat_requires_same_kind_and_func() -> None:
    """``concat_geometries`` requires all inputs to share both ``kind`` and
    ``func_name``, and rejects empty input sequences."""
    coords = np.array([[0.0, 0.0, 1.0, 1.0]])
    geom_a = GeometryResult(title="a", kind="segment", coords=coords, func_name="seg")
    geom_b = GeometryResult(title="b", kind="rectangle", coords=coords, func_name="seg")
    with pytest.raises(ValueError, match="same kind"):
        concat_geometries("merged", [geom_a, geom_b])
    geom_c = GeometryResult(title="c", kind="segment", coords=coords, func_name="other")
    with pytest.raises(ValueError, match="same func_name"):
        concat_geometries("merged", [geom_a, geom_c])
    with pytest.raises(ValueError, match="empty sequence"):
        concat_geometries("merged", [])


# ===========================================================================
# Constructor validation
# ===========================================================================


def test_geometry_unknown_kind_string_raises() -> None:
    """Unknown ``kind`` strings (typos) are rejected at construction time."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind="bogus", coords=np.zeros((1, 2)))


def test_geometry_kind_wrong_type_raises() -> None:
    """``kind`` must be a string or ``KindShape``; integers (and other
    non-string types) are rejected."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind=42, coords=np.zeros((1, 2)))


def test_geometry_empty_title_raises() -> None:
    """An empty ``title`` is forbidden because the title is used as a
    user-facing identifier."""
    with pytest.raises(ValueError):
        GeometryResult(title="", kind=KindShape.POINT, coords=np.zeros((1, 2)))


def test_geometry_coords_not_2d_raises() -> None:
    """``coords`` must always be a 2D array; passing 1D coordinates is a
    validation error."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind=KindShape.POINT, coords=np.zeros(4))


def test_geometry_point_wrong_columns_raises() -> None:
    """``point`` requires exactly 2 columns (x, y)."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind=KindShape.POINT, coords=np.zeros((1, 3)))


def test_geometry_segment_wrong_columns_raises() -> None:
    """``segment`` requires exactly 4 columns (x0, y0, x1, y1)."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind=KindShape.SEGMENT, coords=np.zeros((1, 3)))


def test_geometry_circle_wrong_columns_raises() -> None:
    """``circle`` requires exactly 3 columns (cx, cy, r)."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind=KindShape.CIRCLE, coords=np.zeros((1, 4)))


def test_geometry_ellipse_wrong_columns_raises() -> None:
    """``ellipse`` requires exactly 5 columns (cx, cy, a, b, angle)."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind=KindShape.ELLIPSE, coords=np.zeros((1, 4)))


def test_geometry_rectangle_wrong_columns_raises() -> None:
    """``rectangle`` requires exactly 4 columns (x, y, w, h)."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind=KindShape.RECTANGLE, coords=np.zeros((1, 3)))


def test_geometry_polygon_odd_columns_raises() -> None:
    """``polygon`` coords must come in (x, y) pairs, so an odd column
    count is rejected."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind=KindShape.POLYGON, coords=np.zeros((1, 3)))


def test_geometry_roi_indices_wrong_dim_raises() -> None:
    """``roi_indices`` must be a 1D array; multi-dimensional input is
    rejected."""
    with pytest.raises(ValueError):
        GeometryResult(
            title="t",
            kind=KindShape.POINT,
            coords=np.zeros((1, 2)),
            roi_indices=np.zeros((1, 1)),
        )


def test_geometry_roi_indices_length_mismatch_raises() -> None:
    """``roi_indices`` must have the same length as ``coords`` so that
    each shape can be mapped back to its source ROI."""
    with pytest.raises(ValueError):
        GeometryResult(
            title="t",
            kind=KindShape.POINT,
            coords=np.zeros((1, 2)),
            roi_indices=np.array([0, 1]),
        )


def test_geometry_marker_wrong_columns_raises() -> None:
    """``marker`` requires exactly 2 columns (x, y), like ``point``."""
    with pytest.raises(ValueError):
        GeometryResult(title="t", kind=KindShape.MARKER, coords=np.zeros((1, 3)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
