# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for image ROI geometry classes in
:mod:`sigima.objects.image.roi`.

Covers ``PolygonalROI``, ``RectangularROI`` and ``CircularROI`` validation
and HTML/summary rendering, the public ``create_image_roi`` /
``create_image_roi_around_points`` factories and ``ImageROI`` helpers
(subscript/equality/HTML/from_dict/add_roi/to_mask).
"""

# pylint: disable=invalid-name
# pylint: disable=protected-access

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import (
    ImageObj,
    create_image,
    create_image_roi,
)
from sigima.objects.image.roi import (
    CircularROI,
    ImageROI,
    PolygonalROI,
    RectangularROI,
    create_image_roi_around_points,
)


def _img(width: int = 32, height: int = 32) -> ImageObj:
    """Build a square float64 image with linearly increasing pixel values."""
    data = np.arange(height * width, dtype=np.float64).reshape(height, width)
    return create_image("img", data)


# ===========================================================================
# PolygonalROI
# ===========================================================================


def test_polygonal_roi_invalid_coords_raises() -> None:
    """A polygon needs at least 3 (x, y) pairs (6 coords); 3 raw values
    are not enough and must be rejected."""
    with pytest.raises(ValueError):
        PolygonalROI([0.0, 0.0, 10.0], indices=False)


def test_polygonal_roi_html_rows_few_vertices() -> None:
    """For a polygon with few vertices, the HTML coords table lists each
    vertex literally under a ``Vertices`` row."""
    roi = PolygonalROI([0.0, 0.0, 10.0, 0.0, 5.0, 8.0], indices=False)
    rows = roi.get_coords_html_rows()
    assert rows and "Vertices" in rows[0][0]
    assert "(0" in rows[0][1]


def test_polygonal_roi_html_rows_many_vertices() -> None:
    """For a polygon with many vertices, the HTML coords table collapses
    them into a ``N points`` summary instead of one row per vertex."""
    coords = []
    for i in range(8):
        coords.extend([float(i), float(i * 2)])
    roi = PolygonalROI(coords, indices=False)
    rows = roi.get_coords_html_rows()
    assert "8 points" in rows[0][1]


def test_polygonal_roi_summary() -> None:
    """The plain-text summary of a polygon mentions the vertex count."""
    roi = PolygonalROI([0.0, 0.0, 10.0, 0.0, 5.0, 8.0], indices=False)
    assert "3 vertices" in roi.get_coords_summary()


def test_polygonal_roi_to_mask_inverse() -> None:
    """With ``inverse=True`` a polygonal ROI masks pixels *outside* the
    polygon (centre pixel masked, corner pixel not)."""
    obj = _img()
    roi = PolygonalROI(
        [2.0, 2.0, 20.0, 2.0, 20.0, 20.0, 2.0, 20.0],
        indices=False,
        inverse=True,
    )
    mask = roi.to_mask(obj)
    assert mask.dtype == bool
    assert mask[10, 10]
    assert not mask[0, 0]


# ===========================================================================
# RectangularROI / CircularROI
# ===========================================================================


def test_rectangular_roi_invalid_coords_raises() -> None:
    """A rectangular ROI requires 4 coordinates (x, y, w, h); fewer is
    rejected."""
    with pytest.raises(ValueError):
        RectangularROI([0.0, 0.0, 10.0], indices=False)


def test_rectangular_roi_html_rows_and_summary() -> None:
    """Both the HTML coords table and the plain-text summary of a
    rectangle expose its origin."""
    roi = RectangularROI([1.0, 2.0, 5.0, 6.0], indices=False)
    rows = roi.get_coords_html_rows()
    assert any("Origin" in name for name, _ in rows)
    assert "Origin" in roi.get_coords_summary()


def test_circular_roi_invalid_coords_raises() -> None:
    """A circular ROI requires 3 coordinates (cx, cy, r); fewer is
    rejected."""
    with pytest.raises(ValueError):
        CircularROI([0.0, 0.0], indices=False)


def test_circular_roi_html_rows_and_summary() -> None:
    """Both the HTML coords table and the plain-text summary of a circle
    expose its centre."""
    roi = CircularROI([5.0, 5.0, 3.0], indices=False)
    rows = roi.get_coords_html_rows()
    assert any("Center" in name for name, _ in rows)
    assert "Center" in roi.get_coords_summary()


def test_circular_roi_to_mask_inverse() -> None:
    """With ``inverse=True`` a circular ROI keeps pixels inside the disc
    (regression: inverse logic must not skip the centre pixel)."""
    obj = _img()
    roi = CircularROI([16.0, 16.0, 8.0], indices=False, inverse=True)
    mask = roi.to_mask(obj)
    assert mask[16, 16]


# ===========================================================================
# create_image_roi - validation paths
# ===========================================================================


def test_create_image_roi_unknown_geometry_raises() -> None:
    """``create_image_roi`` rejects unknown geometry strings with a clear
    ``ValueError``."""
    with pytest.raises(ValueError):
        create_image_roi("unknown", [0.0, 0.0, 1.0, 1.0])


def test_create_image_roi_rectangle_invalid_count_raises() -> None:
    """``create_image_roi('rectangle', ...)`` rejects rows with the wrong
    number of values."""
    with pytest.raises(ValueError):
        create_image_roi("rectangle", [[0.0, 0.0, 1.0]])


def test_create_image_roi_circle_invalid_count_raises() -> None:
    """``create_image_roi('circle', ...)`` rejects rows with the wrong
    number of values."""
    with pytest.raises(ValueError):
        create_image_roi("circle", [[0.0, 0.0]])


def test_create_image_roi_polygon_invalid_count_raises() -> None:
    """``create_image_roi('polygon', ...)`` rejects vertex lists with an
    odd number of coordinates (vertices come in (x, y) pairs)."""
    with pytest.raises(ValueError):
        create_image_roi("polygon", [[0.0, 0.0, 1.0]])


def test_create_image_roi_inverse_length_mismatch_raises() -> None:
    """The ``inverse`` sequence must align 1-to-1 with the ROI rows;
    length mismatch raises ``ValueError``."""
    with pytest.raises(ValueError):
        create_image_roi("rectangle", [[0, 0, 1, 1]], inverse=[True, False])


def test_create_image_roi_around_points_no_coords() -> None:
    """``create_image_roi_around_points`` rejects an empty point set."""
    with pytest.raises(ValueError):
        create_image_roi_around_points(np.array([]).reshape(0, 2), "circle")


def test_create_image_roi_around_points_wrong_shape() -> None:
    """``create_image_roi_around_points`` rejects 1D coordinate arrays
    (must be N×2)."""
    with pytest.raises(ValueError):
        create_image_roi_around_points(np.array([1.0, 2.0, 3.0]), "circle")


def test_create_image_roi_around_points_too_few() -> None:
    """``create_image_roi_around_points`` needs at least 2 points to
    estimate a meaningful ROI size; 1 is rejected."""
    with pytest.raises(ValueError):
        create_image_roi_around_points(np.array([[1.0, 2.0]]), "circle")


def test_create_image_roi_around_points_too_close_raises() -> None:
    """Points that are too close to each other yield a degenerate ROI
    size and are rejected with ``ValueError``."""
    with pytest.raises(ValueError):
        create_image_roi_around_points(
            np.array([[10.0, 10.0], [10.5, 10.5]]), "rectangle"
        )


# ===========================================================================
# ImageROI - to_mask edge cases / helpers
# ===========================================================================


def test_imageroi_to_mask_empty() -> None:
    """After ``empty()`` an ``ImageROI`` has no single ROIs and produces
    an all-False mask."""
    obj = _img()
    roi = create_image_roi("rectangle", [0, 0, 4, 4])
    roi.empty()
    assert len(roi) == 0
    mask = roi.to_mask(obj)
    assert mask.dtype == bool and not mask.any()


def test_imageroi_to_mask_all_inverse_union() -> None:
    """When every sub-ROI is inverted, the resulting mask is still a
    boolean mask (union semantics, no crash)."""
    obj = _img()
    roi = create_image_roi(
        "rectangle",
        [[0, 0, 5, 5], [10, 10, 5, 5]],
        inverse=[True, True],
    )
    mask = roi.to_mask(obj)
    assert mask.dtype == bool


def test_imageroi_to_mask_mixed_inverse() -> None:
    """Mixing inverted and non-inverted sub-ROIs combines correctly into
    a single boolean mask."""
    obj = _img()
    roi = create_image_roi(
        "rectangle",
        [[0, 0, 20, 20], [5, 5, 5, 5]],
        inverse=[False, True],
    )
    mask = roi.to_mask(obj)
    assert mask.dtype == bool


def test_imageroi_subscript_set_and_get() -> None:
    """``ImageROI`` supports both ``roi[i] = ...`` subscript assignment
    and the explicit ``set_single_roi`` / ``get_single_roi`` accessors."""
    roi = create_image_roi("rectangle", [0, 0, 4, 4])
    new_single = RectangularROI([1, 2, 3, 4], indices=False)
    roi[0] = new_single
    assert roi[0] is new_single
    roi.set_single_roi(0, RectangularROI([5, 6, 7, 8], indices=False))
    assert roi.get_single_roi(0).coords[0] == 5


def test_imageroi_eq_with_none_and_wrong_type() -> None:
    """``ImageROI.__eq__`` returns ``False`` against ``None`` but raises
    ``TypeError`` against arbitrary unrelated types."""
    roi = create_image_roi("rectangle", [0, 0, 4, 4])
    # pylint: disable=singleton-comparison
    assert (roi == None) is False  # noqa: E711
    with pytest.raises(TypeError):
        _ = roi == "not an roi"


def test_imageroi_get_single_roi_title_default() -> None:
    """When no explicit title was given, ``get_single_roi_title`` returns
    a default starting with ``"ROI"``."""
    roi = create_image_roi("rectangle", [0, 0, 4, 4])
    title = roi.get_single_roi_title(0)
    assert title.startswith("ROI")


def test_imageroi_from_dict_invalid_type_raises() -> None:
    """``ImageROI.from_dict`` rejects unknown ROI ``type`` values with
    ``ValueError`` rather than silently producing an empty ROI."""
    with pytest.raises(ValueError):
        ImageROI.from_dict({"single_rois": [{"type": "BogusROI"}]})


def test_imageroi_from_dict_missing_key_raises() -> None:
    """``ImageROI.from_dict`` rejects empty / missing-key dictionaries
    with ``ValueError``."""
    with pytest.raises(ValueError):
        ImageROI.from_dict({})


def test_imageroi_repr_html_empty_and_filled() -> None:
    """``_repr_html_`` distinguishes the empty case (``"No ROIs"``) from
    the populated case (``"1 ROI"``)."""
    empty = ImageROI()
    html_empty = empty._repr_html_()
    assert "No ROIs" in html_empty

    roi = create_image_roi("rectangle", [0, 0, 4, 4])
    html = roi._repr_html_()
    assert "ROI" in html and "1 ROI" in html


def test_imageroi_add_roi_unsupported_type_raises() -> None:
    """``ImageROI.add_roi`` rejects objects that are not actual ROI
    instances (defensive type check)."""
    with pytest.raises(TypeError):
        ImageROI().add_roi("not an roi")  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
