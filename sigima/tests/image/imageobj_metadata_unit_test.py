# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for less-covered code paths of :py:mod:`sigima.objects.image.object`.

These tests complement ``imageobj_unit_test.py`` (which focuses on the main
``ImageObj`` API and uniform↔non-uniform conversion) by exercising:

- The :py:func:`sigima.objects.image.object.to_builtin` helper used to filter
  metadata values (numeric coercion, strings, iterables, generic objects).
- ``switch_coords_to("uniform")`` from non-uniform coordinates, including the
  failure mode when not enough coordinates are defined.
- The non-uniform branches of ``_compute_xmin/xmax/ymin/ymax`` extent helpers.
- Setting a DICOM-like template, which transparently triggers metadata
  population from arbitrary objects.
"""

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
from sigima.objects.image.object import ImageObj, to_builtin

# ----------------------------------------------------------------------------
# to_builtin
# ----------------------------------------------------------------------------


def test_to_builtin_numeric_types() -> None:
    """Integers and floats are preserved with the smallest builtin type."""
    assert to_builtin(3) == 3 and isinstance(to_builtin(3), int)
    assert to_builtin(3.0) == 3 and isinstance(to_builtin(3.0), int)
    res = to_builtin(2.5)
    assert res == 2.5 and isinstance(res, float)


def test_to_builtin_string() -> None:
    """Non-numeric strings are returned unchanged as ``str``."""
    assert to_builtin("hello") == "hello"


def test_to_builtin_iterable() -> None:
    """Generic iterables are converted to a Python ``list``."""
    assert to_builtin((1, 2, 3)) == [1, 2, 3]


def test_to_builtin_object_with_dict() -> None:
    """Objects exposing ``__dict__`` are serialized to a ``dict``."""

    class Foo:
        """Plain object with two attributes; serialized via ``__dict__``."""

        def __init__(self) -> None:
            """Initialise with two simple attributes."""
            self.a = 1
            self.b = "x"

    result = to_builtin(Foo())
    assert isinstance(result, dict)
    assert result == {"a": 1, "b": "x"}


def test_to_builtin_unsupported() -> None:
    """Objects with no usable conversion path return ``None``."""

    class Opaque:
        """Object with no ``__dict__`` and no recognised conversion path."""

        __slots__ = ()  # No __dict__, not iterable, not numeric, not a string

    assert to_builtin(Opaque()) is None


# ----------------------------------------------------------------------------
# switch_coords_to and extent helpers (non-uniform branch)
# ----------------------------------------------------------------------------


def _make_uniform_image(shape: tuple[int, int] = (4, 5)) -> ImageObj:
    """Create a uniform-coordinate image with deterministic data."""
    data = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
    img = sigima.objects.create_image(title="img", data=data)
    img.set_uniform_coords(dx=2.0, dy=3.0, x0=10.0, y0=-5.0)
    return img


def test_switch_to_non_uniform_then_back_to_uniform() -> None:
    """Round-trip uniform → non-uniform → uniform restores pixel spacing/origin."""
    img = _make_uniform_image()
    dx, dy, x0, y0 = img.dx, img.dy, img.x0, img.y0
    img.switch_coords_to("non-uniform")
    assert not img.is_uniform_coords
    img.switch_coords_to("uniform")
    assert img.is_uniform_coords
    assert pytest.approx(img.dx) == dx
    assert pytest.approx(img.dy) == dy
    assert pytest.approx(img.x0) == x0
    assert pytest.approx(img.y0) == y0


def test_switch_to_uniform_no_op_when_already_uniform() -> None:
    """Switching to ``uniform`` on a uniform image is a no-op."""
    img = _make_uniform_image()
    img.switch_coords_to("uniform")
    assert img.is_uniform_coords


def test_switch_to_uniform_fails_with_insufficient_coords() -> None:
    """Non-uniform → uniform requires at least 2 X and Y coordinates."""
    img = _make_uniform_image(shape=(1, 1))
    img.set_coords(np.array([0.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        img.switch_coords_to("uniform")


def test_extent_helpers_non_uniform() -> None:
    """``_compute_x{min,max}`` and ``_compute_y{min,max}`` use the coord arrays
    when the image is in non-uniform mode."""
    img = _make_uniform_image()
    img.switch_coords_to("non-uniform")
    assert pytest.approx(img.xmin) == img.xcoords[0]
    assert pytest.approx(img.xmax) == img.xcoords[-1]
    assert pytest.approx(img.ymin) == img.ycoords[0]
    assert pytest.approx(img.ymax) == img.ycoords[-1]


def test_extent_helpers_no_data() -> None:
    """Extent helpers must return 0.0 for empty/None data instead of raising."""
    img = ImageObj()
    # Default data is None; the computed properties should not raise
    assert img.xmin == 0.0
    assert img.xmax == 0.0
    assert img.ymin == 0.0
    assert img.ymax == 0.0


def test_extent_helpers_non_uniform_empty_coords() -> None:
    """Empty coord arrays in non-uniform mode produce NaN extents."""
    img = _make_uniform_image()
    img.is_uniform_coords = False
    img.xcoords = np.array([], dtype=float)
    img.ycoords = np.array([], dtype=float)
    assert np.isnan(img.xmin)
    assert np.isnan(img.xmax)
    assert np.isnan(img.ymin)
    assert np.isnan(img.ymax)


# ----------------------------------------------------------------------------
# DICOM template metadata population
# ----------------------------------------------------------------------------


class _FakeDicom:
    """Minimal stand-in for a DICOM dataset (no ``pydicom`` dependency)."""

    ImagePositionPatient = (1.5, 2.5)
    PixelSpacing = (0.1, 0.2)
    PatientName = "John Doe"
    StudyDescription = "Test"
    EmptyField = ""  # Falsy: must be skipped by __set_metadata_from
    GroupLength = 42  # Reserved DICOM-internal name: must be skipped


def test_dicom_template_sets_coords_and_metadata() -> None:
    """Assigning a DICOM-like template configures pixel spacing / origin and
    populates the image metadata with non-empty, non-callable attributes."""
    img = _make_uniform_image()
    img.dicom_template = _FakeDicom()
    # Origin and spacing taken from DICOM tags
    assert pytest.approx(img.x0) == 1.5
    assert pytest.approx(img.y0) == 2.5
    assert pytest.approx(img.dx) == 0.1
    assert pytest.approx(img.dy) == 0.2
    # Metadata populated, GroupLength and empty fields skipped
    assert img.metadata.get("PatientName") == "John Doe"
    assert img.metadata.get("StudyDescription") == "Test"
    assert "GroupLength" not in img.metadata
    assert "EmptyField" not in img.metadata
    # Template stored
    assert img.dicom_template is not None


def test_dicom_template_none_is_no_op() -> None:
    """Setting the template to ``None`` keeps the previous coordinates."""
    img = _make_uniform_image()
    img.dicom_template = None
    assert img.dicom_template is None


def test_dicom_template_missing_image_position() -> None:
    """When ``ImagePositionPatient`` is missing, both ``x0`` and ``y0`` must
    fall back to ``0.0`` (regression test for an operator-precedence bug
    that used to leave ``x0`` set to the bare ``0.0`` scalar while ``y0``
    received the intended fallback)."""

    class _NoIPP:
        """DICOM template stub exposing only ``PixelSpacing``."""

        PixelSpacing = (0.5, 0.25)

    img = _make_uniform_image()
    img.dicom_template = _NoIPP()
    assert img.x0 == 0.0
    assert img.y0 == 0.0
    assert pytest.approx(img.dx) == 0.5
    assert pytest.approx(img.dy) == 0.25


def test_dicom_template_missing_pixel_spacing() -> None:
    """When ``PixelSpacing`` is missing, both ``dx`` and ``dy`` must fall back
    to ``1.0`` (regression test for the same operator-precedence bug)."""

    class _NoPxS:
        """DICOM template stub exposing only ``ImagePositionPatient``."""

        ImagePositionPatient = (3.0, -4.0)

    img = _make_uniform_image()
    img.dicom_template = _NoPxS()
    assert pytest.approx(img.x0) == 3.0
    assert pytest.approx(img.y0) == -4.0
    assert img.dx == 1.0
    assert img.dy == 1.0


def test_dicom_template_missing_both_attributes() -> None:
    """When both DICOM attributes are missing, defaults ``(0,0)`` and ``(1,1)``
    must be applied to the coordinates."""

    class _Empty:
        """DICOM template stub exposing neither attribute."""

    img = _make_uniform_image()
    img.dicom_template = _Empty()
    assert img.x0 == 0.0
    assert img.y0 == 0.0
    assert img.dx == 1.0
    assert img.dy == 1.0


# ----------------------------------------------------------------------------
# _repr_html_ (Jupyter rich representation)
# ----------------------------------------------------------------------------


def test_repr_html_uniform_with_units_and_roi() -> None:
    """``_repr_html_`` produces a non-empty HTML string covering shape, dtype,
    extent, axis units, ROI count and pixel spacing for uniform images."""
    img = _make_uniform_image()
    img.xlabel = "Position"
    img.ylabel = "Position"
    img.zlabel = "Intensity"
    img.xunit = img.yunit = "mm"
    img.zunit = "a.u."
    # Add a single rectangular ROI to exercise the ROI section
    roi = sigima.objects.create_image_roi(
        "rectangle", [0.0, 0.0, img.dx * 2, img.dy * 2], indices=False
    )
    img.roi = roi
    html = img._repr_html_()  # pylint: disable=protected-access
    assert "ImageObj" in html and img.title in html
    # Shape and dtype info
    assert f"{img.data.shape[1]} ×" in html
    assert "float64" in html
    # Units injected in axis labels
    assert "Position" in html and "Intensity" in html and "mm" in html
    # ROI count is rendered
    assert "ROIs:" in html


def test_repr_html_non_uniform() -> None:
    """``_repr_html_`` covers the non-uniform branch, including coord ranges."""
    img = _make_uniform_image()
    img.switch_coords_to("non-uniform")
    html = img._repr_html_()  # pylint: disable=protected-access
    assert "coords" in html
    # Bounds should reference first/last coord values
    assert f"{img.xcoords[0]:.4g}" in html
    assert f"{img.xcoords[-1]:.4g}" in html


if __name__ == "__main__":
    pytest.main([__file__])
