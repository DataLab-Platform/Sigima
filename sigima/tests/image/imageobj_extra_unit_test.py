# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :class:`sigima.objects.ImageObj` helpers.

Covers the ``to_builtin`` recursive conversion helper exposed by
:mod:`sigima.objects.image.roi`, the ``check_points`` validator for ROI
coordinates, the ``create_image_parameters`` factory and ``NewImageParam``
defaults.

The ``to_builtin`` helper from :mod:`sigima.objects.image.object` is covered
by ``imageobj_metadata_unit_test.py``.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects.image import roi as img_roi_mod
from sigima.objects.image.creation import (
    ImageDatatypes,
    ImageTypes,
    NewImageParam,
    create_image_parameters,
)

# ===========================================================================
# objects/image/roi.py - to_builtin helper variants and check_points
# ===========================================================================


def test_image_roi_to_builtin_numeric() -> None:
    """Plain ``int`` and ``float`` values are returned unchanged by
    ``to_builtin`` (identity for builtin numeric types)."""
    assert img_roi_mod.to_builtin(3) == 3
    assert img_roi_mod.to_builtin(2.5) == 2.5


def test_image_roi_to_builtin_bytestring() -> None:
    """Byte strings are decoded to ``str`` so they are JSON/HDF5-friendly."""
    out = img_roi_mod.to_builtin(b"abc")
    assert isinstance(out, str) and "abc" in out


def test_image_roi_to_builtin_sequence_and_mapping() -> None:
    """Lists and dicts pass through ``to_builtin`` unchanged (only
    individual elements are recursively converted)."""
    assert img_roi_mod.to_builtin([1, 2, 3]) == [1, 2, 3]
    assert img_roi_mod.to_builtin({"a": 1}) == {"a": 1}


def test_image_roi_to_builtin_ndarray_and_unsupported() -> None:
    """NumPy arrays are kept as-is, while truly opaque objects (no usable
    conversion path) return ``None``."""
    arr = np.array([1.0, 2.0])
    out = img_roi_mod.to_builtin(arr)
    assert isinstance(out, np.ndarray)

    class Opaque:
        """Sentinel class with no ``__dict__`` and no recognised conversion path."""

    assert img_roi_mod.to_builtin(Opaque()) is None


def test_check_points_valid() -> None:
    """A flat ``float64`` array of even length (pairs of x/y) passes
    ``check_points`` validation."""
    arr = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    assert img_roi_mod.check_points(arr) is True


def test_check_points_invalid() -> None:
    """``check_points`` rejects: integer dtype (``TypeError``), 2D arrays
    (``ValueError``: must be 1D) and odd-length arrays (``ValueError``:
    coords come in pairs)."""
    bad_dtype = np.array([0, 0, 1, 1], dtype=np.int32)
    assert img_roi_mod.check_points(bad_dtype) is False
    with pytest.raises(TypeError):
        img_roi_mod.check_points(bad_dtype, raise_exception=True)
    bad_dim = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    assert img_roi_mod.check_points(bad_dim) is False
    with pytest.raises(ValueError, match="1D array"):
        img_roi_mod.check_points(bad_dim, raise_exception=True)
    bad_len = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    assert img_roi_mod.check_points(bad_len) is False
    with pytest.raises(ValueError, match="pairs"):
        img_roi_mod.check_points(bad_len, raise_exception=True)


# ===========================================================================
# objects/image/creation.py - create_image_parameters branches
# ===========================================================================


def test_create_image_parameters_all_fields() -> None:
    """All optional fields (size, dtype, axis labels and units) are
    propagated onto the parameter object returned by
    ``create_image_parameters``."""
    p = create_image_parameters(
        ImageTypes.ZEROS,
        title="my",
        height=128,
        width=64,
        idtype=ImageDatatypes.UINT8,
        xlabel="x",
        ylabel="y",
        zlabel="z",
        xunit="u",
        yunit="v",
        zunit="w",
    )
    assert p.title == "my"
    assert p.height == 128
    assert p.width == 64
    assert p.dtype == ImageDatatypes.UINT8
    assert p.xlabel == "x"
    assert p.ylabel == "y"
    assert p.zlabel == "z"
    assert p.xunit == "u"
    assert p.yunit == "v"
    assert p.zunit == "w"


def test_image_datatypes_from_numpy_unknown_falls_back_uint8() -> None:
    """NumPy dtypes that have no direct ``ImageDatatypes`` mapping (here:
    ``complex128``) fall back to ``UINT8`` rather than raising."""
    out = ImageDatatypes.from_numpy_dtype(np.dtype("complex128"))
    assert out == ImageDatatypes.UINT8


def test_create_image_parameters_unknown_type_raises() -> None:
    """Passing an unknown image-type marker raises ``ValueError`` instead
    of silently returning a wrong parameter class."""

    class FakeType:
        """Sentinel type that is not a member of ``ImageTypes``."""

    with pytest.raises(ValueError):
        create_image_parameters(FakeType())  # type: ignore[arg-type]


def test_new_image_param_default_generate_2d_data() -> None:
    """``NewImageParam.generate_2d_data`` produces an array of the
    requested shape with the default ``float64`` dtype."""
    p = NewImageParam()
    arr = p.generate_2d_data((4, 5))
    assert arr.shape == (4, 5)
    assert arr.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
