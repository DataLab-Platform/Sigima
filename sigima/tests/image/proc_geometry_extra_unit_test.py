# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :mod:`sigima.proc.image.geometry`.

Covers:

* XYZ polynomial calibration on the X, Y and Z axes (uniform and
  non-uniform coordinates).
* ``Resampling2DParam.update_from_obj`` defaults and validation paths
  (missing ``dx/dy`` / ``width/height``, non-uniform coordinates).
* ``resampling`` interpolation methods (linear, cubic, nearest) and
  single-pixel output.
* ``resize`` rejection of non-uniform coordinates.
* ``transpose`` with non-uniform coordinates and unit/label swap.
* ``set_uniform_coords`` and ``UniformCoordsParam.update_from_obj``.
* ``translate`` on non-uniform coordinates.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

import sigima.params
import sigima.proc.image.geometry as igeom
from sigima.objects import ImageObj, create_image
from sigima.proc.image.geometry import (
    Interpolation2DMethod,
    Resampling2DParam,
    TranslateParam,
    UniformCoordsParam,
    resampling,
    translate,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _img(width: int = 32, height: int = 32) -> ImageObj:
    """Build a square float64 image with linearly increasing pixel values."""
    data = np.arange(height * width, dtype=np.float64).reshape(height, width)
    return create_image("img", data)


def _basic_image(shape: tuple[int, int] = (5, 6)) -> ImageObj:
    """Build a small bare ``ImageObj`` with ramp data and no axis coords."""
    img = ImageObj()
    img.data = np.arange(shape[0] * shape[1], dtype=float).reshape(shape)
    return img


def _make_image(shape: tuple[int, int] = (8, 10)) -> ImageObj:
    """Build an ``ImageObj`` with ramp data and unit pixel spacing."""
    img = ImageObj()
    img.data = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
    img.set_uniform_coords(1.0, 1.0, 0.0, 0.0)
    return img


# ===========================================================================
# XYZ polynomial calibration
# ===========================================================================


def test_calibration_axis_z() -> None:
    """Calibrating the ``z`` axis applies the polynomial directly to the
    pixel values (here a simple linear ``a0 + a1*z`` mapping)."""
    img = _basic_image()
    p = sigima.params.XYZCalibrateParam.create(axis="z", a0=1.0, a1=2.0, a2=0.0, a3=0.0)
    out = igeom.calibration(img, p)
    assert np.allclose(out.data, 1.0 + 2.0 * img.data)


def test_calibration_axis_x_uniform() -> None:
    """Calibrating the ``x`` axis on a uniform-coords image populates
    ``xcoords`` with one value per column."""
    img = _basic_image()
    p = sigima.params.XYZCalibrateParam.create(axis="x", a0=0.0, a1=2.0, a2=0.5, a3=0.1)
    out = igeom.calibration(img, p)
    assert out.xcoords is not None
    assert out.xcoords.shape == (img.data.shape[1],)


def test_calibration_axis_y_uniform() -> None:
    """Calibrating the ``y`` axis on a uniform-coords image populates
    ``ycoords`` with one value per row."""
    img = _basic_image()
    p = sigima.params.XYZCalibrateParam.create(axis="y", a0=0.0, a1=2.0, a2=0.5, a3=0.0)
    out = igeom.calibration(img, p)
    assert out.ycoords is not None
    assert out.ycoords.shape == (img.data.shape[0],)


def test_calibration_axis_x_non_uniform() -> None:
    """X-axis calibration on an image with non-uniform coordinates applies
    the polynomial element-wise to the existing ``xcoords`` array."""
    img = _basic_image()
    img.set_coords(
        np.linspace(0.0, 5.0, img.data.shape[1]) ** 2,
        np.arange(img.data.shape[0], dtype=float),
    )
    p = sigima.params.XYZCalibrateParam.create(axis="x", a0=1.0, a1=1.0, a2=0.0, a3=0.0)
    out = igeom.calibration(img, p)
    assert out.xcoords is not None
    assert np.allclose(out.xcoords, 1.0 + img.xcoords)


def test_calibration_axis_y_non_uniform() -> None:
    """Y-axis calibration on an image with non-uniform coordinates applies
    the polynomial element-wise to the existing ``ycoords`` array."""
    img = _basic_image()
    img.set_coords(
        np.arange(img.data.shape[1], dtype=float),
        np.linspace(0.0, 4.0, img.data.shape[0]) ** 2,
    )
    p = sigima.params.XYZCalibrateParam.create(axis="y", a0=1.0, a1=1.0, a2=0.0, a3=0.0)
    out = igeom.calibration(img, p)
    assert out.ycoords is not None
    assert np.allclose(out.ycoords, 1.0 + img.ycoords)


# ===========================================================================
# Resampling - shape / dxy / missing / non-uniform
# ===========================================================================


def test_resampling_shape_mode() -> None:
    """In ``shape`` mode, ``width``/``height`` directly drive the output
    image dimensions."""
    obj = _img(20, 16)
    p = sigima.params.Resampling2DParam.create(width=10, height=8, mode="shape")
    out = igeom.resampling(obj, p)
    assert out.data.shape == (8, 10)


def test_resampling_dxy_mode() -> None:
    """In ``dxy`` mode the output dimensions are derived from ``dx``/``dy``
    and must be non-empty."""
    obj = _img(20, 16)
    p = sigima.params.Resampling2DParam.create(dx=2.0, dy=2.0, mode="dxy")
    out = igeom.resampling(obj, p)
    assert out.data.shape[0] > 0 and out.data.shape[1] > 0


def test_resampling_dxy_mode_missing_raises() -> None:
    """``dxy`` mode without ``dx``/``dy`` is a configuration error and
    must raise instead of silently using defaults."""
    obj = _img()
    p = sigima.params.Resampling2DParam.create(mode="dxy")
    with pytest.raises(ValueError):
        igeom.resampling(obj, p)


def test_resampling_shape_mode_missing_raises() -> None:
    """``shape`` mode without ``width``/``height`` is a configuration
    error and must raise."""
    obj = _img()
    p = sigima.params.Resampling2DParam.create(mode="shape")
    with pytest.raises(ValueError):
        igeom.resampling(obj, p)


def test_resampling_non_uniform_raises() -> None:
    """Resampling assumes uniform sampling; non-uniform coordinates must
    be rejected to avoid producing visually misleading output."""
    obj = _img()
    obj.set_coords(
        xcoords=np.linspace(0.0, 1.0, obj.data.shape[1]) ** 2,
        ycoords=np.linspace(0.0, 1.0, obj.data.shape[0]) ** 2,
    )
    p = sigima.params.Resampling2DParam.create(width=10, height=8, mode="shape")
    with pytest.raises(ValueError):
        igeom.resampling(obj, p)


def test_resize_non_uniform_raises() -> None:
    """``resize`` (zoom-based) similarly requires uniform coordinates and
    rejects non-uniform input."""
    obj = _img()
    obj.set_coords(
        xcoords=np.linspace(0.0, 1.0, obj.data.shape[1]) ** 2,
        ycoords=np.linspace(0.0, 1.0, obj.data.shape[0]) ** 2,
    )
    p = sigima.params.ResizeParam.create(zoom=2.0)
    with pytest.raises(ValueError):
        igeom.resize(obj, p)


def test_transpose_with_non_uniform_coords() -> None:
    """``transpose`` swaps shape, axis units and labels even when the
    image carries non-uniform coordinates."""
    obj = _img(20, 16)
    obj.xunit = "m"
    obj.yunit = "s"
    obj.xlabel = "X"
    obj.ylabel = "Y"
    obj.set_coords(
        xcoords=np.linspace(0.0, 1.0, obj.data.shape[1]) ** 2,
        ycoords=np.linspace(0.0, 2.0, obj.data.shape[0]) ** 2,
    )
    out = igeom.transpose(obj)
    assert out.data.shape == (obj.data.shape[1], obj.data.shape[0])
    assert out.xunit == "s"
    assert out.yunit == "m"


# ===========================================================================
# UniformCoordsParam / set_uniform_coords
# ===========================================================================


def test_set_uniform_coords_from_non_uniform() -> None:
    """``set_uniform_coords`` rebuilds a uniform grid from non-uniform
    coordinates while keeping the origin at zero (regression: ensure
    ``UniformCoordsParam.update_from_obj`` does not crash on non-uniform)."""
    obj = _img(20, 16)
    obj.set_coords(
        xcoords=np.linspace(0.0, 1.0, obj.data.shape[1]),
        ycoords=np.linspace(0.0, 2.0, obj.data.shape[0]),
    )
    p = sigima.params.UniformCoordsParam()
    p.update_from_obj(obj)
    assert p.x0 == pytest.approx(0.0)
    out = igeom.set_uniform_coords(obj, p)
    assert out.is_uniform_coords


def test_uniform_coords_from_obj_when_already_uniform() -> None:
    """On an already-uniform image, ``UniformCoordsParam.update_from_obj``
    simply mirrors the image's pixel spacing."""
    obj = _img(8, 8)
    p = sigima.params.UniformCoordsParam()
    p.update_from_obj(obj)
    assert p.dx == obj.dx


def test_uniform_coords_param_update_from_non_uniform() -> None:
    """On a non-uniform image, ``UniformCoordsParam.update_from_obj``
    initialises ``x0``/``y0`` from the first coord (here zero)."""
    img = _make_image()
    img.set_coords(
        np.array([0.0, 1.0, 2.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0]),
    )
    p = UniformCoordsParam()
    p.update_from_obj(img)
    assert p.x0 == 0.0
    assert p.y0 == 0.0


def test_uniform_coords_param_update_from_uniform() -> None:
    """On a uniform image, ``UniformCoordsParam.update_from_obj`` copies
    ``dx``/``dy`` from the source image."""
    img = _make_image()
    p = UniformCoordsParam()
    p.update_from_obj(img)
    assert p.dx == 1.0 and p.dy == 1.0


# ===========================================================================
# Resampling2DParam.update_from_obj defaults + validation
# ===========================================================================


def test_resampling_param_update_from_obj_fills_defaults() -> None:
    """``Resampling2DParam.update_from_obj`` populates every field
    (``xmin``/``xmax``, ``ymin``/``ymax``, ``dx``/``dy``,
    ``width``/``height``) from the source image so the GUI shows sensible
    defaults."""
    img = _make_image()
    p = Resampling2DParam()
    p.update_from_obj(img)
    assert p.xmin == img.x0
    assert p.xmax == img.x0 + img.width
    assert p.ymin == img.y0
    assert p.ymax == img.y0 + img.height
    assert p.dx == img.dx
    assert p.dy == img.dy
    assert p.width == img.data.shape[1]
    assert p.height == img.data.shape[0]


# ===========================================================================
# Translate / interpolation methods / single-pixel resample
# ===========================================================================


def test_translate_non_uniform_coords() -> None:
    """``translate`` works on images with non-uniform coordinates and
    returns a non-``None`` result (regression)."""
    img = _make_image()
    img.set_coords(
        np.array([0.0, 1.0, 2.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0]),
    )
    p = TranslateParam.create(dx=1.0, dy=2.0)
    result = translate(img, p)
    assert result is not None


def test_resampling_single_pixel_output() -> None:
    """Edge case: a target window smaller than ``dx``/``dy`` collapses to
    a single output pixel rather than raising."""
    img = _make_image()
    p = Resampling2DParam.create(
        mode="dxy", xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, dx=2.0, dy=2.0
    )
    result = resampling(img, p)
    assert result.data.shape == (1, 1)


def test_resampling_cubic_method() -> None:
    """Cubic interpolation is wired into ``resampling`` and produces
    output of the requested shape."""
    img = _make_image()
    p = Resampling2DParam.create(
        mode="shape", width=15, height=12, method=Interpolation2DMethod.CUBIC
    )
    result = resampling(img, p)
    assert result.data.shape == (12, 15)


def test_resampling_nearest_method() -> None:
    """Nearest-neighbour interpolation is wired into ``resampling`` and
    produces output of the requested shape."""
    img = _make_image()
    p = Resampling2DParam.create(
        mode="shape", width=20, height=16, method=Interpolation2DMethod.NEAREST
    )
    result = resampling(img, p)
    assert result.data.shape == (16, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
