# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry results of images with non-uniform coordinates
-------------------------------------------------------

Analysis results (centroid, peaks, blobs, contours...) are computed in pixel
indices and converted to physical coordinates by
:func:`sigima.proc.image.base.compute_geometry_from_obj`: for non-uniform
coordinates, the conversion follows the pixel centers (``xcoords``, ``ycoords``)
instead of ``dx``, ``dy``, ``x0`` and ``y0``, which do not describe the image.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image as sipi
from sigima.objects import KindShape
from sigima.proc.image.base import compute_geometry_from_obj

NROWS, NCOLS = 32, 40
#: Irregular, strictly increasing pixel centers
XCOORDS = 10.0 + np.cumsum(np.linspace(0.5, 2.0, NCOLS))
YCOORDS = -5.0 + np.cumsum(np.linspace(1.5, 0.25, NROWS))
#: Uniform coordinates (anisotropic pixels, offset origin)
DX, DY, X0, Y0 = 2.0, 0.5, 10.0, -5.0


def _blob_image() -> np.ndarray:
    """Return an image holding a 3x3 square centered on row 20, column 25."""
    data = np.zeros((NROWS, NCOLS))
    data[19:22, 24:27] = 1.0
    return data


def _nonuniform_image(data: np.ndarray) -> sigima.objects.ImageObj:
    """Return an image with irregular non-uniform coordinates."""
    obj = sigima.objects.create_image("nonuniform", data)
    obj.set_coords(XCOORDS.copy(), YCOORDS.copy())
    return obj


def _uniform_pair(
    data: np.ndarray,
) -> tuple[sigima.objects.ImageObj, sigima.objects.ImageObj]:
    """Return an image with uniform coordinates, and the same image with the
    equivalent non-uniform coordinates."""
    uniform = sigima.objects.create_image("uniform", data)
    uniform.set_uniform_coords(DX, DY, X0, Y0)
    nonuniform = uniform.copy()
    nonuniform.switch_coords_to("non-uniform")
    return uniform, nonuniform


@pytest.mark.parametrize("with_roi", [False, True])
def test_centroid_follows_nonuniform_coordinates(with_roi: bool) -> None:
    """The centroid is given at the physical position of the pixel holding it.

    Regression: it was given in pixel indices (``dx``, ``dy``, ``x0`` and ``y0``
    keep their default values for non-uniform coordinates).
    """
    obj = _nonuniform_image(_blob_image())
    if with_roi:
        obj.roi = sigima.objects.create_image_roi(
            "rectangle",
            [
                XCOORDS[20] + 0.1,
                YCOORDS[15] + 0.1,
                XCOORDS[30] - XCOORDS[20],
                YCOORDS[25] - YCOORDS[15],
            ],
        )
    result = sipi.centroid(obj)
    assert result.coords[0] == pytest.approx((XCOORDS[25], YCOORDS[20]))


def test_centroid_ignores_stale_uniform_calibration() -> None:
    """Uniform parameters left over by a switch to non-uniform coordinates are
    not used (regression: the result followed the former calibration)."""
    obj = sigima.objects.create_image("stale", _blob_image())
    obj.set_uniform_coords(0.5, 0.25, 10.0, 20.0)
    obj.switch_coords_to("non-uniform")
    obj.set_coords(obj.xcoords**1.2, obj.ycoords**1.1)
    result = sipi.centroid(obj)
    assert result.coords[0] == pytest.approx((obj.xcoords[25], obj.ycoords[20]))


def test_peak_detection_follows_nonuniform_coordinates() -> None:
    """Detected peaks are given at the physical positions of their pixels."""
    yy, xx = np.mgrid[0:NROWS, 0:NCOLS]
    data = np.zeros((NROWS, NCOLS))
    for row, col in ((8, 10), (22, 30)):
        data += np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / 4.0)
    param = sigima.params.Peak2DDetectionParam.create(size=5)
    indices = sipi.peak_detection(sigima.objects.create_image("i", data), param)
    result = sipi.peak_detection(_nonuniform_image(data), param)
    assert len(indices.coords) == 2
    expected = [(XCOORDS[int(col)], YCOORDS[int(row)]) for col, row in indices.coords]
    np.testing.assert_allclose(result.coords, expected)


#: Results in pixel indices of each kind of shape (polygons padded with NaN)
SHAPES = {
    KindShape.POINT: np.array([[3.0, 4.5], [10.25, 7.0]]),
    KindShape.SEGMENT: np.array([[1.0, 2.0, 6.5, 3.5]]),
    KindShape.CIRCLE: np.array([[5.0, 6.0, 2.5]]),
    KindShape.ELLIPSE: np.array([[5.0, 6.0, 2.5, 1.5, 0.3]]),
    KindShape.POLYGON: np.array([[1.0, 2.0, 5.0, 2.0, 5.0, 6.0, np.nan, np.nan]]),
}


@pytest.mark.parametrize("with_roi", [False, True])
@pytest.mark.parametrize("kind", list(SHAPES))
def test_nonuniform_conversion_matches_uniform(kind: KindShape, with_roi: bool):
    """Non-uniform coordinates describing a uniform grid give the same results as
    the uniform coordinates, for every kind of shape, also in a ROI (whose first
    pixel offsets the coordinates)."""
    data = np.zeros((NROWS, NCOLS))
    images = _uniform_pair(data)
    if with_roi:
        for obj in images:
            obj.roi = sigima.objects.create_image_roi(
                "rectangle", [X0 + 3.3 * DX, Y0 + 2.2 * DY, 10.0 * DX, 8.0 * DY]
            )
            assert obj.get_roi_crop_indices(0)[:2] == (4, 3)
    results = [
        compute_geometry_from_obj("t", kind, obj, lambda _data: SHAPES[kind].copy())
        for obj in images
    ]
    np.testing.assert_allclose(
        results[1].coords, results[0].coords, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    if kind == KindShape.POLYGON:
        assert np.isnan(results[1].coords[0, -2:]).all()


def test_nonuniform_distances_use_the_local_pixel_size() -> None:
    """Radii and semi-axes are scaled by the local pixel size at the shape center,
    and angles are kept."""
    obj = _nonuniform_image(np.zeros((NROWS, NCOLS)))
    col, row = 10, 12
    result = compute_geometry_from_obj(
        "ellipse",
        KindShape.ELLIPSE,
        obj,
        lambda _data: np.array([[float(col), float(row), 3.0, 2.0, 0.4]]),
    )
    size_x = (XCOORDS[col + 1] - XCOORDS[col - 1]) / 2
    size_y = (YCOORDS[row + 1] - YCOORDS[row - 1]) / 2
    scale = (size_x + size_y) / 2
    expected = [XCOORDS[col], YCOORDS[row], 3.0 * scale, 2.0 * scale, 0.4]
    np.testing.assert_allclose(result.coords[0], expected)


def test_uniform_conversion_is_unchanged() -> None:
    """Results of images with uniform coordinates are exactly those of the former
    conversion (pixel size and origin, then offset of the first pixel of the
    ROI)."""
    dx, dy, x0, y0 = 0.3, 0.7, -1.1, 2.9
    obj = sigima.objects.create_image("u", np.zeros((NROWS, NCOLS)))
    obj.set_uniform_coords(dx, dy, x0, y0)
    obj.roi = sigima.objects.create_image_roi(
        "rectangle", [x0 + 2.6 * dx, y0 + 5.2 * dy, 9.0 * dx, 7.0 * dy]
    )
    pixels = np.array([[3.3, 4.7, 1.9]])
    result = compute_geometry_from_obj(
        "circle", KindShape.CIRCLE, obj, lambda _data: pixels.copy()
    )
    expected = pixels.copy()
    expected[:, 0] = dx * expected[:, 0] + x0
    expected[:, 1] = dy * expected[:, 1] + y0
    expected[:, 2] *= (dx + dy) / 2
    ix0, iy0, _ix1, _iy1 = obj.get_roi_crop_indices(0)
    assert (ix0, iy0) != (0, 0)
    expected[:, 0] += ix0 * dx
    expected[:, 1] += iy0 * dy
    assert np.array_equal(result.coords, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
