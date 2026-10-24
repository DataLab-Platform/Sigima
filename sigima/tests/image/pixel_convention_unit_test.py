# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Pixel center convention unit tests
----------------------------------

The center of pixel ``i`` is at ``x0 + i * dx`` (or ``xcoords[i]``): ROIs select the
pixels whose centers they contain.
"""

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image as sipi

#: Non-square image with anisotropic pixels and an offset origin
NROWS, NCOLS = 6, 8
X0, Y0, DX, DY = 10.0, -5.0, 2.0, 0.5
#: Non-uniform coordinates of the same shape
XCOORDS = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0])
YCOORDS = np.array([100.0, 101.0, 103.0, 106.0, 110.0, 115.0])


def make_image(uniform: bool = True) -> sigima.objects.ImageObj:
    """Return a zero image with uniform or non-uniform coordinates."""
    image = sigima.objects.create_image("image", np.zeros((NROWS, NCOLS)))
    if uniform:
        image.set_uniform_coords(DX, DY, X0, Y0)
    else:
        image.set_coords(XCOORDS.copy(), YCOORDS.copy())
    return image


def pixel_center(image: sigima.objects.ImageObj, row: int, col: int) -> tuple:
    """Return the physical coordinates of the center of a pixel."""
    x = float(image.axis_to_physical([col], "x")[0])
    y = float(image.axis_to_physical([row], "y")[0])
    return x, y


def pixel_edges(image: sigima.objects.ImageObj, i0: int, i1: int, axis: str) -> list:
    """Return the outer edges of pixels i0 to i1 (inclusive) along an axis."""
    return image.axis_to_physical([i0 - 0.5, i1 + 0.5], axis).tolist()


def _marker_image(uniform: bool, row: int = 1, col: int = 2) -> sigima.objects.ImageObj:
    """Return an image with a single bright pixel."""
    image = make_image(uniform)
    image.data[row, col] = 1.0
    return image


def _marker_index(image: sigima.objects.ImageObj) -> tuple[int, int]:
    """Return the (row, column) of the bright pixel."""
    # pylint: disable-next=unbalanced-tuple-unpacking
    row, col = np.unravel_index(np.argmax(image.data), image.data.shape)
    return int(row), int(col)


def test_xc_yc_are_the_center_of_the_pixel_grid() -> None:
    """``xc``/``yc`` are midway between the first and last pixel centers."""
    image = make_image()
    assert image.xc == X0 + (NCOLS - 1) * DX / 2
    assert image.yc == Y0 + (NROWS - 1) * DY / 2
    image = make_image(uniform=False)
    assert image.xc == (XCOORDS[0] + XCOORDS[-1]) / 2
    assert image.yc == (YCOORDS[0] + YCOORDS[-1]) / 2


@pytest.mark.parametrize(
    "bounds, expected",
    [
        ((0.5, 3.5), [1, 2, 3]),  # on pixel edges: the pixels displayed inside
        ((0.4, 3.4), [1, 2, 3]),  # regression: pixels 0 to 2 were selected
        ((0.6, 3.6), [1, 2, 3]),
        ((1.0, 4.0), [1, 2, 3, 4]),  # on pixel centers: bounds included
        ((1.0 + 1e-12, 4.0 - 1e-12), [1, 2, 3, 4]),  # rounding noise
        ((-3.0, 0.4), [0]),  # partly outside the image
        ((8.0, 9.0), []),  # outside the image
    ],
)
def test_rectangle_selects_pixels_whose_centers_are_inside(bounds, expected) -> None:
    """A rectangular ROI selects the pixels whose centers lie inside it."""
    image = sigima.objects.create_image("image", np.zeros((4, 8)))
    xmin, xmax = bounds
    image.roi = sigima.objects.create_image_roi(
        "rectangle", [xmin, 0.0, xmax - xmin, 3.0], indices=False
    )
    columns = np.where(~image.roi.to_mask(image)[0])[0].tolist()
    assert columns == expected


def test_rectangle_roi_in_indices_covers_its_pixels() -> None:
    """A rectangular ROI in indices covers its pixels up to their outer edges."""
    for uniform in (True, False):
        image = make_image(uniform)
        roi = sigima.objects.create_image_roi("rectangle", [1, 2, 3, 2], indices=True)
        single_roi = roi.get_single_roi(0)
        x0, y0, width, height = single_roi.get_physical_coords(image)
        assert [x0, x0 + width] == pytest.approx(pixel_edges(image, 1, 3, "x"))
        assert [y0, y0 + height] == pytest.approx(pixel_edges(image, 2, 3, "y"))
        expected = np.ones((NROWS, NCOLS), dtype=bool)
        expected[2:4, 1:4] = False
        assert np.array_equal(roi.to_mask(image), expected)
        physical = sigima.objects.create_image_roi(
            "rectangle", [x0, y0, width, height], indices=False
        )
        assert np.array_equal(physical.to_mask(image), expected)
        assert physical.get_single_roi(0).get_indices_coords(image) == [1, 2, 3, 2]


@pytest.mark.parametrize("uniform", [True, False])
def test_extract_roi_keeps_pixel_positions(uniform: bool) -> None:
    """Extracted pixels keep their physical positions.

    Regression: the origin of the extracted image was the source origin plus the
    (absolute) ROI position, and non-uniform coordinates were not cropped.
    """
    src = make_image(uniform)
    src.data[:] = np.arange(src.data.size).reshape(src.data.shape)
    x_edges, y_edges = pixel_edges(src, 2, 5, "x"), pixel_edges(src, 1, 3, "y")
    param = sigima.objects.ROI2DParam.create(
        geometry="rectangle",
        x0=x_edges[0],
        y0=y_edges[0],
        dx=x_edges[1] - x_edges[0],
        dy=y_edges[1] - y_edges[0],
    )
    for dst in (sipi.extract_roi(src, param), sipi.extract_rois(src, [param])):
        assert np.array_equal(dst.data, src.data[1:4, 2:6])
        assert pixel_center(dst, 0, 0) == pytest.approx(pixel_center(src, 1, 2))
        assert pixel_center(dst, 2, 3) == pytest.approx(pixel_center(src, 3, 5))


def test_roi_grid_selects_every_pixel_once() -> None:
    """The cells of a full-size ROI grid tile the image exactly."""
    src = make_image()
    roi = sigima.proc.image.generate_image_grid_roi(
        src, sigima.params.ROIGridParam.create(nx=2, ny=3, xsize=100, ysize=100)
    )
    counts = np.zeros(src.data.shape, dtype=int)
    for single_roi in roi.single_rois:
        counts += ~single_roi.to_mask(src)
    assert np.all(counts == 1)


def test_roi_geometry_results_use_the_extracted_pixels() -> None:
    """Coordinates computed in a ROI are those of the analyzed pixels.

    Regression: they were offset by the ROI bound instead of the first pixel of
    the ROI, i.e. by up to half a pixel.
    """
    src = _marker_image(True, row=3, col=5)
    expected = sipi.centroid(src).coords[0]
    x_edges, y_edges = pixel_edges(src, 3, 6, "x"), pixel_edges(src, 2, 4, "y")
    src.roi = sigima.objects.create_image_roi(
        "rectangle",
        [x_edges[0] + 0.3, y_edges[0] + 0.1, x_edges[1] - x_edges[0], 1.2],
    )
    assert sipi.centroid(src).coords[0] == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
