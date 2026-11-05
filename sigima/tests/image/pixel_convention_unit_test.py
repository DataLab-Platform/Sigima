# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Pixel center convention unit tests
----------------------------------

The center of pixel ``i`` is at ``x0 + i * dx`` (or ``xcoords[i]``): ROIs select the
pixels whose centers they contain, and exact geometric operations (flips, quarter
turns, transposition) move ROIs, annotations and geometry results with the pixels.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from skimage import measure

import sigima.objects
import sigima.params
import sigima.proc.image as sipi
from sigima.objects import KindShape
from sigima.objects.scalar import GeometryResult
from sigima.proc.image.transformations import transformer

#: Exact image operations and their effect on a pixel array
OPERATIONS: dict[str, tuple[Callable, Callable[[np.ndarray], np.ndarray]]] = {
    "fliph": (sipi.fliph, np.fliplr),
    "flipv": (sipi.flipv, np.flipud),
    "transpose": (sipi.transpose, np.transpose),
    "rotate90": (sipi.rotate90, np.rot90),
    "rotate270": (sipi.rotate270, lambda array: np.rot90(array, 3)),
}

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


@pytest.mark.parametrize("uniform", [True, False])
def test_circle_selects_pixels_whose_centers_are_inside(uniform: bool) -> None:
    """A circular ROI selects the pixels whose centers lie inside it, boundary
    included.

    Regression: on non-uniform grids, the circle was drawn in index space (e.g. 2
    pixels instead of 4 for a circle of radius 1.3 around a pixel center).
    """
    image = make_image(uniform)
    x, y = np.meshgrid(
        image.axis_to_physical(np.arange(NCOLS), "x"),
        image.axis_to_physical(np.arange(NROWS), "y"),
    )
    xc, yc = pixel_center(image, 2, 3)
    for radius in (0.3, 1.3, 2.0, 3.7, 9.9):  # 2.0: pixel centers on the circle
        roi = sigima.objects.create_image_roi("circle", [xc, yc, radius])
        expected = (x - xc) ** 2 + (y - yc) ** 2 <= radius**2 * (1 + 1e-12)
        assert np.array_equal(~roi.to_mask(image), expected), radius


@pytest.mark.parametrize("uniform", [True, False])
def test_polygon_selects_pixels_whose_centers_are_inside(uniform: bool) -> None:
    """A polygonal ROI selects the pixels whose centers lie inside it, edges
    included.

    Regression: the vertices were rounded to the nearest pixel indices, so that
    pixels outside the polygon could be selected (and inside ones left out).
    """
    image = make_image(uniform)
    x, y = np.meshgrid(
        image.axis_to_physical(np.arange(NCOLS), "x"),
        image.axis_to_physical(np.arange(NROWS), "y"),
    )
    # Generic vertices: no pixel center on the edges
    xp = image.axis_to_physical([0.63, 6.37, 2.21], "x")
    yp = image.axis_to_physical([0.71, 1.29, 4.43], "y")
    vertices = np.column_stack([xp, yp])
    roi = sigima.objects.create_image_roi("polygon", np.ravel(vertices).tolist())
    points = np.column_stack([x.ravel(), y.ravel()])
    expected = measure.points_in_poly(points, vertices).reshape(x.shape)
    assert np.array_equal(~roi.to_mask(image), expected)
    # Vertices on pixel centers: the pixels on the edges belong to the polygon
    cells = [(1, 1), (1, 4), (3, 4), (3, 1)]
    corners = [pixel_center(image, row, col) for row, col in cells]
    roi = sigima.objects.create_image_roi("polygon", np.ravel(corners).tolist())
    expected = np.zeros((NROWS, NCOLS), dtype=bool)
    expected[1:4, 1:5] = True
    assert np.array_equal(~roi.to_mask(image), expected)
    if uniform:  # Diagonal edge through pixel centers (straight in index space)
        cells = [(1, 1), (1, 4), (3, 1)]
        corners = [pixel_center(image, row, col) for row, col in cells]
        roi = sigima.objects.create_image_roi("polygon", np.ravel(corners).tolist())
        expected[:] = False
        expected[1, 1:5] = expected[2, 1:3] = expected[3, 1] = True
        assert np.array_equal(~roi.to_mask(image), expected)


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


@pytest.mark.parametrize("uniform", [True, False])
def test_extract_roi_holding_no_pixel_is_refused(uniform: bool) -> None:
    """Extracting regions of interest holding no pixel is refused explicitly.

    Regression: an empty image was created silently.
    """
    src = make_image(uniform)
    (xa, ya), (xb, yb) = pixel_center(src, 1, 2), pixel_center(src, 2, 3)
    hx, hy = abs(xb - xa) / 2, abs(yb - ya) / 2
    create = sigima.objects.ROI2DParam.create
    no_pixel = [
        create(geometry="rectangle", x0=1e6, y0=1e6, dx=1.0, dy=1.0),
        # Between the centers of two columns
        create(geometry="rectangle", x0=xa + hx / 2, y0=ya - hy, dx=hx, dy=2 * hy),
        # No width, on a pixel center
        create(geometry="rectangle", x0=xa, y0=ya - hy, dx=0.0, dy=2 * hy),
        # Bounding box holding pixel centers, circle holding none
        create(
            geometry="circle",
            xc=xa + hx,
            yc=ya + hy,
            r=(max(hx, hy) + np.hypot(hx, hy)) / 2,
        ),
    ]
    for param in no_pixel:
        with pytest.raises(ValueError, match="holds no pixel"):
            sipi.extract_roi(src, param)
        with pytest.raises(ValueError, match="hold no pixel"):
            sipi.extract_rois(src, [param])
    # Regions of interest holding pixels are extracted with those holding none
    with_pixels = create(geometry="rectangle", x0=xa, y0=ya, dx=2 * hx, dy=2 * hy)
    assert sipi.extract_rois(src, no_pixel + [with_pixels]).data.size > 0


def test_extract_roi_grid_cells_holding_no_pixel() -> None:
    """The cells of a ROI grid narrower than a pixel hold no pixel: extracting
    them is refused, the other cells are extracted."""
    src = sigima.objects.create_image("i", np.arange(16.0).reshape(4, 4))
    grid = sipi.generate_image_grid_roi(src, sigima.params.ROIGridParam())
    params = grid.to_params(src)
    empty = [param.to_single_roi(src).to_mask(src).all() for param in params]
    assert any(empty) and not all(empty)
    for param, is_empty in zip(params, empty):
        if is_empty:
            with pytest.raises(ValueError, match="holds no pixel"):
                sipi.extract_roi(src, param)
        else:
            assert sipi.extract_roi(src, param).data.size > 0


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


def _rois(
    image: sigima.objects.ImageObj, curved: bool = True
) -> dict[str, sigima.objects.ImageROI]:
    """Return ROIs of various shapes and conventions for an image.

    Args:
        image: image object
        curved: whether to include circles and polygons (shapes that a non-affine
         mapping of the coordinates does not keep)
    """
    (xe0, xe1), (ye0, ye1) = (
        pixel_edges(image, 1, 3, "x"),
        pixel_edges(image, 1, 2, "y"),
    )
    (xc1, yc1), (xc4, yc3) = pixel_center(image, 1, 1), pixel_center(image, 3, 4)
    xg = 0.3 * xe0 + 0.7 * xe1  # generic bound (neither a center nor an edge)
    # Polygon vertices at generic positions (no pixel center on its edges)
    xp = image.axis_to_physical([0.63, 4.37, 2.21], "x")
    yp = image.axis_to_physical([0.71, 1.29, 4.43], "y")
    rois = {
        "rectangle on edges": [xe0, ye0, xe1 - xe0, ye1 - ye0],
        "rectangle on centers": [xc1, yc1, xc4 - xc1, yc3 - yc1],
        "rectangle generic": [xe0 + 0.1 * (xe1 - xe0), ye0, xg - xe0, ye1 - ye0],
    }
    if curved:
        rois["polygon"] = np.ravel(np.column_stack([xp, yp])).tolist()
    result = {}
    for name, coords in rois.items():
        geometry = name.split()[0]
        result[name] = sigima.objects.create_image_roi(geometry, coords)
        if name == "rectangle on edges":
            result["inverse rectangle"] = sigima.objects.create_image_roi(
                geometry, coords, inverse=True
            )
    result["rectangle in indices"] = sigima.objects.create_image_roi(
        "rectangle", [2, 1, 3, 4], indices=True
    )
    if curved:
        xc, yc = pixel_center(image, 2, 4)
        result["circle"] = sigima.objects.create_image_roi("circle", [xc, yc, 2.6])
    return result


@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_exact_operations_move_rois_with_pixels(operation: str, uniform: bool) -> None:
    """The ROI of the result selects the pixels coming from the source ROI.

    Regression: ROIs were mirrored about ``x0 + width / 2`` instead of the center of
    the pixel grid, shifting circles, polygons and rectangles drawn on pixel edges
    by one pixel (and non-uniform coordinates were ignored).

    On non-uniform grids, flips and quarter turns reverse the pixels along a
    non-uniform axis, which is not an affine mapping of the coordinates: circles
    and polygons are then moved approximately (documented limitation), and only
    checked for the transposition.
    """
    func, move = OPERATIONS[operation]
    src = make_image(uniform)
    curved = uniform or operation == "transpose"
    for name, roi in _rois(src, curved).items():
        src.roi = roi
        dst = func(src)
        expected = move(src.roi.to_mask(src))
        assert np.array_equal(dst.roi.to_mask(dst), expected), name
        assert dst.roi.single_rois[0].inverse == roi.single_rois[0].inverse, name


@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_exact_operations_move_annotations_with_pixels(
    operation: str, uniform: bool
) -> None:
    """An annotation on a pixel center stays on the center of the same pixel.

    Regression: annotations were shifted by one pixel by flips and quarter turns.
    """
    src = _marker_image(uniform)
    x, y = pixel_center(src, *_marker_index(src))
    src.add_graphical_annotation(sigima.objects.PointAnnotation(x=x, y=y))
    dst = OPERATIONS[operation][0](src)
    annotation = dst.get_graphical_annotations()[0]
    assert (annotation.x, annotation.y) == pytest.approx(
        pixel_center(dst, *_marker_index(dst))
    )


@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_geometry_results_move_with_pixels(operation: str, uniform: bool) -> None:
    """Geometry results move like the pixels on which they were computed."""
    src = _marker_image(uniform)
    x, y = pixel_center(src, *_marker_index(src))
    result = GeometryResult("marker", KindShape.POINT, np.array([[x, y]]))
    dst = OPERATIONS[operation][0](src)
    moved = transformer.transform_like_image(result, src, operation)
    assert tuple(moved.coords[0]) == pytest.approx(
        pixel_center(dst, *_marker_index(dst))
    )
    if uniform:
        # The centroid computed on the result image is the moved centroid
        centroid = transformer.transform_like_image(sipi.centroid(src), src, operation)
        assert centroid.coords == pytest.approx(sipi.centroid(dst).coords)


def test_ellipse_geometry_results_are_transformed() -> None:
    """Ellipse results (center, semi-axes, angle) follow exact operations."""
    src = make_image()
    ellipse = GeometryResult(
        "ellipse", KindShape.ELLIPSE, np.array([[12.0, -4.0, 3.0, 1.0, 0.25]])
    )
    moved = transformer.transform_like_image(ellipse, src, "fliph")
    x, y, a, b, theta = moved.coords[0]
    assert (x, y, a, b) == pytest.approx((2 * src.xc - 12.0, -4.0, 3.0, 1.0))
    assert np.tan(theta) == pytest.approx(np.tan(-0.25))
    moved = transformer.transform_like_image(ellipse, src, "transpose")
    x, y, a, b, theta = moved.coords[0]
    assert (x, y, a, b) == pytest.approx((-4.0, 12.0, 3.0, 1.0))
    assert np.tan(theta) == pytest.approx(np.tan(np.pi / 2 - 0.25))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
