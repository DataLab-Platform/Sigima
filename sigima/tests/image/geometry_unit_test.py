# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for geometry computation functions.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pytest
import scipy.ndimage as spi

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import get_test_image, iterate_noisy_images
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_image_translate() -> None:
    """Image translation test."""
    for dx, dy in [(10, 0), (0, 10), (-10, -10)]:
        compfunc = sigima.proc.image.translate
        execenv.print(f"*** Testing image translate: {compfunc.__name__}")
        ima1 = list(iterate_noisy_images(size=128))[0]
        ima2: sigima.objects.ImageObj = compfunc(
            ima1, sigima.params.TranslateParam.create(dx=dx, dy=dy)
        )
        check_scalar_result("Image X translation", ima2.x0, ima1.x0 + dx)
        check_scalar_result("Image Y translation", ima2.y0, ima1.y0 + dy)


def __generic_flip_check(compfunc: callable, expfunc: callable) -> None:
    """Generic flip check function."""
    execenv.print(f"*** Testing image flip: {compfunc.__name__}")
    for ima1 in iterate_noisy_images(size=128):
        execenv.print(f"  {compfunc.__name__}({ima1.data.dtype}): ", end="")
        ima2: sigima.objects.ImageObj = compfunc(ima1)
        check_array_result("Image flip", ima2.data, expfunc(ima1.data))


@pytest.mark.validation
def test_image_fliph() -> None:
    """Image horizontal flip test."""
    __generic_flip_check(sigima.proc.image.fliph, np.fliplr)


@pytest.mark.validation
def test_image_flipv() -> None:
    """Image vertical flip test."""
    __generic_flip_check(sigima.proc.image.flipv, np.flipud)


def __generic_rotate_check(
    func: Callable[[sigima.objects.ImageObj], sigima.objects.ImageObj],
) -> None:
    """Generic rotate check function."""
    angle = int(re.match(r"rotate(\d+)", func.__name__).group(1))
    execenv.print(f"*** Testing image {angle}° rotation:")
    for ima1 in iterate_noisy_images(size=128):
        execenv.print(f"  rotate{angle}({ima1.data.dtype}): ", end="")
        ima2 = func(ima1)
        check_array_result(
            f"Image rotate{angle}", ima2.data, np.rot90(ima1.data, k=angle // 90)
        )


@pytest.mark.validation
def test_image_rotate90() -> None:
    """Image 90° rotation test."""
    __generic_rotate_check(sigima.proc.image.rotate90)


@pytest.mark.validation
def test_image_rotate270() -> None:
    """Image 270° rotation test."""
    __generic_rotate_check(sigima.proc.image.rotate270)


def __get_test_image_with_roi() -> sigima.objects.ImageObj:
    """Get a test image with a predefined ROI."""
    ima = get_test_image("flower.npy")
    ima.roi = sigima.objects.create_image_roi(
        "rectangle", [10.0, 10.0, 50.0, 400.0], indices=False
    )
    return ima


def __check_roi_properties(
    ima1: sigima.objects.ImageObj, ima2: sigima.objects.ImageObj
) -> None:
    """Check that the ROI properties are preserved after transformation."""
    assert ima2.roi.single_rois[0].title == ima1.roi.single_rois[0].title
    assert ima2.roi.single_rois[0].indices == ima1.roi.single_rois[0].indices


def test_roi_rotate90() -> None:
    """Test 90° rotation with ROI transformation."""
    ima = __get_test_image_with_roi()

    # Apply 90° rotation
    rotated = sigima.proc.image.rotate90(ima)

    # Check that ROI coordinates were transformed correctly: transposition, then
    # vertical mirror about the center of the pixel grid (original X axis)
    # Original: [10, 10, 50, 400] -> Expected: [10, 2 * xc - 10 - 50, 400, 50]
    expected_coords = np.array([10.0, 2 * ima.xc - 60.0, 400.0, 50.0])
    actual_coords = rotated.roi.single_rois[0].coords

    assert np.allclose(actual_coords, expected_coords), (
        f"ROI coordinates not transformed correctly. "
        f"Expected {expected_coords}, got {actual_coords}"
    )
    __check_roi_properties(ima, rotated)


def test_roi_rotate270() -> None:
    """Test 270° rotation with ROI transformation."""
    ima = __get_test_image_with_roi()

    # Apply 270° rotation
    rotated = sigima.proc.image.rotate270(ima)

    # Check that ROI coordinates were transformed correctly: transposition, then
    # horizontal mirror about the center of the pixel grid (original Y axis)
    # Original: [10, 10, 50, 400] -> Expected: [2 * yc - 10 - 400, 10, 400, 50]
    expected_coords = np.array([2 * ima.yc - 410.0, 10.0, 400.0, 50.0])
    actual_coords = rotated.roi.single_rois[0].coords

    assert np.allclose(actual_coords, expected_coords), (
        f"ROI coordinates not transformed correctly. "
        f"Expected {expected_coords}, got {actual_coords}"
    )
    __check_roi_properties(ima, rotated)


def __get_anisotropic_image_with_roi() -> sigima.objects.ImageObj:
    """Get a non-square test image with anisotropic pixels and a physical ROI.

    Square images with square pixels are the degenerate case where a rotation about
    the image center happens to agree with the correct axis exchange, so they cannot
    reveal coordinate handling defects."""
    ima = sigima.objects.create_image("aniso", np.arange(15, dtype=float).reshape(3, 5))
    ima.set_uniform_coords(1.0, 10.0, 0.0, 0.0)
    ima.xlabel, ima.ylabel = "position", "time"
    ima.xunit, ima.yunit = "mm", "s"
    ima.roi = sigima.objects.create_image_roi(
        "rectangle", [1.0, 10.0, 2.0, 10.0], indices=False
    )
    return ima


@pytest.mark.parametrize(
    "func", [sigima.proc.image.rotate90, sigima.proc.image.rotate270]
)
def test_rotate90_swaps_pixel_sizes_and_axis_metadata(func: Callable) -> None:
    """A 90° rotation exchanges the two axes, so pixel sizes, labels and units follow.

    Keeping the source pixel sizes silently rescales anisotropic images: a 5 mm × 30 s
    image would be reported as 3 mm × 50 s instead of 30 s × 5 mm."""
    src = __get_anisotropic_image_with_roi()
    dst = func(src)
    check_scalar_result("Rotated pixel width", dst.dx, src.dy)
    check_scalar_result("Rotated pixel height", dst.dy, src.dx)
    check_scalar_result("Rotated physical width", dst.width, src.height)
    check_scalar_result("Rotated physical height", dst.height, src.width)
    assert (dst.xlabel, dst.ylabel) == (src.ylabel, src.xlabel)
    assert (dst.xunit, dst.yunit) == (src.yunit, src.xunit)


@pytest.mark.parametrize(
    "func,expected",
    [
        (sigima.proc.image.rotate90, [10.0, 1.0, 10.0, 2.0]),
        (sigima.proc.image.rotate270, [0.0, 1.0, 10.0, 2.0]),
    ],
)
def test_rotate90_maps_roi_onto_the_rotated_pixels(
    func: Callable, expected: list[float]
) -> None:
    """The rotated ROI must select the same pixels, hence stay inside the image."""
    src = __get_anisotropic_image_with_roi()
    dst = func(src)
    coords = dst.roi.single_rois[0].coords
    check_array_result(f"{func.__name__} ROI", coords, np.array(expected))
    k = 1 if func is sigima.proc.image.rotate90 else 3
    expected_mask = np.rot90(src.roi.to_mask(src), k)
    assert np.array_equal(dst.roi.to_mask(dst), expected_mask), "ROI pixels moved"
    x0, y0, x1, y1 = dst.pixel_range_to_physical([0, 0, *dst.data.shape[::-1]])
    rx0, ry0, width, height = coords
    assert x0 <= rx0 and rx0 + width <= x1, "ROI out of bounds in X"
    assert y0 <= ry0 and ry0 + height <= y1, "ROI out of bounds in Y"


@pytest.mark.parametrize(
    "func,flipfunc",
    [
        (sigima.proc.image.rotate90, sigima.proc.image.flipv),
        (sigima.proc.image.rotate270, sigima.proc.image.fliph),
    ],
)
def test_rotate90_is_transpose_then_flip(func: Callable, flipfunc: Callable) -> None:
    """``numpy.rot90`` is a transpose followed by a flip.

    Coordinates and ROI must follow the very same composition, which is well defined
    for the two quarter turns -- unlike an arbitrary rotation of the ROI shapes."""
    src = __get_anisotropic_image_with_roi()
    dst = func(src)
    ref = flipfunc(sigima.proc.image.transpose(src))
    check_array_result(f"{func.__name__} data", dst.data, ref.data)
    check_array_result(
        f"{func.__name__} ROI",
        dst.roi.single_rois[0].coords,
        ref.roi.single_rois[0].coords,
    )
    assert (dst.dx, dst.dy, dst.x0, dst.y0) == (ref.dx, ref.dy, ref.x0, ref.y0)
    assert (dst.xlabel, dst.ylabel) == (ref.xlabel, ref.ylabel)
    assert (dst.xunit, dst.yunit) == (ref.xunit, ref.yunit)


def test_roi_translation() -> None:
    """Test translation with ROI transformation."""
    ima = __get_test_image_with_roi()

    # Apply translation
    translated = sigima.proc.image.translate(
        ima, sigima.params.TranslateParam.create(dx=10, dy=10)
    )

    # Check that ROI coordinates were transformed correctly
    # Original: [10, 10, 50, 400] -> Expected: [20, 20, 50, 400]
    expected_coords = np.array([20.0, 20.0, 50.0, 400.0])
    actual_coords = translated.roi.single_rois[0].coords

    assert np.allclose(actual_coords, expected_coords), (
        f"ROI coordinates not transformed correctly. "
        f"Expected {expected_coords}, got {actual_coords}"
    )
    __check_roi_properties(ima, translated)


def __get_image_with_annotations() -> sigima.objects.ImageObj:
    """Create a calibrated image with canonical and opaque annotations."""
    image = sigima.objects.create_image(
        "annotations", np.arange(15, dtype=float).reshape(3, 5)
    )
    image.set_uniform_coords(1.0, 2.0, 0.0, 0.0)
    image.set_annotations([{"type": "plotpy_item", "plotpy_json": "{}"}])
    image.add_graphical_annotation(sigima.objects.PointAnnotation(x=1.0, y=2.0))
    return image


def test_annotation_translation() -> None:
    """Canonical annotations follow physical image translations."""
    source = __get_image_with_annotations()
    result = sigima.proc.image.translate(
        source, sigima.params.TranslateParam.create(dx=3.0, dy=-1.0)
    )

    annotation = result.get_graphical_annotations()[0]
    assert isinstance(annotation, sigima.objects.PointAnnotation)
    assert (annotation.x, annotation.y) == (4.0, 1.0)
    assert result.get_annotations()[0]["type"] == "plotpy_item"


@pytest.mark.parametrize(
    "operation,expected",
    [
        (sigima.proc.image.fliph, lambda image: (2 * image.xc - 1.0, 2.0)),
        (sigima.proc.image.flipv, lambda image: (1.0, 2 * image.yc - 2.0)),
        (sigima.proc.image.transpose, lambda _image: (2.0, 1.0)),
    ],
)
def test_annotation_exact_geometry_operations(operation, expected) -> None:
    """Canonical annotations follow exact flips and transposition."""
    source = __get_image_with_annotations()
    expected_point = expected(source)

    result = operation(source)
    annotation = result.get_graphical_annotations()[0]

    assert isinstance(annotation, sigima.objects.PointAnnotation)
    assert (annotation.x, annotation.y) == pytest.approx(expected_point)


@pytest.mark.parametrize(
    "operation,flip_operation",
    [
        (sigima.proc.image.rotate90, sigima.proc.image.flipv),
        (sigima.proc.image.rotate270, sigima.proc.image.fliph),
    ],
)
def test_annotation_quarter_turn_composition(operation, flip_operation) -> None:
    """Quarter-turn annotations follow the same transpose/flip composition."""
    source = __get_image_with_annotations()

    result = operation(source)
    reference = flip_operation(sigima.proc.image.transpose(source))

    assert result.get_graphical_annotations() == reference.get_graphical_annotations()


def test_arbitrary_rotation_clears_only_canonical_annotations() -> None:
    """Arbitrary image rotation preserves opaque data but clears unsafe geometry."""
    source = __get_image_with_annotations()

    result = sigima.proc.image.rotate(
        source, sigima.params.RotateParam.create(angle=30.0)
    )

    assert not result.has_graphical_annotations()
    assert result.get_annotations() == [{"type": "plotpy_item", "plotpy_json": "{}"}]


def test_resize_preserves_data_coordinate_annotations() -> None:
    """Resizing does not move annotations expressed in calibrated coordinates."""
    source = __get_image_with_annotations()
    annotation = source.get_graphical_annotations()[0]

    result = sigima.proc.image.resize(
        source, sigima.params.ResizeParam.create(zoom=2.0)
    )

    assert result.get_graphical_annotations() == [annotation]


@pytest.mark.validation
def test_image_rotate() -> None:
    """Image rotation test."""
    execenv.print("*** Testing image rotation:")
    for ima1 in iterate_noisy_images(size=128):
        for angle in (30.0, 45.0, 60.0, 120.0):
            execenv.print(f"  rotate{angle}({ima1.data.dtype}): ", end="")
            ima2 = sigima.proc.image.rotate(
                ima1, sigima.params.RotateParam.create(angle=angle)
            )
            exp = spi.rotate(ima1.data, angle, reshape=False)
            check_array_result(f"Image rotate{angle}", ima2.data, exp)


@pytest.mark.validation
def test_image_transpose() -> None:
    """Validation test for the image transpose processing."""
    src = get_test_image("flower.npy")
    dst = sigima.proc.image.transpose(src)
    exp = np.swapaxes(src.data, 0, 1)
    check_array_result("Transpose", dst.data, exp)


@pytest.mark.parametrize(
    "func",
    [
        sigima.proc.image.transpose,
        sigima.proc.image.rotate90,
        sigima.proc.image.rotate270,
        sigima.proc.image.fliph,
        sigima.proc.image.flipv,
    ],
)
@pytest.mark.parametrize("uniform", [True, False])
def test_geometry_result_does_not_share_source_memory(
    func: Callable, uniform: bool
) -> None:
    """Results of exact geometric transformations own their data and coordinates.

    Regression: these results were views of the source arrays, so that modifying a
    result in place also modified its source (and conversely).
    """
    src = sigima.objects.create_image("src", np.arange(12.0).reshape(3, 4))
    if not uniform:
        src.set_coords(np.array([1.0, 2.0, 4.0, 8.0]), np.array([10.0, 11.0, 15.0]))
    data, xcoords, ycoords = src.data.copy(), src.xcoords.copy(), src.ycoords.copy()
    dst = func(src)
    assert not np.shares_memory(dst.data, src.data)
    dst.data[...] = -1.0
    src.data[0, 0] = 999.0
    assert dst.data[0, 0] == -1.0
    src.data[0, 0] = data[0, 0]
    check_array_result("Unchanged source data", src.data, data)
    if not uniform:
        for dst_coords in (dst.xcoords, dst.ycoords):
            for src_coords in (src.xcoords, src.ycoords):
                assert not np.shares_memory(dst_coords, src_coords)
        dst.xcoords[...] = -1.0
        dst.ycoords[...] = -1.0
        check_array_result("Unchanged source X coordinates", src.xcoords, xcoords)
        check_array_result("Unchanged source Y coordinates", src.ycoords, ycoords)


def test_image_geometry_nonuniform_coords() -> None:
    """Check that geometry operations propagate non-uniform coordinates.

    ``translate`` and ``transpose`` keep the coordinate arrays in sync with the
    transformed data when the source image uses non-uniform coordinates.
    """
    execenv.print("*** Testing non-uniform coordinate propagation")

    src = get_test_image("flower.npy")
    ny, nx = src.data.shape
    xcoords = np.linspace(0.0, 10.0, nx) ** 1.5  # non-uniform spacing
    ycoords = np.linspace(0.0, 8.0, ny) ** 2  # non-uniform spacing
    src.set_coords(xcoords, ycoords)

    # translate must offset the coordinate arrays (not x0/y0)
    dst_t = sigima.proc.image.translate(
        src, sigima.params.TranslateParam.create(dx=3.0, dy=-2.0)
    )
    assert not dst_t.is_uniform_coords, "translate must keep non-uniform coords"
    check_array_result("Translate non-uniform xcoords", dst_t.xcoords, xcoords + 3.0)
    check_array_result("Translate non-uniform ycoords", dst_t.ycoords, ycoords - 2.0)

    # transpose must swap the coordinate arrays
    dst_tr = sigima.proc.image.transpose(src)
    assert not dst_tr.is_uniform_coords, "transpose must keep non-uniform coords"
    check_array_result("Transpose non-uniform xcoords", dst_tr.xcoords, ycoords)
    check_array_result("Transpose non-uniform ycoords", dst_tr.ycoords, xcoords)
    check_array_result("Transpose non-uniform data", dst_tr.data, src.data.T)


@pytest.mark.validation
def test_image_resampling() -> None:
    """Image resampling test."""
    execenv.print("*** Testing image resampling")

    # Create a test image
    ima1 = get_test_image("flower.npy")
    nrows, ncols = ima1.data.shape
    xlast = ima1.x0 + (ncols - 1) * ima1.dx
    ylast = ima1.y0 + (nrows - 1) * ima1.dy

    # Test 1: Identity resampling (same grid): every pixel is kept
    p1 = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=ncols,
        height=nrows,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
    )
    p1.update_from_obj(ima1)
    dst1 = sigima.proc.image.resampling(ima1, p1)

    check_array_result(
        "Identity resampling", dst1.data, ima1.data.astype(float), rtol=0, atol=0
    )
    check_scalar_result("Identity resampling X0", dst1.x0, ima1.x0)
    check_scalar_result("Identity resampling Y0", dst1.y0, ima1.y0)
    check_scalar_result("Identity resampling dx", dst1.dx, ima1.dx)
    check_scalar_result("Identity resampling dy", dst1.dy, ima1.dy)

    # Test 2: Downsample, same first and last pixel centers
    p2 = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=ncols // 2,
        height=nrows // 2,
        xmin=ima1.x0,
        xmax=xlast,
        ymin=ima1.y0,
        ymax=ylast,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
    )
    dst2 = sigima.proc.image.resampling(ima1, p2)

    check_scalar_result("Downsample X0", dst2.x0, ima1.x0)
    check_scalar_result("Downsample Y0", dst2.y0, ima1.y0)
    check_scalar_result("Downsample shape[0]", dst2.data.shape[0], nrows // 2)
    check_scalar_result("Downsample shape[1]", dst2.data.shape[1], ncols // 2)

    # Check that pixel sizes are adjusted correctly
    expected_dx = (xlast - ima1.x0) / (ncols // 2 - 1)
    expected_dy = (ylast - ima1.y0) / (nrows // 2 - 1)
    check_scalar_result("Downsample dx", dst2.dx, expected_dx, rtol=1e-10)
    check_scalar_result("Downsample dy", dst2.dy, expected_dy, rtol=1e-10)
    assert dst2.data[0, 0] == ima1.data[0, 0]
    assert dst2.data[-1, -1] == ima1.data[-1, -1]

    # Test 3: Use pixel size mode (exact pixel size, default bounds)
    p3 = sigima.params.Resampling2DParam.create(
        mode="dxy",
        dx=ima1.dx * 1.5,
        dy=ima1.dy * 1.5,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
    )
    dst3 = sigima.proc.image.resampling(ima1, p3)

    check_scalar_result("Pixel size mode dx", dst3.dx, ima1.dx * 1.5, rtol=1e-10)
    check_scalar_result("Pixel size mode dy", dst3.dy, ima1.dy * 1.5, rtol=1e-10)
    assert dst3.data.shape == (
        int(np.floor((nrows - 1) / 1.5)) + 1,
        int(np.floor((ncols - 1) / 1.5)) + 1,
    )

    # Test 4: Different interpolation methods
    for method in sigima.enums.Interpolation2DMethod:
        p4 = sigima.params.Resampling2DParam.create(
            mode="shape",
            width=ima1.data.shape[1] // 2,
            height=ima1.data.shape[0] // 2,
            xmin=ima1.x0,
            xmax=ima1.x0 + ima1.width,
            ymin=ima1.y0,
            ymax=ima1.y0 + ima1.height,
            method=method,
        )
        dst4 = sigima.proc.image.resampling(ima1, p4)

        # Basic shape checks
        check_scalar_result(
            f"Method {method} shape[0]", dst4.data.shape[0], ima1.data.shape[0] // 2
        )
        check_scalar_result(
            f"Method {method} shape[1]", dst4.data.shape[1], ima1.data.shape[1] // 2
        )

    # Test 5: fill_value parameter (out-of-bounds sampling)
    execenv.print("  Testing fill_value parameter")

    # Test 5a: Default behavior (fill_value=None should use NaN)
    p5a = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=20,
        height=20,
        xmin=600.0,  # Outside image bounds
        xmax=620.0,
        ymin=600.0,
        ymax=620.0,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
        fill_value=None,
    )
    dst5a = sigima.proc.image.resampling(ima1, p5a)

    # Should be all NaN since sampling outside image bounds
    assert np.all(np.isnan(dst5a.data)), (
        "Expected all NaN values for out-of-bounds sampling with fill_value=None"
    )
    assert dst5a.data.dtype == np.float64, "Expected float64 dtype for NaN result"

    # Test 5b: Custom fill value
    p5b = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=20,
        height=20,
        xmin=600.0,  # Outside image bounds
        xmax=620.0,
        ymin=600.0,
        ymax=620.0,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
        fill_value=123.0,
    )
    dst5b = sigima.proc.image.resampling(ima1, p5b)

    # Should be all 123.0 since sampling outside image bounds
    assert np.all(dst5b.data == 123.0), (
        "Expected all fill values for out-of-bounds sampling"
    )
    assert dst5b.data.dtype == ima1.data.dtype, (
        "Expected same dtype as input for numeric fill value"
    )

    # Test 5c: Partially outside (mix of real data and fill values)
    p5c = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=30,
        height=30,
        xmin=ima1.x0 + ima1.width - 10,  # Partially outside
        xmax=ima1.x0 + ima1.width + 20,
        ymin=ima1.y0 + ima1.height - 10,
        ymax=ima1.y0 + ima1.height + 20,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
        fill_value=99.0,
    )
    dst5c = sigima.proc.image.resampling(ima1, p5c)

    # Should have mix of values
    fill_count = np.sum(dst5c.data == 99.0)
    total_count = dst5c.data.size
    assert fill_count > 0, "Expected some fill values for partially out-of-bounds"
    assert fill_count < total_count, "Expected some real data values"

    # Test 5d: Within bounds should not use fill value
    p5d = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=50,
        height=50,
        xmin=ima1.x0 + 50,  # Within bounds
        xmax=ima1.x0 + 100,
        ymin=ima1.y0 + 50,
        ymax=ima1.y0 + 100,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
        fill_value=999.0,
    )
    dst5d = sigima.proc.image.resampling(ima1, p5d)

    # Should not contain any fill values since all within bounds
    assert not np.any(dst5d.data == 999.0), (
        "No fill values expected for within-bounds sampling"
    )


def _ramp_image(
    nrows: int, ncols: int, x0: float, y0: float, dx: float, dy: float
) -> sigima.objects.ImageObj:
    """Return an image of the affine ramp ``2 x + 3 y`` sampled at pixel centers."""
    x = x0 + np.arange(ncols) * dx
    y = y0 + np.arange(nrows) * dy
    image = sigima.objects.create_image("ramp", 2.0 * x[None, :] + 3.0 * y[:, None])
    image.set_uniform_coords(dx, dy, x0, y0)
    return image


@pytest.mark.parametrize("method", list(sigima.enums.Interpolation2DMethod))
def test_resampling_identity_keeps_every_pixel(
    method: sigima.enums.Interpolation2DMethod,
) -> None:
    """Resampling an image onto its own grid keeps every pixel and its coordinates.

    Regression: the default bounds used to sample half a pixel away from the
    pixel centers, shifting the values and filling a row and a column with NaN.
    """
    for src in (
        sigima.objects.create_image("arange", np.arange(12.0).reshape(3, 4)),
        _ramp_image(5, 7, x0=-3.0, y0=10.0, dx=0.5, dy=2.0),  # anisotropic
    ):
        nrows, ncols = src.data.shape
        p = sigima.params.Resampling2DParam.create(
            mode="shape", width=ncols, height=nrows, method=method
        )
        dst = sigima.proc.image.resampling(src, p)
        assert not np.isnan(dst.data).any()
        check_array_result(f"Identity {method}", dst.data, src.data, atol=1e-9)
        assert (dst.x0, dst.y0) == (src.x0, src.y0)
        check_scalar_result("Identity dx", dst.dx, src.dx, rtol=1e-12)
        check_scalar_result("Identity dy", dst.dy, src.dy, rtol=1e-12)

        p_dxy = sigima.params.Resampling2DParam.create(
            mode="dxy", dx=src.dx, dy=src.dy, method=method
        )
        dst_dxy = sigima.proc.image.resampling(src, p_dxy)
        check_array_result(f"Identity dxy {method}", dst_dxy.data, src.data, atol=1e-9)


def test_resampling_linear_matches_affine_ramp() -> None:
    """Linear resampling of an affine ramp is exact at every output pixel center."""
    src = _ramp_image(5, 7, x0=1.0, y0=-2.0, dx=0.5, dy=0.25)
    p = sigima.params.Resampling2DParam.create(
        mode="shape",
        width=13,
        height=9,
        xmin=1.25,
        xmax=3.5,
        ymin=-2.0,
        ymax=-1.25,
        method=sigima.enums.Interpolation2DMethod.LINEAR,
    )
    dst = sigima.proc.image.resampling(src, p)
    assert (dst.x0, dst.y0) == (1.25, -2.0)
    check_scalar_result("Ramp dx", dst.dx, (3.5 - 1.25) / 12)
    check_scalar_result("Ramp dy", dst.dy, 0.75 / 8)
    x = dst.x0 + np.arange(13) * dst.dx
    y = dst.y0 + np.arange(9) * dst.dy
    check_array_result(
        "Ramp values", dst.data, 2.0 * x[None, :] + 3.0 * y[:, None], atol=1e-12
    )


def test_resampling_single_pixel_image() -> None:
    """A 1×1 image is resampled onto itself with the default parameters."""
    src = sigima.objects.create_image("pixel", np.array([[7.0]]))
    src.set_uniform_coords(0.5, 2.0, 3.0, 4.0)
    for mode in ("shape", "dxy"):
        p = sigima.params.Resampling2DParam.create(mode=mode)
        p.update_from_obj(src)
        dst = sigima.proc.image.resampling(src, p)
        assert dst.data.tolist() == [[7.0]]
        assert (dst.x0, dst.y0, dst.dx, dst.dy) == (3.0, 4.0, 0.5, 2.0)


def test_resampling_pixel_size_mode_keeps_exact_pitch() -> None:
    """In pixel size mode, the pixel size is exact even when it does not divide
    the extent: the last pixel center is the last one not beyond ``xmax``."""
    src = _ramp_image(4, 4, x0=0.0, y0=0.0, dx=1.0, dy=1.0)
    p = sigima.params.Resampling2DParam.create(
        mode="dxy", dx=0.8, dy=1.5, method=sigima.enums.Interpolation2DMethod.LINEAR
    )
    dst = sigima.proc.image.resampling(src, p)
    assert dst.data.shape == (3, 4)
    assert (dst.dx, dst.dy) == (0.8, 1.5)
    x = np.arange(4) * 0.8
    y = np.arange(3) * 1.5
    check_array_result(
        "Pixel size mode values", dst.data, 2.0 * x[None, :] + 3.0 * y[:, None]
    )


@pytest.mark.validation
def test_image_resize() -> None:
    """Image resize test."""
    execenv.print("*** Testing image resize")

    # Test with different zoom factors
    zoom_factors = [0.5, 2.0, 1.5, 0.75]

    for ima1 in iterate_noisy_images(size=128):
        execenv.print(f"  Testing on {ima1.data.dtype} image")

        for zoom in zoom_factors:
            execenv.print(f"    zoom={zoom}: ", end="")

            # Test resize with default parameters
            p = sigima.params.ResizeParam.create(zoom=zoom)
            ima2 = sigima.proc.image.resize(ima1, p)

            # Check that scipy.ndimage.zoom produces the same result
            expected_data = spi.zoom(
                ima1.data, zoom, order=3, mode="constant", cval=0.0, prefilter=True
            )
            check_array_result(f"Resize zoom={zoom}", ima2.data, expected_data)

            # Check that pixel sizes are updated correctly
            if ima1.dx is not None and ima1.dy is not None:
                expected_dx = ima1.dx / zoom
                expected_dy = ima1.dy / zoom
                check_scalar_result(
                    f"Resize dx zoom={zoom}", ima2.dx, expected_dx, rtol=1e-10
                )
                check_scalar_result(
                    f"Resize dy zoom={zoom}", ima2.dy, expected_dy, rtol=1e-10
                )

    # Test different border modes and parameters
    execenv.print("  Testing different border modes and parameters")
    ima_test = get_test_image("flower.npy")

    # Test different modes
    for mode in sigima.enums.BorderMode:
        execenv.print(f"    mode={mode.name}: ", end="")
        p = sigima.params.ResizeParam.create(zoom=1.5, mode=mode, cval=100.0)
        ima_resized = sigima.proc.image.resize(ima_test, p)

        # Compare with scipy implementation
        expected_data = spi.zoom(
            ima_test.data, 1.5, order=3, mode=mode.value, cval=100.0, prefilter=True
        )
        check_array_result(f"Resize mode={mode.name}", ima_resized.data, expected_data)

    # Test different interpolation orders
    execenv.print("  Testing different interpolation orders")
    for order in [0, 1, 2, 3, 4, 5]:
        execenv.print(f"    order={order}: ", end="")
        p = sigima.params.ResizeParam.create(zoom=1.3, order=order, prefilter=False)
        ima_resized = sigima.proc.image.resize(ima_test, p)

        # Compare with scipy implementation
        expected_data = spi.zoom(
            ima_test.data, 1.3, order=order, mode="constant", cval=0.0, prefilter=False
        )
        check_array_result(f"Resize order={order}", ima_resized.data, expected_data)

    # Test with prefilter disabled
    execenv.print("  Testing prefilter parameter")
    for prefilter in [True, False]:
        execenv.print(f"    prefilter={prefilter}: ", end="")
        p = sigima.params.ResizeParam.create(zoom=0.8, prefilter=prefilter)
        ima_resized = sigima.proc.image.resize(ima_test, p)

        # Compare with scipy implementation
        expected_data = spi.zoom(
            ima_test.data, 0.8, order=3, mode="constant", cval=0.0, prefilter=prefilter
        )
        check_array_result(
            f"Resize prefilter={prefilter}", ima_resized.data, expected_data
        )

    # Test edge cases
    execenv.print("  Testing edge cases")

    # Test zoom=1.0 (identity)
    p_identity = sigima.params.ResizeParam.create(zoom=1.0)
    ima_identity = sigima.proc.image.resize(ima_test, p_identity)
    check_array_result("Resize identity zoom=1.0", ima_identity.data, ima_test.data)

    # Test very small zoom
    p_small = sigima.params.ResizeParam.create(zoom=0.1)
    ima_small = sigima.proc.image.resize(ima_test, p_small)
    expected_small = spi.zoom(
        ima_test.data, 0.1, order=3, mode="constant", cval=0.0, prefilter=True
    )
    check_array_result("Resize small zoom=0.1", ima_small.data, expected_small)

    # Test large zoom
    p_large = sigima.params.ResizeParam.create(zoom=5.0)
    ima_large = sigima.proc.image.resize(ima_test, p_large)
    expected_large = spi.zoom(
        ima_test.data, 5.0, order=3, mode="constant", cval=0.0, prefilter=True
    )
    check_array_result("Resize large zoom=5.0", ima_large.data, expected_large)


@pytest.mark.validation
def test_set_uniform_coords() -> None:
    """Test converting from non-uniform to uniform coordinates."""
    execenv.print("*** Testing set_uniform_coords")

    # Test 1: Create an image with non-uniform coordinates
    execenv.print("  Testing non-uniform to uniform conversion")
    ima1 = get_test_image("flower.npy")
    nx, ny = ima1.data.shape[1], ima1.data.shape[0]

    # Create non-uniform coordinates (e.g., quadratic spacing on y-axis)
    xcoords = np.linspace(0.0, 10.0, nx)
    ycoords = np.linspace(0.0, 8.0, ny) ** 2  # Non-uniform spacing
    ima1.set_coords(xcoords, ycoords)

    # Verify it's non-uniform
    assert not ima1.is_uniform_coords, "Image should have non-uniform coordinates"

    # Create parameter and update from object
    p = sigima.params.UniformCoordsParam()
    p.update_from_obj(ima1)

    # Apply conversion
    ima2 = sigima.proc.image.set_uniform_coords(ima1, p)

    # Check that result has uniform coordinates
    assert ima2.is_uniform_coords, "Result should have uniform coordinates"

    # Check that the data is unchanged
    check_array_result("Data preservation", ima2.data, ima1.data)

    # Check that coordinate parameters were extracted correctly
    expected_x0 = xcoords[0]
    expected_y0 = ycoords[0]
    expected_dx = (xcoords[-1] - xcoords[0]) / (nx - 1)
    expected_dy = (ycoords[-1] - ycoords[0]) / (ny - 1)

    check_scalar_result("X0 extraction", ima2.x0, expected_x0, atol=1e-10)
    check_scalar_result("Y0 extraction", ima2.y0, expected_y0, atol=1e-10)
    check_scalar_result("dx extraction", ima2.dx, expected_dx, atol=1e-10)
    check_scalar_result("dy extraction", ima2.dy, expected_dy, atol=1e-10)

    # Test 2: Converting already uniform coordinates (should preserve values)
    execenv.print("  Testing uniform to uniform (identity)")
    ima3 = get_test_image("flower.npy")
    original_x0, original_y0 = ima3.x0, ima3.y0
    original_dx, original_dy = ima3.dx, ima3.dy

    p2 = sigima.params.UniformCoordsParam()
    p2.update_from_obj(ima3)
    ima4 = sigima.proc.image.set_uniform_coords(ima3, p2)

    assert ima4.is_uniform_coords, "Result should have uniform coordinates"
    check_array_result("Data preservation (uniform)", ima4.data, ima3.data)
    check_scalar_result("X0 preservation", ima4.x0, original_x0, atol=1e-10)
    check_scalar_result("Y0 preservation", ima4.y0, original_y0, atol=1e-10)
    check_scalar_result("dx preservation", ima4.dx, original_dx, atol=1e-10)
    check_scalar_result("dy preservation", ima4.dy, original_dy, atol=1e-10)

    # Test 3: Manual parameter specification
    execenv.print("  Testing manual parameter specification")
    ima5 = get_test_image("flower.npy")
    # Create non-uniform coordinates
    ima5.set_coords(np.linspace(5.0, 15.0, nx), np.linspace(10.0, 20.0, ny))

    p3 = sigima.params.UniformCoordsParam.create(x0=5.0, y0=10.0, dx=0.5, dy=0.25)
    ima6 = sigima.proc.image.set_uniform_coords(ima5, p3)

    assert ima6.is_uniform_coords, "Result should have uniform coordinates"
    check_scalar_result("Manual X0", ima6.x0, 5.0, atol=1e-10)
    check_scalar_result("Manual Y0", ima6.y0, 10.0, atol=1e-10)
    check_scalar_result("Manual dx", ima6.dx, 0.5, atol=1e-10)
    check_scalar_result("Manual dy", ima6.dy, 0.25, atol=1e-10)


@pytest.mark.validation
def test_image_calibration() -> None:
    """Validation test for polynomial calibration."""
    execenv.print("*** Testing calibration (polynomial)")

    # Test 1: Z-axis polynomial calibration
    execenv.print("  Testing Z-axis polynomial calibration")
    src = get_test_image("flower.npy")
    # Use smaller coefficients to avoid overflow with uint8 data (0-255 range)
    p = sigima.params.XYZCalibrateParam.create(
        axis="z", a0=10.0, a1=2.0, a2=0.001, a3=0.0
    )
    dst = sigima.proc.image.calibration(src, p)

    # Verify polynomial transformation on data
    src_data_float = src.data.astype(float)
    expected_data = (
        p.a0
        + p.a1 * src_data_float
        + p.a2 * src_data_float**2
        + p.a3 * src_data_float**3
    )
    check_array_result("Z-axis polynomial", dst.data, expected_data)

    # Coordinates should be unchanged
    assert dst.is_uniform_coords
    check_scalar_result("Z-calib: x0", dst.x0, src.x0)
    check_scalar_result("Z-calib: y0", dst.y0, src.y0)
    check_scalar_result("Z-calib: dx", dst.dx, src.dx)
    check_scalar_result("Z-calib: dy", dst.dy, src.dy)

    # Test 2: X-axis polynomial calibration (uniform → non-uniform)
    execenv.print("  Testing X-axis polynomial (uniform → non-uniform)")
    src2 = get_test_image("flower.npy")
    src2.set_uniform_coords(dx=0.5, dy=0.5, x0=0.0, y0=0.0)
    p2 = sigima.params.XYZCalibrateParam.create(
        axis="x", a0=1.0, a1=2.0, a2=0.1, a3=0.0
    )
    dst2 = sigima.proc.image.calibration(src2, p2)

    # After polynomial calibration on X, coordinates should become non-uniform
    assert not dst2.is_uniform_coords, (
        "X-axis polynomial should create non-uniform coords"
    )

    # Verify X coordinates transformation
    x_uniform = src2.x0 + np.arange(src2.data.shape[1]) * src2.dx
    expected_x = p2.a0 + p2.a1 * x_uniform + p2.a2 * x_uniform**2
    check_array_result("X-axis polynomial coords", dst2.xcoords, expected_x)
    # Check that Y coordinates were converted in non-uniform but unchanged
    src2_ycoords = src2.y0 + np.arange(src2.data.shape[0]) * src2.dy
    check_array_result("X-axis polynomial Y coords", dst2.ycoords, src2_ycoords)

    # Data should be unchanged
    check_array_result("X-calib: data preservation", dst2.data, src2.data)

    # Test 3: Y-axis polynomial calibration (non-uniform → non-uniform)
    execenv.print("  Testing Y-axis polynomial (non-uniform → non-uniform)")
    src3 = get_test_image("flower.npy")
    ny = src3.data.shape[0]
    y_nonuniform = np.linspace(0.0, 10.0, ny)
    src3.set_coords(None, y_nonuniform)

    p3 = sigima.params.XYZCalibrateParam.create(
        axis="y", a0=5.0, a1=1.0, a2=0.0, a3=0.05
    )
    dst3 = sigima.proc.image.calibration(src3, p3)

    # Should still be non-uniform
    assert not dst3.is_uniform_coords

    # Verify Y coordinates transformation
    expected_y = p3.a0 + p3.a1 * y_nonuniform + p3.a3 * y_nonuniform**3
    check_array_result("Y-axis polynomial coords", dst3.ycoords, expected_y)

    # Data should be unchanged
    check_array_result("Y-calib: data preservation", dst3.data, src3.data)

    # Test 4: Linear case (a2=a3=0, backward compatibility)
    execenv.print("  Testing linear calibration (a2=a3=0)")
    src4 = get_test_image("flower.npy")
    p4 = sigima.params.XYZCalibrateParam.create(
        axis="x", a0=0.5, a1=2.0, a2=0.0, a3=0.0
    )
    dst4 = sigima.proc.image.calibration(src4, p4)

    # For linear case with uniform input, result should still be non-uniform
    # because we always generate coordinate arrays
    # Verify the transformation is correct
    x_uniform = src4.x0 + np.arange(src4.data.shape[1]) * src4.dx
    expected_x_linear = p4.a0 + p4.a1 * x_uniform
    if dst4.is_uniform_coords:
        # If implementation optimized to keep uniform coords
        check_scalar_result("Linear x0", dst4.x0, expected_x_linear[0])
        check_scalar_result(
            "Linear dx", dst4.dx, expected_x_linear[1] - expected_x_linear[0]
        )
    else:
        # If coordinates are non-uniform
        check_array_result("Linear xcoords", dst4.xcoords, expected_x_linear)


if __name__ == "__main__":
    test_image_fliph()
    test_image_flipv()
    test_image_rotate90()
    test_image_rotate270()
    test_image_rotate()
    test_image_transpose()
    test_image_resampling()
    test_image_resize()
    test_image_translate()
    test_set_uniform_coords()
    test_image_calibration()
