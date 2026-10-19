# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Image ROI advanced unit tests"""

from __future__ import annotations

import numpy as np
import pytest
from skimage import draw

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests import guiutils
from sigima.tests.data import create_multigaussian_image
from sigima.tests.helpers import print_obj_data_dimensions


def test_image_roi_param() -> None:
    """Test image ROI parameter conversion"""
    # Create an image object
    obj = create_multigaussian_image()
    # Create an image ROI
    coords = [100, 150, 200, 250]
    roi = sigima.objects.create_image_roi("rectangle", coords, inverse=False)
    # Convert to parameters
    roiparam = roi.to_params(obj)[0]
    assert isinstance(roiparam, sigima.objects.ROI2DParam), (
        "Parameter should be ROI2DParam"
    )
    # Check that converting back to single ROI gives the same coordinates
    single_roi = roiparam.to_single_roi(obj)
    single_roi_coords = single_roi.get_physical_coords(obj)
    assert np.all(single_roi_coords == np.array(coords)), (
        "Single ROI coordinates mismatch"
    )


def test_image_roi_merge() -> None:
    """Test image ROI merge"""
    # Create an image object with a single ROI, and another one with another ROI.
    # Compute the average of the two objects, and check if the resulting object
    # has the expected ROI (i.e. the union of the original object's ROI).
    obj1 = create_multigaussian_image()
    obj2 = create_multigaussian_image()
    obj2.roi = sigima.objects.create_image_roi(
        "rectangle", [600, 800, 1000, 1200], inverse=False
    )
    obj1.roi = sigima.objects.create_image_roi(
        "rectangle", [500, 750, 1000, 1250], inverse=False
    )

    # Compute the average of the two objects
    obj3 = sigima.proc.image.average([obj1, obj2])
    assert obj3.roi is not None, "Merged object should have a ROI"
    assert len(obj3.roi) == 2, "Merged object should have two single ROIs"
    for single_roi in obj3.roi:
        assert single_roi.get_indices_coords(obj3) in (
            [500, 750, 1000, 1250],
            [600, 800, 1000, 1200],
        ), "Merged object should have the union of the original object's ROIs"


def test_image_roi_combine() -> None:
    """Test `ImageROI.combine_with` method"""
    coords1, coords2 = [600, 800, 1000, 1200], [500, 750, 1000, 1250]
    roi1 = sigima.objects.create_image_roi(
        "rectangle", coords1, indices=True, inverse=False
    )
    roi2 = sigima.objects.create_image_roi(
        "rectangle", coords2, indices=True, inverse=False
    )
    exp_combined = sigima.objects.create_image_roi(
        "rectangle",
        [coords1, coords2],
        indices=True,
        inverse=False,
    )
    # Check that combining two ROIs results in a new ROI with both coordinates:
    roi3 = roi1.combine_with(roi2)
    assert roi3 == exp_combined, "Combined ROI should match expected"
    # Check that combining again with the same ROI does not change it:
    roi3 = roi1.combine_with(roi2)
    assert roi3 == exp_combined, "Combining with the same ROI should not change it"
    # Check that combining with a signal ROI raises an error:
    with pytest.raises(
        TypeError, match=r"Cannot combine([\S ]*)ImageROI([\S ]*)SignalROI"
    ):
        roi1.combine_with(sigima.objects.create_signal_roi([50, 100], indices=True))


SIZE = 200

# Image ROIs:
IROI1 = [100, 100, 75, 100]  # Rectangle
IROI2 = [66, 100, 50]  # Circle
# Polygon (triangle, that is intentionally inside the rectangle, so that this ROI
# has no impact on the mask calculations in the tests)
IROI3 = [100, 100, 100, 150, 150, 133]


def __roi_str(obj: sigima.objects.ImageObj) -> str:
    """Return a string representation of a ImageROI object for context."""
    if obj.roi is None:
        return "None"
    if obj.roi.is_empty():
        return "Empty"
    return ", ".join(
        f"{single_roi.__class__.__name__}({single_roi.get_indices_coords(obj)})"
        for single_roi in obj.roi.single_rois
    )


def __create_test_roi() -> sigima.objects.ImageROI:
    """Create test ROI"""
    roi = sigima.objects.create_image_roi("rectangle", IROI1, inverse=False)
    roi.add_roi(sigima.objects.create_image_roi("circle", IROI2, inverse=False))
    roi.add_roi(sigima.objects.create_image_roi("polygon", IROI3, inverse=False))
    return roi


def __create_test_image() -> sigima.objects.ImageObj:
    """Create test image"""
    param = sigima.objects.NewImageParam.create(height=SIZE, width=SIZE)
    ima = create_multigaussian_image(param)
    ima.data += 1  # Ensure that the image has non-zero values (for ROI check tests)
    return ima


def __test_processing_in_roi(src: sigima.objects.ImageObj) -> None:
    """Run image processing in ROI

    Args:
        src: Source image object (with or without ROI)
    """
    print_obj_data_dimensions(src)
    value = 1
    p = sigima.params.ConstantParam.create(value=value)
    dst = sigima.proc.image.addition_constant(src, p)
    orig = src.data
    new = dst.data
    context = f" [ROI: {__roi_str(src)}]"
    if src.roi is not None and not src.roi.is_empty():
        # A ROI has been set in the source image.
        assert np.all(
            new[IROI1[1] : IROI1[3] + IROI1[1], IROI1[0] : IROI1[2] + IROI1[0]]
            == orig[IROI1[1] : IROI1[3] + IROI1[1], IROI1[0] : IROI1[2] + IROI1[0]]
            + value
        ), f"Image ROI 1 data mismatch{context}"
        assert np.all(
            new[IROI2[1] : IROI1[1] + 1, IROI2[0] : IROI2[0] + 2 * IROI2[2]]
            == orig[IROI2[1] : IROI1[1] + 1, IROI2[0] : IROI2[0] + 2 * IROI2[2]] + value
        ), f"Image ROI 2 data mismatch{context}"
        first_col = min(IROI1[0], IROI2[0] - IROI2[2])
        first_row = min(IROI1[1], IROI2[1] - IROI2[2])
        last_col = max(IROI1[0] + IROI1[2], IROI2[0] + 2 * IROI2[2])
        last_row = max(IROI1[1] + IROI1[3], IROI2[1] + 2 * IROI2[2])
        assert np.all(
            new[:first_row, :first_col] == np.array(orig[:first_row, :first_col], float)
        ), f"Image before ROIs data mismatch{context}"
        assert np.all(new[:first_row, last_col:] == orig[:first_row, last_col:]), (
            f"Image after ROIs data mismatch{context}"
        )
        assert np.all(new[last_row:, :first_col] == orig[last_row:, :first_col]), (
            f"Image before ROIs data mismatch{context}"
        )
        assert np.all(new[last_row:, last_col:] == orig[last_row:, last_col:]), (
            f"Image after ROIs data mismatch{context}"
        )
    else:
        # No ROI has been set in the source image.
        assert np.all(new == orig + value), f"Image data mismatch{context}"


def test_image_roi_processing() -> None:
    """Test image ROI processing"""
    src = __create_test_image()
    base_roi = __create_test_roi()
    empty_roi = sigima.objects.ImageROI()
    for roi in (empty_roi, base_roi):
        src.roi = roi
        __test_processing_in_roi(src)


def test_empty_image_roi() -> None:
    """Test empty image ROI"""
    src = __create_test_image()
    empty_roi = sigima.objects.ImageROI()
    for roi in (None, empty_roi):
        src.roi = roi
        context = f" [ROI: {__roi_str(src)}]"
        assert src.roi is None or src.roi.is_empty(), (
            f"Source object ROI should be empty or None{context}"
        )
        if src.roi is not None:
            # No ROI has been set in the source image
            im1 = sigima.proc.image.extract_roi(src, src.roi.to_params(src))
            assert im1.data.shape == (0, 0), f"Extracted image should be empty{context}"


@pytest.mark.validation
def test_image_extract_rois() -> None:
    """Validation test for image ROI extraction into a single object"""
    src = __create_test_image()
    src.roi = __create_test_roi()
    context = f" [ROI: {__roi_str(src)}]"
    nzroi = f"Non-zero values expected in ROI{context}"
    zroi = f"Zero values expected outside ROI{context}"

    im1 = sigima.proc.image.extract_rois(src, src.roi.to_params(src))

    mask1 = np.zeros(shape=(SIZE, SIZE), dtype=bool)
    mask1[IROI1[1] : IROI1[1] + IROI1[3], IROI1[0] : IROI1[0] + IROI1[2]] = 1
    xc, yc, r = IROI2
    mask2 = np.zeros(shape=(SIZE, SIZE), dtype=bool)
    rr, cc = draw.disk((yc, xc), r)
    mask2[rr, cc] = 1
    mask = mask1 | mask2
    row_min = int(min(IROI1[1], IROI2[1] - r))
    col_min = int(min(IROI1[0], IROI2[0] - r))
    row_max = int(max(IROI1[1] + IROI1[3], IROI2[1] + r))
    col_max = int(max(IROI1[0] + IROI1[2], IROI2[0] + r))
    mask = mask[row_min:row_max, col_min:col_max]

    assert np.all(im1.data[mask] != 0), nzroi
    assert np.all(im1.data[~mask] == 0), zroi
    # Bug fix verification: extracted image should not have ROI defined
    assert im1.roi is None, f"Extracted image should not have ROI defined{context}"


@pytest.mark.validation
def test_image_extract_roi() -> None:
    """Validation test for image ROI extraction into multiple objects"""
    src = __create_test_image()
    src.roi = __create_test_roi()
    context = f" [ROI: {__roi_str(src)}]"
    nzroi = f"Non-zero values expected in ROI{context}"
    roisham = f"ROI shape mismatch{context}"

    images: list[sigima.objects.ImageObj] = []
    for index, single_roi in enumerate(src.roi):
        roiparam = single_roi.to_param(src, index)
        image = sigima.proc.image.extract_roi(src, roiparam)
        images.append(image)
    assert len(images) == 3, f"Three images expected{context}"
    im1, im2 = images[:2]  # pylint: disable=unbalanced-tuple-unpacking
    assert np.all(im1.data != 0), nzroi
    assert im1.data.shape == (IROI1[3], IROI1[2]), roisham
    assert np.all(im2.data != 0), nzroi
    assert im2.data.shape == (IROI2[2] * 2, IROI2[2] * 2), roisham
    mask2 = np.zeros(shape=im2.data.shape, dtype=bool)
    xc = yc = r = IROI2[2]  # Adjust for ROI origin
    rr, cc = draw.disk((yc, xc), r, shape=im2.data.shape)
    mask2[rr, cc] = 1
    assert np.all(im2.maskdata == ~mask2), f"Mask data mismatch{context}"
    # Bug fix verification: extracted images should handle ROI correctly
    # - For rectangular ROI: no ROI should be defined
    # - For circular/polygonal ROI: a new ROI should be created (not copied from source)
    assert images[0].roi is None, f"Rectangular extraction should not have ROI{context}"
    # For circular and polygonal, roi should exist but be different from source
    for idx in [1, 2]:
        if images[idx].roi is not None:
            err_msg = f"Extracted image {idx} ROI should not be same as source{context}"
            assert images[idx].roi is not src.roi, err_msg


def test_roi_coordinates_validation() -> None:
    """Test ROI coordinates validation"""
    # Create a 20x20 Gaussian image, in pixel coordinates (as the ROIs below)
    param = sigima.objects.Gauss2DParam.create(
        a=10.0, height=20, width=20, physical_coords=False
    )
    src = sigima.objects.create_image_from_param(param)

    # Create ROI coordinates
    rect_coords = np.array([4.5, 4.5, 10.0, 10.0])
    circ_coords = np.array([9.5, 9.5, 5.0])
    poly_coords = np.array([5.1, 15.1, 14.7, 12.0, 12.5, 7.0, 5.2, 4.9])

    # Create ROIs
    rect_roi = sigima.objects.create_image_roi(
        "rectangle", rect_coords, title="rectangular"
    )
    circ_roi = sigima.objects.create_image_roi("circle", circ_coords, title="circular")
    poly_roi = sigima.objects.create_image_roi(
        "polygon", poly_coords, title="polygonal"
    )

    # Check that coordinates are correct
    assert np.all(rect_roi.get_single_roi(0).get_physical_coords(src) == rect_coords)
    assert np.all(circ_roi.get_single_roi(0).get_physical_coords(src) == circ_coords)
    assert np.all(poly_roi.get_single_roi(0).get_physical_coords(src) == poly_coords)

    # Check that extracted images have correct data
    for roi in (rect_roi, circ_roi, poly_roi):
        extracted = sigima.proc.image.extract_roi(src, roi.to_params(src)[0])
        assert np.all(extracted.data != 0), "Extracted image should have non-zero data"
        assert extracted.data.shape == (10, 10), "Extracted image shape mismatch"

    # Display the original image and the ROIs
    if guiutils.is_gui_enabled():
        images = [src]
        titles = ["Original Image"]
        for inverse in (False, True):
            for roi in (rect_roi, circ_roi, poly_roi):
                src2 = src.copy()
                roi.get_single_roi(0).inverse = inverse
                src2.roi = roi
                images.append(src2)
                roi_title = roi.get_single_roi(0).title
                mask_str = "mask inside" if inverse else "mask outside"
                titles.append(f"Image with {roi_title} ROI ({mask_str})")
        guiutils.view_images_side_by_side_if_gui(
            images, titles, rows=2, title="Image ROIs"
        )


def test_create_image_roi_inverse_parameter() -> None:
    """Test create_image_roi function with inverse parameter functionality"""
    # Test 1: Single ROI with inverse=True (mask inside)
    roi1 = sigima.objects.create_image_roi("rectangle", [10, 20, 30, 40], inverse=True)
    assert len(roi1) == 1, "Should create one ROI"
    assert roi1.single_rois[0].inverse is True, "ROI should have inverse=True"

    # Test 2: Single ROI with inverse=False (default)
    roi2 = sigima.objects.create_image_roi("rectangle", [10, 20, 30, 40])
    assert roi2.single_rois[0].inverse is False, "Default should be False"

    # Test 3: Multiple ROIs with global inverse parameter
    coords = [[10, 20, 30, 40], [50, 60, 70, 80]]
    roi3 = sigima.objects.create_image_roi("rectangle", coords, inverse=True)
    assert len(roi3) == 2, "Should create two ROIs"
    assert all(single_roi.inverse is True for single_roi in roi3.single_rois), (
        "All ROIs should have inverse=True (internal representation)"
    )

    # Test 4: Multiple ROIs with individual inverse parameters
    inverse_values = [True, False]  # mask inside, then mask outside
    roi4 = sigima.objects.create_image_roi("rectangle", coords, inverse=inverse_values)
    assert len(roi4) == 2, "Should create two ROIs"
    assert roi4.single_rois[0].inverse is True, "First ROI should be True"
    assert roi4.single_rois[1].inverse is False, "Second ROI should be False"

    # Test 5: Circle ROIs with mixed inverse parameters
    circle_coords = [[50, 50, 25], [150, 150, 30]]
    roi5 = sigima.objects.create_image_roi(
        "circle",
        circle_coords,
        inverse=[False, True],  # mask outside, then mask inside
    )
    assert len(roi5) == 2, "Should create two circle ROIs"
    assert roi5.single_rois[0].inverse is False, "First circle should be False"
    assert roi5.single_rois[1].inverse is True, "Second circle should be True"

    # Test 6: Polygon ROIs with varying vertex counts and mixed inverse parameters
    polygon_coords = [
        [0, 0, 10, 0, 5, 8],  # Triangle (3 vertices)
        [20, 20, 30, 20, 30, 30, 20, 30],  # Rectangle (4 vertices)
    ]
    roi6 = sigima.objects.create_image_roi(
        "polygon",
        polygon_coords,
        inverse=[True, False],  # mask inside, then mask outside
    )
    assert len(roi6) == 2, "Should create two polygon ROIs"
    assert roi6.single_rois[0].inverse is True, "Triangle should be True"
    assert roi6.single_rois[1].inverse is False, "Rectangle should be False"


def test_create_image_roi_inverse_parameter_errors() -> None:
    """Test error handling for inverse parameter in create_image_roi"""
    # Test error when inverse parameter count doesn't match ROI count
    coords = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
    # Only 2 values for 3 ROIs
    inverse_params = [
        True,  # mask inside
        False,  # mask outside
    ]

    with pytest.raises(
        ValueError,
        match=r"Number of inverse values \(2\) must match number of ROIs \(3\)",
    ):
        sigima.objects.create_image_roi("rectangle", coords, inverse=inverse_params)

    # Test with too many inverse values
    # 4 values for 3 ROIs
    inverse_params_too_many = [
        True,  # mask inside
        False,  # mask outside
        True,  # mask inside
        False,  # mask outside
    ]
    with pytest.raises(
        ValueError,
        match=r"Number of inverse values \(4\) must match number of ROIs \(3\)",
    ):
        sigima.objects.create_image_roi(
            "rectangle", coords, inverse=inverse_params_too_many
        )


def test_roi_inverse_affects_mask_generation() -> None:
    """Test that inverse parameter affects mask generation correctly"""
    # Create a test image
    img = __create_test_image()

    # Test rectangle ROI with inverse=True vs inverse=False
    rect_coords = [75, 75, 50, 50]  # Rectangle that should be inside image bounds

    # ROI with inverse=True (mask is True inside the rectangle)
    roi_inside = sigima.objects.create_image_roi("rectangle", rect_coords, inverse=True)
    mask_inside = roi_inside.to_mask(img)

    # ROI with inverse=False (mask is True outside the rectangle)
    roi_outside = sigima.objects.create_image_roi(
        "rectangle", rect_coords, inverse=False
    )
    mask_outside = roi_outside.to_mask(img)

    # The two masks should be inverse of each other
    assert np.array_equal(mask_inside, ~mask_outside), (
        "Inside and outside masks should be inverse of each other"
    )

    # Check that inside mask has True values inside the rectangle region
    # For a rectangle [x0, y0, dx, dy], the region is [x0:x0+dx, y0:y0+dy]
    x0, y0, dx, dy = rect_coords
    expected_inside_region = np.zeros_like(img.data, dtype=bool)
    expected_inside_region[y0 : y0 + dy, x0 : x0 + dx] = True

    assert np.array_equal(mask_inside, expected_inside_region), (
        "Inside mask should match expected rectangular region"
    )


def test_roi_inverse_serialization() -> None:
    """Test that inverse parameter is preserved during
    serialization/deserialization"""
    # Create ROIs with mixed inverse parameters
    coords = [[10, 20, 30, 40], [50, 60, 70, 80]]
    inverse_params = [
        True,  # mask inside
        False,  # mask outside
    ]
    original_roi = sigima.objects.create_image_roi(
        "rectangle", coords, inverse=inverse_params
    )

    # Serialize to dictionary
    roi_dict = original_roi.to_dict()

    # Deserialize from dictionary
    restored_roi = sigima.objects.ImageROI.from_dict(roi_dict)

    # Check that inverse parameters are preserved
    assert len(restored_roi) == len(original_roi), "ROI count should be preserved"
    for i in range(len(original_roi)):
        original_inverse = original_roi.single_rois[i].inverse
        restored_inverse = restored_roi.single_rois[i].inverse
        assert original_inverse == restored_inverse, (
            f"inverse parameter for ROI {i} should be preserved "
            f"(expected {original_inverse}, got {restored_inverse})"
        )


def test_roi_inverse_parameter_conversion() -> None:
    """Test that inverse parameter works correctly with parameter conversion"""
    img = __create_test_image()

    # Create ROI with inverse=True (mask inside)
    roi = sigima.objects.create_image_roi("rectangle", [50, 50, 40, 40], inverse=True)

    # Convert to parameters
    params = roi.to_params(img)
    assert len(params) == 1, "Should create one parameter"

    # Check that inverse parameter is preserved in the parameter
    param = params[0]
    assert hasattr(param, "inverse"), "Parameter should have inverse attribute"
    assert param.inverse is True, "Parameter should preserve inverse=True"

    # Create ROI from parameter and check inverse is preserved
    new_roi = sigima.objects.ImageROI.from_params(img, params)
    assert len(new_roi) == 1, "Should recreate one ROI"
    assert new_roi.single_rois[0].inverse is True, (
        "Recreated ROI should have inverse=True (internal representation)"
    )


def test_multiple_rois_inverse_true() -> None:
    """Test multiple ROIs with inverse=True on distinct areas

    This test checks that when multiple ROIs are defined with inverse=True
    on distinct areas of an image, the resulting mask should have True values
    in BOTH ROI areas, not just their intersection.
    """
    # Create a test image
    img = __create_test_image()

    # Define two rectangular ROIs on distinct areas of the image
    # ROI 1: top-left area
    roi1_coords = [30, 30, 40, 40]  # x, y, width, height
    # ROI 2: bottom-right area (distinct from ROI 1)
    roi2_coords = [130, 130, 40, 40]  # x, y, width, height

    # Create ROI with inverse=True for both rectangles
    roi = sigima.objects.create_image_roi(
        "rectangle", [roi1_coords, roi2_coords], inverse=True
    )

    # Generate the mask
    mask = roi.to_mask(img)

    # Expected behavior: mask should be True in BOTH rectangular areas
    # Create expected mask manually
    expected_mask = np.zeros_like(img.data, dtype=bool)

    # ROI 1 area
    x1, y1, w1, h1 = roi1_coords
    expected_mask[y1 : y1 + h1, x1 : x1 + w1] = True

    # ROI 2 area
    x2, y2, w2, h2 = roi2_coords
    expected_mask[y2 : y2 + h2, x2 : x2 + w2] = True

    # Check that the mask has True values in both ROI areas
    assert np.any(mask[y1 : y1 + h1, x1 : x1 + w1]), (
        "Mask should have True values in first ROI area"
    )
    assert np.any(mask[y2 : y2 + h2, x2 : x2 + w2]), (
        "Mask should have True values in second ROI area"
    )

    # Check that the mask matches our expected mask
    assert np.array_equal(mask, expected_mask), (
        "Mask should have True values in both ROI areas and False elsewhere"
    )

    # Verify that the two ROI areas don't overlap (test integrity)
    roi1_mask = np.zeros_like(img.data, dtype=bool)
    roi1_mask[y1 : y1 + h1, x1 : x1 + w1] = True
    roi2_mask = np.zeros_like(img.data, dtype=bool)
    roi2_mask[y2 : y2 + h2, x2 : x2 + w2] = True
    assert not np.any(roi1_mask & roi2_mask), (
        "Test integrity: ROI areas should not overlap"
    )


def test_multiple_rois_mixed_inverse() -> None:
    """Test multiple ROIs with mixed inverse values

    This test checks that when ROIs have mixed inverse values,
    the combination logic works correctly.
    """
    # Create a test image
    img = __create_test_image()

    # Define three rectangular ROIs
    # ROI 1: top-left area (inverse=True - include this area)
    roi1_coords = [30, 30, 40, 40]  # x, y, width, height
    # ROI 2: top-right area (inverse=False - exclude this area)
    roi2_coords = [130, 30, 40, 40]  # x, y, width, height
    # ROI 3: bottom-left area (inverse=True - include this area)
    roi3_coords = [30, 130, 40, 40]  # x, y, width, height

    # Create ROI with mixed inverse values
    roi = sigima.objects.create_image_roi(
        "rectangle",
        [roi1_coords, roi2_coords, roi3_coords],
        inverse=[True, False, True],  # include, exclude, include
    )

    # Generate the mask
    mask = roi.to_mask(img)

    # Expected behavior:
    # - ROI 1 area should have True values (inverse=True)
    # - ROI 2 area should have False values (inverse=False)
    # - ROI 3 area should have True values (inverse=True)
    # - Areas outside all ROIs should have True values (due to ROI 2 being
    #   inverse=False)

    x1, y1, w1, h1 = roi1_coords
    x2, y2, w2, h2 = roi2_coords
    x3, y3, w3, h3 = roi3_coords

    # Check that ROI 1 and ROI 3 areas have True values (inverse=True)
    assert np.all(mask[y1 : y1 + h1, x1 : x1 + w1]), (
        "ROI 1 area should have True values (inverse=True)"
    )
    assert np.all(mask[y3 : y3 + h3, x3 : x3 + w3]), (
        "ROI 3 area should have True values (inverse=True)"
    )

    # Check that ROI 2 area has False values (inverse=False)
    assert not np.any(mask[y2 : y2 + h2, x2 : x2 + w2]), (
        "ROI 2 area should have False values (inverse=False)"
    )

    # Check that areas outside all ROIs have True values (due to ROI 2)
    # For example, the bottom-right corner should be True
    assert np.all(mask[170:190, 170:190]), (
        "Areas outside all ROIs should have True values"
    )


if __name__ == "__main__":
    guiutils.enable_gui()
    test_roi_coordinates_validation()
    # test_image_roi_merge()
    # test_image_roi_combine()
    # test_image_roi_processing()
    # test_empty_image_roi()
    # test_image_extract_rois()
    # test_image_extract_roi()
    # test_create_image_roi_inside_parameter()
    # test_create_image_roi_inside_parameter_errors()
    # test_roi_inverse()
    # test_roi_inside_serialization()
    # test_roi_inside_parameter_conversion()
