# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Regression test: detection ROIs on images with non-unit axis scaling

When an image has non-default pixel spacing (dx, dy ≠ 1.0), ROIs created by
detection algorithms (peak detection, blob detection, …) must be positioned in
**physical** coordinates, not pixel indices.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
from skimage.draw import disk

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.enums import ContourShape
from sigima.objects.image.roi import CircularROI
from sigima.tests.data import get_peak2d_data
from sigima.tests.helpers import validate_detection_rois


def _make_scaled_image(factor: float = 2.0) -> sigima.objects.ImageObj:
    """Return a standard peak-detection test image with a non-unit pixel spacing.

    The underlying pixel data is identical to the default test image, but the
    image's pixel spacing is set to *factor* so that
    ``physical_coord = factor × pixel_index``.

    Args:
        factor: Pixel spacing to apply (default 2.0).

    Returns:
        An ImageObj whose calibration uses *factor* as pixel spacing.
    """
    data, _ = get_peak2d_data(seed=1, multi=False)
    obj = sigima.objects.create_image("scaled_peak_test", data=data)
    obj.set_uniform_coords(dx=float(factor), dy=float(factor), x0=0.0, y0=0.0)
    return obj


def test_peak_detection_rois_with_non_unit_pixel_spacing() -> None:
    """ROIs created by peak detection must be correctly positioned on a scaled image.

    Regression test for the ``indices=True`` bug in
    ``create_image_roi_around_points``: with a pixel spacing of 2.0, the ROI
    bounding boxes must still enclose the physical coordinates of the detected
    peaks.
    """
    obj_base = _make_scaled_image(factor=2.0)

    for roi_geometry in sigima.enums.DetectionROIGeometry:
        obj = obj_base.copy()
        param = sigima.params.Peak2DDetectionParam.create(
            create_rois=True,
            roi_geometry=roi_geometry,
        )
        geometry = sigima.proc.image.peak_detection(obj, param)
        if geometry is None or len(geometry) < 2:
            continue

        sigima.proc.image.apply_detection_rois(obj, geometry)

        # validate_detection_rois checks that each ROI bbox (in physical coords)
        # contains the corresponding detected peak center (also physical coords).
        validate_detection_rois(
            obj,
            geometry.coords,
            create_rois=True,
            roi_geometry=roi_geometry,
        )


def test_peak_detection_rois_with_unit_pixel_spacing() -> None:
    """With default pixel spacing (dx = dy = 1), behaviour is unchanged.

    This test acts as a baseline: the fix must not regress the standard case.
    """
    obj_base = _make_scaled_image(factor=1.0)

    for roi_geometry in sigima.enums.DetectionROIGeometry:
        obj = obj_base.copy()
        param = sigima.params.Peak2DDetectionParam.create(
            create_rois=True,
            roi_geometry=roi_geometry,
        )
        geometry = sigima.proc.image.peak_detection(obj, param)
        if geometry is None or len(geometry) < 2:
            continue

        sigima.proc.image.apply_detection_rois(obj, geometry)
        validate_detection_rois(
            obj,
            geometry.coords,
            create_rois=True,
            roi_geometry=roi_geometry,
        )


def test_roi_center_matches_physical_peak_for_various_spacings() -> None:
    """ROI center must track the physical peak position for several dx values.

    For each spacing factor *s*, a peak detected at pixel index *p* has the
    physical coordinate *s × p*.  The ROI bounding box (in physical coords)
    must contain *s × p*, regardless of *s*.
    """
    for factor in (1.0, 2.0, 0.5, 3.0):
        obj_base = _make_scaled_image(factor=factor)
        obj = obj_base.copy()
        param = sigima.params.Peak2DDetectionParam.create(
            create_rois=True,
            roi_geometry=sigima.enums.DetectionROIGeometry.RECTANGLE,
        )
        geometry = sigima.proc.image.peak_detection(obj, param)
        if geometry is None or len(geometry) < 2:
            continue

        sigima.proc.image.apply_detection_rois(obj, geometry)
        validate_detection_rois(
            obj,
            geometry.coords,
            create_rois=True,
            roi_geometry=sigima.enums.DetectionROIGeometry.RECTANGLE,
        )


def test_contour_roi_circle_with_non_unit_pixel_spacing() -> None:
    """Contour circle ROIs must use physical coordinates on a scaled image.

    Two disks are drawn at pixel centres (50, 50) and (150, 150), each with
    pixel radius 25.  With dx = dy = 2.0 the expected physical coordinates are:
      - circle 0: xc ≈ 100, yc ≈ 100, r ≈ 50
      - circle 1: xc ≈ 300, yc ≈ 300, r ≈ 50
    not the raw pixel values (50/25 and 150/25).
    """
    # Build a synthetic image: two well-separated white disks, dx = dy = 2.0
    data = np.zeros((200, 200), dtype=np.uint8)
    rr0, cc0 = disk((50, 50), 25, shape=data.shape)
    rr1, cc1 = disk((150, 150), 25, shape=data.shape)
    data[rr0, cc0] = 255
    data[rr1, cc1] = 255
    obj = sigima.objects.create_image("circle_scaled", data=data)
    obj.set_uniform_coords(dx=2.0, dy=2.0, x0=0.0, y0=0.0)

    # ContourShapeParam.create_rois is a ValueProp-managed item; set it manually.
    param = sigima.params.ContourShapeParam.create(shape=ContourShape.CIRCLE)
    param.create_rois = True

    geometry = sigima.proc.image.contour_shape(obj, param)
    assert geometry is not None, "contour_shape must detect the circles"
    assert len(geometry) == 2, f"Expected exactly 2 circles, got {len(geometry)}"

    ok = sigima.proc.image.apply_detection_rois(obj, geometry)
    assert ok, "apply_detection_rois must return True when contours are detected"
    assert obj.roi is not None, "ROIs must be created on the image"

    # The ROIs must be CircularROIs in physical coordinates.
    rois = obj.roi.single_rois
    assert len(rois) == 2, f"Expected exactly 2 ROIs, got {len(rois)}"

    for roi in rois:
        assert isinstance(roi, CircularROI), (
            f"Expected CircularROI, got {type(roi).__name__}"
        )

    # Sort by xc so the order is deterministic.
    sorted_rois = sorted(rois, key=lambda roi: roi.coords[0])

    # coords = [xc, yc, r] already in physical units.
    # pixel (50, 50) × dx=2 → physical (100, 100), r=25×2=50
    xc0, yc0, r0 = sorted_rois[0].coords
    assert abs(xc0 - 100.0) < 5.0, f"xc0={xc0} must be near 100 (physical coords)"
    assert abs(yc0 - 100.0) < 5.0, f"yc0={yc0} must be near 100 (physical coords)"
    assert abs(r0 - 50.0) < 5.0, f"r0={r0} must be near 50 (physical units)"

    # pixel (150, 150) × dx=2 → physical (300, 300), r=25×2=50
    xc1, yc1, r1 = sorted_rois[1].coords
    assert abs(xc1 - 300.0) < 5.0, f"xc1={xc1} must be near 300 (physical coords)"
    assert abs(yc1 - 300.0) < 5.0, f"yc1={yc1} must be near 300 (physical coords)"
    assert abs(r1 - 50.0) < 5.0, f"r1={r1} must be near 50 (physical units)"


if __name__ == "__main__":
    test_peak_detection_rois_with_non_unit_pixel_spacing()
    test_peak_detection_rois_with_unit_pixel_spacing()
    test_roi_center_matches_physical_peak_for_various_spacings()
    test_contour_roi_circle_with_non_unit_pixel_spacing()
