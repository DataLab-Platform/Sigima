# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Regression test: detection ROIs on images with non-unit axis scaling

When an image has non-default pixel spacing (dx, dy ≠ 1.0), ROIs created by
detection algorithms (peak detection, blob detection, …) must be positioned in
**physical** coordinates, not pixel indices.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import pytest

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
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


@pytest.mark.validation
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


@pytest.mark.validation
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


@pytest.mark.validation
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


if __name__ == "__main__":
    test_peak_detection_rois_with_non_unit_pixel_spacing()
    test_peak_detection_rois_with_unit_pixel_spacing()
    test_roi_center_matches_physical_peak_for_various_spacings()
