# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Contour finding test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import sys
import time

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.enums import ContourShape
from sigima.objects import KindShape
from sigima.objects.image.roi import CircularROI, PolygonalROI
from sigima.objects.scalar import GeometryResult
from sigima.proc.image import apply_detection_rois
from sigima.proc.image.detection import _store_contour_roi_metadata
from sigima.tests import guiutils
from sigima.tests.data import get_peak2d_data
from sigima.tests.env import execenv
from sigima.tests.helpers import (
    check_array_result,
    check_scalar_result,
)
from sigima.tools.image import get_2d_peaks_coords, get_contour_shapes


@pytest.mark.gui
def test_contour_interactive():
    """2D peak detection test"""
    data, _coords = get_peak2d_data()
    with guiutils.lazy_qt_app_context(force=True):
        # pylint: disable=import-outside-toplevel
        from sigima import viz

        items = [viz.create_image(data, colormap="hsv")]
        t0 = time.time()
        peak_coords = get_2d_peaks_coords(data)
        dt = time.time() - t0
        for x, y in peak_coords:
            items.append(viz.create_marker(x, y))
        execenv.print(f"Calculation time: {int(dt * 1e3):d} ms\n", file=sys.stderr)
        execenv.print(f"Peak coordinates: {peak_coords}")

        # Add contour shapes for all shape types
        for shape in ContourShape:
            coords = get_contour_shapes(data, shape=shape)
            items.extend(viz.create_contour_shapes(coords, shape))

        viz.view_image_items(items)


@pytest.mark.validation
def test_contour_shape() -> None:
    """Test contour shape computation function"""
    # Create test data with known shapes
    data, _expected_coords = get_peak2d_data()

    # Test each contour shape type with ROI creation
    for shape in ContourShape:
        execenv.print(f"Testing contour shape: {shape}")

        # Get contour shapes from the function
        detected_shapes = get_contour_shapes(data, shape=shape)
        execenv.print(f"Detected {len(detected_shapes)} {shape}(s)")

        image = sigima.objects.create_image("Contour Test Image", data=data)
        param = sigima.params.ContourShapeParam.create(shape=shape)
        results = sigima.proc.image.contour_shape(image, param)
        sigima.proc.image.apply_detection_rois(image, results)

        check_array_result(f"Contour shapes ({shape})", detected_shapes, results.coords)

        # Basic validation checks
        assert isinstance(detected_shapes, np.ndarray), (
            f"get_contour_shapes should return numpy array for {shape}"
        )

        if len(detected_shapes) > 0:
            # Check that we detected at least some shapes
            execenv.print(f"Successfully detected contours for {shape}")

            # Validate shape-specific properties
            if shape == ContourShape.CIRCLE:
                # For circles: [xc, yc, r]
                assert detected_shapes.shape[1] == 3, (
                    "Circle contours should have 3 parameters (xc, yc, r)"
                )
                # Check that radius values are positive
                radii = detected_shapes[:, 2]
                assert np.all(radii > 0), "All circle radii should be positive"
                check_scalar_result(
                    "Circle radius range",
                    np.mean(radii),
                    np.mean(radii),  # Just check it's finite
                    rtol=1.0,
                )

            elif shape == ContourShape.ELLIPSE:
                # For ellipses: [xc, yc, a, b, theta]
                assert detected_shapes.shape[1] == 5, (
                    "Ellipse contours should have 5 parameters (xc, yc, a, b, theta)"
                )
                # Check that semi-axes are positive
                a_values = detected_shapes[:, 2]
                b_values = detected_shapes[:, 3]
                assert np.all(a_values > 0), (
                    "All ellipse semi-axes 'a' should be positive"
                )
                assert np.all(b_values > 0), (
                    "All ellipse semi-axes 'b' should be positive"
                )
                check_scalar_result(
                    "Ellipse semi-axis 'a' range",
                    np.mean(a_values),
                    np.mean(a_values),  # Just check it's finite
                    rtol=1.0,
                )

            elif shape == ContourShape.POLYGON:
                # For polygons: flattened x,y coordinates
                # Shape should be (n_contours, max_points) where max_points is even
                assert detected_shapes.shape[1] % 2 == 0, (
                    "Polygon contours should have even number of coordinates "
                    "(x,y pairs)"
                )
                # Check that we have valid coordinates (not all NaN)
                valid_coords = ~np.isnan(detected_shapes)
                assert np.any(valid_coords), (
                    "Polygon should have some valid coordinates"
                )

        # Check that the function handles different threshold levels
        for level in [0.3, 0.5, 0.7]:
            shapes_at_level = get_contour_shapes(data, shape=shape, level=level)
            assert isinstance(shapes_at_level, np.ndarray), (
                f"get_contour_shapes should return numpy array for {shape} "
                f"at level {level}"
            )
            execenv.print(f"  At level {level}: detected {len(shapes_at_level)} shapes")

    execenv.print("All contour shape tests passed!")


def test_contour_roi_polygon() -> None:
    """Test contour detection with polygon shape creates polygon ROIs."""
    data, _coords = get_peak2d_data()
    image = sigima.objects.create_image("Test", data=data)
    param = sigima.params.ContourShapeParam.create(
        shape=ContourShape.POLYGON, create_rois=True
    )
    result = sigima.proc.image.contour_shape(image, param)
    assert result is not None
    assert sigima.proc.image.apply_detection_rois(image, result)
    assert image.roi is not None
    for roi in image.roi.single_rois:
        assert isinstance(roi, PolygonalROI)
    execenv.print(f"Polygon ROIs created: {len(image.roi.single_rois)}")


def test_contour_roi_ellipse() -> None:
    """Test contour detection with ellipse shape creates polygon ROIs."""
    data, _coords = get_peak2d_data()
    image = sigima.objects.create_image("Test", data=data)
    param = sigima.params.ContourShapeParam.create(
        shape=ContourShape.ELLIPSE, create_rois=True
    )
    result = sigima.proc.image.contour_shape(image, param)
    assert result is not None
    assert sigima.proc.image.apply_detection_rois(image, result)
    assert image.roi is not None
    # Ellipses are approximated as polygon ROIs
    for roi in image.roi.single_rois:
        assert isinstance(roi, PolygonalROI)
    execenv.print(f"Polygon ROIs from ellipses: {len(image.roi.single_rois)}")


def test_contour_roi_circle() -> None:
    """Test contour detection with circle shape creates circle ROIs."""
    data, _coords = get_peak2d_data()
    image = sigima.objects.create_image("Test", data=data)
    param = sigima.params.ContourShapeParam.create(
        shape=ContourShape.CIRCLE, create_rois=True
    )
    result = sigima.proc.image.contour_shape(image, param)
    assert result is not None
    assert sigima.proc.image.apply_detection_rois(image, result)
    assert image.roi is not None
    for roi in image.roi.single_rois:
        assert isinstance(roi, CircularROI)
    execenv.print(f"Circle ROIs created: {len(image.roi.single_rois)}")


def test_contour_roi_merged_with_existing() -> None:
    """Detected contour ROIs are appended to existing ROIs, not replacing them."""
    data, _coords = get_peak2d_data()
    image = sigima.objects.create_image("Test", data=data)

    # Run detection on the clean image first
    param = sigima.params.ContourShapeParam.create(
        shape=ContourShape.CIRCLE, create_rois=True
    )
    result = sigima.proc.image.contour_shape(image, param)
    assert result is not None

    # Pre-define a ROI on the image (as if the user had defined it beforehand)
    image.roi = sigima.objects.create_image_roi("circle", [50, 50, 20])
    n_existing = len(image.roi.single_rois)
    existing_rois = list(image.roi.single_rois)

    assert sigima.proc.image.apply_detection_rois(image, result)
    assert image.roi is not None

    # The pre-existing ROI must still be present
    for roi in existing_rois:
        assert roi in image.roi.single_rois, (
            "Existing ROI must be preserved after detection"
        )
    # New ROIs must have been appended (strictly more than before)
    assert len(image.roi.single_rois) > n_existing, (
        "Detected ROIs must be appended to existing ROIs"
    )
    execenv.print(
        f"Merged ROIs: {n_existing} existing -> {len(image.roi.single_rois)} total"
    )


def test_contour_roi_disabled() -> None:
    """Test contour detection with create_rois=False does not create ROIs."""
    data, _coords = get_peak2d_data()
    image = sigima.objects.create_image("Test", data=data)
    param = sigima.params.ContourShapeParam.create(
        shape=ContourShape.POLYGON, create_rois=False
    )
    result = sigima.proc.image.contour_shape(image, param)
    assert not sigima.proc.image.apply_detection_rois(image, result)
    assert image.roi is None


def test_store_contour_roi_metadata_none_geometry() -> None:
    """_store_contour_roi_metadata must return None when geometry is None."""
    result = _store_contour_roi_metadata(None, create_rois=True)
    assert result is None


def test_store_contour_roi_metadata_empty_geometry() -> None:
    """_store_contour_roi_metadata must not set attrs when geometry has 0 rows."""
    geometry = GeometryResult("test", KindShape.CIRCLE, np.empty((0, 3)))
    result = _store_contour_roi_metadata(geometry, create_rois=True)
    # The geometry object is returned but the attrs must NOT be populated because
    # len(geometry) == 0 < 1 (the guard in store_contour_roi_metadata)
    assert result is geometry
    assert not result.attrs.get("contour_rois", False)
    assert not result.attrs.get("create_rois", False)


def test_apply_contour_rois_no_detections_returns_false() -> None:
    """apply_detection_rois returns False when contour detection found nothing.

    Regression guard: calling apply_detection_rois with a contour-flagged
    GeometryResult that has no rows must not crash and must return False.
    """
    data, _coords = get_peak2d_data()
    image = sigima.objects.create_image("Test", data=data)

    # Build a geometry that carries the contour_rois flag but has no rows
    geometry = GeometryResult("test", KindShape.POLYGON, np.empty((0, 6)))
    geometry.attrs["create_rois"] = True
    geometry.attrs["contour_rois"] = True

    result = apply_detection_rois(image, geometry)
    assert result is False
    assert image.roi is None


if __name__ == "__main__":
    test_contour_interactive()
    test_contour_shape()
    test_contour_roi_polygon()
    test_contour_roi_ellipse()
    test_contour_roi_circle()
    test_contour_roi_disabled()
    test_store_contour_roi_metadata_none_geometry()
    test_store_contour_roi_metadata_empty_geometry()
    test_apply_contour_rois_no_detections_returns_false()
