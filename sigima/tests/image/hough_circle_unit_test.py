# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image peak detection test using circle Hough transform
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import numpy as np
import pytest
from skimage.feature import canny

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests import guiutils
from sigima.tests.data import get_peak2d_data
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result, validate_detection_rois
from sigima.tools.image import get_hough_circle_peaks


def __compute_hough_circle_peaks(data):
    """Compute peaks using circle Hough transform, return coordinates"""
    edges = canny(
        data,
        sigma=30,
        low_threshold=0.6,
        high_threshold=0.8,
        use_quantiles=True,
    )
    coords = get_hough_circle_peaks(
        edges, min_radius=25, max_radius=35, min_distance=70
    )
    execenv.print(f"Coordinates: {coords}")
    return edges, coords


def __exec_hough_circle_test(data):
    """Peak detection using circle Hough transform with visualization"""
    # pylint: disable=import-outside-toplevel
    from sigima.tests import vistools

    edges, coords = __compute_hough_circle_peaks(data)
    items = [
        vistools.create_image(data, colormap="gray"),
        vistools.create_image(
            np.array(edges, dtype=np.uint8), colormap="hsv", alpha_function="tanh"
        ),
    ]
    for shapeargs in coords:
        xc, yc, r = shapeargs
        item = vistools.create_circle(xc, yc, r)
        items.append(item)
    vistools.view_image_items(items)


@pytest.mark.validation
def test_image_hough_circle_peaks():
    """Validation test for Hough circle peaks detection"""
    # Create synthetic circular features for testing
    size = 200
    data = np.zeros((size, size), dtype=np.uint8)

    # Add known circles with specific centers and radii
    y, x = np.ogrid[:size, :size]

    # Circle 1: center at (50, 50), radius 30
    ring1 = ((x - 50) ** 2 + (y - 50) ** 2 <= 30**2) & (
        (x - 50) ** 2 + (y - 50) ** 2 >= 28**2
    )
    data[ring1] = 255

    # Circle 2: center at (150, 150), radius 30
    ring2 = ((x - 150) ** 2 + (y - 150) ** 2 <= 30**2) & (
        (x - 150) ** 2 + (y - 150) ** 2 >= 28**2
    )
    data[ring2] = 255

    # Apply edge detection first (hough_circle_peaks expects edge-detected data)
    edges = canny(
        data,
        sigma=1,
        low_threshold=0.1,
        high_threshold=0.2,
    )

    # Expected circle coordinates
    expected_coords = np.array([[50, 50, 30], [150, 150, 30]], dtype=float)

    # Test with and without ROI creation
    for create_rois in (False, True):
        for roi_geometry in sigima.enums.DetectionROIGeometry:
            if (
                not create_rois
                and roi_geometry != sigima.enums.DetectionROIGeometry.RECTANGLE
            ):
                continue

            # Create ImageObj for testing
            obj = sigima.objects.create_image(
                "hough_circle_test", data=edges.astype(np.uint8)
            )

            # Set up parameters for hough_circle_peaks
            param = sigima.params.HoughCircleParam()
            param.min_radius = 25
            param.max_radius = 35
            param.min_distance = 50
            param.create_rois = create_rois
            param.roi_geometry = roi_geometry

            # Test the hough_circle_peaks function
            geometry = sigima.proc.image.hough_circle_peaks(obj, param)
            sigima.proc.image.apply_detection_rois(obj, geometry)

            # Check that we got a GeometryResult
            assert geometry is not None, (
                "hough_circle_peaks should return a GeometryResult"
            )
            assert hasattr(geometry, "coords"), (
                "GeometryResult should have coords attribute"
            )

            coords = geometry.coords
            execenv.print(f"Detected coordinates (ROIs={create_rois}): {coords}")

            # Verify we detected the expected number of circles
            assert coords.shape[0] == expected_coords.shape[0], (
                f"Expected {expected_coords.shape[0]} circles, got {coords.shape[0]}"
            )

            # Check coordinates (tolerance for discretization and edge detection)
            check_array_result(
                "Hough circle centers and radii",
                coords,
                expected_coords,
                atol=5.0,  # Allow 5 pixel tolerance for center coordinates and radius
                sort=True,  # Sort to handle detection order differences
            )

            # Validate ROI creation
            validate_detection_rois(obj, coords, create_rois, roi_geometry)


@pytest.mark.gui
def test_hough_circle():
    """2D peak detection test"""
    data, _coords = get_peak2d_data(multi=False)
    with guiutils.lazy_qt_app_context(force=True):
        __exec_hough_circle_test(data)


if __name__ == "__main__":
    test_hough_circle()
