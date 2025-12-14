# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image peak detection test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import time

import numpy as np
import pytest

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests import guiutils
from sigima.tests.data import get_peak2d_data
from sigima.tests.env import execenv
from sigima.tests.helpers import check_array_result, validate_detection_rois
from sigima.tools.image import get_2d_peaks_coords


def exec_image_peak_detection_func(data: np.ndarray) -> np.ndarray:
    """Execute image peak detection function

    Args:
        data: 2D image data

    Returns:
        2D array of peak coordinates
    """
    t0 = time.time()
    coords = get_2d_peaks_coords(data)
    dt = time.time() - t0
    execenv.print(f"Calculation time: {int(dt * 1e3):d} ms")
    execenv.print(f"  => {coords.tolist()}")
    return coords


def view_image_peak_detection(data: np.ndarray, coords: np.ndarray) -> None:
    """View image peak detection results

    Args:
        data: 2D image data
        coords: Coordinates of detected peaks (shape: (n, 2))
    """
    # pylint: disable=import-outside-toplevel
    from sigima.tests import vistools

    execenv.print("Peak detection results:")
    items = [vistools.create_image(data, colormap="hsv")]
    for x, y in coords:
        items.append(vistools.create_marker(x, y))
    vistools.view_image_items(
        items, name=view_image_peak_detection.__name__, title="Peak Detection"
    )


def test_peak2d_unit():
    """2D peak detection unit test"""
    data, coords_expected = get_peak2d_data(seed=1, multi=False)
    coords = exec_image_peak_detection_func(data)
    assert coords.shape == coords_expected.shape, (
        f"Expected {coords_expected.shape[0]} peaks, got {coords.shape[0]}"
    )
    # Absolute tolerance is set to 2 pixels, as coordinates are in pixel units
    # and the algorithm may detect peaks at slightly different pixel locations
    # Convert coordinates to float64 for dtype compatibility with expected results
    coords_float = coords.astype(np.float64)
    check_array_result(
        "Peak coords (sigima.tools.image.)",
        coords_float,
        coords_expected,
        atol=2,
        sort=True,
    )


@pytest.mark.validation
def test_image_peak_detection():
    """2D peak detection unit test"""
    data, coords_expected = get_peak2d_data(seed=1, multi=False)
    for create_rois in (True, False):
        for roi_geometry in sigima.enums.DetectionROIGeometry:
            if (
                not create_rois
                and roi_geometry != sigima.enums.DetectionROIGeometry.RECTANGLE
            ):
                continue
            obj = sigima.objects.create_image("peak2d_unit_test", data=data)
            param = sigima.params.Peak2DDetectionParam.create(
                create_rois=create_rois, roi_geometry=roi_geometry
            )
            geometry = sigima.proc.image.peak_detection(obj, param)
            # Apply ROIs from detection result
            sigima.proc.image.apply_detection_rois(obj, geometry)
            coords = geometry.coords
            assert coords.shape == coords_expected.shape, (
                f"Expected {coords_expected.shape[0]} peaks, got {coords.shape[0]}"
            )
            # Absolute tolerance is set to 2 pixels, as coordinates are in pixel units
            # and the algorithm may detect peaks at slightly different pixel locations
            check_array_result(
                "Peak coords (comp.)", coords, coords_expected, atol=2, sort=True
            )
            # Validate ROI creation
            validate_detection_rois(obj, coords, create_rois, roi_geometry)


@pytest.mark.gui
def test_peak2d_interactive():
    """2D peak detection interactive test"""
    data, _coords = get_peak2d_data(multi=False)
    coords = exec_image_peak_detection_func(data)
    with guiutils.lazy_qt_app_context(force=True):
        view_image_peak_detection(data, coords)


if __name__ == "__main__":
    test_peak2d_unit()
    test_image_peak_detection()
    test_peak2d_interactive()
