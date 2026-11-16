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
from sigima.tests.data import PeakDataParam, create_peak_image, get_peak2d_data
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
    from sigima import viz

    execenv.print("Peak detection results:")
    items = [viz.create_image(data, colormap="hsv")]
    for x, y in coords:
        items.append(viz.create_marker(x, y))
    viz.view_image_items(
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


def test_peak_image_is_reproducible_and_peaks_are_separated() -> None:
    """A fixed seed produces the same image with well-separated primary peaks."""
    param = sigima.objects.NewImageParam.create(height=256, width=256)
    image = create_peak_image(param, seed=3)
    repeated = create_peak_image(param, seed=3)
    other = create_peak_image(param, seed=6)

    coords = image.metadata["peak_coords"]
    assert coords.shape == (4, 2)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    assert np.min(distances[np.triu_indices(4, 1)]) >= 25.6
    np.testing.assert_array_equal(coords, repeated.metadata["peak_coords"])
    np.testing.assert_array_equal(image.data, repeated.data)
    assert not np.array_equal(coords, other.metadata["peak_coords"])

    unseeded = create_peak_image(param)
    another_unseeded = create_peak_image(param)
    assert not np.array_equal(
        unseeded.metadata["peak_coords"], another_unseeded.metadata["peak_coords"]
    )


def test_peak_data_minimum_distance_is_opt_in() -> None:
    """The first valid layout keeps the original seeded and multi-peak output."""
    first = PeakDataParam.create(size=256)
    second = PeakDataParam.create(size=256)
    original, coords = get_peak2d_data(first, seed=1, multi=True)
    separated, separated_coords = get_peak2d_data(
        second, seed=1, multi=True, min_distance=25.6
    )
    np.testing.assert_array_equal(original, separated)
    np.testing.assert_array_equal(coords, separated_coords)

    single = PeakDataParam.create(size=256, num_peaks=1)
    _, single_coords = get_peak2d_data(single, seed=1, min_distance=25.6)
    assert single_coords.shape == (1, 2)


def test_peak_data_refuses_impossible_minimum_distance() -> None:
    """Impossible spacing cannot silently produce overlapping peaks."""
    param = PeakDataParam.create(size=256, num_peaks=2)
    with pytest.raises(ValueError, match="separat"):
        get_peak2d_data(param, seed=1, min_distance=512)


@pytest.mark.parametrize("min_distance", [-1.0, np.inf, np.nan])
def test_peak_data_refuses_invalid_minimum_distance(min_distance: float) -> None:
    """Invalid distances must not be interpreted as a valid constraint."""
    param = PeakDataParam.create(size=256)
    with pytest.raises(ValueError, match="separation"):
        get_peak2d_data(param, seed=1, min_distance=min_distance)


@pytest.mark.parametrize("dtype", [np.int16, np.float64])
def test_peak2d_signed_integer_range(dtype) -> None:
    """The local ranges of a signed integer image must not overflow.

    Between -30000 and +30000, ``int16`` local ranges wrap around and spurious
    peaks appear in flat regions.
    """
    data = np.full((200, 200), -30000, dtype=dtype)
    data[50, 50] = data[150, 150] = 30000
    data[50, 150] = 0
    coords = get_2d_peaks_coords(data, size=20)
    np.testing.assert_array_equal(sorted(coords.tolist()), [[50, 50], [150, 150]])


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


def test_peak_detection_rois_merged_with_existing():
    """Detected peak ROIs are appended to existing ROIs, not replacing them."""
    data, _coords_expected = get_peak2d_data(seed=1, multi=True)
    obj = sigima.objects.create_image("peak2d_merge_test", data=data)

    # Run detection on the clean image first
    param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
    geometry = sigima.proc.image.peak_detection(obj, param)

    # Pre-define a ROI on the image (as if the user had defined it beforehand)
    obj.roi = sigima.objects.create_image_roi("rectangle", [0, 0, 5, 5])
    existing_rois = list(obj.roi.single_rois)
    n_existing = len(existing_rois)

    assert sigima.proc.image.apply_detection_rois(obj, geometry)
    assert obj.roi is not None

    # The pre-existing ROI must still be present
    for roi in existing_rois:
        assert roi in obj.roi.single_rois, (
            "Existing ROI must be preserved after detection"
        )
    # New ROIs must have been appended
    assert len(obj.roi.single_rois) > n_existing, (
        "Detected ROIs must be appended to existing ROIs"
    )

    roi_count = len(obj.roi)
    assert not sigima.proc.image.apply_detection_rois(obj, geometry)
    assert len(obj.roi) == roi_count


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
