# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Masked array handling in detection functions
---------------------------------------------

Detection functions rely on underlying libraries (scikit-image, OpenCV, SciPy)
that do not support ``numpy.ma.MaskedArray``: the mask is silently ignored.
The ``warn_if_masked`` decorator emits a warning in that case, while passing
the masked array unchanged to the computation.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest
from numpy import ma
from skimage.draw import circle_perimeter, disk

import sigima.objects
import sigima.params
import sigima.proc.image
import sigima.tools.image

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None

MASKED_WARNING_MATCH = "does not support masked arrays"


def _make_two_peaks_image(size: int = 200) -> np.ndarray:
    """Create a test image with two Gaussian peaks at (50, 50) and (150, 150)."""
    y, x = np.mgrid[:size, :size]
    peak1 = np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / (2 * 8.0**2))
    peak2 = np.exp(-((x - 150) ** 2 + (y - 150) ** 2) / (2 * 8.0**2))
    return 1000.0 * (peak1 + peak2)


def _make_blob_image(size: int = 100) -> np.ndarray:
    """Create a test image with a single disk-shaped blob in the center."""
    data = np.zeros((size, size), dtype=np.float64)
    rows, cols = disk((size // 2, size // 2), 10)
    data[rows, cols] = 1.0
    return data


def _mask_around(data: np.ndarray, x: int, y: int, half_size: int) -> ma.MaskedArray:
    """Return a masked array with a square masked region centered on (x, y)."""
    masked = ma.masked_array(data, mask=np.zeros_like(data, dtype=bool))
    masked.mask[y - half_size : y + half_size, x - half_size : x + half_size] = True
    return masked


def test_peaks_masked_array_warns_and_runs() -> None:
    """Peak detection on a masked array warns; the computation still runs.

    The masked array is passed unchanged to the underlying libraries, which do
    not support masks: the exact result is therefore not asserted here (it may
    be unexpected inside or near masked areas — that is what the warning is
    about).
    """
    data = _make_two_peaks_image()
    masked = _mask_around(data, 150, 150, 30)
    with pytest.warns(UserWarning, match=MASKED_WARNING_MATCH):
        coords = sigima.tools.image.get_2d_peaks_coords(masked)
    # The computation runs and returns point coordinates
    assert isinstance(coords, np.ndarray)
    assert coords.ndim == 2 and coords.shape[1] == 2


def test_peaks_plain_array_no_warning() -> None:
    """Peak detection on a plain array does not warn and finds both peaks."""
    data = _make_two_peaks_image()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coords = sigima.tools.image.get_2d_peaks_coords(data)
    assert coords.shape == (2, 2)


def test_peaks_masked_array_without_masked_values_no_warning() -> None:
    """A masked array without any masked value does not warn."""
    data = _make_two_peaks_image()
    masked = ma.masked_array(data, mask=np.zeros_like(data, dtype=bool))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coords = sigima.tools.image.get_2d_peaks_coords(masked)
    assert coords.shape == (2, 2)


@pytest.mark.parametrize(
    "func_name", ["find_blobs_dog", "find_blobs_doh", "find_blobs_log"]
)
def test_blobs_masked_array_warns_and_runs(func_name: str) -> None:
    """Blob detection functions warn on masked arrays and still run."""
    data = _make_blob_image()
    # Mask a corner region (away from the blob)
    masked = _mask_around(data, 10, 10, 10)
    func = getattr(sigima.tools.image, func_name)
    with pytest.warns(UserWarning, match=MASKED_WARNING_MATCH):
        coords = func(masked)
    # The blob at the center must still be detected
    assert coords.shape[0] >= 1
    assert np.allclose(coords[0, :2], (50, 50), atol=3)


@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV is not available")
def test_blobs_opencv_masked_array_warns_and_runs() -> None:
    """OpenCV blob detection warns on masked arrays and still runs."""
    data = _make_blob_image()
    masked = _mask_around(data, 10, 10, 10)
    with pytest.warns(UserWarning, match=MASKED_WARNING_MATCH):
        coords = sigima.tools.image.find_blobs_opencv(masked, blob_color=255)
    assert coords.shape[1] == 3


def test_hough_circle_masked_array_warns_and_runs() -> None:
    """Hough circle detection warns on masked arrays and still runs."""
    data = np.zeros((100, 100), dtype=np.float64)
    rows, cols = circle_perimeter(50, 50, 20)
    data[rows, cols] = 1.0
    masked = _mask_around(data, 10, 10, 10)
    with pytest.warns(UserWarning, match=MASKED_WARNING_MATCH):
        coords = sigima.tools.image.get_hough_circle_peaks(
            masked, min_radius=15, max_radius=25
        )
    assert coords.shape[0] >= 1
    assert np.allclose(coords[0], (50, 50, 20), atol=2)


def test_peak_detection_with_circular_roi_warns() -> None:
    """End-to-end: peak detection on an image with a circular ROI warns.

    A circular ROI produces a masked array (corners of the bounding box are
    masked), which must trigger the masked-array warning while still returning
    a valid geometry result.
    """
    data = _make_two_peaks_image()
    obj = sigima.objects.create_image("masked_roi_test", data=data)
    obj.roi = sigima.objects.create_image_roi("circle", [50, 50, 40])
    param = sigima.params.Peak2DDetectionParam.create(threshold=0.5)
    with pytest.warns(UserWarning, match=MASKED_WARNING_MATCH):
        geometry = sigima.proc.image.peak_detection(obj, param)
    assert geometry is not None
    assert len(geometry) == 1


if __name__ == "__main__":
    test_peaks_masked_array_warns_and_runs()
    test_peaks_plain_array_no_warning()
    test_peaks_masked_array_without_masked_values_no_warning()
    for name in ("find_blobs_dog", "find_blobs_doh", "find_blobs_log"):
        test_blobs_masked_array_warns_and_runs(name)
    if CV2_AVAILABLE:
        test_blobs_opencv_masked_array_warns_and_runs()
    test_hough_circle_masked_array_warns_and_runs()
    test_peak_detection_with_circular_roi_warns()
