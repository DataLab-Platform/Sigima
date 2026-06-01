# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Inverse ROI detection regression test
--------------------------------------

Regression test for: detection functions (blob detection, peak detection,
Hough circles) failing when used with inverse ROI logic.

Three root causes were fixed:

1. ``ImageObj.get_data()`` cropped to the ROI shape bounding box even for
   inverse ROIs, returning a tiny mostly-masked area instead of the full image.
2. Detection functions in ``sigima.tools.image.detection`` passed
   ``MaskedArray`` directly to algorithms that don't support it.
3. ``compute_geometry_from_obj`` applied wrong coordinate offsets for inverse
   ROIs and did not filter out false detections on masked pixels.

See Sigima issue #23.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.env import execenv

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None

# Bounding box of the inverse ROI used in all tests (masks pixels inside).
# Covers the blob at (50, 50) with generous margin so it's fully inside.
_ROI_X0, _ROI_Y0, _ROI_X1, _ROI_Y1 = 25, 25, 75, 75

# Number of blobs that lie entirely outside the ROI region.
_EXPECTED_BLOBS_OUTSIDE = 3


def _create_image_with_blobs() -> sigima.objects.ImageObj:
    """Create a 200x200 test image with 4 well-separated circular blobs.

    Blob centers are at (50, 50), (50, 150), (150, 50), (150, 150) with
    radius 12 px. The inverse ROI [25..75, 25..75] fully covers the first
    blob, so exactly 3 blobs are outside the masked region.

    Returns:
        ImageObj with 4 blobs (normalised float64, range [0, 1])
    """
    size = 200
    data = np.zeros((size, size), dtype=np.float64)
    y_grid, x_grid = np.ogrid[:size, :size]
    for cx, cy in [(50, 50), (50, 150), (150, 50), (150, 150)]:
        mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 < 12**2
        data[mask] = 1.0
    return sigima.objects.create_image("inverse_roi_test", data=data)


def _add_inverse_roi(obj: sigima.objects.ImageObj) -> None:
    """Add an inverse rectangular ROI that masks one blob.

    The ROI rectangle covers the blob at (40, 40). With inverse=True,
    the mask is True inside the rectangle, so detection should find the
    other 3 blobs outside.
    """
    obj.roi = sigima.objects.create_image_roi(
        "rectangle",
        [_ROI_X0, _ROI_Y0, _ROI_X1, _ROI_Y1],
        inverse=True,
    )


def _assert_no_detection_inside_mask(coords: np.ndarray) -> None:
    """Assert that no detection center falls inside the masked ROI region."""
    for row in coords:
        x, y = row[0], row[1]
        assert not (_ROI_X0 <= x <= _ROI_X1 and _ROI_Y0 <= y <= _ROI_Y1), (
            f"Detection at ({x:.0f}, {y:.0f}) is inside the masked region"
        )


def test_blob_dog_inverse_roi():
    """Blob DoG detection with inverse ROI returns results outside the mask."""
    obj = _create_image_with_blobs()
    _add_inverse_roi(obj)

    param = sigima.params.BlobDOGParam.create(
        min_sigma=5.0,
        max_sigma=20.0,
        threshold_rel=0.05,
        overlap=0.5,
        exclude_border=False,
    )
    result = sigima.proc.image.blob_dog(obj, param)

    assert result is not None, "Detection should return a result"
    assert len(result.coords) == _EXPECTED_BLOBS_OUTSIDE, (
        f"Should detect exactly {_EXPECTED_BLOBS_OUTSIDE} blobs outside the "
        f"inverse ROI, got {len(result.coords)}"
    )
    _assert_no_detection_inside_mask(result.coords)
    execenv.print(f"✓ DoG inverse ROI: detected {len(result.coords)} blobs")


def test_blob_log_inverse_roi():
    """Blob LoG detection with inverse ROI returns results outside the mask."""
    obj = _create_image_with_blobs()
    _add_inverse_roi(obj)

    param = sigima.params.BlobLOGParam.create(
        min_sigma=5.0,
        max_sigma=20.0,
        threshold_rel=0.05,
        overlap=0.5,
        exclude_border=False,
    )
    result = sigima.proc.image.blob_log(obj, param)

    assert result is not None, "Detection should return a result"
    assert len(result.coords) == _EXPECTED_BLOBS_OUTSIDE, (
        f"Should detect exactly {_EXPECTED_BLOBS_OUTSIDE} blobs, "
        f"got {len(result.coords)}"
    )
    _assert_no_detection_inside_mask(result.coords)
    execenv.print(f"✓ LoG inverse ROI: detected {len(result.coords)} blobs")


def test_blob_doh_inverse_roi():
    """Blob DoH detection with inverse ROI returns results outside the mask."""
    obj = _create_image_with_blobs()
    _add_inverse_roi(obj)

    param = sigima.params.BlobDOHParam.create(
        min_sigma=5.0,
        max_sigma=20.0,
        threshold_rel=0.05,
        overlap=0.5,
    )
    result = sigima.proc.image.blob_doh(obj, param)

    assert result is not None, "Detection should return a result"
    assert len(result.coords) == _EXPECTED_BLOBS_OUTSIDE, (
        f"Should detect exactly {_EXPECTED_BLOBS_OUTSIDE} blobs, "
        f"got {len(result.coords)}"
    )
    _assert_no_detection_inside_mask(result.coords)
    execenv.print(f"✓ DoH inverse ROI: detected {len(result.coords)} blobs")


@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not installed")
def test_blob_opencv_inverse_roi():
    """Blob OpenCV detection with inverse ROI does not crash or detect inside."""
    obj = _create_image_with_blobs()
    _add_inverse_roi(obj)

    param = sigima.params.BlobOpenCVParam.create(
        min_threshold=10.0,
        max_threshold=200.0,
        min_repeatability=2,
        min_dist_between_blobs=10.0,
        filter_by_color=False,
        filter_by_area=True,
        min_area=10.0,
        max_area=1000.0,
        filter_by_circularity=False,
        filter_by_inertia=False,
        filter_by_convexity=False,
    )
    # Main regression check: this must not raise RuntimeWarning or crash
    result = sigima.proc.image.blob_opencv(obj, param)

    assert result is not None, "Detection should return a result"
    assert len(result.coords) == _EXPECTED_BLOBS_OUTSIDE, (
        f"Should detect exactly {_EXPECTED_BLOBS_OUTSIDE} blobs, "
        f"got {len(result.coords)}"
    )
    _assert_no_detection_inside_mask(result.coords)
    execenv.print(f"✓ OpenCV inverse ROI: detected {len(result.coords)} blobs")


def test_peak2d_inverse_roi():
    """2D peak detection with inverse ROI returns results outside the mask."""
    obj = _create_image_with_blobs()
    _add_inverse_roi(obj)

    param = sigima.params.Peak2DDetectionParam()
    result = sigima.proc.image.peak_detection(obj, param)

    assert result is not None, "Detection should return a result"
    assert len(result.coords) == _EXPECTED_BLOBS_OUTSIDE, (
        f"Should detect exactly {_EXPECTED_BLOBS_OUTSIDE} peaks, "
        f"got {len(result.coords)}"
    )
    _assert_no_detection_inside_mask(result.coords)
    execenv.print(f"✓ Peak2D inverse ROI: detected {len(result.coords)} peaks")


if __name__ == "__main__":
    test_blob_dog_inverse_roi()
    test_blob_log_inverse_roi()
    test_blob_doh_inverse_roi()
    if CV2_AVAILABLE:
        test_blob_opencv_inverse_roi()
    test_peak2d_inverse_roi()
    print("All inverse ROI detection tests passed!")
