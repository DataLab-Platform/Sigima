# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
HDF5 serialization test for blob detection ROIs
-----------------------------------------------

This test verifies that `GeometryResult` objects with ROI creation metadata
can be properly serialized to HDF5.

Regression test for: `DetectionROIGeometry` enum in `geometry.attrs`
cannot be serialized to HDF5.

See issue #7 in Sigima repository: https://github.com/DataLab-Platform/Sigima/issues/7
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image


def create_simple_blob_image() -> np.ndarray:
    """Create a simple test image with blobs.

    Returns:
        Test image with 4 bright spots
    """
    data = np.zeros((100, 100), dtype=float)
    # Add 4 bright spots (blobs)
    for cx, cy in [(25, 25), (25, 75), (75, 25), (75, 75)]:
        y, x = np.ogrid[-cx : 100 - cx, -cy : 100 - cy]
        mask = x * x + y * y <= 100  # radius 10
        data[mask] = 1.0
    return data


def test_detection_roi_geometry_h5_serialization():
    """Test that detection ROI geometry is stored as string, not enum.

    This test verifies the fix for the HDF5 serialization error:
    "NotImplementedError: cannot serialize 'rectangle' of type
    <enum 'DetectionROIGeometry'>"

    The issue was that store_roi_creation_metadata() stored the enum
    directly in geometry.attrs, but h5py cannot serialize enum types.
    """
    # Create test image with blobs
    data = create_simple_blob_image()
    image = sigima.objects.create_image("blob_test", data)

    # Run blob detection with ROI creation enabled
    param = sigima.params.BlobDOGParam()
    param.min_sigma = 5.0
    param.max_sigma = 15.0
    param.create_rois = True
    param.roi_geometry = sigima.enums.DetectionROIGeometry.RECTANGLE
    result = sigima.proc.image.blob_dog(image, param)

    # Verify that blobs were detected
    assert result is not None, "Blob detection should return a result"
    assert len(result.coords) >= 2, "Need at least 2 blobs for meaningful test"

    # KEY ASSERTION: roi_geometry should be stored as string, not enum
    assert "roi_geometry" in result.attrs, "roi_geometry should be in attrs"
    roi_geom_value = result.attrs["roi_geometry"]
    assert isinstance(roi_geom_value, str), (
        f"roi_geometry should be stored as string, not {type(roi_geom_value)}"
    )
    assert roi_geom_value == "rectangle", (
        f"Expected 'rectangle', got '{roi_geom_value}'"
    )

    # Verify this value can be serialized to HDF5
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write to HDF5 - this should NOT raise an error
        with h5py.File(tmp_path, "w") as f:
            grp = f.create_group("test_result")
            for key, value in result.attrs.items():
                grp.attrs[key] = value

        # Read back and verify
        with h5py.File(tmp_path, "r") as f:
            grp = f["test_result"]
            loaded_value = grp.attrs["roi_geometry"]
            assert loaded_value == "rectangle"
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_detection_roi_geometry_h5_serialization()
    print("Test passed!")
