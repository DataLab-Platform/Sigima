# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for image processing parameter validation."""

from __future__ import annotations

import numpy as np
import pytest
from guidata.config import ValidationMode, temporary_validation_mode

from sigima.objects import ImageObj, create_image
from sigima.proc.image import GridParam
from sigima.proc.image.detection import (
    BlobDOGParam,
    BlobOpenCVParam,
    GenericDetectionParam,
    HoughCircleParam,
    blob_opencv,
    hough_circle_peaks,
)
from sigima.proc.image.edges import CannyParam
from sigima.proc.image.extraction import ROIGridParam
from sigima.proc.image.filtering import GaussianFreqFilterParam
from sigima.proc.image.geometry import (
    Resampling2DParam,
    ResizeParam,
    UniformCoordsParam,
    resampling,
)
from sigima.proc.image.preprocessing import BinningParam, binning
from sigima.validation import validate_dataset


def create_test_image() -> ImageObj:
    """Create a small image with uniform unit coordinates."""
    image = create_image("test", np.arange(24.0).reshape(4, 6))
    image.set_uniform_coords(1.0, 1.0, 0.0, 0.0)
    return image


def test_grid_counts_and_zero_sized_roi_cells() -> None:
    """Grid counts are signed and nonzero; zero-sized ROI cells remain valid."""
    assert GridParam.create(direction="col", cols=-3).cols == -3
    assert GridParam.create(direction="row", rows=-4).rows == -4
    assert ROIGridParam.create(nx=1, ny=1, xsize=0, ysize=0).xsize == 0

    with pytest.raises(ValueError, match="Zero is not"):
        GridParam.create(cols=0)
    with pytest.raises(ValueError, match="lower than minimum"):
        ROIGridParam.create(nx=0)


def test_relative_detection_threshold_endpoints() -> None:
    """Relative detection thresholds span the full normalized interval."""
    assert GenericDetectionParam.create(threshold=0.0).threshold == 0.0
    assert GenericDetectionParam.create(threshold=1.0).threshold == 1.0
    with pytest.raises(ValueError, match="lower than minimum"):
        GenericDetectionParam.create(threshold=-0.01)
    with pytest.raises(ValueError, match="greater than maximum"):
        GenericDetectionParam.create(threshold=1.01)


def test_blob_scale_and_hough_radius_intervals() -> None:
    """Blob scales may be equal, while Hough radii must be strictly ordered."""
    validate_dataset(BlobDOGParam.create(min_sigma=2.0, max_sigma=2.0))
    with pytest.raises(ValueError, match="min_sigma must be less"):
        validate_dataset(BlobDOGParam.create(min_sigma=3.0, max_sigma=2.0))

    source = create_test_image()
    with pytest.raises(ValueError, match="min_radius must be strictly less"):
        hough_circle_peaks(source, HoughCircleParam.create(min_radius=2, max_radius=2))
    with pytest.raises(ValueError, match="min_radius must be strictly less"):
        hough_circle_peaks(source, HoughCircleParam.create(min_radius=3, max_radius=2))


@pytest.mark.parametrize(
    ("enabled_field", "lower_field", "upper_field"),
    (
        ("filter_by_area", "min_area", "max_area"),
        ("filter_by_circularity", "min_circularity", "max_circularity"),
        ("filter_by_inertia", "min_inertia_ratio", "max_inertia_ratio"),
        ("filter_by_convexity", "min_convexity", "max_convexity"),
    ),
)
def test_opencv_blob_enabled_filter_intervals(
    enabled_field: str, lower_field: str, upper_field: str
) -> None:
    """Sigima's hook validates an optional interval only when it is enabled."""
    param = BlobOpenCVParam()
    setattr(param, enabled_field, True)
    setattr(param, lower_field, 0.8)
    setattr(param, upper_field, 0.2)
    with pytest.raises(ValueError, match="must be less than or equal"):
        validate_dataset(param)

    setattr(param, enabled_field, False)
    validate_dataset(param)


def test_opencv_blob_ignores_valid_disabled_filter_interval() -> None:
    """A disabled valid interval does not alter the public OpenCV result."""
    pytest.importorskip("cv2")
    ycoords, xcoords = np.ogrid[:100, :100]
    data = np.zeros((100, 100), dtype=float)
    data[(xcoords - 50) ** 2 + (ycoords - 50) ** 2 < 10**2] = 1.0
    image = create_image("blob", data)
    common = {
        "min_threshold": 10.0,
        "max_threshold": 200.0,
        "min_repeatability": 2,
        "min_dist_between_blobs": 10.0,
        "filter_by_color": False,
        "blob_color": 0,
        "filter_by_area": True,
        "min_area": 10.0,
        "max_area": 1000.0,
        "filter_by_circularity": False,
        "filter_by_inertia": False,
        "filter_by_convexity": False,
    }

    first = blob_opencv(
        image,
        BlobOpenCVParam.create(**common, min_circularity=0.1, max_circularity=1.0),
    )
    second = blob_opencv(
        image,
        BlobOpenCVParam.create(**common, min_circularity=0.2, max_circularity=0.9),
    )

    assert first is not None
    assert second is not None
    np.testing.assert_allclose(first.coords, second.coords)


def test_opencv_blob_threshold_and_color_bounds() -> None:
    """Threshold ordering and byte colors apply even with color filtering off."""
    validate_dataset(
        BlobOpenCVParam.create(
            min_threshold=2.0,
            max_threshold=2.0,
            filter_by_color=False,
            blob_color=255,
        )
    )
    with pytest.raises(ValueError, match="min_threshold must be less"):
        validate_dataset(BlobOpenCVParam.create(min_threshold=3.0, max_threshold=2.0))
    with pytest.raises(ValueError, match="greater than maximum"):
        BlobOpenCVParam.create(filter_by_color=False, blob_color=256)


def test_canny_threshold_validation() -> None:
    """Canny quantiles are normalized, while absolute thresholds are unbounded."""
    validate_dataset(
        CannyParam.create(low_threshold=0.5, high_threshold=0.5, use_quantiles=True)
    )
    validate_dataset(
        CannyParam.create(low_threshold=2.0, high_threshold=3.0, use_quantiles=False)
    )
    with pytest.raises(ValueError, match="low_threshold must be less"):
        validate_dataset(
            CannyParam.create(low_threshold=0.8, high_threshold=0.2, use_quantiles=True)
        )
    with pytest.raises(ValueError, match="less than or equal to 1"):
        validate_dataset(
            CannyParam.create(low_threshold=0.5, high_threshold=1.1, use_quantiles=True)
        )


def test_positive_scale_and_identity_binning_bounds() -> None:
    """Scale divisors are positive and one-pixel binning remains an identity."""
    with pytest.raises(ValueError, match="Zero is not"):
        GaussianFreqFilterParam.create(sigma=0.0)
    with pytest.raises(ValueError, match="Zero is not"):
        ResizeParam.create(zoom=0.0)

    source = create_test_image()
    result = binning(source, BinningParam.create(sx=1, sy=1))
    assert np.array_equal(result.data, source.data)


def test_signed_pixel_spacing_is_nonzero() -> None:
    """Image axes may descend, but a zero pixel spacing is invalid."""
    param = UniformCoordsParam.create(dx=-1.0, dy=-2.0)
    assert param.dx == -1.0
    assert param.dy == -2.0
    with pytest.raises(ValueError, match="Zero is not"):
        UniformCoordsParam.create(dx=0.0)


def test_resampling_uses_source_bounds_without_mutation() -> None:
    """Missing output bounds are resolved from the source without being stored."""
    source = create_test_image()
    param = Resampling2DParam.create(mode="shape", width=6, height=4)
    validate_dataset(param, source)
    assert param.xmin is None
    assert param.xmax is None
    assert param.ymin is None
    assert param.ymax is None


def test_resampling_accepts_descending_axes() -> None:
    """Descending extents work in shape mode and with matching negative steps."""
    source = create_test_image()
    shape_param = Resampling2DParam.create(
        mode="shape",
        xmin=6.0,
        xmax=0.0,
        ymin=4.0,
        ymax=0.0,
        dx=0.0,
        dy=0.0,
        width=6,
        height=4,
    )
    result = resampling(source, shape_param)
    assert result.data.shape == (4, 6)
    assert result.dx == -1.0
    assert result.dy == -1.0

    dxy_param = Resampling2DParam.create(
        mode="dxy",
        xmin=6.0,
        xmax=0.0,
        ymin=4.0,
        ymax=0.0,
        dx=-1.0,
        dy=-1.0,
        width=0,
        height=-1,
    )
    dxy_result = resampling(source, dxy_param)
    assert dxy_result.data.shape == (4, 6)
    assert dxy_param.width == 0
    assert dxy_param.height == -1


@pytest.mark.parametrize("step", [1e-200, -1e-200])
def test_resampling_accepts_tiny_signed_pixel_sizes(step: float) -> None:
    """Same-signed extents and steps remain valid when their product underflows."""
    source = create_test_image()
    source.set_uniform_coords(step, step, 0.0, 0.0)
    param = Resampling2DParam.create(mode="dxy", dx=step, dy=step, fill_value=0.0)

    result = resampling(source, param)

    assert result.data.shape == source.data.shape
    assert np.isfinite(result.data).all()
    assert result.dx == step
    assert result.dy == step


def test_resampling_validates_fields_after_mode_switch() -> None:
    """Inactive geometry values are preserved and checked only when activated."""
    source = create_test_image()
    shape_param = Resampling2DParam.create(
        mode="shape", width=6, height=4, dx=0.0, dy=0.0
    )
    resampling(source, shape_param)
    shape_param.mode = "dxy"
    with pytest.raises(ValueError, match="dx and dy must be nonzero"):
        resampling(source, shape_param)
    assert (shape_param.dx, shape_param.dy) == (0.0, 0.0)

    dxy_param = Resampling2DParam.create(mode="dxy", dx=1.0, dy=1.0, width=0, height=-1)
    resampling(source, dxy_param)
    dxy_param.mode = "shape"
    with pytest.raises(ValueError, match="width and height must be at least 1"):
        resampling(source, dxy_param)
    assert (dxy_param.width, dxy_param.height) == (0, -1)

    with temporary_validation_mode(ValidationMode.DISABLED):
        invalid = Resampling2DParam.create(
            mode="dxy", dx=0.0, dy=1.0, width=6, height=4
        )
        with pytest.raises(ValueError, match="dx and dy must be nonzero"):
            resampling(source, invalid)


def test_resampling_rejects_invalid_active_geometry() -> None:
    """Resampling rejects empty extents, missing fields, and mismatched signs."""
    source = create_test_image()
    with pytest.raises(ValueError, match="extents must be nonzero"):
        validate_dataset(
            Resampling2DParam.create(
                mode="shape",
                xmin=0.0,
                xmax=0.0,
                ymin=0.0,
                ymax=4.0,
                width=6,
                height=4,
            ),
            source,
        )
    with pytest.raises(ValueError, match="dx and dy must be specified"):
        validate_dataset(Resampling2DParam.create(mode="dxy"), source)
    with pytest.raises(ValueError, match="same sign"):
        validate_dataset(
            Resampling2DParam.create(
                mode="dxy",
                xmin=6.0,
                xmax=0.0,
                ymin=4.0,
                ymax=0.0,
                dx=1.0,
                dy=-1.0,
            ),
            source,
        )
