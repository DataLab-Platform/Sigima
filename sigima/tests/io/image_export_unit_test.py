"""Unit tests for format-aware image export preparation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sigima.io import (
    ImageExportParam,
    get_supported_export_dtypes,
    prepare_image_for_export,
    write_image,
)
from sigima.io.image import ImageExportParam as ImageExportParamFromImage
from sigima.objects import create_image
from sigima.params import ImageExportParam as ImageExportParamFromParams


def create_fake_dicom() -> SimpleNamespace:
    """Create a minimal mutable DICOM-like template for export-copy tests."""
    return SimpleNamespace(
        **{
            "ImagePositionPatient": [1.5, 2.5],
            "PixelSpacing": [0.1, 0.2],
            "NestedSequence": [{"values": [1, 2]}],
        }
    )


def create_export_param(**values) -> ImageExportParam:
    """Create image export parameters with selected values."""
    param = ImageExportParam.create()
    for name, value in values.items():
        setattr(param, name, value)
    return param


def test_write_image_without_param_uses_original_path() -> None:
    """Check that omitting parameters delegates the original object unchanged."""
    image = create_image("Source", data=np.arange(4).reshape(2, 2))
    with patch("sigima.io.convenience.ImageIORegistry.write") as registry_write:
        write_image("image.png", image)
    registry_write.assert_called_once_with("image.png", image)


def test_write_image_with_param_preserves_source() -> None:
    """Check that prepared export delegates a metadata-preserving copy."""
    source_data = np.linspace(0.0, 1.0, 4).reshape(2, 2)
    image = create_image("Source", data=source_data)
    image.dicom_template = create_fake_dicom()
    image.metadata["instrument"] = {"settings": ["test"]}
    param = create_export_param(normalization="minmax", target_dtype="uint8")

    with patch("sigima.io.convenience.ImageIORegistry.write") as registry_write:
        write_image("image.png", image, param)

    exported_image = registry_write.call_args.args[1]
    assert exported_image is not image
    assert exported_image.metadata == image.metadata
    assert exported_image.metadata is not image.metadata
    assert exported_image.metadata["instrument"] is not image.metadata["instrument"]
    assert (
        exported_image.metadata["instrument"]["settings"]
        is not image.metadata["instrument"]["settings"]
    )
    assert exported_image.dicom_template is not image.dicom_template
    assert (
        exported_image.dicom_template.NestedSequence
        is not image.dicom_template.NestedSequence
    )
    assert exported_image.data.dtype == np.uint8
    np.testing.assert_array_equal(image.data, source_data)


def test_minmax_rescale_float_ramp_to_uint8() -> None:
    """Check min-max rescaling of a float ramp to the full uint8 range."""
    data = np.linspace(-1.0, 1.0, 256)
    param = create_export_param(normalization="minmax", target_dtype="uint8")
    result = prepare_image_for_export(data, ".png", param)
    np.testing.assert_array_equal(result, np.arange(256, dtype=np.uint8))


def test_percentile_and_manual_normalization() -> None:
    """Check percentile-derived and explicit normalization bounds."""
    data = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    percentile_param = create_export_param(
        normalization="percentile",
        low_percentile=0.0,
        high_percentile=75.0,
        target_dtype="uint8",
    )
    percentile_result = prepare_image_for_export(data, "image.png", percentile_param)
    np.testing.assert_array_equal(
        percentile_result, np.array([0, 85, 170, 255, 255], dtype=np.uint8)
    )

    manual_param = create_export_param(
        normalization="manual",
        manual_min=1.0,
        manual_max=3.0,
        target_dtype="uint8",
    )
    manual_result = prepare_image_for_export(data, "png", manual_param)
    np.testing.assert_array_equal(
        manual_result, np.array([0, 0, 128, 255, 255], dtype=np.uint8)
    )


def test_extreme_finite_bounds_rescale_without_overflow() -> None:
    """Check rescaling when the finite bound difference exceeds float64."""
    maximum = np.finfo(np.float64).max
    data = np.array([-maximum, 0.0, maximum])
    param = create_export_param(normalization="minmax", target_dtype="uint8")
    result = prepare_image_for_export(data, ".png", param)
    np.testing.assert_array_equal(result, np.array([0, 128, 255], dtype=np.uint8))


def test_float32_export_rejects_finite_out_of_range_values() -> None:
    """Check that narrowing finite float64 values cannot produce infinity."""
    data = np.array([0.0, float(np.finfo(np.float32).max) * 2.0])
    param = create_export_param(target_dtype="float32")
    with pytest.raises(ValueError, match="outside the finite range of float32"):
        prepare_image_for_export(data, ".tiff", param)


@pytest.mark.parametrize(
    ("normalization", "target_dtype", "expected_dtype"),
    [("minmax", "uint8", np.uint8), ("percentile", "float32", np.float32)],
)
def test_constant_derived_range_rescales_to_target_minimum(
    normalization: str, target_dtype: str, expected_dtype: np.dtype
) -> None:
    """Check deterministic rescaling for constant derived ranges."""
    data = np.full((2, 2), 7.0)
    param = create_export_param(
        normalization=normalization,
        behavior="rescale",
        target_dtype=target_dtype,
    )
    result = prepare_image_for_export(data, ".tiff", param)
    assert result.dtype == expected_dtype
    np.testing.assert_array_equal(result, np.zeros((2, 2), dtype=expected_dtype))


def test_constant_derived_range_clip_keeps_value_before_safe_conversion() -> None:
    """Check clipping semantics for a constant derived range."""
    data = np.full((2, 2), 300.0)
    param = create_export_param(
        normalization="percentile", behavior="clip", target_dtype="uint8"
    )
    result = prepare_image_for_export(data, ".png", param)
    np.testing.assert_array_equal(result, np.full((2, 2), 255, dtype=np.uint8))


def test_clip_and_rescale_are_distinct() -> None:
    """Check direct clipping versus full target-range rescaling."""
    data = np.array([-1.0, 0.0, 1.0, 2.0])
    common = {
        "normalization": "manual",
        "manual_min": 0.0,
        "manual_max": 1.0,
        "target_dtype": "uint8",
    }
    clipped = prepare_image_for_export(
        data, ".png", create_export_param(**common, behavior="clip")
    )
    rescaled = prepare_image_for_export(
        data, ".png", create_export_param(**common, behavior="rescale")
    )
    np.testing.assert_array_equal(clipped, np.array([0, 0, 1, 1], dtype=np.uint8))
    np.testing.assert_array_equal(rescaled, np.array([0, 0, 255, 255], dtype=np.uint8))


def test_nonfinite_policies() -> None:
    """Check error, replacement and clipping policies for NaN and infinities."""
    data = np.array([np.nan, -np.inf, 2.0, 5.0, np.inf])
    with pytest.raises(ValueError, match="NaN or infinite"):
        prepare_image_for_export(data, ".tiff", create_export_param())

    replaced = prepare_image_for_export(
        data,
        ".tiff",
        create_export_param(
            nonfinite_policy="replace",
            replacement_value=3.0,
            target_dtype="float64",
        ),
    )
    np.testing.assert_array_equal(replaced, np.array([3.0, 3.0, 2.0, 5.0, 3.0]))

    clipped = prepare_image_for_export(
        data,
        ".tiff",
        create_export_param(nonfinite_policy="clip", target_dtype="float64"),
    )
    np.testing.assert_array_equal(clipped, np.array([2.0, 2.0, 2.0, 5.0, 5.0]))


def test_all_nonfinite_policies() -> None:
    """Check deterministic handling when no finite source sample exists."""
    data = np.array([np.nan, -np.inf, np.inf])
    with pytest.raises(ValueError, match="NaN or infinite"):
        prepare_image_for_export(data, ".tiff", create_export_param())
    with pytest.raises(ValueError, match="without finite data"):
        prepare_image_for_export(
            data, ".tiff", create_export_param(nonfinite_policy="clip")
        )
    replaced = prepare_image_for_export(
        data,
        ".tiff",
        create_export_param(
            nonfinite_policy="replace",
            replacement_value=4.0,
            target_dtype="float64",
        ),
    )
    np.testing.assert_array_equal(replaced, np.full(3, 4.0))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"normalization": "manual", "manual_min": 1.0, "manual_max": 1.0},
            "finite and ordered",
        ),
        (
            {
                "normalization": "percentile",
                "low_percentile": 90.0,
                "high_percentile": 10.0,
            },
            "Percentile bounds",
        ),
    ],
)
def test_invalid_normalization_ranges(values: dict, message: str) -> None:
    """Check that invalid explicitly requested ranges are rejected."""
    data = np.ones((2, 2))
    with pytest.raises(ValueError, match=message):
        prepare_image_for_export(data, ".png", create_export_param(**values))


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            np.array([-1000, 0, 255, 1000], dtype=np.int64),
            np.array([0, 0, 255, 255], dtype=np.uint8),
        ),
        (
            np.array([0, 255, 1000], dtype=np.uint64),
            np.array([0, 255, 255], dtype=np.uint8),
        ),
    ],
)
def test_integer_conversion_clips_without_wraparound(
    data: np.ndarray, expected: np.ndarray
) -> None:
    """Check safe clipping from signed and unsigned integer arrays."""
    result = prepare_image_for_export(
        data, ".png", create_export_param(target_dtype="uint8")
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("extension", "target_dtype"),
    [
        (".png", "uint16"),
        (".jpg", "float32"),
        (".bmp", "float64"),
        (".jp2", "float32"),
    ],
)
def test_per_format_dtype_constraints(extension: str, target_dtype: str) -> None:
    """Check that unsupported explicit dtype/format combinations are rejected."""
    param = create_export_param(target_dtype=target_dtype)
    with pytest.raises(ValueError, match="is not supported"):
        prepare_image_for_export(np.arange(4), extension, param)


def test_auto_dtype_resolution_and_public_choices() -> None:
    """Check format-aware automatic dtypes and public choices for DataLab."""
    source = np.arange(4, dtype=np.int32)
    assert (
        prepare_image_for_export(source, ".png", create_export_param()).dtype
        == np.uint8
    )
    assert (
        prepare_image_for_export(source, ".jp2", create_export_param()).dtype
        == np.uint16
    )
    assert (
        prepare_image_for_export(source, ".tiff", create_export_param()).dtype
        == np.float64
    )
    npy_result = prepare_image_for_export(source, ".npy", create_export_param())
    assert npy_result.dtype == source.dtype
    assert get_supported_export_dtypes("image.jp2") == ("auto", "uint8", "uint16")


def test_public_exports_and_format_aliases() -> None:
    """Check public parameter imports and equivalent format aliases."""
    assert ImageExportParamFromImage is ImageExportParam
    assert ImageExportParamFromParams is ImageExportParam
    assert get_supported_export_dtypes("jpg") == get_supported_export_dtypes("jpeg")
    assert get_supported_export_dtypes("tif") == get_supported_export_dtypes("tiff")


def test_conditional_parameter_fields() -> None:
    """Check stable conditional activation of normalization-policy fields."""
    param = create_export_param()
    items = {item.get_name(): item for item in param.get_items()}

    param.normalization = "none"
    assert not items["manual_min"].get_prop_value("display", param, "active")
    assert not items["low_percentile"].get_prop_value("display", param, "active")
    param.normalization = "manual"
    assert items["manual_min"].get_prop_value("display", param, "active")
    param.normalization = "percentile"
    assert items["low_percentile"].get_prop_value("display", param, "active")

    param.nonfinite_policy = "error"
    assert not items["replacement_value"].get_prop_value("display", param, "active")
    param.nonfinite_policy = "replace"
    assert items["replacement_value"].get_prop_value("display", param, "active")
