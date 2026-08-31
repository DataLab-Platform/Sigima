"""Unit tests for format-aware image export preparation."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import features as pillow_features

from sigima.io import (
    ImageExportParam,
    ImageIORegistry,
    encode_image_export_data,
    get_image_export_capabilities,
    get_image_export_writer_kwargs,
    get_supported_export_dtypes,
    prepare_image_export_preview,
    prepare_image_for_export,
    validate_image_export_configuration,
    validate_image_export_options,
    write_image,
    write_image_export_data,
)
from sigima.io.image import (
    ImageExportParam as ImageExportParamFromImage,
)
from sigima.io.image import (
    encode_image_export_data as encode_image_export_data_from_image,
)
from sigima.io.image import (
    write_image_export_data as write_image_export_data_from_image,
)
from sigima.io.image.formats import MatImageFormat, TextImageFormat
from sigima.objects import create_image
from sigima.objects.image import ImageObj
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
    image_format = SimpleNamespace(write_with_options=Mock())

    with patch.object(ImageIORegistry, "get_format", return_value=image_format):
        write_image("image.png", image, param)

    exported_image = image_format.write_with_options.call_args.args[1]
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


def test_gamma_and_invert_are_applied_in_normalized_range() -> None:
    """Check gamma correction and inversion before target-range conversion."""
    data = np.array([0.0, 0.25, 1.0])
    param = create_export_param(
        normalization="manual",
        manual_min=0.0,
        manual_max=1.0,
        target_dtype="uint8",
        gamma=2.0,
        invert=True,
    )
    result = prepare_image_for_export(data, ".png", param)
    np.testing.assert_array_equal(result, np.array([255, 239, 0], dtype=np.uint8))


def test_tonal_adjustments_require_normalization() -> None:
    """Reject ambiguous gamma and inversion without a normalization range."""
    with pytest.raises(ValueError, match="require normalization"):
        prepare_image_for_export(
            np.arange(4.0), ".tiff", create_export_param(gamma=2.0)
        )


def test_write_image_forwards_format_options() -> None:
    """Forward writer options to the resolved image format."""
    image = create_image("Source", data=np.arange(4, dtype=np.uint8).reshape(2, 2))
    param = create_export_param(format_options={"compress_level": 7})
    image_format = SimpleNamespace(write_with_options=Mock())

    with patch.object(
        ImageIORegistry, "get_format", return_value=image_format
    ) as get_format:
        write_image("image.png", image, param)

    get_format.assert_called_once()
    image_format.write_with_options.assert_called_once()
    assert image_format.write_with_options.call_args.args[2] == {
        "compress_level": 7,
        "optimize": False,
    }


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


@pytest.mark.parametrize(
    ("extension", "canonical_extension", "raw_preserving", "option_keys"),
    [
        ("bmp", "bmp", False, set()),
        ("png", "png", False, {"compress_level", "optimize"}),
        (
            "jpg",
            "jpg",
            False,
            {"quality", "subsampling", "progressive", "optimize", "smooth"},
        ),
        (
            "jpeg",
            "jpg",
            False,
            {"quality", "subsampling", "progressive", "optimize", "smooth"},
        ),
        (
            "jp2",
            "jp2",
            False,
            {
                "quality_mode",
                "quality_layers",
                "irreversible",
                "progression",
                "num_resolutions",
                "tile_size",
                "plt",
            },
        ),
        (
            "tif",
            "tif",
            False,
            {
                "compression",
                "compression_level",
                "predictor",
                "rows_per_strip",
                "tile_size",
                "resolution",
                "resolution_unit",
                "photometric",
            },
        ),
        (
            "tiff",
            "tif",
            False,
            {
                "compression",
                "compression_level",
                "predictor",
                "rows_per_strip",
                "tile_size",
                "resolution",
                "resolution_unit",
                "photometric",
            },
        ),
        ("npy", "npy", True, set()),
        ("mat", "mat", True, {"do_compression"}),
        ("txt", "txt", True, {"delimiter", "precision"}),
        ("asc", "txt", True, {"delimiter", "precision"}),
        ("csv", "csv", True, {"delimiter", "precision"}),
        ("h5ima", "h5ima", True, set()),
    ],
)
def test_capabilities_cover_writable_extensions(
    extension: str,
    canonical_extension: str,
    raw_preserving: bool,
    option_keys: set[str],
) -> None:
    """Check capabilities for every writable built-in image extension."""
    capabilities = get_image_export_capabilities(f"image.{extension}")
    assert capabilities.canonical_extension == canonical_extension
    assert capabilities.raw_preserving is raw_preserving
    assert {spec.key for spec in capabilities.option_specs} == option_keys


def test_capability_aliases_share_immutable_objects() -> None:
    """Check canonical aliases reuse the same frozen capability objects."""
    assert get_image_export_capabilities("jpg") is get_image_export_capabilities("jpeg")
    assert get_image_export_capabilities("tif") is get_image_export_capabilities("tiff")
    assert get_image_export_capabilities("txt") is get_image_export_capabilities("asc")


@pytest.mark.parametrize(
    ("extension", "options", "error_type", "message"),
    [
        ("png", {"unknown": 1}, ValueError, "Unsupported PNG"),
        ("png", {"optimize": 1}, TypeError, "must be a bool"),
        ("png", {"compress_level": 10}, ValueError, "must be <= 9"),
        ("jpeg", {"subsampling": "invalid"}, ValueError, "must be one of"),
        ("jp2", {"quality_layers": []}, TypeError, "non-empty numeric list"),
        ("jp2", {"quality_layers": [1.0, 0.0]}, ValueError, "must be > 0"),
        ("jp2", {"tile_size": [128]}, TypeError, "must be a pair"),
        ("tiff", {"resolution": [300.0, -1.0]}, ValueError, "must be > 0"),
        ("tiff", {"rows_per_strip": True}, TypeError, "must be an int"),
    ],
)
def test_invalid_format_options_are_rejected(
    extension: str,
    options: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject unknown, mistyped, out-of-range and malformed options."""
    with pytest.raises(error_type, match=message):
        validate_image_export_options(extension, options)


@pytest.mark.parametrize(
    ("extension", "options"),
    [
        (
            "png",
            {"compress_level": 7, "optimize": True},
        ),
        (
            "jpeg",
            {
                "quality": 92,
                "subsampling": "4:4:4",
                "progressive": True,
                "optimize": True,
                "smooth": 2,
            },
        ),
        (
            "jp2",
            {
                "quality_mode": "rates",
                "quality_layers": [8.0, 4.0],
                "irreversible": True,
                "progression": "RLCP",
                "num_resolutions": 4,
                "tile_size": [128, 256],
                "plt": True,
            },
        ),
    ],
)
def test_pillow_writer_options_are_validated_without_translation(
    extension: str, options: dict[str, object]
) -> None:
    """Check Pillow options retain their documented encoder names."""
    shape = (256, 512) if extension == "jp2" else None
    kwargs = get_image_export_writer_kwargs(extension, options, shape=shape)
    expected = dict(options)
    if extension == "jp2":
        expected["tile_size"] = (128, 256)
    assert kwargs == expected
    assert kwargs is not options


@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        (
            "jpeg",
            {
                "quality": 75,
                "subsampling": "4:2:0",
                "progressive": False,
                "optimize": False,
                "smooth": 0,
            },
        ),
        (
            "jp2",
            {
                "quality_mode": "rates",
                "quality_layers": [20.0],
                "irreversible": False,
                "progression": "LRCP",
                "plt": False,
            },
        ),
    ],
)
def test_writer_kwargs_include_declared_defaults(
    extension: str, expected: dict[str, object]
) -> None:
    """Apply declared writer defaults when no explicit options are supplied."""
    assert get_image_export_writer_kwargs(extension, {}) == expected


def test_tiff_writer_options_are_translated() -> None:
    """Check neutral TIFF names are translated for tifffile."""
    options = {
        "compression": "deflate",
        "compression_level": 8,
        "predictor": "horizontal",
        "tile_size": [64, 64],
        "resolution": [300.0, 300.0],
        "resolution_unit": "inch",
        "photometric": "minisblack",
    }
    assert get_image_export_writer_kwargs("image.tiff", options) == {
        "compression": "deflate",
        "compressionargs": {"level": 8},
        "predictor": 2,
        "tile": (64, 64),
        "resolution": (300.0, 300.0),
        "resolutionunit": 2,
        "photometric": "minisblack",
    }


@pytest.mark.parametrize(
    ("dtype", "options", "message"),
    [
        ("uint8", {"tile_size": [16, 17]}, "multiples of 16"),
        (
            "uint8",
            {"rows_per_strip": 32, "tile_size": [64, 64]},
            "may not be used together",
        ),
        (
            "uint8",
            {"compression": "none", "compression_level": 1},
            "not supported",
        ),
        (
            "uint8",
            {"compression": "lzw", "compression_level": 1},
            "not supported",
        ),
        (
            "uint8",
            {"compression": "deflate", "compression_level": 10},
            "between 0 and 9",
        ),
        (
            "uint8",
            {"compression": "zstd", "compression_level": 0},
            "between 1 and 22",
        ),
        (
            "uint8",
            {"compression": "zstd", "compression_level": 23},
            "between 1 and 22",
        ),
        (
            "uint8",
            {"compression": "jpeg", "compression_level": 0},
            "between 1 and 100",
        ),
        (
            "float32",
            {"compression": "deflate", "predictor": "horizontal"},
            "requires an integer dtype",
        ),
        (
            "uint16",
            {"compression": "deflate", "predictor": "floatingpoint"},
            "requires a floating dtype",
        ),
        (
            "uint8",
            {"compression": "none", "predictor": "horizontal"},
            "requires deflate, lzw, or zstd",
        ),
        (
            "float64",
            {"compression": "jpeg"},
            "requires uint8",
        ),
    ],
)
def test_invalid_tiff_export_configurations_are_rejected(
    dtype: str, options: dict[str, object], message: str
) -> None:
    """Reject TIFF combinations unsupported by tifffile or project policy."""
    with pytest.raises(ValueError, match=message):
        validate_image_export_configuration("tiff", dtype, options)


@pytest.mark.parametrize(
    ("filename", "options", "expected_delimiter", "expected_format"),
    [
        ("image.txt", {"delimiter": "tab", "precision": 6}, "\t", "%.6e"),
        ("image.asc", {"delimiter": "comma", "precision": 5}, ",", "%.5e"),
        ("image.csv", {"delimiter": "semicolon", "precision": 3}, ";", "%.3e"),
    ],
)
def test_text_writer_propagates_delimiter_and_precision(
    filename: str,
    options: dict[str, object],
    expected_delimiter: str,
    expected_format: str,
) -> None:
    """Check text options are converted to np.savetxt kwargs."""
    image = create_image("Source", data=np.arange(4.0).reshape(2, 2))
    with patch("sigima.io.image.formats.np.savetxt") as savetxt:
        TextImageFormat().write_with_options(filename, image, options)
    assert savetxt.call_args.kwargs == {
        "fmt": expected_format,
        "delimiter": expected_delimiter,
    }


def test_mat_writer_propagates_compression() -> None:
    """Check MAT compression reaches scipy.io.savemat."""
    image = create_image("Source", data=np.arange(4).reshape(2, 2))
    with patch("sigima.io.image.formats.sio.savemat") as savemat:
        MatImageFormat().write_with_options(
            "image.mat", image, {"do_compression": True}
        )
    assert savemat.call_args.kwargs == {"do_compression": True}
    np.testing.assert_array_equal(savemat.call_args.args[1]["img"], image.data)


@pytest.mark.parametrize("extension", ["npy", "mat", "txt", "asc", "csv", "h5ima"])
def test_auto_preserves_raw_scientific_dtype(extension: str) -> None:
    """Check automatic scientific export preserves every valid image dtype."""
    for dtype in ImageObj.VALID_DTYPES:
        source = np.arange(4).reshape(2, 2).astype(dtype)
        result = prepare_image_for_export(source, extension, create_export_param())
        assert result.dtype == source.dtype
        assert result is not source


def test_jpeg_preview_round_trips_in_memory() -> None:
    """Check JPEG preview reflects encoder quality and preserves shape and dtype."""
    source = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
    low_quality = prepare_image_export_preview(
        source, "jpeg", create_export_param(format_options={"quality": 5})
    )
    high_quality = prepare_image_export_preview(
        source, "jpeg", create_export_param(format_options={"quality": 95})
    )
    assert low_quality.shape == source.shape
    assert low_quality.dtype == np.uint8
    assert not np.array_equal(low_quality, high_quality)


@pytest.mark.parametrize(
    ("shape", "options", "message"),
    [
        (None, {"num_resolutions": 2}, "requires a 2D image shape"),
        ((8, 8), {"num_resolutions": 5}, "must be <= 4"),
        (
            (64, 64),
            {"num_resolutions": 6, "tile_size": [16, 16]},
            "must be <= 5",
        ),
    ],
)
def test_invalid_jp2_resolution_geometry_is_rejected(
    shape: tuple[int, ...] | None,
    options: dict[str, object],
    message: str,
) -> None:
    """Reject explicit JP2 resolution counts unsupported by the geometry."""
    with pytest.raises(ValueError, match=message):
        validate_image_export_configuration("jp2", "uint8", options, shape)


def test_small_jp2_uses_openjpeg_default_resolutions() -> None:
    """Round-trip a small JP2 image without overriding num_resolutions."""
    if not pillow_features.check("jpg_2000"):
        pytest.skip("Pillow was built without JPEG 2000/OpenJPEG support")
    source = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    preview = prepare_image_export_preview(
        source,
        "jp2",
        create_export_param(
            target_dtype="uint8", format_options={"quality_layers": [1.0]}
        ),
    )
    np.testing.assert_array_equal(preview, source)


@pytest.mark.parametrize(
    ("extension", "dtype", "format_options"),
    [
        ("png", "uint8", {}),
        ("tiff", "float32", {}),
        ("jp2", "uint16", {"quality_layers": [1.0], "num_resolutions": 4}),
    ],
)
def test_classic_export_encode_decode_smoke(
    extension: str, dtype: str, format_options: dict[str, object]
) -> None:
    """Encode and decode classic formats through the production helpers."""
    if extension == "jp2" and not pillow_features.check("jpg_2000"):
        pytest.skip("Pillow was built without JPEG 2000/OpenJPEG support")
    source = np.arange(64 * 64).reshape(64, 64).astype(dtype)
    if np.issubdtype(source.dtype, np.floating):
        source /= source.size
    param = create_export_param(target_dtype=dtype, format_options=format_options)
    prepared = prepare_image_for_export(source, extension, param)
    options = validate_image_export_configuration(
        extension, prepared.dtype, format_options, prepared.shape
    )
    preview = prepare_image_export_preview(source, extension, param)
    capabilities = get_image_export_capabilities(extension)
    encoded = encode_image_export_data(prepared, extension, options)
    decoded = iio.imread(
        BytesIO(encoded),
        extension=f".{capabilities.canonical_extension}",
        plugin=capabilities.writer_backend,
    )
    np.testing.assert_array_equal(preview, decoded)
    np.testing.assert_array_equal(decoded, prepared)


def test_option_dictionaries_are_never_mutated() -> None:
    """Check validation and writer dispatch defensively copy caller options."""
    quality_layers = [8.0, 4.0]
    options = {"quality_layers": quality_layers}
    validated = validate_image_export_options("jp2", options)
    assert validated == options
    assert validated is not options
    assert validated["quality_layers"] is not quality_layers

    assert encode_image_export_data_from_image is encode_image_export_data
    assert write_image_export_data_from_image is write_image_export_data

    image = create_image("Source", data=np.arange(4, dtype=np.uint8).reshape(2, 2))
    png_options = {"compress_level": 7}
    param = create_export_param(format_options=png_options)
    image_format = SimpleNamespace(write_with_options=Mock())
    with patch.object(ImageIORegistry, "get_format", return_value=image_format):
        write_image("image.png", image, param)
    forwarded = image_format.write_with_options.call_args.args[2]
    assert forwarded == {"compress_level": 7, "optimize": False}
    assert forwarded is not png_options


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
