"""Format-aware image export preparation."""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass
from io import BytesIO
from numbers import Real
from types import MappingProxyType
from typing import BinaryIO, Literal, Mapping

import guidata.dataset as gds
import imageio.v3 as iio
import numpy as np

from sigima.config import _
from sigima.objects.image import ImageObj

__all__ = [
    "IMAGE_EXPORT_CAPABILITIES",
    "ImageExportCapabilities",
    "ImageExportOptionKind",
    "ImageExportOptionSpec",
    "ImageExportParam",
    "encode_image_export_data",
    "get_image_export_capabilities",
    "get_image_export_writer_kwargs",
    "get_supported_export_dtypes",
    "prepare_image_export_preview",
    "prepare_image_for_export",
    "validate_image_export_configuration",
    "validate_image_export_options",
    "write_image_export_data",
]

ImageExportOptionKind = Literal[
    "bool",
    "int",
    "float",
    "choice",
    "int_pair",
    "float_pair",
    "float_list",
    "string",
]


@dataclass(frozen=True)
class ImageExportOptionSpec:
    """Describe one serializable image writer option."""

    key: str
    value_kind: ImageExportOptionKind
    default: object = None
    choices: tuple[object, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    advanced: bool = False
    allow_none: bool = False
    minimum_inclusive: bool = True


@dataclass(frozen=True)
class ImageExportCapabilities:
    """Describe immutable export capabilities shared by format aliases."""

    format_name: str
    canonical_extension: str
    extension_aliases: tuple[str, ...]
    supported_dtypes: tuple[str, ...]
    raw_preserving: bool
    option_specs: tuple[ImageExportOptionSpec, ...] = ()
    preview_round_trip: bool = False
    native_metadata: bool = False
    writer_backend: Literal["pillow", "tifffile"] | None = None


IMAGE_VALID_DTYPES = tuple(np.dtype(dtype).name for dtype in ImageObj.VALID_DTYPES)

BMP_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "BMP",
    "bmp",
    ("bmp",),
    ("uint8",),
    False,
    preview_round_trip=True,
    writer_backend="pillow",
)
PNG_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "PNG",
    "png",
    ("png",),
    ("uint8",),
    False,
    (
        ImageExportOptionSpec("compress_level", "int", 6, minimum=0, maximum=9),
        ImageExportOptionSpec("optimize", "bool", False),
    ),
    preview_round_trip=True,
    writer_backend="pillow",
)
JPEG_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "JPEG",
    "jpg",
    ("jpg", "jpeg"),
    ("uint8",),
    False,
    (
        ImageExportOptionSpec("quality", "int", 75, minimum=1, maximum=100),
        ImageExportOptionSpec(
            "subsampling", "choice", "4:2:0", choices=("4:4:4", "4:2:2", "4:2:0")
        ),
        ImageExportOptionSpec("progressive", "bool", False),
        ImageExportOptionSpec("optimize", "bool", False),
        ImageExportOptionSpec("smooth", "int", 0, minimum=0, maximum=100),
    ),
    preview_round_trip=True,
    writer_backend="pillow",
)
JP2_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "JPEG 2000",
    "jp2",
    ("jp2",),
    ("uint8", "uint16"),
    False,
    (
        ImageExportOptionSpec(
            "quality_mode", "choice", "rates", choices=("rates", "dB")
        ),
        ImageExportOptionSpec(
            "quality_layers",
            "float_list",
            (20.0,),
            minimum=0.0,
            minimum_inclusive=False,
        ),
        ImageExportOptionSpec("irreversible", "bool", False),
        ImageExportOptionSpec(
            "progression",
            "choice",
            "LRCP",
            choices=("LRCP", "RLCP", "RPCL", "PCRL", "CPRL"),
        ),
        ImageExportOptionSpec(
            "num_resolutions", "int", None, minimum=1, allow_none=True
        ),
        ImageExportOptionSpec(
            "tile_size", "int_pair", None, minimum=1, advanced=True, allow_none=True
        ),
        ImageExportOptionSpec("plt", "bool", False, advanced=True),
    ),
    preview_round_trip=True,
    writer_backend="pillow",
)
TIFF_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "TIFF",
    "tif",
    ("tif", "tiff"),
    ("uint8", "uint16", "float32", "float64"),
    False,
    (
        ImageExportOptionSpec(
            "compression",
            "choice",
            "none",
            choices=("none", "lzw", "deflate", "zstd", "jpeg"),
        ),
        ImageExportOptionSpec(
            "compression_level",
            "int",
            None,
            minimum=0,
            maximum=100,
            advanced=True,
            allow_none=True,
        ),
        ImageExportOptionSpec(
            "predictor",
            "choice",
            "none",
            choices=("none", "horizontal", "floatingpoint"),
        ),
        ImageExportOptionSpec(
            "rows_per_strip", "int", None, minimum=1, advanced=True, allow_none=True
        ),
        ImageExportOptionSpec(
            "tile_size", "int_pair", None, minimum=1, advanced=True, allow_none=True
        ),
        ImageExportOptionSpec(
            "resolution",
            "float_pair",
            None,
            minimum=0.0,
            advanced=True,
            allow_none=True,
            minimum_inclusive=False,
        ),
        ImageExportOptionSpec(
            "resolution_unit",
            "choice",
            "none",
            choices=("none", "inch", "centimeter"),
            advanced=True,
        ),
        ImageExportOptionSpec(
            "photometric",
            "choice",
            "minisblack",
            choices=("minisblack", "miniswhite"),
            advanced=True,
        ),
    ),
    preview_round_trip=True,
    writer_backend="tifffile",
)
NPY_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "NumPy", "npy", ("npy",), IMAGE_VALID_DTYPES, True
)
MAT_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "MAT-File",
    "mat",
    ("mat",),
    IMAGE_VALID_DTYPES,
    True,
    (ImageExportOptionSpec("do_compression", "bool", False),),
)
TEXT_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "Text",
    "txt",
    ("txt", "asc"),
    IMAGE_VALID_DTYPES,
    True,
    (
        ImageExportOptionSpec(
            "delimiter",
            "choice",
            "whitespace",
            choices=("whitespace", "tab", "comma", "semicolon"),
        ),
        ImageExportOptionSpec("precision", "int", 18, minimum=1, maximum=18),
    ),
)
CSV_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "CSV",
    "csv",
    ("csv",),
    IMAGE_VALID_DTYPES,
    True,
    (
        ImageExportOptionSpec(
            "delimiter", "choice", "comma", choices=("comma", "semicolon", "tab")
        ),
        ImageExportOptionSpec("precision", "int", 18, minimum=1, maximum=18),
    ),
)
H5IMA_EXPORT_CAPABILITIES = ImageExportCapabilities(
    "HDF5 image",
    "h5ima",
    ("h5ima",),
    IMAGE_VALID_DTYPES,
    True,
    native_metadata=True,
)

IMAGE_EXPORT_CAPABILITIES = MappingProxyType(
    {
        extension: capabilities
        for capabilities in (
            BMP_EXPORT_CAPABILITIES,
            PNG_EXPORT_CAPABILITIES,
            JPEG_EXPORT_CAPABILITIES,
            JP2_EXPORT_CAPABILITIES,
            TIFF_EXPORT_CAPABILITIES,
            NPY_EXPORT_CAPABILITIES,
            MAT_EXPORT_CAPABILITIES,
            TEXT_EXPORT_CAPABILITIES,
            CSV_EXPORT_CAPABILITIES,
            H5IMA_EXPORT_CAPABILITIES,
        )
        for extension in capabilities.extension_aliases
    }
)


class ImageExportParam(gds.DataSet, title=_("Image export")):
    """Parameters controlling format-aware image export."""

    normalization_prop = gds.GetAttrProp("normalization")
    normalization = gds.ChoiceItem(
        _("Normalization"),
        [
            ("none", _("None")),
            ("minmax", _("Min-max")),
            ("percentile", _("Percentile")),
            ("manual", _("Manual")),
        ],
        default="none",
    ).set_prop("display", store=normalization_prop)
    manual_min = gds.FloatItem(_("Manual minimum"), default=0.0).set_prop(
        "display",
        active=gds.FuncProp(normalization_prop, lambda value: value == "manual"),
    )
    manual_max = gds.FloatItem(_("Manual maximum"), default=1.0).set_prop(
        "display",
        active=gds.FuncProp(normalization_prop, lambda value: value == "manual"),
    )
    low_percentile = gds.FloatItem(
        _("Low percentile"), default=1.0, min=0.0, max=100.0
    ).set_prop(
        "display",
        active=gds.FuncProp(normalization_prop, lambda value: value == "percentile"),
    )
    high_percentile = gds.FloatItem(
        _("High percentile"), default=99.0, min=0.0, max=100.0
    ).set_prop(
        "display",
        active=gds.FuncProp(normalization_prop, lambda value: value == "percentile"),
    )
    behavior = gds.ChoiceItem(
        _("Out-of-range behavior"),
        [("clip", _("Clip")), ("rescale", _("Rescale"))],
        default="rescale",
    )
    target_dtype = gds.ChoiceItem(
        _("Target data type"),
        [
            ("auto", _("Automatic")),
            ("uint8", "uint8"),
            ("uint16", "uint16"),
            ("int16", "int16"),
            ("int32", "int32"),
            ("float32", "float32"),
            ("float64", "float64"),
            ("complex128", "complex128"),
        ],
        default="auto",
    )
    nonfinite_policy_prop = gds.GetAttrProp("nonfinite_policy")
    nonfinite_policy = gds.ChoiceItem(
        _("Non-finite values"),
        [
            ("error", _("Raise an error")),
            ("clip", _("Clip")),
            ("replace", _("Replace")),
        ],
        default="error",
    ).set_prop("display", store=nonfinite_policy_prop)
    replacement_value = gds.FloatItem(_("Replacement value"), default=0.0).set_prop(
        "display",
        active=gds.FuncProp(nonfinite_policy_prop, lambda value: value == "replace"),
    )
    gamma = gds.FloatItem(
        _("Gamma"), default=None, min=0.01, check=False, allow_none=True
    )
    invert = gds.BoolItem(_("Invert"), default=False)
    format_options = gds.DictItem(_("Format options"), default={}).set_prop(
        "display", hide=True
    )


def get_export_extension(filename_or_extension: str) -> str:
    """Return a normalized supported image export extension.

    Args:
        filename_or_extension: File name or extension, with or without a leading dot.

    Returns:
        Normalized extension without a leading dot.

    Raises:
        ValueError: If the extension is not supported by the export backend.
    """
    suffix = osp.splitext(filename_or_extension)[1]
    extension = (suffix or filename_or_extension).lower().lstrip(".")
    if extension not in IMAGE_EXPORT_CAPABILITIES:
        raise ValueError(f"Unsupported image export format: {filename_or_extension!r}")
    return extension


def get_image_export_capabilities(
    filename_or_extension: str,
) -> ImageExportCapabilities:
    """Return immutable capabilities for an image export format.

    Args:
        filename_or_extension: File name or extension, with or without a leading dot.

    Returns:
        Capabilities shared by all aliases of the resolved format.
    """
    return IMAGE_EXPORT_CAPABILITIES[get_export_extension(filename_or_extension)]


def get_supported_export_dtypes(filename_or_extension: str) -> tuple[str, ...]:
    """Return target dtype choices supported by an image export format.

    Args:
        filename_or_extension: File name or extension, with or without a leading dot.

    Returns:
        Supported dtype names, including ``"auto"``.
    """
    capabilities = get_image_export_capabilities(filename_or_extension)
    return ("auto", *capabilities.supported_dtypes)


def validate_numeric_option(spec: ImageExportOptionSpec, value: int | float) -> None:
    """Validate one numeric value against an option specification."""
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"Image export option {spec.key!r} must be finite")
    if spec.minimum is not None:
        below_minimum = (
            value < spec.minimum if spec.minimum_inclusive else value <= spec.minimum
        )
        if below_minimum:
            operator = ">=" if spec.minimum_inclusive else ">"
            raise ValueError(
                f"Image export option {spec.key!r} must be {operator} {spec.minimum}"
            )
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f"Image export option {spec.key!r} must be <= {spec.maximum}")


def validate_image_export_option_value(
    spec: ImageExportOptionSpec, value: object
) -> object:
    """Return a validated defensive value for one writer option."""
    result = value
    if value is None:
        if not spec.allow_none:
            raise TypeError(f"Image export option {spec.key!r} may not be None")
    elif spec.value_kind == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"Image export option {spec.key!r} must be a bool")
    elif spec.value_kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"Image export option {spec.key!r} must be an int")
        result = int(value)
        validate_numeric_option(spec, result)
    elif spec.value_kind == "float":
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Image export option {spec.key!r} must be a float")
        result = float(value)
        validate_numeric_option(spec, result)
    elif spec.value_kind == "choice":
        if value not in spec.choices:
            choices = ", ".join(repr(choice) for choice in spec.choices)
            raise ValueError(
                f"Image export option {spec.key!r} must be one of: {choices}"
            )
    elif spec.value_kind in ("int_pair", "float_pair"):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(f"Image export option {spec.key!r} must be a pair")
        pair = []
        for item in value:
            if spec.value_kind == "int_pair":
                if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                    raise TypeError(
                        f"Image export option {spec.key!r} must be an integer pair"
                    )
                converted = int(item)
            else:
                if isinstance(item, bool) or not isinstance(item, Real):
                    raise TypeError(
                        f"Image export option {spec.key!r} must be a numeric pair"
                    )
                converted = float(item)
            validate_numeric_option(spec, converted)
            pair.append(converted)
        result = tuple(pair)
    elif spec.value_kind == "float_list":
        if not isinstance(value, (list, tuple)) or not value:
            raise TypeError(
                f"Image export option {spec.key!r} must be a non-empty numeric list"
            )
        result = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, Real):
                raise TypeError(
                    f"Image export option {spec.key!r} must be a numeric list"
                )
            converted = float(item)
            validate_numeric_option(spec, converted)
            result.append(converted)
    elif spec.value_kind == "string":
        if not isinstance(value, str):
            raise TypeError(f"Image export option {spec.key!r} must be a string")
    else:
        raise AssertionError(f"Unknown image export option kind: {spec.value_kind!r}")
    return result


def validate_image_export_options(
    filename_or_extension: str, options: Mapping[str, object]
) -> dict[str, object]:
    """Return a validated defensive copy of format-specific writer options.

    Args:
        filename_or_extension: File name or extension, with or without a leading dot.
        options: Serializable writer option mapping.

    Returns:
        Validated option dictionary with copied pair and list values.

    Raises:
        TypeError: If the mapping or an option value has an invalid type.
        ValueError: If a key, choice, range or value shape is invalid.
    """
    if not isinstance(options, Mapping):
        raise TypeError("Image export options must be a mapping")
    capabilities = get_image_export_capabilities(filename_or_extension)
    specs = {spec.key: spec for spec in capabilities.option_specs}
    unknown = set(options) - set(specs)
    if unknown:
        names = ", ".join(sorted(repr(key) for key in unknown))
        raise ValueError(
            f"Unsupported {capabilities.format_name} export options: {names}"
        )
    return {
        key: validate_image_export_option_value(specs[key], value)
        for key, value in options.items()
    }


def validate_image_export_configuration(
    filename_or_extension: str,
    dtype: np.dtype | str | type[np.generic] | None,
    options: Mapping[str, object],
    shape: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Return complete validated options for prepared image data.

    Args:
        filename_or_extension: File name or extension, with or without a leading dot.
        dtype: Prepared image dtype, or None when dtype-specific checks are unavailable.
        options: Explicit serializable writer options.
        shape: Prepared image shape, or None when geometry checks are unavailable.

    Returns:
        Validated option dictionary including defensive copies of non-None defaults.

    Raises:
        ValueError: If options are incompatible with the prepared image data.
    """
    capabilities = get_image_export_capabilities(filename_or_extension)
    validated_options = validate_image_export_options(filename_or_extension, options)
    values = {
        spec.key: validate_image_export_option_value(spec, spec.default)
        for spec in capabilities.option_specs
    }
    values.update(validated_options)
    values = {key: value for key, value in values.items() if value is not None}
    if capabilities.canonical_extension == "jp2":
        num_resolutions = values.get("num_resolutions")
        if num_resolutions is not None:
            if shape is None or len(shape) != 2:
                raise ValueError(
                    "JP2 num_resolutions validation requires a 2D image shape"
                )
            image_height, image_width = (int(size) for size in shape)
            if image_height < 1 or image_width < 1:
                raise ValueError("JP2 image dimensions must be positive")
            tile_size = values.get("tile_size")
            if tile_size is None:
                effective_height, effective_width = image_height, image_width
            else:
                effective_height = min(image_height, tile_size[0])
                effective_width = min(image_width, tile_size[1])
            maximum_resolutions = min(effective_height, effective_width).bit_length()
            if num_resolutions > maximum_resolutions:
                raise ValueError(
                    f"JP2 num_resolutions must be <= {maximum_resolutions} for "
                    "the effective image/tile dimensions"
                )
        return values
    if capabilities.writer_backend != "tifffile":
        return values

    tile_size = values.get("tile_size")
    rows_per_strip = values.get("rows_per_strip")
    if rows_per_strip is not None and tile_size is not None:
        raise ValueError("TIFF rows_per_strip and tile_size may not be used together")
    if tile_size is not None and any(size % 16 for size in tile_size):
        raise ValueError("TIFF tile dimensions must be multiples of 16")

    compression = str(values["compression"])
    compression_level = values.get("compression_level")
    if compression_level is not None:
        if compression in ("none", "lzw"):
            raise ValueError(
                f"TIFF compression_level is not supported with {compression!r} "
                "compression"
            )
        level_ranges = {"deflate": (0, 9), "zstd": (1, 22), "jpeg": (1, 100)}
        minimum, maximum = level_ranges[compression]
        if not minimum <= compression_level <= maximum:
            raise ValueError(
                f"TIFF {compression} compression_level must be between "
                f"{minimum} and {maximum}"
            )

    if dtype is None:
        return values
    prepared_dtype = np.dtype(dtype)
    predictor = values["predictor"]
    predictor_compressions = ("deflate", "lzw", "zstd")
    if predictor == "horizontal":
        if not np.issubdtype(prepared_dtype, np.integer):
            raise ValueError("TIFF horizontal predictor requires an integer dtype")
        if compression not in predictor_compressions:
            raise ValueError(
                "TIFF horizontal predictor requires deflate, lzw, or zstd compression"
            )
    elif predictor == "floatingpoint":
        if not np.issubdtype(prepared_dtype, np.floating):
            raise ValueError("TIFF floatingpoint predictor requires a floating dtype")
        if compression not in predictor_compressions:
            raise ValueError(
                "TIFF floatingpoint predictor requires deflate, lzw, or zstd "
                "compression"
            )
    if compression == "jpeg" and prepared_dtype != np.dtype(np.uint8):
        raise ValueError("TIFF JPEG compression requires uint8 prepared data")
    return values


def get_image_export_writer_kwargs(
    filename_or_extension: str,
    options: Mapping[str, object],
    dtype: np.dtype | str | type[np.generic] | None = None,
    shape: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Translate validated UI-neutral options to imageio backend kwargs."""
    capabilities = get_image_export_capabilities(filename_or_extension)
    writer_options = validate_image_export_configuration(
        filename_or_extension, dtype, options, shape
    )
    if capabilities.writer_backend != "tifffile":
        return writer_options

    kwargs = dict(writer_options)
    compression = kwargs.pop("compression")
    if compression != "none":
        kwargs["compression"] = compression
    compression_level = kwargs.pop("compression_level", None)
    if compression_level is not None:
        kwargs["compressionargs"] = {"level": compression_level}
    predictor = kwargs.pop("predictor")
    if predictor != "none":
        kwargs["predictor"] = {
            "horizontal": 2,
            "floatingpoint": 3,
        }[predictor]
    rows_per_strip = kwargs.pop("rows_per_strip", None)
    if rows_per_strip is not None:
        kwargs["rowsperstrip"] = rows_per_strip
    tile_size = kwargs.pop("tile_size", None)
    if tile_size is not None:
        kwargs["tile"] = tile_size
    resolution_unit = kwargs.pop("resolution_unit")
    if resolution_unit != "none":
        kwargs["resolutionunit"] = {
            "inch": 2,
            "centimeter": 3,
        }[resolution_unit]
    return kwargs


def write_image_export_data(
    destination: str | BinaryIO,
    data: np.ndarray,
    filename_or_extension: str,
    options: Mapping[str, object],
) -> None:
    """Write classic image data through its explicit imageio backend."""
    capabilities = get_image_export_capabilities(filename_or_extension)
    if capabilities.writer_backend is None:
        raise ValueError(
            f"{capabilities.format_name} does not use an image encoding backend"
        )
    array = np.asarray(data)
    kwargs = get_image_export_writer_kwargs(
        filename_or_extension, options, array.dtype, array.shape
    )
    iio.imwrite(
        destination,
        array,
        extension=f".{capabilities.canonical_extension}",
        plugin=capabilities.writer_backend,
        **kwargs,
    )


def encode_image_export_data(
    data: np.ndarray,
    filename_or_extension: str,
    options: Mapping[str, object],
) -> bytes:
    """Encode classic image data to bytes with the normal writer settings."""
    stream = BytesIO()
    write_image_export_data(stream, data, filename_or_extension, options)
    return stream.getvalue()


def resolve_export_dtype(
    source_dtype: np.dtype, extension: str, target_dtype: str
) -> np.dtype:
    """Resolve and validate the target dtype for an export.

    Args:
        source_dtype: Source array dtype.
        extension: Normalized file extension.
        target_dtype: Requested target dtype name.

    Returns:
        Resolved NumPy dtype.

    Raises:
        ValueError: If the requested dtype is invalid for the format.
    """
    capabilities = get_image_export_capabilities(extension)
    supported = capabilities.supported_dtypes
    if target_dtype == "auto":
        if capabilities.raw_preserving:
            return source_dtype
        source_name = source_dtype.name
        if source_name in supported:
            return source_dtype
        if capabilities.canonical_extension in ("bmp", "jpg", "png"):
            return np.dtype("uint8")
        if capabilities.canonical_extension == "jp2":
            return np.dtype("uint16")
        return np.dtype("float64")
    if target_dtype not in supported:
        choices = ", ".join(supported)
        raise ValueError(
            f"Data type {target_dtype!r} is not supported for .{extension}; "
            f"expected one of: {choices}"
        )
    return np.dtype(target_dtype)


def handle_nonfinite_values(data: np.ndarray, param: ImageExportParam) -> np.ndarray:
    """Return a copy of data with the requested non-finite value policy applied.

    Args:
        data: Source image data.
        param: Image export parameters.

    Returns:
        Copied array with finite values.

    Raises:
        ValueError: If non-finite values cannot be handled as requested.
    """
    if param.nonfinite_policy not in ("error", "clip", "replace"):
        raise ValueError(f"Invalid non-finite policy: {param.nonfinite_policy!r}")
    result = np.array(data, copy=True)
    finite_mask = np.isfinite(result)
    if np.all(finite_mask):
        return result
    if param.nonfinite_policy == "error":
        raise ValueError("Image data contains NaN or infinite values")
    if np.iscomplexobj(result):
        raise ValueError("Non-finite handling is not supported for complex image data")
    result = result.astype(np.float64, copy=False)
    if param.nonfinite_policy == "replace":
        if not np.isfinite(param.replacement_value):
            raise ValueError("The non-finite replacement value must be finite")
        result[~finite_mask] = param.replacement_value
        return result
    if param.nonfinite_policy == "clip":
        finite_values = result[finite_mask]
        if finite_values.size == 0:
            raise ValueError("Non-finite values cannot be clipped without finite data")
        finite_min = float(np.min(finite_values))
        finite_max = float(np.max(finite_values))
        return np.nan_to_num(
            result, nan=finite_min, posinf=finite_max, neginf=finite_min
        )
    raise AssertionError("Unreachable non-finite policy")


def get_normalization_bounds(
    data: np.ndarray, param: ImageExportParam
) -> tuple[float, float] | None:
    """Return validated normalization bounds for image data.

    Args:
        data: Finite image data.
        param: Image export parameters.

    Returns:
        Normalization bounds, or ``None`` when normalization is disabled.

    Raises:
        ValueError: If the mode or resulting bounds are invalid.
    """
    if param.normalization == "none":
        return None
    if data.size == 0:
        raise ValueError("Cannot normalize empty image data")
    if np.iscomplexobj(data):
        raise ValueError("Complex image data cannot be normalized")
    if param.normalization == "minmax":
        lower = float(np.min(data))
        upper = float(np.max(data))
    elif param.normalization == "percentile":
        if not 0.0 <= param.low_percentile < param.high_percentile <= 100.0:
            raise ValueError("Percentile bounds must satisfy 0 <= low < high <= 100")
        lower, upper = np.percentile(
            data, (param.low_percentile, param.high_percentile)
        )
        lower, upper = float(lower), float(upper)
    elif param.normalization == "manual":
        lower, upper = float(param.manual_min), float(param.manual_max)
    else:
        raise ValueError(f"Invalid normalization mode: {param.normalization!r}")
    invalid_order = lower >= upper if param.normalization == "manual" else lower > upper
    if not np.isfinite(lower) or not np.isfinite(upper) or invalid_order:
        raise ValueError(
            f"Normalization bounds must be finite and ordered, got "
            f"{lower!r} and {upper!r}"
        )
    return lower, upper


def convert_export_data(
    data: np.ndarray,
    target_dtype: np.dtype,
    bounds: tuple[float, float] | None,
    behavior: str,
    gamma: float | None = None,
    invert: bool = False,
) -> np.ndarray:
    """Convert prepared data without integer overflow or wraparound.

    Args:
        data: Finite source data.
        target_dtype: Resolved target dtype.
        bounds: Optional normalization bounds.
        behavior: Out-of-range behavior, ``"clip"`` or ``"rescale"``.
        gamma: Optional gamma correction exponent.
        invert: Whether to invert values within the normalization range.

    Returns:
        Converted image data.

    Raises:
        ValueError: If the behavior is invalid or complex data needs conversion.
    """
    if behavior not in ("clip", "rescale"):
        raise ValueError(f"Invalid out-of-range behavior: {behavior!r}")
    if np.iscomplexobj(data) and data.dtype != target_dtype:
        raise ValueError("Complex image data cannot be converted to a real data type")
    if gamma is not None and (not np.isfinite(gamma) or gamma <= 0.0):
        raise ValueError("Gamma must be a finite positive value")
    if bounds is None and (gamma is not None or invert):
        raise ValueError("Gamma and inversion require normalization")

    work = np.array(data, copy=True)
    if bounds is not None:
        lower, upper = bounds
        work = np.clip(work, lower, upper)
        if lower == upper:
            normalized = np.zeros_like(work, dtype=np.float64)
        else:
            amplitude = upper - lower
            if np.isfinite(amplitude) and amplitude != 0.0:
                normalized = (work.astype(np.float64) - lower) / amplitude
            else:
                scale = max(abs(lower), abs(upper))
                scaled_lower = lower / scale
                normalized = (work.astype(np.float64) / scale - scaled_lower) / (
                    upper / scale - scaled_lower
                )
        if gamma is not None:
            normalized = np.power(normalized, gamma)
        if invert:
            normalized = 1.0 - normalized
        if behavior == "rescale":
            work = normalized
            if np.issubdtype(target_dtype, np.integer):
                dtype_info = np.iinfo(target_dtype)
                work = work * (dtype_info.max - dtype_info.min) + dtype_info.min
                work = np.rint(work)
        else:
            work = lower + normalized * (upper - lower)

    if np.issubdtype(target_dtype, np.integer):
        dtype_info = np.iinfo(target_dtype)
        work = np.clip(work, dtype_info.min, dtype_info.max)
    elif np.issubdtype(target_dtype, np.floating):
        if not np.all(np.isfinite(work)):
            raise ValueError("Image export conversion produced non-finite values")
        dtype_info = np.finfo(target_dtype)
        if np.any(work < -dtype_info.max) or np.any(work > dtype_info.max):
            raise ValueError(
                f"Image data contains values outside the finite range of "
                f"{target_dtype.name}"
            )
    result = work.astype(target_dtype)
    if np.issubdtype(target_dtype, np.inexact) and not np.all(np.isfinite(result)):
        raise ValueError("Exported floating image data contains non-finite values")
    return result


def prepare_image_for_export(
    data: np.ndarray, filename_or_extension: str, param: ImageExportParam
) -> np.ndarray:
    """Prepare image data for export to a specific file format.

    The source array is never mutated. Normalization, non-finite handling and safe
    dtype conversion are applied before the existing image writer is called.

    Args:
        data: Source image data.
        filename_or_extension: File name or extension, with or without a leading dot.
        param: Image export parameters.

    Returns:
        Prepared image array compatible with the target format.

    Raises:
        ValueError: If parameters or the format/dtype combination are invalid.
    """
    source = np.asarray(data)
    extension = get_export_extension(filename_or_extension)
    target_dtype = resolve_export_dtype(
        source.dtype, extension, str(param.target_dtype)
    )
    finite_data = handle_nonfinite_values(source, param)
    bounds = get_normalization_bounds(finite_data, param)
    return convert_export_data(
        finite_data,
        target_dtype,
        bounds,
        param.behavior,
        param.gamma,
        param.invert,
    )


def prepare_image_export_preview(
    data: np.ndarray, filename_or_extension: str, param: ImageExportParam
) -> np.ndarray:
    """Prepare and round-trip image data for an exact export preview.

    Args:
        data: Source image data.
        filename_or_extension: File name or extension, with or without a leading dot.
        param: Image export parameters.

    Returns:
        Decoded encoded image data, or a prepared copy for scientific formats.
    """
    prepared = prepare_image_for_export(data, filename_or_extension, param)
    writer_options = validate_image_export_configuration(
        filename_or_extension, prepared.dtype, param.format_options, prepared.shape
    )
    capabilities = get_image_export_capabilities(filename_or_extension)
    if not capabilities.preview_round_trip:
        return np.array(prepared, copy=True)
    encoded = encode_image_export_data(prepared, filename_or_extension, writer_options)
    stream = BytesIO(encoded)
    decoded = iio.imread(
        stream,
        extension=f".{capabilities.canonical_extension}",
        plugin=capabilities.writer_backend,
    )
    return np.array(decoded, copy=True)
