"""Format-aware image export preparation."""

from __future__ import annotations

import os.path as osp

import guidata.dataset as gds
import numpy as np

from sigima.config import _

__all__ = [
    "ImageExportParam",
    "get_supported_export_dtypes",
    "prepare_image_for_export",
]

FORMAT_DTYPES = {
    "bmp": ("uint8",),
    "jpg": ("uint8",),
    "jpeg": ("uint8",),
    "png": ("uint8",),
    "jp2": ("uint8", "uint16"),
    "tif": ("uint8", "uint16", "float32", "float64"),
    "tiff": ("uint8", "uint16", "float32", "float64"),
    "npy": ("uint8", "uint16", "float32", "float64"),
}


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
            ("float32", "float32"),
            ("float64", "float64"),
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
    if extension not in FORMAT_DTYPES:
        raise ValueError(f"Unsupported image export format: {filename_or_extension!r}")
    return extension


def get_supported_export_dtypes(filename_or_extension: str) -> tuple[str, ...]:
    """Return target dtype choices supported by an image export format.

    Args:
        filename_or_extension: File name or extension, with or without a leading dot.

    Returns:
        Supported dtype names, including ``"auto"``.
    """
    extension = get_export_extension(filename_or_extension)
    return ("auto", *FORMAT_DTYPES[extension])


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
    supported = FORMAT_DTYPES[extension]
    if target_dtype == "auto":
        if extension == "npy":
            return source_dtype
        source_name = source_dtype.name
        if source_name in supported:
            return source_dtype
        if extension in ("bmp", "jpg", "jpeg", "png"):
            return np.dtype("uint8")
        if extension == "jp2":
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
) -> np.ndarray:
    """Convert prepared data without integer overflow or wraparound.

    Args:
        data: Finite source data.
        target_dtype: Resolved target dtype.
        bounds: Optional normalization bounds.
        behavior: Out-of-range behavior, ``"clip"`` or ``"rescale"``.

    Returns:
        Converted image data.

    Raises:
        ValueError: If the behavior is invalid or complex data needs conversion.
    """
    if behavior not in ("clip", "rescale"):
        raise ValueError(f"Invalid out-of-range behavior: {behavior!r}")
    if np.iscomplexobj(data) and data.dtype != target_dtype:
        raise ValueError("Complex image data cannot be converted to a real data type")

    work = np.array(data, copy=True)
    if bounds is not None:
        lower, upper = bounds
        work = np.clip(work, lower, upper)
        if behavior == "rescale":
            if lower == upper:
                work = np.zeros_like(work, dtype=np.float64)
            else:
                amplitude = upper - lower
                if np.isfinite(amplitude) and amplitude != 0.0:
                    work = (work.astype(np.float64) - lower) / amplitude
                else:
                    scale = max(abs(lower), abs(upper))
                    scaled_lower = lower / scale
                    work = (work.astype(np.float64) / scale - scaled_lower) / (
                        upper / scale - scaled_lower
                    )
            if np.issubdtype(target_dtype, np.integer):
                dtype_info = np.iinfo(target_dtype)
                work = work * (dtype_info.max - dtype_info.min) + dtype_info.min
                work = np.rint(work)

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
    return convert_export_data(finite_data, target_dtype, bounds, param.behavior)
