# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Exposure and level adjustment module
------------------------------------

This module provides functions for adjusting image exposure, contrast, and intensity
levels.

Features include:

- Dynamic range scaling and adjustment
- Various normalization methods (maximum, amplitude, area, energy, RMS)
- Data type preserving transformations

These tools support image enhancement and preprocessing operations that adjust
the intensity distribution of images while preserving their essential characteristics.
"""

from __future__ import annotations

import numpy as np

from sigima.enums import NormalizationMethod
from sigima.tools.checks import check_2d_array
from sigima.tools.image.preprocessing import scale_data_to_min_max

__all__ = [
    "adjust_brightness_contrast",
    "brightness_contrast_context",
    "brightness_contrast_ranges",
    "flatfield",
    "normalize",
]


def _real_dtype_range(dtype: np.dtype) -> tuple[float, float] | None:
    """Return the finite range of a supported real dtype."""
    if np.issubdtype(dtype, np.integer):
        if dtype.itemsize > 4:
            raise TypeError("Brightness and contrast support at most 32-bit integers")
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(dtype, np.floating):
        return None
    raise TypeError("Brightness and contrast adjustment requires a real image dtype")


def _minimum_width(dtype: np.dtype, lower: float, upper: float) -> float:
    """Return a positive representable width for a range editor."""
    if np.issubdtype(dtype, np.integer):
        return 1.0
    widths = []
    for bound in (lower, upper):
        value = dtype.type(bound)
        direction = dtype.type(0.0 if value else 1.0)
        neighbor = np.nextafter(value, direction)
        width = abs(float(value) - float(neighbor))
        if np.isfinite(width) and width > 0.0:
            widths.append(width)
    if widths:
        return max(widths)
    return float(np.nextafter(dtype.type(0.0), dtype.type(1.0)))


def _non_empty_range(
    lower: float, upper: float, minimum_width: float
) -> tuple[float, float]:
    """Return an ordered range with at least *minimum_width*."""
    if lower < upper:
        return lower, upper
    expanded_lower = lower - minimum_width
    expanded_upper = upper + minimum_width
    if np.isfinite(expanded_lower) and expanded_lower >= lower:
        expanded_lower = float(np.nextafter(lower, -np.inf))
    if np.isfinite(expanded_upper) and expanded_upper <= upper:
        expanded_upper = float(np.nextafter(upper, np.inf))
    if np.isfinite(expanded_lower) and expanded_lower < upper:
        if np.isfinite(expanded_upper) and expanded_upper > lower:
            return expanded_lower, expanded_upper
        return expanded_lower, upper
    if np.isfinite(expanded_upper) and expanded_upper > lower:
        return lower, expanded_upper
    raise ValueError("unable to construct a finite non-empty range")


def _normalize_range(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map finite values from a finite range to stable unit coordinates."""
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, lower, upper)
    width = upper - lower
    if np.isfinite(width):
        return (clipped - lower) / width
    scale = max(abs(lower), abs(upper))
    if scale == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    scaled_lower = lower / scale
    scaled_upper = upper / scale
    normalized = (clipped / scale - scaled_lower) / (scaled_upper - scaled_lower)
    return np.clip(normalized, 0.0, 1.0)


def _denormalize_range(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map unit coordinates to a finite range without overflowing its width."""
    width = upper - lower
    if np.isfinite(width):
        return lower + values * width
    scale = max(abs(lower), abs(upper))
    if scale == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    scaled_lower = lower / scale
    scaled_upper = upper / scale
    result = (scaled_lower + values * (scaled_upper - scaled_lower)) * scale
    result = np.clip(result, lower, upper)
    result[values <= 0.0] = lower
    result[values >= 1.0] = upper
    return result


def brightness_contrast_ranges(
    data: np.ndarray, dtype: np.dtype | type
) -> tuple[tuple[float, float] | None, tuple[float, float]]:
    """Return finite observed and output ranges without building a histogram.

    Supports floating-point and up to 32-bit integer dtypes.

    Raises:
        TypeError: If dtype is complex, non-numeric, or an integer wider than 32 bits.
    """
    dtype = np.dtype(dtype)
    dtype_range = _real_dtype_range(dtype)
    values = np.asarray(data).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, dtype_range or (0.0, 1.0)
    observed = (float(np.min(finite)), float(np.max(finite)))
    return observed, dtype_range or observed


def _histogram_auto_range(
    counts: np.ndarray, bin_edges: np.ndarray, tail_fraction: float = 0.01
) -> tuple[float, float]:
    """Return histogram bounds after removing equal mass from both tails."""
    total = int(counts.sum())
    if total == 0:
        return float(bin_edges[0]), float(bin_edges[-1])
    threshold = total * tail_fraction
    lower_index = int(np.searchsorted(np.cumsum(counts), threshold, side="right"))
    upper_index = (
        len(counts)
        - 1
        - int(np.searchsorted(np.cumsum(counts[::-1]), threshold, side="right"))
    )
    lower_index = min(max(lower_index, 0), len(counts) - 1)
    upper_index = min(max(upper_index, lower_index), len(counts) - 1)
    return float(bin_edges[lower_index]), float(bin_edges[upper_index + 1])


def brightness_contrast_context(
    data: np.ndarray, dtype: np.dtype | type
) -> dict[str, object]:
    """Build the portable histogram/range payload for brightness adjustment.

    Args:
        data: Values selected by the source ROI
        dtype: Source image dtype

    Returns:
        JSON-compatible renderer payload

    Raises:
        TypeError: If dtype is not floating-point or an integer of at most 32 bits.
    """
    dtype = np.dtype(dtype)
    values = np.asarray(data).ravel()
    finite = values[np.isfinite(values)]
    observed, output_range = brightness_contrast_ranges(finite, dtype)
    if observed is None:
        domain = (0.0, 1.0)
        minimum_width = _minimum_width(dtype, *domain)
        return {
            "counts": [0] * 256,
            "bin_edges": np.linspace(*domain, 257).tolist(),
            "domain": list(domain),
            "y_max": 1,
            "minimum_width": minimum_width,
            "reset_range": list(domain),
            "auto_range": list(domain),
            "output_range": list(domain),
            "active": False,
        }

    dtype_range = _real_dtype_range(dtype)
    domain = observed if dtype_range is None else dtype_range
    minimum_width = _minimum_width(dtype, *domain)
    histogram_domain = _non_empty_range(*domain, minimum_width)
    normalized = _normalize_range(finite, *histogram_domain)
    counts, normalized_edges = np.histogram(normalized, bins=256, range=(0.0, 1.0))
    bin_edges = _denormalize_range(normalized_edges, *histogram_domain)
    second_highest = int(np.partition(counts, -2)[-2]) if len(counts) > 1 else 0
    mode = int(counts.max(initial=0))
    if second_highest and mode > 2 * second_highest:
        y_max = int(1.5 * second_highest)
    else:
        y_max = mode
    y_max = max(y_max, 1)

    reset_range = dtype_range if dtype == np.dtype(np.uint8) else observed
    reset_range = _non_empty_range(*reset_range, minimum_width)
    auto_range = _non_empty_range(
        *_histogram_auto_range(counts, bin_edges), minimum_width
    )
    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "domain": list(histogram_domain),
        "y_max": y_max,
        "minimum_width": minimum_width,
        "reset_range": list(reset_range),
        "auto_range": list(auto_range),
        "output_range": list(output_range),
        "active": observed[0] < observed[1],
    }


def adjust_brightness_contrast(
    data: np.ndarray,
    minimum: float,
    maximum: float,
    output_range: tuple[float, float],
) -> np.ndarray:
    """Apply a clipped linear intensity remapping while preserving dtype.

    Args:
        data: Input image data
        minimum: Input value mapped to the output minimum
        maximum: Input value mapped to the output maximum
        output_range: Output minimum and maximum

    Returns:
        Remapped array with the input dtype

    Raises:
        TypeError: If data is not floating-point or an integer of at most 32 bits.
    """
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
        raise ValueError("minimum must be finite and strictly less than maximum")
    dtype = data.dtype
    _real_dtype_range(dtype)
    result = np.array(data, copy=True)
    finite = np.isfinite(data)
    if not np.any(finite):
        return result
    output_minimum, output_maximum = output_range
    if minimum == output_minimum and maximum == output_maximum:
        return result
    normalized = _normalize_range(
        np.asarray(data[finite], dtype=np.float64), minimum, maximum
    )
    mapped = _denormalize_range(normalized, output_minimum, output_maximum)
    if np.issubdtype(dtype, np.integer):
        mapped = np.rint(mapped)
    result[finite] = mapped.astype(dtype)
    return result


@check_2d_array(non_constant=True)
def normalize(
    data: np.ndarray,
    parameter: NormalizationMethod = NormalizationMethod.MAXIMUM,
) -> np.ndarray:
    """Normalize input array to a given parameter.

    Args:
        data: Input data
        parameter: Normalization parameter (default: MAXIMUM)

    Returns:
        Normalized array
    """
    if parameter == NormalizationMethod.MAXIMUM:
        return scale_data_to_min_max(data, np.nanmin(data) / np.nanmax(data), 1.0)
    if parameter == NormalizationMethod.AMPLITUDE:
        return scale_data_to_min_max(data, 0.0, 1.0)
    fdata = np.array(data, dtype=float)
    if parameter == NormalizationMethod.AREA:
        return fdata / np.nansum(fdata)
    if parameter == NormalizationMethod.ENERGY:
        return fdata / np.sqrt(np.nansum(fdata * fdata.conjugate()))
    if parameter == NormalizationMethod.RMS:
        return fdata / np.sqrt(np.nanmean(fdata * fdata.conjugate()))
    raise ValueError(f"Unsupported parameter {parameter}")


@check_2d_array
def flatfield(
    rawdata: np.ndarray, flatdata: np.ndarray, threshold: float | None = None
) -> np.ndarray:
    """Compute flat-field correction

    Args:
        rawdata: Raw data
        flatdata: Flat-field data
        threshold: Threshold for flat-field correction (default: None)

    Returns:
        Flat-field corrected data
    """
    dtemp = np.array(rawdata, dtype=float, copy=True) * np.nanmean(flatdata)
    dunif = np.array(flatdata, dtype=float, copy=True)
    dunif[dunif == 0] = 1.0
    dcorr_all = np.array(dtemp / dunif, dtype=rawdata.dtype)
    dcorr = np.array(rawdata, copy=True)
    dcorr[rawdata > threshold] = dcorr_all[rawdata > threshold]
    return dcorr
