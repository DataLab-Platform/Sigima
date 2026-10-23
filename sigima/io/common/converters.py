# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O conversion functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import skimage


def dtypes_to_sorted_short_codes(
    dtypes: Sequence[Any], kind_filter: str | None = None
) -> list[str]:
    """Return sorted short dtype codes for numeric dtypes.

    Convert each input to a numpy dtype and ignore non-numeric types.
    Order:
      - Integer types first, unsigned (and boolean) before signed,
        sorted by itemsize ascending.
      - floats numeric types, sorted by itemsize ascending.
      - complex numeric types, sorted by itemsize ascending.

    Short codes use numpy kind letter plus itemsize in bytes, e.g. "u1", "i2",
    "f8".

    Args:
        dtypes: Sequence of objects acceptable by numpy.dtype (dtype, str, etc.)
        kind_filter: String of dtype kind letters to keep, e.g. "iu" for
         unsigned/signed integers. If empty or None, keep all numeric types

    Returns:
        List of unique short dtype codes in the requested order.
    """
    dtypes = [np.dtype(d).str[1:] for d in dtypes]
    ordered: list[np.dtype] = []

    if kind_filter is None:
        kind_filter = "iubfc"  # all numeric types
    assert kind_filter != "", "kind_filter cannot be empty string"

    # Standard dtype codes in desired order
    bool_codes = ("b1",)
    int_codes = ("u1", "i1", "u2", "i2", "u4", "i4", "u8", "i8")
    float_codes = ("f2", "f4", "f8")
    complex_codes = ("c8", "c16")

    ordered = [
        code
        for code in bool_codes + int_codes + float_codes + complex_codes
        if code in dtypes and code[0] in kind_filter
    ]
    return ordered


def _convert_bool_array(array: np.ndarray) -> np.ndarray:
    """Convert boolean array to uint8."""
    return skimage.util.img_as_ubyte(array)


def _convert_int_array(
    array: np.ndarray, supported_data_types: tuple[np.dtype]
) -> np.ndarray:
    """Convert an integer array to a standard type.

    Select the smallest supported integer dtype that can represent all values in the
    array. If no suitable integer dtype is found, convert the array to the smallest
    supported float type, at least as large as the input, that represents all values
    exactly. A warning is emitted when no supported float type is exact.

    Args:
        array: Input numpy array of integer type.
        supported_data_types: Tuple of supported numpy dtypes for destination object.

    Returns:
        Converted numpy array with the selected dtype.

    Raises:
        ValueError: If no supported dtype can represent the data.
    """
    ordered_codes = dtypes_to_sorted_short_codes(supported_data_types, kind_filter="iu")

    amin = np.min(array) if array.size > 0 else 0
    amax = np.max(array) if array.size > 0 else 0
    for code in ordered_codes:
        info = np.iinfo(code)
        if amin >= info.min and amax <= info.max:
            new_type = np.dtype(code).newbyteorder("=")
            break
    else:
        new_type = _exact_float_dtype(array, amin, amax, supported_data_types)

    return array.astype(new_type, copy=False)


def _exact_float_dtype(
    array: np.ndarray, amin: Any, amax: Any, supported_data_types: tuple[np.dtype]
) -> np.dtype:
    """Return the float dtype used to store an integer array.

    Args:
        array: Input numpy array of integer type.
        amin: Minimum value of the array.
        amax: Maximum value of the array.
        supported_data_types: Tuple of supported numpy dtypes for destination object.

    Returns:
        Smallest supported float dtype, at least as large as the input, whose
        significand represents all values exactly (the largest one otherwise).

    Raises:
        ValueError: If no supported float dtype exists.
    """
    float_codes = dtypes_to_sorted_short_codes(supported_data_types, kind_filter="f")
    if not float_codes:
        raise ValueError("Unsupported data type")
    magnitude = max(abs(int(amin)), abs(int(amax)))
    for code in float_codes:
        dtype = np.dtype(code)
        exact_limit = 2 ** (np.finfo(dtype).nmant + 1)
        if dtype.itemsize >= array.dtype.itemsize and magnitude <= exact_limit:
            return dtype.newbyteorder("=")
    dtype = np.dtype(float_codes[-1])
    warnings.warn(
        f"Integer values up to {magnitude} ({array.dtype}) cannot be represented "
        f"exactly as {dtype}: they are rounded to the nearest {dtype} value.",
        UserWarning,
        stacklevel=4,
    )
    return dtype.newbyteorder("=")


def _convert_float_array(
    array: np.ndarray, supported_data_types: tuple[np.dtype]
) -> np.ndarray:
    """Convert float/complex array to smallest allowed type at least large as current.

    Choose the smallest supported dtype of the same kind ("f" for floats,
    "c" for complex) whose itemsize is greater than or equal to the array's
    itemsize. If no such type exists, fall back to the largest supported
    dtype for that kind.

    Args:
        array: Array to convert.
        supported_data_types: Sequence of allowed dtypes for the destination
            object type.

    Returns:
        Converted array with the selected dtype. If no supported dtype of the
        same kind exists, return the original array.
    """
    kind = array.dtype.kind
    if kind in ["i", "u", "b"]:
        kind = "f"  # convert integers to floats

    itemsize = array.dtype.itemsize

    ordered_codes = dtypes_to_sorted_short_codes(supported_data_types, kind_filter=kind)

    # Filter out any codes that don't match the requested kind (defensive).
    valid_codes: list[str] = []
    for code in ordered_codes:
        try:
            dt = np.dtype(code)
        except TypeError:
            continue
        if dt.kind == kind:
            valid_codes.append(code)

    if not valid_codes:
        # No supported dtype for this kind, return original array.
        raise ValueError("Unsupported data type")

    # Find smallest supported type with itemsize >= current itemsize.
    selected_code: str | None = None
    for code in valid_codes:
        dt = np.dtype(code)
        if dt.itemsize >= itemsize:
            selected_code = code
            break
    else:
        # Fallback to the largest supported type for this kind.
        selected_code = valid_codes[-1]

    new_type = np.dtype(selected_code).newbyteorder("=")
    return array.astype(new_type, copy=False)


def convert_array_to_valid_dtype(
    array: np.ndarray, valid_dtypes: tuple[np.dtype, ...]
) -> np.ndarray:
    """Convert array to the most appropriate valid dtype.

    Converts arrays to one of the valid dtypes, choosing the most appropriate type
    based on the input array's characteristics.

    Args:
        array: array to convert
        valid_dtypes: tuple of valid dtypes

    Returns:
        Converted array with the most appropriate valid dtype.

    Raises:
        TypeError: if input is not a numpy ndarray
        ValueError: if array dtype cannot be converted to any valid type
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")

    if array.dtype in valid_dtypes:
        return array

    kind: str = array.dtype.kind
    if kind in ["f", "c"]:
        return _convert_float_array(array, valid_dtypes)
    if kind == "b":
        return _convert_bool_array(array)
    if kind in ["i", "u"]:
        return _convert_int_array(array, valid_dtypes)

    raise ValueError("Unsupported data type")
