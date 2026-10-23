# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Origin of experimental data
===========================

Objects imported from experimental files or captured from instruments record their
origin in the ``experimental_source`` metadata entry. It is a versioned dictionary
made of JSON types only, so that it stays readable without the reader or the
instrument connector that created it (saved workspaces, DataLab-Web, provenance).

Values are restricted to the JSON types that HDF5 files restore exactly: objects
with non-empty string keys, arrays, strings, booleans, 64-bit integers and finite
numbers. ``null`` is not allowed. NumPy scalars and tuples are converted to Python
values and lists.

It identifies the data origin; it does not make the acquisition reproducible.
The existing ``source`` metadata entry is left untouched.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sigima.objects.image import ImageObj
    from sigima.objects.signal import SignalObj

__all__ = [
    "EXPERIMENTAL_SOURCE_KEY",
    "EXPERIMENTAL_SOURCE_VERSION",
    "get_experimental_source",
    "make_experimental_source",
    "normalize_experimental_source_metadata",
    "set_experimental_source",
]

#: Metadata key of the experimental data origin
EXPERIMENTAL_SOURCE_KEY = "experimental_source"

#: Version of the ``experimental_source`` dictionary layout
EXPERIMENTAL_SOURCE_VERSION = 1

_OPTIONAL_FIELDS = ("entry", "selection", "timestamps", "conversions", "details")

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


def _check_text(name: str, value: Any) -> None:
    """Raise if value is not a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Experimental source `{name}` must be a non-empty string")


def _check_key(key: Any, where: str) -> None:
    """Raise if key cannot be stored as a dictionary key in HDF5 files."""
    if not isinstance(key, str):
        raise TypeError(
            f"Experimental source `{where or 'source'}` keys must be strings: {key!r}"
        )
    # HDF5 paths use "/" and "."; guidata reserves "__seq", "__dict" and "*__type__"
    if (
        not key
        or "/" in key
        or key in (".", "..")
        or key.startswith("__")
        or key.endswith("__type__")
    ):
        raise ValueError(
            f"Experimental source `{where or 'source'}` has an invalid key: {key!r}"
        )


def _normalize(value: Any, where: str = "", strict: bool = True) -> Any:
    """Return a copy of value made of Python JSON types.

    Args:
        value: value to normalize
        where: path of the value in the source, for error messages
        strict: if True, raise on values that HDF5 files do not restore exactly;
         otherwise, convert NumPy values and bytes, and copy other values as is

    Returns:
        Normalized copy of value

    Raises:
        TypeError: if a value or a key has an unsupported type (strict mode)
        ValueError: if a value or a key is invalid (strict mode)
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if strict:
                _check_key(key, where)
            result[key] = _normalize(item, f"{where}.{key}" if where else key, strict)
        return result
    if isinstance(value, (list, tuple)) or (
        not strict and isinstance(value, np.ndarray)
    ):
        return [
            _normalize(item, f"{where}[{index}]", strict)
            for index, item in enumerate(value)
        ]
    return _normalize_scalar(value, where, strict)


def _normalize_scalar(value: Any, where: str, strict: bool) -> Any:
    """Return a copy of a value that is neither a dictionary nor a sequence, made
    of Python JSON types.

    Args:
        value: value to normalize
        where: path of the value in the source, for error messages
        strict: if True, raise on values that HDF5 files do not restore exactly;
         otherwise, convert NumPy values and bytes, and copy other values as is

    Returns:
        Normalized copy of value

    Raises:
        TypeError: if the value has an unsupported type (strict mode)
        ValueError: if the value is invalid (strict mode)
    """
    if isinstance(value, np.generic):
        value = value.item()
    if not strict and isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if strict and not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(
                f"Experimental source `{where}` must be a 64-bit integer: {value}"
            )
        return int(value)
    if isinstance(value, float):
        if strict and not math.isfinite(value):
            raise ValueError(f"Experimental source `{where}` must be finite: {value}")
        return float(value)
    if isinstance(value, str):
        return str(value)
    if not strict:
        return copy.deepcopy(value)
    if value is None:
        raise ValueError(
            f"Experimental source `{where}` must not be None: "
            "null values are not preserved by HDF5 files"
        )
    raise TypeError(
        f"Experimental source `{where}` must contain JSON types, "
        f"not {type(value).__name__}"
    )


def _validate(source: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of source, raise if it is not a valid
    ``experimental_source`` dictionary."""
    if not isinstance(source, dict):
        raise TypeError("Experimental source must be a dictionary")
    source = _normalize(source)
    version = source.get("version")
    # bool is an int subclass: True must not be taken for version 1
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != EXPERIMENTAL_SOURCE_VERSION
    ):
        raise ValueError(f"Unsupported experimental source version: {version!r}")
    _check_text("kind", source.get("kind"))
    _check_text("location", source.get("location"))
    unknown = set(source) - {"version", "kind", "location", *_OPTIONAL_FIELDS}
    if unknown:
        raise ValueError(f"Unknown experimental source fields: {sorted(unknown)}")
    if "entry" in source:
        _check_text("entry", source["entry"])
    for name in ("selection", "timestamps", "details"):
        if name in source and not isinstance(source[name], dict):
            raise TypeError(f"Experimental source `{name}` must be a dictionary")
    if "conversions" in source and not (
        isinstance(source["conversions"], list)
        and all(isinstance(item, str) for item in source["conversions"])
    ):
        raise TypeError("Experimental source `conversions` must be a list of strings")
    return source


def make_experimental_source(
    kind: str,
    location: str,
    *,
    entry: str | None = None,
    selection: dict[str, Any] | None = None,
    timestamps: dict[str, Any] | None = None,
    conversions: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a validated ``experimental_source`` dictionary.

    Values are restricted to the JSON types restored exactly by HDF5 files:
    dictionaries with non-empty string keys (without ``/``, not starting with
    ``__``), lists, strings, booleans, 64-bit integers and finite numbers. ``None``
    is not allowed. NumPy scalars and tuples are converted to Python values and
    lists.

    Args:
        kind: source kind, e.g. ``"spec"``, ``"nexus"``, ``"hdf5"`` or ``"tango"``
        location: file path or instrument address
        entry: item inside the source, e.g. scan, dataset path or attribute name
        selection: selected part of the entry, e.g. columns, indices or slices
        timestamps: timestamps as ISO 8601 strings, e.g. source and reception times
        conversions: human-readable conversions applied to the source values
        details: other JSON-compatible information (units, calibration, quality...)

    Returns:
        Dictionary made of JSON types only

    Raises:
        TypeError: if a field, a value or a key has an invalid type
        ValueError: if a field, a value or a key is invalid
    """
    source: dict[str, Any] = {
        "version": EXPERIMENTAL_SOURCE_VERSION,
        "kind": kind,
        "location": location,
    }
    values = (entry, selection, timestamps, conversions, details)
    for name, value in zip(_OPTIONAL_FIELDS, values):
        if value is not None:
            source[name] = value
    return _validate(source)


def set_experimental_source(obj: SignalObj | ImageObj, source: dict[str, Any]) -> None:
    """Store the experimental data origin in object metadata.

    Args:
        obj: signal or image object
        source: dictionary returned by :func:`make_experimental_source` or
         :func:`get_experimental_source`

    Raises:
        TypeError: if a field, a value or a key has an invalid type
        ValueError: if a field, a value or a key is invalid
    """
    obj.metadata[EXPERIMENTAL_SOURCE_KEY] = _validate(source)


def get_experimental_source(obj: SignalObj | ImageObj) -> dict[str, Any] | None:
    """Return a copy of the experimental data origin stored in object metadata.

    NumPy values, which HDF5 files return for numbers, are converted to Python
    values.

    Args:
        obj: signal or image object

    Returns:
        ``experimental_source`` dictionary, or None if the object has none
    """
    source = obj.metadata.get(EXPERIMENTAL_SOURCE_KEY)
    return None if source is None else _normalize(source, strict=False)


def normalize_experimental_source_metadata(metadata: dict[str, Any]) -> None:
    """Convert the NumPy values of the ``experimental_source`` metadata entry to
    Python values, in place.

    HDF5 files return numbers as NumPy scalars: objects call this function once
    deserialized, so that their metadata keeps JSON types only.

    Args:
        metadata: object metadata
    """
    source = metadata.get(EXPERIMENTAL_SOURCE_KEY)
    if source is not None:
        metadata[EXPERIMENTAL_SOURCE_KEY] = _normalize(source, strict=False)
