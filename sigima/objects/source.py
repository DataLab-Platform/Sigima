# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Origin of experimental data
===========================

Objects imported from experimental files or captured from instruments record their
origin in the ``experimental_source`` metadata entry. It is a versioned dictionary
made of JSON types only, so that it stays readable without the reader or the
instrument connector that created it (saved workspaces, DataLab-Web, provenance).

It identifies the data origin; it does not make the acquisition reproducible.
The existing ``source`` metadata entry is left untouched.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sigima.objects.image import ImageObj
    from sigima.objects.signal import SignalObj

__all__ = [
    "EXPERIMENTAL_SOURCE_KEY",
    "EXPERIMENTAL_SOURCE_VERSION",
    "get_experimental_source",
    "make_experimental_source",
    "set_experimental_source",
]

#: Metadata key of the experimental data origin
EXPERIMENTAL_SOURCE_KEY = "experimental_source"

#: Version of the ``experimental_source`` dictionary layout
EXPERIMENTAL_SOURCE_VERSION = 1

_OPTIONAL_FIELDS = ("entry", "selection", "timestamps", "conversions", "details")


def _check_text(name: str, value: Any) -> None:
    """Raise if value is not a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Experimental source `{name}` must be a non-empty string")


def _validate(source: dict[str, Any]) -> None:
    """Raise if source is not a valid ``experimental_source`` dictionary."""
    if not isinstance(source, dict):
        raise TypeError("Experimental source must be a dictionary")
    if source.get("version") != EXPERIMENTAL_SOURCE_VERSION:
        raise ValueError(
            f"Unsupported experimental source version: {source.get('version')!r}"
        )
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
    try:
        json.dumps(source, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Experimental source must contain JSON types: {exc}") from exc


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
        TypeError: if a field has an invalid type
        ValueError: if a field is invalid or not JSON-compatible
    """
    source: dict[str, Any] = {
        "version": EXPERIMENTAL_SOURCE_VERSION,
        "kind": kind,
        "location": location,
    }
    values = (entry, selection, timestamps, conversions, details)
    for name, value in zip(_OPTIONAL_FIELDS, values):
        if value is not None:
            source[name] = copy.deepcopy(value)
    _validate(source)
    return source


def set_experimental_source(obj: SignalObj | ImageObj, source: dict[str, Any]) -> None:
    """Store the experimental data origin in object metadata.

    Args:
        obj: signal or image object
        source: dictionary returned by :func:`make_experimental_source`

    Raises:
        TypeError: if a field has an invalid type
        ValueError: if a field is invalid or not JSON-compatible
    """
    _validate(source)
    obj.metadata[EXPERIMENTAL_SOURCE_KEY] = copy.deepcopy(source)


def get_experimental_source(obj: SignalObj | ImageObj) -> dict[str, Any] | None:
    """Return a copy of the experimental data origin stored in object metadata.

    Args:
        obj: signal or image object

    Returns:
        ``experimental_source`` dictionary, or None if the object has none
    """
    source = obj.metadata.get(EXPERIMENTAL_SOURCE_KEY)
    return None if source is None else copy.deepcopy(source)
