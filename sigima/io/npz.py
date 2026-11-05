# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
NPZ object archives
===================

In-memory archives of signals and images, used to transfer objects between
applications (e.g. the Web API of DataLab and its clients such as
DataLab-Kernel). An archive is a ZIP file of NumPy arrays (``.npy``, saved
without pickling) and a JSON document; it is not a file format registered for
reading or writing files.

Format (version 1)
------------------

For a signal (:class:`~sigima.objects.SignalObj`):

- ``x.npy``, ``y.npy``: X and Y data (1D arrays);
- ``dx.npy``, ``dy.npy``: X and Y uncertainties (optional).

For an image (:class:`~sigima.objects.ImageObj`):

- ``data.npy``: image data (2D array, its data type is kept);
- ``xcoords.npy``, ``ycoords.npy``: pixel center coordinates along X and Y (1D
  float arrays, only for non-uniform coordinates).

For both, ``metadata.json`` holds:

- ``type``: ``"signal"`` or ``"image"``;
- ``title``, ``xlabel``, ``ylabel``, ``xunit``, ``yunit`` (and ``zlabel``,
  ``zunit`` for images);
- for images, ``x0``, ``y0``, ``dx``, ``dy`` and ``is_uniform_coords``: False when
  the coordinates are given by ``xcoords.npy`` and ``ycoords.npy``. Archives
  without this key (written by earlier versions) have uniform coordinates;
- ``obj_metadata`` (optional): metadata of the object (e.g. analysis results),
  NumPy arrays and scalars being converted to JSON lists and numbers; the
  ``coords`` entries are restored as NumPy arrays.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from sigima.objects import ImageObj, SignalObj

__all__ = ["deserialize_object_from_npz", "serialize_object_to_npz"]


def _json_value(value: Any) -> Any:  # pylint: disable=too-many-return-statements
    """Return a JSON-serializable form of a value (NumPy arrays and scalars).

    Args:
        value: Value, possibly holding NumPy arrays or scalars, in nested
         dictionaries, lists or tuples.

    Returns:
        JSON-serializable value.
    """
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _restore_metadata(serialized: dict[str, Any]) -> dict[str, Any]:
    """Return object metadata read from JSON, ``coords`` entries as arrays.

    Args:
        serialized: Metadata read from ``metadata.json``.

    Returns:
        Metadata, with the ``coords`` lists of geometry results as NumPy arrays.
    """
    result = {}
    for key, value in serialized.items():
        if isinstance(value, dict):
            result[key] = _restore_metadata(value)
        elif isinstance(value, list) and key == "coords":
            result[key] = np.array(value)
        else:
            result[key] = value
    return result


def _write_array(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    """Write a NumPy array to an archive (without pickling)."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    zf.writestr(name, buffer.getvalue())


def _read_array(zf: zipfile.ZipFile, name: str) -> np.ndarray | None:
    """Read a NumPy array from an archive (without unpickling), or None."""
    if name not in zf.namelist():
        return None
    return np.load(io.BytesIO(zf.read(name)), allow_pickle=False)


def serialize_object_to_npz(
    obj: SignalObj | ImageObj, *, compress: bool = True
) -> bytes:
    """Serialize a signal or an image to an NPZ archive.

    Args:
        obj: Signal or image.
        compress: If True (default), use ZIP deflate compression. False is faster
         at the cost of a larger archive (much faster for data that does not
         compress, e.g. noisy images).

    Returns:
        Bytes of the archive.

    Raises:
        TypeError: If *obj* is neither a signal nor an image.
    """
    if isinstance(obj, SignalObj):
        metadata: dict[str, Any] = {
            "type": "signal",
            "title": obj.title,
            "xlabel": obj.xlabel,
            "ylabel": obj.ylabel,
            "xunit": obj.xunit,
            "yunit": obj.yunit,
        }
        arrays = {"x.npy": obj.x, "y.npy": obj.y}
        if obj.dx is not None:
            arrays["dx.npy"] = obj.dx
        if obj.dy is not None:
            arrays["dy.npy"] = obj.dy
    elif isinstance(obj, ImageObj):
        metadata = {
            "type": "image",
            "title": obj.title,
            "xlabel": obj.xlabel,
            "ylabel": obj.ylabel,
            "zlabel": obj.zlabel,
            "xunit": obj.xunit,
            "yunit": obj.yunit,
            "zunit": obj.zunit,
            # Numeric values may be NumPy scalars (e.g. read from HDF5 files)
            "x0": _json_value(obj.x0),
            "y0": _json_value(obj.y0),
            "dx": _json_value(obj.dx),
            "dy": _json_value(obj.dy),
            "is_uniform_coords": bool(obj.is_uniform_coords),
        }
        arrays = {"data.npy": obj.data}
        if not obj.is_uniform_coords:
            arrays["xcoords.npy"] = obj.xcoords
            arrays["ycoords.npy"] = obj.ycoords
    else:
        raise TypeError(f"Unsupported object type: {type(obj).__name__}")
    if obj.metadata:
        metadata["obj_metadata"] = _json_value(obj.metadata)
    buffer = io.BytesIO()
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", compression) as zf:
        for name, array in arrays.items():
            _write_array(zf, name, array)
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False))
    return buffer.getvalue()


def _set_labels(obj: SignalObj | ImageObj, metadata: dict, names: tuple) -> None:
    """Set the labels and units found in the archive metadata."""
    for name in names:
        if metadata.get(name):
            setattr(obj, name, metadata[name])


def _deserialize_signal(zf: zipfile.ZipFile, metadata: dict) -> SignalObj:
    """Return the signal of an archive."""
    x, y = _read_array(zf, "x.npy"), _read_array(zf, "y.npy")
    if x is None or y is None:
        raise ValueError("Invalid signal NPZ: missing x.npy or y.npy")
    obj = SignalObj()
    obj.set_xydata(x, y, dx=_read_array(zf, "dx.npy"), dy=_read_array(zf, "dy.npy"))
    _set_labels(obj, metadata, ("title", "xlabel", "ylabel", "xunit", "yunit"))
    return obj


def _deserialize_image(zf: zipfile.ZipFile, metadata: dict) -> ImageObj:
    """Return the image of an archive."""
    data = _read_array(zf, "data.npy")
    if data is None:
        raise ValueError("Invalid image NPZ: missing data.npy")
    obj = ImageObj()
    obj.data = data
    _set_labels(
        obj,
        metadata,
        ("title", "xlabel", "ylabel", "zlabel", "xunit", "yunit", "zunit"),
    )
    if metadata.get("is_uniform_coords", True):
        obj.x0 = metadata.get("x0", 0.0)
        obj.y0 = metadata.get("y0", 0.0)
        obj.dx = metadata.get("dx", 1.0)
        obj.dy = metadata.get("dy", 1.0)
        return obj
    xcoords, ycoords = _read_array(zf, "xcoords.npy"), _read_array(zf, "ycoords.npy")
    if xcoords is None or ycoords is None:
        raise ValueError(
            "Invalid image NPZ: missing xcoords.npy or ycoords.npy "
            "for non-uniform coordinates"
        )
    nrows, ncols = data.shape
    if xcoords.shape != (ncols,) or ycoords.shape != (nrows,):
        raise ValueError(
            f"Invalid image NPZ: coordinates of shape {xcoords.shape} (X) and "
            f"{ycoords.shape} (Y) do not match image shape {data.shape}"
        )
    obj.set_coords(xcoords, ycoords)
    return obj


def deserialize_object_from_npz(data: bytes) -> SignalObj | ImageObj:
    """Deserialize a signal or an image from an NPZ archive.

    Args:
        data: Bytes of the archive.

    Returns:
        Signal or image.

    Raises:
        ValueError: If the archive is invalid (missing entries, unknown object
         type, coordinates that do not match the image).
    """
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        if "metadata.json" not in zf.namelist():
            raise ValueError("Invalid NPZ format: missing metadata.json")
        metadata = json.loads(zf.read("metadata.json"))
        obj_type = metadata.get("type")
        if obj_type == "signal":
            obj: SignalObj | ImageObj = _deserialize_signal(zf, metadata)
        elif obj_type == "image":
            obj = _deserialize_image(zf, metadata)
        else:
            raise ValueError(f"Unknown object type in NPZ: {obj_type}")
    if metadata.get("obj_metadata"):
        obj.metadata = _restore_metadata(metadata["obj_metadata"])
    return obj
