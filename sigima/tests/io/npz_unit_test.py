# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests of NPZ object archives (:mod:`sigima.io.npz`)."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

import sigima.objects
from sigima.io import deserialize_object_from_npz, serialize_object_to_npz
from sigima.objects import ImageObj, SignalObj


def _signal() -> SignalObj:
    x = np.linspace(0.0, 10.0, 50)
    obj = sigima.objects.create_signal(
        "Signal", x, np.sin(x), labels=("Time", "Amplitude"), units=("s", "V")
    )
    return obj


def _image(shape: tuple[int, int] = (3, 4), dtype=np.float64) -> ImageObj:
    data = np.arange(shape[0] * shape[1]).reshape(shape).astype(dtype)
    return sigima.objects.create_image(
        "Image", data, labels=("X", "Y", "Z"), units=("mm", "µm", "counts")
    )


def _archive(metadata: dict, arrays: dict[str, np.ndarray] | None = None) -> bytes:
    """Build an archive by hand (e.g. as written by another application)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, array in (arrays or {}).items():
            array_buffer = io.BytesIO()
            np.save(array_buffer, array, allow_pickle=False)
            zf.writestr(name, array_buffer.getvalue())
        zf.writestr("metadata.json", json.dumps(metadata))
    return buffer.getvalue()


def _entries(data: bytes) -> tuple[list[str], dict]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return sorted(zf.namelist()), json.loads(zf.read("metadata.json"))


@pytest.mark.parametrize("compress", [False, True])
def test_signal_round_trip(compress: bool) -> None:
    """Data, labels, units and metadata of a signal are kept."""
    obj = _signal()
    obj.metadata["acquisition"] = {"gain": 2.5, "sample": "S1"}
    data = serialize_object_to_npz(obj, compress=compress)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        expected = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        assert zf.getinfo("y.npy").compress_type == expected
    names, metadata = _entries(data)
    assert names == ["metadata.json", "x.npy", "y.npy"]
    assert metadata["type"] == "signal"
    result = deserialize_object_from_npz(data)
    assert isinstance(result, SignalObj)
    np.testing.assert_array_equal(result.x, obj.x)
    np.testing.assert_array_equal(result.y, obj.y)
    assert result.dx is None and result.dy is None
    assert (result.title, result.xlabel, result.ylabel) == (
        "Signal",
        "Time",
        "Amplitude",
    )
    assert (result.xunit, result.yunit) == ("s", "V")
    assert result.metadata["acquisition"] == {"gain": 2.5, "sample": "S1"}


def test_signal_uncertainties() -> None:
    """X and Y uncertainties are kept."""
    obj = _signal()
    obj.set_xydata(obj.x, obj.y, dx=np.full_like(obj.x, 0.01), dy=np.abs(obj.y) / 10)
    result = deserialize_object_from_npz(serialize_object_to_npz(obj))
    np.testing.assert_array_equal(result.dx, obj.dx)
    np.testing.assert_array_equal(result.dy, obj.dy)


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32, float])
def test_image_round_trip(compress: bool, dtype) -> None:
    """Data (and its type), labels, units and uniform coordinates are kept."""
    obj = _image(dtype=dtype)
    obj.set_uniform_coords(0.5, 2.0, -1.0, 3.0)
    data = serialize_object_to_npz(obj, compress=compress)
    names, metadata = _entries(data)
    assert names == ["data.npy", "metadata.json"]
    assert metadata["is_uniform_coords"] is True
    result = deserialize_object_from_npz(data)
    assert isinstance(result, ImageObj)
    np.testing.assert_array_equal(result.data, obj.data)
    assert result.data.dtype == obj.data.dtype
    assert (result.xlabel, result.ylabel, result.zlabel) == ("X", "Y", "Z")
    assert (result.xunit, result.yunit, result.zunit) == ("mm", "µm", "counts")
    assert result.is_uniform_coords
    assert (result.dx, result.dy, result.x0, result.y0) == (0.5, 2.0, -1.0, 3.0)


def test_image_non_uniform_coordinates() -> None:
    """Non-uniform coordinates are kept in xcoords.npy and ycoords.npy."""
    obj = _image()
    obj.set_coords(np.array([1.0, 2.0, 4.0, 8.0]), np.array([10.0, 11.0, 15.0]))
    data = serialize_object_to_npz(obj)
    names, metadata = _entries(data)
    assert names == ["data.npy", "metadata.json", "xcoords.npy", "ycoords.npy"]
    assert metadata["is_uniform_coords"] is False
    result = deserialize_object_from_npz(data)
    assert not result.is_uniform_coords
    np.testing.assert_array_equal(result.xcoords, obj.xcoords)
    np.testing.assert_array_equal(result.ycoords, obj.ycoords)
    np.testing.assert_array_equal(result.data, obj.data)


def test_image_numpy_scalar_coordinates() -> None:
    """NumPy scalars (e.g. read from HDF5 files) are written as JSON numbers."""
    obj = _image()
    # float32 scalars are not JSON-serializable as such
    obj.x0, obj.y0, obj.dx, obj.dy = (np.float32(v) for v in (1.5, 2.5, 0.5, 0.25))
    data = serialize_object_to_npz(obj)
    _names, metadata = _entries(data)
    assert [metadata[key] for key in ("x0", "y0", "dx", "dy")] == [1.5, 2.5, 0.5, 0.25]
    result = deserialize_object_from_npz(data)
    assert (result.x0, result.y0, result.dx, result.dy) == (1.5, 2.5, 0.5, 0.25)


def test_object_metadata() -> None:
    """NumPy values of the metadata are converted; geometry coords restored."""
    obj = _signal()
    obj.metadata = {
        "Geometry_circle": {"coords": np.array([[0.0, 0.0, 1.0]]), "n": np.int64(1)},
        "values": (np.float32(1.5), np.array([1, 2]), np.bool_(True)),
        "label": "ROI-1",
    }
    data = serialize_object_to_npz(obj)
    _names, metadata = _entries(data)
    assert metadata["obj_metadata"]["values"] == [1.5, [1, 2], True]
    result = deserialize_object_from_npz(data)
    coords = result.metadata["Geometry_circle"]["coords"]
    assert isinstance(coords, np.ndarray)
    np.testing.assert_array_equal(coords, [[0.0, 0.0, 1.0]])
    assert result.metadata["Geometry_circle"]["n"] == 1
    assert result.metadata["values"] == [1.5, [1, 2], True]
    assert result.metadata["label"] == "ROI-1"


def test_archive_without_coordinate_flag() -> None:
    """Archives without is_uniform_coords (earlier versions) are uniform."""
    metadata = {"type": "image", "x0": 1.0, "y0": 2.0, "dx": 0.5, "dy": 0.25}
    result = deserialize_object_from_npz(
        _archive(metadata, {"data.npy": np.zeros((3, 4))})
    )
    assert result.is_uniform_coords
    assert (result.dx, result.dy, result.x0, result.y0) == (0.5, 0.25, 1.0, 2.0)


@pytest.mark.parametrize(
    "metadata, arrays, match",
    [
        ({"type": "weird"}, {}, "Unknown object type"),
        ({"type": "signal"}, {"x.npy": np.arange(3.0)}, "missing x.npy or y.npy"),
        ({"type": "image"}, {}, "missing data.npy"),
        (
            {"type": "image", "is_uniform_coords": False},
            {"data.npy": np.zeros((3, 4)), "ycoords.npy": np.arange(3.0)},
            "missing xcoords.npy",
        ),
        (
            {"type": "image", "is_uniform_coords": False},
            {
                "data.npy": np.zeros((3, 4)),
                "xcoords.npy": np.arange(3.0),
                "ycoords.npy": np.arange(3.0),
            },
            "do not match image shape",
        ),
    ],
)
def test_invalid_archives(metadata: dict, arrays: dict, match: str) -> None:
    """Invalid archives are rejected with the reason."""
    with pytest.raises(ValueError, match=match):
        deserialize_object_from_npz(_archive(metadata, arrays))


def test_missing_metadata_and_unsupported_objects() -> None:
    """An archive needs metadata.json; only signals and images are supported."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("x.npy", b"")
    with pytest.raises(ValueError, match="missing metadata.json"):
        deserialize_object_from_npz(buffer.getvalue())
    with pytest.raises(TypeError, match="Unsupported object type"):
        serialize_object_to_npz(object())  # type: ignore[arg-type]


def test_arrays_are_never_unpickled() -> None:
    """Arrays of Python objects (which would need unpickling) are rejected."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        array_buffer = io.BytesIO()
        np.save(array_buffer, np.array([object()]), allow_pickle=True)
        zf.writestr("data.npy", array_buffer.getvalue())
        zf.writestr("metadata.json", json.dumps({"type": "image"}))
    with pytest.raises(ValueError):
        deserialize_object_from_npz(buffer.getvalue())
