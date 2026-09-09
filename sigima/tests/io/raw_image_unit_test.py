"""Unit tests for parameterized RAW image import."""

from __future__ import annotations

import operator
import os.path as osp

import guidata.dataset as gds
import numpy as np
import pytest
from guidata.config import ValidationMode, temporary_validation_mode

from sigima.io import (
    ImageIORegistry,
    RawImageImportParam,
    SignalIORegistry,
    read_image,
    read_images,
)
from sigima.io.base import IOAction
from sigima.io.image import RawImageImportParam as RawImageImportParamFromImage
from sigima.io.image.formats import HDF5ImageFormat, MatImageFormat, TextImageFormat
from sigima.params import RawImageImportParam as RawImageImportParamFromParams


class RecordingWorker:
    """Record progress and optionally cancel after a completed frame."""

    def __init__(self, *, canceled: bool = False, cancel_after: int | None = None):
        self.canceled = canceled
        self.cancel_after = cancel_after
        self.progress_values: list[float] = []

    def set_progress(self, value: float) -> None:
        """Record progress and trigger cancellation when requested."""
        self.progress_values.append(value)
        if self.cancel_after == len(self.progress_values):
            self.canceled = True

    def was_canceled(self) -> bool:
        """Return whether cancellation was requested."""
        return self.canceled


def create_raw_param(**values) -> RawImageImportParam:
    """Create RAW import parameters with selected values."""
    return RawImageImportParam.create(**values)


def write_raw_file(
    filename: str,
    frames: list[np.ndarray],
    *,
    offset: int = 0,
    gap: int = 0,
    little_endian: bool = True,
) -> None:
    """Write test frames with the requested RAW layout."""
    byte_order = "<" if little_endian else ">"
    with open(filename, "wb") as file:
        file.write(b"H" * offset)
        for index, frame in enumerate(frames):
            dtype = frame.dtype.newbyteorder(byte_order)
            file.write(frame.astype(dtype, copy=False).tobytes())
            if index < len(frames) - 1:
                file.write(b"G" * gap)


@pytest.mark.parametrize(
    "dtype_name", ["uint8", "uint16", "int16", "int32", "float32", "float64"]
)
@pytest.mark.parametrize("little_endian", [True, False])
def test_raw_import_dtype_and_byte_order(
    tmp_path, dtype_name: str, little_endian: bool
) -> None:
    """Decode every supported dtype in both byte orders."""
    expected = np.arange(6, dtype=dtype_name).reshape(2, 3)
    filename = str(tmp_path / "matrix.raw")
    write_raw_file(filename, [expected], little_endian=little_endian)
    param = create_raw_param(
        dtype=dtype_name,
        width=3,
        height=2,
        little_endian=little_endian,
    )

    image = read_image(filename, param=param)

    assert image.data.dtype == np.dtype(dtype_name)
    np.testing.assert_array_equal(image.data, expected)
    assert image.title == osp.basename(filename)
    assert image.metadata["source"] == filename


def test_raw_import_multiple_frames_with_offset_and_gap(tmp_path) -> None:
    """Decode all frames while honoring the first offset and inter-frame gap."""
    frames = [
        np.arange(6, dtype=np.int16).reshape(2, 3),
        np.arange(10, 16, dtype=np.int16).reshape(2, 3),
    ]
    filename = str(tmp_path / "sequence.raw")
    write_raw_file(filename, frames, offset=7, gap=5, little_endian=False)
    param = create_raw_param(
        dtype="int16",
        width=3,
        height=2,
        offset=7,
        count=2,
        gap=5,
        little_endian=False,
    )

    images = read_images(filename, param=param)

    assert [image.title for image in images] == ["sequence.raw 00", "sequence.raw 01"]
    assert [image.metadata["source"] for image in images] == [filename, filename]
    for image, expected in zip(images, frames):
        np.testing.assert_array_equal(image.data, expected)


def test_raw_import_pre_cancellation_reads_no_frame(tmp_path, monkeypatch) -> None:
    """Return no object without reading a frame when already canceled."""
    frames = [np.arange(4, dtype=np.uint8).reshape(2, 2)] * 2
    filename = str(tmp_path / "pre-canceled.raw")
    write_raw_file(filename, frames)
    param = create_raw_param(dtype="uint8", width=2, height=2, count=2)
    worker = RecordingWorker(canceled=True)

    def fail_fromfile(*args, **kwargs):
        raise AssertionError("A RAW frame was read after cancellation")

    monkeypatch.setattr(np, "fromfile", fail_fromfile)
    images = ImageIORegistry.read(filename, worker, param=param)

    assert images == []
    assert not worker.progress_values


def test_raw_import_cancellation_stops_after_one_frame(tmp_path, monkeypatch) -> None:
    """Stop RAW frame reads after cancellation is requested by progress."""
    frames = [np.full((2, 2), index, dtype=np.uint8) for index in range(3)]
    filename = str(tmp_path / "canceled.raw")
    write_raw_file(filename, frames)
    param = create_raw_param(dtype="uint8", width=2, height=2, count=3)
    worker = RecordingWorker(cancel_after=1)
    original_fromfile = np.fromfile
    read_count = 0

    def count_fromfile(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        return original_fromfile(*args, **kwargs)

    monkeypatch.setattr(np, "fromfile", count_fromfile)
    images = ImageIORegistry.read(filename, worker, param=param)

    assert read_count == 1
    assert len(images) == 1
    np.testing.assert_array_equal(images[0].data, frames[0])
    assert worker.progress_values == [pytest.approx(1 / 3)]


def test_raw_import_reports_progress_after_each_frame(tmp_path) -> None:
    """Report cumulative progress after every completed RAW frame."""
    frames = [np.full((2, 2), index, dtype=np.uint8) for index in range(3)]
    filename = str(tmp_path / "progress.raw")
    write_raw_file(filename, frames)
    param = create_raw_param(dtype="uint8", width=2, height=2, count=3)
    worker = RecordingWorker()

    images = ImageIORegistry.read(filename, worker, param=param)

    assert len(images) == 3
    assert worker.progress_values == pytest.approx([1 / 3, 2 / 3, 1.0])


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"width": 0}, "width must be positive"),
        ({"height": -1}, "height must be positive"),
        ({"count": 0}, "image count must be positive"),
        ({"offset": -1}, "offset must be nonnegative"),
        ({"gap": -1}, "gap must be nonnegative"),
    ],
)
def test_raw_import_rejects_invalid_layout(
    tmp_path, values: dict, message: str
) -> None:
    """Reject malformed dimensions, count, offset and gap."""
    filename = tmp_path / "invalid.raw"
    filename.write_bytes(b"")
    with temporary_validation_mode(ValidationMode.DISABLED):
        param = create_raw_param(**values)

    with pytest.raises(ValueError, match=message):
        read_images(str(filename), param=param)


@pytest.mark.parametrize(
    ("payload", "message"),
    [(b"\x00" * 7, "truncated"), (b"\x00" * 9, "trailing")],
)
def test_raw_import_requires_exact_file_size(
    tmp_path, payload: bytes, message: str
) -> None:
    """Reject files smaller or larger than the declared RAW layout."""
    filename = tmp_path / "size.raw"
    filename.write_bytes(payload)
    param = create_raw_param(dtype="uint16", width=2, height=2)

    with pytest.raises(ValueError, match=message):
        read_images(str(filename), param=param)


def test_raw_import_requires_dedicated_parameters(tmp_path) -> None:
    """Require RAW-specific parameters and reject unrelated datasets."""
    filename = tmp_path / "required.raw"
    filename.write_bytes(b"\x00")

    with pytest.raises(TypeError, match="RawImageImportParam"):
        read_images(str(filename))
    with pytest.raises(TypeError, match="RawImageImportParam"):
        ImageIORegistry.read(str(filename), param=gds.DataSet())


def test_raw_import_white_is_zero_is_non_destructive_metadata(tmp_path) -> None:
    """Record white-is-zero intent without changing scientific values."""
    expected = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    filename = str(tmp_path / "white-zero.raw")
    write_raw_file(filename, [expected])
    param = create_raw_param(dtype="uint8", width=2, height=2, white_is_zero=True)

    image = read_image(filename, param=param)

    np.testing.assert_array_equal(image.data, expected)
    assert image.metadata["white_is_zero"] is True


def test_raw_format_is_readable_only() -> None:
    """Expose RAW only in read filters and reject RAW writes."""
    raw_format = ImageIORegistry.get_format("image.raw", IOAction.LOAD)
    assert raw_format.info.readable
    assert not raw_format.info.writeable
    assert "*.raw" in ImageIORegistry.get_read_filters()
    assert "*.raw" not in ImageIORegistry.get_write_filters()
    with pytest.raises(NotImplementedError, match="not supported for save"):
        ImageIORegistry.get_format("image.raw", IOAction.SAVE)


def test_raw_public_exports_and_defaults() -> None:
    """Expose RAW parameters through all public import paths with stable defaults."""
    assert RawImageImportParamFromImage is RawImageImportParam
    assert RawImageImportParamFromParams is RawImageImportParam
    param = create_raw_param()
    assert isinstance(param, RawImageImportParam)
    assert param.dtype == "uint16"
    assert param.width == 1280
    assert param.height == 1024
    assert param.offset == 0
    assert param.count == 1
    assert param.gap == 0
    assert param.little_endian is True
    assert param.white_is_zero is False


def test_existing_image_reader_remains_backward_compatible(tmp_path) -> None:
    """Keep parameter-free calls to existing image readers unchanged."""
    expected = np.arange(6, dtype=np.float64).reshape(2, 3)
    filename = tmp_path / "existing.npy"
    np.save(filename, expected)

    images = read_images(str(filename))

    assert len(images) == 1
    np.testing.assert_array_equal(images[0].data, expected)


def test_existing_image_reader_rejects_parameters(tmp_path) -> None:
    """Reject read parameters for formats that do not support them."""
    filename = tmp_path / "existing.npy"
    np.save(filename, np.zeros((2, 2)))

    with pytest.raises(TypeError, match="does not accept read parameters"):
        read_images(str(filename), param=gds.DataSet())


def test_signal_registry_keeps_parameter_free_read_contract(tmp_path) -> None:
    """Keep signal registry reads independent from image parameters."""
    expected = np.array([[0.0, 1.0], [1.0, 2.0]])
    filename = tmp_path / "signal.csv"
    np.savetxt(filename, expected, delimiter=",")

    signals = SignalIORegistry.read(str(filename))

    assert len(signals) == 1
    np.testing.assert_array_equal(signals[0].xydata, expected.T)
    with pytest.raises(TypeError, match="unexpected keyword argument 'param'"):
        operator.methodcaller("read", str(filename), param=gds.DataSet())(
            SignalIORegistry
        )


@pytest.mark.parametrize(
    "format_class", [HDF5ImageFormat, TextImageFormat, MatImageFormat]
)
def test_specialized_image_reader_rejects_parameters(format_class) -> None:
    """Reject parameters consistently before specialized handlers read a file."""
    with pytest.raises(TypeError, match="does not accept read parameters"):
        format_class().read("unused", param=gds.DataSet())
