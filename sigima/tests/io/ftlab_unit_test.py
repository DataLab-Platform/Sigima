# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for :mod:`sigima.io.ftlab`.

Most uncovered lines are validation/error paths in the binary file readers.
We exercise them by feeding crafted byte streams via ``io.BytesIO``.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from sigima.io.ftlab import (
    FTLabImageFile,
    FTLabSignalFile,
    ImageType,
    check_file_header,
    read_length_prefixed_string,
)

# ===========================================================================
# check_file_header
# ===========================================================================


def _valid_header_bytes() -> bytes:
    """6-byte valid header followed by 250 bytes of padding."""
    return struct.pack("<3h", -31609, 0, 8224) + b"\x00" * 250


def test_check_file_header_too_short() -> None:
    """A header shorter than the 6 magic bytes is rejected with the
    documented ``Header is incomplete`` message."""
    with pytest.raises(ValueError, match="Header is incomplete"):
        check_file_header(io.BytesIO(b"\x00\x00\x00"))


def test_check_file_header_unexpected_values() -> None:
    """Magic-bytes mismatch is rejected with the documented ``Unexpected
    values`` message rather than producing garbage data."""
    bad = struct.pack("<3h", 1, 0, 2) + b"\x00" * 250
    with pytest.raises(ValueError, match="Unexpected values"):
        check_file_header(io.BytesIO(bad))


def test_check_file_header_truncated_padding() -> None:
    """A truncated header padding region is also rejected (full 256-byte
    header is required)."""
    short_padding = struct.pack("<3h", -31609, 0, 8224) + b"\x00" * 100
    with pytest.raises(ValueError, match="Header is incomplete"):
        check_file_header(io.BytesIO(short_padding))


def test_check_file_header_valid() -> None:
    """A correct 256-byte header passes validation silently."""
    check_file_header(io.BytesIO(_valid_header_bytes()))  # No exception.


def test_check_file_header_alternative_i1() -> None:
    # The other valid value for i1.
    """FTLab accepts a second valid value for the first header word
    (``-30844``) for backward compatibility with older files."""
    payload = struct.pack("<3h", -30844, 0, 8224) + b"\x00" * 250
    check_file_header(io.BytesIO(payload))  # No exception.


# ===========================================================================
# read_length_prefixed_string
# ===========================================================================


def test_read_length_prefixed_string_truncated_length() -> None:
    """A truncated 4-byte length prefix is reported as a clear
    ``Failed to read string length`` error."""
    with pytest.raises(ValueError, match="Failed to read string length"):
        read_length_prefixed_string(io.BytesIO(b"\x00\x00"))


def test_read_length_prefixed_string_negative_length() -> None:
    """A negative length prefix is invalid and rejected with a
    ``Negative string length`` error."""
    bad = struct.pack("<i", -1)
    with pytest.raises(ValueError, match="Negative string length"):
        read_length_prefixed_string(io.BytesIO(bad))


def test_read_length_prefixed_string_truncated_data() -> None:
    """A length prefix that promises more bytes than the stream contains
    is reported as ``Failed to read data`` (no silent truncation)."""
    payload = struct.pack("<i", 10) + b"abc"
    with pytest.raises(ValueError, match="Failed to read data"):
        read_length_prefixed_string(io.BytesIO(payload))


def test_read_length_prefixed_string_truncated_padding() -> None:
    # Length 3 → odd → padding byte expected, but missing.
    """Odd-length payloads must be followed by a single 0-byte padding;
    a missing pad byte is reported as ``Failed to read padding byte``."""
    payload = struct.pack("<i", 3) + b"abc"
    with pytest.raises(ValueError, match="Failed to read padding byte"):
        read_length_prefixed_string(io.BytesIO(payload))


def test_read_length_prefixed_string_even_length() -> None:
    """Even-length strings are read back verbatim (no padding involved)."""
    payload = struct.pack("<i", 4) + b"test"
    assert read_length_prefixed_string(io.BytesIO(payload)) == "test"


def test_read_length_prefixed_string_odd_length_with_padding() -> None:
    """Odd-length strings followed by their pad byte are read back
    correctly (regression for the pad-handling logic)."""
    payload = struct.pack("<i", 3) + b"abc" + b"\x00"
    assert read_length_prefixed_string(io.BytesIO(payload)) == "abc"


# ===========================================================================
# FTLabSignalFile / FTLabImageFile __repr__
# ===========================================================================


def test_signal_file_repr_after_init() -> None:
    """``repr(FTLabSignalFile)`` includes both the class name and the
    file path so log lines remain self-describing."""
    sig = FTLabSignalFile.__new__(FTLabSignalFile)
    sig.file_path = "/tmp/test.sig"
    sig.x = None  # type: ignore[assignment]
    sig.y = None  # type: ignore[assignment]
    sig.xu = "s"
    sig.yu = "V"
    text = repr(sig)
    assert "FTLabSignalFile" in text
    assert "/tmp/test.sig" in text


def test_image_file_repr_minimal() -> None:
    """``repr(FTLabImageFile)`` includes both the class name and the
    file path even before ``read()`` has been called."""
    img = FTLabImageFile("/tmp/test.ima")
    text = repr(img)
    assert "FTLabImageFile" in text
    assert "/tmp/test.ima" in text


# ===========================================================================
# FTLabSignalFile._read_* error paths
# ===========================================================================


def test_signal_read_real_with_x_range_truncated(tmp_path) -> None:
    """``_read_real_with_x_range`` reports a clear ``Expected N`` error
    when the data block is shorter than the declared sample count."""
    sig = FTLabSignalFile.__new__(FTLabSignalFile)
    sig.file_path = ""
    p = tmp_path / "d.bin"
    p.write_bytes(b"\x00" * 8)  # Only 1 double, expecting 5.
    with open(p, "rb") as fid:
        with pytest.raises(ValueError, match="Expected 5"):
            sig._read_real_with_x_range(fid, 5, 0.0, 1.0)  # pylint: disable=protected-access


def test_signal_read_real_with_x_truncated(tmp_path) -> None:
    """``_read_real_with_x`` reports a clear ``Expected N`` error when
    the (x, y) interleaved block is truncated."""
    sig = FTLabSignalFile.__new__(FTLabSignalFile)
    p = tmp_path / "d.bin"
    p.write_bytes(b"\x00" * 8)
    with open(p, "rb") as fid:
        with pytest.raises(ValueError, match="Expected 6"):
            sig._read_real_with_x(fid, 3)  # pylint: disable=protected-access


def test_signal_read_complex_with_x_range_truncated(tmp_path) -> None:
    """``_read_complex_with_x_range`` reports ``Expected N`` when the
    (re, im) data block is truncated."""
    sig = FTLabSignalFile.__new__(FTLabSignalFile)
    p = tmp_path / "d.bin"
    p.write_bytes(b"\x00" * 8)
    with open(p, "rb") as fid:
        with pytest.raises(ValueError, match="Expected 6"):
            sig._read_complex_with_x_range(fid, 3, 0.0, 1.0)  # pylint: disable=protected-access


def test_signal_read_complex_with_x_truncated(tmp_path) -> None:
    """``_read_complex_with_x`` reports ``Expected N`` when the
    (x, re, im) interleaved block is truncated."""
    sig = FTLabSignalFile.__new__(FTLabSignalFile)
    p = tmp_path / "d.bin"
    p.write_bytes(b"\x00" * 8)
    with open(p, "rb") as fid:
        with pytest.raises(ValueError, match="Expected 9"):
            sig._read_complex_with_x(fid, 3)  # pylint: disable=protected-access


def test_signal_read_real_with_x_range_ok(tmp_path) -> None:
    """``_read_real_with_x_range`` correctly reconstructs the X axis from
    the (xmin, xmax, n) triple when the data is well-formed."""
    sig = FTLabSignalFile.__new__(FTLabSignalFile)
    p = tmp_path / "d.bin"
    p.write_bytes(struct.pack("<3d", 10.0, 20.0, 30.0))
    with open(p, "rb") as fid:
        sig._read_real_with_x_range(fid, 3, 0.0, 1.0)  # pylint: disable=protected-access
    np.testing.assert_array_equal(sig.x, np.linspace(0.0, 2.0, 3))
    np.testing.assert_array_equal(sig.y, np.array([10.0, 20.0, 30.0]))


def test_signal_read_real_with_x_ok(tmp_path) -> None:
    """``_read_real_with_x`` correctly de-interleaves (x, y) pairs."""
    sig = FTLabSignalFile.__new__(FTLabSignalFile)
    p = tmp_path / "d.bin"
    p.write_bytes(struct.pack("<4d", 1.0, 10.0, 2.0, 20.0))
    with open(p, "rb") as fid:
        sig._read_real_with_x(fid, 2)  # pylint: disable=protected-access
    np.testing.assert_array_equal(sig.x, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(sig.y, np.array([10.0, 20.0]))


def test_signal_read_complex_with_x_range_ok(tmp_path) -> None:
    """``_read_complex_with_x_range`` correctly assembles the complex Y
    array from interleaved real/imag doubles."""
    sig = FTLabSignalFile.__new__(FTLabSignalFile)
    p = tmp_path / "d.bin"
    p.write_bytes(struct.pack("<4d", 1.0, 2.0, 3.0, 4.0))
    with open(p, "rb") as fid:
        sig._read_complex_with_x_range(fid, 2, 0.0, 1.0)  # pylint: disable=protected-access
    np.testing.assert_array_equal(sig.y, np.array([1.0 + 2.0j, 3.0 + 4.0j]))


def test_signal_read_complex_with_x_ok(tmp_path) -> None:
    """``_read_complex_with_x`` correctly de-interleaves (x, re, im)
    triples into the X array and the complex Y array."""
    sig = FTLabSignalFile.__new__(FTLabSignalFile)
    p = tmp_path / "d.bin"
    p.write_bytes(struct.pack("<6d", 0.1, 1.0, 2.0, 0.2, 3.0, 4.0))
    with open(p, "rb") as fid:
        sig._read_complex_with_x(fid, 2)  # pylint: disable=protected-access
    np.testing.assert_array_equal(sig.x, np.array([0.1, 0.2]))
    np.testing.assert_array_equal(sig.y, np.array([1.0 + 2.0j, 3.0 + 4.0j]))


# ===========================================================================
# FTLabSignalFile.read full file
# ===========================================================================


def _length_prefixed(s: bytes) -> bytes:
    """Build a length-prefixed FTLab string with the required even-length
    padding byte when needed."""
    pad = b"\x00" if len(s) % 2 else b""
    return struct.pack("<i", len(s)) + s + pad


def _make_signal_file_bytes(stype: int, n: int, version: float = 5.0) -> bytes:
    """Build a complete signal file bytes payload."""
    payload = _valid_header_bytes()
    payload += _length_prefixed(b"title")
    header = [0.0] * 20
    header[0] = float(stype)
    header[1] = float(n)
    header[2] = 1.0  # step
    header[4] = 0.0  # start
    header[19] = version
    payload += struct.pack(f"<{20}d", *header)
    payload += _length_prefixed(b"s")  # xu
    payload += _length_prefixed(b"V")  # yu
    return payload


def test_signal_read_real_full(tmp_path) -> None:
    """End-to-end read of a synthetic FTLab signal file produces an
    ``(x, y)`` pair of the expected shape and values."""
    payload = _make_signal_file_bytes(stype=1, n=3)
    payload += struct.pack("<3d", 10.0, 20.0, 30.0)
    p = tmp_path / "x.sig"
    p.write_bytes(payload)
    sig = FTLabSignalFile(str(p))
    out = sig.read()
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(sig.y, np.array([10.0, 20.0, 30.0]))


def test_signal_read_unsupported_version(tmp_path) -> None:
    """Files with a header version older than the supported set are
    rejected with ``NotImplementedError`` so corrupt data cannot leak in."""
    payload = _make_signal_file_bytes(stype=1, n=3, version=4.0)
    payload += b"\x00" * 24
    p = tmp_path / "x.sig"
    p.write_bytes(payload)
    with pytest.raises(NotImplementedError, match="not supported"):
        FTLabSignalFile(str(p)).read()


def test_signal_read_unsupported_type(tmp_path) -> None:
    """Unknown signal-type codes are rejected with the documented
    ``Unsupported signal type`` error."""
    payload = _make_signal_file_bytes(stype=999, n=3)
    payload += b"\x00" * 24
    p = tmp_path / "x.sig"
    p.write_bytes(payload)
    with pytest.raises(NotImplementedError, match="Unsupported signal type"):
        FTLabSignalFile(str(p)).read()


def test_signal_read_incomplete_header(tmp_path) -> None:
    # Valid file header + title + truncated 20-double header.
    """A truncated 20-double signal header is reported as ``Incomplete
    signal header`` rather than crashing on a struct.unpack error."""
    payload = _valid_header_bytes()
    payload += _length_prefixed(b"title")
    payload += b"\x00" * 8  # Only 1 double instead of 20.
    p = tmp_path / "x.sig"
    p.write_bytes(payload)
    with pytest.raises(ValueError, match="Incomplete signal header"):
        FTLabSignalFile(str(p)).read()


def test_signal_read_missing_file() -> None:
    """A non-existent path raises a clear ``Error opening file`` instead
    of leaking the underlying ``FileNotFoundError``."""
    with pytest.raises(ValueError, match="Error opening file"):
        FTLabSignalFile("/nonexistent/path/to/file.sig").read()


# ===========================================================================
# FTLabImageFile.read full file
# ===========================================================================


def _make_image_file_bytes(
    image_type: int = 101,
    bits: int = 16,
    nb_cols: int = 2,
    nb_lines: int = 2,
    version: float = 7.0,
) -> bytes:
    """Build a synthetic FTLab image-file payload (header + units), ready
    to be appended with the binary pixel data."""
    payload = _valid_header_bytes()
    payload += _length_prefixed(b"img")
    header = [0.0] * 20
    header[0] = float(image_type)
    header[1] = float(bits)
    header[2] = float(nb_cols)
    header[3] = float(nb_lines)
    header[19] = version
    payload += struct.pack(f"<{20}d", *header)
    # Three length-prefixed strings (units).
    for _ in range(3):
        payload += _length_prefixed(b"u")
    return payload


def test_image_read_real_full(tmp_path) -> None:
    """End-to-end read of a synthetic real (uint16) FTLab image file
    returns an array of the expected shape and dtype."""
    payload = _make_image_file_bytes(image_type=101, bits=16, nb_cols=2, nb_lines=2)
    payload += np.array([1, 2, 3, 4], dtype=np.uint16).tobytes()
    p = tmp_path / "x.ima"
    p.write_bytes(payload)
    out = FTLabImageFile(str(p)).read()
    assert out.shape == (2, 2)
    assert out.dtype == np.uint16


def test_image_read_complex_full(tmp_path) -> None:
    """End-to-end read of a synthetic complex (float32 re+im) FTLab
    image file returns a complex-valued array of the expected shape."""
    payload = _make_image_file_bytes(image_type=102, bits=32, nb_cols=2, nb_lines=2)
    payload += np.array([1, 2, 3, 4], dtype=np.float32).tobytes()
    payload += np.array([5, 6, 7, 8], dtype=np.float32).tobytes()
    p = tmp_path / "x.ima"
    p.write_bytes(payload)
    out = FTLabImageFile(str(p)).read()
    assert out.shape == (2, 2)
    assert np.iscomplexobj(out)


def test_image_read_unsupported_version(tmp_path) -> None:
    """Image files with a header version older than the supported set
    are rejected with ``NotImplementedError``."""
    payload = _make_image_file_bytes(version=6.0)
    p = tmp_path / "x.ima"
    p.write_bytes(payload)
    with pytest.raises(NotImplementedError, match="not supported"):
        FTLabImageFile(str(p)).read()


def test_image_read_unsupported_type(tmp_path) -> None:
    """Unknown image-type codes are rejected with ``NotImplementedError``."""
    payload = _make_image_file_bytes(image_type=999)
    p = tmp_path / "x.ima"
    p.write_bytes(payload)
    with pytest.raises(NotImplementedError, match="not supported"):
        FTLabImageFile(str(p)).read()


def test_image_read_incomplete_header(tmp_path) -> None:
    """A truncated 20-double image header surfaces as ``Incomplete image
    header`` instead of a low-level struct error."""
    payload = _valid_header_bytes()
    payload += _length_prefixed(b"img")
    payload += b"\x00" * 8  # Truncated 20-double header.
    p = tmp_path / "x.ima"
    p.write_bytes(payload)
    with pytest.raises(ValueError, match="Incomplete image header"):
        FTLabImageFile(str(p)).read()


def test_image_read_truncated_data(tmp_path) -> None:
    """A header that promises more pixels than the file contains is
    reported as ``Unexpected end of file``."""
    payload = _make_image_file_bytes(image_type=101, bits=16, nb_cols=2, nb_lines=2)
    # No data following the header.
    p = tmp_path / "x.ima"
    p.write_bytes(payload)
    with pytest.raises(ValueError, match="Unexpected end of file"):
        FTLabImageFile(str(p)).read()


def test_image_read_missing_file() -> None:
    """A non-existent path raises a clear ``Error opening file`` for
    images too (parallel to the signal reader behaviour)."""
    with pytest.raises(ValueError, match="Error opening file"):
        FTLabImageFile("/nonexistent/path.ima").read()


def test_image_type_enum() -> None:
    """The ``ImageType`` enum exposes the documented integer codes
    (``101`` real, ``102`` complex) used in image headers."""
    assert ImageType.REAL.value == 101
    assert ImageType.COMPLEX.value == 102


if __name__ == "__main__":
    pytest.main([__file__])
