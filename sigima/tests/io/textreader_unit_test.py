# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for :py:mod:`sigima.io.common.textreader`.

These tests exercise the encoding-fallback logic of ``count_lines`` and
``read_first_n_lines``, which try ``utf-8``, ``utf-8-sig`` and ``latin-1``
in order before reporting failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sigima.io.common.textreader import count_lines, read_first_n_lines


def _write_bytes(tmp_path: Path, name: str, data: bytes) -> Path:
    """Write ``data`` to ``tmp_path / name`` and return the resulting path."""
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_count_lines_utf8(tmp_path: Path) -> None:
    """``count_lines`` should count lines in a plain UTF-8 file."""
    path = _write_bytes(tmp_path, "utf8.txt", "line1\nline2\nline3\n".encode("utf-8"))
    assert count_lines(path) == 3


def test_count_lines_utf8_sig(tmp_path: Path) -> None:
    """``count_lines`` should transparently handle a UTF-8 BOM."""
    content = "a\nb\n".encode("utf-8-sig")
    path = _write_bytes(tmp_path, "bom.txt", content)
    assert count_lines(path) == 2


def test_count_lines_latin1_fallback(tmp_path: Path) -> None:
    """``count_lines`` falls back to ``latin-1`` when UTF-8 decoding fails."""
    # 0xE9 = 'é' in latin-1; this is an invalid stand-alone byte in UTF-8.
    content = b"caf\xe9\nresum\xe9\n"
    path = _write_bytes(tmp_path, "latin1.txt", content)
    assert count_lines(path) == 2


def test_read_first_n_lines_limits(tmp_path: Path) -> None:
    """``read_first_n_lines`` returns at most ``n`` lines from the file."""
    path = _write_bytes(tmp_path, "many.txt", b"a\nb\nc\nd\ne\n")
    first_three = read_first_n_lines(path, n=3)
    assert first_three == "a\nb\nc\n"


def test_read_first_n_lines_latin1_fallback(tmp_path: Path) -> None:
    """``read_first_n_lines`` falls back to ``latin-1`` like ``count_lines``."""
    content = b"caf\xe9\nresum\xe9\n"
    path = _write_bytes(tmp_path, "latin1.txt", content)
    text = read_first_n_lines(path, n=10)
    assert "café" in text
    assert "resumé" in text


def test_count_lines_missing_file(tmp_path: Path) -> None:
    """Reading a non-existent file must raise ``FileNotFoundError``, not IOError."""
    with pytest.raises(FileNotFoundError):
        count_lines(tmp_path / "missing.txt")


if __name__ == "__main__":
    pytest.main([__file__])
