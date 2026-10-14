# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for :class:`sigima.io.base.FormatBase` validation paths.

Covers constructor validation (missing format info / name / extensions /
both readable and writeable flags off / invalid extension string / missing
required package), ``get_filter`` for non-readable / non-writeable formats
and the default ``read`` / ``write`` ``NotImplementedError`` raisers.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import inspect

import numpy as np
import pytest

from sigima.io.base import FormatBase, FormatInfo, IOAction
from sigima.io.image.base import MultipleImagesFormatBase, SingleImageFormatBase
from sigima.io.signal.base import SignalFormatBase
from sigima.objects import create_image


def _make_dummy_format_class(format_info):
    """Build a minimal ``FormatBase`` subclass with the given ``FORMAT_INFO``."""

    class DummyFormat(FormatBase):
        """Bare ``FormatBase`` subclass used solely to exercise base-class behaviour."""

        FORMAT_INFO = format_info

    return DummyFormat


def test_format_base_no_format_info_raises() -> None:
    """Instantiating a ``FormatBase`` subclass without a class-level
    ``FORMAT_INFO`` attribute must raise ``ValueError``."""
    cls = _make_dummy_format_class(None)
    with pytest.raises(ValueError, match="Format info not set"):
        cls()


def test_format_base_no_name_raises() -> None:
    """A ``FormatInfo`` with a ``None`` name is invalid: instantiation must
    fail with an explicit ``Format name not set`` message."""
    info = FormatInfo(name=None, extensions="*.dummy", readable=True, writeable=False)
    cls = _make_dummy_format_class(info)
    with pytest.raises(ValueError, match="Format name not set"):
        cls()


def test_format_base_no_extensions_raises() -> None:
    """A ``FormatInfo`` without file extensions is meaningless: the format
    must refuse to instantiate."""
    info = FormatInfo(name="dummy", extensions=None, readable=True, writeable=False)
    cls = _make_dummy_format_class(info)
    with pytest.raises(ValueError, match="extensions not set"):
        cls()


def test_format_base_neither_readable_nor_writeable_raises() -> None:
    """A format that is neither readable nor writeable is useless and must
    be rejected at construction time."""
    info = FormatInfo(
        name="dummy", extensions="*.dummy", readable=False, writeable=False
    )
    cls = _make_dummy_format_class(info)
    with pytest.raises(ValueError, match="not readable nor writeable"):
        cls()


def test_format_base_invalid_extensions_raises() -> None:
    """An empty extensions string is invalid (empty after splitting); the
    constructor must raise ``Invalid format extensions``."""
    info = FormatInfo(name="dummy", extensions="", readable=True, writeable=False)
    cls = _make_dummy_format_class(info)
    with pytest.raises(ValueError, match="Invalid format extensions"):
        cls()


def test_format_base_missing_required_package_raises() -> None:
    """When ``FormatInfo.requires`` lists a package that cannot be
    imported, the constructor surfaces a clear ``ImportError`` instead of
    failing later inside ``read`` / ``write``."""
    info = FormatInfo(
        name="dummy",
        extensions="*.dummy",
        readable=True,
        writeable=False,
        requires=["nonexistent_package_xyz_12345"],
    )
    cls = _make_dummy_format_class(info)
    with pytest.raises(ImportError, match="requires nonexistent_package"):
        cls()


def test_format_base_get_filter_load_unreadable_returns_none() -> None:
    """``get_filter(LOAD)`` returns ``None`` for a non-readable format so
    the file dialog does not list it."""
    info = FormatInfo(
        name="dummy", extensions="*.dummy", readable=False, writeable=True
    )
    cls = _make_dummy_format_class(info)
    fmt = cls()
    assert fmt.get_filter(IOAction.LOAD) is None


def test_format_base_get_filter_save_unwriteable_returns_none() -> None:
    """``get_filter(SAVE)`` returns ``None`` for a non-writeable format so
    the save dialog does not list it."""
    info = FormatInfo(
        name="dummy", extensions="*.dummy", readable=True, writeable=False
    )
    cls = _make_dummy_format_class(info)
    fmt = cls()
    assert fmt.get_filter(IOAction.SAVE) is None


@pytest.mark.parametrize("method", ["read", "write"])
def test_format_base_unsupported_operation(method: str) -> None:
    """Unsupported defaults preserve the NotImplementedError contract."""
    info = FormatInfo(name="dummy", extensions="*.dummy", readable=True)
    fmt = _make_dummy_format_class(info)()
    args = ("dummy.file",) if method == "read" else ("dummy.file", None)
    with pytest.raises(NotImplementedError, match="dummy is not supported"):
        getattr(fmt, method)(*args)


def test_signal_optional_hooks() -> None:
    """Signal formats need not override hooks when supplying their own reader."""

    class CustomReadFormatBase(SignalFormatBase):
        """Read-only format bypassing the default XY reader."""

        FORMAT_INFO = FormatInfo(name="dummy", extensions="*.dummy", readable=True)

        def read(self, filename, worker=None):
            return []

    fmt = CustomReadFormatBase()
    assert not fmt.read("dummy.file")
    for method, args in (
        (fmt.read_xydata, ("dummy.file",)),
        (fmt.write, ("dummy.file", None)),
    ):
        with pytest.raises(NotImplementedError, match="dummy is not supported"):
            method(*args)


def test_image_optional_writer() -> None:
    """Read-only image formats remain concrete without a writer."""

    class ReadOnlyFormatBase(SingleImageFormatBase):
        """Image reader with no write implementation."""

        FORMAT_INFO = FormatInfo(name="dummy", extensions="*.dummy", readable=True)

        @staticmethod
        def read_data(filename):
            return np.zeros((2, 2))

    fmt = ReadOnlyFormatBase()
    assert fmt.read("dummy.file")[0].data.shape == (2, 2)
    with pytest.raises(NotImplementedError, match="dummy.file is not supported"):
        fmt.write("dummy.file", create_image("dummy", np.zeros((2, 2))))
    assert inspect.isabstract(MultipleImagesFormatBase)
    assert "read_data" in MultipleImagesFormatBase.__abstractmethods__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
