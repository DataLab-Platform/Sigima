# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O registry unit test
"""

from __future__ import annotations

from sigima.config import _
from sigima.io import ImageIORegistry, SignalIORegistry
from sigima.io.base import IOAction, get_file_extensions
from sigima.tests.env import execenv


def test_get_file_extensions() -> None:
    """Test function `get_file_extensions` for I/O registries"""
    extensions = "*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.jp2"
    assert get_file_extensions(extensions) == [
        "bmp",
        "jp2",
        "jpeg",
        "jpg",
        "png",
        "tif",
        "tiff",
    ], "get_file_extensions did not return expected list of extensions"


def __test_io_registry(
    registry: SignalIORegistry | ImageIORegistry,
    default_save_filters: bool = True,
) -> None:
    """Test I/O registry functionality

    Args:
        registry: I/O registry to test
        default_save_filters: Check the default save filter count
    """
    execenv.print("*" * 80)
    execenv.print(f"Testing I/O registry: {registry.__name__}")
    execenv.print("*" * 80)
    formats = registry.get_formats()
    execenv.print(f"Supported formats: {len(formats)}")
    execenv.print(registry.get_format_info(mode="text"))
    load_filters = registry.get_filters(IOAction.LOAD)
    assert (
        len(load_filters.splitlines())
        == len([fmt for fmt in formats if fmt.info.readable]) + 1
    ), "Number of load filters does not match number of formats"
    save_filters = registry.get_filters(IOAction.SAVE)
    if default_save_filters:
        assert (
            len(save_filters.splitlines())
            == len([fmt for fmt in formats if fmt.info.writeable]) + 1
        ), "Number of save filters does not match number of formats"
    execenv.print(f"Readable formats: {load_filters}")
    assert load_filters == registry.get_read_filters()
    execenv.print(f"Writable formats: {save_filters}")
    assert save_filters == registry.get_write_filters()


def test_signal_io_registry() -> None:
    """Test Signal I/O registry functionality"""
    __test_io_registry(SignalIORegistry)


def test_image_io_registry() -> None:
    """Test Image I/O registry functionality"""
    __test_io_registry(ImageIORegistry, default_save_filters=False)
    load_filters = ImageIORegistry.get_read_filters().splitlines()
    grouped_classic_filter = (
        "BMP, JPEG, PNG, TIFF, JPEG2000 (*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.jp2)"
    )
    separate_classic_filters = [
        "BMP (*.bmp)",
        "JPEG (*.jpg *.jpeg)",
        "PNG (*.png)",
        "TIFF (*.tif *.tiff)",
        "JPEG 2000 (*.jp2)",
    ]
    readable_formats = [
        fmt for fmt in ImageIORegistry.get_formats() if fmt.info.readable
    ]
    extensions = [extension for fmt in readable_formats for extension in fmt.extlist]
    expected_load_filters = [
        f"{_('All supported files')} ({'*.' + ' *.'.join(extensions)})",
        *(fmt.get_filter(IOAction.LOAD) for fmt in readable_formats),
    ]
    assert load_filters == expected_load_filters
    assert grouped_classic_filter in expected_load_filters

    save_filters = ImageIORegistry.get_write_filters().splitlines()
    assert not any(
        _("All supported files") in file_filter for file_filter in save_filters
    )
    expected_save_filters = []
    for fmt in ImageIORegistry.get_formats():
        file_filter = fmt.get_filter(IOAction.SAVE)
        if file_filter == grouped_classic_filter:
            expected_save_filters.extend(separate_classic_filters)
        elif file_filter is not None:
            expected_save_filters.append(file_filter)
    assert save_filters == expected_save_filters


if __name__ == "__main__":
    test_signal_io_registry()
    test_image_io_registry()
    test_get_file_extensions()
