# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O registry unit test
"""

from __future__ import annotations

from sigima.io import ImageIORegistry, SignalIORegistry
from sigima.io.base import FormatInfo, IOAction, get_file_extensions
from sigima.io.image.base import ImageFormatBase
from sigima.objects.image import ImageObj
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


def __test_io_registry(registry: SignalIORegistry | ImageIORegistry) -> None:
    """Test I/O registry functionality

    Args:
        registry: I/O registry to test
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
    __test_io_registry(ImageIORegistry)


def test_image_io_registry_preserves_legacy_reader_contract(monkeypatch) -> None:
    """Keep third-party readers using the legacy signature working."""
    monkeypatch.setattr(
        ImageIORegistry,
        "_io_format_instances",
        ImageIORegistry.get_formats().copy(),
    )
    calls = []

    def legacy_read(self, filename: str, worker=None) -> list[ImageObj]:
        """Record a legacy registry read."""
        calls.append((self.info.name, filename, worker))
        return []

    def legacy_write(self, filename: str, obj: ImageObj) -> None:
        """Reject unsupported writes."""
        raise NotImplementedError(f"Writing {obj.title} to {filename} is not supported")

    legacy_format_class = type(
        "LegacyImageFormat",
        (ImageFormatBase,),
        {
            "FORMAT_INFO": FormatInfo(
                name="Legacy test image",
                extensions="*.legacy",
                readable=True,
            ),
            "read": legacy_read,
            "write": legacy_write,
        },
    )

    filename = "image.legacy"
    assert isinstance(
        ImageIORegistry.get_format(filename, IOAction.LOAD), legacy_format_class
    )
    assert ImageIORegistry.read(filename) == []
    assert calls == [("Legacy test image", filename, None)]


if __name__ == "__main__":
    test_signal_io_registry()
    test_image_io_registry()
    test_get_file_extensions()
