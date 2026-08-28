# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Convenience I/O functions

This module provides convenient wrapper functions for input/output operations with
signals and images. These functions offer a simplified interface to the underlying
I/O system, making common tasks easier to perform.
"""

from __future__ import annotations

import copy
import os.path as osp
from typing import Generator, Sequence

import guidata.dataset as gds

from sigima.config import _
from sigima.io.common.basename import format_basenames
from sigima.io.image.base import ImageIORegistry
from sigima.io.image.export import ImageExportParam, prepare_image_for_export
from sigima.io.signal.base import SignalIORegistry
from sigima.objects import ImageObj, SignalObj, TypeObj


class SaveToDirectoryParam(gds.DataSet):
    """Save to directory parameters."""

    def build_filenames(self, objs: list[TypeObj]) -> list[str]:
        """Build filenames according to current parameters."""
        filenames = format_basenames(objs, self.basename + self.extension)
        used: set[str] = set()  # Ensure all filenames are unique.
        for i, filename in enumerate(filenames):
            root, ext = osp.splitext(filename)
            filepath = osp.join(self.directory, filename)
            k = 1
            while (filename in used) or (not self.overwrite and osp.exists(filepath)):
                filename = f"{root}_{k}{ext}"
                filepath = osp.join(self.directory, filename)
                k += 1
            used.add(filename)
            filenames[i] = filename
        return filenames

    def generate_filepath_obj_pairs(
        self, objs: list[TypeObj]
    ) -> Generator[tuple[str, TypeObj], None, None]:
        """Iterate over (filepath, object) pairs to be saved."""
        for filename, obj in zip(self.build_filenames(objs), objs):
            yield osp.join(self.directory, filename), obj

    directory = gds.DirectoryItem(_("Directory"))
    basename = gds.StringItem(
        _("Basename pattern"),
        default="{title}",
        help=_("""Pattern accepts a Python format string.

Standard Python formatting fields may be used, including:
{title}, {index}, {count}, {xlabel}, {xunit}, {ylabel}, {yunit},
{metadata}, {metadata[key]}"""),
    )
    extension = gds.StringItem(
        _("Extension"),
        help=_("File extension with leading dot (e.g. .txt or .csv)"),
        regexp=r"^\.\w+$",
    )
    overwrite = gds.BoolItem(
        _("Overwrite"), default=False, help=_("Overwrite existing files")
    )


def read_signals(filename: str) -> Sequence[SignalObj]:
    """Read a list of signals from a file.

    Args:
        filename: File name.

    Returns:
        List of signals.
    """
    return SignalIORegistry.read(filename)


def read_signal(filename: str) -> SignalObj:
    """Read a signal from a file.

    Args:
        filename: File name.

    Returns:
        Signal.
    """
    return read_signals(filename)[0]


def write_signal(filename: str, signal: SignalObj) -> None:
    """Write a signal to a file.

    Args:
        filename: File name.
        signal: Signal.
    """
    SignalIORegistry.write(filename, signal)


def write_signals(p: SaveToDirectoryParam, signals: list[SignalObj]) -> None:
    """Write a list of signals to a file.

    Args:
        p: Save to directory parameters.
        signals: List of signals.
    """
    for filepath, signal in p.generate_filepath_obj_pairs(signals):
        SignalIORegistry.write(filepath, signal)


def read_images(
    filename: str, *, param: gds.DataSet | None = None
) -> Sequence[ImageObj]:
    """Read a list of images from a file.

    Args:
        filename: File name.
        param: Optional format-specific read parameters. Formats without
         parameter support reject non-None values.

    Returns:
        List of images.
    """
    return ImageIORegistry.read(filename, param=param)


def read_image(filename: str, *, param: gds.DataSet | None = None) -> ImageObj:
    """Read an image from a file.

    Args:
        filename: File name.
        param: Optional format-specific read parameters. Formats without
         parameter support reject non-None values.

    Returns:
        Image.
    """
    return read_images(filename, param=param)[0]


def write_image(
    filename: str, image: ImageObj, param: ImageExportParam | None = None
) -> None:
    """Write an image to a file.

    Args:
        filename: File name.
        image: Image.
        param: Optional format-aware export parameters.
    """
    if param is None:
        ImageIORegistry.write(filename, image)
        return
    exported_image = copy.deepcopy(image)
    exported_image.data = prepare_image_for_export(image.data, filename, param)
    ImageIORegistry.write(filename, exported_image)


def write_images(p: SaveToDirectoryParam, images: list[ImageObj]) -> None:
    """Write a list of images to files in a directory.

    Args:
        p: Save to directory parameters.
        images: List of images.
    """
    for filepath, image in p.generate_filepath_obj_pairs(images):
        ImageIORegistry.write(filepath, image)
