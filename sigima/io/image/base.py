# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image I/O registry
"""

from __future__ import annotations

import abc
import os.path as osp
from typing import Sequence

import numpy as np

from sigima.config import _
from sigima.io.base import BaseIORegistry, FormatBase, IOAction
from sigima.objects.image import ImageObj, create_image
from sigima.worker import CallbackWorkerProtocol


class ImageIORegistry(BaseIORegistry):
    """Metaclass for registering image I/O handler classes"""

    REGISTRY_INFO: str = _("Image I/O formats")

    _io_format_instances: list[ImageFormatBase] = []

    @classmethod
    def get_filters(mcs, action: IOAction) -> str:
        """Return grouped load filters or format-specific save filters."""
        if action == IOAction.LOAD:
            return super().get_filters(action)
        assert action == IOAction.SAVE
        classic_format_name = "BMP, JPEG, PNG, TIFF, JPEG2000"
        classic_save_filters = (
            "BMP (*.bmp)",
            "JPEG (*.jpg *.jpeg)",
            "PNG (*.png)",
            "TIFF (*.tif *.tiff)",
            "JPEG 2000 (*.jp2)",
        )
        filters = []
        for fmt in mcs.get_formats():
            file_filter = fmt.get_filter(action)
            if file_filter is None:
                continue
            if fmt.info.name == classic_format_name:
                filters.extend(classic_save_filters)
            else:
                filters.append(file_filter)
        return "\n".join(filters)


class ImageFormatBaseMeta(ImageIORegistry, abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class ImageFormatBase(abc.ABC, FormatBase, metaclass=ImageFormatBaseMeta):
    """Base image format object.

    This class is used to define the interface for image I/O formats.
    It is an abstract base class that defines the methods that must be
    implemented by any image format class.
    """

    @abc.abstractmethod
    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> Sequence[ImageObj]:
        """Read list of image objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of image objects
        """

    @abc.abstractmethod
    def write(self, filename: str, obj: ImageObj) -> None:
        """Write data to file

        Args:
            filename: file name
            obj: native object (signal or image)

        Raises:
            NotImplementedError: if format is not supported
        """

    def write_with_options(
        self, filename: str, obj: ImageObj, writer_options: dict[str, object]
    ) -> None:
        """Write an image with format-specific options.

        Args:
            filename: File name
            obj: Image object
            writer_options: Format-specific writer options

        Raises:
            ValueError: If this format does not support writer options
        """
        if writer_options:
            raise ValueError(f"{self.info.name} does not support export options")
        self.write(filename, obj)


class SingleImageFormatBase(ImageFormatBase):
    """Base image format object for single image (e.g., TIFF, PNG, etc.)."""

    @staticmethod
    def create_object(filename: str, index: int | None = None) -> ImageObj:
        """Create empty object

        Args:
            filename: File name
            index: Index of object in file

        Returns:
            Image object
        """
        name = osp.basename(filename)
        if index is not None:
            name += f" {index:02d}"
        return create_image(name, metadata={"source": filename})

    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[ImageObj]:
        """Read list of image objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of image objects
        """
        # Default implementation covers the case of a single image:
        obj = self.create_object(filename)
        obj.data = self.read_data(filename)
        unique_values = np.unique(obj.data)
        if len(unique_values) == 2:
            # Binary image: set LUT range to unique values
            obj.zscalemin, obj.zscalemax = unique_values.tolist()
        return [obj]

    @staticmethod
    @abc.abstractmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """

    def write(self, filename: str, obj: ImageObj) -> None:
        """Write data to file

        Args:
            filename: file name
            obj: native object (signal or image)

        Raises:
            NotImplementedError: if format is not supported
        """
        data = obj.data
        self.write_data(filename, data)

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        raise NotImplementedError(f"Writing to {filename} is not supported")


class MultipleImagesFormatBase(SingleImageFormatBase):
    """Base image format object for multiple images (e.g., SIF or SPE).

    Works with read function that returns a NumPy array of 3 dimensions, where
    the first dimension is the number of images.
    """

    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[ImageObj]:
        """Read list of image objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of image objects
        """
        data = self.read_data(filename)
        if len(data.shape) == 3:
            objlist = []
            for idx in range(data.shape[0]):
                obj = self.create_object(filename, index=idx)
                obj.data = data[idx, ::]
                objlist.append(obj)
                if worker is not None:
                    worker.set_progress((idx + 1) / data.shape[0])
                    if worker.was_canceled():
                        break
            return objlist
        obj = self.create_object(filename)
        obj.data = data
        return [obj]
