"""Parameterized RAW image import."""

from __future__ import annotations

import os.path as osp
from typing import BinaryIO

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.io.base import FormatInfo
from sigima.io.image.base import ImageFormatBase
from sigima.objects.image import ImageObj, create_image
from sigima.worker import CallbackWorkerProtocol

__all__ = ["RawImageFormat", "RawImageImportParam"]

RAW_DTYPES = ("uint8", "uint16", "int16", "int32", "float32", "float64")


class RawImageImportParam(gds.DataSet, title=_("RAW image import")):
    """Parameters controlling RAW binary image import."""

    dtype = gds.ChoiceItem(
        _("Image data type"),
        [(name, name) for name in RAW_DTYPES],
        default="uint16",
    )
    width = gds.IntItem(_("Width"), default=1280, min=1)
    height = gds.IntItem(_("Height"), default=1024, min=1)
    offset = gds.IntItem(_("Offset to first image"), default=0, min=0, unit="bytes")
    count = gds.IntItem(_("Number of images"), default=1, min=1)
    gap = gds.IntItem(_("Gap between images"), default=0, min=0, unit="bytes")
    little_endian = gds.BoolItem(_("Little-endian byte order"), default=True)
    white_is_zero = gds.BoolItem(_("White is zero"), default=False)

    @staticmethod
    def create(
        dtype: str = "uint16",
        width: int = 1280,
        height: int = 1024,
        offset: int = 0,
        count: int = 1,
        gap: int = 0,
        little_endian: bool = True,
        white_is_zero: bool = False,
        **kwargs,
    ) -> RawImageImportParam:
        """Create RAW image import parameters."""
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected RAW image parameters: {names}")
        param = RawImageImportParam()
        param.dtype = dtype
        param.width = width
        param.height = height
        param.offset = offset
        param.count = count
        param.gap = gap
        param.little_endian = little_endian
        param.white_is_zero = white_is_zero
        return param


class RawImageFormat(ImageFormatBase):
    """Object representing a headerless RAW image file."""

    FORMAT_INFO = FormatInfo(
        name=_("RAW image"),
        extensions="*.raw",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def validate_param(param: RawImageImportParam) -> None:
        """Validate RAW image dimensions and binary layout parameters."""
        if param.dtype not in RAW_DTYPES:
            choices = ", ".join(RAW_DTYPES)
            raise ValueError(
                f"RAW image data type must be one of {choices}, got {param.dtype!r}"
            )
        if param.width <= 0:
            raise ValueError("RAW image width must be positive")
        if param.height <= 0:
            raise ValueError("RAW image height must be positive")
        if param.count <= 0:
            raise ValueError("RAW image count must be positive")
        if param.offset < 0:
            raise ValueError("RAW image offset must be nonnegative")
        if param.gap < 0:
            raise ValueError("RAW image gap must be nonnegative")

    @staticmethod
    def get_frame_layout(
        filename: str, param: RawImageImportParam
    ) -> tuple[np.dtype, np.dtype, int, int]:
        """Validate and return the RAW frame layout.

        Args:
            filename: File name.
            param: RAW image import parameters.

        Returns:
            Native dtype, file dtype, pixel count and frame size.

        Raises:
            ValueError: If parameters or file size are inconsistent.
        """
        RawImageFormat.validate_param(param)
        native_dtype = np.dtype(param.dtype)
        file_dtype = native_dtype.newbyteorder("<" if param.little_endian else ">")
        pixel_count = param.width * param.height
        frame_size = pixel_count * native_dtype.itemsize
        expected_size = param.offset + param.count * frame_size
        expected_size += (param.count - 1) * param.gap
        actual_size = osp.getsize(filename)
        if actual_size < expected_size:
            raise ValueError(
                f"RAW file is truncated: expected {expected_size} bytes, "
                f"got {actual_size}"
            )
        if actual_size > expected_size:
            raise ValueError(
                f"RAW file has trailing bytes: expected {expected_size} bytes, "
                f"got {actual_size}"
            )

        return native_dtype, file_dtype, pixel_count, frame_size

    @staticmethod
    def read_frame(
        file: BinaryIO,
        param: RawImageImportParam,
        index: int,
        native_dtype: np.dtype,
        file_dtype: np.dtype,
        pixel_count: int,
        frame_size: int,
    ) -> np.ndarray:
        """Read one RAW frame from the current file."""
        frame_offset = param.offset + index * (frame_size + param.gap)
        file.seek(frame_offset)
        frame = np.fromfile(file, dtype=file_dtype, count=pixel_count)
        if frame.size != pixel_count:
            raise ValueError(
                f"RAW frame {index} is truncated: expected {pixel_count} "
                f"values, got {frame.size}"
            )
        return frame.astype(native_dtype, copy=False).reshape(param.height, param.width)

    @staticmethod
    def create_object(filename: str, index: int | None = None) -> ImageObj:
        """Create an empty image object for a RAW frame."""
        name = osp.basename(filename)
        if index is not None:
            name += f" {index:02d}"
        return create_image(name, metadata={"source": filename})

    def read(
        self,
        filename: str,
        worker: CallbackWorkerProtocol | None = None,
        *,
        param: gds.DataSet | None = None,
    ) -> list[ImageObj]:
        """Read one image object per RAW frame.

        Args:
            filename: File name.
            worker: Callback worker object.
            param: RAW image import parameters.

        Returns:
            List of image objects.

        Raises:
            TypeError: If RAW image import parameters are missing or invalid.
        """
        if not isinstance(param, RawImageImportParam):
            raise TypeError("RAW image reading requires RawImageImportParam")
        native_dtype, file_dtype, pixel_count, frame_size = self.get_frame_layout(
            filename, param
        )
        images: list[ImageObj] = []
        with open(filename, "rb") as file:
            for index in range(param.count):
                if worker is not None and worker.was_canceled():
                    break
                frame = self.read_frame(
                    file,
                    param,
                    index,
                    native_dtype,
                    file_dtype,
                    pixel_count,
                    frame_size,
                )
                object_index = index if param.count > 1 else None
                image = self.create_object(filename, index=object_index)
                image.data = frame
                image.metadata["white_is_zero"] = bool(param.white_is_zero)
                images.append(image)
                if worker is not None:
                    worker.set_progress((index + 1) / param.count)
        return images

    def write(self, filename: str, obj: ImageObj) -> None:
        """Reject RAW image writing."""
        raise NotImplementedError(f"Writing to {filename} is not supported")
