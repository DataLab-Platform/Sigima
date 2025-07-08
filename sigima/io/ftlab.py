# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""FT-Lab I/O common functions."""

from __future__ import annotations

import struct
import typing
from enum import Enum

import numpy as np


def check_file_header(fileobj: typing.BinaryIO) -> None:
    """Read and validate the file header.

    Args:
        fileobj: Opened file object in binary mode.

    Raises:
        ValueError: If the header is invalid or incomplete.
    """
    nb_bytes_to_read = 6
    read_bytes = fileobj.read(nb_bytes_to_read)
    if len(read_bytes) != nb_bytes_to_read:
        raise ValueError(f"Header is incomplete (expected {nb_bytes_to_read} bytes).")
    # Unpack the first six bytes to check the header.
    # The first six bytes are expected to be in the format of three little-endian 16-bit
    # signed integers.
    i1, _, i3 = struct.unpack("<3h", read_bytes)
    i1_possible_values = (-31609, -30844)
    i3_expected_value = 8224
    remanining_header_length = 250
    if (i1 in i1_possible_values) and (i3 == i3_expected_value):
        # Skip the rest of the header.
        read_bytes = fileobj.read(remanining_header_length)
        if len(read_bytes) != remanining_header_length:
            raise ValueError(
                f"Header is incomplete (expected {remanining_header_length} bytes)."
            )
    else:
        raise ValueError("Unexpected values in header.")


def read_length_prefixed_string(
    fileobj: typing.BinaryIO, encoding: str = "latin-1"
) -> str:
    """Read a length-prefixed string.

    Args:
        fileobj: Opened file object in binary mode.
        encoding: Encoding to decode the string. Defaults to "latin-1".

    Returns:
        The decoded string.

    Raises:
        ValueError: If the string length is invalid or file is too short.
    """
    nb_bytes_to_read = 4
    read_bytes = fileobj.read(nb_bytes_to_read)
    if len(read_bytes) != nb_bytes_to_read:
        raise ValueError("Failed to read string length.")
    length_to_read = struct.unpack("<i", read_bytes)[0]
    if length_to_read < 0:
        raise ValueError("Negative string length.")
    pad = length_to_read % 2 != 0
    read_string = fileobj.read(length_to_read)
    if len(read_string) != length_to_read:
        raise ValueError("Failed to read data.")
    if pad:
        read_byte = fileobj.read(1)  # Skip padding byte.
        if len(read_byte) != 1:
            raise ValueError("Failed to read padding byte.")
    return read_string.decode(encoding, errors="replace")


class SignalType(Enum):
    """Enum for FT-Lab signal types."""

    REAL_WITH_GIVEN_X_RANGE = {1, 3, 4, 5, 27, 31, 32, 33, 34, 52, 61, 99}
    REAL_WITH_GIVEN_X = {11, 21, 22, 23, 24, 26, 51, 71}
    COMPLEX_WITH_GIVEN_X_RANGE = {2}
    COMPLEX_WITH_GIVEN_X = {12}


class FTLabSignalFile:
    """FT-Lab signal file."""

    def __init__(self, file_path: str) -> None:
        """
        Initialize an FTLabSignalFile object.

        Args:
            file_path: Path to the FT-Lab signal file (.sig).
        """
        self.file_path: str = file_path
        self.x: np.ndarray
        self.y: np.ndarray
        self.xu: str
        self.yu: str

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"FTLabSignalFile("
            f"file_path={self.file_path!r}, "
            f"x_shape={None if self.x is None else self.x.shape}, "
            f"y_shape={None if self.y is None else self.y.shape}, "
            f"xu={self.xu!r}, "
            f"yu={self.yu!r})"
        )

    def _check_header(self, fid: typing.BinaryIO) -> None:
        """Check the file header.

        Args:
            fid: Opened file object in binary mode.
        """
        check_file_header(fid)

    def _read_real_with_x_range(
        self, fid: typing.BinaryIO, n: int, start: float, step: float
    ) -> None:
        """Read real data with a given x range.

        Args:
            fid: Opened file object in binary mode.
            n: Number of data points to read.
            start: Start of the x range.
            step: Step size for the x range.
        """

        self.x = np.linspace(start, start + (n - 1) * step, n)
        self.y = np.fromfile(fid, dtype="<d", count=n)
        if self.y.size != n:
            raise ValueError(f"Expected {n} values, got {self.y.size}")

    def _read_real_with_x(self, fid: typing.BinaryIO, n: int) -> None:
        """Read real data with given x values.

        Args:
            fid: Opened file object in binary mode.
            n: Number of data points to read.
        """
        data = np.fromfile(fid, dtype="<d", count=2 * n)
        if data.size != 2 * n:
            raise ValueError(f"Expected {2 * n} values, got {data.size}")
        self.x = data[::2]
        self.y = data[1::2]

    def _read_complex_with_x_range(
        self, fid: typing.BinaryIO, n: int, start: float, step: float
    ) -> None:
        """Read complex data with a given x range.

        Args:
            fid: Opened file object in binary mode.
            n: Number of data points to read.
            start: Start of the x range.
            step: Step size for the x range.
        """
        self.x = np.linspace(start, start + (n - 1) * step, n)
        data = np.fromfile(fid, dtype="<d", count=2 * n)
        if data.size != 2 * n:
            raise ValueError(f"Expected {2 * n} values, got {data.size}")
        self.y = data[::2] + 1j * data[1::2]

    def _read_complex_with_x(self, fid: typing.BinaryIO, n: int) -> None:
        """Read complex data with given x values.

        Args:
            fid: Opened file object in binary mode.
            n: Number of data points to read.
        """
        data = np.fromfile(fid, dtype="<d", count=3 * n)
        if data.size != 3 * n:
            raise ValueError(f"Expected {3 * n} values, got {data.size}")
        self.x = data[::3]
        self.y = data[1::3] + 1j * data[2::3]

    def read_data(self) -> np.ndarray:
        """
        Read the FT-Lab signal file, populate data and metadata.

        This method reads the signal data from the file, checking the header and
        determining the signal type. It supports various signal formats.

        Returns:
            XY data.

        Raises:
            ValueError: If the file cannot be opened or the format is not recognized.
            NotImplementedError: If the signal type is not supported.
        """
        try:
            with open(self.file_path, "rb") as fid:
                self._check_header(fid)

                # Skip signal title
                _ = read_length_prefixed_string(fid, encoding="latin-1")

                # The following image header is expected to contain 20 double values.
                nb_values = 20
                header = np.fromfile(fid, dtype="<d", count=nb_values)
                if header.size != nb_values:
                    raise ValueError("Incomplete signal header.")

                # Check if the version is supported.
                min_version = 5
                if header[19] < min_version:
                    raise NotImplementedError(
                        f"Signal version {header[19]} is not supported."
                    )

                # The first header value is the signal type.
                stype = int(header[0])
                # The second header value is the number of points.
                n = int(header[1])

                self.xu = read_length_prefixed_string(fid)
                self.yu = read_length_prefixed_string(fid)

                if stype in SignalType.REAL_WITH_GIVEN_X_RANGE.value:
                    start = header[4]
                    step = header[2]
                    self._read_real_with_x_range(fid, n, start, step)
                elif stype in SignalType.REAL_WITH_GIVEN_X.value:
                    self._read_real_with_x(fid, n)
                elif stype in SignalType.COMPLEX_WITH_GIVEN_X_RANGE.value:
                    start = header[4]
                    step = header[2]
                    self._read_complex_with_x_range(fid, n, start, step)
                elif stype in SignalType.COMPLEX_WITH_GIVEN_X.value:
                    self._read_complex_with_x(fid, n)
                else:
                    raise NotImplementedError(f"Unsupported signal type: {stype}")
                return np.vstack((self.x, self.y))

        except OSError as e:
            raise ValueError(f"Error opening file: {e}") from e


def sigread_ftlabsig(filename: str):
    """Read an FT-Lab signal file (.sig) and return the XY data.

    Args:
        filename: Path to FT-Lab signal file (.sig).

    Returns:
        XY data from the signal file.
    """
    sig = FTLabSignalFile(filename)
    return sig.read_data()


class ImageType(Enum):
    """Enum for FT-Lab image types."""

    REAL = 101
    COMPLEX = 102


class FTLabImageFile:
    """Class of an FT-Lab image file (.ima)."""

    def __init__(self, file_path: str) -> None:
        """Initialize an FTLabImageFile object.

        Args:
            file_path: path to an FT-Lab image file (.ima).
        """
        self.file_path: str = file_path
        self.image_type: ImageType
        self.dtype: np.dtype
        self.nb_columns: int
        self.nb_lines: int
        self.data: np.ndarray

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"FTLabImageFile("
            f"file_path={self.file_path!r}, "
            f"image_type={getattr(self, 'image_type', None)}, "
            f"dtype={getattr(self, 'dtype', None)}, "
            f"nb_columns={getattr(self, 'nb_columns', None)}, "
            f"nb_lines={getattr(self, 'nb_lines', None)})"
        )

    def _check_header(self, fid: typing.BinaryIO) -> None:
        """Check the file header.

        Args:
            fid: Opened file object in binary mode.
        """
        check_file_header(fid)

    def _read_image_data(self, fid):
        """Read image data from the file.

        Args:
            fid: Opened file object in binary mode.
            image_type: Type of the image (real or complex).
            dtype: Data type of the image data.
            nb_lines: Number of lines (rows) in the image.
            nb_columns: Number of columns in the image.
        """
        size = self.nb_lines * self.nb_columns
        if self.image_type == ImageType.REAL:
            data = np.fromfile(fid, dtype=self.dtype, count=size)
            if data.size != size:
                raise ValueError("Unexpected end of file while reading image data.")
            return data.reshape((self.nb_columns, self.nb_lines)).T
        if self.image_type == ImageType.COMPLEX:
            real = np.fromfile(fid, dtype=self.dtype, count=size)
            imag = np.fromfile(fid, dtype=self.dtype, count=size)
            if real.size != size or imag.size != size:
                raise ValueError(
                    "Unexpected end of file while reading complex image data."
                )
            return (
                real.reshape((self.nb_columns, self.nb_lines)).T
                + 1j * imag.reshape((self.nb_columns, self.nb_lines)).T
            )
        raise NotImplementedError(f"Image type {self.image_type} is not supported.")

    def read_data(self) -> np.ndarray:
        """Read an image file and return its data.

        Returns:
            Image data.

        Raises:
            ValueError: If the file cannot be opened or the format is not recognized.
            NotImplementedError: If the image type or version is not supported.
        """
        try:
            with open(self.file_path, "rb") as fid:
                self._check_header(fid)

                # Skip image title.
                _ = read_length_prefixed_string(fid, encoding="latin-1")

                # The following image header is expected to contain 20 double values.
                nb_values = 20
                header = np.fromfile(fid, dtype="<d", count=nb_values)
                if header.size != nb_values:
                    raise ValueError("Incomplete image header.")

                # Check if the version is supported.
                min_version = 7
                if header[19] < min_version:
                    raise NotImplementedError(
                        f"Image version {header[19]} is not supported."
                    )

                # The first header value is the image type.
                try:
                    self.image_type = ImageType(int(header[0]))
                except ValueError as exc:
                    raise NotImplementedError(
                        f"Image type {int(header[0])} is not supported."
                    ) from exc

                # Data type.
                data_type_map = {
                    8: np.uint8,
                    16: np.uint16,
                    32: np.float32,
                }
                self.dtype = np.dtype(data_type_map.get(int(header[1])))

                # Size parameters.
                self.nb_columns = int(header[2])
                self.nb_lines = int(header[3])

                # Skip units.
                for _ in range(3):
                    _ = read_length_prefixed_string(fid)

                # Read data.
                return self._read_image_data(fid)
        except OSError as e:
            raise ValueError(f"Error opening file: {e}") from e


def imread_ftlabima(filename: str) -> np.ndarray:
    """Open an FT-Lab image file.

    Args:
        filename: path to FT-Lab image file.

    Returns:
        Image data.
    """
    ftlab_file = FTLabImageFile(filename)
    return ftlab_file.read_data()
