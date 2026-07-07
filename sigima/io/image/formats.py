# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima I/O image formats
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import datetime
import os.path as osp
import re

import imageio.v3 as iio
import numpy as np
import pandas as pd
import scipy.io as sio
import skimage.io
from guidata.io import HDF5Reader, HDF5Writer

import sigima
from sigima.config import _, options
from sigima.io import ftlab
from sigima.io.base import FormatInfo
from sigima.io.common.converters import convert_array_to_valid_dtype
from sigima.io.enums import FileEncoding
from sigima.io.image import funcs
from sigima.io.image.base import (
    ImageFormatBase,
    MultipleImagesFormatBase,
    SingleImageFormatBase,
)
from sigima.objects.image import ImageObj, create_image
from sigima.worker import CallbackWorkerProtocol


class HDF5ImageFormat(ImageFormatBase):
    """Object representing HDF5 image file type"""

    FORMAT_INFO = FormatInfo(
        name="HDF5",
        extensions="*.h5ima",
        readable=True,
        writeable=True,
    )
    GROUP_NAME = "image"

    # pylint: disable=unused-argument
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
        reader = HDF5Reader(filename)
        try:
            with reader.group(self.GROUP_NAME):
                obj = ImageObj()
                obj.deserialize(reader)
        except ValueError as exc:
            raise ValueError("No valid image data found") from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Unexpected error reading HDF5 image from {filename}"
            ) from exc
        finally:
            reader.close()
        return [obj]

    def write(self, filename: str, obj: ImageObj) -> None:
        """Write data to file

        Args:
            filename: file name
            obj: native object (signal or image)

        Raises:
            NotImplementedError: if format is not supported
        """
        assert isinstance(obj, ImageObj), "Object is not an image"
        writer = HDF5Writer(filename)
        with writer.group(self.GROUP_NAME):
            obj.serialize(writer)
        writer.close()


class ClassicsImageFormat(SingleImageFormatBase):
    """Object representing classic image file types"""

    FORMAT_INFO = FormatInfo(
        name="BMP, JPEG, PNG, TIFF, JPEG2000",
        extensions="*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.jp2",
        readable=True,
        writeable=True,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        data = skimage.io.imread(filename)
        if data.ndim == 3:
            # Color image (e.g. RGB/RGBA JPEG or PNG): convert to grayscale while
            # *preserving the original value range*. This is intentionally not
            # done with skimage's ``as_gray=True`` option, which returns a float
            # image normalized to [0, 1] (via ``rgb2gray``): that scaling made
            # the imported values unusable for quantitative analysis (all values
            # <= 1). The conversion below mirrors ImageJ's default behavior: an
            # unweighted mean of the color channels, kept in the source integer
            # range (0-255 for 8-bit, 0-65535 for 16-bit, etc.).
            nchannels = data.shape[2]
            if nchannels == 2:
                # Grayscale + alpha: keep the grayscale channel only
                data = data[:, :, 0]
            else:
                # RGB or RGBA: drop the alpha channel (if any) and average R, G, B
                rgb = data[:, :, :3]
                src_dtype = rgb.dtype
                gray = rgb.mean(axis=2)
                if np.issubdtype(src_dtype, np.integer):
                    gray = np.round(gray)
                data = gray.astype(src_dtype)
        return data

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        ext = osp.splitext(filename)[1].lower()
        if ext in (".bmp", ".jpg", ".jpeg", ".png"):
            if data.dtype is not np.uint8:
                data = data.astype(np.uint8)
        if ext in (".jp2",):
            if data.dtype not in (np.uint8, np.uint16):
                data = data.astype(np.uint16)
        skimage.io.imsave(filename, data, check_contrast=False)


class NumPyImageFormat(SingleImageFormatBase):
    """Object representing NumPy image file type"""

    FORMAT_INFO = FormatInfo(
        name="NumPy",
        extensions="*.npy",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        return convert_array_to_valid_dtype(np.load(filename), ImageObj.VALID_DTYPES)

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        np.save(filename, data)


class NotCoordinatedTextFileError(Exception):
    """Exception raised when a file is not a coordinated text file"""


class CoordinatedTextFileReader:
    """Utility class for reading text files with metadata and coordinates"""

    @staticmethod
    def read_images(filename: str) -> list[ImageObj]:
        """Read list of image objects from coordinated text file.

        Args:
            filename: File name

        Returns:
            List of image objects
        """
        file_metadata = CoordinatedTextFileReader.read_metadata(filename)

        # Validate metadata and raise on inconsistent or missing keys
        CoordinatedTextFileReader.verify_metadata(filename, file_metadata)

        dict_keys = file_metadata.keys()
        allowed_column_header = {
            "X",
            "Y",
            "Z",
            "Zre",
            "Zim",
            "Z Error",
            "Zre Error",
            "Zim Error",
        }
        columns_header = [k for k in dict_keys if k in allowed_column_header]

        # Remove keys that are in columns_header and construct metadata dictionary
        metadata = {
            key: value[0]
            for key, value in file_metadata.items()
            if key not in columns_header
        }
        metadata["source"] = filename

        df = CoordinatedTextFileReader.read_data(filename, columns_header)

        name = osp.basename(filename)

        try:
            # Check if coordinates are uniform or non-uniform
            x_coords = np.sort(df["X"].unique())
            y_coords = np.sort(df["Y"].unique())

            # Check if we have a regular grid structure
            expected_points = len(x_coords) * len(y_coords)
            actual_points = len(df)

            # Extract coordinate and data information
            (zlabel, zunit) = file_metadata.get("Z", file_metadata.get("Zre", ("", "")))
            (xlabel, xunit) = file_metadata.get("X", ("X", ""))
            (ylabel, yunit) = file_metadata.get("Y", ("Y", ""))

            if xlabel is None:
                xlabel = "X"
            if ylabel is None:
                ylabel = "Y"
            if zlabel is None:
                zlabel = "Z"

            xunit = "" if xunit is None else str(xunit)
            yunit = "" if yunit is None else str(yunit)
            zunit = "" if zunit is None else str(zunit)

            if expected_points == actual_points:
                # Regular grid - can use pivot to create 2D array
                data = df.pivot(index="Y", columns="X", values="Z").values
                data = convert_array_to_valid_dtype(data, ImageObj.VALID_DTYPES)

                # Check if coordinates are truly uniform (evenly spaced)
                x_uniform = len(x_coords) >= 2 and np.allclose(
                    np.diff(x_coords), x_coords[1] - x_coords[0], rtol=1e-10
                )
                y_uniform = len(y_coords) >= 2 and np.allclose(
                    np.diff(y_coords), y_coords[1] - y_coords[0], rtol=1e-10
                )

                image = create_image(
                    name,
                    metadata=metadata,
                    data=data,
                    units=(xunit, yunit, zunit),
                    labels=(xlabel, ylabel, zlabel),
                )

                if x_uniform and y_uniform:
                    # Set uniform coordinates
                    dx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
                    dy = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
                    x0 = float(x_coords[0]) if len(x_coords) > 0 else 0.0
                    y0 = float(y_coords[0]) if len(y_coords) > 0 else 0.0
                    image.set_uniform_coords(dx, dy, x0, y0)
                else:
                    # Set non-uniform coordinates
                    image.set_coords(x_coords.astype(float), y_coords.astype(float))
            else:
                # Non-regular grid - cannot create proper 2D array from this data
                raise ValueError(
                    f"File {filename} contains {actual_points} data points "
                    f"but expected {expected_points} for a regular grid "
                    f"({len(x_coords)}×{len(y_coords)}). "
                    "Coordinated text files must contain data on a complete "
                    "rectangular grid."
                )

            images_list = [image]
        except ValueError as exc:
            raise ValueError(f"File {filename} wrong format.\n{exc}") from exc

        if "Z Error" in df.columns:
            # For error data, use the same coordinate structure as the main image
            error_data = df.pivot(index="Y", columns="X", values="Z Error").values

            image_error = create_image(
                name + " error",
                metadata={"source": filename},
                data=error_data,
                units=(
                    file_metadata["X"][1],
                    file_metadata["Y"][1],
                    file_metadata.get(
                        "Z Error",
                        file_metadata.get(
                            "Zre Error",
                            file_metadata.get("Z", file_metadata.get("Zre", ("", ""))),
                        ),
                    )[1],
                ),
                labels=(
                    file_metadata["X"][0],
                    file_metadata["Y"][0],
                    file_metadata.get("Z", file_metadata.get("Zre", ("", "")))[0]
                    + " error",
                ),
            )

            # Apply the same coordinate system as the main image
            if image.is_uniform_coords:
                image_error.set_uniform_coords(image.dx, image.dy, image.x0, image.y0)
            else:
                image_error.set_coords(image.xcoords.copy(), image.ycoords.copy())

            images_list.append(image_error)

        return images_list

    @staticmethod
    def read_metadata(filename: str) -> dict[str, tuple | None]:
        """Read metadata from file

        Args:
            filename: File name

        Returns:
            Metadata dictionary structured as {key: (value, unit)}
            Available keys can be are:
            - nx (value is int)
            - ny (value is int)
            - X (value represents axis label)
            - Y (value represents axis label)
            - Z (value represents axis label)
            - Zre (value represents axis label)
            - Zim (value represents axis label)
            - Z Error (value is none)
            - Zre Error (value is none)
            - Zim Error (value is none)
        """
        metadata = {}

        try:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("#"):
                        break

                    # Remove leading '#' and strip whitespace
                    content = line[1:].strip()

                    # Parse specific patterns
                    parsed = CoordinatedTextFileReader._parse_metadata_line(content)
                    if parsed:
                        key, value_unit = parsed
                        metadata[key] = value_unit

        except (ValueError, IOError) as exc:
            raise ValueError(f"Could not read metadata from file {filename}") from exc

        return metadata

    @staticmethod
    def _parse_metadata_line(line: str) -> tuple[str, tuple] | None:
        """Parse a single metadata line into key-value-unit tuple.

        Args:
            line: Line to parse (without # prefix)

        Returns:
            Tuple of (key, (value, unit)) or None if not parseable
        """
        # Handle special patterns first
        if match := re.match(r"Created by (.*)", line):
            return "author", (match.group(1).strip(), None)

        if match := re.match(
            r"Created on (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)", line
        ):
            date_str, _time_str = match.groups()
            return "creation_date", (date_str, None)
            # Note: creation_time is lost in this simplified version

        if match := re.match(r"Using matrislib ([\d\.a-zA-Z-]+)", line):
            return "software_version", (f"matrislib {match.group(1)}", None)

        # Handle error columns without colons
        if line.startswith(("Z Error", "Zre Error", "Zim Error")):
            if ":" not in line:
                line = line.replace("Error", "Error :", 1)

        # Must contain colon for key-value pairs
        if ":" not in line:
            return None

        # Remove Real(...) or Imaginary(...) wrappers
        line = re.sub(r"(?:Real|Imaginary)\(([^\)]*)\)", r"\1", line)

        # Split on first colon
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        # Parse value and unit
        value, unit = CoordinatedTextFileReader._parse_value_and_unit(rest)

        return key, (value, unit)

    @staticmethod
    def _parse_value_and_unit(
        text: str,
    ) -> tuple[int | float | bool | str | None, str | None]:
        """Parse value and unit from text like 'value (unit)' or just 'value'.

        Intelligently converts values to appropriate types:
        - Booleans: "true"/"false" (case-insensitive) → bool
        - Integers: "123", "-456" → int
        - Floats: "1.23", "-4.56", "1.2e-3" → float
        - None: empty string → None
        - Strings: everything else → str

        Args:
            text: Text to parse

        Returns:
            Tuple of (value, unit) where value can be int, float, bool, str, or None
        """
        text = text.strip()

        # Extract unit in parentheses if present
        unit = None
        if text.endswith(")"):
            if "(" in text:
                parts = text.rsplit("(", 1)
                text = parts[0].strip()
                unit = parts[1].rstrip(")").strip()
                if not unit:
                    unit = None

        # Parse value with intelligent type detection
        if not text:
            value = None
        elif text.lower() in ("true", "false"):
            # Boolean values
            value = text.lower() == "true"
        else:
            # Try to parse as number
            try:
                # Check if it looks like an integer (no decimal point or exponent)
                if "." not in text and "e" not in text.lower():
                    value = int(text)
                else:
                    # Parse as float
                    value = float(text)
            except ValueError:
                # Not a number, keep as string
                value = text

        return value, unit

    @staticmethod
    def verify_metadata(filename: str, metadata: dict[str, tuple | None]) -> None:
        """Verify metadata keys consistency.

        Perform a set of sanity checks on the parsed metadata and raise an
        appropriate exception on failure.

        Args:
            filename: Parsed filename used for error messages.
            metadata: Metadata dictionary parsed from file header.

        Raises:
            NotCoordinatedTextFileError: When file is not a valid format.
            ValueError: When required fields are missing or inconsistent.
        """
        # Check if this is a coordinated text file by looking for key indicators
        has_format_indicators = "software_version" in metadata or (
            "creation_date" in metadata
            and any(col in metadata for col in ["X", "Y", "Z", "Zre", "Zim"])
        )

        if not has_format_indicators:
            raise NotCoordinatedTextFileError(
                f"File {filename} does not appear to be a coordinated text format file "
                "(missing expected metadata structure)"
            )

        columns_header = [k for k in metadata.keys() if k not in ("nx", "ny")]

        # Required columns check
        if "X" not in columns_header or "Y" not in columns_header:
            raise ValueError(
                f"File {filename}: Missing required X, Y columns in header"
            )

        # Z column validation
        has_z = "Z" in columns_header
        has_complex = "Zre" in columns_header or "Zim" in columns_header

        if not (has_z or has_complex):
            raise ValueError(
                f"File {filename}: Must contain either Z column or Zre/Zim columns"
            )

        if has_z and has_complex:
            raise ValueError(
                f"File {filename}: Cannot contain both Z and Zre/Zim columns"
            )

        # Complex Z validation
        if has_complex:
            if ("Zre" in columns_header) ^ ("Zim" in columns_header):
                raise ValueError(
                    f"File {filename}: Both Zre and Zim columns "
                    f"must be present together"
                )

        # Error column validation
        has_z_error = "Z Error" in columns_header
        has_complex_error = (
            "Zre Error" in columns_header or "Zim Error" in columns_header
        )

        if has_z_error and has_complex_error:
            raise ValueError(
                f"File {filename}: Cannot contain both Z Error and "
                f"Zre Error/Zim Error columns"
            )

        if has_complex_error:
            if ("Zre Error" in columns_header) ^ ("Zim Error" in columns_header):
                raise ValueError(
                    f"File {filename}: Both Zre Error and Zim Error columns "
                    f"must be present together"
                )

    @staticmethod
    def _try_df_reading(filename: str, columns_header: list[str]) -> pd.DataFrame:
        """Try to read the data file with various parsing options.

        Args:
            filename: File name
            columns_header: List of column headers to use when reading the data.

        Returns:
            DataFrame containing the image data.

        Raises:
            ValueError: If the file cannot be read with any of the tried options.
        """
        # Define parsing configurations to try in order of preference
        parsing_configs = [
            (encoding, decimal, delimiter)
            for encoding in FileEncoding
            for decimal in (".", ",")
            for delimiter in (r"\s+", ",", ";")
        ]

        last_error = None
        for encoding, decimal, delimiter in parsing_configs:
            try:
                df = pd.read_csv(
                    filename,
                    decimal=decimal,
                    comment="#",
                    delimiter=delimiter,
                    encoding=encoding,
                    names=columns_header,
                )
                # Drop entirely empty columns introduced by trailing delimiters
                df = df.dropna(axis=1, how="all")
                return df

            except (ValueError, UnicodeDecodeError) as exc:
                last_error = exc
                continue

        # If we get here, all parsing attempts failed
        raise ValueError(
            f"Could not read image data from file {filename}. Last error: {last_error}"
        ) from last_error

    @staticmethod
    def read_data(filename: str, columns_header: list[str]) -> pd.DataFrame:
        """Read data and return it.

        Args:
            filename: File name

        Returns:
            Image array data
        """
        # Try several parsing variants (encoding, decimal and delimiter).
        df: pd.DataFrame | None = None

        df = CoordinatedTextFileReader._try_df_reading(filename, columns_header)

        # if Z is present, the image is Real

        if "Zre" in df.columns:
            df["Z"] = df["Zre"] + 1j * df["Zim"]
            df = df.drop(columns=["Zre", "Zim"])
            if "Zre Error" in df.columns:
                df["Z Error"] = df["Zre Error"] + 1j * df["Zim Error"]
                df = df.drop(columns=["Zre Error", "Zim Error"])

        return df


class CoordinatedTextFileWriter:
    """Utility class for writing text files with metadata and coordinates"""

    @staticmethod
    def write_image(filename: str, obj: ImageObj) -> None:
        """Write image object to coordinated text file.

        Args:
            filename: File name to write to
            obj: Image object to write

        Raises:
            ValueError: If image has invalid coordinate system
        """
        # Validate that we can write this image
        if obj.data is None:
            raise ValueError(
                "Cannot write image with no data to coordinated text format"
            )

        # Get coordinate information
        if obj.is_uniform_coords:
            # Generate coordinate arrays for uniform coordinates
            ny, nx = obj.data.shape
            x_coords = obj.x0 + np.arange(nx) * obj.dx
            y_coords = obj.y0 + np.arange(ny) * obj.dy
        else:
            # Use non-uniform coordinates directly
            x_coords = obj.xcoords
            y_coords = obj.ycoords
            if x_coords is None or y_coords is None:
                raise ValueError("Cannot write image with missing coordinate arrays")

        # Create meshgrid for the data
        X, Y = np.meshgrid(x_coords, y_coords)

        # Flatten arrays for CSV output
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = obj.data.flatten()

        # Write file
        with open(filename, "w", encoding="utf-8") as f:
            # Write metadata header
            f.write(f"# Created by Sigima {sigima.__version__}\n")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            f.write(f"# Created on {timestamp}\n")
            f.write(f"# nx: {obj.data.shape[1]}\n")
            f.write(f"# ny: {obj.data.shape[0]}\n")

            # Write axis information
            f.write(f"# X: {obj.xlabel}")
            if obj.xunit:
                f.write(f" ({obj.xunit})")
            f.write("\n")

            f.write(f"# Y: {obj.ylabel}")
            if obj.yunit:
                f.write(f" ({obj.yunit})")
            f.write("\n")

            f.write(f"# Z: {obj.zlabel}")
            if obj.zunit:
                f.write(f" ({obj.zunit})")
            f.write("\n")

            # Write additional metadata if present
            if obj.metadata:
                for key, value in obj.metadata.items():
                    if key not in ("source",):  # Skip internal metadata
                        f.write(f"# {key}: {value}\n")

            # Write data columns
            for x, y, z in zip(x_flat, y_flat, z_flat):
                f.write(f"{x}\t{y}\t{z}\n")


class TextImageFormat(SingleImageFormatBase):
    """Object representing text image file type"""

    FORMAT_INFO = FormatInfo(
        name=_("Text files"),
        extensions="*.txt *.csv *.asc",
        readable=True,
        writeable=True,
    )

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
        # Try to read as coordinated text format first
        # (for .txt/.csv files with metadata and coordinates)
        if filename.lower().endswith((".txt", ".csv")):
            try:
                return CoordinatedTextFileReader.read_images(filename)
            except NotCoordinatedTextFileError:
                # Not a coordinated text file, continue with regular text processing
                pass

        # Read as generic text file
        obj = self.create_object(filename)
        obj.data = self.read_data(filename)
        unique_values = np.unique(obj.data)
        if len(unique_values) == 2:
            # Binary image: set LUT range to unique values
            obj.zscalemin, obj.zscalemax = unique_values.tolist()
        return [obj]

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        for encoding in FileEncoding:
            for decimal in (".", ","):
                for delimiter in (",", ";", r"\s+"):
                    try:
                        df = pd.read_csv(
                            filename,
                            decimal=decimal,
                            delimiter=delimiter,
                            encoding=encoding,
                            header=None,
                        )
                        # Handle the extra column created with trailing delimiters.
                        df = df.dropna(axis=1, how="all")
                        data = df.to_numpy()
                        return convert_array_to_valid_dtype(data, ImageObj.VALID_DTYPES)
                    except ValueError:
                        continue
        raise ValueError(f"Could not read image data from file {filename}.")

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file.

        Args:
            filename: File name.
            data: Image array data.
        """
        if np.issubdtype(data.dtype, np.integer):
            fmt = "%d"
        elif np.issubdtype(data.dtype, np.floating) or np.issubdtype(
            data.dtype, np.complexfloating
        ):
            fmt = "%.18e"
        else:
            raise NotImplementedError(
                f"Writing data of type {data.dtype} to text file is not supported."
            )
        ext = osp.splitext(filename)[1]
        if ext.lower() in (".txt", ".asc", ""):
            np.savetxt(filename, data, fmt=fmt)
        elif ext.lower() == ".csv":
            np.savetxt(filename, data, fmt=fmt, delimiter=",")
        else:
            raise ValueError(f"Unknown text file extension {ext}")

    def write(self, filename: str, obj: ImageObj) -> None:
        """Write data to file

        Args:
            filename: file name
            obj: image object
        """
        if not isinstance(obj, ImageObj):
            raise ValueError("Object is not an image")

        # Check if object has non-uniform coordinates and filename is TXT or CSV
        # If so, use coordinated text format
        ext = osp.splitext(filename)[1].lower()
        if ext in (".txt", ".csv") and not obj.is_uniform_coords:
            try:
                CoordinatedTextFileWriter.write_image(filename, obj)
                return
            except Exception:  # pylint: disable=broad-except
                # Fall back to regular text format if writing fails
                pass

        # Use default text format
        super().write(filename, obj)


class MatImageFormat(SingleImageFormatBase):
    """Object representing MAT-File image file type"""

    FORMAT_INFO = FormatInfo(
        name=_("MAT-Files"),
        extensions="*.mat",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

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
        mat = sio.loadmat(filename)
        allimg: list[ImageObj] = []
        for dname, data in mat.items():
            if dname.startswith("__") or not isinstance(data, np.ndarray):
                continue
            if len(data.shape) != 2:
                continue
            obj = self.create_object(filename)
            obj.data = data
            if dname != "img":
                obj.title += f" ({dname})"
            allimg.append(obj)
        return allimg

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        # This method is not used, as read() is overridden

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        sio.savemat(filename, {"img": data})


class DICOMImageFormat(SingleImageFormatBase):
    """Object representing DICOM image file type"""

    FORMAT_INFO = FormatInfo(
        name="DICOM",
        extensions="*.dcm *.dicom",
        readable=True,
        writeable=False,
        requires=["pydicom"],
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        return funcs.imread_dicom(filename)


class AndorSIFImageFormat(MultipleImagesFormatBase):
    """Object representing an Andor SIF image file type"""

    FORMAT_INFO = FormatInfo(
        name="Andor SIF",
        extensions="*.sif",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        return funcs.imread_sif(filename)


# Generate classes based on the information above:
def generate_imageio_format_classes(
    imageio_formats: list[list[str, str]]
    | list[tuple[str, str]]
    | tuple[tuple[str, str]]
    | tuple[list[str, str]]
    | None = None,
) -> None:
    """Generate classes based on the information above"""
    if imageio_formats is None:
        imageio_formats = options.imageio_formats.get()

    for extensions, name in imageio_formats:
        class_dict = {
            "FORMAT_INFO": FormatInfo(
                name=name, extensions=extensions, readable=True, writeable=False
            ),
            "read_data": staticmethod(
                lambda filename: iio.imread(filename, index=None)
            ),
        }
        class_name = extensions.split()[0].split(".")[1].upper() + "ImageFormat"
        globals()[class_name] = type(
            class_name, (MultipleImagesFormatBase,), class_dict
        )


generate_imageio_format_classes()


class SpiriconImageFormat(SingleImageFormatBase):
    """Object representing Spiricon image file type"""

    FORMAT_INFO = FormatInfo(
        name="Spiricon",
        extensions="*.scor-data",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        return funcs.imread_scor(filename)


class XYZImageFormat(SingleImageFormatBase):
    """Object representing Dürr NDT XYZ image file type"""

    FORMAT_INFO = FormatInfo(
        name="Dürr NDT",
        extensions="*.xyz",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        with open(filename, "rb") as fdesc:
            cols = int(np.fromfile(fdesc, dtype=np.uint16, count=1)[0])
            rows = int(np.fromfile(fdesc, dtype=np.uint16, count=1)[0])
            arr = np.fromfile(fdesc, dtype=np.uint16, count=cols * rows)
            arr = arr.reshape((rows, cols))
        return np.fliplr(arr)


class FTLabImageFormat(SingleImageFormatBase):
    """FT-Lab image file."""

    FORMAT_INFO = FormatInfo(
        name="FT-Lab",
        extensions="*.ima",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read and return data.

        Args:
            filename: Path to FT-Lab file.

        Returns:
            Image data.
        """
        return ftlab.imread_ftlabima(filename)
