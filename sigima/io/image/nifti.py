# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""NIfTI image import (single-file NIfTI-1 and NIfTI-2, ``*.nii`` and ``*.nii.gz``).

2D images are read directly. Slices of 3D volumes are read along the third axis of
the file, as stored: they are not necessarily axial and their display is not an
anatomical orientation. Reading requires NiBabel (``pip install sigima[nifti]``):
without it, the format is not registered, while this module and its parameters stay
importable and :func:`get_nifti_image_info` raises an explicit :class:`ImportError`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os.path as osp
from typing import TYPE_CHECKING, Any, Iterator

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.io.base import FormatInfo, UnsupportedFormatOperationError
from sigima.io.image.base import ImageFormatBase
from sigima.objects.image import ImageObj, create_image
from sigima.worker import CallbackWorkerProtocol

if TYPE_CHECKING:
    import nibabel as nib

__all__ = [
    "NiftiImageFormat",
    "NiftiImageImportParam",
    "NiftiImageInfo",
    "get_nifti_image_info",
]

#: Tolerance on the normalized dot product of the two in-plane axes of the affine
#: transform under which they are considered orthogonal
ORTHOGONALITY_TOLERANCE = 1e-6

#: Metadata key of the NIfTI import provenance
NIFTI_METADATA_KEY = "nifti"

NIBABEL_REQUIRED = (
    "Reading NIfTI files requires NiBabel: install it with "
    "'pip install sigima[nifti]' (or 'pip install nibabel')"
)

#: Spatial units of NiBabel and their DataLab notation
SPATIAL_UNITS = {"meter": "m", "mm": "mm", "micron": "µm"}

#: Conversion of the data types not supported by images, accepted only when the
#: values are preserved exactly
CONVERTED_DTYPES = {
    np.dtype(np.int8): np.dtype(np.int16),
    np.dtype(np.uint32): np.dtype(np.float64),
    np.dtype(np.int64): np.dtype(np.float64),
    np.dtype(np.uint64): np.dtype(np.float64),
    np.dtype(np.longdouble): np.dtype(np.float64),
}

#: Header sizes of NIfTI-1 and NIfTI-2 files, by version
HEADER_SIZES = {348: 1, 540: 2}
#: Number of bytes holding the header size and magic string of both versions
HEADER_PREFIX_SIZE = 348


class NiftiImageImportParam(gds.DataSet, title=_("NIfTI image import")):
    """Slice range of a NIfTI volume to import.

    Bounds are included and indexed from zero along the third axis of the file.
    A single slice is a range whose bounds are equal.
    """

    first_slice = gds.IntItem(_("First slice"), default=0, min=0)
    last_slice = gds.IntItem(_("Last slice"), default=0, min=0)

    @staticmethod
    def create(
        first_slice: int = 0, last_slice: int | None = None, **kwargs
    ) -> NiftiImageImportParam:
        """Create NIfTI image import parameters.

        Args:
            first_slice: Index of the first slice to import.
            last_slice: Index of the last slice to import (included). Defaults to
             ``first_slice`` (single slice).

        Returns:
            NIfTI image import parameters.
        """
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected NIfTI image parameters: {names}")
        param = NiftiImageImportParam()
        param.first_slice = first_slice
        param.last_slice = first_slice if last_slice is None else last_slice
        return param


@dataclasses.dataclass(frozen=True)
class NiftiImageInfo:
    """Header information of a NIfTI file, returned by :func:`get_nifti_image_info`.

    Attributes:
        filename: File name.
        version: NIfTI format version (1 or 2).
        shape: Dimensions of the file, ``(nx, ny)`` or ``(nx, ny, nz)``.
        data_dtype: Data type stored in the file.
        scaled: True if the values are scaled (``scl_slope``, ``scl_inter``).
        image_dtype: Data type of the imported images.
        spatial_unit: Spatial unit declared by the file, empty if unknown.
    """

    filename: str
    version: int
    shape: tuple[int, ...]
    data_dtype: str
    scaled: bool
    image_dtype: str
    spatial_unit: str

    @property
    def width(self) -> int:
        """Number of columns of the imported images (``nx``)."""
        return self.shape[0]

    @property
    def height(self) -> int:
        """Number of rows of the imported images (``ny``)."""
        return self.shape[1]

    @property
    def slice_count(self) -> int:
        """Number of slices along the third axis (1 for a 2D image)."""
        return self.shape[2] if len(self.shape) == 3 else 1

    @property
    def is_volume(self) -> bool:
        """True if a slice selection is required to read the file."""
        return self.slice_count > 1

    @property
    def slice_nbytes(self) -> int:
        """Memory size of one imported slice, in bytes."""
        return self.width * self.height * np.dtype(self.image_dtype).itemsize


@dataclasses.dataclass(frozen=True)
class NiftiCalibration:
    """Local calibration of the imported slices.

    Attributes:
        dx: Pixel size along X.
        dy: Pixel size along Y.
        unit: Unit of the coordinates.
        mode: ``"affine"`` (affine transform), ``"pixdim"`` (voxel sizes of the
         header, without any transform) or ``"pixel"`` (pixel units).
        reason: Reason why the slices are displayed in pixels, empty otherwise.
        affine: Affine transform of the file, None if it is invalid.
        affine_source: ``"sform"``, ``"qform"`` or ``"fallback"``.
    """

    dx: float
    dy: float
    unit: str
    mode: str
    reason: str
    affine: np.ndarray | None
    affine_source: str


def import_nibabel() -> Any:
    """Import NiBabel.

    Returns:
        NiBabel module.

    Raises:
        ImportError: If NiBabel is not installed.
    """
    try:
        import nibabel  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(NIBABEL_REQUIRED) from exc
    return nibabel


def check_shape(filename: str, shape: tuple[int, ...]) -> None:
    """Check that the dimensions of a NIfTI file are supported.

    Args:
        filename: File name.
        shape: Dimensions of the file.

    Raises:
        ValueError: If the file is not a non-empty 2D image or 3D volume.
    """
    name = osp.basename(filename)
    if len(shape) not in (2, 3):
        raise ValueError(
            f"NIfTI file {name} has {len(shape)} dimensions {tuple(shape)}: only "
            "2D images (nx, ny) and 3D volumes (nx, ny, nz) are supported"
        )
    if min(shape) <= 0:
        raise ValueError(f"NIfTI file {name} is empty: dimensions {tuple(shape)}")


def check_dtype(filename: str, dtype: np.dtype) -> None:
    """Check that a NIfTI data type holds real scalar values.

    Args:
        filename: File name.
        dtype: Data type stored in the file.

    Raises:
        ValueError: If the data type is complex, RGB, RGBA or structured.
    """
    if dtype.fields is not None or dtype.kind not in "iuf":
        raise ValueError(
            f"NIfTI file {osp.basename(filename)} holds {dtype} values: only real "
            "scalar values are supported (complex, RGB and RGBA images are not)"
        )


def get_image_dtype(dtype: np.dtype) -> np.dtype:
    """Return the data type of the images holding values of a given type.

    Args:
        dtype: Data type of the values read by NiBabel.

    Returns:
        Supported image data type.

    Raises:
        ValueError: If the data type cannot be converted.
    """
    dtype = dtype.newbyteorder("=")
    if any(dtype == np.dtype(valid) for valid in ImageObj.VALID_DTYPES):
        return dtype
    if dtype in CONVERTED_DTYPES:
        return CONVERTED_DTYPES[dtype]
    raise ValueError(f"NIfTI values of type {dtype} are not supported")


def to_image_array(data: np.ndarray, filename: str, index: int) -> np.ndarray:
    """Return slice values in a supported image data type, without any loss.

    Args:
        data: Slice values read by NiBabel, with axes ``(i, j)``.
        filename: File name.
        index: Slice index.

    Returns:
        Array with axes ``(j, i)``, in native byte order.

    Raises:
        ValueError: If the values cannot be represented exactly.
    """
    dtype = get_image_dtype(data.dtype)
    converted = data.astype(dtype)
    if dtype != data.dtype.newbyteorder("="):
        # Non-finite values are kept: only finite values may be altered
        if not np.array_equal(
            converted.astype(data.dtype), data, equal_nan=dtype.kind == "f"
        ):
            raise ValueError(
                f"Values of slice {index} of NIfTI file {osp.basename(filename)} "
                f"({data.dtype}) cannot be converted to {dtype} without loss"
            )
    return np.ascontiguousarray(converted.T)


@contextlib.contextmanager
def open_nifti(filename: str) -> Iterator[tuple[int, nib.Nifti1Image]]:
    """Open a single-file NIfTI image, keeping one file object open.

    Voxels are read on demand through the data proxy of the image, from the file
    object closed when leaving the context (no memory mapping).

    Args:
        filename: File name.

    Yields:
        NIfTI version (1 or 2) and image.

    Raises:
        ImportError: If NiBabel is not installed.
        ValueError: If the file is not a valid single-file NIfTI image.
    """
    nibabel = import_nibabel()
    # pylint: disable=import-outside-toplevel
    from nibabel.fileholders import FileHolder
    from nibabel.openers import ImageOpener

    name = osp.basename(filename)
    with ImageOpener(filename) as opener:
        try:
            header_bytes = opener.read(HEADER_PREFIX_SIZE)
            version = None
            if len(header_bytes) == HEADER_PREFIX_SIZE:
                for byteorder in ("little", "big"):
                    size = int.from_bytes(header_bytes[:4], byteorder)
                    version = HEADER_SIZES.get(size, version)
            if version is None:
                raise ValueError("unknown header size")
            # NiBabel normalizes the magic string: check it in the file
            magic = header_bytes[344:347] if version == 1 else header_bytes[4:7]
            if magic == f"ni{version}".encode():
                raise ValueError(
                    "only single-file NIfTI images are supported (separate "
                    ".hdr/.img files are not)"
                )
            if magic != f"n+{version}".encode():
                raise ValueError(f"invalid magic string {magic!r}")
            opener.seek(0)
            klass = nibabel.Nifti1Image if version == 1 else nibabel.Nifti2Image
            img = klass.from_file_map(
                {"image": FileHolder(fileobj=opener.fobj)}, mmap=False
            )
        except (FileNotFoundError, PermissionError):
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise ValueError(f"Invalid NIfTI file {name}: {exc}") from exc
        yield version, img


def get_spatial_unit(header: nib.Nifti1Header) -> str:
    """Return the spatial unit declared by a NIfTI header.

    Args:
        header: NIfTI header.

    Returns:
        Unit, empty if unknown.
    """
    return SPATIAL_UNITS.get(header.get_xyzt_units()[0], "")


def get_scaled_dtype(img: nib.Nifti1Image) -> tuple[bool, np.dtype]:
    """Return whether the values are scaled, and the type of the values read.

    Args:
        img: NIfTI image.

    Returns:
        True if the values are scaled, and the data type of the scaled values.
    """
    # pylint: disable=import-outside-toplevel
    from nibabel.volumeutils import apply_read_scaling

    dtype = img.header.get_data_dtype()
    slope, inter = img.dataobj.slope, img.dataobj.inter
    if slope == 1 and inter == 0:
        return False, dtype
    sample = apply_read_scaling(
        np.zeros(1, dtype), np.asanyarray(slope), np.asanyarray(inter)
    )
    return True, sample.dtype


def get_nifti_image_info(filename: str) -> NiftiImageInfo:
    """Return the information needed to select the slices of a NIfTI file.

    The header is read without materializing any voxel. Opening a compressed file
    nevertheless decompresses its beginning.

    Args:
        filename: File name (``*.nii`` or ``*.nii.gz``).

    Returns:
        Header information.

    Raises:
        ImportError: If NiBabel is not installed.
        ValueError: If the file is invalid or holds unsupported data.
    """
    with open_nifti(filename) as (version, img):
        shape = tuple(int(size) for size in img.shape)
        check_shape(filename, shape)
        data_dtype = img.header.get_data_dtype()
        check_dtype(filename, data_dtype)
        scaled, scaled_dtype = get_scaled_dtype(img)
        return NiftiImageInfo(
            filename=filename,
            version=version,
            shape=shape,
            data_dtype=str(data_dtype.newbyteorder("=")),
            scaled=scaled,
            image_dtype=str(get_image_dtype(scaled_dtype)),
            spatial_unit=get_spatial_unit(img.header),
        )


def is_valid_affine(affine: np.ndarray) -> bool:
    """Return True if an affine transform is finite and not degenerate.

    Args:
        affine: 4x4 affine transform.

    Returns:
        True if the affine transform is valid.
    """
    if not np.all(np.isfinite(affine)):
        return False
    matrix = affine[:3, :3]
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0):
        return False
    return abs(np.linalg.det(matrix / norms)) > ORTHOGONALITY_TOLERANCE


def compute_calibration(img: nib.Nifti1Image) -> NiftiCalibration:
    """Compute the local calibration of the slices of a NIfTI image.

    The affine transform follows the NiBabel priority: sform, qform, then the
    fallback transform built from the voxel sizes.

    Args:
        img: NIfTI image.

    Returns:
        Calibration of the slices.
    """
    header = img.header
    sform_code = int(header["sform_code"])
    qform_code = int(header["qform_code"])
    if sform_code > 0:
        source, affine = "sform", header.get_sform()
    elif qform_code > 0:
        source, affine = "qform", header.get_qform()
    else:
        source, affine = "fallback", header.get_best_affine()
    unit = get_spatial_unit(header)

    def pixels(reason: str, valid_affine: np.ndarray | None) -> NiftiCalibration:
        return NiftiCalibration(
            1.0, 1.0, "pixel", "pixel", reason, valid_affine, source
        )

    if not is_valid_affine(affine):
        return pixels(f"invalid {source} transform", None)
    if not unit:
        return pixels("unknown spatial unit", affine)
    if source == "fallback":
        dx, dy = (float(size) for size in header["pixdim"][1:3])
        if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0):
            return pixels("invalid voxel sizes", affine)
        return NiftiCalibration(dx, dy, unit, "pixdim", "", affine, source)
    xaxis, yaxis = affine[:3, 0], affine[:3, 1]
    dx, dy = (float(np.linalg.norm(axis)) for axis in (xaxis, yaxis))
    if abs(np.dot(xaxis, yaxis)) / (dx * dy) > ORTHOGONALITY_TOLERANCE:
        return pixels("sheared slice plane", affine)
    return NiftiCalibration(dx, dy, unit, "affine", "", affine, source)


def create_metadata(
    img: nib.Nifti1Image,
    version: int,
    shape: tuple[int, ...],
    index: int,
    calibration: NiftiCalibration,
) -> dict[str, Any]:
    """Return the import provenance of a slice, with JSON-compatible values.

    Args:
        img: NIfTI image.
        version: NIfTI format version.
        shape: Dimensions of the file.
        index: Slice index along the third axis.
        calibration: Calibration of the slices.

    Returns:
        Import metadata.
    """
    header = img.header
    metadata: dict[str, Any] = {
        "version": version,
        "shape": list(shape),
        "slice_index": index,
        "data_dtype": str(header.get_data_dtype().newbyteorder("=")),
        "spatial_unit": get_spatial_unit(header),
        "sform_code": int(header["sform_code"]),
        "qform_code": int(header["qform_code"]),
        "affine_source": calibration.affine_source,
        "calibration": calibration.mode,
    }
    if calibration.affine is not None:
        metadata["affine"] = [
            [float(value) for value in row] for row in calibration.affine
        ]
    if calibration.reason:
        metadata["calibration_reason"] = calibration.reason
    return metadata


class NiftiImageFormat(ImageFormatBase):
    """Object representing a single-file NIfTI-1 or NIfTI-2 image file."""

    FORMAT_INFO = FormatInfo(
        name="NIfTI",
        extensions="*.nii *.nii.gz",
        readable=True,
        writeable=False,
        requires=["nibabel"],
    )

    @staticmethod
    def get_slice_range(
        filename: str, shape: tuple[int, ...], param: NiftiImageImportParam | None
    ) -> tuple[int, int]:
        """Validate and return the slice range to read.

        Args:
            filename: File name.
            shape: Dimensions of the file.
            param: Slice selection.

        Returns:
            First and last slice indices (included).

        Raises:
            ValueError: If the selection is missing or invalid.
        """
        name = osp.basename(filename)
        count = shape[2] if len(shape) == 3 else 1
        if param is None:
            if count > 1:
                raise ValueError(
                    f"NIfTI file {name} is a volume of {count} slices: a slice "
                    "selection (NiftiImageImportParam) is required"
                )
            return 0, 0
        first, last = param.first_slice, param.last_slice
        if not (
            isinstance(first, (int, np.integer))
            and isinstance(last, (int, np.integer))
            and 0 <= first <= last < count
        ):
            raise ValueError(
                f"Invalid slice selection [{first}, {last}] for NIfTI file {name}: "
                f"indices must satisfy 0 <= first <= last <= {count - 1}"
            )
        return int(first), int(last)

    def read(
        self,
        filename: str,
        worker: CallbackWorkerProtocol | None = None,
        *,
        param: gds.DataSet | None = None,
    ) -> list[ImageObj]:
        """Read a 2D NIfTI image, or the selected slices of a 3D volume.

        Slice ``k`` of the file gives an image whose ``data[j, i]`` is the voxel
        ``(i, j, k)``. Returns an empty list if the worker is cancelled.

        Args:
            filename: File name.
            worker: Callback worker object.
            param: Slice selection, required for a 3D volume.

        Returns:
            One image per selected slice, by increasing index.

        Raises:
            TypeError: If ``param`` is not a NIfTI image import parameter set.
            ValueError: If the file, its dimensions, its data or the selection are
             invalid.
        """
        if param is not None and not isinstance(param, NiftiImageImportParam):
            raise TypeError(
                "NIfTI image reading requires NiftiImageImportParam, "
                f"got {type(param).__name__}"
            )

        def canceled() -> bool:
            return worker is not None and worker.was_canceled()

        name = osp.basename(filename)
        with open_nifti(filename) as (version, img):
            shape = tuple(int(size) for size in img.shape)
            check_shape(filename, shape)
            first, last = self.get_slice_range(filename, shape, param)
            check_dtype(filename, img.header.get_data_dtype())
            calibration = compute_calibration(img)
            is_volume = len(shape) == 3 and shape[2] > 1
            images: list[ImageObj] = []
            for index in range(first, last + 1):
                if canceled():
                    return []
                try:
                    data = (
                        img.dataobj[:, :, index]
                        if len(shape) == 3
                        else img.dataobj[:, :]
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    raise ValueError(
                        f"Cannot read slice {index} of NIfTI file {name}: {exc}"
                    ) from exc
                title = f"{name} [k={index}]" if is_volume else name
                metadata = {
                    "source": filename,
                    NIFTI_METADATA_KEY: create_metadata(
                        img, version, shape, index, calibration
                    ),
                }
                unit = calibration.unit
                image = create_image(
                    title,
                    to_image_array(np.asarray(data), filename, index),
                    metadata=metadata,
                    units=(unit, unit, ""),
                )
                image.set_uniform_coords(calibration.dx, calibration.dy)
                images.append(image)
                if worker is not None:
                    worker.set_progress((index - first + 1) / (last - first + 1))
            if canceled():
                return []
        return images

    def write(self, filename: str, obj: ImageObj) -> None:
        """Reject NIfTI writing."""
        raise UnsupportedFormatOperationError(f"Writing to {filename} is not supported")
