# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for NIfTI image import."""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp

import numpy as np
import pytest
from guidata.config import ValidationMode, temporary_validation_mode

from sigima.io import (
    ImageIORegistry,
    NiftiImageImportParam,
    RawImageImportParam,
    get_nifti_image_info,
    read_image,
    read_images,
)
from sigima.io.base import FormatInfo, IOAction
from sigima.io.image.base import ImageFormatBase
from sigima.io.image.nifti import NiftiImageFormat
from sigima.objects.image import ImageObj

nib = pytest.importorskip("nibabel")
ArrayProxy = pytest.importorskip("nibabel.arrayproxy").ArrayProxy


class RecordingWorker:
    """Record progress and optionally cancel after a completed slice."""

    def __init__(self, *, canceled: bool = False, cancel_after: int | None = None):
        self.canceled = canceled
        self.cancel_after = cancel_after
        self.progress_values: list[float] = []

    def set_progress(self, value: float) -> None:
        """Record progress and trigger cancellation when requested."""
        self.progress_values.append(value)
        if self.cancel_after == len(self.progress_values):
            self.canceled = True

    def was_canceled(self) -> bool:
        """Return whether cancellation was requested."""
        return self.canceled


def encoded_volume(shape=(4, 3, 5), dtype=np.int16) -> np.ndarray:
    """Return a volume whose voxel (i, j, k) is 100*i + 10*j + k."""
    i, j, k = np.meshgrid(*(np.arange(size) for size in shape), indexing="ij")
    return (100 * i + 10 * j + k).astype(dtype)


def save_nifti(
    filename: str,
    data: np.ndarray,
    *,
    version: int = 1,
    affine: np.ndarray | None = None,
    endianness: str | None = None,
    units: str = "mm",
) -> str:
    """Save a single-file NIfTI image (sform code 2 when an affine is given)."""
    klass = nib.Nifti1Image if version == 1 else nib.Nifti2Image
    header = None if endianness is None else klass.header_class(endianness=endianness)
    affine = np.diag([0.5, 2.0, 3.0, 1.0]) if affine is None else affine
    img = klass(data, affine, header=header, dtype=data.dtype)
    img.header.set_xyzt_units(units)
    nib.save(img, filename)
    return filename


def patch_header(filename: str, **fields) -> None:
    """Change header fields of an uncompressed NIfTI file, keeping its data."""
    header = nib.load(filename).header.copy()
    for name, value in fields.items():
        header[name] = value
    with open(filename, "r+b") as file:
        header.write_to(file)


def check_encoded_slice(image: ImageObj, k: int, shape=(4, 3)) -> None:
    """Check that image data[j, i] is the encoded voxel (i, j, k)."""
    nx, ny = shape
    assert image.data.shape == (ny, nx)
    for j in range(ny):
        for i in range(nx):
            assert image.data[j, i] == 100 * i + 10 * j + k


# Registry -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename", ["volume.nii", "volume.nii.gz", "DIR.d/VOLUME.NII.GZ", "a.b.Nii"]
)
def test_nifti_format_is_selected_by_suffix(filename: str) -> None:
    """NIfTI files are recognized by their (compound) suffix, whatever the case."""
    fmt = ImageIORegistry.get_format(filename, IOAction.LOAD)
    assert isinstance(fmt, NiftiImageFormat)
    assert fmt.extlist == ["nii", "nii.gz"]


@pytest.mark.parametrize("filename", ["volume.nii.gz.bak", ".nii.gz", "volume.gz"])
def test_nifti_format_rejects_other_suffixes(filename: str) -> None:
    """Only names ending with a NIfTI suffix after a non-empty stem are NIfTI."""
    with pytest.raises(NotImplementedError):
        ImageIORegistry.get_format(filename, IOAction.LOAD)


def test_nifti_format_is_read_only() -> None:
    """NIfTI is listed in the read filters only."""
    assert "NIfTI (*.nii *.nii.gz)" in ImageIORegistry.get_read_filters().splitlines()
    assert "*.nii" in ImageIORegistry.get_read_filters().splitlines()[0]
    assert "nii" not in ImageIORegistry.get_write_filters()
    with pytest.raises(NotImplementedError):
        ImageIORegistry.get_format("volume.nii", IOAction.SAVE)


def test_longest_suffix_wins_over_registration_order(monkeypatch) -> None:
    """A ``*.gz`` format does not take ``*.nii.gz`` files, and keeps ``*.gz``."""
    monkeypatch.setattr(
        ImageIORegistry, "_io_format_instances", ImageIORegistry.get_formats().copy()
    )

    def read(self, filename, worker=None, *, param=None):  # pylint: disable=unused-argument
        """Read nothing."""
        return []

    gz_format_class = type(
        "GzImageFormat",
        (ImageFormatBase,),
        {
            "FORMAT_INFO": FormatInfo(
                name="Gzip test image", extensions="*.gz", readable=True
            ),
            "read": read,
            "write": lambda self, filename, obj: None,
        },
    )
    assert isinstance(
        ImageIORegistry.get_format("volume.nii.gz", IOAction.LOAD), NiftiImageFormat
    )
    assert isinstance(
        ImageIORegistry.get_format("archive.GZ", IOAction.LOAD), gz_format_class
    )


# Header information and parameters --------------------------------------------


def test_nifti_image_info(tmp_path) -> None:
    """Header information is read without voxel data."""
    filename = save_nifti(str(tmp_path / "volume.nii.gz"), encoded_volume())
    info = get_nifti_image_info(filename)
    assert info.version == 1
    assert info.shape == (4, 3, 5)
    assert (info.width, info.height, info.slice_count) == (4, 3, 5)
    assert info.is_volume
    assert info.data_dtype == info.image_dtype == "int16"
    assert not info.scaled
    assert info.spatial_unit == "mm"
    assert info.slice_nbytes == 4 * 3 * 2


def test_nifti_import_param_create() -> None:
    """Parameters default to a single slice and reject unknown arguments."""
    param = NiftiImageImportParam.create(first_slice=3)
    assert (param.first_slice, param.last_slice) == (3, 3)
    param = NiftiImageImportParam.create(first_slice=1, last_slice=4)
    assert (param.first_slice, param.last_slice) == (1, 4)
    with pytest.raises(TypeError, match="width"):
        NiftiImageImportParam.create(width=3)


# Pixels -----------------------------------------------------------------------


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("extension", [".nii", ".nii.gz"])
@pytest.mark.parametrize("endianness", ["<", ">"])
def test_read_slice_range(tmp_path, version, extension, endianness) -> None:
    """Selected slices are returned in order, with voxel (i, j, k) at [j, i]."""
    filename = save_nifti(
        str(tmp_path / f"volume{extension}"),
        encoded_volume(),
        version=version,
        endianness=endianness,
    )
    param = NiftiImageImportParam.create(first_slice=1, last_slice=3)
    images = read_images(filename, param=param)
    assert [image.title for image in images] == [
        f"volume{extension} [k={k}]" for k in (1, 2, 3)
    ]
    for k, image in zip((1, 2, 3), images):
        assert image.data.dtype == np.dtype(np.int16)
        assert image.data.dtype.isnative
        check_encoded_slice(image, k)
        metadata = image.metadata["nifti"]
        assert metadata["version"] == version
        assert metadata["slice_index"] == k
        assert metadata["shape"] == [4, 3, 5]
        assert image.metadata["source"] == filename
        json.dumps(metadata)
    assert read_image(filename, param=param).title == images[0].title


def test_read_2d_image_and_single_slice_volume(tmp_path) -> None:
    """2D images and single-slice volumes need no selection."""
    data2d = encoded_volume((4, 3, 1))[:, :, 0]
    filename = save_nifti(str(tmp_path / "image.nii"), data2d)
    (image,) = read_images(filename)
    assert image.title == "image.nii"
    check_encoded_slice(image, 0)
    filename = save_nifti(str(tmp_path / "slice.nii"), encoded_volume((4, 3, 1)))
    (image,) = read_images(filename)
    assert image.title == "slice.nii"
    assert not get_nifti_image_info(filename).is_volume


def test_read_scaled_values(tmp_path) -> None:
    """scl_slope and scl_inter are applied, giving floating-point values."""
    filename = save_nifti(str(tmp_path / "scaled.nii"), encoded_volume())
    patch_header(filename, scl_slope=0.5, scl_inter=10.0)
    info = get_nifti_image_info(filename)
    assert info.scaled
    assert (info.data_dtype, info.image_dtype) == ("int16", "float64")
    (image,) = read_images(filename, param=NiftiImageImportParam.create(2))
    assert image.data.dtype == np.dtype(np.float64)
    expected = 0.5 * encoded_volume()[:, :, 2].T.astype(float) + 10.0
    np.testing.assert_array_equal(image.data, expected)


@pytest.mark.parametrize(
    "dtype, image_dtype",
    [
        (np.uint8, np.uint8),
        (np.int8, np.int16),
        (np.uint32, np.float64),
        (np.int64, np.float64),
        (np.uint64, np.float64),
        (np.float64, np.float64),
    ],
)
def test_exact_dtype_conversions(tmp_path, dtype, image_dtype) -> None:
    """Unsupported data types are converted when no value changes."""
    data = encoded_volume((2, 3, 2)).astype(dtype)
    filename = save_nifti(str(tmp_path / "typed.nii"), data)
    assert get_nifti_image_info(filename).image_dtype == np.dtype(image_dtype).name
    (image,) = read_images(filename, param=NiftiImageImportParam.create(1))
    assert image.data.dtype == np.dtype(image_dtype)
    check_encoded_slice(image, 1, shape=(2, 3))


def test_lossy_conversion_is_refused(tmp_path) -> None:
    """Integers not representable exactly as float64 are refused."""
    data = np.full((2, 3, 1), 2**53 + 1, dtype=np.int64)
    filename = save_nifti(str(tmp_path / "large.nii"), data)
    with pytest.raises(ValueError, match="without loss"):
        read_images(filename)


def test_non_finite_values_are_kept(tmp_path) -> None:
    """NaN and infinite values are neither replaced nor clipped."""
    data = np.zeros((3, 2), dtype=np.float32)
    data[0, 0], data[1, 0], data[2, 1] = np.nan, np.inf, -np.inf
    (image,) = read_images(save_nifti(str(tmp_path / "nan.nii"), data))
    assert np.isnan(image.data[0, 0])
    assert image.data[0, 1] == np.inf and image.data[1, 2] == -np.inf


def test_complex_and_rgb_images_are_refused(tmp_path) -> None:
    """Only real scalar values are supported."""
    data = encoded_volume((2, 3, 1))[:, :, 0].astype(np.complex64)
    filename = save_nifti(str(tmp_path / "complex.nii"), data)
    with pytest.raises(ValueError, match="real scalar"):
        get_nifti_image_info(filename)
    with pytest.raises(ValueError, match="real scalar"):
        read_images(filename)
    rgb = np.zeros((2, 3), dtype=[("R", "u1"), ("G", "u1"), ("B", "u1")])
    filename = save_nifti(str(tmp_path / "rgb.nii"), rgb)
    with pytest.raises(ValueError, match="real scalar"):
        read_images(filename)


# Geometry ---------------------------------------------------------------------


def read_calibrated_slice(filename: str) -> ImageObj:
    """Read the first slice of a file."""
    return read_images(filename, param=NiftiImageImportParam.create(0))[0]


def rotation_z(angle: float) -> np.ndarray:
    """Return a 3x3 rotation matrix around Z."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([0.5, 2.0, 3.0]),
        np.diag([-0.5, 2.0, 3.0]),
        rotation_z(np.pi / 6) @ np.diag([0.5, 2.0, 3.0]),
    ],
    ids=["anisotropic", "reflection", "oblique"],
)
def test_affine_calibration(tmp_path, matrix) -> None:
    """Pixel sizes are the norms of the in-plane columns of the affine."""
    affine = np.eye(4)
    affine[:3, :3], affine[:3, 3] = matrix, [10.0, -20.0, 5.0]
    filename = save_nifti(str(tmp_path / "v.nii"), encoded_volume(), affine=affine)
    image = read_calibrated_slice(filename)
    assert image.dx == pytest.approx(0.5) and image.dy == pytest.approx(2.0)
    assert (image.x0, image.y0) == (0.0, 0.0)
    assert (image.xunit, image.yunit) == ("mm", "mm")
    metadata = image.metadata["nifti"]
    assert metadata["calibration"] == "affine"
    assert metadata["affine_source"] == "sform"
    np.testing.assert_allclose(metadata["affine"], affine)
    assert "calibration_reason" not in metadata


def test_qform_and_fallback_calibration(tmp_path) -> None:
    """The qform is used without sform, then the header voxel sizes."""
    filename = str(tmp_path / "qform.nii")
    img = nib.Nifti1Image(encoded_volume(), np.diag([0.5, 2.0, 3.0, 1.0]))
    img.set_sform(None, code=0)
    img.set_qform(np.diag([0.25, 4.0, 3.0, 1.0]), code=1)
    img.header.set_xyzt_units("micron")
    nib.save(img, filename)
    image = read_calibrated_slice(filename)
    assert (image.dx, image.dy, image.xunit) == (0.25, 4.0, "µm")
    assert image.metadata["nifti"]["affine_source"] == "qform"
    patch_header(filename, qform_code=0)
    image = read_calibrated_slice(filename)
    assert (image.dx, image.dy, image.xunit) == (0.25, 4.0, "µm")
    metadata = image.metadata["nifti"]
    assert (metadata["affine_source"], metadata["calibration"]) == (
        "fallback",
        "pixdim",
    )


@pytest.mark.parametrize(
    "sform, units, reason",
    [
        (np.array([[1, 0.5, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]]), "mm",
         "sheared slice plane"),
        (np.diag([0.5, 0.0, 3.0, 1.0]), "mm", "invalid sform transform"),
        (np.diag([0.5, np.nan, 3.0, 1.0]), "mm", "invalid sform transform"),
        (np.diag([0.5, 2.0, 3.0, 1.0]), "unknown", "unknown spatial unit"),
    ],
    ids=["sheared", "degenerate", "non-finite", "unknown-unit"],
)  # fmt: skip
def test_pixel_calibration(tmp_path, sform, units, reason) -> None:
    """Images are in pixels when the calibration is not reliable."""
    filename = save_nifti(str(tmp_path / "v.nii"), encoded_volume(), units=units)
    patch_header(filename, srow_x=sform[0], srow_y=sform[1], srow_z=sform[2])
    image = read_calibrated_slice(filename)
    assert (image.dx, image.dy, image.x0, image.y0) == (1.0, 1.0, 0.0, 0.0)
    assert image.xunit == image.yunit == "pixel"
    metadata = image.metadata["nifti"]
    assert metadata["calibration"] == "pixel"
    assert metadata["calibration_reason"] == reason
    assert ("affine" in metadata) == reason.startswith(("sheared", "unknown"))
    check_encoded_slice(image, 0)
    json.dumps(metadata)


# Errors -----------------------------------------------------------------------


def test_volume_requires_valid_selection(tmp_path) -> None:
    """A volume requires a selection within its slices."""
    filename = save_nifti(str(tmp_path / "volume.nii"), encoded_volume())
    with pytest.raises(ValueError, match="slice selection"):
        read_images(filename)
    for first, last in ((0, 5), (3, 2), (-1, 1)):
        param = NiftiImageImportParam.create(first_slice=0)
        with temporary_validation_mode(ValidationMode.DISABLED):
            param.first_slice, param.last_slice = first, last
        with pytest.raises(ValueError, match="Invalid slice selection"):
            read_images(filename, param=param)
    raw_param = RawImageImportParam.create(width=4, height=3)
    with pytest.raises(TypeError, match="NiftiImageImportParam"):
        read_images(filename, param=raw_param)


def test_unsupported_dimensions_are_refused(tmp_path) -> None:
    """Files with four dimensions are refused."""
    data = np.zeros((2, 3, 4, 2), dtype=np.int16)
    filename = save_nifti(str(tmp_path / "time.nii"), data)
    with pytest.raises(ValueError, match="4 dimensions"):
        get_nifti_image_info(filename)
    with pytest.raises(ValueError, match="4 dimensions"):
        read_images(filename, param=NiftiImageImportParam.create(0))


def test_invalid_files_are_refused(tmp_path) -> None:
    """Invalid, truncated or paired files give explicit errors."""
    filename = str(tmp_path / "garbage.nii")
    with open(filename, "wb") as file:
        file.write(os.urandom(16) + bytes(400))
    with pytest.raises(ValueError, match="Invalid NIfTI file garbage.nii"):
        get_nifti_image_info(filename)
    filename = str(tmp_path / "garbage.nii.gz")
    with open(filename, "wb") as file:
        file.write(b"not a gzip stream" * 30)
    with pytest.raises(ValueError, match="Invalid NIfTI file garbage.nii.gz"):
        read_images(filename)
    with pytest.raises(FileNotFoundError):
        read_images(str(tmp_path / "missing.nii"))

    filename = save_nifti(str(tmp_path / "pair.nii"), encoded_volume())
    patch_header(filename, magic=b"ni1")
    with pytest.raises(ValueError, match="single-file"):
        get_nifti_image_info(filename)
    patch_header(filename, magic=b"xyz")
    with pytest.raises(ValueError, match="magic"):
        read_images(filename, param=NiftiImageImportParam.create(0))
    filename = save_nifti(str(tmp_path / "pair2.nii"), encoded_volume(), version=2)
    patch_header(filename, magic=b"ni2")
    with pytest.raises(ValueError, match="single-file"):
        get_nifti_image_info(filename)

    for name in ("truncated.nii", "truncated.nii.gz"):
        filename = save_nifti(str(tmp_path / name), encoded_volume((40, 30, 5)))
        with open(filename, "rb") as file:
            content = file.read()
        if name.endswith(".gz"):
            content = gzip.compress(gzip.decompress(content)[:3000])
        else:
            content = content[:3000]
        with open(filename, "wb") as file:
            file.write(content)
        assert get_nifti_image_info(filename).slice_count == 5
        with pytest.raises(ValueError, match="Cannot read slice 4"):
            read_images(filename, param=NiftiImageImportParam.create(4))


# Reading behavior -------------------------------------------------------------


def test_cancellation_returns_no_image(tmp_path) -> None:
    """A canceled reading returns no partial result."""
    filename = save_nifti(str(tmp_path / "volume.nii.gz"), encoded_volume())
    param = NiftiImageImportParam.create(first_slice=0, last_slice=4)
    fmt = ImageIORegistry.get_format(filename, IOAction.LOAD)
    assert fmt.read(filename, RecordingWorker(canceled=True), param=param) == []
    worker = RecordingWorker(cancel_after=2)
    assert fmt.read(filename, worker, param=param) == []
    assert worker.progress_values == [0.2, 0.4]
    worker = RecordingWorker()
    assert len(fmt.read(filename, worker, param=param)) == 5
    assert worker.progress_values == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])


@pytest.mark.parametrize("extension", [".nii", ".nii.gz"])
def test_file_is_released_after_reading(tmp_path, extension) -> None:
    """The file can be deleted after reading or a failed reading."""
    filename = save_nifti(str(tmp_path / f"volume{extension}"), encoded_volume())
    read_images(filename, param=NiftiImageImportParam.create(2))
    get_nifti_image_info(filename)
    with pytest.raises(ValueError):
        read_images(filename)
    os.remove(filename)
    assert not osp.exists(filename)


def test_only_selected_slices_are_read(tmp_path, monkeypatch) -> None:
    """The volume is never loaded as a whole."""
    filename = save_nifti(str(tmp_path / "volume.nii.gz"), encoded_volume())
    keys = []
    getitem = ArrayProxy.__getitem__

    def recording_getitem(self, key):
        keys.append(key)
        return getitem(self, key)

    def forbidden_array(self, *args, **kwargs):
        raise AssertionError("The whole volume must not be read")

    monkeypatch.setattr(ArrayProxy, "__getitem__", recording_getitem)
    monkeypatch.setattr(ArrayProxy, "__array__", forbidden_array)
    get_nifti_image_info(filename)
    assert not keys
    param = NiftiImageImportParam.create(first_slice=1, last_slice=2)
    images = read_images(filename, param=param)
    assert len(images) == 2
    assert keys == [(slice(None), slice(None), 1), (slice(None), slice(None), 2)]
