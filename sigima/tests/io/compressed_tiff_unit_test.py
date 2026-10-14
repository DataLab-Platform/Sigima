# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Testing compressed TIFF import and export.

Regression test: TIFF files compressed with LZW, JPEG or CCITT could not be opened
("requires the 'imagecodecs' package"), because ``tifffile`` only embeds decoders
for a few compression schemes. The same limitation affected format-aware TIFF
export with LZW or JPEG compression. ``imagecodecs`` is now a Sigima dependency.
See https://github.com/DataLab-Platform/DataLab/issues/381.
"""

from __future__ import annotations

import os.path as osp

import numpy as np
import pytest
import tifffile
from PIL import Image

from sigima.io import ImageExportParam, read_image, write_image
from sigima.objects import create_image
from sigima.tests.helpers import WorkdirRestoringTempDir

# CCITT decoders are only provided by imagecodecs >= 2026 (Python >= 3.11)
CCITT_SUPPORTED = tifffile.COMPRESSION.CCITT_T6 in tifffile.TIFF.DECOMPRESSORS


def make_gradient(shape: tuple[int, int], dtype: type = np.uint8) -> np.ndarray:
    """Return a smooth gradient image, suitable for lossy compression checks."""
    rows, cols = np.indices(shape)
    maxval = np.iinfo(dtype).max
    return ((rows + cols) * maxval // (shape[0] + shape[1] - 2)).astype(dtype)


@pytest.mark.parametrize("compression", ["tiff_lzw", "tiff_deflate", "packbits"])
def test_read_lossless_compressed_grayscale_tiff(compression: str) -> None:
    """Grayscale TIFF files with lossless compression are read back exactly."""
    data = make_gradient((32, 48))
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, f"{compression}.tif")
        Image.fromarray(data).save(path, compression=compression)

        image = read_image(path)

    assert image.data.dtype == np.uint8
    np.testing.assert_array_equal(image.data, data)


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_read_lzw_color_tiff(mode: str) -> None:
    """LZW-compressed color TIFF files are read and converted to grayscale."""
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., 0] = 30
    rgba[..., 1] = 150
    rgba[..., 2] = 240
    rgba[..., 3] = 255
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, f"{mode}.tif")
        Image.fromarray(rgba).convert(mode).save(path, compression="tiff_lzw")

        image = read_image(path)

    assert image.data.shape == (16, 16)
    assert np.all(image.data == round((30 + 150 + 240) / 3))


def test_read_jpeg_compressed_tiff() -> None:
    """JPEG-compressed TIFF files are read with a bounded compression error."""
    data = make_gradient((32, 32))
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "jpeg.tif")
        Image.fromarray(data).save(path, compression="jpeg", quality=95)

        image = read_image(path)

    assert image.data.shape == data.shape
    assert np.max(np.abs(image.data.astype(int) - data.astype(int))) <= 8


@pytest.mark.skipif(not CCITT_SUPPORTED, reason="CCITT codecs are not available")
def test_read_ccitt_group4_tiff() -> None:
    """Bilevel TIFF files compressed with CCITT Group 4 are read back exactly."""
    data = np.zeros((32, 32), dtype=bool)
    data[8:24, 4:28] = True
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "group4.tif")
        Image.fromarray(data).save(path, compression="group4")

        image = read_image(path)

    np.testing.assert_array_equal(image.data.astype(bool), data)


@pytest.mark.parametrize(
    ("compression", "predictor"),
    [("lzw", "none"), ("lzw", "horizontal"), ("zstd", "none"), ("deflate", "none")],
)
def test_export_lossless_compressed_tiff(compression: str, predictor: str) -> None:
    """Format-aware TIFF export with lossless compression round-trips exactly."""
    data = make_gradient((32, 48), np.uint16)
    param = ImageExportParam()
    param.format_options = {"compression": compression, "predictor": predictor}
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "export.tif")
        write_image(path, create_image("Source", data), param)

        image = read_image(path)

    assert image.data.dtype == np.uint16
    np.testing.assert_array_equal(image.data, data)


def test_export_jpeg_compressed_tiff() -> None:
    """Format-aware TIFF export with JPEG compression is readable."""
    data = make_gradient((32, 32))
    param = ImageExportParam()
    param.format_options = {"compression": "jpeg", "compression_level": 95}
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "export.tif")
        write_image(path, create_image("Source", data), param)

        image = read_image(path)

    assert image.data.dtype == np.uint8
    assert np.max(np.abs(image.data.astype(int) - data.astype(int))) <= 8
