# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Testing that color image import preserves the original value range.

Regression test: opening an RGB JPEG/PNG used to normalize pixel values to
[0, 1] (because of skimage's ``as_gray=True`` option), which made the imported
values unusable for quantitative analysis. The reader now converts color images
to grayscale while keeping the source integer range (matching ImageJ's default).
"""

from __future__ import annotations

import os.path as osp

import numpy as np
import skimage.io

from sigima.io.image.formats import ClassicsImageFormat
from sigima.tests.helpers import WorkdirRestoringTempDir


def test_rgb_png_preserves_value_range() -> None:
    """An RGB PNG should be read back in the 0-255 range, not normalized."""
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "rgb.png")
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[..., 0] = 30
        rgb[..., 1] = 150
        rgb[..., 2] = 240
        skimage.io.imsave(path, rgb, check_contrast=False)

        data = ClassicsImageFormat.read_data(path)

        assert data.ndim == 2, "Color image should be collapsed to grayscale"
        assert data.dtype == np.uint8, "Source integer dtype should be preserved"
        assert data.max() > 1, "Values must not be normalized to [0, 1]"
        # Unweighted mean of the channels, rounded (ImageJ default behavior)
        assert data[0, 0] == round((30 + 150 + 240) / 3)


def test_grayscale_encoded_as_rgb_is_exact() -> None:
    """A grayscale image stored as RGB (R==G==B) must be read back exactly."""
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "gray_as_rgb.png")
        value = 200
        rgb = np.full((8, 8, 3), value, dtype=np.uint8)
        skimage.io.imsave(path, rgb, check_contrast=False)

        data = ClassicsImageFormat.read_data(path)

        assert data.dtype == np.uint8
        assert np.all(data == value), "Identical channels must yield exact values"


def test_rgba_png_drops_alpha() -> None:
    """An RGBA PNG should ignore the alpha channel when converting to gray."""
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "rgba.png")
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., :3] = 200
        rgba[..., 3] = 10  # Nearly transparent, must not affect the result
        skimage.io.imsave(path, rgba, check_contrast=False)

        data = ClassicsImageFormat.read_data(path)

        assert data.ndim == 2
        assert np.all(data == 200), "Alpha channel must be ignored"


def test_grayscale_png_is_unchanged() -> None:
    """A genuine single-channel grayscale image keeps its native range."""
    with WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "gray.png")
        gray = np.arange(64, dtype=np.uint8).reshape(8, 8) * 3
        skimage.io.imsave(path, gray, check_contrast=False)

        data = ClassicsImageFormat.read_data(path)

        assert data.dtype == np.uint8
        assert np.array_equal(data, gray)


if __name__ == "__main__":
    test_rgb_png_preserves_value_range()
    test_grayscale_encoded_as_rgb_is_exact()
    test_rgba_png_drops_alpha()
    test_grayscale_png_is_unchanged()
    print("All tests passed.")
