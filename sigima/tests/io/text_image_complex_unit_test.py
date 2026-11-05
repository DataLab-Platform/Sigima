# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Round trip of complex images through text files (Issue #61)."""

from __future__ import annotations

import numpy as np
import pytest

from sigima.io import ImageExportParam, read_image, write_image
from sigima.objects import create_image

COMPLEX_DATA = np.array(
    [[0.0, 1.0 + 1.0j, -2.5 - 3.0j], [4.0j, -1.0, 1e-300 - 1e300j]],
    dtype=np.complex128,
)


@pytest.mark.parametrize("ext", [".txt", ".asc", ".csv"])
def test_complex_text_round_trip(tmp_path, ext: str) -> None:
    """A complex image written to a text file is read back unchanged."""
    path = str(tmp_path / f"complex{ext}")
    write_image(path, create_image("Complex image", COMPLEX_DATA))
    image = read_image(path)
    assert image.data.dtype == np.complex128
    np.testing.assert_array_equal(image.data, COMPLEX_DATA)


@pytest.mark.parametrize("ext", [".txt", ".asc", ".csv"])
@pytest.mark.parametrize("delimiter", ["whitespace", "tab", "comma", "semicolon"])
def test_complex_text_round_trip_with_options(
    tmp_path, ext: str, delimiter: str
) -> None:
    """The format-aware export path is read back with any delimiter."""
    if ext == ".csv" and delimiter == "whitespace":
        pytest.skip("CSV files do not accept whitespace delimiters")
    path = str(tmp_path / f"complex{ext}")
    param = ImageExportParam()
    param.format_options = {"delimiter": delimiter, "precision": 18}
    write_image(path, create_image("Complex image", COMPLEX_DATA), param)
    image = read_image(path)
    assert image.data.dtype == np.complex128
    np.testing.assert_array_equal(image.data, COMPLEX_DATA)


def test_complex_text_round_trip_low_precision(tmp_path) -> None:
    """Values written with a reduced precision are read back within it."""
    path = str(tmp_path / "complex.txt")
    param = ImageExportParam()
    param.format_options = {"precision": 3}
    write_image(path, create_image("Complex image", COMPLEX_DATA), param)
    image = read_image(path)
    assert image.data.dtype == np.complex128
    np.testing.assert_allclose(image.data, COMPLEX_DATA, rtol=1e-3)


@pytest.mark.parametrize("ext", [".txt", ".csv"])
def test_real_text_round_trip_unchanged(tmp_path, ext: str) -> None:
    """Real-valued text exports are still read back as real images."""
    data = np.arange(12, dtype=np.float64).reshape(3, 4) - 5.5
    path = str(tmp_path / f"real{ext}")
    write_image(path, create_image("Real image", data))
    image = read_image(path)
    assert not np.iscomplexobj(image.data)
    np.testing.assert_array_equal(image.data, data)


def test_non_numeric_text_is_still_refused(tmp_path) -> None:
    """Text cells that are not numbers are refused, as before."""
    path = tmp_path / "text.txt"
    path.write_text("a b c\nd e f\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_image(str(path))
