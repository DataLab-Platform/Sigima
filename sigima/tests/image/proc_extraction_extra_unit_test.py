# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :mod:`sigima.proc.image.extraction`.

Covers:

* ``average_profile`` boundary handling: swapped row/col bounds and
  ``-1`` default sentinel values for both vertical and horizontal
  directions.
* ``line_profile`` vertical / horizontal direction branches.
* ``segment_profile`` basic execution.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import ImageObj
from sigima.proc.image.extraction import (
    AverageProfileParam,
    LineProfileParam,
    SegmentProfileParam,
    average_profile,
    line_profile,
    segment_profile,
)


def _make_image(shape: tuple[int, int] = (8, 10)) -> ImageObj:
    """Build an ``ImageObj`` with ``arange`` data and unit pixel coordinates."""
    img = ImageObj()
    img.data = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
    img.set_uniform_coords(1.0, 1.0, 0.0, 0.0)
    return img


@pytest.mark.parametrize("direction", ["horizontal", "vertical"])
def test_average_profile_swapped_bounds(direction: str) -> None:
    """``average_profile`` must transparently swap inverted bounds (e.g.
    ``row1=5, row2=2``) so the user can pass them in either order without
    getting an empty profile, regardless of direction."""
    img = _make_image(shape=(8, 10))
    p = AverageProfileParam.create(direction=direction, row1=5, row2=2, col1=8, col2=1)
    sig = average_profile(img, p)
    assert p.row1 < p.row2
    assert p.col1 < p.col2
    assert sig.y.size > 0


@pytest.mark.parametrize(
    "direction, expected_axis",
    [("horizontal", 1), ("vertical", 0)],
)
def test_average_profile_default_bounds_minus_one(
    direction: str, expected_axis: int
) -> None:
    """The sentinel value ``-1`` for ``row2`` / ``col2`` is resolved to the
    last valid index of the image, covering the whole array by default,
    for both directions."""
    img = _make_image(shape=(4, 5))
    p = AverageProfileParam.create(
        direction=direction, row1=0, row2=-1, col1=0, col2=-1
    )
    sig = average_profile(img, p)
    assert p.row2 == img.data.shape[0] - 1
    assert p.col2 == img.data.shape[1] - 1
    assert sig.y.size == img.data.shape[expected_axis]


@pytest.mark.parametrize(
    "direction, kwargs, expected_axis",
    [
        ("vertical", {"col": 2}, 0),
        ("horizontal", {"row": 3}, 1),
    ],
)
def test_line_profile_directions(
    direction: str, kwargs: dict, expected_axis: int
) -> None:
    """Line profile yields exactly ``height``/``width`` samples (one per
    row/column) depending on the direction."""
    img = _make_image()
    p = LineProfileParam.create(direction=direction, **kwargs)
    sig = line_profile(img, p)
    assert sig.y.size == img.data.shape[expected_axis]


def test_segment_profile_basic() -> None:
    """Smoke test: ``segment_profile`` returns a non-empty signal for an
    arbitrary diagonal segment inside the image."""
    img = _make_image()
    p = SegmentProfileParam.create(row1=0, col1=0, row2=5, col2=5)
    sig = segment_profile(img, p)
    assert sig.y.size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
