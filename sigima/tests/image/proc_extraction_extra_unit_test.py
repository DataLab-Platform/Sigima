# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :mod:`sigima.proc.image.extraction`.

Covers:

* ``average_profile`` boundary handling: swapped corners and default
  (``None``) bounds for both vertical and horizontal directions.
* ``line_profile`` vertical / horizontal direction branches.
* ``segment_profile`` basic execution.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import copy

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
    """``average_profile`` accepts the corners of its rectangle in any order, so
    that the user can pass them in either order without getting an empty profile,
    regardless of direction."""
    img = _make_image(shape=(8, 10))
    p = AverageProfileParam.create(direction=direction, x1=8, y1=5, x2=1, y2=2)
    sig = average_profile(img, p)
    ordered = AverageProfileParam.create(direction=direction, x1=1, y1=2, x2=8, y2=5)
    expected = average_profile(img, ordered)
    assert np.array_equal(sig.x, expected.x) and np.array_equal(sig.y, expected.y)
    # The parameters are not modified: they apply unchanged to other images
    assert (p.x1, p.y1, p.x2, p.y2) == (8, 5, 1, 2)


@pytest.mark.parametrize(
    "direction, expected_axis",
    [("horizontal", 1), ("vertical", 0)],
)
def test_average_profile_default_bounds_none(
    direction: str, expected_axis: int
) -> None:
    """Bounds left to None are the edges of the image, covering the whole array by
    default, for both directions."""
    img = _make_image(shape=(4, 5))
    p = AverageProfileParam.create(direction=direction)
    sig = average_profile(img, p)
    assert (p.x1, p.y1, p.x2, p.y2) == (None, None, None, None)
    assert sig.y.size == img.data.shape[expected_axis]


@pytest.mark.parametrize(
    "direction, kwargs, expected_axis",
    [
        ("vertical", {"x": 2}, 0),
        ("horizontal", {"y": 3}, 1),
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
    p = SegmentProfileParam.create(x1=0, y1=0, x2=5, y2=5)
    sig = segment_profile(img, p)
    assert sig.y.size > 0


def _values(p) -> dict:
    """Return the item values of a parameter set."""
    return {item.get_name(): getattr(p, item.get_name()) for item in p.get_items()}


@pytest.mark.parametrize(
    "func, p",
    [
        (line_profile, LineProfileParam.create(direction="horizontal", y=3)),
        (line_profile, LineProfileParam.create(direction="vertical", x=4)),
        (segment_profile, SegmentProfileParam.create(x1=1, y1=1, x2=4, y2=3)),
        (
            average_profile,
            AverageProfileParam.create(direction="horizontal", y1=3, y2=1),
        ),
    ],
)
def test_profile_parameters_apply_unchanged_to_other_images(func, p) -> None:
    """Parameters are not modified by a computation: they apply unchanged to
    another image."""
    small, large = _make_image(shape=(4, 5)), _make_image(shape=(8, 10))
    before = _values(p)
    expected = func(large, copy.deepcopy(p))
    func(small, p)
    assert _values(p) == before
    assert np.array_equal(func(large, p).y, expected.y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
