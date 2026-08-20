# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test creation of PlotPy ROI items from an :class:`sigima.objects.ImageObj`.

Regression test for the inverse rectangular ROI rendering bug: the PlotPy ROI
item must be built from the rectangle's own shape, not from the extraction
bounding box (which, for an inverse ROI, covers the whole image).
"""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
from sigima.objects import ImageObj

pytest.importorskip("plotpy")


def _make_image() -> ImageObj:
    """Return a 100x100 float64 image with a unit coordinate system."""
    obj = ImageObj(title="Test")
    obj.data = np.zeros((100, 100), dtype=np.float64)
    obj.set_uniform_coords(1.0, 1.0, 0.0, 0.0)
    return obj


class _RecordingBuilder:
    """Minimal ``make`` stub recording the rectangle coordinates."""

    def __init__(self) -> None:
        self.rect_coords: tuple[float, float, float, float] | None = None

    def annotated_rectangle(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        _title: str = "",
    ) -> object:
        """Record the rectangle coordinates and return a sentinel item."""
        self.rect_coords = (x0, y0, x1, y1)
        return object()


@pytest.mark.parametrize("inverse", [False, True])
def test_rectangular_roi_item_uses_shape_not_extraction_box(
    monkeypatch: pytest.MonkeyPatch, inverse: bool
) -> None:
    """Render a rectangular ROI as its own shape, even when inverse."""
    from sigima.viz import viz_plotpy

    builder = _RecordingBuilder()
    monkeypatch.setattr(viz_plotpy, "make", builder)

    obj = _make_image()
    obj.roi = sigima.objects.create_image_roi(
        "rectangle", [20.0, 30.0, 40.0, 25.0], inverse=inverse
    )

    create_items = getattr(viz_plotpy, "__create_image_roi_items")
    items = create_items(obj)

    assert len(items) == 1
    assert builder.rect_coords == (20.0, 30.0, 60.0, 55.0)
