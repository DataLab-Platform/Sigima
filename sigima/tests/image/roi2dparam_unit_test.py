# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI image parameters unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
import pytest

from sigima.objects import ImageObj, ROI2DParam
from sigima.proc.image.extraction import extract_roi
from sigima.tests import guiutils
from sigima.tests.env import execenv


def __create_roi_2d_parameters() -> gds.DataSetGroup:
    """Create a group of ROI parameters."""
    p_circ = ROI2DParam("Circular")
    p_circ.geometry = "circle"
    p_circ.xc, p_circ.yc, p_circ.r = 100, 200, 50
    p_rect = ROI2DParam("Rectangular")
    p_rect.geometry = "rectangle"
    p_rect.x0, p_rect.y0, p_rect.dx, p_rect.dy = 50, 150, 150, 250
    p_poly = ROI2DParam("Polygonal")
    p_poly.geometry = "polygon"
    p_poly.points = np.array([50.0, 150.0, 150.0, 150.0, 150.0, 250.0, 50.0, 250.0])
    params = [p_circ, p_rect, p_poly]
    return gds.DataSetGroup(params, title="ROI Parameters")


def test_roi_2d_param_unit():
    """ROI parameters unit test."""
    group = __create_roi_2d_parameters()
    for param in group.datasets:
        execenv.print(param)


@pytest.mark.gui
def test_roi_2d_param_interactive():
    """ROI parameters interactive test."""
    with guiutils.lazy_qt_app_context(force=True):
        group = __create_roi_2d_parameters()
        if group.edit():
            for param in group.datasets:
                execenv.print(param)


def _create_image(rows: int = 40, cols: int = 50) -> ImageObj:
    """Create a uniform test image with 1.0 px spacing starting at origin."""
    obj = ImageObj(title="Test")
    obj.data = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    obj.set_uniform_coords(1.0, 1.0, 0.0, 0.0)
    return obj


class TestInverseROI2DParamExtraction:
    """Test ROI2DParam.get_data, get_bounding_box_indices and extract_roi
    for inverse ROIs.
    """

    # ------------------------------------------------------------------
    # get_bounding_box_indices
    # ------------------------------------------------------------------
    def test_inverse_rect_bbox_indices_is_full_image(self):
        """Inverse rectangle: bounding-box indices cover the whole image."""
        obj = _create_image()
        p = ROI2DParam()
        p.geometry = "rectangle"
        p.x0, p.y0, p.dx, p.dy = 5.0, 5.0, 20.0, 15.0
        p.inverse = True
        ix0, iy0, ix1, iy1 = p.get_bounding_box_indices(obj)
        assert (ix0, iy0) == (0, 0)
        assert (ix1, iy1) == (obj.data.shape[1], obj.data.shape[0])

    def test_normal_rect_bbox_indices_is_shape(self):
        """Normal rectangle: bounding-box indices match the rectangle coordinates."""
        obj = _create_image()
        p = ROI2DParam()
        p.geometry = "rectangle"
        p.x0, p.y0, p.dx, p.dy = 5.0, 5.0, 20.0, 15.0
        p.inverse = False
        ix0, iy0, ix1, iy1 = p.get_bounding_box_indices(obj)
        # With dx=1.0, dy=1.0, x0=0: pixel index == physical coordinate
        assert ix0 == 5 and iy0 == 5
        assert ix1 == 25 and iy1 == 20

    # ------------------------------------------------------------------
    # get_data
    # ------------------------------------------------------------------
    def test_inverse_rect_get_data_returns_full_image(self):
        """ROI2DParam.get_data with inverse=True must return the full image array."""
        obj = _create_image()
        p = ROI2DParam()
        p.geometry = "rectangle"
        p.x0, p.y0, p.dx, p.dy = 5.0, 5.0, 20.0, 15.0
        p.inverse = True
        data = p.get_data(obj)
        assert data.shape == obj.data.shape

    def test_normal_rect_get_data_returns_crop(self):
        """ROI2DParam.get_data with inverse=False must return a cropped array."""
        obj = _create_image()
        p = ROI2DParam()
        p.geometry = "rectangle"
        p.x0, p.y0, p.dx, p.dy = 5.0, 5.0, 20.0, 15.0
        p.inverse = False
        data = p.get_data(obj)
        assert data.shape == (15, 20)

    # ------------------------------------------------------------------
    # get_extracted_roi
    # ------------------------------------------------------------------
    def test_inverse_rect_get_extracted_roi_is_not_none(self):
        """Inverse rectangle: get_extracted_roi must return the inverse ROI,
        not None.
        """
        obj = _create_image()
        p = ROI2DParam()
        p.geometry = "rectangle"
        p.x0, p.y0, p.dx, p.dy = 5.0, 5.0, 20.0, 15.0
        p.inverse = True
        roi = p.get_extracted_roi(obj)
        assert roi is not None

    def test_normal_rect_get_extracted_roi_is_none(self):
        """Normal rectangle: get_extracted_roi must return None (simple crop)."""
        obj = _create_image()
        p = ROI2DParam()
        p.geometry = "rectangle"
        p.x0, p.y0, p.dx, p.dy = 5.0, 5.0, 20.0, 15.0
        p.inverse = False
        assert p.get_extracted_roi(obj) is None

    # ------------------------------------------------------------------
    # extract_roi end-to-end
    # ------------------------------------------------------------------
    def test_extract_roi_inverse_rect_has_full_image_shape(self):
        """extract_roi with an inverse rectangle ROI must produce the full image."""
        obj = _create_image()
        p = ROI2DParam()
        p.geometry = "rectangle"
        p.x0, p.y0, p.dx, p.dy = 5.0, 5.0, 20.0, 15.0
        p.inverse = True
        dst = extract_roi(obj, p)
        assert dst.data.shape == obj.data.shape

    def test_extract_roi_inverse_rect_origin_unchanged(self):
        """extract_roi with an inverse ROI must preserve the image origin."""
        obj = _create_image()
        p = ROI2DParam()
        p.geometry = "rectangle"
        p.x0, p.y0, p.dx, p.dy = 5.0, 5.0, 20.0, 15.0
        p.inverse = True
        dst = extract_roi(obj, p)
        assert dst.x0 == obj.x0
        assert dst.y0 == obj.y0

    def test_extract_roi_normal_rect_is_cropped(self):
        """extract_roi with a normal rectangle ROI must produce a cropped image."""
        obj = _create_image()
        p = ROI2DParam()
        p.geometry = "rectangle"
        p.x0, p.y0, p.dx, p.dy = 5.0, 5.0, 20.0, 15.0
        p.inverse = False
        dst = extract_roi(obj, p)
        assert dst.data.shape == (15, 20)
        assert dst.x0 == obj.x0 + 5.0
        assert dst.y0 == obj.y0 + 5.0


if __name__ == "__main__":
    test_roi_2d_param_interactive()
