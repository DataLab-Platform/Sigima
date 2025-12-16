# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image offset correction unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests import guiutils
from sigima.tests.data import create_noisy_gaussian_image


@pytest.mark.gui
def test_image_offset_correction_interactive() -> None:
    """Image offset correction interactive test."""
    with guiutils.lazy_qt_app_context(force=True):
        # pylint: disable=import-outside-toplevel
        from plotpy.builder import make
        from plotpy.items import RectangleShape
        from plotpy.tools import RectangleTool
        from plotpy.widgets.selectdialog import SelectDialog, select_with_shape_tool

        from sigima.tests import vistools

        i1 = create_noisy_gaussian_image()
        shape: RectangleShape = select_with_shape_tool(
            None,
            RectangleTool,
            make.image(i1.data, interpolation="nearest", eliminate_outliers=1.0),
            "Select background area",
            tooldialogclass=SelectDialog,
        )
        if shape is not None:
            param = sigima.objects.ROI2DParam()
            # pylint: disable=unbalanced-tuple-unpacking
            ix0, iy0, ix1, iy1 = i1.physical_to_indices(shape.get_rect())
            param.x0, param.y0, param.dx, param.dy = ix0, iy0, ix1 - ix0, iy1 - iy0
            i2 = sigima.proc.image.offset_correction(i1, param)
            i3 = sigima.proc.image.clip(i2, sigima.params.ClipParam.create(lower=0))
            vistools.view_images_side_by_side(
                [i1, i3],
                titles=["Original image", "Corrected image"],
                title="Image offset correction and thresholding",
            )


@pytest.mark.validation
def test_image_offset_correction() -> None:
    """Image offset correction validation test."""
    i1 = create_noisy_gaussian_image()
    p = sigima.objects.ROI2DParam.create(x0=0, y0=0, dx=10, dy=10)
    i2 = sigima.proc.image.offset_correction(i1, p)

    # Check that the offset correction has been applied
    ix0, iy0 = int(p.x0), int(p.y0)
    ix1, iy1 = int(p.x0 + p.dx), int(p.y0 + p.dy)
    offset = np.mean(i1.data[iy0:iy1, ix0:ix1])
    assert np.allclose(i2.data, i1.data - offset), "Offset correction failed"


def test_image_offset_correction_lut_range() -> None:
    """Image offset correction LUT range regression test.

    Verify that the LUT range is NOT copied from the original image when processing.
    This is a regression test for the bug where the original image's LUT range
    (zscalemin/zscalemax) was incorrectly copied to the result image, causing
    incorrect visualization when the data range changes significantly.
    """
    i1 = create_noisy_gaussian_image()

    # Simulate user setting a specific LUT range on the original image
    # (as would happen when viewing in DataLab)
    i1.zscalemin = 50.0
    i1.zscalemax = 200.0

    p = sigima.objects.ROI2DParam.create(x0=0, y0=0, dx=10, dy=10)
    i2 = sigima.proc.image.offset_correction(i1, p)

    # The result image should NOT have the original LUT range copied
    # because the data values have changed significantly
    assert i2.zscalemin is None, (
        f"LUT range should not be copied from original: zscalemin={i2.zscalemin}"
    )
    assert i2.zscalemax is None, (
        f"LUT range should not be copied from original: zscalemax={i2.zscalemax}"
    )


if __name__ == "__main__":
    test_image_offset_correction_interactive()
    test_image_offset_correction()
    test_image_offset_correction_lut_range()
