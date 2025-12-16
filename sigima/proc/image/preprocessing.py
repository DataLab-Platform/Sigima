# -*- coding: utf-8 -*-
"""
Image preprocessing functions.

This module consolidates preprocessing functions that were previously scattered
across different modules (exposure, geometry, fourier). All functions in this
module operate on high-level ImageObj objects and use parameter classes from
the sigima.proc framework.
"""

from __future__ import annotations

import guidata.dataset as gds
import numpy as np

import sigima.enums
import sigima.tools.image
from sigima.config import _
from sigima.enums import PadLocation2D
from sigima.objects.image import ImageObj
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import dst_1_to_1

__all__ = [
    "BinningParam",
    "ZeroPadding2DParam",
    "binning",
    "zero_padding",
]


class BinningParam(gds.DataSet):
    """Binning parameters."""

    sx = gds.IntItem(
        _("Cluster size (X)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along X-axis."),
    )
    sy = gds.IntItem(
        _("Cluster size (Y)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along Y-axis."),
    )
    operation = gds.ChoiceItem(
        _("Operation"),
        sigima.enums.BinningOperation,
        default=sigima.enums.BinningOperation.SUM,
    )
    dtypes = ["dtype"] + ImageObj.get_valid_dtypenames()
    dtype_str = gds.ChoiceItem(
        _("Data type"),
        list(zip(dtypes, dtypes)),
        help=_("Output image data type."),
    )
    change_pixel_size = gds.BoolItem(
        _("Change pixel size"),
        default=True,
        help=_(
            "If checked, pixel size is updated according to binning factors. "
            "Users who prefer to work with pixel coordinates may want to uncheck this."
        ),
    )


@computation_function()
def binning(src: ImageObj, p: BinningParam) -> ImageObj:
    """Binning: image pixel binning (or aggregation).

    Depending on the algorithm, the input image may be cropped to fit an integer
    number of blocks.

    Args:
        src: source image
        p: parameters

    Returns:
        Output image

    Raises:
        ValueError: if source image has non-uniform coordinates
    """
    if not src.is_uniform_coords:
        raise ValueError("Binning only works with images having uniform coordinates")
    # Create destination image
    dst = dst_1_to_1(
        src,
        "binning",
        f"{p.sx}x{p.sy},{p.operation},change_pixel_size={p.change_pixel_size}",
    )
    dst.data = sigima.tools.image.binning(
        src.data,
        sx=p.sx,
        sy=p.sy,
        operation=p.operation,
        dtype=None if p.dtype_str == "dtype" else p.dtype_str,
    )
    if p.change_pixel_size:
        if not np.isnan(src.dx) and not np.isnan(src.dy):
            # Update coordinates with new pixel spacing
            new_dx = src.dx * p.sx
            new_dy = src.dy * p.sy
            dst.set_uniform_coords(new_dx, new_dy, src.x0, src.y0)
    return dst


class ZeroPadding2DParam(gds.DataSet):
    """Zero padding parameters for 2D images"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__obj: ImageObj | None = None

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from image"""
        self.__obj = obj
        self.choice_callback(None, self.strategy)

    def choice_callback(self, item, value):  # pylint: disable=unused-argument
        """Callback to update padding values"""
        if self.__obj is None:
            return
        rows, cols = self.__obj.data.shape
        if value == "next_pow2":
            self.rows = 2 ** int(np.ceil(np.log2(rows))) - rows
            self.cols = 2 ** int(np.ceil(np.log2(cols))) - cols
        elif value == "multiple_of_64":
            self.rows = (64 - rows % 64) if rows % 64 != 0 else 0
            self.cols = (64 - cols % 64) if cols % 64 != 0 else 0

    strategies = ("next_pow2", "multiple_of_64", "custom")
    _prop = gds.GetAttrProp("strategy")
    strategy = gds.ChoiceItem(
        _("Padding strategy"), zip(strategies, strategies), default=strategies[-1]
    ).set_prop("display", store=_prop, callback=choice_callback)

    _func_prop = gds.FuncProp(_prop, lambda x: x == "custom")
    rows = gds.IntItem(_("Rows to add"), min=0, default=0).set_prop(
        "display", active=_func_prop
    )
    cols = gds.IntItem(_("Columns to add"), min=0, default=0).set_prop(
        "display", active=_func_prop
    )
    position = gds.ChoiceItem(
        _("Padding position"), PadLocation2D, default=PadLocation2D.BOTTOM_RIGHT
    )


@computation_function()
def zero_padding(
    src: ImageObj,
    p: ZeroPadding2DParam | None = None,
) -> ImageObj:
    """Zero-padding: add zeros to image borders.

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    if p is None:
        p = ZeroPadding2DParam.create()

    if p.strategy == "custom":
        suffix = f"rows={p.rows}, cols={p.cols}"
    else:
        suffix = f"strategy={p.strategy}"
    suffix += f", position={p.position}"

    dst = dst_1_to_1(src, "zero_padding", suffix)
    result = sigima.tools.image.zero_padding(
        src.data,
        rows=p.rows,
        cols=p.cols,
        position=p.position,
    )
    dst.data = result
    return dst
