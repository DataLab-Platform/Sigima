# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Extraction computation module
-----------------------------

This module provides functions to extract sub-regions
and intensity profiles from images.

Main features include:

- Extraction of regions of interest (ROIs)
- Extraction of line, segment, average, and radial intensity profiles

These functions are useful for isolating specific image zones and for analyzing signal
intensity along defined paths.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

from typing import Callable

import guidata.dataset as gds
import numpy as np
from numpy import ma

import sigima.tools.image
from sigima.config import _
from sigima.objects.image import ImageObj, ImageROI, RectangularROI, ROI2DParam
from sigima.objects.signal import SignalObj
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import dst_1_to_1, dst_1_to_1_signal

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "AverageProfileParam",
    "LineProfileParam",
    "ROIGridParam",
    "RadialProfileParam",
    "SegmentProfileParam",
    "average_profile",
    "extract_roi",
    "extract_rois",
    "generate_image_grid_roi",
    "line_profile",
    "radial_profile",
    "segment_profile",
]


@computation_function()
def extract_rois(src: ImageObj, params: list[ROI2DParam]) -> ImageObj:
    """Extract multiple regions of interest from data

    Args:
        src: input image object
        params: list of ROI parameters

    Returns:
        Output image object
    """
    # Initialize ix0, iy0 with maximum values:
    iy0, ix0 = iymax, ixmax = src.data.shape
    # Initialize ix1, iy1 with minimum values:
    iy1, ix1 = iymin, ixmin = 0, 0
    for p in params:
        x0i, y0i, x1i, y1i = p.get_bounding_box_indices(src)
        ix0, iy0, ix1, iy1 = min(ix0, x0i), min(iy0, y0i), max(ix1, x1i), max(iy1, y1i)
    ix0, iy0 = max(ix0, ixmin), max(iy0, iymin)
    ix1, iy1 = min(ix1, ixmax), min(iy1, iymax)

    suffix = None
    if len(params) == 1:
        p = params[0]
        suffix = p.get_suffix()
    dst = dst_1_to_1(src, "extract_rois", suffix)
    if src.is_uniform_coords:
        dst.set_uniform_coords(
            dst.dx, dst.dy, dst.x0 + ix0 * src.dx, dst.y0 + iy0 * src.dy
        )
    else:
        dst.set_coords(src.xcoords[iy0:iy1], src.ycoords[ix0:ix1])
    dst.roi = None

    src2 = src.copy()
    src2.roi = ImageROI.from_params(src2, params)
    src2.data[src2.maskdata] = 0
    dst.data = src2.data[iy0:iy1, ix0:ix1]
    return dst


@computation_function()
def extract_roi(src: ImageObj, p: ROI2DParam) -> ImageObj:
    """Extract single ROI

    Args:
        src: input image object
        p: ROI parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "extract_roi", p.get_suffix())
    dst.data = p.get_data(src).copy()
    dst.roi = p.get_extracted_roi(src)
    x0, y0, _x1, _y1 = p.get_bounding_box_physical()
    if src.is_uniform_coords:
        dst.set_uniform_coords(dst.dx, dst.dy, dst.x0 + x0, dst.y0 + y0)
    else:
        dst.set_coords(src.xcoords + x0, src.ycoords + y0)
    return dst


class Direction(gds.LabeledEnum):
    """Direction choice"""

    INCREASING = "increasing", _("increasing")
    DECREASING = "decreasing", _("decreasing")


class ROIGridParam(gds.DataSet):
    """ROI Grid parameters"""

    # optional Python-level hook, no Qt
    on_geometry_changed: Callable | None = None

    # pylint: disable=unused-argument
    def geometry_changed(self, item, value) -> None:
        """Notify host (if any) that geometry changed."""
        if callable(self.on_geometry_changed):
            self.on_geometry_changed()  # pylint: disable=not-callable

    _b_group0 = gds.BeginGroup(_("Geometry"))
    ny = gds.IntItem(f"N<sub>y</sub> ({_('rows')})", default=3, nonzero=True).set_prop(
        "display", callback=geometry_changed
    )
    nx = (
        gds.IntItem(f"N<sub>x</sub> ({_('columns')})", default=3, nonzero=True)
        .set_prop("display", callback=geometry_changed)
        .set_pos(col=1)
    )
    xtranslation = gds.IntItem(
        _("X translation"),
        default=50,
        min=0,
        max=100,
        unit="%",
        slider=True,
    ).set_prop("display", callback=geometry_changed)
    ytranslation = gds.IntItem(
        _("Y translation"),
        default=50,
        min=0,
        max=100,
        unit="%",
        slider=True,
    ).set_prop("display", callback=geometry_changed)
    xsize = gds.IntItem(
        f"X size ({_('column size')})",
        default=50,
        min=0,
        max=100,
        unit="%",
        slider=True,
    ).set_prop("display", callback=geometry_changed)
    ysize = gds.IntItem(
        f"Y size ({_('row size')})",
        default=50,
        min=0,
        max=100,
        unit="%",
        slider=True,
    ).set_prop("display", callback=geometry_changed)
    xstep = gds.IntItem(
        f"X step ({_('column spacing')})",
        default=100,
        min=1,
        max=200,
        unit="%",
        slider=True,
        help=_(
            "Horizontal spacing between ROI centers, as a percentage of the "
            "automatically computed cell width (100% = evenly distributed grid)"
        ),
    ).set_prop("display", callback=geometry_changed)
    ystep = gds.IntItem(
        f"Y step ({_('row spacing')})",
        default=100,
        min=1,
        max=200,
        unit="%",
        slider=True,
        help=_(
            "Vertical spacing between ROI centers, as a percentage of the "
            "automatically computed cell height (100% = evenly distributed grid)"
        ),
    ).set_prop("display", callback=geometry_changed)
    _e_group0 = gds.EndGroup(_("Geometry"))
    _b_group1 = gds.BeginGroup(_("ROI titles"))
    base_name = gds.StringItem(_("Base name"), default="ROI").set_prop(
        "display", callback=geometry_changed
    )
    name_pattern = gds.StringItem(
        _("Name pattern"), default="{base}({r},{c})"
    ).set_prop("display", callback=geometry_changed)
    xdirection = gds.ChoiceItem(_("X direction"), Direction).set_prop(
        "display", callback=geometry_changed
    )
    ydirection = (
        gds.ChoiceItem(_("Y direction"), Direction)
        .set_prop("display", callback=geometry_changed)
        .set_pos(col=1)
    )
    _e_group1 = gds.EndGroup(_("ROI titles"))


def generate_image_grid_roi(src: ImageObj, p: ROIGridParam) -> ImageROI:
    """Create a grid of rectangular ROIs from an image object.

    Args:
        obj: The image object to create the ROI for.
        p: ROIGridParam object containing the grid parameters.

    Returns:
        The created ROI object.
    """
    dx_cell = src.width / p.nx
    dy_cell = src.height / p.ny
    dx = dx_cell * p.xsize / 100.0
    dy = dy_cell * p.ysize / 100.0
    # Apply step multipliers to cell spacing
    dx_step = dx_cell * p.xstep / 100.0
    dy_step = dy_cell * p.ystep / 100.0
    xtrans = src.width * (p.xtranslation - 50.0) / 100.0
    ytrans = src.height * (p.ytranslation - 50.0) / 100.0
    lbl_rows = range(p.ny)
    if p.ydirection == Direction.DECREASING:
        lbl_rows = range(p.ny - 1, -1, -1)
    lbl_cols = range(p.nx)
    if p.xdirection == Direction.DECREASING:
        lbl_cols = range(p.nx - 1, -1, -1)
    ptn: str = p.name_pattern
    roi = ImageROI()
    for ir in range(p.ny):
        for ic in range(p.nx):
            x0 = src.x0 + (ic + 0.5) * dx_step + xtrans - 0.5 * dx
            y0 = src.y0 + (ir + 0.5) * dy_step + ytrans - 0.5 * dy
            nir, nic = lbl_rows[ir], lbl_cols[ic]
            try:
                title = ptn.format(base=p.base_name, r=nir + 1, c=nic + 1)
            except Exception:  # pylint: disable=broad-except
                title = f"ROI({nir + 1},{nic + 1})"
            roi.add_roi(RectangularROI([x0, y0, dx, dy], indices=False, title=title))
    return roi


class LineProfileParam(gds.DataSet):
    """Horizontal or vertical profile parameters"""

    _prop = gds.GetAttrProp("direction")
    _directions = (("horizontal", _("horizontal")), ("vertical", _("vertical")))
    direction = gds.ChoiceItem(_("Direction"), _directions, radio=True).set_prop(
        "display", store=_prop
    )
    row = gds.IntItem(_("Row"), default=0, min=0).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "horizontal")
    )
    col = gds.IntItem(_("Column"), default=0, min=0).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "vertical")
    )


@computation_function()
def line_profile(src: ImageObj, p: LineProfileParam) -> SignalObj:
    """Compute horizontal or vertical profile

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the profile
    """
    data = src.get_masked_view()
    p.row = min(p.row, data.shape[0] - 1)
    p.col = min(p.col, data.shape[1] - 1)
    if p.direction == "horizontal":
        suffix, shape_index, pdata = f"row={p.row}", 1, data[p.row, :]
    else:
        suffix, shape_index, pdata = f"col={p.col}", 0, data[:, p.col]
    pdata: ma.MaskedArray
    x = np.arange(data.shape[shape_index])[~pdata.mask]
    y = np.array(pdata, dtype=float)[~pdata.mask]
    dst = dst_1_to_1_signal(src, "profile", suffix)
    dst.set_xydata(x, y)
    return dst


class SegmentProfileParam(gds.DataSet):
    """Segment profile parameters"""

    row1 = gds.IntItem(_("Start row"), default=0, min=0)
    col1 = gds.IntItem(_("Start column"), default=0, min=0)
    row2 = gds.IntItem(_("End row"), default=0, min=0)
    col2 = gds.IntItem(_("End column"), default=0, min=0)


def csline(data: np.ndarray, row0, col0, row1, col1) -> tuple[np.ndarray, np.ndarray]:
    """Return intensity profile of data along a line

    Args:
        data: 2D array
        row0, col0: start point
        row1, col1: end point
    """
    # Keep coordinates inside the image
    row0 = max(0, min(row0, data.shape[0] - 1))
    col0 = max(0, min(col0, data.shape[1] - 1))
    row1 = max(0, min(row1, data.shape[0] - 1))
    col1 = max(0, min(col1, data.shape[1] - 1))
    # Keep coordinates in the right order
    row0, row1 = min(row0, row1), max(row0, row1)
    col0, col1 = min(col0, col1), max(col0, col1)
    # Extract the line
    line = np.zeros((2, max(abs(row1 - row0), abs(col1 - col0)) + 1), dtype=int)
    line[0, :] = np.linspace(row0, row1, line.shape[1]).astype(int)
    line[1, :] = np.linspace(col0, col1, line.shape[1]).astype(int)
    # Interpolate the line
    y = np.ma.array(data[line[0], line[1]], float).filled(np.nan)
    x = np.arange(y.size)
    return x, y


@computation_function()
def segment_profile(src: ImageObj, p: SegmentProfileParam) -> SignalObj:
    """Compute segment profile

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the segment profile
    """
    data = src.get_masked_view()
    p.row1 = min(p.row1, data.shape[0] - 1)
    p.col1 = min(p.col1, data.shape[1] - 1)
    p.row2 = min(p.row2, data.shape[0] - 1)
    p.col2 = min(p.col2, data.shape[1] - 1)
    suffix = f"({p.row1}, {p.col1})-({p.row2}, {p.col2})"
    x, y = csline(data, p.row1, p.col1, p.row2, p.col2)
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]  # Remove NaN values
    dst = dst_1_to_1_signal(src, "segment_profile", suffix)
    dst.set_xydata(np.array(x, dtype=float), np.array(y, dtype=float))
    return dst


class AverageProfileParam(gds.DataSet):
    """Average horizontal or vertical profile parameters"""

    _directions = (("horizontal", _("horizontal")), ("vertical", _("vertical")))
    direction = gds.ChoiceItem(_("Direction"), _directions, radio=True)
    _hgroup_begin = gds.BeginGroup(_("Profile rectangular area"))
    row1 = gds.IntItem(_("Row 1"), default=0, min=0)
    row2 = gds.IntItem(_("Row 2"), default=-1, min=-1)
    col1 = gds.IntItem(_("Column 1"), default=0, min=0)
    col2 = gds.IntItem(_("Column 2"), default=-1, min=-1)
    _hgroup_end = gds.EndGroup(_("Profile rectangular area"))


@computation_function()
def average_profile(src: ImageObj, p: AverageProfileParam) -> SignalObj:
    """Compute horizontal or vertical average profile

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the average profile
    """
    data = src.get_masked_view()
    if p.row2 == -1:
        p.row2 = data.shape[0] - 1
    if p.col2 == -1:
        p.col2 = data.shape[1] - 1
    if p.row1 > p.row2:
        p.row1, p.row2 = p.row2, p.row1
    if p.col1 > p.col2:
        p.col1, p.col2 = p.col2, p.col1
    p.row1 = min(p.row1, data.shape[0] - 1)
    p.row2 = min(p.row2, data.shape[0] - 1)
    p.col1 = min(p.col1, data.shape[1] - 1)
    p.col2 = min(p.col2, data.shape[1] - 1)
    suffix = f"{p.direction}, rows=[{p.row1}, {p.row2}], cols=[{p.col1}, {p.col2}]"
    if p.direction == "horizontal":
        x, axis = np.arange(p.col1, p.col2 + 1), 0
    else:
        x, axis = np.arange(p.row1, p.row2 + 1), 1
    y = ma.mean(data[p.row1 : p.row2 + 1, p.col1 : p.col2 + 1], axis=axis)
    dst = dst_1_to_1_signal(src, "average_profile", suffix)
    dst.set_xydata(x, y)
    return dst


class RadialProfileParam(gds.DataSet):
    """Radial profile parameters"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__obj: ImageObj | None = None

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from image"""
        self.__obj = obj
        self.x0 = obj.xc
        self.y0 = obj.yc

    def choice_callback(self, item, value):  # pylint: disable=unused-argument
        """Callback for choice item"""
        if self.__obj is None:
            return
        if value == "centroid":
            self.y0, self.x0 = sigima.tools.image.get_centroid_fourier(
                self.__obj.get_masked_view()
            )
        elif value == "center":
            self.x0, self.y0 = self.__obj.xc, self.__obj.yc

    _prop = gds.GetAttrProp("center")
    center = gds.ChoiceItem(
        _("Center position"),
        (
            ("centroid", _("Image centroid")),
            ("center", _("Image center")),
            ("user", _("User-defined")),
        ),
        default="centroid",
    ).set_prop("display", store=_prop, callback=choice_callback)

    _func_prop = gds.FuncProp(_prop, lambda x: x == "user")
    _xyl = "<sub>" + _("Center") + "</sub>"
    x0 = gds.FloatItem(f"X{_xyl}", default=0.0, unit="pixel").set_prop(
        "display", active=_func_prop
    )
    y0 = gds.FloatItem(f"Y{_xyl}", default=0.0, unit="pixel").set_prop(
        "display", active=_func_prop
    )


@computation_function()
def radial_profile(src: ImageObj, p: RadialProfileParam) -> SignalObj:
    """Compute radial profile around the centroid
    with :py:func:`sigima.tools.image.get_radial_profile`

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the radial profile
    """
    data = src.get_masked_view()
    if p.center == "centroid":
        y0, x0 = sigima.tools.image.get_centroid_fourier(data)
    elif p.center == "center":
        x0, y0 = src.xc, src.yc
    else:
        x0, y0 = p.x0, p.y0
    suffix = f"center=({x0:.3f}, {y0:.3f})"
    dst = dst_1_to_1_signal(src, "radial_profile", suffix)
    x, y = sigima.tools.image.get_radial_profile(data, (x0, y0))
    dst.set_xydata(x, y)
    return dst
