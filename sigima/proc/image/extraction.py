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
from sigima.objects.image.object import PIXEL_INDEX_TOLERANCE
from sigima.objects.signal import SignalObj
from sigima.proc.base import new_signal_result
from sigima.proc.contracts import EXACT, ReplayQualification
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import dst_1_to_1

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


@computation_function(
    operation_id="sigima.image.extract_rois",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
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
        dst.set_coords(src.xcoords[ix0:ix1].copy(), src.ycoords[iy0:iy1].copy())
    dst.roi = None

    src2 = src.copy()
    src2.roi = ImageROI.from_params(src2, params)
    src2.data[src2.maskdata] = 0
    dst.data = src2.data[iy0:iy1, ix0:ix1]
    return dst


@computation_function(
    operation_id="sigima.image.extract_roi",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
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
    # The extracted pixels keep their physical positions (and the kept ROI, which
    # is expressed in physical coordinates, stays on them)
    ix0, iy0, ix1, iy1 = p.get_bounding_box_indices(src)
    nrows, ncols = src.data.shape
    ix0, iy0 = max(0, ix0), max(0, iy0)
    ix1, iy1 = min(ncols, ix1), min(nrows, iy1)
    if src.is_uniform_coords:
        dst.set_uniform_coords(
            dst.dx, dst.dy, src.x0 + ix0 * src.dx, src.y0 + iy0 * src.dy
        )
    else:
        dst.set_coords(src.xcoords[ix0:ix1].copy(), src.ycoords[iy0:iy1].copy())
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
    ny = gds.IntItem(f"N<sub>y</sub> ({_('rows')})", default=3, min=1).set_prop(
        "display", callback=geometry_changed
    )
    nx = (
        gds.IntItem(f"N<sub>x</sub> ({_('columns')})", default=3, min=1)
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


def _cell_pixel_range(
    src: ImageObj, start: float, end: float, axis: str
) -> tuple[int, int]:
    """Return the pixels of a grid cell along an axis.

    A pixel belongs to the cell when its center lies in ``[start, end[``, so that
    adjacent cells never share a pixel, even when their common bound lies on a
    pixel center.

    Args:
        src: image object
        start: physical coordinate where the cell starts
        end: physical coordinate where the cell ends
        axis: ``"x"`` or ``"y"``

    Returns:
        ``(i0, i1)``: pixels ``i0`` to ``i1 - 1`` (not clipped to the image)
    """
    f_start, f_end = src.axis_to_float_indices([start, end], axis)
    i0 = int(np.ceil(f_start - PIXEL_INDEX_TOLERANCE))
    i1 = int(np.ceil(f_end - PIXEL_INDEX_TOLERANCE))
    return i0, max(i0, i1)


def generate_image_grid_roi(src: ImageObj, p: ROIGridParam) -> ImageROI:
    """Create a grid of rectangular ROIs from an image object.

    The grid is laid out over the image up to the outer edges of its pixels (the
    actual extent of its axes, also for non-uniform coordinates), then each cell
    is made of the pixels whose centers lie in it, from the start of the cell
    included to its end excluded: the cells of a grid with 100 % sizes and steps
    tile the pixels of the image exactly. The rectangle of each ROI goes from
    the outer edges of its pixels.

    Args:
        obj: The image object to create the ROI for.
        p: ROIGridParam object containing the grid parameters.

    Returns:
        The created ROI object.
    """
    nrows, ncols = src.data.shape
    left, right = (float(v) for v in src.axis_to_physical([-0.5, ncols - 0.5], "x"))
    top, bottom = (float(v) for v in src.axis_to_physical([-0.5, nrows - 0.5], "y"))
    width, height = right - left, bottom - top
    dx_cell = width / p.nx
    dy_cell = height / p.ny
    dx = dx_cell * p.xsize / 100.0
    dy = dy_cell * p.ysize / 100.0
    # Apply step multipliers to cell spacing
    dx_step = dx_cell * p.xstep / 100.0
    dy_step = dy_cell * p.ystep / 100.0
    xtrans = width * (p.xtranslation - 50.0) / 100.0
    ytrans = height * (p.ytranslation - 50.0) / 100.0
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
            x0 = left + (ic + 0.5) * dx_step + xtrans - 0.5 * dx
            y0 = top + (ir + 0.5) * dy_step + ytrans - 0.5 * dy
            ix0, ix1 = _cell_pixel_range(src, x0, x0 + dx, "x")
            iy0, iy1 = _cell_pixel_range(src, y0, y0 + dy, "y")
            rx0, ry0, rx1, ry1 = src.pixel_range_to_physical([ix0, iy0, ix1, iy1])
            nir, nic = lbl_rows[ir], lbl_cols[ic]
            try:
                title = ptn.format(base=p.base_name, r=nir + 1, c=nic + 1)
            except Exception:  # pylint: disable=broad-except
                title = f"ROI({nir + 1},{nic + 1})"
            roi.add_roi(
                RectangularROI(
                    [rx0, ry0, rx1 - rx0, ry1 - ry0], indices=False, title=title
                )
            )
    return roi


def _pixel_centers(src: ImageObj, axis: str) -> np.ndarray:
    """Return the physical coordinates of the pixel centers along an axis.

    Args:
        src: image object
        axis: ``"x"`` (columns) or ``"y"`` (rows)

    Returns:
        Coordinates of the pixel centers
    """
    size = src.data.shape[1 if axis == "x" else 0]
    return src.axis_to_physical(np.arange(size, dtype=float), axis)


def _nearest_pixels(
    src: ImageObj, values: np.ndarray | list[float], axis: str, name: str
) -> np.ndarray:
    """Return the indices of the pixels whose centers are the nearest to coordinates.

    Args:
        src: image object
        values: physical coordinates along the axis
        axis: ``"x"`` (columns) or ``"y"`` (rows)
        name: name of the coordinates, for error messages

    Returns:
        Pixel indices (ties go to the next pixel)

    Raises:
        ValueError: if a coordinate lies outside the image (beyond the outer edges
         of its pixels)
    """
    size = src.data.shape[1 if axis == "x" else 0]
    findices = np.atleast_1d(src.axis_to_float_indices(values, axis))
    outside = (findices < -0.5 - PIXEL_INDEX_TOLERANCE) | (
        findices > size - 0.5 + PIXEL_INDEX_TOLERANCE
    )
    if np.any(outside):
        raise ValueError(
            f"{name} = {np.atleast_1d(values)[outside][0]:g} lies outside the image"
        )
    return np.clip(np.floor(findices + 0.5), 0, size - 1).astype(int)


def _pixel_spacing(src: ImageObj) -> float:
    """Return the smallest distance between the centers of neighboring pixels.

    Args:
        src: image object

    Returns:
        Smallest pixel spacing along X and Y (uniform or non-uniform coordinates)
    """
    spacings = []
    for axis in ("x", "y"):
        centers = _pixel_centers(src, axis)
        if centers.size > 1:
            spacings.append(float(np.min(np.abs(np.diff(centers)))))
    if not spacings:
        # Single pixel: any positive step gives the same profile
        spacings = [abs(float(src.dx)) if src.is_uniform_coords else 1.0]
    return min(spacings)


def _check_same_units(src: ImageObj, name: str) -> None:
    """Check that distances can be computed in the image plane.

    Args:
        src: image object
        name: name of the computation, for error messages

    Raises:
        ValueError: if the X and Y units of the image differ
    """
    if (src.xunit or "") != (src.yunit or ""):
        raise ValueError(
            f"{name}: the X and Y units of the image differ "
            f"({src.xunit!r} and {src.yunit!r}), so that distances are undefined"
        )


def _axis_units(src: ImageObj, axis: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """Return the units and labels of a profile along an axis of the image."""
    if axis == "x":
        return (src.xunit, src.zunit), (src.xlabel, src.zlabel)
    return (src.yunit, src.zunit), (src.ylabel, src.zlabel)


class LineProfileParam(gds.DataSet):
    """Horizontal or vertical profile parameters

    The profile is extracted along the row (or column) of pixels whose center is
    the nearest to the given coordinate, in the physical units of the image.
    """

    _prop = gds.GetAttrProp("direction")
    _directions = (("horizontal", _("horizontal")), ("vertical", _("vertical")))
    direction = gds.ChoiceItem(_("Direction"), _directions, radio=True).set_prop(
        "display", store=_prop
    )
    y = gds.FloatItem(
        "Y", default=0.0, help=_("Y coordinate of the row (nearest pixel center)")
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "horizontal"))
    x = gds.FloatItem(
        "X", default=0.0, help=_("X coordinate of the column (nearest pixel center)")
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "vertical"))

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from image: profiles through the center of the image"""
        self.x, self.y = obj.xc, obj.yc


@computation_function(
    operation_id="sigima.image.line_profile",
    contract_version=2,
    qualification=ReplayQualification(contract_version=2, comparison=EXACT),
)
def line_profile(src: ImageObj, p: LineProfileParam) -> SignalObj:
    """Compute horizontal or vertical profile

    The row (or column) is the one whose pixel centers are the nearest to the
    coordinate given in the physical units of the image. The X axis of the profile
    holds the coordinates of the pixel centers, with the unit and label of the
    corresponding axis of the image.

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the profile

    Raises:
        ValueError: if the coordinate lies outside the image
    """
    data = src.get_masked_view()
    if p.direction == "horizontal":
        (row,) = _nearest_pixels(src, [p.y], "y", "Y")
        suffix, axis, pdata = f"y={p.y:g}", "x", data[row, :]
    else:
        (col,) = _nearest_pixels(src, [p.x], "x", "X")
        suffix, axis, pdata = f"x={p.x:g}", "y", data[:, col]
    keep = ~ma.getmaskarray(pdata)
    x = _pixel_centers(src, axis)[keep]
    y = np.array(pdata, dtype=float)[keep]
    units, labels = _axis_units(src, axis)
    dst = new_signal_result(src, "profile", suffix, units, labels)
    dst.set_xydata(x, y)
    return dst


class SegmentProfileParam(gds.DataSet):
    """Segment profile parameters

    The segment goes from (X1, Y1) to (X2, Y2), in the physical units of the image.
    """

    x1 = gds.FloatItem("X<sub>1</sub>", default=0.0, help=_("Start point"))
    y1 = gds.FloatItem("Y<sub>1</sub>", default=0.0, help=_("Start point")).set_pos(
        col=1
    )
    x2 = gds.FloatItem("X<sub>2</sub>", default=0.0, help=_("End point"))
    y2 = gds.FloatItem("Y<sub>2</sub>", default=0.0, help=_("End point")).set_pos(col=1)

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from image: segment along the middle row"""
        x_centers = _pixel_centers(obj, "x")
        self.x1, self.x2 = float(x_centers[0]), float(x_centers[-1])
        self.y1 = self.y2 = obj.yc


def csline(data: np.ndarray, row0, col0, row1, col1) -> tuple[np.ndarray, np.ndarray]:
    """Return intensity profile of data along a line, in pixel indices

    Args:
        data: 2D array
        row0, col0: start point
        row1, col1: end point

    Returns:
        Sample indices and values along the line, from the start point to the end
        point (nearest pixels, one sample per pixel along the longest direction)
    """
    # Keep coordinates inside the image
    row0 = max(0, min(row0, data.shape[0] - 1))
    col0 = max(0, min(col0, data.shape[1] - 1))
    row1 = max(0, min(row1, data.shape[0] - 1))
    col1 = max(0, min(col1, data.shape[1] - 1))
    # Sample from the start point to the end point (rows and columns are not
    # reordered independently, which would mirror anti-diagonal segments)
    count = max(abs(row1 - row0), abs(col1 - col0)) + 1
    rows = np.rint(np.linspace(row0, row1, count)).astype(int)
    cols = np.rint(np.linspace(col0, col1, count)).astype(int)
    y = np.ma.array(data[rows, cols], float).filled(np.nan)
    x = np.arange(y.size)
    return x, y


@computation_function(
    operation_id="sigima.image.segment_profile",
    contract_version=2,
    qualification=ReplayQualification(contract_version=2, comparison=EXACT),
)
def segment_profile(src: ImageObj, p: SegmentProfileParam) -> SignalObj:
    """Compute segment profile

    The segment is sampled at regular steps, equal to the smallest distance between
    the centers of neighboring pixels, from (X1, Y1) to (X2, Y2): each sample takes
    the value of the pixel whose center is the nearest (no interpolation). The X
    axis of the profile is the distance from the start point, in the units of the
    image.

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the segment profile

    Raises:
        ValueError: if an end point lies outside the image, or if the X and Y units
         of the image differ
    """
    _check_same_units(src, "Segment profile")
    _nearest_pixels(src, [p.x1, p.x2], "x", "X")
    _nearest_pixels(src, [p.y1, p.y2], "y", "Y")
    length = float(np.hypot(p.x2 - p.x1, p.y2 - p.y1))
    step = _pixel_spacing(src)
    count = int(np.floor(length / step + PIXEL_INDEX_TOLERANCE)) + 1
    dist = np.arange(count) * step
    fraction = dist / length if length > 0 else np.zeros(count)
    cols = _nearest_pixels(src, p.x1 + (p.x2 - p.x1) * fraction, "x", "X")
    rows = _nearest_pixels(src, p.y1 + (p.y2 - p.y1) * fraction, "y", "Y")
    values = src.get_masked_view()[rows, cols]
    keep = ~ma.getmaskarray(values)
    suffix = f"({p.x1:g}, {p.y1:g})-({p.x2:g}, {p.y2:g})"
    units, labels = _axis_units(src, "x")
    dst = new_signal_result(src, "segment_profile", suffix, units, labels)
    dst.set_xydata(dist[keep], np.array(values, dtype=float)[keep])
    return dst


class AverageProfileParam(gds.DataSet):
    """Average horizontal or vertical profile parameters

    The rectangular area goes from (X1, Y1) to (X2, Y2), in the physical units of
    the image (None: edge of the image). It holds the pixels whose centers lie
    inside it, bounds included.
    """

    _directions = (("horizontal", _("horizontal")), ("vertical", _("vertical")))
    direction = gds.ChoiceItem(_("Direction"), _directions, radio=True)
    _hgroup_begin = gds.BeginGroup(_("Profile rectangular area"))
    _none_help = _("None: edge of the image")
    x1 = gds.FloatItem("X<sub>1</sub>", default=None, allow_none=True, help=_none_help)
    y1 = gds.FloatItem(
        "Y<sub>1</sub>", default=None, allow_none=True, help=_none_help
    ).set_pos(col=1)
    x2 = gds.FloatItem("X<sub>2</sub>", default=None, allow_none=True, help=_none_help)
    y2 = gds.FloatItem(
        "Y<sub>2</sub>", default=None, allow_none=True, help=_none_help
    ).set_pos(col=1)
    _hgroup_end = gds.EndGroup(_("Profile rectangular area"))


@computation_function(
    operation_id="sigima.image.average_profile",
    contract_version=2,
    qualification=ReplayQualification(contract_version=2, comparison=EXACT),
)
def average_profile(src: ImageObj, p: AverageProfileParam) -> SignalObj:
    """Compute horizontal or vertical average profile

    The pixels whose centers lie in the rectangle (bounds included, like
    rectangular ROIs) are averaged along the columns (horizontal profile) or the
    rows (vertical profile). The X axis of the profile holds the coordinates of the
    pixel centers, with the unit and label of the corresponding axis of the image.

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the average profile

    Raises:
        ValueError: if no pixel center lies in the rectangle
    """
    data = src.get_masked_view()
    nrows, ncols = data.shape
    left, top, right, bottom = src.pixel_range_to_physical([0, 0, ncols, nrows])
    x1 = left if p.x1 is None else p.x1
    y1 = top if p.y1 is None else p.y1
    x2 = right if p.x2 is None else p.x2
    y2 = bottom if p.y2 is None else p.y2
    ix0, iy0, ix1, iy1 = src.physical_to_pixel_range([x1, y1, x2, y2])
    ix0, iy0 = max(0, ix0), max(0, iy0)
    ix1, iy1 = min(ncols, ix1), min(nrows, iy1)
    if ix1 <= ix0 or iy1 <= iy0:
        raise ValueError(
            f"Average profile: no pixel center lies in the rectangle "
            f"({x1:g}, {y1:g})-({x2:g}, {y2:g})"
        )
    suffix = f"{p.direction}, x=[{x1:g}, {x2:g}], y=[{y1:g}, {y2:g}]"
    area = data[iy0:iy1, ix0:ix1]
    if p.direction == "horizontal":
        axis = "x"
        x, y = _pixel_centers(src, "x")[ix0:ix1], ma.mean(area, axis=0)
    else:
        axis = "y"
        x, y = _pixel_centers(src, "y")[iy0:iy1], ma.mean(area, axis=1)
    units, labels = _axis_units(src, axis)
    dst = new_signal_result(src, "average_profile", suffix, units, labels)
    dst.set_xydata(x, y)
    return dst


class RadialProfileParam(gds.DataSet):
    """Radial profile parameters

    The center is given in the physical units of the image.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__obj: ImageObj | None = None

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from image"""
        self.__obj = obj
        self.xc = obj.xc
        self.yc = obj.yc

    def choice_callback(self, item, value):  # pylint: disable=unused-argument
        """Callback for choice item"""
        if self.__obj is None:
            return
        if value == "centroid":
            self.xc, self.yc = _centroid(self.__obj, self.__obj.get_masked_view())
        elif value == "center":
            self.xc, self.yc = self.__obj.xc, self.__obj.yc

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
    xc = gds.FloatItem(f"X{_xyl}", default=0.0).set_prop("display", active=_func_prop)
    yc = gds.FloatItem(f"Y{_xyl}", default=0.0).set_prop("display", active=_func_prop)


def _centroid(src: ImageObj, data: np.ndarray) -> tuple[float, float]:
    """Return the centroid of the image, in physical coordinates."""
    yi, xi = sigima.tools.image.get_centroid_fourier(data)
    return (
        float(src.axis_to_physical([xi], "x")[0]),
        float(src.axis_to_physical([yi], "y")[0]),
    )


@computation_function(
    operation_id="sigima.image.radial_profile",
    contract_version=2,
    qualification=ReplayQualification(contract_version=2, comparison=EXACT),
)
def radial_profile(src: ImageObj, p: RadialProfileParam) -> SignalObj:
    """Compute radial profile around the centroid, the image center or a given point
    with :py:func:`sigima.tools.image.get_radial_profile`

    The center and the distances are physical: the pixels are binned by their
    distance to the center, in steps equal to the smallest distance between the
    centers of neighboring pixels, so that the profile is circular whatever the
    pixel sizes and for non-uniform coordinates. The X axis of the profile is the
    distance to the center (mirrored), in the units of the image.

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the radial profile

    Raises:
        ValueError: if the X and Y units of the image differ
    """
    _check_same_units(src, "Radial profile")
    data = src.get_masked_view()
    if p.center == "centroid":
        xc, yc = _centroid(src, data)
    elif p.center == "center":
        xc, yc = src.xc, src.yc
    else:
        xc, yc = p.xc, p.yc
    suffix = f"center=({xc:.3f}, {yc:.3f})"
    units, labels = _axis_units(src, "x")
    dst = new_signal_result(src, "radial_profile", suffix, units, labels)
    x, y = sigima.tools.image.get_radial_profile(
        data,
        (xc, yc),
        xcoords=_pixel_centers(src, "x"),
        ycoords=_pixel_centers(src, "y"),
        step=_pixel_spacing(src),
    )
    dst.set_xydata(x, y)
    return dst
