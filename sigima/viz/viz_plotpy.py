# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Visualization tools for Sigima (PlotPy backend)
===============================================

This module provides PlotPy-based visualization utilities for Sigima objects.
It offers interactive plotting with Qt dialogs, supporting curve and image display,
ROI visualization, and geometry result overlays.
"""

from __future__ import annotations

import os
import sys
from typing import Generator, Literal

import numpy as np
import plotpy.tools
from guidata.qthelpers import exec_dialog as guidata_exec_dialog
from plotpy.builder import make
from plotpy.config import CONF
from plotpy.items import (
    AnnotatedCircle,
    AnnotatedEllipse,
    AnnotatedPoint,
    AnnotatedPolygon,
    AnnotatedRectangle,
    AnnotatedSegment,
    AnnotatedShape,
    AnnotatedXRange,
    AnnotatedYRange,
    CurveItem,
    ImageItem,
    LabelItem,
    Marker,
    MaskedImageItem,
    MaskedXYImageItem,
)
from plotpy.items.shape.range import XRangeSelection
from plotpy.plot import (
    BasePlot,
    BasePlotOptions,
    PlotDialog,
    PlotOptions,
    SyncPlotDialog,
)
from plotpy.styles import LINESTYLES, ShapeParam
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from sigima.config import _
from sigima.enums import ContourShape
from sigima.objects import (
    CircularROI,
    GeometryResult,
    ImageObj,
    KindShape,
    PolygonalROI,
    RectangularROI,
    SegmentROI,
    SignalObj,
)
from sigima.tools import coordinates
from sigima.viz.annotation_plotpy import (
    annotations_to_plotpy_items,
    load_legacy_plotpy_items,
)


# Optional imports for test environment integration
def _get_default_name(suffix: str | None = None) -> str:
    """Return default name based on script name.

    Fallback if sigima.tests not available.
    """
    name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    if suffix is not None:
        name += "_" + suffix
    return name


def _debug_print(message: str) -> None:
    """Debug print (fallback if sigima.tests.env not available)."""
    if os.environ.get("SIGIMA_DEBUG", "").lower() in ("1", "true"):
        print(message)


# Try to import from sigima.tests for full functionality in test environment
try:
    from sigima.tests.env import execenv

    debug_print = execenv.print
except ImportError:
    debug_print = _debug_print

try:
    from sigima.tests.helpers import get_default_test_name
except ImportError:
    get_default_test_name = _get_default_name


QAPP: QW.QApplication | None = None

WIDGETS: list[QW.QWidget] = []


# ---------------------------------------------------------------------------
# Curve-clipped Signal ROI rendering (mirrors DataLab's adapter behavior)
# ---------------------------------------------------------------------------

#: Color palette used to cycle through Signal ROI fill colors when several
#: ROIs are defined on the same signal. Mirrors DataLab's ``tab10``-inspired
#: palette so ROI colors stay consistent across DataLab, DataLab-Kernel and
#: Sigima viewers.
_ROI_FILL_COLORS: tuple[str, ...] = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
)
#: Alpha (0..255) used for the translucent fill of Signal ROIs.
_ROI_FILL_ALPHA: int = 90


def roi_color_for_index(index: int) -> QG.QColor:
    """Return the Signal ROI fill color for the given ROI index (cycles
    through the predefined palette)."""
    name = _ROI_FILL_COLORS[index % len(_ROI_FILL_COLORS)]
    color = QG.QColor(name)
    color.setAlpha(_ROI_FILL_ALPHA)
    return color


class _CurveClippedXRangeSelection(XRangeSelection):
    """X-range selection whose translucent fill is clipped vertically to the
    underlying signal curve (instead of filling the whole canvas height).

    Mirrors :class:`datalab.adapters_plotpy.roi.signal._CurveClippedXRangeSelection`
    so Sigima's PlotPy viewer renders Signal ROIs identically to DataLab.
    """

    def __init__(
        self,
        _min: float | None = None,
        _max: float | None = None,
        shapeparam=None,
    ) -> None:
        super().__init__(_min, _max, shapeparam)
        self._curve_x: np.ndarray | None = None
        self._curve_y: np.ndarray | None = None
        self._fill_color: QG.QColor | None = None

    def set_signal_curve(self, x: np.ndarray, y: np.ndarray) -> None:
        """Attach the signal curve coordinates used to clip the fill area."""
        self._curve_x = np.asarray(x, dtype=float)
        self._curve_y = np.asarray(y, dtype=float)

    def set_fill_color(self, color: QG.QColor) -> None:
        """Override the fill color for this ROI instance."""
        self._fill_color = QG.QColor(color)

    def _build_curve_polygon(  # pylint: disable=too-many-return-statements
        self,
        xMap,  # pylint: disable=invalid-name
        yMap,  # pylint: disable=invalid-name
        rct,
    ):
        """Build a polygon following the signal curve between ``self._min``
        and ``self._max`` with a baseline at ``y=0`` (clamped to the visible
        canvas, with a fallback to canvas bottom for log-Y)."""
        if self._curve_x is None or self._curve_y is None:
            return None
        x_arr = self._curve_x
        y_arr = self._curve_y
        if x_arr.size < 2:
            return None
        finite = np.isfinite(x_arr) & np.isfinite(y_arr)
        if not finite.all():
            x_arr = x_arr[finite]
            y_arr = y_arr[finite]
            if x_arr.size < 2:
                return None
        xmin, xmax = self._min, self._max
        if xmin is None or xmax is None:
            return None
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if not np.all(np.diff(x_arr) >= 0):
            order = np.argsort(x_arr)
            x_arr = x_arr[order]
            y_arr = y_arr[order]
        x0 = max(xmin, float(x_arr[0]))
        x1 = min(xmax, float(x_arr[-1]))
        if x1 <= x0:
            return None
        mask = (x_arr >= x0) & (x_arr <= x1)
        xs_in = x_arr[mask]
        ys_in = y_arr[mask]
        y_left = float(np.interp(x0, x_arr, y_arr))
        y_right = float(np.interp(x1, x_arr, y_arr))
        xs = np.concatenate(([x0], xs_in, [x1]))
        ys = np.concatenate(([y_left], ys_in, [y_right]))
        keep = np.concatenate(([True], np.diff(xs) > 0))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 2:
            return None
        baseline_y = self._compute_baseline_y(yMap, rct)
        if baseline_y is None:
            return None
        polygon = QG.QPolygonF()
        polygon.append(QC.QPointF(xMap.transform(xs[0]), baseline_y))
        for xv, yv in zip(xs, ys):
            polygon.append(QC.QPointF(xMap.transform(xv), yMap.transform(yv)))
        polygon.append(QC.QPointF(xMap.transform(xs[-1]), baseline_y))
        return polygon

    def _compute_baseline_y(self, yMap, rct):  # pylint: disable=invalid-name
        """Return the canvas y-coordinate of the polygon baseline."""
        plot = self.plot()
        is_log_y = False
        if plot is not None:
            is_log_y = plot.get_axis_scale(self.yAxis()) == "log"
        if is_log_y:
            return rct.bottom()
        baseline_y = yMap.transform(0.0)
        if not np.isfinite(baseline_y):
            return rct.bottom()
        return max(rct.top(), min(rct.bottom(), baseline_y))

    def draw(self, painter, xMap, yMap, canvasRect):
        """Draw the ROI: filled polygon clipped to the curve (or a fallback
        rectangle), surrounded by the usual handle decoration."""
        plot = self.plot()
        if not plot:
            return
        if self.selected:
            pen = self.sel_pen
            sym = self.sel_symbol
        else:
            pen = self.pen
            sym = self.symbol

        rct = QC.QRectF(plot.canvas().contentsRect())
        rct.setLeft(xMap.transform(self._min))
        rct.setRight(xMap.transform(self._max))

        if self._fill_color is not None:
            brush = QG.QBrush(self._fill_color)
            edge_color = QG.QColor(self._fill_color)
            edge_color.setAlpha(255)
            pen = QG.QPen(pen)
            pen.setColor(edge_color)
        else:
            brush = self.brush

        polygon = self._build_curve_polygon(xMap, yMap, rct)
        painter.save()
        painter.setPen(QC.Qt.NoPen)
        painter.setBrush(brush)
        if polygon is not None:
            painter.drawPolygon(polygon)
        else:
            painter.fillRect(rct, brush)
        painter.restore()

        painter.setPen(pen)
        painter.drawLine(rct.topLeft(), rct.bottomLeft())
        painter.drawLine(rct.topRight(), rct.bottomRight())

        dash = QG.QPen(pen)
        dash.setStyle(QC.Qt.DashLine)
        dash.setWidth(1)
        painter.setPen(dash)
        cx = rct.center().x()
        painter.drawLine(QC.QPointF(cx, rct.top()), QC.QPointF(cx, rct.bottom()))

        if self.can_resize() and not self.is_readonly():
            painter.setPen(pen)
            x0, x1, y = self.get_handles_pos()
            sym.drawSymbol(painter, QC.QPointF(x0, y))
            sym.drawSymbol(painter, QC.QPointF(x1, y))


class _CurveClippedAnnotatedXRange(AnnotatedXRange):  # pylint: disable=abstract-method
    """Annotated X-range selection whose underlying shape is a
    :class:`_CurveClippedXRangeSelection` (curve-clipped fill + per-instance
    color).

    ``get_tr_size`` is intentionally not overridden: ``AnnotatedXRange`` (from
    PlotPy) does not override it either, so the abstract-method warning is a
    false positive inherited from upstream."""

    SHAPE_CLASS = _CurveClippedXRangeSelection


CONF.set("plot", "title/font/size", 11)


def __ensure_qapp() -> QW.QApplication:
    """Ensure that a QApplication instance exists."""
    global QAPP  # pylint: disable=global-statement
    if QAPP is None:
        QAPP = QW.QApplication.instance()
        if QAPP is None:
            QAPP = QW.QApplication([])  # type: ignore[assignment]
    return QAPP


def __exec_dialog(dlg: QW.QDialog) -> None:
    """Execute a dialog, supporting Sphinx-Gallery scraping."""
    global WIDGETS  # pylint: disable=global-statement,global-variable-not-assigned
    gallery_building = os.getenv("SPHINX_GALLERY_BUILDING")
    if gallery_building:
        dlg.show()
        WIDGETS.append(dlg)
    else:
        guidata_exec_dialog(dlg)


TEST_NB = {}

# Default image parameters
IMAGE_PARAMETERS = {
    "interpolation": "nearest",
    "eliminate_outliers": 0.1,
    "colormap": "viridis",
}

#: Curve colors
COLORS = (
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
)


def __style_generator() -> Generator[tuple[str, str], None, None]:
    """Cycling through curve styles"""
    while True:
        for linestyle in LINESTYLES:
            for color in COLORS:
                yield (color, linestyle)


make.style = __style_generator()


def __generate_widget_name_title(
    name: str | None, title: str | None
) -> tuple[str, str]:
    """Return (default) widget name and title

    Args:
        name: Name of the widget, or None to use a default name
        title: Title of the widget, or None to use a default title

    Returns:
        A tuple (name, title) where:
        - `name` is the widget name, which is either the provided name or a default
        - `title` is the widget title, which is either the provided title or a default
    """
    if name is None:
        TEST_NB[name] = TEST_NB.setdefault(name, 0) + 1
        name = get_default_test_name(f"{TEST_NB[name]:02d}")
    if title is None:
        title = f"{_('Plot dialog')} `{name}`"
    return name, title


def __create_curve_dialog(
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    size: tuple[int, int] | None = None,
) -> PlotDialog:
    """Create Curve Dialog

    Args:
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        size: Size of the dialog as a tuple (width, height), or None for default size

    Returns:
        A `PlotDialog` instance configured for curve plotting
    """
    name, title = __generate_widget_name_title(name, title)
    win = PlotDialog(
        edit=False,
        toolbar=True,
        title=title,
        options=PlotOptions(
            title=title,
            type="curve",
            xlabel=xlabel,
            ylabel=ylabel,
            xunit=xunit,
            yunit=yunit,
            curve_antialiasing=True,
        ),
        size=(800, 600) if size is None else size,
    )
    win.setObjectName(name)
    return win


def create_curve(x: np.ndarray, y: np.ndarray, title: str | None = None) -> CurveItem:
    """Create a curve item from x and y data

    Args:
        x: X data array
        y: Y data array
        title: Label for the curve, or None for no label

    Returns:
        A `CurveItem` representing the curve
    """
    item = make.mcurve(x, y, label=title)
    return item


def create_image(
    data: np.ndarray,
    title: str | None = None,
    interpolation: str = "linear",
    colormap: str | None = None,
    alpha_function: str | None = None,
    xdata: list[float] | None = None,
    ydata: list[float] | None = None,
) -> ImageItem:
    """Create an image item from data

    Args:
        data: 2D array of image data
        title: Title for the image
        interpolation: Interpolation method for image display
        colormap: Colormap for the image, or None for default
        alpha_function: Alpha function for the image, or None for default
        xdata: X coordinates for the image (optional)
        ydata: Y coordinates for the image (optional)
    """
    item = make.image(
        data,
        title=title,
        interpolation=interpolation,
        colormap=colormap,
        xdata=[None, None] if xdata is None else xdata,
        ydata=[None, None] if ydata is None else ydata,
        eliminate_outliers=2.0,
        alpha_function=alpha_function,
    )
    return item


def create_contour_shapes(
    coords: np.ndarray, shape: ContourShape
) -> list[AnnotatedShape]:
    """Create plotpy items for a specific contour shape.

    Args:
        coords: Coordinates of the contours
        shape: ContourShape enum value

    Returns:
        List of plotpy items representing the detected contours
    """
    items = []
    debug_print(f"Coordinates ({shape}s): {coords}")
    for shapeargs in coords:
        if shape == ContourShape.CIRCLE:
            xc, yc, r = shapeargs
            x0, y0, x1, y1 = coordinates.circle_to_diameter(xc, yc, r)
            item = make.circle(x0, y0, x1, y1)
        elif shape == ContourShape.ELLIPSE:
            xc, yc, a, b, theta = shapeargs
            coords_ellipse = coordinates.ellipse_to_diameters(xc, yc, a, b, theta)
            x0, y0, x1, y1, x2, y2, x3, y3 = coords_ellipse
            item = make.ellipse(x0, y0, x1, y1, x2, y2, x3, y3)
        else:  # ContourShape.POLYGON
            # `shapeargs` is a flattened array of x, y coordinates
            x, y = shapeargs[::2], shapeargs[1::2]
            item = make.polygon(x, y, closed=False)
        items.append(item)
    return items


def create_circle(
    xc: float,
    yc: float,
    r: float,
    label: str | None = None,
) -> AnnotatedCircle:
    """Create an annotated circle item

    Args:
        xc: X-coordinate of the circle center
        yc: Y-coordinate of the circle center
        r: Radius of the circle
        label: Label for the circle, or None for no label

    Returns:
        An `AnnotatedCircle` object representing the circle
    """
    item = make.annotated_circle(xc - r, yc, xc + r, yc, label, show_computations=False)
    item.label.labelparam.bgalpha = 0.5
    item.label.labelparam.anchor = "T"
    item.label.labelparam.yc = 10
    item.label.labelparam.update_item(item.label)
    p: ShapeParam = item.shape.shapeparam
    p.line.color = "#ff9933"
    p.line.width = 2
    p.symbol.facecolor = "#ffb366"
    p.symbol.edgecolor = "#ff9933"
    p.symbol.marker = "Ellipse"
    p.symbol.size = 5
    p.update_item(item.shape)
    item.set_movable(False)
    item.set_resizable(False)
    item.set_selectable(False)
    return item


def create_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str | None = None,
) -> AnnotatedSegment:
    """Create an annotated segment item

    Args:
        x0: X-coordinate of the start point
        y0: Y-coordinate of the start point
        x1: X-coordinate of the end point
        y1: Y-coordinate of the end point
        label: Label for the segment, or None for no label

    Returns:
        An `AnnotatedSegment` object representing the segment
    """
    item = make.annotated_segment(x0, y0, x1, y1, label, show_computations=False)
    item.label.labelparam.bgalpha = 0.5
    item.label.labelparam.anchor = "T"
    item.label.labelparam.yc = 10
    item.label.labelparam.update_item(item.label)
    p: ShapeParam = item.shape.shapeparam
    p.line.color = "#33ff00"
    p.line.width = 5
    p.symbol.facecolor = "#26be00"
    p.symbol.edgecolor = "#33ff00"
    p.symbol.marker = "Ellipse"
    p.symbol.size = 11
    p.update_item(item.shape)
    item.set_movable(False)
    item.set_resizable(False)
    item.set_selectable(False)
    return item


def create_cursor(
    orientation: Literal["h", "v", "x"],
    position: float | tuple[float, float],
    label: str,
) -> Marker:
    """Create a horizontal or vertical cursor item

    Args:
        orientation: 'h' for horizontal cursor, 'v' for vertical cursor,
         'x' for crosshair
        position: Position of the cursor along the relevant axis,
         or (x, y) tuple for crosshair
        label: Label format string for the cursor

    Returns:
        A `Marker` representing the cursor
    """
    if orientation == "h":
        cursor = make.hcursor(position, label=label)
    elif orientation == "v":
        cursor = make.vcursor(position, label=label)
    elif orientation == "x":
        assert isinstance(position, tuple) and len(position) == 2
        cursor = make.xcursor(position[0], position[1], label=label)
    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    cursor.set_resizable(False)
    cursor.set_movable(False)
    cursor.set_selectable(False)
    cursor.markerparam.line.color = "#a7ff33"
    cursor.markerparam.line.width = 2
    cursor.markerparam.symbol.marker = "NoSymbol"
    cursor.markerparam.text.textcolor = "#ffffff"
    cursor.markerparam.text.background_color = "#000000"
    cursor.markerparam.text.background_alpha = 0.5
    cursor.markerparam.text.font.bold = True
    cursor.markerparam.update_item(cursor)
    return cursor


def create_range(
    orientation: Literal["h", "v"], pos_min: float, pos_max: float, title: str
) -> AnnotatedXRange | AnnotatedYRange:
    """Create a horizontal or vertical range item

    Args:
        orientation: 'h' for horizontal range, 'v' for vertical range
        pos_min: Minimum position of the range along the relevant axis
        pos_max: Maximum position of the range along the relevant axis
        title: Title for the range

    Returns:
        An `AnnotatedXRange` or `AnnotatedYRange` representing the range
    """
    if orientation == "h":
        item = make.annotated_xrange(
            pos_min, pos_max, title=title, show_computations=False
        )
    elif orientation == "v":
        item = make.annotated_yrange(
            pos_min, pos_max, title=title, show_computations=False
        )
    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    item.label.labelparam.bgalpha = 0.5
    item.label.labelparam.anchor = "L"
    item.label.labelparam.xc = 20
    item.label.labelparam.update_item(item.label)
    item.set_movable(False)
    item.set_resizable(False)
    item.set_selectable(False)
    return item


def create_label(text: str) -> LabelItem:
    """Create a text label item

    Args:
        text: Text content of the label

    Returns:
        A `LabelItem` representing the text label
    """
    item = make.label(text, "TL", (0, 0), "TL")
    return item


def create_marker(x: float, y: float, title: str | None = None) -> Marker:
    """Create a marker item

    Args:
        x: x coordinate
        y: y coordinate
        title: title of the marker

    Returns:
        A `Marker` representing the marker
    """
    if title is None:
        return make.marker((x, y))
    return make.marker((x, y), "%.3f", title)


def __make_marker_item(x0: float, y0: float, fmt: str, title: str) -> Marker:
    """Make marker item

    Args:
        x0: x coordinate
        y0: y coordinate
        fmt: numeric format (e.g. '%.3f')
        title: title of the marker
    """
    if np.isnan(x0):
        mstyle = "-"

        def label(x, y):  # pylint: disable=unused-argument
            return (title + ": " + fmt) % y

    elif np.isnan(y0):
        mstyle = "|"

        def label(x, y):  # pylint: disable=unused-argument
            return (title + ": " + fmt) % x

    else:
        mstyle = "+"
        txt = title + ": (" + fmt + ", " + fmt + ")"

        def label(x, y):
            return txt % (x, y)

    return make.marker(
        position=(x0, y0),
        markerstyle=mstyle,
        label_cb=label,
        linestyle="DashLine",
        color="yellow",
    )


def __create_curve_item(
    obj: SignalObj | tuple[np.ndarray, np.ndarray], title: str | None = None
) -> CurveItem:
    """Create a curve item from a SignalObj or (xdata, ydata) tuple

    Args:
        obj: Signal object or tuple of (xdata, ydata)
        title: Title for the curve item

    Returns:
        A `CurveItem` representing the signal data
    """
    if isinstance(obj, (tuple, list)):
        xdata, ydata = obj
        title = title or ""
    else:
        assert obj.xydata is not None
        xdata, ydata = obj.xydata
        title = obj.title or title or ""
    # Only display the real part for signals (for simplicity):
    item = create_curve(xdata.real, ydata.real, title=title)
    item.param.line.width = 1.25
    item.param.update_item(item)
    return item


def __create_curve_roi_items(obj: SignalObj) -> list[AnnotatedXRange]:
    """Create signal ROI items from a SignalObj

    Args:
        obj: Signal object

    Returns:
        A list of `AnnotatedXRange` items representing the ROIs
    """
    items = []
    if obj.roi is not None and not obj.roi.is_empty():
        # Curve coordinates used to clip the ROI fill vertically (so each
        # ROI follows the underlying signal curve, with a baseline at y=0).
        x_curve, y_curve = (None, None)
        if obj.xydata is not None:
            x_curve, y_curve = obj.xydata
        for roi_idx, single_roi in enumerate(obj.roi):
            assert isinstance(single_roi, SegmentROI)
            x0, x1 = single_roi.get_physical_coords(obj)
            roi_item = _CurveClippedAnnotatedXRange(x0, x1)
            roi_item.setTitle(single_roi.title)
            roi_item.shape.set_style("plot", "range")
            if x_curve is not None and y_curve is not None:
                roi_item.shape.set_signal_curve(x_curve, y_curve)
            roi_item.shape.set_fill_color(roi_color_for_index(roi_idx))
            roi_item.label.labelparam.anchor = "T"
            roi_item.label.labelparam.xc = 20
            roi_item.label.labelparam.update_item(roi_item.label)
            roi_item.set_movable(False)
            roi_item.set_resizable(False)
            roi_item.set_selectable(False)
            items.append(roi_item)
    return items


def __create_image_item(
    obj: ImageObj | np.ndarray, title: str | None = None, **kwargs
) -> MaskedImageItem | MaskedXYImageItem:
    """Create an image item from an ImageObj

    Args:
        obj: Image object or 2D numpy array
        title: Title for the image item
        **kwargs: Additional parameters for image display
         (e.g., interpolation, colormap)

    Returns:
        A `MaskedImageItem` or `MaskedXYImageItem` representing the image
    """
    if isinstance(obj, ImageObj):
        data = obj.data
        mask = obj.maskdata
        title = obj.title or title or ""
    elif isinstance(obj, np.ndarray):
        data = obj
        mask = np.zeros_like(data, dtype=bool)
        title = title or ""
    else:
        raise TypeError(f"Unsupported image type: {type(obj)}")
    imparameters = IMAGE_PARAMETERS.copy()
    imparameters.update(kwargs)
    if isinstance(obj, ImageObj) and not obj.is_uniform_coords:
        x, y = obj.xcoords, obj.ycoords
        item = make.maskedxyimage(
            x, y, data, mask, title=title, show_mask=True, **imparameters
        )
    else:
        item = make.maskedimage(data, mask, title=title, show_mask=True, **imparameters)
        if isinstance(obj, ImageObj):
            x0, y0, dx, dy = obj.x0, obj.y0, obj.dx, obj.dy
            item.param.xmin, item.param.xmax = x0, x0 + dx * data.shape[1]
            item.param.ymin, item.param.ymax = y0, y0 + dy * data.shape[0]
            item.param.update_item(item)
    return item


def __create_image_roi_items(obj: ImageObj) -> list[AnnotatedShape]:
    """Create image ROI items from an ImageObj

    Args:
        obj: Image object

    Returns:
        A list of `AnnotatedShape` items representing the ROIs
    """
    items = []
    if obj.roi is not None and not obj.roi.is_empty():
        for single_roi in obj.roi:
            if isinstance(single_roi, RectangularROI):
                x0, y0, x1, y1 = single_roi.get_bounding_box(obj)
                roi_item = make.annotated_rectangle(x0, y0, x1, y1, single_roi.title)
            elif isinstance(single_roi, CircularROI):
                xc, yc, r = single_roi.get_physical_coords(obj)
                x0, y0, x1, y1 = coordinates.circle_to_diameter(xc, yc, r)
                roi_item = make.annotated_circle(x0, y0, x1, y1, single_roi.title)
            elif isinstance(single_roi, PolygonalROI):
                coords = single_roi.get_physical_coords(obj)
                points = np.array(coords).reshape(-1, 2)
                roi_item = AnnotatedPolygon(points)
                roi_item.annotationparam.title = single_roi.title
                roi_item.set_style("plot", "shape/drag")
                roi_item.annotationparam.update_item(roi_item)
            items.append(roi_item)
    return items


def __create_plot_items_from_geometry(
    result: GeometryResult,
) -> list[
    AnnotatedPoint
    | Marker
    | AnnotatedRectangle
    | AnnotatedCircle
    | AnnotatedSegment
    | AnnotatedEllipse
    | AnnotatedPolygon
]:
    """Create plot items from a GeometryResult object

    Args:
        result: The GeometryResult object to convert

    Returns:
        A list of plot items corresponding to the geometry result
    """
    items = []
    for coords in result.coords:
        title = result.title or ""
        if result.kind == KindShape.POINT:
            x0, y0 = coords
            item = AnnotatedPoint(x0, y0)
            sparam: ShapeParam = item.shape.shapeparam
            sparam.symbol.marker = "Ellipse"
            sparam.symbol.size = 6
            sparam.sel_symbol.marker = "Ellipse"
            sparam.sel_symbol.size = 6
            aparam = item.annotationparam
            aparam.title = title
            sparam.update_item(item.shape)
            aparam.update_item(item)
        elif result.kind == KindShape.MARKER:
            x0, y0 = coords
            item = __make_marker_item(x0, y0, "%.3f", title)
        elif result.kind == KindShape.RECTANGLE:
            x0, y0, dx, dy = coords
            item = make.annotated_rectangle(x0, y0, x0 + dx, y0 + dy, title=title)
        elif result.kind == KindShape.CIRCLE:
            xc, yc, r = coords
            x0, y0, x1, y1 = coordinates.circle_to_diameter(xc, yc, r)
            item = make.annotated_circle(x0, y0, x1, y1, title=title)
        elif result.kind == KindShape.SEGMENT:
            x0, y0, x1, y1 = coords
            item = make.annotated_segment(x0, y0, x1, y1, title=title)
        elif result.kind == KindShape.ELLIPSE:
            xc, yc, a, b, t = coords
            coords = coordinates.ellipse_to_diameters(xc, yc, a, b, t)
            x0, y0, x1, y1, x2, y2, x3, y3 = coords
            item = make.annotated_ellipse(x0, y0, x1, y1, x2, y2, x3, y3, title=title)
        elif result.kind == KindShape.POLYGON:
            x, y = coords[::2], coords[1::2]
            item = make.polygon(x, y, title=title, closed=False)
        else:
            raise TypeError(f"Unsupported GeometryResult type: {type(result)}")
        item.set_movable(False)
        item.set_resizable(False)
        item.set_selectable(False)

        if isinstance(item, AnnotatedShape):
            shapeparam: ShapeParam = item.shape.shapeparam
            shapeparam.line.width = 2
            shapeparam.update_item(item.shape)
            item.annotationparam.show_computations = False
            item.annotationparam.show_label = bool(title)
            item.annotationparam.update_item(item)
            item.label.labelparam.anchor = "T"
            item.label.labelparam.yc = 10
            item.label.labelparam.update_item(item.label)

        items.append(item)

    return items


def __generate_object_name(title: str, fallback: str) -> str:
    """Generate a valid object name from a title string

    Args:
        title: The title string to convert
        fallback: Fallback name to use if title is empty or invalid

    Returns:
        A valid object name derived from the title or the fallback name
    """
    if title:
        obj_name = "".join(c if c.isalnum() else "_" for c in title)
        if obj_name:
            return obj_name
    return fallback


def view_curve_items(
    items: list[CurveItem],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    add_legend: bool = True,
    datetime_format: str | None = None,
    object_name: str = "",
) -> None:
    """Create a curve dialog and plot items

    Args:
        items: List of `CurveItem` objects to plot
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        add_legend: Whether to add a legend to the plot, default is True
        datetime_format: Datetime format for x-axis if x data is datetime, or None
        object_name: Object name for the dialog (for screenshot functionality)
    """
    __ensure_qapp()
    win = __create_curve_dialog(
        name=name, title=title, xlabel=xlabel, ylabel=ylabel, xunit=xunit, yunit=yunit
    )
    win.setObjectName(object_name or __generate_object_name(title, "curve_dialog"))
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    if add_legend:
        plot.add_item(make.legend())
    if datetime_format is not None:
        plot.set_axis_datetime("bottom", format=datetime_format)
    __exec_dialog(win)
    make.style = __style_generator()  # Reset style generator for next call


def view_curves(
    data_or_objs: list[SignalObj | np.ndarray | tuple[np.ndarray, np.ndarray]]
    | SignalObj
    | np.ndarray
    | tuple[np.ndarray, np.ndarray],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    show_roi: bool = True,
    show_annotations: bool = True,
    object_name: str = "",
) -> None:
    """Create a curve dialog and plot curves

    Args:
        data_or_objs: Single `SignalObj` or `np.ndarray`, or a list/tuple of these,
         or a list/tuple of (xdata, ydata) pairs
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        show_roi: Whether to show ROIs defined in `SignalObj` instances, default is True
         (ignored if `data_or_objs` is not a `SignalObj`)
        show_annotations: Whether to show canonical and historical annotations,
         default is True
        object_name: Object name for the dialog (for screenshot functionality)
    """
    __ensure_qapp()
    if isinstance(data_or_objs, (tuple, list)):
        datalist = data_or_objs
    else:
        datalist = [data_or_objs]
    items = []
    datetime_format = None
    for data_or_obj in datalist:
        if isinstance(data_or_obj, SignalObj):
            xlabel = xlabel or data_or_obj.xlabel or ""
            ylabel = ylabel or data_or_obj.ylabel or ""
            xunit = xunit or data_or_obj.xunit or ""
            yunit = yunit or data_or_obj.yunit or ""
            if data_or_obj.is_x_datetime():
                datetime_format = data_or_obj.metadata.get("x_datetime_format")
                if datetime_format is None:
                    unit = data_or_obj.xunit if data_or_obj.xunit else "s"
                    if unit in ("ns", "us", "ms"):
                        datetime_format = "%H:%M:%S.%f"
                    else:
                        datetime_format = "%H:%M:%S"
        item = __create_curve_item(data_or_obj)
        if isinstance(data_or_obj, SignalObj) and show_roi:
            items.extend(__create_curve_roi_items(data_or_obj))
        items.append(item)
        if isinstance(data_or_obj, SignalObj) and show_annotations:
            items.extend(
                annotations_to_plotpy_items(data_or_obj.get_graphical_annotations())
            )
            items.extend(load_legacy_plotpy_items(data_or_obj))
    view_curve_items(
        items,
        name=name,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xunit=xunit,
        yunit=yunit,
        datetime_format=datetime_format,
        object_name=object_name,
    )
    make.style = __style_generator()  # Reset style generator for next call


def __create_image_dialog(
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    size: tuple[int, int] | None = None,
    object_name: str = "",
) -> PlotDialog:
    """Create Image Dialog

    Args:
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        size: Size of the dialog as a tuple (width, height), or None for default size
        object_name: Object name for the dialog (for screenshot functionality)

    Returns:
        A `PlotDialog` instance configured for image plotting
    """
    __ensure_qapp()
    name, title = __generate_widget_name_title(name, title)
    win = PlotDialog(
        edit=False,
        toolbar=True,
        title=title,
        options=PlotOptions(
            title=title,
            type="image",
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            xunit=xunit,
            yunit=yunit,
            zunit=zunit,
        ),
        size=(800, 600) if size is None else size,
    )
    win.setObjectName(object_name or name)
    for toolklass in (
        plotpy.tools.LabelTool,
        plotpy.tools.VCursorTool,
        plotpy.tools.HCursorTool,
        plotpy.tools.XCursorTool,
        plotpy.tools.AnnotatedRectangleTool,
        plotpy.tools.AnnotatedCircleTool,
        plotpy.tools.AnnotatedEllipseTool,
        plotpy.tools.AnnotatedSegmentTool,
        plotpy.tools.AnnotatedPointTool,
    ):
        win.get_manager().add_tool(toolklass, switch_to_default_tool=True)
    return win


def view_image_items(
    items: list[ImageItem],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    show_itemlist: bool = False,
    object_name: str = "",
) -> None:
    """Create an image dialog and show items

    Args:
        items: List of `ImageItem` objects to display
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        show_itemlist: Whether to show the item list panel in the dialog,
         default is False
        object_name: Object name for the dialog (for screenshot functionality)
    """
    __ensure_qapp()
    win = __create_image_dialog(
        name=name,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        xunit=xunit,
        yunit=yunit,
        zunit=zunit,
        object_name=object_name,
    )
    if show_itemlist:
        win.manager.get_itemlist_panel().show()
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    __exec_dialog(win)


# pylint: disable=too-many-positional-arguments
def view_images(
    data_or_objs: list[ImageObj | np.ndarray] | ImageObj | np.ndarray,
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    results: list[GeometryResult] | GeometryResult | None = None,
    show_roi: bool = True,
    show_annotations: bool = True,
    object_name: str = "",
    **kwargs,
) -> None:
    """Create an image dialog and show images

    Args:
        data_or_objs: Single `ImageObj` or `np.ndarray`, or a list/tuple of these
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        results: Single `GeometryResult` or list of these to overlay on images, or None
         if no overlay is needed.
        show_roi: Whether to show ROIs defined in `ImageObj` instances, default is True
         (ignored if `data_or_objs` is not a `ImageObj`)
        show_annotations: Whether to show canonical and historical annotations,
         default is True
        object_name: Object name for the dialog (for screenshot functionality)
        **kwargs: Additional keyword arguments to pass to `make.maskedimage()`
    """
    __ensure_qapp()
    if isinstance(data_or_objs, (tuple, list)):
        datalist = data_or_objs
    else:
        datalist = [data_or_objs]
    imparameters = IMAGE_PARAMETERS.copy()
    imparameters.update(kwargs)
    items = []
    image_title: str | None = None
    for data_or_obj in datalist:
        if isinstance(data_or_obj, ImageObj):
            data = data_or_obj.data
            if data_or_obj.title:
                image_title = data_or_obj.title
            if data_or_obj.xlabel and xlabel is None:
                xlabel = data_or_obj.xlabel
            if data_or_obj.ylabel and ylabel is None:
                ylabel = data_or_obj.ylabel
            if data_or_obj.zlabel and zlabel is None:
                zlabel = data_or_obj.zlabel
            if data_or_obj.xunit and xunit is None:
                xunit = data_or_obj.xunit
            if data_or_obj.yunit and yunit is None:
                yunit = data_or_obj.yunit
            if data_or_obj.zunit and zunit is None:
                zunit = data_or_obj.zunit
        elif isinstance(data_or_obj, np.ndarray):
            data = data_or_obj
        else:
            raise TypeError(f"Unsupported data type: {type(data_or_obj)}")
        # Display real and imaginary parts of complex images.
        assert data is not None
        if np.issubdtype(data.dtype, np.complexfloating):
            re_title = f"Re({image_title})" if image_title is not None else "Real"
            im_title = f"Im({image_title})" if image_title is not None else "Imaginary"
            items.append(__create_image_item(data.real, title=re_title, **imparameters))
            items.append(__create_image_item(data.imag, title=im_title, **imparameters))
        else:
            items.append(
                __create_image_item(data_or_obj, title=image_title, **imparameters)
            )
        if isinstance(data_or_obj, ImageObj) and show_roi:
            items.extend(__create_image_roi_items(data_or_obj))
        if isinstance(data_or_obj, ImageObj) and show_annotations:
            items.extend(
                annotations_to_plotpy_items(data_or_obj.get_graphical_annotations())
            )
            items.extend(load_legacy_plotpy_items(data_or_obj))
    if results is not None:
        if isinstance(results, GeometryResult):
            results = [results]
        if not isinstance(results, (list, tuple)):
            raise TypeError(f"Unsupported results type: {type(results)}")
        for res in results:
            items.extend(__create_plot_items_from_geometry(res))
    view_image_items(
        items,
        name=name,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        xunit=xunit,
        yunit=yunit,
        zunit=zunit,
        object_name=object_name,
    )


def view_curves_and_images(
    data_or_objs: list[SignalObj | np.ndarray | ImageObj | np.ndarray],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    object_name: str = "",
    show_annotations: bool = True,
) -> None:
    """View signals, then images in two successive dialogs

    Args:
        data_or_objs: List of `SignalObj`, `ImageObj`, `np.ndarray` or a mix of these
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        object_name: Object name for the dialog (for screenshot functionality)
        show_annotations: Whether to show canonical and historical annotations,
         default is True
    """
    __ensure_qapp()
    if isinstance(data_or_objs, (tuple, list)):
        objs = data_or_objs
    else:
        objs = [data_or_objs]
    sig_objs = [obj for obj in objs if isinstance(obj, (SignalObj, np.ndarray))]
    if sig_objs:
        view_curves(
            sig_objs,
            name=name,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xunit=xunit,
            yunit=yunit,
            object_name=f"{object_name}_curves",
            show_annotations=show_annotations,
        )
    ima_objs = [obj for obj in objs if isinstance(obj, (ImageObj, np.ndarray))]
    if ima_objs:
        view_images(
            ima_objs,
            name=name,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            xunit=xunit,
            yunit=yunit,
            zunit=zunit,
            object_name=f"{object_name}_images",
            show_annotations=show_annotations,
        )


def __compute_grid(
    num_objects: int, max_cols: int = 4, fixed_num_rows: int | None = None
) -> tuple[int, int]:
    """Compute number of rows and columns for a grid of images

    Args:
        num_objects: Total number of objects to display
        max_cols: Maximum number of columns in the grid
        fixed_num_rows: Fixed number of rows, if specified

    Returns:
        A tuple (num_rows, num_cols) representing the grid dimensions
    """
    num_cols = min(num_objects, max_cols)
    if fixed_num_rows is not None:
        num_rows = fixed_num_rows
        num_cols = (num_objects + num_rows - 1) // num_rows
    else:
        num_rows = (num_objects + num_cols - 1) // num_cols
    return num_rows, num_cols


def view_images_side_by_side(
    images: list[ImageItem | np.ndarray | ImageObj],
    titles: list[str] | None = None,
    share_axes: bool = True,
    rows: int | None = None,
    maximized: bool = False,
    title: str | None = None,
    results: list[GeometryResult] | GeometryResult | None = None,
    show_roi: bool = True,
    show_annotations: bool = True,
    object_name: str = "",
    **kwargs,
) -> None:
    """Show sequence of images

    Args:
        images: List of `ImageItem`, `np.ndarray`, or `ImageObj` objects to display
        titles: List of titles for each image
        share_axes: Whether to share axes across plots, default is True
        rows: Fixed number of rows in the grid, or None to compute automatically
        maximized: Whether to show the dialog maximized, default is False
        title: Title of the dialog, or None for a default title
        results: Single `GeometryResult` or list of these to overlay on images, or None
         if no overlay is needed.
        show_roi: Whether to show ROIs defined in `ImageObj` instances, default is True
         (ignored if `images` do not contain `ImageObj` instances)
        show_annotations: Whether to show canonical and historical annotations,
         default is True
        object_name: Object name for the dialog widget (used for screenshot filename)
        **kwargs: Additional keyword arguments to pass to `make.maskedimage()`
    """
    __ensure_qapp()
    # pylint: disable=too-many-nested-blocks
    rows, cols = __compute_grid(len(images), fixed_num_rows=rows, max_cols=4)
    dlg = SyncPlotDialog(title=title)
    dlg.setObjectName(
        object_name or __generate_object_name(title, "images_side_by_side")
    )
    imparameters = IMAGE_PARAMETERS.copy()
    imparameters.update(kwargs)
    if not isinstance(titles, (list, tuple)):
        titles = [titles] * len(images)
    elif len(titles) != len(images):
        raise ValueError("Length of titles must match length of images")
    if not isinstance(results, (list, tuple)):
        results = [results] * len(images)
    elif len(results) != len(images):
        raise ValueError("Length of results must match length of images")
    for idx, (img, result, imtitle) in enumerate(zip(images, results, titles)):
        row = idx // cols
        col = idx % cols
        imtitle = img.title if isinstance(img, ImageObj) else imtitle
        plot = BasePlot(options=BasePlotOptions(title=imtitle))
        other_items = []
        if isinstance(img, (MaskedImageItem, ImageItem)):
            item = img
        else:
            item = __create_image_item(img, title=imtitle, **imparameters)
            if isinstance(img, ImageObj) and show_roi:
                other_items.extend(__create_image_roi_items(img))
            if isinstance(img, ImageObj) and show_annotations:
                other_items.extend(
                    annotations_to_plotpy_items(img.get_graphical_annotations())
                )
                other_items.extend(load_legacy_plotpy_items(img))
        plot.add_item(item)
        for other_item in other_items:
            plot.add_item(other_item)
        if result is not None:
            if not isinstance(result, GeometryResult):
                raise TypeError(f"Unsupported results type: {type(result)}")
            overlay_items = __create_plot_items_from_geometry(result)
            for overlay_item in overlay_items:
                plot.add_item(overlay_item)
        dlg.add_plot(row, col, plot, sync=share_axes)
    dlg.finalize_configuration()
    if maximized:
        dlg.showMaximized()
    elif os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        # Set explicit size for proper rendering in headless mode
        # Qt size hints don't work reliably without a display
        dlg.resize(20 + 440 * cols, 20 + 400 * rows)
    __exec_dialog(dlg)
