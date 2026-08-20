# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""PlotPy adapter for canonical and historical graphical annotations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from guidata.io import JSONReader
from plotpy.builder import make
from plotpy.io import load_items
from plotpy.items import (
    AnnotatedCircle,
    AnnotatedEllipse,
    AnnotatedObliqueRectangle,
    AnnotatedPoint,
    AnnotatedPolygon,
    AnnotatedRectangle,
    AnnotatedSegment,
    AnnotatedShape,
    AnnotatedXRange,
    AnnotatedYRange,
    LabelItem,
    Marker,
)
from qtpy import QtCore as QC
from qwt import QwtPlotMarker

from sigima.objects.annotations import (
    AnnotationLabel,
    AnnotationStyle,
    Axis,
    CircleAnnotation,
    CursorAnnotation,
    CursorOrientation,
    EllipseAnnotation,
    FillStyle,
    GraphicalAnnotation,
    MarkerStyle,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    StrokeStyle,
    TextAnchor,
    TextAnnotation,
    TextStyle,
    annotation_to_dict,
    is_graphical_annotation_dict,
)

if TYPE_CHECKING:
    from sigima.objects.base import BaseObj

_ANCHORS = {
    TextAnchor.TOP_LEFT: "TL",
    TextAnchor.TOP: "T",
    TextAnchor.TOP_RIGHT: "TR",
    TextAnchor.LEFT: "L",
    TextAnchor.CENTER: "C",
    TextAnchor.RIGHT: "R",
    TextAnchor.BOTTOM_LEFT: "BL",
    TextAnchor.BOTTOM: "B",
    TextAnchor.BOTTOM_RIGHT: "BR",
}

_LINE_STYLES = {
    "solid": "SolidLine",
    "dashed": "DashLine",
    "dotted": "DotLine",
    "dashdot": "DashDotLine",
    "-": "SolidLine",
    "--": "DashLine",
    ":": "DotLine",
    "-.": "DashDotLine",
}

_MARKERS = {
    "circle": "Ellipse",
    "square": "Rect",
    "diamond": "Diamond",
    "cross": "Cross",
    "x": "XCross",
    "triangle-up": "UTriangle",
    "triangle-down": "DTriangle",
    "none": "NoSymbol",
}

_PLOTPY_ANCHOR_POSITIONS = {
    "TL": (0.0, 1.0),
    "T": (0.5, 1.0),
    "TR": (1.0, 1.0),
    "L": (0.0, 0.5),
    "C": (0.5, 0.5),
    "R": (1.0, 0.5),
    "BL": (0.0, 0.0),
    "B": (0.5, 0.0),
    "BR": (1.0, 0.0),
}

_PLOTPY_TEXT_ANCHORS = {value: key for key, value in _ANCHORS.items()}


@dataclass(frozen=True)
class PlotPyMigrationReport:
    """Result of an explicit historical PlotPy migration."""

    converted_count: int
    preserved_count: int
    diagnostics: tuple[str, ...]
    applied: bool


class AxesLabelItem(LabelItem):
    """PlotPy label positioned in normalized axes coordinates."""

    def __init__(self, text: str, position: tuple[float, float], labelparam) -> None:
        super().__init__(text, labelparam)
        self.axes_position = position

    def get_origin(self, xMap, yMap, canvasRect) -> tuple[float, float]:
        """Return the normalized axes position in canvas coordinates."""
        x, y = self.axes_position
        return (
            canvasRect.left() + x * canvasRect.width(),
            canvasRect.bottom() - y * canvasRect.height(),
        )


def _shape_corners(annotation: RectangleAnnotation) -> np.ndarray:
    """Return oriented rectangle corners in PlotPy order."""
    half_width = annotation.width / 2
    half_height = annotation.height / 2
    corners = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ]
    )
    cos_angle = math.cos(annotation.angle)
    sin_angle = math.sin(annotation.angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    return corners @ rotation.T + np.array([annotation.x, annotation.y])


def _annotation_options(annotation: GraphicalAnnotation) -> dict[str, Any]:
    """Return common PlotPy builder options for an annotated shape."""
    label = annotation.label
    return {
        "title": annotation.title,
        "show_label": bool(label and label.visible and label.text),
        "show_computations": bool(label and label.visible and label.text),
        "show_subtitle": False,
        "readonly": annotation.locked,
    }


def _configure_shape(item: AnnotatedShape, annotation: GraphicalAnnotation) -> None:
    """Apply common canonical state and style to a PlotPy shape item."""
    item.setVisible(annotation.visible)
    item.setZ(annotation.z_index)
    item.set_readonly(annotation.locked)
    item.setTitle(annotation.title)
    if annotation.label is not None and annotation.label.text:
        text = annotation.label.text
        item.set_info_callback(lambda _item, value=text: value)
        item.set_label_visible(annotation.label.visible)

    shape_param = item.shape.shapeparam
    stroke = annotation.style.stroke
    fill = annotation.style.fill
    marker = annotation.style.marker
    shape_param.line.color = stroke.color or "#000000"
    shape_param.line.width = stroke.width
    shape_param.line.style = _LINE_STYLES.get(
        stroke.dash if isinstance(stroke.dash, str) else "solid", "SolidLine"
    )
    if isinstance(item, (AnnotatedXRange, AnnotatedYRange)):
        shape_param.fill = fill.color or "#000000"
        shape_param.shade = fill.opacity if fill.color is not None else 0.0
    else:
        shape_param.fill.color = fill.color or "#000000"
        shape_param.fill.alpha = fill.opacity if fill.color is not None else 0.0
        shape_param.fill.style = "SolidPattern" if fill.color is not None else "NoBrush"
    shape_param.symbol.marker = _MARKERS.get(marker.symbol, marker.symbol)
    shape_param.symbol.size = round(marker.size)
    shape_param.symbol.edgecolor = marker.color or stroke.color or "#000000"
    shape_param.symbol.facecolor = marker.color or stroke.color or "#000000"
    shape_param.symbol.alpha = stroke.opacity
    if hasattr(shape_param, "readonly"):
        shape_param.readonly = annotation.locked
    shape_param.update_item(item.shape)

    for pen in (item.shape.pen, item.shape.sel_pen):
        color = pen.color()
        color.setAlphaF(stroke.opacity)
        pen.setColor(color)
        if isinstance(stroke.dash, tuple):
            pen.setStyle(QC.Qt.CustomDashLine)
            pen.setDashPattern(list(stroke.dash))


def _configure_label(item: LabelItem, annotation: TextAnnotation) -> None:
    """Apply canonical state and text style to a PlotPy label item."""
    item.setVisible(annotation.visible)
    item.setZ(annotation.z_index)
    item.set_readonly(annotation.locked)
    item.setTitle(annotation.title)
    param = item.labelparam
    style = annotation.style.text
    param.font.family = style.family or param.font.family
    param.font.size = round(style.size)
    param.font.bold = style.bold
    param.font.italic = style.italic
    param.color = style.color
    param.bgcolor = style.background_color or "#ffffff"
    param.bgalpha = style.background_opacity
    param.update_item(item)


def annotation_to_plotpy_item(annotation: GraphicalAnnotation):
    """Convert one canonical annotation to a native PlotPy item."""
    options = _annotation_options(annotation)
    item: Any
    if isinstance(annotation, PointAnnotation):
        item = make.annotated_point(annotation.x, annotation.y, **options)
    elif isinstance(annotation, SegmentAnnotation):
        item = make.annotated_segment(
            annotation.x0, annotation.y0, annotation.x1, annotation.y1, **options
        )
    elif isinstance(annotation, RectangleAnnotation):
        corners = _shape_corners(annotation)
        if math.isclose(annotation.angle, 0.0, abs_tol=1e-12):
            item = make.annotated_rectangle(
                corners[0, 0],
                corners[0, 1],
                corners[2, 0],
                corners[2, 1],
                **options,
            )
        else:
            item = AnnotatedObliqueRectangle(*corners.ravel())
    elif isinstance(annotation, CircleAnnotation):
        item = make.annotated_circle(
            annotation.cx - annotation.radius,
            annotation.cy,
            annotation.cx + annotation.radius,
            annotation.cy,
            **options,
        )
    elif isinstance(annotation, EllipseAnnotation):
        cos_angle = math.cos(annotation.angle)
        sin_angle = math.sin(annotation.angle)
        dx = annotation.radius_x * cos_angle
        dy = annotation.radius_x * sin_angle
        ex = -annotation.radius_y * sin_angle
        ey = annotation.radius_y * cos_angle
        item = make.annotated_ellipse(
            annotation.cx - dx,
            annotation.cy - dy,
            annotation.cx + dx,
            annotation.cy + dy,
            annotation.cx - ex,
            annotation.cy - ey,
            annotation.cx + ex,
            annotation.cy + ey,
            **options,
        )
    elif isinstance(annotation, (PolylineAnnotation, PolygonAnnotation)):
        item = make.annotated_polygon(np.asarray(annotation.points), **options)
        item.set_closed(isinstance(annotation, PolygonAnnotation))
    elif isinstance(annotation, TextAnnotation):
        anchor = _ANCHORS[annotation.anchor]
        offset = tuple(round(value) for value in annotation.offset)
        if annotation.coordinate_space.value == "data":
            item = make.label(
                annotation.text,
                (annotation.x, annotation.y),
                offset,
                anchor,
                title=annotation.title,
            )
        else:
            template = make.label(
                annotation.text, "TL", offset, anchor, title=annotation.title
            )
            item = AxesLabelItem(
                annotation.text,
                (annotation.x, annotation.y),
                template.labelparam,
            )
        _configure_label(item, annotation)
        return item
    elif isinstance(annotation, CursorAnnotation):
        if annotation.orientation == CursorOrientation.CROSSHAIR:
            assert isinstance(annotation.position, tuple)
            position = annotation.position
            markerstyle = "+"
        elif annotation.orientation == CursorOrientation.VERTICAL:
            assert isinstance(annotation.position, float)
            position = (annotation.position, 0.0)
            markerstyle = "|"
        else:
            assert isinstance(annotation.position, float)
            position = (0.0, annotation.position)
            markerstyle = "-"
        stroke = annotation.style.stroke
        item = make.marker(
            position=position,
            markerstyle=markerstyle,
            movable=not annotation.locked,
            readonly=annotation.locked,
            color=stroke.color,
            linewidth=stroke.width,
        )
        item.setVisible(annotation.visible)
        item.setZ(annotation.z_index)
        item.setTitle(annotation.title)
        return item
    elif isinstance(annotation, RangeAnnotation):
        builder = (
            make.annotated_xrange
            if annotation.axis == Axis.X
            else make.annotated_yrange
        )
        item = builder(annotation.start, annotation.end, **options)
    else:  # pragma: no cover - protected by the closed model hierarchy
        raise TypeError(f"Unsupported annotation type: {type(annotation).__name__}")
    _configure_shape(item, annotation)
    return item


def annotations_to_plotpy_items(
    annotations: list[GraphicalAnnotation],
) -> list[Any]:
    """Convert canonical annotations to PlotPy items in layer order."""
    return [
        annotation_to_plotpy_item(annotation)
        for annotation in sorted(annotations, key=lambda item: item.z_index)
    ]


def _load_legacy_range(payload: dict[str, Any]):
    """Load a historical range affected by PlotPy's deserialize ordering bug."""
    document = json.loads(payload["plotpy_json"])
    item_keys = document.get("plot_items")
    if not isinstance(item_keys, list) or len(item_keys) != 1:
        raise ValueError("Expected one PlotPy range item")
    item_key = item_keys[0]
    item_data = document[item_key]
    if item_key.startswith("AnnotatedXRange"):
        builder = make.annotated_xrange
    elif item_key.startswith("AnnotatedYRange"):
        builder = make.annotated_yrange
    else:
        raise ValueError("Payload is not an annotated PlotPy range")
    annotation_param = item_data.get("annotationparam", {})
    item = builder(
        item_data["min"],
        item_data["max"],
        title=annotation_param.get("title"),
        show_label=annotation_param.get("show_label"),
        show_computations=annotation_param.get("show_computations"),
        show_subtitle=annotation_param.get("show_subtitle"),
        readonly=annotation_param.get("readonly"),
        private=annotation_param.get("private"),
    )
    shape_data = item_data.get("shapeparam", {})
    shape_param = item.shape.shapeparam
    for name in ("line", "sel_line"):
        line_data = shape_data.get(name, {})
        line_param = getattr(shape_param, name)
        line_param.style = line_data.get("style", line_param.style)
        line_param.color = line_data.get("color", line_param.color)
        line_param.width = line_data.get("width", line_param.width)
    shape_param.fill = shape_data.get("fill", shape_param.fill)
    shape_param.shade = shape_data.get("shade", shape_param.shade)
    shape_param.update_item(item.shape)
    item.setVisible(item_data.get("visible", True))
    return item


def _load_legacy_plotpy_payload(payload: dict[str, Any]) -> list[Any]:
    """Load one historical payload, including known PlotPy range defects."""
    try:
        return load_items(JSONReader(payload["plotpy_json"]))
    except TypeError:
        return [_load_legacy_range(payload)]


def load_legacy_plotpy_items(obj: BaseObj) -> list[Any]:
    """Load historical PlotPy payloads without mutating the object."""
    items = []
    for payload in obj.get_annotations():
        if is_graphical_annotation_dict(payload) or "plotpy_json" not in payload:
            continue
        try:
            items.extend(_load_legacy_plotpy_payload(payload))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return items


def _legacy_title(item: AnnotatedShape) -> str:
    """Return an item's persisted title as plain text."""
    return str(item.title().text())


def _legacy_style(item: AnnotatedShape) -> AnnotationStyle:
    """Convert the portable part of a PlotPy annotated-shape style."""
    pen = item.shape.pen
    brush = item.shape.brush
    symbol = item.shape.symbol
    pen_color = pen.color()
    brush_color = brush.color()
    symbol_pen = symbol.pen()
    symbol_brush = symbol.brush()
    return AnnotationStyle(
        stroke=StrokeStyle(
            color=pen_color.name(),
            width=pen.widthF(),
            opacity=pen_color.alphaF(),
        ),
        fill=FillStyle(
            color=brush_color.name() if brush_color.alphaF() > 0 else None,
            opacity=brush_color.alphaF(),
        ),
        marker=MarkerStyle(
            symbol="circle",
            size=symbol.size().width(),
            color=(
                symbol_brush.color().name()
                if symbol_brush.color().alphaF() > 0
                else symbol_pen.color().name()
            ),
        ),
    )


def _legacy_common(item: Any) -> dict[str, Any]:
    """Return canonical fields shared by migrated PlotPy items."""
    common = {
        "title": str(item.title().text()),
        "visible": item.isVisible(),
        "locked": item.is_readonly(),
        "z_index": round(item.z()),
        "extensions": {"plotpy": {"item_class": type(item).__name__}},
    }
    if isinstance(item, AnnotatedShape):
        common["style"] = _legacy_style(item)
        title = str(item.title().text())
        if title:
            common["label"] = AnnotationLabel(
                text=title, visible=item.is_label_visible()
            )
    return common


def plotpy_item_to_annotation(item: Any) -> GraphicalAnnotation | None:
    """Convert a known historical PlotPy item to a canonical annotation."""
    common = _legacy_common(item)
    annotation = None
    if isinstance(item, AnnotatedPoint):
        x, y = item.get_pos()
        annotation = PointAnnotation(x=x, y=y, **common)
    elif isinstance(item, AnnotatedSegment):
        x0, y0, x1, y1 = item.get_rect()
        annotation = SegmentAnnotation(x0=x0, y0=y0, x1=x1, y1=y1, **common)
    elif isinstance(item, AnnotatedObliqueRectangle):
        points = np.asarray(item.shape.points)
        center = points.mean(axis=0)
        edge_x = points[1] - points[0]
        edge_y = points[3] - points[0]
        annotation = RectangleAnnotation(
            x=center[0],
            y=center[1],
            width=np.linalg.norm(edge_x),
            height=np.linalg.norm(edge_y),
            angle=math.atan2(edge_x[1], edge_x[0]),
            **common,
        )
    elif isinstance(item, AnnotatedRectangle):
        x0, y0, x1, y1 = item.get_rect()
        annotation = RectangleAnnotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            width=abs(x1 - x0),
            height=abs(y1 - y0),
            **common,
        )
    elif isinstance(item, AnnotatedCircle):
        x0, y0, x1, y1 = item.get_xdiameter()
        annotation = CircleAnnotation(
            cx=(x0 + x1) / 2,
            cy=(y0 + y1) / 2,
            radius=math.hypot(x1 - x0, y1 - y0) / 2,
            **common,
        )
    elif isinstance(item, AnnotatedEllipse):
        x0, y0, x1, y1 = item.get_xdiameter()
        x2, y2, x3, y3 = item.get_ydiameter()
        annotation = EllipseAnnotation(
            cx=(x0 + x1) / 2,
            cy=(y0 + y1) / 2,
            radius_x=math.hypot(x1 - x0, y1 - y0) / 2,
            radius_y=math.hypot(x3 - x2, y3 - y2) / 2,
            angle=math.atan2(y1 - y0, x1 - x0),
            **common,
        )
    elif isinstance(item, AnnotatedPolygon):
        points = tuple(map(tuple, item.get_points()))
        annotation_class = PolygonAnnotation if item.is_closed() else PolylineAnnotation
        annotation = annotation_class(points=points, **common)
    elif isinstance(item, (AnnotatedXRange, AnnotatedYRange)):
        start, end = item.get_range()
        axis = Axis.X if isinstance(item, AnnotatedXRange) else Axis.Y
        annotation = RangeAnnotation(axis=axis, start=start, end=end, **common)
    elif isinstance(item, LabelItem):
        if item.G in _PLOTPY_ANCHOR_POSITIONS:
            x, y = _PLOTPY_ANCHOR_POSITIONS[item.G]
            coordinate_space = "axes"
        elif isinstance(item.G, tuple):
            x, y = item.G
            coordinate_space = "data"
        else:
            return None
        param = item.labelparam
        style = AnnotationStyle(
            text=TextStyle(
                family=param.font.family,
                size=param.font.size,
                bold=param.font.bold,
                italic=param.font.italic,
                color=param.color,
                background_color=param.bgcolor,
                background_opacity=param.bgalpha,
            )
        )
        annotation = TextAnnotation(
            text=item.get_plain_text(),
            x=x,
            y=y,
            coordinate_space=coordinate_space,
            anchor=_PLOTPY_TEXT_ANCHORS.get(item.anchor, TextAnchor.TOP_LEFT),
            offset=tuple(item.C),
            style=style,
            **common,
        )
    elif isinstance(item, Marker):
        line_style = item.lineStyle()
        if line_style == QwtPlotMarker.VLine:
            orientation = CursorOrientation.VERTICAL
            position = item.xValue()
        elif line_style == QwtPlotMarker.HLine:
            orientation = CursorOrientation.HORIZONTAL
            position = item.yValue()
        elif line_style == QwtPlotMarker.Cross:
            orientation = CursorOrientation.CROSSHAIR
            position = (item.xValue(), item.yValue())
        else:
            return PointAnnotation(x=item.xValue(), y=item.yValue(), **common)
        annotation = CursorAnnotation(
            orientation=orientation, position=position, **common
        )
    return annotation


def migrate_legacy_plotpy_annotations(
    obj: BaseObj, *, dry_run: bool = False
) -> PlotPyMigrationReport:
    """Replace fully recognized historical PlotPy payloads with canonical data.

    Unknown, malformed, or only partially supported payloads remain byte-for-byte
    represented by their original dictionary.

    Args:
        obj: Object containing historical PlotPy payloads.
        dry_run: If True, inspect migration without modifying the object.

    Returns:
        Structured migration report.
    """
    migrated_count = 0
    preserved_count = 0
    diagnostics = []
    output = []
    for payload in obj.get_annotations():
        if is_graphical_annotation_dict(payload) or "plotpy_json" not in payload:
            output.append(payload)
            continue
        try:
            items = _load_legacy_plotpy_payload(payload)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            output.append(payload)
            preserved_count += 1
            diagnostics.append(
                f"Malformed PlotPy payload preserved: {type(exc).__name__}: {exc}"
            )
            continue
        converted = [plotpy_item_to_annotation(item) for item in items]
        if not items or any(annotation is None for annotation in converted):
            output.append(payload)
            preserved_count += 1
            class_names = ", ".join(type(item).__name__ for item in items) or "empty"
            diagnostics.append(f"Unsupported PlotPy payload preserved: {class_names}")
            continue
        output.extend(annotation_to_dict(annotation) for annotation in converted)
        migrated_count += len(converted)
    applied = bool(migrated_count and not dry_run)
    if applied:
        obj.set_annotations(output)
    return PlotPyMigrationReport(
        converted_count=migrated_count,
        preserved_count=preserved_count,
        diagnostics=tuple(diagnostics),
        applied=applied,
    )
