# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Migration of historical PlotPy annotations to the canonical model."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sigima.objects.annotations.model import (
    AnnotationLabel,
    AnnotationStyle,
    Axis,
    CircleAnnotation,
    CoordinateSpace,
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
)
from sigima.objects.annotations.serialization import annotation_to_dict

if TYPE_CHECKING:
    from sigima.objects.base import BaseObj

__all__ = [
    "LegacyPlotPyMigrationReport",
    "legacy_plotpy_payload_to_annotations",
    "migrate_legacy_plotpy_annotations",
]


_ANCHORS = {
    "TL": TextAnchor.TOP_LEFT,
    "T": TextAnchor.TOP,
    "TR": TextAnchor.TOP_RIGHT,
    "L": TextAnchor.LEFT,
    "C": TextAnchor.CENTER,
    "R": TextAnchor.RIGHT,
    "BL": TextAnchor.BOTTOM_LEFT,
    "B": TextAnchor.BOTTOM,
    "BR": TextAnchor.BOTTOM_RIGHT,
}

_AXES_POSITIONS = {
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

_DASH_STYLES = {
    "SolidLine": "solid",
    "DashLine": "dash",
    "DotLine": "dot",
    "DashDotLine": "dashdot",
    "DashDotDotLine": (6.0, 3.0, 1.0, 3.0, 1.0, 3.0),
}

_MARKER_SYMBOLS = {
    "NoSymbol": "none",
    "Ellipse": "circle",
    "Rect": "square",
    "Diamond": "diamond",
    "Cross": "x",
    "Plus": "cross",
    "TriangleUp": "triangle-up",
    "TriangleDown": "triangle-down",
    "TriangleLeft": "triangle-left",
    "TriangleRight": "triangle-right",
    "Star1": "star",
    "Star2": "asterisk",
}


@dataclass(frozen=True)
class LegacyPlotPyMigrationReport:
    """Summary of one legacy annotation migration attempt."""

    converted_count: int
    preserved_count: int
    diagnostics: tuple[str, ...]
    applied: bool


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return *value* as a mapping, or an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def _float(value: Any, name: str) -> float:
    """Return a finite floating-point value."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _array(value: Any, name: str) -> Sequence[Any]:
    """Unwrap guidata's JSON array representation."""
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
        and value[0] == "array"
    ):
        value = value[1]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return value


def _points(item: Mapping[str, Any], minimum: int) -> list[tuple[float, float]]:
    """Return PlotPy points as finite coordinate pairs."""
    result = []
    for index, point in enumerate(_array(item.get("points"), "points")):
        values = _array(point, f"points[{index}]")
        if len(values) != 2:
            raise ValueError(f"points[{index}] must contain two coordinates")
        result.append(
            (
                _float(values[0], f"points[{index}].x"),
                _float(values[1], f"points[{index}].y"),
            )
        )
    if len(result) < minimum:
        raise ValueError(f"points must contain at least {minimum} entries")
    return result


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return the Euclidean distance between two points."""
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _center(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float]:
    """Return the midpoint of two points."""
    return (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0


def _anchor(value: Any) -> TextAnchor:
    """Convert a PlotPy anchor name to the canonical enum."""
    return _ANCHORS.get(str(value), TextAnchor.TOP_LEFT)


def _line_style(parameters: Mapping[str, Any]) -> StrokeStyle:
    """Convert a PlotPy line parameter group."""
    line = _mapping(parameters.get("line"))
    style = str(line.get("style", "SolidLine"))
    if style in ("NoPen", "NoLine"):
        return StrokeStyle(color=None, width=0.0, opacity=0.0)
    return StrokeStyle(
        color=str(line.get("color", "#ff9933")),
        width=_float(line.get("width", 1.0), "line width"),
        dash=_DASH_STYLES.get(style, "solid"),
    )


def _fill_style(parameters: Mapping[str, Any]) -> FillStyle:
    """Convert PlotPy shape or range fill parameters."""
    fill = parameters.get("fill")
    if isinstance(fill, Mapping):
        style = str(fill.get("style", "NoBrush"))
        opacity = (
            0.0
            if style in ("NoBrush", "NoPattern")
            else _float(fill.get("alpha", 1.0), "fill opacity")
        )
        return FillStyle(color=str(fill.get("color", "#000000")), opacity=opacity)
    if isinstance(fill, str):
        return FillStyle(
            color=fill,
            opacity=_float(parameters.get("shade", 0.0), "range opacity"),
        )
    return FillStyle()


def _marker_style(parameters: Mapping[str, Any]) -> MarkerStyle:
    """Convert a PlotPy symbol parameter group."""
    symbol = _mapping(parameters.get("symbol"))
    name = str(symbol.get("marker", "NoSymbol"))
    return MarkerStyle(
        symbol=_MARKER_SYMBOLS.get(name, "circle"),
        size=_float(symbol.get("size", 6.0), "marker size"),
        color=str(symbol.get("facecolor", symbol.get("edgecolor", "#ff9933"))),
    )


def _text_style(label_parameters: Mapping[str, Any]) -> TextStyle:
    """Convert PlotPy label parameters."""
    font = _mapping(label_parameters.get("font"))
    family = font.get("family")
    if family in (None, "", "default"):
        family = None
    return TextStyle(
        family=str(family) if family is not None else None,
        size=_float(font.get("size", 10.0), "font size"),
        bold=bool(font.get("bold", False)),
        italic=bool(font.get("italic", False)),
        color=str(label_parameters.get("color", "#000000")),
        background_color=str(label_parameters.get("bgcolor", "#ffffff")),
        background_opacity=_float(
            label_parameters.get("bgalpha", 0.0), "background opacity"
        ),
    )


def _style(item: Mapping[str, Any]) -> AnnotationStyle:
    """Convert the style groups shared by PlotPy item families."""
    parameters = _mapping(item.get("shapeparam"))
    if not parameters:
        parameters = _mapping(item.get("markerparam"))
    label_parameters = _mapping(item.get("labelparam"))
    if not label_parameters:
        label_parameters = _mapping(parameters.get("text"))
        if label_parameters:
            font = _mapping(label_parameters.get("font"))
            label_parameters = {
                "font": font,
                "color": label_parameters.get("textcolor", "#000000"),
                "bgcolor": label_parameters.get("background_color", "#ffffff"),
                "bgalpha": label_parameters.get("background_alpha", 0.0),
            }
    return AnnotationStyle(
        stroke=_line_style(parameters),
        fill=_fill_style(parameters),
        marker=_marker_style(parameters),
        text=_text_style(label_parameters),
    )


def _label(item: Mapping[str, Any]) -> AnnotationLabel | None:
    """Return the label attached to an annotated PlotPy shape."""
    text = item.get("text")
    if not isinstance(text, str) or not text:
        return None
    annotation_parameters = _mapping(item.get("annotationparam"))
    label_parameters = _mapping(item.get("labelparam"))
    return AnnotationLabel(
        text=text,
        visible=bool(annotation_parameters.get("show_label", True)),
        anchor=_anchor(label_parameters.get("anchor", "TL")),
        offset=(
            _float(label_parameters.get("xc", 0.0), "label x offset"),
            _float(label_parameters.get("yc", 0.0), "label y offset"),
        ),
    )


def _common(item: Mapping[str, Any], item_class: str) -> dict[str, Any]:
    """Return canonical fields shared by migrated items."""
    annotation_parameters = _mapping(item.get("annotationparam"))
    shape_parameters = _mapping(item.get("shapeparam"))
    title = annotation_parameters.get("title") or shape_parameters.get("label") or ""
    return {
        "visible": bool(item.get("visible", True)),
        "locked": bool(
            annotation_parameters.get("readonly", False)
            or shape_parameters.get("readonly", False)
        ),
        "z_index": int(round(_float(item.get("z", 0.0), "z index"))),
        "title": str(title),
        "style": _style(item),
        "label": _label(item),
        "extensions": {"plotpy": {"item_class": item_class}},
    }


def _item_to_annotation(
    item_class: str, item: Mapping[str, Any]
) -> GraphicalAnnotation:
    """Convert one decoded PlotPy item to a canonical annotation."""
    common = _common(item, item_class)
    if item_class == "AnnotatedPoint":
        point = _points(item, 1)[0]
        annotation = PointAnnotation(x=point[0], y=point[1], **common)
    elif item_class == "AnnotatedSegment":
        points = _points(item, 2)
        first, second = points[0], points[1]
        annotation = SegmentAnnotation(
            x0=first[0], y0=first[1], x1=second[0], y1=second[1], **common
        )
    elif item_class in ("AnnotatedRectangle", "AnnotatedObliqueRectangle"):
        points = _points(item, 4)
        center = _center(points[0], points[2])
        annotation = RectangleAnnotation(
            x=center[0],
            y=center[1],
            width=_distance(points[0], points[1]),
            height=_distance(points[1], points[2]),
            angle=math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]),
            **common,
        )
    elif item_class == "AnnotatedCircle":
        points = _points(item, 2)
        first, second = points[0], points[1]
        center = _center(first, second)
        annotation = CircleAnnotation(
            cx=center[0], cy=center[1], radius=_distance(first, second) / 2.0, **common
        )
    elif item_class == "AnnotatedEllipse":
        points = _points(item, 4)
        first, second, third, fourth = points[0], points[1], points[2], points[3]
        center = _center(first, second)
        annotation = EllipseAnnotation(
            cx=center[0],
            cy=center[1],
            radius_x=_distance(first, second) / 2.0,
            radius_y=_distance(third, fourth) / 2.0,
            angle=math.atan2(first[1] - second[1], first[0] - second[0]),
            **common,
        )
    elif item_class == "AnnotatedPolygon":
        points = _points(item, 2)
        if bool(item.get("closed", True)):
            annotation = PolygonAnnotation(points=tuple(points), **common)
        else:
            annotation = PolylineAnnotation(points=tuple(points), **common)
    elif item_class == "LabelItem":
        label_parameters = _mapping(item.get("labelparam"))
        common["label"] = None
        common["title"] = str(label_parameters.get("label", ""))
        if bool(label_parameters.get("abspos", False)):
            axes_position = _AXES_POSITIONS.get(
                str(label_parameters.get("absg", "TL")), (0.0, 1.0)
            )
            x, y = axes_position[0], axes_position[1]
            coordinate_space = CoordinateSpace.AXES
        else:
            x = _float(label_parameters.get("xg", 0.0), "label x")
            y = _float(label_parameters.get("yg", 0.0), "label y")
            coordinate_space = CoordinateSpace.DATA
        annotation = TextAnnotation(
            text=str(item.get("text", label_parameters.get("contents", ""))),
            x=x,
            y=y,
            coordinate_space=coordinate_space,
            anchor=_anchor(label_parameters.get("anchor", "TL")),
            offset=(
                _float(label_parameters.get("xc", 0.0), "label x offset"),
                _float(label_parameters.get("yc", 0.0), "label y offset"),
            ),
            **common,
        )
    elif item_class == "Marker":
        marker_parameters = _mapping(item.get("markerparam"))
        marker_style = str(marker_parameters.get("markerstyle", "NoLine"))
        x = _float(item.get("x", 0.0), "marker x")
        y = _float(item.get("y", 0.0), "marker y")
        common["label"] = None
        if marker_style == "VLine":
            annotation = CursorAnnotation(
                orientation=CursorOrientation.VERTICAL, position=x, **common
            )
        elif marker_style == "HLine":
            annotation = CursorAnnotation(
                orientation=CursorOrientation.HORIZONTAL, position=y, **common
            )
        elif marker_style == "Cross":
            annotation = CursorAnnotation(
                orientation=CursorOrientation.CROSSHAIR, position=(x, y), **common
            )
        else:
            annotation = PointAnnotation(x=x, y=y, **common)
    elif item_class in ("AnnotatedXRange", "XRangeSelection"):
        annotation = RangeAnnotation(
            axis=Axis.X,
            start=_float(item.get("min"), "range minimum"),
            end=_float(item.get("max"), "range maximum"),
            **common,
        )
    elif item_class in ("AnnotatedYRange", "YRangeSelection"):
        annotation = RangeAnnotation(
            axis=Axis.Y,
            start=_float(item.get("min"), "range minimum"),
            end=_float(item.get("max"), "range maximum"),
            **common,
        )
    else:
        raise ValueError(f"unsupported PlotPy annotation class {item_class!r}")
    return annotation


def _item_class(key: str, fallback: Any = None) -> str:
    """Return the PlotPy class name encoded in a JSON item key."""
    match = re.fullmatch(r"(.+)_\d+", key)
    if match is not None:
        return match.group(1)
    if isinstance(fallback, str) and fallback:
        return fallback
    return key


def legacy_plotpy_payload_to_annotations(
    payload: Mapping[str, Any],
) -> list[GraphicalAnnotation]:
    """Convert one historical ``plotpy_json`` payload without importing PlotPy."""
    if not isinstance(payload, Mapping):
        raise TypeError("legacy PlotPy payload must be a mapping")
    json_text = payload.get("plotpy_json")
    if not isinstance(json_text, str):
        raise ValueError("legacy PlotPy payload has no JSON string")
    document = json.loads(json_text)
    if not isinstance(document, Mapping):
        raise ValueError("legacy PlotPy JSON root must be an object")
    item_keys = document.get("plot_items")
    if not isinstance(item_keys, list):
        item_keys = [key for key in document if key != "plot_items"]
    annotations = []
    for key in item_keys:
        if not isinstance(key, str):
            raise ValueError("legacy PlotPy item key must be a string")
        item = document.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"legacy PlotPy item {key!r} must be an object")
        annotations.append(
            _item_to_annotation(_item_class(key, payload.get("item_class")), item)
        )
    if not annotations:
        raise ValueError("legacy PlotPy payload contains no items")
    return annotations


def migrate_legacy_plotpy_annotations(
    obj: BaseObj, *, dry_run: bool = False
) -> LegacyPlotPyMigrationReport:
    """Replace supported historical PlotPy payloads on *obj* with canonical ones."""
    stored = obj.get_annotations()
    migrated: list[dict[str, Any]] = []
    converted_count = 0
    preserved_count = 0
    diagnostics = []
    for index, payload in enumerate(stored):
        if not isinstance(payload, Mapping) or "plotpy_json" not in payload:
            migrated.append(payload)
            preserved_count += 1
            continue
        try:
            annotations = legacy_plotpy_payload_to_annotations(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            migrated.append(payload)
            preserved_count += 1
            diagnostics.append(f"annotation {index}: {exc}")
            continue
        migrated.extend(annotation_to_dict(annotation) for annotation in annotations)
        converted_count += len(annotations)
    applied = converted_count > 0 and not dry_run
    if applied:
        obj.set_annotations(migrated)
    return LegacyPlotPyMigrationReport(
        converted_count=converted_count,
        preserved_count=preserved_count,
        diagnostics=tuple(diagnostics),
        applied=applied,
    )
