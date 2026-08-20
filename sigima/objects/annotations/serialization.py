# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""JSON serialization for renderer-independent graphical annotations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sigima.objects.annotations.model import (
    AnnotationKind,
    AnnotationLabel,
    AnnotationStyle,
    CircleAnnotation,
    CursorAnnotation,
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
    TextAnnotation,
    TextStyle,
)

ANNOTATION_FORMAT = "sigima.annotation"
ANNOTATION_VERSION = "1.0"

_COMMON_FIELDS = {
    "format",
    "version",
    "id",
    "kind",
    "visible",
    "locked",
    "z_index",
    "title",
    "style",
    "label",
    "metadata",
    "extensions",
}

_KIND_FIELDS = {
    AnnotationKind.POINT: {"x", "y"},
    AnnotationKind.SEGMENT: {"x0", "y0", "x1", "y1"},
    AnnotationKind.RECTANGLE: {"x", "y", "width", "height", "angle"},
    AnnotationKind.CIRCLE: {"cx", "cy", "radius"},
    AnnotationKind.ELLIPSE: {"cx", "cy", "radius_x", "radius_y", "angle"},
    AnnotationKind.POLYLINE: {"points"},
    AnnotationKind.POLYGON: {"points"},
    AnnotationKind.TEXT: {
        "text",
        "x",
        "y",
        "coordinate_space",
        "anchor",
        "offset",
    },
    AnnotationKind.CURSOR: {"orientation", "position"},
    AnnotationKind.RANGE: {"axis", "start", "end"},
}

_ANNOTATION_CLASSES = {
    AnnotationKind.POINT: PointAnnotation,
    AnnotationKind.SEGMENT: SegmentAnnotation,
    AnnotationKind.RECTANGLE: RectangleAnnotation,
    AnnotationKind.CIRCLE: CircleAnnotation,
    AnnotationKind.ELLIPSE: EllipseAnnotation,
    AnnotationKind.POLYLINE: PolylineAnnotation,
    AnnotationKind.POLYGON: PolygonAnnotation,
    AnnotationKind.TEXT: TextAnnotation,
    AnnotationKind.CURSOR: CursorAnnotation,
    AnnotationKind.RANGE: RangeAnnotation,
}


def _json_value(value: Any) -> Any:
    """Convert immutable model collections to mutable JSON values."""
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _style_to_dict(style: AnnotationStyle) -> dict[str, Any]:
    """Serialize an annotation style."""
    return {
        "stroke": {
            "color": style.stroke.color,
            "width": style.stroke.width,
            "opacity": style.stroke.opacity,
            "dash": _json_value(style.stroke.dash),
        },
        "fill": {
            "color": style.fill.color,
            "opacity": style.fill.opacity,
        },
        "marker": {
            "symbol": style.marker.symbol,
            "size": style.marker.size,
            "color": style.marker.color,
        },
        "text": {
            "family": style.text.family,
            "size": style.text.size,
            "bold": style.text.bold,
            "italic": style.text.italic,
            "color": style.text.color,
            "background_color": style.text.background_color,
            "background_opacity": style.text.background_opacity,
        },
    }


def _check_keys(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """Reject unknown fields in normalized structures."""
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {path} field(s): {names}")


def _style_from_dict(data: Any) -> AnnotationStyle:
    """Deserialize an annotation style."""
    if not isinstance(data, Mapping):
        raise TypeError("annotation style must be an object")
    _check_keys(data, {"stroke", "fill", "marker", "text"}, "style")

    stroke_data = data.get("stroke", {})
    fill_data = data.get("fill", {})
    marker_data = data.get("marker", {})
    text_data = data.get("text", {})
    for name, value in (
        ("stroke", stroke_data),
        ("fill", fill_data),
        ("marker", marker_data),
        ("text", text_data),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"annotation {name} style must be an object")

    _check_keys(stroke_data, {"color", "width", "opacity", "dash"}, "stroke")
    _check_keys(fill_data, {"color", "opacity"}, "fill")
    _check_keys(marker_data, {"symbol", "size", "color"}, "marker")
    _check_keys(
        text_data,
        {
            "family",
            "size",
            "bold",
            "italic",
            "color",
            "background_color",
            "background_opacity",
        },
        "text",
    )
    return AnnotationStyle(
        stroke=StrokeStyle(**stroke_data),
        fill=FillStyle(**fill_data),
        marker=MarkerStyle(**marker_data),
        text=TextStyle(**text_data),
    )


def _label_to_dict(label: AnnotationLabel) -> dict[str, Any]:
    """Serialize an annotation label."""
    return {
        "text": label.text,
        "visible": label.visible,
        "anchor": label.anchor.value,
        "offset": list(label.offset),
    }


def _label_from_dict(data: Any) -> AnnotationLabel | None:
    """Deserialize an optional annotation label."""
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise TypeError("annotation label must be an object or null")
    _check_keys(data, {"text", "visible", "anchor", "offset"}, "label")
    return AnnotationLabel(**data)


def annotation_to_dict(annotation: GraphicalAnnotation) -> dict[str, Any]:
    """Serialize a graphical annotation to a JSON-compatible dictionary."""
    if not isinstance(annotation, GraphicalAnnotation):
        raise TypeError("annotation must be a GraphicalAnnotation")
    data = {
        "format": ANNOTATION_FORMAT,
        "version": ANNOTATION_VERSION,
        "id": annotation.id,
        "kind": annotation.kind.value,
        "visible": annotation.visible,
        "locked": annotation.locked,
        "z_index": annotation.z_index,
        "title": annotation.title,
        "style": _style_to_dict(annotation.style),
        "label": (
            _label_to_dict(annotation.label) if annotation.label is not None else None
        ),
        "metadata": _json_value(annotation.metadata),
        "extensions": _json_value(annotation.extensions),
    }
    if isinstance(annotation, PointAnnotation):
        data.update(x=annotation.x, y=annotation.y)
    elif isinstance(annotation, SegmentAnnotation):
        data.update(
            x0=annotation.x0,
            y0=annotation.y0,
            x1=annotation.x1,
            y1=annotation.y1,
        )
    elif isinstance(annotation, RectangleAnnotation):
        data.update(
            x=annotation.x,
            y=annotation.y,
            width=annotation.width,
            height=annotation.height,
            angle=annotation.angle,
        )
    elif isinstance(annotation, CircleAnnotation):
        data.update(cx=annotation.cx, cy=annotation.cy, radius=annotation.radius)
    elif isinstance(annotation, EllipseAnnotation):
        data.update(
            cx=annotation.cx,
            cy=annotation.cy,
            radius_x=annotation.radius_x,
            radius_y=annotation.radius_y,
            angle=annotation.angle,
        )
    elif isinstance(annotation, (PolylineAnnotation, PolygonAnnotation)):
        data["points"] = [list(point) for point in annotation.points]
    elif isinstance(annotation, TextAnnotation):
        data.update(
            text=annotation.text,
            x=annotation.x,
            y=annotation.y,
            coordinate_space=annotation.coordinate_space.value,
            anchor=annotation.anchor.value,
            offset=list(annotation.offset),
        )
    elif isinstance(annotation, CursorAnnotation):
        data.update(
            orientation=annotation.orientation.value,
            position=_json_value(annotation.position),
        )
    elif isinstance(annotation, RangeAnnotation):
        data.update(
            axis=annotation.axis.value, start=annotation.start, end=annotation.end
        )
    else:  # pragma: no cover - protected by the closed model hierarchy
        raise TypeError(f"Unsupported annotation type: {type(annotation).__name__}")
    return data


def is_graphical_annotation_dict(data: Any) -> bool:
    """Return whether a dictionary declares the canonical annotation format."""
    return isinstance(data, Mapping) and data.get("format") == ANNOTATION_FORMAT


def annotation_from_dict(data: Mapping[str, Any]) -> GraphicalAnnotation:
    """Deserialize and validate a canonical graphical annotation dictionary."""
    if not isinstance(data, Mapping):
        raise TypeError("annotation data must be an object")
    if data.get("format") != ANNOTATION_FORMAT:
        raise ValueError(f"Unsupported annotation format: {data.get('format')!r}")
    if data.get("version") != ANNOTATION_VERSION:
        raise ValueError(f"Unsupported annotation version: {data.get('version')!r}")
    try:
        kind = AnnotationKind(data["kind"])
    except KeyError as exc:
        raise ValueError("Missing annotation kind") from exc
    except ValueError as exc:
        raise ValueError(f"Unsupported annotation kind: {data.get('kind')!r}") from exc
    _check_keys(data, _COMMON_FIELDS | _KIND_FIELDS[kind], "annotation")

    common = {
        "id": data["id"],
        "visible": data.get("visible", True),
        "locked": data.get("locked", False),
        "z_index": data.get("z_index", 0),
        "title": data.get("title", ""),
        "style": _style_from_dict(data.get("style", {})),
        "label": _label_from_dict(data.get("label")),
        "metadata": data.get("metadata", {}),
        "extensions": data.get("extensions", {}),
    }
    geometry = {name: data[name] for name in _KIND_FIELDS[kind]}
    annotation_class = _ANNOTATION_CLASSES[kind]
    return annotation_class(**common, **geometry)
