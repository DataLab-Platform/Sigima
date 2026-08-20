# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Plotly JSON renderer for canonical graphical annotations."""

from __future__ import annotations

import math
from typing import Any

from sigima.objects.annotations import (
    Axis,
    CircleAnnotation,
    CursorAnnotation,
    CursorOrientation,
    EllipseAnnotation,
    GraphicalAnnotation,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    TextAnchor,
    TextAnnotation,
)

__all__ = [
    "annotation_to_plotly_spec",
    "annotations_to_plotly_spec",
]


_ANCHORS = {
    TextAnchor.TOP_LEFT: ("left", "top"),
    TextAnchor.TOP: ("center", "top"),
    TextAnchor.TOP_RIGHT: ("right", "top"),
    TextAnchor.LEFT: ("left", "middle"),
    TextAnchor.CENTER: ("center", "middle"),
    TextAnchor.RIGHT: ("right", "middle"),
    TextAnchor.BOTTOM_LEFT: ("left", "bottom"),
    TextAnchor.BOTTOM: ("center", "bottom"),
    TextAnchor.BOTTOM_RIGHT: ("right", "bottom"),
}

_DASH_STYLES = {
    "-": "solid",
    "solid": "solid",
    "--": "dash",
    "dash": "dash",
    "dashed": "dash",
    ":": "dot",
    "dot": "dot",
    "dotted": "dot",
    "-.": "dashdot",
    "dashdot": "dashdot",
}


def _empty_spec() -> dict[str, list[dict[str, Any]]]:
    """Return an empty Plotly overlay specification."""
    return {"traces": [], "shapes": [], "annotations": []}


def _color_with_opacity(color: str | None, opacity: float) -> str:
    """Return a Plotly color preserving opacity for hexadecimal colors."""
    if color is None:
        return "rgba(0,0,0,0)"
    if opacity == 1.0:
        return color
    value = color.lstrip("#")
    if len(value) in (3, 4):
        value = "".join(character * 2 for character in value)
    if len(value) in (6, 8):
        try:
            red, green, blue = (
                int(value[index : index + 2], 16) for index in (0, 2, 4)
            )
        except ValueError:
            return color
        source_alpha = int(value[6:8], 16) / 255 if len(value) == 8 else 1.0
        return f"rgba({red},{green},{blue},{opacity * source_alpha:.6g})"
    return color


def _dash_style(dash: str | tuple[float, ...]) -> str:
    """Return a Plotly-compatible line dash value."""
    if isinstance(dash, str):
        return _DASH_STYLES.get(dash, dash)
    return ",".join(f"{value:g}px" for value in dash)


def _line_spec(annotation: GraphicalAnnotation) -> dict[str, Any]:
    """Return a Plotly line specification."""
    stroke = annotation.style.stroke
    return {
        "color": _color_with_opacity(stroke.color, stroke.opacity),
        "width": stroke.width,
        "dash": _dash_style(stroke.dash),
    }


def _shape_spec(annotation: GraphicalAnnotation, **geometry: Any) -> dict[str, Any]:
    """Return a styled Plotly shape specification."""
    return {
        **geometry,
        "line": _line_spec(annotation),
        "fillcolor": _color_with_opacity(
            annotation.style.fill.color, annotation.style.fill.opacity
        ),
        "editable": not annotation.locked,
        "visible": annotation.visible,
        "name": annotation.title or annotation.kind.value,
    }


def _path(points: list[tuple[float, float]], closed: bool) -> str:
    """Return a Plotly SVG path using data coordinates."""
    commands = [f"M {points[0][0]:.12g},{points[0][1]:.12g}"]
    commands.extend(f"L {x:.12g},{y:.12g}" for x, y in points[1:])
    if closed:
        commands.append("Z")
    return " ".join(commands)


def _rotated_points(
    center_x: float,
    center_y: float,
    points: list[tuple[float, float]],
    angle: float,
) -> list[tuple[float, float]]:
    """Rotate points around a center by an angle in radians."""
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        (
            center_x + x * cosine - y * sine,
            center_y + x * sine + y * cosine,
        )
        for x, y in points
    ]


def _rectangle_shape(annotation: RectangleAnnotation) -> dict[str, Any]:
    """Return a Plotly shape for a possibly rotated rectangle."""
    half_width = annotation.width / 2
    half_height = annotation.height / 2
    if annotation.angle == 0.0:
        return _shape_spec(
            annotation,
            type="rect",
            x0=annotation.x - half_width,
            y0=annotation.y - half_height,
            x1=annotation.x + half_width,
            y1=annotation.y + half_height,
        )
    points = _rotated_points(
        annotation.x,
        annotation.y,
        [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ],
        annotation.angle,
    )
    return _shape_spec(annotation, type="path", path=_path(points, closed=True))


def _ellipse_shape(annotation: EllipseAnnotation) -> dict[str, Any]:
    """Return a Plotly shape for a possibly rotated ellipse."""
    if annotation.angle == 0.0:
        return _shape_spec(
            annotation,
            type="circle",
            x0=annotation.cx - annotation.radius_x,
            y0=annotation.cy - annotation.radius_y,
            x1=annotation.cx + annotation.radius_x,
            y1=annotation.cy + annotation.radius_y,
        )
    points = []
    for index in range(72):
        angle = 2 * math.pi * index / 72
        points.append(
            (
                annotation.radius_x * math.cos(angle),
                annotation.radius_y * math.sin(angle),
            )
        )
    rotated = _rotated_points(annotation.cx, annotation.cy, points, annotation.angle)
    return _shape_spec(annotation, type="path", path=_path(rotated, closed=True))


def _styled_text(annotation: GraphicalAnnotation, text: str) -> str:
    """Return text decorated with Plotly-supported HTML style tags."""
    if annotation.style.text.italic:
        text = f"<i>{text}</i>"
    if annotation.style.text.bold:
        text = f"<b>{text}</b>"
    return text


def _text_spec(
    annotation: GraphicalAnnotation,
    text: str,
    x: float,
    y: float,
    anchor: TextAnchor,
    offset: tuple[float, float],
    xref: str = "x",
    yref: str = "y",
) -> dict[str, Any]:
    """Return a styled Plotly text annotation specification."""
    xanchor, yanchor = _ANCHORS[anchor]
    text_style = annotation.style.text
    font = {
        "size": text_style.size,
        "color": text_style.color,
    }
    if text_style.family is not None:
        font["family"] = text_style.family
    return {
        "text": _styled_text(annotation, text),
        "x": x,
        "y": y,
        "xref": xref,
        "yref": yref,
        "xanchor": xanchor,
        "yanchor": yanchor,
        "xshift": offset[0],
        "yshift": offset[1],
        "showarrow": False,
        "font": font,
        "bgcolor": _color_with_opacity(
            text_style.background_color, text_style.background_opacity
        ),
        "visible": annotation.visible,
        "captureevents": not annotation.locked,
    }


def _label_location(
    annotation: GraphicalAnnotation,
) -> tuple[float, float, str, str] | None:
    """Return the Plotly position and references for an attached label."""
    if isinstance(annotation, PointAnnotation):
        location = (annotation.x, annotation.y, "x", "y")
    elif isinstance(annotation, SegmentAnnotation):
        location = (
            (annotation.x0 + annotation.x1) / 2,
            (annotation.y0 + annotation.y1) / 2,
            "x",
            "y",
        )
    elif isinstance(annotation, RectangleAnnotation):
        location = (annotation.x, annotation.y, "x", "y")
    elif isinstance(annotation, (CircleAnnotation, EllipseAnnotation)):
        location = (annotation.cx, annotation.cy, "x", "y")
    elif isinstance(annotation, (PolylineAnnotation, PolygonAnnotation)):
        location = (
            sum(point[0] for point in annotation.points) / len(annotation.points),
            sum(point[1] for point in annotation.points) / len(annotation.points),
            "x",
            "y",
        )
    elif isinstance(annotation, CursorAnnotation):
        if annotation.orientation == CursorOrientation.CROSSHAIR:
            assert isinstance(annotation.position, tuple)
            location = (*annotation.position, "x", "y")
        else:
            assert isinstance(annotation.position, float)
            if annotation.orientation == CursorOrientation.VERTICAL:
                location = (annotation.position, 1.0, "x", "paper")
            else:
                location = (1.0, annotation.position, "paper", "y")
    elif isinstance(annotation, RangeAnnotation):
        center = (annotation.start + annotation.end) / 2
        if annotation.axis == Axis.X:
            location = (center, 1.0, "x", "paper")
        else:
            location = (1.0, center, "paper", "y")
    else:
        location = None
    return location


def _label_spec(annotation: GraphicalAnnotation) -> dict[str, Any] | None:
    """Return an attached label specification when visible."""
    label = annotation.label
    if label is None or not label.visible or not label.text:
        return None
    location = _label_location(annotation)
    if location is None:
        return None
    x, y, xref, yref = location
    return _text_spec(
        annotation,
        label.text,
        x,
        y,
        label.anchor,
        label.offset,
        xref=xref,
        yref=yref,
    )


def annotation_to_plotly_spec(
    annotation: GraphicalAnnotation,
) -> dict[str, list[dict[str, Any]]]:
    """Convert one canonical annotation to a Plotly JSON overlay specification."""
    spec = _empty_spec()
    if not annotation.visible:
        return spec

    if isinstance(annotation, PointAnnotation):
        marker = annotation.style.marker
        marker_color = marker.color or annotation.style.stroke.color
        trace_marker: dict[str, Any] = {
            "symbol": "circle" if marker.symbol == "none" else marker.symbol,
            "size": marker.size,
            "color": _color_with_opacity(
                marker_color,
                0.0 if marker.symbol == "none" else annotation.style.stroke.opacity,
            ),
            "line": {
                "color": _color_with_opacity(
                    annotation.style.stroke.color,
                    annotation.style.stroke.opacity,
                ),
                "width": annotation.style.stroke.width,
            },
        }
        spec["traces"].append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [annotation.x],
                "y": [annotation.y],
                "marker": trace_marker,
                "name": annotation.title or annotation.kind.value,
                "showlegend": False,
                "meta": {"annotation_id": annotation.id},
            }
        )
    elif isinstance(annotation, SegmentAnnotation):
        spec["shapes"].append(
            _shape_spec(
                annotation,
                type="line",
                x0=annotation.x0,
                y0=annotation.y0,
                x1=annotation.x1,
                y1=annotation.y1,
            )
        )
    elif isinstance(annotation, RectangleAnnotation):
        spec["shapes"].append(_rectangle_shape(annotation))
    elif isinstance(annotation, CircleAnnotation):
        spec["shapes"].append(
            _shape_spec(
                annotation,
                type="circle",
                x0=annotation.cx - annotation.radius,
                y0=annotation.cy - annotation.radius,
                x1=annotation.cx + annotation.radius,
                y1=annotation.cy + annotation.radius,
            )
        )
    elif isinstance(annotation, EllipseAnnotation):
        spec["shapes"].append(_ellipse_shape(annotation))
    elif isinstance(annotation, PolylineAnnotation):
        spec["shapes"].append(
            _shape_spec(
                annotation,
                type="path",
                path=_path(list(annotation.points), closed=False),
            )
        )
    elif isinstance(annotation, PolygonAnnotation):
        spec["shapes"].append(
            _shape_spec(
                annotation,
                type="path",
                path=_path(list(annotation.points), closed=True),
            )
        )
    elif isinstance(annotation, TextAnnotation):
        reference = "paper" if annotation.coordinate_space.value == "axes" else "x"
        y_reference = "paper" if reference == "paper" else "y"
        spec["annotations"].append(
            _text_spec(
                annotation,
                annotation.text,
                annotation.x,
                annotation.y,
                annotation.anchor,
                annotation.offset,
                xref=reference,
                yref=y_reference,
            )
        )
    elif isinstance(annotation, CursorAnnotation):
        positions: list[dict[str, Any]] = []
        if annotation.orientation in (
            CursorOrientation.VERTICAL,
            CursorOrientation.CROSSHAIR,
        ):
            x_position = (
                annotation.position[0]
                if isinstance(annotation.position, tuple)
                else annotation.position
            )
            positions.append(
                {
                    "type": "line",
                    "x0": x_position,
                    "x1": x_position,
                    "y0": 0.0,
                    "y1": 1.0,
                    "xref": "x",
                    "yref": "paper",
                }
            )
        if annotation.orientation in (
            CursorOrientation.HORIZONTAL,
            CursorOrientation.CROSSHAIR,
        ):
            y_position = (
                annotation.position[1]
                if isinstance(annotation.position, tuple)
                else annotation.position
            )
            positions.append(
                {
                    "type": "line",
                    "x0": 0.0,
                    "x1": 1.0,
                    "y0": y_position,
                    "y1": y_position,
                    "xref": "paper",
                    "yref": "y",
                }
            )
        spec["shapes"].extend(
            _shape_spec(annotation, **position) for position in positions
        )
    elif isinstance(annotation, RangeAnnotation):
        if annotation.axis == Axis.X:
            geometry = {
                "type": "rect",
                "x0": annotation.start,
                "x1": annotation.end,
                "y0": 0.0,
                "y1": 1.0,
                "xref": "x",
                "yref": "paper",
            }
        else:
            geometry = {
                "type": "rect",
                "x0": 0.0,
                "x1": 1.0,
                "y0": annotation.start,
                "y1": annotation.end,
                "xref": "paper",
                "yref": "y",
            }
        spec["shapes"].append(_shape_spec(annotation, **geometry))
    else:  # pragma: no cover - protected by the closed model hierarchy
        raise TypeError(f"Unsupported annotation type: {type(annotation).__name__}")

    label = _label_spec(annotation)
    if label is not None:
        spec["annotations"].append(label)
    return spec


def annotations_to_plotly_spec(
    annotations: list[GraphicalAnnotation],
) -> dict[str, list[dict[str, Any]]]:
    """Convert canonical annotations in deterministic layer order."""
    spec = _empty_spec()
    for annotation in sorted(annotations, key=lambda item: item.z_index):
        converted = annotation_to_plotly_spec(annotation)
        for key, values in spec.items():
            values.extend(converted[key])
    return spec
