# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Geometric transformations for graphical annotations."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from sigima.objects.annotations.model import (
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
    TextAnnotation,
)


class AnnotationTransformError(ValueError):
    """Raised when a transformed annotation is not exactly representable."""


def _normalize_angle(angle: float) -> float:
    """Normalize an angle to the [-pi, pi] interval."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _rotate_point(
    x: float, y: float, angle: float, center: tuple[float, float]
) -> tuple[float, float]:
    """Rotate one point around a center."""
    cx, cy = center
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    dx, dy = x - cx, y - cy
    return (
        cx + cos_angle * dx - sin_angle * dy,
        cy + sin_angle * dx + cos_angle * dy,
    )


def _scale_point(
    x: float, y: float, sx: float, sy: float, center: tuple[float, float]
) -> tuple[float, float]:
    """Scale one point around a center."""
    cx, cy = center
    return cx + sx * (x - cx), cy + sy * (y - cy)


def _map_points(annotation: GraphicalAnnotation, function) -> GraphicalAnnotation:
    """Map all explicit points of a point-based annotation."""
    if isinstance(annotation, PointAnnotation):
        x, y = function(annotation.x, annotation.y)
        return replace(annotation, x=x, y=y)
    if isinstance(annotation, SegmentAnnotation):
        x0, y0 = function(annotation.x0, annotation.y0)
        x1, y1 = function(annotation.x1, annotation.y1)
        return replace(annotation, x0=x0, y0=y0, x1=x1, y1=y1)
    if isinstance(annotation, (PolylineAnnotation, PolygonAnnotation)):
        points = tuple(function(x, y) for x, y in annotation.points)
        return replace(annotation, points=points)
    if isinstance(annotation, TextAnnotation):
        if annotation.coordinate_space.value == "axes":
            return annotation
        x, y = function(annotation.x, annotation.y)
        return replace(annotation, x=x, y=y)
    raise TypeError(f"Unsupported point-based annotation: {type(annotation).__name__}")


def _require_quarter_turn(angle: float) -> int:
    """Return a quarter-turn count or raise for an oblique axis primitive."""
    turns = round(angle / (math.pi / 2.0))
    if not math.isclose(angle, turns * math.pi / 2.0, abs_tol=1e-12):
        raise AnnotationTransformError(
            "Axis cursor and range annotations only support quarter-turn rotations"
        )
    return turns % 4


def _rotate_cursor(
    annotation: CursorAnnotation,
    angle: float,
    center: tuple[float, float],
) -> CursorAnnotation:
    """Rotate an axis cursor by a representable quarter turn."""
    turns = _require_quarter_turn(angle)
    cx, cy = center
    if annotation.orientation == CursorOrientation.CROSSHAIR:
        assert isinstance(annotation.position, tuple)
        position = _rotate_point(*annotation.position, angle, center)
        return replace(annotation, position=position)
    if annotation.orientation == CursorOrientation.HORIZONTAL:
        assert isinstance(annotation.position, float)
        point = _rotate_point(cx, annotation.position, angle, center)
        orientation = (
            CursorOrientation.HORIZONTAL
            if turns % 2 == 0
            else CursorOrientation.VERTICAL
        )
    else:
        assert isinstance(annotation.position, float)
        point = _rotate_point(annotation.position, cy, angle, center)
        orientation = (
            CursorOrientation.VERTICAL
            if turns % 2 == 0
            else CursorOrientation.HORIZONTAL
        )
    position = point[1] if orientation == CursorOrientation.HORIZONTAL else point[0]
    return replace(annotation, orientation=orientation, position=position)


def _rotate_range(
    annotation: RangeAnnotation,
    angle: float,
    center: tuple[float, float],
) -> RangeAnnotation:
    """Rotate an axis range by a representable quarter turn."""
    turns = _require_quarter_turn(angle)
    cx, cy = center
    if annotation.axis == Axis.X:
        points = (
            _rotate_point(annotation.start, cy, angle, center),
            _rotate_point(annotation.end, cy, angle, center),
        )
        axis = Axis.X if turns % 2 == 0 else Axis.Y
    else:
        points = (
            _rotate_point(cx, annotation.start, angle, center),
            _rotate_point(cx, annotation.end, angle, center),
        )
        axis = Axis.Y if turns % 2 == 0 else Axis.X
    coordinate = 0 if axis == Axis.X else 1
    return replace(
        annotation,
        axis=axis,
        start=points[0][coordinate],
        end=points[1][coordinate],
    )


def translate_annotation(
    annotation: GraphicalAnnotation, dx: float, dy: float
) -> GraphicalAnnotation:
    """Translate an annotation in data coordinates."""
    if isinstance(
        annotation,
        (PointAnnotation, SegmentAnnotation, PolylineAnnotation, PolygonAnnotation),
    ) or isinstance(annotation, TextAnnotation):
        return _map_points(annotation, lambda x, y: (x + dx, y + dy))
    if isinstance(annotation, (RectangleAnnotation,)):
        return replace(annotation, x=annotation.x + dx, y=annotation.y + dy)
    if isinstance(annotation, (CircleAnnotation, EllipseAnnotation)):
        return replace(annotation, cx=annotation.cx + dx, cy=annotation.cy + dy)
    if isinstance(annotation, CursorAnnotation):
        if annotation.orientation == CursorOrientation.CROSSHAIR:
            assert isinstance(annotation.position, tuple)
            return replace(
                annotation,
                position=(annotation.position[0] + dx, annotation.position[1] + dy),
            )
        assert isinstance(annotation.position, float)
        delta = dy if annotation.orientation == CursorOrientation.HORIZONTAL else dx
        return replace(annotation, position=annotation.position + delta)
    if isinstance(annotation, RangeAnnotation):
        delta = dx if annotation.axis == Axis.X else dy
        return replace(
            annotation,
            start=annotation.start + delta,
            end=annotation.end + delta,
        )
    raise TypeError(f"Unsupported annotation type: {type(annotation).__name__}")


def rotate_annotation(
    annotation: GraphicalAnnotation,
    angle: float,
    center: tuple[float, float] = (0.0, 0.0),
) -> GraphicalAnnotation:
    """Rotate an annotation counterclockwise around a center."""

    def point_transform(x: float, y: float) -> tuple[float, float]:
        return _rotate_point(x, y, angle, center)

    if isinstance(
        annotation,
        (PointAnnotation, SegmentAnnotation, PolylineAnnotation, PolygonAnnotation),
    ) or isinstance(annotation, TextAnnotation):
        return _map_points(annotation, point_transform)
    if isinstance(annotation, RectangleAnnotation):
        x, y = point_transform(annotation.x, annotation.y)
        return replace(
            annotation, x=x, y=y, angle=_normalize_angle(annotation.angle + angle)
        )
    if isinstance(annotation, CircleAnnotation):
        cx, cy = point_transform(annotation.cx, annotation.cy)
        return replace(annotation, cx=cx, cy=cy)
    if isinstance(annotation, EllipseAnnotation):
        cx, cy = point_transform(annotation.cx, annotation.cy)
        return replace(
            annotation,
            cx=cx,
            cy=cy,
            angle=_normalize_angle(annotation.angle + angle),
        )
    if isinstance(annotation, CursorAnnotation):
        return _rotate_cursor(annotation, angle, center)
    if isinstance(annotation, RangeAnnotation):
        return _rotate_range(annotation, angle, center)
    raise TypeError(f"Unsupported annotation type: {type(annotation).__name__}")


def flip_annotation_horizontally(
    annotation: GraphicalAnnotation, cx: float = 0.0
) -> GraphicalAnnotation:
    """Flip an annotation around the vertical line ``x=cx``."""
    if isinstance(annotation, CursorAnnotation):
        if annotation.orientation == CursorOrientation.HORIZONTAL:
            return annotation
        if annotation.orientation == CursorOrientation.VERTICAL:
            assert isinstance(annotation.position, float)
            return replace(annotation, position=2 * cx - annotation.position)
        assert isinstance(annotation.position, tuple)
        return replace(
            annotation,
            position=(2 * cx - annotation.position[0], annotation.position[1]),
        )
    if isinstance(annotation, RangeAnnotation):
        if annotation.axis == Axis.Y:
            return annotation
        return replace(
            annotation, start=2 * cx - annotation.end, end=2 * cx - annotation.start
        )
    transformed = scale_annotation(annotation, -1.0, 1.0, center=(cx, 0.0))
    return transformed


def flip_annotation_vertically(
    annotation: GraphicalAnnotation, cy: float = 0.0
) -> GraphicalAnnotation:
    """Flip an annotation around the horizontal line ``y=cy``."""
    if isinstance(annotation, CursorAnnotation):
        if annotation.orientation == CursorOrientation.VERTICAL:
            return annotation
        if annotation.orientation == CursorOrientation.HORIZONTAL:
            assert isinstance(annotation.position, float)
            return replace(annotation, position=2 * cy - annotation.position)
        assert isinstance(annotation.position, tuple)
        return replace(
            annotation,
            position=(annotation.position[0], 2 * cy - annotation.position[1]),
        )
    if isinstance(annotation, RangeAnnotation):
        if annotation.axis == Axis.X:
            return annotation
        return replace(
            annotation, start=2 * cy - annotation.end, end=2 * cy - annotation.start
        )
    transformed = scale_annotation(annotation, 1.0, -1.0, center=(0.0, cy))
    return transformed


def transpose_annotation(annotation: GraphicalAnnotation) -> GraphicalAnnotation:
    """Transpose an annotation by exchanging its X and Y axes."""
    if isinstance(annotation, CursorAnnotation):
        if annotation.orientation == CursorOrientation.CROSSHAIR:
            assert isinstance(annotation.position, tuple)
            return replace(annotation, position=annotation.position[::-1])
        orientation = (
            CursorOrientation.VERTICAL
            if annotation.orientation == CursorOrientation.HORIZONTAL
            else CursorOrientation.HORIZONTAL
        )
        return replace(annotation, orientation=orientation)
    if isinstance(annotation, RangeAnnotation):
        axis = Axis.Y if annotation.axis == Axis.X else Axis.X
        return replace(annotation, axis=axis)
    if isinstance(annotation, RectangleAnnotation):
        return replace(
            annotation,
            x=annotation.y,
            y=annotation.x,
            angle=_normalize_angle(math.pi / 2.0 - annotation.angle),
        )
    if isinstance(annotation, CircleAnnotation):
        return replace(annotation, cx=annotation.cy, cy=annotation.cx)
    if isinstance(annotation, EllipseAnnotation):
        return replace(
            annotation,
            cx=annotation.cy,
            cy=annotation.cx,
            angle=_normalize_angle(math.pi / 2.0 - annotation.angle),
        )
    return _map_points(annotation, lambda x, y: (y, x))


def _scale_rectangle(
    annotation: RectangleAnnotation,
    sx: float,
    sy: float,
    center: tuple[float, float],
) -> RectangleAnnotation:
    """Scale a rectangle when the transformed edges remain orthogonal."""
    cos_angle = math.cos(annotation.angle)
    sin_angle = math.sin(annotation.angle)
    edge_x = np.array(
        [annotation.width * cos_angle * sx, annotation.width * sin_angle * sy]
    )
    edge_y = np.array(
        [-annotation.height * sin_angle * sx, annotation.height * cos_angle * sy]
    )
    if np.linalg.norm(edge_x) and np.linalg.norm(edge_y):
        dot = float(np.dot(edge_x, edge_y))
        tolerance = 1e-12 * float(np.linalg.norm(edge_x) * np.linalg.norm(edge_y))
        if not math.isclose(dot, 0.0, abs_tol=tolerance):
            raise AnnotationTransformError(
                "An anisotropically scaled rotated rectangle becomes a parallelogram"
            )
    x, y = _scale_point(annotation.x, annotation.y, sx, sy, center)
    if np.linalg.norm(edge_x):
        angle = math.atan2(edge_x[1], edge_x[0])
    elif np.linalg.norm(edge_y):
        angle = math.atan2(edge_y[1], edge_y[0]) - math.pi / 2.0
    else:
        angle = annotation.angle
    return replace(
        annotation,
        x=x,
        y=y,
        width=float(np.linalg.norm(edge_x)),
        height=float(np.linalg.norm(edge_y)),
        angle=_normalize_angle(angle),
    )


def _scale_ellipse(
    annotation: EllipseAnnotation,
    sx: float,
    sy: float,
    center: tuple[float, float],
) -> EllipseAnnotation:
    """Scale an ellipse through singular-value decomposition."""
    cos_angle = math.cos(annotation.angle)
    sin_angle = math.sin(annotation.angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=float)
    transform = (
        np.diag([sx, sy])
        @ rotation
        @ np.diag([annotation.radius_x, annotation.radius_y])
    )
    axes, radii, _ = np.linalg.svd(transform)
    x_axis = axes[:, 0]
    angle = math.atan2(x_axis[1], x_axis[0])
    cx, cy = _scale_point(annotation.cx, annotation.cy, sx, sy, center)
    return replace(
        annotation,
        cx=cx,
        cy=cy,
        radius_x=float(radii[0]),
        radius_y=float(radii[1]),
        angle=_normalize_angle(angle),
    )


def scale_annotation(
    annotation: GraphicalAnnotation,
    sx: float,
    sy: float,
    center: tuple[float, float] = (0.0, 0.0),
) -> GraphicalAnnotation:
    """Scale an annotation around a center."""

    def point_transform(x: float, y: float) -> tuple[float, float]:
        return _scale_point(x, y, sx, sy, center)

    if isinstance(
        annotation,
        (PointAnnotation, SegmentAnnotation, PolylineAnnotation, PolygonAnnotation),
    ) or isinstance(annotation, TextAnnotation):
        return _map_points(annotation, point_transform)
    if isinstance(annotation, RectangleAnnotation):
        return _scale_rectangle(annotation, sx, sy, center)
    if isinstance(annotation, CircleAnnotation):
        cx, cy = point_transform(annotation.cx, annotation.cy)
        if math.isclose(abs(sx), abs(sy)):
            return replace(annotation, cx=cx, cy=cy, radius=annotation.radius * abs(sx))
        ellipse = EllipseAnnotation(
            id=annotation.id,
            visible=annotation.visible,
            locked=annotation.locked,
            z_index=annotation.z_index,
            title=annotation.title,
            style=annotation.style,
            label=annotation.label,
            metadata=annotation.metadata,
            extensions=annotation.extensions,
            cx=cx,
            cy=cy,
            radius_x=annotation.radius * abs(sx),
            radius_y=annotation.radius * abs(sy),
        )
        return ellipse
    if isinstance(annotation, EllipseAnnotation):
        return _scale_ellipse(annotation, sx, sy, center)
    if isinstance(annotation, CursorAnnotation):
        if annotation.orientation == CursorOrientation.CROSSHAIR:
            assert isinstance(annotation.position, tuple)
            return replace(annotation, position=point_transform(*annotation.position))
        assert isinstance(annotation.position, float)
        cx, cy = center
        if annotation.orientation == CursorOrientation.HORIZONTAL:
            return replace(annotation, position=cy + sy * (annotation.position - cy))
        return replace(annotation, position=cx + sx * (annotation.position - cx))
    if isinstance(annotation, RangeAnnotation):
        origin = center[0] if annotation.axis == Axis.X else center[1]
        factor = sx if annotation.axis == Axis.X else sy
        return replace(
            annotation,
            start=origin + factor * (annotation.start - origin),
            end=origin + factor * (annotation.end - origin),
        )
    raise TypeError(f"Unsupported annotation type: {type(annotation).__name__}")


def transform_annotation(
    annotation: GraphicalAnnotation, operation: str, **kwargs: Any
) -> GraphicalAnnotation:
    """Apply a named geometric operation and return a new annotation."""
    if operation == "translate":
        return translate_annotation(
            annotation, kwargs.get("dx", 0), kwargs.get("dy", 0)
        )
    if operation == "rotate":
        return rotate_annotation(
            annotation, kwargs.get("angle", 0), kwargs.get("center", (0, 0))
        )
    if operation == "fliph":
        return flip_annotation_horizontally(annotation, kwargs.get("cx", 0))
    if operation == "flipv":
        return flip_annotation_vertically(annotation, kwargs.get("cy", 0))
    if operation == "transpose":
        return transpose_annotation(annotation)
    if operation == "scale":
        return scale_annotation(
            annotation,
            kwargs.get("sx", 1),
            kwargs.get("sy", 1),
            kwargs.get("center", (0, 0)),
        )
    raise ValueError(f"Unknown annotation transformation: {operation}")
