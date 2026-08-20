# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Renderer-independent graphical annotation model."""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence


class AnnotationKind(str, enum.Enum):
    """Supported graphical annotation primitives."""

    POINT = "point"
    SEGMENT = "segment"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    TEXT = "text"
    CURSOR = "cursor"
    RANGE = "range"


class CoordinateSpace(str, enum.Enum):
    """Coordinate space used by an annotation."""

    DATA = "data"
    AXES = "axes"


class CursorOrientation(str, enum.Enum):
    """Cursor orientation."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    CROSSHAIR = "crosshair"


class Axis(str, enum.Enum):
    """Plot axis."""

    X = "x"
    Y = "y"


class TextAnchor(str, enum.Enum):
    """Text anchor relative to its position."""

    TOP_LEFT = "top-left"
    TOP = "top"
    TOP_RIGHT = "top-right"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM = "bottom"
    BOTTOM_RIGHT = "bottom-right"


def _validate_number(value: float, name: str) -> None:
    """Validate that a value is a finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is finite and non-negative."""
    _validate_number(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _freeze_json(value: Any, path: str) -> Any:
    """Return an immutable copy of a JSON-compatible value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _normalize_points(
    points: Sequence[Sequence[float]], minimum: int, name: str
) -> tuple[tuple[float, float], ...]:
    """Validate and normalize a sequence of 2D points."""
    normalized = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two values")
        x, y = point
        _validate_number(x, f"{name}[{index}].x")
        _validate_number(y, f"{name}[{index}].y")
        normalized.append((float(x), float(y)))
    if len(normalized) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} points")
    return tuple(normalized)


@dataclass(frozen=True)
class StrokeStyle:
    """Line appearance shared by shape annotations."""

    color: str | None = "#ff9933"
    width: float = 1.0
    opacity: float = 1.0
    dash: str | tuple[float, ...] = "solid"

    def __post_init__(self) -> None:
        """Validate stroke style values."""
        if self.color is not None and not isinstance(self.color, str):
            raise TypeError("stroke color must be a string or None")
        _validate_non_negative(self.width, "stroke width")
        _validate_number(self.opacity, "stroke opacity")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("stroke opacity must be between 0 and 1")
        if isinstance(self.dash, str):
            if not self.dash:
                raise ValueError("stroke dash name must not be empty")
        else:
            dash = tuple(float(value) for value in self.dash)
            if not dash:
                raise ValueError("stroke dash pattern must not be empty")
            for value in dash:
                _validate_non_negative(value, "stroke dash value")
            object.__setattr__(self, "dash", dash)


@dataclass(frozen=True)
class FillStyle:
    """Fill appearance shared by closed shape annotations."""

    color: str | None = None
    opacity: float = 0.0

    def __post_init__(self) -> None:
        """Validate fill style values."""
        if self.color is not None and not isinstance(self.color, str):
            raise TypeError("fill color must be a string or None")
        _validate_number(self.opacity, "fill opacity")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("fill opacity must be between 0 and 1")


@dataclass(frozen=True)
class MarkerStyle:
    """Point marker appearance."""

    symbol: str = "circle"
    size: float = 6.0
    color: str | None = None

    def __post_init__(self) -> None:
        """Validate marker style values."""
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("marker symbol must be a non-empty string")
        _validate_non_negative(self.size, "marker size")
        if self.color is not None and not isinstance(self.color, str):
            raise TypeError("marker color must be a string or None")


@dataclass(frozen=True)
class TextStyle:
    """Text appearance."""

    family: str | None = None
    size: float = 10.0
    bold: bool = False
    italic: bool = False
    color: str = "#000000"
    background_color: str | None = None
    background_opacity: float = 0.0

    def __post_init__(self) -> None:
        """Validate text style values."""
        _validate_non_negative(self.size, "text size")
        if not isinstance(self.color, str):
            raise TypeError("text color must be a string")
        if self.background_color is not None and not isinstance(
            self.background_color, str
        ):
            raise TypeError("text background color must be a string or None")
        _validate_number(self.background_opacity, "text background opacity")
        if not 0.0 <= self.background_opacity <= 1.0:
            raise ValueError("text background opacity must be between 0 and 1")


@dataclass(frozen=True)
class AnnotationStyle:
    """Renderer-independent annotation style."""

    stroke: StrokeStyle = field(default_factory=StrokeStyle)
    fill: FillStyle = field(default_factory=FillStyle)
    marker: MarkerStyle = field(default_factory=MarkerStyle)
    text: TextStyle = field(default_factory=TextStyle)


@dataclass(frozen=True)
class AnnotationLabel:
    """Optional label attached to a graphical annotation."""

    text: str = ""
    visible: bool = True
    anchor: TextAnchor = TextAnchor.TOP
    offset: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate and normalize label values."""
        if not isinstance(self.text, str):
            raise TypeError("label text must be a string")
        anchor = TextAnchor(self.anchor)
        if len(self.offset) != 2:
            raise ValueError("label offset must contain exactly two values")
        x_offset, y_offset = self.offset
        _validate_number(x_offset, "label x offset")
        _validate_number(y_offset, "label y offset")
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "offset", (float(x_offset), float(y_offset)))


@dataclass(frozen=True)
class GraphicalAnnotation:
    """Base class for renderer-independent graphical annotations."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    visible: bool = True
    locked: bool = False
    z_index: int = 0
    title: str = ""
    style: AnnotationStyle = field(default_factory=AnnotationStyle)
    label: AnnotationLabel | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    KIND: ClassVar[AnnotationKind | None] = None

    def __post_init__(self) -> None:
        """Validate and freeze common annotation fields."""
        if self.KIND is None:
            raise TypeError("GraphicalAnnotation cannot be instantiated directly")
        if not isinstance(self.visible, bool):
            raise TypeError("annotation visible flag must be a boolean")
        if not isinstance(self.locked, bool):
            raise TypeError("annotation locked flag must be a boolean")
        if not isinstance(self.id, str):
            raise TypeError("annotation id must be a string")
        try:
            uuid.UUID(self.id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("annotation id must be a valid UUID") from exc
        if isinstance(self.z_index, bool) or not isinstance(self.z_index, int):
            raise TypeError("annotation z_index must be an integer")
        if not isinstance(self.title, str):
            raise TypeError("annotation title must be a string")
        if not isinstance(self.style, AnnotationStyle):
            raise TypeError("annotation style must be an AnnotationStyle")
        if self.label is not None and not isinstance(self.label, AnnotationLabel):
            raise TypeError("annotation label must be an AnnotationLabel or None")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))
        object.__setattr__(
            self, "extensions", _freeze_json(self.extensions, "extensions")
        )

    @property
    def kind(self) -> AnnotationKind:
        """Return the annotation discriminator."""
        assert self.KIND is not None
        return self.KIND


@dataclass(frozen=True)
class PointAnnotation(GraphicalAnnotation):
    """Point annotation in data coordinates."""

    x: float = 0.0
    y: float = 0.0

    KIND: ClassVar[AnnotationKind] = AnnotationKind.POINT

    def __post_init__(self) -> None:
        """Validate point coordinates."""
        super().__post_init__()
        _validate_number(self.x, "x")
        _validate_number(self.y, "y")


@dataclass(frozen=True)
class SegmentAnnotation(GraphicalAnnotation):
    """Line segment annotation in data coordinates."""

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    KIND: ClassVar[AnnotationKind] = AnnotationKind.SEGMENT

    def __post_init__(self) -> None:
        """Validate segment coordinates."""
        super().__post_init__()
        for name in ("x0", "y0", "x1", "y1"):
            _validate_number(getattr(self, name), name)


@dataclass(frozen=True)
class RectangleAnnotation(GraphicalAnnotation):
    """Possibly rotated rectangle annotation in data coordinates."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0

    KIND: ClassVar[AnnotationKind] = AnnotationKind.RECTANGLE

    def __post_init__(self) -> None:
        """Validate rectangle coordinates."""
        super().__post_init__()
        _validate_number(self.x, "x")
        _validate_number(self.y, "y")
        _validate_non_negative(self.width, "width")
        _validate_non_negative(self.height, "height")
        _validate_number(self.angle, "angle")


@dataclass(frozen=True)
class CircleAnnotation(GraphicalAnnotation):
    """Circle annotation in data coordinates."""

    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0

    KIND: ClassVar[AnnotationKind] = AnnotationKind.CIRCLE

    def __post_init__(self) -> None:
        """Validate circle coordinates."""
        super().__post_init__()
        _validate_number(self.cx, "cx")
        _validate_number(self.cy, "cy")
        _validate_non_negative(self.radius, "radius")


@dataclass(frozen=True)
class EllipseAnnotation(GraphicalAnnotation):
    """Possibly rotated ellipse annotation in data coordinates."""

    cx: float = 0.0
    cy: float = 0.0
    radius_x: float = 0.0
    radius_y: float = 0.0
    angle: float = 0.0

    KIND: ClassVar[AnnotationKind] = AnnotationKind.ELLIPSE

    def __post_init__(self) -> None:
        """Validate ellipse coordinates."""
        super().__post_init__()
        _validate_number(self.cx, "cx")
        _validate_number(self.cy, "cy")
        _validate_non_negative(self.radius_x, "radius_x")
        _validate_non_negative(self.radius_y, "radius_y")
        _validate_number(self.angle, "angle")


@dataclass(frozen=True)
class PolylineAnnotation(GraphicalAnnotation):
    """Open polyline annotation in data coordinates."""

    points: tuple[tuple[float, float], ...] = ()

    KIND: ClassVar[AnnotationKind] = AnnotationKind.POLYLINE

    def __post_init__(self) -> None:
        """Validate polyline points."""
        super().__post_init__()
        object.__setattr__(self, "points", _normalize_points(self.points, 2, "points"))


@dataclass(frozen=True)
class PolygonAnnotation(GraphicalAnnotation):
    """Closed polygon annotation in data coordinates."""

    points: tuple[tuple[float, float], ...] = ()

    KIND: ClassVar[AnnotationKind] = AnnotationKind.POLYGON

    def __post_init__(self) -> None:
        """Validate polygon points."""
        super().__post_init__()
        object.__setattr__(self, "points", _normalize_points(self.points, 3, "points"))


@dataclass(frozen=True)
class TextAnnotation(GraphicalAnnotation):
    """Standalone text anchored in data or normalized axes coordinates."""

    text: str = ""
    x: float = 0.0
    y: float = 0.0
    coordinate_space: CoordinateSpace = CoordinateSpace.DATA
    anchor: TextAnchor = TextAnchor.TOP_LEFT
    offset: tuple[float, float] = (0.0, 0.0)

    KIND: ClassVar[AnnotationKind] = AnnotationKind.TEXT

    def __post_init__(self) -> None:
        """Validate text annotation values."""
        super().__post_init__()
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_number(self.x, "x")
        _validate_number(self.y, "y")
        if len(self.offset) != 2:
            raise ValueError("text offset must contain exactly two values")
        x_offset, y_offset = self.offset
        _validate_number(x_offset, "text x offset")
        _validate_number(y_offset, "text y offset")
        object.__setattr__(
            self, "coordinate_space", CoordinateSpace(self.coordinate_space)
        )
        object.__setattr__(self, "anchor", TextAnchor(self.anchor))
        object.__setattr__(self, "offset", (float(x_offset), float(y_offset)))


@dataclass(frozen=True)
class CursorAnnotation(GraphicalAnnotation):
    """Horizontal, vertical, or crosshair cursor annotation."""

    orientation: CursorOrientation = CursorOrientation.VERTICAL
    position: float | tuple[float, float] = 0.0

    KIND: ClassVar[AnnotationKind] = AnnotationKind.CURSOR

    def __post_init__(self) -> None:
        """Validate cursor position for its orientation."""
        super().__post_init__()
        orientation = CursorOrientation(self.orientation)
        if orientation == CursorOrientation.CROSSHAIR:
            if not isinstance(self.position, (tuple, list)) or len(self.position) != 2:
                raise ValueError("crosshair position must contain x and y")
            x, y = self.position
            _validate_number(x, "cursor x")
            _validate_number(y, "cursor y")
            position: float | tuple[float, float] = (float(x), float(y))
        else:
            if isinstance(self.position, (tuple, list)):
                raise ValueError("axis cursor position must be a scalar")
            _validate_number(self.position, "cursor position")
            position = float(self.position)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class RangeAnnotation(GraphicalAnnotation):
    """Highlighted interval along one plot axis."""

    axis: Axis = Axis.X
    start: float = 0.0
    end: float = 0.0

    KIND: ClassVar[AnnotationKind] = AnnotationKind.RANGE

    def __post_init__(self) -> None:
        """Validate and normalize range bounds."""
        super().__post_init__()
        _validate_number(self.start, "range start")
        _validate_number(self.end, "range end")
        start, end = sorted((float(self.start), float(self.end)))
        object.__setattr__(self, "axis", Axis(self.axis))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
