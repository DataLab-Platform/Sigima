# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for canonical annotation transformations."""

import math

import pytest

from sigima.objects.annotations import (
    AnnotationTransformError,
    Axis,
    CircleAnnotation,
    CursorAnnotation,
    CursorOrientation,
    EllipseAnnotation,
    PointAnnotation,
    PolygonAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    TextAnnotation,
    rotate_annotation,
    scale_annotation,
    transform_annotation,
    translate_annotation,
    transpose_annotation,
)


def test_translate_preserves_identity_and_common_fields() -> None:
    """Check point translation without changing annotation identity."""
    source = PointAnnotation(x=1, y=2, title="Peak", metadata={"source": "test"})

    result = translate_annotation(source, 3, 4)

    assert (result.x, result.y) == (4, 6)
    assert result.id == source.id
    assert result.title == source.title
    assert result.metadata == source.metadata


def test_rotate_rectangle_and_polygon() -> None:
    """Check rotation of oriented and vertex-based geometries."""
    rectangle = RectangleAnnotation(x=1, y=0, width=2, height=1)
    polygon = PolygonAnnotation(points=((0, 0), (1, 0), (0, 1)))

    rotated_rectangle = rotate_annotation(rectangle, math.pi / 2)
    rotated_polygon = rotate_annotation(polygon, math.pi / 2)

    assert rotated_rectangle.x == pytest.approx(0)
    assert rotated_rectangle.y == pytest.approx(1)
    assert rotated_rectangle.angle == pytest.approx(math.pi / 2)
    assert rotated_polygon.points[1] == pytest.approx((0, 1))


def test_anisotropic_circle_scale_returns_ellipse() -> None:
    """Check exact circle conversion under anisotropic scaling."""
    circle = CircleAnnotation(cx=1, cy=2, radius=3)

    result = scale_annotation(circle, 2, 4)

    assert isinstance(result, EllipseAnnotation)
    assert (result.cx, result.cy) == (2, 8)
    assert (result.radius_x, result.radius_y) == (6, 12)
    assert result.id == circle.id


def test_anisotropic_rotated_rectangle_is_rejected() -> None:
    """Check that non-representable parallelograms are not approximated."""
    rectangle = RectangleAnnotation(x=0, y=0, width=2, height=1, angle=math.pi / 4)

    with pytest.raises(AnnotationTransformError, match="parallelogram"):
        scale_annotation(rectangle, 2, 1)


def test_axis_primitives_transpose_and_rotate() -> None:
    """Check exact transformations of cursor and range primitives."""
    cursor = CursorAnnotation(orientation=CursorOrientation.VERTICAL, position=2)
    interval = RangeAnnotation(axis=Axis.X, start=1, end=3)

    transposed = transpose_annotation(cursor)
    rotated = rotate_annotation(interval, math.pi / 2)

    assert transposed.orientation == CursorOrientation.HORIZONTAL
    assert transposed.position == 2
    assert rotated.axis == Axis.Y
    assert (rotated.start, rotated.end) == pytest.approx((1, 3))


def test_oblique_axis_primitive_rotation_is_rejected() -> None:
    """Check that oblique infinite axis primitives are not approximated."""
    cursor = CursorAnnotation(orientation=CursorOrientation.HORIZONTAL, position=2)

    with pytest.raises(AnnotationTransformError, match="quarter-turn"):
        rotate_annotation(cursor, math.pi / 4)


def test_axes_text_is_not_transformed() -> None:
    """Check that overlay text remains fixed in normalized axes coordinates."""
    text = TextAnnotation(text="Title", x=0.1, y=0.9, coordinate_space="axes")

    result = transform_annotation(text, "translate", dx=10, dy=20)

    assert result is text


def test_range_scale_normalizes_reversed_bounds() -> None:
    """Check range normalization after a negative scale."""
    interval = RangeAnnotation(axis=Axis.Y, start=1, end=3)

    result = scale_annotation(interval, 1, -1)

    assert (result.start, result.end) == (-3, -1)
