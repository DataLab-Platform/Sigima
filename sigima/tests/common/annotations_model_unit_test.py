# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the renderer-independent annotation model."""

from dataclasses import FrozenInstanceError

import pytest

from sigima.objects.annotations import (
    AnnotationKind,
    AnnotationLabel,
    AnnotationStyle,
    Axis,
    CircleAnnotation,
    CursorAnnotation,
    CursorOrientation,
    EllipseAnnotation,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    StrokeStyle,
    TextAnnotation,
)


def test_all_annotation_primitives() -> None:
    """Check that every canonical primitive can be constructed."""
    annotations = [
        PointAnnotation(x=1, y=2),
        SegmentAnnotation(x0=0, y0=1, x1=2, y1=3),
        RectangleAnnotation(x=1, y=2, width=3, height=4, angle=0.5),
        CircleAnnotation(cx=1, cy=2, radius=3),
        EllipseAnnotation(cx=1, cy=2, radius_x=3, radius_y=4, angle=0.5),
        PolylineAnnotation(points=((0, 0), (1, 1))),
        PolygonAnnotation(points=((0, 0), (1, 0), (0, 1))),
        TextAnnotation(text="Peak", x=1, y=2),
        CursorAnnotation(orientation=CursorOrientation.CROSSHAIR, position=(1, 2)),
        RangeAnnotation(axis=Axis.X, start=2, end=1),
    ]

    assert [annotation.kind for annotation in annotations] == list(AnnotationKind)
    assert annotations[-1].start == 1
    assert annotations[-1].end == 2


def test_annotation_is_deeply_immutable() -> None:
    """Check that common and nested values cannot be modified after creation."""
    annotation = PointAnnotation(
        x=1,
        y=2,
        metadata={"source": {"names": ["a", "b"]}},
        extensions={"plotpy": {"custom": True}},
    )

    with pytest.raises(FrozenInstanceError):
        annotation.x = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        annotation.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        annotation.metadata["source"]["new"] = "value"  # type: ignore[index]
    assert annotation.metadata["source"]["names"] == ("a", "b")


def test_common_style_and_label() -> None:
    """Check renderer-neutral style and label values."""
    annotation = SegmentAnnotation(
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        style=AnnotationStyle(
            stroke=StrokeStyle(color="#123456", width=2, dash=(4, 2))
        ),
        label=AnnotationLabel(text="Distance", offset=(2, 4)),
        locked=True,
        z_index=5,
    )

    assert annotation.style.stroke.dash == (4.0, 2.0)
    assert annotation.label is not None
    assert annotation.label.offset == (2.0, 4.0)
    assert annotation.locked
    assert annotation.z_index == 5


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PointAnnotation(x=float("nan")),
        lambda: CircleAnnotation(radius=-1),
        lambda: PolygonAnnotation(points=((0, 0), (1, 1))),
        lambda: CursorAnnotation(orientation=CursorOrientation.CROSSHAIR, position=1),
        lambda: PointAnnotation(metadata={"bad": object()}),
        lambda: StrokeStyle(opacity=2),
    ],
)
def test_invalid_annotation_values(factory) -> None:
    """Check that invalid or non-portable values are rejected."""
    with pytest.raises((TypeError, ValueError)):
        factory()
