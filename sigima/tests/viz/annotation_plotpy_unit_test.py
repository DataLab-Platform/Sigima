# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for canonical annotation integration with PlotPy."""

# pylint: disable=import-outside-toplevel

import dataclasses
import importlib.util
import math

import numpy as np
import pytest

import sigima.objects
from sigima.objects import (
    AnnotationStyle,
    Axis,
    CircleAnnotation,
    CursorAnnotation,
    CursorOrientation,
    EllipseAnnotation,
    MarkerStyle,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    TextAnnotation,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("plotpy") is None, reason="PlotPy not installed"
)


def test_all_annotation_primitives_create_native_items() -> None:
    """Check conversion of every canonical primitive to a PlotPy item."""
    from plotpy.items import (
        AnnotatedCircle,
        AnnotatedEllipse,
        AnnotatedObliqueRectangle,
        AnnotatedPoint,
        AnnotatedPolygon,
        AnnotatedSegment,
        AnnotatedXRange,
        LabelItem,
        Marker,
    )

    from sigima.viz.annotation_plotpy import (
        AxesLabelItem,
        annotations_to_plotpy_items,
    )

    annotations = [
        PointAnnotation(x=1, y=2),
        SegmentAnnotation(x0=0, y0=0, x1=1, y1=1),
        RectangleAnnotation(x=1, y=2, width=3, height=4, angle=math.pi / 4),
        CircleAnnotation(cx=1, cy=2, radius=3),
        EllipseAnnotation(cx=1, cy=2, radius_x=3, radius_y=4),
        PolylineAnnotation(points=((0, 0), (1, 1))),
        PolygonAnnotation(points=((0, 0), (1, 0), (0, 1))),
        TextAnnotation(text="Data", x=1, y=2),
        TextAnnotation(text="Axes", x=0.1, y=0.9, coordinate_space="axes"),
        CursorAnnotation(orientation=CursorOrientation.CROSSHAIR, position=(1, 2)),
        RangeAnnotation(axis=Axis.X, start=1, end=2),
    ]

    items = annotations_to_plotpy_items(annotations)

    assert [type(item) for item in items] == [
        AnnotatedPoint,
        AnnotatedSegment,
        AnnotatedObliqueRectangle,
        AnnotatedCircle,
        AnnotatedEllipse,
        AnnotatedPolygon,
        AnnotatedPolygon,
        LabelItem,
        AxesLabelItem,
        Marker,
        AnnotatedXRange,
    ]
    assert not items[5].is_closed()
    assert items[6].is_closed()


def test_all_canonical_marker_symbols_create_valid_plotpy_markers() -> None:
    """Check that every portable marker name maps to a valid PlotPy symbol."""
    from sigima.viz.annotation_plotpy import annotations_to_plotpy_items

    expected_markers = {
        "circle": "Ellipse",
        "square": "Rect",
        "diamond": "Diamond",
        "cross": "Cross",
        "x": "XCross",
        "triangle-up": "UTriangle",
        "triangle-down": "DTriangle",
        "triangle-left": "LTriangle",
        "triangle-right": "RTriangle",
        "star": "Star1",
        "asterisk": "Star2",
        "none": "NoSymbol",
    }

    for symbol, expected in expected_markers.items():
        annotation = PointAnnotation(
            style=AnnotationStyle(marker=MarkerStyle(symbol=symbol))
        )
        [item] = annotations_to_plotpy_items([annotation])

        assert item.shape.shapeparam.symbol.marker == expected


def test_legacy_plotpy_payload_load_and_migration() -> None:
    """Check explicit migration of a known historical PlotPy payload."""
    from guidata.io import JSONWriter
    from plotpy.builder import make
    from plotpy.io import save_items
    from plotpy.items import AnnotatedPoint

    from sigima.viz.annotation_plotpy import (
        load_legacy_plotpy_items,
        migrate_legacy_plotpy_annotations,
    )

    source_item = make.annotated_point(3.0, 4.0, title="Legacy point")
    writer = JSONWriter(None)
    save_items(writer, [source_item])
    payload = {
        "type": "plotpy_item",
        "item_class": "AnnotatedPoint",
        "plotpy_json": writer.get_json(),
    }
    obj = sigima.objects.create_signal("legacy", np.arange(5), np.arange(5))
    obj.set_annotations([payload, {"consumer": "unknown"}])

    loaded = load_legacy_plotpy_items(obj)
    preview = migrate_legacy_plotpy_annotations(obj, dry_run=True)
    assert preview.converted_count == 1
    assert not preview.applied
    assert obj.get_annotations() == [payload, {"consumer": "unknown"}]

    report = migrate_legacy_plotpy_annotations(obj)

    assert len(loaded) == 1
    assert isinstance(loaded[0], AnnotatedPoint)
    assert report.converted_count == 1
    assert report.applied
    [annotation] = obj.get_graphical_annotations()
    assert isinstance(annotation, PointAnnotation)
    assert (annotation.x, annotation.y) == (3.0, 4.0)
    assert {"consumer": "unknown"} in obj.get_annotations()
    assert migrate_legacy_plotpy_annotations(obj).converted_count == 0


def test_all_known_legacy_plotpy_types_are_migrated() -> None:
    """Check migration coverage for the historical DataLab PlotPy surface."""
    from guidata.io import JSONWriter
    from plotpy.builder import make
    from plotpy.io import save_items

    from sigima.viz.annotation_plotpy import migrate_legacy_plotpy_annotations

    items = [
        make.annotated_point(1, 2),
        make.annotated_segment(0, 0, 1, 1),
        make.annotated_rectangle(0, 0, 2, 3),
        make.annotated_circle(0, 0, 2, 0),
        make.annotated_ellipse(0, 0, 2, 0, 1, -2, 1, 2),
        make.annotated_polygon(np.array([[0, 0], [1, 0], [0, 1]])),
        make.label("Legacy label", (1, 2), (3, 4), "TL"),
        make.marker(position=(1, 2), markerstyle="+"),
        make.annotated_xrange(1, 2),
        make.annotated_yrange(3, 4),
    ]
    obj = sigima.objects.create_signal("legacy", np.arange(5), np.arange(5))
    payloads = []
    for item in items:
        writer = JSONWriter(None)
        save_items(writer, [item])
        payloads.append(
            {
                "type": "plotpy_item",
                "item_class": type(item).__name__,
                "plotpy_json": writer.get_json(),
            }
        )
    obj.set_annotations(payloads)

    report = migrate_legacy_plotpy_annotations(obj)
    annotations = obj.get_graphical_annotations()

    assert report.converted_count == len(items)
    assert not report.diagnostics
    assert [type(annotation) for annotation in annotations] == [
        PointAnnotation,
        SegmentAnnotation,
        RectangleAnnotation,
        CircleAnnotation,
        EllipseAnnotation,
        PolygonAnnotation,
        TextAnnotation,
        CursorAnnotation,
        RangeAnnotation,
        RangeAnnotation,
    ]


def _assert_same_geometry(first, second) -> None:
    """Check that two canonical annotations have the same type and geometry."""
    assert type(first) is type(second)
    for annotation_field in dataclasses.fields(first):
        name = annotation_field.name
        if name in ("id", "style", "label", "metadata", "extensions"):
            continue
        first_value, second_value = getattr(first, name), getattr(second, name)
        if isinstance(first_value, (float, tuple)):
            np.testing.assert_allclose(
                np.asarray(first_value, dtype=float),
                np.asarray(second_value, dtype=float),
                atol=1e-12,
                err_msg=name,
            )
        else:
            assert first_value == second_value, name


def _make_plain_shapes() -> list:
    """Return one plain PlotPy shape of each supported family."""
    from plotpy.builder import make
    from plotpy.items import (
        ObliqueRectangleShape,
        PointShape,
        XRangeSelection,
        YRangeSelection,
    )

    rectangle = make.rectangle(1, 2, 5, 4, "Plain rectangle")
    rectangle.shapeparam.update_param(rectangle)
    rectangle.shapeparam.line.color = "#112233"
    rectangle.shapeparam.update_item(rectangle)
    return [
        PointShape(1, 2),
        make.segment(0, 1, 3, 5, "Plain segment"),
        rectangle,
        ObliqueRectangleShape(0, 0, 4, 0, 4, 2, 0, 2),
        make.circle(0, 0, 2, 0),
        make.ellipse(0, 0, 4, 0, 2, -1, 2, 1),
        make.polygon(np.array([0.0, 2.0, 1.0]), np.array([0.0, 0.0, 3.0]), True),
        make.polygon(np.array([0.0, 2.0]), np.array([0.0, 1.0]), False),
        XRangeSelection(1, 2),
        YRangeSelection(3, 4),
    ]


_PLAIN_SHAPE_ANNOTATION_TYPES = [
    PointAnnotation,
    SegmentAnnotation,
    RectangleAnnotation,
    RectangleAnnotation,
    EllipseAnnotation,
    EllipseAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RangeAnnotation,
]


def test_plain_plotpy_shapes_are_converted() -> None:
    """Plain PlotPy shapes must convert like their annotated variants.

    Live items and their historical JSON payloads must give the same annotation, and
    rendering the annotation back must keep its geometry.
    """
    from guidata.io import JSONWriter
    from plotpy.io import save_items

    from sigima.objects.annotations.legacy_plotpy import (
        legacy_plotpy_payload_to_annotations,
    )
    from sigima.viz.annotation_plotpy import (
        annotation_to_plotpy_item,
        plotpy_item_to_annotation,
    )

    items = _make_plain_shapes()
    annotations = [plotpy_item_to_annotation(item) for item in items]
    assert [type(annotation) for annotation in annotations] == (
        _PLAIN_SHAPE_ANNOTATION_TYPES
    )

    for item, annotation in zip(items, annotations):
        writer = JSONWriter(None)
        save_items(writer, [item])
        payload = {
            "type": "plotpy_item",
            "item_class": type(item).__name__,
            "plotpy_json": writer.get_json(),
        }
        [migrated] = legacy_plotpy_payload_to_annotations(payload)
        _assert_same_geometry(migrated, annotation)
        assert migrated.title == annotation.title
        _assert_same_geometry(
            plotpy_item_to_annotation(annotation_to_plotpy_item(annotation)),
            annotation,
        )

    point, segment, rectangle = annotations[:3]
    assert (point.x, point.y) == (1.0, 2.0)
    assert point.style.marker.symbol == "none"  # Shape whose style was never set
    assert (segment.x0, segment.y0, segment.x1, segment.y1) == (0.0, 1.0, 3.0, 5.0)
    assert segment.title == "Plain segment"
    assert (rectangle.x, rectangle.y, rectangle.width, rectangle.height) == (
        3.0,
        3.0,
        4.0,
        2.0,
    )
    assert rectangle.title == "Plain rectangle"
    assert rectangle.style.stroke.color == "#112233"
    assert rectangle.extensions == {"plotpy": {"item_class": "RectangleShape"}}


def test_specialized_plotpy_shapes_are_not_converted() -> None:
    """Subclasses of plain PlotPy shapes may hold data that would be lost."""
    from plotpy.items import RectangleShape

    from sigima.viz.annotation_plotpy import plotpy_item_to_annotation

    class SpecializedRectangle(RectangleShape):
        """Rectangle holding application-specific data."""

    assert plotpy_item_to_annotation(SpecializedRectangle(0, 0, 1, 1)) is None


def test_unknown_legacy_item_is_preserved() -> None:
    """Check that migration leaves unsupported PlotPy items untouched."""
    from guidata.io import JSONWriter
    from plotpy.builder import make
    from plotpy.io import save_items

    from sigima.viz.annotation_plotpy import migrate_legacy_plotpy_annotations

    writer = JSONWriter(None)
    save_items(writer, [make.curve([0, 1], [1, 2])])
    payload = {"type": "plotpy_item", "plotpy_json": writer.get_json()}
    obj = sigima.objects.create_signal("legacy", np.arange(5), np.arange(5))
    obj.set_annotations([payload])

    report = migrate_legacy_plotpy_annotations(obj)

    assert report.converted_count == 0
    assert report.preserved_count == 1
    assert report.diagnostics
    assert obj.get_annotations() == [payload]


def test_rotated_rectangle_and_ellipse_round_trip() -> None:
    """Rotated rectangles must apply options and ellipses must keep their angle."""
    from sigima.viz.annotation_plotpy import (
        annotation_to_plotpy_item,
        plotpy_item_to_annotation,
    )

    rect = RectangleAnnotation(
        x=1.0, y=2.0, width=3.0, height=4.0, angle=0.5, title="Rotated", locked=True
    )
    rect_item = annotation_to_plotpy_item(rect)
    assert rect_item.title().text() == "Rotated"
    assert not rect_item.is_label_visible()
    assert not rect_item.annotationparam.show_computations
    assert not rect_item.annotationparam.show_subtitle
    assert rect_item.is_readonly()

    ellipse = EllipseAnnotation(cx=5.0, cy=6.0, radius_x=4.0, radius_y=2.0, angle=0.25)
    ellipse_item = annotation_to_plotpy_item(ellipse)
    round_tripped = plotpy_item_to_annotation(ellipse_item)
    assert isinstance(round_tripped, EllipseAnnotation)
    assert round_tripped.angle == pytest.approx(0.25)
