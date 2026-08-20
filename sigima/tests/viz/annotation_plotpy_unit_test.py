# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for canonical annotation integration with PlotPy."""

import math

import numpy as np
from guidata.io import JSONWriter
from plotpy.builder import make
from plotpy.io import save_items
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
from sigima.viz.annotation_plotpy import (
    AxesLabelItem,
    annotations_to_plotpy_items,
    load_legacy_plotpy_items,
    migrate_legacy_plotpy_annotations,
)


def test_all_annotation_primitives_create_native_items() -> None:
    """Check conversion of every canonical primitive to a PlotPy item."""
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
    expected_markers = {
        "circle": "Ellipse",
        "square": "Rect",
        "diamond": "Diamond",
        "cross": "Cross",
        "x": "XCross",
        "triangle-up": "UTriangle",
        "triangle-down": "DTriangle",
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


def test_unknown_legacy_item_is_preserved() -> None:
    """Check that migration leaves unsupported PlotPy items untouched."""
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
