# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Tests for migration of historical PlotPy annotation payloads."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sigima.objects import (
    CircleAnnotation,
    CursorAnnotation,
    EllipseAnnotation,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    TextAnnotation,
    create_signal,
)
from sigima.objects.annotations.legacy_plotpy import (
    legacy_plotpy_payload_to_annotations,
    migrate_legacy_plotpy_annotations,
)


def _legacy_payload(item_class: str, item: dict) -> dict:
    """Return a historical DataLab PlotPy payload."""
    item_key = f"{item_class}_001"
    return {
        "type": "plotpy_item",
        "item_class": item_class,
        "plotpy_json": json.dumps({item_key: item, "plot_items": [item_key]}),
    }


def test_migrate_legacy_plotpy_rectangle_without_plotpy() -> None:
    """A historical rectangle must become a canonical annotation."""
    payload = _legacy_payload(
        "AnnotatedRectangle",
        {
            "annotationparam": {"title": "Legacy rectangle"},
            "shapeparam": {
                "line": {
                    "style": "DashLine",
                    "color": "#112233",
                    "width": 2.0,
                },
                "fill": {
                    "style": "SolidPattern",
                    "color": "#445566",
                    "alpha": 0.25,
                },
            },
            "points": [
                "array",
                [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
                "float64",
            ],
            "closed": True,
            "visible": True,
        },
    )
    obj = create_signal("legacy", np.arange(5), np.arange(5))
    obj.set_annotations([payload])

    report = migrate_legacy_plotpy_annotations(obj)

    assert report.converted_count == 1
    assert report.applied
    [annotation] = obj.get_graphical_annotations()
    assert isinstance(annotation, RectangleAnnotation)
    assert (annotation.x, annotation.y) == (2.0, 1.0)
    assert (annotation.width, annotation.height, annotation.angle) == (4.0, 2.0, 0.0)
    assert annotation.title == "Legacy rectangle"
    assert annotation.style.stroke.color == "#112233"
    assert annotation.style.stroke.dash == "dash"
    assert annotation.style.fill.opacity == 0.25


@pytest.mark.parametrize(
    ("item_class", "item", "expected_type"),
    [
        (
            "AnnotatedPoint",
            {"points": ["array", [[1.0, 2.0]], "float64"]},
            PointAnnotation,
        ),
        (
            "AnnotatedSegment",
            {"points": ["array", [[0.0, 0.0], [3.0, 4.0]], "float64"]},
            SegmentAnnotation,
        ),
        (
            "AnnotatedCircle",
            {"points": ["array", [[0.0, 0.0], [4.0, 0.0]], "float64"]},
            CircleAnnotation,
        ),
        (
            "AnnotatedEllipse",
            {
                "points": [
                    "array",
                    [[4.0, 0.0], [0.0, 0.0], [2.0, -1.0], [2.0, 1.0]],
                    "float64",
                ]
            },
            EllipseAnnotation,
        ),
        (
            "AnnotatedPolygon",
            {
                "points": [
                    "array",
                    [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]],
                    "float64",
                ],
                "closed": True,
            },
            PolygonAnnotation,
        ),
        (
            "AnnotatedPolygon",
            {
                "points": ["array", [[0.0, 0.0], [2.0, 1.0]], "float64"],
                "closed": False,
            },
            PolylineAnnotation,
        ),
        (
            "LabelItem",
            {
                "labelparam": {
                    "label": "Label #1",
                    "anchor": "TL",
                    "abspos": False,
                    "xg": 1.0,
                    "yg": 2.0,
                    "xc": 3,
                    "yc": 4,
                },
                "text": "Legacy label",
            },
            TextAnnotation,
        ),
        (
            "Marker",
            {"markerparam": {"markerstyle": "Cross"}, "x": 1.0, "y": 2.0},
            CursorAnnotation,
        ),
        (
            "AnnotatedXRange",
            {"min": 1.0, "max": 5.0},
            RangeAnnotation,
        ),
        (
            "AnnotatedYRange",
            {"min": 2.0, "max": 6.0},
            RangeAnnotation,
        ),
        (
            "PointShape",
            {"points": ["array", [[1.0, 2.0]], "float64"]},
            PointAnnotation,
        ),
        (
            "SegmentShape",
            {
                "points": [
                    "array",
                    [[0.0, 0.0], [3.0, 4.0], [1.5, 2.0]],
                    "float64",
                ]
            },
            SegmentAnnotation,
        ),
        (
            "RectangleShape",
            {
                "points": [
                    "array",
                    [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
                    "float64",
                ]
            },
            RectangleAnnotation,
        ),
        (
            "ObliqueRectangleShape",
            {
                "points": [
                    "array",
                    [
                        [0.0, 0.0],
                        [4.0, 0.0],
                        [4.0, 2.0],
                        [0.0, 2.0],
                        [0.0, 1.0],
                        [4.0, 1.0],
                    ],
                    "float64",
                ]
            },
            RectangleAnnotation,
        ),
        (
            "EllipseShape",
            {
                "points": [
                    "array",
                    [[4.0, 0.0], [0.0, 0.0], [2.0, -1.0], [2.0, 1.0]],
                    "float64",
                ]
            },
            EllipseAnnotation,
        ),
        (
            "PolygonShape",
            {
                "points": [
                    "array",
                    [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]],
                    "float64",
                ],
                "closed": True,
            },
            PolygonAnnotation,
        ),
        (
            "PolygonShape",
            {
                "points": ["array", [[0.0, 0.0], [2.0, 1.0]], "float64"],
                "closed": False,
            },
            PolylineAnnotation,
        ),
        (
            "XRangeSelection",
            {"min": 1.0, "max": 5.0},
            RangeAnnotation,
        ),
        (
            "YRangeSelection",
            {"min": 2.0, "max": 6.0},
            RangeAnnotation,
        ),
    ],
)
def test_all_known_legacy_plotpy_types_are_supported(
    item_class: str, item: dict, expected_type: type
) -> None:
    """Every historical DataLab annotation family must have a canonical type."""
    annotations = legacy_plotpy_payload_to_annotations(
        _legacy_payload(item_class, item)
    )

    assert len(annotations) == 1
    assert isinstance(annotations[0], expected_type)


def test_migrate_legacy_plain_plotpy_shapes() -> None:
    """Plain PlotPy shapes must keep their geometry, title, lock state and style.

    DataLab <= 1.3 stored the rectangles and segments drawn on signals as plain
    PlotPy shapes, whose title and lock state are shape parameters.
    """
    shape_parameters = {
        "label": "Plain rectangle",
        "readonly": True,
        "line": {"style": "SolidLine", "color": "#112233", "width": 2.0},
    }
    rectangle = _legacy_payload(
        "RectangleShape",
        {
            "shapeparam": shape_parameters,
            "points": [
                "array",
                [[1.0, 2.0], [5.0, 2.0], [5.0, 4.0], [1.0, 4.0]],
                "float64",
            ],
            "closed": True,
            "visible": False,
            "z": 3.0,
        },
    )
    segment = _legacy_payload(
        "SegmentShape",
        {
            "shapeparam": {"label": "Plain segment"},
            "points": [
                "array",
                [[0.0, 1.0], [3.0, 5.0], [1.5, 3.0]],
                "float64",
            ],
            "closed": False,
        },
    )
    obj = create_signal("legacy", np.arange(5), np.arange(5))
    obj.set_annotations([rectangle, segment])

    report = migrate_legacy_plotpy_annotations(obj)

    assert report.converted_count == 2
    assert not report.diagnostics
    rect_annotation, segment_annotation = obj.get_graphical_annotations()
    assert isinstance(rect_annotation, RectangleAnnotation)
    assert (rect_annotation.x, rect_annotation.y) == (3.0, 3.0)
    assert (rect_annotation.width, rect_annotation.height) == (4.0, 2.0)
    assert rect_annotation.angle == 0.0
    assert rect_annotation.title == "Plain rectangle"
    assert rect_annotation.label is None
    assert rect_annotation.locked
    assert not rect_annotation.visible
    assert rect_annotation.z_index == 3
    assert rect_annotation.style.stroke.color == "#112233"
    assert rect_annotation.style.stroke.width == 2.0
    assert rect_annotation.extensions == {"plotpy": {"item_class": "RectangleShape"}}
    assert isinstance(segment_annotation, SegmentAnnotation)
    assert (
        segment_annotation.x0,
        segment_annotation.y0,
        segment_annotation.x1,
        segment_annotation.y1,
    ) == (0.0, 1.0, 3.0, 5.0)
    assert segment_annotation.title == "Plain segment"
    assert not segment_annotation.locked


def test_specialized_plotpy_shapes_are_not_migrated() -> None:
    """Shapes derived from plain shapes must not be mistaken for them."""
    payload = _legacy_payload(
        "RectangleSVGShape",
        {
            "points": [
                "array",
                [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
                "float64",
            ]
        },
    )
    with pytest.raises(ValueError, match="unsupported PlotPy annotation class"):
        legacy_plotpy_payload_to_annotations(payload)


def test_migration_is_idempotent_and_preserves_unknown_payloads() -> None:
    """Unsupported consumers must survive migration unchanged."""
    supported = _legacy_payload(
        "AnnotatedPoint", {"points": ["array", [[1.0, 2.0]], "float64"]}
    )
    unsupported = _legacy_payload("CurveItem", {"x": [0, 1], "y": [1, 2]})
    opaque = {"consumer": "unknown", "payload": {"keep": True}}
    obj = create_signal("legacy", np.arange(5), np.arange(5))
    obj.set_annotations([opaque, supported, unsupported])

    preview = migrate_legacy_plotpy_annotations(obj, dry_run=True)
    assert preview.converted_count == 1
    assert not preview.applied
    assert obj.get_annotations() == [opaque, supported, unsupported]

    report = migrate_legacy_plotpy_annotations(obj)
    assert report.converted_count == 1
    assert report.preserved_count == 2
    assert report.diagnostics
    assert obj.get_annotations()[0] == opaque
    assert obj.get_annotations()[-1] == unsupported

    second_report = migrate_legacy_plotpy_annotations(obj)
    assert second_report.converted_count == 0
    assert obj.get_annotations()[0] == opaque
    assert obj.get_annotations()[-1] == unsupported


@pytest.mark.parametrize(
    ("plotpy_marker", "expected_symbol"),
    [
        ("Cross", "cross"),
        ("XCross", "x"),
        ("UTriangle", "triangle-up"),
        ("DTriangle", "triangle-down"),
        ("LTriangle", "triangle-left"),
        ("RTriangle", "triangle-right"),
        ("Star1", "star"),
        ("Star2", "asterisk"),
    ],
)
def test_migrate_legacy_plotpy_marker_symbols(
    plotpy_marker: str, expected_symbol: str
) -> None:
    """PlotPy SymbolParam marker names must map to their canonical symbols."""
    payload = _legacy_payload(
        "AnnotatedPoint",
        {
            "points": ["array", [[1.0, 2.0]], "float64"],
            "shapeparam": {"symbol": {"marker": plotpy_marker}},
        },
    )
    annotations = legacy_plotpy_payload_to_annotations(payload)
    assert len(annotations) == 1
    annotation = annotations[0]
    assert isinstance(annotation, PointAnnotation)
    assert annotation.style.marker.symbol == expected_symbol
