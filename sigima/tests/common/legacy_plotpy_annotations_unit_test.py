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
