# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the Plotly canonical annotation adapter."""

from __future__ import annotations

import importlib
import json
import sys

from sigima.objects.annotations import (
    Axis,
    CircleAnnotation,
    CoordinateSpace,
    CursorAnnotation,
    CursorOrientation,
    EllipseAnnotation,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    TextAnnotation,
)


def test_annotation_adapter_does_not_import_plotly() -> None:
    """The JSON adapter must not require or load the Plotly Python package."""
    modules_before = set(sys.modules)
    importlib.import_module("sigima.viz.annotation_plotly")
    imported_modules = set(sys.modules) - modules_before
    assert not any(
        name == "plotly" or name.startswith("plotly.") for name in imported_modules
    )


def test_all_canonical_annotations_produce_json_plotly_specs() -> None:
    """All canonical primitives must produce serializable Plotly overlays."""
    from sigima.viz.annotation_plotly import annotations_to_plotly_spec

    annotations = [
        PointAnnotation(x=1.0, y=2.0),
        SegmentAnnotation(x0=0.0, y0=1.0, x1=2.0, y1=3.0),
        RectangleAnnotation(x=2.0, y=3.0, width=4.0, height=2.0, angle=0.2),
        CircleAnnotation(cx=3.0, cy=4.0, radius=1.0),
        EllipseAnnotation(cx=4.0, cy=5.0, radius_x=2.0, radius_y=1.0, angle=0.3),
        PolylineAnnotation(points=((0.0, 0.0), (1.0, 2.0))),
        PolygonAnnotation(points=((0.0, 0.0), (2.0, 0.0), (1.0, 2.0))),
        TextAnnotation(
            text="axes text", x=0.5, y=0.9, coordinate_space=CoordinateSpace.AXES
        ),
        CursorAnnotation(orientation=CursorOrientation.CROSSHAIR, position=(1.0, 2.0)),
        RangeAnnotation(axis=Axis.X, start=2.0, end=4.0),
    ]

    spec = annotations_to_plotly_spec(annotations)

    assert set(spec) == {"traces", "shapes", "annotations"}
    assert len(spec["traces"]) == 1
    assert len(spec["shapes"]) == 9
    assert len(spec["annotations"]) == 1
    assert spec["annotations"][0]["xref"] == "paper"
    assert spec["annotations"][0]["yref"] == "paper"
    json.dumps(spec, allow_nan=False)


def test_hidden_annotations_are_omitted() -> None:
    """Hidden canonical annotations must not create Plotly overlay entries."""
    from sigima.viz.annotation_plotly import annotation_to_plotly_spec

    spec = annotation_to_plotly_spec(PointAnnotation(x=1.0, y=2.0, visible=False))

    assert spec == {"traces": [], "shapes": [], "annotations": []}
