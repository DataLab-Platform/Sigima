# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for graphical annotation serialization and JSON Schema."""

import copy
import json
from importlib import resources

import jsonschema
import pytest

from sigima.objects.annotations import (
    AnnotationLabel,
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
    TextAnnotation,
    annotation_from_dict,
    annotation_to_dict,
)


@pytest.fixture
def annotations():
    """Return one instance of every canonical annotation primitive."""
    common = {"metadata": {"author": "Sigima"}, "extensions": {"test": [1, 2]}}
    return [
        PointAnnotation(x=1, y=2, label=AnnotationLabel(text="Point"), **common),
        SegmentAnnotation(x0=0, y0=1, x1=2, y1=3, **common),
        RectangleAnnotation(x=1, y=2, width=3, height=4, angle=0.5, **common),
        CircleAnnotation(cx=1, cy=2, radius=3, **common),
        EllipseAnnotation(cx=1, cy=2, radius_x=3, radius_y=4, angle=0.5, **common),
        PolylineAnnotation(points=((0, 0), (1, 1)), **common),
        PolygonAnnotation(points=((0, 0), (1, 0), (0, 1)), **common),
        TextAnnotation(text="Peak", x=1, y=2, offset=(3, 4), **common),
        CursorAnnotation(
            orientation=CursorOrientation.CROSSHAIR, position=(1, 2), **common
        ),
        RangeAnnotation(axis=Axis.Y, start=1, end=2, **common),
    ]


@pytest.fixture
def annotation_schema():
    """Load the packaged graphical annotation schema."""
    path = resources.files("sigima.objects.annotations").joinpath("schema-v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_annotation_round_trip(annotations) -> None:
    """Check lossless model-to-JSON-to-model round-trips."""
    for annotation in annotations:
        data = annotation_to_dict(annotation)
        json_data = json.loads(json.dumps(data, allow_nan=False))
        assert annotation_from_dict(json_data) == annotation


def test_all_annotations_match_schema(annotations, annotation_schema) -> None:
    """Check all serialized primitives against the packaged schema."""
    validator = jsonschema.Draft202012Validator(
        annotation_schema, format_checker=jsonschema.FormatChecker()
    )
    for annotation in annotations:
        validator.validate(annotation_to_dict(annotation))


def test_schema_and_deserializer_reject_unknown_field(
    annotations, annotation_schema
) -> None:
    """Check that normalized fields cannot silently drift."""
    data = annotation_to_dict(annotations[0])
    data["unexpected"] = True

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(annotation_schema).validate(data)
    with pytest.raises(ValueError, match="Unknown annotation field"):
        annotation_from_dict(data)


def test_schema_and_deserializer_reject_missing_geometry(
    annotations, annotation_schema
) -> None:
    """Check that required primitive coordinates are enforced."""
    data = copy.deepcopy(annotation_to_dict(annotations[0]))
    del data["x"]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(annotation_schema).validate(data)
    with pytest.raises(KeyError):
        annotation_from_dict(data)


def test_deserializer_rejects_future_version(annotations) -> None:
    """Check that unsupported canonical versions are never reinterpreted."""
    data = annotation_to_dict(annotations[0])
    data["version"] = "2.0"

    with pytest.raises(ValueError, match="Unsupported annotation version"):
        annotation_from_dict(data)
