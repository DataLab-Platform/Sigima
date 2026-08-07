# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for annotation import/export."""

import tempfile
from pathlib import Path

from sigima.io import (
    read_annotations,
    read_graphical_annotations,
    write_annotations,
    write_graphical_annotations,
)
from sigima.objects import PointAnnotation, SegmentAnnotation, annotation_to_dict
from sigima.objects.signal.creation import create_signal


def test_write_read_annotations():
    """Test writing and reading annotations to/from file."""
    # Create test annotations
    annotations = [
        {
            "type": "plotpy_item",
            "item_class": "AnnotatedRectangle",
            "plotpy_json": "{}",
        },
        {
            "type": "plotpy_item",
            "item_class": "AnnotatedCircle",
            "plotpy_json": "{}",
        },
    ]

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".dlabann", delete=False) as f:
        filepath = f.name

    try:
        write_annotations(filepath, annotations)

        # Read back
        read_ann = read_annotations(filepath)

        # Verify
        assert read_ann == annotations
        assert len(read_ann) == 2
        assert read_ann[0]["item_class"] == "AnnotatedRectangle"
        assert read_ann[1]["item_class"] == "AnnotatedCircle"

    finally:
        # Clean up
        Path(filepath).unlink(missing_ok=True)


def test_write_read_annotations_with_object():
    """Test writing and reading annotations using signal object."""
    # Create signal with annotations
    obj = create_signal("Test")
    obj.set_annotations(
        [
            {"type": "label", "text": "Peak 1"},
            {"type": "label", "text": "Peak 2"},
        ]
    )

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".dlabann", delete=False) as f:
        filepath = f.name

    try:
        write_annotations(filepath, obj.get_annotations())

        # Read back into new object
        obj2 = create_signal("Test 2")
        annotations = read_annotations(filepath)
        obj2.set_annotations(annotations)

        # Verify
        assert obj2.get_annotations() == obj.get_annotations()
        assert len(obj2.get_annotations()) == 2

    finally:
        # Clean up
        Path(filepath).unlink(missing_ok=True)


def test_write_read_graphical_annotations(tmp_path):
    """Test typed graphical annotation file round-trip."""
    filepath = str(tmp_path / "canonical.dlabann")
    annotations = [
        PointAnnotation(x=1, y=2, title="Peak"),
        SegmentAnnotation(x0=0, y0=1, x1=2, y1=3),
    ]

    write_graphical_annotations(filepath, annotations)

    assert read_graphical_annotations(filepath) == annotations
    assert len(read_annotations(filepath)) == 2


def test_read_graphical_annotations_ignores_opaque_entries(tmp_path):
    """Test that typed file reads leave opaque entries to the raw API."""
    filepath = str(tmp_path / "mixed.dlabann")
    annotation = PointAnnotation(x=1, y=2)
    opaque = {"type": "plotpy_item", "plotpy_json": "{}"}
    write_annotations(filepath, [opaque, annotation_to_dict(annotation)])

    assert read_graphical_annotations(filepath) == [annotation]
    assert read_annotations(filepath)[0] == opaque


if __name__ == "__main__":
    test_write_read_annotations()
    test_write_read_annotations_with_object()
    print("All tests passed!")
