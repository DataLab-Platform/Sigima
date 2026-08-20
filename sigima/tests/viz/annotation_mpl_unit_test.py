# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for canonical annotation rendering with Matplotlib."""

from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from matplotlib.text import Text

from sigima.objects import (
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
)
from sigima.viz.annotation_mpl import add_annotations_to_axes


def test_all_annotation_primitives_create_expected_artists() -> None:
    """Check structural rendering of every canonical primitive."""
    figure, axes = plt.subplots()
    annotations = [
        PointAnnotation(x=1, y=2, z_index=1),
        SegmentAnnotation(x0=0, y0=0, x1=1, y1=1, z_index=2),
        RectangleAnnotation(x=1, y=2, width=3, height=4, z_index=3),
        CircleAnnotation(cx=1, cy=2, radius=3, z_index=4),
        EllipseAnnotation(cx=1, cy=2, radius_x=3, radius_y=4, z_index=5),
        PolylineAnnotation(points=((0, 0), (1, 1)), z_index=6),
        PolygonAnnotation(points=((0, 0), (1, 0), (0, 1)), z_index=7),
        TextAnnotation(text="Peak", x=1, y=2, z_index=8),
        CursorAnnotation(
            orientation=CursorOrientation.CROSSHAIR, position=(1, 2), z_index=9
        ),
        RangeAnnotation(axis=Axis.X, start=1, end=2, z_index=10),
    ]

    artists = add_annotations_to_axes(axes, annotations)

    assert len(artists) == 11
    assert sum(isinstance(artist, Line2D) for artist in artists) == 5
    artist_types = Counter(map(type, artists))
    assert artist_types[Rectangle] == 2
    assert artist_types[Circle] == 1
    assert artist_types[Ellipse] == 1
    assert artist_types[Polygon] == 1
    assert sum(isinstance(artist, Text) for artist in artists) == 1
    assert [artist.get_zorder() for artist in artists] == sorted(
        artist.get_zorder() for artist in artists
    )
    plt.close(figure)


def test_axes_text_uses_normalized_transform() -> None:
    """Check that overlay text uses the normalized axes coordinate system."""
    figure, axes = plt.subplots()
    annotation = TextAnnotation(text="Overlay", x=0.1, y=0.9, coordinate_space="axes")

    artists = add_annotations_to_axes(axes, [annotation])

    assert len(artists) == 1
    assert artists[0].get_transform() != axes.transData
    plt.close(figure)


def test_hidden_annotation_creates_no_artist() -> None:
    """Check annotation visibility at the renderer boundary."""
    figure, axes = plt.subplots()

    artists = add_annotations_to_axes(axes, [PointAnnotation(visible=False)])

    assert not artists
    plt.close(figure)
