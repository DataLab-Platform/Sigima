# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Interactive visual tests for renderer-independent graphical annotations."""

from __future__ import annotations

import math

import numpy as np
import pytest

import sigima.objects as sio
from sigima.tests import guiutils


def _label(text: str, offset: tuple[float, float] = (0.0, 8.0)) -> sio.AnnotationLabel:
    """Create a visible annotation label."""
    return sio.AnnotationLabel(
        text=text,
        anchor=sio.TextAnchor.BOTTOM,
        offset=offset,
    )


def _style(
    color: str,
    *,
    width: float = 2.0,
    dash: str | tuple[float, ...] = "solid",
    fill: str | None = None,
    fill_opacity: float = 0.0,
    marker: str = "circle",
    marker_size: float = 8.0,
) -> sio.AnnotationStyle:
    """Create a high-contrast style for visual inspection."""
    return sio.AnnotationStyle(
        stroke=sio.StrokeStyle(color=color, width=width, dash=dash),
        fill=sio.FillStyle(color=fill, opacity=fill_opacity),
        marker=sio.MarkerStyle(symbol=marker, size=marker_size, color=color),
        text=sio.TextStyle(
            size=10,
            bold=True,
            color="#ffffff",
            background_color="#20242b",
            background_opacity=0.85,
        ),
    )


def _create_image(title: str) -> sio.ImageObj:
    """Create a calibrated image with enough contrast for overlays."""
    y_grid, x_grid = np.mgrid[0:96, 0:96]
    data = 0.35 * x_grid + 0.2 * y_grid + 15.0 * np.sin(x_grid / 9.0)
    image = sio.create_image(
        title,
        data,
        units=("mm", "mm", "a.u."),
        labels=("X", "Y", "Intensity"),
    )
    image.set_uniform_coords(1.0, 1.0, 0.0, 0.0)
    return image


def _create_signal(title: str) -> sio.SignalObj:
    """Create a calibrated signal suitable for cursor and range overlays."""
    x_data = np.linspace(0.0, 4.0 * math.pi, 600)
    y_data = np.sin(x_data) + 0.18 * np.sin(3.0 * x_data)
    return sio.create_signal(
        title,
        x_data,
        y_data,
        units=("s", "V"),
        labels=("Time", "Amplitude"),
    )


@pytest.mark.gui
def test_geometric_annotations_interactive() -> None:
    """Visually inspect point, segment, rectangle, circle, ellipse and paths."""
    image = _create_image("Geometric annotation primitives")
    annotations = [
        sio.PointAnnotation(
            x=12,
            y=78,
            title="Point",
            label=_label("Point"),
            style=_style("#ffffff", marker="diamond", marker_size=11),
        ),
        sio.SegmentAnnotation(
            x0=7,
            y0=12,
            x1=36,
            y1=29,
            title="Segment",
            label=_label("Segment"),
            style=_style("#00e5ff", width=3),
        ),
        sio.RectangleAnnotation(
            x=29,
            y=65,
            width=28,
            height=15,
            angle=math.radians(18),
            title="Rectangle",
            label=_label("Oriented rectangle"),
            style=_style("#ffea00", fill="#ffea00", fill_opacity=0.18, marker_size=5),
        ),
        sio.CircleAnnotation(
            cx=68,
            cy=73,
            radius=12,
            title="Circle",
            label=_label("Circle"),
            style=_style("#ff4081", fill="#ff4081", fill_opacity=0.16),
        ),
        sio.EllipseAnnotation(
            cx=73,
            cy=43,
            radius_x=17,
            radius_y=8,
            angle=math.radians(-30),
            title="Ellipse",
            label=_label("Oriented ellipse"),
            style=_style("#76ff03", fill="#76ff03", fill_opacity=0.16),
        ),
        sio.PolylineAnnotation(
            points=((7, 44), (16, 55), (25, 43), (37, 53)),
            title="Polyline",
            label=_label("Open polyline", (0, 11)),
            style=_style("#e040fb", width=3, dash="dashed"),
        ),
        sio.PolygonAnnotation(
            points=((47, 10), (67, 8), (84, 20), (61, 31)),
            title="Polygon",
            label=_label("Closed polygon"),
            style=_style("#ff9100", width=3, fill="#ff9100", fill_opacity=0.22),
        ),
    ]
    image.set_graphical_annotations(annotations)

    assert {annotation.kind for annotation in annotations} == {
        sio.AnnotationKind.POINT,
        sio.AnnotationKind.SEGMENT,
        sio.AnnotationKind.RECTANGLE,
        sio.AnnotationKind.CIRCLE,
        sio.AnnotationKind.ELLIPSE,
        sio.AnnotationKind.POLYLINE,
        sio.AnnotationKind.POLYGON,
    }
    with guiutils.lazy_qt_app_context(force=True):
        from sigima import viz  # pylint: disable=import-outside-toplevel

        viz.view_images(
            image,
            title="Visual check: geometric annotations",
            show_annotations=True,
        )


@pytest.mark.gui
def test_text_cursor_and_range_annotations_interactive() -> None:
    """Visually inspect data/axes text, cursors and X/Y ranges on a signal."""
    signal = _create_signal("Text, cursor and range annotations")
    guide_style = _style("#d81b60", width=2, dash="dashdot")
    annotations = [
        sio.TextAnnotation(
            text="Data coordinates",
            x=math.pi / 2,
            y=1.18,
            anchor=sio.TextAnchor.BOTTOM,
            offset=(0, 8),
            title="Data text",
            style=_style("#ffffff"),
            z_index=8,
        ),
        sio.TextAnnotation(
            text="Normalized axes coordinates",
            x=0.02,
            y=0.97,
            coordinate_space=sio.CoordinateSpace.AXES,
            anchor=sio.TextAnchor.TOP_LEFT,
            offset=(4, -4),
            title="Axes text",
            style=_style("#ffffff"),
            z_index=9,
        ),
        sio.CursorAnnotation(
            orientation=sio.CursorOrientation.VERTICAL,
            position=math.pi,
            title="Vertical cursor",
            label=_label("Vertical cursor"),
            style=guide_style,
            z_index=5,
        ),
        sio.CursorAnnotation(
            orientation=sio.CursorOrientation.HORIZONTAL,
            position=0.55,
            title="Horizontal cursor",
            label=_label("Horizontal cursor"),
            style=_style("#00acc1", width=2, dash="dotted"),
            z_index=5,
        ),
        sio.CursorAnnotation(
            orientation=sio.CursorOrientation.CROSSHAIR,
            position=(3.0 * math.pi / 2.0, -1.18),
            title="Crosshair cursor",
            label=_label("Crosshair"),
            style=_style("#fdd835", width=2, dash="dashed"),
            z_index=6,
        ),
        sio.RangeAnnotation(
            axis=sio.Axis.X,
            start=2.0 * math.pi,
            end=2.6 * math.pi,
            title="X range",
            label=_label("X range"),
            style=_style("#7b1fa2", fill="#ab47bc", fill_opacity=0.22, width=2),
            z_index=1,
        ),
        sio.RangeAnnotation(
            axis=sio.Axis.Y,
            start=-0.25,
            end=0.25,
            title="Y range",
            label=_label("Y range"),
            style=_style("#2e7d32", fill="#66bb6a", fill_opacity=0.18, width=2),
            z_index=2,
        ),
    ]
    signal.set_graphical_annotations(annotations)

    assert {annotation.kind for annotation in annotations} == {
        sio.AnnotationKind.TEXT,
        sio.AnnotationKind.CURSOR,
        sio.AnnotationKind.RANGE,
    }
    with guiutils.lazy_qt_app_context(force=True):
        from sigima import viz  # pylint: disable=import-outside-toplevel

        viz.view_curves(
            signal,
            title="Visual check: text, cursors and ranges",
            show_annotations=True,
        )


@pytest.mark.gui
def test_annotation_styles_and_states_interactive() -> None:
    """Visually inspect markers, strokes, fills, layers and locked state."""
    marker_image = _create_image("Marker and stroke styles")
    marker_names = (
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
    )
    colors = (
        "#ffffff",
        "#00e5ff",
        "#ffea00",
        "#ff4081",
        "#76ff03",
        "#e040fb",
        "#ff9100",
    )
    marker_annotations = [
        sio.PointAnnotation(
            x=10 + index * 12,
            y=77,
            title=name,
            label=_label(name, (0, 10)),
            style=_style(color, marker=name, marker_size=12),
        )
        for index, (name, color) in enumerate(zip(marker_names, colors))
    ]
    marker_annotations.extend(
        [
            sio.SegmentAnnotation(
                x0=8,
                y0=53,
                x1=86,
                y1=53,
                title="Solid",
                label=_label("solid"),
                style=_style("#ffffff", width=4),
            ),
            sio.SegmentAnnotation(
                x0=8,
                y0=39,
                x1=86,
                y1=39,
                title="Dashed",
                label=_label("dashed"),
                style=_style("#00e5ff", width=3, dash="dashed"),
            ),
            sio.SegmentAnnotation(
                x0=8,
                y0=25,
                x1=86,
                y1=25,
                title="Custom dash",
                label=_label("custom dash"),
                style=_style("#ffea00", width=3, dash=(8, 3, 2, 3)),
            ),
        ]
    )
    marker_image.set_graphical_annotations(marker_annotations)

    state_image = _create_image("Fill, layer and interaction states")
    state_annotations = [
        sio.RectangleAnnotation(
            x=35,
            y=48,
            width=44,
            height=35,
            angle=math.radians(-12),
            title="Back layer",
            label=_label("Back layer (z=1)"),
            style=_style("#00bcd4", fill="#00bcd4", fill_opacity=0.28),
            z_index=1,
        ),
        sio.CircleAnnotation(
            cx=54,
            cy=49,
            radius=22,
            title="Front layer",
            label=_label("Front layer (z=3)"),
            style=_style("#ff4081", fill="#ff4081", fill_opacity=0.35),
            z_index=3,
        ),
        sio.PointAnnotation(
            x=18,
            y=82,
            title="Movable point",
            label=_label("Movable"),
            style=_style("#76ff03", marker="diamond", marker_size=13),
            locked=False,
            z_index=5,
        ),
        sio.PointAnnotation(
            x=78,
            y=82,
            title="Locked point",
            label=_label("Locked"),
            style=_style("#ffea00", marker="square", marker_size=13),
            locked=True,
            z_index=5,
        ),
        sio.PointAnnotation(
            x=48,
            y=12,
            title="Hidden point",
            label=_label("Must not be visible"),
            style=_style("#ffffff", marker_size=18),
            visible=False,
            z_index=10,
        ),
        sio.TextAnnotation(
            text="The hidden point and label must be absent",
            x=0.5,
            y=0.04,
            coordinate_space=sio.CoordinateSpace.AXES,
            anchor=sio.TextAnchor.BOTTOM,
            style=_style("#ffffff"),
            z_index=11,
        ),
    ]
    state_image.set_graphical_annotations(state_annotations)

    assert len(marker_image.get_graphical_annotations()) == 10
    assert len(state_image.get_graphical_annotations()) == 6
    with guiutils.lazy_qt_app_context(force=True):
        from sigima import viz  # pylint: disable=import-outside-toplevel

        viz.view_images_side_by_side(
            [marker_image, state_image],
            share_axes=False,
            title="Visual check: annotation styles and states",
            show_annotations=True,
        )


if __name__ == "__main__":
    guiutils.enable_gui()
    test_geometric_annotations_interactive()
    test_text_cursor_and_range_annotations_interactive()
    test_annotation_styles_and_states_interactive()
