# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Interactive visual gallery for Sigima's autonomous Plotly backend."""

from __future__ import annotations

import atexit
import importlib.util
import tempfile
import webbrowser
from html import escape
from pathlib import Path

import numpy as np
import pytest

from sigima.objects import (
    GeometryResult,
    KindShape,
    create_image,
    create_image_roi,
    create_signal,
    create_signal_roi,
)
from sigima.objects.annotations import (
    AnnotationLabel,
    AnnotationStyle,
    Axis,
    CircleAnnotation,
    CoordinateSpace,
    CursorAnnotation,
    CursorOrientation,
    EllipseAnnotation,
    FillStyle,
    MarkerStyle,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    StrokeStyle,
    TextAnnotation,
    TextStyle,
)
from sigima.viz.plotly_spec import build_curve_figure_spec, build_image_figure_spec
from sigima.viz.viz_plotly import figure_from_spec

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(
        importlib.util.find_spec("plotly") is None,
        reason="Plotly not installed",
    ),
]

_GALLERY_FIGURES: list[tuple[str, str]] = []


def _add_gallery_figure(name: str, spec: dict) -> None:
    """Validate a figure spec and add its HTML fragment to the gallery."""
    import plotly.io as pio  # pylint: disable=import-outside-toplevel

    figure = figure_from_spec(spec)
    assert figure.data
    figure_json = figure.to_plotly_json()
    assert "data" in figure_json
    assert "layout" in figure_json
    fragment = pio.to_html(figure, include_plotlyjs=False, full_html=False)
    _GALLERY_FIGURES.append((name, fragment))


def _build_gallery_html() -> str:
    """Return a standalone tabbed HTML document for accumulated figures."""
    from plotly import offline  # pylint: disable=import-outside-toplevel

    navigation = []
    panels = []
    for index, (name, fragment) in enumerate(_GALLERY_FIGURES):
        selected = " selected" if index == 0 else ""
        hidden = "" if index == 0 else " hidden"
        navigation.append(
            f'<button class="gallery-tab{selected}" data-index="{index}">'
            f"{escape(name)}</button>"
        )
        panels.append(
            f'<section class="gallery-panel{hidden}" id="panel-{index}">'
            f"<h2>{escape(name)}</h2>{fragment}</section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sigima Plotly visual gallery</title>
<style>
:root {{ color-scheme: light; --ink: #1d252c; --line: #c8d0d6; --accent: #006d77; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: var(--ink); background: #f4f7f8;
        font-family: "Segoe UI", sans-serif; }}
.gallery-shell {{ display: grid; grid-template-columns: minmax(220px, 280px) 1fr;
                  min-height: 100vh; }}
.gallery-nav {{ padding: 20px 0; background: #ffffff;
                border-right: 1px solid var(--line); }}
.gallery-nav h1 {{ margin: 0 20px 18px; font-size: 18px; letter-spacing: 0; }}
.gallery-tab {{ width: 100%; padding: 11px 20px; border: 0;
                border-top: 1px solid #e6ebee; color: inherit;
                background: transparent; text-align: left; cursor: pointer; }}
.gallery-tab:last-child {{ border-bottom: 1px solid #e6ebee; }}
.gallery-tab:hover {{ background: #edf5f5; }}
.gallery-tab.selected {{ color: #ffffff; background: var(--accent); }}
.gallery-content {{ min-width: 0; padding: 24px; overflow: auto; }}
.gallery-panel h2 {{ margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }}
.gallery-panel.hidden {{ display: none; }}
@media (max-width: 760px) {{
  .gallery-shell {{ grid-template-columns: 1fr; }}
  .gallery-nav {{ border-right: 0; border-bottom: 1px solid var(--line); }}
}}
</style>
<script>{offline.get_plotlyjs()}</script>
</head>
<body>
<main class="gallery-shell">
<nav class="gallery-nav"><h1>Sigima Plotly gallery</h1>{"".join(navigation)}</nav>
<div class="gallery-content">{"".join(panels)}</div>
</main>
<script>
document.querySelectorAll('.gallery-tab').forEach((tab) => {{
  tab.addEventListener('click', () => {{
        document.querySelectorAll('.gallery-tab').forEach(
            (item) => item.classList.remove('selected')
        );
        document.querySelectorAll('.gallery-panel').forEach(
            (item) => item.classList.add('hidden')
        );
    tab.classList.add('selected');
    document.getElementById(`panel-${{tab.dataset.index}}`).classList.remove('hidden');
    window.dispatchEvent(new Event('resize'));
  }});
}});
</script>
</body>
</html>"""


def _open_gallery() -> None:
    """Write and open the autonomous gallery after the GUI test process exits."""
    if not _GALLERY_FIGURES:
        return
    gallery_path = Path(tempfile.gettempdir()) / "sigima-plotly-gallery.html"
    gallery_path.write_text(_build_gallery_html(), encoding="utf-8")
    print(f"Sigima Plotly gallery: {gallery_path}")
    webbrowser.open(gallery_path.as_uri())


atexit.register(_open_gallery)


@pytest.mark.gui
def test_canonical_annotations_plotly_gallery() -> None:
    """Display every canonical annotation primitive on one image."""
    image = create_image("Canonical annotations", data=np.zeros((100, 100)))
    image.set_graphical_annotations(
        [
            PointAnnotation(
                x=12,
                y=15,
                title="Point",
                label=AnnotationLabel("Point"),
            ),
            SegmentAnnotation(
                x0=8,
                y0=30,
                x1=35,
                y1=42,
                title="Segment",
                label=AnnotationLabel("Segment"),
            ),
            RectangleAnnotation(
                x=28,
                y=68,
                width=28,
                height=14,
                angle=0.35,
                title="Rotated rectangle",
                label=AnnotationLabel("Rectangle"),
            ),
            CircleAnnotation(
                cx=58,
                cy=22,
                radius=10,
                title="Circle",
                label=AnnotationLabel("Circle"),
            ),
            EllipseAnnotation(
                cx=75,
                cy=55,
                radius_x=16,
                radius_y=8,
                angle=-0.45,
                title="Rotated ellipse",
                label=AnnotationLabel("Ellipse"),
            ),
            PolylineAnnotation(
                points=((5, 90), (20, 80), (35, 92), (48, 82)),
                title="Polyline",
                label=AnnotationLabel("Polyline"),
            ),
            PolygonAnnotation(
                points=((62, 78), (80, 70), (92, 88), (72, 95)),
                title="Polygon",
                label=AnnotationLabel("Polygon"),
            ),
            TextAnnotation(
                text="Axes coordinates",
                x=0.02,
                y=0.98,
                coordinate_space=CoordinateSpace.AXES,
            ),
            CursorAnnotation(
                orientation=CursorOrientation.CROSSHAIR,
                position=(50, 50),
                title="Crosshair",
                label=AnnotationLabel("Cursor"),
            ),
            RangeAnnotation(
                axis=Axis.X,
                start=40,
                end=55,
                title="X range",
                label=AnnotationLabel("Range"),
                style=AnnotationStyle(fill=FillStyle("#00a896", 0.18)),
            ),
        ]
    )

    spec = build_image_figure_spec(image, colormap="gray")

    assert len(spec["layout"]["shapes"]) == 9
    assert len(spec["layout"]["annotations"]) == 10
    _add_gallery_figure("Canonical annotations", spec)


@pytest.mark.gui
def test_annotation_styles_plotly_gallery() -> None:
    """Display annotation color, dash, fill, marker, text, and lock states."""
    image = create_image("Annotation styles", data=np.zeros((80, 120)))
    styles = [
        ("Solid", "solid", "#e63946"),
        ("Dashed", "dash", "#457b9d"),
        ("Dotted", "dot", "#2a9d8f"),
        ("Dash-dot", "dashdot", "#f4a261"),
    ]
    annotations = []
    for index, (label, dash, color) in enumerate(styles):
        y_value = 12 + index * 14
        annotations.append(
            SegmentAnnotation(
                x0=8,
                y0=y_value,
                x1=62,
                y1=y_value,
                title=label,
                locked=index % 2 == 0,
                label=AnnotationLabel(label),
                style=AnnotationStyle(
                    stroke=StrokeStyle(color=color, width=2 + index, dash=dash)
                ),
            )
        )
    annotations.extend(
        [
            PointAnnotation(
                x=84,
                y=18,
                title="Diamond",
                label=AnnotationLabel("Diamond"),
                style=AnnotationStyle(
                    marker=MarkerStyle("diamond", 14, "#e76f51"),
                    stroke=StrokeStyle("#7f2d1d", 2),
                ),
            ),
            RectangleAnnotation(
                x=88,
                y=48,
                width=28,
                height=20,
                title="Translucent fill",
                label=AnnotationLabel("Fill"),
                style=AnnotationStyle(
                    stroke=StrokeStyle("#264653", 3),
                    fill=FillStyle("#e9c46a", 0.45),
                ),
            ),
            TextAnnotation(
                text="Bold italic text",
                x=0.98,
                y=0.06,
                coordinate_space=CoordinateSpace.AXES,
                style=AnnotationStyle(
                    text=TextStyle(
                        size=15,
                        bold=True,
                        italic=True,
                        color="#1d3557",
                        background_color="#a8dadc",
                        background_opacity=0.8,
                    )
                ),
            ),
        ]
    )
    image.set_graphical_annotations(annotations)

    spec = build_image_figure_spec(image, colormap="gray")

    assert len(spec["data"]) == 2
    assert len(spec["layout"]["shapes"]) == 5
    _add_gallery_figure("Styles and states", spec)


@pytest.mark.gui
def test_signal_roi_and_errors_plotly_gallery() -> None:
    """Display multiple signals, error bars, styles, ROI, and annotations."""
    x_values = np.linspace(0, 4 * np.pi, 160)
    signal = create_signal(
        "Measurement",
        x=x_values,
        y=np.sin(x_values),
        dy=np.full_like(x_values, 0.08),
    )
    signal.xlabel = "Time"
    signal.xunit = "s"
    signal.ylabel = "Amplitude"
    signal.yunit = "V"
    signal.roi = create_signal_roi([2.5, 6.5], title="Analysis interval")
    signal.set_graphical_annotations(
        [
            CursorAnnotation(
                orientation=CursorOrientation.VERTICAL,
                position=np.pi,
                label=AnnotationLabel("Phase marker"),
            )
        ]
    )
    reference = create_signal("Reference", x=x_values, y=0.6 * np.cos(x_values))
    reference.metadata["color"] = "#e76f51"
    reference.metadata["linestyle"] = "DashLine"

    spec = build_curve_figure_spec([signal, reference])

    assert len(spec["data"]) == 3
    assert "error_y" in spec["data"][0]
    assert len(spec["layout"]["shapes"]) == 1
    _add_gallery_figure("Signals, ROI, and errors", spec)


@pytest.mark.gui
def test_image_results_and_coordinates_plotly_gallery() -> None:
    """Display calibrated images, ROI masks, and every geometry result kind."""
    x_grid, y_grid = np.meshgrid(np.linspace(-3, 3, 90), np.linspace(-2, 2, 60))
    data = np.exp(-(x_grid**2 + y_grid**2)) + 0.35 * np.exp(
        -((x_grid - 1.4) ** 2 + (y_grid + 0.5) ** 2) / 0.25
    )
    image = create_image("Calibrated image", data=data)
    image.xlabel = "X"
    image.xunit = "mm"
    image.ylabel = "Y"
    image.yunit = "mm"
    image.zlabel = "Intensity"
    image.set_coords(
        np.linspace(-3, 3, data.shape[1]) ** 3 / 9,
        np.linspace(-2, 2, data.shape[0]) ** 3 / 4,
    )
    image.roi = create_image_roi("circle", [0.0, 0.0, 1.2], title="Circular ROI")
    results = [
        GeometryResult.from_coords("Peak", KindShape.POINT, np.array([[0, 0]])),
        GeometryResult.from_coords(
            "Centroid", KindShape.MARKER, np.array([[0.15, -0.1]])
        ),
        GeometryResult.from_coords(
            "Bounds", KindShape.RECTANGLE, np.array([[-1.8, -1.2, 3.6, 2.4]])
        ),
        GeometryResult.from_coords("Radius", KindShape.CIRCLE, np.array([[0, 0, 0.8]])),
        GeometryResult.from_coords(
            "Diameter", KindShape.SEGMENT, np.array([[-0.8, 0, 0.8, 0]])
        ),
        GeometryResult.from_coords(
            "Fit", KindShape.ELLIPSE, np.array([[0, 0, 1.5, 0.7, 0.4]])
        ),
        GeometryResult.from_coords(
            "Envelope",
            KindShape.POLYGON,
            np.array([[-2, -0.4, -0.8, -1.4, 1.5, -1, 2, 0.8, 0, 1.6]]),
        ),
    ]

    spec = build_image_figure_spec(image, results=results, colormap="Viridis")

    assert len(spec["data"]) == 4
    assert len(spec["layout"]["shapes"]) == 8
    assert len(spec["layout"]["annotations"]) == 8
    _add_gallery_figure("Images, ROI, and geometry", spec)
