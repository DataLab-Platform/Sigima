# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the optional Plotly visualization backend."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sigima.objects import GeometryResult, KindShape, create_image, create_signal
from sigima.objects.annotations import CircleAnnotation, TextAnnotation

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None,
    reason="Plotly not installed",
)


def test_curve_and_image_specs_materialize_as_plotly_figures() -> None:
    """Pure Sigima specs must pass Plotly's runtime schema validation."""
    plotly_spec = importlib.import_module("sigima.viz.plotly_spec")
    viz_plotly = importlib.import_module("sigima.viz.viz_plotly")

    signal = create_signal(
        "Signal", x=np.arange(4, dtype=float), y=np.array([1.0, 3.0, 2.0, 4.0])
    )
    signal.set_graphical_annotations([CircleAnnotation(cx=2.0, cy=2.0, radius=0.5)])
    image = create_image("Image", data=np.arange(12, dtype=float).reshape(3, 4))
    image.set_graphical_annotations([TextAnnotation(text="Peak", x=2.0, y=1.0)])
    result = GeometryResult.from_coords(
        "Detected point", KindShape.POINT, np.array([[2.0, 1.0]])
    )

    curve_figure = viz_plotly.figure_from_spec(
        plotly_spec.build_curve_figure_spec(signal)
    )
    image_figure = viz_plotly.figure_from_spec(
        plotly_spec.build_image_figure_spec(image, results=result)
    )

    assert len(curve_figure.data) == 1
    assert len(curve_figure.layout.shapes) == 1
    assert len(image_figure.data) == 2
    assert len(image_figure.layout.annotations) == 2


def test_view_curves_uses_plotly_show(monkeypatch) -> None:
    """The public Plotly viewer must materialize and display its figure."""
    go = importlib.import_module("plotly.graph_objects")
    viz_plotly = importlib.import_module("sigima.viz.viz_plotly")

    shown = []

    def record_figure(figure) -> None:
        shown.append(figure)

    monkeypatch.setattr(go.Figure, "show", record_figure)

    viz_plotly.view_curves(np.array([1.0, 2.0, 1.0]))

    assert len(shown) == 1
    assert list(shown[0].data[0].y) == [1.0, 2.0, 1.0]
