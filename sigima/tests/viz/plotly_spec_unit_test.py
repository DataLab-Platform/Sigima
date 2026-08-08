# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for dependency-free Plotly figure specifications."""

from __future__ import annotations

import importlib
import json
import sys

import numpy as np

from sigima.objects import (
    GeometryResult,
    KindShape,
    create_image,
    create_image_roi,
    create_signal,
    create_signal_roi,
)
from sigima.objects.annotations import PointAnnotation, TextAnnotation


def test_plotly_spec_module_does_not_import_plotly() -> None:
    """Building JSON figure specs must not load the Plotly Python package."""
    modules_before = set(sys.modules)
    importlib.import_module("sigima.viz.plotly_spec")
    imported_modules = set(sys.modules) - modules_before
    assert not any(
        name == "plotly" or name.startswith("plotly.") for name in imported_modules
    )


def test_build_curve_figure_spec_with_errors_and_annotations() -> None:
    """Signal specs must include data, errors, labels, styles, and overlays."""
    from sigima.viz.plotly_spec import build_curve_figure_spec

    signal = create_signal(
        "Measured signal",
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([1.0, 3.0, 2.0]),
        dx=np.array([0.1, 0.1, 0.1]),
        dy=np.array([0.2, 0.3, 0.2]),
    )
    signal.xlabel = "Time"
    signal.xunit = "s"
    signal.ylabel = "Amplitude"
    signal.yunit = "V"
    signal.set_graphical_annotations([PointAnnotation(x=1.0, y=3.0)])

    spec = build_curve_figure_spec(signal)

    assert len(spec["data"]) == 2
    assert spec["data"][0]["error_x"]["array"] == [0.1, 0.1, 0.1]
    assert spec["data"][0]["error_y"]["array"] == [0.2, 0.3, 0.2]
    assert spec["layout"]["xaxis"]["title"]["text"] == "Time (s)"
    assert spec["layout"]["yaxis"]["title"]["text"] == "Amplitude (V)"
    json.dumps(spec, allow_nan=False)


def test_build_image_figure_spec_with_coordinates_mask_and_annotations() -> None:
    """Image specs must preserve calibrated coordinates and strict JSON values."""
    from sigima.viz.plotly_spec import build_image_figure_spec

    image = create_image(
        "Calibrated image",
        data=np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]),
    )
    image.xlabel = "X"
    image.xunit = "mm"
    image.ylabel = "Y"
    image.yunit = "mm"
    image.zlabel = "Value"
    image.zunit = "a.u."
    image.set_uniform_coords(dx=0.5, dy=2.0, x0=10.0, y0=20.0)
    image.roi = create_image_roi("rectangle", [0, 0, 2, 1], indices=True)
    image.set_graphical_annotations([TextAnnotation(text="origin", x=10.0, y=20.0)])

    spec = build_image_figure_spec(image)

    assert spec["data"][0]["x"] == [10.0, 10.5, 11.0]
    assert spec["data"][0]["y"] == [20.0, 22.0]
    assert spec["data"][0]["z"][0][1] is None
    assert spec["data"][0]["colorbar"]["title"]["text"] == "Value (a.u.)"
    assert len(spec["data"]) == 2
    assert len(spec["layout"]["annotations"]) == 2
    assert spec["layout"]["annotations"][1]["text"] == "origin"
    json.dumps(spec, allow_nan=False)


def test_raw_arrays_produce_strict_json_specs() -> None:
    """Raw curve and image arrays must be accepted without Sigima objects."""
    from sigima.viz.plotly_spec import (
        build_curve_figure_spec,
        build_image_figure_spec,
    )

    curve_spec = build_curve_figure_spec(np.array([1.0, np.inf, 2.0]))
    image_spec = build_image_figure_spec(np.arange(6).reshape(2, 3))

    assert curve_spec["data"][0]["y"] == [1.0, None, 2.0]
    assert image_spec["data"][0]["x"] == [0, 1, 2]
    json.dumps(curve_spec, allow_nan=False)
    json.dumps(image_spec, allow_nan=False)


def test_roi_and_geometry_overlays_are_portable_json() -> None:
    """ROI and every GeometryResult kind must produce portable overlays."""
    from sigima.viz.plotly_spec import (
        build_geometry_overlay,
        build_image_roi_overlay,
        build_signal_roi_overlay,
    )

    signal = create_signal(
        "ROI signal",
        x=np.linspace(0.0, 4.0, 9),
        y=np.array([0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0]),
    )
    signal.roi = create_signal_roi([1.0, 3.0], title="Signal ROI")
    image = create_image("ROI image", data=np.arange(100).reshape(10, 10))
    image.roi = create_image_roi("rectangle", [2.0, 3.0, 4.0, 2.0], title="Image ROI")
    results = [
        GeometryResult.from_coords("Point", KindShape.POINT, np.array([[1, 2]])),
        GeometryResult.from_coords("Marker", KindShape.MARKER, np.array([[2, 3]])),
        GeometryResult.from_coords(
            "Rectangle", KindShape.RECTANGLE, np.array([[1, 1, 3, 2]])
        ),
        GeometryResult.from_coords("Circle", KindShape.CIRCLE, np.array([[3, 3, 1]])),
        GeometryResult.from_coords(
            "Segment", KindShape.SEGMENT, np.array([[0, 0, 4, 4]])
        ),
        GeometryResult.from_coords(
            "Ellipse", KindShape.ELLIPSE, np.array([[4, 4, 2, 1, 0.3]])
        ),
        GeometryResult.from_coords(
            "Polygon", KindShape.POLYGON, np.array([[0, 0, 2, 0, 1, 2]])
        ),
    ]

    signal_overlay = build_signal_roi_overlay(signal)
    image_overlay = build_image_roi_overlay(image)
    geometry_overlay = build_geometry_overlay(results)

    assert len(signal_overlay["traces"]) == 1
    assert len(image_overlay["shapes"]) == 1
    assert len(image_overlay["annotations"]) == 1
    assert len(geometry_overlay["traces"]) == 2
    assert len(geometry_overlay["shapes"]) == 7
    assert len(geometry_overlay["annotations"]) == 7
    json.dumps(signal_overlay, allow_nan=False)
    json.dumps(image_overlay, allow_nan=False)
    json.dumps(geometry_overlay, allow_nan=False)
