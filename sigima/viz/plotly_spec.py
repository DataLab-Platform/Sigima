# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Dependency-free builders for Plotly-compatible JSON figure specifications."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sigima.objects import (
    CircularROI,
    GeometryResult,
    ImageObj,
    KindShape,
    PolygonalROI,
    RectangularROI,
    SignalObj,
)
from sigima.viz.annotation_plotly import annotations_to_plotly_spec

__all__ = [
    "build_curve_figure_spec",
    "build_geometry_overlay",
    "build_image_figure_spec",
    "build_image_roi_overlay",
    "build_signal_roi_overlay",
    "merge_plotly_overlay",
]


PLOTLY_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
)
PLOTLY_DASHES = ("solid", "dash", "dashdot", "dot")
MASK_OPACITY = 0.35
ROI_FILL_ALPHA = 0.35
ROI_FILL_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

_DASH_STYLES = {
    "SolidLine": "solid",
    "DashLine": "dash",
    "DashDotLine": "dashdot",
    "DashDotDotLine": "dot",
    "-": "solid",
    "--": "dash",
    "-.": "dashdot",
    ":": "dot",
}


def _json_value(value: Any) -> Any:
    """Return a strict JSON-compatible copy of a NumPy-derived value."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        value = abs(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _array_to_json(array: Any) -> list[Any]:
    """Convert an array-like value to strict JSON-compatible nested lists."""
    values = np.asarray(array)
    if np.iscomplexobj(values):
        values = np.abs(values)
    return _json_value(values.tolist())


def _format_axis_title(label: str | None, unit: str | None) -> str:
    """Return an axis title with an optional parenthesized unit."""
    label = label or ""
    if not unit:
        return label
    return f"{label} ({unit})" if label else f"({unit})"


def _metadata_option(obj: SignalObj | ImageObj, name: str, default: Any) -> Any:
    """Read an object metadata option without mutating its defaults."""
    return obj.metadata.get(f"__{name}", default)


def _line_style(obj: SignalObj, index: int) -> dict[str, Any]:
    """Return a Plotly line style for a signal object."""
    color = PLOTLY_COLORS[index % len(PLOTLY_COLORS)]
    dash = PLOTLY_DASHES[(index // len(PLOTLY_COLORS)) % len(PLOTLY_DASHES)]
    metadata = obj.metadata
    color = metadata.get("color", color)
    dash = _DASH_STYLES.get(
        metadata.get("linestyle", dash), metadata.get("linestyle", dash)
    )
    return {
        "color": color,
        "dash": dash,
        "width": metadata.get("linewidth", 1),
    }


def _color_with_alpha(color: str, alpha: float) -> str:
    """Return an RGBA Plotly color for a hexadecimal color when possible."""
    value = color.lstrip("#")
    if len(value) == 6:
        try:
            red, green, blue = (
                int(value[index : index + 2], 16) for index in (0, 2, 4)
            )
        except ValueError:
            return color
        return f"rgba({red},{green},{blue},{alpha:.6g})"
    return color


def merge_plotly_overlay(
    figure_spec: dict[str, Any], overlay: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Return a figure specification containing an additional Plotly overlay."""
    merged = {
        **figure_spec,
        "data": list(figure_spec.get("data", [])),
        "layout": dict(figure_spec.get("layout", {})),
    }
    merged["data"].extend(overlay.get("traces", []))
    for key in ("shapes", "annotations"):
        values = list(merged["layout"].get(key, []))
        values.extend(overlay.get(key, []))
        if values:
            merged["layout"][key] = values
    return merged


def _empty_overlay() -> dict[str, list[dict[str, Any]]]:
    """Return an empty Plotly overlay specification."""
    return {"traces": [], "shapes": [], "annotations": []}


def _overlay_label(text: str, x: float, y: float) -> dict[str, Any]:
    """Return a compact Plotly label for an ROI or geometry overlay."""
    return {
        "text": text,
        "x": x,
        "y": y,
        "showarrow": False,
        "font": {"size": 10, "color": "#333333"},
        "bgcolor": "rgba(255,255,255,0.8)",
        "bordercolor": "rgba(80,80,80,0.5)",
        "borderwidth": 1,
        "borderpad": 3,
        "xanchor": "left",
        "yanchor": "bottom",
    }


def _plotly_path(points: list[tuple[float, float]]) -> str:
    """Return a closed Plotly path for physical-coordinate points."""
    path = "M " + " L ".join(f"{x:.12g},{y:.12g}" for x, y in points)
    return f"{path} Z"


def build_signal_roi_overlay(obj: SignalObj) -> dict[str, list[dict[str, Any]]]:
    """Build Plotly overlays for signal ROIs, clipped to the signal curve."""
    overlay = _empty_overlay()
    if obj.roi is None or obj.roi.is_empty():
        return overlay
    x_values = np.asarray(obj.x, dtype=float)
    y_values = np.asarray(obj.y, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size >= 2:
        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]
    for index, roi in enumerate(obj.roi):
        start, end = roi.get_physical_coords(obj)
        start, end = sorted((float(start), float(end)))
        color = ROI_FILL_COLORS[index % len(ROI_FILL_COLORS)]
        label = roi.title or f"ROI {index + 1}"
        if x_values.size >= 2:
            clipped_start = max(float(x_values[0]), start)
            clipped_end = min(float(x_values[-1]), end)
            if clipped_end > clipped_start:
                mask = (x_values >= clipped_start) & (x_values <= clipped_end)
                roi_x = np.concatenate(([clipped_start], x_values[mask], [clipped_end]))
                roi_y = np.concatenate(
                    (
                        [float(np.interp(clipped_start, x_values, y_values))],
                        y_values[mask],
                        [float(np.interp(clipped_end, x_values, y_values))],
                    )
                )
                overlay["traces"].append(
                    {
                        "type": "scatter",
                        "mode": "none",
                        "x": _array_to_json(roi_x),
                        "y": _array_to_json(roi_y),
                        "fill": "tozeroy",
                        "fillcolor": _color_with_alpha(color, ROI_FILL_ALPHA),
                        "name": label,
                        "hoverinfo": "name",
                        "showlegend": False,
                    }
                )
                continue
        overlay["shapes"].append(
            {
                "type": "rect",
                "x0": start,
                "x1": end,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": color, "width": 1},
                "fillcolor": _color_with_alpha(color, ROI_FILL_ALPHA),
                "name": label,
            }
        )
    return overlay


def build_image_roi_overlay(obj: ImageObj) -> dict[str, list[dict[str, Any]]]:
    """Build Plotly shape and label overlays for image ROIs."""
    overlay = _empty_overlay()
    if obj.roi is None or obj.roi.is_empty():
        return overlay
    for index, roi in enumerate(obj.roi):
        label = roi.title or f"ROI {index + 1}"
        line = {"color": "#ff3333", "width": 2}
        if isinstance(roi, RectangularROI):
            x0, y0, x1, y1 = roi.get_bounding_box(obj)
            shape = {
                "type": "rect",
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "line": line,
                "name": label,
            }
            label_x, label_y = (x0 + x1) / 2, y0
        elif isinstance(roi, CircularROI):
            center_x, center_y, radius = roi.get_physical_coords(obj)
            shape = {
                "type": "circle",
                "x0": center_x - radius,
                "y0": center_y - radius,
                "x1": center_x + radius,
                "y1": center_y + radius,
                "line": line,
                "name": label,
            }
            label_x, label_y = center_x, center_y - radius
        elif isinstance(roi, PolygonalROI):
            coords = np.asarray(roi.get_physical_coords(obj)).reshape(-1, 2)
            points = [(float(x), float(y)) for x, y in coords]
            shape = {
                "type": "path",
                "path": _plotly_path(points),
                "line": line,
                "name": label,
            }
            label_x = float(coords[:, 0].mean())
            label_y = float(coords[:, 1].min())
        else:  # pragma: no cover - protected by the closed ROI hierarchy
            raise TypeError(f"Unsupported image ROI type: {type(roi).__name__}")
        overlay["shapes"].append(shape)
        overlay["annotations"].append(_overlay_label(label, label_x, label_y))
    return overlay


def _ellipse_points(coords: np.ndarray) -> list[tuple[float, float]]:
    """Return physical-coordinate points for a GeometryResult ellipse."""
    center_x, center_y, radius_x, radius_y, angle = coords
    cosine = math.cos(angle)
    sine = math.sin(angle)
    points = []
    for index in range(72):
        phase = 2 * math.pi * index / 72
        x_value = radius_x * math.cos(phase)
        y_value = radius_y * math.sin(phase)
        points.append(
            (
                float(center_x + x_value * cosine - y_value * sine),
                float(center_y + x_value * sine + y_value * cosine),
            )
        )
    return points


def build_geometry_overlay(
    results: GeometryResult | list[GeometryResult] | tuple[GeometryResult, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Build Plotly overlays from one or more geometry results."""
    overlay = _empty_overlay()
    result_list = list(results) if isinstance(results, (list, tuple)) else [results]
    line = {"color": "#ffff00", "width": 2, "dash": "dash"}
    for result in result_list:
        for coords in result.coords:
            label_x: float | None = None
            label_y: float | None = None
            if result.kind == KindShape.POINT:
                x0, y0 = coords
                overlay["traces"].append(
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": [float(x0)],
                        "y": [float(y0)],
                        "marker": {
                            "color": "#ffff00",
                            "size": 8,
                            "line": {"color": "#000000", "width": 1},
                        },
                        "showlegend": False,
                        "name": result.title,
                    }
                )
                label_x, label_y = float(x0), float(y0)
            elif result.kind == KindShape.MARKER:
                x0, y0 = coords
                overlay["traces"].append(
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": [float(x0)],
                        "y": [float(y0)],
                        "marker": {
                            "symbol": "cross",
                            "color": "#ffff00",
                            "size": 12,
                        },
                        "showlegend": False,
                        "name": result.title,
                    }
                )
                overlay["shapes"].extend(
                    (
                        {
                            "type": "line",
                            "x0": float(x0),
                            "x1": float(x0),
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": line,
                        },
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 1,
                            "xref": "paper",
                            "y0": float(y0),
                            "y1": float(y0),
                            "line": line,
                        },
                    )
                )
                label_x, label_y = float(x0), float(y0)
            elif result.kind == KindShape.RECTANGLE:
                x0, y0, width, height = coords
                overlay["shapes"].append(
                    {
                        "type": "rect",
                        "x0": float(x0),
                        "y0": float(y0),
                        "x1": float(x0 + width),
                        "y1": float(y0 + height),
                        "line": line,
                    }
                )
                label_x, label_y = float(x0), float(y0)
            elif result.kind == KindShape.CIRCLE:
                center_x, center_y, radius = coords
                overlay["shapes"].append(
                    {
                        "type": "circle",
                        "x0": float(center_x - radius),
                        "y0": float(center_y - radius),
                        "x1": float(center_x + radius),
                        "y1": float(center_y + radius),
                        "line": line,
                    }
                )
                label_x, label_y = float(center_x + radius), float(center_y)
            elif result.kind == KindShape.SEGMENT:
                x0, y0, x1, y1 = coords
                overlay["shapes"].append(
                    {
                        "type": "line",
                        "x0": float(x0),
                        "y0": float(y0),
                        "x1": float(x1),
                        "y1": float(y1),
                        "line": line,
                    }
                )
                label_x, label_y = float((x0 + x1) / 2), float((y0 + y1) / 2)
            elif result.kind == KindShape.ELLIPSE:
                points = _ellipse_points(coords)
                overlay["shapes"].append(
                    {"type": "path", "path": _plotly_path(points), "line": line}
                )
                label_x, label_y = float(coords[0]), float(coords[1])
            elif result.kind == KindShape.POLYGON:
                finite_coords = coords[np.isfinite(coords)]
                points = [(float(x), float(y)) for x, y in finite_coords.reshape(-1, 2)]
                overlay["shapes"].append(
                    {"type": "path", "path": _plotly_path(points), "line": line}
                )
                label_x = sum(point[0] for point in points) / len(points)
                label_y = sum(point[1] for point in points) / len(points)
            else:  # pragma: no cover - protected by KindShape validation
                raise TypeError(f"Unsupported geometry kind: {result.kind}")
            if label_x is not None and label_y is not None:
                overlay["annotations"].append(
                    _overlay_label(result.title, label_x, label_y)
                )
    return overlay


def _normalize_curve_items(
    data_or_objs: list[Any] | tuple[Any, Any] | SignalObj | np.ndarray,
) -> list[Any]:
    """Return a normalized list of curve inputs."""
    if isinstance(data_or_objs, (SignalObj, np.ndarray)):
        return [data_or_objs]
    if isinstance(data_or_objs, tuple) and len(data_or_objs) == 2:
        return [data_or_objs]
    if isinstance(data_or_objs, list):
        return data_or_objs
    raise TypeError(f"Unsupported curve data type: {type(data_or_objs).__name__}")


def _signal_trace(obj: SignalObj, index: int) -> dict[str, Any]:
    """Return the main Plotly trace for a signal object."""
    line = _line_style(obj, index)
    x_values = _array_to_json(obj.x)
    y_values = _array_to_json(obj.y)
    trace: dict[str, Any] = {
        "type": "scatter",
        "mode": "lines",
        "x": x_values,
        "y": y_values,
        "line": line,
        "name": obj.title or f"Signal {index + 1}",
    }
    curve_style = _metadata_option(obj, "curvestyle", "Lines")
    if curve_style == "Sticks":
        baseline = float(_metadata_option(obj, "baseline", 0.0))
        stick_x: list[float | None] = []
        stick_y: list[float | None] = []
        for x_value, y_value in zip(x_values, y_values):
            stick_x.extend((x_value, x_value, None))
            stick_y.extend((baseline, y_value, None))
        trace["x"] = stick_x
        trace["y"] = stick_y
        trace["line"] = {"color": line["color"], "width": line["width"]}
    elif curve_style == "Steps":
        trace["line"] = {**line, "shape": "hv"}
    else:
        if obj.dx is not None:
            trace["error_x"] = {
                "type": "data",
                "array": _array_to_json(obj.dx),
                "visible": True,
            }
        if obj.dy is not None:
            trace["error_y"] = {
                "type": "data",
                "array": _array_to_json(obj.dy),
                "visible": True,
            }
    shade = float(_metadata_option(obj, "shade", 0.0))
    if shade > 0.0:
        trace["fill"] = "tozeroy"
        trace["fillcolor"] = _color_with_alpha(line["color"], shade)
    return trace


def _curve_axes(
    first_obj: SignalObj | None,
    xlabel: str | None,
    ylabel: str | None,
    xunit: str | None,
    yunit: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return Plotly X and Y axis specifications for curves."""
    x_axis: dict[str, Any] = {
        "title": {
            "text": _format_axis_title(
                xlabel or getattr(first_obj, "xlabel", None),
                xunit or getattr(first_obj, "xunit", None),
            )
        },
        "showgrid": True,
        "gridcolor": "rgba(0,0,0,0.1)",
    }
    y_axis: dict[str, Any] = {
        "title": {
            "text": _format_axis_title(
                ylabel or getattr(first_obj, "ylabel", None),
                yunit or getattr(first_obj, "yunit", None),
            )
        },
        "showgrid": True,
        "gridcolor": "rgba(0,0,0,0.1)",
    }
    if first_obj is not None:
        if first_obj.xscalelog:
            x_axis["type"] = "log"
        if first_obj.yscalelog:
            y_axis["type"] = "log"
        if not first_obj.autoscale:
            x_axis["range"] = [first_obj.xscalemin, first_obj.xscalemax]
            y_axis["range"] = [first_obj.yscalemin, first_obj.yscalemax]
    return x_axis, y_axis


def build_curve_figure_spec(
    data_or_objs: list[Any] | tuple[Any, Any] | SignalObj | np.ndarray,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    show_roi: bool = True,
    show_annotations: bool = True,
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    """Build a Plotly-compatible JSON figure specification for curves."""
    items = _normalize_curve_items(data_or_objs)
    traces: list[dict[str, Any]] = []
    first_obj = next((item for item in items if isinstance(item, SignalObj)), None)
    overlays = []
    for index, item in enumerate(items):
        if isinstance(item, SignalObj):
            traces.append(_signal_trace(item, index))
            if show_roi:
                overlays.append(build_signal_roi_overlay(item))
            if show_annotations:
                overlays.append(
                    annotations_to_plotly_spec(item.get_graphical_annotations())
                )
        elif isinstance(item, tuple) and len(item) == 2:
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": _array_to_json(item[0]),
                    "y": _array_to_json(item[1]),
                    "line": {
                        "color": PLOTLY_COLORS[index % len(PLOTLY_COLORS)],
                        "dash": PLOTLY_DASHES[
                            (index // len(PLOTLY_COLORS)) % len(PLOTLY_DASHES)
                        ],
                    },
                    "name": f"Curve {index + 1}",
                }
            )
        elif isinstance(item, np.ndarray):
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": list(range(len(item))),
                    "y": _array_to_json(item),
                    "line": {
                        "color": PLOTLY_COLORS[index % len(PLOTLY_COLORS)],
                        "dash": PLOTLY_DASHES[
                            (index // len(PLOTLY_COLORS)) % len(PLOTLY_DASHES)
                        ],
                    },
                    "name": f"Curve {index + 1}",
                }
            )
        else:
            raise TypeError(f"Unsupported curve data type: {type(item).__name__}")

    x_axis, y_axis = _curve_axes(first_obj, xlabel, ylabel, xunit, yunit)
    figure: dict[str, Any] = {
        "data": traces,
        "layout": {
            "title": {"text": title or (first_obj.title if first_obj else "Curves")},
            "xaxis": x_axis,
            "yaxis": y_axis,
            "template": "plotly_white",
            "showlegend": len(items) > 1,
            "width": width,
            "height": height,
        },
    }
    for overlay in overlays:
        figure = merge_plotly_overlay(figure, overlay)
    return figure


def _image_coords(obj: ImageObj) -> tuple[list[Any], list[Any]]:
    """Return JSON-compatible pixel-center coordinates for an image object."""
    if not obj.is_uniform_coords:
        return _array_to_json(obj.xcoords), _array_to_json(obj.ycoords)
    row_count, column_count = obj.data.shape[:2]
    x_coords = obj.x0 + np.arange(column_count) * obj.dx
    y_coords = obj.y0 + np.arange(row_count) * obj.dy
    return _array_to_json(x_coords), _array_to_json(y_coords)


def build_image_figure_spec(
    data_or_obj: ImageObj | np.ndarray,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    results: GeometryResult | list[GeometryResult] | None = None,
    show_roi: bool = True,
    show_annotations: bool = True,
    colormap: str | None = None,
    width: int = 640,
    height: int = 520,
) -> dict[str, Any]:
    """Build a Plotly-compatible JSON figure specification for one image."""
    if isinstance(data_or_obj, ImageObj):
        obj = data_or_obj
        data = obj.data
        x_coords, y_coords = _image_coords(obj)
        image_title = title or obj.title or "Image"
        x_title = _format_axis_title(xlabel or obj.xlabel, xunit or obj.xunit)
        y_title = _format_axis_title(ylabel or obj.ylabel, yunit or obj.yunit)
        z_title = _format_axis_title(zlabel or obj.zlabel, zunit or obj.zunit)
        colorscale = colormap or _metadata_option(obj, "colormap", "viridis")
        if _metadata_option(obj, "invert_colormap", False):
            colorscale = f"{colorscale}_r"
    elif isinstance(data_or_obj, np.ndarray):
        obj = None
        data = data_or_obj
        row_count, column_count = data.shape[:2]
        x_coords = list(range(column_count))
        y_coords = list(range(row_count))
        image_title = title or "Image"
        x_title = _format_axis_title(xlabel, xunit)
        y_title = _format_axis_title(ylabel, yunit)
        z_title = _format_axis_title(zlabel, zunit)
        colorscale = colormap or "viridis"
    else:
        raise TypeError(f"Unsupported image data type: {type(data_or_obj).__name__}")

    heatmap: dict[str, Any] = {
        "type": "heatmap",
        "z": _array_to_json(data),
        "x": x_coords,
        "y": y_coords,
        "colorscale": colorscale,
    }
    if z_title:
        heatmap["colorbar"] = {"title": {"text": z_title}}
    figure: dict[str, Any] = {
        "data": [heatmap],
        "layout": {
            "title": {"text": image_title},
            "xaxis": {"title": {"text": x_title}},
            "yaxis": {
                "title": {"text": y_title},
                "autorange": "reversed",
                "scaleanchor": "x",
                "constrain": "domain",
            },
            "template": "plotly_white",
            "showlegend": False,
            "width": width,
            "height": height,
        },
    }
    if obj is not None:
        if obj.xscalelog:
            figure["layout"]["xaxis"]["type"] = "log"
        if obj.yscalelog:
            figure["layout"]["yaxis"]["type"] = "log"
        if not obj.autoscale:
            figure["layout"]["xaxis"]["range"] = [obj.xscalemin, obj.xscalemax]
            figure["layout"]["yaxis"]["range"] = [obj.yscalemax, obj.yscalemin]
        if obj.maskdata is not None:
            mask = np.where(obj.maskdata, 1.0, np.nan)
            figure["data"].append(
                {
                    "type": "heatmap",
                    "z": _array_to_json(mask),
                    "x": x_coords,
                    "y": y_coords,
                    "colorscale": [
                        [0.0, f"rgba(255,0,0,{MASK_OPACITY})"],
                        [1.0, f"rgba(255,0,0,{MASK_OPACITY})"],
                    ],
                    "showscale": False,
                    "hoverinfo": "skip",
                }
            )
        if show_roi:
            figure = merge_plotly_overlay(figure, build_image_roi_overlay(obj))
        if show_annotations:
            figure = merge_plotly_overlay(
                figure,
                annotations_to_plotly_spec(obj.get_graphical_annotations()),
            )
    if results is not None:
        figure = merge_plotly_overlay(figure, build_geometry_overlay(results))
    return figure
