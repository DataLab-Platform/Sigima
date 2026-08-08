# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Interactive Plotly visualization backend for Sigima."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sigima.objects import ImageObj, SignalObj
from sigima.viz.plotly_spec import (
    ROI_FILL_COLORS,
    _array_to_json,
    _format_axis_title,
    build_curve_figure_spec,
    build_image_figure_spec,
)

__all__ = [
    "create_circle",
    "create_contour_shapes",
    "create_cursor",
    "create_curve",
    "create_image",
    "create_label",
    "create_marker",
    "create_range",
    "create_segment",
    "figure_from_spec",
    "roi_color_for_index",
    "view_curve_items",
    "view_curves",
    "view_curves_and_images",
    "view_image_items",
    "view_images",
    "view_images_side_by_side",
]


def figure_from_spec(spec: dict[str, Any]):
    """Materialize a Plotly figure from a dependency-free JSON specification."""
    import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel

    return go.Figure(spec)


def roi_color_for_index(index: int) -> str:
    """Return the ROI fill color for the given index."""
    return ROI_FILL_COLORS[index % len(ROI_FILL_COLORS)]


def _show_spec(spec: dict[str, Any]) -> None:
    """Display a Plotly figure specification using the configured renderer."""
    figure_from_spec(spec).show()


def create_curve(
    x: np.ndarray, y: np.ndarray, title: str | None = None
) -> dict[str, Any]:
    """Create a Plotly scatter trace from X and Y data."""
    return {
        "type": "scatter",
        "mode": "lines",
        "x": _array_to_json(x),
        "y": _array_to_json(y),
        "name": title or "Curve",
    }


def create_image(
    data: np.ndarray,
    title: str | None = None,
    interpolation: str = "linear",
    colormap: str | None = None,
    alpha_function: str | None = None,
    xdata: list[float] | None = None,
    ydata: list[float] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Create a Plotly heatmap trace from image data."""
    del interpolation, alpha_function, kwargs
    row_count, column_count = data.shape[:2]
    return {
        "type": "heatmap",
        "z": _array_to_json(data),
        "x": xdata if xdata is not None else list(range(column_count)),
        "y": ydata if ydata is not None else list(range(row_count)),
        "colorscale": colormap or "viridis",
        "name": title or "Image",
    }


def create_contour_shapes(coords: np.ndarray, shape) -> list[dict[str, Any]]:
    """Create Plotly shapes for detected contour coordinates."""
    shape_name = getattr(shape, "name", str(shape)).lower()
    shapes = []
    for values in coords:
        if shape_name == "circle":
            center_x, center_y, radius = values
            shapes.append(
                create_circle(float(center_x), float(center_y), float(radius))
            )
        elif shape_name == "ellipse":
            center_x, center_y, radius_x, radius_y, angle = values
            points = []
            for index in range(72):
                phase = 2 * math.pi * index / 72
                cosine = math.cos(angle)
                sine = math.sin(angle)
                x_value = radius_x * math.cos(phase)
                y_value = radius_y * math.sin(phase)
                points.append(
                    (
                        center_x + x_value * cosine - y_value * sine,
                        center_y + x_value * sine + y_value * cosine,
                    )
                )
            path = "M " + " L ".join(f"{x:.12g},{y:.12g}" for x, y in points)
            shapes.append(
                {
                    "type": "path",
                    "path": f"{path} Z",
                    "line": {"color": "#ff9933", "width": 2},
                }
            )
        else:
            points = list(zip(values[::2], values[1::2]))
            path = "M " + " L ".join(f"{x:.12g},{y:.12g}" for x, y in points)
            shapes.append(
                {
                    "type": "path",
                    "path": f"{path} Z",
                    "line": {"color": "#ff9933", "width": 2},
                }
            )
    return shapes


def create_circle(
    xc: float, yc: float, r: float, label: str | None = None, **kwargs
) -> dict[str, Any]:
    """Create a Plotly circle shape."""
    del kwargs
    return {
        "type": "circle",
        "x0": xc - r,
        "y0": yc - r,
        "x1": xc + r,
        "y1": yc + r,
        "line": {"color": "#ff9933", "width": 2},
        "name": label or "Circle",
    }


def create_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Create a Plotly line shape."""
    del kwargs
    return {
        "type": "line",
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "line": {"color": "#33ff00", "width": 3},
        "name": label or "Segment",
    }


def create_cursor(
    orientation: str,
    position: float | tuple[float, float],
    label: str,
) -> list[dict[str, Any]]:
    """Create horizontal, vertical, or crosshair Plotly cursor shapes."""
    shapes = []
    line = {"color": "#a7ff33", "width": 2, "dash": "dash"}
    if orientation in ("v", "x"):
        x_position = position[0] if isinstance(position, tuple) else position
        shapes.append(
            {
                "type": "line",
                "x0": x_position,
                "x1": x_position,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": line,
                "name": label,
            }
        )
    if orientation in ("h", "x"):
        y_position = position[1] if isinstance(position, tuple) else position
        shapes.append(
            {
                "type": "line",
                "x0": 0,
                "x1": 1,
                "xref": "paper",
                "y0": y_position,
                "y1": y_position,
                "line": line,
                "name": label,
            }
        )
    if not shapes:
        raise ValueError("Orientation must be 'h', 'v', or 'x'")
    return shapes


def create_range(
    orientation: str,
    pos_min: float,
    pos_max: float,
    title: str,
    **kwargs,
) -> dict[str, Any]:
    """Create a horizontal or vertical Plotly range shape."""
    del kwargs
    if orientation == "h":
        geometry = {
            "x0": pos_min,
            "x1": pos_max,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
        }
    elif orientation == "v":
        geometry = {
            "x0": 0,
            "x1": 1,
            "xref": "paper",
            "y0": pos_min,
            "y1": pos_max,
        }
    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    return {
        "type": "rect",
        **geometry,
        "line": {"color": "#ff9933", "width": 1},
        "fillcolor": "rgba(255,153,51,0.2)",
        "name": title,
    }


def create_label(text: str) -> dict[str, Any]:
    """Create a Plotly text annotation in the upper-left plot corner."""
    return {
        "text": text,
        "x": 0,
        "y": 1,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "top",
        "showarrow": False,
    }


def create_marker(x: float, y: float, title: str | None = None) -> dict[str, Any]:
    """Create a Plotly point marker trace."""
    return {
        "type": "scatter",
        "mode": "markers",
        "x": [x],
        "y": [y],
        "marker": {"symbol": "cross", "size": 10, "color": "yellow"},
        "name": title or "Marker",
        "showlegend": False,
    }


def _flatten_items(items: list[Any]) -> list[dict[str, Any]]:
    """Flatten lists returned by multi-shape creation helpers."""
    flattened = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(_flatten_items(item))
        else:
            flattened.append(item)
    return flattened


def _items_spec(items: list[Any], title: str | None, image: bool) -> dict[str, Any]:
    """Build a Plotly figure spec from low-level creation helper results."""
    data = []
    shapes = []
    annotations = []
    for item in _flatten_items(items):
        item_type = item.get("type")
        if item_type in ("scatter", "heatmap", "image"):
            data.append(item)
        elif item_type in ("circle", "line", "path", "rect"):
            shapes.append(item)
        elif "text" in item:
            annotations.append(item)
        else:
            raise TypeError(f"Unsupported Plotly item: {item!r}")
    layout: dict[str, Any] = {
        "title": {"text": title or ("Images" if image else "Curves")},
        "template": "plotly_white",
    }
    if shapes:
        layout["shapes"] = shapes
    if annotations:
        layout["annotations"] = annotations
    if image:
        layout["yaxis"] = {
            "autorange": "reversed",
            "scaleanchor": "x",
            "constrain": "domain",
        }
    return {"data": data, "layout": layout}


def view_curve_items(
    items: list[Any],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    add_legend: bool = True,
    datetime_format: str | None = None,
    object_name: str = "",
) -> None:
    """Display low-level Plotly curve items."""
    del name, datetime_format, object_name
    spec = _items_spec(items, title, image=False)
    spec["layout"].update(
        {
            "xaxis": {"title": {"text": _format_axis_title(xlabel, xunit)}},
            "yaxis": {"title": {"text": _format_axis_title(ylabel, yunit)}},
            "showlegend": add_legend,
        }
    )
    _show_spec(spec)


def view_image_items(
    items: list[Any],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    show_itemlist: bool = False,
    object_name: str = "",
) -> None:
    """Display low-level Plotly image items."""
    del name, zlabel, zunit, show_itemlist, object_name
    spec = _items_spec(items, title, image=True)
    spec["layout"]["xaxis"] = {"title": {"text": _format_axis_title(xlabel, xunit)}}
    spec["layout"]["yaxis"].update(
        {"title": {"text": _format_axis_title(ylabel, yunit)}}
    )
    _show_spec(spec)


def view_curves(
    data_or_objs,
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    show_roi: bool = True,
    show_annotations: bool = True,
    object_name: str = "",
    **kwargs,
) -> None:
    """Display signals or curve arrays in an interactive Plotly figure."""
    del name, object_name
    spec = build_curve_figure_spec(
        data_or_objs,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xunit=xunit,
        yunit=yunit,
        show_roi=show_roi,
        show_annotations=show_annotations,
        width=kwargs.get("width", 640),
        height=kwargs.get("height", 480),
    )
    _show_spec(spec)


def view_images(
    data_or_objs,
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    results=None,
    show_roi: bool = True,
    show_annotations: bool = True,
    object_name: str = "",
    **kwargs,
) -> None:
    """Display images in interactive Plotly figures."""
    del name, object_name
    if isinstance(data_or_objs, (list, tuple)):
        view_images_side_by_side(
            list(data_or_objs),
            title=title,
            results=results,
            show_roi=show_roi,
            show_annotations=show_annotations,
            **kwargs,
        )
        return
    spec = build_image_figure_spec(
        data_or_objs,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        xunit=xunit,
        yunit=yunit,
        zunit=zunit,
        results=results,
        show_roi=show_roi,
        show_annotations=show_annotations,
        colormap=kwargs.get("colormap"),
        width=kwargs.get("width", 640),
        height=kwargs.get("height", 520),
    )
    _show_spec(spec)


def view_images_side_by_side(
    images: list[np.ndarray | ImageObj],
    titles: list[str] | None = None,
    share_axes: bool = True,
    rows: int | None = None,
    maximized: bool = False,
    title: str | None = None,
    results=None,
    show_roi: bool = True,
    show_annotations: bool = True,
    object_name: str = "",
    **kwargs,
) -> None:
    """Display images in a grid of Plotly subplots."""
    del maximized, object_name
    from plotly.subplots import make_subplots  # pylint: disable=import-outside-toplevel

    row_count = rows or max(1, math.ceil(len(images) / min(4, len(images))))
    column_count = math.ceil(len(images) / row_count)
    subplot_titles = titles or [
        image.title if isinstance(image, ImageObj) else f"Image {index + 1}"
        for index, image in enumerate(images)
    ]
    figure = make_subplots(
        rows=row_count,
        cols=column_count,
        subplot_titles=subplot_titles,
        shared_xaxes=share_axes,
        shared_yaxes=share_axes,
    )
    if results is None:
        result_items = [None] * len(images)
    elif isinstance(results, (list, tuple)) and len(results) == len(images):
        result_items = list(results)
    else:
        result_items = [results] * len(images)
    for index, image in enumerate(images):
        row = index // column_count + 1
        column = index % column_count + 1
        spec = build_image_figure_spec(
            image,
            results=result_items[index],
            show_roi=show_roi,
            show_annotations=show_annotations,
            colormap=kwargs.get("colormap"),
        )
        for trace in spec["data"]:
            figure.add_trace(trace, row=row, col=column)
        for shape in spec["layout"].get("shapes", []):
            figure.add_shape(shape, row=row, col=column)
        for annotation in spec["layout"].get("annotations", []):
            figure.add_annotation(annotation, row=row, col=column)
        figure.update_yaxes(
            autorange="reversed",
            scaleanchor=f"x{index + 1 if index else ''}",
            constrain="domain",
            row=row,
            col=column,
        )
    figure.update_layout(
        title={"text": title or "Images"},
        template="plotly_white",
        width=kwargs.get("width", 640 * column_count),
        height=kwargs.get("height", 520 * row_count),
    )
    figure.show()


def view_curves_and_images(
    data_or_objs,
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    object_name: str = "",
    show_annotations: bool = True,
    **kwargs,
) -> None:
    """Display mixed signals and images in successive Plotly figures."""
    del name, object_name
    objects = (
        list(data_or_objs)
        if isinstance(data_or_objs, (list, tuple))
        else [data_or_objs]
    )
    curves = [obj for obj in objects if isinstance(obj, SignalObj)]
    images = [obj for obj in objects if isinstance(obj, ImageObj)]
    if curves:
        view_curves(
            curves,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xunit=xunit,
            yunit=yunit,
            show_annotations=show_annotations,
            **kwargs,
        )
    if images:
        view_images(
            images,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            xunit=xunit,
            yunit=yunit,
            zunit=zunit,
            show_annotations=show_annotations,
            **kwargs,
        )
