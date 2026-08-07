# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Matplotlib renderer for canonical graphical annotations."""

from __future__ import annotations

import math
from typing import Any

from matplotlib import patches, transforms
from matplotlib.colors import to_rgba

from sigima.objects.annotations import (
    Axis,
    CircleAnnotation,
    CursorAnnotation,
    CursorOrientation,
    EllipseAnnotation,
    GraphicalAnnotation,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    TextAnchor,
    TextAnnotation,
)

_ANCHOR_ALIGNMENT = {
    TextAnchor.TOP_LEFT: ("left", "top"),
    TextAnchor.TOP: ("center", "top"),
    TextAnchor.TOP_RIGHT: ("right", "top"),
    TextAnchor.LEFT: ("left", "center"),
    TextAnchor.CENTER: ("center", "center"),
    TextAnchor.RIGHT: ("right", "center"),
    TextAnchor.BOTTOM_LEFT: ("left", "bottom"),
    TextAnchor.BOTTOM: ("center", "bottom"),
    TextAnchor.BOTTOM_RIGHT: ("right", "bottom"),
}

_MARKERS = {
    "circle": "o",
    "square": "s",
    "diamond": "D",
    "cross": "+",
    "x": "x",
    "triangle-up": "^",
    "triangle-down": "v",
    "none": "",
}


def _line_kwargs(annotation: GraphicalAnnotation) -> dict[str, Any]:
    """Return Matplotlib line keyword arguments for an annotation."""
    stroke = annotation.style.stroke
    return {
        "color": stroke.color or "none",
        "linewidth": stroke.width,
        "alpha": stroke.opacity,
        "linestyle": stroke.dash if isinstance(stroke.dash, str) else "-",
        "zorder": annotation.z_index,
    }


def _apply_custom_dash(line, annotation: GraphicalAnnotation) -> None:
    """Apply a custom dash sequence to a line artist."""
    dash = annotation.style.stroke.dash
    if isinstance(dash, tuple):
        line.set_dashes(dash)


def _patch_kwargs(annotation: GraphicalAnnotation) -> dict[str, Any]:
    """Return Matplotlib patch keyword arguments for an annotation."""
    stroke = annotation.style.stroke
    fill = annotation.style.fill
    edgecolor = (
        to_rgba(stroke.color, stroke.opacity) if stroke.color is not None else "none"
    )
    facecolor = to_rgba(fill.color, fill.opacity) if fill.color is not None else "none"
    linestyle = stroke.dash if isinstance(stroke.dash, str) else "-"
    return {
        "edgecolor": edgecolor,
        "facecolor": facecolor,
        "linewidth": stroke.width,
        "linestyle": linestyle,
        "zorder": annotation.z_index,
    }


def _text_kwargs(annotation: GraphicalAnnotation) -> dict[str, Any]:
    """Return Matplotlib text keyword arguments for an annotation."""
    style = annotation.style.text
    kwargs = {
        "fontsize": style.size,
        "fontweight": "bold" if style.bold else "normal",
        "fontstyle": "italic" if style.italic else "normal",
        "color": style.color,
        "zorder": annotation.z_index,
    }
    if style.family is not None:
        kwargs["fontfamily"] = style.family
    if style.background_color is not None:
        kwargs["bbox"] = {
            "facecolor": to_rgba(style.background_color, style.background_opacity),
            "edgecolor": "none",
        }
    return kwargs


def _offset_transform(ax, base_transform, offset: tuple[float, float]):
    """Return a transform shifted by an offset expressed in display points."""
    return transforms.offset_copy(
        base_transform,
        fig=ax.figure,
        x=offset[0],
        y=offset[1],
        units="points",
    )


def _add_text(
    ax,
    annotation: GraphicalAnnotation,
    text: str,
    x: float,
    y: float,
    anchor: TextAnchor,
    offset: tuple[float, float],
    transform,
):
    """Add styled annotation text to axes."""
    horizontal, vertical = _ANCHOR_ALIGNMENT[anchor]
    return ax.text(
        x,
        y,
        text,
        horizontalalignment=horizontal,
        verticalalignment=vertical,
        transform=_offset_transform(ax, transform, offset),
        **_text_kwargs(annotation),
    )


def _label_location(ax, annotation: GraphicalAnnotation):
    """Return a label anchor position and transform for an annotation."""
    if isinstance(annotation, PointAnnotation):
        return annotation.x, annotation.y, ax.transData
    if isinstance(annotation, SegmentAnnotation):
        return (
            (annotation.x0 + annotation.x1) / 2,
            (annotation.y0 + annotation.y1) / 2,
            ax.transData,
        )
    if isinstance(annotation, RectangleAnnotation):
        return annotation.x, annotation.y, ax.transData
    if isinstance(annotation, (CircleAnnotation, EllipseAnnotation)):
        return annotation.cx, annotation.cy, ax.transData
    if isinstance(annotation, (PolylineAnnotation, PolygonAnnotation)):
        x = sum(point[0] for point in annotation.points) / len(annotation.points)
        y = sum(point[1] for point in annotation.points) / len(annotation.points)
        return x, y, ax.transData
    if isinstance(annotation, CursorAnnotation):
        if annotation.orientation == CursorOrientation.CROSSHAIR:
            assert isinstance(annotation.position, tuple)
            return *annotation.position, ax.transData
        assert isinstance(annotation.position, float)
        if annotation.orientation == CursorOrientation.VERTICAL:
            return annotation.position, 1.0, ax.get_xaxis_transform()
        return 1.0, annotation.position, ax.get_yaxis_transform()
    if isinstance(annotation, RangeAnnotation):
        center = (annotation.start + annotation.end) / 2
        if annotation.axis == Axis.X:
            return center, 1.0, ax.get_xaxis_transform()
        return 1.0, center, ax.get_yaxis_transform()
    return None


def _add_label(ax, annotation: GraphicalAnnotation):
    """Add an attached annotation label when visible."""
    label = annotation.label
    if label is None or not label.visible or not label.text:
        return None
    location = _label_location(ax, annotation)
    if location is None:
        return None
    x, y, transform = location
    return _add_text(
        ax,
        annotation,
        label.text,
        x,
        y,
        label.anchor,
        label.offset,
        transform,
    )


def add_annotation_to_axes(ax, annotation: GraphicalAnnotation) -> list[Any]:
    """Add one canonical graphical annotation to Matplotlib axes."""
    if not annotation.visible:
        return []
    artists = []
    line_kwargs = _line_kwargs(annotation)
    if isinstance(annotation, PointAnnotation):
        marker = _MARKERS.get(
            annotation.style.marker.symbol, annotation.style.marker.symbol
        )
        (line,) = ax.plot(
            [annotation.x],
            [annotation.y],
            marker=marker,
            markersize=annotation.style.marker.size,
            markerfacecolor=annotation.style.marker.color or line_kwargs["color"],
            markeredgecolor=line_kwargs["color"],
            linestyle="",
            alpha=annotation.style.stroke.opacity,
            zorder=annotation.z_index,
        )
        artists.append(line)
    elif isinstance(annotation, SegmentAnnotation):
        (line,) = ax.plot(
            [annotation.x0, annotation.x1],
            [annotation.y0, annotation.y1],
            **line_kwargs,
        )
        _apply_custom_dash(line, annotation)
        artists.append(line)
    elif isinstance(annotation, RectangleAnnotation):
        patch = patches.Rectangle(
            (annotation.x - annotation.width / 2, annotation.y - annotation.height / 2),
            annotation.width,
            annotation.height,
            **_patch_kwargs(annotation),
        )
        patch.set_transform(
            transforms.Affine2D().rotate_around(
                annotation.x, annotation.y, annotation.angle
            )
            + ax.transData
        )
        ax.add_patch(patch)
        artists.append(patch)
    elif isinstance(annotation, CircleAnnotation):
        patch = patches.Circle(
            (annotation.cx, annotation.cy),
            annotation.radius,
            **_patch_kwargs(annotation),
        )
        ax.add_patch(patch)
        artists.append(patch)
    elif isinstance(annotation, EllipseAnnotation):
        patch = patches.Ellipse(
            (annotation.cx, annotation.cy),
            2 * annotation.radius_x,
            2 * annotation.radius_y,
            angle=math.degrees(annotation.angle),
            **_patch_kwargs(annotation),
        )
        ax.add_patch(patch)
        artists.append(patch)
    elif isinstance(annotation, PolylineAnnotation):
        x, y = zip(*annotation.points)
        (line,) = ax.plot(x, y, **line_kwargs)
        _apply_custom_dash(line, annotation)
        artists.append(line)
    elif isinstance(annotation, PolygonAnnotation):
        patch = patches.Polygon(
            annotation.points, closed=True, **_patch_kwargs(annotation)
        )
        ax.add_patch(patch)
        artists.append(patch)
    elif isinstance(annotation, TextAnnotation):
        transform = (
            ax.transData
            if annotation.coordinate_space.value == "data"
            else ax.transAxes
        )
        artists.append(
            _add_text(
                ax,
                annotation,
                annotation.text,
                annotation.x,
                annotation.y,
                annotation.anchor,
                annotation.offset,
                transform,
            )
        )
    elif isinstance(annotation, CursorAnnotation):
        if annotation.orientation == CursorOrientation.HORIZONTAL:
            assert isinstance(annotation.position, float)
            artists.append(ax.axhline(annotation.position, **line_kwargs))
        elif annotation.orientation == CursorOrientation.VERTICAL:
            assert isinstance(annotation.position, float)
            artists.append(ax.axvline(annotation.position, **line_kwargs))
        else:
            assert isinstance(annotation.position, tuple)
            artists.append(ax.axvline(annotation.position[0], **line_kwargs))
            artists.append(ax.axhline(annotation.position[1], **line_kwargs))
    elif isinstance(annotation, RangeAnnotation):
        kwargs = _patch_kwargs(annotation)
        if annotation.axis == Axis.X:
            artists.append(ax.axvspan(annotation.start, annotation.end, **kwargs))
        else:
            artists.append(ax.axhspan(annotation.start, annotation.end, **kwargs))
    else:  # pragma: no cover - protected by the closed model hierarchy
        raise TypeError(f"Unsupported annotation type: {type(annotation).__name__}")

    label = _add_label(ax, annotation)
    if label is not None:
        artists.append(label)
    return artists


def add_annotations_to_axes(ax, annotations: list[GraphicalAnnotation]) -> list[Any]:
    """Add canonical annotations to axes in deterministic layer order."""
    artists = []
    for annotation in sorted(annotations, key=lambda item: item.z_index):
        artists.extend(add_annotation_to_axes(ax, annotation))
    return artists
