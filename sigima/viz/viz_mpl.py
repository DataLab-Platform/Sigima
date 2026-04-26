# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Visualization tools for Sigima (Matplotlib backend)
====================================================

This module provides matplotlib-based visualization utilities for Sigima objects,
as an alternative to the PlotPy-based viz_plotpy.py. It maintains API
compatibility with the PlotPy backend while using matplotlib for rendering.

Key differences from viz_plotpy.py:
- Uses matplotlib figures instead of Qt PlotDialog
- No interactive editing tools (view-only mode)
- Simpler, more lightweight implementation
- No Qt dependency beyond matplotlib's backends
- Automatic mask visualization with semi-transparent red overlay
- Many helper functions raise NotImplementedError (use view_* functions directly)
"""

from __future__ import annotations

# pylint: disable=import-error
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import numpy as np
from matplotlib import patches  # pyright: ignore[reportMissingModuleSource]

from sigima.objects import (
    CircularROI,
    GeometryResult,
    ImageObj,
    KindShape,
    PolygonalROI,
    RectangularROI,
    SegmentROI,
    SignalObj,
)

# Style configuration
COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive"]
LINESTYLES = ["-", "--", "-.", ":"]
MASK_OPACITY = 0.35  # Opacity for mask overlay

#: Color palette used to cycle through Signal ROI fill colors when several
#: ROIs are defined on the same signal. Mirrors the ``tab10``-inspired
#: palette used by DataLab so that ROI colors stay consistent across the
#: DataLab GUI, DataLab-Kernel and Sigima viewers.
ROI_FILL_COLORS = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
)
#: Translucency (matplotlib alpha) used for the ROI fill.
ROI_FILL_ALPHA = 0.35


def roi_color_for_index(index: int) -> str:
    """Return the ROI fill color (hex string) for the given ROI index.

    The index is taken modulo the palette size so colors cycle when more
    ROIs than palette entries are defined.
    """
    return ROI_FILL_COLORS[index % len(ROI_FILL_COLORS)]


def _get_image_extent_and_aspect(obj: ImageObj) -> tuple[list[float], float]:
    """Compute matplotlib extent and aspect ratio from image physical coordinates.

    ImageObj uses physical coordinates defined by:
    - x0, y0: Origin (center of top-left pixel)
    - dx, dy: Pixel spacing

    For matplotlib's imshow:
    - extent defines pixel edges: [left, right, bottom, top]
    - aspect ratio is dx/dy to preserve physical proportions

    With origin="upper", matplotlib expects:
    - extent = [xmin - dx/2, xmax + dx/2, ymax + dy/2, ymin - dy/2]

    Args:
        obj: ImageObj with physical coordinate attributes

    Returns:
        Tuple of (extent, aspect_ratio) where:
        - extent: [left, right, bottom, top] for imshow
        - aspect_ratio: dx/dy for proper physical display
    """
    nrows, ncols = obj.data.shape[:2]
    x0, y0 = obj.x0, obj.y0
    dx, dy = obj.dx, obj.dy

    # Compute pixel centers range (as in Sigima)
    xmin = x0  # Center of leftmost column
    xmax = x0 + (ncols - 1) * dx  # Center of rightmost column
    ymin = y0  # Center of topmost row
    ymax = y0 + (nrows - 1) * dy  # Center of bottommost row

    # Convert to pixel edges for matplotlib extent
    # extent = [left, right, bottom, top]
    # For origin="upper", bottom is ymax and top is ymin
    left = xmin - dx / 2
    right = xmax + dx / 2
    bottom = ymax + dy / 2  # Lower edge of bottom-most pixel
    top = ymin - dy / 2  # Upper edge of top-most pixel

    extent = [left, right, bottom, top]

    # Aspect ratio preserves physical pixel proportions
    aspect_ratio = dx / dy

    return extent, aspect_ratio


def _get_next_style(index: int) -> tuple[str, str]:
    """Get color and linestyle for the next plot item.

    Args:
        index: Sequential index of the item to style

    Returns:
        A tuple (color, linestyle) for styling the plot item
    """
    color = COLORS[index % len(COLORS)]
    linestyle = LINESTYLES[(index // len(COLORS)) % len(LINESTYLES)]
    return color, linestyle


def view_curves(
    data_or_objs: list[SignalObj | np.ndarray | tuple[np.ndarray, np.ndarray]]
    | SignalObj
    | np.ndarray
    | tuple[np.ndarray, np.ndarray],
    name: str | None = None,  # Qt-specific  # pylint: disable=unused-argument
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    show_roi: bool = True,
    object_name: str = "",  # Qt-specific  # pylint: disable=unused-argument
) -> None:
    """Create a matplotlib figure and plot curves.

    Args:
        data_or_objs: Single `SignalObj` or `np.ndarray`, or a list/tuple of these,
         or a list/tuple of (xdata, ydata) pairs
        name: Name of the dialog (unused in matplotlib - kept for API compatibility
         with PlotPy version)
        title: Title of the plot, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        show_roi: Whether to show ROIs defined in `SignalObj` instances, default is True
         (ignored if `data_or_objs` is not a `SignalObj`)
        object_name: Object name for screenshot functionality (unused in matplotlib -
         kept for API compatibility with PlotPy version)
    """
    if isinstance(data_or_objs, (tuple, list)):
        datalist = data_or_objs
    else:
        datalist = [data_or_objs]

    fig, ax = plt.subplots(figsize=(10, 6))
    if title:
        fig.suptitle(title)

    # Track labels/units from SignalObj
    x_label = xlabel
    y_label = ylabel
    x_unit = xunit
    y_unit = yunit

    for idx, data_or_obj in enumerate(datalist):
        color, linestyle = _get_next_style(idx)

        if isinstance(data_or_obj, SignalObj):
            # It's a SignalObj
            obj = data_or_obj
            xdata, ydata = obj.xydata
            label = obj.title or f"Signal {idx + 1}"

            # Update labels/units from first SignalObj
            if idx == 0:
                x_label = x_label or obj.xlabel or ""
                y_label = y_label or obj.ylabel or ""
                x_unit = x_unit or obj.xunit or ""
                y_unit = y_unit or obj.yunit or ""

            # Plot signal
            ax.plot(xdata, ydata, color=color, linestyle=linestyle, label=label)

            # Plot ROIs if requested
            if show_roi and obj.roi:
                x_arr = np.asarray(xdata, dtype=float)
                y_arr = np.asarray(ydata, dtype=float)
                if x_arr.size >= 2 and y_arr.size == x_arr.size:
                    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
                    x_arr = x_arr[finite]
                    y_arr = y_arr[finite]
                have_curve = x_arr.size >= 2 and y_arr.size == x_arr.size
                if have_curve:
                    order = np.argsort(x_arr)
                    x_arr = x_arr[order]
                    y_arr = y_arr[order]
                for roi_idx, single_roi in enumerate(obj.roi):
                    assert isinstance(single_roi, SegmentROI)
                    x0, x1 = single_roi.get_physical_coords(obj)
                    roi_color = roi_color_for_index(roi_idx)
                    roi_label = f"{label} ROI {roi_idx + 1}" if roi_idx == 0 else None
                    if have_curve:
                        x_lo = max(float(x_arr[0]), min(x0, x1))
                        x_hi = min(float(x_arr[-1]), max(x0, x1))
                        if x_hi > x_lo:
                            mask = (x_arr >= x_lo) & (x_arr <= x_hi)
                            xs_in = x_arr[mask]
                            ys_in = y_arr[mask]
                            y_left = float(np.interp(x_lo, x_arr, y_arr))
                            y_right = float(np.interp(x_hi, x_arr, y_arr))
                            xs = np.concatenate(([x_lo], xs_in, [x_hi]))
                            ys = np.concatenate(([y_left], ys_in, [y_right]))
                            ax.fill_between(
                                xs,
                                ys,
                                0.0,
                                color=roi_color,
                                alpha=ROI_FILL_ALPHA,
                                linewidth=0,
                                label=roi_label,
                            )
                            continue
                    # Fallback: full-height vertical strip
                    ax.axvspan(
                        x0,
                        x1,
                        alpha=ROI_FILL_ALPHA,
                        color=roi_color,
                        label=roi_label,
                    )

        elif isinstance(data_or_obj, tuple) and len(data_or_obj) == 2:
            # Tuple of (x, y) arrays
            xdata, ydata = data_or_obj
            ax.plot(
                xdata, ydata, color=color, linestyle=linestyle, label=f"Curve {idx + 1}"
            )

        elif isinstance(data_or_obj, np.ndarray):
            # Just y data, use indices for x
            ydata = data_or_obj
            xdata = np.arange(len(ydata))
            ax.plot(
                xdata, ydata, color=color, linestyle=linestyle, label=f"Curve {idx + 1}"
            )

        else:
            raise TypeError(f"Unsupported data type: {type(data_or_obj)}")

    # Set axis labels with units
    if x_label:
        ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
    if y_label:
        ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# pylint: disable=too-many-positional-arguments
def view_images(
    data_or_objs: list[ImageObj | np.ndarray] | ImageObj | np.ndarray,
    name: str | None = None,  # Qt-specific  # pylint: disable=unused-argument
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    results: list[GeometryResult] | GeometryResult | None = None,
    show_roi: bool = True,
    object_name: str = "",  # Qt-specific  # pylint: disable=unused-argument
    **kwargs,
) -> None:
    """Create a matplotlib figure and show images.

    Args:
        data_or_objs: Single `ImageObj` or `np.ndarray`, or a list/tuple of these
        name: Name of the dialog (unused in matplotlib - kept for API compatibility
         with PlotPy version)
        title: Title of the plot, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        results: Single `GeometryResult` or list of these to overlay on images, or None
         if no overlay is needed.
        show_roi: Whether to show ROIs defined in `ImageObj` instances, default is True
         (ignored if `data_or_objs` is not a `ImageObj`)
        object_name: Object name for screenshot functionality (unused in matplotlib -
         kept for API compatibility with PlotPy version)
        **kwargs: Additional keyword arguments (e.g., colormap settings)
    """
    if isinstance(data_or_objs, (tuple, list)):
        datalist = data_or_objs
    else:
        datalist = [data_or_objs]

    # Determine subplot layout
    n_images = len(datalist)
    if n_images == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    elif n_images <= 4:
        fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))
        if n_images == 1:
            axes = [axes]
    else:
        ncols = min(4, n_images)
        nrows = (n_images + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
        axes = axes.flatten()

    if title:
        fig.suptitle(title)

    # Track labels/units from ImageObj
    x_label = xlabel
    y_label = ylabel
    z_label = zlabel
    x_unit = xunit
    y_unit = yunit
    z_unit = zunit

    for idx, (ax, data_or_obj) in enumerate(zip(axes, datalist)):
        if isinstance(data_or_obj, ImageObj):
            # It's an ImageObj
            obj = data_or_obj
            data = obj.data
            img_title = obj.title or f"Image {idx + 1}"

            # Update labels/units from first ImageObj
            if idx == 0:
                x_label = x_label or obj.xlabel or ""
                y_label = y_label or obj.ylabel or ""
                z_label = z_label or obj.zlabel or ""
                x_unit = x_unit or obj.xunit or ""
                y_unit = y_unit or obj.yunit or ""
                z_unit = z_unit or obj.zunit or ""

            # Handle complex images (show real and imaginary separately)
            if np.issubdtype(data.dtype, np.complexfloating):
                # Skip complex handling in this simple implementation
                # Just show magnitude
                data = np.abs(data)
                img_title = f"|{img_title}|"

        elif isinstance(data_or_obj, np.ndarray):
            # NumPy array
            data = data_or_obj
            img_title = f"Image {idx + 1}"
        else:
            raise TypeError(f"Unsupported data type: {type(data_or_obj)}")

        # Compute extent and aspect ratio for ImageObj, use defaults for arrays
        if isinstance(data_or_obj, ImageObj):
            extent, aspect_ratio = _get_image_extent_and_aspect(data_or_obj)
        else:
            nrows_img, ncols_img = data.shape[:2]
            extent = [-0.5, ncols_img - 0.5, nrows_img - 0.5, -0.5]
            aspect_ratio = 1.0

        # Display image
        im = ax.imshow(
            data,
            cmap=kwargs.get("colormap", "viridis"),
            origin="upper",
            extent=extent,
            aspect=aspect_ratio,
        )
        ax.set_title(img_title)

        # Overlay mask if ImageObj has maskdata
        if isinstance(data_or_obj, ImageObj) and data_or_obj.maskdata is not None:
            # Create semi-transparent red overlay for masked areas
            mask = data_or_obj.maskdata
            mask_rgba = np.zeros((*mask.shape, 4))
            mask_rgba[mask, :] = [1, 0, 0, MASK_OPACITY]
            ax.imshow(mask_rgba, origin="upper", extent=extent)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if z_label:
            cbar.set_label(f"{z_label} ({z_unit})" if z_unit else z_label)

        # Set axis labels
        if x_label:
            ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
        if y_label:
            ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)

        # Overlay ROIs if requested
        if show_roi and isinstance(data_or_obj, ImageObj) and data_or_obj.roi:
            for single_roi in data_or_obj.roi.single_rois:
                _add_single_roi_to_axes(ax, single_roi)

        # Overlay geometry results
        if results is not None:
            result_list = results if isinstance(results, (list, tuple)) else [results]
            for result in result_list:
                _add_geometry_to_axes(ax, result)

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def _add_single_roi_to_axes(
    ax: plt.Axes, roi: RectangularROI | CircularROI | PolygonalROI
) -> None:
    """Add single ROI overlay to matplotlib axes.

    Args:
        ax: Matplotlib axes object
        roi: Single ROI object (RectangularROI, CircularROI, or PolygonalROI)
    """
    if isinstance(roi, RectangularROI):
        # coords = [x0, y0, dx, dy]
        x0, y0, dx, dy = roi.coords
        rect = patches.Rectangle(
            (x0, y0),
            dx,
            dy,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="ROI",
        )
        ax.add_patch(rect)
    elif isinstance(roi, CircularROI):
        # coords = [xc, yc, r]
        xc, yc, r = roi.coords
        circle = patches.Circle(
            (xc, yc), r, linewidth=2, edgecolor="red", facecolor="none", label="ROI"
        )
        ax.add_patch(circle)
    elif isinstance(roi, PolygonalROI):
        # coords = [x0, y0, x1, y1, x2, y2, ...]
        points = roi.coords.reshape(-1, 2)
        polygon = patches.Polygon(
            points,
            closed=True,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="ROI",
        )
        ax.add_patch(polygon)


def _add_geometry_to_axes(ax: plt.Axes, result: GeometryResult) -> None:
    """Add geometry result overlay to matplotlib axes.

    Iterates over all rows in result.coords to draw each geometric shape.
    Supports POINT, MARKER, RECTANGLE, CIRCLE, SEGMENT, ELLIPSE, and POLYGON.

    Args:
        ax: Matplotlib axes object
        result: GeometryResult object with shape information (coords is 2D array)
    """
    # Iterate over all rows in coords (each row is one shape)
    for coords in result.coords:
        if result.kind == KindShape.POINT:
            x0, y0 = coords
            ax.plot(
                x0,
                y0,
                marker="o",
                markersize=6,
                color="yellow",
                markeredgecolor="black",
                markeredgewidth=1,
            )
        elif result.kind == KindShape.MARKER:
            x0, y0 = coords
            # Marker with crosshair style (matching PlotPy behavior)
            ax.axhline(y0, color="yellow", linestyle="--", linewidth=1, alpha=0.7)
            ax.axvline(x0, color="yellow", linestyle="--", linewidth=1, alpha=0.7)
            ax.plot(
                x0,
                y0,
                marker="+",
                markersize=10,
                color="yellow",
                markeredgewidth=2,
            )
        elif result.kind == KindShape.RECTANGLE:
            x0, y0, dx, dy = coords
            rect = patches.Rectangle(
                (x0, y0),
                dx,
                dy,
                linewidth=2,
                edgecolor="yellow",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
        elif result.kind == KindShape.CIRCLE:
            xc, yc, r = coords
            circle = patches.Circle(
                (xc, yc),
                r,
                linewidth=2,
                edgecolor="yellow",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(circle)
        elif result.kind == KindShape.SEGMENT:
            x0, y0, x1, y1 = coords
            ax.plot([x0, x1], [y0, y1], "y--", linewidth=2)
        elif result.kind == KindShape.ELLIPSE:
            # For ellipse, coords are (xc, yc, a, b, theta)
            # Matplotlib Ellipse uses (center_x, center_y), width, height, angle_degrees
            xc, yc, a, b, theta = coords
            ellipse = patches.Ellipse(
                (xc, yc),
                2 * a,
                2 * b,
                angle=np.degrees(theta),
                linewidth=2,
                edgecolor="yellow",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(ellipse)
        elif result.kind == KindShape.POLYGON:
            x = coords[::2]
            y = coords[1::2]
            ax.plot(x, y, "y--", linewidth=2, marker="o", markersize=4)


def view_images_side_by_side(
    images: list[np.ndarray | ImageObj],
    titles: list[str] | None = None,
    share_axes: bool = True,
    rows: int | None = None,
    maximized: bool = False,  # Qt-specific  # pylint: disable=unused-argument
    title: str | None = None,
    results: list[GeometryResult] | GeometryResult | None = None,
    show_roi: bool = True,
    object_name: str = "",  # Qt-specific  # pylint: disable=unused-argument
    **kwargs,
) -> None:
    """Show sequence of images side by side.

    Args:
        images: List of `np.ndarray` or `ImageObj` objects to display
        titles: List of titles for each image
        share_axes: Whether to share axes across plots, default is True
        rows: Fixed number of rows in the grid, or None to compute automatically
        maximized: Whether to show the dialog maximized (unused in matplotlib - kept
         for API compatibility with PlotPy version)
        title: Title of the figure, or None for a default title
        results: Single `GeometryResult` or list of these to overlay on images, or None
         if no overlay is needed.
        show_roi: Whether to show ROIs defined in `ImageObj` instances, default is True
        object_name: Object name for screenshot functionality (unused in matplotlib -
         kept for API compatibility with PlotPy version)
        **kwargs: Additional keyword arguments (e.g., colormap settings)
    """
    n_images = len(images)

    # Compute grid layout
    if rows is not None:
        nrows = rows
        ncols = (n_images + nrows - 1) // nrows
    else:
        ncols = min(4, n_images)
        nrows = (n_images + ncols - 1) // ncols

    # Create figure
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 6 * nrows),
        sharex=share_axes,
        sharey=share_axes,
    )

    if title:
        fig.suptitle(title)

    # Flatten axes for easier iteration
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Prepare titles list
    if titles is None:
        titles = [None] * n_images
    elif len(titles) != n_images:
        raise ValueError("Length of titles must match length of images")

    # Prepare results list
    if results is None:
        results_list = [None] * n_images
    elif isinstance(results, (list, tuple)):
        if len(results) != n_images:
            raise ValueError("Length of results must match length of images")
        results_list = results
    else:
        results_list = [results] * n_images

    # Plot each image
    for idx, (ax, img, img_title, result) in enumerate(
        zip(axes, images, titles, results_list)
    ):
        # Extract data
        if isinstance(img, ImageObj):
            data = img.data
            img_title = img_title or img.title or f"Image {idx + 1}"
            is_image_obj = True
        elif isinstance(img, np.ndarray):
            data = img
            img_title = img_title or f"Image {idx + 1}"
            is_image_obj = False
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Compute extent and aspect ratio for ImageObj, use defaults for arrays
        if is_image_obj:
            extent, aspect_ratio = _get_image_extent_and_aspect(img)
        else:
            nrows_img, ncols_img = data.shape[:2]
            extent = [-0.5, ncols_img - 0.5, nrows_img - 0.5, -0.5]
            aspect_ratio = 1.0

        # Display image
        im = ax.imshow(
            data,
            cmap=kwargs.get("colormap", "viridis"),
            origin="upper",
            extent=extent,
            aspect=aspect_ratio,
        )
        ax.set_title(img_title)

        # Overlay mask if ImageObj has maskdata
        if is_image_obj and img.maskdata is not None:
            # Create semi-transparent red overlay for masked areas
            mask = img.maskdata
            mask_rgba = np.zeros((*mask.shape, 4))
            mask_rgba[mask, :] = [1, 0, 0, MASK_OPACITY]
            ax.imshow(mask_rgba, origin="upper", extent=extent)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Overlay ROIs
        if show_roi and is_image_obj and img.roi:
            for roi in img.roi:
                _add_single_roi_to_axes(ax, roi)

        # Overlay geometry results
        if result is not None:
            _add_geometry_to_axes(ax, result)

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def view_curves_and_images(
    data_or_objs: list[SignalObj | np.ndarray | ImageObj | np.ndarray],
    name: str | None = None,  # Qt-specific: unused in matplotlib implementation
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    object_name: str = "",  # Qt-specific: unused in matplotlib implementation
) -> None:
    """View signals, then images in two successive matplotlib figures.

    Args:
        data_or_objs: List of `SignalObj`, `ImageObj`, `np.ndarray` or a mix of these
        name: Name of the dialog (unused in matplotlib - kept for API compatibility
         with PlotPy version)
        title: Title of the plot, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        object_name: Object name for screenshot functionality (unused in matplotlib -
         kept for API compatibility with PlotPy version)
    """
    if isinstance(data_or_objs, (tuple, list)):
        objs = data_or_objs
    else:
        objs = [data_or_objs]

    # Separate signals and images
    sig_objs = []
    ima_objs = []

    for obj in objs:
        if isinstance(obj, SignalObj):
            sig_objs.append(obj)
        elif isinstance(obj, ImageObj):
            ima_objs.append(obj)
        elif isinstance(obj, np.ndarray):
            # Assume 1D is signal, 2D is image
            if obj.ndim == 1:
                sig_objs.append(obj)
            elif obj.ndim == 2:
                ima_objs.append(obj)

    # Display signals
    if sig_objs:
        view_curves(
            sig_objs,
            name=name,
            title=f"{title} - Curves" if title else None,
            xlabel=xlabel,
            ylabel=ylabel,
            xunit=xunit,
            yunit=yunit,
            object_name=f"{object_name}_curves",
        )

    # Display images
    if ima_objs:
        view_images(
            ima_objs,
            name=name,
            title=f"{title} - Images" if title else None,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            xunit=xunit,
            yunit=yunit,
            zunit=zunit,
            object_name=f"{object_name}_images",
        )


# Stub implementations for PlotPy-specific functions not supported in Matplotlib
# These raise NotImplementedError to alert users when trying to use unsupported features


def create_curve(x: np.ndarray, y: np.ndarray, title: str | None = None):
    """Not implemented - use view_curves() instead."""
    raise NotImplementedError(
        "create_curve() is not supported in matplotlib backend. "
        "Use view_curves() directly instead."
    )


def create_image(data: np.ndarray, **kwargs):
    """Not implemented - use view_images() instead."""
    raise NotImplementedError(
        "create_image() is not supported in matplotlib backend. "
        "Use view_images() directly instead."
    )


def create_contour_shapes(coords: np.ndarray, shape):
    """Not implemented in matplotlib backend."""
    raise NotImplementedError(
        "create_contour_shapes() is not supported in matplotlib backend."
    )


def create_circle(xc: float, yc: float, r: float, label: str | None = None):
    """Not implemented in matplotlib backend."""
    raise NotImplementedError("create_circle() is not supported in matplotlib backend.")


def create_segment(
    x0: float, y0: float, x1: float, y1: float, label: str | None = None
):
    """Not implemented in matplotlib backend."""
    raise NotImplementedError(
        "create_segment() is not supported in matplotlib backend."
    )


def create_cursor(orientation, position, label: str):
    """Not implemented in matplotlib backend."""
    raise NotImplementedError("create_cursor() is not supported in matplotlib backend.")


def create_range(orientation, pos_min: float, pos_max: float, title: str):
    """Not implemented in matplotlib backend."""
    raise NotImplementedError("create_range() is not supported in matplotlib backend.")


def create_label(text: str):
    """Not implemented in matplotlib backend."""
    raise NotImplementedError("create_label() is not supported in matplotlib backend.")


def create_marker(x: float, y: float, title: str | None = None):
    """Not implemented in matplotlib backend."""
    raise NotImplementedError("create_marker() is not supported in matplotlib backend.")


def view_curve_items(items: list, **kwargs):
    """Not implemented - use view_curves() instead."""
    raise NotImplementedError(
        "view_curve_items() is not supported in matplotlib backend. "
        "Use view_curves() directly instead."
    )


def view_image_items(items: list, **kwargs):
    """Not implemented - use view_images() instead."""
    raise NotImplementedError(
        "view_image_items() is not supported in matplotlib backend. "
        "Use view_images() directly instead."
    )


def guidata_exec_dialog(dialog):
    """Not implemented in matplotlib backend.

    This is a Qt-specific function from guidata that executes a dialog."""
    raise NotImplementedError(
        "guidata_exec_dialog() is not supported in matplotlib backend. "
        "This is a Qt-specific function."
    )
