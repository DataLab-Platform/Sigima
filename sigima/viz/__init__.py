# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Visualization tools for Sigima
==============================

This module provides visualization utilities for Sigima objects, useful for:
- Interactive testing and debugging
- Data analysis in Jupyter notebooks
- Quick visual inspection of processing results

The module automatically selects between PlotPy and Matplotlib backends based on
availability and configuration settings.

The backend selection follows this priority:
1. Environment variable SIGIMA_VIZ_BACKEND (if set)
2. Configuration option sigima.config.options.viz_backend
3. Auto-detection (PlotPy preferred, Matplotlib as fallback)

Backend selection logic:
- "auto": Try PlotPy first, fall back to Matplotlib
- "plotpy": Use PlotPy (raise ImportError if not available)
- "matplotlib": Use Matplotlib (raise ImportError if not available)

Module exports:
- BACKEND_NAME: Name of the selected backend ("plotpy" or "matplotlib")
- BACKEND_SOURCE: How the backend was selected ("env", "config", or "auto")
- All public functions from the selected backend module

Example usage::

    from sigima import viz

    # View signal objects
    viz.view_curves([signal1, signal2], title="My signals")

    # View image objects with ROIs
    viz.view_images([image], show_roi=True)

    # Compare images side by side
    viz.view_images_side_by_side([img1, img2], titles=["Before", "After"])
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import type hints for static analysis (Pylance, Pylint, mypy)
    # These imports only happen during type checking, not at runtime
    import numpy as np

    from sigima.objects import GeometryResult, ImageObj

    # Type stub declarations for public API
    # These tell static analyzers what functions are available
    BACKEND_NAME: str
    BACKEND_SOURCE: str

    # pylint: disable=unused-argument

    def view_curves(
        curves: list,
        titles: list[str] | None = None,
        title: str | None = None,
        maximized: bool = False,
        results: list[GeometryResult] | GeometryResult | None = None,
        show_roi: bool = True,
        object_name: str = "",
        **kwargs,
    ) -> None:
        """Display multiple curves in a dialog."""

    def view_images(
        images: list,
        titles: list[str] | None = None,
        title: str | None = None,
        maximized: bool = False,
        results: list[GeometryResult] | GeometryResult | None = None,
        show_roi: bool = True,
        object_name: str = "",
        **kwargs,
    ) -> None:
        """Display multiple images in a dialog."""

    def view_images_side_by_side(
        images: list,
        titles: list[str] | None = None,
        share_axes: bool = True,
        rows: int | None = None,
        maximized: bool = False,
        title: str | None = None,
        results: list[GeometryResult] | GeometryResult | None = None,
        show_roi: bool = True,
        object_name: str = "",
        **kwargs,
    ) -> None:
        """Display images side by side in a grid layout."""

    def view_curves_and_images(
        curves: list,
        images: list,
        curve_titles: list[str] | None = None,
        image_titles: list[str] | None = None,
        title: str | None = None,
        maximized: bool = False,
        results: list[GeometryResult] | GeometryResult | None = None,
        show_roi: bool = True,
        object_name: str = "",
        **kwargs,
    ) -> None:
        """Display curves and images together in a dialog."""

    def view_curve_items(
        items: list,
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
        """Display curve items in a plot dialog."""

    def view_image_items(
        items: list,
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
        """Display image items in a plot dialog."""

    def create_curve(x: np.ndarray, y: np.ndarray, title: str | None = None) -> Any:
        """Create a curve item from x and y data."""

    def create_image(
        data: np.ndarray,
        title: str | None = None,
        interpolation: str = "linear",
        colormap: str | None = None,
        alpha_function: str | None = None,
        xdata: list[float] | None = None,
        ydata: list[float] | None = None,
        **kwargs,
    ) -> Any:
        """Create an image item from array data."""
        return object()

    def create_contour_shapes(
        obj: ImageObj,
        threshold: float,
        kind: str = "polygon",
    ) -> list[Any]:
        """Create contour shape items from image object."""
        return []

    def create_circle(
        xc: float,
        yc: float,
        r: float,
        title: str | None = None,
        **kwargs,
    ) -> Any:
        """Create a circle annotation item."""
        return object()

    def create_segment(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        title: str | None = None,
        **kwargs,
    ) -> Any:
        """Create a segment annotation item."""
        return object()

    def create_cursor(
        orientation: str,
        position: float | tuple[float, float],
        label: str,
    ) -> Any:
        """Create a cursor marker item."""
        return object()

    def create_range(
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        title: str | None = None,
        **kwargs,
    ) -> Any:
        """Create a range annotation item."""
        return object()

    def create_label(text: str) -> Any:
        """Create a text label item."""
        return object()

    def create_marker(x: float, y: float, title: str | None = None) -> Any:
        """Create a marker item at specified coordinates."""
        return object()


def _select_backend() -> tuple[str, str]:
    """Select visualization backend based on configuration and availability.

    Returns:
        Tuple of (backend_name, source) where:
        - backend_name: "plotpy" or "matplotlib"
        - source: How the backend was selected ("env", "config", "auto")

    Raises:
        ImportError: If no suitable backend is available or selected backend not found
    """
    # pylint: disable=import-outside-toplevel
    # pylint: disable=unused-import

    # Priority 1: Environment variable
    env_backend = os.environ.get("SIGIMA_VIZ_BACKEND", "").lower()
    if env_backend in ("plotpy", "matplotlib", "auto"):
        requested = env_backend
        source = "env"
    else:
        # Priority 2: Configuration option
        try:
            from sigima.config import options

            requested = options.viz_backend.get(sync_env=False).lower()
            source = "config"
        except Exception:  # pylint: disable=broad-exception-caught
            requested = "auto"
            source = "auto"

    # Try to import based on request
    if requested == "plotpy":
        try:
            import plotpy  # noqa: F401

            return ("plotpy", source)
        except ImportError as exc:
            raise ImportError(
                "PlotPy backend requested but PlotPy is not installed. "
                "Install with: pip install PlotPy"
            ) from exc

    elif requested == "matplotlib":
        try:
            import matplotlib  # noqa: F401

            return ("matplotlib", source)
        except ImportError as exc:
            raise ImportError(
                "Matplotlib backend requested but Matplotlib is not installed. "
                "Install with: pip install matplotlib"
            ) from exc

    else:  # "auto"
        # Try PlotPy first
        try:
            import plotpy  # noqa: F401

            return ("plotpy", source)
        except ImportError:
            pass

        # Fall back to Matplotlib
        try:
            import matplotlib  # noqa: F401

            return ("matplotlib", source)
        except ImportError:
            pass

        # Neither available
        raise ImportError(
            "No visualization backend available. Please install either:\n"
            "  - PlotPy: pip install PlotPy (recommended for interactive features)\n"
            "  - Matplotlib: pip install matplotlib (simpler, view-only)"
        )


# Lazy backend initialization - deferred until first attribute access
_BACKEND_MODULE = None
_BACKEND_NAME: str | None = None
_BACKEND_SOURCE: str | None = None
_INITIALIZING = False  # Flag to prevent recursion

# Public API: Set default values for BACKEND_NAME and BACKEND_SOURCE
# These will be updated when the backend is initialized
BACKEND_NAME = None  # Will be "plotpy" or "matplotlib" after initialization
BACKEND_SOURCE = None  # Will be "env", "config", or "auto" after initialization


def _initialize_backend():
    """Initialize backend on first use (lazy loading)."""
    # pylint: disable=global-statement
    global _BACKEND_MODULE, _BACKEND_NAME, _BACKEND_SOURCE, _INITIALIZING
    global BACKEND_NAME, BACKEND_SOURCE

    if _BACKEND_MODULE is not None:
        return  # Already initialized

    if _INITIALIZING:
        return  # Prevent recursion during import

    _INITIALIZING = True
    try:
        _BACKEND_NAME, _BACKEND_SOURCE = _select_backend()

        # Update public API variables
        BACKEND_NAME = _BACKEND_NAME
        BACKEND_SOURCE = _BACKEND_SOURCE

        # Import selected backend using importlib to avoid triggering __getattr__
        if _BACKEND_NAME == "plotpy":
            _BACKEND_MODULE = importlib.import_module(".viz_plotpy", package=__name__)
        elif _BACKEND_NAME == "matplotlib":
            _BACKEND_MODULE = importlib.import_module(".viz_mpl", package=__name__)
    finally:
        _INITIALIZING = False


def __getattr__(name: str):
    """Lazy loading of backend attributes."""
    # Handle special/dunder attributes that inspect might access
    # Raise AttributeError immediately to avoid backend initialization
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # For functions in __all__, try to initialize backend and forward to backend module
    # If backend is not available, return a placeholder function
    if name in __all__:
        try:
            _initialize_backend()
            if _BACKEND_MODULE is not None:
                return getattr(_BACKEND_MODULE, name)
        except ImportError:
            pass  # Fall through to placeholder

        # Return a placeholder function that will raise an error when called
        def _placeholder(*args, **kwargs):
            raise ImportError(
                f"Function '{name}' requires a visualization backend. "
                "Please install either PlotPy or Matplotlib."
            )

        _placeholder.__name__ = name
        _placeholder.__doc__ = f"Placeholder for {name} (backend not available)"
        return _placeholder

    # For other attributes, raise AttributeError
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    """Return list of available attributes (with lazy initialization)."""
    try:
        _initialize_backend()
        base_attrs = ["BACKEND_NAME", "BACKEND_SOURCE"]
        backend_attrs = [
            name for name in dir(_BACKEND_MODULE) if not name.startswith("_")
        ]
        return base_attrs + backend_attrs
    except ImportError:
        # During pytest collection or in environments without visualization backends,
        # return minimal attributes to avoid breaking inspect.getmembers()
        return ["BACKEND_NAME", "BACKEND_SOURCE"]


# Define __all__ to include expected public API
__all__ = [
    "BACKEND_NAME",
    "BACKEND_SOURCE",
    # Creation functions
    "create_circle",
    "create_contour_shapes",
    "create_cursor",
    "create_curve",
    "create_image",
    "create_label",
    "create_marker",
    "create_range",
    "create_segment",
    # Viewer functions
    "view_curve_items",
    "view_curves",
    "view_curves_and_images",
    "view_image_items",
    "view_images",
    "view_images_side_by_side",
]
