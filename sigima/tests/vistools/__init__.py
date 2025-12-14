# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Visualization tools dispatcher for Sigima tests
===============================================

This module automatically selects between PlotPy and Matplotlib backends based on
availability and configuration settings.

The backend selection follows this priority:
1. Environment variable SIGIMA_VISTOOLS_BACKEND (if set)
2. Configuration option sigima.config.options.vistools_backend
3. Auto-detection (PlotPy preferred, Matplotlib as fallback)

Backend selection logic:
- "auto": Try PlotPy first, fall back to Matplotlib
- "plotpy": Use PlotPy (raise ImportError if not available)
- "matplotlib": Use Matplotlib (raise ImportError if not available)

Module exports:
- BACKEND_NAME: Name of the selected backend ("plotpy" or "matplotlib")
- BACKEND_SOURCE: How the backend was selected ("env", "config", or "auto")
- All public functions from the selected backend module
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import type hints for static analysis (Pylance, Pylint, mypy)
    # These imports only happen during type checking, not at runtime
    import numpy as np

    from sigima.objects import GeometryResult, ImageObj

    # Type stub declarations for public API
    # These tell static analyzers what functions are available
    BACKEND_NAME: str
    BACKEND_SOURCE: str

    def view_curves(
        curves: list,
        titles: list[str] | None = None,
        title: str | None = None,
        maximized: bool = False,
        results: list[GeometryResult] | GeometryResult | None = None,
        show_roi: bool = True,
        object_name: str = "",
        **kwargs,
    ) -> None: ...

    def view_images(
        images: list,
        titles: list[str] | None = None,
        title: str | None = None,
        maximized: bool = False,
        results: list[GeometryResult] | GeometryResult | None = None,
        show_roi: bool = True,
        object_name: str = "",
        **kwargs,
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    def view_curve_items(
        items: list, title: str | None = None, object_name: str = ""
    ) -> None: ...

    def view_image_items(
        items: list, title: str | None = None, object_name: str = ""
    ) -> None: ...

    def create_curve(x: np.ndarray, y: np.ndarray, title: str | None = None): ...

    def create_image(
        data: np.ndarray,
        title: str | None = None,
        x0: float | None = None,
        y0: float | None = None,
        dx: float | None = None,
        dy: float | None = None,
        **kwargs,
    ): ...

    def create_contour_shapes(
        obj: ImageObj,
        threshold: float,
        kind: str = "polygon",
    ) -> list: ...

    def create_circle(
        xc: float,
        yc: float,
        r: float,
        title: str | None = None,
        **kwargs,
    ): ...

    def create_segment(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        title: str | None = None,
        **kwargs,
    ): ...

    def create_cursor(
        x: float,
        y: float,
        title: str | None = None,
        **kwargs,
    ): ...

    def create_range(
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        title: str | None = None,
        **kwargs,
    ): ...

    def create_label(text: str): ...

    def create_marker(x: float, y: float, title: str | None = None): ...


# Determine which backend to use
_backend_name: str | None = None
_backend_source: str = "auto"


def _select_backend() -> tuple[str, str]:
    """Select visualization backend based on configuration and availability.

    Returns:
        Tuple of (backend_name, source) where:
        - backend_name: "plotpy" or "matplotlib"
        - source: How the backend was selected ("env", "config", "auto")

    Raises:
        ImportError: If no suitable backend is available or selected backend not found
    """
    # Priority 1: Environment variable
    env_backend = os.environ.get("SIGIMA_VISTOOLS_BACKEND", "").lower()
    if env_backend in ("plotpy", "matplotlib", "auto"):
        requested = env_backend
        source = "env"
    else:
        # Priority 2: Configuration option
        try:
            from sigima.config import options

            requested = options.vistools_backend.get(sync_env=False).lower()
            source = "config"
        except Exception:  # pylint: disable=broad-except
            requested = "auto"
            source = "auto"

    # Try to import based on request
    if requested == "plotpy":
        try:
            import plotpy  # noqa: F401 # pylint: disable=unused-import,import-outside-toplevel

            return ("plotpy", source)
        except ImportError as exc:
            raise ImportError(
                "PlotPy backend requested but PlotPy is not installed. "
                "Install with: pip install PlotPy"
            ) from exc

    elif requested == "matplotlib":
        try:
            import matplotlib  # noqa: F401 # pylint: disable=unused-import,import-outside-toplevel

            return ("matplotlib", source)
        except ImportError as exc:
            raise ImportError(
                "Matplotlib backend requested but Matplotlib is not installed. "
                "Install with: pip install matplotlib"
            ) from exc

    else:  # "auto"
        # Try PlotPy first
        try:
            import plotpy  # noqa: F401 # pylint: disable=unused-import,import-outside-toplevel

            return ("plotpy", source)
        except ImportError:
            pass

        # Fall back to Matplotlib
        try:
            import matplotlib  # noqa: F401 # pylint: disable=unused-import,import-outside-toplevel

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
_backend_module = None
_backend_name = None
_backend_source = None
_initializing = False  # Flag to prevent recursion


def _initialize_backend():
    """Initialize backend on first use (lazy loading)."""
    global _backend_module, _backend_name, _backend_source, _initializing

    if _backend_module is not None:
        return  # Already initialized

    if _initializing:
        return  # Prevent recursion during import

    _initializing = True
    try:
        _backend_name, _backend_source = _select_backend()

        # Import selected backend using importlib to avoid triggering __getattr__
        import importlib

        if _backend_name == "plotpy":
            _backend_module = importlib.import_module(
                ".vistools_plotpy", package=__name__
            )
        elif _backend_name == "matplotlib":
            _backend_module = importlib.import_module(".vistools_mpl", package=__name__)
    finally:
        _initializing = False


def __getattr__(name: str):
    """Lazy loading of backend attributes."""
    # Handle special/dunder attributes that inspect might access
    # Raise AttributeError immediately to avoid backend initialization
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    if name in ("BACKEND_NAME", "BACKEND_SOURCE", "_backend_module"):
        try:
            _initialize_backend()
        except ImportError:
            # During pytest collection or in environments without backends
            raise AttributeError(
                f"module '{__name__}' has no attribute '{name}'"
            ) from None
        if name == "BACKEND_NAME":
            return _backend_name
        if name == "BACKEND_SOURCE":
            return _backend_source
        if name == "_backend_module":
            return _backend_module

    # For any other attribute, initialize backend and forward to backend module
    try:
        _initialize_backend()
        return getattr(_backend_module, name)
    except ImportError:
        # Backend not available - raise AttributeError for clean failure
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None
    except AttributeError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None


def __dir__():
    """Return list of available attributes (with lazy initialization)."""
    try:
        _initialize_backend()
        base_attrs = ["BACKEND_NAME", "BACKEND_SOURCE"]
        backend_attrs = [
            name for name in dir(_backend_module) if not name.startswith("_")
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
