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


# Select backend at module import time
_backend_name, _backend_source = _select_backend()

# Import all symbols from selected backend
if _backend_name == "plotpy":
    from . import vistools_plotpy as _backend_module
    from .vistools_plotpy import *  # noqa: F401,F403 # pylint: disable=wildcard-import,unused-wildcard-import
elif _backend_name == "matplotlib":
    from . import vistools_mpl as _backend_module
    from .vistools_mpl import *  # noqa: F401,F403 # pylint: disable=wildcard-import,unused-wildcard-import

# Export backend information
BACKEND_NAME = _backend_name
BACKEND_SOURCE = _backend_source

# Build __all__ list: backend info + all public functions from backend module
__all__ = ["BACKEND_NAME", "BACKEND_SOURCE"] + [
    name
    for name in dir(_backend_module)
    if not name.startswith("_") and name not in ("BACKEND_NAME", "BACKEND_SOURCE")
]
