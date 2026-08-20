# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test viz API compatibility between backends
"""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import inspect
import sys

import pytest


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401  # pylint: disable=unused-import

        return True
    except ImportError:
        return False


def get_public_functions(module) -> set[str]:
    """Get all public function names from a module.

    Args:
        module: Module object to inspect

    Returns:
        Set of public function names (not starting with '_') that are
        defined in the module (not just imported from elsewhere)
    """
    return {
        name
        for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and obj.__module__ == module.__name__
    }


@pytest.mark.skipif(
    "matplotlib" not in sys.modules and not _has_matplotlib(),
    reason="matplotlib not available",
)
def test_matplotlib_backend_has_all_plotpy_functions():
    """Test that matplotlib backend implements stubs for all PlotPy functions.

    This ensures that all public API functions from PlotPy backend are at least
    present in the Matplotlib backend (even if they raise NotImplementedError).
    """
    # Import both backend modules directly
    from sigima.viz import viz_mpl, viz_plotpy

    plotpy_funcs = get_public_functions(viz_plotpy)
    mpl_funcs = get_public_functions(viz_mpl)

    # Functions that are expected to be missing in matplotlib
    # (internal helpers that don't need to be exposed)
    expected_missing = set()

    # Check for missing functions
    missing_funcs = plotpy_funcs - mpl_funcs - expected_missing

    if missing_funcs:
        missing_list = "\n  - ".join(sorted(missing_funcs))
        pytest.fail(
            "Matplotlib backend is missing the following functions:"
            f"\n  - {missing_list}\n\n"
            "These functions should be added as stubs that raise NotImplementedError."
        )


def test_annotation_visibility_parameter_has_backend_parity() -> None:
    """Check the public annotation visibility switch on both backends."""
    from sigima.viz import viz_mpl, viz_plotpy

    for function_name in (
        "view_curves",
        "view_images",
        "view_images_side_by_side",
        "view_curves_and_images",
    ):
        for backend in (viz_mpl, viz_plotpy):
            parameter = inspect.signature(getattr(backend, function_name)).parameters[
                "show_annotations"
            ]
            assert parameter.default is True


def test_backend_selection_env_var(monkeypatch):
    """Test that SIGIMA_VIZ_BACKEND environment variable works."""
    import importlib

    # Check if matplotlib is available
    try:
        import matplotlib  # noqa: F401  # pylint: disable=unused-import
    except ImportError:
        pytest.skip("matplotlib not available")

    # Test with matplotlib
    monkeypatch.setenv("SIGIMA_VIZ_BACKEND", "matplotlib")
    # Reload the module to pick up new env var
    if "sigima.viz" in sys.modules:
        importlib.reload(sys.modules["sigima.viz"])

    # Access the reloaded module directly from sys.modules
    viz = sys.modules["sigima.viz"]
    # Trigger lazy initialization by accessing a function from __all__
    _ = viz.view_images

    assert viz.BACKEND_NAME == "matplotlib"
    assert viz.BACKEND_SOURCE == "env"


def test_backend_selection_option(monkeypatch):
    """Test that configuration option viz_backend works."""
    import importlib

    # Check if matplotlib is available
    try:
        import matplotlib  # noqa: F401  # pylint: disable=unused-import
    except ImportError:
        pytest.skip("matplotlib not available")

    from sigima.config import options

    # Clear env var to test config option
    monkeypatch.delenv("SIGIMA_VIZ_BACKEND", raising=False)

    # Test with matplotlib - use .set() method
    options.viz_backend.set("matplotlib")
    # Reload the module to pick up new config option
    if "sigima.viz" in sys.modules:
        importlib.reload(sys.modules["sigima.viz"])

    # Access the reloaded module directly from sys.modules
    viz = sys.modules["sigima.viz"]
    # Trigger lazy initialization by accessing a function from __all__
    _ = viz.view_images

    assert viz.BACKEND_NAME == "matplotlib"
    assert viz.BACKEND_SOURCE == "config"


def test_backend_info_available():
    """Test that backend information is exposed."""
    # Check if any backend is available
    try:
        import matplotlib  # noqa: F401  # pylint: disable=unused-import

        backend_available = True
    except ImportError:
        try:
            import plotpy  # noqa: F401  # pylint: disable=unused-import

            backend_available = True
        except ImportError:
            backend_available = False

    if not backend_available:
        pytest.skip("No visualization backend available")

    from sigima import viz

    # Trigger lazy initialization by accessing a function from __all__
    _ = viz.view_images

    assert hasattr(viz, "BACKEND_NAME")
    assert hasattr(viz, "BACKEND_SOURCE")
    assert viz.BACKEND_NAME in ("plotpy", "matplotlib")
    assert viz.BACKEND_SOURCE in ("env", "config", "auto")


if __name__ == "__main__":
    pytest.main([__file__])
