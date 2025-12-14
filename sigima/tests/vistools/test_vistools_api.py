# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test vistools API compatibility between backends
"""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import inspect
import sys

import pytest


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401

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
    from . import vistools_mpl, vistools_plotpy

    plotpy_funcs = get_public_functions(vistools_plotpy)
    mpl_funcs = get_public_functions(vistools_mpl)

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


def test_backend_selection_env_var(monkeypatch):
    """Test that SIGIMA_VISTOOLS_BACKEND environment variable works."""
    import importlib

    # Check if matplotlib is available
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not available")

    # Test with matplotlib
    monkeypatch.setenv("SIGIMA_VISTOOLS_BACKEND", "matplotlib")
    # Reload the module to pick up new env var
    if "sigima.tests.vistools" in sys.modules:
        importlib.reload(sys.modules["sigima.tests.vistools"])

    from sigima.tests import vistools

    assert vistools.BACKEND_NAME == "matplotlib"
    assert vistools.BACKEND_SOURCE == "env"


def test_backend_selection_option(monkeypatch):
    """Test that configuration option vistools_backend works."""
    import importlib

    # Check if matplotlib is available
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not available")

    from sigima.config import options

    # Clear env var to test config option
    monkeypatch.delenv("SIGIMA_VISTOOLS_BACKEND", raising=False)

    # Test with matplotlib - use .set() method
    options.vistools_backend.set("matplotlib")
    # Reload the module to pick up new config option
    if "sigima.tests.vistools" in sys.modules:
        importlib.reload(sys.modules["sigima.tests.vistools"])

    from sigima.tests import vistools

    assert vistools.BACKEND_NAME == "matplotlib"
    assert vistools.BACKEND_SOURCE == "config"


def test_backend_info_available():
    """Test that backend information is exposed."""
    # Check if any backend is available
    try:
        import matplotlib  # noqa: F401

        backend_available = True
    except ImportError:
        try:
            import plotpy  # noqa: F401

            backend_available = True
        except ImportError:
            backend_available = False

    if not backend_available:
        pytest.skip("No visualization backend available")

    from sigima.tests import vistools

    assert hasattr(vistools, "BACKEND_NAME")
    assert hasattr(vistools, "BACKEND_SOURCE")
    assert vistools.BACKEND_NAME in ("plotpy", "matplotlib")
    assert vistools.BACKEND_SOURCE in ("env", "config", "auto")


if __name__ == "__main__":
    test_matplotlib_backend_has_all_plotpy_functions()
    test_backend_info_available()
    test_backend_selection_env_var()
    test_backend_selection_option()
