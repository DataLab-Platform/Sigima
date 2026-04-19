# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for backend selection logic in :mod:`sigima.viz`.

The backend stub functions in :mod:`sigima.viz.viz_mpl` are intentionally
omitted from coverage (see ``.coveragerc``) and are not tested here, since
they are trivial ``raise NotImplementedError`` placeholders.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

# ===========================================================================
# Backend availability detection (CI may have neither PlotPy nor Matplotlib)
# ===========================================================================

HAS_PLOTPY = importlib.util.find_spec("plotpy") is not None
HAS_MPL = importlib.util.find_spec("matplotlib") is not None
HAS_ANY_BACKEND = HAS_PLOTPY or HAS_MPL


# ===========================================================================
# Backend selection (sigima.viz)
# ===========================================================================


def _reload_viz():
    """Reload the viz package so module-level state is reset."""
    if "sigima.viz" in sys.modules:
        del sys.modules["sigima.viz"]
    return importlib.import_module("sigima.viz")


@pytest.mark.skipif(not HAS_PLOTPY, reason="PlotPy not installed")
def test_select_backend_via_env_plotpy(monkeypatch):
    """Setting ``SIGIMA_VIZ_BACKEND=plotpy`` selects the PlotPy backend via env."""
    monkeypatch.setenv("SIGIMA_VIZ_BACKEND", "plotpy")
    viz = _reload_viz()
    name, source = viz._select_backend()  # pylint: disable=protected-access
    assert source == "env"
    assert name in ("plotpy", "matplotlib") or name == "plotpy"


@pytest.mark.skipif(not HAS_MPL, reason="Matplotlib not installed")
def test_select_backend_via_env_matplotlib(monkeypatch):
    """Setting ``SIGIMA_VIZ_BACKEND=matplotlib`` forces the Matplotlib backend."""
    monkeypatch.setenv("SIGIMA_VIZ_BACKEND", "matplotlib")
    viz = _reload_viz()
    name, source = viz._select_backend()  # pylint: disable=protected-access
    assert source == "env"
    assert name == "matplotlib"


@pytest.mark.skipif(not HAS_ANY_BACKEND, reason="No viz backend installed")
def test_select_backend_via_env_auto(monkeypatch):
    """``SIGIMA_VIZ_BACKEND=auto`` is a recognised value that triggers detection."""
    monkeypatch.setenv("SIGIMA_VIZ_BACKEND", "auto")
    viz = _reload_viz()
    name, source = viz._select_backend()  # pylint: disable=protected-access
    assert source == "env"
    assert name in ("plotpy", "matplotlib")


@pytest.mark.skipif(not HAS_ANY_BACKEND, reason="No viz backend installed")
def test_select_backend_unrecognized_env_falls_back(monkeypatch):
    """An unrecognised ``SIGIMA_VIZ_BACKEND`` value must not crash: the
    selection logic must fall back to the configuration / auto-detection
    code path instead of raising."""
    monkeypatch.setenv("SIGIMA_VIZ_BACKEND", "garbage")
    viz = _reload_viz()
    name, source = viz._select_backend()  # pylint: disable=protected-access
    assert source in ("config", "auto")
    assert name in ("plotpy", "matplotlib")


def test_dunder_attribute_raises(monkeypatch):
    """Accessing an unknown ``__dunder__`` attribute must raise ``AttributeError``
    so that ``hasattr``-based introspection (e.g. by IPython) keeps working."""
    monkeypatch.delenv("SIGIMA_VIZ_BACKEND", raising=False)
    viz = _reload_viz()
    with pytest.raises(AttributeError):
        viz.__nonexistent_dunder__  # noqa: B018  # pylint: disable=pointless-statement


def test_unknown_attribute_raises(monkeypatch):
    """Names not in ``__all__`` must raise ``AttributeError`` rather than
    silently returning ``None`` from the lazy ``__getattr__`` hook."""
    monkeypatch.delenv("SIGIMA_VIZ_BACKEND", raising=False)
    viz = _reload_viz()
    with pytest.raises(AttributeError):
        viz.not_in_all  # noqa: B018  # pylint: disable=pointless-statement


def test_dir_returns_known_attrs(monkeypatch):
    """``dir(sigima.viz)`` must expose ``BACKEND_NAME``/``BACKEND_SOURCE`` so
    that interactive auto-completion can surface the resolved backend."""
    monkeypatch.delenv("SIGIMA_VIZ_BACKEND", raising=False)
    viz = _reload_viz()
    listing = dir(viz)
    assert "BACKEND_NAME" in listing
    assert "BACKEND_SOURCE" in listing


@pytest.mark.skipif(not HAS_ANY_BACKEND, reason="No viz backend installed")
def test_lazy_attribute_access_initializes_backend(monkeypatch):
    """Backend selection is deferred until first use: ``BACKEND_NAME`` starts as
    ``None`` and gets populated only when a public attribute is touched.
    This keeps ``import sigima.viz`` cheap and side-effect-free."""
    monkeypatch.delenv("SIGIMA_VIZ_BACKEND", raising=False)
    viz = _reload_viz()
    # Before access, BACKEND_NAME is None.
    assert viz.BACKEND_NAME is None
    # Accessing a function in __all__ triggers lazy initialization.
    func = viz.view_curves
    assert callable(func)
    # Now BACKEND_NAME should be populated.
    assert viz.BACKEND_NAME in ("plotpy", "matplotlib")
    assert viz.BACKEND_SOURCE in ("env", "config", "auto")


if __name__ == "__main__":
    pytest.main([__file__])
