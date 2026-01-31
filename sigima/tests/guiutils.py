# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Utilities to manage GUI activation for tests executed with pytest
or as standalone scripts.

?? This module must not import any Qt-related module at the top level,
    as Qt is an optional dependency of Sigima.
"""

from __future__ import annotations

import os
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Optional

from sigima.tests import SIGIMA_TESTS_GUI_ENV

if TYPE_CHECKING:
    # ?? Type-only: no runtime Qt import
    from qtpy.QtWidgets import QApplication


# Single source of truth (module-global); None means "not forced".
_FORCED_GUI: bool | None = None


def enable_gui(on: bool | None = True) -> None:
    """Force GUI mode on/off (or reset to auto when None)."""
    global _FORCED_GUI  # pylint: disable=global-statement
    _FORCED_GUI = on


def is_gui_enabled() -> bool:
    """Return True if GUI mode is enabled."""
    # 1) explicit override
    if _FORCED_GUI is not None:
        return _FORCED_GUI
    # 2) pytest --gui, exposed by conftest via env var (see below)
    if os.environ.get(SIGIMA_TESTS_GUI_ENV, "") in ("1", "true", "True"):
        return True
    return False


class DummyRequest:
    """
    Dummy request object to simulate pytest --gui when running a test manually.

    Example usage:
        test_x(request=DummyRequest(gui=True))
    """

    def __init__(self, gui: bool = True):
        self.config = types.SimpleNamespace()
        self.config.getoption = lambda name: gui if name == "--gui" else None


@contextmanager
def lazy_qt_app_context(
    *, exec_loop: bool = False, force: bool | None = None
) -> Generator[Optional[QApplication], None, None]:
    """Provide a Qt app context lazily; no-op if GUI is disabled.

    Args:
        exec_loop: Run the Qt event loop (e.g. when showing a non-blocking widget).
        force: None ? auto (use is_gui_enabled());
               True ? force GUI ON (always create Qt app);
               False ? force GUI OFF (no-op).

    Yields:
        The QApplication instance if enabled, else None.

    .. note::

       This context manager is useful for tests that require a Qt application context,
       but should be used with caution to avoid unnecessary Qt imports. For tests
       that are exclusively GUI-based, option `force=True` can be used to ensure
       the Qt application context is always created. For tests that must be executable
       without a GUI, option `force` may be skipped so that operations inside the
       context are only performed if the GUI is enabled.
    """
    enabled = is_gui_enabled() if force is None else force
    if not enabled:
        # No Qt import, block executes as a no-op context
        yield None
        return

    # Lazy import: only when enabled
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context(exec_loop=exec_loop) as qt_app:
        yield qt_app


def _viz_call_if_gui(func_name: str, *args, **kwargs) -> bool:
    """Call sigima.viz.<func_name>(...) only if GUI is enabled.

    Returns:
        True if the call executed (GUI enabled or forced), else False.
    """
    with lazy_qt_app_context() as app:
        if app is None:
            return False
        from sigima import viz  # pylint: disable=import-outside-toplevel

        getattr(viz, func_name)(*args, **kwargs)
        return True


def view_curves_if_gui(*args, **kwargs) -> None:
    """Create a curve dialog and plot curves if GUI mode enabled.

    Args:
        data_or_objs: Single `SignalObj` or `np.ndarray`, or a list/tuple of these,
         or a list/tuple of (xdata, ydata) pairs
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
    """
    return _viz_call_if_gui("view_curves", *args, **kwargs)


def view_images_if_gui(*args, **kwargs) -> None:
    """Show sequence of images if GUI mode enabled.

    Args:
        data_or_objs: Single `ImageObj` or `np.ndarray`, or a list/tuple of these
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
    """
    return _viz_call_if_gui("view_images", *args, **kwargs)


def view_curves_and_images_if_gui(*args, **kwargs) -> None:
    """View signals, then images in two successive dialogs if GUI mode enabled.

    Args:
        data_or_objs: List of `SignalObj`, `ImageObj`, `np.ndarray` or a mix of these
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
    """
    return _viz_call_if_gui("view_curves_and_images", *args, **kwargs)


def view_images_side_by_side_if_gui(*args, **kwargs) -> None:
    """Show sequence of images side-by-side if GUI mode enabled.

    Args:
        images: List of `ImageItem`, `np.ndarray`, or `ImageObj` objects to display
        titles: List of titles for each image
        share_axes: Whether to share axes across plots, default is True
        rows: Fixed number of rows in the grid, or None to compute automatically
        maximized: Whether to show the dialog maximized, default is False
        title: Title of the dialog, or None for a default title
    """
    return _viz_call_if_gui("view_images_side_by_side", *args, **kwargs)
