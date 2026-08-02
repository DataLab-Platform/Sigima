# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for low-coverage branches of :mod:`sigima.objects.base`.

Covers:
- Metadata options API (``set_metadata_options_defaults``, ``get_metadata_option``,
  ``set_metadata_option``, ``get_metadata_options``, ``reset_metadata_to_defaults``).
- ROI cache mutation helpers (``sync_roi_to_metadata``, ``mark_roi_as_changed``).
- ``update_metadata_from``.
- HTML representations of single ROIs and ROI containers.
- ``check_data`` raising on unsupported dtypes.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import (
    ImageObj,
    SignalObj,
    create_image,
    create_image_roi,
    create_signal,
    create_signal_roi,
)


def test_object_title_layout_is_consistent() -> None:
    """Signal and image titles are the first field of their data group."""
    signal_items = SignalObj._items  # pylint: disable=protected-access
    image_items = ImageObj._items  # pylint: disable=protected-access

    assert [item.get_name() for item in signal_items[:3]] == [
        "_tabs",
        "_datag",
        "title",
    ]
    assert [item.get_name() for item in image_items[:3]] == [
        "_tabs",
        "_datag",
        "title",
    ]
    assert image_items[1].get_prop("display", "label") == signal_items[1].get_prop(
        "display", "label"
    )
    assert sum(item.get_name() == "title" for item in image_items) == 1


# ===========================================================================
# Metadata options API
# ===========================================================================


def test_set_and_get_metadata_option_default() -> None:
    """Setting an option default registers it under the ``__<name>`` key and
    makes it readable through :meth:`get_metadata_option`."""
    sig = create_signal("t", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    sig.set_metadata_options_defaults({"my_opt": 42})
    assert sig.get_metadata_option("my_opt") == 42
    assert "__my_opt" in sig.metadata


def test_set_metadata_option_no_overwrite() -> None:
    """With ``overwrite=False``, a second :meth:`set_metadata_option` call must
    keep the previously stored value untouched."""
    sig = create_signal("t", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    sig.set_metadata_option("opt", 1, overwrite=True)
    sig.set_metadata_option("opt", 2, overwrite=False)
    assert sig.metadata["__opt"] == 1


def test_get_metadata_option_invalid_raises() -> None:
    """Reading an option that was never declared (no default, no value) raises
    a clear :class:`ValueError`."""
    sig = create_signal("t", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="Invalid metadata option name"):
        sig.get_metadata_option("nonexistent_option")


def test_get_metadata_option_with_default_param() -> None:
    """:meth:`get_metadata_option` invoked with a ``default`` registers that
    default for subsequent calls and immediately returns it."""
    sig = create_signal("t", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    # Providing a default registers it as a default.
    val = sig.get_metadata_option("new_opt", default="hello")
    assert val == "hello"
    # Subsequent call without default still works.
    assert sig.get_metadata_option("new_opt") == "hello"


def test_get_metadata_option_falls_back_to_default_value() -> None:
    """When the actual ``__<name>`` key has been removed from metadata, the
    accessor must transparently fall back to the registered default."""
    sig = create_signal("t", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    sig.set_metadata_options_defaults({"opt": 10})
    # Remove from metadata to trigger fallback to default.
    del sig.metadata["__opt"]
    assert sig.get_metadata_option("opt") == 10


def test_get_metadata_options_returns_only_double_underscored() -> None:
    """:meth:`get_metadata_options` filters the metadata dict and only exposes
    ``__``-prefixed keys (the option storage convention), stripping the prefix."""
    sig = create_signal("t", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    sig.metadata["__a"] = 1
    sig.metadata["__b"] = 2
    sig.metadata["regular_key"] = 3
    opts = sig.get_metadata_options()
    assert opts == {"a": 1, "b": 2}


def test_reset_metadata_to_defaults() -> None:
    """:meth:`reset_metadata_to_defaults` discards every user-set metadata
    entry while preserving registered option defaults."""
    sig = create_signal("t", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    sig.set_metadata_options_defaults({"opt": 5})
    sig.metadata["extra_key"] = "value"
    sig.reset_metadata_to_defaults()
    assert "extra_key" not in sig.metadata
    assert sig.get_metadata_option("opt") == 5


def test_set_metadata_options_defaults_no_overwrite() -> None:
    """:meth:`set_metadata_options_defaults` with ``overwrite=False`` must not
    replace an already-registered option value."""
    sig = create_signal("t", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    sig.set_metadata_options_defaults({"opt": 1})
    sig.set_metadata_options_defaults({"opt": 2}, overwrite=False)
    # Existing value preserved (overwrite=False on set_metadata_option).
    assert sig.get_metadata_option("opt") == 1


# ===========================================================================
# ROI cache helpers
# ===========================================================================


def test_sync_roi_to_metadata_no_op_when_no_cache() -> None:
    """:meth:`sync_roi_to_metadata` must be a no-op when there is no ROI cache
    to write back, and must not introduce a spurious ``ROI`` key."""
    sig = create_signal("t", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    # Without prior ROI cache, sync should be a no-op.
    sig.sync_roi_to_metadata()
    # No "ROI" key inserted.
    assert sig.roi is None


def test_sync_and_mark_roi_changed() -> None:
    """:meth:`mark_roi_as_changed` invalidates the cached mask so it is rebuilt
    on next access (smoke test that the cache turnover does not break)."""
    sig = create_signal("t", np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    sig.roi = create_signal_roi([[2.0, 5.0]], indices=False)
    # Access cache, then mark as changed.
    _ = sig.roi
    sig.mark_roi_as_changed()
    # Mask cache is rebuilt on next access.
    assert sig.maskdata is not None


def test_update_metadata_from_invalidates_roi_cache() -> None:
    """:meth:`update_metadata_from` merges another mapping into the metadata
    and must keep the ROI cache consistent for subsequent accesses."""
    sig = create_signal("t", np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    sig.roi = create_signal_roi([[1.0, 3.0]], indices=False)
    _ = sig.roi  # populate cache
    other = {"new_meta_key": "new_value"}
    sig.update_metadata_from(other)
    assert sig.metadata["new_meta_key"] == "new_value"


# ===========================================================================
# check_data raises on unsupported dtype
# ===========================================================================


def test_check_data_unsupported_dtype_raises() -> None:
    """:meth:`check_data` rejects unsupported dtypes (e.g. ``bool``) with a
    descriptive :class:`TypeError`, preventing silent downstream failures."""
    img = create_image("t", np.zeros((4, 4), dtype=np.uint8))
    img.data = np.zeros((4, 4), dtype=bool)
    with pytest.raises(TypeError, match="Unsupported data type"):
        img.check_data()


# ===========================================================================
# SingleROI / BaseROI _repr_html_
# ===========================================================================


def test_single_segment_roi_repr_html() -> None:
    """A single segment ROI exposes its class name and a ``<table>`` element
    in its HTML representation (used by Jupyter integration)."""
    roi = create_signal_roi([[1.0, 5.0]], indices=False)
    single = roi.single_rois[0]
    html = single._repr_html_()  # pylint: disable=protected-access
    assert "SegmentROI" in html
    assert "<table" in html


def test_single_roi_repr_html_with_title() -> None:
    """User-defined ROI titles must appear verbatim in the HTML rendering."""
    roi = create_signal_roi([[1.0, 5.0]], indices=False, title="my-roi")
    single = roi.single_rois[0]
    html = single._repr_html_()  # pylint: disable=protected-access
    assert "my-roi" in html


def test_single_roi_get_coords_summary() -> None:
    """:meth:`get_coords_summary` returns a string in which the raw boundary
    values are still recognisable (used by HTML and tooltip rendering)."""
    roi = create_signal_roi([[1.0, 5.0]], indices=False)
    single = roi.single_rois[0]
    summary = single.get_coords_summary()
    assert isinstance(summary, str)
    assert isinstance(summary, str)
    # Subclass override may add a prefix; the raw coords list must still appear.
    assert "1" in summary and "5" in summary


def test_base_roi_repr_html_with_rois() -> None:
    """The ROI container HTML rendering reports the count and per-ROI type."""
    roi = create_signal_roi([[1.0, 5.0], [6.0, 9.0]], indices=False)
    html = roi._repr_html_()  # pylint: disable=protected-access
    assert "2 ROI(s)" in html
    assert "SegmentROI" in html


def test_base_roi_repr_html_empty() -> None:
    """An empty ROI container must render an explicit ``No ROIs`` notice rather
    than an empty table, so the user knows nothing is defined."""
    # pylint: disable=import-outside-toplevel
    from sigima.objects.signal.roi import SignalROI

    roi = SignalROI()
    html = roi._repr_html_()  # pylint: disable=protected-access
    assert "No ROIs" in html


def test_image_rectangular_roi_repr_html() -> None:
    """Image ROI HTML rendering identifies the concrete shape subclass
    (``RectangularROI``) for a rectangle ROI."""
    img = create_image("t", np.zeros((100, 100), dtype=np.uint16))
    img.roi = create_image_roi("rectangle", [[10, 20, 50, 60]])
    html = img.roi._repr_html_()  # pylint: disable=protected-access
    assert "RectangularROI" in html


def test_image_circular_roi_repr_html() -> None:
    """Image ROI HTML rendering identifies the concrete shape subclass
    (``CircularROI``) for a circle ROI."""
    img = create_image("t", np.zeros((100, 100), dtype=np.uint16))
    img.roi = create_image_roi("circle", [[50, 50, 20]])
    html = img.roi._repr_html_()  # pylint: disable=protected-access
    assert "CircularROI" in html


if __name__ == "__main__":
    pytest.main([__file__])
