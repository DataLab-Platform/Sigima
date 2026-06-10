# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for :mod:`sigima.objects.scalar.common`.

Covers branches not exercised by the existing test suite:
- ``set_display_preferences`` clearing path
- ``apply_visible_only_filter`` empty-headers fallback
- ``ResultHtmlGenerator`` callable column formats, single-row transpose
- ``_get_row_headers`` with no roi indices and out-of-bounds indices

Note: ``format_legend_value`` is exhaustively covered by
``format_legend_value_unit_test.py``; no test for it is repeated here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sigima.objects.scalar.common import (
    DataFrameManager,
    DisplayPreferencesManager,
    ResultHtmlGenerator,
)
from sigima.objects.scalar.table import TableResult

# ===========================================================================
# DisplayPreferencesManager
# ===========================================================================


def _make_table(headers=("a", "b", "c"), rows=None) -> TableResult:
    """Build a small :class:`TableResult` for HTML / preference tests."""
    rows = rows or [[1.0, 2.0, 3.0]]
    return TableResult(title="t", headers=list(headers), data=rows)


def test_set_display_preferences_clears_attr() -> None:
    """When all columns become visible again, the ``hidden_headers`` attribute
    must be removed (rather than stored as an empty list) to keep ``attrs``
    clean for downstream serialization."""
    table = _make_table()
    headers = list(table.headers)
    # First, hide one column.
    DisplayPreferencesManager.set_display_preferences(
        table, {"a": True, "b": False, "c": True}, headers
    )
    assert "hidden_headers" in table.attrs
    # Now set everything visible: the attr must be removed.
    DisplayPreferencesManager.set_display_preferences(
        table, {"a": True, "b": True, "c": True}, headers
    )
    assert "hidden_headers" not in table.attrs


def test_set_display_preferences_no_op_when_attr_missing() -> None:
    """Setting all columns visible on a table that never had hidden columns
    must remain a no-op and not introduce a ``hidden_headers`` entry."""
    table = _make_table()
    headers = list(table.headers)
    DisplayPreferencesManager.set_display_preferences(
        table, {"a": True, "b": True, "c": True}, headers
    )
    assert "hidden_headers" not in table.attrs


def test_get_display_preferences_with_list_attr() -> None:
    """``hidden_headers`` may be persisted as a Python ``list``; the getter
    must accept that representation and return a per-column visibility map."""
    table = _make_table()
    table.attrs["hidden_headers"] = ["b"]  # stored as list
    prefs = DisplayPreferencesManager.get_display_preferences(
        table, list(table.headers)
    )
    assert prefs == {"a": True, "b": False, "c": True}


# ===========================================================================
# DataFrameManager.apply_visible_only_filter
# ===========================================================================


def test_apply_visible_only_filter_empty_returns_original() -> None:
    """When no requested header matches a column, the filter degrades to a
    no-op and returns the original DataFrame instance (identity check)."""
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    out = DataFrameManager.apply_visible_only_filter(df, visible_headers=["zzz"])
    # No headers match → the filter falls back to returning the original df.
    assert out is df


def test_apply_visible_only_filter_keeps_roi_index() -> None:
    """The ``roi_index`` column is structural and must always be preserved by
    the visibility filter, even when not requested explicitly."""
    df = pd.DataFrame({"roi_index": [0, 1], "a": [1, 2], "b": [3, 4]})
    out = DataFrameManager.apply_visible_only_filter(df, visible_headers=["a"])
    assert list(out.columns) == ["roi_index", "a"]


# ===========================================================================
# ResultHtmlGenerator
# ===========================================================================


def test_generate_html_default_format() -> None:
    """The default HTML rendering uses the underlined ``<u>`` style for the
    table title and shows the title verbatim."""
    table = _make_table(rows=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    html = ResultHtmlGenerator.generate_html(table)
    assert "<u>" in html
    assert table.title in html


def test_generate_html_callable_column_format() -> None:
    """Per-column ``column_formats`` may be a callable; the HTML generator
    must invoke it for each cell of the targeted column."""
    table = _make_table(rows=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    table.attrs["column_formats"] = {"a": lambda v: f"<<{v:.1f}>>"}
    html = ResultHtmlGenerator.generate_html(table)
    assert "&lt;&lt;1.0&gt;&gt;" in html or "<<1.0>>" in html


def test_generate_html_single_row_transposed() -> None:
    """With ``transpose_single_row=True``, a one-row table becomes a
    label/value vertical layout, which is the preferred presentation in
    DataLab for scalar result groups."""
    table = _make_table(rows=[[1.0, 2.0, 3.0]])
    html = ResultHtmlGenerator.generate_html(table, transpose_single_row=True)
    # Transposed view: column header is "Value" and row labels are original col names.
    assert "Value" in html
    for col in table.headers:
        assert col in html


def test_generate_html_single_row_no_transpose() -> None:
    """With ``transpose_single_row=False`` the standard horizontal layout is
    used, so the ``Value`` row label of the transposed view is absent."""
    table = _make_table(rows=[[1.0, 2.0, 3.0]])
    html = ResultHtmlGenerator.generate_html(table, transpose_single_row=False)
    assert "Value" not in html


def test_generate_html_with_roi_indices() -> None:
    """When ``roi_indices`` are set, the row headers must show the ROI labels
    (``ROI 0``, ``ROI 1`` ...)."""
    table = TableResult(
        title="t",
        headers=["a"],
        data=[[1.0], [2.0]],
        roi_indices=[0, 1],
    )
    html = ResultHtmlGenerator.generate_html(table, transpose_single_row=False)
    assert "ROI 0" in html
    assert "ROI 1" in html


def test_generate_html_with_no_roi_index_constant() -> None:
    """Rows tagged with the ``NO_ROI`` sentinel (-1) must produce empty row
    headers; in particular ``ROI -1`` must never be displayed."""
    # NO_ROI = -1 should give an empty header (no "ROI -1").
    table = TableResult(
        title="t",
        headers=["a"],
        data=[[1.0], [2.0]],
        roi_indices=[-1, -1],
    )
    html = ResultHtmlGenerator.generate_html(table, transpose_single_row=False)
    assert "ROI -1" not in html


def test_get_row_headers_no_roi_indices() -> None:
    """Without ROI indices, ``_get_row_headers`` returns one empty string per
    row (placeholders preserving alignment in the rendered HTML)."""
    table = _make_table(rows=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    headers = ResultHtmlGenerator._get_row_headers(  # pylint: disable=protected-access
        table, roi_indices=None, obj=None
    )
    assert headers == ["", ""]


def test_get_row_headers_out_of_bounds_keeps_default() -> None:
    """When the requested ROI index is beyond what the parent object exposes,
    the helper must keep the generic ``ROI N`` label rather than crashing."""

    # When roi_idx is beyond the obj's available ROIs, the default "ROI N" is kept.
    class DummyROI:
        """ROI container exposing zero entries."""

        single_rois = []  # No ROIs available

        def get_single_roi_title(self, _idx):  # pragma: no cover - not called
            """Sentinel implementation; should never be invoked."""
            return "should not be called"

    class DummyObj:
        """Minimal stand-in object exposing the ``roi`` attribute."""

        roi = DummyROI()

    headers = ResultHtmlGenerator._get_row_headers(  # pylint: disable=protected-access
        _make_table(),
        roi_indices=[5],
        obj=DummyObj(),
    )
    assert headers == ["ROI 5"]


def test_get_row_headers_uses_obj_roi_title() -> None:
    """When the parent object exposes a ROI with a custom title, that title
    overrides the generic ``ROI N`` placeholder in row headers."""

    class DummyROI:
        """ROI container with a single entry and a custom title accessor."""

        single_rois = [object()]

        def get_single_roi_title(self, idx):
            """Return a recognisable per-index title for assertions."""
            return f"Custom-{idx}"

    class DummyObj:
        """Minimal stand-in object exposing the ``roi`` attribute."""

        roi = DummyROI()

    headers = ResultHtmlGenerator._get_row_headers(  # pylint: disable=protected-access
        _make_table(),
        roi_indices=[0],
        obj=DummyObj(),
    )
    assert headers == ["Custom-0"]


if __name__ == "__main__":
    pytest.main([__file__])
