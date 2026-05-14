# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Markers tables (XY/X/Y) unit tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest

import sigima.params
import sigima.proc.signal
from sigima.objects import TableKind, TableResult
from sigima.tests import guiutils
from sigima.tests.data import create_paracetamol_signal
from sigima.tests.env import execenv


@pytest.mark.gui
def test_signal_extract_peak_positions_interactive() -> None:
    """Interactive test: visualize peak positions on the paracetamol spectrum.

    Displays the paracetamol spectrum together with a cross marker placed at
    each ``(x, y)`` couple returned by :func:`extract_peak_positions`. Useful
    to visually validate the rendering used by DataLab for
    :attr:`~sigima.objects.TableKind.XY_MARKERS` tables.
    """
    # pylint: disable=import-outside-toplevel
    from sigima import viz

    src = create_paracetamol_signal()
    param = sigima.params.PeakDetectionParam.create(threshold=20, min_dist=5)
    result = sigima.proc.signal.extract_peak_positions(src, param)
    execenv.print(
        f"Detected {len(result.data)} peak(s) on '{src.title}' "
        f"(threshold={param.threshold}%, min_dist={param.min_dist}pts)"
    )

    items = [viz.create_curve(src.x, src.y, title=src.title)]
    x_header, y_header = result.headers[0], result.headers[1]
    for x_val, y_val in zip(result.col(x_header), result.col(y_header)):
        items.append(viz.create_marker(x_val, y_val))

    with guiutils.lazy_qt_app_context(force=True):
        viz.view_curve_items(items, title=f"Peak positions [{src.title}]")


@pytest.mark.validation
def test_signal_extract_peak_positions() -> None:
    """Validate `extract_peak_positions` on the paracetamol spectrum.

    The function must return an XY-markers table whose ``(x, y)`` couples
    match the samples returned by `peak_detection` (same algorithm).
    """
    src = create_paracetamol_signal()

    param = sigima.params.PeakDetectionParam.create(threshold=20, min_dist=5)
    result = sigima.proc.signal.extract_peak_positions(src, param)

    # Result type and metadata
    assert isinstance(result, TableResult)
    assert result.kind == TableKind.XY_MARKERS
    assert result.is_xy_markers()
    assert not result.is_x_markers()
    assert not result.is_y_markers()
    # Headers reflect the source signal axis labels (with units when set);
    # paracetamol.txt provides ``2 theta (°)`` / ``Intensity``.
    x_header, y_header = result.headers[0], result.headers[1]
    assert x_header == "2 theta (°)"
    assert y_header == "Intensity"
    assert len(result.data) > 0, "At least one peak should be detected"
    # Markers tables must request a row-index column when displayed
    assert result.attrs.get("show_row_index") is True

    # Cross-check with the existing peak_detection function (same algorithm)
    with pytest.warns(DeprecationWarning, match="peak_detection is deprecated"):
        expected = sigima.proc.signal.peak_detection(src, param)
    xs = np.array(result.col(x_header))
    ys = np.array(result.col(y_header))
    assert xs.size == expected.x.size
    np.testing.assert_allclose(np.sort(xs), np.sort(expected.x))
    np.testing.assert_allclose(ys[np.argsort(xs)], expected.y[np.argsort(expected.x)])

    # Each (x, y) couple must lie on the source signal samples
    for x_val, y_val in zip(xs, ys):
        idx = int(np.argmin(np.abs(src.x - x_val)))
        assert src.x[idx] == pytest.approx(x_val)
        assert src.y[idx] == pytest.approx(y_val)


def test_signal_extract_peak_positions_no_peak() -> None:
    """Empty result when no peak passes the threshold."""
    src = create_paracetamol_signal()
    # Threshold so high that no peak is detected
    param = sigima.params.PeakDetectionParam.create(threshold=100, min_dist=1)
    result = sigima.proc.signal.extract_peak_positions(src, param)
    assert result.kind == TableKind.XY_MARKERS
    assert result.data == []
    assert result.roi_indices is None


def test_table_kind_xy_markers_construction() -> None:
    """XY_MARKERS TableResult can be built from arbitrary (x, y) couples."""
    xs = [1.0, 2.5, 4.2, 7.8]
    ys = [10.0, -3.5, 0.0, 42.0]
    result = TableResult.from_rows(
        title="Arbitrary XY markers",
        headers=["x", "y"],
        rows=[[x, y] for x, y in zip(xs, ys)],
        kind=TableKind.XY_MARKERS,
    )
    assert result.is_xy_markers()
    assert not result.is_x_markers()
    assert not result.is_y_markers()
    assert result.col("x") == xs
    assert result.col("y") == ys


def test_table_kind_x_markers_construction() -> None:
    """X_MARKERS TableResult holds arbitrary X positions only."""
    xs = [1.5, 3.14, 5.0, 9.81]
    result = TableResult.from_rows(
        title="Arbitrary X markers",
        headers=["x"],
        rows=[[x] for x in xs],
        kind=TableKind.X_MARKERS,
    )
    assert result.is_x_markers()
    assert not result.is_xy_markers()
    assert not result.is_y_markers()
    assert list(result.headers) == ["x"]
    assert result.col("x") == xs


def test_table_kind_y_markers_construction() -> None:
    """Y_MARKERS TableResult holds arbitrary Y positions only."""
    ys = [-1.0, 0.0, 2.71, 100.0]
    result = TableResult.from_rows(
        title="Arbitrary Y markers",
        headers=["y"],
        rows=[[y] for y in ys],
        kind=TableKind.Y_MARKERS,
    )
    assert result.is_y_markers()
    assert not result.is_xy_markers()
    assert not result.is_x_markers()
    assert list(result.headers) == ["y"]
    assert result.col("y") == ys


def test_show_row_index_html_rendering() -> None:
    """When ``show_row_index`` is set, HTML rows are labelled ``#0, #1, ...``."""
    xs = [10.0, 20.0, 30.0]
    result = TableResult.from_rows(
        title="X markers with index",
        headers=["x"],
        rows=[[x] for x in xs],
        kind=TableKind.X_MARKERS,
        attrs={"show_row_index": True},
    )
    html = result.to_html(transpose_single_row=False)
    for i in range(len(xs)):
        assert f"#{i}" in html, f"Expected row index '#{i}' in HTML output"


if __name__ == "__main__":
    test_signal_extract_peak_positions_interactive()
    test_signal_extract_peak_positions()
    test_signal_extract_peak_positions_no_peak()
    test_table_kind_xy_markers_construction()
    test_table_kind_x_markers_construction()
    test_table_kind_y_markers_construction()
    test_show_row_index_html_rendering()
