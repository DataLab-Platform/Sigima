# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :class:`sigima.objects.scalar.table.TableResult`,
:class:`TableResultBuilder` and the ``concat_tables`` / ``filter_table_by_roi``
helpers.

Covers validation, naming, equality, helpers (``is_*``), HTML rendering,
column formats, ``as_dict`` / ``value`` lookup, builder validation paths
(``set_global_function`` / ``add``), ``compute_with_column_funcs`` and
table concatenation rules.
"""

# pylint: disable=invalid-name
# pylint: disable=protected-access

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sigima.objects import SignalObj
from sigima.objects.scalar.table import (
    NO_ROI,
    TableKind,
    TableResult,
    TableResultBuilder,
    concat_tables,
    filter_table_by_roi,
)

# ===========================================================================
# Naming and kind helpers
# ===========================================================================


def test_table_result_kind_string_unknown_kept() -> None:
    """Free-form ``kind`` strings (not in ``TableKind``) are kept as-is
    and used as the table's ``name`` for display purposes."""
    res = TableResult(title="t", kind="custom-x", headers=["a"], data=[[1.0]])
    assert res.kind == "custom-x"
    assert res.name == "custom-x"


def test_table_result_name_with_string_kind() -> None:
    """With a string ``kind`` the table's ``name`` mirrors the string."""
    res = TableResult(title="t", kind="custom_kind", headers=["a"], data=[[1.0]])
    assert res.name == "custom_kind"


def test_table_result_name_with_enum_kind() -> None:
    """With a ``TableKind`` enum, the ``name`` resolves to the enum's
    string ``.value``."""
    res = TableResult(title="t", kind=TableKind.STATISTICS, headers=["a"], data=[[1.0]])
    assert res.name == TableKind.STATISTICS.value


def test_table_result_is_helpers() -> None:
    """``is_statistics`` / ``is_pulse_features`` / ``is_custom`` are
    mutually exclusive predicates that match the configured ``kind``."""
    stat = TableResult(
        title="t", kind=TableKind.STATISTICS, headers=["a"], data=[[1.0]]
    )
    assert stat.is_statistics()
    assert not stat.is_pulse_features()
    assert not stat.is_custom()
    pf = TableResult(
        title="t", kind=TableKind.PULSE_FEATURES, headers=["a"], data=[[1.0]]
    )
    assert pf.is_pulse_features()
    cu = TableResult(title="t", kind=TableKind.CUSTOM, headers=["a"], data=[[1.0]])
    assert cu.is_custom()


# ===========================================================================
# String / HTML / formats
# ===========================================================================


def test_table_result_str_and_repr_html() -> None:
    """``str(table)`` includes the title and ``_repr_html_`` produces a
    real HTML ``<table>`` element (Jupyter rich display)."""
    res = TableResult(
        title="My table",
        headers=["a", "b"],
        data=[[1.0, 2.0], [3.0, 4.0]],
        roi_indices=[NO_ROI, 0],
    )
    text = str(res)
    assert "My table" in text
    html = res._repr_html_()
    assert "<table" in html


def test_table_result_str_representation_default() -> None:
    """The default ``str()`` representation includes the class name so
    debug logs are unambiguous."""
    res = TableResult(title="t", headers=["a"], data=[[1.0]])
    assert "TableResult" in str(res)


def test_table_result_set_and_clear_column_formats() -> None:
    """Per-column format strings can be set, retrieved and cleared
    independently of the data."""
    res = TableResult(title="t", headers=["a"], data=[[1.0]])
    res.set_column_formats({"a": ".3f"})
    assert res.get_column_formats() == {"a": ".3f"}
    res.set_column_formats({})
    assert not res.get_column_formats()


# ===========================================================================
# value() and as_dict() lookup paths
# ===========================================================================


def test_table_result_as_dict_and_value_errors() -> None:
    """On a multi-row table without ROI indices, both ``as_dict()`` and
    ``value()`` raise ``ValueError`` (ambiguous lookup); on a multi-row
    table with ROI indices, an unknown ROI raises ``KeyError``."""
    res = TableResult(title="t", headers=["a"], data=[[1.0], [2.0]])
    with pytest.raises(ValueError):
        res.as_dict()
    with pytest.raises(ValueError):
        res.value("a")
    res2 = TableResult(
        title="t", headers=["a"], data=[[1.0], [2.0]], roi_indices=[0, 1]
    )
    with pytest.raises(KeyError):
        res2.as_dict(roi=99)
    with pytest.raises(KeyError):
        res2.value("a", roi=99)


def test_table_result_value_ambiguous_no_roi_indices() -> None:
    """``value()`` cannot pick a row when there are no ROI indices and
    several rows match: it raises an ``Ambiguous`` error."""
    res = TableResult(title="t", headers=["a"], data=[[1.0], [2.0]])
    with pytest.raises(ValueError, match="Ambiguous"):
        res.value("a", roi=None)


def test_table_result_value_ambiguous_with_roi() -> None:
    """Even with explicit ROI indices, several rows for the same ROI is
    still ambiguous when ``value()`` is asked for that ROI."""
    res = TableResult(title="t", headers=["a"], data=[[1.0], [2.0]], roi_indices=[0, 0])
    with pytest.raises(ValueError, match="Ambiguous"):
        res.value("a", roi=0)


def test_table_result_value_unique_match() -> None:
    """When exactly one row matches the requested ROI, ``value()``
    returns that cell's value."""
    res = TableResult(
        title="t", headers=["a"], data=[[1.0], [2.0]], roi_indices=[NO_ROI, 0]
    )
    assert res.value("a", roi=0) == 2.0


def test_table_result_as_dict_returns_dict() -> None:
    """On a single-row table, ``as_dict()`` returns a {header: value}
    mapping covering every column."""
    res = TableResult(
        title="t", headers=["a", "b"], data=[[1.0, 2.0]], roi_indices=[NO_ROI]
    )
    d = res.as_dict(roi=None)
    assert d == {"a": 1.0, "b": 2.0}


def test_table_result_as_dict_ambiguous_with_roi() -> None:
    """``as_dict(roi=...)`` raises the same ambiguity error as ``value``
    when several rows share the same ROI index."""
    res = TableResult(title="t", headers=["a"], data=[[1.0], [2.0]], roi_indices=[0, 0])
    with pytest.raises(ValueError, match="Ambiguous"):
        res.as_dict(roi=0)


# ===========================================================================
# TableResultBuilder validation
# ===========================================================================


def test_table_builder_set_global_function_errors() -> None:
    """``set_global_function`` rejects callables that take no arguments
    (the global function is expected to receive the source data)."""
    builder = TableResultBuilder("t")

    def no_args() -> None:
        """No-arg helper used to trigger the validation error."""
        return None

    with pytest.raises(ValueError):
        builder.set_global_function(no_args)


def test_table_builder_global_func_wrong_annotation() -> None:
    """``set_global_function`` rejects callables whose argument is not
    annotated as ``tuple[np.ndarray, np.ndarray]`` (or compatible)."""
    builder = TableResultBuilder("t")

    def bad(_x: "int") -> None:
        """Helper with a wrong argument annotation (``int``)."""
        return None

    with pytest.raises(ValueError):
        builder.set_global_function(bad)


def test_table_builder_global_func_non_dataclass_return() -> None:
    """``set_global_function`` rejects callables whose return annotation
    is not a ``@dataclass`` (the builder needs the dataclass fields to
    derive column names)."""
    builder = TableResultBuilder("t")

    def bad(_x):
        """Helper whose return annotation will be set to ``int``."""
        return None

    bad.__annotations__ = {"return": int}
    with pytest.raises(ValueError):
        builder.set_global_function(bad)


def test_table_builder_global_func_with_dataclass_annotation() -> None:
    """``set_global_function`` accepts a callable returning a dataclass; the
    builder must store the function as the global computation function."""

    @dataclasses.dataclass
    class Result:
        """Single-field dataclass standing in for a real builder result."""

        x: float

    def good(_xy):
        """Helper computation function returning a ``Result`` dataclass."""
        return Result(x=1.0)

    good.__annotations__ = {
        "_xy": "tuple[np.ndarray, np.ndarray]",
        "return": Result,
    }
    builder = TableResultBuilder("t")
    builder.set_global_function(good)
    assert builder.global_func is good


def test_table_builder_add_column_func_errors() -> None:
    """``add`` rejects column callables that take no arguments."""
    builder = TableResultBuilder("t")

    def bad_func() -> float:
        """No-arg helper used to trigger the validation error."""
        return 0.0

    with pytest.raises(ValueError):
        builder.add(bad_func, "x")


def test_table_builder_add_func_wrong_annotation() -> None:
    """``add`` rejects callables whose argument annotation is not the
    expected (x, y) NumPy tuple."""
    builder = TableResultBuilder("t")

    def bad(_x: "int"):
        """Helper with a wrong argument annotation (``int``)."""
        return 0.0

    with pytest.raises(ValueError):
        builder.add(bad, "col")


def test_table_builder_add_func_wrong_return() -> None:
    """``add`` rejects callables that return non-numeric values (here
    annotated as ``str``)."""
    builder = TableResultBuilder("t")

    def bad(_x) -> "str":
        """Helper whose return annotation is ``str`` (non-numeric)."""
        return ""

    with pytest.raises(ValueError):
        builder.add(bad, "col")


def test_table_builder_check_value_non_numeric_raises() -> None:
    """The internal ``__check_value`` helper rejects non-numeric column
    values at runtime, mirroring the static annotation checks."""
    check = TableResultBuilder._TableResultBuilder__check_value
    with pytest.raises(ValueError):
        check(object())


def test_table_builder_compute_with_column_funcs() -> None:
    """End-to-end check: a builder configured with two column callables
    produces a table whose headers contain both column names and at
    least one row of computed data."""
    sig = SignalObj()
    sig.set_xydata(np.linspace(0.0, 1.0, 16), np.linspace(2.0, 3.0, 16))

    builder = TableResultBuilder("stats", kind=TableKind.STATISTICS)
    builder.add(lambda data: float(np.mean(data[1])), "mean_y")
    builder.add(lambda data: float(np.max(data[1])), "max_y")
    res = builder.compute(sig)
    assert "mean_y" in res.headers
    assert "max_y" in res.headers
    assert len(res.data) >= 1


# ===========================================================================
# concat_tables / filter_table_by_roi
# ===========================================================================


def test_concat_tables_empty_and_mismatched() -> None:
    """``concat_tables`` returns an empty table for an empty input list,
    and rejects tables with mismatched headers."""
    empty = concat_tables("merged", [])
    assert empty.headers == []
    assert not empty.data
    a = TableResult(title="a", headers=["x"], data=[[1.0]])
    b = TableResult(title="b", headers=["y"], data=[[2.0]])
    with pytest.raises(ValueError):
        concat_tables("merged", [a, b])


def test_concat_tables_kind_demotion() -> None:
    """When concatenated tables have different ``kind`` values, the
    result is demoted to ``CUSTOM`` to avoid a misleading kind tag."""
    a = TableResult(title="a", headers=["x"], data=[[1.0]], kind=TableKind.STATISTICS)
    b = TableResult(
        title="b", headers=["x"], data=[[2.0]], kind=TableKind.PULSE_FEATURES
    )
    out = concat_tables("merged", [a, b])
    assert out.kind == TableKind.CUSTOM
    assert len(out.data) == 2


def test_filter_table_by_roi_round_trip() -> None:
    """``filter_table_by_roi`` keeps only the rows whose ``roi_indices``
    match the requested ROI (here exactly one)."""
    res = TableResult(
        title="t",
        headers=["a"],
        data=[[1.0], [2.0], [3.0]],
        roi_indices=[NO_ROI, 0, 1],
    )
    out = filter_table_by_roi(res, roi=0)
    assert len(out.data) == 1


def test_filter_table_by_roi_no_indices() -> None:
    """Without ROI indices and ``roi=None``, ``filter_table_by_roi`` is
    a pass-through (the input table is returned unchanged)."""
    res = TableResult(title="t", headers=["a"], data=[[1.0]])
    out = filter_table_by_roi(res, roi=None)
    assert len(out.data) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
