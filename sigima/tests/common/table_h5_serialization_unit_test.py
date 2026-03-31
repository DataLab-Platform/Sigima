# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
TableResult HDF5 serialization fixes — unit tests
--------------------------------------------------

Tests covering the changes introduced in ``fix/HDF5_format`` through actual
HDF5 file round-trips (write → read → verify).

1. **to_dict sanitization** — callable values in ``attrs`` are stripped so that
   HDF5 serialization does not fail.
2. **__check_value enum handling** — str-based enum values (``LabeledEnum``)
   survive HDF5 serialization as plain strings.
3. **column_formats API** — per-column format strings survive the HDF5
   round-trip.
"""

from __future__ import annotations

import enum
import os.path as osp

import pytest
from numpy import ma

import sigima.io
from sigima.objects import (
    TableResult,
    TableResultBuilder,
)
from sigima.tests.data import create_paracetamol_signal, create_test_signal_rois
from sigima.tests.helpers import WorkdirRestoringTempDir


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
class _SampleStrEnum(str, enum.Enum):
    """Str-based enum mimicking ``guidata.dataset.LabeledEnum``."""

    GAUSSIAN = "Gaussian"
    LORENTZIAN = "Lorentzian"


def _h5_roundtrip(sig):
    """Write a signal to HDF5, read it back, return the restored signal."""
    with WorkdirRestoringTempDir() as tmpdir:
        fpath = osp.join(tmpdir, "test.h5sig")
        sigima.io.write_signal(fpath, sig)
        return sigima.io.read_signal(fpath)


def _strip_none(d: dict) -> dict:
    """Recursively strip None values from a dict (HDF5Writer cannot store None)."""
    return {
        k: _strip_none(v) if isinstance(v, dict) else v
        for k, v in d.items()
        if v is not None
    }


def _roundtrip_table(table: TableResult) -> TableResult:
    """Perform a full HDF5 round-trip and return the restored TableResult."""
    sig = create_paracetamol_signal()
    # HDF5Writer.write_dict cannot serialise None values, so strip them
    # before storing — this mirrors what DataLab does at the adapter level.
    sig.metadata["table_test"] = _strip_none(table.to_dict())
    restored_sig = _h5_roundtrip(sig)
    return TableResult.from_dict(restored_sig.metadata["table_test"])


def _create_dummy_signal():
    """Create a simple SignalObj with one ROI."""
    sig = create_paracetamol_signal()
    roi = list(create_test_signal_rois(sig))[0]
    sig.roi = roi
    return sig


# ===================================================================
#  1. to_dict — callable sanitization (HDF5 round-trip)
# ===================================================================


class TestSanitizeCallablesH5:
    """Callable values in attrs must be stripped *before* HDF5 write so that
    the file can be created without error and read back correctly.
    """

    def test_callable_stripped_after_roundtrip(self) -> None:
        """A callable attr is absent in the HDF5-restored TableResult."""
        table = TableResult(
            title="T",
            headers=["col1"],
            data=[[1.0]],
            roi_indices=[-1],
            func_name="test",
            attrs={"method": "peak", "callback": print},
        )
        restored = _roundtrip_table(table)
        assert restored.attrs["method"] == "peak"
        assert "callback" not in restored.attrs

    def test_nested_callable_stripped_after_roundtrip(self) -> None:
        """A callable nested in a sub-dict of attrs is removed after HDF5 I/O."""
        table = TableResult(
            title="T",
            headers=["col1"],
            data=[[1.0]],
            roi_indices=[-1],
            func_name="test",
            attrs={"info": {"label": "ok", "fn": abs}},
        )
        restored = _roundtrip_table(table)
        assert restored.attrs["info"] == {"label": "ok"}

    def test_non_callable_values_survive_roundtrip(self) -> None:
        """str, int, float values in attrs survive the HDF5 round-trip."""
        table = TableResult(
            title="T",
            headers=["col1"],
            data=[[1.0]],
            roi_indices=[-1],
            func_name="test",
            attrs={"s": "text", "i": 42, "f": 3.14},
        )
        restored = _roundtrip_table(table)
        assert restored.attrs["s"] == "text"
        assert restored.attrs["i"] == 42
        assert restored.attrs["f"] == pytest.approx(3.14)

    def test_all_callable_attrs_gives_empty_after_roundtrip(self) -> None:
        """If attrs contained only callables, restored attrs is empty."""
        table = TableResult(
            title="T",
            headers=["col1"],
            data=[[1.0]],
            roi_indices=[-1],
            func_name="test",
            attrs={"fn1": abs, "fn2": lambda: None},
        )
        restored = _roundtrip_table(table)
        # Same as "=={}""
        assert not restored.attrs

    def test_title_and_data_survive_roundtrip(self) -> None:
        """Title, headers, and numeric data survive even with tainted attrs."""
        table = TableResult(
            title="Stats",
            headers=["min", "max"],
            data=[[1.5, 9.8], [2.3, 7.6]],
            roi_indices=[-1, 0],
            func_name="test",
            attrs={"clean": "yes", "dirty": lambda: 0},
        )
        restored = _roundtrip_table(table)
        assert restored.title == "Stats"
        assert restored.headers == ["min", "max"]
        assert len(restored.data) == 2
        assert restored.data[0][0] == pytest.approx(1.5)
        assert restored.data[1][1] == pytest.approx(7.6)


# ===================================================================
#  2. __check_value — enum handling (HDF5 round-trip)
# ===================================================================


class TestEnumHandlingH5:
    """Str-based enum values (e.g. ``SignalShape``) must be converted to plain
    ``str`` by ``__check_value`` so they can be serialised to HDF5.
    """

    def test_str_enum_survives_roundtrip(self) -> None:
        """A str-based enum value ends up as a plain str after HDF5 I/O."""
        sig = _create_dummy_signal()
        builder = TableResultBuilder("Shapes")
        builder.add(lambda _data: _SampleStrEnum.GAUSSIAN, "shape")
        builder.add(ma.mean, "mean")
        table = builder.compute(sig)

        restored = _roundtrip_table(table)
        for row in restored.data:
            assert isinstance(row[0], str)
            assert isinstance(row[1], float)

    def test_str_enum_value_preserved_after_roundtrip(self) -> None:
        """The original str value of the enum is preserved after HDF5 I/O."""
        sig = _create_dummy_signal()
        builder = TableResultBuilder("Shapes")
        builder.add(lambda _data: _SampleStrEnum.LORENTZIAN, "shape")
        table = builder.compute(sig)

        restored = _roundtrip_table(table)
        assert restored.data[0][0] == str(_SampleStrEnum.LORENTZIAN)

    def test_mixed_str_and_float_survive_roundtrip(self) -> None:
        """A table mixing plain str and float columns round-trips correctly."""
        sig = _create_dummy_signal()
        builder = TableResultBuilder("Mixed")
        builder.add(lambda _data: "label", "tag")
        builder.add(ma.mean, "mean")
        builder.add(ma.min, "min")
        table = builder.compute(sig)

        restored = _roundtrip_table(table)
        assert restored.data[0][0] == "label"
        assert isinstance(restored.data[0][1], float)
        assert isinstance(restored.data[0][2], float)


# ===================================================================
#  3. column_formats API (HDF5 round-trip)
# ===================================================================


class TestColumnFormatsH5:
    """``column_formats`` stored in ``attrs`` must survive the HDF5 round-trip."""

    def test_formats_survive_roundtrip(self) -> None:
        """Per-column formats set via the API are retrieved after HDF5 I/O."""
        table = TableResult(
            title="T",
            headers=["x", "y"],
            data=[[1.0, 2.0]],
            roi_indices=[-1],
            func_name="test",
        )
        table.set_column_formats({"x": ".2e", "y": ".3g"})

        restored = _roundtrip_table(table)
        assert restored.get_column_formats() == {"x": ".2e", "y": ".3g"}

    def test_empty_formats_survive_roundtrip(self) -> None:
        """A table with no column_formats keeps an empty dict after HDF5 I/O."""
        table = TableResult(
            title="T",
            headers=["a"],
            data=[[1.0]],
            roi_indices=[-1],
            func_name="test",
        )

        restored = _roundtrip_table(table)
        # Same as "=={}""
        assert not restored.get_column_formats()

    def test_builder_formats_survive_roundtrip(self) -> None:
        """Formats set on the builder are present in the result after HDF5 I/O."""
        sig = _create_dummy_signal()
        builder = TableResultBuilder("Stats")
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")
        builder.set_column_formats({"min": ".2e", "max": ".3g"})
        table = builder.compute(sig)

        restored = _roundtrip_table(table)
        assert restored.get_column_formats() == {"min": ".2e", "max": ".3g"}

    def test_overwritten_formats_survive_roundtrip(self) -> None:
        """Only the last set of formats is present after HDF5 I/O."""
        table = TableResult(
            title="T",
            headers=["a", "b"],
            data=[[1.0, 2.0]],
            roi_indices=[-1],
            func_name="test",
        )
        table.set_column_formats({"a": ".2f", "b": ".3g"})
        table.set_column_formats({"a": ".5e"})

        restored = _roundtrip_table(table)
        assert restored.get_column_formats() == {"a": ".5e"}
