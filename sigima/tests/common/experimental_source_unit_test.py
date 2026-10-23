# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the experimental data origin stored in object metadata
"""

from __future__ import annotations

import json
import os.path as osp

import numpy as np
import pytest

import sigima.io
from sigima.objects import (
    EXPERIMENTAL_SOURCE_KEY,
    create_image,
    create_signal,
    get_experimental_source,
    make_experimental_source,
    set_experimental_source,
)
from sigima.tests.helpers import WorkdirRestoringTempDir


def _spec_source() -> dict:
    """Return a typical source of a signal imported from a SPEC scan."""
    return make_experimental_source(
        "spec",
        "C:/data/sample.spec",
        entry="12.1",
        selection={"x": "Energy", "y": "I0"},
        timestamps={"source": "2026-10-20T08:15:00+02:00"},
        conversions=["int32 -> float64"],
        details={"command": "ascan th 0 1 10 0.1", "motors": {"th": 0.5}},
    )


def test_make_experimental_source() -> None:
    """Optional fields are kept only when provided."""
    source = make_experimental_source("tango", "tango://host:10000/lab/cam/1")
    assert source == {
        "version": 1,
        "kind": "tango",
        "location": "tango://host:10000/lab/cam/1",
    }
    assert _spec_source()["selection"] == {"x": "Energy", "y": "I0"}


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"kind": "", "location": "f"}, ValueError),
        ({"kind": "spec", "location": " "}, ValueError),
        ({"kind": "spec", "location": "f", "entry": ""}, ValueError),
        ({"kind": "spec", "location": "f", "selection": ["x"]}, TypeError),
        ({"kind": "spec", "location": "f", "conversions": "a"}, TypeError),
        ({"kind": "spec", "location": "f", "details": {"v": float("nan")}}, ValueError),
        ({"kind": "spec", "location": "f", "details": {"v": None}}, ValueError),
        ({"kind": "spec", "location": "f", "selection": {"a": [None]}}, ValueError),
        ({"kind": "spec", "location": "f", "details": {1: "a"}}, TypeError),
        ({"kind": "spec", "location": "f", "details": {"": 1}}, ValueError),
        ({"kind": "spec", "location": "f", "details": {"a/b": 1}}, ValueError),
        ({"kind": "spec", "location": "f", "details": {"__seq": 1}}, ValueError),
        ({"kind": "spec", "location": "f", "details": {"v": 2**63}}, ValueError),
        ({"kind": "spec", "location": "f", "details": {"v": 1j}}, TypeError),
        ({"kind": "spec", "location": "f", "details": {"v": b"a"}}, TypeError),
    ],
    ids=[
        "kind",
        "location",
        "entry",
        "selection",
        "conversions",
        "nan",
        "none",
        "none-in-list",
        "int-key",
        "empty-key",
        "slash-key",
        "reserved-key",
        "int64-overflow",
        "complex",
        "bytes",
    ],
)
def test_invalid_experimental_source(kwargs: dict, error: type) -> None:
    """Invalid fields and values that HDF5 files do not restore are rejected."""
    with pytest.raises(error):
        make_experimental_source(**kwargs)


def test_numpy_values_are_converted() -> None:
    """NumPy scalars and tuples are stored as Python values and lists."""
    source = make_experimental_source(
        "spec",
        "f",
        selection={"indices": (np.int32(1), np.int64(2))},
        details={
            "gain": np.float32(0.5),
            "valid": np.bool_(True),
            "name": np.str_("a"),
        },
    )
    assert source["selection"] == {"indices": [1, 2]}
    assert source["details"] == {"gain": 0.5, "valid": True, "name": "a"}
    for value in (*source["selection"]["indices"], *source["details"].values()):
        assert not isinstance(value, np.generic)
    json.dumps(source)


def test_set_rejects_unknown_version_and_fields() -> None:
    """Only the current layout can be stored."""
    obj = create_signal("s", np.arange(3.0), np.arange(3.0))
    with pytest.raises(ValueError):
        set_experimental_source(obj, {**_spec_source(), "version": 2})
    with pytest.raises(ValueError):
        set_experimental_source(obj, {**_spec_source(), "extra": 1})
    assert get_experimental_source(obj) is None


def test_source_is_copied() -> None:
    """Stored and returned sources are independent of the caller's dictionary."""
    source = _spec_source()
    obj = create_signal("s", np.arange(3.0), np.arange(3.0))
    set_experimental_source(obj, source)
    source["details"]["motors"]["th"] = 9.0
    returned = get_experimental_source(obj)
    assert returned["details"]["motors"]["th"] == 0.5
    returned["entry"] = "1.1"
    assert obj.metadata[EXPERIMENTAL_SOURCE_KEY]["entry"] == "12.1"
    assert get_experimental_source(obj.copy()) == _spec_source()


def _assert_json_types(value) -> None:
    """Assert that value is made of Python JSON types only."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str
            _assert_json_types(item)
    elif isinstance(value, list):
        for item in value:
            _assert_json_types(item)
    else:
        assert type(value) in (str, int, float, bool), repr(value)


def _round_trip(signal, image) -> tuple:
    """Write and read objects with Sigima's native signal and image formats."""
    with WorkdirRestoringTempDir() as tmpdir:
        sigima.io.write_signal(osp.join(tmpdir, "s.h5sig"), signal)
        sigima.io.write_image(osp.join(tmpdir, "i.h5ima"), image)
        return (
            sigima.io.read_signal(osp.join(tmpdir, "s.h5sig")),
            sigima.io.read_image(osp.join(tmpdir, "i.h5ima")),
        )


def test_source_survives_file_round_trip() -> None:
    """The source is restored by Sigima's native signal and image formats, with
    its JSON types, so that it can be stored again."""
    signal = create_signal("s", np.arange(3.0), np.arange(3.0))
    image = create_image("i", np.zeros((2, 3)))
    for obj in (signal, image):
        set_experimental_source(obj, _spec_source())
    for obj in _round_trip(signal, image):
        source = get_experimental_source(obj)
        assert source == _spec_source()
        assert type(source["version"]) is int
        _assert_json_types(source)
        _assert_json_types(obj.metadata[EXPERIMENTAL_SOURCE_KEY])
        json.dumps(source)
        set_experimental_source(obj, source)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        [[0, 1], [2, 3]],
        [{"a": 1}, {"b": "x"}],
        {"a": {"b": [1.5, -2]}},
        [True, False],
        "µm é",
        "",
        ["a", 1, 2.5, False],
        ".",
        -(2**63),
        2**63 - 1,
    ],
    ids=[
        "empty-list",
        "empty-dict",
        "nested-list",
        "list-of-dicts",
        "nested-dict",
        "bools",
        "unicode",
        "empty-string",
        "mixed-list",
        "dot",
        "int64-min",
        "int64-max",
    ],
)
def test_json_values_survive_file_round_trip(value) -> None:
    """Every accepted JSON value is restored with its type."""
    source = make_experimental_source(
        "hdf5", "f.h5", selection={"axes": [".", "x"]}, details={"v": value}
    )
    signal = create_signal("s", np.arange(3.0), np.arange(3.0))
    image = create_image("i", np.zeros((2, 3)))
    for obj in (signal, image):
        set_experimental_source(obj, source)
    for obj in _round_trip(signal, image):
        restored = get_experimental_source(obj)
        assert restored == source
        _assert_json_types(restored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
