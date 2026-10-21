# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the experimental data origin stored in object metadata
"""

from __future__ import annotations

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
        ({"kind": "spec", "location": "f", "details": {"v": np.int64(1)}}, ValueError),
        ({"kind": "spec", "location": "f", "details": {"v": float("nan")}}, ValueError),
    ],
    ids=["kind", "location", "entry", "selection", "conversions", "numpy", "nan"],
)
def test_invalid_experimental_source(kwargs: dict, error: type) -> None:
    """Invalid fields and non-JSON values are rejected."""
    with pytest.raises(error):
        make_experimental_source(**kwargs)


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


def test_source_survives_file_round_trip() -> None:
    """The source is restored by Sigima's native signal and image formats."""
    signal = create_signal("s", np.arange(3.0), np.arange(3.0))
    image = create_image("i", np.zeros((2, 3)))
    for obj in (signal, image):
        set_experimental_source(obj, _spec_source())
    with WorkdirRestoringTempDir() as tmpdir:
        sigima.io.write_signal(osp.join(tmpdir, "s.h5sig"), signal)
        sigima.io.write_image(osp.join(tmpdir, "i.h5ima"), image)
        restored = (
            sigima.io.read_signal(osp.join(tmpdir, "s.h5sig")),
            sigima.io.read_image(osp.join(tmpdir, "i.h5ima")),
        )
    for obj in restored:
        assert get_experimental_source(obj) == _spec_source()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
