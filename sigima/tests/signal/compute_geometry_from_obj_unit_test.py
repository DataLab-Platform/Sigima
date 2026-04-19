# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for :func:`sigima.proc.signal.base.compute_geometry_from_obj`.
"""

# `simple_signal` is a pytest fixture passed by name to test functions; this is
# the idiomatic pytest pattern but pylint flags it as ``redefined-outer-name``.
# pylint: disable=redefined-outer-name

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import KindShape, create_signal
from sigima.proc.signal.base import compute_geometry_from_obj


@pytest.fixture()
def simple_signal():
    """Provide a 100-point sinusoid as a generic input signal for the
    geometry-from-callable tests below."""
    x = np.linspace(0.0, 10.0, 100)
    y = np.sin(x)
    return create_signal("test", x, y)


def test_returns_none_when_func_returns_none(simple_signal) -> None:
    """When the user-supplied callable returns ``None``, the wrapper must
    propagate ``None`` rather than constructing an empty result object."""
    out = compute_geometry_from_obj("title", "point", simple_signal, lambda x, y: None)
    assert out is None


def test_skips_empty_results(simple_signal) -> None:
    """An empty NumPy array from the callable yields ``None`` (no shape
    can be built from zero coordinates)."""
    out = compute_geometry_from_obj(
        "title", "point", simple_signal, lambda x, y: np.array([])
    )
    assert out is None


def test_skips_malformed_1d_odd_length(simple_signal) -> None:
    # Length 3 is odd and shape != "segment" → should be skipped → None.
    """A 1D result of odd length cannot be reshaped to ``(N, 2)`` point
    coordinates and must be silently skipped (returns ``None``)."""
    out = compute_geometry_from_obj(
        "title", "point", simple_signal, lambda x, y: np.array([1.0, 2.0, 3.0])
    )
    assert out is None


def test_skips_malformed_higher_dim(simple_signal) -> None:
    """3D arrays returned by the callable cannot represent geometry and
    are skipped (returns ``None``)."""
    out = compute_geometry_from_obj(
        "title",
        "point",
        simple_signal,
        lambda x, y: np.zeros((2, 2, 2)),
    )
    assert out is None


def test_segment_shape_with_4_coords_string(simple_signal) -> None:
    """With ``shape="segment"`` and a 4-value 1D array, the wrapper builds
    a ``SEGMENT`` result with coords of shape ``(1, 4)``."""
    out = compute_geometry_from_obj(
        "segment_title",
        "segment",
        simple_signal,
        lambda x, y: np.array([0.0, 1.0, 2.0, 3.0]),
    )
    assert out is not None
    assert out.kind == KindShape.SEGMENT
    assert out.coords.shape == (1, 4)


def test_segment_shape_with_kindshape_enum(simple_signal) -> None:
    """Passing the ``KindShape.SEGMENT`` enum value (instead of the
    ``"segment"`` string) yields the same result."""
    out = compute_geometry_from_obj(
        "segment_title",
        KindShape.SEGMENT,
        simple_signal,
        lambda x, y: np.array([0.0, 1.0, 2.0, 3.0]),
    )
    assert out is not None
    assert out.kind == KindShape.SEGMENT


def test_marker_shape(simple_signal) -> None:
    """With ``shape="marker"`` and a 2-value 1D array, the wrapper builds
    a ``MARKER`` result."""
    out = compute_geometry_from_obj(
        "marker_title",
        "marker",
        simple_signal,
        lambda x, y: np.array([1.0, 2.0]),
    )
    assert out is not None
    assert out.kind == KindShape.MARKER


def test_unknown_shape_falls_back_to_point(simple_signal) -> None:
    """An unknown ``shape`` string is not an error: it falls back to
    ``KindShape.POINT`` so the user always gets some result."""
    out = compute_geometry_from_obj(
        "title",
        "totally_unknown",
        simple_signal,
        lambda x, y: np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert out is not None
    assert out.kind == KindShape.POINT


def test_point_shape_from_string(simple_signal) -> None:
    """With ``shape="point"`` and a 2D ``(N, 2)`` array, the wrapper builds
    a ``POINT`` result preserving the ``(N, 2)`` coords shape."""
    out = compute_geometry_from_obj(
        "title",
        "point",
        simple_signal,
        lambda x, y: np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert out is not None
    assert out.kind == KindShape.POINT
    assert out.coords.shape == (2, 2)


def test_func_with_extra_args(simple_signal) -> None:
    """Extra positional arguments after the callable are forwarded to it
    (here, a ``scale`` factor used inside the callable)."""

    def func(x, y, scale):  # pylint: disable=unused-argument
        """Sentinel callable that returns ``[[scale, 2.0]]`` so the test
        can assert that ``scale`` was forwarded correctly."""
        return np.array([[scale * 1.0, 2.0]])

    out = compute_geometry_from_obj("title", "point", simple_signal, func, 5.0)
    assert out is not None
    assert out.coords[0, 0] == 5.0


def test_skips_2d_with_only_one_column(simple_signal) -> None:
    """A 2D array with only one column does not represent ``(x, y)``
    point coordinates and must be skipped (returns ``None``)."""
    out = compute_geometry_from_obj(
        "title", "point", simple_signal, lambda x, y: np.array([[1.0], [2.0]])
    )
    assert out is None


if __name__ == "__main__":
    pytest.main([__file__])
