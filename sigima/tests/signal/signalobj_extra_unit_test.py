# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :class:`sigima.objects.SignalObj` internals.

Covers data-type validation, copy/setter behavior, HTML representation,
``create_signal_parameters`` factory branches and the polynomial signal
title generator (:class:`sigima.objects.PolyParam.generate_title`).
"""

# pylint: disable=invalid-name
# pylint: disable=protected-access

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import PolyParam, SignalObj
from sigima.objects.signal.creation import (
    SignalTypes,
    create_signal_parameters,
)
from sigima.objects.signal.object import validate_and_convert_dtype
from sigima.objects.signal.roi import create_signal_roi

# ===========================================================================
# validate_and_convert_dtype
# ===========================================================================


def test_validate_and_convert_dtype_integer_to_float() -> None:
    """Integer arrays are promoted to float64."""
    arr = np.array([1, 2, 3], dtype=np.int32)
    out = validate_and_convert_dtype(arr)
    assert out.dtype == np.float64


def test_validate_and_convert_dtype_invalid_dtype() -> None:
    """Non-floating, non-integer arrays raise ValueError."""
    arr = np.array([True, False, True], dtype=np.bool_)
    with pytest.raises(ValueError, match="Invalid data type"):
        validate_and_convert_dtype(arr)


# ===========================================================================
# SignalObj copy / set_data_type / setters validation
# ===========================================================================


def _make_signal(n: int = 16) -> SignalObj:
    """Build a single-period sine ``SignalObj`` titled ``"cov"``."""
    sig = SignalObj(title="cov")
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x)
    sig.set_xydata(x, y)
    return sig


def test_signalobj_copy_invalid_dtype_raises() -> None:
    """Copy with an unsupported dtype raises RuntimeError."""
    sig = _make_signal()
    with pytest.raises(RuntimeError, match="float64/complex128"):
        sig.copy(dtype=np.int32)


def test_signalobj_set_data_type_raises() -> None:
    """``set_data_type`` is not supported for signals."""
    sig = _make_signal()
    with pytest.raises(RuntimeError, match="not support"):
        sig.set_data_type(np.float32)


def test_signalobj_set_x_non_monotonic_raises() -> None:
    """Setting non-monotonic X data raises ValueError."""
    sig = _make_signal()
    bad_x = np.array([0.0, 1.0, 0.5, 2.0])
    with pytest.raises(ValueError, match="monotonic"):
        sig.x = np.tile(bad_x, len(sig.x) // 4)


def test_signalobj_dx_dy_roundtrip() -> None:
    """Setting and reading dx/dy uncertainties works as expected."""
    sig = _make_signal(n=8)
    assert sig.dx is None
    assert sig.dy is None
    dx = np.full_like(sig.x, 0.01)
    dy = np.full_like(sig.y, 0.02)
    sig.dx = dx
    sig.dy = dy
    assert sig.dx is not None and np.allclose(sig.dx, 0.01)
    assert sig.dy is not None and np.allclose(sig.dy, 0.02)
    sig.dx = None
    sig.dy = None
    assert sig.dx is None
    assert sig.dy is None


def test_signalobj_set_dx_without_data_raises() -> None:
    """Setting dx on an empty SignalObj raises ValueError."""
    sig = SignalObj(title="empty")
    with pytest.raises((ValueError, AttributeError, AssertionError)):
        sig.dx = np.array([1.0, 2.0])


# ===========================================================================
# SignalObj._repr_html_
# ===========================================================================


def test_signalobj_repr_html_basic() -> None:
    """``_repr_html_`` produces a non-empty HTML string."""
    sig = _make_signal()
    html = sig._repr_html_()
    assert "<table" in html and "</table>" in html
    assert "SignalObj" in html
    assert "Points" in html


def test_signalobj_repr_html_with_units_and_roi() -> None:
    """``_repr_html_`` includes axis labels, units and ROI count when set."""
    sig = _make_signal()
    sig.xlabel = "Time"
    sig.xunit = "s"
    sig.ylabel = "Voltage"
    sig.yunit = "V"
    sig.roi = create_signal_roi([[0.1, 0.5]], indices=False)
    html = sig._repr_html_()
    assert "Time" in html
    assert "Voltage" in html
    assert "] s" in html
    assert "] V" in html
    assert "ROIs" in html


def test_signalobj_repr_html_empty() -> None:
    """``_repr_html_`` works with an empty SignalObj (no data)."""
    sig = SignalObj(title="empty")
    sig.set_xydata(None, None)
    html = sig._repr_html_()
    assert "N/A" in html


# ===========================================================================
# create_signal_parameters
# ===========================================================================


def test_create_signal_parameters_all_fields() -> None:
    """All optional fields (size, bounds, labels, units) are propagated
    onto the parameter object returned by ``create_signal_parameters``."""
    p = create_signal_parameters(
        SignalTypes.ZERO,
        title="my",
        size=64,
        xmin=0.0,
        xmax=10.0,
        xlabel="x",
        ylabel="y",
        xunit="s",
        yunit="V",
    )
    assert p.title == "my"
    assert p.size == 64
    assert p.xmin == 0.0
    assert p.xmax == 10.0
    assert p.xlabel == "x"
    assert p.ylabel == "y"
    assert p.xunit == "s"
    assert p.yunit == "V"


def test_create_signal_parameters_unknown_type_raises() -> None:
    """Passing an unknown signal-type marker raises ``ValueError`` instead
    of silently returning ``None`` or a wrong parameter class."""

    class FakeType:
        """Sentinel type that is not a member of ``SignalTypes``."""

    with pytest.raises(ValueError):
        create_signal_parameters(FakeType())  # type: ignore[arg-type]


# ===========================================================================
# PolyParam.generate_title
# ===========================================================================


def test_polyparam_generate_title_default() -> None:
    """Default coefficients ``a0=1, a1=1`` produce a title containing both
    a constant ``1`` and the linear term ``x``."""
    p = PolyParam()
    title = p.generate_title()
    # Default: a0=1, a1=1 -> "1+x"
    assert "1" in title and "x" in title


def test_polyparam_generate_title_unit_and_negative() -> None:
    """Title generation uses unit coefficients without explicit ``1*`` and
    correctly inserts the negative sign for negative coefficients."""
    p = PolyParam()
    p.a0 = 0.0
    p.a1 = 1.0
    p.a2 = -1.0
    p.a3 = 2.0
    p.a4 = 0.0
    p.a5 = -1.0
    title = p.generate_title()
    assert "x" in title
    assert "-x^2" in title
    assert "x^3" in title
    assert "-x^5" in title


def test_polyparam_generate_title_negative_one_linear() -> None:
    """Single linear term ``-x`` (a1 = -1, all others 0) is rendered as
    the canonical ``"-x"`` string."""
    p = PolyParam()
    p.a0 = 0.0
    p.a1 = -1.0
    p.a2 = 0.0
    p.a3 = 0.0
    p.a4 = 0.0
    p.a5 = 0.0
    assert p.generate_title() == "-x"


def test_polyparam_generate_title_zero() -> None:
    """All-zero coefficients render as ``"0"`` rather than an empty string
    (avoids generating an empty plot title)."""
    p = PolyParam()
    p.a0 = 0.0
    p.a1 = 0.0
    p.a2 = 0.0
    p.a3 = 0.0
    p.a4 = 0.0
    p.a5 = 0.0
    assert p.generate_title() == "0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
