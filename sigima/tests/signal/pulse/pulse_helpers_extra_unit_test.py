# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for low-level helpers in :mod:`sigima.tools.signal.pulse`.

Covers shape heuristics, range/mean helpers, polarity detection, amplitude
estimation and crossing-time helpers.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

from sigima.tools.signal import pulse as p_mod
from sigima.tools.signal.pulse import SignalShape


def _square_pulse(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Build a normalised rectangular pulse on ``[0, 1]`` with ``n`` samples."""
    x = np.linspace(0.0, 1.0, n)
    y = np.zeros_like(x)
    y[(x > 0.3) & (x < 0.7)] = 1.0
    return x, y


def _step_pulse(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Build a single rising-edge step signal on ``[0, 1]`` with ``n`` samples."""
    x = np.linspace(0.0, 1.0, n)
    y = np.where(x > 0.5, 1.0, 0.0)
    return x, y


def test_heuristically_recognize_shape_step_and_square() -> None:
    """``heuristically_recognize_shape`` correctly classifies a clean step
    as ``STEP`` and a clean rectangular pulse as ``SQUARE``."""
    x_step, y_step = _step_pulse()
    assert p_mod.heuristically_recognize_shape(x_step, y_step) == SignalShape.STEP
    x_sq, y_sq = _square_pulse()
    assert p_mod.heuristically_recognize_shape(x_sq, y_sq) == SignalShape.SQUARE


def test_heuristically_recognize_shape_invalid_signal() -> None:
    """Mismatched lengths, too-short input or constant arrays must raise
    ``InvalidSignalError`` rather than returning a misleading shape."""
    with pytest.raises(p_mod.InvalidSignalError):
        p_mod.heuristically_recognize_shape(np.array([0.0, 1.0]), np.array([0.0]))
    with pytest.raises(p_mod.InvalidSignalError):
        p_mod.heuristically_recognize_shape(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    x = np.linspace(0.0, 1.0, 10)
    with pytest.raises(p_mod.InvalidSignalError):
        p_mod.heuristically_recognize_shape(x, np.zeros_like(x))


def test_get_range_mean_y_empty_returns_nan() -> None:
    """When the requested ``value_range`` does not intersect ``y``, the
    helper returns ``NaN`` instead of crashing on the empty selection."""
    x = np.linspace(0.0, 1.0, 10)
    y = np.ones_like(x)
    out = p_mod.get_range_mean_y(x, y, value_range=(2.0, 3.0))
    assert np.isnan(out)


def test_get_start_and_end_range() -> None:
    """``get_start_range`` covers ``[xmin, xmin + fraction*span]`` and
    ``get_end_range`` covers ``[xmax - fraction*span, xmax]``."""
    x = np.linspace(0.0, 10.0, 100)
    s_lo, s_hi = p_mod.get_start_range(x, fraction=0.1)
    e_lo, e_hi = p_mod.get_end_range(x, fraction=0.1)
    assert s_lo == pytest.approx(0.0)
    assert s_hi == pytest.approx(1.0)
    assert e_hi == pytest.approx(10.0)
    assert e_lo == pytest.approx(9.0)


def test_detect_polarity_step_positive_and_negative() -> None:
    """Polarity for a step is ``+1`` for a rising edge and ``-1`` for a
    falling edge."""
    x = np.linspace(0.0, 1.0, 200)
    y_pos = np.where(x > 0.5, 1.0, 0.0)
    y_neg = np.where(x > 0.5, 0.0, 1.0)
    assert p_mod.detect_polarity(x, y_pos, signal_shape=SignalShape.STEP) == 1
    assert p_mod.detect_polarity(x, y_neg, signal_shape=SignalShape.STEP) == -1


def test_detect_polarity_unknown_signal_shape() -> None:
    """An unrecognised ``signal_shape`` argument is rejected with a clear
    ``ValueError`` instead of returning ``0`` silently."""
    x = np.linspace(0.0, 1.0, 50)
    y = np.zeros_like(x)
    with pytest.raises(ValueError):
        p_mod.detect_polarity(x, y, signal_shape="weird")  # type: ignore[arg-type]


def test_get_amplitude_step_and_square() -> None:
    """``get_amplitude`` returns ~1.0 (within 5%) for both a unit-height
    step and a unit-height square pulse."""
    x_step, y_step = _step_pulse()
    amp_step = p_mod.get_amplitude(x_step, y_step, signal_shape=SignalShape.STEP)
    assert amp_step == pytest.approx(1.0, rel=0.05)
    x_sq, y_sq = _square_pulse()
    amp_sq = p_mod.get_amplitude(x_sq, y_sq, signal_shape=SignalShape.SQUARE)
    assert amp_sq == pytest.approx(1.0, rel=0.05)


def test_get_amplitude_unknown_signal_shape() -> None:
    """An unrecognised ``signal_shape`` argument is rejected with a clear
    ``ValueError`` (mirrors ``detect_polarity`` behaviour)."""
    x = np.linspace(0.0, 1.0, 50)
    y = np.zeros_like(x)
    with pytest.raises(ValueError):
        p_mod.get_amplitude(x, y, signal_shape="bogus")  # type: ignore[arg-type]


def test_find_crossing_at_ratio_invalid_ratio() -> None:
    """``find_crossing_at_ratio`` requires ``ratio`` in ``[0, 1]``: out of
    range values raise ``ValueError``."""
    x, y = _step_pulse()
    with pytest.raises(ValueError):
        p_mod.find_crossing_at_ratio(x, y, ratio=1.5)


def test_find_crossing_at_ratio_basic() -> None:
    """For a unit step at ``x = 0.5``, the half-amplitude crossing is
    located within ``[0.4, 0.6]``."""
    x, y = _step_pulse()
    crossing = p_mod.find_crossing_at_ratio(
        x, y, ratio=0.5, signal_shape=SignalShape.STEP
    )
    assert crossing is not None
    assert 0.4 < crossing < 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
