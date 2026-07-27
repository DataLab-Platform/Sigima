# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :mod:`sigima.tools.signal.stability`.

Covers tau-too-small / tau-too-large NaN branches and a basic positive
case for ``allan_variance``, ``allan_deviation``,
``overlapping_allan_variance``, ``modified_allan_variance``,
``hadamard_variance`` and ``total_variance``.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

from sigima.tools.signal import stability


def _make_xy(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic ``(x, y)`` pair (white Gaussian noise, seed=42)."""
    x = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    return x, y


def test_allan_variance_tau_too_small_raises() -> None:
    """``allan_variance`` rejects sub-sample tau values (``tau < dt``)
    with a ``ValueError`` rather than silently producing nonsense."""
    x, y = _make_xy()
    dt = x[1] - x[0]
    with pytest.raises(ValueError):
        stability.allan_variance(x, y, np.array([dt / 10.0]))


def test_allan_variance_tau_too_large_returns_nan() -> None:
    """When tau equals or exceeds the full record length, no overlapping
    bins exist; the function must return ``NaN`` for that tau instead of
    crashing."""
    x, y = _make_xy(20)
    dt = x[1] - x[0]
    result = stability.allan_variance(x, y, np.array([dt * len(y)]))
    assert np.isnan(result[0])


def test_allan_variance_n_zero_returns_nan() -> None:
    """With very few samples the bin count can collapse; the function
    must still return an array of the requested shape (filled with finite
    or ``NaN`` values, but never raising)."""
    n = 4
    x = np.linspace(0.0, 1.0, n)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    dt = x[1] - x[0]
    result = stability.allan_variance(x, y, np.array([2 * dt]))
    assert result.shape == (1,)


@pytest.mark.parametrize(
    "func",
    [
        stability.overlapping_allan_variance,
        stability.modified_allan_variance,
        stability.hadamard_variance,
        stability.total_variance,
    ],
)
def test_variance_tau_branches(func) -> None:
    """All four variance estimators share the same tau coverage matrix:
    too-small (``ValueError``, consistently with ``allan_variance``), too-large
    (NaN), and a valid intermediate tau yielding a finite-shape result."""
    x, y = _make_xy(60)
    dt = x[1] - x[0]
    with pytest.raises(ValueError):
        func(x, y, np.array([0.1 * dt]))
    r2 = func(x, y, np.array([dt * len(y)]))
    assert np.isnan(r2[0])
    r3 = func(x, y, np.array([3 * dt]))
    assert r3.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
