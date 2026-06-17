# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for replace_special_values (signal)
-----------------------------------------------

Tests for replacing NaN, +Inf and -Inf values in signals using the various
strategies provided by :func:`sigima.proc.signal.replace_special_values`.
"""

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.proc.signal as sips
from sigima.enums import ReplacementStrategySignal as S
from sigima.proc.base import ReplaceSpecialValuesSignalParam
from sigima.tools.signal.replace_values import count_special_values


def _make_signal(
    y: np.ndarray, x: np.ndarray | None = None
) -> sigima.objects.SignalObj:
    """Helper: create a SignalObj from x/y arrays."""
    if x is None:
        x = np.arange(len(y), dtype=float)
    return sigima.objects.create_signal("test", x, y)


# ---------------------------------------------------------------------------
# Fixed value strategies
# ---------------------------------------------------------------------------


class TestFixedValueStrategies:
    """Test replacement with fixed values (zero, min, max, mean, median)."""

    @pytest.fixture()
    def signal_with_nan(self):
        """Fixture: a signal containing NaN values."""
        y = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        return _make_signal(y)

    def test_replace_zero(self, signal_with_nan):
        """Test replacement of NaN values with zero."""
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.ZERO, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sips.replace_special_values(signal_with_nan, p)
        assert not np.any(np.isnan(dst.y))
        assert dst.y[1] == 0.0
        assert dst.y[3] == 0.0

    def test_replace_min(self, signal_with_nan):
        """Test replacement of NaN values with the minimum of valid data."""
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.MIN, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sips.replace_special_values(signal_with_nan, p)
        assert dst.y[1] == 1.0
        assert dst.y[3] == 1.0

    def test_replace_max(self, signal_with_nan):
        """Test replacement of NaN values with the maximum of valid data."""
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.MAX, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sips.replace_special_values(signal_with_nan, p)
        assert dst.y[1] == 5.0
        assert dst.y[3] == 5.0

    def test_replace_mean(self, signal_with_nan):
        """Test replacement of NaN values with the mean of valid data."""
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.MEAN, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sips.replace_special_values(signal_with_nan, p)
        expected_mean = np.mean([1.0, 3.0, 5.0])
        np.testing.assert_allclose(dst.y[1], expected_mean)
        np.testing.assert_allclose(dst.y[3], expected_mean)

    def test_replace_median(self, signal_with_nan):
        """Test replacement of NaN values with the median of valid data."""
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.MEDIAN, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sips.replace_special_values(signal_with_nan, p)
        expected_median = np.median([1.0, 3.0, 5.0])
        np.testing.assert_allclose(dst.y[1], expected_median)

    def test_replace_constant(self, signal_with_nan):
        """Test replacement of NaN values with a user-specified constant."""
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.CONSTANT,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_constant_value=42.0,
        )
        dst = sips.replace_special_values(signal_with_nan, p)
        assert not np.any(np.isnan(dst.y))
        assert dst.y[1] == 42.0
        assert dst.y[3] == 42.0


# ---------------------------------------------------------------------------
# Removal strategies (signal only)
# ---------------------------------------------------------------------------


class TestRemovalStrategies:
    """Test delete, forward fill, and backward fill."""

    def test_delete(self):
        """Test deletion of NaN values."""
        x = np.array([0.0, 1.0, 2.5, 4.5, 7.0])
        y = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        src = _make_signal(y, x)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.DELETE, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sips.replace_special_values(src, p)
        assert len(dst.y) == 4
        np.testing.assert_array_equal(dst.y, [1.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(dst.x, [0.0, 2.5, 4.5, 7.0])

    def test_delete_warns_uniform_sampling(self):
        """Test that deletion of NaN values emits a warning about uniform sampling."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        y[50] = np.nan
        src = _make_signal(y, x)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.DELETE, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        with pytest.warns(UserWarning, match="uniformly sampled"):
            sips.replace_special_values(src, p)

    def test_forward_fill(self):
        """Test forward fill of NaN values."""
        y = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.FORWARD_FILL,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sips.replace_special_values(src, p)
        np.testing.assert_array_equal(dst.y, [1.0, 1.0, 1.0, 4.0, 5.0])

    def test_forward_fill_leading_nan(self):
        """Test forward fill when leading values are NaN."""
        y = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.FORWARD_FILL,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sips.replace_special_values(src, p)
        np.testing.assert_array_equal(dst.y, [3.0, 3.0, 3.0, 4.0, 5.0])

    def test_backward_fill(self):
        """Test backward fill of NaN values."""
        y = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.BACKWARD_FILL,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sips.replace_special_values(src, p)
        np.testing.assert_array_equal(dst.y, [1.0, 4.0, 4.0, 4.0, 5.0])

    def test_backward_fill_trailing_nan(self):
        """Test backward fill when trailing values are NaN."""
        y = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.BACKWARD_FILL,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sips.replace_special_values(src, p)
        np.testing.assert_array_equal(dst.y, [1.0, 2.0, 3.0, 3.0, 3.0])


# ---------------------------------------------------------------------------
# Interpolation strategies
# ---------------------------------------------------------------------------


class TestInterpolationStrategies:
    """Test interpolation-based replacement."""

    @pytest.fixture()
    def signal_with_gap(self):
        """Fixture: a signal containing NaN values with valid data on both sides."""
        x = np.arange(10, dtype=float)
        y = 2.0 * x + 1.0  # linear: y = 2x + 1
        y[3] = np.nan
        y[7] = np.nan
        return _make_signal(y, x)

    def test_interp_linear(self, signal_with_gap):
        """Test linear interpolation of NaN values."""
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.INTERP_LINEAR,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sips.replace_special_values(signal_with_gap, p)
        # Linear data → perfect reconstruction
        np.testing.assert_allclose(dst.y[3], 7.0, atol=1e-10)
        np.testing.assert_allclose(dst.y[7], 15.0, atol=1e-10)

    def test_interp_cubic(self, signal_with_gap):
        """Test cubic interpolation of NaN values."""
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.INTERP_CUBIC,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sips.replace_special_values(signal_with_gap, p)
        np.testing.assert_allclose(dst.y[3], 7.0, atol=1e-6)

    def test_interp_pchip(self, signal_with_gap):
        """Test PCHIP interpolation of NaN values."""
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.INTERP_PCHIP,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sips.replace_special_values(signal_with_gap, p)
        np.testing.assert_allclose(dst.y[3], 7.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Neighbor strategies
# ---------------------------------------------------------------------------


class TestNeighborStrategies:
    """Test neighbor-based replacement."""

    def test_neighbor_mean(self):
        """Test replacement of NaN values with the mean of neighboring valid data."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.NEIGHBOR_MEAN,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sips.replace_special_values(src, p)
        # Neighbors of index 2 are [2.0, 4.0] → mean = 3.0
        np.testing.assert_allclose(dst.y[2], 3.0)

    def test_neighbor_median(self):
        """Test replacement of NaN values with the median of neighboring valid data."""
        y = np.array([1.0, 2.0, np.nan, 8.0, 5.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.NEIGHBOR_MEDIAN,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sips.replace_special_values(src, p)
        # Neighbors of index 2 are [2.0, 8.0] → median = 5.0
        np.testing.assert_allclose(dst.y[2], 5.0)

    def test_neighbor_min(self):
        """Test replacement of NaN values with the minimum of neighboring valid data."""
        y = np.array([1.0, 2.0, np.nan, 8.0, 5.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.NEIGHBOR_MIN,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sips.replace_special_values(src, p)
        # Neighbors of index 2 are [2.0, 8.0] → min = 2.0
        np.testing.assert_allclose(dst.y[2], 2.0)

    def test_neighbor_max(self):
        """Test replacement of NaN values with the maximum of neighboring valid data."""
        y = np.array([1.0, 2.0, np.nan, 8.0, 5.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.NEIGHBOR_MAX,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sips.replace_special_values(src, p)
        # Neighbors of index 2 are [2.0, 8.0] → max = 8.0
        np.testing.assert_allclose(dst.y[2], 8.0)


# ---------------------------------------------------------------------------
# Multiple targets (NaN + Inf)
# ---------------------------------------------------------------------------


class TestMultipleTargets:
    """Test replacing NaN, +Inf and -Inf simultaneously."""

    def test_all_three_targets(self):
        """Test replacement of NaN, +Inf and -Inf values in the same signal."""
        y = np.array([1.0, np.nan, np.inf, -np.inf, 5.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.ZERO,
            posinf_strategy=S.MAX,
            neginf_strategy=S.MIN,
        )
        dst = sips.replace_special_values(src, p)
        assert dst.y[1] == 0.0  # NaN → 0
        assert dst.y[2] == 5.0  # +inf → max of valid data
        assert dst.y[3] == 0.0  # -inf → min (0.0 is now min after NaN→0)

    def test_none_strategy_skips(self):
        """Test that the NONE strategy skips replacement."""
        y = np.array([1.0, np.nan, 3.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.NONE, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sips.replace_special_values(src, p)
        assert np.isnan(dst.y[1])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases."""

    def test_no_special_values(self):
        """Test that a signal with no special values is unchanged."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.ZERO, posinf_strategy=S.ZERO, neginf_strategy=S.ZERO
        )
        dst = sips.replace_special_values(src, p)
        np.testing.assert_array_equal(dst.y, y)

    def test_all_nan(self):
        """Test that a signal with all NaN values is replaced correctly."""
        y = np.array([np.nan, np.nan, np.nan])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.ZERO, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sips.replace_special_values(src, p)
        np.testing.assert_array_equal(dst.y, [0.0, 0.0, 0.0])

    def test_posinf_only(self):
        """Test replacement of +Inf values with zero."""
        y = np.array([1.0, np.inf, 3.0])
        src = _make_signal(y)
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.NONE, posinf_strategy=S.ZERO, neginf_strategy=S.NONE
        )
        dst = sips.replace_special_values(src, p)
        assert dst.y[1] == 0.0
        assert dst.y[0] == 1.0
        assert dst.y[2] == 3.0


# ---------------------------------------------------------------------------
# Validation test (required by the test framework)
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_signal_replace_special_values() -> None:
    """Validation test for the signal replace_special_values processing."""
    # Use separate data per category to avoid cross-contamination
    # (e.g. mean of data containing Inf is NaN)
    y_nan = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_all = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 6.0, 7.0, 8.0])

    # Test fixed strategies on NaN-only data
    for strategy in (S.ZERO, S.MIN, S.MAX, S.MEAN, S.MEDIAN):
        src = _make_signal(y_nan.copy())
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=strategy,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sigima.proc.signal.replace_special_values(src, p)
        assert not np.any(np.isnan(dst.y))

    # Test all three targets with non-stat strategies
    src = _make_signal(y_all.copy())
    p = ReplaceSpecialValuesSignalParam.create(
        nan_strategy=S.ZERO,
        posinf_strategy=S.ZERO,
        neginf_strategy=S.ZERO,
    )
    dst = sigima.proc.signal.replace_special_values(src, p)
    assert not np.any(np.isnan(dst.y))
    assert not np.any(np.isinf(dst.y))

    # Test interpolation strategies
    for strategy in (S.INTERP_LINEAR, S.INTERP_CUBIC, S.INTERP_PCHIP):
        src = _make_signal(y_nan.copy())
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=strategy,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sigima.proc.signal.replace_special_values(src, p)
        assert not np.any(np.isnan(dst.y))

    # Test neighbor strategies
    for strategy in (
        S.NEIGHBOR_MIN,
        S.NEIGHBOR_MAX,
        S.NEIGHBOR_MEAN,
        S.NEIGHBOR_MEDIAN,
    ):
        src = _make_signal(y_nan.copy())
        p = ReplaceSpecialValuesSignalParam.create(
            nan_strategy=strategy,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sigima.proc.signal.replace_special_values(src, p)
        assert not np.any(np.isnan(dst.y))

    # Test constant strategy
    src = _make_signal(y_nan.copy())
    p = ReplaceSpecialValuesSignalParam.create(
        nan_strategy=S.CONSTANT,
        posinf_strategy=S.NONE,
        neginf_strategy=S.NONE,
        nan_constant_value=-999.0,
    )
    dst = sigima.proc.signal.replace_special_values(src, p)
    assert not np.any(np.isnan(dst.y))
    assert dst.y[1] == -999.0


# ---------------------------------------------------------------------------
# Count special values utility
# ---------------------------------------------------------------------------


class TestCountSpecialValues:
    """Test the count_special_values utility."""

    def test_count_mixed(self):
        """Test counting of NaN, +Inf and -Inf values in a mixed array."""
        y = np.array([1.0, np.nan, np.inf, -np.inf, 5.0, np.nan])

        counts = count_special_values(y)
        assert counts == {"nan": 2, "posinf": 1, "neginf": 1}

    def test_count_none(self):
        """Test counting when there are no special values."""
        y = np.array([1.0, 2.0, 3.0])

        counts = count_special_values(y)
        assert counts == {"nan": 0, "posinf": 0, "neginf": 0}

    def test_count_all_nan(self):
        """Test counting when all values are NaN."""
        y = np.array([np.nan, np.nan])

        counts = count_special_values(y)
        assert counts == {"nan": 2, "posinf": 0, "neginf": 0}
