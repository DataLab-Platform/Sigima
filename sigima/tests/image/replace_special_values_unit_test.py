# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for replace_special_values (image)
----------------------------------------------

Tests for replacing NaN, +Inf and -Inf values in images using the various
strategies provided by :func:`sigima.proc.image.replace_special_values`.
"""

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.proc.image
import sigima.proc.image as sipi
from sigima.enums import ReplacementStrategyImage as S
from sigima.proc.base import ReplaceSpecialValuesImageParam


def _make_image(data: np.ndarray) -> sigima.objects.ImageObj:
    """Helper: create an ImageObj from a 2-D array."""
    return sigima.objects.create_image("test", data)


# ---------------------------------------------------------------------------
# Fixed value strategies
# ---------------------------------------------------------------------------


class TestFixedValueStrategies:
    """Test replacement with fixed values (zero, min, max, mean, median)."""

    @pytest.fixture()
    def image_with_nan(self):
        data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])
        return _make_image(data)

    def test_replace_zero(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.ZERO, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        assert not np.any(np.isnan(dst.data))
        assert dst.data[0, 1] == 0.0
        assert dst.data[1, 2] == 0.0

    def test_replace_min(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.MIN, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        valid_min = np.nanmin(image_with_nan.data)
        assert dst.data[0, 1] == pytest.approx(valid_min)
        assert dst.data[1, 2] == pytest.approx(valid_min)

    def test_replace_max(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.MAX, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        valid_max = np.nanmax(image_with_nan.data)
        assert dst.data[0, 1] == pytest.approx(valid_max)
        assert dst.data[1, 2] == pytest.approx(valid_max)

    def test_replace_mean(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.MEAN, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        valid_mean = np.nanmean(image_with_nan.data)
        assert dst.data[0, 1] == pytest.approx(valid_mean)

    def test_replace_median(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.MEDIAN, posinf_strategy=S.NONE, neginf_strategy=S.NONE
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        valid_median = np.nanmedian(image_with_nan.data)
        assert dst.data[0, 1] == pytest.approx(valid_median)

    def test_replace_constant(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.CONSTANT,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_constant_value=42.0,
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        assert not np.any(np.isnan(dst.data))
        assert dst.data[0, 1] == 42.0
        assert dst.data[1, 2] == 42.0


# ---------------------------------------------------------------------------
# Neighbor strategies
# ---------------------------------------------------------------------------


class TestNeighborStrategies:
    """Test N-neighbor replacement strategies."""

    @pytest.fixture()
    def image_with_nan(self):
        data = np.ones((5, 5), dtype=float) * 4.0
        data[2, 2] = np.nan
        return _make_image(data)

    def test_neighbor_mean(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.NEIGHBOR_MEAN,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        assert not np.any(np.isnan(dst.data))
        assert dst.data[2, 2] == pytest.approx(4.0)

    def test_neighbor_median(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.NEIGHBOR_MEDIAN,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        assert not np.any(np.isnan(dst.data))
        assert dst.data[2, 2] == pytest.approx(4.0)

    def test_neighbor_min(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.NEIGHBOR_MIN,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        assert not np.any(np.isnan(dst.data))
        assert dst.data[2, 2] == pytest.approx(4.0)

    def test_neighbor_max(self, image_with_nan):
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.NEIGHBOR_MAX,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sipi.replace_special_values(image_with_nan, p)
        assert not np.any(np.isnan(dst.data))
        assert dst.data[2, 2] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Multiple targets
# ---------------------------------------------------------------------------


class TestMultipleTargets:
    """Test independent processing of NaN, +Inf and -Inf."""

    def test_all_three_targets(self):
        # Strategies are applied sequentially: NaN first, then +inf, then -inf.
        # After NaN→ZERO, the data min includes 0.0, so -inf→MIN gives 0.0.
        data = np.array([[1.0, np.nan, 3.0], [np.inf, 5.0, -np.inf], [7.0, 8.0, 9.0]])
        src = _make_image(data)
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.ZERO,
            posinf_strategy=S.MAX,
            neginf_strategy=S.MIN,
        )
        dst = sipi.replace_special_values(src, p)
        assert not np.any(np.isnan(dst.data))
        assert not np.any(np.isinf(dst.data))
        assert dst.data[0, 1] == 0.0  # NaN → zero
        assert dst.data[1, 0] == pytest.approx(9.0)  # +inf → max(after NaN→0)
        assert dst.data[1, 2] == pytest.approx(0.0)  # -inf → min(after NaN→0)

    def test_none_leaves_unchanged(self):
        data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        src = _make_image(data)
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.NONE,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sipi.replace_special_values(src, p)
        assert np.isnan(dst.data[0, 1])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge conditions and special inputs."""

    def test_no_special_values(self):
        data = np.arange(9, dtype=float).reshape(3, 3)
        src = _make_image(data)
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.ZERO,
            posinf_strategy=S.ZERO,
            neginf_strategy=S.ZERO,
        )
        dst = sipi.replace_special_values(src, p)
        np.testing.assert_array_equal(dst.data, data)

    def test_posinf_only(self):
        data = np.array([[1.0, np.inf], [3.0, 4.0]])
        src = _make_image(data)
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.NONE,
            posinf_strategy=S.ZERO,
            neginf_strategy=S.NONE,
        )
        dst = sipi.replace_special_values(src, p)
        assert dst.data[0, 1] == 0.0
        assert not np.any(np.isinf(dst.data))


# ---------------------------------------------------------------------------
# Validation test (required by the test framework)
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_image_replace_special_values() -> None:
    """Validation test for the image replace_special_values processing."""
    # Use NaN-only data for stat-based strategies (mean of Inf is NaN)
    data_nan = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])
    data_all = np.array([[1.0, np.nan, 3.0], [np.inf, 5.0, -np.inf], [7.0, 8.0, 9.0]])

    # Test fixed strategies on NaN-only data
    for strategy in (S.ZERO, S.MIN, S.MAX, S.MEAN, S.MEDIAN):
        src = _make_image(data_nan.copy())
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=strategy,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
        )
        dst = sigima.proc.image.replace_special_values(src, p)
        assert not np.any(np.isnan(dst.data))

    # Test all three targets with non-stat strategies
    src = _make_image(data_all.copy())
    p = ReplaceSpecialValuesImageParam.create(
        nan_strategy=S.ZERO,
        posinf_strategy=S.ZERO,
        neginf_strategy=S.ZERO,
    )
    dst = sigima.proc.image.replace_special_values(src, p)
    assert not np.any(np.isnan(dst.data))
    assert not np.any(np.isinf(dst.data))

    # Test neighbor strategies
    data_smooth = np.arange(25, dtype=float).reshape(5, 5)
    data_smooth[2, 2] = np.nan
    for strategy in (S.NEIGHBOR_MIN, S.NEIGHBOR_MAX, S.NEIGHBOR_MEAN, S.NEIGHBOR_MEDIAN):
        src = _make_image(data_smooth.copy())
        p = ReplaceSpecialValuesImageParam.create(
            nan_strategy=strategy,
            posinf_strategy=S.NONE,
            neginf_strategy=S.NONE,
            nan_neighbor_size=1,
        )
        dst = sigima.proc.image.replace_special_values(src, p)
        assert not np.any(np.isnan(dst.data))

    # Test constant strategy
    src = _make_image(data_nan.copy())
    p = ReplaceSpecialValuesImageParam.create(
        nan_strategy=S.CONSTANT,
        posinf_strategy=S.NONE,
        neginf_strategy=S.NONE,
        nan_constant_value=-999.0,
    )
    dst = sigima.proc.image.replace_special_values(src, p)
    assert not np.any(np.isnan(dst.data))
    assert dst.data[0, 1] == -999.0


# ---------------------------------------------------------------------------
# Count special values utility
# ---------------------------------------------------------------------------


class TestCountSpecialValues2D:
    """Test the count_special_values_2d utility."""

    def test_count_mixed(self):
        data = np.array([[1.0, np.nan], [np.inf, -np.inf]])
        from sigima.tools.image.replace_values import count_special_values_2d

        counts = count_special_values_2d(data)
        assert counts == {"nan": 1, "posinf": 1, "neginf": 1}

    def test_count_none(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        from sigima.tools.image.replace_values import count_special_values_2d

        counts = count_special_values_2d(data)
        assert counts == {"nan": 0, "posinf": 0, "neginf": 0}
