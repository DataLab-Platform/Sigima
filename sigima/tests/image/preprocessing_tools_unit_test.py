# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for low-level functions in :mod:`sigima.tools.image.preprocessing`.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigima.enums import BinningOperation
from sigima.tools.image.preprocessing import (
    _USE_NEW_SHAPE_API,
    binning,
    distance_matrix,
    fit_circle_model,
    fit_ellipse_model,
    get_absolute_level,
    scale_data_to_min_max,
    zero_padding,
)

# ===========================================================================
# fit_circle_model / fit_ellipse_model
# ===========================================================================


def _circle_contour(xc: float, yc: float, r: float, n: int = 64) -> np.ndarray:
    """Sample ``n`` evenly-spaced points along the circle of centre ``(xc, yc)``
    and radius ``r`` (returned as an ``(n, 2)`` array)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([xc + r * np.cos(theta), yc + r * np.sin(theta)])


def _ellipse_contour(
    xc: float, yc: float, a: float, b: float, theta0: float = 0.0, n: int = 128
) -> np.ndarray:
    """Sample ``n`` points along an ellipse with semi-axes ``(a, b)`` rotated
    by ``theta0`` and centred on ``(xc, yc)``."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    ct, st = np.cos(theta0), np.sin(theta0)
    xr = xc + ct * x - st * y
    yr = yc + st * x + ct * y
    return np.column_stack([xr, yr])


def test_fit_circle_model_success() -> None:
    """``fit_circle_model`` recovers the radius (and one of the centre coords)
    of a noise-free circular contour sampled densely."""
    contour = _circle_contour(10.0, 20.0, 5.0)
    result = fit_circle_model(contour)
    assert result is not None
    xc, _yc, r = result
    assert xc == pytest.approx(10.0, abs=1e-6) or xc == pytest.approx(20.0, abs=1e-6)
    assert r == pytest.approx(5.0, abs=1e-6)


def test_fit_circle_model_failure() -> None:
    # Degenerate input: not enough non-collinear points
    """Degenerate input (only two collinear points) must not raise: the fit
    is allowed to return ``None`` or a tuple, but never crash."""
    contour = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = fit_circle_model(contour)
    # Either None or some result; at least it should not raise
    assert result is None or isinstance(result, tuple)


def test_fit_ellipse_model_success() -> None:
    """``fit_ellipse_model`` returns the 5 ellipse parameters for a clean
    elliptical contour."""
    contour = _ellipse_contour(0.0, 0.0, 5.0, 3.0)
    result = fit_ellipse_model(contour)
    assert result is not None
    assert len(result) == 5


def test_fit_ellipse_model_failure() -> None:
    """Three collinear points cannot describe an ellipse: the fit must
    fail gracefully (return ``None`` or a tuple) rather than raise."""
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = fit_ellipse_model(contour)
    assert result is None or isinstance(result, tuple)


def test_fit_circle_model_ground_truth() -> None:
    """Verify that ``fit_circle_model`` recovers **all** circle parameters
    (xc, yc, radius) from a noise-free contour.

    Note: the functions swap x/y because scikit-image models interpret the
    contour columns as (row, col), so the returned ``xc`` corresponds to the
    contour's second column and ``yc`` to the first.
    """
    if not _USE_NEW_SHAPE_API:
        pytest.skip()
    xc_in, yc_in, r_in = 7.0, -5.0, 12.0
    contour = _circle_contour(xc_in, yc_in, r_in, n=256)
    result = fit_circle_model(contour)
    assert result is not None
    # xc/yc are swapped by the row/col → x/y conversion
    yc, xc, r = result
    assert xc == pytest.approx(xc_in, abs=1e-6)
    assert yc == pytest.approx(yc_in, abs=1e-6)
    assert r == pytest.approx(r_in, abs=1e-6)


def test_fit_ellipse_model_ground_truth() -> None:
    """Verify that ``fit_ellipse_model`` recovers **all** ellipse parameters
    (xc, yc, a, b, theta) from a noise-free contour.

    Note: the functions swap x/y and a/b because scikit-image models interpret
    the contour columns as (row, col), so centre and semi-axes are transposed.
    """
    if not _USE_NEW_SHAPE_API:
        pytest.skip()
    xc_in, yc_in = -3.0, 5.0
    a_in, b_in = 4.0, 8.0
    theta_in = np.pi / 6
    contour = _ellipse_contour(xc_in, yc_in, a_in, b_in, theta0=theta_in, n=256)
    result = fit_ellipse_model(contour)
    assert result is not None
    # xc/yc are swapped by the row/col → x/y conversion
    yc, xc, a, b, theta = result
    assert xc == pytest.approx(xc_in, abs=1e-4)
    assert yc == pytest.approx(yc_in, abs=1e-4)
    assert a == pytest.approx(a_in, abs=1e-4)
    assert b == pytest.approx(b_in, abs=1e-4)
    # The fitted angle is expected to differ by π/2,
    # theta along y axis instead of x axis
    assert theta == pytest.approx(theta_in + np.pi / 2, abs=1e-4)


# ===========================================================================
# get_absolute_level
# ===========================================================================


def test_get_absolute_level_valid() -> None:
    """``get_absolute_level`` linearly maps a fractional level in ``[0, 1]``
    to the corresponding absolute value within the data range."""
    data = np.array([0.0, 5.0, 10.0])
    assert get_absolute_level(data, 0.0) == 0.0
    assert get_absolute_level(data, 1.0) == 10.0
    assert get_absolute_level(data, 0.5) == 5.0


@pytest.mark.parametrize("level", [-0.1, 1.1, "abc", None])
def test_get_absolute_level_invalid_raises(level) -> None:
    """Out-of-range or non-numeric levels are rejected with ``ValueError``,
    preventing silent garbage values from propagating."""
    with pytest.raises(ValueError):
        get_absolute_level(np.array([0.0, 1.0]), level)


# ===========================================================================
# distance_matrix
# ===========================================================================


def test_distance_matrix_basic() -> None:
    """``distance_matrix`` returns the upper-triangular pairwise Euclidean
    distance matrix (lower triangle zeroed via ``np.triu``)."""
    coords = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]
    dm = distance_matrix(coords)
    assert dm.shape == (3, 3)
    # Upper triangle only (np.triu)
    assert dm[0, 1] == pytest.approx(3.0)
    assert dm[0, 2] == pytest.approx(4.0)
    assert dm[1, 2] == pytest.approx(5.0)
    assert dm[1, 0] == 0.0  # lower triangle zeroed


# ===========================================================================
# binning
# ===========================================================================


def test_binning_sum_average() -> None:
    """``binning`` with ``sum`` aggregates pixel groups by addition while
    ``average`` produces the per-group mean (with dtype preserved)."""
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    result = binning(data, sx=2, sy=1, operation="sum")
    assert result.shape == (2, 2)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, np.array([[3, 7], [11, 15]]))

    result = binning(data, sx=2, sy=1, operation="average")
    np.testing.assert_array_equal(result, np.array([[1, 3], [5, 7]]))


@pytest.mark.parametrize("operation", ["median", "min", "max"])
def test_binning_other_operations(operation) -> None:
    """Non-arithmetic binning operations (``median``/``min``/``max``) must
    produce arrays of the expected reduced shape."""
    data = np.arange(16, dtype=float).reshape(4, 4)
    result = binning(data, sx=2, sy=2, operation=operation)
    assert result.shape == (2, 2)


def test_binning_with_enum() -> None:
    """``binning`` accepts both string operation names and ``BinningOperation``
    enum members for the same behaviour."""
    data = np.ones((4, 4), dtype=np.int32)
    result = binning(data, sx=2, sy=2, operation=BinningOperation.SUM)
    np.testing.assert_array_equal(result, 4 * np.ones((2, 2), dtype=np.int32))


def test_binning_invalid_operation_raises() -> None:
    """An unknown operation string raises ``ValueError`` with a clear
    ``Invalid operation`` message rather than silently fall back."""
    data = np.ones((4, 4))
    with pytest.raises(ValueError, match="Invalid operation"):
        binning(data, sx=2, sy=2, operation="bogus")


def test_binning_with_dtype_override() -> None:
    """The optional ``dtype`` argument forces the output dtype, which is
    required when the natural sum would overflow the input integer type."""
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    result = binning(data, sx=2, sy=2, operation="sum", dtype=np.float64)
    assert result.dtype == np.float64


# ===========================================================================
# scale_data_to_min_max
# ===========================================================================


def test_scale_data_to_min_max_basic() -> None:
    """``scale_data_to_min_max`` rescales an array linearly so its min/max
    match the requested target bounds."""
    data = np.array([[0.0, 5.0], [10.0, 20.0]])
    out = scale_data_to_min_max(data, 0.0, 1.0)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_scale_data_to_min_max_no_op() -> None:
    # Hit the early-return branch where (dmin, dmax) already equals (zmin, zmax).
    """When the data range already matches the target, the function must
    short-circuit and return the input array (identity, no copy)."""
    data = np.array([[0.0, 5.0], [10.0, 10.0]])
    out = scale_data_to_min_max(data, 0.0, 10.0)
    assert out is data  # identity returned


# ===========================================================================
# zero_padding
# ===========================================================================


def test_zero_padding_bottom_right() -> None:
    """``position='bottom-right'`` keeps the original block at the top-left
    of the output and fills the appended rows/columns with zeros."""
    data = np.ones((2, 3), dtype=np.int16)
    out = zero_padding(data, rows=2, cols=4, position="bottom-right")
    assert out.shape == (4, 7)
    # Top-left block preserved
    np.testing.assert_array_equal(out[:2, :3], data)
    # Padding zone is zeros
    assert out[2:, :].sum() == 0
    assert out[:, 3:].sum() == 0


def test_zero_padding_around() -> None:
    """``position='around'`` centres the original block in the padded output."""
    data = np.ones((2, 2), dtype=np.int16)
    out = zero_padding(data, rows=2, cols=2, position="around")
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out[1:3, 1:3], data)


def test_zero_padding_negative_raises() -> None:
    """Negative padding sizes are forbidden and rejected at validation time."""
    data = np.ones((2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        zero_padding(data, rows=-1, cols=0)


def test_zero_padding_invalid_position_raises() -> None:
    """Unknown ``position`` strings raise ``ValueError`` with an explicit
    ``Invalid position`` message."""
    data = np.ones((2, 2))
    with pytest.raises(ValueError, match="Invalid position"):
        zero_padding(data, rows=1, cols=1, position="middle")


if __name__ == "__main__":
    pytest.main([__file__])
