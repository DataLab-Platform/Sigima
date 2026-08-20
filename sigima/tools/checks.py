# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Checks for 1D and 2D NumPy arrays used in tools (:mod:`sigima.tools.checks`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import numpy as np
from numpy import ma


@dataclass(frozen=True)
class ArrayValidationRules:
    """Hold 1-D array validation rules."""

    #: Label used in error messages (e.g., "x" or "y")
    label: str
    #: Whether to enforce 1-D.
    require_1d: bool = True
    #: Check minimum size
    min_size: int | None = None
    #: Expected dtype (np.issubdtype). Use None to skip.
    dtype: type | None = None
    #: Whether to enforce finite values only.
    finite_only: bool = False
    #: Whether to enforce non-decreasing order.
    sorted_: bool = False
    #: Whether to enforce constant spacing (within rtol).
    evenly_spaced: bool = False
    #: Relative tolerance for regular spacing.
    rtol: float = 1e-5


def _validate_array_1d(arr: np.ndarray, *, rules: ArrayValidationRules) -> None:
    """Validate a single 1D NumPy array according to the provided rules.

    Args:
        arr: Array to validate.
        rules: Validation rules to apply.

    Raises:
        ValueError: If shape constraint is violated.
        ValueError: If size constraint is violated.
        ValueError: If finite constraint is violated.
        ValueError: If order constraint is violated.
        ValueError: If spacing constraint is violated.
        TypeError: If dtype does not match.
    """
    if rules.require_1d and arr.ndim != 1:
        raise ValueError(f"{rules.label} must be 1-D.")
    if rules.min_size is not None and arr.size < rules.min_size:
        raise ValueError(f"{rules.label} must have at least {rules.min_size} elements.")
    if rules.dtype is not None and not np.issubdtype(arr.dtype, rules.dtype):
        raise TypeError(
            f"{rules.label} must be of type {rules.dtype}, but got {arr.dtype}."
        )
    if rules.finite_only and not np.all(np.isfinite(arr)):
        raise ValueError(f"{rules.label} must contain only finite values.")
    if rules.sorted_ and arr.size > 1 and not np.all(np.diff(arr) > 0.0):
        raise ValueError(f"{rules.label} must be sorted in ascending order.")
    if rules.evenly_spaced and arr.size > 1:
        dx = np.diff(arr)
        if not np.allclose(dx, np.mean(dx), rtol=rules.rtol):
            raise ValueError(f"{rules.label} must be evenly spaced.")


def check_1d_array(
    func: Callable[..., Any] | None = None,
    *,
    require_1d: bool = True,
    min_size: int | None = None,
    dtype: type | None = np.inexact,
    finite_only: bool = False,
    sorted_: bool = False,
    evenly_spaced: bool = False,
    rtol: float = 1e-5,
    label: str = "array",
) -> Callable:
    """Decorator to check a single 1D NumPy array.

    Can be used with or without parentheses.

    Args:
        require_1d: Whether to check if the array is 1-D.
        min_size: Minimum size of the array.
        dtype: Expected dtype of the array (np.issubdtype). Use None to skip.
        finite_only: Whether to check if the array contains only finite values.
        sorted_: Whether to check if the array is sorted in ascending order.
        evenly_spaced: Whether to check if the array is evenly spaced.
        rtol: Relative tolerance for regular spacing.
        label: Label for error messages (e.g., "x", "y").

    Returns:
        Decorated function with pre-checks on the single array.
    """

    def decorator(inner_func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(inner_func)
        def wrapper(arr: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            _validate_array_1d(
                arr,
                rules=ArrayValidationRules(
                    label=label,
                    require_1d=require_1d,
                    min_size=min_size,
                    dtype=dtype,
                    finite_only=finite_only,
                    sorted_=sorted_,
                    evenly_spaced=evenly_spaced,
                    rtol=rtol,
                ),
            )
            return inner_func(arr, *args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def check_1d_arrays(
    func: Callable[..., Any] | None = None,
    *,
    x_require_1d: bool = True,
    x_min_size: int | None = None,
    x_dtype: type | None = np.floating,
    x_finite_only: bool = False,
    x_sorted: bool = False,
    x_evenly_spaced: bool = False,
    y_require_1d: bool = True,
    y_min_size: int | None = None,
    y_dtype: type | None = np.inexact,
    y_finite_only: bool = False,
    same_size: bool = True,
    rtol: float = 1e-5,
) -> Callable:
    """Decorator to check paired 1D NumPy arrays (x, y).

    Can be used with or without parentheses.

    Args:
        func: Function to decorate.
        x_require_1d: Whether to check if x is 1-D.
        x_min_size: Minimum size of x.
        x_dtype: Expected dtype of x (np.issubdtype). Use None to skip.
        x_finite_only: Whether to check if x contains only finite values.
        x_sorted: Whether to check if x is sorted in ascending order.
        x_evenly_spaced: Whether to check if x is evenly spaced.
        y_require_1d: Whether to check if y is 1-D.
        y_min_size: Minimum size of y.
        y_dtype: Expected dtype of y (np.issubdtype). Use None to skip.
        y_finite_only: Whether to check if y contains only finite values.
        same_size: Whether to check that x and y have the same size.
        rtol: Relative tolerance for regular spacing (used for x).

    Returns:
        Decorated function with pre-checks on x/y.
    """

    def decorator(inner_func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(inner_func)
        def wrapper(x: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            _validate_array_1d(
                x,
                rules=ArrayValidationRules(
                    label="x",
                    require_1d=x_require_1d,
                    min_size=x_min_size,
                    dtype=x_dtype,
                    finite_only=x_finite_only,
                    sorted_=x_sorted,
                    evenly_spaced=x_evenly_spaced,
                    rtol=rtol,
                ),
            )
            _validate_array_1d(
                y,
                rules=ArrayValidationRules(
                    label="y",
                    require_1d=y_require_1d,
                    min_size=y_min_size,
                    dtype=y_dtype,
                    finite_only=y_finite_only,
                    sorted_=False,
                    evenly_spaced=False,
                    rtol=rtol,
                ),
            )
            if same_size and x.size != y.size:
                raise ValueError("x and y must have the same size.")
            return inner_func(x, y, *args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def check_2d_array(
    func: Callable[..., Any] | None = None,
    *,
    ndim: int = 2,
    dtype: type | None = None,
    non_constant: bool = False,
    finite_only: bool = False,
) -> Callable:
    """
    Decorator to check input for functions operating on 2D NumPy arrays (e.g. images).

    Can be used with parentheses:

    .. code-block:: python

        @check_2d_array(ndim=3, dtype=np.uint8)
        def process_image(image: np.ndarray) -> np.ndarray:
            # Process the image
            return image

    Or without parentheses (default arguments):

    .. code-block:: python

        @check_2d_array
        def process_image(image: np.ndarray) -> np.ndarray:
            # Process the image
            return image

    Args:
        ndim: Expected number of dimensions.
        dtype: Expected dtype.
        non_constant: Whether to check that the array has dynamic range.
        finite_only: Whether to check that all values are finite.

    Returns:
        Decorated function with pre-checks on data.
    """

    def decorator(inner_func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(inner_func)
        def wrapper(data: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            # === Check input array
            if data.ndim != ndim:
                raise ValueError(f"Input array must be {ndim}D, got {data.ndim}D.")
            if dtype is not None and not np.issubdtype(data.dtype, dtype):
                raise TypeError(
                    f"Input array must be of type {dtype}, got {data.dtype}."
                )
            if non_constant:
                dmin, dmax = np.nanmin(data), np.nanmax(data)
                if dmin == dmax:
                    raise ValueError("Input array has no dynamic range.")
            if finite_only and not np.all(np.isfinite(data)):
                raise ValueError("Input array contains non-finite values.")
            # === Call the original function
            return inner_func(data, *args, **kwargs)

        return wrapper

    if func is not None:
        # Usage: `@check_2d_array`
        return decorator(func)
    # Usage: `@check_2d_array(...)`
    return decorator


def warn_if_masked(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator warning when the input array is a masked array.

    Some algorithms rely on underlying libraries (scikit-image, OpenCV, SciPy)
    that do not support :class:`numpy.ma.MaskedArray` inputs: the mask is
    silently ignored, so results may be unexpected inside or near masked areas.

    If the first positional argument of the decorated function is a masked
    array containing at least one masked value, a :class:`UserWarning` is
    emitted. The array is passed unchanged to the decorated function.

    Args:
        func: Function to decorate. Its first positional argument must be a
         NumPy array (possibly masked).

    Returns:
        Decorated function emitting a warning on masked array input.
    """

    @wraps(func)
    def wrapper(data: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        if (
            isinstance(data, ma.MaskedArray)
            and data.mask is not ma.nomask
            and np.any(data.mask)
        ):
            warnings.warn(
                f"{func.__name__} does not support masked arrays (often "
                "produced by ROIs): the mask is ignored, so results may be "
                "unexpected inside or near masked areas.\n"
                "Workaround: extract the ROI data and replace masked/NaN "
                "values with a neutral value (e.g., 0) before calling this "
                "function.",
                UserWarning,
                stacklevel=2,
            )
        return func(data, *args, **kwargs)

    return wrapper


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    """Normalize a convolution/deconvolution kernel if needed.

    This utility function can normalize the kernel to sum to 1.0.

    Args:
        kernel: The kernel array to normalize.

    Returns:
        The normalized kernel if it's not already normalized and if its sum is not
        zero, otherwise the original kernel.

    Note:
        A kernel is considered normalized if ``np.isclose(sum(kernel), 1.0)``.
    """
    kernel_sum = np.sum(kernel)
    if not np.isclose(kernel_sum, 1.0) and kernel_sum != 0.0:
        return kernel / kernel_sum
    return kernel
