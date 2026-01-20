"""
Signal/Image Preprocessing
--------------------------

This module contains utility functions for preprocessing and transforming image data:

- Binning and scaling operations
- Zero padding for Fourier analysis
- Utility functions for data transformation
- Compatibility helpers for scikit-image API changes

.. note::
    All functions in this module are also available directly in the parent
    `sigima.tools.image` package.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.spatial as spt
from numpy import ma
from packaging.version import Version
from skimage import __version__, measure

from sigima.enums import BinningOperation
from sigima.tools.checks import check_2d_array

# Check scikit-image version for API compatibility
# Version 0.26.0 introduced breaking changes to CircleModel and EllipseModel:
# - Old API: model.estimate(contour) + model.params
# - New API: model.from_estimate(contour) + model.center/radius/axis_lengths properties
# Reference: https://github.com/scikit-image/scikit-image/commit/6d6e7924cca105320690f716a14c5bd11055bf43
_SKIMAGE_VERSION = Version(__version__)
_USE_NEW_SHAPE_API = _SKIMAGE_VERSION >= Version("0.26.0")


def fit_circle_model(contour: np.ndarray) -> tuple[float, float, float] | None:
    """Fit circle model to contour with version compatibility.

    Args:
        contour: Contour coordinates array (N, 2)

    Returns:
        Tuple (xc, yc, radius) or None if fitting fails
    """
    if _USE_NEW_SHAPE_API:
        model = measure.CircleModel.from_estimate(contour)
        if model:
            return model.center[0], model.center[1], model.radius
    else:
        model = measure.CircleModel()
        if model.estimate(contour):
            yc, xc, radius = model.params
            return xc, yc, radius
    return None


def fit_ellipse_model(
    contour: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    """Fit ellipse model to contour with version compatibility.

    Args:
        contour: Contour coordinates array (N, 2)

    Returns:
        Tuple (xc, yc, a, b, theta) or None if fitting fails,
        where a and b are semi-major and semi-minor axes
    """
    if _USE_NEW_SHAPE_API:
        model = measure.EllipseModel.from_estimate(contour)
        if model:
            xc, yc = model.center[0], model.center[1]
            a, b = model.axis_lengths[0], model.axis_lengths[1]
            return xc, yc, a, b, model.theta
    else:
        model = measure.EllipseModel()
        if model.estimate(contour):
            yc, xc, b, a, theta = model.params
            return xc, yc, a, b, theta
    return None


def get_absolute_level(data: np.ndarray, level: float) -> float:
    """Get absolute level from relative level

    Args:
        data: Input data
        level: Relative level (0.0 to 1.0)

    Returns:
        Absolute level

    Raises:
        ValueError: If level is not a float between 0.0 and 1.0
    """
    if not isinstance(level, (int, float)) or level < 0.0 or level > 1.0:
        raise ValueError("Level must be a number between 0.0 and 1.0")
    return np.nanmin(data) + level * (np.nanmax(data) - np.nanmin(data))


def distance_matrix(coords: list) -> np.ndarray:
    """Return distance matrix from coords

    Args:
        coords: List of coordinates

    Returns:
        Distance matrix
    """
    return np.triu(spt.distance.cdist(coords, coords, "euclidean"))


@check_2d_array
def binning(
    data: np.ndarray,
    sx: int,
    sy: int,
    operation: BinningOperation | str,
    dtype=None,
) -> np.ndarray:
    """Perform image pixel binning

    Args:
        data: Input data
        sx: Binning size along x (number of pixels to bin together)
        sy: Binning size along y (number of pixels to bin together)
        operation: Binning operation
        dtype: Output data type (default: None, i.e. same as input)

    Returns:
        Binned data
    """
    # Convert enum to string value if needed
    if isinstance(operation, BinningOperation):
        operation = operation.value

    ny, nx = data.shape
    shape = (ny // sy, sy, nx // sx, sx)
    try:
        bdata = data[: ny - ny % sy, : nx - nx % sx].reshape(shape)
    except ValueError as err:
        raise ValueError("Binning is not a multiple of image dimensions") from err
    if operation == "sum":
        bdata = np.array(bdata, dtype=float).sum(axis=(-1, 1))
    elif operation == "average":
        bdata = bdata.mean(axis=(-1, 1))
    elif operation == "median":
        bdata = ma.median(bdata, axis=(-1, 1))
    elif operation == "min":
        bdata = bdata.min(axis=(-1, 1))
    elif operation == "max":
        bdata = bdata.max(axis=(-1, 1))
    else:
        valid = ", ".join(op.value for op in BinningOperation)
        raise ValueError(f"Invalid operation {operation} (valid values: {valid})")
    return np.array(bdata, dtype=data.dtype if dtype is None else np.dtype(dtype))


@check_2d_array(non_constant=True)
def scale_data_to_min_max(
    data: np.ndarray, zmin: float | int, zmax: float | int
) -> np.ndarray:
    """Scale array `data` to fit [zmin, zmax] dynamic range

    Args:
        data: Input data
        zmin: Minimum value of output data
        zmax: Maximum value of output data

    Returns:
        Scaled data
    """
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    if dmin == zmin and dmax == zmax:
        return data
    fdata = np.array(data, dtype=float)
    fdata -= dmin
    fdata *= float(zmax - zmin) / (dmax - dmin)
    fdata += float(zmin)
    return np.array(fdata, data.dtype)


@check_2d_array
def zero_padding(
    data: np.ndarray,
    rows: int = 0,
    cols: int = 0,
    position: Literal["bottom-right", "around"] = "bottom-right",
) -> np.ndarray:
    """
    Zero-pad a 2D image by adding rows and/or columns.

    Args:
        data: 2D input image (grayscale)
        rows: Number of rows to add in total (default: 0)
        cols: Number of columns to add in total (default: 0)
        position: Padding placement strategy:
            - "bottom-right": all padding is added to the bottom and right
            - "around": padding is split equally on top/bottom and left/right

    Returns:
        The padded 2D image as a NumPy array.

    Raises:
        ValueError: If the input is not a 2D array or if padding values are negative.
    """
    if rows < 0 or cols < 0:
        raise ValueError("Padding values must be non-negative")

    if position == "bottom-right":
        pad_width = ((0, rows), (0, cols))
    elif position == "around":
        pad_width = (
            (rows // 2, rows - rows // 2),
            (cols // 2, cols - cols // 2),
        )
    else:
        raise ValueError(f"Invalid position: {position}")

    return np.pad(data, pad_width, mode="constant", constant_values=0)
