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
_SKIMAGE_VERSION = Version(__version__)
_USE_NEW_SHAPE_API = _SKIMAGE_VERSION >= Version("0.26.0")


def fit_circle_model(contour: np.ndarray) -> tuple[float, float, float] | None:
    """Fit circle model to contour with version compatibility.

    Args:
        contour: Contour coordinates array (N, 2), in scikit-image's ``(row, col)``
         order -- i.e. ``(y, x)`` -- as returned by
         :func:`skimage.measure.find_contours`

    Returns:
        Tuple (xc, yc, radius) in ``(x, y)`` order, or None if fitting fails
    """
    # pylint: disable=no-member
    if _USE_NEW_SHAPE_API:
        model = measure.CircleModel.from_estimate(contour)
        if model:
            # model.center is (row, col) = (y, x), swap to (x, y)
            return model.center[1], model.center[0], model.radius
    else:
        model = measure.CircleModel()
        if model.estimate(contour):
            yc, xc, radius = model.params
            return xc, yc, radius
    return None


def _estimate_ellipse_params(
    contour: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    """Direct least-squares ellipse fit (Halir & Flusser) -- fallback only.

    This is a faithful, self-contained copy of scikit-image's
    ``EllipseModel`` estimation algorithm. It exists **solely** as a fallback
    for :func:`fit_ellipse_model` and is *not* meant to replace the library:
    it is invoked only when the installed scikit-image raises while fitting an
    otherwise valid contour.

    The only deliberate deviation from the upstream code is that the
    eigen-decomposition is forced back to its real part. This works around an
    upstream bug where ``numpy.linalg.eig`` returns complex eigenvalues and
    eigenvectors for noise-free contours, making ``EllipseModel._estimate``
    crash on ``phi %= np.pi`` with a ``TypeError`` (scikit-image issue #7013).
    The bug only affects scikit-image 0.26.x combined with recent NumPy; it is
    fixed upstream in 0.27 (PR scikit-image/scikit-image#8054), which applies
    exactly the same ``.real`` correction. Once Sigima's minimum scikit-image
    is >= 0.27 this helper and the fallback in :func:`fit_ellipse_model` can be
    removed.

    Args:
        contour: Contour coordinates array (N, 2), columns interpreted as
         ``(x, y)`` (same convention as scikit-image's ``EllipseModel``).

    Returns:
        Tuple ``(x0, y0, width, height, phi)`` (centre, axis lengths and
        rotation angle), or ``None`` if the fit fails.
    """
    data = np.asarray(contour, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 5:
        return None

    # Normalize value range to avoid misfitting due to numeric errors if the
    # relative distances are small compared to absolute distances.
    origin = data.mean(axis=0)
    data = data - origin
    scale = data.std()
    if scale < np.finfo(float).tiny:
        return None
    data = data / scale

    x = data[:, 0]
    y = data[:, 1]

    # Quadratic and linear parts of the design matrix [eqns. 15, 16] from
    # Halir & Flusser.
    d1 = np.vstack([x**2, x * y, y**2]).T
    d2 = np.vstack([x, y, np.ones_like(x)]).T
    s1 = d1.T @ d1
    s2 = d1.T @ d2
    s3 = d2.T @ d2
    c1 = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
    try:
        reduced = np.linalg.inv(c1) @ (s1 - s2 @ np.linalg.inv(s3) @ s2.T)
    except np.linalg.LinAlgError:
        return None

    eig_vals, eig_vecs = np.linalg.eig(reduced)
    # Work around scikit-image #7013: numpy may return complex eigenvalues and
    # eigenvectors for this real, non-symmetric matrix; keep the real part.
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real

    # Eigenvector must meet constraint 4ac - b^2 > 0 to be valid.
    cond = 4 * np.multiply(eig_vecs[0, :], eig_vecs[2, :]) - np.power(eig_vecs[1, :], 2)
    a1 = eig_vecs[:, (cond > 0)]
    if 0 in a1.shape or len(a1.ravel()) != 3:
        return None
    a, b, c = a1.ravel()
    a2 = -np.linalg.inv(s3) @ s2.T @ a1
    d, f, g = a2.ravel()

    # Coefficients of an ellipse in general form:
    # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0.
    b /= 2.0
    d /= 2.0
    f /= 2.0

    denom = b**2.0 - a * c
    if denom == 0:
        return None
    x0 = (c * d - b * f) / denom
    y0 = (a * f - b * d) / denom

    numerator = a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g
    term = np.sqrt((a - c) ** 2 + 4 * b**2)
    denominator1 = denom * (term - (a + c))
    denominator2 = denom * (-term - (a + c))
    width = np.sqrt(2 * numerator / denominator1)
    height = np.sqrt(2 * numerator / denominator2)

    phi = 0.5 * np.arctan((2.0 * b) / (a - c))
    if a > c:
        phi += 0.5 * np.pi

    # Sometimes small fluctuations in data cause height and width to swap.
    if width < height:
        width, height = height, width
        phi += np.pi / 2
    phi %= np.pi

    params = np.nan_to_num([x0, y0, width, height, phi]).real
    params[:4] *= scale
    params[:2] += origin
    return tuple(params)


def fit_ellipse_model(
    contour: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    """Fit ellipse model to contour with version compatibility.

    The fit is delegated to scikit-image's ``EllipseModel`` (the source of
    truth across all supported versions). As a defensive measure, if the
    library raises a ``TypeError`` -- which scikit-image 0.26.x does on valid
    contours because of upstream bug #7013, see
    :func:`_estimate_ellipse_params` -- we fall back to a local, equivalent
    implementation that is immune to that bug. The fallback is otherwise never
    exercised, so working environments keep using the library unchanged.

    Args:
        contour: Contour coordinates array (N, 2), in scikit-image's ``(row, col)``
         order -- i.e. ``(y, x)`` -- as returned by
         :func:`skimage.measure.find_contours`

    Returns:
        Tuple (xc, yc, a, b, theta) in ``(x, y)`` order, or None if fitting fails,
        where a and b are semi-major and semi-minor axes
    """
    # pylint: disable=no-member
    try:
        if _USE_NEW_SHAPE_API:
            model = measure.EllipseModel.from_estimate(contour)
            if not model:
                return None
            # model.center is (row, col) = (y, x), swap to (x, y)
            # model.axis_lengths is (semi_row, semi_col), swap to (semi_x, semi_y)
            xc, yc = model.center[1], model.center[0]
            a, b = model.axis_lengths[1], model.axis_lengths[0]
            return xc, yc, a, b, model.theta
        model = measure.EllipseModel()
        if not model.estimate(contour):
            return None
        yc, xc, b, a, theta = model.params
        return xc, yc, a, b, theta
    except TypeError:
        # scikit-image issue #7013: EllipseModel crashes on a valid contour
        # because numpy.linalg.eig returned complex eigenvectors. Retry with
        # the bug-free local implementation (see _estimate_ellipse_params).
        params = _estimate_ellipse_params(contour)
        if params is None:
            return None
        x0, y0, width, height, phi = params
        # _estimate_ellipse_params follows scikit-image's (x, y) convention,
        # i.e. centre = (x0, y0) and axis_lengths = (width, height). Apply the
        # same (row, col) -> (x, y) swap as the library branches above.
        return y0, x0, height, width, phi


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
