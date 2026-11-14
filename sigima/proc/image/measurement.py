# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Measurement computation module
------------------------------

This module provides tools for extracting quantitative information from images,
such as object centroids, enclosing circles, and region-based statistics.

Main features include:

- Centroid and enclosing circle computation
- Region/property measurements
- Statistical analysis of image regions

These functions are useful for image quantification and morphometric analysis.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

import numpy as np
from numpy import ma

import sigima.tools.image
from sigima.config import _
from sigima.objects import (
    NO_ROI,
    GeometryResult,
    ImageObj,
    KindShape,
    SignalObj,
    TableKind,
    TableResult,
    TableResultBuilder,
)
from sigima.proc.base import new_signal_result
from sigima.proc.contracts import EXACT, ReplayQualification, tolerance
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import compute_geometry_from_obj
from sigima.proc.image.extraction import _check_same_units, _pixel_centers

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "centroid",
    "enclosing_circle",
    "horizontal_projection",
    "stats",
    "vertical_projection",
]


def get_centroid_coords(data: np.ndarray) -> np.ndarray:
    """Return centroid coordinates
    with :py:func:`sigima.tools.image.get_centroid_auto`

    Args:
        data: input data

    Returns:
        Centroid coordinates
    """
    y, x = sigima.tools.image.get_centroid_auto(data)
    return np.array([(x, y)])


@computation_function(
    operation_id="sigima.image.centroid",
    contract_version=2,
    qualification=ReplayQualification(
        contract_version=2,
        comparison=tolerance(
            (
                "Floating-point reductions and linear algebra (NumPy, SciPy, scikit-"
                "image) differ in the last bits across library versions"
            ),
            atol=1e-11,
            rtol=1e-14,
            same_environment="exact",
        ),
    ),
)
def centroid(image: ImageObj) -> GeometryResult | None:
    """Compute centroid
    with :py:func:`sigima.tools.image.get_centroid_fourier`

    Args:
        image: input image

    Returns:
        Centroid coordinates
    """
    return compute_geometry_from_obj(
        "centroid", KindShape.MARKER, image, get_centroid_coords
    )


def get_enclosing_circle_coords(data: np.ndarray) -> np.ndarray:
    """Return the minimum circle enclosing pixel centers above the midrange.

    Args:
        data: input data

    Returns:
        Circle center and radius in pixel coordinates.
    """
    x, y, r = sigima.tools.image.get_enclosing_circle(data)
    return np.array([[x, y, r]])


@computation_function(
    operation_id="sigima.image.enclosing_circle",
    contract_version=3,
    qualification=ReplayQualification(
        contract_version=3,
        comparison=tolerance(
            "Floating-point geometry may differ in the last bits across platforms",
            atol=1e-12,
            rtol=1e-12,
            same_environment="exact",
        ),
    ),
)
def enclosing_circle(image: ImageObj) -> GeometryResult | None:
    """Compute the minimum physical circle enclosing pixel centers above midrange.

    The threshold is ``min + 0.5 * (max - min)`` in each selected region.
    X and Y units must be equal.

    Args:
        image: input image

    Returns:
        Circle center and radius for each ROI.
    """
    _check_same_units(image, "Enclosing circle")
    xcoords = _pixel_centers(image, "x")
    ycoords = _pixel_centers(image, "y")
    circles = []
    roi_indices = []
    for index in image.iterate_roi_indices():
        if index is None:
            data = image.get_data()
        else:
            roi = image.roi.get_single_roi(index)
            data = ma.masked_array(image.data, mask=roi.to_mask(image))
        circles.append(
            sigima.tools.image.get_enclosing_circle(
                data, xcoords=xcoords, ycoords=ycoords
            )
        )
        roi_indices.append(NO_ROI if index is None else index)
    if not circles:
        return None
    return GeometryResult(
        "enclosing_circle",
        KindShape.CIRCLE,
        np.asarray(circles),
        np.asarray(roi_indices),
    )


def __calc_snr_without_warning(data: np.ndarray) -> float:
    """Calculate SNR based on <z>/σ(z), ignoring warnings

    Args:
        data: input data

    Returns:
        Signal-to-noise ratio
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = ma.mean(data) / ma.std(data)
    return snr


@computation_function(
    operation_id="sigima.image.stats",
    contract_version=1,
    qualification=ReplayQualification(
        contract_version=1,
        comparison=tolerance(
            (
                "Floating-point reductions and linear algebra (NumPy, SciPy, scikit-"
                "image) differ in the last bits across library versions"
            ),
            atol=1e-09,
            rtol=1e-14,
            same_environment="exact",
        ),
    ),
)
def stats(obj: ImageObj) -> TableResult:
    """Compute statistics on an image

    Args:
        obj: input image object

    Returns:
        Result properties
    """
    builder = TableResultBuilder(_("Image statistics"), kind=TableKind.STATISTICS)
    builder.add(ma.min, "min")
    builder.add(ma.max, "max")
    builder.add(ma.mean, "mean")
    builder.add(ma.median, "median")
    builder.add(ma.std, "std")
    builder.add(__calc_snr_without_warning, "snr")
    builder.add(ma.ptp, "ptp")
    builder.add(ma.sum, "sum")
    return builder.compute(obj)


@computation_function(
    operation_id="sigima.image.horizontal_projection",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def horizontal_projection(image: ImageObj) -> SignalObj:
    """Compute the sum of pixel intensities along each col. (projection on the x-axis).

    Args:
        image: Input image object.

    Returns:
        Signal object containing the profile.
    """
    dst_signal = new_signal_result(
        image,
        "horizontal_projection",
        units=(image.xunit, image.zunit),
        labels=(image.xlabel, image.zlabel),
    )
    x = np.linspace(image.x0, image.x0 + image.width - image.dx, image.data.shape[1])
    y = image.data.sum(axis=0, dtype=float)
    dst_signal.set_xydata(x, y)
    return dst_signal


@computation_function(
    operation_id="sigima.image.vertical_projection",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def vertical_projection(image: ImageObj) -> SignalObj:
    """Compute the sum of pixel intensities along each row (projection on the y-axis).

    Args:
        image: Input image object.

    Returns:
        Signal object containing the profile.
    """
    dst_signal = new_signal_result(
        image,
        "vertical_projection",
        units=(image.yunit, image.zunit),
        labels=(image.ylabel, image.zlabel),
    )
    x = np.linspace(image.y0, image.y0 + image.height - image.dy, image.data.shape[0])
    y = image.data.sum(axis=1, dtype=float)
    dst_signal.set_xydata(x, y)
    return dst_signal
