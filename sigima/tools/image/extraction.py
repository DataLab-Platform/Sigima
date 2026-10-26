"""
Signal/Image Data Extraction
-----------------------------

This module contains functions for extracting information and features from image data:

- Radial profile extraction
- Statistical analysis and feature extraction functions
- Data analysis utilities
"""

from __future__ import annotations

import numpy as np

from sigima.tools.checks import check_2d_array


@check_2d_array
def get_radial_profile(
    data: np.ndarray,
    center: tuple[float, float],
    *,
    xcoords: np.ndarray | None = None,
    ycoords: np.ndarray | None = None,
    step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radial profile of image data

    The pixels are binned by their distance to the center (distance divided by
    *step*, rounded down), and the values of each bin are averaged. Only the bins
    holding pixels are allocated, so that the memory used never exceeds the size
    of the image, whatever the step.

    Args:
        data: Input data (2D array)
        center: Coordinates of the center of the profile (x, y), in the same units
         as *xcoords* and *ycoords* (pixel indices by default)
        xcoords: Coordinates of the pixel centers along X (columns); by default,
         the column indices
        ycoords: Coordinates of the pixel centers along Y (rows); by default, the
         row indices
        step: Width of the distance bins, in the same units

    Returns:
        Radial profile (X, Y) where X is the distance from the center (1D array,
        mirrored: from minus to plus the largest distance, in steps of *step*;
        distances holding no pixel are left out) and Y is the average value of
        pixels at this distance (1D array)

    Raises:
        ValueError: if *step* is not positive and finite, or if the center or the
         pixel coordinates are not finite
    """
    if not (np.isfinite(step) and step > 0):
        raise ValueError(f"The bin width must be positive and finite (got {step})")
    if xcoords is None:
        xcoords = np.arange(data.shape[1], dtype=float)
    if ycoords is None:
        ycoords = np.arange(data.shape[0], dtype=float)
    x, y = np.meshgrid(np.asarray(xcoords, float), np.asarray(ycoords, float))
    x0, y0 = center
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)  # Calculate distance to the center
    if not np.all(np.isfinite(r)):
        raise ValueError("The center and the pixel coordinates must be finite")
    # Bin of each pixel, robust to rounding noise when distances are multiples of
    # the step: only the bins holding pixels are kept (sorted), each pixel being
    # given the position of its bin among them
    bins, index = np.unique(np.floor(r / step + 1e-9).ravel(), return_inverse=True)

    # Average over the same distance (distances holding no pixel, possible with
    # anisotropic pixels or non-uniform coordinates, are left out of the profile)
    tbin = np.bincount(index, data.ravel())  # Sum of pixel values at each distance
    nr = np.bincount(index)  # Number of pixels at each distance

    yhalf = tbin / nr  # this is the half radial profile
    xhalf = bins * step
    # Mirror it to get the full radial profile (the first element is the center)
    skip = 1 if bins[0] == 0 else 0
    yprofile = np.concatenate((yhalf[::-1], yhalf[skip:]))
    # The x axis is the distance from the center (0 is the center)
    xprofile = np.concatenate((-xhalf[::-1], xhalf[skip:]))

    return xprofile, yprofile
