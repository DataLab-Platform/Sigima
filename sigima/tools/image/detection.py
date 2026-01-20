# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Detection algorithms module
---------------------------

This module provides various object detection algorithms for image analysis.

Features include:

- Blob detection using multiple algorithms (DoG, DoH, LoG, OpenCV)
- Peak detection with configurable thresholds and neighborhood sizes
- Hough transform-based circle detection
- Contour shape fitting (circles, ellipses, polygons)
- Utility functions for coordinate processing

These tools support automated feature extraction and object identification
in images for scientific and technical applications.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as spi
from numpy import ma
from skimage import exposure, feature, measure, transform

from sigima.enums import ContourShape
from sigima.tools.checks import check_2d_array
from sigima.tools.image.preprocessing import (
    distance_matrix,
    fit_circle_model,
    fit_ellipse_model,
    get_absolute_level,
)


@check_2d_array(non_constant=True)
def get_2d_peaks_coords(
    data: np.ndarray, size: int | None = None, level: float = 0.5
) -> np.ndarray:
    """Detect peaks in image data, return coordinates.

    If neighborhoods size is None, default value is the highest value
    between 50 pixels and the 1/40th of the smallest image dimension.

    Detection threshold level is relative to difference
    between data maximum and minimum values.

    Args:
        data: Input data
        size: Neighborhood size (default: None)
        level: Relative level (default: 0.5)

    Returns:
        Coordinates of peaks
    """
    if size is None:
        size = max(min(data.shape) // 40, 50)
    data_max = spi.maximum_filter(data, size)
    data_min = spi.minimum_filter(data, size)
    data_diff = data_max - data_min
    diff = (data_max - data_min) > get_absolute_level(data_diff, level)
    maxima = data == data_max
    maxima[diff == 0] = 0
    labeled, _num_objects = spi.label(maxima)
    slices = spi.find_objects(labeled)
    coords = []
    for dy, dx in slices:
        x_center = int(0.5 * (dx.start + dx.stop - 1))
        y_center = int(0.5 * (dy.start + dy.stop - 1))
        coords.append((x_center, y_center))
    if len(coords) > 1:
        # Eventually removing duplicates
        dist = distance_matrix(coords)
        for index in reversed(np.unique(np.where((dist < size) & (dist > 0))[1])):
            coords.pop(index)
    return np.array(coords)


def get_contour_shapes(
    data: np.ndarray | ma.MaskedArray,
    shape: ContourShape = ContourShape.ELLIPSE,
    level: float = 0.5,
) -> np.ndarray:
    """Find iso-valued contours in a 2D array, above relative level (.5 means FWHM).

    Args:
        data: Input data
        shape: Shape to fit. Default is ELLIPSE
        level: Relative level (default: 0.5)

    Returns:
        Coordinates of shapes fitted to contours
    """
    # pylint: disable=too-many-locals
    contours = measure.find_contours(data, level=get_absolute_level(data, level))
    coords = []
    for contour in contours:
        # `contour` is a (N, 2) array (rows, cols): we need to check if all those
        # coordinates are masked: if so, we skip this contour
        if isinstance(data, ma.MaskedArray) and np.all(
            data.mask[contour[:, 0].astype(int), contour[:, 1].astype(int)]
        ):
            continue
        if shape == ContourShape.CIRCLE:
            result = fit_circle_model(contour)
            if result:
                xc, yc, r = result
                if r > 1.0:
                    coords.append([xc, yc, r])
        elif shape == ContourShape.ELLIPSE:
            result = fit_ellipse_model(contour)
            if result:
                xc, yc, a, b, theta = result
                if a > 1.0 and b > 1.0:
                    coords.append([xc, yc, a, b, theta])
        elif shape == ContourShape.POLYGON:
            # `contour` is a (N, 2) array (rows, cols): we need to convert it
            # to a list of x, y coordinates flattened in a single list
            coords.append(contour[:, ::-1].flatten())
        else:
            raise NotImplementedError(f"Invalid contour shape {shape}")
    if shape == ContourShape.POLYGON:
        # `coords` is a list of arrays of shape (N, 2) where N is the number of points
        # that can vary from one array to another, so we need to padd with NaNs each
        # array to get a regular array:
        max_len = max(coord.shape[0] for coord in coords)
        arr = np.full((len(coords), max_len), np.nan)
        for i_row, coord in enumerate(coords):
            arr[i_row, : coord.shape[0]] = coord
        return arr
    return np.array(coords)


@check_2d_array(non_constant=True)
def get_hough_circle_peaks(
    data: np.ndarray,
    min_radius: float | None = None,
    max_radius: float | None = None,
    nb_radius: int | None = None,
    min_distance: int = 1,
) -> np.ndarray:
    """Detect peaks in image from circle Hough transform, return circle coordinates.

    Args:
        data: Input data
        min_radius: Minimum radius (default: None)
        max_radius: Maximum radius (default: None)
        nb_radius: Number of radii (default: None)
        min_distance: Minimum distance between circles (default: 1)

    Returns:
        Coordinates of circles
    """
    assert min_radius is not None and max_radius is not None and max_radius > min_radius
    if nb_radius is None:
        nb_radius = max_radius - min_radius + 1
    hough_radii = np.arange(
        min_radius, max_radius + 1, (max_radius - min_radius + 1) // nb_radius
    )
    hough_res = transform.hough_circle(data, hough_radii)
    _accums, cx, cy, radii = transform.hough_circle_peaks(
        hough_res, hough_radii, min_xdistance=min_distance, min_ydistance=min_distance
    )
    return np.vstack([cx, cy, radii]).T


# MARK: Blob detection -----------------------------------------------------------------


def __blobs_to_coords(blobs: np.ndarray) -> np.ndarray:
    """Convert blobs to coordinates

    Args:
        blobs: Blobs

    Returns:
        Coordinates
    """
    cy, cx, radii = blobs.T
    coords = np.vstack([cx, cy, radii]).T
    return coords


@check_2d_array(non_constant=True)
def find_blobs_dog(
    data: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 30,
    overlap: float = 0.5,
    threshold_rel: float = 0.2,
    exclude_border: bool = True,
) -> np.ndarray:
    """
    Finds blobs in the given grayscale image using the Difference of Gaussians
    (DoG) method.

    Args:
        data: The grayscale input image.
        min_sigma: The minimum blob radius in pixels.
        max_sigma: The maximum blob radius in pixels.
        overlap: The minimum overlap between two blobs in pixels. For instance, if two
         blobs are detected with radii of 10 and 12 respectively, and the ``overlap``
         is set to 0.5, then the area of the smaller blob will be ignored and only the
         area of the larger blob will be returned.
        threshold_rel: The absolute lower bound for scale space maxima. Local maxima
         smaller than ``threshold_rel`` are ignored. Reduce this to detect blobs with
         less intensities.
        exclude_border: If ``True``, exclude blobs from detection if they are too
         close to the border of the image. Border size is ``min_sigma``.

    Returns:
        Coordinates of blobs
    """
    # Use scikit-image's Difference of Gaussians (DoG) method
    blobs = feature.blob_dog(
        data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        overlap=overlap,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
    )
    return __blobs_to_coords(blobs)


@check_2d_array(non_constant=True)
def find_blobs_doh(
    data: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 30,
    overlap: float = 0.5,
    log_scale: bool = False,
    threshold_rel: float = 0.2,
) -> np.ndarray:
    """
    Finds blobs in the given grayscale image using the Determinant of Hessian
    (DoH) method.

    Args:
        data: The grayscale input image.
        min_sigma: The minimum blob radius in pixels.
        max_sigma: The maximum blob radius in pixels.
        overlap: The minimum overlap between two blobs in pixels. For instance, if two
         blobs are detected with radii of 10 and 12 respectively, and the ``overlap``
         is set to 0.5, then the area of the smaller blob will be ignored and only the
         area of the larger blob will be returned.
        log_scale: If ``True``, the radius of each blob is returned as ``sqrt(sigma)``
         for each detected blob.
        threshold_rel: The absolute lower bound for scale space maxima. Local maxima
         smaller than ``threshold_rel`` are ignored. Reduce this to detect blobs with
         less intensities.

    Returns:
        Coordinates of blobs
    """
    # Use scikit-image's Determinant of Hessian (DoH) method to detect blobs
    blobs = feature.blob_doh(
        data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=int(max_sigma - min_sigma + 1),
        threshold=None,
        threshold_rel=threshold_rel,
        overlap=overlap,
        log_scale=log_scale,
    )
    return __blobs_to_coords(blobs)


@check_2d_array(non_constant=True)
def find_blobs_log(
    data: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 30,
    overlap: float = 0.5,
    log_scale: bool = False,
    threshold_rel: float = 0.2,
    exclude_border: bool = True,
) -> np.ndarray:
    """Finds blobs in the given grayscale image using the Laplacian of Gaussian
    (LoG) method.

    Args:
        data: The grayscale input image.
        min_sigma: The minimum blob radius in pixels.
        max_sigma: The maximum blob radius in pixels.
        overlap: The minimum overlap between two blobs in pixels. For instance, if
         two blobs are detected with radii of 10 and 12 respectively, and the
         ``overlap`` is set to 0.5, then the area of the smaller blob will be ignored
         and only the area of the larger blob will be returned.
        log_scale: If ``True``, the radius of each blob is returned as ``sqrt(sigma)``
         for each detected blob.
        threshold_rel: The absolute lower bound for scale space maxima. Local maxima
         smaller than ``threshold_rel`` are ignored. Reduce this to detect blobs with
         less intensities.
        exclude_border: If ``True``, exclude blobs from detection if they are too
         close to the border of the image. Border size is ``min_sigma``.

    Returns:
        Coordinates of blobs
    """
    # Use scikit-image's Laplacian of Gaussian (LoG) method to detect blobs
    blobs = feature.blob_log(
        data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=int(max_sigma - min_sigma + 1),
        threshold=None,
        threshold_rel=threshold_rel,
        overlap=overlap,
        log_scale=log_scale,
        exclude_border=exclude_border,
    )
    return __blobs_to_coords(blobs)


def remove_overlapping_disks(coords: np.ndarray) -> np.ndarray:
    """Remove overlapping disks among coordinates

    Args:
        coords: The coordinates of the disks

    Returns:
        The coordinates of the disks with overlapping disks removed
    """
    # Get the radii of each disk from the coordinates
    radii = coords[:, 2]
    # Calculate the distance between the center of each pair of disks
    dist = np.sqrt(np.sum((coords[:, None, :2] - coords[:, :2]) ** 2, axis=-1))
    # Create a boolean mask where the distance between the centers
    # is less than the sum of the radii
    mask = dist < (radii[:, None] + radii)
    # Find the indices of overlapping disks
    overlapping_indices = np.argwhere(mask)
    # Remove the smaller disk from each overlapping pair
    for i, j in overlapping_indices:
        if i != j:
            if radii[i] < radii[j]:
                coords[i] = [np.nan, np.nan, np.nan]
            else:
                coords[j] = [np.nan, np.nan, np.nan]
    # Remove rows with NaN values
    coords = coords[~np.isnan(coords).any(axis=1)]
    return coords


# pylint: disable=too-many-positional-arguments
@check_2d_array(non_constant=True)
def find_blobs_opencv(
    data: np.ndarray,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
    min_repeatability: int | None = None,
    min_dist_between_blobs: float | None = None,
    filter_by_color: bool | None = None,
    blob_color: int | None = None,
    filter_by_area: bool | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    filter_by_circularity: bool | None = None,
    min_circularity: float | None = None,
    max_circularity: float | None = None,
    filter_by_inertia: bool | None = None,
    min_inertia_ratio: float | None = None,
    max_inertia_ratio: float | None = None,
    filter_by_convexity: bool | None = None,
    min_convexity: float | None = None,
    max_convexity: float | None = None,
) -> np.ndarray:
    """
    Finds blobs in the given grayscale image using OpenCV's SimpleBlobDetector.

    Args:
        data: The grayscale input image.
        min_threshold: The minimum blob intensity.
        max_threshold: The maximum blob intensity.
        min_repeatability: The minimum number of times a blob is detected
         before it is reported.
        min_dist_between_blobs: The minimum distance between blobs.
        filter_by_color: If ``True``, blobs are filtered by color.
        blob_color: The color of the blobs to filter by.
        filter_by_area: If ``True``, blobs are filtered by area.
        min_area: The minimum blob area.
        max_area: The maximum blob area.
        filter_by_circularity: If ``True``, blobs are filtered by circularity.
        min_circularity: The minimum blob circularity.
        max_circularity: The maximum blob circularity.
        filter_by_inertia: If ``True``, blobs are filtered by inertia.
        min_inertia_ratio: The minimum blob inertia ratio.
        max_inertia_ratio: The maximum blob inertia ratio.
        filter_by_convexity: If ``True``, blobs are filtered by convexity.
        min_convexity: The minimum blob convexity.
        max_convexity: The maximum blob convexity.

    Returns:
        Coordinates of blobs
    """
    # Note:
    # Importing OpenCV inside the function in order to eventually raise an ImportError
    # when the function is called and OpenCV is not installed. This error will be
    # handled by DataLab and the user will be informed that OpenCV is required to use
    # this function.
    import cv2  # pylint: disable=import-outside-toplevel

    params = cv2.SimpleBlobDetector_Params()
    if min_threshold is not None:
        params.minThreshold = min_threshold
    if max_threshold is not None:
        params.maxThreshold = max_threshold
    if min_repeatability is not None:
        params.minRepeatability = min_repeatability
    if min_dist_between_blobs is not None:
        params.minDistBetweenBlobs = min_dist_between_blobs
    if filter_by_color is not None:
        params.filterByColor = filter_by_color
    if blob_color is not None:
        params.blobColor = blob_color
    if filter_by_area is not None:
        params.filterByArea = filter_by_area
    if min_area is not None:
        params.minArea = min_area
    if max_area is not None:
        params.maxArea = max_area
    if filter_by_circularity is not None:
        params.filterByCircularity = filter_by_circularity
    if min_circularity is not None:
        params.minCircularity = min_circularity
    if max_circularity is not None:
        params.maxCircularity = max_circularity
    if filter_by_inertia is not None:
        params.filterByInertia = filter_by_inertia
    if min_inertia_ratio is not None:
        params.minInertiaRatio = min_inertia_ratio
    if max_inertia_ratio is not None:
        params.maxInertiaRatio = max_inertia_ratio
    if filter_by_convexity is not None:
        params.filterByConvexity = filter_by_convexity
    if min_convexity is not None:
        params.minConvexity = min_convexity
    if max_convexity is not None:
        params.maxConvexity = max_convexity
    detector = cv2.SimpleBlobDetector_create(params)
    image = exposure.rescale_intensity(data, out_range=np.uint8)
    keypoints = detector.detect(image)
    if keypoints:
        coords = cv2.KeyPoint_convert(keypoints)
        radii = 0.5 * np.array([kp.size for kp in keypoints])
        blobs = np.vstack([coords[:, 1], coords[:, 0], radii]).T
        blobs = remove_overlapping_disks(blobs)
    else:
        blobs = np.array([]).reshape((0, 3))
    return __blobs_to_coords(blobs)
