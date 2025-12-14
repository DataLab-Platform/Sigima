# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image centroid computation test

Comparing different algorithms for centroid calculation:

- SciPy (measurements.center_of_mass)
- OpenCV (moments)
- Method based on moments
- Method based on Fourier (Sigima's algorithm)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy import ma
from skimage import measure

import sigima.objects
import sigima.proc.image
import sigima.tools.image
from sigima.config import _
from sigima.tests import guiutils
from sigima.tests.data import (
    create_noisy_gaussian_image,
    get_laser_spot_data,
    get_test_image,
)
from sigima.tests.env import execenv
from sigima.tests.helpers import check_scalar_result


def get_centroid_from_moments(data: np.ndarray) -> tuple[int, int]:
    """Computing centroid from image moments

    Args:
        data: 2D array of image data

    Returns:
        Tuple with centroid coordinates (y, x)
    """
    y, x = np.ogrid[: data.shape[0], : data.shape[1]]
    imx, imy = data.sum(axis=0)[None, :], data.sum(axis=1)[:, None]
    m00 = np.array(data, dtype=float).sum() or 1.0
    m10 = (np.array(imx, dtype=float) * x).sum() / m00
    m01 = (np.array(imy, dtype=float) * y).sum() / m00
    return int(m01), int(m10)


def get_centroid_with_cv2(data: np.ndarray) -> tuple[int, int]:
    """Compute centroid from moments with OpenCV

    Args:
        data: 2D array of image data

    Returns:
        Tuple with centroid coordinates (y, x)
    """
    import cv2  # pylint: disable=import-outside-toplevel

    m = cv2.moments(data)
    col = int(m["m10"] / m["m00"])
    row = int(m["m01"] / m["m00"])
    return row, col


def __compare_centroid_funcs(data: np.ndarray) -> None:
    """Compare different centroid computation methods

    Args:
        data: 2D array of image data
    """
    # pylint: disable=import-outside-toplevel
    from sigima.tests import vistools

    items = []
    items += [vistools.create_image(data, interpolation="nearest")]
    # Computing centroid coordinates
    for name, func in (
        # ("SciPy", spi.center_of_mass),
        # ("OpenCV", get_centroid_with_cv2),
        ("scikit-image", measure.centroid),
        # ("Moments", get_centroid_from_moments),
        ("Fourier", sigima.tools.image.get_centroid_fourier),
        ("Auto", sigima.tools.image.get_centroid_auto),
        ("Projected Profile Median", sigima.tools.image.get_projected_profile_centroid),
        (
            "Projected Profile Barycenter",
            lambda d: sigima.tools.image.get_projected_profile_centroid(
                d, method="barycenter"
            ),
        ),
    ):
        try:
            t0 = time.time()
            y, x = func(data)
            dt = time.time() - t0
            label = "  " + f"{_('Centroid')}[{name}] (x=%s, y=%s)"
            execenv.print(label % (x, y))
            cursor = vistools.create_cursor("x", (x, y), label=label)
            items.append(cursor)
            execenv.print(f"    Calculation time: {int(dt * 1e3):d} ms")
        except ImportError:
            execenv.print(f"    Unable to compute {name}: missing module")
    vistools.view_image_items(items)


@pytest.mark.gui
def test_image_centroid_interactive() -> None:
    """Interactive test for image centroid computation

    This test will display the centroid of laser spot data using different methods.
    It will also print the centroid coordinates and computation time for each method.
    """
    with guiutils.lazy_qt_app_context(force=True):
        centroid_test_data = get_test_image("centroid_test.npy").data
        for data in get_laser_spot_data() + [centroid_test_data]:
            execenv.print(f"Data[dtype={data.dtype},shape={data.shape}]")
            # Testing with masked arrays
            __compare_centroid_funcs(data.view(ma.MaskedArray))


def __check_centroid(
    image: sigima.objects.ImageObj, expected_x: float, expected_y: float, debug_str: str
) -> None:
    """Check centroid computation

    Args:
        image: Image object to compute centroid from
        expected_x: Expected x coordinate of the centroid
        expected_y: Expected y coordinate of the centroid
        debug_str: Debug string for logging
    """
    geometry = sigima.proc.image.centroid(image)
    x, y = geometry.coords[0]
    check_scalar_result(f"Centroid X [{debug_str}]", x, expected_x, atol=1.0)
    check_scalar_result(f"Centroid Y [{debug_str}]", y, expected_y, atol=1.0)


@pytest.mark.validation
def test_image_centroid() -> None:
    """Test centroid computation"""
    param = sigima.objects.NewImageParam.create(height=500, width=500)
    image = create_noisy_gaussian_image(param, center=(-2.0, 3.0), add_annotations=True)
    circle_roi = sigima.objects.create_image_roi("circle", [200, 325, 10], indices=True)
    for roi, x0, y0 in (
        (None, 0.0, 0.0),
        (None, 100.0, 100.0),
        (circle_roi, 0.0, 0.0),
        (circle_roi, 100.0, 100.0),  # Test for regression like #106
    ):
        image.roi = roi
        image.set_uniform_coords(image.dx, image.dy, x0, y0)
        debug_str = f"{roi}, x0: {x0}, y0: {y0}"
        __check_centroid(image, 200.0 + x0, 325.0 + y0, debug_str)


if __name__ == "__main__":
    test_image_centroid_interactive()
    test_image_centroid()
