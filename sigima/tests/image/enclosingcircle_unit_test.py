# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Enclosing circle test

Testing enclsoing circle function on various test images.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import numpy as np
import pytest

import sigima.objects
import sigima.proc.image
import sigima.tools.image
from sigima.config import _
from sigima.tests import guiutils
from sigima.tests.data import RingParam, create_ring_image, get_laser_spot_data
from sigima.tests.env import execenv
from sigima.tests.helpers import check_scalar_result


def __enclosingcircle_test(data):
    """Enclosing circle test function"""
    # pylint: disable=import-outside-toplevel
    from sigima import viz

    items = []
    items += [viz.create_image(data, interpolation="nearest")]

    # Computing centroid coordinates
    row, col = sigima.tools.image.get_centroid_auto(data)
    label = _("Centroid") + " (%d, %d)"
    execenv.print(label % (row, col))
    cursor = viz.create_cursor("x", (col, row), label=label)
    items.append(cursor)

    x, y, radius = sigima.tools.image.get_enclosing_circle(data)
    circle = viz.create_circle(x, y, radius)
    items.append(circle)
    execenv.print(x, y, radius)
    execenv.print("")

    viz.view_image_items(items)


@pytest.mark.gui
def test_enclosing_circle_interactive():
    """Interactive test for enclosing circle computation."""
    with guiutils.lazy_qt_app_context(force=True):
        for data in get_laser_spot_data():
            __enclosingcircle_test(data)


@pytest.mark.validation
def test_image_enclosing_circle():
    """Test enclosing circle on a ring image."""
    p = RingParam.create(image_size=200, xc=100, yc=100, radius=30, thickness=5)
    # Create a ring image, so that the outer circle radius is radius + thickness:
    obj = create_ring_image(p)
    execenv.print("Testing enclosing circle on a ring image...")
    ex, ey, er = sigima.tools.image.get_enclosing_circle(obj.data)
    geometry = sigima.proc.image.enclosing_circle(obj)
    x, y, r = geometry.coords[0]
    execenv.print(geometry)
    assert ex == x and ey == y and er == r, (
        f"Enclosing circle test failed: expected ({ex}, {ey}, {er}), "
        f"got ({x}, {y}, {r})"
    )
    check_scalar_result("Enclosing circle", er, p.radius + p.thickness, rtol=0.002)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_enclosing_circle_in_physical_coordinates(nonuniform):
    """The physical circle is minimal and encloses all thresholded pixel centers."""
    data = np.zeros((6, 8))
    data[1:5, 2:7] = 1.0
    image = sigima.objects.create_image("rectangle", data)
    image.xunit = image.yunit = "mm"
    if nonuniform:
        image.set_coords(
            np.array([0, 1, 2, 4, 5, 7, 10, 11], dtype=float),
            np.array([0, 1, 3, 5, 10, 11], dtype=float),
        )
        expected = [6.0, 5.5, np.hypot(4.0, 4.5)]
    else:
        image.set_uniform_coords(2.0, 3.0, 5.0, -4.0)
        expected = [13.0, 3.5, np.hypot(4.0, 4.5)]
    result = sigima.proc.image.enclosing_circle(image)
    assert result is not None
    np.testing.assert_allclose(result.coords[0], expected)


def test_enclosing_circle_rejects_different_units():
    """A physical circle cannot combine axes with different units."""
    data = np.zeros((8, 8))
    data[2:6, 2:6] = 1.0
    image = sigima.objects.create_image("rectangle", data)
    image.xunit, image.yunit = "mm", "s"
    with pytest.raises(ValueError, match="units"):
        sigima.proc.image.enclosing_circle(image)


def test_enclosing_circle_uses_midrange_and_strict_threshold():
    """Only pixel centers strictly above the midpoint of the data range count."""
    data = np.full((3, 5), 1.0)
    data[1, 1], data[1, 3] = 3.0, 5.0
    image = sigima.objects.create_image("midrange", data)
    result = sigima.proc.image.enclosing_circle(image)
    assert result is not None
    np.testing.assert_allclose(result.coords[0], [3, 1, 0])


def test_enclosing_circle_three_boundary_points():
    """An acute triangle requires three points, not merely a diameter."""
    data = np.zeros((7, 7))
    data[1, 1] = data[1, 5] = data[5, 3] = 1.0
    image = sigima.objects.create_image("triangle", data)
    result = sigima.proc.image.enclosing_circle(image)
    assert result is not None
    np.testing.assert_allclose(result.coords[0], [3, 2.5, 2.5])


def test_enclosing_circle_overlapping_rois_are_independent():
    """Other ROIs cannot contribute pixels to an individual ROI's circle."""
    data = np.zeros((9, 9))
    data[4, 1] = data[4, 3] = 10.0
    data[2, 4] = 100.0
    image = sigima.objects.create_image("two ROIs", data)
    image.roi = sigima.objects.create_image_roi("circle", [2, 4, 2])
    image.roi.add_roi(sigima.objects.create_image_roi("circle", [4, 4, 2]))
    result = sigima.proc.image.enclosing_circle(image)
    assert result is not None
    np.testing.assert_allclose(result.coords, [[2, 4, 1], [4, 2, 0]])
    np.testing.assert_array_equal(result.roi_indices, [0, 1])


def test_enclosing_circle_inverse_roi():
    """An inverse ROI analyzes its selected pixels outside the shape."""
    data = np.zeros((9, 9))
    data[4, 2] = data[4, 6] = 10.0
    data[4, 4] = 100.0
    image = sigima.objects.create_image("inverse ROI", data)
    image.roi = sigima.objects.create_image_roi("circle", [4, 4, 1], inverse=True)
    result = sigima.proc.image.enclosing_circle(image)
    assert result is not None
    np.testing.assert_allclose(result.coords[0], [4, 4, 2])


if __name__ == "__main__":
    test_enclosing_circle_interactive()
    test_image_enclosing_circle()
