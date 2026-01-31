# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Enclosing circle test

Testing enclsoing circle function on various test images.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import pytest

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


if __name__ == "__main__":
    test_enclosing_circle_interactive()
    test_image_enclosing_circle()
