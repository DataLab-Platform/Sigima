# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image profiles in physical coordinates: anisotropic pixels, non-uniform
coordinates, refusals and renamed parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image as sipi


def _image(data: np.ndarray) -> sigima.objects.ImageObj:
    obj = sigima.objects.create_image("i", data)
    obj.xunit = obj.yunit = "mm"
    return obj


def test_radial_profile_anisotropic_pixels() -> None:
    """With dx != dy, distances are physical: a disk gives a step profile."""
    xc, yc, radius = 0.0, 0.0, 3.0
    xcoords = np.arange(-40, 41) * 0.1  # dx = 0.1
    ycoords = np.arange(-20, 21) * 0.2  # dy = 0.2
    xx, yy = np.meshgrid(xcoords, ycoords)
    disk = (np.hypot(xx - xc, yy - yc) <= radius).astype(float)
    obj = _image(disk)
    obj.set_uniform_coords(0.1, 0.2, xcoords[0], ycoords[0])
    param = sigima.params.RadialProfileParam.create(center="user", xc=xc, yc=yc)
    sig = sipi.radial_profile(obj, param)
    assert np.allclose(np.diff(sig.x), 0.1)  # Step: smallest pixel spacing
    half = sig.x >= 0
    assert np.all(sig.y[half][sig.x[half] < radius - 0.1] == 1.0)
    assert np.all(sig.y[half][sig.x[half] > radius + 0.1] == 0.0)


def test_profiles_non_uniform_coordinates() -> None:
    """Profiles of images with non-uniform coordinates use the pixel centers."""
    data = np.arange(20.0).reshape(4, 5)
    obj = _image(data)
    xcoords = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    ycoords = np.array([0.0, 0.5, 1.5, 3.0])
    obj.set_coords(xcoords, ycoords)
    # Line profile: nearest row center to y = 1.2 is row 2 (1.5)
    sig = sipi.line_profile(obj, sigima.params.LineProfileParam.create(y=1.2))
    assert np.array_equal(sig.x, xcoords) and np.array_equal(sig.y, data[2])
    # Vertical: nearest column center to x = 5 is column 3 (6.0)
    sig = sipi.line_profile(
        obj, sigima.params.LineProfileParam.create(direction="vertical", x=5.0)
    )
    assert np.array_equal(sig.x, ycoords) and np.array_equal(sig.y, data[:, 3])
    # Average profile: centers in [1, 6] x [0.5, 3]
    param = sigima.params.AverageProfileParam.create(x1=1, y1=0.5, x2=6, y2=3)
    sig = sipi.average_profile(obj, param)
    assert np.array_equal(sig.x, xcoords[1:4])
    assert np.allclose(sig.y, data[1:4, 1:4].mean(axis=0))
    # Radial profile: steps of the smallest spacing (0.5); distances holding no
    # pixel are left out (no NaN)
    param = sigima.params.RadialProfileParam.create(center="user", xc=0, yc=0)
    sig = sipi.radial_profile(obj, param)
    assert np.all(np.isfinite(sig.y))
    assert np.allclose(sig.x / 0.5, np.rint(sig.x / 0.5))
    assert sig.x[len(sig.x) // 2] == 0.0 and np.allclose(sig.x, -sig.x[::-1])


def test_radial_profile_refuses_different_units() -> None:
    """Distances are undefined when the X and Y units differ."""
    obj = _image(np.ones((5, 5)))
    obj.yunit = "s"
    with pytest.raises(ValueError, match="units"):
        sipi.radial_profile(obj, sigima.params.RadialProfileParam.create())


@pytest.mark.parametrize(
    "paramclass, kwargs",
    [
        (sigima.params.LineProfileParam, {"row": 1}),
        (sigima.params.SegmentProfileParam, {"row1": 1}),
        (sigima.params.AverageProfileParam, {"col2": -1}),
        (sigima.params.RadialProfileParam, {"x0": 1.0}),
    ],
)
def test_former_parameter_names_are_refused(paramclass, kwargs) -> None:
    """Parameters in pixel indices were renamed: former names fail explicitly."""
    with pytest.raises((AttributeError, TypeError)):
        paramclass.create(**kwargs)


def test_default_parameters_from_image() -> None:
    """Defaults set from an image lie inside it (center, middle row)."""
    obj = _image(np.arange(30.0).reshape(5, 6))
    obj.set_uniform_coords(2.0, 3.0, 100.0, -50.0)
    line = sigima.params.LineProfileParam()
    line.update_from_obj(obj)
    assert (line.x, line.y) == (obj.xc, obj.yc)
    sipi.line_profile(obj, line)
    segment = sigima.params.SegmentProfileParam()
    segment.update_from_obj(obj)
    sig = sipi.segment_profile(obj, segment)
    assert np.array_equal(sig.y, obj.data[2])
    radial = sigima.params.RadialProfileParam()
    radial.update_from_obj(obj)
    assert (radial.xc, radial.yc) == (obj.xc, obj.yc)
