# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image profiles in physical coordinates: anisotropic pixels, non-uniform
coordinates, refusals and renamed parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image as sipi
import sigima.tools.image


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


def _radial_profile_all_bins(data, center, xcoords, ycoords, step):
    """Former implementation of ``get_radial_profile``, allocating every bin from
    the center to the largest distance."""
    x, y = np.meshgrid(xcoords, ycoords)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = np.floor(r / step + 1e-9).astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    bins = np.flatnonzero(nr)
    yhalf = tbin[bins] / nr[bins]
    xhalf = bins * step
    skip = 1 if bins[0] == 0 else 0
    return (
        np.concatenate((-xhalf[::-1], xhalf[skip:])),
        np.concatenate((yhalf[::-1], yhalf[skip:])),
    )


@pytest.mark.parametrize(
    "xcoords, ycoords, center, step",
    [
        (np.arange(53.0), np.arange(37.0), (17.3, 20.6), 1.0),  # uniform
        (np.arange(53.0), np.arange(37.0), (26.0, 18.0), 1.0),  # pixel center
        (np.arange(53) * 0.1, np.arange(37) * 0.2, (2.6, 3.5), 0.1),  # anisotropic
        (np.cumsum(np.linspace(0.1, 2.0, 53)), np.arange(37.0), (30.0, 9.5), 0.1),
    ],
)
def test_radial_profile_same_as_all_bins(xcoords, ycoords, center, step) -> None:
    """Accumulating only the bins holding pixels gives bit-identical profiles."""
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 20.0, (ycoords.size, xcoords.size))
    for values in (data, data.astype(np.uint16), np.ma.masked_less(data, 90.0)):
        x, y = sigima.tools.image.get_radial_profile(
            values, center, xcoords=xcoords, ycoords=ycoords, step=step
        )
        xref, yref = _radial_profile_all_bins(values, center, xcoords, ycoords, step)
        assert np.array_equal(x, xref) and np.array_equal(y, yref)


def test_radial_profile_tiny_spacing() -> None:
    """The memory used does not depend on the pixel spacing (regression: a tiny
    spacing allocated one bin per step up to the largest distance, i.e. several
    GB for a few pixels)."""
    data = np.arange(25.0).reshape(5, 5)
    obj = _image(data)
    obj.set_coords(np.array([0.0, 1e-8, 1.0, 2.0, 3.0]), np.arange(5.0))
    param = sigima.params.RadialProfileParam.create(center="user", xc=0.0, yc=0.0)
    tracemalloc.start()
    try:
        sig = sipi.radial_profile(obj, param)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 10e6
    assert sig.x.size <= 2 * data.size - 1
    assert np.all(np.isfinite(sig.y)) and np.allclose(sig.x, -sig.x[::-1])
    assert sig.y[sig.x.size // 2] == data[0, 0]


def test_radial_profile_refusals() -> None:
    """Coincident pixel centers, invalid bin widths and centers are refused
    explicitly (regression: obscure error after divisions by zero)."""
    obj = _image(np.ones((5, 5)))
    obj.set_coords(np.array([0.0, 0.0, 1.0, 2.0, 3.0]), np.arange(5.0))
    param = sigima.params.RadialProfileParam.create(center="user", xc=1.0, yc=1.0)
    with pytest.raises(ValueError, match="coincide along X"):
        sipi.radial_profile(obj, param)
    data = np.ones((5, 5))
    for step in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="bin width"):
            sigima.tools.image.get_radial_profile(data, (2.0, 2.0), step=step)
    for center in ((np.nan, 2.0), (2.0, np.inf)):
        with pytest.raises(ValueError, match="finite"):
            sigima.tools.image.get_radial_profile(data, center)


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


SEGMENT_DATA = np.arange(100.0).reshape(10, 10)


def _segment(x1, y1, x2, y2, obj=None) -> sigima.objects.SignalObj:
    """Return the segment profile from (x1, y1) to (x2, y2)."""
    obj = _image(SEGMENT_DATA) if obj is None else obj
    param = sigima.params.SegmentProfileParam.create(x1=x1, y1=y1, x2=x2, y2=y2)
    return sipi.segment_profile(obj, param)


def _segment_profile_1_3(data, row0, col0, row1, col1) -> np.ndarray:
    """Segment profile of Sigima 1.3 (``csline``, in pixel indices), without its
    reordering of rows and columns, which mirrored anti-diagonal segments."""
    count = max(abs(row1 - row0), abs(col1 - col0)) + 1
    rows = np.linspace(row0, row1, count).astype(int)
    cols = np.linspace(col0, col1, count).astype(int)
    return data[rows, cols].astype(float)


def test_segment_profile_end_points_included() -> None:
    """Both end points are sampled (regression: the diagonal of a 10 x 10 image
    was sampled in steps of the pixel size, up to pixel (8, 8), duplicating some
    pixels and omitting the last one)."""
    sig = _segment(0, 0, 9, 9)
    assert np.array_equal(sig.y, np.diag(SEGMENT_DATA))
    assert sig.y[-1] == 99.0
    assert np.allclose(sig.x, np.linspace(0.0, 9.0 * np.sqrt(2.0), 10))
    # Oblique segment: one sample per column (the axis spanning the most pixels),
    # each taking the nearest pixel, the end pixel included
    sig = _segment(0, 0, 9, 4)
    rows = np.floor(4 * np.arange(10) / 9 + 0.5).astype(int)
    assert np.array_equal(sig.y, SEGMENT_DATA[rows, np.arange(10)])
    assert sig.y[-1] == SEGMENT_DATA[4, 9]
    assert np.allclose(sig.x, np.linspace(0.0, np.hypot(9.0, 4.0), 10))


@pytest.mark.parametrize(
    "start, end",
    [
        ((0, 0), (9, 9)),  # diagonal
        ((9, 0), (0, 9)),  # anti-diagonal
        ((8, 1), (2, 7)),  # anti-diagonal, inside the image
        ((0, 4), (9, 4)),  # horizontal
        ((9, 4), (0, 4)),  # horizontal, reversed
        ((3, 0), (3, 9)),  # vertical
    ],
)
def test_segment_profile_like_sigima_1_3(start, end) -> None:
    """On images without calibration, axis-aligned and diagonal segments give the
    profiles of Sigima 1.3."""
    (x1, y1), (x2, y2) = start, end
    expected = _segment_profile_1_3(SEGMENT_DATA, y1, x1, y2, x2)
    assert np.array_equal(_segment(x1, y1, x2, y2).y, expected)


@pytest.mark.parametrize(
    "x1, y1, x2, y2, expected",
    [
        (0, 0, 0.6, 0, [0.0, 1.0]),  # shorter than a pixel: end pixel sampled
        (0, 0, 0.4, 0.3, [0.0]),  # both end points in the same pixel
        (2, 3, 2, 3, [32.0]),  # zero length
    ],
)
def test_segment_profile_short_segments(x1, y1, x2, y2, expected) -> None:
    """Segments shorter than a pixel sample the pixels of both end points."""
    sig = _segment(x1, y1, x2, y2)
    assert np.array_equal(sig.y, expected)
    assert np.allclose(sig.x, np.linspace(0, np.hypot(x2 - x1, y2 - y1), len(sig.x)))


def test_segment_profile_tiny_spacings() -> None:
    """The number of samples does not depend on the pixel spacings (regression:
    a tiny spacing gave hundreds of millions of samples for a few pixels)."""
    data = np.arange(25.0).reshape(5, 5)
    # Non-uniform coordinates with a tiny spacing
    obj = _image(data)
    obj.set_coords(np.array([0.0, 1e-8, 1.0, 2.0, 3.0]), np.arange(5.0))
    sig = _segment(0, 0, 3, 4, obj)
    assert sig.y.size == 5
    assert sig.y[0] == data[0, 0] and sig.y[-1] == data[4, 4]
    # Strongly anisotropic pixels
    obj = _image(data)
    obj.set_uniform_coords(1e-8, 1.0, 0.0, 0.0)
    sig = _segment(0, 0, 4e-8, 4, obj)
    assert np.array_equal(sig.y, np.diag(data))
    assert np.allclose(sig.x, np.linspace(0.0, np.hypot(4e-8, 4.0), 5))
