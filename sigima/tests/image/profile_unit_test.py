# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction unit test

Profiles are defined and returned in the physical coordinates of the image: the
tests use calibrated images (origin, anisotropic pixel sizes, non-uniform
coordinates), whose expected profiles are given by the pixel indices.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import create_sincos_image
from sigima.tests.helpers import check_array_result

WIDTH, HEIGHT = 256, 128
# Calibration: X = X0 + DX * column, Y = Y0 + DY * row
X0, Y0, DX, DY = -10.0, 5.0, 0.5, 2.0


def _image(calibrated: bool = True) -> sigima.objects.ImageObj:
    """Return the test image, calibrated with anisotropic pixels by default."""
    newparam = sigima.objects.NewImageParam.create(
        dtype=sigima.objects.ImageDatatypes.UINT16, height=HEIGHT, width=WIDTH
    )
    ima = create_sincos_image(newparam)
    if calibrated:
        ima.set_uniform_coords(DX, DY, X0, Y0)
    ima.xunit = ima.yunit = "mm"
    ima.xlabel, ima.ylabel = "Position X", "Position Y"
    return ima


@pytest.mark.validation
def test_line_profile() -> None:
    """Test line profile computation"""
    ima = _image()

    # Horizontal profile: row whose center is the nearest to Y (row 100 at 205)
    param = sigima.params.LineProfileParam.create(
        y=Y0 + 100.4 * DY, direction="horizontal"
    )
    sig = sigima.proc.image.line_profile(ima, param)
    check_array_result("Horizontal line profile", sig.y, ima.data[100, :].astype(float))
    check_array_result("Horizontal X axis", sig.x, X0 + DX * np.arange(WIDTH))
    assert (sig.xunit, sig.xlabel) == ("mm", "Position X")

    # Vertical profile: column 50; X axis along Y, with the Y unit and label
    param = sigima.params.LineProfileParam.create(
        x=X0 + 49.6 * DX, direction="vertical"
    )
    sig = sigima.proc.image.line_profile(ima, param)
    check_array_result("Vertical line profile", sig.y, ima.data[:, 50].astype(float))
    check_array_result("Vertical X axis", sig.x, Y0 + DY * np.arange(HEIGHT))
    assert sig.xlabel == "Position Y"

    # Coordinates outside the image are refused (beyond the outer pixel edges)
    for direction, kwargs in (
        ("horizontal", {"y": Y0 - 0.51 * DY}),
        ("vertical", {"x": X0 + (WIDTH - 0.49) * DX}),
    ):
        param = sigima.params.LineProfileParam.create(direction=direction, **kwargs)
        with pytest.raises(ValueError, match="outside the image"):
            sigima.proc.image.line_profile(ima, param)


@pytest.mark.validation
def test_segment_profile() -> None:
    """Test segment profile computation"""
    ima = _image(calibrated=False)
    ima.set_uniform_coords(0.5, 0.5, X0, Y0)  # Isotropic: one sample per pixel

    # Vertical segment along column 20, from row 10 to row 120
    param = sigima.params.SegmentProfileParam.create(
        x1=X0 + 20 * 0.5, y1=Y0 + 10 * 0.5, x2=X0 + 20 * 0.5, y2=Y0 + 120 * 0.5
    )
    sig = sigima.proc.image.segment_profile(ima, param)
    check_array_result("Segment profile", sig.y, ima.data[10:121, 20].astype(float))
    check_array_result("Segment distances", sig.x, 0.5 * np.arange(111))
    assert sig.xunit == "mm"

    # Anti-diagonal segment: sampled from its start point (row 0, column 30) to its
    # end point (row 30, column 0), each sample taking the nearest pixel
    param = sigima.params.SegmentProfileParam.create(
        x1=X0 + 30 * 0.5, y1=Y0, x2=X0, y2=Y0 + 30 * 0.5
    )
    sig = sigima.proc.image.segment_profile(ima, param)
    fraction = np.arange(43) * 0.5 / np.hypot(15.0, 15.0)
    rows = np.floor(30 * fraction + 0.5).astype(int)
    cols = np.floor(30 - 30 * fraction + 0.5).astype(int)
    check_array_result(
        "Anti-diagonal segment", sig.y, ima.data[rows, cols].astype(float)
    )
    assert sig.y[0] == ima.data[0, 30]

    # End points outside the image and different X/Y units are refused
    param = sigima.params.SegmentProfileParam.create(x1=X0, y1=Y0, x2=X0, y2=Y0 + 1e3)
    with pytest.raises(ValueError, match="outside the image"):
        sigima.proc.image.segment_profile(ima, param)
    ima.yunit = "s"
    param = sigima.params.SegmentProfileParam.create(x1=X0, y1=Y0, x2=X0, y2=Y0 + 1)
    with pytest.raises(ValueError, match="units"):
        sigima.proc.image.segment_profile(ima, param)


@pytest.mark.validation
def test_average_profile() -> None:
    """Test average profile computation"""
    ima = _image()
    # Pixels whose centers lie in the rectangle: rows 10 to 100, columns 20 to 230
    param = sigima.params.AverageProfileParam.create(
        x1=X0 + 19.6 * DX, y1=Y0 + 10 * DY, x2=X0 + 230.4 * DX, y2=Y0 + 100.4 * DY
    )
    area = ima.data[10:101, 20:231].astype(float)

    param.direction = "horizontal"
    sig = sigima.proc.image.average_profile(ima, param)
    check_array_result("Horizontal average profile", sig.y, area.mean(axis=0))
    check_array_result("Horizontal X axis", sig.x, X0 + DX * np.arange(20, 231))

    param.direction = "vertical"
    sig = sigima.proc.image.average_profile(ima, param)
    check_array_result("Vertical average profile", sig.y, area.mean(axis=1))
    check_array_result("Vertical X axis", sig.x, Y0 + DY * np.arange(10, 101))
    assert sig.xlabel == "Position Y"

    # Default bounds: the whole image
    sig = sigima.proc.image.average_profile(
        ima, sigima.params.AverageProfileParam.create(direction="horizontal")
    )
    check_array_result("Whole image", sig.y, ima.data.astype(float).mean(axis=0))

    # A rectangle holding no pixel center is refused
    param = sigima.params.AverageProfileParam.create(x1=X0 + 0.1 * DX, x2=X0 + 0.4 * DX)
    with pytest.raises(ValueError, match="no pixel center"):
        sigima.proc.image.average_profile(ima, param)


def __test_radial_profile_center(
    obj: sigima.objects.ImageObj, p: sigima.params.RadialProfileParam
) -> sigima.objects.SignalObj:
    """Test radial profile computation with given center.

    Args:
        obj: Image object
        p: Radial profile parameters

    Returns:
        Signal object containing the radial profile
    """
    sig = sigima.proc.image.radial_profile(obj, p)
    assert sig is not None
    assert len(sig.x) == len(sig.y)
    assert len(sig.y) > 0
    # Check that profile values are within expected range
    assert np.all(np.isfinite(sig.y))
    assert np.all(sig.y >= 0)  # Pixel values should be non-negative
    return sig


@pytest.mark.validation
def test_radial_profile() -> None:
    """Test radial profile computation"""
    ima = _image(calibrated=False)

    # Test radial profile with centroid center
    param = sigima.params.RadialProfileParam.create(center="centroid")
    param.update_from_obj(ima)
    __test_radial_profile_center(ima, param)

    # Test radial profile with image center
    param = sigima.params.RadialProfileParam.create(center="center")
    param.update_from_obj(ima)
    __test_radial_profile_center(ima, param)

    # Test radial profile with user-defined center
    xc, yc = WIDTH // 2, HEIGHT // 2
    param = sigima.params.RadialProfileParam.create(center="user", xc=xc, yc=yc)
    sig = __test_radial_profile_center(ima, param)

    # Test that the x-axis represents distance from center (symmetric around 0)
    assert sig.x[0] < 0  # First element should be negative
    assert sig.x[-1] > 0  # Last element should be positive
    assert sig.x[len(sig.x) // 2] == 0  # Center should be at distance 0

    # Calibrated image (isotropic pixels): same profile, physical distances
    calibrated = ima.copy()
    calibrated.set_uniform_coords(0.25, 0.25, X0, Y0)
    param = sigima.params.RadialProfileParam.create(
        center="user", xc=X0 + 0.25 * xc, yc=Y0 + 0.25 * yc
    )
    csig = __test_radial_profile_center(calibrated, param)
    check_array_result("Calibrated radial profile", csig.y, sig.y)
    check_array_result("Calibrated distances", csig.x, 0.25 * sig.x)
    for center in ("center", "centroid"):
        param = sigima.params.RadialProfileParam.create(center=center)
        expected = sigima.proc.image.radial_profile(ima, param)
        result = sigima.proc.image.radial_profile(calibrated, param)
        check_array_result(
            f"Calibrated radial profile ({center})", result.y, expected.y
        )


if __name__ == "__main__":
    test_line_profile()
    test_segment_profile()
    test_average_profile()
    test_radial_profile()
