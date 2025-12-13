# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Image grid ROI unit tests"""

from __future__ import annotations

import os.path as osp
from copy import deepcopy

import guidata.dataset as gds
import numpy as np
from numpy.testing import assert_array_equal
from pytest import approx

from sigima.io import read_roi_grid, write_roi_grid
from sigima.objects import ImageObj, ImageROI, create_image
from sigima.proc.image.extraction import (
    Direction,
    ROIGridParam,
    extract_roi,
    generate_image_grid_roi,
)
from sigima.tests.data import create_grid_of_gaussian_images
from sigima.tests.helpers import WorkdirRestoringTempDir


def _roi_by_title(roi: list[ImageROI], title: str) -> ImageROI:
    """Get ROI by title."""
    for r in roi:
        if getattr(r, "title", None) == title:
            return r
    raise KeyError(title)


def test_roi_grid_basic_geometry() -> None:
    """2x2 grid, 50% size, centered (50% translations)."""
    img = create_grid_of_gaussian_images()  # synthetic image with known geometry
    p = ROIGridParam()
    p.nx = p.ny = 2
    p.xsize = p.ysize = 50
    p.xtranslation = p.ytranslation = 50
    p.xdirection = p.ydirection = Direction.INCREASING
    p.base_name = "ROI"
    p.name_pattern = "{base}({r},{c})"

    # src.roi must stay untouched (pure builder)
    assert img.roi is None

    roi = generate_image_grid_roi(img, p)

    # 4 rectangles created
    items = list(roi)
    assert len(items) == 4

    # Titles present
    titles = {r.title for r in items}
    assert {"ROI(1,1)", "ROI(1,2)", "ROI(2,1)", "ROI(2,2)"} <= titles

    # Check one rectangle’s geometry (top-left label)
    r11 = _roi_by_title(roi, "ROI(1,1)")
    _x0, _y0, dx, dy = r11.get_physical_coords(img)  # uses indices=False path
    # Each cell: width/2 by height/2; ROI takes 50% of that
    assert dx == approx((img.width / 2) * 0.5)
    assert dy == approx((img.height / 2) * 0.5)

    # Source image must still be unmodified
    assert img.roi is None


def test_labeling_changes_with_direction_but_geometry_set_is_invariant() -> None:
    """Flipping directions relabels cells
    but the set of rectangles (geometry) stays the same."""
    img = create_grid_of_gaussian_images()
    base = ROIGridParam()
    base.nx = base.ny = 2
    base.xsize = base.ysize = 50
    base.xtranslation = base.ytranslation = 50
    base.base_name = "ROI"
    base.name_pattern = "{base}({r},{c})"

    # Increasing both
    p_inc = deepcopy(base)
    p_inc.xdirection = p_inc.ydirection = Direction.INCREASING
    roi_inc = generate_image_grid_roi(img, p_inc)
    geoms_inc = sorted(
        (r.get_physical_coords(img) for r in roi_inc), key=lambda t: (t[0], t[1])
    )

    # Decreasing both
    p_dec = deepcopy(base)
    p_dec.xdirection = p_dec.ydirection = Direction.DECREASING
    roi_dec = generate_image_grid_roi(img, p_dec)
    geoms_dec = sorted(
        (r.get_physical_coords(img) for r in roi_dec), key=lambda t: (t[0], t[1])
    )

    # Same rectangles, just different titles
    for (x0a, y0a, dxa, dya), (x0b, y0b, dxb, dyb) in zip(geoms_inc, geoms_dec):
        assert x0a == approx(x0b)
        assert y0a == approx(y0b)
        assert dxa == approx(dxb)
        assert dya == approx(dyb)


def test_translation_semantics_delta() -> None:
    """Changing translation by +10% moves rectangles by 10% of image size."""
    img = create_grid_of_gaussian_images()
    p1 = ROIGridParam()
    p1.nx = p1.ny = 2
    p1.xsize = p1.ysize = 50
    p1.xtranslation = p1.ytranslation = 50  # centered
    p1.xdirection = p1.ydirection = Direction.INCREASING

    p2 = deepcopy(p1)
    p2.xtranslation = 60  # +10% shift in X

    roi1 = generate_image_grid_roi(img, p1)
    roi2 = generate_image_grid_roi(img, p2)

    r11_1 = _roi_by_title(roi1, "ROI(1,1)")
    r11_2 = _roi_by_title(roi2, "ROI(1,1)")

    x0_1, y0_1, dx1, dy1 = r11_1.get_physical_coords(img)
    x0_2, y0_2, dx2, dy2 = r11_2.get_physical_coords(img)

    # Width should be unchanged; position should shift by exactly 10% of image width
    assert dx1 == approx(dx2)
    assert dy1 == approx(dy2)
    assert (x0_2 - x0_1) == approx(0.10 * img.width)
    assert (y0_2 - y0_1) == approx(0.00 * img.height)


def test_invalid_name_pattern_falls_back() -> None:
    """Malformed pattern should not break: titles fall back to 'ROI(r,c)'."""
    img = create_grid_of_gaussian_images()
    p = ROIGridParam()
    p.nx = p.ny = 1
    p.xsize = p.ysize = 50
    p.xtranslation = p.ytranslation = 50
    p.xdirection = p.ydirection = Direction.INCREASING
    p.base_name = "ANY"
    p.name_pattern = "{this_will_raise}"  # invalid placeholders

    roi = generate_image_grid_roi(img, p)
    titles = [r.title for r in roi]
    assert titles == ["ROI(1,1)"]  # see fallback in implementation


def test_zero_size_is_allowed_currently() -> None:
    """Current behavior: 0% sizes produce degenerate rectangles (dx==0 or dy==0)."""
    img = create_grid_of_gaussian_images()
    p = ROIGridParam()
    p.nx = p.ny = 2
    p.xsize = 0
    p.ysize = 50
    p.xtranslation = p.ytranslation = 50

    roi = generate_image_grid_roi(img, p)
    # All ROIs exist; all have dx == 0
    for r in roi:
        _x0, _y0, dx, dy = r.get_physical_coords(img)
        assert dx == approx(0.0)
        assert dy > 0.0


def _make_positional_image(h=6, w=9, dx=1.0, dy=1.0, x0=0.0, y0=0.0) -> ImageObj:
    """
    pixel(y, x) = 1000*y + x  (strictly monotone in both axes)
    Choosing H=6, W=9 allows clean 2x3 tiling (cell=3x3).
    """
    data = np.add.outer(
        1000 * np.arange(h, dtype=np.int32), np.arange(w, dtype=np.int32)
    )
    img = create_image("positional", data)
    img.set_uniform_coords(dx, dy, x0, y0)
    img.roi = None
    return img


def test_roi_grid_extract_matches_pattern() -> None:
    """Test that the extracted ROIs match the expected pattern."""
    img = _make_positional_image(h=6, w=9)  # 6x9 → ny=2, nx=3 → cells are 3x3

    # Build a full-coverage grid: each ROI == cell (no gaps/overlaps)
    p = ROIGridParam()
    p.nx, p.ny = 3, 2
    p.xsize = p.ysize = 100  # ROI size == cell size
    p.xtranslation = p.ytranslation = 50  # centered on each cell
    p.xdirection = p.ydirection = Direction.INCREASING
    p.base_name = "ROI"
    p.name_pattern = "{base}({r},{c})"

    roi = generate_image_grid_roi(img, p)  # pure builder, no mutation
    # Sanity: full coverage, 3*2 cells, all 3x3

    params = roi.to_params(img)
    for rparam in params:
        # Reference window from indices
        x0, y0, x1, y1 = rparam.get_bounding_box_indices(img)
        ref = img.data[y0:y1, x0:x1]

        # Extract via computation function
        extracted = extract_roi(img, rparam)
        out = extracted.data

        # 1) Pixel-exact equality
        assert_array_equal(out, ref)

        # 2) Dimensions are the expected 3x3
        assert out.shape == (3, 3)

        # 3) Physical origin is consistent with bounding box (dx=dy=1, x0=y0=0)
        px0, py0, _px1, _py1 = rparam.get_bounding_box_physical()
        assert extracted.x0 == px0
        assert extracted.y0 == py0


def test_roi_grid_extract_with_translation() -> None:
    """Shift the grid by +10% in X (of full image width) and verify that each ROI
    moves accordingly — the extracted content matches the shifted reference.
    """
    img = _make_positional_image(h=6, w=9)  # width=9 → +10% shift = 0.9 pixel

    # Base centered grid (2x3)
    p1 = ROIGridParam()
    p1.nx, p1.ny = 3, 2
    p1.xsize = p1.ysize = 100
    p1.xtranslation = p1.ytranslation = 50
    p1.xdirection = p1.ydirection = Direction.INCREASING
    p1.base_name = "ROI"
    p1.name_pattern = "{base}({r},{c})"

    # Shifted grid (+10% in X)
    p2 = deepcopy(p1)
    p2.xtranslation = 60

    roi1 = generate_image_grid_roi(img, p1)
    roi2 = generate_image_grid_roi(img, p2)

    # Compare first ROI of each row/col "logically" (same label), but expect
    # a one-pixel shift when rounding indices. To avoid rounding heuristics,
    # we compare against each ROI's own bounding box-derived slice.
    roiparams1, roiparams2 = roi1.to_params(img), roi2.to_params(img)
    for rp1, rp2 in zip(roiparams1, roiparams2):
        ref1 = img.data[
            rp1.get_bounding_box_indices(img)[1] : rp1.get_bounding_box_indices(img)[3],
            rp1.get_bounding_box_indices(img)[0] : rp1.get_bounding_box_indices(img)[2],
        ]
        ref2 = img.data[
            rp2.get_bounding_box_indices(img)[1] : rp2.get_bounding_box_indices(img)[3],
            rp2.get_bounding_box_indices(img)[0] : rp2.get_bounding_box_indices(img)[2],
        ]
        out1 = extract_roi(img, rp1).data
        out2 = extract_roi(img, rp2).data
        # Both extractions must match their own references exactly
        assert_array_equal(out1, ref1)
        assert_array_equal(out2, ref2)


def test_roi_grid_import_export() -> None:
    """Test the import and export of ROI grids."""
    p = ROIGridParam()
    p.nx, p.ny = 3, 2
    p.xsize = p.ysize = 100
    p.xtranslation = p.ytranslation = 50
    p.xdirection = p.ydirection = Direction.INCREASING
    p.base_name = "ROI"
    p.name_pattern = "{base}({r},{c})"

    with WorkdirRestoringTempDir() as temp_dir:
        path = osp.join(temp_dir, "test_roi_grid.json")
        write_roi_grid(path, p)
        new_p = read_roi_grid(path)

    gds.assert_datasets_equal(new_p, p, "Imported ROI grid does not match original")


def test_roi_grid_custom_step() -> None:
    """Test grid ROI with custom xstep/ystep parameters.

    This tests the bug fix for cases where ROI spacing differs from evenly
    distributed grid (e.g., laser spot arrays with gaps between spots).
    """
    # Create a test image
    img = create_image(
        title="Test Grid",
        data=np.random.rand(200, 300),
    )

    # Test Case 1: Default behavior (100% step = evenly distributed)
    p_default = ROIGridParam()
    p_default.nx = p_default.ny = 3
    p_default.xsize = p_default.ysize = 30  # 30% of cell size
    p_default.xtranslation = p_default.ytranslation = 50  # centered
    p_default.xstep = p_default.ystep = 100  # evenly distributed

    roi_default = generate_image_grid_roi(img, p_default)
    items_default = list(roi_default)
    assert len(items_default) == 9

    # Get spacing between first two ROIs in X direction
    r11 = _roi_by_title(roi_default, "ROI(1,1)")
    r12 = _roi_by_title(roi_default, "ROI(1,2)")
    x0_r11, _, _, _ = r11.get_physical_coords(img)
    x0_r12, _, _, _ = r12.get_physical_coords(img)
    default_x_spacing = x0_r12 - x0_r11

    # Expected: width / nx
    expected_default_spacing = img.width / p_default.nx
    assert default_x_spacing == approx(expected_default_spacing)

    # Test Case 2: Tighter spacing (50% step = half the cell width)
    p_tight = deepcopy(p_default)
    p_tight.xstep = p_tight.ystep = 50  # Half spacing

    roi_tight = generate_image_grid_roi(img, p_tight)
    items_tight = list(roi_tight)
    assert len(items_tight) == 9

    r11_tight = _roi_by_title(roi_tight, "ROI(1,1)")
    r12_tight = _roi_by_title(roi_tight, "ROI(1,2)")
    x0_r11_tight, _, _, _ = r11_tight.get_physical_coords(img)
    x0_r12_tight, _, _, _ = r12_tight.get_physical_coords(img)
    tight_x_spacing = x0_r12_tight - x0_r11_tight

    # Should be half of default spacing
    expected_tight_spacing = (img.width / p_tight.nx) * 0.5
    assert tight_x_spacing == approx(expected_tight_spacing)
    assert tight_x_spacing == approx(default_x_spacing * 0.5)

    # Test Case 3: Wider spacing (150% step)
    p_wide = deepcopy(p_default)
    p_wide.xstep = p_wide.ystep = 150  # 1.5x spacing

    roi_wide = generate_image_grid_roi(img, p_wide)
    items_wide = list(roi_wide)
    assert len(items_wide) == 9

    r11_wide = _roi_by_title(roi_wide, "ROI(1,1)")
    r12_wide = _roi_by_title(roi_wide, "ROI(1,2)")
    x0_r11_wide, _, _, _ = r11_wide.get_physical_coords(img)
    x0_r12_wide, _, _, _ = r12_wide.get_physical_coords(img)
    wide_x_spacing = x0_r12_wide - x0_r11_wide

    # Should be 1.5x of default spacing
    expected_wide_spacing = (img.width / p_wide.nx) * 1.5
    assert wide_x_spacing == approx(expected_wide_spacing)
    assert wide_x_spacing == approx(default_x_spacing * 1.5)

    # Test Case 4: Different X and Y steps
    p_mixed = deepcopy(p_default)
    p_mixed.xstep = 80
    p_mixed.ystep = 120

    roi_mixed = generate_image_grid_roi(img, p_mixed)
    items_mixed = list(roi_mixed)
    assert len(items_mixed) == 9

    # Check X spacing
    r11_mixed = _roi_by_title(roi_mixed, "ROI(1,1)")
    r12_mixed = _roi_by_title(roi_mixed, "ROI(1,2)")
    x0_r11_mixed, y0_r11_mixed, _, _ = r11_mixed.get_physical_coords(img)
    x0_r12_mixed, _, _, _ = r12_mixed.get_physical_coords(img)
    mixed_x_spacing = x0_r12_mixed - x0_r11_mixed

    # Check Y spacing
    r21_mixed = _roi_by_title(roi_mixed, "ROI(2,1)")
    _, y0_r21_mixed, _, _ = r21_mixed.get_physical_coords(img)
    mixed_y_spacing = y0_r21_mixed - y0_r11_mixed

    assert mixed_x_spacing == approx((img.width / p_mixed.nx) * 0.8)
    assert mixed_y_spacing == approx((img.height / p_mixed.ny) * 1.2)


if __name__ == "__main__":
    test_roi_grid_custom_step()
    test_roi_grid_import_export()
