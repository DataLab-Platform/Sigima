# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the matplotlib visualization backend (viz_mpl.py)

These tests verify the matplotlib-specific implementation of visualization features:
- Physical coordinates (extent) for images
- Aspect ratio correction (dx/dy)
- Geometry result overlays (POINT, MARKER, RECTANGLE, CIRCLE, SEGMENT, ELLIPSE, POLYGON)

Tests are designed to run headlessly (no plt.show()) unless --gui flag is passed.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from sigima.objects import (
    GeometryResult,
    ImageObj,
    KindShape,
    create_image,
    create_image_roi,
    create_signal,
)


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401  # pylint: disable=unused-import

        return True
    except ImportError:
        return False


# Skip all tests in this module if matplotlib is not available
pytestmark = pytest.mark.skipif(
    "matplotlib" not in sys.modules and not _has_matplotlib(),
    reason="matplotlib not available",
)


@pytest.fixture
def mock_plt_show(monkeypatch):
    """Mock plt.show() to prevent blocking during tests."""
    import matplotlib.pyplot as plt

    shown_figures = []

    def mock_show():
        # Capture figures instead of showing them
        shown_figures.extend(plt.get_fignums())
        plt.close("all")

    monkeypatch.setattr(plt, "show", mock_show)
    return shown_figures


@pytest.fixture
def image_with_physical_coords() -> ImageObj:
    """Create an ImageObj with non-trivial physical coordinates.

    Returns:
        ImageObj with x0=100, y0=200, dx=0.5, dy=0.25
    """
    data = np.random.rand(50, 100).astype(np.float32)
    image = create_image("Physical Coords Test", data=data)
    # Set physical coordinates: origin at (100, 200), spacing 0.5x0.25
    image.set_uniform_coords(dx=0.5, dy=0.25, x0=100.0, y0=200.0)
    return image


@pytest.fixture
def image_default_coords() -> ImageObj:
    """Create an ImageObj with default pixel coordinates.

    Returns:
        ImageObj with default x0=0, y0=0, dx=1, dy=1
    """
    data = np.random.rand(64, 64).astype(np.float32)
    return create_image("Default Coords Test", data=data)


class TestGetImageExtentAndAspect:
    """Tests for _get_image_extent_and_aspect() helper function."""

    def test_default_coordinates(self, image_default_coords: ImageObj):
        """Test extent computation with default pixel coordinates."""
        from sigima.viz.viz_mpl import _get_image_extent_and_aspect

        extent, aspect = _get_image_extent_and_aspect(image_default_coords)

        # Default: x0=0, y0=0, dx=1, dy=1, 64x64 image
        # Expected extent: [-0.5, 63.5, 63.5, -0.5] (left, right, bottom, top)
        assert len(extent) == 4
        assert extent[0] == pytest.approx(-0.5)  # left
        assert extent[1] == pytest.approx(63.5)  # right
        assert extent[2] == pytest.approx(63.5)  # bottom (with origin=upper)
        assert extent[3] == pytest.approx(-0.5)  # top
        assert aspect == pytest.approx(1.0)  # dx/dy = 1

    def test_physical_coordinates(self, image_with_physical_coords: ImageObj):
        """Test extent computation with physical coordinates."""
        from sigima.viz.viz_mpl import _get_image_extent_and_aspect

        obj = image_with_physical_coords
        extent, aspect = _get_image_extent_and_aspect(obj)

        # Image is 50 rows x 100 cols
        # x0=100, dx=0.5 -> xmin=100, xmax=100 + 99*0.5 = 149.5
        # y0=200, dy=0.25 -> ymin=200, ymax=200 + 49*0.25 = 212.25
        # extent = [left, right, bottom, top]
        #        = [xmin-dx/2, xmax+dx/2, ymax+dy/2, ymin-dy/2]
        #        = [99.75, 149.75, 212.375, 199.875]
        assert extent[0] == pytest.approx(99.75)  # left
        assert extent[1] == pytest.approx(149.75)  # right
        assert extent[2] == pytest.approx(212.375)  # bottom
        assert extent[3] == pytest.approx(199.875)  # top

        # Aspect ratio = dx/dy = 0.5/0.25 = 2.0
        assert aspect == pytest.approx(2.0)

    def test_anisotropic_pixels(self):
        """Test aspect ratio with anisotropic (non-square) pixels."""
        from sigima.viz.viz_mpl import _get_image_extent_and_aspect

        data = np.zeros((10, 20))
        image = create_image("Anisotropic", data=data)
        image.set_uniform_coords(dx=2.0, dy=0.5, x0=0, y0=0)

        extent, aspect = _get_image_extent_and_aspect(image)

        # dx=2, dy=0.5 -> aspect = 4.0
        assert aspect == pytest.approx(4.0)


class TestViewImagesPhysicalCoords:
    """Tests for view_images() with physical coordinates."""

    def test_view_images_uses_extent(
        self, mock_plt_show, image_with_physical_coords: ImageObj
    ):
        """Test that view_images() applies physical coordinates via extent."""
        from sigima.viz import viz_mpl

        # Should not raise and should create a figure
        viz_mpl.view_images(image_with_physical_coords, title="Physical Coords")

        assert len(mock_plt_show) > 0, "Expected at least one figure to be shown"

    def test_view_images_numpy_array(self, mock_plt_show):
        """Test view_images() with raw NumPy array uses pixel coordinates."""
        from sigima.viz import viz_mpl

        data = np.random.rand(32, 32)
        viz_mpl.view_images(data, title="NumPy Array")

        assert len(mock_plt_show) > 0

    def test_view_images_multiple(
        self, mock_plt_show, image_with_physical_coords, image_default_coords
    ):
        """Test view_images() with multiple ImageObj instances."""
        from sigima.viz import viz_mpl

        viz_mpl.view_images(
            [image_with_physical_coords, image_default_coords],
            title="Multiple Images",
        )

        assert len(mock_plt_show) > 0


class TestViewImagesSideBySide:
    """Tests for view_images_side_by_side() with physical coordinates."""

    def test_side_by_side_physical_coords(
        self, mock_plt_show, image_with_physical_coords
    ):
        """Test side-by-side view applies physical coordinates."""
        from sigima.viz import viz_mpl

        viz_mpl.view_images_side_by_side(
            [image_with_physical_coords, image_with_physical_coords],
            titles=["Image 1", "Image 2"],
        )

        assert len(mock_plt_show) > 0

    def test_side_by_side_mixed_types(self, mock_plt_show, image_with_physical_coords):
        """Test side-by-side view with mixed ImageObj and NumPy arrays."""
        from sigima.viz import viz_mpl

        numpy_array = np.random.rand(50, 50)
        viz_mpl.view_images_side_by_side(
            [image_with_physical_coords, numpy_array],
            titles=["ImageObj", "NumPy"],
        )

        assert len(mock_plt_show) > 0


class TestGeometryOverlays:
    """Tests for geometry result overlays (POINT, MARKER, etc.)."""

    @pytest.fixture
    def test_image(self) -> ImageObj:
        """Create a simple test image."""
        data = np.random.rand(100, 100).astype(np.float32)
        return create_image("Geometry Test", data=data)

    def test_point_overlay(self, mock_plt_show, test_image):
        """Test POINT geometry overlay renders correctly."""
        from sigima.viz import viz_mpl

        result = GeometryResult.from_coords(
            coords=np.array([[50, 50]]),
            kind=KindShape.POINT,
            title="Test Point",
        )

        viz_mpl.view_images(test_image, results=result)
        assert len(mock_plt_show) > 0

    def test_marker_overlay(self, mock_plt_show, test_image):
        """Test MARKER geometry overlay (crosshair) renders correctly."""
        from sigima.viz import viz_mpl

        result = GeometryResult.from_coords(
            coords=np.array([[25, 75]]),
            kind=KindShape.MARKER,
            title="Test Marker",
        )

        viz_mpl.view_images(test_image, results=result)
        assert len(mock_plt_show) > 0

    def test_rectangle_overlay(self, mock_plt_show, test_image):
        """Test RECTANGLE geometry overlay renders correctly."""
        from sigima.viz import viz_mpl

        result = GeometryResult.from_coords(
            coords=np.array([[10, 10, 30, 40]]),
            kind=KindShape.RECTANGLE,
            title="Test Rectangle",
        )

        viz_mpl.view_images(test_image, results=result)
        assert len(mock_plt_show) > 0

    def test_circle_overlay(self, mock_plt_show, test_image):
        """Test CIRCLE geometry overlay renders correctly."""
        from sigima.viz import viz_mpl

        result = GeometryResult.from_coords(
            coords=np.array([[50, 50, 20]]),
            kind=KindShape.CIRCLE,
            title="Test Circle",
        )

        viz_mpl.view_images(test_image, results=result)
        assert len(mock_plt_show) > 0

    def test_segment_overlay(self, mock_plt_show, test_image):
        """Test SEGMENT geometry overlay renders correctly."""
        from sigima.viz import viz_mpl

        result = GeometryResult.from_coords(
            coords=np.array([[10, 10, 90, 90]]),
            kind=KindShape.SEGMENT,
            title="Test Segment",
        )

        viz_mpl.view_images(test_image, results=result)
        assert len(mock_plt_show) > 0

    def test_ellipse_overlay(self, mock_plt_show, test_image):
        """Test ELLIPSE geometry overlay renders correctly."""
        from sigima.viz import viz_mpl

        # Ellipse: (xc, yc, a, b, theta)
        result = GeometryResult.from_coords(
            coords=np.array([[50, 50, 30, 15, 0.5]]),
            kind=KindShape.ELLIPSE,
            title="Test Ellipse",
        )

        viz_mpl.view_images(test_image, results=result)
        assert len(mock_plt_show) > 0

    def test_polygon_overlay(self, mock_plt_show, test_image):
        """Test POLYGON geometry overlay renders correctly."""
        from sigima.viz import viz_mpl

        # Polygon: x0, y0, x1, y1, x2, y2, ...
        result = GeometryResult.from_coords(
            coords=np.array([[20, 20, 80, 20, 80, 80, 20, 80]]),
            kind=KindShape.POLYGON,
            title="Test Polygon",
        )

        viz_mpl.view_images(test_image, results=result)
        assert len(mock_plt_show) > 0

    def test_multiple_points(self, mock_plt_show, test_image):
        """Test multiple POINT geometries in a single result."""
        from sigima.viz import viz_mpl

        # Multiple points (each row is a separate point)
        result = GeometryResult.from_coords(
            coords=np.array([[20, 30], [50, 50], [80, 70]]),
            kind=KindShape.POINT,
            title="Multiple Points",
        )

        viz_mpl.view_images(test_image, results=result)
        assert len(mock_plt_show) > 0

    def test_multiple_results(self, mock_plt_show, test_image):
        """Test multiple different geometry results overlaid on same image."""
        from sigima.viz import viz_mpl

        point_result = GeometryResult.from_coords(
            coords=np.array([[50, 50]]),
            kind=KindShape.POINT,
            title="Point",
        )
        circle_result = GeometryResult.from_coords(
            coords=np.array([[50, 50, 20]]),
            kind=KindShape.CIRCLE,
            title="Circle",
        )

        viz_mpl.view_images(test_image, results=[point_result, circle_result])
        assert len(mock_plt_show) > 0


class TestMaskOverlay:
    """Tests for mask visualization via ROI."""

    def test_mask_overlay_via_roi(self, mock_plt_show):
        """Test that ROI-generated mask is shown with red overlay."""
        from sigima.viz import viz_mpl

        data = np.random.rand(50, 50).astype(np.float32)
        image = create_image("Masked Image", data=data)

        # Add a rectangular ROI to generate a mask
        # create_image_roi returns an ImageROI container with the single ROI
        roi = create_image_roi("rectangle", [10, 10, 20, 20], indices=True)
        image.roi = roi

        viz_mpl.view_images(image, title="Image with ROI Mask")
        assert len(mock_plt_show) > 0


class TestViewCurves:
    """Basic tests for view_curves() to ensure it works with SignalObj."""

    def test_view_curves_signal_obj(self, mock_plt_show):
        """Test view_curves with SignalObj."""
        from sigima.viz import viz_mpl

        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        signal = create_signal("Sine Wave", x=x, y=y)

        viz_mpl.view_curves(signal, title="Signal Test")
        assert len(mock_plt_show) > 0

    def test_view_curves_numpy_arrays(self, mock_plt_show):
        """Test view_curves with raw NumPy arrays."""
        from sigima.viz import viz_mpl

        x = np.linspace(0, 5, 50)
        y = np.exp(-x)

        viz_mpl.view_curves((x, y), title="NumPy Arrays")
        assert len(mock_plt_show) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
