# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI coordinate setter methods unit tests
=========================================

Tests for set_physical_coords() and set_indices_coords() methods
in BaseSingleImageROI, RectangularROI, and CircularROI classes.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import ImageObj
from sigima.objects.image.roi import CircularROI, PolygonalROI, RectangularROI


def create_test_image(
    dx: float = 1.0, dy: float = 1.0, x0: float = 0.0, y0: float = 0.0
) -> ImageObj:
    """Create a test image with specified coordinate system.

    Args:
        dx: Pixel width in physical units
        dy: Pixel height in physical units
        x0: X origin in physical units
        y0: Y origin in physical units

    Returns:
        ImageObj: Test image object
    """
    obj = ImageObj(title="Test Image")
    obj.data = np.zeros((100, 100), dtype=np.float64)
    obj.set_uniform_coords(dx, dy, x0, y0)
    return obj


class TestRectangularROISetters:
    """Test set_physical_coords and set_indices_coords for RectangularROI"""

    def test_set_physical_coords_indices_mode(self):
        """Test setting physical coords when ROI is in indices mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create ROI in indices mode: rectangle from (10, 20) with delta (20, 20)
        roi = RectangularROI([10, 20, 20, 20], indices=True, title="Test ROI")

        # Set new physical coordinates: (15.0, 25.0) with size (10.0, 10.0)
        new_physical = np.array([15.0, 25.0, 10.0, 10.0])
        roi.set_physical_coords(obj, new_physical)

        # Verify coords were converted and stored as indices (deltas)
        assert roi.indices is True
        # Physical (15.0, 25.0, 10.0, 10.0) means corners (15.0, 25.0) to (25.0, 35.0)
        # x0: (15.0 - 10.0) / 0.5 = 10, x1: (25.0 - 10.0) / 0.5 = 30
        # y0: (25.0 - 20.0) / 0.5 = 10, y1: (35.0 - 20.0) / 0.5 = 30
        # Stored as [ix0, iy0, dx, dy] = [10, 10, 20, 20]
        np.testing.assert_array_equal(roi.coords, [10, 10, 20, 20])

        # Verify we can retrieve the same physical coords
        retrieved = roi.get_physical_coords(obj)
        np.testing.assert_allclose(retrieved, new_physical, rtol=1e-10)

    def test_set_physical_coords_physical_mode(self):
        """Test setting physical coords when ROI is in physical mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create ROI in physical mode
        roi = RectangularROI([15.0, 25.0, 10.0, 10.0], indices=False, title="Test ROI")

        # Set new physical coordinates
        new_physical = np.array([20.0, 30.0, 15.0, 12.0])
        roi.set_physical_coords(obj, new_physical)

        # Verify coords were stored directly as floats
        assert roi.indices is False
        np.testing.assert_array_equal(roi.coords, new_physical)

        # Verify we can retrieve the same physical coords
        retrieved = roi.get_physical_coords(obj)
        np.testing.assert_allclose(retrieved, new_physical, rtol=1e-10)

    def test_set_indices_coords_indices_mode(self):
        """Test setting indices coords when ROI is in indices mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create ROI in indices mode
        roi = RectangularROI([10, 20, 20, 20], indices=True, title="Test ROI")

        # Set new indices coordinates
        new_indices = np.array([5, 8, 15, 12])
        roi.set_indices_coords(obj, new_indices)

        # Verify coords were stored directly
        assert roi.indices is True
        np.testing.assert_array_equal(roi.coords, new_indices)

        # Verify we can retrieve the same indices coords
        retrieved = roi.get_indices_coords(obj)
        np.testing.assert_array_equal(retrieved, new_indices)

    def test_set_indices_coords_physical_mode(self):
        """Test setting indices coords when ROI is in physical mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create ROI in physical mode
        roi = RectangularROI([15.0, 25.0, 10.0, 10.0], indices=False, title="Test ROI")

        # Set new indices coordinates: (10, 10, 20, 20)
        new_indices = np.array([10, 10, 20, 20])
        roi.set_indices_coords(obj, new_indices)

        # Verify coords were converted to physical and stored
        assert roi.indices is False
        # Indices (10, 10, 20, 20) -> Physical (15.0, 25.0, 10.0, 10.0)
        # x0: 10 * 0.5 + 10.0 = 15.0, y0: 10 * 0.5 + 20.0 = 25.0
        # dx: 20 * 0.5 = 10.0, dy: 20 * 0.5 = 10.0
        expected_physical = np.array([15.0, 25.0, 10.0, 10.0])
        np.testing.assert_allclose(roi.coords, expected_physical, rtol=1e-10)

        # Verify we can retrieve the same indices coords
        retrieved = roi.get_indices_coords(obj)
        np.testing.assert_array_equal(retrieved, new_indices)

    def test_roundtrip_physical_to_indices(self):
        """Test roundtrip conversion: physical -> indices -> physical"""
        obj = create_test_image(dx=2.0, dy=3.0, x0=5.0, y0=7.0)

        roi = RectangularROI([25.0, 37.0, 20.0, 30.0], indices=False, title="Test ROI")

        # Get physical coords
        original_physical = roi.get_physical_coords(obj)

        # Convert to indices and back
        indices = roi.get_indices_coords(obj)
        roi.set_indices_coords(obj, np.array(indices))
        retrieved_physical = roi.get_physical_coords(obj)

        # Should be identical
        np.testing.assert_allclose(
            retrieved_physical, original_physical, rtol=1e-10, atol=1e-10
        )

    def test_roundtrip_indices_to_physical(self):
        """Test roundtrip conversion: indices -> physical -> indices"""
        obj = create_test_image(dx=2.0, dy=3.0, x0=5.0, y0=7.0)

        roi = RectangularROI([10, 10, 20, 15], indices=True, title="Test ROI")

        # Get indices coords (should return delta format)
        original_indices = roi.get_indices_coords(obj)
        assert original_indices == [10, 10, 20, 15]

        # Convert to physical and back
        physical = roi.get_physical_coords(obj)
        # physical should be [25.0, 37.0, 40.0, 45.0] based on:
        # ix0=10 -> 10*2+5=25, iy0=10 -> 10*3+7=37
        # dx=20 -> 20*2=40, dy=15 -> 15*3=45
        assert physical == [25.0, 37.0, 40.0, 45.0]

        roi.set_physical_coords(obj, np.array(physical))
        # Should store deltas: [ix0, iy0, dx, dy] = [10, 10, 20, 15]
        assert roi.coords.tolist() == [10, 10, 20, 15]

        retrieved_indices = roi.get_indices_coords(obj)
        # Should retrieve the same indices
        np.testing.assert_array_equal(retrieved_indices, original_indices)


class TestCircularROISetters:
    """Test set_physical_coords and set_indices_coords for CircularROI"""

    def test_set_physical_coords_indices_mode(self):
        """Test setting physical coords when ROI is in indices mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create circular ROI in indices mode: center (50, 50), radius 20 pixels
        roi = CircularROI([50, 50, 20], indices=True, title="Test Circle")

        # Set new physical coordinates: center (30.0, 35.0), radius 10.0
        new_physical = np.array([30.0, 35.0, 10.0])
        roi.set_physical_coords(obj, new_physical)

        # Verify coords were converted and stored as indices
        assert roi.indices is True
        # Physical center (30.0, 35.0) -> Indices (40, 30)
        # xc: (30.0 - 10.0) / 0.5 = 40, yc: (35.0 - 20.0) / 0.5 = 30
        # Physical radius 10.0 -> Index radius 20 (10.0 / 0.5)
        expected = np.array([40.0, 30.0, 20.0])
        np.testing.assert_allclose(roi.coords, expected, rtol=1e-10)

        # Verify we can retrieve the same physical coords
        retrieved = roi.get_physical_coords(obj)
        np.testing.assert_allclose(retrieved, new_physical, rtol=1e-10)

    def test_set_physical_coords_physical_mode(self):
        """Test setting physical coords when ROI is in physical mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create circular ROI in physical mode
        roi = CircularROI([30.0, 35.0, 10.0], indices=False, title="Test Circle")

        # Set new physical coordinates
        new_physical = np.array([40.0, 45.0, 15.0])
        roi.set_physical_coords(obj, new_physical)

        # Verify coords were stored directly as floats
        assert roi.indices is False
        np.testing.assert_array_equal(roi.coords, new_physical)

        # Verify we can retrieve the same physical coords
        retrieved = roi.get_physical_coords(obj)
        np.testing.assert_allclose(retrieved, new_physical, rtol=1e-10)

    def test_set_indices_coords_indices_mode(self):
        """Test setting indices coords when ROI is in indices mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create circular ROI in indices mode
        roi = CircularROI([50, 50, 20], indices=True, title="Test Circle")

        # Set new indices coordinates
        new_indices = np.array([30, 40, 15])
        roi.set_indices_coords(obj, new_indices)

        # Verify coords were stored directly
        assert roi.indices is True
        np.testing.assert_array_equal(roi.coords, new_indices)

        # Verify we can retrieve the same indices coords
        retrieved = roi.get_indices_coords(obj)
        np.testing.assert_allclose(retrieved, new_indices, rtol=1e-10)

    def test_set_indices_coords_physical_mode(self):
        """Test setting indices coords when ROI is in physical mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create circular ROI in physical mode
        roi = CircularROI([30.0, 35.0, 10.0], indices=False, title="Test Circle")

        # Set new indices coordinates: center (40, 30), radius 20
        new_indices = np.array([40, 30, 20])
        roi.set_indices_coords(obj, new_indices)

        # Verify coords were converted to physical and stored
        assert roi.indices is False
        # Indices center (40, 30) -> Physical (30.0, 35.0)
        # xc: 40 * 0.5 + 10.0 = 30.0, yc: 30 * 0.5 + 20.0 = 35.0
        # Index radius 20 -> Physical radius 10.0 (20 * 0.5)
        expected_physical = np.array([30.0, 35.0, 10.0])
        np.testing.assert_allclose(roi.coords, expected_physical, rtol=1e-10)

        # Verify we can retrieve the same indices coords
        retrieved = roi.get_indices_coords(obj)
        np.testing.assert_allclose(retrieved, new_indices, rtol=1e-10)

    def test_roundtrip_physical_to_indices(self):
        """Test roundtrip conversion: physical -> indices -> physical"""
        obj = create_test_image(dx=2.0, dy=3.0, x0=5.0, y0=7.0)

        roi = CircularROI([45.0, 67.0, 20.0], indices=False, title="Test Circle")

        # Get physical coords
        original_physical = roi.get_physical_coords(obj)

        # Convert to indices and back
        indices = roi.get_indices_coords(obj)
        roi.set_indices_coords(obj, np.array(indices))
        retrieved_physical = roi.get_physical_coords(obj)

        # Should be identical (with some tolerance for floating point)
        np.testing.assert_allclose(
            retrieved_physical, original_physical, rtol=1e-10, atol=1e-10
        )

    def test_roundtrip_indices_to_physical(self):
        """Test roundtrip conversion: indices -> physical -> indices"""
        obj = create_test_image(dx=2.0, dy=3.0, x0=5.0, y0=7.0)

        roi = CircularROI([20, 15, 10], indices=True, title="Test Circle")

        # Get indices coords
        original_indices = roi.get_indices_coords(obj)

        # Convert to physical and back
        physical = roi.get_physical_coords(obj)
        roi.set_physical_coords(obj, np.array(physical))
        retrieved_indices = roi.get_indices_coords(obj)

        # Should be identical (with some tolerance for floating point)
        np.testing.assert_allclose(
            retrieved_indices, original_indices, rtol=1e-10, atol=1e-10
        )

    def test_anisotropic_pixels(self):
        """Test circular ROI with non-square pixels (dx != dy)"""
        # Create image with anisotropic pixels
        obj = create_test_image(dx=1.0, dy=2.0, x0=0.0, y0=0.0)

        # Create circular ROI in physical coordinates
        roi = CircularROI([50.0, 50.0, 10.0], indices=False, title="Anisotropic Circle")

        # Get indices (should account for different pixel sizes)
        indices = roi.get_indices_coords(obj)

        # Set indices and verify physical coords are preserved
        roi.set_indices_coords(obj, np.array(indices))
        retrieved_physical = roi.get_physical_coords(obj)

        np.testing.assert_allclose(retrieved_physical, [50.0, 50.0, 10.0], rtol=1e-10)


class TestPolygonalROISetters:
    """Test set_physical_coords and set_indices_coords for PolygonalROI"""

    def test_set_physical_coords_indices_mode(self):
        """Test setting physical coords when ROI is in indices mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create polygonal ROI in indices mode: triangle
        roi = PolygonalROI([10, 10, 30, 10, 20, 30], indices=True, title="Test Polygon")

        # Set new physical coordinates: a different triangle
        new_physical = np.array([15.0, 25.0, 25.0, 25.0, 20.0, 35.0])
        roi.set_physical_coords(obj, new_physical)

        # Verify coords were converted and stored as indices
        assert roi.indices is True
        # Physical to indices conversion
        expected = np.array([10, 10, 30, 10, 20, 30])
        np.testing.assert_allclose(roi.coords, expected, rtol=1e-10)

        # Verify we can retrieve the same physical coords
        retrieved = roi.get_physical_coords(obj)
        np.testing.assert_allclose(retrieved, new_physical, rtol=1e-10)

    def test_set_physical_coords_physical_mode(self):
        """Test setting physical coords when ROI is in physical mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create polygonal ROI in physical mode
        roi = PolygonalROI(
            [15.0, 25.0, 25.0, 25.0, 20.0, 35.0], indices=False, title="Test Polygon"
        )

        # Set new physical coordinates
        new_physical = np.array([20.0, 30.0, 30.0, 30.0, 25.0, 40.0])
        roi.set_physical_coords(obj, new_physical)

        # Verify coords were stored directly as floats
        assert roi.indices is False
        np.testing.assert_array_equal(roi.coords, new_physical)

        # Verify we can retrieve the same physical coords
        retrieved = roi.get_physical_coords(obj)
        np.testing.assert_allclose(retrieved, new_physical, rtol=1e-10)

    def test_set_indices_coords_indices_mode(self):
        """Test setting indices coords when ROI is in indices mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create polygonal ROI in indices mode
        roi = PolygonalROI([10, 10, 30, 10, 20, 30], indices=True, title="Test Polygon")

        # Set new indices coordinates: a quadrilateral
        new_indices = np.array([5, 5, 15, 5, 15, 15, 5, 15])
        roi.set_indices_coords(obj, new_indices)

        # Verify coords were stored directly
        assert roi.indices is True
        np.testing.assert_array_equal(roi.coords, new_indices)

        # Verify we can retrieve the same indices coords
        retrieved = roi.get_indices_coords(obj)
        np.testing.assert_array_equal(retrieved, new_indices)

    def test_set_indices_coords_physical_mode(self):
        """Test setting indices coords when ROI is in physical mode"""
        obj = create_test_image(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

        # Create polygonal ROI in physical mode
        roi = PolygonalROI(
            [15.0, 25.0, 25.0, 25.0, 20.0, 35.0], indices=False, title="Test Polygon"
        )

        # Set new indices coordinates
        new_indices = np.array([10, 10, 30, 10, 20, 30])
        roi.set_indices_coords(obj, new_indices)

        # Verify coords were converted to physical and stored
        assert roi.indices is False
        # Indices to physical conversion: multiply by pixel size and add origin
        # x: 10 * 0.5 + 10.0 = 15.0, 30 * 0.5 + 10.0 = 25.0, 20 * 0.5 + 10.0 = 20.0
        # y: 10 * 0.5 + 20.0 = 25.0, 10 * 0.5 + 20.0 = 25.0, 30 * 0.5 + 20.0 = 35.0
        expected_physical = np.array([15.0, 25.0, 25.0, 25.0, 20.0, 35.0])
        np.testing.assert_allclose(roi.coords, expected_physical, rtol=1e-10)

        # Verify we can retrieve the same indices coords
        retrieved = roi.get_indices_coords(obj)
        np.testing.assert_array_equal(retrieved, new_indices)

    def test_polygon_with_many_vertices(self):
        """Test polygon with many vertices"""
        obj = create_test_image(dx=1.0, dy=1.0, x0=0.0, y0=0.0)

        # Create a hexagon in physical mode
        hexagon_coords = [
            30.0,
            20.0,  # vertex 1
            40.0,
            20.0,  # vertex 2
            45.0,
            30.0,  # vertex 3
            40.0,
            40.0,  # vertex 4
            30.0,
            40.0,  # vertex 5
            25.0,
            30.0,  # vertex 6
        ]
        roi = PolygonalROI(hexagon_coords, indices=False, title="Hexagon")

        # Convert to indices and back
        indices = roi.get_indices_coords(obj)
        roi.set_indices_coords(obj, np.array(indices))
        retrieved_physical = roi.get_physical_coords(obj)

        np.testing.assert_allclose(retrieved_physical, hexagon_coords, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_size_rectangle(self):
        """Test rectangle with zero width or height"""
        obj = create_test_image()
        roi = RectangularROI([10, 10, 0, 20], indices=True, title="Zero Width")

        # Should handle gracefully
        physical = roi.get_physical_coords(obj)
        assert physical[2] == 0.0  # Width should be 0

        # Setting coords should work
        roi.set_physical_coords(obj, np.array([5.0, 5.0, 0.0, 10.0]))
        # [5.0, 5.0, 0.0, 10.0] means x0=5, y0=5, dx=0, dy=10
        # So corners are (5, 5) to (5, 15)
        # In indices (with dx=1, dy=1, x0=0, y0=0): (5, 5) to (5, 15)
        # Stored as [ix0, iy0, dx, dy] = [5, 5, 0, 10]
        assert roi.coords[0] == 5  # ix0
        assert roi.coords[1] == 5  # iy0
        assert roi.coords[2] == 0  # dx (zero width)
        assert roi.coords[3] == 10  # dy

    def test_negative_coordinates(self):
        """Test ROI with negative coordinates"""
        obj = create_test_image(dx=1.0, dy=1.0, x0=-50.0, y0=-50.0)

        # Create rectangle with negative physical coordinates
        roi = RectangularROI(
            [-10.0, -10.0, 20.0, 20.0], indices=False, title="Negative"
        )

        # Convert to indices and back
        indices = roi.get_indices_coords(obj)
        roi.set_indices_coords(obj, np.array(indices))
        retrieved_physical = roi.get_physical_coords(obj)

        np.testing.assert_allclose(retrieved_physical, [-10.0, -10.0, 20.0, 20.0])

    def test_large_offset_coordinates(self):
        """Test ROI with large coordinate offsets"""
        obj = create_test_image(dx=0.1, dy=0.1, x0=1000.0, y0=2000.0)

        # Create circle with large offset
        roi = CircularROI([1050.0, 2050.0, 5.0], indices=False, title="Large Offset")

        # Convert to indices and back
        indices = roi.get_indices_coords(obj)
        roi.set_indices_coords(obj, np.array(indices))
        retrieved_physical = roi.get_physical_coords(obj)

        np.testing.assert_allclose(retrieved_physical, [1050.0, 2050.0, 5.0])

    def test_backwards_drawn_rectangle(self):
        """Test rectangle drawn backwards (from bottom-right to top-left).

        When a rectangle is drawn graphically by dragging from bottom-right to
        top-left (instead of top-left to bottom-right), the x1 < x0 and y1 < y0.
        This should still produce valid ROI coordinates with positive Δx and Δy.

        Regression test for bug: backwards-drawn rectangles caused NaN statistics
        because negative Δx/Δy resulted in empty masks.
        """
        obj = create_test_image()

        # Simulate backwards drawing: start at (50, 50), drag to (10, 10)
        # This means x0=50, y0=50, x1=10, y1=10 from get_rect()
        coords = RectangularROI.rect_to_coords(50.0, 50.0, 10.0, 10.0)

        # Coordinates should be normalized: (10, 10, 40, 40)
        assert coords[0] == 10.0, f"x0 should be 10.0, got {coords[0]}"
        assert coords[1] == 10.0, f"y0 should be 10.0, got {coords[1]}"
        assert coords[2] == 40.0, f"Δx should be 40.0, got {coords[2]}"
        assert coords[3] == 40.0, f"Δy should be 40.0, got {coords[3]}"

        # Create ROI with these normalized coordinates
        roi = RectangularROI(coords.tolist(), indices=False, title="Backwards")

        # Mask should work correctly (not empty)
        mask = roi.to_mask(obj)
        # For normal (non-inverse) ROI, mask is False inside the ROI
        pixels_inside_roi = np.sum(~mask)
        assert pixels_inside_roi > 0, "ROI mask should contain pixels"

        # Verify bounding box is correct
        x0, y0, x1, y1 = roi.get_bounding_box(obj)
        assert x0 == 10.0 and y0 == 10.0 and x1 == 50.0 and y1 == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
