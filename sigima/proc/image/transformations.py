# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry transformations module
===============================

This module provides a unified interface for applying geometric transformations
to both geometry results (:class:`sigima.objects.GeometryResult`) and ROI objects
using the shape coordinate system (:mod:`sigima.objects.shape`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sigima.objects.annotations import transform_annotation
from sigima.objects.scalar import GeometryResult, KindShape
from sigima.objects.shape import (
    CircleCoordinates,
    EllipseCoordinates,
    PointCoordinates,
    PolygonCoordinates,
    RectangleCoordinates,
    SegmentCoordinates,
)

if TYPE_CHECKING:
    from sigima.objects import CircularROI, ImageObj, PolygonalROI, RectangularROI


__all__ = [
    "GeometryTransformer",
    "transformer",
]


class GeometryTransformer:
    """
    Singleton class for applying transformations to geometry objects.

    Provides a unified interface for transforming both GeometryResult and ROI
    objects using the shape coordinate system.
    """

    _instance: GeometryTransformer | None = None

    def __new__(cls) -> GeometryTransformer:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the transformer (only once due to singleton)."""
        if self._initialized:  # pylint: disable=access-member-before-definition
            return

        # Mapping from GeometryResult kinds to shape coordinate classes
        self._geometry_shape_map: dict[
            KindShape,
            type[
                RectangleCoordinates
                | CircleCoordinates
                | PolygonCoordinates
                | SegmentCoordinates
            ],
        ] = {
            KindShape.POINT: PointCoordinates,
            KindShape.RECTANGLE: RectangleCoordinates,
            KindShape.CIRCLE: CircleCoordinates,
            KindShape.ELLIPSE: EllipseCoordinates,
            KindShape.POLYGON: PolygonCoordinates,
            KindShape.SEGMENT: SegmentCoordinates,
            KindShape.MARKER: PointCoordinates,
        }

        # Mapping from ROI types to shape coordinate classes (lazy loaded)
        self._roi_shape_map: dict[type, type] = {}

        self._initialized = True

    def _get_roi_shape_map(
        self,
    ) -> dict[
        type[CircularROI | RectangularROI | PolygonalROI],
        type[RectangleCoordinates | CircleCoordinates | PolygonCoordinates],
    ]:
        """Lazy load ROI shape mapping to avoid circular imports."""
        if not self._roi_shape_map:
            # pylint: disable=import-outside-toplevel
            from sigima.objects.image import CircularROI, PolygonalROI, RectangularROI

            self._roi_shape_map = {
                RectangularROI: RectangleCoordinates,
                CircularROI: CircleCoordinates,
                PolygonalROI: PolygonCoordinates,
            }
        return self._roi_shape_map

    def transform_geometry(
        self, geometry: GeometryResult, operation: str, **kwargs: Any
    ) -> GeometryResult:
        """
        Transform a GeometryResult and return a new one.

        Args:
            geometry: The GeometryResult to transform.
            operation: Operation name ('rotate', 'translate', 'fliph', 'flipv',
                      'transpose', 'scale').
            **kwargs: Operation-specific parameters.

        Returns:
            New GeometryResult with transformed coordinates.

        Raises:
            ValueError: If operation is unknown or geometry kind is unsupported.
        """
        coord_class = self._geometry_shape_map.get(geometry.kind)
        if coord_class is None:
            raise ValueError(f"Unsupported geometry kind: {geometry.kind}")

        # Transform each row of coordinates
        transformed_coords = []
        for row in geometry.coords:
            # Create coordinate object for this row
            shape_coords = coord_class(row.copy())

            # Apply transformation
            self._apply_operation(shape_coords, operation, **kwargs)

            transformed_coords.append(shape_coords.data)

        # Create new GeometryResult with transformed coordinates
        return GeometryResult(
            title=geometry.title,
            kind=geometry.kind,
            coords=np.array(transformed_coords),
            roi_indices=(
                geometry.roi_indices.copy()
                if geometry.roi_indices is not None
                else None
            ),
            attrs=geometry.attrs.copy(),
            func_name=geometry.func_name,
        )

    def transform_single_roi(
        self,
        single_roi: RectangularROI | CircularROI | PolygonalROI,
        operation: str,
        **kwargs: Any,
    ) -> None:
        """
        Transform ROI coordinates inplace.

        Args:
            single_roi: ROI object with .coords attribute.
            operation: Operation name.
            **kwargs: Operation-specific parameters.

        Raises:
            ValueError: If ROI type is unsupported or operation is unknown.
        """
        roi_shape_map = self._get_roi_shape_map()
        coord_class = roi_shape_map.get(type(single_roi))
        if coord_class is None:
            raise ValueError(f"Unsupported ROI type: {type(single_roi)}")

        # Create shape coordinates and transform
        shape_coords = coord_class(single_roi.coords.copy())
        self._apply_operation(shape_coords, operation, **kwargs)

        # Update ROI coordinates inplace
        single_roi.coords[:] = shape_coords.data

    def transform_roi(self, image: ImageObj, operation: str, **kwargs: Any) -> None:
        """
        Transform all ROI coordinates in an ImageObj inplace.

        Args:
            image: Image object whose ROI coordinates will be transformed.
            operation: Operation name.
            **kwargs: Operation-specific parameters.
        """
        if image.roi is None or image.roi.is_empty():
            return

        # Import here to avoid circular imports
        # pylint: disable=import-outside-toplevel
        from sigima.objects.image import ImageROI

        # Determine ROI type and set up appropriate classes
        new_roi = ImageROI()

        # Transform each single ROI
        for single_roi in image.roi.single_rois:
            coords = single_roi.coords.copy()
            roi_class = single_roi.__class__

            # Create shape coordinates and transform
            roi_shape_map = self._get_roi_shape_map()
            coord_class = roi_shape_map.get(roi_class)
            if coord_class is None:
                raise ValueError(f"Unsupported ROI type: {roi_class}")

            shape_coords = coord_class(coords)
            self._apply_operation(shape_coords, operation, **kwargs)

            new_coords = shape_coords.data
            new_single_roi = roi_class(new_coords, single_roi.indices, single_roi.title)
            new_roi.add_roi(new_single_roi)

        image.roi = new_roi

    def transform_annotations(
        self, image: ImageObj, operation: str, **kwargs: Any
    ) -> None:
        """Transform all canonical annotations of an image inplace.

        Opaque application-specific annotation payloads are preserved unchanged.

        Args:
            image: Image object whose canonical annotations will be transformed.
            operation: Operation name.
            **kwargs: Operation-specific parameters.
        """
        annotations = image.get_graphical_annotations()
        if not annotations:
            return
        transformed = [
            transform_annotation(annotation, operation, **kwargs)
            for annotation in annotations
        ]
        image.set_graphical_annotations(transformed, preserve_opaque=True)

    def _apply_operation(
        self,
        shape_coords: (
            PointCoordinates
            | RectangleCoordinates
            | CircleCoordinates
            | EllipseCoordinates
            | PolygonCoordinates
            | SegmentCoordinates
        ),
        operation: str,
        **kwargs: Any,
    ) -> None:
        """
        Apply the specified operation to shape coordinates.

        Args:
            shape_coords: Shape coordinate object to transform.
            operation: Operation name.
            **kwargs: Operation-specific parameters.

        Raises:
            ValueError: If operation is unknown.
        """
        if operation == "rotate":
            angle = kwargs.get("angle", 0)
            center = kwargs.get("center", (0, 0))
            shape_coords.rotate(angle, center)
        elif operation == "translate":
            dx = kwargs.get("dx", 0)
            dy = kwargs.get("dy", 0)
            shape_coords.translate(dx, dy)
        elif operation == "fliph":
            cx = kwargs.get("cx", 0.0)
            shape_coords.fliph(cx)
        elif operation == "flipv":
            cy = kwargs.get("cy", 0.0)
            shape_coords.flipv(cy)
        elif operation == "transpose":
            shape_coords.transpose()
        elif operation == "scale":
            sx = kwargs.get("sx", 1.0)
            sy = kwargs.get("sy", 1.0)
            center = kwargs.get("center", (0, 0))
            shape_coords.scale(sx, sy, center)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # Convenience methods for common operations
    def rotate(
        self,
        obj: GeometryResult | RectangularROI | CircularROI | PolygonalROI,
        angle: float,
        center: tuple[float, float],
    ) -> GeometryResult | None:
        """
        Rotate geometry or ROI by given angle around center.

        Args:
            obj: GeometryResult or single ROI object.
            angle: Rotation angle in radians.
            center: Center of rotation (x, y).

        Returns:
            New GeometryResult if input was GeometryResult, None if ROI (inplace).
        """
        if isinstance(obj, GeometryResult):
            return self.transform_geometry(obj, "rotate", angle=angle, center=center)
        self.transform_single_roi(obj, "rotate", angle=angle, center=center)
        return None

    def translate(
        self,
        obj: GeometryResult | RectangularROI | CircularROI | PolygonalROI,
        dx: float,
        dy: float,
    ) -> GeometryResult | None:
        """
        Translate geometry or ROI by given offset.

        Args:
            obj: GeometryResult or single ROI object.
            dx: Translation in x direction.
            dy: Translation in y direction.

        Returns:
            New GeometryResult if input was GeometryResult, None if ROI (inplace).
        """
        if isinstance(obj, GeometryResult):
            return self.transform_geometry(obj, "translate", dx=dx, dy=dy)
        self.transform_single_roi(obj, "translate", dx=dx, dy=dy)
        return None

    def fliph(
        self,
        obj: GeometryResult | RectangularROI | CircularROI | PolygonalROI,
        cx: float,
    ) -> GeometryResult | None:
        """
        Flip geometry or ROI horizontally around given x-coordinate.

        Args:
            obj: GeometryResult or single ROI object.
            cx: X-coordinate of flip axis.

        Returns:
            New GeometryResult if input was GeometryResult, None if ROI (inplace).
        """
        if isinstance(obj, GeometryResult):
            return self.transform_geometry(obj, "fliph", cx=cx)
        self.transform_single_roi(obj, "fliph", cx=cx)
        return None

    def flipv(
        self,
        obj: GeometryResult | RectangularROI | CircularROI | PolygonalROI,
        cy: float,
    ) -> GeometryResult | None:
        """
        Flip geometry or ROI vertically around given y-coordinate.

        Args:
            obj: GeometryResult or single ROI object.
            cy: Y-coordinate of flip axis.

        Returns:
            New GeometryResult if input was GeometryResult, None if ROI (inplace).
        """
        if isinstance(obj, GeometryResult):
            return self.transform_geometry(obj, "flipv", cy=cy)
        self.transform_single_roi(obj, "flipv", cy=cy)
        return None

    def transpose(
        self, obj: GeometryResult | RectangularROI | CircularROI | PolygonalROI
    ) -> GeometryResult | None:
        """
        Transpose geometry or ROI (swap x and y coordinates).

        Args:
            obj: GeometryResult or single ROI object.

        Returns:
            New GeometryResult if input was GeometryResult, None if ROI (inplace).
        """
        if isinstance(obj, GeometryResult):
            return self.transform_geometry(obj, "transpose")
        self.transform_single_roi(obj, "transpose")
        return None

    def scale(
        self,
        obj: GeometryResult | RectangularROI | CircularROI | PolygonalROI,
        sx: float,
        sy: float,
        center: tuple[float, float],
    ) -> GeometryResult | None:
        """
        Scale geometry or ROI by given factors around center.

        Args:
            obj: GeometryResult or single ROI object.
            sx: Scale factor in x direction.
            sy: Scale factor in y direction.
            center: Center of scaling (x, y).

        Returns:
            New GeometryResult if input was GeometryResult, None if ROI (inplace).
        """
        if isinstance(obj, GeometryResult):
            return self.transform_geometry(obj, "scale", sx=sx, sy=sy, center=center)
        self.transform_single_roi(obj, "scale", sx=sx, sy=sy, center=center)
        return None


#: Global singleton instance of GeometryTransformer for applying geometric
#: transformations to geometry results and ROI objects.
transformer = GeometryTransformer()
