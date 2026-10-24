# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry transformations module
===============================

This module provides a unified interface for applying geometric transformations
to both geometry results (:class:`sigima.objects.GeometryResult`) and ROI objects
using the shape coordinate system (:mod:`sigima.objects.shape`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

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
    "IMAGE_OPERATIONS",
    "GeometryTransformer",
    "image_axis_mirror",
    "image_operation_steps",
    "transformer",
]

#: Exact image operations (pixels are moved, not resampled) whose effect on
#: coordinates is given by :func:`image_operation_steps`
IMAGE_OPERATIONS = ("fliph", "flipv", "transpose", "rotate90", "rotate270")


def image_axis_mirror(
    image: ImageObj, axis: Literal["x", "y"]
) -> Callable[[np.ndarray], np.ndarray]:
    """Return how the reversal of the pixels along an axis moves coordinates.

    The pixel ``i`` (centered on its coordinate) receives the value of the pixel
    ``n - 1 - i``: a point at fractional index ``u`` moves to index ``n - 1 - u``.
    For uniform coordinates, this is the mirror symmetry about the center of the
    pixel grid (:attr:`ImageObj.xc` or :attr:`ImageObj.yc`).

    Args:
        image: image whose pixels are reversed (its coordinates are kept)
        axis: ``"x"`` (columns) or ``"y"`` (rows)

    Returns:
        Function mapping physical coordinates along the axis
    """
    size = image.data.shape[1 if axis == "x" else 0]

    def mirror(values: np.ndarray) -> np.ndarray:
        indices = image.axis_to_float_indices(values, axis)
        return image.axis_to_physical((size - 1) - indices, axis)

    return mirror


def _mirror_step(
    src: ImageObj, src_axis: Literal["x", "y"], axis: Literal["x", "y"]
) -> tuple[str, dict[str, Any]]:
    """Return the step mirroring coordinates along ``axis`` by the reversal of the
    pixels of ``src`` along ``src_axis`` (the two differ after a transposition)."""
    if src.is_uniform_coords:
        if axis == "x":
            return "fliph", {"cx": src.xc if src_axis == "x" else src.yc}
        return "flipv", {"cy": src.xc if src_axis == "x" else src.yc}
    return "remap", {"f" + axis: image_axis_mirror(src, src_axis)}


def image_operation_steps(
    src: ImageObj, operation: str
) -> list[tuple[str, dict[str, Any]]]:
    """Return how an exact image operation moves physical coordinates.

    The steps move a point of ``src`` to the position of the same pixel content in
    the result of the operation, for ROIs, annotations and geometry results.

    For uniform coordinates, the steps are affine and move every shape exactly.
    For non-uniform coordinates, reversing the pixels along a non-uniform axis is
    not an affine mapping of the coordinates: points, bounds and unrotated
    rectangles are moved exactly, but a circle does not stay a circle and the
    edges of a polygon do not stay straight, so that these shapes are moved
    approximately (see :meth:`~sigima.objects.shape.BaseCoordinates.remap`).

    Args:
        src: source image of the operation
        operation: one of :data:`IMAGE_OPERATIONS` (``numpy.rot90`` is a
         transposition followed by a vertical reversal, and three quarter turns a
         transposition followed by a horizontal reversal)

    Returns:
        List of (operation, keyword arguments) for :class:`GeometryTransformer`

    Raises:
        ValueError: if the operation is unknown
    """
    if operation == "fliph":
        return [_mirror_step(src, "x", "x")]
    if operation == "flipv":
        return [_mirror_step(src, "y", "y")]
    if operation == "transpose":
        return [("transpose", {})]
    if operation == "rotate90":
        return [("transpose", {}), _mirror_step(src, "x", "y")]
    if operation == "rotate270":
        return [("transpose", {}), _mirror_step(src, "y", "x")]
    raise ValueError(f"Unknown image operation: {operation}")


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
            new_single_roi = roi_class(
                new_coords, single_roi.indices, single_roi.title, single_roi.inverse
            )
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
        elif operation == "remap":
            shape_coords.remap(kwargs.get("fx"), kwargs.get("fy"))
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def transform_like_image(
        self, geometry: GeometryResult, src: ImageObj, operation: str
    ) -> GeometryResult:
        """Move a geometry result of an image as an exact image operation moves the
        pixels (see :func:`image_operation_steps`).

        Args:
            geometry: geometry result computed on ``src``
            src: source image of the operation
            operation: one of :data:`IMAGE_OPERATIONS`

        Returns:
            New geometry result, expressed in the coordinates of the result image
        """
        for step, kwargs in image_operation_steps(src, operation):
            geometry = self.transform_geometry(geometry, step, **kwargs)
        return geometry

    def apply_image_operation(
        self, dst: ImageObj, src: ImageObj, operation: str
    ) -> None:
        """Move the ROIs and canonical annotations of the result of an exact image
        operation with its pixels (they are copied from ``src`` beforehand).

        Args:
            dst: result image, holding the ROIs and annotations of ``src``
            src: source image of the operation
            operation: one of :data:`IMAGE_OPERATIONS`
        """
        if dst.roi is not None and any(r.indices for r in dst.roi.single_rois):
            # Steps move physical coordinates: express ROIs defined in indices in
            # the physical coordinates of the source image first
            # pylint: disable=import-outside-toplevel
            from sigima.objects.image import ImageROI

            physical_roi = ImageROI()
            for single_roi in dst.roi.single_rois:
                physical_roi.add_roi(
                    single_roi.__class__(
                        np.array(single_roi.get_physical_coords(src), dtype=float),
                        False,
                        single_roi.title,
                        single_roi.inverse,
                    )
                )
            dst.roi = physical_roi
        for step, kwargs in image_operation_steps(src, operation):
            self.transform_roi(dst, step, **kwargs)
            self.transform_annotations(dst, step, **kwargs)

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
