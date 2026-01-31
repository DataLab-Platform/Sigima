# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Coordinates objects module
==========================

Coordinate objects are 2D geometric entities that are used to represent and manipulate
shapes in a two-dimensional space. Coordinates are typically defined in the physical
coordinate system (as opposed to the image pixel coordinate system).
Geometric transformations, such as translation, rotation, and scaling, can be applied
to those objects to manipulate their position and size.

Applications
------------

Coordinate objects have two main applications:

1. ROIs (Regions of Interest) for image processing tasks.
2. Geometry results of signal and image processing tasks.

In both cases, geometric transformations can be applied to coordinate objects (rotation,
translation, horizontal flipping, vertical flipping, transposition, and scaling), in
order to apply the same transformation as the one applied to the image or signal data.

.. note::

    Geometric transformations currently support only the necessary transformations
    for image processing (not for signal processing).

Design choices
--------------

Coordinate objects are built around a `data` attribute that holds the characteristics
defining the shape in the form of a 1D array:

- (x, y) coordinates are stored in a flattened format: (x1, y1, x2, y2, ..., xn, yn)
- Point coordinates are represented as (x, y).
- Segment coordinates are represented as (x1, y1, x2, y2),
  where (x1, y1) is the start point and (x2, y2) is the end point.
- Rectangle coordinates are represented as (x0, y0, dx, dy),
  where (x0, y0) is the top-left corner and (dx, dy) are the width and height.
- Circle coordinates are represented as (x, y, r),
  where (x, y) is the center and r is the radius.
- Ellipse coordinates are represented as (x, y, a, b),
  where (x, y) is the center, a is the semi-major axis, and b is the semi-minor axis.
- Polygon coordinates are represented as a flattened array of (x, y) pairs:
  (x1, y1, x2, y2, ..., xn, yn).

At construction, the coordinates are validated to ensure they conform to the expected
format for each shape type.
"""

from __future__ import annotations

import abc

import numpy as np

from sigima.objects.base import HTML_TABLE_CSS


class BaseCoordinates(abc.ABC):
    """Base class for 2D coordinates representation of shapes.

    Args:
        data: 1D array or list of characteristics defining the shape.
    """

    VALID_SHAPE: tuple[int, int] | None = None  # To be defined by subclasses
    REQUIRES_EVEN_NUMBER_OF_VALUES: bool = False

    def __init__(self, points: np.ndarray | list[int] | list[float]):
        self.data = np.array(points, dtype=float)
        self.validate()

    def validate(self) -> None:
        """Validate the coordinates.

        Raises:
            ValueError: If the coordinates are invalid.
        """
        if self.data.ndim != 1:
            raise ValueError(
                f"Invalid {self.__class__.__name__} coordinates ndim: "
                f"{self.data.ndim} (expected 1)"
            )
        if self.VALID_SHAPE is not None and self.data.shape != self.VALID_SHAPE:
            raise ValueError(
                f"Invalid {self.__class__.__name__} coordinates shape: "
                f"{self.data.shape} (expected {self.VALID_SHAPE})"
            )
        if self.REQUIRES_EVEN_NUMBER_OF_VALUES and self.data.size % 2 != 0:
            raise ValueError(
                f"Invalid {self.__class__.__name__} coordinates: "
                f"even number of values expected (got {self.data.size})"
            )

    def transform_affine(self, matrix: np.ndarray) -> None:
        """Apply a 2D affine transformation to the coordinates inplace.

        Args:
            matrix: 3x3 affine transformation matrix.
        """
        coords = self.data
        if coords.size % 2 == 0:
            pts = coords.reshape(-1, 2)
            homo = np.c_[pts, np.ones(len(pts))]
            out = (homo @ matrix.T)[:, :2].reshape(-1)
            self.data[:] = out
        else:
            pts = coords[:2].reshape(1, 2)
            homo = np.c_[pts, np.ones(1)]
            out = (homo @ matrix.T)[:, :2].reshape(-1)
            self.data[:2] = out

    def copy(self) -> BaseCoordinates:
        """Return a copy of the coordinate object.

        Returns:
            BaseCoordinates: A new object with the same data.
        """
        return self.__class__(self.data.copy())

    def rotate(self, angle: float, center: tuple[float, float] = (0, 0)) -> None:
        """Rotate coordinates by a given angle around a center inplace.

        Args:
            angle: Rotation angle in radians (counterclockwise).
            center: Center of rotation (x, y). Defaults to (0, 0).
        """
        cx, cy = center
        c, s = np.cos(angle), np.sin(angle)
        t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
        t2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], float)
        matrix = t2 @ r @ t1
        self.transform_affine(matrix)

    def translate(self, dx: float, dy: float) -> None:
        """Translate coordinates by (dx, dy) inplace.

        Args:
            dx: Translation along x-axis.
            dy: Translation along y-axis.
        """
        matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)
        self.transform_affine(matrix)

    def fliph(self, cx: float = 0.0) -> None:
        """Flip coordinates horizontally around a vertical line x=cx inplace.

        Args:
            cx: x-coordinate of the vertical axis. Defaults to 0.0.
        """
        matrix = np.array([[-1, 0, 2 * cx], [0, 1, 0], [0, 0, 1]], float)
        self.transform_affine(matrix)

    def flipv(self, cy: float = 0.0) -> None:
        """Flip coordinates vertically around a horizontal line y=cy inplace.

        Args:
            cy: y-coordinate of the horizontal axis. Defaults to 0.0.
        """
        matrix = np.array([[1, 0, 0], [0, -1, 2 * cy], [0, 0, 1]], float)
        self.transform_affine(matrix)

    def transpose(self) -> None:
        """Transpose coordinates (swap x and y) inplace."""
        matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)
        self.transform_affine(matrix)

    def scale(self, sx: float, sy: float, center: tuple[float, float] = (0, 0)) -> None:
        """Scale coordinates by (sx, sy) around a center inplace.

        Args:
            sx: Scaling factor along x-axis.
            sy: Scaling factor along y-axis.
            center: Center of scaling (x, y). Defaults to (0, 0).
        """
        cx, cy = center
        t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
        s = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)
        t2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], float)
        matrix = t2 @ s @ t1
        self.transform_affine(matrix)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks."""
        coord_type = type(self).__name__

        # Format the data array nicely
        if self.data.size <= 10:
            data_str = ", ".join(f"{v:.4g}" for v in self.data)
        else:
            first = ", ".join(f"{v:.4g}" for v in self.data[:5])
            last = ", ".join(f"{v:.4g}" for v in self.data[-5:])
            data_str = f"{first}, ..., {last}"

        html = f"""
        {HTML_TABLE_CSS}
        <div class="sigima-html-container">
            <div class="sigima-html-title">{coord_type}</div>
            <table class="sigima-html-table">
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Size</td><td>{self.data.size} values</td></tr>
                <tr><td>Data</td><td>[{data_str}]</td></tr>
            </table>
        </div>
        """
        return html


class PointCoordinates(BaseCoordinates):
    """Class for point coordinates.

    Args:
        data: 1D array point coordinates (x, y).
    """

    VALID_SHAPE = (2,)


class SegmentCoordinates(BaseCoordinates):
    """Class for segment coordinates.

    Args:
        data: 1D array point coordinates (x1, y1, x2, y2).
    """

    VALID_SHAPE = (4,)


class RectangleCoordinates(BaseCoordinates):
    """Class for a rectangle coordinates.

    Args:
        data: 1D array point coordinates (x0, y0, dx, dy).
    """

    VALID_SHAPE = (4,)

    def transform_affine(self, matrix: np.ndarray) -> None:
        """
        Apply 2D affine transformation to rectangle coordinates.

        For rectangles in (x0, y0, dx, dy) format, we transform the corner points
        and then convert back to (x0, y0, dx, dy) format.

        Args:
            matrix: 3x3 affine transformation matrix.
        """
        x0, y0, dx, dy = self.data

        # Convert to corner coordinates
        x1, y1 = x0, y0
        x2, y2 = x0 + dx, y0 + dy

        # Transform both corners
        corners = np.array([[x1, y1], [x2, y2]])
        homo = np.c_[corners, np.ones(2)]
        transformed = (homo @ matrix.T)[:, :2]

        # Get the new bounding box
        x_min, y_min = transformed.min(axis=0)
        x_max, y_max = transformed.max(axis=0)

        # Update data with new (x0, y0, dx, dy)
        self.data[:] = [x_min, y_min, x_max - x_min, y_max - y_min]

    def rotate(self, angle: float, center: tuple[float, float] = (0, 0)) -> None:
        """Rotate rectangle by a given angle around a center.

        For rectangles, rotation may change the bounding box, so we transform
        all corners and compute the new axis-aligned bounding rectangle.

        Args:
            angle: Rotation angle in radians (counterclockwise).
            center: Center of rotation (x, y). Defaults to (0, 0).
        """
        cx, cy = center
        c, s = np.cos(angle), np.sin(angle)
        t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
        t2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], float)
        matrix = t2 @ r @ t1
        self.transform_affine(matrix)


class CircleCoordinates(BaseCoordinates):
    """Class for a circle coordinates.

    Args:
        data: 1D array point coordinates (x, y, r).
    """

    VALID_SHAPE = (3,)


class EllipseCoordinates(BaseCoordinates):
    """Class for an ellipse coordinates.

    Args:
        data: 1D array point coordinates (x, y, a, b).
    """

    VALID_SHAPE = (4,)


class PolygonCoordinates(BaseCoordinates):
    """Class for a polygon coordinates.

    Args:
        data: 1D array point coordinates (x1, y1, x2, y2, ..., xn, yn).
    """

    VALID_SHAPE = None
    REQUIRES_EVEN_NUMBER_OF_VALUES = True
