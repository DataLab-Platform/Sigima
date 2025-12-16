# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image ROI classes
=================

This module defines ROI (Region of Interest) classes and utilities for images.

The module provides:

- `ROI2DParam`: Parameter class for 2D image ROIs
- `BaseSingleImageROI`: Base class for single image ROIs
- `RectangularROI`: Rectangular ROI implementation
- `CircularROI`: Circular ROI implementation
- `PolygonalROI`: Polygonal ROI implementation
- `ImageROI`: Container for multiple image ROIs
- `create_image_roi`: Factory function for creating image ROIs
- `create_image_roi_around_points`: Function for creating image ROIs around points
- Utility functions for coordinate handling

These classes handle ROI definitions, parameter conversion, and mask generation
for image processing operations.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import abc
import re
from collections.abc import ByteString, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Type

import guidata.dataset as gds
import numpy as np
from skimage import draw

import sigima.enums
import sigima.tools.image
from sigima.config import _
from sigima.objects import base

if TYPE_CHECKING:
    from sigima.objects.image.object import ImageObj


def to_builtin(obj) -> str | int | float | list | dict | np.ndarray | None:
    """Convert an object implementing a numeric value or collection
    into the corresponding builtin/NumPy type.

    Return None if conversion fails."""
    try:
        return int(obj) if int(obj) == float(obj) else float(obj)
    except (TypeError, ValueError):
        pass
    if isinstance(obj, ByteString):
        return str(obj)
    if isinstance(obj, Sequence):
        return str(obj) if len(obj) == len(str(obj)) else list(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray):
        return obj
    return None


def check_points(value: np.ndarray, raise_exception: bool = False) -> bool:
    """Check if value is a valid 1D array of coordinates (X, Y) pairs.

    Args:
        value: value to check
        raise_exception: if True, raise an exception on invalid value

    Returns:
        True if value is valid, False otherwise
    """
    if not np.issubdtype(value.dtype, np.floating):
        if raise_exception:
            raise TypeError("Coordinates must be floats")
        return False
    if value.ndim != 1:
        if raise_exception:
            raise ValueError("Coordinates must be a 1D array")
        return False
    if len(value) % 2 != 0:
        if raise_exception:
            raise ValueError("Coordinates must contain pairs (X, Y)")
        return False
    return True


class ROI2DParam(base.BaseROIParam["ImageObj", "BaseSingleImageROI"]):
    """Image ROI parameters"""

    def get_comment(self) -> str | None:
        """Get comment for ROI parameters"""
        return _(
            "This is a set of parameters defining a <b>Region of Interest</b> "
            "(ROI) in an image. The parameters can be used to create a ROI object "
            "or to extract data from an image using the ROI.<br><br>"
            "All values are expressed in physical coordinates (floats). "
            "The conversion to pixel coordinates is done by taking into account "
            "the image pixel size and origin.<br>"
        )

    title = gds.StringItem(_("ROI title"), default="")

    # Note: the ROI coordinates are expressed in pixel coordinates (integers)
    # => That is the only way to handle ROI parametrization for image objects.
    #    Otherwise, we would have to ask the user to systematically provide the
    #    physical coordinates: that would be cumbersome and error-prone.

    _geometry_prop = gds.GetAttrProp("geometry")
    _rfp = gds.FuncProp(_geometry_prop, lambda x: x != "rectangle")
    _cfp = gds.FuncProp(_geometry_prop, lambda x: x != "circle")
    _pfp = gds.FuncProp(_geometry_prop, lambda x: x != "polygon")

    # Do not declare it as a static method: not supported by Python 3.9
    def _lbl(name: str, index: int):  # pylint: disable=no-self-argument
        """Returns name<sub>index</sub>"""
        return f"{name}<sub>{index}</sub>"

    geometries = ("rectangle", "circle", "polygon")
    geometry = gds.ChoiceItem(
        _("Geometry"), list(zip(geometries, geometries)), default="rectangle"
    ).set_prop("display", store=_geometry_prop, hide=True)

    # Parameters for rectangular ROI geometry:
    _tlcorner = gds.BeginGroup(_("Top left corner")).set_prop("display", hide=_rfp)
    x0 = gds.FloatItem(_lbl("X", 0), default=0).set_prop("display", hide=_rfp)
    y0 = (
        gds.FloatItem(_lbl("Y", 0), default=0).set_pos(1).set_prop("display", hide=_rfp)
    )
    _e_tlcorner = gds.EndGroup(_("Top left corner"))
    dx = gds.FloatItem("ΔX", default=0).set_prop("display", hide=_rfp)
    dy = gds.FloatItem("ΔY", default=0).set_pos(1).set_prop("display", hide=_rfp)

    # Parameters for circular ROI geometry:
    _cgroup = gds.BeginGroup(_("Center coordinates")).set_prop("display", hide=_cfp)
    xc = gds.FloatItem(_lbl("X", "C"), default=0).set_prop("display", hide=_cfp)
    yc = (
        gds.FloatItem(_lbl("Y", "C"), default=0)
        .set_pos(1)
        .set_prop("display", hide=_cfp)
    )
    _e_cgroup = gds.EndGroup(_("Center coordinates"))
    r = gds.FloatItem(_("Radius"), default=0).set_prop("display", hide=_cfp)

    # Parameters for polygonal ROI geometry:
    points = gds.FloatArrayItem(_("Coordinates"), check_callback=check_points).set_prop(
        "display", hide=_pfp
    )

    # Parameter for ROI mask behavior:
    inverse = gds.BoolItem(
        _("Inverse ROI logic"),
        default=False,
        help=_(
            "When disabled (default), the ROI defines an area inside the shape on "
            "which to focus (masking data outside).\n"
            "When enabled, the ROI logic is inverted to focus on data outside the "
            "shape (masking data inside)."
        ),
    )

    def to_single_roi(
        self, obj: ImageObj
    ) -> PolygonalROI | RectangularROI | CircularROI:
        """Convert parameters to single ROI

        Args:
            obj: image object (used for conversion of pixel to physical coordinates)

        Returns:
            Single ROI
        """
        if self.geometry == "rectangle":
            return RectangularROI.from_param(obj, self)
        if self.geometry == "circle":
            return CircularROI.from_param(obj, self)
        if self.geometry == "polygon":
            return PolygonalROI.from_param(obj, self)
        raise ValueError(f"Unknown ROI geometry type: {self.geometry}")

    def get_suffix(self) -> str:
        """Get suffix text representation for ROI extraction"""
        if re.match(base.GENERIC_ROI_TITLE_REGEXP, self.title) or not self.title:
            if self.geometry == "rectangle":
                return f"x0={self.x0},y0={self.y0},dx={self.dx},dy={self.dy}"
            if self.geometry == "circle":
                return f"xc={self.xc},yc={self.yc},r={self.r}"
            if self.geometry == "polygon":
                return "polygon"
            raise ValueError(f"Unknown ROI geometry type: {self.geometry}")
        return self.title

    def get_extracted_roi(self, obj: ImageObj) -> ImageROI | None:
        """Get extracted ROI, i.e. the remaining ROI after extracting ROI from image.

        Args:
            obj: image object (used for conversion of pixel to physical coordinates)

        When extracting ROIs from an image to multiple images (i.e. one image per ROI),
        this method returns the ROI that has to be kept in the destination image. This
        is not necessary for a rectangular ROI: the destination image is simply a crop
        of the source image according to the ROI coordinates. But for a circular ROI or
        a polygonal ROI, the destination image is a crop of the source image according
        to the bounding box of the ROI. Thus, to avoid any loss of information, a ROI
        has to be defined for the destination image: this is the ROI returned by this
        method. It's simply the same as the source ROI, but with coordinates adjusted
        to the destination image. One may called this ROI the "extracted ROI".
        """
        if self.geometry == "rectangle":
            return None
        single_roi = self.to_single_roi(obj)
        roi = ImageROI()
        roi.add_roi(single_roi)
        return roi

    def get_bounding_box_physical(self) -> tuple[int, int, int, int]:
        """Get bounding box (physical coordinates)"""
        if self.geometry == "circle":
            x0, y0 = self.xc - self.r, self.yc - self.r
            x1, y1 = self.xc + self.r, self.yc + self.r
        elif self.geometry == "rectangle":
            x0, y0, x1, y1 = self.x0, self.y0, self.x0 + self.dx, self.y0 + self.dy
        else:
            self.points: np.ndarray
            x0, y0 = self.points[::2].min(), self.points[1::2].min()
            x1, y1 = self.points[::2].max(), self.points[1::2].max()
        return x0, y0, x1, y1

    def get_bounding_box_indices(self, obj: ImageObj) -> tuple[int, int, int, int]:
        """Get bounding box (pixel coordinates)"""
        x0, y0, x1, y1 = self.get_bounding_box_physical()
        ix0, iy0 = obj.physical_to_indices((x0, y0))
        ix1, iy1 = obj.physical_to_indices((x1, y1))
        return ix0, iy0, ix1, iy1

    def get_data(self, obj: ImageObj) -> np.ndarray:
        """Get data in ROI

        Args:
            obj: image object

        Returns:
            Data in ROI
        """
        ix0, iy0, ix1, iy1 = self.get_bounding_box_indices(obj)
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(obj.data.shape[1], ix1), min(obj.data.shape[0], iy1)
        return obj.data[iy0:iy1, ix0:ix1]


class BaseSingleImageROI(base.BaseSingleROI["ImageObj", ROI2DParam], abc.ABC):
    """Base class for single image ROI

    Args:
        coords: ROI edge coordinates (floats)
        indices: if True, coordinates are indices, if False, they are physical values
        title: ROI title
        inverse: inverse ROI logic (default: False)

    .. note::

        The image ROI coords are expressed in physical coordinates (floats). The
        conversion to pixel coordinates is done in :class:`sigima.objects.ImageObj`
        (see :meth:`sigima.objects.ImageObj.physical_to_indices`). Most of the time,
        the physical coordinates are the same as the pixel coordinates, but this
        is not always the case (e.g. after image binning), so it's better to keep the
        physical coordinates in the ROI object: this will help reusing the ROI with
        different images (e.g. with different pixel sizes).
    """

    def __init__(
        self,
        coords: np.ndarray | list[int] | list[float],
        indices: bool,
        title: str = "ROI",
        inverse: bool = False,
    ) -> None:
        super().__init__(coords, indices, title)
        self.inverse = inverse

    def to_dict(self) -> dict:
        """Convert ROI to dictionary

        Returns:
            Dictionary
        """
        result = super().to_dict()
        result["inverse"] = self.inverse
        return result

    @classmethod
    def from_dict(cls, dictdata: dict):
        """Convert dictionary to ROI

        Args:
            dictdata: dictionary

        Returns:
            ROI
        """
        # Get inverse parameter
        inverse = dictdata.get("inverse", False)  # Default: normal ROI behavior
        return cls(dictdata["coords"], dictdata["indices"], dictdata["title"], inverse)

    @abc.abstractmethod
    def get_bounding_box(self, obj: ImageObj) -> tuple[float, float, float, float]:
        """Get bounding box (physical coordinates)

        Args:
            obj: image object
        """


class PolygonalROI(BaseSingleImageROI):
    """Polygonal ROI

    Args:
        coords: ROI edge coordinates
        title: title
        inverse: inverse ROI logic (default: False)

    Raises:
        ValueError: if number of coordinates is odd

    .. note:: The image ROI coords are expressed in physical coordinates (floats)
    """

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) % 2 != 0:
            raise ValueError("Edge indices must be pairs of X, Y values")

    # pylint: disable=unused-argument
    @classmethod
    def from_param(cls: PolygonalROI, obj: ImageObj, param: ROI2DParam) -> PolygonalROI:
        """Create ROI from parameters

        Args:
            obj: image object
            param: parameters
        """
        return cls(
            param.points,
            indices=False,
            title=param.title,
            inverse=param.inverse,
        )

    def get_bounding_box(self, obj: ImageObj) -> tuple[float, float, float, float]:
        """Get bounding box (physical coordinates)

        Args:
            obj: image object
        """
        coords = self.get_physical_coords(obj)
        x_edges, y_edges = coords[::2], coords[1::2]
        return min(x_edges), min(y_edges), max(x_edges), max(y_edges)

    def to_mask(self, obj: ImageObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        roi_mask = np.ones_like(obj.data, dtype=bool)
        indices = self.get_indices_coords(obj)
        rows = np.array(indices[1::2], dtype=float)  # y coordinates
        cols = np.array(indices[::2], dtype=float)  # x coordinates
        rr, cc = draw.polygon(rows, cols, shape=obj.data.shape)

        if self.inverse:
            # Inverse ROI: mask inside the polygon (True inside, False outside)
            roi_mask[:] = False
            roi_mask[rr, cc] = True
        else:
            # Normal ROI: mask outside the polygon (False inside, True outside)
            roi_mask[rr, cc] = False
        return roi_mask

    def to_param(self, obj: ImageObj, index: int) -> ROI2DParam:
        """Convert ROI to parameters

        Args:
            obj: object (image), for physical-indices coordinates conversion
            index: ROI index
        """
        gtitle = base.get_generic_roi_title(index)
        param = ROI2DParam(gtitle)
        param.title = self.title or gtitle
        param.geometry = "polygon"
        param.points = np.array(self.get_physical_coords(obj))
        param.inverse = self.inverse
        return param


class RectangularROI(BaseSingleImageROI):
    """Rectangular ROI

    Args:
        coords: ROI edge coordinates (x0, y0, dx, dy)
        title: title
        inverse: inverse ROI logic (default: False)

    .. note:: The image ROI coords are expressed in physical coordinates (floats)
    """

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) != 4:
            raise ValueError("Rectangle ROI requires 4 coordinates")

    # pylint: disable=unused-argument
    @classmethod
    def from_param(
        cls: RectangularROI, obj: ImageObj, param: ROI2DParam
    ) -> RectangularROI:
        """Create ROI from parameters

        Args:
            obj: image object
            param: parameters
        """
        x0, y0, x1, y1 = param.get_bounding_box_physical()
        return cls(
            [x0, y0, x1 - x0, y1 - y0],
            indices=False,
            title=param.title,
            inverse=param.inverse,
        )

    def get_bounding_box(self, obj: ImageObj) -> tuple[float, float, float, float]:
        """Get bounding box (physical coordinates)

        Args:
            obj: image object
        """
        x0, y0, dx, dy = self.get_physical_coords(obj)
        return x0, y0, x0 + dx, y0 + dy

    def get_physical_coords(self, obj: ImageObj) -> list[float]:
        """Return physical coords

        Args:
            obj: image object

        Returns:
            Physical coords
        """
        if self.indices:
            ix0, iy0, idx, idy = self.coords
            x0, y0, x1, y1 = obj.indices_to_physical([ix0, iy0, ix0 + idx, iy0 + idy])
            return [x0, y0, x1 - x0, y1 - y0]
        return self.coords

    def set_physical_coords(self, obj: ImageObj, coords: np.ndarray) -> None:
        """Set physical coords

        Args:
            obj: object (signal/image)
            coords: physical coords
        """
        if self.indices:
            x0, y0, dx, dy = coords
            ix0, iy0, ix1, iy1 = obj.physical_to_indices([x0, y0, x0 + dx, y0 + dy])
            self.coords = np.array([ix0, iy0, ix1 - ix0, iy1 - iy0], dtype=int)
        else:
            self.coords = np.array(coords, dtype=float)

    def get_indices_coords(self, obj: ImageObj) -> list[int]:
        """Return indices coords

        Args:
            obj: image object

        Returns:
            Indices coords
        """
        if self.indices:
            return self.coords.tolist()
        ix0, iy0, ix1, iy1 = obj.physical_to_indices(self.get_bounding_box(obj))
        return [ix0, iy0, ix1 - ix0, iy1 - iy0]

    def set_indices_coords(self, obj: ImageObj, coords: np.ndarray) -> None:
        """Set indices coords

        Args:
            obj: object (signal/image)
            coords: indices coords
        """
        if self.indices:
            self.coords = coords
        else:
            ix0, iy0, idx, idy = coords
            x0, y0, x1, y1 = obj.indices_to_physical([ix0, iy0, ix0 + idx, iy0 + idy])
            self.coords = np.array([x0, y0, x1 - x0, y1 - y0])

    def to_mask(self, obj: ImageObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        roi_mask = np.ones_like(obj.data, dtype=bool)
        ix0, iy0, idx, idy = self.get_indices_coords(obj)
        rr, cc = draw.rectangle((iy0, ix0), extent=(idy, idx), shape=obj.data.shape)

        if self.inverse:
            # Inverse ROI: mask inside the rectangle (True inside, False outside)
            roi_mask[:] = False
            roi_mask[rr, cc] = True
        else:
            # Normal ROI: mask outside the rectangle (False inside, True outside)
            roi_mask[rr, cc] = False
        return roi_mask

    def to_param(self, obj: ImageObj, index: int) -> ROI2DParam:
        """Convert ROI to parameters

        Args:
            obj: object (image), for physical-indices coordinates conversion
            index: ROI index
        """
        gtitle = base.get_generic_roi_title(index)
        param = ROI2DParam(gtitle)
        param.title = self.title or gtitle
        param.geometry = "rectangle"
        param.x0, param.y0, param.dx, param.dy = self.get_physical_coords(obj)
        param.inverse = self.inverse
        return param

    @staticmethod
    def rect_to_coords(
        x0: int | float, y0: int | float, x1: int | float, y1: int | float
    ) -> np.ndarray:
        """Convert rectangle to coordinates

        Args:
            x0: x0 (first corner)
            y0: y0 (first corner)
            x1: x1 (opposite corner)
            y1: y1 (opposite corner)

        Returns:
            Rectangle coordinates (x0, y0, Δx, Δy) with positive Δx and Δy

        Note:
            Coordinates are normalized so that Δx and Δy are always positive.
            This handles rectangles drawn "backwards" (from bottom-right to top-left).
        """
        # Normalize coordinates to ensure Δx and Δy are positive
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        return np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=type(x0))


class CircularROI(BaseSingleImageROI):
    """Circular ROI

    Args:
        coords: ROI edge coordinates (xc, yc, r)
        title: title
        inverse: inverse ROI logic (default: False)

    .. note:: The image ROI coords are expressed in physical coordinates (floats)
    """

    # pylint: disable=unused-argument
    @classmethod
    def from_param(cls: CircularROI, obj: ImageObj, param: ROI2DParam) -> CircularROI:
        """Create ROI from parameters

        Args:
            obj: image object
            param: parameters
        """
        x0, y0, x1, y1 = param.get_bounding_box_physical()
        ixc, iyc = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        ir = (x1 - x0) * 0.5
        return cls(
            [ixc, iyc, ir],
            indices=False,
            title=param.title,
            inverse=param.inverse,
        )

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) != 3:
            raise ValueError("Circle ROI requires 3 coordinates")

    def get_bounding_box(self, obj: ImageObj) -> tuple[float, float, float, float]:
        """Get bounding box (physical coordinates)

        Args:
            obj: image object
        """
        xc, yc, r = self.get_physical_coords(obj)
        return xc - r, yc - r, xc + r, yc + r

    def get_physical_coords(self, obj: ImageObj) -> np.ndarray:
        """Return physical coords

        Args:
            obj: image object

        Returns:
            Physical coords
        """
        if self.indices:
            ixc, iyc, ir = self.coords
            x0, y0, x1, y1 = obj.indices_to_physical(
                [ixc - ir, iyc - ir, ixc + ir, iyc + ir]
            )
            return [0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0)]
        return self.coords

    def set_physical_coords(self, obj: ImageObj, coords: np.ndarray) -> None:
        """Set physical coords

        Args:
            obj: object (signal/image)
            coords: physical coords
        """
        if self.indices:
            xc, yc, r = coords
            ix0, iy0, ix1, iy1 = obj.physical_to_indices(
                [xc - r, yc - r, xc + r, yc + r]
            )
            self.coords = np.array(
                [0.5 * (ix0 + ix1), 0.5 * (iy0 + iy1), 0.5 * (ix1 - ix0)]
            )
        else:
            self.coords = np.array(coords, dtype=float)

    def get_indices_coords(self, obj: ImageObj) -> list[float]:
        """Return indices coords

        Args:
            obj: image object

        Returns:
            Indices coords
        """
        if self.indices:
            return self.coords
        ix0, iy0, ix1, iy1 = obj.physical_to_indices(
            self.get_bounding_box(obj), as_float=True
        )
        ixc, iyc = (ix0 + ix1) * 0.5, (iy0 + iy1) * 0.5
        ir = (ix1 - ix0) * 0.5
        return [ixc, iyc, ir]

    def set_indices_coords(self, obj: ImageObj, coords: np.ndarray) -> None:
        """Set indices coords

        Args:
            obj: object (signal/image)
            coords: indices coords
        """
        if self.indices:
            self.coords = coords
        else:
            ixc, iyc, ir = coords
            x0, y0, x1, y1 = obj.indices_to_physical(
                [ixc - ir, iyc - ir, ixc + ir, iyc + ir]
            )
            self.coords = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0)])

    def to_mask(self, obj: ImageObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        roi_mask = np.ones_like(obj.data, dtype=bool)
        ixc, iyc, ir = self.get_indices_coords(obj)
        yxratio = obj.dy / obj.dx
        rr, cc = draw.ellipse(iyc, ixc, ir / yxratio, ir, shape=obj.data.shape)

        if self.inverse:
            # Inverse ROI: mask inside the circle (True inside, False outside)
            roi_mask[:] = False
            roi_mask[rr, cc] = True
        else:
            # Normal ROI: mask outside the circle (False inside, True outside)
            roi_mask[rr, cc] = False
        return roi_mask

    def to_param(self, obj: ImageObj, index: int) -> ROI2DParam:
        """Convert ROI to parameters

        Args:
            obj: object (image), for physical-indices coordinates conversion
            index: ROI index
        """
        gtitle = base.get_generic_roi_title(index)
        param = ROI2DParam(gtitle)
        param.title = self.title or gtitle
        param.geometry = "circle"
        param.xc, param.yc, param.r = self.get_physical_coords(obj)
        param.inverse = self.inverse
        return param

    @staticmethod
    def rect_to_coords(
        x0: int | float, y0: int | float, x1: int | float, y1: int | float
    ) -> np.ndarray:
        """Convert rectangle to circle coordinates

        Args:
            x0: x0 (top-left corner)
            y0: y0 (top-left corner)
            x1: x1 (bottom-right corner)
            y1: y1 (bottom-right corner)

        Returns:
            Circle coordinates (xc, yc, r)
        """
        xc, yc, r = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0)
        return np.array([xc, yc, r], dtype=type(x0))


class ImageROI(base.BaseROI["ImageObj", BaseSingleImageROI, ROI2DParam]):
    """Image Regions of Interest

    Args:
        inverse: if True, ROI is outside the region
    """

    PREFIX = "i"

    @staticmethod
    def get_compatible_single_roi_classes() -> list[Type[BaseSingleImageROI]]:
        """Return compatible single ROI classes"""
        return [RectangularROI, CircularROI, PolygonalROI]

    def to_mask(self, obj: ImageObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        if not self.single_rois:
            # If no single ROIs, the mask is empty (no ROI defined)
            mask = np.ones_like(obj.data, dtype=bool)
            mask.fill(False)
            return mask

        # Check inverse values to determine combination strategy
        inverse_values = [roi.inverse for roi in self.single_rois]
        all_normal = not any(inverse_values)  # All inverse=False (normal ROI)
        all_inverse = all(inverse_values)  # All inverse=True (inverse ROI)

        if all_normal:
            # All ROIs have inverse=False: AND them together (intersection)
            mask = np.ones_like(obj.data, dtype=bool)
            for roi in self.single_rois:
                mask &= roi.to_mask(obj)
        elif all_inverse:
            # All ROIs have inverse=True: OR them together (union)
            mask = np.zeros_like(obj.data, dtype=bool)
            for roi in self.single_rois:
                mask |= roi.to_mask(obj)
        else:
            # Mixed inverse values: complex combination
            # Start with all True (full mask)
            mask = np.ones_like(obj.data, dtype=bool)

            # First apply all inverse=False ROIs (intersections)
            for roi in self.single_rois:
                if not roi.inverse:
                    mask &= roi.to_mask(obj)

            # Then apply inverse=True ROIs (add their areas)
            inside_mask = np.zeros_like(obj.data, dtype=bool)
            for roi in self.single_rois:
                if roi.inverse:
                    inside_mask |= roi.to_mask(obj)

            # Combine: mask outside regions AND include inside regions
            mask = mask | inside_mask

        return mask


def create_image_roi(
    geometry: Literal["rectangle", "circle", "polygon"],
    coords: np.ndarray | list[float] | list[list[float]],
    indices: bool = False,
    title: str = "",
    inverse: bool | list[bool] = False,
) -> ImageROI:
    """Create Image Regions of Interest (ROI) object.
    More ROIs can be added to the object after creation, using the `add_roi` method.

    Args:
        geometry: ROI type ('rectangle', 'circle', 'polygon')
        coords: ROI coords (physical coordinates), `[x0, y0, dx, dy]` for a rectangle,
         `[xc, yc, r]` for a circle, or `[x0, y0, x1, y1, ...]` for a polygon (lists or
         NumPy arrays are accepted). For multiple ROIs, nested lists or NumPy arrays are
         accepted but with a common geometry type (e.g.
         `[[xc1, yc1, r1], [xc2, yc2, r2], ...]` for circles).
        indices: if True, coordinates are indices, if False, they are physical values
         (default to False)
        title: title
        inverse: ROI logic behavior. Controls whether the ROI logic is inversed
         (default: False, meaning normal ROI that focuses on data inside the shape).
         When True, the ROI logic is inversed to focus on data outside the shape.
         Can be a single boolean (applied to all ROIs) or a list of booleans
         (one per ROI for individual control).

    Returns:
        Regions of Interest (ROI) object

    Raises:
        ValueError: if ROI type is unknown, if the number of coordinates is invalid,
         or if the number of inverse values doesn't match the number of ROIs

    Examples:
        Create a single rectangle ROI (defaults to normal behavior):
        >>> roi = create_image_roi("rectangle", [10, 20, 30, 40])

        Create a single rectangle ROI with inverse logic explicitly:
        >>> roi = create_image_roi("rectangle", [10, 20, 30, 40], inverse=True)

        Create multiple rectangles with global inverse parameter:
        >>> coords = [[10, 20, 30, 40], [50, 60, 70, 80]]
        >>> roi = create_image_roi("rectangle", coords, inverse=False)

        Create multiple rectangles with individual inverse parameters:
        >>> coords = [[10, 20, 30, 40], [50, 60, 70, 80]]
        >>> inverse_values = [True, False]  # First inside, second outside
        >>> roi = create_image_roi(
        ...     "rectangle", coords, inverse=inverse_values
        ... )

        Create polygons with varying vertex counts:
        >>> polygon_coords = [[0, 0, 10, 0, 5, 8], [20, 20, 30, 20, 30, 30, 20, 30]]
        >>> inverse_values = [False, True]  # First outside, second inside
        >>> roi = create_image_roi(
        ...     "polygon", polygon_coords, inverse=inverse_values
        ... )
    """
    # Handle coordinates - try to create numpy array, fall back to list for irregular
    try:
        coords = np.array(coords, float)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        coord_list = coords
        coord_count = len(coords)
    except ValueError:
        # Handle irregular case: polygons with varying vertex counts
        coord_list = [np.array(coord, float) for coord in coords]
        coord_count = len(coord_list)

    # Handle inverse parameter - can be single value or list
    if isinstance(inverse, bool):
        inverse_values = [inverse] * coord_count
    else:
        inverse_values = list(inverse)
        if len(inverse_values) != coord_count:
            raise ValueError(
                f"Number of inverse values ({len(inverse_values)}) must "
                f"match number of ROIs ({coord_count})"
            )

    roi = ImageROI()
    if geometry == "rectangle":
        if isinstance(coord_list, np.ndarray) and coord_list.shape[1] != 4:
            raise ValueError("Rectangle ROI requires 4 coordinates")
        for coord_row, inverse_val in zip(coord_list, inverse_values):
            roi.add_roi(RectangularROI(coord_row, indices, title, inverse_val))
    elif geometry == "circle":
        if isinstance(coord_list, np.ndarray) and coord_list.shape[1] != 3:
            raise ValueError("Circle ROI requires 3 coordinates")
        for coord_row, inverse_val in zip(coord_list, inverse_values):
            roi.add_roi(CircularROI(coord_row, indices, title, inverse_val))
    elif geometry == "polygon":
        if isinstance(coord_list, np.ndarray) and coord_list.shape[1] % 2 != 0:
            raise ValueError("Polygon ROI requires pairs of X, Y coordinates")
        for coord_row, inverse_val in zip(coord_list, inverse_values):
            roi.add_roi(PolygonalROI(coord_row, indices, title, inverse_val))
    else:
        raise ValueError(f"Unknown ROI type: {geometry}")
    return roi


def create_image_roi_around_points(
    coords: np.ndarray, roi_geometry: sigima.enums.DetectionROIGeometry
) -> ImageROI:
    """Create ROIs around given point coordinates.

    Args:
        coords: Coordinates of points (shape: (n, 2))
        roi_geometry: ROI geometry type (rectangle or circle)

    Returns:
        ImageROI object containing rectangles or circles around each point

    Raises:
        ValueError: If less than 2 points are provided (cannot determine ROI size),
         or if points are too close together resulting in too small ROI size,
         or if an invalid ROI geometry is specified.
    """
    assert roi_geometry in ("rectangle", "circle")
    if coords.size == 0:
        raise ValueError("No coordinates provided to create ROIs")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"Coordinates array must have shape (n, 2), got {coords.shape}"
        )
    if coords.shape[0] < 2:
        raise ValueError(
            "At least two points are required to automatically determine ROI size"
        )

    # Calculate ROI size based on minimum distance between points
    dist = sigima.tools.image.distance_matrix(coords)
    dist_min = dist[dist != 0].min()
    assert dist_min > 0
    if roi_geometry == "rectangle":
        # For rectangles, account for diagonal to avoid overlap
        radius = int(0.5 * dist_min / np.sqrt(2) - 1)
    else:  # circle
        # For circles, use half the minimum distance directly
        radius = int(0.5 * dist_min - 1)
    if radius < 1:
        raise ValueError(
            "Calculated ROI size is too small. Points may be too close together."
        )
    roi_coords = []
    for x, y in coords:
        if roi_geometry == "rectangle":
            x0, y0 = x - radius, y - radius
            dx, dy = 2 * radius, 2 * radius
            roi_coords.append([x0, y0, dx, dy])
        else:  # circle
            roi_coords.append([x, y, radius])
    return create_image_roi(roi_geometry, roi_coords, indices=True)
