# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal ROI utilities
====================

This module provides Region of Interest (ROI) classes and utilities for signal objects.

The module includes:

- `ROI1DParam`: Parameter class for 1D signal ROIs
- `SegmentROI`: Single ROI representing a segment of a signal
- `SignalROI`: Collection of signal ROIs with operations
- `create_signal_roi`: Factory function for creating signal ROI objects

These classes enable defining and working with regions of interest in 1D signal data,
supporting operations like data extraction, masking, and parameter conversion.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.objects import base

if TYPE_CHECKING:
    from sigima.objects.signal.object import SignalObj


class ROI1DParam(base.BaseROIParam["SignalObj", "SegmentROI"]):
    """Signal ROI parameters"""

    # Note: in this class, the ROI parameters are stored as X coordinates

    title = gds.StringItem(_("ROI title"), default="")
    xmin = gds.FloatItem(_("First point coordinate"), default=0.0)
    xmax = gds.FloatItem(_("Last point coordinate"), default=1.0)

    def to_single_roi(self, obj: SignalObj) -> SegmentROI:
        """Convert parameters to single ROI

        Args:
            obj: signal object

        Returns:
            Single ROI
        """
        assert isinstance(self.xmin, float) and isinstance(self.xmax, float)
        return SegmentROI([self.xmin, self.xmax], False, title=self.title)

    def get_data(self, obj: SignalObj) -> np.ndarray:
        """Get signal data in ROI

        Args:
            obj: signal object

        Returns:
            Data in ROI
        """
        assert isinstance(self.xmin, float) and isinstance(self.xmax, float)
        imin, imax = np.searchsorted(obj.x, [self.xmin, self.xmax])
        return np.array([obj.x[imin:imax], obj.y[imin:imax]])


class SegmentROI(base.BaseSingleROI["SignalObj", ROI1DParam]):
    """Segment ROI

    Args:
        coords: ROI coordinates (xmin, xmax)
        title: ROI title
    """

    # Note: in this class, the ROI parameters are stored as X indices

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) != 2:
            raise ValueError("Invalid ROI segment coords (2 values expected)")
        if self.coords[0] >= self.coords[1]:
            raise ValueError("Invalid ROI segment coords (xmin >= xmax)")

    def get_coords_html_rows(self) -> list[tuple[str, str]]:
        """Return HTML table rows describing the segment coordinates."""
        xmin, xmax = self.coords
        coord_type = "indices" if self.indices else "physical"
        return [
            (f"X min ({coord_type})", f"{xmin:.4g}"),
            (f"X max ({coord_type})", f"{xmax:.4g}"),
        ]

    def get_coords_summary(self) -> str:
        """Return a short summary of the segment coordinates."""
        xmin, xmax = self.coords
        return f"X: [{xmin:.4g}, {xmax:.4g}]"

    def get_data(self, obj: SignalObj) -> tuple[np.ndarray, np.ndarray]:
        """Get signal data in ROI

        Args:
            obj: signal object

        Returns:
            Data in ROI
        """
        imin, imax = self.get_indices_coords(obj)
        return obj.x[imin:imax], obj.y[imin:imax]

    def to_mask(self, obj: SignalObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        mask = np.ones_like(obj.xydata, dtype=bool)
        imin, imax = self.get_indices_coords(obj)
        mask[:, imin:imax] = False
        return mask

    # pylint: disable=unused-argument
    def to_param(self, obj: SignalObj, index: int) -> ROI1DParam:
        """Convert ROI to parameters

        Args:
            obj: object (signal), for physical-indices coordinates conversion
            index: ROI index
        """
        gtitle = base.get_generic_roi_title(index)
        param = ROI1DParam(gtitle)
        param.title = self.title or gtitle
        param.xmin, param.xmax = self.get_physical_coords(obj)
        return param


class SignalROI(base.BaseROI["SignalObj", SegmentROI, ROI1DParam]):
    """Signal Regions of Interest

    Args:
        inverse: if True, ROI is outside the region
    """

    PREFIX = "s"

    def union(self) -> SignalROI:
        """Return union of ROIs"""
        if not self.single_rois:
            return SignalROI()
        coords = np.array([roi.coords for roi in self.single_rois])
        # Merge overlapping segments:
        sorted_coords = coords[coords[:, 0].argsort()]
        merged_coords = [sorted_coords[0].tolist()]
        for current in sorted_coords[1:]:
            last = merged_coords[-1]
            if current[0] <= last[1]:  # Overlap
                last[1] = max(last[1], current[1])  # Merge
            else:
                merged_coords.append(current.tolist())
        # Create new SignalROI with merged segments:
        roi = create_signal_roi(merged_coords)
        return roi

    def clipped(self, x_min: float, x_max: float) -> SignalROI:
        """Remove parts of ROIs outside the signal range

        Args:
            x_min: signal minimum X value
            x_max: signal maximum X value

        Returns:
            SignalROI object containing ROIs clipped to the specified signal range.
        """
        new_roi = SignalROI()
        for roi in self.single_rois:
            roi_min, roi_max = roi.coords
            if roi_max < x_min or roi_min > x_max:
                # ROI completely outside signal range: skip it
                continue
            # Clip ROI to signal range:
            new_roi_min = max(roi_min, x_min)
            new_roi_max = min(roi_max, x_max)
            new_roi.add_roi(
                SegmentROI(np.array([new_roi_min, new_roi_max], float), indices=False)
            )
        return new_roi

    def inverted(self, x_min: float, x_max: float) -> SignalROI:
        """Return inverted ROI (inside/outside).

        Args:
            x_min: signal minimum X value
            x_max: signal maximum X value
        Returns:
            Inverted ROI
        """
        clipped_roi = self.clipped(x_min, x_max)
        union_roi = clipped_roi.union()
        roi_delimiter_list = np.array(
            [roi.coords for roi in union_roi.single_rois]
        ).reshape(-1)

        if len(roi_delimiter_list) == 0:
            # No ROIs: inverted ROI is the whole signal
            raise ValueError("No ROIs defined, cannot invert")
        if len(roi_delimiter_list) % 2 != 0:
            # Odd number of delimiters: add signal limits
            raise ValueError("Internal error: odd number of ROI delimiters")

        if roi_delimiter_list[0] == x_min:
            # First delimiter is signal min: remove it
            roi_delimiter_list = roi_delimiter_list[1:]
        else:
            # Add signal min as first delimiter
            roi_delimiter_list = np.insert(roi_delimiter_list, 0, x_min)

        if roi_delimiter_list[-1] == x_max:
            # Last delimiter is signal max: remove it
            roi_delimiter_list = roi_delimiter_list[:-1]
        else:
            # Add signal max as last delimiter
            roi_delimiter_list = np.append(roi_delimiter_list, x_max)

        return create_signal_roi(np.array(roi_delimiter_list).reshape(-1, 2))

    @staticmethod
    def get_compatible_single_roi_classes() -> list[Type[SegmentROI]]:
        """Return compatible single ROI classes"""
        return [SegmentROI]

    def to_mask(self, obj: SignalObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        mask = np.ones_like(obj.xydata, dtype=bool)
        if self.single_rois:
            for roi in self.single_rois:
                mask &= roi.to_mask(obj)
        else:
            # If no single ROIs, the mask is empty (no ROI defined)
            mask[:] = False
        return mask


def create_signal_roi(
    coords: np.ndarray | list[float] | list[list[float]],
    indices: bool = False,
    title: str = "",
) -> SignalROI:
    """Create Signal Regions of Interest (ROI) object.
    More ROIs can be added to the object after creation, using the `add_roi` method.

    Args:
        coords: single ROI coordinates `[xmin, xmax]`, or multiple ROIs coordinates
         `[[xmin1, xmax1], [xmin2, xmax2], ...]` (lists or NumPy arrays)
        indices: if True, coordinates are indices, if False, they are physical values
         (default to False for signals)
        title: title

    Returns:
        Regions of Interest (ROI) object

    Raises:
        ValueError: if the number of coordinates is not even
    """
    coords = np.array(coords, float)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    roi = SignalROI()
    for row in coords:
        roi.add_roi(SegmentROI(row, indices=indices, title=title))
    return roi
