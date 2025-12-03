# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal object class
===================

This module provides the main SignalObj class for handling 1D signal data.

The module includes:

- `SignalObj`: Main class for signal data management and operations

The SignalObj class supports:
- Signal data storage with x, y coordinates
- Error bars (dx, dy) for uncertainty quantification
- Metadata and annotations
- ROI (Region of Interest) operations
- Axis labels and units management
- Copy operations with type conversion
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Type

import guidata.dataset as gds
import numpy as np
import pandas as pd

from sigima.config import _
from sigima.objects import base
from sigima.objects.signal.constants import (
    DATETIME_X_FORMAT_KEY,
    DATETIME_X_KEY,
    DEFAULT_DATETIME_FORMAT,
    VALID_TIME_UNITS,
)
from sigima.objects.signal.roi import SignalROI


def validate_and_convert_dtype(x: np.ndarray) -> np.ndarray:
    """Check if data type is valid, convert integer to float64 if needed.

    Args:
        x: Input array

    Returns:
        Original array if data type is valid, array converted to float64 if integer

    Raises:
        ValueError: If data type is not valid
    """
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float64)
    if x.dtype not in SignalObj.VALID_DTYPES:
        raise ValueError(
            f"Invalid data type: {x.dtype}. "
            f"Valid types: {', '.join(str(dt) for dt in SignalObj.VALID_DTYPES)}"
        )
    return x


class SignalObj(gds.DataSet, base.BaseObj[SignalROI]):
    """Signal object"""

    PREFIX = "s"
    VALID_DTYPES = (np.float32, np.float64, np.complex128)

    _tabs = gds.BeginTabGroup("all")

    _datag = gds.BeginGroup(_("Data and metadata"))
    title = gds.StringItem(_("Signal title"), default=_("Untitled"))
    xydata = gds.FloatArrayItem(_("Data"), transpose=True, minmax="rows")
    metadata = gds.DictItem(_("Metadata"), default={})  # type: ignore[assignment]
    annotations = gds.StringItem(_("Annotations"), default="").set_prop(
        "display",
        hide=True,
    )  # Annotations (JSON). Use get/set_annotations() API  # type: ignore[assignment]
    _e_datag = gds.EndGroup(_("Data and metadata"))

    _unitsg = gds.BeginGroup(_("Titles / Units"))
    title = gds.StringItem(_("Signal title"), default=_("Untitled"))
    _tabs_u = gds.BeginTabGroup("units")
    _unitsx = gds.BeginGroup(_("X-axis"))
    xlabel = gds.StringItem(_("Title"), default="")
    xunit = gds.StringItem(_("Unit"), default="")
    _e_unitsx = gds.EndGroup(_("X-axis"))
    _unitsy = gds.BeginGroup(_("Y-axis"))
    ylabel = gds.StringItem(_("Title"), default="")
    yunit = gds.StringItem(_("Unit"), default="")
    _e_unitsy = gds.EndGroup(_("Y-axis"))
    _e_tabs_u = gds.EndTabGroup("units")
    _e_unitsg = gds.EndGroup(_("Titles / Units"))

    _scalesg = gds.BeginGroup(_("Scales"))
    _prop_autoscale = gds.GetAttrProp("autoscale")
    autoscale = gds.BoolItem(_("Auto scale"), default=True).set_prop(
        "display", store=_prop_autoscale
    )
    _tabs_b = gds.BeginTabGroup("bounds")
    _boundsx = gds.BeginGroup(_("X-axis"))
    xscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    xscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    xscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsx = gds.EndGroup(_("X-axis"))
    _boundsy = gds.BeginGroup(_("Y-axis"))
    yscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    yscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    yscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsy = gds.EndGroup(_("Y-axis"))
    _e_tabs_b = gds.EndTabGroup("bounds")
    _e_scalesg = gds.EndGroup(_("Scales"))

    _e_tabs = gds.EndTabGroup("all")

    def __init__(self, title=None, comment=None, icon=""):
        """Constructor

        Args:
            title: title
            comment: comment
            icon: icon
        """
        gds.DataSet.__init__(self, title, comment, icon)
        base.BaseObj.__init__(self)

    @staticmethod
    def get_roi_class() -> Type[SignalROI]:
        """Return ROI class"""
        return SignalROI

    def copy(
        self,
        title: str | None = None,
        dtype: np.dtype | None = None,
        all_metadata: bool = False,
    ) -> SignalObj:
        """Copy object.

        Args:
            title: title
            dtype: data type
            all_metadata: if True, copy all metadata, otherwise only basic metadata

        Returns:
            Copied object
        """
        title = self.title if title is None else title
        obj = SignalObj(title=title)
        obj.title = title
        obj.xlabel = self.xlabel
        obj.ylabel = self.ylabel
        obj.xunit = self.xunit
        obj.yunit = self.yunit
        if dtype not in (None, float, complex, np.complex128):
            raise RuntimeError("Signal data only supports float64/complex128 dtype")
        obj.metadata = base.deepcopy_metadata(self.metadata, all_metadata=all_metadata)
        obj.annotations = self.annotations
        obj.xydata = np.array(self.xydata, copy=True, dtype=dtype)
        obj.autoscale = self.autoscale
        obj.xscalelog = self.xscalelog
        obj.xscalemin = self.xscalemin
        obj.xscalemax = self.xscalemax
        obj.yscalelog = self.yscalelog
        obj.yscalemin = self.yscalemin
        obj.yscalemax = self.yscalemax
        return obj

    def set_data_type(self, dtype: np.dtype) -> None:  # pylint: disable=unused-argument
        """Change data type.

        Args:
            Data type
        """
        raise RuntimeError("Setting data type is not support for signals")

    def set_xydata(
        self,
        x: np.ndarray | list | None,
        y: np.ndarray | list | None,
        dx: np.ndarray | list | None = None,
        dy: np.ndarray | list | None = None,
    ) -> None:
        """Set xy data

        Args:
            x: x data
            y: y data
            dx: dx data (optional: error bars). Use None to reset dx data to None,
             or provide array to set new dx data.
            dy: dy data (optional: error bars). Use None to reset dy data to None,
             or provide array to set new dy data.
        """
        if x is None and y is None:
            # Using empty arrays (this allows initialization of the object without data)
            x = np.array([], dtype=np.float64)
            y = np.array([], dtype=np.float64)
        if x is None and y is not None:
            # If x is None, we create a default x array based on the length of y
            assert isinstance(y, (list, np.ndarray))
            x = np.arange(len(y), dtype=np.float64)
        if x is not None:
            x = np.array(x)
        if y is not None:
            y = np.array(y)
        if dx is not None:
            dx = np.array(dx)
        if dy is not None:
            dy = np.array(dy)
        if dx is None and dy is None:
            xydata = np.vstack([x, y])
        else:
            if dx is None:
                dx = np.full_like(x, np.nan)
            if dy is None:
                dy = np.full_like(y, np.nan)
            assert x is not None and y is not None
            xydata = np.vstack((x, y, dx, dy))
        self.xydata = validate_and_convert_dtype(xydata)

    def __get_x(self) -> np.ndarray | None:
        """Get x data"""
        if self.xydata is not None:
            x: np.ndarray = self.xydata[0]
            # We have to ensure that x is a floating point array, because if y is
            # complex, the whole xydata array will be complex, and we need to avoid
            # any unintended type promotion.
            return x.real.astype(float)
        return None

    def __set_x(self, data: np.ndarray | list[float]) -> None:
        """Set x data"""
        assert isinstance(self.xydata, np.ndarray)
        assert isinstance(data, (list, np.ndarray))
        data = np.array(data, dtype=float)
        assert data.shape[0] == self.xydata.shape[1], (
            "X data size must match Y data size"
        )
        if not np.all(np.diff(data) >= 0.0):
            raise ValueError("X data must be monotonic (sorted in ascending order)")
        self.xydata[0] = validate_and_convert_dtype(data)

    def __get_y(self) -> np.ndarray | None:
        """Get y data"""
        if self.xydata is not None:
            return self.xydata[1]
        return None

    def __set_y(self, data: np.ndarray | list[float]) -> None:
        """Set y data"""
        assert isinstance(self.xydata, np.ndarray)
        assert isinstance(data, (list, np.ndarray))
        data = np.array(data)
        assert data.shape[0] == self.xydata.shape[1], (
            "Y data size must match X data size"
        )
        assert np.issubdtype(data.dtype, np.inexact), "Y data must be float or complex"
        self.xydata[1] = validate_and_convert_dtype(data)

    def __get_dx(self) -> np.ndarray | None:
        """Get dx data"""
        if self.xydata is not None and len(self.xydata) == 4:
            dx: np.ndarray = self.xydata[2]
            if np.all(np.isnan(dx)):
                return None
            return dx.real.astype(float)
        return None

    def __set_dx(self, data: np.ndarray | list[float] | None) -> None:
        """Set dx data"""
        if data is None:
            data = np.full_like(self.x, np.nan)
        assert isinstance(data, (list, np.ndarray))
        data = np.array(data)
        if self.xydata is None:
            raise ValueError("Signal data not initialized")
        assert data.shape[0] == self.xydata.shape[1], (
            "dx data size must match X data size"
        )
        if len(self.xydata) == 2:
            self.xydata = np.vstack((self.xydata, np.zeros((2, self.xydata.shape[1]))))
        self.xydata[2] = validate_and_convert_dtype(data)

    def __get_dy(self) -> np.ndarray | None:
        """Get dy data"""
        if self.xydata is not None and len(self.xydata) == 4:
            dy: np.ndarray = self.xydata[3]
            if np.all(np.isnan(dy)):
                return None
            return dy
        return None

    def __set_dy(self, data: np.ndarray | list[float] | None) -> None:
        """Set dy data"""
        if data is None:
            data = np.full_like(self.x, np.nan)
        assert isinstance(data, (list, np.ndarray))
        data = np.array(data)
        if self.xydata is None:
            raise ValueError("Signal data not initialized")
        assert data.shape[0] == self.xydata.shape[1], (
            "dy data size must match X data size"
        )
        if len(self.xydata) == 2:
            self.xydata = np.vstack((self.xydata, np.zeros((2, self.xydata.shape[1]))))
        self.xydata[3] = validate_and_convert_dtype(data)

    x = property(__get_x, __set_x)
    y = data = property(__get_y, __set_y)
    dx = property(__get_dx, __set_dx)
    dy = property(__get_dy, __set_dy)

    def get_data(self, roi_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).

        Args:
            roi_index: ROI index

        Returns:
            Data
        """
        if self.roi is None or roi_index is None:
            assert isinstance(self.xydata, np.ndarray)
            return self.x, self.y
        single_roi = self.roi.get_single_roi(roi_index)
        return single_roi.get_data(self)

    def physical_to_indices(self, coords: list[float]) -> list[int]:
        """Convert coordinates from physical (real world) to indices (pixel)

        Args:
            coords: coordinates

        Returns:
            Indices
        """
        assert isinstance(self.x, np.ndarray)
        return [int(np.abs(self.x - x).argmin()) for x in coords]

    def indices_to_physical(self, indices: list[int]) -> list[float]:
        """Convert coordinates from indices to physical (real world)

        Args:
            indices: indices

        Returns:
            Coordinates
        """
        # We take the real part of the x data to avoid `ComplexWarning` warnings
        # when creating and manipulating the `XRangeSelection` shape (`plotpy`)
        return self.x.real[indices].tolist()

    def is_x_datetime(self) -> bool:
        """Check if x data represents datetime values.

        Returns:
            True if x data represents datetime values, False otherwise
        """
        return self.metadata.get(DATETIME_X_KEY, False)

    def set_x_from_datetime(
        self,
        dt_array: np.ndarray | list,
        unit: str = "s",
        format_str: str | None = None,
    ) -> None:
        """Set x values from datetime objects or strings.

        This method converts datetime data to float timestamps (Unix time: seconds
        since 1970-01-01) for efficient storage and computation. The datetime context
        is preserved through metadata.

        Note: X values are always stored as Unix timestamps (seconds since 1970-01-01)
        regardless of the 'unit' parameter. The 'unit' parameter is stored in metadata
        and used only for axis labeling when plotting.

        Args:
            dt_array: Array of datetime objects, datetime strings, or numpy datetime64
            unit: Time unit label for display. Options: 's' (seconds),
             'ms' (milliseconds), 'us' (microseconds), 'ns' (nanoseconds),
             'min' (minutes), 'h' (hours). Default is 's'. This parameter only
             affects the axis label, not the stored data.
            format_str: Format string for datetime display. If None, uses default.

        Raises:
            ValueError: If unit is not valid

        Example:
            >>> from datetime import datetime
            >>> signal = SignalObj()
            >>> timestamps = [datetime(2025, 1, 1, 10, 0, 0),
            ...               datetime(2025, 1, 1, 10, 0, 1)]
            >>> signal.set_x_from_datetime(timestamps, unit='s')
            >>> signal.is_x_datetime()
            True
            >>> # X data is stored as Unix timestamps (seconds since 1970)
            >>> signal.x[0] > 1.7e9  # Year 2025
            True
        """
        if unit not in VALID_TIME_UNITS:
            raise ValueError(
                f"Invalid unit: {unit}. Must be one of: {', '.join(VALID_TIME_UNITS)}"
            )

        # Convert to pandas datetime (handles strings, datetime objects, etc.)
        dt_series = pd.to_datetime(dt_array)

        # Convert to float timestamp in seconds (pandas epoch is in nanoseconds)
        # Note: We always store as Unix timestamps (seconds since 1970-01-01)
        # regardless of the 'unit' parameter, which is only for display purposes
        timestamp_seconds = dt_series.astype(np.int64) / 1e9

        # Convert to numpy array (pandas may return Float64Index)
        x_float = np.array(timestamp_seconds, dtype=np.float64)

        # Check if signal already has data with matching size
        if self.xydata is not None and self.xydata.shape[1] == len(x_float):
            # Signal already has matching data, just update x
            self.x = x_float
        else:
            # Initialize or reinitialize signal with x data (y will be zeros)
            y_placeholder = np.zeros_like(x_float)
            self.set_xydata(x_float, y_placeholder)

        # Store metadata
        self.metadata[DATETIME_X_KEY] = True
        self.metadata[DATETIME_X_FORMAT_KEY] = (
            format_str if format_str is not None else DEFAULT_DATETIME_FORMAT
        )
        # Store unit in xunit attribute (more intuitive than metadata)
        self.xunit = unit

    def get_x_as_datetime(self) -> np.ndarray:
        """Get x values as datetime objects if x is datetime data.

        Returns x data as numpy datetime64 array if the signal contains datetime data,
        otherwise returns the regular x data as floats.

        Returns:
            Array of datetime64 objects if x is datetime data, otherwise regular x array

        Example:
            >>> signal.set_x_from_datetime([datetime(2025, 1, 1, 10, 0, 0)])
            >>> dt_values = signal.get_x_as_datetime()
            >>> isinstance(dt_values[0], np.datetime64)
            True
        """
        if not self.is_x_datetime():
            return self.x
        # X values are always stored as Unix timestamps (seconds since 1970-01-01)
        # regardless of the 'unit' parameter
        x_float = self.x

        # Convert seconds to datetime using pandas
        return pd.to_datetime(x_float, unit="s").to_numpy()
