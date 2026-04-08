# -*- coding: utf-8 -*-
# Licensed under the terms of the BSD 3-Clause
# (see sigima/LICENSE for details)

"""
General analysis functions
==========================

This module provides general analysis functions for signal objects:

- Histogram computation
- Other analysis operations

.. note::

    Most operations use standard NumPy/SciPy functions or custom analysis routines.
"""

from __future__ import annotations

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.objects import (
    SignalObj,
    TableKind,
    TableResult,
    TableResultBuilder,
)
from sigima.objects.scalar.common import format_legend_value
from sigima.proc.base import HistogramParam, new_signal_result
from sigima.proc.decorator import computation_function
from sigima.tools.signal import dynamic, features, pulse


@computation_function()
def histogram(src: SignalObj, p: HistogramParam) -> SignalObj:
    """Compute histogram with :py:func:`numpy.histogram`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    data = src.get_masked_view().compressed()
    suffix = p.get_suffix(data)  # Also updates p.lower and p.upper

    # Compute histogram:
    y, bin_edges = np.histogram(data, bins=p.bins, range=(p.lower, p.upper))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Note: we use the `new_signal_result` function to create the result signal object
    # because the `dst_1_to_1` would copy the source signal, which is not what we want
    # here (we want a brand new signal object).
    dst = new_signal_result(
        src,
        "histogram",
        suffix=suffix,
        units=(src.yunit, ""),
        labels=(src.ylabel, _("Counts")),
    )
    dst.set_xydata(x, y)
    dst.set_metadata_option("shade", 0.5)
    dst.set_metadata_option("curvestyle", "Steps")
    return dst


class PulseFeaturesParam(gds.DataSet, title=_("Pulse features")):
    """Pulse features parameters."""

    signal_shape = gds.ChoiceItem(
        _("Signal shape"),
        [
            (None, _("Auto")),
            ("step", _("Step")),
            ("square", _("Square")),
        ],
        default=None,
        help=_("Signal type: auto-detect, step, or square."),
    )
    xstartmin = gds.FloatItem(
        _("Start baseline min"),
        default=0.0,
        help=_("Lower X boundary for the start baseline"),
    )
    xstartmax = gds.FloatItem(
        _("Start baseline max"),
        default=0.0,
        help=_("Upper X boundary for the start baseline"),
    )
    xendmin = gds.FloatItem(
        _("End baseline min"),
        default=1.0,
        help=_("Lower X boundary for the end baseline"),
    )
    xendmax = gds.FloatItem(
        _("End baseline max"),
        default=1.0,
        help=_("Upper X boundary for the end baseline"),
    )
    reference_levels = gds.ChoiceItem(
        _("Rise/Fall time"),
        [
            ((5, 95), _("5% - 95% (High precision)")),
            ((10, 90), _("10% - 90% (IEEE standard)")),
            ((20, 80), _("20% - 80% (Noisy signals)")),
            ((25, 75), _("25% - 75% (Alternative)")),
        ],
        default=(10, 90),
        help=_(
            "Reference levels for rise/fall time measurement. "
            "10%-90% is the IEEE standard for digital signal analysis."
        ),
    )

    def update_from_obj(self, obj: SignalObj) -> None:
        """Update parameters from a signal object."""
        self.xstartmin, self.xstartmax = pulse.get_start_range(obj.x)
        self.xendmin, self.xendmax = pulse.get_end_range(obj.x)


@computation_function()
def extract_pulse_features(obj: SignalObj, p: PulseFeaturesParam) -> TableResult:
    """Extract pulse features.

    Args:
        obj: The signal object from which to extract features.
        p: The pulse features parameters.

    Returns:
        An object containing the pulse features.
    """
    start_ratio, stop_ratio = p.reference_levels

    def func_extract_pulse_features(xydata: tuple[np.ndarray, np.ndarray]):
        """Extract pulse features (internal function).

        Args:
            xydata: Tuple of (x, y) data arrays

        Returns:
            Pulse features dataclass
        """
        x, y = xydata

        # When processing ROI data, the start/end ranges from parameters might be
        # outside the ROI's x-range. In that case, use None to auto-detect ranges.
        start_range = [p.xstartmin, p.xstartmax]
        end_range = [p.xendmin, p.xendmax]

        # Check if ranges are within the data's x-range
        x_min, x_max = x.min(), x.max()
        if (
            start_range[0] < x_min
            or start_range[1] > x_max
            or end_range[0] < x_min
            or end_range[1] > x_max
        ):
            # Ranges are outside ROI bounds - use auto-detection
            start_range = None
            end_range = None

        return pulse.extract_pulse_features(
            x,
            y,
            signal_shape=p.signal_shape,
            start_range=start_range,
            end_range=end_range,
            start_ratio=start_ratio / 100.0,
            stop_ratio=stop_ratio / 100.0,
        )

    builder = TableResultBuilder(_("Pulse features"), kind=TableKind.PULSE_FEATURES)
    builder.set_global_function(func_extract_pulse_features)
    builder.hide_columns(
        ["xstartmin", "xstartmax", "xendmin", "xendmax", "xplateaumin", "xplateaumax"]
    )
    builder.set_column_formats({"*": format_legend_value})
    return builder.compute(obj)


@computation_function()
def sampling_rate_period(obj: SignalObj) -> TableResult:
    """Compute sampling rate and period
    using the following functions:

    - fs: :py:func:`sigima.tools.signal.dynamic.sampling_rate`
    - T: :py:func:`sigima.tools.signal.dynamic.sampling_period`

    Args:
        obj: source signal

    Returns:
        Result properties with sampling rate and period
    """
    table = TableResultBuilder(_("Sampling rate and period"))
    table.add(lambda xy: dynamic.sampling_rate(xy[0]), "fs")
    table.add(lambda xy: dynamic.sampling_period(xy[0]), "T")
    return table.compute(obj)


@computation_function()
def contrast(obj: SignalObj) -> TableResult:
    """Compute contrast with :py:func:`sigima.tools.signal.misc.contrast`"""
    table = TableResultBuilder(_("Contrast"))
    table.add(lambda xy: features.contrast(xy[1]), "contrast")
    return table.compute(obj)


@computation_function()
def x_at_minmax(obj: SignalObj) -> TableResult:
    """
    Compute the smallest argument at the minima and the smallest argument at the maxima.

    Args:
        obj: The signal object.

    Returns:
        An object containing the x-values at the minima and the maxima.
    """
    table = TableResultBuilder(_("X at min/max"))
    table.add(lambda xy: xy[0][np.argmin(xy[1])], "X@Ymin")
    table.add(lambda xy: xy[0][np.argmax(xy[1])], "X@Ymax")
    return table.compute(obj)
