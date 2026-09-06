# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see sigima/LICENSE for details)

"""
Feature extraction and analysis functions
=========================================

This module provides feature extraction and analysis functions for signal objects:

- Peak detection
- Full Width at Half Maximum (FWHM) and related measurements
- Statistical analysis
- Bandwidth calculations
- Dynamic parameters (ENOB, SNR, SINAD, THD, SFDR)

.. note::

    Most operations use functions from :mod:`sigima.tools.signal` for actual
    computations.
"""

from __future__ import annotations

import warnings

import guidata.dataset as gds
import numpy as np
import scipy.integrate as spt

from sigima.config import _
from sigima.enums import PowerUnit
from sigima.objects import (
    GeometryResult,
    KindShape,
    SignalObj,
    TableKind,
    TableResult,
    TableResultBuilder,
)
from sigima.objects.signal.creation import create_signal
from sigima.proc.base import dst_1_to_1
from sigima.proc.decorator import computation_function
from sigima.proc.signal.base import compute_geometry_from_obj
from sigima.tools.signal import dynamic, features, peakdetection, pulse


class PeakDetectionParam(gds.DataSet, title=_("Peak detection")):
    """Peak detection parameters"""

    threshold = gds.FloatItem(_("Threshold"), default=0.1, min=0.0, max=100.0)
    min_dist = gds.IntItem(_("Minimum distance"), default=1, min=1)


@computation_function()
def peak_detection(src: SignalObj, p: PeakDetectionParam) -> SignalObj:
    """Peak detection with
    :py:func:`sigima.tools.signal.peakdetection.peak_indices`

    .. deprecated::
        Use :py:func:`extract_peak_positions` to detect peaks (returns a
        :class:`~sigima.objects.TableResult` of XY markers, suitable for
        graphical overlay), then :py:func:`markers_table_to_signal` if you
        also need a child signal with sticks. This two-step workflow makes
        the user intent explicit and aligns the signal API with the image
        API (which is analysis-only via
        :class:`~sigima.objects.GeometryResult`).

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    warnings.warn(
        "sigima.proc.signal.peak_detection is deprecated and will be removed in a "
        "future release: use extract_peak_positions to detect peaks, then "
        "markers_table_to_signal if a sticks signal is also needed.",
        DeprecationWarning,
        stacklevel=2,
    )
    dst = dst_1_to_1(
        src, "peak_detection", f"threshold={p.threshold}%, min_dist={p.min_dist}pts"
    )
    x, y = src.get_data()
    indices = peakdetection.peak_indices(
        y, thres=p.threshold * 0.01, min_dist=p.min_dist
    )
    dst.set_xydata(x[indices], y[indices])
    dst.set_metadata_option("curvestyle", "Sticks")
    return dst


def _resolve_xy_columns(table: TableResult) -> tuple[int, int]:
    """Locate the ``x`` and ``y`` column indices in an XY-markers table.

    Falls back to the first two columns when explicit ``x`` / ``y`` headers
    are missing, mirroring the tolerance of DataLab's PlotPy adapter.

    Args:
        table: source table

    Returns:
        Tuple ``(x_index, y_index)``.

    Raises:
        ValueError: when the table has fewer than 2 columns.
    """
    headers = list(table.headers)
    if len(headers) < 2:
        raise ValueError("XY-markers table must have at least 2 columns")
    lowered = [h.strip().lower() for h in headers]
    try:
        ix = lowered.index("x")
    except ValueError:
        ix = 0
    try:
        iy = lowered.index("y")
    except ValueError:
        iy = 1 if ix != 1 else 0
    if ix == iy:
        raise ValueError("XY-markers table x and y columns must differ")
    return ix, iy


def markers_table_to_signal(
    table: TableResult, ref: SignalObj | None = None
) -> SignalObj:
    """Convert an XY-markers table to a sticks signal.

    Builds a new :class:`~sigima.objects.SignalObj` from the ``(x, y)``
    rows of an :attr:`~sigima.objects.TableKind.XY_MARKERS` table. The
    resulting signal carries ``curvestyle="Sticks"`` so that it is
    rendered as delta functions, matching the historical output of the
    deprecated :py:func:`peak_detection`.

    The table's ``x`` and ``y`` columns are located by header name (case
    and whitespace insensitive); the first two columns are used as a
    fallback when explicit headers are missing. Extra columns are ignored.

    Args:
        table: source XY-markers table (typically produced by
         :py:func:`extract_peak_positions`).
        ref: optional reference signal whose ``xlabel`` / ``ylabel`` /
         ``xunit`` / ``yunit`` are inherited by the new signal.

    Returns:
        New signal whose samples are the table's ``(x, y)`` couples,
        rendered as sticks.

    Raises:
        ValueError: when ``table`` is not an XY-markers table or has fewer
         than 2 columns / valid x/y columns.
    """
    if not isinstance(table, TableResult):
        raise TypeError("table must be a TableResult instance")
    if not table.is_xy_markers():
        raise ValueError(
            f"table must be an XY-markers TableResult (got kind={table.kind!r})"
        )
    ix, iy = _resolve_xy_columns(table)
    rows = table.data
    if rows:
        x = np.asarray([row[ix] for row in rows], dtype=float)
        y = np.asarray([row[iy] for row in rows], dtype=float)
    else:
        x = np.empty(0, dtype=float)
        y = np.empty(0, dtype=float)
    headers = list(table.headers)
    if ref is not None:
        labels = (ref.xlabel, ref.ylabel)
        units = (ref.xunit, ref.yunit)
    else:
        labels = (headers[ix], headers[iy])
        units = ("", "")
    title = f"{table.title} \u2192 sticks"
    if ref is not None and ref.title:
        title = f"{table.title}({ref.title}) \u2192 sticks"
    signal = create_signal(title, x=x, y=y, labels=labels, units=units)
    signal.set_metadata_option("curvestyle", "Sticks")
    return signal


@computation_function()
def extract_peak_positions(obj: SignalObj, p: PeakDetectionParam) -> TableResult:
    """Extract peak positions as an XY-markers table.

    Detects peaks with
    :py:func:`sigima.tools.signal.peakdetection.peak_indices` and returns a
    :class:`~sigima.objects.TableResult` of kind
    :attr:`~sigima.objects.TableKind.XY_MARKERS`. Each row holds the
    ``(x, y)`` coordinates of a detected peak. Suitable for highlighting
    remarkable points such as spectral lines (e.g. gamma-ray spectra) or
    pulse positions: DataLab renders such tables as cross markers at the
    corresponding ``(x, y)`` positions.

    Args:
        obj: source signal
        p: peak detection parameters

    Returns:
        Table result with columns ``x`` and ``y``, one row per detected peak.
    """
    rows: list[list] = []
    roi_idx: list[int] = []
    for i_roi in obj.iterate_roi_indices():
        x, y = obj.get_data(i_roi)
        indices = peakdetection.peak_indices(
            y, thres=p.threshold * 0.01, min_dist=p.min_dist
        )
        for idx in indices:
            rows.append([float(x[idx]), float(y[idx])])
            roi_idx.append(-1 if i_roi is None else int(i_roi))

    def _axis_header(default: str, label: str, unit: str) -> str:
        """Build a column header from a signal axis label/unit pair."""
        text = label or default
        if unit:
            text = f"{text} ({unit})"
        return text

    return TableResult.from_rows(
        title=_("Peak positions"),
        headers=[
            _axis_header("x", obj.xlabel, obj.xunit),
            _axis_header("y", obj.ylabel, obj.yunit),
        ],
        rows=rows,
        roi_indices=roi_idx if rows else None,
        kind=TableKind.XY_MARKERS,
        attrs={
            "threshold": p.threshold,
            "min_dist": p.min_dist,
            "show_row_index": True,
        },
    )


class FWHMParam(
    gds.DataSet,
    title=_("FWHM"),
    comment=_(
        "<u>Methods and trade-offs:</u><br><br>"
        "•&nbsp;Zero-crossing: Fast, sensitive to noise<br>"
        "•&nbsp;Gaussian fit: Good for symmetric peaks, assumes Gaussian shape<br>"
        "•&nbsp;Lorentzian fit: Suitable for peaks with long tails, dominated by "
        "collisional or lifetime broadening<br>"
        "•&nbsp;Voigt fit: Most accurate for spectroscopic data, or laser lines "
        "broadened by both Doppler and collisional effects<br>"
    ),
):
    """FWHM parameters"""

    methods = (
        ("zero-crossing", _("Zero-crossing")),
        ("gauss", _("Gaussian fit")),
        ("lorentz", _("Lorentzian fit")),
        ("voigt", _("Voigt fit")),
    )
    method = gds.ChoiceItem(_("Method"), methods, default="zero-crossing")
    xmin = gds.FloatItem(
        "X<sub>MIN</sub>",
        default=None,
        check=False,
        help=_("Lower X boundary (empty for no limit, i.e. start of the signal)"),
    )
    xmax = gds.FloatItem(
        "X<sub>MAX</sub>",
        default=None,
        check=False,
        help=_("Upper X boundary (empty for no limit, i.e. end of the signal)"),
    ).set_prop("display", col=1)

    def validate_parameters(self, *context: object) -> None:
        """Validate optional measurement boundaries."""
        del context
        if self.xmin is not None and self.xmax is not None and self.xmin >= self.xmax:
            raise ValueError("xmin must be strictly less than xmax")


@computation_function()
def fwhm(obj: SignalObj, param: FWHMParam) -> GeometryResult | None:
    """Compute FWHM with :py:func:`sigima.tools.signal.pulse.fwhm`

    Args:
        obj: source signal
        param: parameters

    Returns:
        Segment coordinates
    """
    return compute_geometry_from_obj(
        "fwhm",
        KindShape.SEGMENT,
        obj,
        pulse.fwhm,
        param.method,
        param.xmin,
        param.xmax,
    )


@computation_function()
def fw1e2(obj: SignalObj) -> GeometryResult | None:
    """Compute FW at 1/e² with :py:func:`sigima.tools.signal.pulse.fw1e2`

    Args:
        obj: source signal

    Returns:
        Segment coordinates
    """
    return compute_geometry_from_obj("fw1e2", KindShape.SEGMENT, obj, pulse.fw1e2)


# Note: we do not specify title of the dataset here because it's a generic parameter
# used in multiple functions (this avoids that the same title is displayed in GUI
# for different functions)
class OrdinateParam(gds.DataSet):
    """Ordinate parameter."""

    y = gds.FloatItem("y", default=0.0)


@computation_function()
def full_width_at_y(obj: SignalObj, p: OrdinateParam) -> GeometryResult | None:
    """
    Compute full width at a given y value for a signal object.

    Args:
        obj: The signal object containing x and y data.
        p: The ordinate parameter dataset

    Returns:
        Segment coordinates
    """
    return compute_geometry_from_obj(
        "∆X", KindShape.SEGMENT, obj, pulse.full_width_at_y, p.y
    )


@computation_function()
def x_at_y(obj: SignalObj, p: OrdinateParam) -> GeometryResult | None:
    """
    Compute the smallest x-value at a given y-value for a signal object.

    Args:
        obj: The signal object containing x and y data.
        p: The parameter dataset for finding the abscissa.

    Returns:
         A GeometryResult with a cross marker at the (x, y) position.
    """

    def compute_x_at_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Helper function to compute x at y value."""
        x_values = features.find_x_values_at_y(x, y, p.y)
        x_result = x_values[0] if len(x_values) > 0 else np.nan
        return np.array([x_result, p.y])

    return compute_geometry_from_obj(
        f"x|y={p.y}",
        KindShape.MARKER,
        obj,
        compute_x_at_y,
    )


# Note: we do not specify title of the dataset here because it's a generic parameter
# used in multiple functions (this avoids that the same title is displayed in GUI
# for different functions)
class AbscissaParam(gds.DataSet):
    """Abscissa parameter."""

    x = gds.FloatItem("x", default=0.0)


@computation_function()
def y_at_x(obj: SignalObj, p: AbscissaParam) -> GeometryResult | None:
    """
    Compute the smallest y-value at a given x-value for a signal object.

    Args:
        obj: The signal object containing x and y data.
        p: The parameter dataset for finding the ordinate.

    Returns:
         A GeometryResult with a cross marker at the (x, y) position.
    """

    def compute_y_at_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Helper function to compute y at x value."""
        y_result = features.find_y_at_x_value(x, y, p.x)
        return np.array([p.x, y_result])

    return compute_geometry_from_obj(
        f"y|x={p.x}",
        KindShape.MARKER,
        obj,
        compute_y_at_x,
    )


@computation_function()
def stats(obj: SignalObj) -> TableResult:
    """Compute statistics on a signal

    Args:
        obj: source signal

    Returns:
        Result properties object
    """
    table = TableResultBuilder(_("Signal statistics"), kind=TableKind.STATISTICS)
    table.add(lambda xy: np.nanmin(xy[1]), "min")
    table.add(lambda xy: np.nanmax(xy[1]), "max")
    table.add(lambda xy: np.nanmean(xy[1]), "mean")
    table.add(lambda xy: np.nanmedian(xy[1]), "median")
    table.add(lambda xy: np.nanstd(xy[1]), "std")
    table.add(lambda xy: np.nanmean(xy[1]) / np.nanstd(xy[1]), "snr")
    table.add(lambda xy: np.nanmax(xy[1]) - np.nanmin(xy[1]), "ptp")
    table.add(lambda xy: np.nansum(xy[1]), "sum")
    table.add(lambda xy: spt.trapezoid(xy[1], xy[0]), "trapz")
    return table.compute(obj)


@computation_function()
def bandwidth_3db(obj: SignalObj) -> GeometryResult | None:
    """Compute bandwidth at -3 dB with
    :py:func:`sigima.tools.signal.misc.bandwidth`

    .. note::

       The bandwidth is defined as the range of frequencies over which the signal
       maintains a certain level relative to its peak.

    .. warning::

        The signal is assumed to be smooth enough for the bandwidth calculation to be
        meaningful. If the signal contains excessive noise, multiple peaks, or is not
        sufficiently continuous, the computed bandwidth may not accurately represent the
        true -3dB range. It is recommended to preprocess the signal to ensure reliable
        results.

    Args:
        obj: Source signal.

    Returns:
        Result shape with bandwidth.
    """
    return compute_geometry_from_obj(
        "bandwidth", KindShape.SEGMENT, obj, features.find_bandwidth_coordinates, -3.0
    )


class DynamicParam(gds.DataSet, title=_("Dynamic parameters")):
    """Parameters for dynamic range computation (ENOB, SNR, SINAD, THD, SFDR)"""

    full_scale = gds.FloatItem(
        _("Full scale"), default=0.16, min=0.0, nonzero=True, unit="V"
    )
    unit = gds.ChoiceItem(
        _("Unit"),
        [(PowerUnit.DBC, "dBc"), (PowerUnit.DBFS, "dBFS")],
        default=PowerUnit.DBC,
        help=_("Unit for SINAD"),
    )
    nb_harm = gds.IntItem(
        _("Number of harmonics"),
        default=5,
        min=1,
        help=_("Number of harmonics to consider for THD"),
    )


@computation_function()
def dynamic_parameters(src: SignalObj, p: DynamicParam) -> TableResult:
    """Compute Dynamic parameters
    using the following functions:

    - Freq: :py:func:`sigima.tools.signal.dynamic.sinus_frequency`
    - ENOB: :py:func:`sigima.tools.signal.dynamic.enob`
    - SNR: :py:func:`sigima.tools.signal.dynamic.snr`
    - SINAD: :py:func:`sigima.tools.signal.dynamic.sinad`
    - THD: :py:func:`sigima.tools.signal.dynamic.thd`
    - SFDR: :py:func:`sigima.tools.signal.dynamic.sfdr`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result properties with ENOB, SNR, SINAD, THD, SFDR
    """
    unit: PowerUnit = p.unit
    table = TableResultBuilder(_("Dynamic parameters"))
    table.add(lambda xy: dynamic.sinus_frequency(xy[0], xy[1]), "freq")
    table.add(lambda xy: dynamic.enob(xy[0], xy[1], p.full_scale), "enob")
    table.add(lambda xy: dynamic.snr(xy[0], xy[1], unit), "snr")
    table.add(lambda xy: dynamic.sinad(xy[0], xy[1], unit), "sinad")
    table.add(
        lambda xy: dynamic.thd(xy[0], xy[1], p.full_scale, unit, p.nb_harm), "thd"
    )
    table.add(lambda xy: dynamic.sfdr(xy[0], xy[1], p.full_scale, unit), "sfdr")
    return table.compute(src)
