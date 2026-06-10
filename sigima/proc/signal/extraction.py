# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal extraction and ROI operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import numpy as np

from sigima.objects import ROI1DParam, SignalObj
from sigima.proc.base import dst_1_to_1
from sigima.proc.decorator import computation_function


@computation_function()
def extract_rois(src: SignalObj, params: list[ROI1DParam]) -> SignalObj:
    """Extract multiple regions of interest from data

    Args:
        src: source signal
        params: list of ROI parameters

    Returns:
        Signal with multiple regions of interest
    """
    suffix = None
    if len(params) == 1:
        p: ROI1DParam = params[0]
        suffix = f"{p.xmin:.3g}≤x≤{p.xmax:.3g}"
    dst = dst_1_to_1(src, "extract_rois", suffix)
    x, y = src.get_data()
    src_dx, src_dy = src.dx, src.dy
    xout = np.full_like(x, np.nan)
    yout = np.full_like(y, np.nan)
    dxout = np.full_like(x, np.nan) if src_dx is not None else None
    dyout = np.full_like(y, np.nan) if src_dy is not None else None
    for p in params:
        idx1, idx2 = np.searchsorted(x, p.xmin), np.searchsorted(x, p.xmax)
        slice0 = slice(idx1, idx2)
        xout[slice0], yout[slice0] = x[slice0], y[slice0]
        if dxout is not None:
            dxout[slice0] = src_dx[slice0]
        if dyout is not None:
            dyout[slice0] = src_dy[slice0]
    nans = np.isnan(xout) | np.isnan(yout)
    keep = ~nans
    dst.set_xydata(
        xout[keep],
        yout[keep],
        dx=None if dxout is None else dxout[keep],
        dy=None if dyout is None else dyout[keep],
    )
    # Remove ROI from destination signal: the extracted data no longer needs ROI
    dst.roi = None
    return dst


@computation_function()
def extract_roi(src: SignalObj, p: ROI1DParam) -> SignalObj:
    """Extract single region of interest from data

    Args:
        src: source signal
        p: ROI parameters

    Returns:
        Signal with single region of interest
    """
    dst = dst_1_to_1(src, "extract_roi", f"{p.xmin:.3g}≤x≤{p.xmax:.3g}")
    x, y = p.get_data(src).copy()
    # Slice uncertainty data the same way ROI1DParam.get_data slices x/y
    src_x = src.x
    imin, imax = np.searchsorted(src_x, [p.xmin, p.xmax])
    src_dx, src_dy = src.dx, src.dy
    dx = None if src_dx is None else src_dx[imin:imax].copy()
    dy = None if src_dy is None else src_dy[imin:imax].copy()
    dst.set_xydata(x, y, dx=dx, dy=dy)
    # Remove ROI from destination signal: the extracted data no longer needs ROI
    dst.roi = None
    return dst
