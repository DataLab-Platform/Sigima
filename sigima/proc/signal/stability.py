# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see sigima/LICENSE for details)

"""
Stability analysis functions
============================

This module provides stability analysis functions for signal objects:

- Allan variance and deviation
- Overlapping Allan variance
- Modified Allan variance
- Hadamard variance
- Total variance

.. note::

    All operations use functions from :mod:`sigima.tools.signal.stability` for
    actual computations.
"""

from __future__ import annotations

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.objects import SignalObj
from sigima.proc.contract_types import EXACT, ReplayQualification
from sigima.proc.decorator import computation_function
from sigima.tools.signal import stability

from .base import dst_1_to_1


class AllanVarianceParam(gds.DataSet, title=_("Allan variance")):
    """Allan variance parameters"""

    max_tau = gds.IntItem("Max τ", default=100, min=1, unit="pts")


@computation_function(
    operation_id="sigima.signal.allan_variance",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def allan_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Allan variance with
    :py:func:`sigima.tools.signal.stability.allan_variance`.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "allan_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    avar = stability.allan_variance(x, y, tau_values)
    dst.set_xydata(tau_values, avar)
    return dst


@computation_function(
    operation_id="sigima.signal.allan_deviation",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def allan_deviation(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Allan deviation with
    :py:func:`sigima.tools.signal.stability.allan_deviation`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "allan_deviation", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    adev = stability.allan_deviation(x, y, tau_values)
    dst.set_xydata(tau_values, adev)
    return dst


@computation_function(
    operation_id="sigima.signal.overlapping_allan_variance",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def overlapping_allan_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Overlapping Allan variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "overlapping_allan_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    oavar = stability.overlapping_allan_variance(x, y, tau_values)
    dst.set_xydata(tau_values, oavar)
    return dst


@computation_function(
    operation_id="sigima.signal.modified_allan_variance",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def modified_allan_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Modified Allan variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "modified_allan_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    mavar = stability.modified_allan_variance(x, y, tau_values)
    dst.set_xydata(tau_values, mavar)
    return dst


@computation_function(
    operation_id="sigima.signal.hadamard_variance",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def hadamard_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Hadamard variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "hadamard_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    hvar = stability.hadamard_variance(x, y, tau_values)
    dst.set_xydata(tau_values, hvar)
    return dst


@computation_function(
    operation_id="sigima.signal.total_variance",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def total_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Total variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "total_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    tvar = stability.total_variance(x, y, tau_values)
    dst.set_xydata(tau_values, tvar)
    return dst


@computation_function(
    operation_id="sigima.signal.time_deviation",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def time_deviation(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Time Deviation (TDEV).

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "time_deviation", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    tdev = stability.time_deviation(x, y, tau_values)
    dst.set_xydata(tau_values, tdev)
    return dst
