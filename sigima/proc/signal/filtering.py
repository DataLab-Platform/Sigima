# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see sigima/LICENSE for details)

"""
Filtering processing functions for signal objects
=================================================

This module provides filtering operations for signal objects:

- Gaussian filter
- Moving average and median filters
- Wiener filter
- Frequency filters (low-pass, high-pass, band-pass, band-stop)
- Noise addition functions

.. note::

    Uses zero-phase filtering when possible for better phase response.
"""

from __future__ import annotations

import warnings
from typing import Callable

import guidata.dataset as gds
import numpy as np
import scipy.ndimage as spi
import scipy.signal as sps

from sigima.config import _
from sigima.enums import FilterType, FrequencyFilterMethod, PadLocation1D
from sigima.objects import (
    NormalDistribution1DParam,
    PoissonDistribution1DParam,
    SignalObj,
    UniformDistribution1DParam,
    create_signal_from_param,
)
from sigima.objects.base import (
    NormalDistributionParam,
    PoissonDistributionParam,
    UniformDistributionParam,
)
from sigima.proc.base import GaussianParam, MovingAverageParam, MovingMedianParam
from sigima.proc.decorator import computation_function
from sigima.proc.signal.arithmetic import addition
from sigima.proc.signal.base import Wrap1to1Func, dst_1_to_1, restore_data_outside_roi
from sigima.proc.signal.fourier import ZeroPadding1DParam, zero_padding
from sigima.tools.signal import fourier


@computation_function()
def gaussian_filter(src: SignalObj, p: GaussianParam) -> SignalObj:
    """Compute gaussian filter with :py:func:`scipy.ndimage.gaussian_filter`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(spi.gaussian_filter, sigma=p.sigma)(src)


@computation_function()
def moving_average(src: SignalObj, p: MovingAverageParam) -> SignalObj:
    """Compute moving average with :py:func:`scipy.ndimage.uniform_filter`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(
        spi.uniform_filter, size=p.n, mode=p.mode, func_name="moving_average"
    )(src)


@computation_function()
def moving_median(src: SignalObj, p: MovingMedianParam) -> SignalObj:
    """Compute moving median with :py:func:`scipy.ndimage.median_filter`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(
        spi.median_filter, size=p.n, mode=p.mode, func_name="moving_median"
    )(src)


@computation_function()
def wiener(src: SignalObj) -> SignalObj:
    """Compute Wiener filter with :py:func:`scipy.signal.wiener`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(sps.wiener)(src)


def get_nyquist_frequency(obj: SignalObj) -> float:
    """Return the Nyquist frequency of a signal object

    Args:
        obj: signal object

    Returns:
        Nyquist frequency
    """
    fs = float(obj.x.size - 1) / (obj.x[-1] - obj.x[0])
    return fs / 2.0


class BaseHighLowBandParam(gds.DataSet, title=_("Filter")):
    """Base class for high-pass, low-pass, band-pass and band-stop filters"""

    TYPE = FilterType.LOWPASS
    _type_prop = gds.GetAttrProp("TYPE")

    # Must be overwriten by the child class
    _method_prop = gds.GetAttrProp("method")
    method = gds.ChoiceItem(
        _("Filter method"),
        [
            (FrequencyFilterMethod.BUTTERWORTH, "Butterworth"),
            (FrequencyFilterMethod.BESSEL, "Bessel"),
            (FrequencyFilterMethod.CHEBYSHEV1, "Chebyshev I"),
            (FrequencyFilterMethod.CHEBYSHEV2, "Chebyshev II"),
            (FrequencyFilterMethod.ELLIPTIC, "Elliptic"),
            (FrequencyFilterMethod.BRICKWALL, "Brickwall"),
        ],
    ).set_prop("display", store=_method_prop)

    def get_filter_func(self) -> Callable:
        """Get the scipy filter function corresponding to the method."""
        filter_funcs = {
            FrequencyFilterMethod.BESSEL: sps.bessel,
            FrequencyFilterMethod.BUTTERWORTH: sps.butter,
            FrequencyFilterMethod.CHEBYSHEV1: sps.cheby1,
            FrequencyFilterMethod.CHEBYSHEV2: sps.cheby2,
            FrequencyFilterMethod.ELLIPTIC: sps.ellip,
        }
        return filter_funcs.get(self.method)

    order = gds.IntItem(_("Filter order"), default=3, min=1).set_prop(
        "display",
        active=gds.FuncProp(
            _method_prop, lambda x: x != FrequencyFilterMethod.BRICKWALL
        ),
    )
    cut0 = gds.FloatItem(
        _("Low cutoff frequency"), min=0.0, nonzero=True, unit="Hz", allow_none=True
    )
    cut1 = gds.FloatItem(
        _("High cutoff frequency"), min=0.0, nonzero=True, unit="Hz", allow_none=True
    ).set_prop(
        "display",
        hide=gds.FuncProp(
            _type_prop, lambda x: x in (FilterType.LOWPASS, FilterType.HIGHPASS)
        ),
    )
    rp = gds.FloatItem(
        _("Passband ripple"), min=0.0, default=1.0, nonzero=True, unit="dB"
    ).set_prop(
        "display",
        active=gds.FuncProp(
            _method_prop,
            lambda x: (
                x in (FrequencyFilterMethod.CHEBYSHEV1, FrequencyFilterMethod.ELLIPTIC)
            ),
        ),
    )
    rs = gds.FloatItem(
        _("Stopband attenuation"), min=0.0, default=60.0, nonzero=True, unit="dB"
    ).set_prop(
        "display",
        active=gds.FuncProp(
            _method_prop,
            lambda x: (
                x in (FrequencyFilterMethod.CHEBYSHEV2, FrequencyFilterMethod.ELLIPTIC)
            ),
        ),
    )

    _zp_prop = gds.GetAttrProp("zero_padding")
    zero_padding = gds.BoolItem(
        _("Zero padding"),
        default=True,
    ).set_prop(
        "display",
        active=gds.FuncProp(
            _method_prop, lambda x: x == FrequencyFilterMethod.BRICKWALL
        ),
        store=_zp_prop,
    )
    nfft = gds.IntItem(
        _("Minimum FFT points number"),
        default=0,
    ).set_prop(
        "display",
        active=gds.FuncPropMulti(
            [_method_prop, _zp_prop],
            lambda x, y: x == FrequencyFilterMethod.BRICKWALL and y,
        ),
    )

    def validate_parameters(self, *context: object) -> None:
        """Validate active cutoff frequencies against the filter method."""
        if not context or not isinstance(context[0], SignalObj):
            raise ValueError("filter validation requires a source SignalObj")
        if self.cut0 is None or self.cut0 <= 0.0:
            raise ValueError("cut0 must be strictly positive")

        is_band_filter = self.TYPE in (FilterType.BANDPASS, FilterType.BANDSTOP)
        if is_band_filter:
            if self.cut1 is None or self.cut1 <= 0.0:
                raise ValueError("cut1 must be strictly positive for band filters")
            if self.method == FrequencyFilterMethod.BRICKWALL:
                if self.cut0 > self.cut1:
                    raise ValueError("cut0 must be less than or equal to cut1")
            elif self.cut0 >= self.cut1:
                raise ValueError("cut0 must be strictly less than cut1")

        if self.method != FrequencyFilterMethod.BRICKWALL:
            f_nyquist = get_nyquist_frequency(context[0])
            active_cutoffs = [self.cut0]
            if is_band_filter:
                active_cutoffs.append(self.cut1)
            if any(cutoff >= f_nyquist for cutoff in active_cutoffs):
                raise ValueError(
                    "IIR cutoff frequencies must be strictly below Nyquist"
                )

    def update_from_obj(self, obj: SignalObj) -> None:
        """Update the filter parameters from a signal object

        Args:
            obj: signal object
        """
        f_nyquist = get_nyquist_frequency(obj)
        if self.cut0 is None:
            if self.TYPE == FilterType.LOWPASS:
                self.cut0 = 0.1 * f_nyquist
            elif self.TYPE == FilterType.HIGHPASS:
                self.cut0 = 0.9 * f_nyquist
            elif self.TYPE == FilterType.BANDPASS:
                self.cut0 = 0.1 * f_nyquist
                self.cut1 = 0.9 * f_nyquist
            elif self.TYPE == FilterType.BANDSTOP:
                self.cut0 = 0.4 * f_nyquist
                self.cut1 = 0.6 * f_nyquist

    def get_filter_params(self, obj: SignalObj) -> tuple[float | str, float | str]:
        """Return the filter parameters (a and b) as a tuple. These parameters are used
        in the scipy.signal filter functions (eg. `scipy.signal.filtfilt`).

        Args:
            obj: signal object

        Returns:
            tuple: filter parameters
        """
        f_nyquist = get_nyquist_frequency(obj)
        args: list[float | str | tuple[float, ...]] = [self.order]  # type: ignore
        if self.method == FrequencyFilterMethod.CHEBYSHEV1:
            args += [self.rp]
        elif self.method == FrequencyFilterMethod.CHEBYSHEV2:
            args += [self.rs]
        elif self.method == FrequencyFilterMethod.ELLIPTIC:
            args += [self.rp, self.rs]
        if self.TYPE in (FilterType.HIGHPASS, FilterType.LOWPASS):
            args += [self.cut0 / f_nyquist]
        else:
            args += [[self.cut0 / f_nyquist, self.cut1 / f_nyquist]]
        args += [self.TYPE.value]
        return self.get_filter_func()(*args)


class LowPassFilterParam(BaseHighLowBandParam):
    """Low-pass filter parameters"""

    TYPE = FilterType.LOWPASS

    # Redefine cut0 just to change its label (instead of "Low cutoff frequency")
    cut0 = gds.FloatItem(
        _("Cutoff frequency"), min=0, nonzero=True, unit="Hz", allow_none=True
    )


class HighPassFilterParam(BaseHighLowBandParam):
    """High-pass filter parameters"""

    TYPE = FilterType.HIGHPASS

    # Redefine cut0 just to change its label (instead of "High cutoff frequency")
    cut0 = gds.FloatItem(
        _("Cutoff frequency"), min=0, nonzero=True, unit="Hz", allow_none=True
    )


class BandPassFilterParam(BaseHighLowBandParam):
    """Band-pass filter parameters"""

    TYPE = FilterType.BANDPASS


class BandStopFilterParam(BaseHighLowBandParam):
    """Band-stop filter parameters"""

    TYPE = FilterType.BANDSTOP


def frequency_filter(src: SignalObj, p: BaseHighLowBandParam) -> SignalObj:
    """Compute frequency filter (low-pass, high-pass, band-pass, band-stop),
    with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object

    .. note::

        Uses zero-phase filtering (`filtfilt`) when possible for better phase response.
        If numerical instability occurs (e.g., singular matrix errors), automatically
        falls back to forward filtering (`lfilter`) with a warning. This ensures
        cross-platform compatibility while maintaining optimal filtering when possible.
    """
    name = f"{p.TYPE.value}"
    suffix = ""
    if p.method != FrequencyFilterMethod.BRICKWALL:
        suffix = f"order={p.order:d}, "
    if p.TYPE in (FilterType.LOWPASS, FilterType.HIGHPASS):
        suffix += f"cutoff={p.cut0:.2f}"
    else:
        suffix += f"cutoff={p.cut0:.2f}:{p.cut1:.2f}"
    dst = dst_1_to_1(src, name, suffix)

    if p.method == FrequencyFilterMethod.BRICKWALL:
        original_size = src.y.size
        src_padded = src.copy()
        if p.zero_padding and p.nfft is not None:
            size_padded = ZeroPadding1DParam.next_power_of_two(max(p.nfft, src.y.size))
            n_to_add = size_padded - src.y.size
            if n_to_add > 0:
                src_padded = zero_padding(
                    src_padded,
                    ZeroPadding1DParam.create(
                        location=PadLocation1D.APPEND,
                        strategy="custom",
                        n=n_to_add,
                    ),
                )
        x_padded, y_padded = src_padded.get_data()
        x, y = fourier.brickwall_filter(
            x_padded, y_padded, p.TYPE.value, p.cut0, p.cut1
        )
        # Trim back to original size if padding was applied
        x = x[:original_size]
        y = y[:original_size]
        dst.set_xydata(x, y)
    else:
        b, a = p.get_filter_params(dst)
        try:
            # Prefer zero-phase filtering
            dst.y = sps.filtfilt(b, a, dst.y)
        except np.linalg.LinAlgError:
            # Fallback to forward filtering if filtfilt fails due to numerical issues
            warnings.warn(
                "Zero-phase filtering failed due to numerical instability. "
                "Using forward filtering instead.",
                UserWarning,
                stacklevel=2,
            )
            dst.y = sps.lfilter(b, a, dst.y)

    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def lowpass(src: SignalObj, p: LowPassFilterParam) -> SignalObj:
    """Compute low-pass filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


@computation_function()
def highpass(src: SignalObj, p: HighPassFilterParam) -> SignalObj:
    """Compute high-pass filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


@computation_function()
def bandpass(src: SignalObj, p: BandPassFilterParam) -> SignalObj:
    """Compute band-pass filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


@computation_function()
def bandstop(src: SignalObj, p: BandStopFilterParam) -> SignalObj:
    """Compute band-stop filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


# Noise addition functions
@computation_function()
def add_gaussian_noise(src: SignalObj, p: NormalDistributionParam) -> SignalObj:
    """Add normal noise to the input signal.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    param = NormalDistribution1DParam()  # Do not confuse with NormalDistributionParam
    gds.update_dataset(param, p)
    param.xmin = src.x[0]
    param.xmax = src.x[-1]
    param.size = src.x.size
    noise = create_signal_from_param(param)
    dst = dst_1_to_1(src, "add_gaussian_noise", f"µ={p.mu}, σ={p.sigma}")
    dst.xydata = addition([src, noise]).xydata
    return dst


@computation_function()
def add_poisson_noise(src: SignalObj, p: PoissonDistributionParam) -> SignalObj:
    """Add Poisson noise to the input signal.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    param = PoissonDistribution1DParam()  # Do not confuse with PoissonDistributionParam
    gds.update_dataset(param, p)
    param.xmin = src.x[0]
    param.xmax = src.x[-1]
    param.size = src.x.size
    noise = create_signal_from_param(param)
    dst = dst_1_to_1(src, "add_poisson_noise", f"λ={p.lam}")
    dst.xydata = addition([src, noise]).xydata
    return dst


@computation_function()
def add_uniform_noise(src: SignalObj, p: UniformDistributionParam) -> SignalObj:
    """Add uniform noise to the input signal.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    param = UniformDistribution1DParam()  # Do not confuse with UniformDistributionParam
    gds.update_dataset(param, p)
    param.xmin = src.x[0]
    param.xmax = src.x[-1]
    param.size = src.x.size
    noise = create_signal_from_param(param)
    dst = dst_1_to_1(src, "add_uniform_noise", f"low={p.vmin}, high={p.vmax}")
    dst.xydata = addition([src, noise]).xydata
    return dst
