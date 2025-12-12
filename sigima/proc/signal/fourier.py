# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see sigima/LICENSE for details)

"""
Fourier transform and frequency domain operations
=================================================

This module provides Fourier transform and frequency domain operations:

- FFT and inverse FFT
- Magnitude and phase spectrum
- Power spectral density (PSD)

.. note::

    Most operations use functions from :mod:`sigima.tools.signal.fourier` for actual
    computations.
"""

from __future__ import annotations

from math import ceil, log2

import guidata.dataset as gds

from sigima.config import _
from sigima.enums import PadLocation1D
from sigima.objects import SignalObj
from sigima.proc.base import FFTParam, SpectrumParam
from sigima.proc.decorator import computation_function
from sigima.proc.signal.base import dst_1_to_1
from sigima.tools.signal import fourier


class ZeroPadding1DParam(gds.DataSet, title=_("Zero padding")):
    """Zero-padding parameters for signals.

    This class manages parameters for applying zero-padding to signals,
    commonly used to improve FFT resolution or prepare signals for convolution.

    .. important::

        For strategies other than "custom", the number of points to add (``n``)
        is **automatically calculated** based on the signal size. However, this
        calculation requires knowledge of the signal, so you **must call**
        :meth:`update_from_obj` before using the parameters.

    Example usage:

    .. code-block:: python

        import sigima.params
        import sigima.proc.signal as sips

        # Create the parameter object
        param = sigima.params.ZeroPadding1DParam.create(strategy="next_pow2")

        # IMPORTANT: Update parameters from the signal to compute 'n'
        param.update_from_obj(signal)

        # Now the parameters are ready to use
        result = sips.zero_padding(signal, param)

    Attributes:

    - strategies: Available strategies ("next_pow2", "double", "triple", "custom").
    - strategy: Choice item for selecting the zero-padding strategy.

      - ``"next_pow2"``: Pad to the next power of 2 (optimal for FFT)
      - ``"double"``: Double the signal length
      - ``"triple"``: Triple the signal length
      - ``"custom"``: Use a user-specified number of points

    - location: Where to add the padding ("append", "prepend", or "both").
    - n: Number of points to add as padding. For "custom" strategy, this is
      user-specified. For other strategies, it is computed automatically
      by :meth:`update_from_obj`.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize zero padding parameters.

        Args:
            *args: Variable length argument list passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.__obj: SignalObj | None = None

    def update_from_obj(self, obj: SignalObj) -> None:
        """Update parameters based on a signal object.

        This method computes the number of padding points (``n``) based on
        the selected strategy and the actual signal size. **This must be called
        before using the parameters** for strategies other than "custom".

        Args:
            obj: Signal object from which to compute the padding parameters.
        """
        self.__obj = obj
        self.strategy_callback(None, self.strategy)

    @staticmethod
    def next_power_of_two(size: int) -> int:
        """Compute the next power of two greater than or equal to the given size.

        Args:
            size: The input integer.

        Returns:
            The smallest power of two greater than or equal to 'size'.
        """
        return 2 ** (ceil(log2(size)))

    def strategy_callback(self, _, value):
        """Callback for strategy choice item.

        Args:
            _: Unused argument (in this context).
            value: The selected strategy value.
        """
        if self.__obj is None:
            return
        assert self.__obj.x is not None
        size = self.__obj.x.size
        if value == "next_pow2":
            self.n = self.next_power_of_two(size) - size
        elif value == "double":
            self.n = size
        elif value == "triple":
            self.n = 2 * size

    strategies = ("next_pow2", "double", "triple", "custom")
    _prop = gds.GetAttrProp("strategy")
    strategy = gds.ChoiceItem(
        _("Strategy"), zip(strategies, strategies), default=strategies[0]
    ).set_prop("display", store=_prop, callback=strategy_callback)
    location = gds.ChoiceItem(
        _("Location"),
        PadLocation1D,
        default=PadLocation1D.APPEND,
        help=_("Where to add the padding"),
    )
    _func_prop = gds.FuncProp(_prop, lambda x: x == "custom")
    n = gds.IntItem(
        _("Number of points"), min=1, default=1, help=_("Number of points to add")
    ).set_prop("display", active=_func_prop)


@computation_function()
def zero_padding(src: SignalObj, p: ZeroPadding1DParam) -> SignalObj:
    """Compute zero padding with :py:func:`sigima.tools.signal.fourier.zero_padding`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    if p.strategy == "custom":
        suffix = f"n={p.n}"
    else:
        suffix = f"strategy={p.strategy}"

    assert p.n is not None
    if p.location == PadLocation1D.APPEND:
        n_prepend = 0
        n_append = p.n
    elif p.location == PadLocation1D.PREPEND:
        n_prepend = p.n
        n_append = 0
    else:
        # At this point, we must have BOTH (last option)
        assert p.location == PadLocation1D.BOTH
        n_prepend = p.n // 2
        n_append = p.n - n_prepend

    dst = dst_1_to_1(src, "zero_padding", suffix)
    x, y = src.get_data()
    x_padded, y_padded = fourier.zero_padding(x, y, n_prepend, n_append)
    dst.set_xydata(x_padded, y_padded)

    return dst


@computation_function()
def fft(src: SignalObj, p: FFTParam | None = None) -> SignalObj:
    """Compute FFT with :py:func:`sigima.tools.signal.fourier.fft1d`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    if p is None:
        p = FFTParam()
    dst = dst_1_to_1(src, "fft")
    x, y = src.get_data()
    fft_x, fft_y = fourier.fft1d(x, y, shift=p.shift)
    dst.set_xydata(fft_x, fft_y)
    dst.save_attr_to_metadata("xunit", "Hz" if dst.xunit == "s" else "")
    dst.save_attr_to_metadata("yunit", "")
    dst.save_attr_to_metadata("xlabel", _("Frequency"))
    return dst


@computation_function()
def ifft(src: SignalObj) -> SignalObj:
    """Compute the inverse FFT with :py:func:`sigima.tools.signal.fourier.ifft1d`.

    Args:
        src: Source signal.

    Returns:
        Result signal object.
    """
    dst = dst_1_to_1(src, "ifft")
    f, sp = src.get_data()
    x, y = fourier.ifft1d(f, sp)
    dst.set_xydata(x, y)
    dst.restore_attr_from_metadata("xunit", "s" if src.xunit == "Hz" else "")
    dst.restore_attr_from_metadata("yunit", "")
    dst.restore_attr_from_metadata("xlabel", "")
    return dst


@computation_function()
def magnitude_spectrum(src: SignalObj, p: SpectrumParam | None = None) -> SignalObj:
    """Compute magnitude spectrum.

    This function computes the magnitude spectrum of a signal using
    :py:func:`sigima.tools.signal.fourier.magnitude_spectrum`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    decibel = bool(p is not None and p.decibel)
    dst = dst_1_to_1(src, "magnitude_spectrum", f"dB={decibel}")
    x, y = src.get_data()
    mag_x, mag_y = fourier.magnitude_spectrum(x, y, decibel=decibel)
    dst.set_xydata(mag_x, mag_y)
    dst.xlabel = _("Frequency")
    dst.xunit = "Hz" if dst.xunit == "s" else ""
    dst.yunit = "dB" if decibel else ""
    return dst


@computation_function()
def phase_spectrum(src: SignalObj) -> SignalObj:
    """Compute phase spectrum.

    This function computes the phase spectrum of a signal using
    :py:func:`sigima.tools.signal.fourier.phase_spectrum`

    Args:
        src: Source signal.

    Returns:
        Result signal object.
    """
    dst = dst_1_to_1(src, "phase_spectrum")
    x, y = src.get_data()
    phase_x, phase_y = fourier.phase_spectrum(x, y)
    dst.set_xydata(phase_x, phase_y)
    dst.xlabel = _("Frequency")
    dst.xunit = "Hz" if dst.xunit == "s" else ""
    dst.yunit = ""
    return dst


@computation_function()
def psd(src: SignalObj, p: SpectrumParam | None = None) -> SignalObj:
    """Compute power spectral density with :py:func:`sigima.tools.signal.fourier.psd`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    decibel = p is not None and p.decibel
    dst = dst_1_to_1(src, "psd", f"dB={decibel}")
    x, y = src.get_data()
    psd_x, psd_y = fourier.psd(x, y, decibel=decibel)
    dst.set_xydata(psd_x, psd_y)
    dst.xlabel = _("Frequency")
    dst.xunit = "Hz" if dst.xunit == "s" else ""
    dst.yunit = "dB/Hz" if decibel else ""
    return dst
