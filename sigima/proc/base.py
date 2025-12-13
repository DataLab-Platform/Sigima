# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Common computation objects (see parent package :mod:`sigima.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the sigima.params module.

from __future__ import annotations

from typing import TypeVar, cast

import guidata.dataset as gds
import numpy as np

from sigima import ImageObj, SignalObj, create_signal
from sigima.config import _, options
from sigima.enums import (
    AngleUnit,
    FilterMode,
    MathOperator,
    NormalizationMethod,
    SignalsToImageOrientation,
)
from sigima.proc.title_formatting import get_default_title_formatter

# NOTE: This module is a shared utilities library that defines common parameter classes
# used by multiple other modules (signal processing, image processing, etc.).
# Unlike other modules, the parameter classes DEFINED in this module should NOT be
# included in __all__ because they are imported and re-exported by the modules that
# use them, and including them here would create Sphinx cross-reference conflicts.
# The sigima.params module serves as the central API point that imports and re-exports
# all parameter classes from their canonical locations.
__all__ = [
    "dst_1_to_1",
    "dst_2_to_1",
    "dst_n_to_1",
    "new_signal_result",
]


class ArithmeticParam(gds.DataSet, title=_("Arithmetic")):
    """Arithmetic parameters"""

    def get_operation(self) -> str:
        """Return the operation string"""
        o, a, b = self.operator, self.factor, self.constant
        b_added = False
        if a == 0.0:
            if o in ("+", "-"):
                txt = "obj3 = obj1"
            elif b == 0.0:
                txt = "obj3 = 0"
            else:
                txt = f"obj3 = {b}"
                b_added = True
        elif a == 1.0:
            txt = f"obj3 = obj1 {o} obj2"
        else:
            txt = f"obj3 = (obj1 {o} obj2) × {a}"
        if b != 0.0 and not b_added:
            txt += f" + {b}"
        return txt

    def update_operation(self, _item, _value):  # pylint: disable=unused-argument
        """Update the operation item"""
        self.operation = self.get_operation()

    operator = gds.ChoiceItem(
        _("Operator"), MathOperator, default=MathOperator.ADD
    ).set_prop("display", callback=update_operation)
    factor = (
        gds.FloatItem(_("Factor"), default=1.0)
        .set_pos(col=1)
        .set_prop("display", callback=update_operation)
    )
    constant = (
        gds.FloatItem(_("Constant"), default=0.0)
        .set_pos(col=1)
        .set_prop("display", callback=update_operation)
    )
    operation = gds.StringItem(_("Operation"), default="").set_prop(
        "display", active=False
    )
    restore_dtype = gds.BoolItem(
        _("Convert to `obj1` data type"), label=_("Result"), default=True
    )


class GaussianParam(gds.DataSet, title=_("Gaussian filter")):
    """Gaussian filter parameters."""

    sigma = gds.FloatItem(
        "σ",
        default=1.0,
        min=0.0,
        help=_("Standard deviation of the Gaussian filter"),
    )


HELP_MODE = _("""Mode of the filter:
- 'reflect': Reflect the data at the boundary
- 'constant': Pad with a constant value
- 'nearest': Pad with the nearest value
- 'mirror': Reflect the data at the boundary with the data itself
- 'wrap': Circular boundary""")


class MovingAverageParam(gds.DataSet, title=_("Moving average")):
    """Moving average parameters"""

    n = gds.IntItem(_("Size of the moving window"), default=3, min=1)
    mode = gds.ChoiceItem(
        _("Mode"), FilterMode, default=FilterMode.REFLECT, help=HELP_MODE
    )


class MovingMedianParam(gds.DataSet, title=_("Moving median")):
    """Moving median parameters"""

    n = gds.IntItem(_("Size of the moving window"), default=3, min=1, even=False)
    mode = gds.ChoiceItem(
        _("Mode"), FilterMode, default=FilterMode.NEAREST, help=HELP_MODE
    )


class ClipParam(gds.DataSet, title=_("Clip")):
    """Data clipping parameters"""

    lower = gds.FloatItem(_("Lower clipping value"), check=False)
    upper = gds.FloatItem(_("Upper clipping value"), check=False)


class NormalizeParam(gds.DataSet, title=_("Normalize")):
    """Normalize parameters"""

    method = gds.ChoiceItem(_("Normalize with respect to"), NormalizationMethod)


class HistogramParam(gds.DataSet, title=_("Histogram")):
    """Histogram parameters"""

    def get_suffix(self, data: np.ndarray) -> str:
        """Return suffix for the histogram computation

        Args:
            data: data array
        """
        suffix = f"bins={self.bins:d}"
        if self.lower is not None:
            suffix += f", ymin={self.lower:.3f}"
        else:
            self.lower = np.min(data)
        if self.upper is not None:
            suffix += f", ymax={self.upper:.3f}"
        else:
            self.upper = np.max(data)
        return suffix

    bins = gds.IntItem(_("Number of bins"), default=256, min=1)
    lower = gds.FloatItem(_("Lower limit"), default=None, check=False)
    upper = gds.FloatItem(_("Upper limit"), default=None, check=False)


class FFTParam(gds.DataSet, title=_("FFT")):
    """FFT parameters"""

    shift = gds.BoolItem(_("Shift"), help=_("Shift zero frequency to center"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift = options.fft_shift_enabled.get()


class SpectrumParam(gds.DataSet, title=_("Spectrum")):
    """Spectrum parameters."""

    decibel = gds.BoolItem(_("Output in decibel (dB)"), default=False)


class ConstantParam(gds.DataSet, title=_("Constant")):
    """Parameter used to set a constant value to used in operations"""

    value = gds.FloatItem(_("Constant value"))


class AngleUnitParam(gds.DataSet, title=_("Angle unit")):
    """Choice of angle unit."""

    unit = gds.ChoiceItem(
        _("Angle unit"),
        AngleUnit,
        default=AngleUnit.RADIAN,
        help=_("Unit of angle measurement"),
    )


class PhaseParam(gds.DataSet, title=_("Phase")):
    """Parameters for phase computation."""

    unwrap = gds.BoolItem(
        "unwrap", default=True, help=_("Unwrapping removes discontinuities in phase")
    )
    unit = gds.ChoiceItem(
        _("Unit"),
        AngleUnit,
        default=AngleUnit.DEGREE,
        help=_("Unit of angle measurement"),
    )


class SignalsToImageParam(gds.DataSet, title=_("Signals to image")):
    """Parameters for assembling signals into an image."""

    orientation = gds.ChoiceItem(
        _("Orientation"),
        SignalsToImageOrientation,
        default=SignalsToImageOrientation.ROWS,
        help=_("Stack signals as rows or columns in the output image"),
    )
    _prop = gds.GetAttrProp("normalize")
    normalize = gds.BoolItem(
        _("Normalize"),
        default=False,
        help=_("Normalize each signal before combining"),
    )
    normalize_method = gds.ChoiceItem(
        _("Normalization method"),
        NormalizationMethod,
        default=NormalizationMethod.MAXIMUM,
        help=_("Method used for normalization"),
    ).set_prop("display", active=_prop)


# MARK: Helper functions for creating result objects -----------------------------------

Obj = TypeVar("Obj", bound="SignalObj | ImageObj")


def dst_1_to_1(src: Obj, name: str, suffix: str | None = None) -> Obj:
    """Create a result object, for processing functions that take a single
    signal or image object as input and return a single signal or image object (1-to-1).

    .. note::

        Data of the result object is copied from the source object (`src`).
        This initial data is usually replaced by the processing function, but it may
        also be used to initialize the result object as part of the processing function.

    Args:
        src: source signal or image object
        name: name of the function. The title format depends on the configured
         title formatter (SimpleTitleFormatter creates readable titles,
         PlaceholderTitleFormatter creates DataLab-compatible placeholder titles).
        suffix: suffix to add to the title. Optional.

    Returns:
        Result signal or image object
    """
    formatter = get_default_title_formatter()
    title = formatter.format_1_to_1_title(name, suffix)
    dst = src.copy(title=title)
    return cast(Obj, dst)


def dst_n_to_1(src_list: list[Obj], name: str, suffix: str | None = None) -> Obj:
    """Create a result object, for processing functions that take a list of signal or
    image objects as input and return a single signal or image object (n-to-1).

    .. note::

        Data of the result object is copied from the first source object
        (`src_list[0]`). This initial data is usually replaced by the processing
        function, but it may also be used to initialize the result object as part
        of the processing function.

    Args:
        src_list: list of input signal or image objects
        name: name of the processing function. The title format depends on the
         configured title formatter (SimpleTitleFormatter creates readable titles,
         PlaceholderTitleFormatter creates DataLab-compatible placeholder titles).
        suffix: suffix to add to the title

    Returns:
        Result signal or image object
    """
    if not isinstance(src_list, list) or len(src_list) <= 1:
        raise ValueError("src_list must be a list of at least 2 objects")
    all_sigs = all(isinstance(obj, SignalObj) for obj in src_list)
    all_imgs = all(isinstance(obj, ImageObj) for obj in src_list)
    if not (all_sigs or all_imgs):
        raise ValueError("src_list must be a list of SignalObj or ImageObj objects")

    formatter = get_default_title_formatter()
    title = formatter.format_n_to_1_title(name, len(src_list), suffix)

    if any(np.issubdtype(obj.data.dtype, complex) for obj in src_list):
        dst_dtype = complex
    else:
        dst_dtype = float
    dst = src_list[0].copy(title=title, dtype=dst_dtype)
    dst.roi = None
    for src_obj in src_list:
        if src_obj.roi is not None:
            if dst.roi is None:
                dst.roi = src_obj.roi.copy()
            else:
                dst.roi.add_roi(src_obj.roi)
    return dst


# Note about `src2` parameter:
# ----------------------------
# The `src2` parameter is currently not used in the function, but it is included
# to maintain a consistent interface with other similar functions (e.g., `dst_n_to_1`).
# This may be useful in the future if we want to extend the functionality.
#
# pylint: disable=unused-argument
def dst_2_to_1(src1: Obj, src2: Obj, name: str, suffix: str | None = None) -> Obj:
    """Create a result  object, for processing functions that take two signal or
    image objects as input and return a single signal or image object (2-to-1).

    .. note::

        Data of the result object is copied from the first source object (`src1`).
        This initial data is usually replaced by the processing function, but it may
        also be used to initialize the result object as part of the processing function.

    Args:
        src1: input signal or image object
        src2: input signal or image object
        name: name of the processing function. The title format depends on the
         configured title formatter (SimpleTitleFormatter creates readable titles,
         PlaceholderTitleFormatter creates DataLab-compatible placeholder titles).
        suffix: suffix to add to the title

    Returns:
        Output signal or image object
    """
    formatter = get_default_title_formatter()
    title = formatter.format_2_to_1_title(name, suffix)
    dst = src1.copy(title=title)
    return dst


def new_signal_result(
    src: SignalObj | ImageObj,
    name: str,
    suffix: str | None = None,
    units: tuple[str, str] | None = None,
    labels: tuple[str, str] | None = None,
) -> SignalObj:
    """Create new signal object as a result of a `compute_1_to_1` function

    As opposed to the `dst_1_to_1` functions, this function creates a new signal object
    without copying the original object metadata, except for the "source" entry.

    Args:
        src: input signal or image object
        name: name of the processing function. The title format depends on the
         configured title formatter (SimpleTitleFormatter creates readable titles,
         PlaceholderTitleFormatter creates DataLab-compatible placeholder titles).
        suffix: suffix to add to the title
        units: units of the output signal
        labels: labels of the output signal

    Returns:
        Output signal object
    """
    formatter = get_default_title_formatter()
    title = formatter.format_1_to_1_title(name, suffix)
    dst = create_signal(title=title, units=units, labels=labels)
    if (source := src.metadata.get("source")) is not None:
        dst.metadata["source"] = source
    return dst
