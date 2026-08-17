# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Common enum definitions for Sigima processing."""

from __future__ import annotations

import guidata.dataset as gds

from sigima.config import _


class AngleUnit(gds.LabeledEnum):
    """Angle units."""

    RADIAN = "rad"
    DEGREE = "°"


class BinningOperation(gds.LabeledEnum):
    """Binning operations for image processing."""

    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"


class ContourShape(gds.LabeledEnum):
    """Contour shapes for image processing."""

    ELLIPSE = "ellipse", _("Ellipse")
    CIRCLE = "circle", _("Circle")
    POLYGON = "polygon", _("Polygon")


class BorderMode(gds.LabeledEnum):
    """Border modes for filtering and image processing."""

    CONSTANT = "constant"
    NEAREST = "nearest"
    REFLECT = "reflect"
    WRAP = "wrap"
    MIRROR = "mirror"


class MathOperator(gds.LabeledEnum):
    """Mathematical operators for data operations."""

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "×"
    DIVIDE = "/"


class FilterMode(gds.LabeledEnum):
    """Filter modes for signal and image processing."""

    REFLECT = "reflect"
    CONSTANT = "constant"
    NEAREST = "nearest"
    MIRROR = "mirror"
    WRAP = "wrap"


class WaveletMode(gds.LabeledEnum):
    """Wavelet transform modes."""

    CONSTANT = "constant"
    EDGE = "edge"
    SYMMETRIC = "symmetric"
    REFLECT = "reflect"
    WRAP = "wrap"


class ThresholdMethod(gds.LabeledEnum):
    """Thresholding methods for wavelet denoising."""

    SOFT = "soft"
    HARD = "hard"


class ShrinkageMethod(gds.LabeledEnum):
    """Shrinkage methods for wavelet denoising."""

    BAYES_SHRINK = "BayesShrink"
    VISU_SHRINK = "VisuShrink"


class PadLocation1D(gds.LabeledEnum):
    """Padding location for 1D signal processing."""

    APPEND = "append"
    PREPEND = "prepend"
    BOTH = "both"


class PadLocation2D(gds.LabeledEnum):
    """Padding location for 2D image processing."""

    BOTTOM_RIGHT = "bottom-right", _("Bottom-right")
    AROUND = "around", _("Around")


class PowerUnit(gds.LabeledEnum):
    """Power spectral density units."""

    DBC = "dBc"
    DBFS = "dBFS"


class WindowingMethod(gds.LabeledEnum):
    """Windowing methods enumeration."""

    BARTHANN = "barthann", "Barthann"
    BARTLETT = "bartlett", "Bartlett"
    BLACKMAN = "blackman", "Blackman"
    BLACKMAN_HARRIS = "blackman_harris", "Blackman-Harris"
    BOHMAN = "bohman", "Bohman"
    BOXCAR = "boxcar", "Boxcar"
    COSINE = "cosine", _("Cosine")
    EXPONENTIAL = "exponential", _("Exponential")
    FLAT_TOP = "flat_top", _("Flat Top")
    GAUSSIAN = "gaussian", _("Gaussian")
    HAMMING = "hamming", "Hamming"
    HANN = "hann", "Hann"
    KAISER = "kaiser", "Kaiser"
    LANCZOS = "lanczos", "Lanczos"
    NUTTALL = "nuttall", "Nuttall"
    PARZEN = "parzen", "Parzen"
    TAYLOR = "taylor", "Taylor"
    TUKEY = "tukey", "Tukey"


class Interpolation1DMethod(gds.LabeledEnum):
    """Methods for 1D interpolation and resampling."""

    LINEAR = "linear", _("Linear")
    SPLINE = "spline", _("Spline")
    QUADRATIC = "quadratic", _("Quadratic")
    CUBIC = "cubic", _("Cubic")
    BARYCENTRIC = "barycentric", _("Barycentric")
    PCHIP = "pchip", _("PCHIP")


class Interpolation2DMethod(gds.LabeledEnum):
    """Methods for 2D interpolation and resampling."""

    NEAREST = "nearest", _("Nearest")
    LINEAR = "linear", _("Linear")
    CUBIC = "cubic", _("Cubic")


class NormalizationMethod(gds.LabeledEnum):
    """Normalization methods for signal processing."""

    MAXIMUM = "maximum", _("Maximum")
    AMPLITUDE = "amplitude", _("Amplitude")
    AREA = "area", _("Area")
    ENERGY = "energy", _("Energy")
    RMS = "rms", _("RMS")


class FilterType(gds.LabeledEnum):
    """Filter types"""

    LOWPASS = "lowpass", "lowpass"
    HIGHPASS = "highpass", "highpass"
    BANDPASS = "bandpass", "bandpass"
    BANDSTOP = "bandstop", "bandstop"


class FrequencyFilterMethod(gds.LabeledEnum):
    """Frequency filter methods for signal processing."""

    BESSEL = "bessel", "Bessel"
    BRICKWALL = "brickwall", _("Brickwall")
    BUTTERWORTH = "butterworth", "Butterworth"
    CHEBYSHEV1 = "chebyshev1", "Chebyshev I"
    CHEBYSHEV2 = "chebyshev2", "Chebyshev II"
    ELLIPTIC = "elliptic", _("Elliptic")


class SignalShape(gds.LabeledEnum):
    """Signal shapes."""

    STEP = "step"
    SQUARE = "square"


class SignalsToImageOrientation(gds.LabeledEnum):
    """Orientation for assembling signals into an image."""

    ROWS = "rows", _("Each signal becomes a row")
    COLUMNS = "columns", _("Each signal becomes a column")


class DetectionROIGeometry(gds.LabeledEnum):
    """Geometries for 2D peak detection ROIs."""

    CIRCLE = "circle", _("Circle")
    RECTANGLE = "rectangle", _("Rectangle")


class ReplacementStrategySignal(gds.LabeledEnum):
    """Replacement strategies for special values (NaN/Inf) in signals."""

    NONE = "none", _("Do nothing")
    ZERO = "zero", _("Replace with zero")
    CONSTANT = "constant", _("Replace with constant")
    MIN = "min", _("Replace with minimum")
    MAX = "max", _("Replace with maximum")
    MEAN = "mean", _("Replace with mean")
    MEDIAN = "median", _("Replace with median")
    DELETE = "delete", _("Delete points")
    FORWARD_FILL = "ffill", _("Forward fill (previous value)")
    BACKWARD_FILL = "bfill", _("Backward fill (next value)")
    INTERP_LINEAR = "interp_linear", _("Interpolation: Linear")
    INTERP_SPLINE = "interp_spline", _("Interpolation: Spline")
    INTERP_QUADRATIC = "interp_quadratic", _("Interpolation: Quadratic")
    INTERP_CUBIC = "interp_cubic", _("Interpolation: Cubic")
    INTERP_PCHIP = "interp_pchip", _("Interpolation: PCHIP")
    NEIGHBOR_MIN = "neighbor_min", _("N-neighbor minimum")
    NEIGHBOR_MAX = "neighbor_max", _("N-neighbor maximum")
    NEIGHBOR_MEAN = "neighbor_mean", _("N-neighbor mean")
    NEIGHBOR_MEDIAN = "neighbor_median", _("N-neighbor median")


class ReplacementStrategyImage(gds.LabeledEnum):
    """Replacement strategies for special values (NaN/Inf) in images."""

    NONE = "none", _("Do nothing")
    ZERO = "zero", _("Replace with zero")
    CONSTANT = "constant", _("Replace with constant")
    MIN = "min", _("Replace with minimum")
    MAX = "max", _("Replace with maximum")
    MEAN = "mean", _("Replace with mean")
    MEDIAN = "median", _("Replace with median")
    NEIGHBOR_MIN = "neighbor_min", _("N-neighbor minimum")
    NEIGHBOR_MAX = "neighbor_max", _("N-neighbor maximum")
    NEIGHBOR_MEAN = "neighbor_mean", _("N-neighbor mean")
    NEIGHBOR_MEDIAN = "neighbor_median", _("N-neighbor median")
