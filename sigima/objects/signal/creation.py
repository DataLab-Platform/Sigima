# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal creation utilities
========================

This module provides functions and parameter classes for creating new signals.

The module includes:

- `create_signal_from_param`: Factory function for creating SignalObj instances
  from parameters
- `SignalTypes`: Enumeration of supported signal generation types
- `NewSignalParam` and subclasses: Parameter classes for signal generation
- Factory functions and registration utilities

These utilities support creating signals from various sources:
- Synthetic data (zeros, random distributions, analytical functions)
- Periodic functions (sine, cosine, square, etc.)
- Step functions, chirps, pulses
- Custom user-defined signals
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal, Type

import guidata.dataset as gds
import numpy as np
import scipy.constants
import scipy.signal as sps

from sigima.config import _
from sigima.enums import SignalShape
from sigima.objects import base
from sigima.objects.signal.object import SignalObj
from sigima.tools.signal.pulse import GaussianModel, LorentzianModel, VoigtModel


def create_signal(
    title: str,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
    metadata: dict | None = None,
    units: tuple[str, str] | None = None,
    labels: tuple[str, str] | None = None,
) -> SignalObj:
    """Create a new Signal object.

    Args:
        title: signal title
        x: X data
        y: Y data
        dx: dX data (optional: error bars)
        dy: dY data (optional: error bars)
        metadata: signal metadata
        units: X, Y units (tuple of strings)
        labels: X, Y labels (tuple of strings)

    Returns:
        Signal object
    """
    assert isinstance(title, str)
    signal = SignalObj(title=title)
    signal.title = title
    signal.set_xydata(x, y, dx=dx, dy=dy)
    if units is not None:
        signal.xunit, signal.yunit = units
    if labels is not None:
        signal.xlabel, signal.ylabel = labels
    if metadata is not None:
        signal.metadata.update(metadata)
    return signal


class SignalTypes(gds.LabeledEnum):
    """Signal types"""

    #: Signal filled with zero
    ZERO = "zero", _("Zero")
    #: Random signal (normal distribution)
    NORMAL_DISTRIBUTION = "normal_distribution", _("Normal distribution")
    #: Random signal (Poisson distribution)
    POISSON_DISTRIBUTION = "poisson_distribution", _("Poisson distribution")
    #: Random signal (uniform distribution)
    UNIFORM_DISTRIBUTION = "uniform_distribution", _("Uniform distribution")
    #: Gaussian function
    GAUSS = "gauss", _("Gaussian")
    #: Lorentzian function
    LORENTZ = "lorentz", _("Lorentzian")
    #: Voigt function
    VOIGT = "voigt", _("Voigt")
    #: Planck function
    PLANCK = "planck", _("Blackbody (Planck)")
    #: Sinusoid
    SINE = "sine", _("Sine")
    #: Cosinusoid
    COSINE = "cosine", _("Cosine")
    #: Sawtooth function
    SAWTOOTH = "sawtooth", _("Sawtooth")
    #: Triangle function
    TRIANGLE = "triangle", _("Triangle")
    #: Square function
    SQUARE = "square", _("Square")
    #: Cardinal sine
    SINC = "sinc", _("Cardinal sine")
    #: Linear chirp
    LINEARCHIRP = "linearchirp", _("Linear chirp")
    #: Step function
    STEP = "step", _("Step")
    #: Exponential function
    EXPONENTIAL = "exponential", _("Exponential")
    #: Logistic function
    LOGISTIC = "logistic", _("Logistic")
    #: Pulse function
    PULSE = "pulse", _("Pulse")
    #: Step pulse function (with configurable rise time)
    STEP_PULSE = "step_pulse", _("Step pulse")
    #: Square pulse function (with configurable rise/fall times)
    SQUARE_PULSE = "square_pulse", _("Square pulse")
    #: Polynomial function
    POLYNOMIAL = "polynomial", _("Polynomial")
    #: Custom function
    CUSTOM = "custom", _("Custom")


DEFAULT_TITLE = _("Untitled signal")


class NewSignalParam(gds.DataSet):
    """New signal dataset.

    Subclasses can optionally implement a ``generate_title()`` method to provide
    automatic title generation based on their parameters. This method should return
    a string containing the generated title, or an empty string if no title can be
    generated.

    Example::

        def generate_title(self) -> str:
            '''Generate a title based on current parameters.'''
            return f"MySignal(param1={self.param1},param2={self.param2})"
    """

    title = gds.StringItem(_("Title"), default=DEFAULT_TITLE)
    size = gds.IntItem(
        _("N<sub>points</sub>"),
        help=_("Total number of points in the signal"),
        min=1,
        default=500,
    )
    xmin = gds.FloatItem("x<sub>min</sub>", default=-10.0)
    xmax = gds.FloatItem("x<sub>max</sub>", default=10.0).set_prop("display", col=1)
    xlabel = gds.StringItem(_("X label"), default="")
    xunit = gds.StringItem(_("X unit"), default="").set_prop("display", col=1)
    ylabel = gds.StringItem(_("Y label"), default="")
    yunit = gds.StringItem(_("Y unit"), default="").set_prop("display", col=1)

    # As it is the last item of the dataset, the separator will be hidden if no other
    # items are present after it (i.e. when derived classes do not add any new items
    # or when the NewSignalParam class is used alone).
    sep = gds.SeparatorItem()

    def generate_x_data(self) -> np.ndarray:
        """Generate x data based on current parameters."""
        return np.linspace(self.xmin, self.xmax, self.size)

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        return self.generate_x_data(), np.zeros(self.size)


SIGNAL_TYPE_PARAM_CLASSES = {}


def register_signal_parameters_class(stype: SignalTypes, param_class) -> None:
    """Register a parameters class for a given signal type.

    Args:
        stype: signal type
        param_class: parameters class
    """
    SIGNAL_TYPE_PARAM_CLASSES[stype] = param_class


def __get_signal_parameters_class(stype: SignalTypes) -> Type[NewSignalParam]:
    """Get parameters class for a given signal type.

    Args:
        stype: signal type

    Returns:
        Parameters class

    Raises:
        ValueError: if no parameters class is registered for the given signal type
    """
    try:
        return SIGNAL_TYPE_PARAM_CLASSES[stype]
    except KeyError as exc:
        raise ValueError(
            f"Image type {stype} has no parameters class registered"
        ) from exc


def check_all_signal_parameters_classes() -> None:
    """Check all registered parameters classes."""
    for stype, param_class in SIGNAL_TYPE_PARAM_CLASSES.items():
        assert __get_signal_parameters_class(stype) is param_class


def create_signal_parameters(
    stype: SignalTypes,
    title: str | None = None,
    size: int | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    **kwargs: dict,
) -> NewSignalParam:
    """Create parameters for a given signal type.

    Args:
        stype: signal type
        title: signal title
        size: signal size (number of points)
        xmin: minimum x value
        xmax: maximum x value
        xlabel: x axis label
        ylabel: y axis label
        xunit: x axis unit
        yunit: y axis unit
        **kwargs: additional parameters (specific to the signal type)

    Returns:
        Parameters object for the given signal type
    """
    pclass = __get_signal_parameters_class(stype)
    p = pclass.create(**kwargs)
    if title is not None:
        p.title = title
    if size is not None:
        p.size = size
    if xmin is not None:
        p.xmin = xmin
    if xmax is not None:
        p.xmax = xmax
    if xlabel is not None:
        p.xlabel = xlabel
    if ylabel is not None:
        p.ylabel = ylabel
    if xunit is not None:
        p.xunit = xunit
    if yunit is not None:
        p.yunit = yunit
    return p


class ZeroParam(NewSignalParam, title=_("Zero")):
    """Parameters for zero signal."""

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        x = self.generate_x_data()
        return x, np.zeros_like(x)


register_signal_parameters_class(SignalTypes.ZERO, ZeroParam)


class UniformDistribution1DParam(
    NewSignalParam, base.UniformDistributionParam, title=_("Uniform distribution")
):
    """Uniform-distribution signal parameters."""

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        x = self.generate_x_data()
        rng = np.random.default_rng(self.seed)
        assert self.vmin is not None
        assert self.vmax is not None
        y = self.vmin + rng.random(len(x)) * (self.vmax - self.vmin)
        return x, y


register_signal_parameters_class(
    SignalTypes.UNIFORM_DISTRIBUTION, UniformDistribution1DParam
)


class NormalDistribution1DParam(
    NewSignalParam, base.NormalDistributionParam, title=_("Normal distribution")
):
    """Normal-distribution signal parameters."""

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        x = self.generate_x_data()
        rng = np.random.default_rng(self.seed)
        assert self.mu is not None
        assert self.sigma is not None
        y = rng.normal(self.mu, self.sigma, len(x))
        return x, y


register_signal_parameters_class(
    SignalTypes.NORMAL_DISTRIBUTION, NormalDistribution1DParam
)


class PoissonDistribution1DParam(
    NewSignalParam, base.PoissonDistributionParam, title=_("Poisson distribution")
):
    """Poisson-distribution signal parameters."""

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        x = self.generate_x_data()
        rng = np.random.default_rng(self.seed)
        assert self.lam is not None
        y = rng.poisson(lam=self.lam, size=len(x))
        return x, y


register_signal_parameters_class(
    SignalTypes.POISSON_DISTRIBUTION, PoissonDistribution1DParam
)


class BaseGaussLorentzVoigtParam(NewSignalParam):
    """Base parameters for Gaussian, Lorentzian and Voigt functions"""

    STYPE: Type[SignalTypes] | None = None

    a = gds.FloatItem("A", default=1.0)
    y0 = gds.FloatItem("y<sub>0</sub>", default=0.0).set_pos(col=1)
    sigma = gds.FloatItem("σ", default=1.0)
    mu = gds.FloatItem("μ", default=0.0).set_pos(col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        assert isinstance(self.STYPE, SignalTypes)
        return (
            f"{self.STYPE.name.lower()}(A={self.a:.3g},σ={self.sigma:.3g},"
            f"μ={self.mu:.3g},y0={self.y0:.3g})"
        )

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        func = {
            SignalTypes.GAUSS: GaussianModel.func,
            SignalTypes.LORENTZ: LorentzianModel.func,
            SignalTypes.VOIGT: VoigtModel.func,
        }[self.STYPE]
        y = func(x, self.a, self.sigma, self.mu, self.y0)
        return x, y

    def get_expected_features(
        self, start_ratio: float = 0.1, stop_ratio: float = 0.9
    ) -> ExpectedFeatures:
        """Calculate expected pulse features for this signal.

        Args:
            start_ratio: Start ratio for rise time calculation
            stop_ratio: Stop ratio for rise time calculation

        Returns:
            ExpectedFeatures dataclass with all expected values
        """
        if self.a is None or self.sigma is None:
            raise ValueError("Parameters 'a' and 'sigma' must be set")
        if self.a == 0 or self.sigma <= 0:
            raise ValueError("Parameter 'a' must be non-zero and 'sigma' positive")

        polarity = 1 if self.a > 0 else -1

        # For Gaussian: peak amplitude is a / (sigma * sqrt(2*pi))
        # This gives the actual maximum value of the Gaussian function
        amplitude = abs(self.a) / (self.sigma * np.sqrt(2 * np.pi))

        if self.STYPE == SignalTypes.GAUSS:
            # Gaussian rise time: t_r = 2.563 * sigma (10% to 90%)
            rise_time = 2.563 * self.sigma
        elif self.STYPE == SignalTypes.LORENTZ:
            # Lorentzian rise time: 2*sigma*sqrt(1/start_ratio - 1/stop_ratio)
            rise_time = 2 * self.sigma * np.sqrt(1 / start_ratio - 1 / stop_ratio)
        elif self.STYPE == SignalTypes.VOIGT:
            # Voigt rise time: approximate as Gaussian for simplicity
            rise_time = 2.563 * self.sigma
        else:
            raise ValueError(f"Unsupported signal type: {self.STYPE}")

        # For Gaussian signals centered at mu
        x_center = self.mu if self.mu is not None else 0.0

        # Gaussian-specific calculations
        if self.STYPE == SignalTypes.GAUSS:
            # Time at 50% amplitude (FWHM calculation)
            fwhm = 2.355 * self.sigma  # Full Width at Half Maximum for Gaussian
            # x50 is the 50% crossing on the rise (left side of peak)
            x50 = x_center - self.sigma * np.sqrt(-2 * np.log(0.5))  # ~0.833σ

            # Rise time from left 20% to left 80% (one-sided)
            # For amplitude ratios: x = mu ± sigma * sqrt(-2 * ln(ratio))
            t_20_left = x_center - self.sigma * np.sqrt(-2 * np.log(0.2))  # ~1.794σ
            t_80_left = x_center - self.sigma * np.sqrt(-2 * np.log(0.8))  # ~0.668σ
            actual_rise_time = abs(t_80_left - t_20_left)

            # Fall time (symmetric for Gaussian)
            fall_time = actual_rise_time

            # Foot duration: For Gaussian, use approximation based on sigma
            # Since Gaussian has no true flat foot, this is an approximation
            foot_duration = 1.5 * self.sigma  # Empirically derived approximation

        else:
            # For Lorentzian and Voigt, use approximations
            x50 = x_center
            actual_rise_time = rise_time  # Use calculated rise_time
            fall_time = rise_time
            if self.STYPE == SignalTypes.LORENTZ:
                fwhm = 2 * self.sigma
            else:
                fwhm = 2.355 * self.sigma
            foot_duration = 2 * self.sigma  # Approximation

        return ExpectedFeatures(
            signal_shape=SignalShape.SQUARE,
            polarity=polarity,
            amplitude=amplitude,
            rise_time=actual_rise_time,
            offset=self.y0 if self.y0 is not None else 0.0,
            x50=x50,
            x100=x_center,  # Maximum is at center for Gaussian
            foot_duration=foot_duration,
            fall_time=fall_time,
            fwhm=fwhm,
        )

    def get_feature_tolerances(self) -> FeatureTolerances:
        """Get absolute tolerance values for pulse feature validation.

        Returns:
            FeatureTolerances dataclass with adjusted tolerances for Gaussian signals
        """
        # Gaussian signals may need slightly more relaxed tolerances due to smoothness
        return FeatureTolerances(
            rise_time=0.3,  # Slightly higher tolerance for Gaussian rise time
            fall_time=0.3,  # Match rise time tolerance
            x100=0.1,  # Tighter tolerance for maximum position (should be exact)
            fwhm=0.2,  # Reasonable tolerance for FWHM
        )

    def get_crossing_time(self, edge: Literal["rise", "fall"], ratio: float) -> float:
        """Get the theoretical crossing time for the specified edge and ratio.

        Args:
            edge: Which edge to calculate ("rise" or "fall")
            ratio: Crossing ratio (0.0 to 1.0)

        Returns:
            Theoretical crossing time for the specified edge and ratio
        """
        if self.a is None or self.sigma is None or self.mu is None:
            raise ValueError("Parameters 'a', 'sigma', and 'mu' must be set")
        if self.a == 0 or self.sigma <= 0:
            raise ValueError("Parameter 'a' must be non-zero and 'sigma' positive")
        if not 0.0 < ratio < 1.0:
            raise ValueError("Ratio must be between 0.0 and 1.0")

        if self.STYPE != SignalTypes.GAUSS:
            raise NotImplementedError(
                "Crossing time calculation is only implemented for Gaussian signals"
            )

        # For Gaussian: x = mu ± sigma * sqrt(-2 * ln(ratio))
        delta_x = self.sigma * np.sqrt(-2 * np.log(ratio))
        if edge == "rise":
            return self.mu - delta_x
        if edge == "fall":
            return self.mu + delta_x
        raise ValueError("Edge must be 'rise' or 'fall'")


class GaussParam(
    BaseGaussLorentzVoigtParam,
    title=_("Gaussian"),
    comment="y = y<sub>0</sub> + "
    "A/(σ √(2π)) exp(-((x - μ)<sup>2</sup>) / (2 σ<sup>2</sup>))",
):
    """Parameters for Gaussian function."""

    STYPE = SignalTypes.GAUSS


register_signal_parameters_class(SignalTypes.GAUSS, GaussParam)


class LorentzParam(
    BaseGaussLorentzVoigtParam,
    title=_("Lorentzian"),
    comment="y = y<sub>0</sub> + A/(π σ (1 + ((x - μ)/σ)<sup>2</sup>))",
):
    """Parameters for Lorentzian function."""

    STYPE = SignalTypes.LORENTZ


register_signal_parameters_class(SignalTypes.LORENTZ, LorentzParam)


class VoigtParam(
    BaseGaussLorentzVoigtParam,
    title=_("Voigt"),
    comment="y = y<sub>0</sub> + "
    "A Re[exp(-z<sup>2</sup>) erfc(-j z)] / (σ √(2π)), "
    "with z = (x - μ - j σ) / (σ √2)",
):
    """Parameters for Voigt function."""

    STYPE = SignalTypes.VOIGT


register_signal_parameters_class(SignalTypes.VOIGT, VoigtParam)


class PlanckParam(
    NewSignalParam,
    title=_("Blackbody (Planck)"),
    comment="y = (2 h c<sup>2</sup>) / "
    "(λ<sup>5</sup> (exp(h c / (λ k<sub>B</sub> T)) - 1))",
):
    """Planck radiation law."""

    xmin = gds.FloatItem(
        "λ<sub>min</sub>", default=1e-7, unit="m", min=0.0, nonzero=True
    )
    xmax = gds.FloatItem(
        "λ<sub>max</sub>", default=1e-4, unit="m", min=0.0, nonzero=True
    ).set_prop("display", col=1)
    T = gds.FloatItem(
        "T", default=293.0, unit="K", min=0.0, nonzero=True, help=_("Temperature")
    )

    def generate_title(self) -> str:
        """Generate a title based on current parameters.

        Returns:
            Title string.
        """
        return f"planck(T={self.T:.3g}K)"

    @classmethod
    def func(cls, wavelength: np.ndarray, temperature: float) -> np.ndarray:
        """Compute the Planck function.

        Args:
            wavelength: Wavelength (m).
            T: Temperature (K).

        Returns:
            Spectral radiance (W m<sup>-2</sup> sr<sup>-1</sup> Hz<sup>-1</sup>).
        """
        h = scipy.constants.h  # Planck constant (J·s)
        c = scipy.constants.c  # Speed of light (m/s)
        k = scipy.constants.k  # Boltzmann constant (J/K)
        c1 = 2 * h * c**2
        c2 = (h * c) / k
        denom = np.exp(c2 / (wavelength * temperature)) - 1.0
        spectral_radiance = c1 / (wavelength**5 * (denom))
        return spectral_radiance

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (wavelength, spectral radiance) arrays.
        """
        wavelength = self.generate_x_data()
        assert self.T is not None
        y = self.func(wavelength, self.T)
        return wavelength, y


register_signal_parameters_class(SignalTypes.PLANCK, PlanckParam)


class FreqUnits(enum.Enum):
    """Frequency units"""

    HZ = "Hz"
    KHZ = "kHz"
    MHZ = "MHz"
    GHZ = "GHz"

    @classmethod
    def convert_in_hz(cls, value, unit):
        """Convert value in Hz"""
        factor = {cls.HZ: 1, cls.KHZ: 1e3, cls.MHZ: 1e6, cls.GHZ: 1e9}.get(unit)
        if factor is None:
            raise ValueError(f"Unknown unit: {unit}")
        return value * factor


class BasePeriodicParam(NewSignalParam):
    """Parameters for periodic functions"""

    STYPE: Type[SignalTypes] | None = None

    def get_frequency_in_hz(self):
        """Return frequency in Hz"""
        return FreqUnits.convert_in_hz(self.freq, self.freq_unit)

    # Redefining some parameters with more appropriate defaults
    xunit = gds.StringItem(_("X unit"), default="s")

    a = gds.FloatItem("A", default=1.0)
    offset = gds.FloatItem("y<sub>0</sub>", default=0.0).set_pos(col=1)
    freq = gds.FloatItem("f", default=1.0)
    freq_unit = gds.ChoiceItem(_("Unit"), FreqUnits, default=FreqUnits.HZ).set_pos(
        col=1
    )
    phase = gds.FloatItem("φ", default=0.0, unit="°")

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        assert isinstance(self.STYPE, SignalTypes)
        freq_hz = self.get_frequency_in_hz()
        title = (
            f"{self.STYPE.name.lower()}(f={freq_hz:.3g}Hz,"
            f"A={self.a:.3g},y0={self.offset:.3g},φ={self.phase:.3g}°)"
        )
        return title

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        func = {
            SignalTypes.SINE: np.sin,
            SignalTypes.COSINE: np.cos,
            SignalTypes.SAWTOOTH: sps.sawtooth,
            SignalTypes.TRIANGLE: triangle_func,
            SignalTypes.SQUARE: sps.square,
            SignalTypes.SINC: np.sinc,
        }[self.STYPE]
        freq = self.get_frequency_in_hz()
        y = self.a * func(2 * np.pi * freq * x + np.deg2rad(self.phase)) + self.offset
        return x, y


class SineParam(
    BasePeriodicParam, title=_("Sine"), comment="y = y<sub>0</sub> + A sin(2π f x + φ)"
):
    """Parameters for sine function."""

    STYPE = SignalTypes.SINE


register_signal_parameters_class(SignalTypes.SINE, SineParam)


class CosineParam(
    BasePeriodicParam,
    title=_("Cosine"),
    comment="y = y<sub>0</sub> + A cos(2π f x + φ)",
):
    """Parameters for cosine function."""

    STYPE = SignalTypes.COSINE


register_signal_parameters_class(SignalTypes.COSINE, CosineParam)


class SawtoothParam(
    BasePeriodicParam,
    title=_("Sawtooth"),
    comment="y = y<sub>0</sub> + A (2 (f x + φ/(2π) - |f x + φ/(2π) + 1/2|))",
):
    """Parameters for sawtooth function."""

    STYPE = SignalTypes.SAWTOOTH


register_signal_parameters_class(SignalTypes.SAWTOOTH, SawtoothParam)


class TriangleParam(
    BasePeriodicParam,
    title=_("Triangle"),
    comment="y = y<sub>0</sub> + A sawtooth(2π f x + φ, width=0.5)",
):
    """Parameters for triangle function."""

    STYPE = SignalTypes.TRIANGLE


register_signal_parameters_class(SignalTypes.TRIANGLE, TriangleParam)


class SquareParam(
    BasePeriodicParam,
    title=_("Square"),
    comment="y = y<sub>0</sub> + A sgn(sin(2π f x + φ))",
):
    """Parameters for square function."""

    STYPE = SignalTypes.SQUARE


register_signal_parameters_class(SignalTypes.SQUARE, SquareParam)


class SincParam(
    BasePeriodicParam,
    title=_("Cardinal sine"),
    comment="y = y<sub>0</sub> + A sinc(f x + φ)",
):
    """Parameters for cardinal sine function."""

    STYPE = SignalTypes.SINC


register_signal_parameters_class(SignalTypes.SINC, SincParam)


class LinearChirpParam(
    NewSignalParam,
    title=_("Linear chirp"),
    comment="y = y<sub>0</sub> + a sin(φ<sub>0</sub> "
    "+ 2π (f<sub>0</sub> x + 0.5 k x²))",
):
    """Linear chirp function."""

    a = gds.FloatItem("A", default=1.0, help=_("Amplitude"))
    phi0 = gds.FloatItem(
        "φ<sub>0</sub>", default=0.0, help=_("Initial phase")
    ).set_prop("display", col=1)
    k = gds.FloatItem("k", default=1.0, help=_("Chirp rate (f<sup>-2</sup>)"))
    offset = gds.FloatItem(
        "y<sub>0</sub>", default=0.0, help=_("Vertical offset")
    ).set_prop("display", col=1)
    f0 = gds.FloatItem("f<sub>0</sub>", default=1.0, help=_("Initial frequency (Hz)"))

    def generate_title(self) -> str:
        """Generate a title based on current parameters.

        Returns:
            Title string.
        """
        return (
            f"chirp(A={self.a:.3g},"
            f"k={self.k:.3g},"
            f"f0={self.f0:.3g},"
            f"φ0={self.phi0:.3g},"
            f"y0={self.offset:.3g})"
        )

    @classmethod
    def func(
        cls, x: np.ndarray, a: float, k: float, f0: float, phi0: float, offset: float
    ) -> np.ndarray:
        """Compute the linear chirp function.

        Args:
            x: X data array.
            a: Amplitude.
            k: Chirp rate (s<sup>-2</sup>).
            f0: Initial frequency (Hz).
            phi0: Initial phase.
            offset: Vertical offset.

        Returns:
            Y data array computed using the chirp function.
        """
        phase = phi0 + 2 * np.pi * (f0 * x + 0.5 * k * x**2)
        return offset + a * np.sin(phase)

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        assert self.a is not None
        assert self.k is not None
        assert self.f0 is not None
        assert self.phi0 is not None
        assert self.offset is not None
        x = self.generate_x_data()
        y = self.func(x, self.a, self.k, self.f0, self.phi0, self.offset)
        return x, y


register_signal_parameters_class(SignalTypes.LINEARCHIRP, LinearChirpParam)


class StepParam(NewSignalParam, title=_("Step")):
    """Parameters for step function."""

    a1 = gds.FloatItem("A<sub>1</sub>", default=0.0)
    a2 = gds.FloatItem("A<sub>2</sub>", default=1.0).set_pos(col=1)
    x0 = gds.FloatItem("x<sub>0</sub>", default=0.0)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"step(a1={self.a1:.3g},a2={self.a2:.3g},x0={self.x0:.3g})"

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        y = np.ones_like(x) * self.a1
        y[x > self.x0] = self.a2
        return x, y


register_signal_parameters_class(SignalTypes.STEP, StepParam)


class ExponentialParam(
    NewSignalParam, title=_("Exponential"), comment="y = A exp(B x) + y<sub>0</sub>"
):
    """Parameters for exponential function."""

    a = gds.FloatItem("A", default=1.0)
    offset = gds.FloatItem("y<sub>0</sub>", default=0.0)
    exponent = gds.FloatItem("B", default=1.0)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"exponential(A={self.a:.3g},B={self.exponent:.3g},y0={self.offset:.3g})"

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        y = self.a * np.exp(self.exponent * x) + self.offset
        return x, y


register_signal_parameters_class(SignalTypes.EXPONENTIAL, ExponentialParam)


class LogisticParam(
    NewSignalParam,
    title=_("Logistic"),
    comment="y = y<sub>0</sub> + A / (1 + exp(-k (x - x<sub>0</sub>)))",
):
    """Logistic function."""

    a = gds.FloatItem("A", default=1.0, help=_("Amplitude"))
    x0 = gds.FloatItem(
        "x<sub>0</sub>", default=0.0, help=_("Horizontal offset")
    ).set_prop("display", col=1)
    k = gds.FloatItem("k", default=1.0, help=_("Growth or decay rate"))
    offset = gds.FloatItem(
        "y<sub>0</sub>", default=0.0, help=_("Vertical offset")
    ).set_prop("display", col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters.

        Returns:
            Title string.
        """
        return (
            f"logistic(A={self.a:.3g},"
            f"k={self.k:.3g},"
            f"x0={self.x0:.3g},"
            f"y0={self.offset:.3g})"
        )

    @classmethod
    def func(
        cls, x: np.ndarray, a: float, k: float, x0: float, offset: float
    ) -> np.ndarray:
        """Compute the logistic function.

        Args:
            x: X data array.
            a: Amplitude.
            k: Growth or decay rate.
            x0: Horizontal offset.
            offset: Vertical offset.

        Returns:
            Y data array computed using the logistic function.
        """
        return offset + a / (1.0 + np.exp(-k * (x - x0)))

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays.
        """
        assert self.a is not None
        assert self.k is not None
        assert self.x0 is not None
        assert self.offset is not None
        x = self.generate_x_data()
        y = self.func(x, self.a, self.k, self.x0, self.offset)
        return x, y


register_signal_parameters_class(SignalTypes.LOGISTIC, LogisticParam)


class PulseParam(NewSignalParam, title=_("Pulse")):
    """Parameters for pulse function."""

    amp = gds.FloatItem("Amplitude", default=1.0)
    start = gds.FloatItem(_("Start"), default=0.0).set_pos(col=1)
    offset = gds.FloatItem(_("Offset"), default=10.0)
    stop = gds.FloatItem(_("End"), default=5.0).set_pos(col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return (
            f"pulse(start={self.start:.3g},stop={self.stop:.3g},"
            f"offset={self.offset:.3g},amp={self.amp:.3g})"
        )

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        y = np.full_like(x, self.offset)
        y[(x >= self.start) & (x <= self.stop)] += self.amp
        return x, y


register_signal_parameters_class(SignalTypes.PULSE, PulseParam)


@dataclass
class ExpectedFeatures:
    """Expected pulse feature values for validation."""

    signal_shape: SignalShape
    polarity: int
    amplitude: float
    rise_time: float  # Rise time between specified ratios
    offset: float
    x50: float
    x100: float  # Time at 100% amplitude (maximum)
    foot_duration: float
    fall_time: float | None = None  # Fall time between specified ratios
    fwhm: float | None = None


@dataclass
class FeatureTolerances:
    """Absolute tolerance values for pulse feature validation."""

    polarity: float = 1e-8
    amplitude: float = 0.5
    rise_time: float = 0.2
    offset: float = 0.5
    x50: float = 0.1
    x100: float = 0.6  # Tolerance for time at 100% amplitude
    foot_duration: float = 0.5
    fall_time: float = 1.0
    fwhm: float = 0.5


class BasePulseParam(NewSignalParam):
    """Base class for pulse signal parameters."""

    SEED = 0

    # Redefine NewSignalParam parameters with more appropriate defaults
    xmin = gds.FloatItem(_("Start time"), default=0.0)
    xmax = gds.FloatItem(_("End time"), default=10.0)
    size = gds.IntItem(_("Number of points"), default=1000, min=1)

    # Specific pulse parameters
    offset = gds.FloatItem(_("Initial value"), default=0.0)
    amplitude = gds.FloatItem(_("Amplitude"), default=5.0).set_pos(col=1)
    noise_amplitude = gds.FloatItem(_("Noise amplitude"), default=0.2, min=0.0)
    x_rise_start = gds.FloatItem(_("Rise start time"), default=3.0, min=0.0)
    total_rise_time = gds.FloatItem(_("Total rise time"), default=2.0, min=0.0).set_pos(
        col=1
    )

    def get_crossing_time(self, edge: Literal["rise", "fall"], ratio: float) -> float:
        """Get the theoretical crossing time for the specified edge and ratio.

        Args:
            edge: Which edge to calculate ("rise" or "fall")
            ratio: Crossing ratio (0.0 to 1.0)

        Returns:
            Theoretical crossing time for the specified edge and ratio
        """
        if edge == "rise":
            return self.x_rise_start + ratio * self.total_rise_time
        raise NotImplementedError(
            "Fall edge crossing time not implemented for this signal type"
        )

    def get_expected_features(
        self, start_ratio: float = 0.1, stop_ratio: float = 0.9
    ) -> ExpectedFeatures:
        """Calculate expected pulse features for this signal.

        Args:
            start_ratio: Start ratio for rise time calculation
            stop_ratio: Stop ratio for rise time calculation

        Returns:
            ExpectedFeatures dataclass with all expected values
        """
        y_end_value = self.offset + self.amplitude
        return ExpectedFeatures(
            signal_shape=SignalShape.STEP,
            polarity=1 if y_end_value > self.offset else -1,
            amplitude=abs(y_end_value - self.offset),
            rise_time=(stop_ratio - start_ratio) * self.total_rise_time,
            offset=self.offset,
            x50=self.x_rise_start + 0.5 * self.total_rise_time,
            x100=self.x_rise_start + self.total_rise_time,
            foot_duration=self.x_rise_start - self.xmin,
        )

    def get_feature_tolerances(self) -> FeatureTolerances:
        """Get absolute tolerance values for pulse feature validation.

        Returns:
            FeatureTolerances dataclass with default tolerance values
        """
        return FeatureTolerances()


class StepPulseParam(BasePulseParam, title=_("Step pulse with noise")):
    """Parameters for generating step signals with configurable rise time."""

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return (
            f"step_pulse(rise_time={self.total_rise_time:.3g},"
            f"x_start={self.x_rise_start:.3g},offset={self.offset:.3g},"
            f"amp={self.amplitude:.3g})"
        )

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a noisy step signal with a linear rise.

        The function creates a time vector and generates a signal that starts at
        `offset`, rises linearly to `offset + amplitude` starting at `x_rise_start` over
        a duration of `total_rise_time`, and remains at the final value afterwards.
        Gaussian noise is added to the signal.

        Returns:
            Tuple containing the time vector and noisy step signal.
        """
        # time vector
        x = self.generate_x_data()

        # Calculate final value from offset and amplitude
        y_final = self.offset + self.amplitude

        # creating the signal
        rise_end_time = self.x_rise_start + self.total_rise_time
        y = np.piecewise(
            x,
            [
                x < self.x_rise_start,
                (x >= self.x_rise_start) & (x < rise_end_time),
                x >= rise_end_time,
            ],
            [
                self.offset,
                lambda t: (
                    self.offset
                    + (y_final - self.offset)
                    * (t - self.x_rise_start)
                    / self.total_rise_time
                ),
                y_final,
            ],
        )
        rdg = np.random.default_rng(self.SEED)
        noise = rdg.normal(0, self.noise_amplitude, size=len(y))
        y_noisy = y + noise

        return x, y_noisy


register_signal_parameters_class(SignalTypes.STEP_PULSE, StepPulseParam)


class SquarePulseParam(BasePulseParam, title=_("Square pulse with noise")):
    """Parameters for generating square signals with configurable rise/fall times."""

    # Redefine NewSignalParam parameters with more appropriate defaults
    xmax = gds.FloatItem(_("End time"), default=20.0)

    # Specific square pulse parameters
    fwhm = gds.FloatItem(_("Full Width at Half Maximum"), default=5.5, min=0.0)
    total_fall_time = gds.FloatItem(_("Total fall time"), default=5.0, min=0.0).set_pos(
        col=1
    )

    @property
    def square_duration(self) -> float:
        """Calculate the square duration from FWHM and total rise/fall times."""
        return self.fwhm - 0.5 * self.total_rise_time - 0.5 * self.total_fall_time

    def get_plateau_range(self) -> tuple[float, float]:
        """Get the theoretical plateau range (start, end) for the square signal.

        Returns:
            Tuple with (start, end) times of the plateau
        """
        return (
            self.x_rise_start + self.total_rise_time,
            self.x_rise_start + self.total_rise_time + self.square_duration,
        )

    def get_crossing_time(self, edge: Literal["rise", "fall"], ratio: float) -> float:
        """Get the theoretical crossing time for the specified edge and ratio.

        Args:
            edge: Which edge to calculate ("rise" or "fall")
            ratio: Crossing ratio (0.0 to 1.0)

        Returns:
            Theoretical crossing time for the specified edge and ratio
        """
        if edge == "rise":
            return super().get_crossing_time(edge, ratio)
        if edge == "fall":
            t_start_fall = (
                self.x_rise_start + self.total_rise_time + self.square_duration
            )
            return t_start_fall + ratio * self.total_fall_time
        raise ValueError("edge must be 'rise' or 'fall'")

    def get_expected_features(
        self, start_ratio: float = 0.1, stop_ratio: float = 0.9
    ) -> ExpectedFeatures:
        """Calculate expected pulse features for this signal.

        Args:
            start_ratio: Start ratio for rise time calculation
            stop_ratio: Stop ratio for rise time calculation

        Returns:
            ExpectedFeatures dataclass with all expected values
        """
        features = super().get_expected_features(start_ratio, stop_ratio)
        features.signal_shape = SignalShape.SQUARE
        features.fall_time = np.abs(stop_ratio - start_ratio) * self.total_fall_time
        features.fwhm = self.fwhm
        return features

    def get_feature_tolerances(self) -> FeatureTolerances:
        """Get absolute tolerance values for square signal feature validation.

        Returns:
            FeatureTolerances dataclass with square-specific tolerance values
        """
        return FeatureTolerances(
            x100=0.8,  # Looser tolerance for square signals
        )

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return (
            f"square_pulse(rise_time={self.total_rise_time:.3g},"
            f"fall_time={self.total_fall_time:.3g},"
            f"fwhm={self.fwhm:.3g},offset={self.offset:.3g},"
            f"amp={self.amplitude:.3g})"
        )

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic square-like signal with configurable parameters.

        Generates a synthetic square-like signal with configurable rise, plateau,
        and fall times, and adds Gaussian noise.

        Returns:
            Tuple containing the time vector and noisy square signal.
        """
        # time vector
        x = self.generate_x_data()

        # Calculate high value from offset and amplitude
        y_high = self.offset + self.amplitude

        x_rise_end = self.x_rise_start + self.total_rise_time
        x_start_fall = self.x_rise_start + self.total_rise_time + self.square_duration
        # creating the signal
        y = np.piecewise(
            x,
            [
                x < self.x_rise_start,
                (x >= self.x_rise_start) & (x < x_rise_end),
                (x >= x_rise_end) & (x < x_start_fall),
                (x >= x_start_fall) & (x < x_start_fall + self.total_fall_time),
                x >= self.total_fall_time + x_start_fall,
            ],
            [
                self.offset,
                lambda t: (
                    self.offset
                    + (y_high - self.offset)
                    * (t - self.x_rise_start)
                    / self.total_rise_time
                ),
                y_high,
                lambda t: y_high
                - (y_high - self.offset) * (t - x_start_fall) / self.total_fall_time,
                self.offset,
            ],
        )
        rdg = np.random.default_rng(self.SEED)
        noise = rdg.normal(0, self.noise_amplitude, size=len(y))
        y_noisy = y + noise

        return x, y_noisy


register_signal_parameters_class(SignalTypes.SQUARE_PULSE, SquarePulseParam)


class PolyParam(NewSignalParam, title=_("Polynomial")):
    """Parameters for polynomial function."""

    a0 = gds.FloatItem("a0", default=1.0)
    a3 = gds.FloatItem("a3", default=0.0).set_pos(col=1)
    a1 = gds.FloatItem("a1", default=1.0)
    a4 = gds.FloatItem("a4", default=0.0).set_pos(col=1)
    a2 = gds.FloatItem("a2", default=0.0)
    a5 = gds.FloatItem("a5", default=0.0).set_pos(col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        coeffs = [self.a0, self.a1, self.a2, self.a3, self.a4, self.a5]
        terms = []
        for i, coeff in enumerate(coeffs):
            if coeff == 0:
                continue
            # Format coefficient
            if i == 0:
                # Constant term
                terms.append(f"{coeff:.3g}")
            elif i == 1:
                # Linear term
                if coeff == 1:
                    terms.append("x")
                elif coeff == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff:.3g}x")
            else:
                # Higher order terms
                if coeff == 1:
                    terms.append(f"x^{i}")
                elif coeff == -1:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{coeff:.3g}x^{i}")

        if not terms:
            return "0"

        # Join terms with + or - signs
        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += term
            else:
                result += f"+{term}"

        return result

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        x = self.generate_x_data()
        y = np.polyval([self.a5, self.a4, self.a3, self.a2, self.a1, self.a0], x)
        return x, y


register_signal_parameters_class(SignalTypes.POLYNOMIAL, PolyParam)


class CustomSignalParam(NewSignalParam, title=_("Custom signal")):
    """Parameters for custom signal (e.g. manually defined experimental data)."""

    size = gds.IntItem(_("N<sub>points</sub>"), default=10).set_prop(
        "display", active=False
    )
    xmin = gds.FloatItem("x<sub>min</sub>", default=0.0).set_prop(
        "display", active=False
    )
    xmax = gds.FloatItem("x<sub>max</sub>", default=1.0).set_prop(
        "display", active=False, col=1
    )

    xyarray = gds.FloatArrayItem(
        "XY Values",
        format="%g",
    )

    def setup_array(
        self,
        size: int | None = None,
        xmin: float | None = None,
        xmax: float | None = None,
    ) -> None:
        """Setup the xyarray from size, xmin and xmax (use the current values is not
        provided)

        Args:
            size: xyarray size (default: None)
            xmin: X min (default: None)
            xmax: X max (default: None)
        """
        self.size = size or self.size
        self.xmin = xmin or self.xmin
        self.xmax = xmax or self.xmax
        x_arr = np.linspace(self.xmin, self.xmax, self.size)  # type: ignore
        self.xyarray = np.vstack((x_arr, x_arr)).T

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"custom(size={self.size})"

    def generate_1d_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D data based on current parameters.

        Returns:
            Tuple of (x, y) arrays
        """
        self.setup_array(size=self.size, xmin=self.xmin, xmax=self.xmax)
        x, y = self.xyarray.T
        return x, y


register_signal_parameters_class(SignalTypes.CUSTOM, CustomSignalParam)


check_all_signal_parameters_classes()


def triangle_func(xarr: np.ndarray) -> np.ndarray:
    """Triangle function

    Args:
        xarr: x data
    """
    # ignore warning, as type hint is not handled properly in upstream library
    return sps.sawtooth(xarr, width=0.5)  # type: ignore[no-untyped-def]


SIG_NB = 0


def get_next_signal_number() -> int:
    """Get the next signal number.

    This function is used to keep track of the number of signals created.
    It is typically used to generate unique titles for new signals.

    Returns:
        int: new signal number
    """
    global SIG_NB  # pylint: disable=global-statement
    SIG_NB += 1
    return SIG_NB


def create_signal_from_param(param: NewSignalParam) -> SignalObj:
    """Create a new Signal object from parameters.

    Args:
        param: new signal parameters

    Returns:
        Signal object

    Raises:
        NotImplementedError: if the signal type is not supported
    """
    # Generate data first, as some `generate_title()` methods may depend on it:
    x, y = param.generate_1d_data()
    # Check if user has customized the title or left it as default/empty
    use_generated_title = not param.title or param.title == DEFAULT_TITLE
    if use_generated_title:
        # Try to generate a descriptive title
        gen_title = getattr(param, "generate_title", lambda: "")()
        if gen_title:
            title = gen_title
        else:
            # No generated title available, use default with number
            title = f"{DEFAULT_TITLE} {get_next_signal_number():d}"
    else:
        # User has set a custom title, use it as-is
        title = param.title
    signal = create_signal(
        title,
        x,
        y,
        units=(param.xunit, param.yunit),
        labels=(param.xlabel, param.ylabel),
    )
    return signal
