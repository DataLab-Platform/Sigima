# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Pulse analysis (see parent package :mod:`sigima.tools.signal`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.ndimage
import scipy.optimize  # type: ignore
import scipy.special

from sigima.enums import SignalShape
from sigima.tools.checks import check_1d_arrays
from sigima.tools.signal import features, filtering, peakdetection

PEAK_PARAMETERIZATION = "height"


class LegacyPeakParameterizationError(ValueError):
    """Raised when the historical area-based peak API is used."""


class PulseFitModel(abc.ABC):
    """Base class for 1D pulse fit models"""

    @classmethod
    @abc.abstractmethod
    def evaluate(cls, x, amplitude, sigma, x0, y0):
        """Evaluate the peak model parameterized by its signed height."""

    @classmethod
    @abc.abstractmethod
    def area_from_amplitude(cls, amplitude, sigma):
        """Return the integrated area corresponding to a peak height."""

    @classmethod
    @abc.abstractmethod
    def amplitude_from_area(cls, area, sigma):
        """Return the peak height corresponding to an integrated area."""

    @classmethod
    @abc.abstractmethod
    def fwhm(cls, sigma):
        """Return function FWHM"""

    @staticmethod
    def _validate_relative_level(sigma: float, ratio: float) -> None:
        """Validate inputs used to locate a relative peak level."""
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        if not 0.0 < ratio < 1.0:
            raise ValueError("Ratio must be between 0 and 1")

    @classmethod
    @abc.abstractmethod
    def relative_level_offset(cls, sigma: float, ratio: float) -> float:
        """Return the positive center offset at a relative peak level."""

    @classmethod
    def half_max_segment(cls, amplitude, sigma, x0, y0):
        """Return segment coordinates for y=half-maximum intersection"""
        hwhm = 0.5 * cls.fwhm(sigma)
        yhm = 0.5 * amplitude + y0
        return x0 - hwhm, yhm, x0 + hwhm, yhm

    @classmethod
    def func(cls, *args, **kwargs):
        """Reject the historical area-based evaluation API."""
        raise LegacyPeakParameterizationError(
            "PeakModel.func() used the ambiguous area-based 'amp' parameter. "
            "Use evaluate(..., amplitude, ...) with a peak height, or convert "
            "the old value with amplitude_from_area(area, sigma)."
        )

    @classmethod
    def get_amp_from_amplitude(cls, *args, **kwargs):
        """Reject the historical ambiguous area conversion helper."""
        raise LegacyPeakParameterizationError(
            "get_amp_from_amplitude() returned an integrated area. "
            "Use area_from_amplitude(amplitude, sigma)."
        )

    @classmethod
    def amplitude(cls, *args, **kwargs):
        """Reject the historical ambiguous height conversion helper."""
        raise LegacyPeakParameterizationError(
            "amplitude(amp, sigma) interpreted 'amp' as an integrated area. "
            "Use amplitude_from_area(area, sigma)."
        )


class GaussianModel(PulseFitModel):
    """1-dimensional Gaussian fit model"""

    @classmethod
    def evaluate(cls, x, amplitude, sigma, x0, y0):
        """Evaluate a Gaussian with signed peak height ``amplitude``."""
        return amplitude * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + y0

    @classmethod
    def area_from_amplitude(cls, amplitude, sigma):
        """Return the Gaussian area for a peak height and standard deviation."""
        return amplitude * (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def amplitude_from_area(cls, area, sigma):
        """Return the Gaussian peak height for an integrated area."""
        return area / (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def fwhm(cls, sigma):
        """Return function FWHM"""
        return 2 * sigma * np.sqrt(2 * np.log(2))

    @classmethod
    def relative_level_offset(cls, sigma: float, ratio: float) -> float:
        """Return the positive center offset at a relative peak level."""
        cls._validate_relative_level(sigma, ratio)
        return sigma * np.sqrt(-2 * np.log(ratio))


class LorentzianModel(PulseFitModel):
    """1-dimensional Lorentzian fit model"""

    @classmethod
    def evaluate(cls, x, amplitude, sigma, x0, y0):
        """Evaluate a Lorentzian with signed peak height ``amplitude``."""
        return amplitude / (1 + ((x - x0) / sigma) ** 2) + y0

    @classmethod
    def area_from_amplitude(cls, amplitude, sigma):
        """Return the Lorentzian area for a peak height and scale."""
        return amplitude * (sigma * np.pi)

    @classmethod
    def amplitude_from_area(cls, area, sigma):
        """Return the Lorentzian peak height for an integrated area."""
        return area / (sigma * np.pi)

    @classmethod
    def fwhm(cls, sigma):
        """Return function FWHM"""
        return 2 * sigma

    @classmethod
    def relative_level_offset(cls, sigma: float, ratio: float) -> float:
        """Return the positive center offset at a relative peak level."""
        cls._validate_relative_level(sigma, ratio)
        return sigma * np.sqrt(1.0 / ratio - 1.0)


class VoigtModel(PulseFitModel):
    """1-dimensional Voigt fit model"""

    @classmethod
    def evaluate(cls, x, amplitude, sigma, x0, y0):
        """Evaluate an equal-width Voigt with signed peak height ``amplitude``."""
        # pylint: disable=no-member
        z = (x - x0 + 1j * sigma) / (sigma * np.sqrt(2.0))
        center = scipy.special.wofz(1j / np.sqrt(2.0)).real
        return y0 + amplitude * scipy.special.wofz(z).real / center

    @classmethod
    def area_from_amplitude(cls, amplitude, sigma):
        """Return the equal-width Voigt area for a peak height and scale."""
        center = scipy.special.wofz(1j / np.sqrt(2.0)).real
        return amplitude * sigma * np.sqrt(2 * np.pi) / center

    @classmethod
    def amplitude_from_area(cls, area, sigma):
        """Return the equal-width Voigt peak height for an integrated area."""
        center = scipy.special.wofz(1j / np.sqrt(2.0)).real
        return area * center / (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def fwhm(cls, sigma):
        """Return the exact full width at half maximum."""
        return 2.0 * cls.relative_level_offset(sigma, 0.5)

    @classmethod
    def relative_level_offset(cls, sigma: float, ratio: float) -> float:
        """Return the positive center offset at a relative peak level."""
        cls._validate_relative_level(sigma, ratio)

        def difference(scaled_offset: float) -> float:
            value = cls.evaluate(np.array([scaled_offset]), 1.0, 1.0, 0.0, 0.0)[0]
            return float(value) - ratio

        upper = 1.0
        for _index in range(64):
            if difference(upper) <= 0.0:
                root = scipy.optimize.brentq(difference, 0.0, upper)
                return sigma * root
            upper *= 2.0
        raise RuntimeError("Unable to bracket the requested Voigt relative level")


# MARK: Pulse analysis -----------------------------------------------------------------


class PulseAnalysisError(Exception):
    """Base exception for pulse analysis errors."""


class InvalidSignalError(PulseAnalysisError):
    """Raised when signal data is invalid or insufficient."""


class PolarityDetectionError(PulseAnalysisError):
    """Raised when polarity cannot be determined."""


def heuristically_recognize_shape(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    step_threshold_ratio: float = 0.5,
) -> SignalShape:
    """
    Heuristically determine the shape of the signal: 'step' or 'square'.

    Args:
        x: 1D array of x values (e.g., time or position).
        y: 1D array of y values corresponding to x.
        start_range: Range for the start baseline.
        end_range: Range for the end baseline.
        step_threshold_ratio: Threshold ratio to distinguish step from square pulse.
         If step amplitude > threshold_ratio * total_amplitude, classify as step.

    Returns:
        Signal shape, either SignalShape.STEP or SignalShape.SQUARE.

    Raises:
        InvalidSignalError: If signal data is invalid.
    """
    if x.size != y.size:
        raise InvalidSignalError("x and y arrays must have the same length")
    if x.size < 3:
        raise InvalidSignalError("Signal must have at least 3 data points")

    # if ranges are None, use the first and last points
    if start_range is None:
        start_range = (x[0], x[0])
    if end_range is None:
        end_range = (x[-1], x[-1])

    step_amplitude = get_amplitude(
        x, y, start_range, end_range, signal_shape=SignalShape.STEP
    )
    total_amplitude = np.max(y) - np.min(y)

    if total_amplitude == 0:
        raise InvalidSignalError("Signal has zero amplitude")

    if np.abs(step_amplitude) > np.abs(step_threshold_ratio * total_amplitude):
        signal_shape = SignalShape.STEP
    else:
        signal_shape = SignalShape.SQUARE

    return signal_shape


def _detect_square_polarity(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    plateau_range: tuple[float, float] | None = None,
    y_start: float | None = None,
    y_end: float | None = None,
) -> int:
    """Detect the polarity of a square pulse in a signal based on baseline regions.

    Args:
        x: The array of x-values (typically time or sample indices).
        y: The array of y-values (signal amplitudes) corresponding to `x`.
        start_range: The x range for the initial baseline (before the pulse).
        end_range: The x range for the final baseline (after the pulse).
        plateau_range: The x range for the plateau region, if applicable.
         If None, uses the reduced y-values.
        y_start: The y value of the baseline at the start of the pulse.
        y_end: The y value of the baseline at the end of the pulse.

    Returns:
        1 if the pulse is positive, -1 if negative, or 0 if indeterminate.
    """
    if start_range is None:
        start_range = (x[0], x[0])
    if end_range is None:
        end_range = (x[-1], x[-1])

    # reduce x and y outside the base level
    y_red = y[np.logical_and(x >= start_range[1], x <= end_range[0])]
    if len(y_red) == 0:
        return 0

    if plateau_range is None:
        max_y = np.max(y_red)
        min_y = np.min(y_red)
    else:
        max_y = min_y = get_range_mean_y(x, y, plateau_range)
    positive_score = negative_score = 0

    y_start = get_range_mean_y(x, y, start_range) if y_start is None else y_start
    y_end = get_range_mean_y(x, y, end_range) if y_end is None else y_end

    if max_y > y_start and max_y > y_end:
        positive_score = (max_y - y_start) ** 2 + (max_y - y_end) ** 2
    if min_y < y_start and min_y < y_end:
        negative_score = (min_y - y_start) ** 2 + (min_y - y_end) ** 2
    return int(np.sign(positive_score - negative_score))


def detect_polarity(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    plateau_range: tuple[float, float] | None = None,
    signal_shape: SignalShape | None = None,
    fraction: float = 0.05,
) -> int:
    """Get step curve polarity.

    Args:
        x: Array of x-values (abscisse).
        y: Array of y-values (ordinate).
        start_range: Range for the start baseline.
        end_range: Range for the end baseline.
        plateau_range: Range for the plateau.
        signal_shape: Shape of the signal.
        fraction: Fraction of the x-range to use for baseline and plateau calculations.

    Returns:
        Polarity of the step (1 for positive, -1 for negative).

    Raises:
        PolarityDetectionError: If polarity cannot be determined.
        ValueError: If signal shape is unknown.
    """
    if start_range is None:
        start_range = get_start_range(x, fraction)
    if end_range is None:
        end_range = get_end_range(x, fraction)

    if signal_shape is None:
        signal_shape = heuristically_recognize_shape(x, y, start_range, end_range)

    y_start = get_range_mean_y(x, y, start_range)
    y_end = get_range_mean_y(x, y, end_range)

    if signal_shape == SignalShape.STEP:
        if y_start < y_end:
            return 1
        if y_start > y_end:
            return -1

        raise PolarityDetectionError(
            "Polarity could not be determined. Check signal data and baseline ranges."
        )
    if signal_shape == SignalShape.SQUARE:
        # Try square polarity detection first
        try:
            return _detect_square_polarity(
                x,
                y,
                start_range,
                end_range,
                plateau_range,
                y_start,
                y_end,
            )
        except (PolarityDetectionError, IndexError, ValueError) as exc:
            # If square detection fails, try Gaussian-like approach
            # This handles Gaussian signals that are misclassified as SQUARE
            baseline_mean = (y_start + y_end) / 2
            max_y = np.max(y)
            min_y = np.min(y)
            peak_value = (
                max_y if max_y - baseline_mean > baseline_mean - min_y else min_y
            )

            if peak_value > baseline_mean:
                return 1
            if peak_value < baseline_mean:
                return -1
            raise PolarityDetectionError(
                "Polarity could not be determined. Check signal data and "
                "baseline ranges."
            ) from exc

    raise ValueError(
        f"\nUnknown signal shape '{signal_shape}'. Use 'step' or 'square'."
    )


def get_start_range(x: np.ndarray, fraction: float = 0.05) -> tuple[float, float]:
    """Get start range based on fraction of x-range.

    Args:
        x: 1D array of x values.
        fraction: Fraction of the x-range to use for the start range.

    Returns:
        Tuple representing the start range (min, max).
    """
    x_fraction = fraction * (x[-1] - x[0])
    return (x[0], x[0] + x_fraction)


def get_end_range(x: np.ndarray, fraction: float = 0.05) -> tuple[float, float]:
    """Get end range based on fraction of x-range.

    Args:
        x: 1D array of x values.
        fraction: Fraction of the x-range to use for the end range.

    Returns:
        Tuple representing the end range (min, max).
    """
    x_fraction = fraction * (x[-1] - x[0])
    return (x[-1] - x_fraction, x[-1])


def get_plateau_range(
    x: np.ndarray,
    y: np.ndarray,
    polarity: int,
    fraction: float = 0.05,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Get plateau range around the max y-value based on fraction of x-range.

    The plateau is identified as the longest continuous region with values >= 90% of
    the maximum, constrained to be after the start baseline and before the end baseline.
    This ensures the plateau is detected in the correct temporal region of the signal,
    avoiding isolated spikes.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        polarity: Polarity of the signal (1 for positive, -1 for negative).
        fraction: Fraction of the x-range to use for the plateau range.
        start_range: Start baseline range (optional, for constraining search).
        end_range: End baseline range (optional, for constraining search).

    Returns:
        Tuple representing the plateau range (min, max).
    """
    # Get baseline ranges if not provided
    if start_range is None:
        start_range = get_start_range(x, fraction)
    if end_range is None:
        end_range = get_end_range(x, fraction)

    # Constrain search to region between end of start baseline and start of end baseline
    # This avoids the circular dependency with find_crossing_at_ratio
    mask = (x > start_range[1]) & (x < end_range[0])

    # Apply polarity correction
    y_polarity_corrected = y if polarity == 1 else np.max(y) - y

    # Find points >= 90% of maximum within the constrained region
    # (lower threshold for plateau)
    y_masked = y_polarity_corrected[mask]
    if len(y_masked) == 0:
        # Fallback: use full signal if constrained region is empty
        y_masked = y_polarity_corrected
        mask = np.ones_like(x, dtype=bool)

    # Use 90% threshold to capture the full plateau, not just the peak
    threshold = 0.9 * np.max(y_masked)
    above_threshold = y_masked >= threshold

    # Find the longest continuous region above threshold
    # This identifies the plateau, not isolated spikes
    changes = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
    start_indices = np.where(changes == 1)[0]
    end_indices = np.where(changes == -1)[0]

    if len(start_indices) == 0:
        # Fallback: if no regions found, use the maximum point
        max_idx = np.argmax(y_masked)
        x_fraction = fraction * (x[-1] - x[0])
        x_masked = x[mask]
        x_center = x_masked[max_idx]
        return (x_center - 0.5 * x_fraction, x_center + 0.5 * x_fraction)

    # Find the longest continuous region
    region_lengths = end_indices - start_indices
    longest_region_idx = np.argmax(region_lengths)
    plateau_start_idx = start_indices[longest_region_idx]
    plateau_end_idx = end_indices[longest_region_idx] - 1  # -1 because end is exclusive

    x_masked = x[mask]
    return (x_masked[plateau_start_idx], x_masked[plateau_end_idx])


def get_range_mean_y(
    x: np.ndarray,
    y: np.ndarray,
    value_range: tuple[float, float],
) -> float:
    """Get mean y-value in a given x-range.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        value_range: Tuple representing the x-range (min, max).

    Returns:
        Mean y-value in the specified x-range, or NaN if no points in range.
    """
    y_range = y[np.logical_and(x >= value_range[0], x <= value_range[1])]
    if len(y_range) == 0:
        return np.nan
    return float(np.mean(y_range))


@check_1d_arrays(x_sorted=True)
def get_amplitude(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    plateau_range: tuple[float, float] | None = None,
    signal_shape: SignalShape | str | None = None,
    fraction: float = 0.05,
) -> float:
    """Get curve amplitude.

    This function uses plateau-mean detection which is robust to noise but introduces
    a systematic error (~2-3%) for smooth pulse shapes (Gaussian, Lorentzian, etc.).
    This is an acceptable trade-off for robustness across all signal types.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        start_range: Range for the start baseline.
        end_range: Range for the end baseline.
        plateau_range: Range for the plateau.
        signal_shape: Shape of the signal.
        fraction: Fraction of the x-range to use for baseline and plateau calculations
         if start, end, or plateau are None.

    Returns:
        Amplitude of the step.

    .. warning::

        For noise-free smooth peaks (Gaussian, Lorentzian), this method introduces
        a systematic offset of approximately +2-3% due to plateau-mean detection.
        For accurate results with known signal shapes, use fitting methods instead
        (e.g., fwhm with method='gauss' for Gaussian signals).
    """
    if signal_shape is None:
        signal_shape = heuristically_recognize_shape(x, y, start_range, end_range)

    if start_range is None:
        start_range = get_start_range(x, fraction)
    if end_range is None:
        end_range = get_end_range(x, fraction)

    if signal_shape == SignalShape.STEP:
        min_level = get_range_mean_y(x, y, start_range)
        max_level = get_range_mean_y(x, y, end_range)
    elif signal_shape == SignalShape.SQUARE:
        try:
            polarity = detect_polarity(
                x, y, start_range, end_range, signal_shape=signal_shape
            )
        except PolarityDetectionError:
            # If polarity cannot be determined, use total amplitude
            return np.max(y) - np.min(y)

        if plateau_range is None:
            plateau_range = get_plateau_range(
                x, y, polarity, fraction, start_range, end_range
            )

        # reverse y if polarity is negative
        y_positive = y * polarity
        # compute base level
        min_level = get_range_mean_y(x, y_positive, start_range)
        max_level = get_range_mean_y(x, y_positive, plateau_range)
    else:
        raise ValueError("Unknown signal type. Use 'step' or 'square'.")

    return np.abs(min_level - max_level)


@check_1d_arrays(x_sorted=True)
def find_crossing_at_ratio(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float = 0.1,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    signal_shape: SignalShape | str | None = None,
    fraction: float = 0.05,
    warn_multiple_crossings: bool = False,
) -> float | None:
    """Find the x-value at which the signal crosses a specified fractional amplitude.

    Calculates the x-value at which a normalized step signal crosses a specified
    fractional amplitude.

    This function normalizes the input signal `y` relative to the baseline level defined
    by `start` and the amplitude between `start` and `end`. It accounts for the
    polarity of the step (rising or falling) and then finds the x-position where the
    normalized signal crosses the specified `ratio` fraction of the step height.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        ratio: The fractional amplitude (between 0 and 1) at which to find the
         crossing time. For example, 0.5 corresponds to the half-maximum crossing.
        start_range: Tuple defining the start baseline region (initial plateau).
        end_range: Tuple defining the end baseline region (final plateau).
        signal_shape: Shape of the signal. If None, it will be heuristically
         determined.
        fraction: Fraction of the x-range to use for baseline calculations if
         start_range or end_range are None.
        warn_multiple_crossings: If True, a warning is issued when multiple crossings
         are found.

    Returns:
        The x-value where the normalized signal crosses the specified fractional
        amplitude.

    Raises:
        ValueError: If `ratio` is not between 0 and 1.
        InvalidSignalError: If the signal is invalid or if polarity cannot be
         determined.
    """
    # pylint: disable=too-many-return-statements
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be between 0 and 1")
    if signal_shape is None:
        signal_shape = heuristically_recognize_shape(x, y, start_range, end_range)
    try:
        polarity = detect_polarity(
            x,
            y,
            start_range,
            end_range,
            signal_shape=signal_shape,
        )
    except PolarityDetectionError as exc:
        raise InvalidSignalError(f"Cannot determine crossing time: {exc}") from exc
    amplitude = get_amplitude(
        x, y, start_range, end_range, signal_shape=signal_shape, fraction=fraction
    )
    y_positive = y * polarity
    if start_range is None:
        start_range = get_start_range(x, fraction)
    if end_range is None:
        end_range = get_end_range(x, fraction)
    y_start = get_range_mean_y(x, y_positive, start_range)
    if amplitude == 0.0:
        return None
    y_norm = (y_positive - y_start) / amplitude

    # Constrain search to the rise/fall edge region (between baselines)
    # This prevents finding spurious crossings in the baseline regions
    mask = (x > start_range[1]) & (x < end_range[0])
    x_search = x[mask]
    y_norm_search = y_norm[mask]

    if len(x_search) == 0:
        return None

    # Special handling for low ratios (0% to 10%):
    # For these, we need to avoid finding crossings in the baseline noise.
    # Strategy: Find the 10% crossing first, then search backwards to find
    # the requested crossing point near the actual rise edge.
    if ratio < 0.1:
        try:
            # First find the 10% crossing point as a reference
            roots_10pct = features.find_x_values_at_y(x_search, y_norm_search, 0.1)
            if len(roots_10pct) > 0:
                x_10pct = roots_10pct[0]
                # Now search for the requested ratio only in the region
                # leading up to the 10% crossing
                mask_near_edge = x_search <= x_10pct
                x_near_edge = x_search[mask_near_edge]
                y_norm_near_edge = y_norm_search[mask_near_edge]
                roots = features.find_x_values_at_y(
                    x_near_edge, y_norm_near_edge, ratio
                )
                # Return the crossing closest to the 10% point
                if len(roots) > 0:
                    return roots[-1]  # Last crossing before 10% point
        except ValueError:
            pass
        # Fallback to regular search if 10% crossing not found
        try:
            roots = features.find_x_values_at_y(x_search, y_norm_search, ratio)
        except ValueError:
            return None
    else:
        # For higher ratios, regular search works fine
        try:
            roots = features.find_x_values_at_y(x_search, y_norm_search, ratio)
        except ValueError:
            return None

    if len(roots) == 0:
        return None
    if len(roots) > 1 and warn_multiple_crossings:
        warnings.warn(
            f"Multiple crossing points found at ratio {ratio}. "
            f"Returning first at x={roots[0]:.6f}"
        )
    return roots[0]


def _find_gaussian_crossing_times(
    x: np.ndarray,
    y: np.ndarray,
    start_ratio: float,
    stop_ratio: float,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
) -> tuple[float, float] | None:
    """Find crossing times for Gaussian signals with proper handling of multiple
    crossings.

    For Gaussian signals, we want:
    - Left side crossing at start_ratio
    - Right side crossing at stop_ratio

    This gives a meaningful "rise time" across the Gaussian peak.

    Returns:
        Tuple of (start_time, stop_time) or None if calculation fails.
    """
    try:
        # Get signal properties
        signal_shape = heuristically_recognize_shape(x, y, start_range, end_range)
        polarity = detect_polarity(
            x, y, start_range, end_range, signal_shape=signal_shape
        )
        amplitude = get_amplitude(
            x, y, start_range, end_range, signal_shape=signal_shape
        )

        y_positive = y * polarity
        y_start = get_range_mean_y(x, y_positive, start_range)

        if amplitude == 0.0:
            return None

        y_norm = (y_positive - y_start) / amplitude

        # Find all crossing points
        start_roots = features.find_x_values_at_y(x, y_norm, start_ratio)
        stop_roots = features.find_x_values_at_y(x, y_norm, stop_ratio)

        # For true Gaussian signals, we expect exactly 2 crossings per ratio
        # (left and right sides). However, for truncated signals (e.g., in
        # extract_pulse_features), we might only see 1 crossing (left side only).
        if len(start_roots) == 1 and len(stop_roots) == 1:
            # Truncated Gaussian (left side only) - use these crossings directly
            start_time = start_roots[0]
            stop_time = stop_roots[0]
            return (start_time, stop_time)
        if len(start_roots) != 2 or len(stop_roots) != 2:
            # Not a clean Gaussian-style signal
            return None

        # Additional check: Gaussian signals should be roughly symmetric
        # Square signals often have asymmetric crossing patterns
        center_x = (x[0] + x[-1]) / 2
        start_center = (start_roots[0] + start_roots[1]) / 2
        stop_center = (stop_roots[0] + stop_roots[1]) / 2

        # If crossings are very asymmetric or far from signal center,
        # it's likely a square signal, not Gaussian
        max_center_offset = (x[-1] - x[0]) * 0.05  # 5% of signal range (stricter)
        if (
            abs(start_center - center_x) > max_center_offset
            or abs(stop_center - center_x) > max_center_offset
        ):
            # Too asymmetric for a Gaussian
            return None

        # For Gaussian signals, calculate rise/fall time on ONE side only
        # start_roots[0] = left side, start_roots[1] = right side
        # stop_roots[0] = left side, stop_roots[1] = right side
        if stop_ratio > start_ratio:
            # Rise time: use left side only (start_ratio to stop_ratio)
            start_time = start_roots[0]  # Left side at start_ratio
            stop_time = stop_roots[0]  # Left side at stop_ratio
        else:
            # Fall time: use right side only (start_ratio to stop_ratio)
            start_time = start_roots[1]  # Right side at start_ratio
            stop_time = stop_roots[1]  # Right side at stop_ratio

        return (start_time, stop_time)

    except (ValueError, PolarityDetectionError, InvalidSignalError):
        return None


def _get_rise_time_traditional(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    start_ratio: float,
    stop_ratio: float,
) -> float | None:
    """Internal function for traditional ratio-based rise time calculation.

    This avoids recursion by providing the core traditional implementation.
    For Gaussian signals, it uses special multi-crossing logic.
    """
    # Check if this might be a Gaussian signal by detecting signal shape
    # Only try Gaussian method for signals that have clean symmetric crossings
    try:
        gaussian_result = _find_gaussian_crossing_times(
            x, y, start_ratio, stop_ratio, start_range, end_range
        )
        if gaussian_result is not None:
            start_time, stop_time = gaussian_result
            return abs(stop_time - start_time)
    except (InvalidSignalError, PolarityDetectionError):
        pass  # Fall back to traditional method

    # Traditional method for step/square signals
    # Find crossing times for both ratios
    start_time = find_crossing_at_ratio(x, y, start_ratio, start_range, end_range)
    stop_time = find_crossing_at_ratio(x, y, stop_ratio, start_range, end_range)

    if start_time is None or stop_time is None:
        return None

    # Validate that start_time and stop_time are in correct order
    if (stop_ratio > start_ratio and start_time >= stop_time) or (
        stop_ratio < start_ratio and stop_time >= start_time
    ):
        return None

    return abs(stop_time - start_time)


def get_rise_time_estimated(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    start_ratio: float = 0.1,
    stop_ratio: float = 0.9,
) -> float | None:
    """Calculates rise time using heuristic foot detection and 50% crossing estimation.

    This method uses a more robust approach:
    1. Find the true start of the systematic rise (foot end time)
    2. Find the 50% amplitude crossing
    3. Estimate the full rise duration from these two points
    4. Calculate the requested ratio-based rise time from the estimated parameters

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_range: Tuple defining the start plateau region (before the rise).
        end_range: Tuple defining the end plateau region (after the rise).
        start_ratio: Fraction of the step height at which the rise starts.
        stop_ratio: Fraction of the step height at which the rise ends.

    Returns:
        The estimated rise time between the specified ratios.
    """
    if start_range is None:
        start_range = get_start_range(x)
    if end_range is None:
        end_range = get_end_range(x)

    # Step 1: Find the true start of the rise (foot end)
    try:
        foot_end_time = heuristically_find_rise_start_time(x, y, start_range)
    except InvalidSignalError:
        foot_end_time = None
    if foot_end_time is None:
        # Fallback to traditional method if heuristic fails
        return _get_rise_time_traditional(
            x, y, start_range, end_range, start_ratio, stop_ratio
        )

    # Step 2: Find the 50% crossing point
    t_50_percent = find_crossing_at_ratio(x, y, 0.5, start_range, end_range)
    if t_50_percent is None:
        # Fallback to traditional method if 50% crossing not found
        return _get_rise_time_traditional(
            x, y, start_range, end_range, start_ratio, stop_ratio
        )

    # Step 3: Estimate the full rise duration
    # If we assume linear rise: t_50% = t_start + 0.5 * total_rise_time
    # Therefore: total_rise_time = 2 * (t_50% - t_start)
    estimated_total_rise_time = 2.0 * (t_50_percent - foot_end_time)

    # Validate the estimation makes sense
    if estimated_total_rise_time <= 0:
        warnings.warn(
            f"Invalid rise time estimation: foot_end ({foot_end_time:.3f}) >= "
            f"50% crossing ({t_50_percent:.3f}). Using fallback method."
        )
        return _get_rise_time_traditional(
            x, y, start_range, end_range, start_ratio, stop_ratio
        )

    # Step 4: Calculate ratio-based times from the estimated parameters
    estimated_start_time = foot_end_time + start_ratio * estimated_total_rise_time
    estimated_stop_time = foot_end_time + stop_ratio * estimated_total_rise_time

    return estimated_stop_time - estimated_start_time


def get_rise_time(
    x: np.ndarray,
    y: np.ndarray,
    start_ratio: float = 0.1,
    stop_ratio: float = 0.9,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> float | None:
    """Calculates the rise time of a step-like signal between two defined plateaus.

    The rise time is defined as the time it takes for the signal to increase from
    start_ratio to stop_ratio of the total amplitude change.

    For rise time calculations (stop_ratio > start_ratio), this function
    automatically uses an improved estimation method that combines heuristic foot
    detection with 50% crossing analysis for better accuracy in noisy signals.

    For fall time calculations (stop_ratio < start_ratio), it uses the
    traditional ratio-based method.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_ratio: Fraction of the step height at which the rise starts.
         Default is 0.1 (i.e., 10% of the step height).
        stop_ratio: Fraction of the step height at which the rise ends.
         Default is 0.9 (i.e., 90% of the step height).
        start_range: Tuple defining the start plateau region (before the rise).
        end_range: Tuple defining the end plateau region (after the rise).

    Returns:
        The rise time (difference between the stop and start ratio crossings).

    Raises:
        ValueError: If ratios are not between 0 and 1.
    """
    if start_ratio < 0.0 or start_ratio > 1.0:
        raise ValueError("start_ratio must be between 0 and 1")
    if stop_ratio < 0.0 or stop_ratio > 1.0:
        raise ValueError("stop_ratio must be between 0 and 1")

    if start_range is None:
        start_range = get_start_range(x)
    if end_range is None:
        end_range = get_end_range(x)

    # For rise time calculations (stop > start), try traditional method first
    # and use enhanced method only when needed for robustness
    if stop_ratio > start_ratio:
        # Check if we have a degenerate range (single point) - use traditional only
        if start_range[0] == start_range[1]:
            return _get_rise_time_traditional(
                x, y, start_range, end_range, start_ratio, stop_ratio
            )

        # Try traditional method first (better for clean signals)
        traditional_result = _get_rise_time_traditional(
            x, y, start_range, end_range, start_ratio, stop_ratio
        )

        if traditional_result is not None:
            # Check if this is a Gaussian signal - if so, trust the Gaussian result
            gaussian_result = _find_gaussian_crossing_times(
                x, y, start_ratio, stop_ratio, start_range, end_range
            )
            if gaussian_result is not None:
                # Gaussian detection succeeded - trust this result
                return traditional_result

            # For non-Gaussian signals, check if enhanced method needed
            # Estimate signal quality by checking foot end detection reliability
            foot_end_time = heuristically_find_rise_start_time(x, y, start_range)
            if foot_end_time is not None:
                # Compare traditional result with enhanced method
                enhanced_result = get_rise_time_estimated(
                    x, y, start_range, end_range, start_ratio, stop_ratio
                )

                if enhanced_result is not None:
                    # If results differ significantly, signal might be noisy
                    max_result = max(traditional_result, enhanced_result)
                    relative_diff = abs(traditional_result - enhanced_result)
                    relative_diff /= max_result

                    # Use enhanced method if significant discrepancy (indicating noise)
                    if relative_diff > 0.3:  # 30% threshold
                        return enhanced_result

            return traditional_result

        # If traditional method fails, try enhanced method
        enhanced_result = get_rise_time_estimated(
            x, y, start_range, end_range, start_ratio, stop_ratio
        )
        if enhanced_result is not None:
            return enhanced_result

    # Traditional method for fall time calculations or as fallback
    result = _get_rise_time_traditional(
        x, y, start_range, end_range, start_ratio, stop_ratio
    )

    if result is None:
        warnings.warn(
            "Could not determine start or stop time for the step rise. Returning None."
        )

    return result


def get_fall_time(
    x: np.ndarray,
    y: np.ndarray,
    start_ratio: float = 0.9,
    stop_ratio: float = 0.1,
    plateau_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    fraction: float = 0.05,
) -> float | None:
    """Calculates the fall time of a step-like signal between two defined plateaus.

    The fall time is defined as the time it takes for the signal to decrease from
    start_ratio to stop_ratio of the total amplitude change.

    This function internally reverses the signal and applies the rise time calculation
    to determine the fall time.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_ratio: Fraction of the step height at which the fall starts.
         Default is 0.9 (i.e., 90% of the step height).
        stop_ratio: Fraction of the step height at which the fall ends.
         Default is 0.1 (i.e., 10% of the step height).
        plateau_range: Tuple defining the plateau region (top of the step).
         If None, uses the peak y-value.
        end_range: Tuple defining the end plateau region (after the fall).
        fraction: Fraction of the x-range to use for baseline calculations if
         plateau_range is None.

    Returns:
        The fall time (difference between the stop and start ratio crossings).

    Raises:
        ValueError: If start_ratio is not greater than stop_ratio or if ratios are
         not between 0 and 1.
    """
    if start_ratio < 0.0 or start_ratio > 1.0:
        raise ValueError("start_ratio must be between 0 and 1")
    if stop_ratio < 0.0 or stop_ratio > 1.0:
        raise ValueError("stop_ratio must be between 0 and 1")
    if start_ratio <= stop_ratio:
        raise ValueError("For fall time, start_ratio must be greater than stop_ratio")

    # Check if this might be a Gaussian signal
    try:
        if end_range is None:
            end_range = get_end_range(x)
        start_range = get_start_range(x)

        signal_shape = heuristically_recognize_shape(x, y, start_range, end_range)
        if signal_shape in ["square", "gaussian"]:  # Gaussian often detected as square
            # For Gaussian signals, use symmetric crossing calculation
            gaussian_result = _find_gaussian_crossing_times(
                x, y, start_ratio, stop_ratio, start_range, end_range
            )
            if gaussian_result is not None:
                start_time, stop_time = gaussian_result
                return abs(stop_time - start_time)
    except (InvalidSignalError, PolarityDetectionError):
        pass  # Fall back to traditional method

    # Traditional method for step/square signals
    if plateau_range is None:
        ymax_idx = np.argmax(y)
    else:
        # Use the index of the lower plateau boundary as the peak index:
        plateau_mask = (x >= plateau_range[0]) & (x <= plateau_range[1])
        if not np.any(plateau_mask):
            raise InvalidSignalError("No data points found in plateau_range")
        ymax_idx = np.nonzero(plateau_mask)[0][0]
    # For fall time calculation, we need a proper baseline range in the fall segment
    # Use a small portion near the peak as baseline instead of degenerate range
    x_fall = x[ymax_idx:]
    if len(x_fall) > 10:  # Ensure we have enough points
        # Use the plateau if provided
        if plateau_range is None:
            # Use first fraction% of the fall segment as baseline, but at least 2 points
            baseline_points = max(2, int(fraction * len(x_fall)))
            fall_baseline_range = (x_fall[0], x_fall[baseline_points])
        else:
            fall_baseline_range = plateau_range
    else:
        # Fallback to a small range around the peak
        fall_baseline_range = (x[ymax_idx], x[min(ymax_idx + 2, len(x) - 1)])
    fall_time = get_rise_time(
        x[ymax_idx:],
        y[ymax_idx:],
        start_ratio,
        stop_ratio,
        fall_baseline_range,
        end_range,
    )
    return fall_time


@check_1d_arrays(x_sorted=True)
def heuristically_find_rise_start_time(
    x: np.ndarray, y: np.ndarray, start_range: tuple[float, float]
) -> float | None:
    """Finds the point where a step signal begins its systematic rise from baseline.

    This function uses multiple strategies to detect the true start of a step
    transition:
    1. Trend analysis to identify sustained directional change
    2. Moving window statistics to detect consistent deviations
    3. Gradient analysis for backup detection

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_range: Tuple defining the lower plateau region (start of the step).

    Returns:
        The x-value of the first systematic rise, or None if no such value is found.

    Raises:
        InvalidSignalError: If insufficient data is provided.
    """
    if y.size < 20:
        raise InvalidSignalError(
            "Insufficient data for statistical analysis (need ≥20 points)"
        )

    start_indices = np.nonzero(x >= start_range[1])[0]
    if len(start_indices) == 0:
        raise InvalidSignalError("No data points found after start_baseline_range")

    start_idx = start_indices[0]

    # Calculate baseline statistics from the start_range
    baseline_mask = (x >= start_range[0]) & (x <= start_range[1])
    if np.sum(baseline_mask) < 5:
        raise InvalidSignalError("Insufficient baseline data")

    baseline_y = y[baseline_mask]
    baseline_mean = np.mean(baseline_y)
    baseline_std = np.std(baseline_y)

    # Strategy 1: Look for consistent upward trend
    # Use a moving window to detect sustained increase
    window_size = max(5, len(y) // 100)  # Adaptive window size

    for i in range(start_idx, len(y) - window_size):
        # Get a window of data points
        window_y = y[i : i + window_size]
        window_mean = np.mean(window_y)

        # Check if this window is significantly above baseline
        if window_mean > baseline_mean + 1.5 * baseline_std:
            # Verify this is the start of a trend by checking if subsequent windows
            # continue to be elevated
            confirmation_windows = 0
            for j in range(
                i + window_size,
                min(i + 3 * window_size, len(y) - window_size),
                window_size,
            ):
                next_window_y = y[j : j + window_size]
                next_window_mean = np.mean(next_window_y)
                if next_window_mean >= window_mean:
                    confirmation_windows += 1

            # If we have at least one confirming window, this looks like the start
            if confirmation_windows >= 1:
                return x[i]

    # Strategy 2: Gradient-based detection with smoothing
    # Smooth the signal to reduce noise effects
    y_smooth = scipy.ndimage.gaussian_filter1d(y, sigma=2.0)

    dy = np.gradient(y_smooth, x)

    # Find the region to search (after start_range)
    search_mask = x >= start_range[1]
    search_x = x[search_mask]
    search_dy = dy[search_mask]

    if len(search_dy) > 10:
        # Look for sustained positive gradient
        baseline_dy = dy[(x >= start_range[0]) & (x <= start_range[1])]
        dy_baseline_std = np.std(baseline_dy) if len(baseline_dy) > 0 else 0.1

        # Find first point where gradient is consistently above noise level
        gradient_threshold = 3 * dy_baseline_std

        for i in range(len(search_dy) - 3):
            # Check if gradient is above threshold for several consecutive points
            if np.all(search_dy[i : i + 3] > gradient_threshold):
                return search_x[i]

    # Strategy 3: Fall back to simple threshold above baseline
    # Look for first point that's consistently above baseline + 2*std
    threshold_y = baseline_mean + 2.0 * baseline_std

    search_y = y[search_mask]
    for i in range(len(search_y) - 2):
        if np.all(search_y[i : i + 3] > threshold_y):
            return search_x[i]

    return None


def get_rise_start_time(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    threshold: float | None = None,
) -> float:
    """Find the rise start time of a step signal using multiple strategies.

    This function tries multiple approaches to find the rise start time:
    1. Uses threshold crossing if threshold is provided
    2. Uses heuristic detection as fallback
    3. Validates results to ensure they make physical sense

    Args:
        x: 1D array of x values (e.g., time or position).
        y: 1D array of y values (same size as x).
        start_range: A range (min, max) representing the initial flat region ("foot").
        end_range: A range (min, max) representing the final high region after the rise.
        threshold: If provided, use this fractional amplitude (0-1) to determine the
         end of the foot. If None, use heuristic detection.

    Returns:
        The rise start time (foot end time).

    Raises:
        InvalidSignalError: If rise start time cannot be determined.
    """
    # Try heuristic detection first as it's often more reliable for step detection
    try:
        heuristic_result = heuristically_find_rise_start_time(x, y, start_range)
    except InvalidSignalError:
        heuristic_result = None

    # Try threshold method if requested
    threshold_result = None
    if threshold is not None:
        try:
            threshold_result = find_crossing_at_ratio(
                x, y, threshold, start_range, end_range
            )
        except InvalidSignalError:
            pass

    # Validate and choose the best result
    if heuristic_result is not None and threshold_result is not None:
        # If both methods give results, prefer the one that's more reasonable
        # For step signals, the heuristic should give a later (more accurate) time
        # than a low threshold crossing
        if threshold <= 0.2 and heuristic_result > threshold_result:
            return heuristic_result
        return threshold_result
    if heuristic_result is not None:
        return heuristic_result
    if threshold_result is not None:
        return threshold_result

    # Last resort: look for the steepest part of the signal
    dy = np.gradient(y, x)
    search_mask = (x >= start_range[1]) & (x <= end_range[0])
    if np.any(search_mask):
        search_indices = np.where(search_mask)[0]
        max_dy_idx = search_indices[np.argmax(dy[search_mask])]
        # Move back to find the start of the steep region
        for i in range(max_dy_idx, max(0, max_dy_idx - 50), -1):
            if dy[i] <= dy[max_dy_idx] * 0.1:  # 10% of max gradient
                return x[i]
        return x[max_dy_idx]

    raise InvalidSignalError("Could not determine rise start time with any method")


@check_1d_arrays(x_sorted=True)
def full_width_at_y(
    x: np.ndarray,
    y: np.ndarray,
    level: float,
) -> tuple[float, float, float, float]:
    """Compute the full width at a given y level of a square shaped signal using
    zero-crossing method.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        level: The Y level at which to compute the width

    Returns:
        Full width segment coordinates (x1, level, x2, level)
    """
    tmax_idx = np.argmax(y)

    roots1 = features.find_x_values_at_y(
        x[0 : tmax_idx + 1],
        y[0 : tmax_idx + 1],
        level,
    )
    if len(roots1) > 1:
        warnings.warn("Multiple crossing points found. Returning first.")
    roots2 = features.find_x_values_at_y(
        x[tmax_idx:],
        y[tmax_idx:],
        level,
    )
    if len(roots2) > 1:
        warnings.warn("Multiple crossing points found. Returning last.")
    t1 = roots1[0] if len(roots1) > 0 else np.nan
    t2 = roots2[-1] if len(roots2) > 0 else np.nan
    return t1, level, t2, level


def full_width_at_ratio(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    fraction: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Calculate the full width at a specified ratio of the amplitude for a pulse signal.

    This function determines the two crossing points (x1, x2) where the normalized
    signal crosses a given ratio of its amplitude, and returns these points along with
    the corresponding y-level.

    Args:
        x: 1D array of x-values.
        y: 1D array of y-values.
        ratio: Ratio (between 0 and 1) of the amplitude at which to measure the width.
        start_range: Range of x-values to estimate the start baseline.
        end_range: Range of x-values to estimate the end baseline.
        fraction: Fraction of the x-range to use for baseline calculations if
         start_range or end_range are None.

    Returns:
        Full width segment coordinates (x1, level, x2, level), where x1 and x2 are the
         crossing points at the specified ratio, and level is the corresponding y-value.

    Raises:
        ValueError: If the amplitude of the signal is zero.
        RuntimeWarning: If the polarity cannot be determined, returns NaN for crossing
         times.

    Notes:
        - The function normalizes the signal based on the detected amplitude and
          polarity.
        - The crossing times are computed using `features.find_first_x_at_y_value`
          function.
    """
    amplitude = get_amplitude(x, y, start_range, end_range)

    try:
        polarity = detect_polarity(x, y, start_range, end_range)
    except PolarityDetectionError as e:
        raise InvalidSignalError(f"Cannot determine width at ratio: {e}") from e

    if start_range is None:
        start_range = get_start_range(x, fraction)
    if end_range is None:
        end_range = get_end_range(x, fraction)
    start_baseline = get_range_mean_y(x, y * polarity, start_range)

    if amplitude == 0:
        raise InvalidSignalError(
            "Amplitude of your square signal is zero. Check your data."
        )

    y_norm = np.asarray(polarity * (y - start_baseline) / amplitude, dtype=y.dtype.type)

    level = y.dtype.type(ratio * polarity * amplitude + start_baseline)

    tmax_idx = np.argmax(y_norm)

    try:
        roots1 = features.find_x_values_at_y(
            x[0 : tmax_idx + 1],
            y_norm[0 : tmax_idx + 1],
            ratio,
        )
    except ValueError:
        roots1 = []
    if len(roots1) > 1:
        warnings.warn("Multiple crossing points found. Returning first.")
    x1 = roots1[0] if len(roots1) > 0 else np.nan
    try:
        roots2 = features.find_x_values_at_y(
            x[tmax_idx:],
            y_norm[tmax_idx:],
            ratio,
        )
    except ValueError:
        roots2 = []
    if len(roots2) > 1:
        warnings.warn("Multiple crossing points found. Returning last.")
    x2 = roots2[-1] if len(roots2) > 0 else np.nan
    return x1, level, x2, level


def fwhm(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["zero-crossing", "gauss", "lorentz", "voigt"] = "zero-crossing",
    xmin: float | None = None,
    xmax: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute Full Width at Half Maximum (FWHM) of the input data.

    Two types of methods are supported:

      - "zero-crossing": Fast interpolation-based method that works directly with
        signal values (no normalization). Shape-independent and robust.
      - "gauss", "lorentz", "voigt": Curve fitting methods for specific signal
        shapes. Very accurate when the signal matches the model.

    Args:
        x: 1D array of x-values.
        y: 1D array of y-values.
        method: Calculation method.
        xmin: Lower X bound for the fitting. Defaults to None (no lower bound,
         i.e. the fitting starts from the first point).
        xmax: Upper X bound for the fitting. Defaults to None (no upper bound,
         i.e. the fitting ends at the last point)

    Returns:
        FWHM segment coordinates (x1, y1, x2, y2)

    .. warning::

        The zero-crossing method uses amplitude normalization via
        :py:func:`get_amplitude`,
        which introduces a small systematic offset (~2-3%) for smooth peaks
        (Gaussian, Lorentzian, etc.) but is robust to noise. For accurate results
        with known signal shapes, use fitting methods (e.g., method='gauss').
    """
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, peakdetection.xpeak(x, y)
    if isinstance(xmin, float):
        indices = np.where(x >= xmin)[0]
        x = x[indices]
        y = y[indices]
    if isinstance(xmax, float):
        indices = np.where(x <= xmax)[0]
        x = x[indices]
        y = y[indices]

    if method == "zero-crossing":
        x1, y1, x2, y2 = full_width_at_ratio(x, y, 0.5)
        return x1, y1, x2, y2

    try:
        fit_model_class: type[PulseFitModel] = {
            "gauss": GaussianModel,
            "lorentz": LorentzianModel,
            "voigt": VoigtModel,
        }[method]
    except KeyError as exc:
        raise ValueError(f"Invalid method {method}") from exc

    def func(params) -> np.ndarray:
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - fit_model_class.evaluate(x, *params)

    amplitude = dy
    (amplitude, sigma, mu, base), _ier = scipy.optimize.leastsq(
        func, np.array([amplitude, sigma, mu, base])
    )
    return fit_model_class.half_max_segment(amplitude, sigma, mu, base)


@check_1d_arrays(x_sorted=True)
def fw1e2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Full Width at 1/e² of the input data (using a Gaussian model fitting).

    Args:
        x: 1D array of x-values.
        y: 1D array of y-values.

    Returns:
        FW at 1/e² segment coordinates
    """
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, peakdetection.xpeak(x, y)
    p_in = np.array([dy, sigma, mu, base])

    def func(params):
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - GaussianModel.evaluate(x, *params)

    p_out, _ier = scipy.optimize.leastsq(func, p_in)
    amplitude, sigma, mu, base = p_out
    hw = 2 * sigma
    yhm = amplitude / np.e**2 + base
    return mu - hw, yhm, mu + hw, yhm


@dataclass
class PulseFeatures:
    """Extracted features from a pulse signal.

    Attributes:
        signal_shape: The shape of the signal (step or square).
        polarity: The polarity of the signal (1 for positive, -1 for negative).
        amplitude: The amplitude of the signal.
        offset: The baseline offset of the signal.
        foot_duration: The duration of the foot (flat region before rise) of the signal.
        xstartmin: The minimum x-value of the start baseline region.
        xstartmax: The maximum x-value of the start baseline region.
        xendmin: The minimum x-value of the end baseline region.
        xendmax: The maximum x-value of the end baseline region.
        xplateaumin: The minimum x-value of the plateau region (if applicable).
        xplateaumax: The maximum x-value of the plateau region (if applicable).
        rise_time: The rise time of the signal (time from start_ratio to stop_ratio).
        fall_time: The fall time of the signal (time from stop_ratio to start_ratio).
        fwhm: The full width at half maximum of the signal.
        x50: The time at which the signal reaches 50% of its amplitude.
        x100: The time at which the signal reaches its maximum amplitude.
    """

    signal_shape: SignalShape = SignalShape.STEP
    polarity: int = 1
    amplitude: float = 0.0
    offset: float = 0.0
    foot_duration: float = 0.0
    xstartmin: float = 0.0
    xstartmax: float = 0.0
    xendmin: float = 0.0
    xendmax: float = 0.0
    xplateaumin: float | None = None
    xplateaumax: float | None = None
    rise_time: float | None = None
    fall_time: float | None = None
    fwhm: float | None = None
    x0: float | None = None
    x50: float | None = None
    x100: float | None = None


def extract_pulse_features(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    start_ratio: float = 0.1,
    stop_ratio: float = 0.9,
    signal_shape: SignalShape | None = None,
    fraction: float = 0.05,
    denoise: bool = True,
) -> PulseFeatures:
    """Extract various pulse features from the input signal.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values (signal).
        start_range: Interval for the first plateau (baseline).
        end_range: Interval for the second plateau (peak).
        signal_shape: Signal type (None for auto-detection).
        start_ratio: Fraction for rise start.
        stop_ratio: Fraction for rise end.
        fraction: Fraction of the x-range to use for baseline calculations if
         start_range or end_range are None.
        denoise: If True, apply a denoising filter to the signal before analysis.

    Returns:
        Pulse features.

    Raises:
        ValueError: If input parameters are invalid.
    """
    if start_ratio > 1.0 or start_ratio < 0.0:
        raise ValueError("start_ratio must be between 0 and 1")
    if stop_ratio > 1.0 or stop_ratio < 0.0:
        raise ValueError("stop_ratio must be between 0 and 1")
    if start_ratio >= stop_ratio:
        raise ValueError("start_ratio must be less than stop_ratio")

    # Initialize ranges if None (auto-detection)
    if start_range is None:
        start_range = get_start_range(x, fraction)
    if end_range is None:
        end_range = get_end_range(x, fraction)

    if signal_shape is None:
        signal_shape = heuristically_recognize_shape(x, y, start_range, end_range)
    if not isinstance(signal_shape, (SignalShape, str)):
        raise ValueError("Invalid signal_shape")

    if denoise:
        y = filtering.denoise_preserve_shape(y)[0]

    polarity = detect_polarity(x, y, start_range, end_range, signal_shape=signal_shape)
    plateau_range = None
    if signal_shape == SignalShape.SQUARE:
        plateau_range = get_plateau_range(
            x, y, polarity, fraction, start_range, end_range
        )
    amplitude = get_amplitude(x, y, start_range, end_range, plateau_range, signal_shape)

    ymax_idx = np.argmax(y)

    if signal_shape == SignalShape.STEP:
        rise_time = get_rise_time(x, y, start_ratio, stop_ratio, start_range, end_range)
        x0 = get_rise_start_time(x, y, start_range, end_range)
        x50 = find_crossing_at_ratio(x, y, 0.5, start_range, end_range)
        fall_time = None
        fwhm_val = None
    else:  # is square
        rise_time = get_rise_time(
            x[0 : ymax_idx + 1],
            y[0 : ymax_idx + 1],
            start_ratio,
            stop_ratio,
            start_range,
            (x[ymax_idx], x[ymax_idx]),
        )
        # Check if this is a Gaussian signal - if so, x50 should be on the rise
        gaussian_result = _find_gaussian_crossing_times(
            x, y, start_ratio, stop_ratio, start_range, end_range
        )
        if gaussian_result is not None:
            # For Gaussian signals, x50 is the 50% crossing on the rise (left side)
            x50 = find_crossing_at_ratio(
                x[0 : ymax_idx + 1],
                y[0 : ymax_idx + 1],
                0.5,
                start_range,
                (x[ymax_idx], x[ymax_idx]),
            )
        else:
            # For square signals, x50 is at the 50% crossing
            x50 = find_crossing_at_ratio(
                x[0 : ymax_idx + 1],
                y[0 : ymax_idx + 1],
                0.5,
                start_range,
                (x[ymax_idx], x[ymax_idx]),
            )
        fall_time = get_fall_time(
            x, y, stop_ratio, start_ratio, plateau_range, end_range
        )
        x0 = get_rise_start_time(
            x[0 : ymax_idx + 1],
            y[0 : ymax_idx + 1],
            start_range,
            (x[ymax_idx], x[ymax_idx]),
        )
        x1, _, x2, _ = fwhm(x, y, "zero-crossing")
        fwhm_val = x2 - x1  # full width at half maximum
        mean_x_sampling_time = float(np.mean(np.diff(x)))
        if fwhm_val <= 10 * mean_x_sampling_time:
            # if the fwhm is smaller than 10 times the mean sampling time, we cannot
            # rely on rising and falling times, as the pulse is too narrow
            fall_time = None
            rise_time = None

    if x50 is None:
        # No x50 found (e.g., for very narrow pulses or Gaussian signals)
        x100 = x[ymax_idx]
    else:
        # For step/square signals, calculate conventional x100
        # Check if this is likely a Gaussian signal (symmetric crossings)
        gaussian_result = _find_gaussian_crossing_times(
            x, y, start_ratio, stop_ratio, start_range, end_range
        )
        if gaussian_result is not None:
            # Gaussian signal: x100 should be at the actual maximum
            x100 = x[ymax_idx]
        else:
            # Step/square signal: use traditional extrapolation from x0/x50
            x100 = x50 + (x50 - x0)

    return PulseFeatures(
        signal_shape=signal_shape,
        polarity=polarity,
        amplitude=amplitude,
        offset=get_range_mean_y(x, y * polarity, start_range),
        foot_duration=x0 - x[0],
        xstartmin=start_range[0],
        xstartmax=start_range[1],
        xendmin=end_range[0],
        xendmax=end_range[1],
        xplateaumin=None if plateau_range is None else plateau_range[0],
        xplateaumax=None if plateau_range is None else plateau_range[1],
        rise_time=rise_time,
        fall_time=fall_time,
        fwhm=fwhm_val,
        x0=x0,
        x50=x50,
        x100=x100,
    )
