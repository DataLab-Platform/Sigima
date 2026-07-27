# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Stability Analysis (see parent package :mod:`sigima.algorithms.signal`)

This module implements the classical frequency-stability estimators. All of them take
*fractional frequency* samples ``y`` regularly sampled at ``dt = x[1] - x[0]``, and an
array of averaging times ``tau``. The averaging factor is ``m = round(tau / dt)``.

Estimators either average adjacent intervals (Allan family) or reject a linear
frequency drift (Hadamard). For white frequency noise of standard deviation ``σ``, the
expectations are ``σ²/m`` for the Allan, overlapping Allan, Hadamard and total
variances, and ``σ²·(m² + 1)/(2·m³)`` for the modified Allan variance.

References:
    IEEE Std 1139-2008, *Definitions of Physical Quantities for Fundamental Frequency
    and Time Metrology — Random Instabilities*.

    W. J. Riley, *Handbook of Frequency Stability Analysis*, NIST Special Publication
    1065, 2008.
"""

from __future__ import annotations

import numpy as np

from sigima.tools.checks import check_1d_arrays


def _averaging_factor(tau: float, dt: float) -> int:
    """Return the number of samples averaged over the duration ``tau``.

    Args:
        tau: Averaging time
        dt: Sampling interval

    Returns:
        Averaging factor ``m = tau / dt``, rounded to the nearest integer

    Raises:
        ValueError: If ``tau`` is shorter than the sampling interval
    """
    m = int(round(tau / dt))
    if m < 1:
        raise ValueError(f"Tau value {tau} is smaller than the sampling interval {dt}")
    return m


def _running_mean(y: np.ndarray, m: int) -> np.ndarray:
    """Return the overlapping averages of ``m`` consecutive samples.

    Args:
        y: Input values
        m: Number of samples per average

    Returns:
        Array of ``len(y) - m + 1`` overlapping averages
    """
    csum = np.concatenate(([0.0], np.cumsum(y)))
    return (csum[m:] - csum[:-m]) / m


def _centered(y: np.ndarray) -> np.ndarray:
    """Return the input values with their mean removed.

    All the estimators of this module are built from differences of averages, which are
    invariant under a constant frequency offset. Removing the mean beforehand therefore
    leaves the results unchanged while avoiding the loss of significance that a large
    offset would cause in the cumulative sums.

    Args:
        y: Input values

    Returns:
        Centered values, as a float array
    """
    values = np.asarray(y, dtype=float)
    return values - np.mean(values)


@check_1d_arrays(x_evenly_spaced=True)
def allan_variance(x: np.ndarray, y: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    """
    Calculate the Allan variance for given time and measurement values at specified
    tau values.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Allan deviation time values

    Returns:
        Allan variance values
    """
    dt = np.mean(np.diff(x))  # Sampling interval

    allan_var = []
    for tau in tau_values:
        m = _averaging_factor(tau, dt)
        if m > len(y) // 2:
            # Tau too large for reliable statistics
            allan_var.append(np.nan)
            continue

        # Calculate the clusters/bins
        clusters = y[: len(y) - (len(y) % m)].reshape(-1, m)
        bin_means = clusters.mean(axis=1)

        # Calculate Allan variance using the definition
        # σ²(τ) = 1/(2(N-1)) Σ(y_(i+1) - y_i)²
        # where y_i are the bin means
        squared_diff = np.sum(np.diff(bin_means) ** 2)
        n = len(bin_means) - 1

        if n > 0:
            var = squared_diff / (2.0 * n)
            allan_var.append(var)
        else:
            allan_var.append(np.nan)

    return np.array(allan_var)


@check_1d_arrays(x_evenly_spaced=True)
def allan_deviation(x: np.ndarray, y: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    """
    Calculate the Allan deviation for given time and measurement values at specified
    tau values.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Allan deviation time values

    Returns:
        Allan deviation values
    """
    return np.sqrt(allan_variance(x, y, tau_values))


@check_1d_arrays(x_evenly_spaced=True)
def overlapping_allan_variance(
    x: np.ndarray, y: np.ndarray, tau_values: np.ndarray
) -> np.ndarray:
    """
    Calculate the Overlapping Allan variance for given time and measurement values.

    The overlapping estimator uses every possible pair of adjacent averaging intervals
    instead of the contiguous ones only:

    .. math::

        \\sigma^2_y(\\tau) = \\frac{1}{2 (N - 2m + 1)}
        \\sum_{j} \\left( \\bar{y}_{j+m} - \\bar{y}_j \\right)^2

    where :math:`\\bar{y}_j` is the average of ``m`` consecutive samples starting at
    index ``j`` and :math:`m = \\tau / dt`. It estimates the same quantity as the Allan
    variance, with a better confidence.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Allan deviation time values

    Returns:
        Overlapping Allan variance values
    """
    dt = np.mean(np.diff(x))  # Sampling interval
    values = _centered(y)

    overlapping_var = []
    for tau in tau_values:
        m = _averaging_factor(tau, dt)
        if m > len(values) // 2:
            # Averaging time too long for reliable statistics
            overlapping_var.append(np.nan)
            continue

        avg = _running_mean(values, m)
        # Differences between *adjacent averaging intervals*, hence the stride m:
        diff = avg[m:] - avg[:-m]
        overlapping_var.append(np.mean(diff**2) / 2.0)

    return np.array(overlapping_var)


@check_1d_arrays(x_evenly_spaced=True)
def modified_allan_variance(
    x: np.ndarray, y: np.ndarray, tau_values: np.ndarray
) -> np.ndarray:
    """
    Calculate the Modified Allan variance for given time and measurement values.

    The modified Allan variance adds an averaging of ``m`` consecutive first differences
    to the overlapping Allan variance:

    .. math::

        \\mathrm{Mod}\\,\\sigma^2_y(\\tau) = \\frac{1}{2 M} \\sum_{j}
        \\left( \\frac{1}{m} \\sum_{i=j}^{j+m-1}
        \\left( \\bar{y}_{i+m} - \\bar{y}_i \\right) \\right)^2

    This extra averaging makes the estimator able to discriminate white from flicker
    phase modulation, which the Allan variance cannot. It coincides with the Allan
    variance at :math:`\\tau = dt`.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Modified Allan deviation time values

    Returns:
        Modified Allan variance values
    """
    dt = np.mean(np.diff(x))  # Sampling interval
    values = _centered(y)

    mod_allan_var = []
    for tau in tau_values:
        m = _averaging_factor(tau, dt)
        if m > len(values) // 3:
            # Averaging time too long for reliable statistics
            mod_allan_var.append(np.nan)
            continue

        avg = _running_mean(values, m)
        diff = avg[m:] - avg[:-m]
        # The phase-averaging step specific to the modified Allan variance:
        smoothed = _running_mean(diff, m)
        mod_allan_var.append(np.mean(smoothed**2) / 2.0)

    return np.array(mod_allan_var)


@check_1d_arrays(x_evenly_spaced=True)
def hadamard_variance(
    x: np.ndarray, y: np.ndarray, tau_values: np.ndarray
) -> np.ndarray:
    """
    Calculate the Hadamard variance for given time and measurement values.

    The Hadamard variance is built on *second* differences of adjacent averaging
    intervals:

    .. math::

        H\\sigma^2_y(\\tau) = \\frac{1}{6 (N - 3m + 1)} \\sum_{j}
        \\left( \\bar{y}_{j+2m} - 2 \\bar{y}_{j+m} + \\bar{y}_j \\right)^2

    Second differences cancel any linear frequency drift, which makes this estimator
    the usual choice for oscillators exhibiting ageing. For white frequency noise it
    has the same expectation as the Allan variance.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Hadamard deviation time values

    Returns:
        Hadamard variance values
    """
    dt = np.mean(np.diff(x))  # Sampling interval
    values = _centered(y)

    hadamard_var = []
    for tau in tau_values:
        m = _averaging_factor(tau, dt)
        if m > len(values) // 3:
            # Averaging time too long for reliable statistics
            hadamard_var.append(np.nan)
            continue

        avg = _running_mean(values, m)
        # Second differences between *adjacent averaging intervals*, hence the stride m:
        diff = avg[2 * m :] - 2.0 * avg[m:-m] + avg[: -2 * m]
        hadamard_var.append(np.mean(diff**2) / 6.0)

    return np.array(hadamard_var)


@check_1d_arrays(x_evenly_spaced=True)
def total_variance(x: np.ndarray, y: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    """
    Calculate the Total variance for given time and measurement values.

    The total variance estimates the same quantity as the Allan variance, but computes
    it on the phase data extended by reflection about both ends of the record:

    .. math::

        \\mathrm{Tot}\\,\\sigma^2_y(\\tau) = \\frac{1}{2 \\tau^2 (N - 2)}
        \\sum_{i} \\left( x^*_{i-m} - 2 x^*_i + x^*_{i+m} \\right)^2

    where :math:`x^*` is the extended phase sequence. Because every averaging time uses
    the whole record, the confidence at long tau is much better than that of the Allan
    variance, at the cost of a small end effect of the order of :math:`1/N`.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Total variance time values

    Returns:
        Total variance values
    """
    dt = np.mean(np.diff(x))  # Sampling interval
    values = _centered(y)

    # Phase data, then its doubly-reflected extension:
    phase = np.concatenate(([0.0], np.cumsum(values) * dt))
    size = phase.size
    mirror = phase[1 : size - 1][::-1]
    extended = np.concatenate(
        (2.0 * phase[0] - mirror, phase, 2.0 * phase[-1] - mirror)
    )
    offset = size - 2
    index = np.arange(1, size - 1)

    total_var = []
    for tau in tau_values:
        m = _averaging_factor(tau, dt)
        if m > len(values) // 2:
            # Averaging time too long for reliable statistics
            total_var.append(np.nan)
            continue

        second_diff = (
            extended[offset + index - m]
            - 2.0 * extended[offset + index]
            + extended[offset + index + m]
        )
        total_var.append(np.mean(second_diff**2) / (2.0 * (m * dt) ** 2))

    return np.array(total_var)


@check_1d_arrays(x_evenly_spaced=True)
def time_deviation(x: np.ndarray, y: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    """
    Calculate the Time Deviation (TDEV) for given time and measurement values.

    The time deviation is derived from the *modified* Allan deviation:

    .. math::

        \\mathrm{TDEV}(\\tau) =
        \\frac{\\tau}{\\sqrt{3}} \\, \\mathrm{Mod}\\,\\sigma_y(\\tau)

    Args:
        x: Time array
        y: Measured values array
        tau_values: Time deviation time values

    Returns:
        Time deviation values
    """
    taus = np.asarray(tau_values, dtype=float)
    mod_allan_var = modified_allan_variance(x, y, taus)
    return np.sqrt(mod_allan_var) * taus / np.sqrt(3.0)
