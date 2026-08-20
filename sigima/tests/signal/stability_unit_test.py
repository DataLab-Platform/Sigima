# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal stability analysis unit test.

The estimators are validated against *exact* analytical references rather than
statistical ones. Every variance estimator in :mod:`sigima.tools.signal.stability` is
a quadratic form ``Q(y) = yᵀ·A·y`` of the input samples, so for independent samples of
variance ``σ²`` its expectation is ``E[Q] = σ²·tr(A)``. The trace is obtained by summing
``Q`` over the canonical basis, which yields a deterministic reference completely free
of Monte-Carlo noise.

For white frequency noise sampled at ``dt``, with ``m = τ/dt``, the expectations are:

==============================  ==========================================
Estimator                       Expectation
==============================  ==========================================
Allan variance                  ``σ²/m``
Overlapping Allan variance      ``σ²/m``
Modified Allan variance         ``σ²·(m² + 1)/(2·m³)``
Hadamard variance               ``σ²/m``
Total variance                  ``σ²/m`` (up to an O(1/N) end-effect bias)
Time deviation                  ``σ·sqrt((m² + 1)/(6·m))``
==============================  ==========================================

For a linear frequency drift ``y = a·t``, the Allan and overlapping Allan variances are
``(a·τ)²/2``, whereas the Hadamard variance rejects the drift and vanishes.

References:
    IEEE Std 1139-2008, *Definitions of Physical Quantities for Fundamental Frequency
    and Time Metrology*; W. J. Riley, *Handbook of Frequency Stability Analysis*, NIST
    Special Publication 1065, 2008.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
from sigima.tests.helpers import check_array_result
from sigima.tools.signal import stability

#: Number of samples used by the exact-expectation references. Kept small on purpose:
#: the trace is accumulated by looping over the canonical basis, so the cost is O(N²).
N_POINTS = 384

#: Averaging factors (``m = τ/dt``) covered by the references, with ``dt = 1``.
TAUS = np.arange(1, 17, dtype=float)

#: Standard deviation of the synthetic white frequency noise.
SIGMA = 1.0


def exact_expectation(
    func: Callable[..., np.ndarray], n_points: int, tau_values: np.ndarray
) -> np.ndarray:
    """Return the exact expectation of a quadratic estimator for white noise.

    The estimator is a quadratic form of the samples, hence ``E[Q] = σ²·tr(A)`` for
    independent samples of variance ``σ²``. The trace is accumulated by evaluating the
    estimator on each canonical basis vector, which makes the reference deterministic.

    Args:
        func: Estimator taking ``(x, y, tau_values)`` and returning one value per tau
        n_points: Number of samples of the synthetic series
        tau_values: Averaging times, in the same unit as ``x``

    Returns:
        Expected estimator values for unit-variance white noise
    """
    x = np.arange(n_points, dtype=float)
    total = np.zeros(len(tau_values))
    unit = np.zeros(n_points)
    for index in range(n_points):
        unit[index] = 1.0
        total += func(x, unit, tau_values)
        unit[index] = 0.0
    return total


def white_noise_signal(
    n_points: int = N_POINTS, seed: int = 42
) -> sigima.objects.SignalObj:
    """Return a signal object holding white frequency noise.

    Args:
        n_points: Number of samples
        seed: Seed of the random generator, for reproducibility

    Returns:
        Signal object with unit sampling interval
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, SIGMA, n_points)
    return sigima.objects.create_signal(
        "White frequency noise", np.arange(n_points, dtype=float), values
    )


def drift_signal(
    slope: float = 0.01, n_points: int = N_POINTS
) -> sigima.objects.SignalObj:
    """Return a signal object holding a pure linear frequency drift.

    Args:
        slope: Drift slope, per sample
        n_points: Number of samples

    Returns:
        Signal object with unit sampling interval
    """
    time = np.arange(n_points, dtype=float)
    return sigima.objects.create_signal("Linear frequency drift", time, slope * time)


def compute(func: Callable, obj: sigima.objects.SignalObj) -> np.ndarray:
    """Run a stability computation function over :data:`TAUS`.

    Args:
        func: Computation function from :mod:`sigima.proc.signal`
        obj: Source signal object

    Returns:
        Estimator values, one per tau
    """
    param = sigima.params.AllanVarianceParam.create(max_tau=int(TAUS[-1]))
    result = func(obj, param)
    check_array_result("Tau axis", result.x, TAUS, verbose=False)
    return result.y


@pytest.mark.validation
def test_signal_allan_variance() -> None:
    """Validate the Allan variance against its analytical expectation."""
    check_array_result(
        "Allan variance, white noise",
        exact_expectation(stability.allan_variance, N_POINTS, TAUS),
        SIGMA**2 / TAUS,
    )
    # A pure linear frequency drift gives AVAR(τ) = (a·τ)²/2:
    slope = 0.01
    check_array_result(
        "Allan variance, linear drift",
        compute(sigima.proc.signal.allan_variance, drift_signal(slope)),
        (slope * TAUS) ** 2 / 2.0,
    )


@pytest.mark.validation
def test_signal_allan_deviation() -> None:
    """Validate the Allan deviation as the square root of the Allan variance."""
    obj = white_noise_signal()
    check_array_result(
        "Allan deviation is the square root of the Allan variance",
        compute(sigima.proc.signal.allan_deviation, obj),
        np.sqrt(compute(sigima.proc.signal.allan_variance, obj)),
    )
    slope = 0.01
    check_array_result(
        "Allan deviation, linear drift",
        compute(sigima.proc.signal.allan_deviation, drift_signal(slope)),
        slope * TAUS / np.sqrt(2.0),
    )


@pytest.mark.validation
def test_signal_overlapping_allan_variance() -> None:
    """Validate the overlapping Allan variance against its analytical expectation."""
    # The overlapping estimator shares the expectation of the Allan variance and only
    # improves its confidence: any residual factor depending on m would be a defect.
    check_array_result(
        "Overlapping Allan variance, white noise",
        exact_expectation(stability.overlapping_allan_variance, N_POINTS, TAUS),
        SIGMA**2 / TAUS,
    )
    slope = 0.01
    check_array_result(
        "Overlapping Allan variance, linear drift",
        compute(sigima.proc.signal.overlapping_allan_variance, drift_signal(slope)),
        (slope * TAUS) ** 2 / 2.0,
    )


@pytest.mark.validation
def test_signal_modified_allan_variance() -> None:
    """Validate the modified Allan variance against its analytical expectation."""
    mvar = exact_expectation(stability.modified_allan_variance, N_POINTS, TAUS)
    check_array_result(
        "Modified Allan variance, white noise",
        mvar,
        SIGMA**2 * (TAUS**2 + 1.0) / (2.0 * TAUS**3),
    )
    # For white frequency noise the MVAR/AVAR ratio tends to 1/2, which is the property
    # making MVAR able to discriminate white from flicker phase modulation:
    ratio = mvar / exact_expectation(stability.allan_variance, N_POINTS, TAUS)
    assert ratio[0] == pytest.approx(1.0), "MVAR must equal AVAR at τ = dt"
    assert ratio[-1] == pytest.approx(0.5, abs=0.01), "MVAR/AVAR must tend to 1/2"
    # The computation function must reproduce the underlying estimator:
    obj = white_noise_signal()
    check_array_result(
        "Modified Allan variance, computation function",
        compute(sigima.proc.signal.modified_allan_variance, obj),
        stability.modified_allan_variance(obj.x, obj.y, TAUS),
    )


@pytest.mark.validation
def test_signal_hadamard_variance() -> None:
    """Validate the Hadamard variance against its analytical expectation."""
    check_array_result(
        "Hadamard variance, white noise",
        exact_expectation(stability.hadamard_variance, N_POINTS, TAUS),
        SIGMA**2 / TAUS,
    )
    # The defining property of the Hadamard variance is its immunity to linear frequency
    # drift, to which the Allan variance is sensitive:
    drift = drift_signal(0.01)
    hvar = compute(sigima.proc.signal.hadamard_variance, drift)
    avar = compute(sigima.proc.signal.allan_variance, drift)
    assert np.all(np.abs(hvar) < 1e-12 * avar[-1]), (
        f"Hadamard variance must reject linear frequency drift, got {hvar}"
    )


@pytest.mark.validation
def test_signal_total_variance() -> None:
    """Validate the total variance against its analytical expectation."""
    # TOTVAR estimates the same quantity as AVAR; the reflected extension of the phase
    # data leaves an end effect of the order of 1/N, hence the relaxed tolerance.
    check_array_result(
        "Total variance, white noise",
        exact_expectation(stability.total_variance, N_POINTS, TAUS),
        SIGMA**2 / TAUS,
        rtol=0.01,
    )
    # TOTVAR must not degenerate into the plain Allan variance: it uses every sample at
    # every tau, so the two estimators differ on any finite record.
    obj = white_noise_signal()
    totvar = compute(sigima.proc.signal.total_variance, obj)
    avar = compute(sigima.proc.signal.allan_variance, obj)
    assert not np.allclose(totvar[1:], avar[1:]), (
        "Total variance must not be a duplicate of the Allan variance"
    )


@pytest.mark.validation
def test_signal_time_deviation() -> None:
    """Validate the time deviation against its analytical expectation."""
    obj = white_noise_signal()
    # TDEV is defined from the *modified* Allan variance: TDEV(τ) = τ·MDEV(τ)/√3
    mvar = compute(sigima.proc.signal.modified_allan_variance, obj)
    check_array_result(
        "Time deviation is τ·MDEV/√3",
        compute(sigima.proc.signal.time_deviation, obj),
        TAUS * np.sqrt(mvar) / np.sqrt(3.0),
    )
    # Expectation for white frequency noise, derived from the MVAR expectation:
    tdev = (
        TAUS
        * np.sqrt(exact_expectation(stability.modified_allan_variance, N_POINTS, TAUS))
        / np.sqrt(3.0)
    )
    check_array_result(
        "Time deviation, white noise",
        tdev,
        SIGMA * np.sqrt((TAUS**2 + 1.0) / (6.0 * TAUS)),
    )


def test_stability_shortest_tau_is_defined() -> None:
    """Check that every estimator is defined at the shortest averaging time.

    The estimators used to return NaN for τ = dt, although all of them are perfectly
    defined there — the modified Allan variance even coincides with the Allan variance
    at that point.
    """
    obj = white_noise_signal()
    for func in (
        sigima.proc.signal.allan_variance,
        sigima.proc.signal.allan_deviation,
        sigima.proc.signal.overlapping_allan_variance,
        sigima.proc.signal.modified_allan_variance,
        sigima.proc.signal.hadamard_variance,
        sigima.proc.signal.total_variance,
        sigima.proc.signal.time_deviation,
    ):
        values = compute(func, obj)
        assert np.isfinite(values[0]), f"{func.__name__} is not defined at τ = dt"
        assert values[0] > 0, f"{func.__name__} must be positive at τ = dt"


def test_stability_rejects_unevenly_spaced_time() -> None:
    """Check that unevenly spaced time values are rejected."""
    x = np.array([0.0, 1.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    for func in (
        stability.allan_variance,
        stability.overlapping_allan_variance,
        stability.modified_allan_variance,
        stability.hadamard_variance,
        stability.total_variance,
        stability.time_deviation,
    ):
        with pytest.raises(ValueError, match="evenly spaced"):
            func(x, y, np.array([1.0]))


if __name__ == "__main__":
    test_signal_allan_variance()
    test_signal_allan_deviation()
    test_signal_overlapping_allan_variance()
    test_signal_modified_allan_variance()
    test_signal_hadamard_variance()
    test_signal_total_variance()
    test_signal_time_deviation()
    test_stability_shortest_tau_is_defined()
    test_stability_rejects_unevenly_spaced_time()
