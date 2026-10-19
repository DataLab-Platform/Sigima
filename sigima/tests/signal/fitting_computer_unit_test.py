# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the :class:`sigima.tools.signal.fitting.FitComputer` API.

These tests exercise the public ``FitComputer`` API surface without going
through high-level computation functions, focusing on the small classmethod
helpers, configuration paths, and the concrete fit computer classes
(linear, polynomial, gaussian, lorentzian, voigt, exponential,
two-half-gaussian and planckian).
"""

# pylint: disable=invalid-name

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.constants

from sigima.objects import PlanckParam, create_signal_from_param
from sigima.tools.signal import fitting as fit_mod

# ===========================================================================
# FitComputer base class — args/kwargs handling and validation
# ===========================================================================


def test_args_kwargs_to_list_with_args() -> None:
    """Positional fit-parameter values are passed straight through as a
    list (no name lookup needed)."""
    out = fit_mod.LinearFitComputer.args_kwargs_to_list(2.0, 3.0)
    assert out == [2.0, 3.0]


def test_args_kwargs_to_list_with_kwargs() -> None:
    """Keyword fit-parameter values are reordered to match the computer's
    declared ``PARAMS_NAMES`` order."""
    out = fit_mod.LinearFitComputer.args_kwargs_to_list(a=2.0, b=3.0)
    assert out == [2.0, 3.0]


def test_args_kwargs_to_list_mixed_raises() -> None:
    """Mixing positional and keyword fit parameters is ambiguous and
    rejected with ``ValueError``."""
    with pytest.raises(ValueError):
        fit_mod.LinearFitComputer.args_kwargs_to_list(2.0, b=3.0)


def test_args_kwargs_to_list_too_many_args() -> None:
    """Passing more positional values than the computer has parameters
    raises ``ValueError`` (no silent truncation)."""
    with pytest.raises(ValueError):
        fit_mod.LinearFitComputer.args_kwargs_to_list(1.0, 2.0, 3.0)


def test_args_kwargs_to_list_missing_kwarg() -> None:
    """Omitting a required keyword parameter (here ``b``) raises
    ``ValueError`` rather than producing a partially-initialised list."""
    with pytest.raises(ValueError):
        fit_mod.LinearFitComputer.args_kwargs_to_list(a=2.0)


def test_args_kwargs_to_list_no_params_no_kwargs_raises() -> None:
    """Computer subclass without PARAMS_NAMES requires kwargs."""

    class _Empty(fit_mod.FitComputer):  # pylint: disable=abstract-method
        """Subclass with an empty ``PARAMS_NAMES`` to exercise the
        ``args_kwargs_to_list`` validation path."""

        PARAMS_NAMES: tuple = ()

    with pytest.raises(ValueError):
        _Empty.args_kwargs_to_list()


def test_check_params_missing_raises() -> None:
    """``check_params`` enforces that all declared ``PARAMS_NAMES`` are
    provided; missing ones (here ``b``) raise ``ValueError``."""
    x = np.linspace(0.0, 1.0, 16)
    y = np.zeros_like(x)
    fit = fit_mod.LinearFitComputer(x, y)
    with pytest.raises(ValueError):
        fit.check_params(a=2.0)  # missing 'b'


def test_create_params_includes_fit_type_and_residual() -> None:
    """``create_params`` always emits a ``fit_type`` tag and a
    ``residual_rms`` value (here zero on perfect data)."""
    x = np.linspace(0.0, 1.0, 8)
    y = 2.0 * x + 1.0
    fit = fit_mod.LinearFitComputer(x, y)
    params = fit.create_params(y, a=2.0, b=1.0)
    assert params["fit_type"] == "linear"
    assert params["residual_rms"] == pytest.approx(0.0, abs=1e-9)


def test_polynomial_fit_invalid_degree_raises() -> None:
    """A polynomial fit of degree 0 is degenerate (constant) and rejected
    at construction time with ``ValueError``."""
    x = np.linspace(0.0, 1.0, 8)
    y = np.zeros_like(x)
    with pytest.raises(ValueError):
        fit_mod.PolynomialFitComputer(x, y, degree=0)


def test_polynomial_infer_param_names() -> None:
    """``infer_param_names_from_kwargs`` for polynomials accepts the
    canonical ``a``/``b``/``c``... letters and returns them in order."""
    names = fit_mod.PolynomialFitComputer.infer_param_names_from_kwargs(
        {"a": 1.0, "b": 2.0, "c": 3.0}
    )
    assert names == ("a", "b", "c")


def test_polynomial_infer_param_names_empty_raises() -> None:
    """``infer_param_names_from_kwargs`` rejects a kwargs dict that does
    not contain any of the known polynomial coefficient letters."""
    with pytest.raises(ValueError):
        fit_mod.PolynomialFitComputer.infer_param_names_from_kwargs({"unknown": 1.0})


# ===========================================================================
# Concrete fit computers — `evaluate` and `fit`
# ===========================================================================


def test_linear_fit_recovers_coefficients() -> None:
    """On noiseless data the linear fit recovers the exact slope and
    intercept and the fitted curve matches ``y`` element-wise."""
    x = np.linspace(0.0, 10.0, 64)
    y = 3.0 * x + 1.5
    fit = fit_mod.LinearFitComputer(x, y)
    fitted_y, params = fit.fit()
    assert params["a"] == pytest.approx(3.0, rel=1e-6)
    assert params["b"] == pytest.approx(1.5, rel=1e-6)
    assert fitted_y == pytest.approx(y, abs=1e-6)


def test_planckian_evaluate_with_negative_x_returns_baseline() -> None:
    """The Planckian model is undefined for ``x≤0`` and falls back to
    the baseline ``y0`` to avoid producing NaNs."""
    x = np.array([-1.0, -2.0, -3.0])
    out = fit_mod.PlanckianFitComputer.evaluate(x, amp=1.0, x0=500.0, sigma=1.0, y0=0.5)
    assert np.allclose(out, 0.5)


# ===========================================================================
# Multi-peak fit computers — base class infer/evaluate
# ===========================================================================


def test_multi_peak_infer_param_names_no_amplitude_raises() -> None:
    """Multi-peak ``infer_param_names_from_kwargs`` requires at least one
    ``amplitude_*`` key; a kwargs dict without any is rejected."""
    with pytest.raises(ValueError):
        fit_mod.MultiGaussianFitComputer.infer_param_names_from_kwargs({"y0": 0.0})


def test_multi_gaussian_evaluate_two_peaks() -> None:
    """``MultiGaussianFitComputer.evaluate`` produces a sum of Gaussians,
    each with its own height, width and center, plus a baseline."""
    x = np.linspace(-5.0, 5.0, 300)
    out = fit_mod.MultiGaussianFitComputer.evaluate(
        x,
        amplitude_1=1.0,
        sigma_1=0.5,
        x0_1=-2.0,
        amplitude_2=0.5,
        sigma_2=0.7,
        x0_2=2.0,
        y0=0.1,
    )
    assert out.shape == x.shape
    # Should be > baseline near the peaks
    assert out[np.argmin(np.abs(x + 2.0))] > 0.5
    assert out[np.argmin(np.abs(x - 2.0))] > 0.3


def test_multi_lorentzian_evaluate_one_peak() -> None:
    """``MultiLorentzianFitComputer.evaluate`` works for a single peak
    (boundary case: degenerate ``multi`` fit)."""
    x = np.linspace(-5.0, 5.0, 200)
    out = fit_mod.MultiLorentzianFitComputer.evaluate(
        x,
        amplitude_1=1.0,
        sigma_1=0.5,
        x0_1=0.0,
        y0=0.0,
    )
    assert out.shape == x.shape


@pytest.mark.parametrize(
    ("fit_type", "computer"),
    [
        ("multigaussian", fit_mod.MultiGaussianFitComputer),
        ("multilorentzian", fit_mod.MultiLorentzianFitComputer),
    ],
)
def test_multi_peak_fit_params_reproduce_model_on_arbitrary_axis(
    fit_type, computer
) -> None:
    """Canonical multi-peak metadata includes centers and reproduces the model."""
    values = {
        "amplitude_1": 2.0,
        "sigma_1": 0.6,
        "x0_1": -1.5,
        "amplitude_2": -0.75,
        "sigma_2": 0.9,
        "x0_2": 1.25,
        "y0": 0.2,
    }
    params = fit_mod.create_fit_params(
        fit_type, values, residual_rms=0.01, interactive=True
    )

    assert params["fit_params_version"] == fit_mod.FIT_PARAMS_VERSION
    assert params["peak_parameterization"] == fit_mod.PEAK_PARAMETERIZATION
    assert params["interactive"] is True
    assert params["x0_1"] == values["x0_1"]
    assert params["x0_2"] == values["x0_2"]
    for x in (np.linspace(-5.0, 5.0, 201), np.linspace(-8.0, 8.0, 321)):
        np.testing.assert_allclose(
            fit_mod.evaluate_fit(x, **params), computer.evaluate(x, **values)
        )


def test_multi_peak_fit_schema_rejects_missing_center() -> None:
    """Every multi-peak component requires its detected center coordinate."""
    params = fit_mod.create_fit_params(
        "multigaussian",
        {
            "amplitude_1": 2.0,
            "sigma_1": 0.6,
            "x0_1": -1.5,
            "amplitude_2": 0.75,
            "sigma_2": 0.9,
            "x0_2": 1.25,
            "y0": 0.2,
        },
    )
    params.pop("x0_2")

    with pytest.raises(ValueError, match="x0_2"):
        fit_mod.validate_fit_params(params)


# ===========================================================================
# Module-level helpers: evaluate_fit, infer_param_names
# ===========================================================================


def test_fitting_evaluate_fit_unknown_type_raises() -> None:
    """The module-level ``evaluate_fit`` dispatcher rejects unknown
    ``fit_type`` values rather than silently returning zeros."""
    with pytest.raises(ValueError):
        fit_mod.evaluate_fit(np.linspace(0, 1, 10), fit_type="not_a_real_fit")


@pytest.mark.parametrize(
    "updates",
    [
        {"fit_params_version": None},
        {"fit_params_version": fit_mod.FIT_PARAMS_VERSION + 1},
        {"peak_parameterization": "area"},
        {"amp": 1.0},
        {"amplitude": True},
        {"sigma": np.nan},
    ],
)
def test_peak_fit_schema_rejects_incoherent_metadata(updates: dict) -> None:
    """Peak fit evaluation rejects absent, future or contradictory schemas."""
    params = fit_mod.create_fit_params(
        "gaussian",
        {"amplitude": 1.0, "sigma": 0.5, "x0": 0.0, "y0": 0.1},
    )
    params.update(updates)
    with pytest.raises(ValueError):
        fit_mod.evaluate_fit(np.linspace(-1.0, 1.0, 10), **params)


def test_unversioned_legacy_peak_fit_schema_has_dedicated_error() -> None:
    """An unversioned area key is identified as historical, not malformed v2."""
    params = {"fit_type": "gaussian", "amp": 1.0, "sigma": 0.5, "x0": 0.0, "y0": 0.1}
    with pytest.raises(fit_mod.pulse.LegacyPeakParameterizationError):
        fit_mod.evaluate_fit(np.linspace(-1.0, 1.0, 10), **params)


def test_multi_peak_fit_schema_rejects_non_contiguous_indices() -> None:
    """Multi-peak parameter indices must form a contiguous sequence from one."""
    params = {
        "fit_type": "multigaussian",
        "fit_params_version": fit_mod.FIT_PARAMS_VERSION,
        "peak_parameterization": fit_mod.PEAK_PARAMETERIZATION,
        "amplitude_1": 1.0,
        "sigma_1": 0.5,
        "x0_1": -1.0,
        "amplitude_3": 0.5,
        "sigma_3": 0.7,
        "x0_3": 1.0,
        "y0": 0.1,
    }
    with pytest.raises(ValueError, match="missing=.*amplitude_2"):
        fit_mod.evaluate_fit(np.linspace(-2.0, 2.0, 20), **params)


@pytest.mark.parametrize(
    ("values", "metadata"),
    [
        ({"amplitude": True, "sigma": 0.5, "x0": 0.0, "y0": 0.0}, {}),
        (
            {"amplitude": 1.0, "sigma": 0.5, "x0": 0.0, "y0": 0.0},
            {"residual_rms": np.nan},
        ),
    ],
)
def test_create_fit_params_rejects_invalid_numbers(values, metadata) -> None:
    """The canonical builder validates values before float coercion."""
    with pytest.raises(ValueError):
        fit_mod.create_fit_params("gaussian", values, **metadata)


def test_fitting_infer_param_names_from_kwargs_default() -> None:
    """The base ``FitComputer.infer_param_names_from_kwargs`` simply
    returns the kwargs keys (in insertion order)."""
    names = fit_mod.FitComputer.infer_param_names_from_kwargs({"a": 1, "b": 2})
    assert names == ("a", "b")


def test_fitting_polynomial_fit_with_initial_params_dict() -> None:
    """Cover the explicit initial params override branch."""
    x = np.linspace(-3.0, 3.0, 60)
    y = 1.0 + 2.0 * x + 0.5 * x**2 + np.random.default_rng(0).normal(0, 0.05, x.size)
    fc = fit_mod.PolynomialFitComputer(x, y, degree=2)
    fitted, params = fc.fit()
    assert fitted.shape == x.shape
    assert "a" in params and "b" in params and "c" in params


def test_fitting_exponential_growth_initial_params() -> None:
    """Cover the growth branch of ExponentialFitComputer.compute_initial_params."""
    x = np.linspace(0, 5, 50)
    y = 1.0 - np.exp(-0.5 * x)  # rising
    fc = fit_mod.ExponentialFitComputer(x, y)
    p = fc.compute_initial_params()
    assert "a" in p and "b" in p and "y0" in p


# ===========================================================================
# Initial parameter estimates — transition models (CDF, sigmoid)
# ===========================================================================


def _erf_transition() -> tuple[np.ndarray, np.ndarray]:
    """Return an erf transition: amplitude 2, mu 0.5, sigma 1.2, baseline 1."""
    x = np.linspace(-5.0, 5.0, 201)
    y = fit_mod.CDFFitComputer.evaluate(x, 2.0, 0.5, 1.2, 1.0)
    return x, y


def _logistic_transition() -> tuple[np.ndarray, np.ndarray]:
    """Return a logistic transition: amplitude 4, k 2, x0 -1, offset 1."""
    x = np.linspace(-5.0, 5.0, 201)
    y = fit_mod.SigmoidFitComputer.evaluate(x, 4.0, 2.0, -1.0, 1.0)
    return x, y


@pytest.mark.parametrize(
    ("computer_cls", "data_func", "center_name"),
    [
        (fit_mod.CDFFitComputer, _erf_transition, "mu"),
        (fit_mod.SigmoidFitComputer, _logistic_transition, "x0"),
    ],
)
def test_transition_initial_params_lie_strictly_inside_bounds(
    computer_cls, data_func, center_name
) -> None:
    """Seeds must be usable starting points, not values sitting on a bound.

    ``(x_max + abs(x_min)) / 2`` used to pin the centre seed on the upper
    bound of its own admissible interval for any range symmetric about zero.
    """
    x, y = data_func()
    fc = computer_cls(x, y)
    initial = fc.compute_initial_params()
    bounds = fc.compute_bounds(**initial)
    for name, (low, high) in zip(computer_cls.PARAMS_NAMES, bounds):
        assert low < initial[name] < high, f"{name}={initial[name]} on bound"
    assert initial[center_name] == pytest.approx(0.0)


def test_cdf_initial_params_match_amplitude_and_baseline_scales() -> None:
    """``amplitude * erf(...)`` spans ``2 * amplitude`` peak-to-peak."""
    x, y = _erf_transition()
    initial = fit_mod.CDFFitComputer(x, y).compute_initial_params()
    assert initial["amplitude"] == pytest.approx(2.0, rel=0.05)
    assert initial["baseline"] == pytest.approx(1.0, rel=0.05)


@pytest.mark.parametrize(
    ("computer_cls", "data_func"),
    [
        (fit_mod.CDFFitComputer, _erf_transition),
        (fit_mod.SigmoidFitComputer, _logistic_transition),
    ],
)
def test_transition_fits_recover_descending_step(computer_cls, data_func) -> None:
    """A falling transition is as fittable as a rising one.

    The amplitude used to be bounded to positive values, which confined the
    optimiser to the ascending half-space.
    """
    x, y = data_func()
    y_down = y[::-1].copy()
    for data in (y, y_down):
        y_fitted, _params = computer_cls(x, data).fit()
        residual = np.sqrt(np.mean((data - y_fitted) ** 2))
        assert residual < 1e-3 * (data.max() - data.min())


# ===========================================================================
# Planckian model — documented degeneracy
# ===========================================================================


@pytest.mark.parametrize("k", [0.5, 2.0, 3.0])
def test_planckian_model_is_scale_degenerate(k: float) -> None:
    """``(amp, x0, sigma)`` are not separately identifiable.

    The model is strictly invariant under ``x0 -> k*x0``, ``sigma -> k*sigma``,
    ``amp -> amp/k**5``. This test pins the degeneracy down so that any future
    reparameterization is a deliberate decision, not an accident.
    """
    x = np.linspace(0.1, 10.0, 500)
    ref = fit_mod.PlanckianFitComputer.evaluate(x, 1.0, 2.0, 1.5, 0.3)
    alt = fit_mod.PlanckianFitComputer.evaluate(x, 1.0 / k**5, k * 2.0, k * 1.5, 0.3)
    assert np.allclose(alt, ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_planckian_peak_is_not_at_x0(sigma: float) -> None:
    """``x0`` is a scale parameter, not the peak abscissa.

    The maximum sits at approximately ``1.007 * x0 / sigma``; it coincides
    with ``x0`` only when ``sigma == 1``.
    """
    x = np.linspace(0.05, 20.0, 20001)
    y = fit_mod.PlanckianFitComputer.evaluate(x, 1.0, 1.0, sigma, 0.0)
    assert x[np.argmax(y)] == pytest.approx(1.007 / sigma, rel=0.01)


# ===========================================================================
# Planckian fit — physically scaled data
# ===========================================================================


def _blackbody(temperature: float, xmin: float, xmax: float) -> tuple:
    """Return a blackbody spectral radiance curve (wavelengths in meters)."""
    param = PlanckParam.create(T=temperature, xmin=xmin, xmax=xmax)
    signal = create_signal_from_param(param)
    return signal.x, signal.y


def test_planckian_initial_params_match_data_peak() -> None:
    """The initial guess must reproduce the data peak, not a 147× lower one.

    ``amp`` used to be seeded with the Y range, while the model with
    ``sigma == 1`` peaks at ``amp / (exp(5) - 1)``.
    """
    x, y = _blackbody(296.0, 1e-7, 1e-4)
    computer = fit_mod.PlanckianFitComputer(x, y)
    initial = computer.compute_initial_params()
    y_init = fit_mod.PlanckianFitComputer.evaluate(x, **initial)
    assert y_init.max() == pytest.approx(y.max(), rel=0.01)
    bounds = computer.compute_bounds(**initial)
    for name, (low, high) in zip(fit_mod.PlanckianFitComputer.PARAMS_NAMES, bounds):
        assert low <= initial[name] <= high, f"{name} outside its bounds"


@pytest.mark.parametrize(
    ("temperature", "xmin", "xmax"),
    [
        (296.0, 1e-7, 1e-4),  # DataLab's default blackbody signal
        (1000.0, 1e-7, 1e-4),
        (5800.0, 1e-7, 3e-6),
        (296.0, 1.2e-5, 1e-4),  # peak just outside the window
    ],
)
@pytest.mark.parametrize("x_scale", [1.0, 1e6, 1e9])  # m, µm, nm
def test_planckian_fit_recovers_blackbody(
    temperature: float, xmin: float, xmax: float, x_scale: float
) -> None:
    """The automatic fit converges on blackbody data whatever the units.

    It used to return its initial guess unchanged when the wavelength was in
    meters. The temperature and the emissivity are derived from the
    identifiable combinations ``sigma / x0`` and ``amp * x0**5``.
    """
    x, y = _blackbody(temperature, xmin, xmax)
    y_fitted, params = fit_mod.planckian_fit(x * x_scale, y)
    assert np.sqrt(np.mean((y - y_fitted) ** 2)) < 1e-6 * np.ptp(y)
    x0_m = params["x0"] / x_scale
    c2 = scipy.constants.h * scipy.constants.c / scipy.constants.k
    assert c2 * params["sigma"] / (5 * x0_m) == pytest.approx(temperature, rel=1e-6)
    emissivity = (
        params["amp"] * x0_m**5 / (2 * scipy.constants.h * scipy.constants.c**2)
    )
    assert emissivity == pytest.approx(1.0, rel=1e-6)


def test_planckian_fit_with_noise_and_baseline() -> None:
    """Noise and a baseline offset do not prevent convergence."""
    x, y = _blackbody(296.0, 1e-7, 1e-4)
    rng = np.random.default_rng(0)
    y_noisy = y + 0.05 * y.max() + rng.normal(0, 0.01 * y.max(), y.size)
    y_fitted, params = fit_mod.planckian_fit(x, y_noisy)
    assert y_fitted.max() == pytest.approx(y_noisy.max(), rel=0.03)
    assert np.sqrt(np.mean((y_noisy - y_fitted) ** 2)) < 0.015 * np.ptp(y_noisy)
    assert params["y0"] == pytest.approx(0.05 * y.max(), rel=0.2)


# ===========================================================================
# Exponential fit — X unit independence
# ===========================================================================


def test_exponential_fit_short_time_constant() -> None:
    """A 1 ms time constant with X in seconds is fitted (b = -1000).

    ``b`` used to be bounded to ``[-10, 10]``, which made the initial guess
    infeasible and the fit fall back to it.
    """
    x = np.linspace(0, 5e-3, 500)
    y = 2.0 * np.exp(-x / 1e-3) + 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y_fitted, params = fit_mod.exponential_fit(x, y)
    assert params["b"] == pytest.approx(-1000.0, rel=1e-6)
    assert params["a"] == pytest.approx(2.0, rel=1e-6)
    assert params["y0"] == pytest.approx(0.1, rel=1e-6)
    assert np.allclose(y_fitted, y)


def test_exponential_evaluate_clips_exponent_only() -> None:
    """``evaluate`` protects against overflow without altering large rates."""
    x = np.linspace(0, 1e-3, 11)
    y = fit_mod.ExponentialFitComputer.evaluate(x, 1.0, -1000.0, 0.0)
    assert np.allclose(y, np.exp(-1000.0 * x))
    y_big = fit_mod.ExponentialFitComputer.evaluate(np.array([1e6]), 1.0, 1.0, 0.0)
    assert np.isfinite(y_big).all()


# ===========================================================================
# Model formulas
# ===========================================================================


@pytest.mark.parametrize("fit_type", sorted(fit_mod.FIT_TYPE_MAPPING))
def test_every_fit_type_has_a_formula(fit_type: str) -> None:
    """Interactive fit dialogs display ``FORMULA``: none may be missing."""
    formula = fit_mod.FIT_TYPE_MAPPING[fit_type].FORMULA
    assert formula.startswith("y = ")


def test_polynomial_formula_depends_on_degree() -> None:
    """The polynomial formula lists one term per coefficient."""
    x = np.linspace(-1.0, 1.0, 10)
    computer = fit_mod.PolynomialFitComputer(x, x**3, degree=3)
    assert computer.get_formula() == "y = a x<sup>3</sup> + b x<sup>2</sup> + c x + d"
    linear = fit_mod.LinearFitComputer(x, x)
    assert linear.get_formula() == fit_mod.LinearFitComputer.FORMULA
