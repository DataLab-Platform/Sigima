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

import numpy as np
import pytest

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
