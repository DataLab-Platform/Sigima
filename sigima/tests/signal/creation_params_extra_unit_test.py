# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for parametric signal creation classes
in :mod:`sigima.objects.signal.creation`.

Covers ``get_expected_features`` and ``get_crossing_time`` methods of
``LorentzParam``, ``VoigtParam``, ``GaussParam``, ``SquarePulseParam`` and
the shared ``BaseGaussLorentzVoigtParam`` validation paths.
"""

# pylint: disable=invalid-name
# pylint: disable=protected-access

from __future__ import annotations

import json

import numpy as np
import pytest
from guidata.dataset import dataset_to_json

from sigima.objects import GaussParam, LorentzParam, SquarePulseParam, VoigtParam
from sigima.objects.signal.creation import (
    CREATION_PARAMS_VERSION,
    convert_legacy_peak_creation_params,
    validate_peak_creation_params,
)
from sigima.tools.signal.pulse import (
    PEAK_PARAMETERIZATION,
    GaussianModel,
    LegacyPeakParameterizationError,
    LorentzianModel,
    VoigtModel,
)

# ===========================================================================
# LorentzParam / VoigtParam.get_expected_features
# ===========================================================================


def test_lorentz_get_expected_features() -> None:
    """For a Lorentzian, the FWHM is exactly ``2*sigma`` and ``rise_time``
    follows the closed-form ``2*sigma*sqrt(1/start - 1/stop)``; positive
    amplitude maps to polarity ``+1``."""
    p = LorentzParam()
    p.amplitude = 1.0
    p.sigma = 0.5
    p.mu = 0.0
    p.y0 = 0.0
    feats = p.get_expected_features(start_ratio=0.1, stop_ratio=0.9)
    # FWHM for Lorentzian: 2 * sigma
    assert feats.fwhm == pytest.approx(2 * 0.5)
    expected_rt = 0.5 * (np.sqrt(1 / 0.1 - 1) - np.sqrt(1 / 0.9 - 1))
    assert feats.rise_time == pytest.approx(expected_rt)
    assert feats.polarity == 1
    assert feats.amplitude == pytest.approx(1.0)


def test_voigt_get_expected_features() -> None:
    """Voigt features follow the normalized model and preserve peak height."""
    p = VoigtParam()
    p.amplitude = -1.0
    p.sigma = 0.3
    p.mu = 1.0
    p.y0 = 0.5
    feats = p.get_expected_features()
    start_offset = VoigtModel.relative_level_offset(0.3, 0.1)
    stop_offset = VoigtModel.relative_level_offset(0.3, 0.9)
    assert feats.rise_time == pytest.approx(start_offset - stop_offset)
    assert feats.fwhm == pytest.approx(2 * VoigtModel.relative_level_offset(0.3, 0.5))
    assert feats.polarity == -1
    assert feats.amplitude == pytest.approx(1.0)
    assert feats.offset == pytest.approx(0.5)


# ===========================================================================
# GaussParam validation in get_expected_features / get_crossing_time
# ===========================================================================


def test_gauss_get_crossing_time_invalid_edge() -> None:
    """``get_crossing_time`` rejects edge names other than ``rise``/``fall``
    and the rising edge time is necessarily before ``mu`` while the falling
    edge is after."""
    p = GaussParam()
    p.amplitude = 1.0
    p.sigma = 0.5
    p.mu = 0.0
    with pytest.raises(ValueError, match="rise.*fall"):
        p.get_crossing_time("invalid", 0.5)
    rise = p.get_crossing_time("rise", 0.5)
    fall = p.get_crossing_time("fall", 0.5)
    assert rise < 0 < fall


@pytest.mark.parametrize(
    ("param_class", "model"),
    [
        (GaussParam, GaussianModel),
        (LorentzParam, LorentzianModel),
        (VoigtParam, VoigtModel),
    ],
)
def test_peak_creation_height_schema(param_class, model) -> None:
    """Peak creators serialize and evaluate signed height parameters."""
    p = param_class.create(
        size=1001,
        xmin=-5.0,
        xmax=5.0,
        amplitude=-2.5,
        sigma=0.7,
        mu=0.0,
        y0=0.75,
    )
    x, y = p.generate_1d_data()
    center = np.argmin(np.abs(x - p.mu))
    assert y[center] == pytest.approx(p.y0 + p.amplitude)
    assert p.get_expected_features().amplitude == pytest.approx(abs(p.amplitude))
    assert p.get_expected_features().fwhm == pytest.approx(
        2 * model.relative_level_offset(p.sigma, 0.5)
    )

    payload = json.loads(dataset_to_json(p))
    assert payload["amplitude"] == pytest.approx(p.amplitude)
    assert "a" not in payload
    assert payload["creation_params_version"] == CREATION_PARAMS_VERSION
    assert payload["peak_parameterization"] == PEAK_PARAMETERIZATION
    validate_peak_creation_params(payload)


@pytest.mark.parametrize(
    ("param_class", "model"),
    [
        (GaussParam, GaussianModel),
        (LorentzParam, LorentzianModel),
        (VoigtParam, VoigtModel),
    ],
)
def test_legacy_peak_creation_params_conversion(param_class, model) -> None:
    """Legacy areas require explicit, non-mutating conversion."""
    x = np.linspace(-5.0, 5.0, 200)
    amplitude, sigma, mu, y0 = -2.5, 0.7, 0.3, 0.75
    legacy = {
        "class_module": param_class.__module__,
        "class_name": param_class.__name__,
        "a": model.area_from_amplitude(amplitude, sigma),
        "sigma": sigma,
        "mu": mu,
        "y0": y0,
        "extension": "preserved",
    }
    original = legacy.copy()

    with pytest.raises(
        LegacyPeakParameterizationError,
        match="convert_legacy_peak_creation_params",
    ):
        validate_peak_creation_params(legacy)

    converted = convert_legacy_peak_creation_params(legacy)
    assert legacy == original
    assert converted["extension"] == "preserved"
    assert converted["amplitude"] == pytest.approx(amplitude)
    validate_peak_creation_params(converted)
    expected = model.evaluate(x, amplitude, sigma, mu, y0)
    actual = model.evaluate(x, converted["amplitude"], sigma, mu, y0)
    np.testing.assert_allclose(actual, expected)


def test_legacy_peak_creation_python_api_is_rejected() -> None:
    """The removed ``a`` API never aliases an area to a peak height."""
    with pytest.raises(LegacyPeakParameterizationError, match="integrated area"):
        GaussParam.create(a=1.0)

    param = GaussParam()
    with pytest.raises(LegacyPeakParameterizationError, match="integrated area"):
        param.a = 1.0
    with pytest.raises(LegacyPeakParameterizationError, match="integrated area"):
        _ = param.a


def test_legacy_peak_creation_conversion_rejects_mixed_keys() -> None:
    """Explicit conversion never overwrites an existing height silently."""
    payload = {
        "class_module": GaussParam.__module__,
        "class_name": GaussParam.__name__,
        "a": 1.0,
        "amplitude": 2.0,
        "sigma": 1.0,
        "mu": 0.0,
        "y0": 0.0,
    }

    with pytest.raises(ValueError, match="mix"):
        convert_legacy_peak_creation_params(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"class_module": "other.module"},
        {"mu": None},
        {"y0": None},
        {"amplitude": True},
        {"sigma": np.nan},
    ],
)
def test_peak_creation_schema_rejects_identity_and_missing_values(change) -> None:
    """Raw peak payload validation runs before guidata can supply defaults."""
    payload = json.loads(dataset_to_json(GaussParam.create(amplitude=1.0)))
    payload.update(change)

    with pytest.raises(ValueError):
        validate_peak_creation_params(payload)


# ===========================================================================
# SquarePulseParam.get_crossing_time
# ===========================================================================


def test_square_pulse_get_crossing_time_branches() -> None:
    """Pulse model rejects unknown edge names, but for the supported
    ``rise`` / ``fall`` the falling crossing time is always after the
    rising one for a normal pulse with default parameters."""
    p = SquarePulseParam()
    # Use defaults; ensure rise/fall yield distinct increasing times.
    t_rise = p.get_crossing_time("rise", 0.5)
    t_fall = p.get_crossing_time("fall", 0.5)
    assert t_fall > t_rise
    with pytest.raises(ValueError, match="rise.*fall"):
        p.get_crossing_time("middle", 0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
