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

import numpy as np
import pytest

from sigima.objects import GaussParam, LorentzParam, SquarePulseParam, VoigtParam

# ===========================================================================
# LorentzParam / VoigtParam.get_expected_features
# ===========================================================================


def test_lorentz_get_expected_features() -> None:
    """For a Lorentzian, the FWHM is exactly ``2*sigma`` and ``rise_time``
    follows the closed-form ``2*sigma*sqrt(1/start - 1/stop)``; positive
    amplitude maps to polarity ``+1``."""
    p = LorentzParam()
    p.a = 1.0
    p.sigma = 0.5
    p.mu = 0.0
    p.y0 = 0.0
    feats = p.get_expected_features(start_ratio=0.1, stop_ratio=0.9)
    # FWHM for Lorentzian: 2 * sigma
    assert feats.fwhm == pytest.approx(2 * 0.5)
    # rise_time uses sqrt(1/0.1 - 1/0.9)
    expected_rt = 2 * 0.5 * np.sqrt(1 / 0.1 - 1 / 0.9)
    assert feats.rise_time == pytest.approx(expected_rt)
    assert feats.polarity == 1


def test_voigt_get_expected_features() -> None:
    """Voigt features use the Gaussian approximation (``2.563*sigma`` for
    rise time, ``2.355*sigma`` for FWHM) and propagate the offset; a
    negative amplitude must yield polarity ``-1``."""
    p = VoigtParam()
    p.a = -1.0  # Negative amplitude => polarity -1
    p.sigma = 0.3
    p.mu = 1.0
    p.y0 = 0.5
    feats = p.get_expected_features()
    # Voigt approximated as Gaussian: rise_time = 2.563 * sigma
    assert feats.rise_time == pytest.approx(2.563 * 0.3)
    assert feats.fwhm == pytest.approx(2.355 * 0.3)
    assert feats.polarity == -1
    assert feats.offset == pytest.approx(0.5)


# ===========================================================================
# GaussParam validation in get_expected_features / get_crossing_time
# ===========================================================================


def test_gauss_get_crossing_time_invalid_edge() -> None:
    """``get_crossing_time`` rejects edge names other than ``rise``/``fall``
    and the rising edge time is necessarily before ``mu`` while the falling
    edge is after."""
    p = GaussParam()
    p.a = 1.0
    p.sigma = 0.5
    p.mu = 0.0
    with pytest.raises(ValueError, match="rise.*fall"):
        p.get_crossing_time("invalid", 0.5)
    rise = p.get_crossing_time("rise", 0.5)
    fall = p.get_crossing_time("fall", 0.5)
    assert rise < 0 < fall


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
