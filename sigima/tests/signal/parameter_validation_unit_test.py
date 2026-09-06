# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for signal processing parameter validation."""

from __future__ import annotations

import numpy as np
import pytest
from guidata.config import ValidationMode, temporary_validation_mode

from sigima.enums import FrequencyFilterMethod, WindowingMethod
from sigima.objects import create_signal
from sigima.proc.base import HistogramParam
from sigima.proc.signal.analysis import (
    PulseFeaturesParam,
    extract_pulse_features,
    histogram,
)
from sigima.proc.signal.features import DynamicParam, FWHMParam, PeakDetectionParam
from sigima.proc.signal.filtering import (
    BandPassFilterParam,
    LowPassFilterParam,
    lowpass,
)
from sigima.proc.signal.processing import (
    Resampling1DParam,
    WindowingParam,
    apply_window,
    resampling,
)
from sigima.validation import validate_dataset


def create_test_signal():
    """Create a uniformly sampled signal with a 50 Hz Nyquist frequency."""
    x = np.linspace(0.0, 1.0, 101)
    return create_signal("test", x=x, y=np.sin(2.0 * np.pi * x))


def test_histogram_range_validation() -> None:
    """Equal histogram limits are valid, while reversed limits are rejected."""
    src = create_test_signal()
    result = histogram(src, HistogramParam.create(bins=4, lower=0.0, upper=0.0))
    assert result.x.size == 4

    with pytest.raises(ValueError, match="lower must be less"):
        histogram(src, HistogramParam.create(lower=1.0, upper=0.0))


@pytest.mark.parametrize(
    ("xmin", "xmax", "nbpts"),
    [(0.0, 1.0, 5), (1.0, 0.0, 5), (0.5, 0.5, 3), (0.5, 0.5, 1)],
)
def test_resampling_1d_nbpts_preserves_supported_domains(
    xmin: float, xmax: float, nbpts: int
) -> None:
    """Point-count mode supports ascending, descending, and repeated X."""
    src = create_test_signal()
    src.y = 2.0 * src.x + 1.0
    param = Resampling1DParam.create(
        mode="nbpts", xmin=xmin, xmax=xmax, nbpts=nbpts, dx=0.0
    )

    result = resampling(src, param)
    expected_x = np.linspace(xmin, xmax, nbpts)

    np.testing.assert_allclose(result.x, expected_x)
    np.testing.assert_allclose(result.y, 2.0 * expected_x + 1.0)
    assert param.dx == 0.0


@pytest.mark.parametrize(
    ("xmin", "xmax", "dx"),
    [(0.0, 1.0, 0.25), (1.0, 0.0, -0.25), (0.5, 0.5, 0.1), (0.5, 0.5, -0.1)],
)
def test_resampling_1d_dx_supports_signed_and_point_domains(
    xmin: float, xmax: float, dx: float
) -> None:
    """Step mode supports both orientations and point-like domains."""
    src = create_test_signal()
    src.y = 2.0 * src.x + 1.0
    param = Resampling1DParam.create(mode="dx", xmin=xmin, xmax=xmax, dx=dx, nbpts=0)

    result = resampling(src, param)
    expected_x = np.arange(xmin, xmax + dx / 2, dx)

    np.testing.assert_allclose(result.x, expected_x)
    np.testing.assert_allclose(result.y, 2.0 * expected_x + 1.0)
    assert param.nbpts == 0


def test_resampling_1d_expanded_call_supports_descending_domain() -> None:
    """Expanded arguments use the same mode-aware validation contract."""
    src = create_test_signal()
    src.y = 2.0 * src.x + 1.0

    result = resampling(src, mode="nbpts", xmin=1.0, xmax=0.0, nbpts=5, dx=0.0)

    expected_x = np.linspace(1.0, 0.0, 5)
    np.testing.assert_allclose(result.x, expected_x)
    np.testing.assert_allclose(result.y, 2.0 * expected_x + 1.0)


def test_resampling_1d_rejects_only_invalid_active_fields() -> None:
    """Mode changes preserve inactive values and validate them only when active."""
    src = create_test_signal()

    with pytest.raises(ValueError, match="xmin and xmax"):
        resampling(src, Resampling1DParam.create(mode="nbpts", nbpts=10))
    with pytest.raises(ValueError, match="dx must be non-zero"):
        resampling(
            src,
            Resampling1DParam.create(mode="dx", xmin=0.0, xmax=1.0, dx=0.0, nbpts=10),
        )
    for xmin, xmax, dx in ((0.0, 1.0, -0.1), (1.0, 0.0, 0.1)):
        with pytest.raises(ValueError, match="dx sign must match"):
            resampling(
                src,
                Resampling1DParam.create(
                    mode="dx", xmin=xmin, xmax=xmax, dx=dx, nbpts=10
                ),
            )

    point_count_param = Resampling1DParam.create(
        mode="nbpts", xmin=0.0, xmax=1.0, nbpts=3, dx=0.0
    )
    resampling(src, point_count_param)
    point_count_param.mode = "dx"
    with pytest.raises(ValueError, match="dx must be non-zero"):
        resampling(src, point_count_param)
    assert point_count_param.dx == 0.0

    step_param = Resampling1DParam.create(
        mode="dx", xmin=0.0, xmax=1.0, dx=0.5, nbpts=0
    )
    resampling(src, step_param)
    step_param.mode = "nbpts"
    with pytest.raises(ValueError, match="nbpts must be at least 1"):
        resampling(src, step_param)
    assert step_param.nbpts == 0

    with temporary_validation_mode(ValidationMode.DISABLED):
        invalid = Resampling1DParam.create(
            mode="dx", xmin=0.0, xmax=1.0, dx=0.0, nbpts=3
        )
        with pytest.raises(ValueError, match="dx must be non-zero"):
            resampling(src, invalid)


def test_windowing_sigma_validation_depends_on_method() -> None:
    """Only Gaussian windowing requires a nonzero sigma."""
    src = create_test_signal()
    with temporary_validation_mode(ValidationMode.STRICT):
        param = WindowingParam.create(method=WindowingMethod.HAMMING, sigma=0.0)
        result = apply_window(src, param)
    np.testing.assert_allclose(result.y, src.y * np.hamming(src.y.size))

    param.method = WindowingMethod.GAUSSIAN
    with pytest.raises(ValueError, match="sigma must be non-zero"):
        apply_window(src, param)
    assert param.sigma == 0.0

    with temporary_validation_mode(ValidationMode.DISABLED):
        invalid = WindowingParam.create(method=WindowingMethod.GAUSSIAN, sigma=0.0)
        with pytest.raises(ValueError, match="sigma must be non-zero"):
            apply_window(src, invalid)

    validate_dataset(
        WindowingParam.create(method=WindowingMethod.GAUSSIAN, sigma=-0.5), src
    )


def test_filter_validation_depends_on_method() -> None:
    """Only IIR cutoffs are bounded by Nyquist and strictly ordered."""
    src = create_test_signal()
    nyquist = 50.0

    validate_dataset(
        LowPassFilterParam.create(
            method=FrequencyFilterMethod.BUTTERWORTH, cut0=nyquist - 1.0
        ),
        src,
    )
    with pytest.raises(ValueError, match="below Nyquist"):
        validate_dataset(
            LowPassFilterParam.create(
                method=FrequencyFilterMethod.BUTTERWORTH, cut0=nyquist
            ),
            src,
        )

    validate_dataset(
        LowPassFilterParam.create(
            method=FrequencyFilterMethod.BRICKWALL, cut0=2.0 * nyquist
        ),
        src,
    )
    validate_dataset(
        BandPassFilterParam.create(
            method=FrequencyFilterMethod.BRICKWALL, cut0=10.0, cut1=10.0
        ),
        src,
    )
    with pytest.raises(ValueError, match="strictly less than cut1"):
        validate_dataset(
            BandPassFilterParam.create(
                method=FrequencyFilterMethod.BUTTERWORTH, cut0=10.0, cut1=10.0
            ),
            src,
        )


def test_filter_validation_requires_source_context() -> None:
    """Dynamic Nyquist validation requires its source signal."""
    param = LowPassFilterParam.create(
        method=FrequencyFilterMethod.BUTTERWORTH, cut0=1.0
    )
    with pytest.raises(ValueError, match="source SignalObj"):
        validate_dataset(param)


def test_filter_nfft_preserves_negative_minimum_semantics() -> None:
    """Negative nfft values remain inert or equivalent to the source-size floor."""
    src = create_test_signal()

    iir = lowpass(
        src,
        LowPassFilterParam.create(
            method=FrequencyFilterMethod.BUTTERWORTH, cut0=10.0, nfft=-1
        ),
    )
    assert iir.y.size == src.y.size

    no_padding = lowpass(
        src,
        LowPassFilterParam.create(
            method=FrequencyFilterMethod.BRICKWALL,
            cut0=10.0,
            zero_padding=False,
            nfft=-1,
        ),
    )
    no_padding_large_nfft = lowpass(
        src,
        LowPassFilterParam.create(
            method=FrequencyFilterMethod.BRICKWALL,
            cut0=10.0,
            zero_padding=False,
            nfft=4096,
        ),
    )
    negative_nfft = lowpass(
        src,
        LowPassFilterParam.create(
            method=FrequencyFilterMethod.BRICKWALL,
            cut0=10.0,
            zero_padding=True,
            nfft=-1,
        ),
    )
    zero_nfft = lowpass(
        src,
        LowPassFilterParam.create(
            method=FrequencyFilterMethod.BRICKWALL,
            cut0=10.0,
            zero_padding=True,
            nfft=0,
        ),
    )

    np.testing.assert_allclose(no_padding.y, no_padding_large_nfft.y)
    np.testing.assert_allclose(negative_nfft.y, zero_nfft.y)


def test_analysis_range_validation() -> None:
    """FWHM stays non-empty, while pulse baselines may be point-like."""
    with pytest.raises(ValueError, match="xmin must be strictly less"):
        validate_dataset(FWHMParam.create(xmin=1.0, xmax=1.0))

    validate_dataset(PulseFeaturesParam())
    with pytest.raises(ValueError, match="xstartmin must be less than or equal"):
        validate_dataset(
            PulseFeaturesParam.create(
                xstartmin=1.0, xstartmax=0.0, xendmin=1.0, xendmax=2.0
            )
        )
    with pytest.raises(ValueError, match="xendmin must be less than or equal"):
        validate_dataset(
            PulseFeaturesParam.create(
                xstartmin=0.0, xstartmax=1.0, xendmin=2.0, xendmax=1.0
            )
        )


def test_pulse_features_accepts_point_baselines_in_public_calls() -> None:
    """Default and explicit point baselines produce usable pulse features."""
    x = np.linspace(0.0, 1.0, 101)
    y = np.clip((x - 0.4) / 0.2, 0.0, 1.0)
    src = create_signal("step", x=x, y=y)
    param = PulseFeaturesParam.create(
        signal_shape="step",
        xstartmin=0.0,
        xstartmax=0.0,
        xendmin=1.0,
        xendmax=1.0,
    )
    original_ranges = (
        param.xstartmin,
        param.xstartmax,
        param.xendmin,
        param.xendmax,
    )

    results = (
        extract_pulse_features(src),
        extract_pulse_features(src, param),
        extract_pulse_features(
            src,
            signal_shape="step",
            xstartmin=0.0,
            xstartmax=0.0,
            xendmin=1.0,
            xendmax=1.0,
        ),
    )

    assert original_ranges == (
        param.xstartmin,
        param.xstartmax,
        param.xendmin,
        param.xendmax,
    )
    for result in results:
        row = result.to_dataframe().iloc[0]
        assert row["polarity"] == 1
        metrics = np.asarray(
            [row["amplitude"], row["offset"], row["rise_time"], row["x50"]],
            dtype=float,
        )
        assert np.isfinite(metrics).all()
        assert row["x50"] == pytest.approx(0.5, abs=0.02)


def test_signal_scalar_bounds() -> None:
    """Percentage and full-scale domains reject values outside their contracts."""
    assert PeakDetectionParam.create(threshold=100.0).threshold == 100.0
    with pytest.raises(ValueError, match="greater than maximum"):
        PeakDetectionParam.create(threshold=100.1)
    with pytest.raises(ValueError, match="Zero is not"):
        DynamicParam.create(full_scale=0.0)
