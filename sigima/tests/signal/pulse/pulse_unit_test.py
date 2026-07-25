# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the `sigima.tools.signal.pulse` module.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np
import pytest

import sigima.proc.signal
from sigima.enums import SignalShape
from sigima.objects import create_signal
from sigima.objects.signal import (
    ExpectedFeatures,
    FeatureTolerances,
    GaussParam,
    SquarePulseParam,
    StepPulseParam,
)
from sigima.tests import guiutils
from sigima.tests.data import get_test_signal
from sigima.tests.helpers import check_scalar_result
from sigima.tests.signal.pulse import (
    view_baseline_plateau_and_curve,
    view_pulse_features,
)
from sigima.tools.signal import filtering, pulse


@dataclass
class PulseTestData:
    """Container for pulse test data with metadata."""

    x: np.ndarray
    y: np.ndarray
    signal_type: Literal["step", "square", "gaussian"]
    is_generated: bool
    description: str
    expected_features: ExpectedFeatures | None = None
    tolerances: FeatureTolerances | None = None


def iterate_square_pulse_data() -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Iterate over real square pulse data for testing."""
    for basename in ("boxcar.npy", "square2.npy"):
        obj = get_test_signal(basename)
        yield obj.x, obj.y


def iterate_step_pulse_data() -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Iterate over real step pulse data for testing."""
    for basename in ("step.npy",):
        obj = get_test_signal(basename)
        yield obj.x, obj.y


def iterate_all_step_test_data(
    start_ratio: float = 0.1, stop_ratio: float = 0.9
) -> Generator[PulseTestData, None, None]:
    """Iterate over all step pulse test data (generated and real).

    Args:
        start_ratio: Start ratio for feature calculation
        stop_ratio: Stop ratio for feature calculation

    Yields:
        PulseTestData objects with both generated and real step signals
    """
    # Generated step data
    params = create_test_step_params()
    x, y = params.generate_1d_data()
    yield PulseTestData(
        x=x,
        y=y,
        signal_type="step",
        is_generated=True,
        description="Generated step signal",
        expected_features=params.get_expected_features(start_ratio, stop_ratio),
        tolerances=params.get_feature_tolerances(),
    )

    # Real step data
    for idx, (x, y) in enumerate(iterate_step_pulse_data(), 1):
        yield PulseTestData(
            x=x,
            y=y,
            signal_type="step",
            is_generated=False,
            description=f"Real step signal #{idx}",
        )


def iterate_all_square_test_data(
    start_ratio: float = 0.1, stop_ratio: float = 0.9
) -> Generator[PulseTestData, None, None]:
    """Iterate over all square pulse test data (generated and real).

    Args:
        start_ratio: Start ratio for feature calculation
        stop_ratio: Stop ratio for feature calculation

    Yields:
        PulseTestData objects with both generated and real square signals
    """
    # Generated square data
    params = create_test_square_params()
    x, y = params.generate_1d_data()
    yield PulseTestData(
        x=x,
        y=y,
        signal_type="square",
        is_generated=True,
        description="Generated square signal",
        expected_features=params.get_expected_features(start_ratio, stop_ratio),
        tolerances=params.get_feature_tolerances(),
    )

    # Real square data
    for idx, (x, y) in enumerate(iterate_square_pulse_data(), 1):
        yield PulseTestData(
            x=x,
            y=y,
            signal_type="square",
            is_generated=False,
            description=f"Real square signal #{idx}",
        )


def iterate_all_gaussian_test_data(
    start_ratio: float = 0.1, stop_ratio: float = 0.9
) -> Generator[PulseTestData, None, None]:
    """Iterate over all Gaussian pulse test data (generated only).

    Args:
        start_ratio: Start ratio for feature calculation
        stop_ratio: Stop ratio for feature calculation

    Yields:
        PulseTestData objects with generated Gaussian signals
    """
    # Generated Gaussian data
    params = create_test_gaussian_params()
    x, y = params.generate_1d_data()
    yield PulseTestData(
        x=x,
        y=y,
        signal_type="gaussian",
        is_generated=True,
        description="Generated Gaussian signal",
        expected_features=params.get_expected_features(start_ratio, stop_ratio),
        tolerances=params.get_feature_tolerances(),
    )


def create_test_gaussian_params() -> GaussParam:
    """Create GaussParam with explicit test values."""
    params = GaussParam()
    # Explicit values to ensure test stability
    params.xmin = -10.0
    params.xmax = 10.0
    params.size = 1000
    params.amplitude = 5.0
    params.y0 = 0.0
    params.sigma = 2.0
    params.mu = 0.0
    return params


def create_test_step_params() -> StepPulseParam:
    """Create StepPulseParam with explicit test values."""
    params = StepPulseParam()
    # Explicit values to ensure test stability
    params.xmin = 0.0
    params.xmax = 10.0
    params.size = 1000
    params.offset = 0.0
    params.amplitude = 5.0
    params.noise_amplitude = 0.2
    params.x_rise_start = 3.0
    params.total_rise_time = 2.0
    return params


def create_test_square_params() -> SquarePulseParam:
    """Create SquarePulseParam with explicit test values."""
    params = SquarePulseParam()
    # Explicit values to ensure test stability
    params.xmin = 0.0
    params.xmax = 20.0
    params.size = 1000
    params.offset = 0.0
    params.amplitude = 5.0
    params.noise_amplitude = 0.2
    params.x_rise_start = 3.0
    params.total_rise_time = 2.0
    params.fwhm = 5.5
    params.total_fall_time = 5.0
    return params


@dataclass
class AnalysisParams:
    """Parameters for pulse analysis."""

    start_ratio: float = 0.1
    stop_ratio: float = 0.9
    start_range: tuple[float, float] = (0.0, 3.0)
    end_range: tuple[float, float] = (6.0, 8.0)


def _test_shape_recognition_case(
    signal_type: Literal["step", "square", "gaussian"],
    expected_shape: SignalShape,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> None:
    """Helper function to test shape recognition for different signal configurations.

    Args:
        signal_type: Signal shape type
        expected_shape: Expected SignalShape result
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for shape recognition (optional)
        end_range: End baseline range for shape recognition (optional)
    """
    # Generate signal
    if signal_type == "step":
        step_params = create_test_step_params()
        step_params.offset = y_initial
        step_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = step_params.generate_1d_data()
    elif signal_type == "square":
        square_params = create_test_square_params()
        square_params.offset = y_initial
        square_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = square_params.generate_1d_data()
    else:  # gaussian
        gaussian_params = create_test_gaussian_params()
        gaussian_params.y0 = y_initial
        gaussian_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = gaussian_params.generate_1d_data()

    # Create title
    polarity_desc = "positive" if y_final_or_high > y_initial else "negative"
    title = f"{signal_type.capitalize()}, {polarity_desc} polarity | Shape recognition"
    if start_range is None:
        title += " (auto-detection)"

    # Test shape recognition
    if start_range is not None and end_range is not None:
        shape = pulse.heuristically_recognize_shape(x, y_noisy, start_range, end_range)
    else:
        shape = pulse.heuristically_recognize_shape(x, y_noisy)

    assert shape == expected_shape, f"Expected {expected_shape}, got {shape}"
    guiutils.view_curves_if_gui([[x, y_noisy]], title=f"{title}: {shape}")

    # Test auto-detection if requested and ranges were provided
    if start_range is not None:
        shape_auto = pulse.heuristically_recognize_shape(x, y_noisy)
        assert shape_auto == expected_shape, (
            f"Auto-detection: Expected {expected_shape}, got {shape_auto}"
        )


def _test_shape_recognition_with_data(
    test_data: PulseTestData,
    expected_shape: SignalShape,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> None:
    """Test shape recognition using PulseTestData.

    Args:
        test_data: Test data container
        expected_shape: Expected SignalShape result
        start_range: Start baseline range for shape recognition (optional)
        end_range: End baseline range for shape recognition (optional)
    """
    x, y = test_data.x, test_data.y
    title = f"{test_data.description} | Shape recognition"

    # Test shape recognition
    if start_range is not None and end_range is not None:
        shape = pulse.heuristically_recognize_shape(x, y, start_range, end_range)
        title += " (with ranges)"
    else:
        shape = pulse.heuristically_recognize_shape(x, y)
        title += " (auto-detection)"

    assert shape == expected_shape, f"Expected {expected_shape}, got {shape}"
    guiutils.view_curves_if_gui([[x, y]], title=f"{title}: {shape}")


def test_heuristically_recognize_shape() -> None:
    """Unit test for the `pulse.heuristically_recognize_shape` function.

    This test verifies that the function correctly identifies the shape of various
    noisy signals (step and square) generated with different parameters. It checks the
    recognition both with and without specifying regions of interest.

    Test cases:
        - Step signal with default parameters.
        - Step signal with specified regions.
        - Square signal with default parameters.
        - Step signal with custom initial and final values.
        - Square signal with custom initial and high values.

    """
    tsc = _test_shape_recognition_case
    # Step signals with positive polarity
    tsc("step", SignalShape.STEP, 0.0, 5.0, (0.0, 2.0), (4.0, 8.0))
    # Step signals with negative polarity
    tsc("step", SignalShape.STEP, 5.0, 2.0, (0.0, 2.0), (4.0, 8.0))
    # Square signals with positive polarity
    tsc("square", SignalShape.SQUARE, 0.0, 5.0, (0.0, 2.0), (12.0, 14.0))
    # Square signals with negative polarity
    tsc("square", SignalShape.SQUARE, 5.0, 2.0, (0.0, 2.0), (12.0, 14.0))
    # Gaussian signals with positive polarity
    tsc("gaussian", SignalShape.SQUARE, 0.0, 5.0)

    # Test with real data
    for test_data in iterate_all_step_test_data():
        if not test_data.is_generated:
            _test_shape_recognition_with_data(test_data, SignalShape.STEP)

    for test_data in iterate_all_square_test_data():
        if not test_data.is_generated:
            _test_shape_recognition_with_data(test_data, SignalShape.SQUARE)


def _test_polarity_detection_case(
    signal_type: Literal["step", "square", "gaussian"],
    polarity_desc: str,
    expected_polarity: int,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> None:
    """Helper function to test polarity detection for different signal configurations.

    Args:
        signal_type: Signal shape type
        polarity_desc: Description of polarity ("positive" or "negative")
        expected_polarity: Expected polarity result (1 or -1)
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for polarity detection (optional)
        end_range: End baseline range for polarity detection (optional)
    """
    # Generate signal
    if signal_type == "step":
        step_params = create_test_step_params()
        step_params.offset = y_initial
        step_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = step_params.generate_1d_data()
    elif signal_type == "square":
        square_params = create_test_square_params()
        square_params.offset = y_initial
        square_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = square_params.generate_1d_data()
    else:  # gaussian
        gaussian_params = create_test_gaussian_params()
        gaussian_params.y0 = y_initial
        gaussian_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = gaussian_params.generate_1d_data()

    # Create title
    title = f"{signal_type}, detection {polarity_desc} polarity"
    if start_range is None:
        title += " (auto)"

    # Test polarity detection
    if start_range is not None and end_range is not None:
        polarity = pulse.detect_polarity(x, y_noisy, start_range, end_range)
    else:
        polarity = pulse.detect_polarity(x, y_noisy)

    check_scalar_result(title, polarity, expected_polarity)
    guiutils.view_curves_if_gui([[x, y_noisy]], title=f"{title}: {polarity}")

    # Test auto-detection if requested and ranges were provided
    if start_range is not None:
        polarity_auto = pulse.detect_polarity(x, y_noisy)
        check_scalar_result(f"{title} (auto)", polarity_auto, expected_polarity)


def _test_polarity_detection_with_data(
    test_data: PulseTestData,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> None:
    """Test polarity detection using PulseTestData.

    Args:
        test_data: Test data container
        start_range: Start baseline range for polarity detection (optional)
        end_range: End baseline range for polarity detection (optional)
    """
    x, y = test_data.x, test_data.y
    title = f"{test_data.description} | Polarity detection"

    # Test polarity detection
    if start_range is not None and end_range is not None:
        polarity = pulse.detect_polarity(x, y, start_range, end_range)
        title += " (with ranges)"
    else:
        polarity = pulse.detect_polarity(x, y)
        title += " (auto-detection)"

    # For real data, we just verify it returns a valid polarity
    assert polarity in (1, -1), f"Expected polarity to be 1 or -1, got {polarity}"
    guiutils.view_curves_if_gui([[x, y]], title=f"{title}: {polarity}")


def test_detect_polarity() -> None:
    """Unit test for the `pulse.detect_polarity` function.

    This test verifies the correct detection of signal polarity for both step and
    square signals, with various initial and final values, and using different detection
    intervals.

    Test cases covered:
    - Positive polarity detection for step and square signals.
    - Negative polarity detection for step and square signals with inverted amplitude.
    - Detection with and without explicit interval arguments.
    """
    tpdc = _test_polarity_detection_case
    # Step signals with positive polarity
    tpdc("step", "positive", 1, 0.0, 5.0, (0.0, 2.0), (4.0, 8.0))
    # Step signals with negative polarity
    tpdc("step", "negative", -1, 5.0, 2.0, (0.0, 2.0), (4.0, 8.0))
    # Square signals with positive polarity
    tpdc("square", "positive", 1, 0.0, 5.0, (0.0, 2.0), (12.0, 14.0))
    # Square signals with negative polarity
    tpdc("square", "negative", -1, 5.0, 2.0, (0.0, 2.0), (12.0, 14.0))
    # Gaussian signals with positive polarity (use baseline ranges at extremes)
    tpdc("gaussian", "positive", 1, 0.0, 5.0, (-9.0, -7.0), (7.0, 9.0))
    # Gaussian signals with negative polarity
    tpdc("gaussian", "negative", -1, 5.0, 2.0, (-9.0, -7.0), (7.0, 9.0))

    # Test with real data
    for test_data in iterate_all_step_test_data():
        if not test_data.is_generated:
            _test_polarity_detection_with_data(test_data)

    for test_data in iterate_all_square_test_data():
        if not test_data.is_generated:
            _test_polarity_detection_with_data(test_data)


def _test_amplitude_case(
    signal_type: Literal["step", "square", "gaussian"],
    polarity_desc: str,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    plateau_range: tuple[float, float] | None = None,
    atol: float = 0.2,
    rtol: float = 0.1,
) -> None:
    """Helper function to test amplitude calculation for different signal configs.

    Args:
        signal_type: Signal shape type
        polarity_desc: Description of polarity ("positive" or "negative")
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for amplitude calculation
        end_range: End baseline range for amplitude calculation
        plateau_range: Plateau range for square signals (optional)
        atol: Absolute tolerance for amplitude comparison
        rtol: Relative tolerance for auto-detection comparison
    """
    # Generate signal and calculate expected amplitude
    if signal_type == "step":
        step_params = create_test_step_params()
        step_params.offset = y_initial
        step_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = step_params.generate_1d_data()
        expected_features = step_params.get_expected_features()
        expected_amp = expected_features.amplitude
    elif signal_type == "square":
        square_params = create_test_square_params()
        square_params.offset = y_initial
        square_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = square_params.generate_1d_data()
        expected_features = square_params.get_expected_features()
        expected_amp = expected_features.amplitude
    else:  # gaussian
        gaussian_params = create_test_gaussian_params()
        gaussian_params.y0 = y_initial
        gaussian_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = gaussian_params.generate_1d_data()
        expected_features = gaussian_params.get_expected_features()
        expected_amp = expected_features.amplitude

    # Create title
    title = (
        f"{signal_type.capitalize()}, {polarity_desc} polarity | "
        f"Get {signal_type} amplitude"
    )
    if plateau_range is None:
        title += " (without plateau)"

    # Test with explicit ranges
    if plateau_range is not None:
        amp = pulse.get_amplitude(x, y_noisy, start_range, end_range, plateau_range)
    else:
        amp = pulse.get_amplitude(x, y_noisy, start_range, end_range)

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_baseline_plateau_and_curve(
                x,
                y_noisy,
                f"{title}: {amp:.3f}",
                signal_type,
                start_range,
                end_range,
                plateau_range,
            )

    check_scalar_result(title, amp, expected_amp, atol=atol)

    # Test auto-detection
    amplitude_auto = pulse.get_amplitude(x, y_noisy)
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_baseline_plateau_and_curve(
                x,
                y_noisy,
                f"{title}: {amp:.3f} (auto)",
                signal_type,
                pulse.get_start_range(x),
                pulse.get_end_range(x),
                pulse.get_plateau_range(x, y_noisy, expected_features.polarity),
            )
    check_scalar_result(f"{title} (auto)", amplitude_auto, expected_amp, rtol=rtol)


def _test_amplitude_with_data(
    test_data: PulseTestData,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    plateau_range: tuple[float, float] | None = None,
) -> None:
    """Test amplitude calculation using PulseTestData.

    Args:
        test_data: Test data container
        start_range: Start baseline range (optional)
        end_range: End baseline range (optional)
        plateau_range: Plateau range for square signals (optional)
    """
    x, y = test_data.x, test_data.y
    title = f"{test_data.description} | Amplitude calculation"

    # Calculate amplitude
    if start_range is not None and end_range is not None:
        if plateau_range is not None:
            amp = pulse.get_amplitude(x, y, start_range, end_range, plateau_range)
        else:
            amp = pulse.get_amplitude(x, y, start_range, end_range)
        title += " (with ranges)"
    else:
        amp = pulse.get_amplitude(x, y)
        title += " (auto-detection)"

    # For real data, just verify we get a reasonable value
    assert amp > 0, f"Expected positive amplitude, got {amp}"

    # Check against expected if available
    if test_data.expected_features is not None:
        check_scalar_result(
            title,
            amp,
            test_data.expected_features.amplitude,
            atol=test_data.tolerances.amplitude if test_data.tolerances else 0.2,
        )

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_baseline_plateau_and_curve(
                x,
                y,
                f"{title}: {amp:.3f}",
                test_data.signal_type,
                start_range or pulse.get_start_range(x),
                end_range or pulse.get_end_range(x),
                plateau_range,
            )


def test_get_amplitude() -> None:
    """Unit test for the `pulse.get_amplitude` function.

    This test verifies the correct calculation of the amplitude of step and square
    signals, both with and without specified regions of interest. It checks the
    amplitude for both positive and negative polarities using theoretical calculations.

    Test cases:
        - Step signal with positive polarity.
        - Step signal with negative polarity.
        - Square signal with positive polarity.
        - Square signal with negative polarity.
        - Gaussian signal with positive polarity.
        - Gaussian signal with negative polarity.

        - Step signal with custom initial and final values.
        - Square signal with custom initial and high values.
    """
    tac = _test_amplitude_case
    # Step signals
    tac("step", "positive", 0.0, 5.0, (0.0, 2.0), (6.0, 8.0))
    tac("step", "negative", 5.0, 2.0, (0.0, 2.0), (6.0, 8.0))
    # Square signals with plateau
    tac("square", "positive", 0.0, 5.0, (0.0, 2.0), (12.0, 14.0), (5.5, 6.5))
    tac("square", "negative", 5.0, 2.0, (0.0, 2.0), (12.0, 14.0), (5.5, 6.5), rtol=0.25)
    # Square signals without plateau (auto-detected plateau)
    tac("square", "positive", 0.0, 5.0, (0.0, 2.0), (12.0, 14.0), atol=0.7)
    tac("square", "negative", 5.0, 2.0, (0.0, 2.0), (12.0, 14.0), atol=0.7, rtol=0.25)
    # Gaussian signals
    tac("gaussian", "positive", 0.0, 5.0, (-9.0, -7.0), (7.0, 9.0), atol=0.6)
    tac("gaussian", "negative", 5.0, 2.0, (-9.0, -7.0), (7.0, 9.0), atol=0.6)

    # Test with real data (auto-detection only, as we don't know optimal ranges)
    for test_data in iterate_all_step_test_data():
        if not test_data.is_generated:
            _test_amplitude_with_data(test_data)

    for test_data in iterate_all_square_test_data():
        if not test_data.is_generated:
            _test_amplitude_with_data(test_data)


def _test_crossing_ratio_time_case(
    signal_type: Literal["step", "square", "gaussian"],
    polarity_desc: str,
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    ratio: float,
    edge: Literal["rise", "fall"] = "rise",
    atol: float = 0.1,
    rtol: float = 0.1,
) -> None:
    """Helper function to test crossing ratio time for different signal configurations.

    Args:
        signal_type: Signal shape type
        polarity_desc: Description of polarity ("positive" or "negative")
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for crossing time calculation
        end_range: End baseline range for crossing time calculation
        ratio: Crossing ratio (0.0 to 1.0)
        edge: Which edge to calculate for square signals
        atol: Absolute tolerance for crossing time comparison
        rtol: Relative tolerance for auto-detection comparison
    """
    # Generate signal and calculate expected crossing time
    if signal_type == "step":
        step_params = create_test_step_params()
        step_params.offset = y_initial
        step_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = step_params.generate_1d_data()
        # Calculate crossing time for the specific ratio
        expected_ct = step_params.get_crossing_time("rise", ratio)
    elif signal_type == "square":
        square_params = create_test_square_params()
        square_params.offset = y_initial
        square_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = square_params.generate_1d_data()
        # For square signals, calculate crossing time based on edge and ratio
        expected_ct = square_params.get_crossing_time(edge, ratio)
    else:  # gaussian
        gaussian_params = create_test_gaussian_params()
        gaussian_params.y0 = y_initial
        gaussian_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = gaussian_params.generate_1d_data()
        # Calculate crossing time for the specific ratio
        expected_ct = gaussian_params.get_crossing_time("rise", ratio)

    # Create title
    title = (
        f"{signal_type.capitalize()}, {polarity_desc} polarity | "
        f"Get crossing time at {ratio:.1%}"
    )
    if signal_type == "square":
        title += f" ({edge} edge)"

    # Using the same denoise algorithm as in `extract_pulse_features`
    y_noisy = filtering.denoise_preserve_shape(y_noisy)[0]

    # Test with explicit ranges
    ct = pulse.find_crossing_at_ratio(x, y_noisy, ratio, start_range, end_range)
    check_scalar_result(title, ct, expected_ct, atol=atol)

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # polarity = pulse.detect_polarity(x, y_noisy, start_range, end_range)
            # plateau_range = pulse.get_plateau_range(x, y_noisy, polarity)
            view_baseline_plateau_and_curve(
                x,
                y_noisy,
                f"{title}: {ct:.3f}",
                signal_type,
                start_range,
                end_range,
                plateau_range=None,
                vcursors={f"Crossing at {ratio:.1%}": ct},
            )

    # Test auto-detection
    ct_auto = pulse.find_crossing_at_ratio(x, y_noisy, ratio)
    check_scalar_result(f"{title} (auto)", ct_auto, expected_ct, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ratio", [0.2, 0.5, 0.8])
def test_get_crossing_ratio_time(ratio: float) -> None:
    """Unit test for the `pulse.find_crossing_at_ratio` function.

    This test verifies the correct calculation of the crossing time at a given ratio
    for both positive and negative polarity step signals using theoretical calculations
    based on the signal generation parameters.

    Test cases:
        - Step signal with positive polarity.
        - Step signal with negative polarity.
    """
    tcrtc = _test_crossing_ratio_time_case

    tcrtc("step", "positive", 0.0, 5.0, (0.0, 2.0), (6.0, 8.0), ratio)
    tcrtc("step", "negative", 5.0, 2.0, (0.0, 2.0), (6.0, 8.0), ratio)
    # Gaussian signals (test that functions work, even if results are less meaningful)
    tcrtc("gaussian", "positive", 0.0, 5.0, (-9.0, -7.0), (7.0, 9.0), ratio, atol=1.0)
    tcrtc("gaussian", "negative", 5.0, 2.0, (-9.0, -7.0), (7.0, 9.0), ratio, atol=1.0)


def _test_rise_time_case(
    signal_type: Literal["step", "square", "gaussian"],
    polarity_desc: Literal["positive", "negative"],
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    start_ratio: float,
    stop_ratio: float,
    noise_amplitude: float = 0.1,
    atol: float = 0.1,
    rtol: float = 0.1,
) -> None:
    """Helper function to test step rise time for different signal configurations.

    Args:
        signal_type: Signal shape type
        polarity_desc: Description of polarity
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for rise time calculation
        end_range: End baseline range for rise time calculation
        start_ratio: Starting amplitude ratio for rise time measurement
        stop_ratio: Stopping amplitude ratio (e.g., 0.8 for 80%)
        noise_amplitude: Noise level for signal generation
        atol: Absolute tolerance for rise time comparison
        rtol: Relative tolerance for auto-detection comparison
    """
    rise_or_fall = "Rise" if polarity_desc == "positive" else "Fall"

    if noise_amplitude == 0.0:
        atol /= 10.0  # Tighter check for clean signals

    # Generate signal and calculate expected rise time
    if signal_type == "step":
        step_params = create_test_step_params()
        step_params.offset = y_initial
        step_params.amplitude = y_final_or_high - y_initial
        step_params.noise_amplitude = noise_amplitude
        x, y_noisy = step_params.generate_1d_data()
        expected_features = step_params.get_expected_features(start_ratio, stop_ratio)
    elif signal_type == "square":
        square_params = create_test_square_params()
        square_params.offset = y_initial
        square_params.amplitude = y_final_or_high - y_initial
        square_params.noise_amplitude = noise_amplitude
        x, y_noisy = square_params.generate_1d_data()
        expected_features = square_params.get_expected_features(start_ratio, stop_ratio)
    else:  # gaussian
        gaussian_params = create_test_gaussian_params()
        gaussian_params.y0 = y_initial
        gaussian_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = gaussian_params.generate_1d_data()
        expected_features = gaussian_params.get_expected_features(
            start_ratio, stop_ratio
        )

    # Create title
    noise_desc = "clean" if noise_amplitude == 0 else "noisy"
    title = (
        f"{signal_type.capitalize()}, {polarity_desc} polarity | "
        f"Get {rise_or_fall.lower()} time ({noise_desc})"
    )

    # Test with explicit ranges
    rise_time = pulse.get_rise_time(
        x, y_noisy, start_ratio, stop_ratio, start_range, end_range
    )

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from sigima import viz

            ct1 = pulse.find_crossing_at_ratio(
                x, y_noisy, start_ratio, start_range, end_range
            )
            ct2 = pulse.find_crossing_at_ratio(
                x, y_noisy, stop_ratio, start_range, end_range
            )
            item = viz.create_range(
                "h",
                ct1,
                ct2,
                f"{rise_or_fall} time {start_ratio:.0%}-"
                f"{stop_ratio:.0%} = {rise_time:.3f}",
            )

            view_baseline_plateau_and_curve(
                x,
                y_noisy,
                f"{title}: {rise_time:.3f}",
                signal_type,
                start_range,
                end_range,
                plateau_range=None,
                other_items=[item],
            )

    check_scalar_result(title, rise_time, expected_features.rise_time, atol=atol)

    # Test auto-detection
    rise_time_auto = pulse.get_rise_time(
        x, y_noisy, start_ratio=start_ratio, stop_ratio=stop_ratio
    )
    check_scalar_result(
        f"{title} (auto)", rise_time_auto, expected_features.rise_time, rtol=rtol
    )


@pytest.mark.parametrize("noise_amplitude", [0.1, 0.0])
def test_get_rise_time(noise_amplitude: float) -> None:
    """Unit test for the `pulse.get_rise_time` function.

    This test verifies the correct calculation of the rise time for step signals with
    both positive and negative polarity using theoretical calculations based on
    signal generation parameters.

    Test cases (including noisy and clean signals):
        - Step signal with positive polarity (20%-80% rise time).
        - Step signal with negative polarity (20%-80% rise time).
    """
    trtc = _test_rise_time_case
    # Standard 20%-80% rise time parameters
    start_ratio, stop_ratio = 0.2, 0.8

    # Step signals with positive polarity
    na = noise_amplitude
    trtc("step", "positive", 0.0, 5.0, (0, 2), (6, 8), start_ratio, stop_ratio, na)
    trtc("step", "negative", 5.0, 2.0, (0, 2), (6, 8), start_ratio, stop_ratio, na)
    # Gaussian signals (test that functions work, even if results are less meaningful)
    trtc(
        "gaussian",
        "positive",
        0.0,
        5.0,
        (-9.0, -7.0),
        (7.0, 9.0),
        start_ratio,
        stop_ratio,
        na,
        atol=1.0,
    )
    trtc(
        "gaussian",
        "negative",
        5.0,
        2.0,
        (-9.0, -7.0),
        (7.0, 9.0),
        start_ratio,
        stop_ratio,
        na,
        atol=1.0,
    )

    # Test with real data (only for noise_amplitude=0.1 to avoid duplication)
    if noise_amplitude == 0.1:
        for test_data in iterate_all_step_test_data():
            if not test_data.is_generated:
                _test_rise_time_with_data(test_data, start_ratio, stop_ratio)

        for test_data in iterate_all_square_test_data():
            if not test_data.is_generated:
                _test_rise_time_with_data(test_data, start_ratio, stop_ratio)


def _test_rise_time_with_data(
    test_data: PulseTestData,
    start_ratio: float,
    stop_ratio: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
) -> None:
    """Test rise time calculation using PulseTestData.

    Args:
        test_data: Test data container
        start_ratio: Starting amplitude ratio for rise time measurement
        stop_ratio: Stopping amplitude ratio for rise time measurement
        start_range: Start baseline range (optional)
        end_range: End baseline range (optional)
    """
    x, y = test_data.x, test_data.y
    title = f"{test_data.description} | Rise time"

    # Calculate rise time
    if start_range is not None and end_range is not None:
        rise_time = pulse.get_rise_time(
            x, y, start_ratio, stop_ratio, start_range, end_range
        )
        title += " (with ranges)"
    else:
        rise_time = pulse.get_rise_time(x, y, start_ratio, stop_ratio)
        title += " (auto-detection)"

    # For real data, just verify we get a reasonable value
    assert rise_time > 0, f"Expected positive rise time, got {rise_time}"

    # Check against expected if available
    if test_data.expected_features is not None:
        check_scalar_result(
            title,
            rise_time,
            test_data.expected_features.rise_time,
            atol=test_data.tolerances.rise_time if test_data.tolerances else 0.2,
        )

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from sigima import viz

            sr = start_range or pulse.get_start_range(x)
            er = end_range or pulse.get_end_range(x)
            ct1 = pulse.find_crossing_at_ratio(x, y, start_ratio, sr, er)
            ct2 = pulse.find_crossing_at_ratio(x, y, stop_ratio, sr, er)
            item = viz.create_range(
                "h",
                ct1,
                ct2,
                f"Rise time {start_ratio:.0%}-{stop_ratio:.0%} = {rise_time:.3f}",
            )
            view_baseline_plateau_and_curve(
                x,
                y,
                f"{title}: {rise_time:.3f}",
                test_data.signal_type,
                sr,
                er,
                plateau_range=None,
                other_items=[item],
            )


# pylint: disable=too-many-positional-arguments
def _test_fall_time_case(
    signal_type: Literal["square", "gaussian"],
    polarity_desc: Literal["positive", "negative"],
    y_initial: float,
    y_final_or_high: float,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    plateau_range: tuple[float, float],
    start_ratio: float,
    stop_ratio: float,
    noise_amplitude: float = 0.1,
    atol: float = 0.1,
    rtol: float = 0.1,
) -> None:
    """Helper function to test fall time for different signal configurations.

    Args:
        signal_type: Type of signal ("square" or "gaussian")
        polarity_desc: Description of polarity
        y_initial: Initial signal value
        y_final_or_high: Final value (step) or high value (square)
        start_range: Start baseline range for fall time calculation
        end_range: End baseline range for fall time calculation
        plateau_range: Plateau range for square signals
        start_ratio: Starting amplitude ratio for fall time measurement
        stop_ratio: Stopping amplitude ratio (e.g., 0.8 for 80%)
        noise_amplitude: Noise level for signal generation
        atol: Absolute tolerance for fall time comparison
        rtol: Relative tolerance for auto-detection comparison
    """
    if noise_amplitude == 0.0:
        atol /= 10.0  # Tighter check for clean signals

    # Generate signal and calculate expected fall time
    if signal_type == "square":
        square_params = create_test_square_params()
        square_params.offset = y_initial
        square_params.amplitude = y_final_or_high - y_initial
        square_params.noise_amplitude = noise_amplitude
        x, y_noisy = square_params.generate_1d_data()
        expected_features = square_params.get_expected_features(start_ratio, stop_ratio)
    else:  # gaussian
        gaussian_params = create_test_gaussian_params()
        gaussian_params.y0 = y_initial
        gaussian_params.amplitude = y_final_or_high - y_initial
        x, y_noisy = gaussian_params.generate_1d_data()
        expected_features = gaussian_params.get_expected_features(
            start_ratio, stop_ratio
        )

    # Create title
    noise_desc = "clean" if noise_amplitude == 0 else "noisy"
    signal_desc = signal_type.capitalize()
    title = f"{signal_desc}, {polarity_desc} polarity | Get fall time ({noise_desc})"

    # Using the same denoise algorithm as in `extract_pulse_features`
    y_noisy = filtering.denoise_preserve_shape(y_noisy)[0]

    # Test with explicit ranges
    fall_time = pulse.get_fall_time(
        x, y_noisy, start_ratio, stop_ratio, plateau_range, end_range
    )

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from sigima import viz

            ct1 = pulse.find_crossing_at_ratio(
                x, y_noisy[::-1], start_ratio, start_range, end_range
            )
            ct1 = x[-1] - ct1  # Adjust for reversed x
            ct2 = pulse.find_crossing_at_ratio(
                x, y_noisy[::-1], stop_ratio, start_range, end_range
            )
            ct2 = x[-1] - ct2  # Adjust for reversed x
            item = viz.create_range(
                "h",
                ct1,
                ct2,
                f"Fall time {start_ratio:.0%}-{stop_ratio:.0%} = {fall_time:.3f}",
            )

            view_baseline_plateau_and_curve(
                x,
                y_noisy,
                f"{title}: {fall_time:.3f}",
                signal_type,
                start_range,
                end_range,
                plateau_range=plateau_range,
                other_items=[item],
            )

    check_scalar_result(
        f"Get fall time ({noise_desc})",
        fall_time,
        expected_features.fall_time,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("noise_amplitude", [0.1, 0.0])
def test_get_fall_time(noise_amplitude: float) -> None:
    """Unit test for the `pulse.get_fall_time` function.

    This test verifies the correct calculation of the fall time for signals with
    both positive and negative polarity using theoretical calculations based on
    signal generation parameters.

    Test cases (including noisy and clean signals):
        - Square signal with positive polarity (20%-80% fall time).
        - Square signal with negative polarity (20%-80% fall time).
        - Gaussian signal with positive polarity (function test only).
        - Gaussian signal with negative polarity (function test only).
    """
    tftc = _test_fall_time_case

    # Square signals with plateau
    na = noise_amplitude
    tftc(
        "square",
        "positive",
        0.0,
        5.0,
        (0.0, 2.0),
        (12.0, 14.0),
        (5.5, 6.5),
        0.8,
        0.2,
        na,
    )
    tftc(
        "square",
        "negative",
        5.0,
        2.0,
        (0.0, 2.0),
        (12.0, 14.0),
        (5.5, 6.5),
        0.8,
        0.2,
        na,
    )
    # Gaussian signals (test that functions work, even if results are less meaningful)
    tftc(
        "gaussian",
        "positive",
        0.0,
        5.0,
        (-9.0, -7.0),
        (7.0, 9.0),
        (-1.0, 1.0),
        0.8,
        0.2,
        na,
        atol=1.0,
    )
    tftc(
        "gaussian",
        "negative",
        5.0,
        2.0,
        (-9.0, -7.0),
        (7.0, 9.0),
        (-1.0, 1.0),
        0.8,
        0.2,
        na,
        atol=1.0,
    )

    # Test with real data (only for noise_amplitude=0.1 to avoid duplication)
    if noise_amplitude == 0.1:
        for test_data in iterate_all_square_test_data():
            if not test_data.is_generated:
                _test_fall_time_with_data(test_data, 0.8, 0.2)


def _test_fall_time_with_data(
    test_data: PulseTestData,
    start_ratio: float,
    stop_ratio: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    plateau_range: tuple[float, float] | None = None,
) -> None:
    """Test fall time calculation using PulseTestData.

    Args:
        test_data: Test data container
        start_ratio: Starting amplitude ratio for fall time measurement
        stop_ratio: Stopping amplitude ratio for fall time measurement
        start_range: Start baseline range (optional)
        end_range: End baseline range (optional)
        plateau_range: Plateau range (optional)
    """
    x, y = test_data.x, test_data.y
    title = f"{test_data.description} | Fall time"

    # Using the same denoise algorithm as in `extract_pulse_features`
    y = filtering.denoise_preserve_shape(y)[0]

    # Calculate fall time
    if plateau_range is not None and end_range is not None:
        fall_time = pulse.get_fall_time(
            x, y, start_ratio, stop_ratio, plateau_range, end_range
        )
        title += " (with ranges)"
    else:
        # Auto-detect ranges
        sr = start_range or pulse.get_start_range(x)
        er = end_range or pulse.get_end_range(x)
        polarity = pulse.detect_polarity(x, y, sr, er)
        pr = plateau_range or pulse.get_plateau_range(x, y, polarity)
        fall_time = pulse.get_fall_time(x, y, start_ratio, stop_ratio, pr, er)
        title += " (auto-detection)"

    # For real data, fall_time might be None for some signals
    if fall_time is None:
        # This is acceptable for some real data
        return

    # Verify we get a reasonable value
    assert fall_time > 0, f"Expected positive fall time, got {fall_time}"

    # Check against expected if available
    if test_data.expected_features is not None:
        check_scalar_result(
            title,
            fall_time,
            test_data.expected_features.fall_time,
            atol=test_data.tolerances.fall_time if test_data.tolerances else 0.2,
        )

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from sigima import viz

            sr = start_range or pulse.get_start_range(x)
            er = end_range or pulse.get_end_range(x)
            polarity = pulse.detect_polarity(x, y, sr, er)
            pr = plateau_range or pulse.get_plateau_range(x, y, polarity)

            ct1 = pulse.find_crossing_at_ratio(x, y[::-1], start_ratio, sr, er)
            ct1 = x[-1] - ct1
            ct2 = pulse.find_crossing_at_ratio(x, y[::-1], stop_ratio, sr, er)
            ct2 = x[-1] - ct2

            item = viz.create_range(
                "h",
                ct1,
                ct2,
                f"Fall time {start_ratio:.0%}-{stop_ratio:.0%} = {fall_time:.3f}",
            )
            view_baseline_plateau_and_curve(
                x,
                y,
                f"{title}: {fall_time:.3f}",
                test_data.signal_type,
                sr,
                er,
                plateau_range=pr,
                other_items=[item],
            )


def test_heuristically_find_rise_start_time() -> None:
    """Unit test for the `pulse.heuristically_find_rise_start_time` function.

    This test verifies that the function correctly identifies the end time of the foot
    (baseline) region in a step signal with a sharp rise, ensuring accurate detection
    even in the presence of noise.
    """
    # Generate a signal with baseline until t=3, then rising from t=3 to t=5
    step_params = create_test_step_params()
    x, y = step_params.generate_1d_data()
    # Use proper baseline range that doesn't include the rising portion
    time = pulse.heuristically_find_rise_start_time(x, y, (0, 2.5))
    if time is not None:
        # Expected time should be x_rise_start (3.0) - the start of the rise
        # This is when the foot (baseline) region ends
        expected_foot_end_time = step_params.x_rise_start
        check_scalar_result(
            "heuristically find foot end time",
            time,
            expected_foot_end_time,
            atol=0.2,  # Allow reasonable tolerance for noisy signals
        )
    else:
        # If the function returns None, that's unexpected for this signal
        pytest.fail(
            "heuristically_find_rise_start_time returned None for a clear step signal"
        )
    time_str = f"{time:.3f}" if time is not None else "None"
    guiutils.view_curves_if_gui([[x, y]], title=f"Rise start time = {time_str}")


def test_get_rise_start_time() -> None:
    """Unit test for the `pulse.get_rise_start_time ` function."""
    # Generate a step signal with a sharp rise at t=5
    step_params = create_test_step_params()
    x, y = step_params.generate_1d_data()

    # Use start_range before the step, end_range after
    start_range, end_range, threshold = (0, 2), (6, 8), 0.1

    x0 = pulse.get_rise_start_time(x, y, start_range, end_range, threshold=threshold)
    foot_duration = x0 - x[0]  # Since x[0] = 0.0 in this case

    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            # polarity = pulse.detect_polarity(x, y_noisy, start_range, end_range)
            # plateau_range = pulse.get_plateau_range(x, y_noisy, polarity)
            title = f"Foot duration={foot_duration:.3f}, x_end={x0:.3f}, "
            title += f"threshold={threshold:.3f}"
            view_baseline_plateau_and_curve(
                x,
                y,
                title,
                "step",
                start_range,
                end_range,
                plateau_range=None,
                vcursors={"Foot duration end": x0},
            )

    check_scalar_result("foot_info x_end", x0, step_params.x_rise_start, atol=0.2)


def __check_features(
    features: pulse.PulseFeatures,
    expected: ExpectedFeatures,
    tolerances: FeatureTolerances,
) -> None:
    """Helper function to validate extracted pulse features against expected values.

    Args:
        features: Extracted pulse features.
        expected: Expected feature values for validation.
        tolerances: Tolerance values for each feature.
    """
    signal_shape = features.signal_shape
    # Get signal shape string for error messages (handle both string and enum)
    shape_str = signal_shape if isinstance(signal_shape, str) else signal_shape.value
    # Validate numerical features
    for field in dataclasses.fields(features):
        value = getattr(features, field.name)
        expected_value = getattr(expected, field.name, None)
        if expected_value is None:
            continue  # Skip fields without expected values
        tolerance = getattr(tolerances, field.name, None)
        if tolerance is None:
            assert value == expected_value, (
                f"[{shape_str}] {field.name}: Expected {expected_value}, got {value}"
            )
        else:
            check_scalar_result(
                f"[{shape_str}] {field.name}",
                value,
                expected_value,
                atol=tolerance,
            )


def _extract_and_validate_step_features(
    x: np.ndarray,
    y: np.ndarray,
    analysis: AnalysisParams,
    expected: ExpectedFeatures,
    signal_params: StepPulseParam,
) -> pulse.PulseFeatures:
    """Helper function to extract and validate step signal features.

    Args:
        x: X data array
        y: Y data array
        analysis: Analysis parameters for pulse feature extraction
        expected: Expected feature values for validation
        signal_params: Step signal parameters for tolerance calculation

    Returns:
        Extracted pulse features
    """
    # Extract features while ignoring FWHM warnings for noisy signals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features = pulse.extract_pulse_features(
            x,
            y,
            analysis.start_range,
            analysis.end_range,
            analysis.start_ratio,
            analysis.stop_ratio,
        )

    # Visualize results if GUI is available
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_pulse_features(
                x, y, "Step signal feature extraction", "step", features
            )

    # Validate that we got the correct type
    assert isinstance(features, pulse.PulseFeatures), (
        f"Expected PulseFeatures, got {type(features)}"
    )

    # Validate signal shape
    assert features.signal_shape == SignalShape.STEP, (
        f"Expected signal_shape to be STEP, but got {features.signal_shape}"
    )

    # Get tolerance values
    tolerances = signal_params.get_feature_tolerances()

    # Validate numerical features
    __check_features(features, expected, tolerances)

    # Validate that step-specific features are None
    assert features.fall_time is None, (
        f"Expected fall_time to be None for step signal, but got {features.fall_time}"
    )
    assert features.fwhm is None, (
        f"Expected fwhm to be None for step signal, but got {features.fwhm}"
    )

    return features


def _extract_and_validate_step_features_from_data(
    test_data: PulseTestData,
) -> pulse.PulseFeatures:
    """Helper function to extract and validate step signal features from test data.

    Args:
        test_data: Test data container

    Returns:
        Extracted pulse features
    """
    x, y = test_data.x, test_data.y

    # Auto-detect ranges
    start_range = pulse.get_start_range(x)
    end_range = pulse.get_end_range(x)

    # Extract features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features = pulse.extract_pulse_features(x, y, start_range, end_range)

    # Visualize results if GUI is available
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_pulse_features(
                x, y, f"{test_data.description} | Feature extraction", "step", features
            )

    # Validate that we got the correct type
    assert isinstance(features, pulse.PulseFeatures), (
        f"Expected PulseFeatures, got {type(features)}"
    )

    # Validate signal shape
    assert features.signal_shape == SignalShape.STEP, (
        f"Expected signal_shape to be STEP, but got {features.signal_shape}"
    )

    # If we have expected features, validate against them
    if test_data.expected_features is not None and test_data.tolerances is not None:
        __check_features(features, test_data.expected_features, test_data.tolerances)

    return features


def _extract_and_validate_square_features(
    x: np.ndarray,
    y: np.ndarray,
    analysis: AnalysisParams,
    expected: ExpectedFeatures,
    signal_params: SquarePulseParam,
) -> pulse.PulseFeatures:
    """Helper function to extract and validate square signal features.

    Args:
        x: X data array
        y: Y data array
        analysis: Analysis parameters for pulse feature extraction
        expected: Expected feature values for validation
        signal_params: Square signal parameters for tolerance calculation

    Returns:
        Extracted pulse features
    """
    # Extract features while ignoring FWHM warnings for noisy signals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features = pulse.extract_pulse_features(
            x,
            y,
            analysis.start_range,
            analysis.end_range,
            analysis.start_ratio,
            analysis.stop_ratio,
        )

    # Visualize results if GUI is available
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_pulse_features(
                x, y, "Square signal feature extraction", "square", features
            )

    # Validate that we got the correct type
    assert isinstance(features, pulse.PulseFeatures), (
        f"Expected PulseFeatures, got {type(features)}"
    )

    # Validate signal shape
    assert features.signal_shape == SignalShape.SQUARE, (
        f"Expected signal_shape to be SQUARE, but got {features.signal_shape}"
    )

    # Get tolerance values
    tolerances = signal_params.get_feature_tolerances()

    # Validate numerical features
    __check_features(features, expected, tolerances)

    return features


def _extract_and_validate_square_features_from_data(
    test_data: PulseTestData,
) -> pulse.PulseFeatures:
    """Helper function to extract and validate square signal features from test data.

    Args:
        test_data: Test data container

    Returns:
        Extracted pulse features
    """
    x, y = test_data.x, test_data.y

    # Auto-detect ranges
    start_range = pulse.get_start_range(x)
    end_range = pulse.get_end_range(x)

    # Extract features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features = pulse.extract_pulse_features(x, y, start_range, end_range)

    # Visualize results if GUI is available
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_pulse_features(
                x,
                y,
                f"{test_data.description} | Feature extraction",
                "square",
                features,
            )

    # Validate that we got the correct type
    assert isinstance(features, pulse.PulseFeatures), (
        f"Expected PulseFeatures, got {type(features)}"
    )

    # Validate signal shape
    assert features.signal_shape == SignalShape.SQUARE, (
        f"Expected signal_shape to be SQUARE, but got {features.signal_shape}"
    )

    # If we have expected features, validate against them
    if test_data.expected_features is not None and test_data.tolerances is not None:
        __check_features(features, test_data.expected_features, test_data.tolerances)

    return features


def _extract_and_validate_gaussian_features(
    x: np.ndarray,
    y: np.ndarray,
    analysis: AnalysisParams,
    expected: ExpectedFeatures,
    signal_params: GaussParam,
) -> pulse.PulseFeatures:
    """Helper function to extract and validate Gaussian signal features.

    Args:
        x: X data array
        y: Y data array
        analysis: Analysis parameters for pulse feature extraction
        expected: Expected feature values for validation
        signal_params: Gaussian signal parameters for tolerance calculation

    Returns:
        Extracted pulse features
    """
    # Extract features while ignoring FWHM warnings for noisy signals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features = pulse.extract_pulse_features(
            x,
            y,
            analysis.start_range,
            analysis.end_range,
            analysis.start_ratio,
            analysis.stop_ratio,
        )

    # Visualize results if GUI is available
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_pulse_features(
                x, y, "Gaussian signal feature extraction", "gaussian", features
            )

    # Validate that we got the correct type
    assert isinstance(features, pulse.PulseFeatures), (
        f"Expected PulseFeatures, got {type(features)}"
    )

    # Validate signal shape (Gaussian is recognized as SQUARE)
    assert features.signal_shape == SignalShape.SQUARE, (
        f"Expected signal_shape to be SQUARE, but got {features.signal_shape}"
    )

    # Get tolerance values
    tolerances = signal_params.get_feature_tolerances()

    # Validate numerical features
    __check_features(features, expected, tolerances)

    return features


def test_step_feature_extraction() -> None:
    """Test feature extraction for step signals.

    Validates that pulse feature extraction correctly identifies and measures
    all relevant parameters for a step signal, including polarity, amplitude,
    rise time, timing features, and baseline characteristics.
    """
    # Define signal parameters
    signal_params = create_test_step_params()

    # Define analysis parameters
    analysis = AnalysisParams()

    # Calculate expected values
    expected = signal_params.get_expected_features(
        start_ratio=analysis.start_ratio,
        stop_ratio=analysis.stop_ratio,
    )

    # Generate test signal
    x, y = signal_params.generate_1d_data()

    # Extract and validate features
    _extract_and_validate_step_features(x, y, analysis, expected, signal_params)

    # Test with real data
    for test_data in iterate_all_step_test_data():
        if not test_data.is_generated:
            _extract_and_validate_step_features_from_data(test_data)


def test_square_feature_extraction() -> None:
    """Test feature extraction for square signals.

    Validates that pulse feature extraction correctly identifies and measures
    all relevant parameters for a square signal, including polarity, amplitude,
    rise/fall times, FWHM, timing features, and baseline characteristics.
    """
    # Define signal parameters with custom ranges for square signal
    signal_params = create_test_square_params()

    # Define analysis parameters with custom ranges for square signal
    analysis = AnalysisParams(
        start_range=(0.0, 2.5),
        end_range=(15.0, 17.0),
    )

    # Calculate expected values
    expected = signal_params.get_expected_features(
        start_ratio=analysis.start_ratio,
        stop_ratio=analysis.stop_ratio,
    )

    # Generate test signal
    x, y = signal_params.generate_1d_data()

    # Extract and validate features
    _extract_and_validate_square_features(x, y, analysis, expected, signal_params)

    # Test with real data
    for test_data in iterate_all_square_test_data():
        if not test_data.is_generated:
            _extract_and_validate_square_features_from_data(test_data)


def test_gaussian_feature_extraction() -> None:
    """Test feature extraction for Gaussian signals.

    Validates that pulse feature extraction correctly identifies and measures
    all relevant parameters for a Gaussian signal, including polarity, amplitude,
    rise/fall times, timing features, and baseline characteristics using the
    improved Gaussian-aware algorithms.
    """
    # Define signal parameters with appropriate ranges for Gaussian signal
    signal_params = create_test_gaussian_params()

    # Define analysis parameters with ranges suitable for Gaussian signal
    analysis = AnalysisParams(
        start_range=(-9.0, -7.0),
        end_range=(7.0, 9.0),
        start_ratio=0.2,  # 20%
        stop_ratio=0.8,  # 80%
    )

    # Calculate expected values
    expected = signal_params.get_expected_features(
        start_ratio=analysis.start_ratio,
        stop_ratio=analysis.stop_ratio,
    )

    # Generate test signal
    x, y = signal_params.generate_1d_data()

    # Extract and validate features
    _extract_and_validate_gaussian_features(x, y, analysis, expected, signal_params)


@pytest.mark.validation
def test_signal_extract_pulse_features() -> None:
    """Validation test for extract_pulse_features computation function.

    Tests the extract_pulse_features function for both step and square signals,
    validating that all computed parameters match expected theoretical values.
    """
    # Test STEP signal feature extraction
    step_params = create_test_step_params()
    x_step, y_step = step_params.generate_1d_data()
    sig_step = create_signal("Test Step Signal", x_step, y_step)

    # Define step analysis parameters
    p_step = sigima.proc.signal.PulseFeaturesParam()
    p_step.xstartmin = 0.0
    p_step.xstartmax = 3.0
    p_step.xendmin = 6.0
    p_step.xendmax = 8.0
    p_step.reference_levels = (10, 90)

    # Calculate expected step features using the DataSet method
    start_ratio, stop_ratio = p_step.reference_levels
    expected_step = step_params.get_expected_features(
        start_ratio / 100.0, stop_ratio / 100.0
    )
    tolerances_step = step_params.get_feature_tolerances()

    # Extract and validate step features
    table_step = sigima.proc.signal.extract_pulse_features(sig_step, p_step)
    tdict_step = table_step.as_dict()
    features_step = pulse.PulseFeatures(**tdict_step)
    __check_features(features_step, expected_step, tolerances_step)

    # Visualize results if GUI is available
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_pulse_features(
                x_step, y_step, "Step signal feature extraction", "step", features_step
            )

    # Test SQUARE signal feature extraction
    square_params = create_test_square_params()
    x_square, y_square = square_params.generate_1d_data()
    sig_square = create_signal("Test Square Signal", x_square, y_square)

    # Define square analysis parameters
    p_square = sigima.proc.signal.PulseFeaturesParam()
    p_square.xstartmin = 0
    p_square.xstartmax = 2.5
    p_square.xendmin = 15
    p_square.xendmax = 17
    p_square.reference_levels = (10, 90)

    # Calculate expected square features using the DataSet method
    start_ratio, stop_ratio = p_square.reference_levels
    expected_square = square_params.get_expected_features(
        start_ratio / 100.0, stop_ratio / 100.0
    )

    # Extract and validate square features
    table_square = sigima.proc.signal.extract_pulse_features(sig_square, p_square)
    tdict_square = table_square.as_dict()
    features_square = pulse.PulseFeatures(**tdict_square)
    tolerances_square = square_params.get_feature_tolerances()
    __check_features(features_square, expected_square, tolerances_square)

    # Visualize results if GUI is available
    with guiutils.lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            view_pulse_features(
                x_square,
                y_square,
                "Square signal feature extraction",
                "square",
                features_square,
            )


if __name__ == "__main__":
    guiutils.enable_gui()
    # test_heuristically_recognize_shape()
    # test_detect_polarity()
    test_get_amplitude()
    test_get_crossing_ratio_time(0.2)
    test_get_crossing_ratio_time(0.5)
    test_get_crossing_ratio_time(0.8)
    test_get_rise_time(0.1)
    test_get_rise_time(0.0)
    test_get_fall_time(0.1)
    test_get_fall_time(0.0)
    test_heuristically_find_rise_start_time()
    test_get_rise_start_time()
    test_step_feature_extraction()
    test_square_feature_extraction()
    test_gaussian_feature_extraction()
    test_signal_extract_pulse_features()
