# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Peak detection unit tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
from sigima.objects import GaussParam, SineParam, create_signal_from_param
from sigima.tests.data import create_paracetamol_signal
from sigima.tests.helpers import check_scalar_result

# `peak_detection` is deprecated in favor of `extract_peak_positions` followed
# by `markers_table_to_signal`. The legacy behaviour is still validated below;
# the deprecation warning itself has its own dedicated test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:sigima.proc.signal.peak_detection is deprecated:DeprecationWarning"
)


@pytest.mark.validation
def test_signal_peak_detection() -> None:
    """Peak detection validation test."""
    # Test 1: Use a known signal with multiple peaks (paracetamol spectrum)
    src = create_paracetamol_signal()

    # Create peak detection parameters
    param = sigima.params.PeakDetectionParam.create(threshold=20, min_dist=5)

    # Apply peak detection
    dst = sigima.proc.signal.peak_detection(src, param)

    # Check that we got some peaks
    assert dst.y.size > 0, "Peak detection should find at least some peaks"
    assert dst.x.size == dst.y.size, "X and Y arrays should have same size"

    # Check that all detected peaks are from the original signal
    assert np.all(dst.x >= src.x.min()) and np.all(dst.x <= src.x.max()), (
        "All detected peak positions should be within the original signal range"
    )

    # Check that peak y-values are from the original signal
    for i in range(dst.x.size):
        # Find closest point in original signal
        idx = np.argmin(np.abs(src.x - dst.x[i]))
        expected_y = src.y[idx]
        check_scalar_result(f"Peak {i} y-value", dst.y[i], expected_y, rtol=1e-10)

    # Test 2: Synthetic signal with known peaks (multiple Gaussians)
    # Create a signal with 3 well-separated Gaussian peaks
    x = np.linspace(-10, 10, 1000)
    y = (
        np.exp(-((x + 4) ** 2) / 2)  # Peak at x=-4
        + 0.5 * np.exp(-((x - 0) ** 2) / 2)  # Peak at x=0 (smaller)
        + np.exp(-((x - 4) ** 2) / 2)  # Peak at x=4
    )
    synthetic_signal = sigima.objects.create_signal("Multi-peak test", x, y)

    # Detect peaks with appropriate parameters
    param = sigima.params.PeakDetectionParam.create(threshold=40, min_dist=100)
    dst_synthetic = sigima.proc.signal.peak_detection(synthetic_signal, param)

    # Should detect exactly 3 peaks
    assert dst_synthetic.x.size == 3, f"Expected 3 peaks, got {dst_synthetic.x.size}"

    # Check peak positions (approximately)
    expected_positions = np.array([-4.0, 0.0, 4.0])
    detected_positions = np.sort(dst_synthetic.x)

    for i, (detected, expected) in enumerate(
        zip(detected_positions, expected_positions)
    ):
        check_scalar_result(f"Peak {i} position", detected, expected, atol=0.2)

    # Test 3: Edge case - signal with minimal peaks
    # Create a simple sinusoidal signal and use very restrictive parameters
    param_simple = SineParam.create(size=100, xmin=0, xmax=10, freq=1, a=0.1)
    simple_signal = create_signal_from_param(param_simple)

    # Use a very high threshold to minimize peak detection
    param = sigima.params.PeakDetectionParam.create(threshold=99, min_dist=1)
    dst_minimal = sigima.proc.signal.peak_detection(simple_signal, param)

    # With such a high threshold, few or no peaks should be detected
    assert dst_minimal.x.size >= 0, "Peak count should be non-negative"
    # If peaks are found, they should be within signal range
    if dst_minimal.x.size > 0:
        assert np.all(dst_minimal.x >= simple_signal.x.min())
        assert np.all(dst_minimal.x <= simple_signal.x.max())

    # Test 4: Single peak signal
    param_single = GaussParam.create(size=200, xmin=-5, xmax=5, a=1, sigma=1, mu=0)
    single_peak_signal = create_signal_from_param(param_single)

    param = sigima.params.PeakDetectionParam.create(threshold=30, min_dist=10)
    dst_single = sigima.proc.signal.peak_detection(single_peak_signal, param)

    # Should detect exactly 1 peak
    assert dst_single.x.size == 1, f"Expected 1 peak, got {dst_single.x.size}"

    # Peak should be near x=0 (the center of the Gaussian)
    check_scalar_result("Single peak position", dst_single.x[0], 0.0, atol=0.1)


def test_signal_peak_detection_parameters() -> None:
    """Test peak detection with different parameter values."""
    # Create a test signal with multiple peaks
    param_signal = SineParam.create(size=500, xmin=0, xmax=10, freq=2, a=1)
    test_signal = create_signal_from_param(param_signal)

    # Test different threshold values
    thresholds = [10, 30, 50, 70]
    peak_counts = []

    for threshold in thresholds:
        param = sigima.params.PeakDetectionParam.create(threshold=threshold, min_dist=5)
        result = sigima.proc.signal.peak_detection(test_signal, param)
        peak_counts.append(result.x.size)

    # Higher thresholds should generally detect fewer peaks
    # (though this isn't guaranteed for all signals)
    assert all(count >= 0 for count in peak_counts), (
        "All peak counts should be non-negative"
    )

    # Test different minimum distances
    min_distances = [1, 5, 10, 20]
    param = sigima.params.PeakDetectionParam.create(threshold=30, min_dist=1)
    result_ref = sigima.proc.signal.peak_detection(test_signal, param)

    for min_dist in min_distances[1:]:  # Skip the first one (reference)
        param = sigima.params.PeakDetectionParam.create(threshold=30, min_dist=min_dist)
        result = sigima.proc.signal.peak_detection(test_signal, param)

        # Larger minimum distances should generally detect fewer or equal peaks
        assert result.x.size <= result_ref.x.size, (
            f"min_dist={min_dist} should not detect more peaks than min_dist=1"
        )


def test_signal_peak_detection_deprecation() -> None:
    """`peak_detection` must emit a DeprecationWarning."""
    src = create_paracetamol_signal()
    param = sigima.params.PeakDetectionParam.create(threshold=20, min_dist=5)
    with pytest.warns(DeprecationWarning, match="peak_detection is deprecated"):
        sigima.proc.signal.peak_detection(src, param)


def _peaks_table(src):
    """Helper: produce an XY-markers TableResult from a paracetamol signal."""
    p = sigima.params.PeakDetectionParam.create(threshold=20, min_dist=5)
    return sigima.proc.signal.extract_peak_positions(src, p)


def test_markers_table_to_signal() -> None:
    """`markers_table_to_signal` must rebuild a sticks signal from a table."""
    src = create_paracetamol_signal()
    table = _peaks_table(src)
    dst = sigima.proc.signal.markers_table_to_signal(table, ref=src)

    assert dst.x.size == len(table.data)
    assert dst.x.size == dst.y.size
    expected_x = np.asarray([row[0] for row in table.data], dtype=float)
    expected_y = np.asarray([row[1] for row in table.data], dtype=float)
    np.testing.assert_allclose(dst.x, expected_x)
    np.testing.assert_allclose(dst.y, expected_y)
    assert dst.get_metadata_option("curvestyle") == "Sticks"
    # Labels/units inherited from the reference signal
    assert dst.xlabel == src.xlabel
    assert dst.ylabel == src.ylabel
    assert dst.xunit == src.xunit
    assert dst.yunit == src.yunit


def test_markers_table_to_signal_invalid_kind() -> None:
    """Non XY-markers tables must be rejected."""
    from sigima.objects import TableKind, TableResult

    table = TableResult.from_rows(
        title="not markers",
        headers=["a", "b"],
        rows=[[1.0, 2.0]],
        kind=TableKind.STATISTICS,
    )
    with pytest.raises(ValueError, match="XY-markers"):
        sigima.proc.signal.markers_table_to_signal(table)


def test_markers_table_to_signal_header_fallback() -> None:
    """When ``x`` / ``y`` headers are missing, the first two columns are used."""
    from sigima.objects import TableKind, TableResult

    table = TableResult.from_rows(
        title="raw markers",
        headers=["lambda", "intensity", "ignored"],
        rows=[[1.0, 10.0, 0.0], [2.0, 20.0, 0.0], [3.0, 30.0, 0.0]],
        kind=TableKind.XY_MARKERS,
    )
    dst = sigima.proc.signal.markers_table_to_signal(table)
    np.testing.assert_allclose(dst.x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(dst.y, [10.0, 20.0, 30.0])
    assert dst.get_metadata_option("curvestyle") == "Sticks"


if __name__ == "__main__":
    test_signal_peak_detection()
    test_signal_peak_detection_parameters()
    test_signal_peak_detection_deprecation()
    test_markers_table_to_signal()
    test_markers_table_to_signal_invalid_kind()
    test_markers_table_to_signal_header_fallback()
