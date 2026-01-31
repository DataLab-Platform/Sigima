# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for plateau detection algorithm with real square pulse data.

This test suite validates that the plateau detection algorithm correctly identifies
the plateau region in real-world square pulse signals.
"""

from __future__ import annotations

import pytest

from sigima.tests import guiutils
from sigima.tests.data import get_test_signal
from sigima.tests.signal.pulse import view_baseline_plateau_and_curve
from sigima.tools.signal import pulse

# Test data configuration: (basename, expected_interval)
PLATEAU_TEST_DATA = [
    ("boxcar.npy", (0.0, 1.4e-7)),  # Slightly wider to accommodate full plateau
    ("square2.npy", (0.0, 2.4e-4)),  # Slightly wider to accommodate full plateau
]


@pytest.mark.parametrize("basename,expected_interval", PLATEAU_TEST_DATA)
def test_plateau_detection(
    basename: str, expected_interval: tuple[float, float]
) -> None:
    """Parametric test for plateau detection on real square pulse data.

    Args:
        basename: Name of the test data file
        expected_interval: Expected (min, max) interval containing the plateau

    This test ensures that the plateau detection algorithm correctly identifies
    the plateau region within the expected interval for various real signals.
    The test also includes optional visualization when run with GUI enabled.
    """
    # Load real data
    obj = get_test_signal(basename)
    x, y = obj.x, obj.y

    # Auto-detect polarity and plateau
    start_range = pulse.get_start_range(x)
    end_range = pulse.get_end_range(x)
    polarity = pulse.detect_polarity(x, y, start_range, end_range)
    plateau_min, plateau_max = pulse.get_plateau_range(x, y, polarity)

    expected_min, expected_max = expected_interval

    with guiutils.lazy_qt_app_context() as qt_app:
        title = (
            f"{basename} - Plateau Detection Test\n"
            f"Detected: [{plateau_min:.3e}, {plateau_max:.3e}]"
        )
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from sigima import viz

            item = viz.create_range(
                "h",
                expected_min,
                expected_max,
                "Expected plateau interval",
            )
            view_baseline_plateau_and_curve(
                x,
                y,
                title,
                "square",
                start_range or pulse.get_start_range(x),
                end_range or pulse.get_end_range(x),
                (plateau_min, plateau_max),
                other_items=[item],
            )

    # Validate plateau is within expected interval
    assert plateau_min >= expected_min, (
        f"{basename}: Plateau start {plateau_min:.3e} is before expected "
        f"interval [{expected_min:.3e}, {expected_max:.3e}]"
    )
    assert plateau_max <= expected_max, (
        f"{basename}: Plateau end {plateau_max:.3e} is after expected "
        f"interval [{expected_min:.3e}, {expected_max:.3e}]"
    )
    assert plateau_min < plateau_max, (
        f"{basename}: Invalid plateau range [{plateau_min:.3e}, {plateau_max:.3e}]"
    )


def all_tests() -> None:
    """Run all plateau detection tests."""
    for basename, expected_interval in PLATEAU_TEST_DATA:
        print(f"Testing {basename}...")
        test_plateau_detection(basename, expected_interval)
        print(f"✓ {basename} passed\n")


if __name__ == "__main__":
    guiutils.enable_gui()
    all_tests()
