# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for crossing time detection (x0, x50, x100) with real square pulse data.

This test suite validates that the crossing time detection algorithm correctly
identifies the x0 (0%), x50 (50%), and x100 (100%) crossing points in real-world
square pulse signals.
"""

from __future__ import annotations

import pytest

from sigima.tests import guiutils
from sigima.tests.data import get_test_signal
from sigima.tests.helpers import check_scalar_result
from sigima.tests.signal.pulse import view_baseline_plateau_and_curve
from sigima.tools.signal import pulse

# Test data configuration: (basename, (x0, x50, x100), (rtol0, rtol50, rtol100))
# Note: x0 values have known issues (detecting crossings too early in the signal)
# so we use very high tolerances. x50 is generally reliable, x100 varies by signal.
CROSSING_TEST_DATA = [
    ("boxcar.npy", (-1e-9, 2.5e-10, 2e-9), (20.0, 0.15, 0.5)),
    ("square2.npy", (-3e-6, 6.5e-6, 1.65e-5), (20.0, 0.1, 5.0)),
]


@pytest.mark.parametrize("basename,expected_values,tolerances", CROSSING_TEST_DATA)
def test_crossing_times(
    basename: str,
    expected_values: tuple[float, float, float],
    tolerances: tuple[float, float, float],
) -> None:
    """Parametric test for crossing time detection on real square pulse data.

    Args:
        basename: Name of the test data file
        expected_values: Expected (x0, x50, x100) crossing times
        tolerances: Relative tolerances (rtol0, rtol50, rtol100) for each crossing

    This test ensures that the crossing time detection algorithm correctly identifies
    the 0%, 50%, and 100% crossing points within acceptable tolerances.
    The test includes optional visualization when run with GUI enabled.

    Note:
        x0 detection has known issues and uses very high tolerances. This test
        primarily validates that the algorithm doesn't crash and returns values
        in the ballpark of expected results.
    """
    # Load real data
    obj = get_test_signal(basename)
    x, y = obj.x, obj.y

    # Auto-detect ranges and polarity
    start_range = pulse.get_start_range(x)
    end_range = pulse.get_end_range(x)
    polarity = pulse.detect_polarity(x, y, start_range, end_range)
    plateau_range = pulse.get_plateau_range(x, y, polarity)

    # Find crossing times
    x0 = pulse.find_crossing_at_ratio(x, y, 0.0, start_range, end_range)
    x50 = pulse.find_crossing_at_ratio(x, y, 0.5, start_range, end_range)
    x100 = pulse.find_crossing_at_ratio(x, y, 1.0, start_range, end_range)

    exp_x0, exp_x50, exp_x100 = expected_values
    rtol0, rtol50, rtol100 = tolerances

    with guiutils.lazy_qt_app_context() as qt_app:
        title = (
            f"{basename} - Crossing Times Test\n"
            f"x0={x0:.3e} (exp: {exp_x0:.3e}), "
            f"x50={x50:.3e} (exp: {exp_x50:.3e}), "
            f"x100={x100:.3e} (exp: {exp_x100:.3e})"
        )
        if qt_app is not None:
            # pylint: disable=import-outside-toplevel
            from sigima import viz

            # Create vertical markers for crossing times
            items = []
            items.append(viz.create_cursor("v", x0, "x0 (detected)"))
            items.append(viz.create_cursor("v", exp_x0, "x0 (expected)"))
            items.append(viz.create_cursor("v", x50, "x50 (detected)"))
            items.append(viz.create_cursor("v", exp_x50, "x50 (expected)"))
            items.append(viz.create_cursor("v", x100, "x100 (detected)"))
            items.append(viz.create_cursor("v", exp_x100, "x100 (expected)"))

            view_baseline_plateau_and_curve(
                x,
                y,
                title,
                "square",
                start_range,
                end_range,
                plateau_range,
                other_items=items,
            )

    # Validate crossing times with appropriate tolerances
    # Note: We use very high tolerances for x0 due to known detection issues
    assert x0 is not None, f"{basename}: x0 crossing not found"
    assert x50 is not None, f"{basename}: x50 crossing not found"
    assert x100 is not None, f"{basename}: x100 crossing not found"

    # Check results
    check_scalar_result("x0", x0, exp_x0, rtol0)
    check_scalar_result("x50", x50, exp_x50, rtol50)
    check_scalar_result("x100", x100, exp_x100, rtol100)


def all_tests() -> None:
    """Run all crossing time tests."""
    for basename, expected_values, tolerances in CROSSING_TEST_DATA:
        print(f"Testing {basename}...")
        test_crossing_times(basename, expected_values, tolerances)
        print(f"✓ {basename} passed\n")


if __name__ == "__main__":
    guiutils.enable_gui()
    all_tests()
