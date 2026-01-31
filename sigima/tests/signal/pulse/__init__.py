# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for pulse analysis functions.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from sigima.tools.signal import pulse


def view_baseline_plateau_and_curve(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    signal_type: Literal["step", "square"],
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    plateau_range: tuple[float, float] | None = None,
    vcursors: dict[str, float] | None = None,
    other_items: list | None = None,
) -> None:
    """Helper function to visualize signal with baselines and plateau.

    Args:
        x: X data.
        y: Y data.
        title: Title for the plot.
        signal_type: Signal shape type
        start_range: Start baseline range.
        end_range: End baseline range.
        plateau_range: Plateau range for square signals (optional).
        vcursors: Dictionary of vertical cursors to display (optional).
        other_items: Additional items to display (optional).
    """
    # pylint: disable=import-outside-toplevel
    from sigima import viz

    ys = pulse.get_range_mean_y(x, y, start_range)
    ye = pulse.get_range_mean_y(x, y, end_range)
    xs0, xs1 = start_range
    xe0, xe1 = end_range
    items = [
        viz.create_curve(x, y, title="Noisy signal"),
        viz.create_segment(xs0, ys, xs1, ys, "Start baseline"),
        viz.create_segment(xe0, ye, xe1, ye, "End baseline"),
    ]
    if signal_type == "square":
        if plateau_range is None:
            polarity = pulse.detect_polarity(x, y, start_range, end_range)
            plateau_range = pulse.get_plateau_range(x, y, polarity)
        xp0, xp1 = plateau_range
        yp = pulse.get_range_mean_y(x, y, plateau_range)
        items.append(viz.create_segment(xp0, yp, xp1, yp, "Plateau"))
    if vcursors is not None:
        for label, xt in vcursors.items():
            items.append(viz.create_cursor("v", xt, label))
    if other_items is not None:
        items.extend(other_items)

    viz.view_curve_items(items, title=title)


def view_pulse_features(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    signal_type: Literal["step", "square"],
    features: pulse.PulseFeatures,
) -> None:
    """Helper function to visualize pulse features.

    Args:
        x: X data.
        y: Y data.
        title: Title for the plot.
        signal_type: Signal shape type
        features: Extracted pulse features.
    """
    # pylint: disable=import-outside-toplevel
    from sigima import viz

    params_text = "<br>".join(
        [
            f"<b>Extracted {signal_type} parameters:</b>",
            f"Polarity: {features.polarity}",
            f"Amplitude: {features.amplitude}",
            f"Rise time: {features.rise_time}",
            f"Fall time: {features.fall_time}",
            f"FWHM: {features.fwhm}",
            f"Offset: {features.offset}",
            f"T50: {features.x50}",
            f"X100: {features.x100}",
            f"Foot duration: {features.foot_duration}",
        ]
    )
    view_baseline_plateau_and_curve(
        x,
        y,
        title,
        signal_type,
        [features.xstartmin, features.xstartmax],
        [features.xendmin, features.xendmax],
        plateau_range=None,
        other_items=[viz.create_label(params_text)],
    )
