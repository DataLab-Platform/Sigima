# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Spectral helpers shared by the signal and image Fourier analysis modules.
"""

from __future__ import annotations

import numpy as np


def to_decibel(values: np.ndarray, factor: float) -> np.ndarray:
    """Convert non-negative spectral values to a decibel scale.

    Spectral values are exactly zero wherever the data has no content at a given
    frequency -- synthetic data, zero padding or a notch filter all produce them.
    Taking the logarithm as-is yields ``-inf`` samples, which propagate to
    autoscaling, statistics and file export, and emit a divide-by-zero warning.
    Values are therefore floored at the machine epsilon of the peak value, i.e. about
    313 dB below the peak for the amplitude convention: far below any physically
    meaningful level, yet finite. Being relative to the peak, the floor does not
    depend on the unit of the data.

    Args:
        values: Non-negative spectral values.
        factor: 20 for amplitude quantities, 10 for power quantities.

    Returns:
        Values expressed in decibel (``-inf`` everywhere for an identically zero
        spectrum).
    """
    peak = np.max(values)
    if peak <= 0.0:
        # Identically zero spectrum: there is no dynamic range to represent
        return np.full(values.shape, -np.inf)
    return factor * np.log10(np.maximum(values, peak * np.finfo(float).eps))
