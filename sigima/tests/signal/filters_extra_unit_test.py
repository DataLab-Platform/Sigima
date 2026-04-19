# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Additional unit tests for :mod:`sigima.proc.signal.filtering`.

Covers ``update_from_obj`` default-cutoff branches for the high-pass,
low-pass, band-pass and band-stop filter parameter classes.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects import SignalObj
from sigima.proc.signal.filtering import (
    BandPassFilterParam,
    BandStopFilterParam,
    HighPassFilterParam,
    LowPassFilterParam,
)


def _make_signal(n: int = 200) -> SignalObj:
    """Build a two-tone signal (5 Hz + 50 Hz) suitable for filter tests."""
    sig = SignalObj()
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * 5 * x) + 0.3 * np.sin(2 * np.pi * 50 * x)
    sig.set_xydata(x, y)
    return sig


@pytest.mark.parametrize(
    "param_cls, two_cutoffs",
    [
        (HighPassFilterParam, False),
        (LowPassFilterParam, False),
        (BandPassFilterParam, True),
        (BandStopFilterParam, True),
    ],
)
def test_filter_update_from_obj_default_cutoffs(param_cls, two_cutoffs) -> None:
    """When cut-off frequencies are left ``None`` on a filter parameter,
    ``update_from_obj`` must derive sensible positive defaults from the
    signal's sampling rate (covers all 4 filter param classes)."""
    sig = _make_signal()
    p = param_cls()
    p.cut0 = None
    if two_cutoffs:
        p.cut1 = None
    p.update_from_obj(sig)
    assert p.cut0 is not None and p.cut0 > 0
    if two_cutoffs:
        assert p.cut1 is not None and p.cut1 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
