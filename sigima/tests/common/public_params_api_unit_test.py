"""Unit tests for the public parameter API."""
# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

from __future__ import annotations

import sigima.params
import sigima.proc.image
from sigima.proc.image.detection import DetectionROIParam


def test_detection_roi_param_public_exports() -> None:
    """Check DetectionROIParam public exports."""
    assert "DetectionROIParam" in sigima.proc.image.__all__
    assert "DetectionROIParam" in sigima.params.__all__
    assert sigima.proc.image.DetectionROIParam is DetectionROIParam
    assert sigima.params.DetectionROIParam is DetectionROIParam
