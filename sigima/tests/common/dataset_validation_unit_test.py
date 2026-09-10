# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for parameter DataSet validation helpers."""

from __future__ import annotations

import guidata.dataset as gds
import pytest
from guidata.config import ValidationMode, temporary_validation_mode

from sigima.validation import validate_dataset


class PlainParam(gds.DataSet):
    """DataSet without relational validation."""

    value = gds.FloatItem("Value", default=1.0)


class RecordingParam(gds.DataSet):
    """DataSet recording relational validation context."""

    value = gds.FloatItem("Value", default=1.0)

    def validate_parameters(self, *context: object) -> None:
        """Record validation context."""
        self.validation_context = context


class RejectingParam(gds.DataSet):
    """DataSet rejecting an invalid relation."""

    lower = gds.FloatItem("Lower", default=1.0)
    upper = gds.FloatItem("Upper", default=0.0)

    def validate_parameters(self, *context: object) -> None:
        """Reject reversed bounds."""
        del context
        if self.lower > self.upper:
            raise ValueError("lower must be less than or equal to upper")


def test_validate_dataset_without_hook() -> None:
    """DataSets without a validation hook are accepted."""
    validate_dataset(PlainParam())


def test_validate_dataset_passes_context() -> None:
    """Execution context is forwarded unchanged to the validation hook."""
    param = RecordingParam()
    context = object()

    validate_dataset(param, context, "extra")

    assert param.validation_context == (context, "extra")


def test_validate_dataset_propagates_value_error() -> None:
    """Relational validation errors propagate to the caller."""
    with pytest.raises(ValueError, match="lower must be less"):
        validate_dataset(RejectingParam())


def test_inactive_parameter_values_survive_json_round_trip() -> None:
    """Conditional values remain unchanged through DataSet JSON conversion."""
    from sigima.proc.image.geometry import Resampling2DParam
    from sigima.proc.signal.processing import Resampling1DParam, WindowingParam

    params_and_values = (
        (
            Resampling1DParam.create(mode="nbpts", xmin=0.0, xmax=1.0, nbpts=3, dx=0.0),
            ("mode", "nbpts", "dx", 0.0),
        ),
        (
            Resampling2DParam.create(mode="dxy", dx=1.0, dy=1.0, width=0, height=-1),
            ("mode", "dxy", "width", 0, "height", -1),
        ),
        (
            WindowingParam.create(method="hamming", sigma=0.0),
            ("sigma", 0.0),
        ),
    )

    for param, expected in params_and_values:
        restored = gds.json_to_dataset(gds.dataset_to_json(param))
        for name, value in zip(expected[::2], expected[1::2]):
            assert getattr(restored, name) == value


@pytest.mark.gui
@pytest.mark.parametrize("validation_mode", list(ValidationMode))
def test_parameter_bounds_qt_forms(validation_mode: ValidationMode) -> None:
    """Signed ROI and grid values remain editable in real DataSet dialogs."""
    import numpy as np
    from guidata.dataset.qtwidgets import DataSetEditDialog

    from sigima.objects import ImageObj
    from sigima.objects.image.roi import RectangularROI
    from sigima.proc.image import GridParam
    from sigima.tests import guiutils

    image = ImageObj(title="Reversed axes")
    image.data = np.zeros((12, 12), dtype=float)
    image.set_uniform_coords(-1.0, -2.0, 10.0, 20.0)

    with temporary_validation_mode(validation_mode):
        roi_param = RectangularROI([2, 3, 4, 2], indices=True, inverse=True).to_param(
            image, 0
        )
        grid_param = GridParam.create(direction="col", cols=-3)

        values_before = (
            (roi_param.dx, roi_param.dy, roi_param.inverse),
            (grid_param.cols, grid_param.direction),
        )
        with guiutils.lazy_qt_app_context(force=True) as app:
            assert app is not None
            dialogs = (DataSetEditDialog(roi_param), DataSetEditDialog(grid_param))
            assert all(dialog.edit_layout for dialog in dialogs)
            for dialog in dialogs:
                assert all(layout.check_all_values() for layout in dialog.edit_layout)
                for layout in dialog.edit_layout:
                    layout.accept_changes()
                dialog.close()

        assert values_before == (
            (roi_param.dx, roi_param.dy, roi_param.inverse),
            (grid_param.cols, grid_param.direction),
        )
