# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the shared execution path of operation contracts."""

# Test-only operations declare signatures they do not use.
# pylint: disable=unused-argument

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

import sigima
import sigima.objects
import sigima.params
import sigima.proc.image
import sigima.proc.signal
from sigima.config import options
from sigima.objects import SignalObj
from sigima.proc import contracts
from sigima.proc.contracts.alignment import (
    SMALLEST_GRID_LINEAR_HOLD,
    SOURCE_GRID_LINEAR,
    SOURCE_GRID_LINEAR_HOLD,
    XAlignmentError,
)
from sigima.proc.contracts.execution import (
    ExecutionError,
    PreparedExecution,
    default_roles,
    describe_call,
    execute,
    normalize_outputs,
    prepare_execution,
)
from sigima.proc.decorator import computation_function


@computation_function()
def no_contract(src: SignalObj, p: sigima.params.NormalizeParam) -> SignalObj:
    """Test-only function without operation contract."""
    return src.copy()


def _signal(x, y, title="s", dy=None) -> SignalObj:
    sig = sigima.objects.create_signal(title, np.asarray(x, float), np.asarray(y))
    if dy is not None:
        sig.dy = np.asarray(dy, float)
    return sig


def _contract(func) -> contracts.OperationContract:
    contract = contracts.contract_for_function(func)
    assert contract is not None
    return contract


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _same_signal(first: SignalObj, second: SignalObj) -> None:
    assert np.array_equal(first.xydata, second.xydata, equal_nan=True)
    assert first.xydata.dtype == second.xydata.dtype


def test_describe_qualified_call() -> None:
    """A qualified call is described with identity, rule, roles and parameters."""
    src = _signal([0, 1, 2], [1.0, 2.0, 4.0])
    param = sigima.params.NormalizeParam.create(method="maximum")
    description = describe_call(sigima.proc.signal.normalize, [src], param)
    assert description["operation"] == {
        "id": "sigima.signal.normalize",
        "contract_version": 1,
    }
    assert description["qualification"] == {
        "status": "qualified",
        "comparison": {"rule": "exact"},
    }
    assert description["pattern"] == "1_to_1" and description["roles"] == ["source"]
    assert description["parameters"] == {"method": "maximum"}
    assert not description["limits"] and not description["options"]
    assert not description["metadata_read"]
    assert description["implementation"] == {
        "package": "sigima",
        "version": sigima.__version__,
        "python_name": "sigima.proc.signal.processing.normalize",
    }
    assert _json(description) == description


def test_identity_is_kept_without_qualification() -> None:
    """Unqualified contracts and mismatched inputs keep the operation identity."""
    src = _signal([0, 1, 2], [1.0, 2.0, 4.0])
    description = describe_call(sigima.proc.signal.phase_spectrum, [src])
    assert description["operation"] == {
        "id": "sigima.signal.phase_spectrum",
        "contract_version": 1,
    }
    assert description["qualification"] == {
        "status": "not_qualified",
        "comparison": None,
    }
    assert description["parameters"] is not None and not description["parameters"]
    param = sigima.params.ConstantParam.create(value=2.0)
    constant = describe_call(sigima.proc.signal.addition_constant, [src], param)
    assert constant["qualification"]["comparison"] == {"rule": "exact"}
    assert constant["parameters"] == {"value": 2.0}
    image = sigima.objects.create_image("i", np.zeros((2, 2)))
    mismatch = describe_call(sigima.proc.signal.addition_constant, [image], param)
    assert mismatch["operation"] == {
        "id": "sigima.signal.addition_constant",
        "contract_version": 1,
    }
    assert mismatch["limits"] == ["inputs_do_not_match_contract"]
    bad = describe_call(sigima.proc.signal.addition_constant, [src], [param])
    assert bad["parameters"] is None and bad["limits"] == ["parameters_not_encoded"]
    sources = [_signal([0, 1], [1.0, 2.0]) for _ in range(3)]
    average = describe_call(sigima.proc.signal.average, sources)
    assert average["roles"] == ["sources"] * 3 and average["pattern"] == "n_to_1"


def test_describe_call_without_contract() -> None:
    """A function without contract is described without identity."""
    src = _signal([0, 1, 2], [1.0, 2.0, 4.0])
    param = sigima.params.NormalizeParam.create(method="maximum")
    description = describe_call(no_contract, [src], param)
    assert description["operation"] is None
    assert description["qualification"]["status"] == "no_contract"
    assert description["roles"] == ["source"]
    assert description["parameters"] == {"method": "maximum"}
    assert not description["context"]
    assert describe_call(no_contract, [src, src], roles=["a", "b"])["roles"] == [
        "a",
        "b",
    ]
    assert default_roles(2) == ["source", "operand"]
    assert default_roles(3) == ["sources"] * 3


def test_image_roi_convention_is_recorded_and_checked() -> None:
    """Calls on images with ROIs record the pixel convention; replays require it."""
    rule = {"rule": "sigima.image.roi.pixel_centers", "version": 1}
    assert dict(contracts.IMAGE_ROI_CONVENTION) == rule
    contract = _contract(sigima.proc.image.absolute)
    image = sigima.objects.create_image("i", np.arange(-32.0, 32.0).reshape(8, 8))
    # Without ROI, nothing is recorded nor required
    assert not describe_call(contract.function, [image])["context"]
    ordinary = prepare_execution(contract, [image], param=None)
    assert ordinary.context == {"x_alignment": None}
    prepare_execution(contract, [image], values={}, context={})
    # With a ROI, the rule is recorded by both the description and the preparation
    image.roi = sigima.objects.create_image_roi("rectangle", [1, 1, 4, 4])
    description = describe_call(contract.function, [image])
    assert description["context"] == {"roi_convention": rule}
    assert _json(description) == description
    ordinary = prepare_execution(contract, [image], param=None)
    assert ordinary.context == {"x_alignment": None, "roi_convention": rule}
    reference = ordinary.run()
    replay = prepare_execution(contract, [image], values={}, context=ordinary.context)
    assert np.array_equal(replay.run().data, reference.data)
    # Calls recorded without the rule (before Sigima 1.4) or with another one
    for recorded in (
        {"x_alignment": None},
        {"roi_convention": None},
        {"roi_convention": {"rule": "sigima.image.roi.pixel_centers", "version": 2}},
        {"roi_convention": {"rule": "other.rule", "version": 1}},
    ):
        with pytest.raises(ExecutionError) as info:
            prepare_execution(contract, [image], values={}, context=recorded)
        assert info.value.code == "unsupported_context"
    with pytest.raises(ExecutionError, match="before Sigima 1.4"):
        prepare_execution(contract, [image], values={}, context={})
    # An empty ROI selects every pixel, as before: nothing is required
    image.roi = sigima.objects.create_image_roi("rectangle", [1, 1, 4, 4])
    image.roi.single_rois.clear()
    assert not describe_call(contract.function, [image])["context"]
    # Any input image carrying ROIs (here the operand) makes the call depend on it
    source = sigima.objects.create_image("s", np.ones((8, 8)))
    operand = image.copy()
    operand.roi = sigima.objects.create_image_roi("circle", [4, 4, 2])
    difference = _contract(sigima.proc.image.difference)
    context = describe_call(difference.function, [source, operand])["context"]
    assert context == {"roi_convention": rule}
    # Signal ROIs are not concerned
    signal = _signal([0, 1, 2, 3], [-2.0, 0.0, 1.0, 4.0])
    signal.roi = sigima.objects.create_signal_roi([[0.5, 2.5]])
    normalize = _contract(sigima.proc.signal.normalize)
    assert not describe_call(normalize.function, [signal], None)["context"]
    assert contracts.image_roi_convention([signal]) is None


def _roi_params(count: int) -> list[sigima.objects.ROI2DParam]:
    """Rectangles whose bounds do not lie on pixel centers."""
    return [
        sigima.objects.ROI2DParam.create(
            geometry="rectangle", x0=1.5 + 3 * i, y0=2.5, dx=2.0, dy=3.0
        )
        for i in range(count)
    ]


@pytest.mark.parametrize(
    "func, param",
    [
        (sigima.proc.image.extract_roi, _roi_params(1)[0]),
        (sigima.proc.image.extract_rois, _roi_params(2)),
        (sigima.proc.image.erase, _roi_params(1)[0]),
        (sigima.proc.image.erase, _roi_params(2)),
        (sigima.proc.image.offset_correction, _roi_params(1)[0]),
    ],
)
def test_image_roi_parameters_depend_on_the_convention(func, param) -> None:
    """Operations taking image ROIs as parameters record and require the pixel
    convention, also on images without regions of interest (regression: a replay
    recorded before Sigima 1.4, which selected other pixels, was accepted)."""
    rule = dict(contracts.IMAGE_ROI_CONVENTION)
    contract = _contract(func)
    image = sigima.objects.create_image("i", np.arange(64.0).reshape(8, 8))
    assert image.roi is None
    assert contracts.image_roi_convention([image], contract) == rule
    description = _json(describe_call(func, [image], param))
    assert description["context"] == {"roi_convention": rule}
    assert not description["limits"]
    ordinary = prepare_execution(contract, [image], param=param)
    assert ordinary.context == {"x_alignment": None, "roi_convention": rule}
    reference = ordinary.run()
    values = description["parameters"]
    replay = prepare_execution(
        contract, [image], values=values, context=ordinary.context
    )
    assert np.array_equal(replay.run().data, reference.data)
    for recorded in (
        {},
        {"x_alignment": None},
        {"roi_convention": None},
        {"roi_convention": {"rule": "sigima.image.roi.pixel_centers", "version": 2}},
    ):
        with pytest.raises(ExecutionError) as info:
            prepare_execution(contract, [image], values=values, context=recorded)
        assert info.value.code == "unsupported_context"


def test_operations_taking_image_roi_parameters() -> None:
    """The operations depending on the convention through their parameters are
    detected from their contracts; signal ROI parameters are not concerned."""
    registry = contracts.build_contract_registry()
    found = {
        contract.operation_id
        for contract in registry.contracts
        if contracts.image_roi_convention([], contract) is not None
    }
    assert found == {
        "sigima.image.erase",
        "sigima.image.extract_roi",
        "sigima.image.extract_rois",
        "sigima.image.offset_correction",
    }
    signal = _signal([0, 1, 2, 3], [-2.0, 0.0, 1.0, 4.0])
    roi = sigima.objects.ROI1DParam.create(xmin=0.5, xmax=2.5)
    description = describe_call(sigima.proc.signal.extract_roi, [signal], roi)
    assert not description["context"]


def test_ordinary_and_replay_give_the_same_result() -> None:
    """Ordinary processing and replay share the same preparation."""
    src = _signal([0, 1, 2, 3], [-2.0, 0.0, 1.0, 4.0])
    contract = _contract(sigima.proc.signal.normalize)
    param = sigima.params.NormalizeParam.create(method="amplitude")
    ordinary = prepare_execution(contract, [src], param=param)
    assert ordinary.context == {"x_alignment": None}
    reference = ordinary.run()
    values = _json(describe_call(contract.function, [src], param)["parameters"])
    replay = prepare_execution(contract, [src], values=values, context=ordinary.context)
    _same_signal(replay.run(), reference)
    # A replay works on detached copies; ordinary processing on the inputs.
    assert ordinary.args[0] is src
    copy = replay.args[0]
    assert copy is not src and copy.roi is None and copy.title == src.title
    assert copy.metadata == src.metadata and _same_signal(copy, src) is None
    with pytest.raises(TypeError):
        prepare_execution(contract, [src])
    with pytest.raises(ExecutionError) as info:
        prepare_execution(contract, [src], values={"method": "Maximum"}, context={})
    assert info.value.code == "invalid_parameters"
    with pytest.raises(ExecutionError) as info:
        prepare_execution(contract, [src, src], param=param)
    assert info.value.code == "input_mismatch"


def test_source_grid_policy_is_applied_once() -> None:
    """The difference aligns its operand once, with the same refusals."""
    source = _signal([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    operand = _signal([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0, 3.0, 4.0])
    contract = _contract(sigima.proc.signal.difference)
    ordinary = prepare_execution(contract, [source, operand], param=None)
    assert ordinary.context["x_alignment"] == {
        "rule": SOURCE_GRID_LINEAR,
        "version": 1,
        "interpolated": True,
    }
    assert ordinary.args[0] is source and ordinary.args[1] is not operand
    assert operand.x.size == 5
    replay = prepare_execution(
        contract, [source, operand], values={}, context=_json(ordinary.context)
    )
    _same_signal(replay.run(), ordinary.run())
    for record in (None, {"rule": SOURCE_GRID_LINEAR_HOLD, "version": 1}):
        with pytest.raises(XAlignmentError) as info:
            prepare_execution(
                contract, [source, operand], values={}, context={"x_alignment": record}
            )
        assert info.value.code == "unknown_rule"
    uncertain = _signal(operand.x, operand.y, dy=np.full(5, 0.1))
    for context in (None, ordinary.context):
        with pytest.raises(XAlignmentError) as info:
            if context is None:
                prepare_execution(contract, [source, uncertain], param=None)
            else:
                prepare_execution(
                    contract, [source, uncertain], values={}, context=context
                )
        assert info.value.code == "uncertainty"


def test_historical_source_grid_policy() -> None:
    """Two-input operations keep the historical alignment, as declared."""
    contract = _contract(sigima.proc.signal.division)
    assert contract.x_alignment == SOURCE_GRID_LINEAR_HOLD
    source = _signal([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    operand = _signal([0.5, 1.5, 2.5], [2.0, 4.0, 8.0], dy=[0.1, 0.1, 0.1])
    prepared = prepare_execution(contract, [source, operand], param=None)
    assert prepared.context["x_alignment"] == {
        "rule": SOURCE_GRID_LINEAR_HOLD,
        "version": 1,
        "interpolated": True,
    }
    aligned = prepared.args[1]
    # Constant extension at both ends, uncertainties dropped, inputs untouched.
    assert np.array_equal(aligned.x, source.x)
    assert np.array_equal(aligned.y, [2.0, 3.0, 6.0, 8.0])
    assert aligned.dy is None and operand.dy is not None
    assert aligned.title == "s (interpolated)"
    same = prepare_execution(contract, [source, source], param=None)
    assert same.args[1] is source and not same.context["x_alignment"]["interpolated"]
    for name in ("interpolate", "replace_x_by_other_y", "evaluate_fit"):
        assert _contract(getattr(sigima.proc.signal, name)).x_alignment is None


def test_historical_smallest_grid_policy() -> None:
    """List operations align on the smallest grid, as both editions did."""
    contract = _contract(sigima.proc.signal.average)
    assert contract.x_alignment == SMALLEST_GRID_LINEAR_HOLD
    first = _signal([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0], dy=[1, 1, 1, 1])
    small = _signal([0.5, 1.5, 2.5], [1.0, 1.0, 1.0])
    tie = _signal([0.0, 1.0, 2.0], [3.0, 3.0, 3.0])
    same = _signal([0.5, 1.5, 2.5], [5.0, 5.0, 5.0], dy=[2, 2, 2])
    prepared = prepare_execution(contract, [first, small, tie, same], param=None)
    record = prepared.context["x_alignment"]
    assert record == {
        "rule": SMALLEST_GRID_LINEAR_HOLD,
        "version": 1,
        "interpolated": [0, 2],
        "target": 1,
    }
    (aligned,) = prepared.args
    assert aligned[1] is small and aligned[3] is same
    assert np.array_equal(aligned[0].y, [1.0, 3.0, 5.0]) and aligned[0].dy is None
    assert np.array_equal(aligned[2].y, [3.0, 3.0, 3.0])
    result = prepared.run()
    replay = prepare_execution(
        contract, [first, small, tie, same], values={}, context=_json(prepared.context)
    )
    _same_signal(replay.run(), result)
    to_image = _contract(sigima.proc.signal.signals_to_image)
    param = sigima.params.SignalsToImageParam()
    image = prepare_execution(to_image, [first, small], param=param).run()
    assert 3 in image.data.shape


def test_global_options_are_recorded_and_restored() -> None:
    """Options read by a computation are recorded, then restored for a replay."""
    contract = _contract(sigima.proc.signal.convolution)
    src = _signal(np.arange(8.0), [0, 0, 1, 2, 1, 0, 0, 0])
    kernel = _signal(np.arange(8.0), [0, 0, 0, 1, 1, 0, 0, 0])
    with options.auto_normalize_kernel.context(False):
        ordinary = prepare_execution(contract, [src, kernel], param=None)
        assert ordinary.context["options"] == {"auto_normalize_kernel": False}
        reference = ordinary.run()
        # The description of the call records the same values.
        description = describe_call(contract.function, [src, kernel])
        assert description["options"] == ordinary.context["options"]
    image = sigima.objects.create_image("i", np.ones((3, 3)))
    image_conv = describe_call(sigima.proc.image.convolution, [image, image])
    assert set(image_conv["options"]) == {"auto_normalize_kernel"}
    before = options.auto_normalize_kernel.get()
    with options.auto_normalize_kernel.context(True):
        replay = prepare_execution(
            contract, [src, kernel], values={}, context=_json(ordinary.context)
        )
        _same_signal(replay.run(), reference)
        assert options.auto_normalize_kernel.get() is True
        normalized = prepare_execution(contract, [src, kernel], param=None).run()
    assert not np.array_equal(normalized.y, reference.y)
    assert options.auto_normalize_kernel.get() == before
    with pytest.raises(ExecutionError) as info:
        prepare_execution(
            contract, [src, kernel], values={}, context={"x_alignment": None}
        )
    assert info.value.code == "unsupported_context"


def test_prepared_execution_is_picklable() -> None:
    """A prepared execution runs after a pickle round trip (process pools)."""
    contract = _contract(sigima.proc.signal.extract_rois)
    src = _signal(np.arange(10.0), np.arange(10.0) ** 2)
    rois = [sigima.objects.ROI1DParam.create(xmin=1.0, xmax=3.0)]
    prepared = prepare_execution(
        contract, [src], values=_json(contract.encode_parameters([rois])), context={}
    )
    copy = pickle.loads(pickle.dumps(prepared))
    assert isinstance(copy, PreparedExecution)
    _same_signal(execute(copy), prepared.run())


def test_metadata_read_are_described() -> None:
    """The metadata read by a computation are listed by input position."""
    x = np.linspace(0.0, 1.0, 11)
    data = _signal(x, 2.0 * x + 1.0)
    fit = sigima.proc.signal.linear_fit(data)
    other = _signal(np.linspace(0.0, 2.0, 5), np.zeros(5))
    description = describe_call(sigima.proc.signal.evaluate_fit, [fit, other])
    assert description["roles"] == ["source", "operand"]
    assert description["metadata_read"] == [
        {"role": "source", "position": 0, "key": "fit_params"}
    ]
    assert _json(description) == description
    mismatch = describe_call(sigima.proc.signal.evaluate_fit, [fit])
    assert not mismatch["metadata_read"]


def test_normalize_outputs() -> None:
    """Outputs are returned by role; an optional result may be absent."""
    src = _signal(np.linspace(-5, 5, 101), np.exp(-(np.linspace(-5, 5, 101) ** 2)))
    fwhm = _contract(sigima.proc.signal.fwhm)
    assert normalize_outputs(fwhm, None) == [("result", None)]
    result = prepare_execution(fwhm, [src], param=sigima.params.FWHMParam()).run()
    assert normalize_outputs(fwhm, result) == [("result", result)]
    stats = _contract(sigima.proc.signal.stats)
    table = prepare_execution(stats, [src], param=None).run()
    assert normalize_outputs(stats, table)[0][1] is table
    for bad in (None, src):
        with pytest.raises(ExecutionError) as info:
            normalize_outputs(stats, bad)
        assert info.value.code == "unexpected_output"
    image_contract = _contract(sigima.proc.image.line_profile)
    assert image_contract.outputs[0].kind == "signal"
    with pytest.raises(ExecutionError):
        normalize_outputs(
            image_contract, sigima.objects.create_image("i", np.ones((2, 2)))
        )
