# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for operation contracts and parameter encoding."""

# Test-only operations declare signatures they do not use.
# pylint: disable=unused-argument

from __future__ import annotations

import inspect
import json
import math
import pickle
import sys

import guidata.dataset as gds
import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image
import sigima.proc.signal
from sigima.enums import NormalizationMethod
from sigima.objects import ImageObj, SignalObj
from sigima.objects.scalar.geometry import GeometryResult
from sigima.objects.scalar.table import TableResult
from sigima.proc import contracts
from sigima.proc.alignment import SOURCE_GRID_LINEAR
from sigima.proc.annotations import (
    AnnotationError,
    resolve_annotation,
    sequence_item,
    split_optional,
)
from sigima.proc.contract_types import (
    EXACT,
    ComparisonRule,
    OutputTolerance,
    ReplayQualification,
    tolerance,
)
from sigima.proc.decorator import (
    ComputationMetadata,
    computation_function,
    find_computation_functions,
    get_computation_metadata,
    is_computation_function,
)


class DummyParam(gds.DataSet):
    """Parameters exercising every encoded item type."""

    factor = gds.FloatItem("Factor", default=1.0)
    count = gds.IntItem("Count", default=2)
    enabled = gds.BoolItem("Enabled", default=True)
    label = gds.StringItem("Label", default="a")
    mode = gds.ChoiceItem("Mode", ((0, "zero"), (1, "one")), default=0)
    method = gds.ChoiceItem("Method", NormalizationMethod)
    hidden = gds.FloatItem("Hidden", default=0.0).set_prop("data", transient=True)


@computation_function(
    operation_id="test.signal.dummy", contract_version=2, aliases=("test.signal.old",)
)
def dummy(src: SignalObj, p: DummyParam) -> SignalObj:
    """Test-only signal 1-to-1 operation."""
    return src.copy()


@computation_function(operation_id="test.signal.twin", contract_version=1)
def twin(src: SignalObj, p: DummyParam) -> SignalObj:
    """Test-only operation sharing nothing with ``dummy``."""
    return src.copy()


@computation_function(operation_id="test.signal.dummy", contract_version=1)
def duplicate(src: SignalObj) -> SignalObj:
    """Test-only operation reusing ``dummy``'s identifier."""
    return src.copy()


@computation_function(operation_id="test.signal.pair", contract_version=1)
def pair(src1: SignalObj, src2: SignalObj, p: DummyParam) -> SignalObj:
    """Test-only signal 2-to-1 operation."""
    return src1.copy()


@computation_function(operation_id="test.signal.triple", contract_version=1)
def triple(src1: SignalObj, src2: SignalObj, src3: SignalObj) -> SignalObj:
    """Test-only 3-to-1 operation without declared roles: ambiguous."""
    return src1.copy()


@computation_function(
    operation_id="test.signal.triple_roles",
    contract_version=1,
    roles=("reference", "measure", "background"),
)
def triple_roles(src1: SignalObj, src2: SignalObj, src3: SignalObj) -> SignalObj:
    """Test-only 3-to-1 operation with declared roles."""
    return src1.copy()


@computation_function(operation_id="test.image.average", contract_version=1)
def image_average(src_list: list[ImageObj], p: DummyParam | None = None) -> ImageObj:
    """Test-only image n-to-1 operation with an optional parameter set."""
    return src_list[0].copy()


@computation_function(operation_id="test.image.profile", contract_version=1)
def image_profile(src: ImageObj) -> SignalObj:
    """Test-only image-to-signal conversion."""
    return sigima.objects.create_signal("p", np.arange(3.0), np.zeros(3))


@computation_function(operation_id="test.signal.detect", contract_version=1)
def detect(src: SignalObj, p: DummyParam) -> GeometryResult | None:
    """Test-only analysis with an optional result."""
    return None


@computation_function(operation_id="test.image.table", contract_version=1)
def image_table(src: ImageObj) -> TableResult:
    """Test-only analysis returning a table."""
    raise NotImplementedError


@computation_function(operation_id="test.signal.rois", contract_version=1)
def rois(
    src: SignalObj, params: sigima.objects.ROI1DParam | list[sigima.objects.ROI1DParam]
) -> SignalObj:
    """Test-only operation taking one or several ROI parameter sets."""
    return src.copy()


@computation_function(
    operation_id="test.signal.aligned",
    contract_version=3,
    x_alignment=SOURCE_GRID_LINEAR,
    qualification=ReplayQualification(contract_version=2),
)
def aligned(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Test-only aligned operation whose qualification is outdated."""
    return src1.copy()


@computation_function(
    operation_id="test.signal.misaligned",
    contract_version=1,
    x_alignment=SOURCE_GRID_LINEAR,
)
def misaligned(src: SignalObj) -> SignalObj:
    """Test-only 1-to-1 operation declaring a 2-to-1 policy."""
    return src.copy()


@computation_function(
    operation_id="test.signal.unknown_policy",
    contract_version=1,
    x_alignment="test.unknown.policy",
)
def unknown_policy(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Test-only operation declaring an unknown policy."""
    return src1.copy()


@computation_function(
    operation_id="test.signal.unknown_option",
    contract_version=1,
    reads_options=("no_such_option",),
)
def unknown_option(src: SignalObj) -> SignalObj:
    """Test-only operation declaring an unknown global option."""
    return src.copy()


@computation_function(operation_id="test.signal.bad_order", contract_version=1)
def bad_order(p: list[DummyParam], src: SignalObj) -> SignalObj:
    """Test-only operation with parameters before its input."""
    return src.copy()


@computation_function(operation_id="test.signal.bad_return", contract_version=1)
def bad_return(src: SignalObj) -> float:
    """Test-only operation with an unsupported return type."""
    return 0.0


@computation_function(operation_id="test.signal.bad_argument", contract_version=1)
def bad_argument(src: SignalObj, factor: float) -> SignalObj:
    """Test-only operation with an unsupported argument type."""
    return src.copy()


@computation_function(
    operation_id="test.signal.bad_roles", contract_version=1, roles=("a", "b")
)
def bad_roles(src: SignalObj) -> SignalObj:
    """Test-only operation declaring too many roles."""
    return src.copy()


def _f0() -> SignalObj:
    return sigima.objects.create_signal(
        "signal",
        np.array([0.0, 0.25, 0.5, 0.75]),
        np.array([-2.0, 0.0, 1.0, 4.0]),
        units=("s", ""),
    )


def test_declared_operation_ids() -> None:
    """Every exported computation function declares its identity, by convention.

    Identifiers are written next to each function; the convention is
    ``sigima.<family>.<exported name>``, version 1 at the first declaration.
    Exceptions to the convention must be listed here explicitly.
    """
    exceptions: dict[str, str] = {}
    registry = contracts.build_contract_registry()
    families = (("signal", sigima.proc.signal), ("image", sigima.proc.image))
    count = 0
    for family, module in families:
        for name in module.__all__:
            func = getattr(module, name)
            if not (inspect.isfunction(func) and is_computation_function(func)):
                continue
            count += 1
            metadata = get_computation_metadata(func)
            expected = exceptions.get(f"{family}.{name}", f"sigima.{family}.{name}")
            assert metadata.operation_id == expected
            assert registry.for_function(func) is not None
    assert len(registry.contracts) == count


def test_catalogue_contracts() -> None:
    """The whole catalogue is described by generic contracts."""
    registry = contracts.build_contract_registry()
    patterns = {contract.pattern for contract in registry.contracts}
    assert patterns == {"1_to_1", "2_to_1", "n_to_1", "1_to_0"}
    signals_to_image = registry.for_function(sigima.proc.signal.signals_to_image)
    assert signals_to_image.pattern == "n_to_1"
    assert signals_to_image.inputs == (contracts.Role("sources", "signal", None),)
    assert signals_to_image.outputs == (contracts.Role("result", "image"),)
    profile = registry.for_function(sigima.proc.image.line_profile)
    assert profile.pattern == "1_to_1" and profile.outputs[0].kind == "signal"
    fwhm = registry.for_function(sigima.proc.signal.fwhm)
    assert fwhm.pattern == "1_to_0"
    assert fwhm.outputs == (contracts.Role("result", "geometry", optional=True),)
    stats = registry.for_function(sigima.proc.signal.stats)
    assert stats.outputs == (contracts.Role("result", "table"),)
    rois = registry.for_function(sigima.proc.signal.extract_rois)
    assert rois.paramclass is None
    assert rois.parameters == (
        contracts.ParameterSpec("params", sigima.objects.ROI1DParam, "list"),
    )
    erase = registry.for_function(sigima.proc.image.erase)
    assert erase.parameters[0].container == "single_or_list"
    fft = registry.for_function(sigima.proc.signal.fft)
    assert fft.parameters[0].optional and fft.paramclass is sigima.params.FFTParam
    tophat = registry.for_function(sigima.proc.image.denoise_tophat)
    assert tophat.paramclass is sigima.params.MorphologyParam
    for func in (sigima.proc.signal.convolution, sigima.proc.image.deconvolution):
        assert registry.for_function(func).reads_options == ("auto_normalize_kernel",)
    qualified = sorted(c.operation_id for c in registry.contracts if c.qualified)
    assert qualified == ["sigima.signal.difference", "sigima.signal.normalize"]


def test_default_registry_needs_no_test_dependency(monkeypatch) -> None:
    """The default registry imports only public runtime modules (no pytest)."""
    declared = [
        getattr(__import__(modname, fromlist=[name]), name)
        for modname, name, _doc in find_computation_functions()
    ]
    declared = [f for f in declared if get_computation_metadata(f).operation_id]
    monkeypatch.setitem(sys.modules, "_pytest", None)
    monkeypatch.setitem(sys.modules, "_pytest.mark", None)
    monkeypatch.delitem(sys.modules, "sigima.proc.validation", raising=False)
    registry = contracts.build_contract_registry()
    assert "sigima.proc.validation" not in sys.modules
    assert declared
    assert all(registry.for_function(func) is not None for func in declared)


def test_normalize_contract() -> None:
    """The normalization contract is qualified, signal 1-to-1, version 1."""
    contract = contracts.get_operation_contract("sigima.signal.normalize", 1)
    assert contract.function is sigima.proc.signal.normalize
    assert contract.pattern == "1_to_1" and contract.qualified
    assert contract.inputs == (contracts.Role("source", "signal"),)
    assert contract.outputs == (contracts.Role("result", "signal"),)
    assert contract.paramclass is sigima.params.NormalizeParam
    assert contract.parameter_schema_version == 2
    assert contracts.contract_for_function(sigima.proc.signal.normalize) is contract
    constant = contracts.contract_for_function(sigima.proc.signal.addition_constant)
    assert constant.operation_id == "sigima.signal.addition_constant"
    assert not constant.qualified and constant.qualification_status == "not_qualified"
    assert constant.comparison is None
    assert contract.comparison is contracts.EXACT
    with pytest.raises(contracts.UnknownOperationError):
        contracts.get_operation_contract("sigima.signal.unknown", 1)
    with pytest.raises(contracts.IncompatibleContractError):
        contracts.get_operation_contract("sigima.signal.normalize", 2)
    assert contract.x_alignment is None


def test_difference_contract() -> None:
    """The difference contract is qualified, signal 2-to-1, with X alignment."""
    contract = contracts.get_operation_contract("sigima.signal.difference", 1)
    assert contract.function is sigima.proc.signal.difference
    assert contract.pattern == "2_to_1" and contract.qualified
    assert contract.inputs == (
        contracts.Role("source", "signal"),
        contracts.Role("operand", "signal"),
    )
    assert contract.outputs == (contracts.Role("result", "signal"),)
    assert contract.paramclass is None
    assert contract.x_alignment == "sigima.signal.x_alignment.source_grid_linear"
    assert contracts.contract_for_function(sigima.proc.signal.difference) is contract
    assert contracts.parameters_from_values(contract, {}) is None


def test_signature_calls_and_pickling_unchanged() -> None:
    """Decorated signature, both calling styles and pickling are preserved."""
    func = sigima.proc.signal.normalize
    assert list(inspect.signature(func).parameters) == ["src", "p", "method"]
    src = _f0()
    by_param = func(src, sigima.params.NormalizeParam.create(method="amplitude"))
    # pylint: disable-next=no-value-for-parameter
    by_kwarg = func(src, method="amplitude")
    assert np.array_equal(by_param.y, by_kwarg.y)
    assert pickle.loads(pickle.dumps(func)) is func
    metadata = get_computation_metadata(func)
    assert ComputationMetadata(**metadata.__dict__) == metadata
    legacy = ComputationMetadata("name", "doc")
    assert (legacy.operation_id, legacy.contract_version, legacy.aliases) == (
        None,
        None,
        (),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operation_id": "Bad.ID", "contract_version": 1},
        {"operation_id": "a..b", "contract_version": 1},
        {"operation_id": "sigima.signal.x"},
        {"operation_id": "sigima.signal.x", "contract_version": 0},
        {"operation_id": "sigima.signal.x", "contract_version": True},
        {"contract_version": 1},
        {"aliases": ("sigima.signal.y",)},
        {"operation_id": "a.b", "contract_version": 1, "aliases": ("a.b",)},
        {"operation_id": "a.b", "contract_version": 1, "aliases": "a.c"},
    ],
)
def test_decorator_rejects_invalid_identity(kwargs) -> None:
    """Identity arguments are validated when the decorator is created."""
    with pytest.raises(ValueError):
        computation_function(**kwargs)


def test_registry_aliases_and_errors() -> None:
    """A test-only registry resolves aliases and rejects inconsistencies."""
    registry = contracts.build_contract_registry([dummy, twin])
    assert registry.get("test.signal.old", 2).function is dummy
    assert registry.get("test.signal.dummy", 2).paramclass is DummyParam
    assert not registry.get("test.signal.dummy", 2).qualified
    assert registry.for_function(twin).operation_id == "test.signal.twin"
    with pytest.raises(contracts.ContractDeclarationError):
        contracts.build_contract_registry([dummy, duplicate])
    with pytest.raises(contracts.ContractDeclarationError):
        contracts.build_contract_registry([triple])
    paired = contracts.build_contract_registry([pair]).get("test.signal.pair", 1)
    assert paired.pattern == "2_to_1" and paired.paramclass is DummyParam
    assert paired.x_alignment is None and not paired.qualified


def test_input_kinds() -> None:
    """Input kinds follow the roles; any valid signal state is accepted."""
    contract = contracts.get_operation_contract("sigima.signal.normalize", 1)
    assert contract.input_kinds(1) == ["signal"]
    assert contract.input_kinds(0) is None and contract.input_kinds(2) is None
    assert not hasattr(contract, "check_preconditions")


def test_parameters_to_values() -> None:
    """Enums by value, choices by key, transient excluded, non-finite encoded."""
    param = sigima.params.NormalizeParam.create(method="maximum")
    assert contracts.parameters_to_values(param) == {"method": "maximum"}
    assert not contracts.parameters_to_values(None)
    dummy_param = DummyParam.create(method=NormalizationMethod.RMS, factor=math.inf)
    assert contracts.parameters_to_values(dummy_param) == {
        "factor": {"$float": "Infinity"},
        "count": 2,
        "enabled": True,
        "label": "a",
        "mode": 0,
        "method": "rms",
    }
    clip = sigima.params.ClipParam()
    values = contracts.parameters_to_values(clip)
    assert all(not isinstance(v, float) or math.isfinite(v) for v in values.values())
    json.dumps(values, allow_nan=False)


def test_parameters_to_values_arrays() -> None:
    """Array items are encoded with their dtype and shape; object arrays are not."""

    class ArrayParam(gds.DataSet):
        """Array parameter."""

        data = gds.FloatArrayItem("Data", default=np.zeros(3))

    values = contracts.parameters_to_values(ArrayParam())
    assert set(values["data"]) == {"$ndarray"}
    assert values["data"]["$ndarray"]["dtype"] == "<f8"
    param = ArrayParam()
    param.data = np.array([None, 1], dtype=object)
    with pytest.raises(contracts.ParameterEncodingError):
        contracts.parameters_to_values(param)


def test_parameters_from_values_round_trip() -> None:
    """JSON -> DataSet -> JSON is the identity; integers become floats."""
    registry = contracts.build_contract_registry([dummy])
    contract = registry.get("test.signal.dummy", 2)
    values = {
        "factor": 1,
        "count": 3,
        "enabled": False,
        "label": "b",
        "mode": 1,
        "method": "amplitude",
    }
    param = contracts.parameters_from_values(contract, values)
    assert isinstance(param, DummyParam)
    assert param.factor == 1.0 and isinstance(param.factor, float)
    assert param.method is NormalizationMethod.AMPLITUDE
    assert contracts.parameters_to_values(param) == dict(values, factor=1.0)
    nan = contracts.parameters_from_values(
        contract, dict(values, factor={"$float": "NaN"})
    )
    assert math.isnan(nan.factor)
    normalize = contracts.get_operation_contract("sigima.signal.normalize", 1)
    restored = contracts.parameters_from_values(normalize, {"method": "maximum"})
    assert contracts.parameters_to_values(restored) == {"method": "maximum"}


@pytest.mark.parametrize(
    "values",
    [
        {"method": "Maximum"},
        {"method": "MAXIMUM"},
        {"method": 0},
        {},
        {"method": "maximum", "extra": 1},
        ["method", "maximum"],
    ],
)
def test_parameters_from_values_refuses(values) -> None:
    """Labels, member names, indexes, missing and unknown items are refused."""
    contract = contracts.get_operation_contract("sigima.signal.normalize", 1)
    with pytest.raises(contracts.InvalidParametersError):
        contracts.parameters_from_values(contract, values)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("factor", "1.0"),
        ("factor", True),
        ("factor", {"$float": "inf"}),
        ("count", 1.5),
        ("count", True),
        ("enabled", 1),
        ("label", 3),
        ("mode", 2),
        ("mode", True),
    ],
)
def test_parameters_from_values_type_checks(name, value) -> None:
    """Each item type only accepts its own JSON type."""
    contract = contracts.build_contract_registry([dummy]).get("test.signal.dummy", 2)
    values = {
        "factor": 1.0,
        "count": 3,
        "enabled": False,
        "label": "b",
        "mode": 1,
        "method": "amplitude",
    }
    values[name] = value
    with pytest.raises(contracts.InvalidParametersError):
        contracts.parameters_from_values(contract, values)


def test_generic_inference() -> None:
    """Structures are inferred from annotations, for any family or container."""
    registry = contracts.build_contract_registry(
        [triple_roles, image_average, image_profile, detect, image_table, rois]
    )
    contract = registry.get("test.signal.triple_roles", 1)
    assert contract.pattern == "3_to_1"
    assert [role.name for role in contract.inputs] == [
        "reference",
        "measure",
        "background",
    ]
    contract = registry.get("test.image.average", 1)
    assert contract.pattern == "n_to_1"
    assert contract.inputs == (contracts.Role("sources", "image", None),)
    assert contract.parameters == (
        contracts.ParameterSpec("p", DummyParam, "single", optional=True),
    )
    assert contract.input_kinds(3) == ["image"] * 3
    assert contract.input_kinds(0) is None
    assert contract.input_role_names(2) == ["sources", "sources"]
    contract = registry.get("test.image.profile", 1)
    assert (contract.pattern, contract.outputs[0].kind) == ("1_to_1", "signal")
    contract = registry.get("test.signal.detect", 1)
    assert contract.pattern == "1_to_0"
    assert contract.outputs == (contracts.Role("result", "geometry", optional=True),)
    assert contract.outputs[0].is_artifact
    assert registry.get("test.image.table", 1).outputs[0].kind == "table"
    contract = registry.get("test.signal.rois", 1)
    assert contract.parameters[0].container == "single_or_list"
    assert contract.paramclass is None


def test_unknown_operation_discovered_by_metadata() -> None:
    """An operation unknown to Sigima is described from its declarations only."""
    registry = contracts.build_contract_registry([aligned])
    contract = registry.get("test.signal.aligned", 3)
    assert contract.x_alignment == SOURCE_GRID_LINEAR
    assert contract.qualification == ReplayQualification(contract_version=2)
    assert contract.qualification_status == "qualification_outdated"
    assert not contract.qualified and contract.comparison is None
    with pytest.raises(contracts.IncompatibleContractError):
        registry.get("test.signal.aligned", 2)


@pytest.mark.parametrize(
    "func",
    [
        triple,
        misaligned,
        unknown_policy,
        unknown_option,
        bad_order,
        bad_return,
        bad_argument,
        bad_roles,
    ],
)
def test_inconsistent_declarations(func) -> None:
    """Ambiguous or inconsistent declarations are reported explicitly."""
    with pytest.raises(contracts.ContractDeclarationError, match=func.__name__):
        contracts.build_contract_registry([func])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"qualification": ReplayQualification(1)},
        {"x_alignment": SOURCE_GRID_LINEAR},
        {"reads_options": ("auto_normalize_kernel",)},
        {"roles": ("source",)},
        {"operation_id": "a.b", "contract_version": 1, "qualification": 1},
        {"operation_id": "a.b", "contract_version": 1, "x_alignment": "Bad"},
        {"operation_id": "a.b", "contract_version": 1, "reads_options": "opt"},
        {"operation_id": "a.b", "contract_version": 1, "roles": ()},
        {"operation_id": "a.b", "contract_version": 1, "roles": ("a", "a")},
        {"operation_id": "a.b", "contract_version": 1, "roles": ("Source",)},
    ],
)
def test_decorator_rejects_invalid_declarations(kwargs) -> None:
    """Contract declarations are validated when the decorator is created."""
    with pytest.raises(ValueError):
        computation_function(**kwargs)


def test_comparison_rules() -> None:
    """Comparison rules are validated and round-trip as plain JSON values."""
    assert EXACT.exact and EXACT.to_dict() == {"rule": "exact"}
    assert ComparisonRule.from_dict({"rule": "exact"}) is EXACT
    rule = tolerance("FFT rounding", rtol=1e-12, same_environment="exact")
    assert not rule.exact
    values = json.loads(json.dumps(rule.to_dict(), allow_nan=False))
    assert values == {
        "rule": "tolerance",
        "outputs": {"result": {"atol": 0.0, "rtol": 1e-12}},
        "same_environment": "exact",
        "justification": "FFT rounding",
    }
    assert ComparisonRule.from_dict(values) == rule
    for bad in (
        lambda: tolerance("", atol=1.0),
        lambda: tolerance("why"),
        lambda: tolerance("why", atol=-1.0),
        lambda: tolerance("why", atol=math.inf),
        lambda: tolerance("why", atol=1.0, same_environment="close"),
        lambda: ComparisonRule(justification="exact needs none"),
        lambda: ComparisonRule((OutputTolerance("result", 1.0),) * 2, "exact", "x"),
        lambda: ReplayQualification(0),
        lambda: ReplayQualification(True),
        lambda: ReplayQualification(1, comparison={"rule": "exact"}),
    ):
        with pytest.raises(ValueError):
            bad()
    for bad_dict in (
        None,
        {"rule": "close"},
        {"rule": "tolerance", "outputs": {}, "same_environment": "exact"},
        dict(values, outputs={"result": {"atol": 1.0}}),
    ):
        with pytest.raises(ValueError):
            ComparisonRule.from_dict(bad_dict)


def test_contract_descriptors_pickle() -> None:
    """Descriptors and decorated functions with declarations are picklable."""
    contract = contracts.get_operation_contract("sigima.signal.difference", 1)
    for value in (
        contract.inputs,
        contract.outputs,
        contract.qualification,
        tolerance("why", atol=1.0),
        contracts.ParameterSpec("p", DummyParam, "list"),
    ):
        assert pickle.loads(pickle.dumps(value)) == value
    func = sigima.proc.signal.difference
    assert pickle.loads(pickle.dumps(func)) is func
    assert get_computation_metadata(func).qualification == contract.qualification


def test_annotation_resolution() -> None:
    """String annotations are resolved without evaluation, also on Python 3.9."""
    namespace = {"SignalObj": SignalObj, "sigima": sigima}
    assert resolve_annotation("SignalObj", namespace) is SignalObj
    assert resolve_annotation(SignalObj, namespace) is SignalObj
    optional = resolve_annotation("SignalObj | None", namespace)
    assert split_optional(optional) == (SignalObj, True)
    items = resolve_annotation("list[sigima.params.NormalizeParam]", namespace)
    assert sequence_item(items) is sigima.params.NormalizeParam
    assert sequence_item(resolve_annotation("tuple[int, ...]", namespace)) is int
    assert sequence_item(resolve_annotation("tuple[int, str]", namespace)) is None
    assert sequence_item(SignalObj) is None
    for bad in ("Unknown", "sigima.nothing", "1 + 2", "SignalObj(", "lambda: 0"):
        with pytest.raises(AnnotationError):
            resolve_annotation(bad, namespace)
    with pytest.raises(AnnotationError):
        split_optional(resolve_annotation("None", namespace))
