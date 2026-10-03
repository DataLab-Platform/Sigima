# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for operation contracts and parameter encoding."""

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
import sigima.proc.signal
from sigima.enums import NormalizationMethod
from sigima.objects import SignalObj
from sigima.proc import contracts
from sigima.proc.decorator import (
    ComputationMetadata,
    computation_function,
    find_computation_functions,
    get_computation_metadata,
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
def pair(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Test-only 2-to-1 operation: not supported by contracts yet."""
    return src1.copy()


def _f0() -> SignalObj:
    return sigima.objects.create_signal(
        "signal",
        np.array([0.0, 0.25, 0.5, 0.75]),
        np.array([-2.0, 0.0, 1.0, 4.0]),
        units=("s", ""),
    )


def test_only_normalize_declares_an_operation_id() -> None:
    """Only the signal normalization declares a contract."""
    declared = []
    for modname, name, _doc in find_computation_functions():
        func = getattr(__import__(modname, fromlist=[name]), name)
        if get_computation_metadata(func).operation_id:
            declared.append(f"{func.__module__}.{func.__qualname__}")
    assert declared == ["sigima.proc.signal.processing.normalize"]


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
    assert contract.parameter_schema_version == 1
    assert contracts.contract_for_function(sigima.proc.signal.normalize) is contract
    assert contracts.contract_for_function(sigima.proc.signal.addition_constant) is None
    with pytest.raises(contracts.UnknownOperationError):
        contracts.get_operation_contract("sigima.signal.unknown", 1)
    with pytest.raises(contracts.IncompatibleContractError):
        contracts.get_operation_contract("sigima.signal.normalize", 2)


def test_signature_calls_and_pickling_unchanged() -> None:
    """Decorated signature, both calling styles and pickling are preserved."""
    func = sigima.proc.signal.normalize
    assert list(inspect.signature(func).parameters) == ["src", "p", "method"]
    src = _f0()
    by_param = func(src, sigima.params.NormalizeParam.create(method="amplitude"))
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
        contracts.build_contract_registry([pair])


def test_preconditions() -> None:
    """ROI and uncertainty rows fail the qualified preconditions."""
    contract = contracts.get_operation_contract("sigima.signal.normalize", 1)
    src = _f0()
    assert contract.check_preconditions([src]) is None
    assert contract.check_preconditions([]) == "input_count"
    with_dy = src.copy()
    with_dy.dy = np.full(4, 0.1)
    assert contract.check_preconditions([with_dy]) == "no_uncertainty"
    with_roi = src.copy()
    with_roi.roi = sigima.objects.create_signal_roi([0.0, 0.5])
    assert contract.check_preconditions([with_roi]) == "no_roi"


def test_parameters_to_values() -> None:
    """Enums by value, choices by key, transient excluded, non-finite encoded."""
    param = sigima.params.NormalizeParam.create(method="maximum")
    assert contracts.parameters_to_values(param) == {"method": "maximum"}
    assert contracts.parameters_to_values(None) == {}
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


def test_parameters_to_values_refuses_arrays() -> None:
    """Array items cannot be encoded."""

    class ArrayParam(gds.DataSet):
        """Array parameter."""

        data = gds.FloatArrayItem("Data", default=np.zeros(3))

    with pytest.raises(contracts.ParameterEncodingError):
        contracts.parameters_to_values(ArrayParam())


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
