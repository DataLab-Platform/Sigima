# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Operation contracts
===================

An operation contract states what a computation function does, independently of
its Python name: a stable identifier, an integer contract version, the processing
pattern, typed input and output roles, the parameter class and the preconditions
under which replay is qualified.

Contracts are declared with :func:`sigima.proc.decorator.computation_function`
(``operation_id``, ``contract_version``, ``aliases``) and collected by scanning
``sigima.proc``. Declaring a contract does not qualify it for replay:
qualification is a separate, test-backed decision listed in this module.

Parameters cross application boundaries as plain JSON values
(:func:`parameters_to_values`) and are rebuilt only into the contract's own
parameter class (:func:`parameters_from_values`). No class name is ever read from
data.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import math
import sys
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any, Callable

import guidata.dataset as gds
import numpy as np
from guidata.dataset.datatypes import (
    BeginGroup,
    ComputedProp,
    EndGroup,
)

from sigima.objects import ImageObj, SignalObj
from sigima.proc.decorator import (
    find_computation_functions,
    get_computation_metadata,
    is_computation_function,
)

__all__ = [
    "ContractDeclarationError",
    "ContractError",
    "IncompatibleContractError",
    "InvalidParametersError",
    "OperationContract",
    "ParameterEncodingError",
    "Role",
    "UnknownOperationError",
    "build_contract_registry",
    "contract_for_function",
    "get_operation_contract",
    "parameters_from_values",
    "parameters_to_values",
]


class ContractError(Exception):
    """Base class of operation contract errors."""


class UnknownOperationError(ContractError, LookupError):
    """Raised when no contract has the requested identifier or alias."""


class IncompatibleContractError(ContractError):
    """Raised when the requested contract version is not supported."""


class ContractDeclarationError(ContractError):
    """Raised when declared contracts are inconsistent (duplicates, shapes)."""


class InvalidParametersError(ContractError, ValueError):
    """Raised when parameter values do not match the contract's parameter class."""


class ParameterEncodingError(ContractError, ValueError):
    """Raised when a parameter set cannot be encoded as plain JSON values."""


_NON_FINITE = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}


@dataclasses.dataclass(frozen=True)
class Role:
    """Typed input or output role of an operation.

    Attributes:
        name: Role name (e.g. ``"source"``, ``"result"``).
        kind: Object kind (``"signal"`` or ``"image"``).
        cardinality: Number of objects bound to the role.
    """

    name: str
    kind: str
    cardinality: int = 1


def _check_signal(obj: Any) -> bool:
    return isinstance(obj, SignalObj)


def _check_real_dtype(obj: Any) -> bool:
    return np.asarray(obj.x).dtype.kind == "f" and np.asarray(obj.y).dtype.kind == "f"


def _check_no_roi(obj: Any) -> bool:
    return obj.roi is None


def _check_no_uncertainty(obj: Any) -> bool:
    return obj.dx is None and obj.dy is None


# Ordered: the first failing check names the refusal reason.
_PRECONDITION_CHECKS: dict[str, Callable[[Any], bool]] = {
    "signal": _check_signal,
    "real_dtype": _check_real_dtype,
    "no_roi": _check_no_roi,
    "no_uncertainty": _check_no_uncertainty,
}


@dataclasses.dataclass(frozen=True)
class _Qualification:
    parameter_schema_version: int
    preconditions: tuple[str, ...]


# Contracts qualified for replay, each backed by exact reference tests.
_QUALIFIED: dict[tuple[str, int], _Qualification] = {
    ("sigima.signal.normalize", 1): _Qualification(
        parameter_schema_version=1,
        preconditions=("signal", "real_dtype", "no_roi", "no_uncertainty"),
    ),
}


@dataclasses.dataclass(frozen=True)
class OperationContract:
    """Contract of one operation.

    Attributes:
        operation_id: Stable identifier.
        contract_version: Integer contract version.
        pattern: Processing pattern (e.g. ``"1_to_1"``).
        inputs: Ordered input roles.
        outputs: Ordered output roles.
        paramclass: Parameter class, or None for parameterless operations.
        parameter_schema_version: Version of the parameter encoding.
        function: The decorated Sigima function.
        aliases: Earlier identifiers of the operation.
        qualified: True if replay of this contract is qualified.
        preconditions: Names of the checks that inputs must pass for replay.
    """

    operation_id: str
    contract_version: int
    pattern: str
    inputs: tuple[Role, ...]
    outputs: tuple[Role, ...]
    paramclass: type[gds.DataSet] | None
    parameter_schema_version: int
    function: Callable
    aliases: tuple[str, ...] = ()
    qualified: bool = False
    preconditions: tuple[str, ...] = ()

    def check_preconditions(self, inputs: Sequence[Any]) -> str | None:
        """Return the name of the first failing precondition, or None.

        Args:
            inputs: Input objects, in role order.
        """
        expected = sum(role.cardinality for role in self.inputs)
        if len(inputs) != expected:
            return "input_count"
        for name in self.preconditions:
            check = _PRECONDITION_CHECKS[name]
            if not all(check(obj) for obj in inputs):
                return name
        return None


def _annotation_name(func: Callable, annotation: Any) -> Any:
    """Resolve a (possibly stringized) annotation against the function module."""
    if isinstance(annotation, str):
        module = sys.modules.get(func.__module__)
        return getattr(module, annotation, annotation)
    return annotation


def _is_signal_annotation(annotation: Any) -> bool:
    if isinstance(annotation, str):
        return annotation == "SignalObj"
    return isinstance(annotation, type) and issubclass(annotation, SignalObj)


def _infer_signal_1_to_1(func: Callable) -> type[gds.DataSet] | None:
    """Return the parameter class of a signal 1-to-1 function.

    Raises:
        ContractDeclarationError: If the signature is not a signal 1-to-1 one.
    """
    sig = inspect.signature(func)
    # Keyword-only parameters are the DataSet items expanded by the decorator.
    params = [
        p for p in sig.parameters.values() if p.kind != inspect.Parameter.KEYWORD_ONLY
    ]
    ret = _annotation_name(func, sig.return_annotation)
    first = _annotation_name(func, params[0].annotation) if params else None
    paramclass = None
    if len(params) == 2:
        paramclass = _annotation_name(func, params[1].annotation)
        if not (
            isinstance(paramclass, type)
            and issubclass(paramclass, gds.DataSet)
            and not issubclass(paramclass, (SignalObj, ImageObj))
        ):
            paramclass = None
    if (
        not params
        or not _is_signal_annotation(first)
        or not _is_signal_annotation(ret)
        or (len(params) == 2 and paramclass is None)
        or len(params) > 2
    ):
        raise ContractDeclarationError(
            f"{func.__module__}.{func.__qualname__}: only signal 1-to-1 "
            "operations can declare a contract in this version"
        )
    return paramclass


def _build_contract(func: Callable) -> OperationContract:
    metadata = get_computation_metadata(func)
    paramclass = _infer_signal_1_to_1(func)
    key = (metadata.operation_id, metadata.contract_version)
    qualification = _QUALIFIED.get(key)
    return OperationContract(
        operation_id=metadata.operation_id,
        contract_version=metadata.contract_version,
        pattern="1_to_1",
        inputs=(Role("source", "signal"),),
        outputs=(Role("result", "signal"),),
        paramclass=paramclass,
        parameter_schema_version=(
            qualification.parameter_schema_version if qualification else 1
        ),
        function=func,
        aliases=metadata.aliases,
        qualified=qualification is not None,
        preconditions=qualification.preconditions if qualification else (),
    )


class ContractRegistry:
    """Contracts declared by a set of computation functions.

    Args:
        functions: Decorated computation functions; those without an operation
         identifier are ignored.

    Raises:
        ContractDeclarationError: On duplicate identifiers, aliases colliding with
         an identifier, or an unsupported operation shape.
    """

    def __init__(self, functions: Iterable[Callable]) -> None:
        self._contracts: dict[str, OperationContract] = {}
        self._aliases: dict[str, str] = {}
        self._by_function: dict[int, OperationContract] = {}
        for func in functions:
            if not is_computation_function(func):
                continue
            metadata = get_computation_metadata(func)
            if metadata.operation_id is None or id(func) in self._by_function:
                continue
            contract = _build_contract(func)
            if contract.operation_id in self._contracts:
                raise ContractDeclarationError(
                    f"Duplicate operation identifier: {contract.operation_id}"
                )
            self._contracts[contract.operation_id] = contract
            self._by_function[id(func)] = contract
            for alias in contract.aliases:
                if alias in self._aliases:
                    raise ContractDeclarationError(f"Duplicate alias: {alias}")
                self._aliases[alias] = contract.operation_id
        collisions = set(self._aliases) & set(self._contracts)
        if collisions:
            raise ContractDeclarationError(
                f"Aliases collide with operation identifiers: {sorted(collisions)}"
            )

    @property
    def contracts(self) -> tuple[OperationContract, ...]:
        """All contracts, sorted by operation identifier."""
        return tuple(self._contracts[key] for key in sorted(self._contracts))

    def get(self, operation_id: str, contract_version: int) -> OperationContract:
        """Return the contract for an identifier (or alias) and version.

        Raises:
            UnknownOperationError: If the identifier is unknown.
            IncompatibleContractError: If the version is not supported.
        """
        key = self._aliases.get(operation_id, operation_id)
        contract = self._contracts.get(key)
        if contract is None:
            raise UnknownOperationError(f"Unknown operation: {operation_id!r}")
        if contract_version != contract.contract_version:
            raise IncompatibleContractError(
                f"{contract.operation_id}: contract version {contract_version!r} "
                f"is not supported (current: {contract.contract_version})"
            )
        return contract

    def for_function(self, func: Callable) -> OperationContract | None:
        """Return the contract declared by *func* (identity lookup), if any."""
        return self._by_function.get(id(func))


def build_contract_registry(
    functions: Iterable[Callable] | None = None,
) -> ContractRegistry:
    """Build a contract registry.

    Args:
        functions: Functions to register. If None, scan ``sigima.proc``.
    """
    if functions is None:
        functions = [
            getattr(importlib.import_module(modname), name)
            for modname, name, _doc in find_computation_functions()
        ]
    return ContractRegistry(functions)


_REGISTRY: ContractRegistry | None = None


def _default_registry() -> ContractRegistry:
    global _REGISTRY  # pylint: disable=global-statement
    if _REGISTRY is None:
        _REGISTRY = build_contract_registry()
    return _REGISTRY


def get_operation_contract(
    operation_id: str, contract_version: int
) -> OperationContract:
    """Return the Sigima contract for an identifier (or alias) and version.

    Raises:
        UnknownOperationError: If the identifier is unknown.
        IncompatibleContractError: If the version is not supported.
    """
    return _default_registry().get(operation_id, contract_version)


def contract_for_function(func: Callable) -> OperationContract | None:
    """Return the Sigima contract declared by *func*, or None."""
    return _default_registry().for_function(func)


def _parameter_items(paramclass: type[gds.DataSet]) -> list[gds.DataItem]:
    """Return the items that make up the parameter values of *paramclass*."""
    items = []
    for item in paramclass._items:  # pylint: disable=protected-access
        if isinstance(item, (BeginGroup, EndGroup, gds.ButtonItem)):
            continue
        if item.get_prop("data", "transient", False):
            continue
        if isinstance(item.get_prop("data", "computed", None), ComputedProp):
            continue
        items.append(item)
    return items


def _encode_scalar(name: str, value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"$float": "NaN"}
        if math.isinf(value):
            return {"$float": "Infinity" if value > 0 else "-Infinity"}
        return value
    raise ParameterEncodingError(
        f"Parameter {name!r}: values of type {type(value).__name__} cannot be encoded"
    )


def parameters_to_values(param: gds.DataSet | None) -> dict[str, Any]:
    """Encode a parameter set as plain JSON values.

    Enum choices are encoded by their stable ``.value``, other choices by their
    stored key. Non-finite floats use ``{"$float": "Infinity" | "-Infinity" |
    "NaN"}``. Transient and computed items are excluded.

    Args:
        param: Parameter set instance actually used, or None.

    Returns:
        Mapping of item names to JSON values (empty for None).

    Raises:
        ParameterEncodingError: For arrays, dictionaries, nested DataSets, dates
         and other non-scalar values.
    """
    if param is None:
        return {}
    values: dict[str, Any] = {}
    for item in _parameter_items(type(param)):
        name = item.get_name()
        raw = item.get_value(param)
        enum_cls = getattr(item, "_enum_cls", None)
        if isinstance(item, gds.ChoiceItem) and enum_cls is not None:
            values[name] = None if raw is None else enum_cls[raw].value
        elif isinstance(raw, (list, tuple)) and isinstance(
            item, gds.MultipleChoiceItem
        ):
            values[name] = [_encode_scalar(name, v) for v in raw]
        else:
            values[name] = _encode_scalar(name, raw)
    return values


def _decode_float(name: str, value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) != {"$float"} or value["$float"] not in _NON_FINITE:
            raise InvalidParametersError(f"Parameter {name!r}: invalid float")
        return _NON_FINITE[value["$float"]]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParametersError(f"Parameter {name!r}: a number is expected")
    return float(value)


def _decode_item(item: gds.DataItem, value: Any) -> Any:
    """Return the raw stored value of *item* for the JSON *value*."""
    name = item.get_name()
    if value is None:
        if not item.get_prop("data", "allow_none", False):
            raise InvalidParametersError(f"Parameter {name!r} cannot be null")
        return None
    enum_cls: type[Enum] | None = getattr(item, "_enum_cls", None)
    if isinstance(item, gds.ChoiceItem) and enum_cls is not None:
        for member in enum_cls:
            if isinstance(value, str) and value == member.value:
                return member.name
        raise InvalidParametersError(
            f"Parameter {name!r}: {value!r} is not a value of {enum_cls.__name__}"
        )
    if isinstance(item, gds.FloatItem):
        raw = _decode_float(name, value)
    elif isinstance(item, gds.IntItem):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidParametersError(f"Parameter {name!r}: an integer is expected")
        raw = value
    elif isinstance(item, gds.BoolItem):
        if not isinstance(value, bool):
            raise InvalidParametersError(f"Parameter {name!r}: a boolean is expected")
        raw = value
    elif isinstance(item, gds.StringItem):
        if not isinstance(value, str):
            raise InvalidParametersError(f"Parameter {name!r}: a string is expected")
        raw = value
    elif isinstance(item, gds.ChoiceItem):
        raw = list(value) if isinstance(value, list) else value
        if not isinstance(item, gds.MultipleChoiceItem) and isinstance(value, bool):
            raise InvalidParametersError(f"Parameter {name!r}: invalid choice key")
    else:
        raise InvalidParametersError(
            f"Parameter {name!r}: items of type {type(item).__name__} are not supported"
        )
    try:
        valid = item.check_value(raw, raise_exception=True)
    except NotImplementedError:
        valid = True
    except Exception as exc:  # pylint: disable=broad-except
        raise InvalidParametersError(f"Parameter {name!r}: {exc}") from exc
    if valid is False:
        raise InvalidParametersError(f"Parameter {name!r}: invalid value {value!r}")
    return raw


def parameters_from_values(
    contract: OperationContract, values: dict[str, Any]
) -> gds.DataSet | None:
    """Rebuild the parameter set of *contract* from plain JSON values.

    Only the contract's own parameter class is used. Every item is required;
    unknown items are refused; enum choices are accepted only by ``.value``.

    Args:
        contract: Operation contract.
        values: Mapping produced by :func:`parameters_to_values`.

    Returns:
        A new parameter instance, or None for a parameterless contract.

    Raises:
        InvalidParametersError: If *values* do not match the parameter class.
    """
    if not isinstance(values, dict) or not all(isinstance(k, str) for k in values):
        raise InvalidParametersError("Parameters must be a JSON object")
    paramclass = contract.paramclass
    if paramclass is None:
        if values:
            raise InvalidParametersError(f"{contract.operation_id} has no parameters")
        return None
    items = _parameter_items(paramclass)
    names = [item.get_name() for item in items]
    unknown = sorted(set(values) - set(names))
    missing = [name for name in names if name not in values]
    if unknown or missing:
        raise InvalidParametersError(
            f"{contract.operation_id}: unknown parameters {unknown}, "
            f"missing parameters {missing}"
        )
    param = paramclass()
    for item in items:
        raw = _decode_item(item, values[item.get_name()])
        # Already checked: bypass the global guidata validation mode.
        setattr(param, f"_{item.get_name()}", raw)
    return param
