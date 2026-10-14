# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Operation contracts
===================

An operation contract states what a computation function does, independently of
its Python name: a stable identifier, an integer contract version, the processing
pattern, typed input and output roles, the parameter arguments, the execution
policy, the global options read and, when tested, the replay qualification.

Every property is declared next to the function with
:func:`sigima.proc.decorator.computation_function` (``operation_id``,
``contract_version``, ``aliases``, ``qualification``, ``x_alignment``,
``reads_options``, ``roles``) or inferred from its signature. This module only
discovers the declarations of the public computation packages
(``sigima.proc.signal`` and ``sigima.proc.image``), infers the structure,
assembles the contracts, detects collisions and resolves identifiers: it holds
no data naming an operation.

Declaring a contract does not qualify it for replay: a qualification names the
contract version actually tested (see
:class:`sigima.proc.contract_types.ReplayQualification`).

Parameters cross application boundaries as plain JSON values
(:meth:`OperationContract.encode_parameters`, :func:`parameters_to_values`) and
are rebuilt only into the classes declared by the local contract
(:meth:`OperationContract.decode_parameters`, :func:`parameters_from_values`),
see :mod:`sigima.proc.parameter_codec`. No class name is ever read from data.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import sys
import typing
from collections.abc import Iterable, Sequence
from typing import Any, Callable

import guidata.dataset as gds
import numpy as np

from sigima.config import OptionField, options
from sigima.objects import ImageObj, SignalObj
from sigima.objects.scalar.geometry import GeometryResult
from sigima.objects.scalar.table import TableResult
from sigima.proc.alignment import XAlignmentError
from sigima.proc.annotations import (
    AnnotationError,
    resolve_annotation,
    sequence_item,
    split_optional,
    union_members,
)
from sigima.proc.contract_types import (
    EXACT,
    ComparisonRule,
    ContractDeclarationError,
    ContractError,
    IncompatibleContractError,
    InvalidParametersError,
    OutputTolerance,
    ParameterEncodingError,
    ParameterSpec,
    ReplayQualification,
    Role,
    UnknownOperationError,
    tolerance,
)
from sigima.proc.decorator import (
    ComputationMetadata,
    get_computation_metadata,
    is_computation_function,
)
from sigima.proc.parameter_codec import (
    PARAMETER_CODEC_VERSION,
    decode_dataset,
    decode_parameters,
    encode_dataset,
    encode_parameters,
)
from sigima.proc.policies import check_x_alignment_record, get_x_alignment_policy

__all__ = [
    "EXACT",
    "ComparisonRule",
    "ContractDeclarationError",
    "ContractError",
    "IncompatibleContractError",
    "InvalidParametersError",
    "OperationContract",
    "OutputTolerance",
    "ParameterEncodingError",
    "ParameterSpec",
    "ReplayQualification",
    "Role",
    "UnknownOperationError",
    "XAlignmentError",
    "build_contract_registry",
    "contract_for_function",
    "decode_parameters",
    "encode_parameters",
    "get_operation_contract",
    "parameters_from_values",
    "parameters_to_values",
    "tolerance",
]


_OBJECT_CLASSES: tuple[tuple[type, str], ...] = (
    (SignalObj, "signal"),
    (ImageObj, "image"),
)
_ARTIFACT_CLASSES: tuple[tuple[type, str], ...] = (
    (TableResult, "table"),
    (GeometryResult, "geometry"),
)


def _is_class(tp: Any) -> bool:
    # On Python 3.9, isinstance(list[X], type) is True: exclude generic aliases.
    return isinstance(tp, type) and typing.get_origin(tp) is None


def _kind_of(tp: Any, classes: tuple[tuple[type, str], ...]) -> str | None:
    if _is_class(tp):
        for cls, kind in classes:
            if issubclass(tp, cls):
                return kind
    return None


def _object_kind(obj: Any) -> str | None:
    """Return the kind of a live object (``"signal"``, ``"image"`` or None)."""
    return _kind_of(type(obj), _OBJECT_CLASSES)


# Provisional replay preconditions of qualified contracts. They restrict replay
# to the states that the current state fingerprints and comparisons can verify;
# they are removed once those are generalized. They name no operation.


def _check_real_dtype(obj: Any) -> bool:
    if isinstance(obj, SignalObj):
        return (
            np.asarray(obj.x).dtype.kind == "f" and np.asarray(obj.y).dtype.kind == "f"
        )
    return np.asarray(obj.data).dtype.kind in "biuf"


def _check_no_roi(obj: Any) -> bool:
    return obj.roi is None


def _check_no_uncertainty(obj: Any) -> bool:
    # Only signals carry uncertainty rows (an image's dx/dy are its pixel size).
    return not isinstance(obj, SignalObj) or (obj.dx is None and obj.dy is None)


# Ordered: the first failing check names the refusal reason.
_PROVISIONAL_PRECONDITIONS: dict[str, Callable[[Any], bool]] = {
    "real_dtype": _check_real_dtype,
    "no_roi": _check_no_roi,
    "no_uncertainty": _check_no_uncertainty,
}


@dataclasses.dataclass(frozen=True)
class OperationContract:
    """Contract of one operation.

    Attributes:
        operation_id: Stable identifier.
        contract_version: Integer contract version.
        pattern: Processing pattern: ``"1_to_1"``, ``"2_to_1"``, ``"n_to_1"``
         (variable number of inputs), ``"1_to_0"`` (analysis result), or more
         generally ``"<k>_to_1"`` / ``"<k>_to_0"``.
        inputs: Ordered input roles.
        outputs: Ordered output roles.
        paramclass: Parameter class when the operation takes a single parameter
         set, None otherwise.
        parameter_schema_version: Version of the parameter encoding (see
         :mod:`sigima.proc.parameter_codec`).
        function: The decorated Sigima function.
        aliases: Earlier identifiers of the operation.
        parameters: Parameter arguments, in call order.
        qualification: Declared replay qualification, or None.
        x_alignment: X-alignment policy applied to the inputs, or None.
        reads_options: Global options read by the computation.
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
    parameters: tuple[ParameterSpec, ...] = ()
    qualification: ReplayQualification | None = None
    x_alignment: str | None = None
    reads_options: tuple[str, ...] = ()

    @property
    def qualification_status(self) -> str:
        """``"qualified"``, ``"not_qualified"`` or ``"qualification_outdated"``.

        A qualification declared for another contract version than the current
        one is outdated: it does not qualify the current version.
        """
        if self.qualification is None:
            return "not_qualified"
        if self.qualification.contract_version != self.contract_version:
            return "qualification_outdated"
        return "qualified"

    @property
    def qualified(self) -> bool:
        """True if replay of this contract version is qualified."""
        return self.qualification_status == "qualified"

    @property
    def comparison(self) -> ComparisonRule | None:
        """Comparison rule of the qualification, or None if not qualified."""
        return self.qualification.comparison if self.qualified else None

    @property
    def preconditions(self) -> tuple[str, ...]:
        """Names of the provisional checks that inputs must pass for replay."""
        return tuple(_PROVISIONAL_PRECONDITIONS) if self.qualified else ()

    def input_kinds(self, count: int) -> list[str] | None:
        """Return the kind expected for each of *count* inputs, or None.

        Args:
            count: Number of input objects.

        Returns:
            One kind per input, in role order, or None if *count* does not fit
            the input roles.
        """
        fixed = sum(role.cardinality or 0 for role in self.inputs)
        variadic = [role for role in self.inputs if role.variadic]
        if count < fixed + len(variadic) or (not variadic and count != fixed):
            return None
        kinds: list[str] = []
        for role in self.inputs:
            number = role.cardinality if role.cardinality else count - fixed
            kinds.extend([role.kind] * number)
        return kinds

    def input_role_names(self, count: int) -> list[str] | None:
        """Return the role name of each of *count* inputs, or None."""
        if self.input_kinds(count) is None:
            return None
        fixed = sum(role.cardinality or 0 for role in self.inputs)
        names: list[str] = []
        for role in self.inputs:
            names.extend([role.name] * (role.cardinality or count - fixed))
        return names

    def encode_parameters(self, arguments: Sequence[Any]) -> dict[str, Any]:
        """Encode the parameter arguments actually passed to the function.

        Args:
            arguments: Parameter arguments, in call order (after the inputs).

        Raises:
            ParameterEncodingError: If an argument cannot be encoded.
        """
        return encode_parameters(self.parameters, arguments)

    def decode_parameters(self, values: Any) -> list[Any]:
        """Rebuild the parameter arguments from :meth:`encode_parameters` values.

        Raises:
            InvalidParametersError: If *values* do not match the declarations.
        """
        try:
            return decode_parameters(self.parameters, values)
        except InvalidParametersError as exc:
            raise InvalidParametersError(f"{self.operation_id}: {exc}") from exc

    def check_preconditions(self, inputs: Sequence[Any]) -> str | None:
        """Return the name of the first failing precondition, or None.

        Args:
            inputs: Input objects, in role order.
        """
        kinds = self.input_kinds(len(inputs))
        if kinds is None:
            return "input_count"
        for obj, kind in zip(inputs, kinds):
            if _object_kind(obj) != kind:
                return kind
        for name in self.preconditions:
            check = _PROVISIONAL_PRECONDITIONS[name]
            if not all(check(obj) for obj in inputs):
                return name
        return None

    def prepare_inputs(
        self, inputs: Sequence[Any], context: dict[str, Any] | None = None
    ) -> tuple[list[Any], dict[str, Any]]:
        """Return the inputs to pass to the function and the execution context.

        The original inputs are never modified. Ordinary processing calls this
        without *context*; a replay passes the recorded context, whose policy
        must be the one declared by the contract.

        Args:
            inputs: Input objects, in role order.
            context: Recorded execution context, for a replay.

        Returns:
            ``(inputs, context)``; *context* holds ``x_alignment`` when the
            contract aligns its inputs.

        Raises:
            XAlignmentError: If a recorded policy is unknown or the inputs cannot
             be aligned.
        """
        if self.x_alignment is None:
            return list(inputs), {}
        if context is not None:
            check_x_alignment_record(context.get("x_alignment"), self.x_alignment)
        policy = get_x_alignment_policy(self.x_alignment)
        prepared, record = policy.apply(list(inputs))
        return prepared, {"x_alignment": record}


def _declaration_error(func: Callable, message: str) -> ContractDeclarationError:
    return ContractDeclarationError(f"{func.__module__}.{func.__qualname__}: {message}")


def _is_paramclass(tp: Any) -> bool:
    return (
        _is_class(tp)
        and issubclass(tp, gds.DataSet)
        and _kind_of(tp, _OBJECT_CLASSES) is None
    )


def _parameter_spec(name: str, tp: Any, optional: bool) -> ParameterSpec | None:
    """Return the parameter argument designated by *tp*, or None."""
    members = union_members(tp)
    singles = [member for member in members if _is_paramclass(member)]
    lists = [sequence_item(member) for member in members]
    lists = [item for item in lists if _is_paramclass(item)]
    if len(members) == 1 and singles:
        return ParameterSpec(name, singles[0], "single", optional)
    if len(members) == 1 and lists:
        return ParameterSpec(name, lists[0], "list", optional)
    if len(members) == 2 and len(singles) == 1 and lists == singles:
        return ParameterSpec(name, singles[0], "single_or_list", optional)
    return None


@dataclasses.dataclass(frozen=True)
class _Structure:
    pattern: str
    inputs: tuple[Role, ...]
    outputs: tuple[Role, ...]
    parameters: tuple[ParameterSpec, ...]


def _namespace(func: Callable) -> dict[str, Any]:
    """Return the globals of the module defining *func* (annotation namespace)."""
    module = sys.modules.get(func.__module__)
    return vars(module) if module is not None else {}


def _infer_inputs(
    func: Callable, metadata: ComputationMetadata
) -> tuple[tuple[Role, ...], tuple[ParameterSpec, ...]]:
    """Infer the input roles and parameter arguments of *func*."""
    namespace = _namespace(func)
    objects: list[tuple[str, bool]] = []
    parameters: list[ParameterSpec] = []
    for param in inspect.signature(func).parameters.values():
        # Keyword-only parameters are the DataSet items expanded by the decorator.
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise _declaration_error(func, "variable positional arguments")
        if param.annotation is inspect.Parameter.empty:
            raise _declaration_error(func, f"argument {param.name!r} is not annotated")
        try:
            tp, optional = split_optional(
                resolve_annotation(param.annotation, namespace)
            )
        except AnnotationError as exc:
            raise _declaration_error(func, str(exc)) from exc
        item = sequence_item(tp)
        kind = _kind_of(tp, _OBJECT_CLASSES)
        item_kind = None if item is None else _kind_of(item, _OBJECT_CLASSES)
        if kind is not None or item_kind is not None:
            if optional:
                raise _declaration_error(
                    func, f"object argument {param.name!r} cannot be optional"
                )
            if parameters:
                raise _declaration_error(
                    func, "object arguments must precede parameter arguments"
                )
            objects.append((kind or item_kind, item_kind is not None))
            continue
        spec = _parameter_spec(param.name, tp, optional)
        if spec is None:
            raise _declaration_error(
                func, f"unsupported annotation of {param.name!r}: {param.annotation}"
            )
        parameters.append(spec)
    if not objects:
        raise _declaration_error(func, "an operation needs at least one input object")
    if metadata.roles is not None:
        if len(metadata.roles) != len(objects):
            raise _declaration_error(
                func, f"{len(metadata.roles)} roles declared for {len(objects)} inputs"
            )
        names = metadata.roles
    elif len(objects) == 1:
        names = ("sources",) if objects[0][1] else ("source",)
    elif len(objects) == 2 and not any(variadic for _k, variadic in objects):
        names = ("source", "operand")
    else:
        raise _declaration_error(
            func, "input roles cannot be inferred: declare them with roles=..."
        )
    inputs = tuple(
        Role(name, kind, None if variadic else 1)
        for name, (kind, variadic) in zip(names, objects)
    )
    return inputs, tuple(parameters)


def _infer_structure(func: Callable, metadata: ComputationMetadata) -> _Structure:
    """Infer the structure of the contract declared by *func*.

    Raises:
        ContractDeclarationError: If the signature is not supported or ambiguous.
    """
    inputs, parameters = _infer_inputs(func, metadata)
    namespace = _namespace(func)
    try:
        ret, optional = split_optional(
            resolve_annotation(inspect.signature(func).return_annotation, namespace)
        )
    except AnnotationError as exc:
        raise _declaration_error(func, f"return annotation: {exc}") from exc
    kind = _kind_of(ret, _OBJECT_CLASSES)
    artifact = _kind_of(ret, _ARTIFACT_CLASSES)
    if kind is None and artifact is None:
        raise _declaration_error(func, f"unsupported return annotation: {ret!r}")
    outputs = (Role("result", kind or artifact, optional=optional),)
    variadic = [role for role in inputs if role.variadic]
    if variadic and len(inputs) > 1:
        raise _declaration_error(func, "a variable input role must be the only one")
    suffix = "0" if artifact else "1"
    prefix = "n" if variadic else str(len(inputs))
    return _Structure(f"{prefix}_to_{suffix}", inputs, outputs, parameters)


def _build_contract(func: Callable) -> OperationContract:
    metadata = get_computation_metadata(func)
    structure = _infer_structure(func, metadata)
    if metadata.x_alignment is not None:
        try:
            policy = get_x_alignment_policy(metadata.x_alignment)
        except XAlignmentError as exc:
            raise _declaration_error(func, str(exc)) from exc
        if structure.pattern not in policy.patterns or any(
            role.kind != "signal" for role in structure.inputs
        ):
            raise _declaration_error(
                func,
                f"X-alignment policy {policy.identifier} does not apply to a "
                f"{structure.pattern} operation on {structure.inputs}",
            )
    for name in metadata.reads_options:
        if not isinstance(getattr(options, name, None), OptionField):
            raise _declaration_error(func, f"unknown global option {name!r}")
    single = structure.parameters
    paramclass = (
        single[0].paramclass
        if len(single) == 1 and single[0].container == "single"
        else None
    )
    return OperationContract(
        operation_id=metadata.operation_id,
        contract_version=metadata.contract_version,
        pattern=structure.pattern,
        inputs=structure.inputs,
        outputs=structure.outputs,
        paramclass=paramclass,
        parameter_schema_version=PARAMETER_CODEC_VERSION,
        function=func,
        aliases=metadata.aliases,
        parameters=structure.parameters,
        qualification=metadata.qualification,
        x_alignment=metadata.x_alignment,
        reads_options=metadata.reads_options,
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


#: Packages whose exported computation functions are collected by default. Only
#: public runtime packages are imported (``sigima.proc.validation``, for one,
#: needs pytest, which is not available at runtime).
COMPUTATION_PACKAGES = ("sigima.proc.signal", "sigima.proc.image")


def _exported_computation_functions() -> list[Callable]:
    """Return the computation functions exported by :data:`COMPUTATION_PACKAGES`."""
    functions: list[Callable] = []
    for package in COMPUTATION_PACKAGES:
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if (
                inspect.isfunction(obj)
                and is_computation_function(obj)
                and all(obj is not f for f in functions)
            ):
                functions.append(obj)
    return functions


def build_contract_registry(
    functions: Iterable[Callable] | None = None,
) -> ContractRegistry:
    """Build a contract registry.

    Args:
        functions: Functions to register. If None, collect the computation
         functions exported by :data:`COMPUTATION_PACKAGES`.
    """
    if functions is None:
        functions = _exported_computation_functions()
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


def parameters_to_values(param: gds.DataSet | None) -> dict[str, Any]:
    """Encode a parameter set as plain JSON values.

    Enum choices are encoded by their stable ``.value``, other choices by their
    key. Non-finite floats use ``{"$float": "Infinity" | "-Infinity" | "NaN"}``;
    arrays and other composite values use the typed forms of
    :mod:`sigima.proc.parameter_codec`. Transient and computed items are
    excluded.

    Args:
        param: Parameter set instance actually used, or None.

    Returns:
        Mapping of item names to JSON values (empty for None).

    Raises:
        ParameterEncodingError: For values of unsupported types.
    """
    if param is None:
        return {}
    return encode_dataset(param)


def parameters_from_values(
    contract: OperationContract, values: dict[str, Any]
) -> gds.DataSet | None:
    """Rebuild the parameter set of *contract* from plain JSON values.

    Only the contract's own parameter class is used. Every item is required;
    unknown items are refused; enum choices are accepted only by ``.value``. For
    operations taking other parameter arguments than a single parameter set, use
    :func:`sigima.proc.parameter_codec.decode_parameters`.

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
    if not contract.parameters:
        if values:
            raise InvalidParametersError(f"{contract.operation_id} has no parameters")
        return None
    if contract.paramclass is None:
        raise InvalidParametersError(
            f"{contract.operation_id}: parameters are not a single parameter set"
        )
    try:
        return decode_dataset(contract.paramclass, values)
    except InvalidParametersError as exc:
        raise InvalidParametersError(f"{contract.operation_id}: {exc}") from exc
