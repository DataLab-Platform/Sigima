# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Execution of operation contracts
================================

One scientific path for ordinary processing and replay, whatever the
application:

- :func:`describe_call` describes a call as plain data: identity of the
  operation (kept as soon as it is known), qualification, pattern, ordered roles,
  effective parameters encoded by the contract (or the reason why they could not
  be), informative implementation.
- :func:`prepare_execution` decodes the parameters (for a replay), applies the
  execution policy declared by the contract exactly once (X alignment) and the
  global options read by the computation, and returns a :class:`PreparedExecution`
  holding the function, its arguments and the context to record. The application
  runs it itself (:meth:`PreparedExecution.run` or :func:`execute`), in its own
  process, worker or subprocess: a prepared execution is picklable.
- :func:`normalize_outputs` returns the outputs of a result by role.

Errors carry a stable ``code`` (:class:`ExecutionError`); scientific refusals of a
policy keep their own exception
(:class:`~sigima.proc.contracts.alignment.XAlignmentError`) in ordinary processing
and in replay alike. Nothing here depends on a
provenance format.
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
from collections.abc import Iterator, Sequence
from importlib import metadata
from typing import Any, Callable

from sigima.config import options
from sigima.objects import ImageObj, SignalObj
from sigima.objects.scalar.geometry import GeometryResult
from sigima.objects.scalar.table import TableResult
from sigima.proc.contracts.codec import PARAMETER_CODEC_VERSION, encode_value
from sigima.proc.contracts.declarations import (
    ContractError,
    InvalidParametersError,
    ParameterEncodingError,
)
from sigima.proc.contracts.policies import (
    check_x_alignment_record,
    get_x_alignment_policy,
)
from sigima.proc.contracts.registry import OperationContract, contract_for_function

__all__ = [
    "EXECUTION_ERROR_CODES",
    "ExecutionError",
    "PreparedExecution",
    "default_roles",
    "describe_call",
    "execute",
    "implementation_of",
    "normalize_outputs",
    "prepare_execution",
]

#: Stable codes of :class:`ExecutionError`.
EXECUTION_ERROR_CODES = (
    "input_mismatch",
    "invalid_parameters",
    "unsupported_context",
    "unexpected_output",
)

_KINDS: tuple[tuple[type, str], ...] = (
    (SignalObj, "signal"),
    (ImageObj, "image"),
    (TableResult, "table"),
    (GeometryResult, "geometry"),
)

_MISSING = object()


class ExecutionError(ContractError):
    """Raised when a call cannot be prepared or its result is unexpected.

    Args:
        code: Stable code, one of :data:`EXECUTION_ERROR_CODES`.
        message: Human-readable detail.
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


def _kind(value: Any) -> str | None:
    for cls, kind in _KINDS:
        if isinstance(value, cls):
            return kind
    return None


def default_roles(count: int) -> list[str]:
    """Return the role names of a call without contract, by number of inputs.

    ``source`` for one input, ``source`` and ``operand`` for two, ``sources``
    for each input of a list.
    """
    if count == 1:
        return ["source"]
    if count == 2:
        return ["source", "operand"]
    return ["sources"] * count


def implementation_of(func: Callable) -> dict[str, Any]:
    """Return the informative implementation descriptor of *func*.

    It is never resolved into a contract nor imported back.
    """
    module = getattr(func, "__module__", None) or ""
    package = module.split(".", 1)[0] or None
    version = getattr(sys.modules.get(package or ""), "__version__", None)
    if version is None and package:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = None
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return {
        "package": package,
        "version": version,
        "python_name": f"{module}.{qualname}" if module else qualname,
    }


def _parameter_arguments(contract: OperationContract, param: Any) -> list[Any]:
    """Return the parameter arguments of a call from what the caller passed."""
    count = len(contract.parameters)
    if count == 0:
        if param is not None:
            raise ExecutionError(
                "invalid_parameters", f"{contract.operation_id} has no parameters"
            )
        return []
    if count == 1:
        return [param]
    if not isinstance(param, (list, tuple)) or len(param) != count:
        raise ExecutionError(
            "invalid_parameters",
            f"{contract.operation_id} takes {count} parameter arguments",
        )
    return list(param)


def _inputs_match(contract: OperationContract, inputs: Sequence[Any]) -> bool:
    kinds = contract.input_kinds(len(inputs))
    return kinds is not None and all(
        _kind(obj) == kind for obj, kind in zip(inputs, kinds)
    )


def describe_call(
    func: Callable,
    inputs: Sequence[Any],
    param: Any = None,
    *,
    roles: Sequence[str] | None = None,
    contract: OperationContract | None | object = _MISSING,
) -> dict[str, Any]:
    """Describe a call of a computation function as plain data.

    The identity of the operation is kept as soon as it is known, whether or not
    the contract is qualified for replay.

    Args:
        func: Computation function.
        inputs: Input objects, in role order (a list input is flattened).
        param: Parameter argument actually passed (a parameter set, a list of
         parameter sets or None), or a tuple of them for an operation taking
         several parameter arguments.
        roles: Role names to use when *func* has no contract (by default, see
         :func:`default_roles`).
        contract: Contract of *func*, if already resolved (None for none); by
         default, looked up with :func:`~sigima.proc.contracts.contract_for_function`.

    Returns:
        ``{"operation", "qualification", "pattern", "roles", "parameters",
        "parameter_codec", "options", "limits", "implementation"}``:
        ``operation`` is ``{"id", "contract_version"}`` or None;
        ``qualification`` is ``{"status", "comparison"}`` (status ``qualified``,
        ``not_qualified``, ``qualification_outdated`` or ``no_contract``;
        comparison rule as plain data when qualified); ``parameters`` are the
        encoded parameters or None when they could not be encoded
        (``parameters_not_encoded`` in ``limits``); ``options`` are the current
        values of the global options read by the computation (see
        :func:`prepare_execution`), to be recorded in the execution context.
    """
    if contract is _MISSING:
        contract = contract_for_function(func)
    limits: list[str] = []
    description: dict[str, Any] = {
        "operation": None,
        "qualification": {"status": "no_contract", "comparison": None},
        "pattern": None,
        "roles": list(roles) if roles is not None else default_roles(len(inputs)),
        "parameters": None,
        "parameter_codec": PARAMETER_CODEC_VERSION,
        "options": {},
        "limits": limits,
        "implementation": implementation_of(func),
    }
    if contract is None:
        try:
            description["parameters"] = _encode_without_contract(param)
        except (ParameterEncodingError, ValueError):
            limits.append("parameters_not_encoded")
        return description
    assert isinstance(contract, OperationContract)
    description["operation"] = {
        "id": contract.operation_id,
        "contract_version": contract.contract_version,
    }
    comparison = contract.comparison
    description["qualification"] = {
        "status": contract.qualification_status,
        "comparison": None if comparison is None else comparison.to_dict(),
    }
    description["pattern"] = contract.pattern
    description["options"] = _current_options(contract)
    if _inputs_match(contract, inputs):
        description["roles"] = contract.input_role_names(len(inputs))
    else:
        limits.append("inputs_do_not_match_contract")
    try:
        description["parameters"] = contract.encode_parameters(
            _parameter_arguments(contract, param)
        )
    except (ParameterEncodingError, ExecutionError):
        limits.append("parameters_not_encoded")
    return description


def _current_options(contract: OperationContract) -> dict[str, Any]:
    """Return the current values of the global options read by *contract*."""
    return {name: getattr(options, name).get() for name in contract.reads_options}


def _encode_without_contract(param: Any) -> dict[str, Any]:
    """Encode the parameters of a call without contract (informative only)."""
    # pylint: disable=import-outside-toplevel
    import guidata.dataset as gds

    from sigima.proc.contracts.codec import encode_dataset

    if param is None:
        return {}
    if isinstance(param, gds.DataSet):
        return encode_dataset(param)
    return {"value": encode_value("value", param)}


@dataclasses.dataclass(frozen=True)
class PreparedExecution:
    """A computation ready to run, with the context to record.

    Attributes:
        function: Computation function.
        args: Positional arguments: prepared inputs (a list input as a list),
         then parameter arguments.
        context: Execution context to record: ``x_alignment`` (record of the
         policy applied, or None) and, when the computation reads global options,
         ``options`` (their values).
        options: Global option values applied while the function runs.
    """

    function: Callable
    args: tuple[Any, ...]
    context: dict[str, Any]
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    def applied_options(self) -> contextlib.AbstractContextManager[None]:
        """Return a context applying the global options, restored on exit.

        For applications that run :attr:`function` (or the same function from
        their catalogue) themselves with :attr:`args`.
        """
        return _options_applied(self.options)

    def run(self) -> Any:
        """Run the function with the recorded global options, then restore them."""
        with self.applied_options():
            return self.function(*self.args)


def execute(prepared: PreparedExecution) -> Any:
    """Run *prepared* (module-level helper for process pools and workers)."""
    return prepared.run()


@contextlib.contextmanager
def _options_applied(values: dict[str, Any]) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        for name, value in values.items():
            stack.enter_context(getattr(options, name).context(value))
        yield


def _call_arguments(
    contract: OperationContract, inputs: list[Any], arguments: list[Any]
) -> tuple[Any, ...]:
    fixed = sum(role.cardinality or 0 for role in contract.inputs)
    args: list[Any] = []
    index = 0
    for role in contract.inputs:
        if role.variadic:
            count = len(inputs) - fixed
            args.append(list(inputs[index : index + count]))
        else:
            count = role.cardinality
            args.extend(inputs[index : index + count])
        index += count
    return (*args, *arguments)


def prepare_execution(
    contract: OperationContract,
    inputs: Sequence[Any],
    *,
    param: Any = _MISSING,
    values: Any = _MISSING,
    context: dict[str, Any] | None = None,
) -> PreparedExecution:
    """Prepare a computation, in ordinary processing or for a replay.

    The inputs are never modified. Ordinary processing passes the parameter
    argument actually used (*param*) and no *context*; a replay passes the
    recorded encoded parameters (*values*) and the recorded *context*, whose
    policy and global options must be those declared by the contract. A replay
    works on detached copies of the inputs (all metadata included), so that the
    live objects are protected even when the computation runs in the same
    process.

    Args:
        contract: Operation contract.
        inputs: Input objects, in role order (a list input is flattened).
        param: Parameter argument (see :func:`describe_call`).
        values: Encoded parameters, for a replay.
        context: Recorded execution context, for a replay.

    Returns:
        The prepared execution.

    Raises:
        ExecutionError: ``input_mismatch`` if the inputs do not fit the roles,
         ``invalid_parameters`` if the parameters cannot be decoded or do not fit,
         ``unsupported_context`` if a recorded global option is missing.
        XAlignmentError: If a recorded policy is unknown or the inputs cannot be
         aligned (scientific refusal, also in ordinary processing).
    """
    if (param is _MISSING) == (values is _MISSING):
        raise TypeError("Pass exactly one of param or values")
    if not _inputs_match(contract, inputs):
        raise ExecutionError(
            "input_mismatch",
            f"{contract.operation_id}: inputs do not match roles {contract.inputs}",
        )
    if values is not _MISSING:
        try:
            arguments = contract.decode_parameters(values)
        except InvalidParametersError as exc:
            raise ExecutionError("invalid_parameters", str(exc)) from exc
    else:
        arguments = _parameter_arguments(contract, param)
    replay = context is not None
    option_values: dict[str, Any] = {}
    if contract.reads_options:
        if replay:
            recorded = context.get("options")
            if not isinstance(recorded, dict) or any(
                name not in recorded for name in contract.reads_options
            ):
                raise ExecutionError(
                    "unsupported_context",
                    f"Global options {list(contract.reads_options)} were not recorded",
                )
            option_values = {name: recorded[name] for name in contract.reads_options}
        else:
            option_values = _current_options(contract)
    record = None
    prepared_inputs = list(inputs)
    if replay:
        prepared_inputs = [obj.copy(all_metadata=True) for obj in prepared_inputs]
    if contract.x_alignment is not None:
        if replay:
            check_x_alignment_record(context.get("x_alignment"), contract.x_alignment)
        policy = get_x_alignment_policy(contract.x_alignment)
        prepared_inputs, record = policy.apply(prepared_inputs)
    exec_context: dict[str, Any] = {"x_alignment": record}
    if option_values:
        exec_context["options"] = dict(option_values)
    return PreparedExecution(
        function=contract.function,
        args=_call_arguments(contract, prepared_inputs, arguments),
        context=exec_context,
        options=option_values,
    )


def normalize_outputs(
    contract: OperationContract, result: Any
) -> list[tuple[str, Any]]:
    """Return the outputs of *result* by role.

    Args:
        contract: Operation contract.
        result: Value returned by the function.

    Returns:
        ``[(role, value)]`` in output order; *value* is None for an optional
        output without result (e.g. a detection that found nothing).

    Raises:
        ExecutionError: ``unexpected_output`` if *result* does not match the
         output roles.
    """
    (role,) = contract.outputs
    if result is None:
        if role.optional:
            return [(role.name, None)]
        raise ExecutionError(
            "unexpected_output", f"{contract.operation_id} returned no result"
        )
    if _kind(result) != role.kind:
        raise ExecutionError(
            "unexpected_output",
            f"{contract.operation_id} returned {type(result).__name__}, "
            f"expected {role.kind}",
        )
    return [(role.name, result)]
