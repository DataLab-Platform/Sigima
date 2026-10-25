# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Computation function decorator and utilities
(see parent package :mod:`sigima.computation`)
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import os.path as osp
import pkgutil
import re
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Callable, Literal, ParamSpec, TypeVar

import guidata.dataset as gds
import makefun

from sigima.objects.scalar.geometry import GeometryResult
from sigima.objects.scalar.table import TableResult
from sigima.proc.contracts._annotations import (
    AnnotationError,
    resolve_annotation,
    split_optional,
)
from sigima.proc.contracts.declarations import NAME_PATTERN, ReplayQualification
from sigima.validation import validate_dataset

# NOTE: Parameter classes should NOT be included in __all__ to avoid Sphinx
# cross-reference conflicts. All parameter classes are re-exported through
# sigima.params module which serves as the single source of truth for the
# public API. Only utility functions should be exported from this module.
__all__ = [
    "computation_function",
    "find_computation_functions",
    "get_computation_metadata",
    "is_computation_function",
    "observe_calls",
]

# Marker attribute used by @computation_function and introspection
COMPUTATION_METADATA_ATTR = "__computation_function_metadata"

# Same character set as recipe identifiers, e.g. "sigima.signal.normalize"
OPERATION_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")

P = ParamSpec("P")
R = TypeVar("R")

#: Observer of computation calls (see :func:`observe_calls`), None by default.
_CALL_OBSERVER: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "sigima_call_observer", default=None
)
#: True while an observed call runs: nested calls are not observed.
_IN_OBSERVED_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sigima_in_observed_call", default=False
)


@contextlib.contextmanager
def observe_calls(
    observer: Callable[[Callable, tuple], Callable[[Any], None] | None],
) -> Iterator[None]:
    """Observe the outermost calls of computation functions within the context.

    Used by test and audit tools (e.g. the replay qualification tests): nothing
    is observed by default, and observation never changes a result.

    Args:
        observer: ``(function, args) -> on_result or None``, called before each
         outermost call with the decorated function and the positional arguments
         passed to the original function (parameter sets included); the returned
         callable, if any, receives the result.
    """
    token = _CALL_OBSERVER.set(observer)
    try:
        yield
    finally:
        _CALL_OBSERVER.reset(token)


def _begin_observation(
    function: Callable, args: Sequence[Any]
) -> tuple[Any, Callable[[Any], None] | None]:
    """Notify the observer of an outermost call; return ``(token, on_result)``."""
    observer = _CALL_OBSERVER.get()
    if observer is None or _IN_OBSERVED_CALL.get():
        return None, None
    on_result = observer(function, tuple(args))
    return _IN_OBSERVED_CALL.set(True), on_result


def _end_observation(token: Any) -> None:
    if token is not None:
        _IN_OBSERVED_CALL.reset(token)


@dataclasses.dataclass(frozen=True)
class ComputationMetadata:
    """Metadata for a computation function.

    Attributes:
        name: The name of the computation function.
        description: A description or docstring for the computation function.
        operation_id: Stable operation identifier, or None if the function
         declares no operation contract.
        contract_version: Contract version (integer >= 1) when ``operation_id``
         is set, None otherwise.
        aliases: Earlier operation identifiers of the same operation.
        qualification: Replay qualification of a tested contract version, or
         None.
        x_alignment: Identifier of the X-alignment policy applied to the
         inputs, or None.
        reads_options: Names of the :data:`sigima.config.options` read by the
         computation itself.
        reads_metadata: ``((role, (key, ...)), ...)``: metadata keys of input
         objects read by the computation itself.
        roles: Explicit input role names, or None to infer them.
    """

    name: str
    description: str
    operation_id: str | None = None
    contract_version: int | None = None
    aliases: tuple[str, ...] = ()
    qualification: ReplayQualification | None = None
    x_alignment: str | None = None
    reads_options: tuple[str, ...] = ()
    reads_metadata: tuple[tuple[str, tuple[str, ...]], ...] = ()
    roles: tuple[str, ...] | None = None


def _as_names(value: Sequence[str] | None, label: str) -> tuple[str, ...] | None:
    """Return *value* as a tuple of unique names, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        raise ValueError(f"{label} must be a sequence of names, not a string")
    names = tuple(value)
    if any(not isinstance(n, str) or not NAME_PATTERN.match(n) for n in names):
        raise ValueError(f"Invalid {label}: {names!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must be unique")
    return names


def _as_metadata_reads(
    value: Mapping[str, Sequence[str]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return ``reads_metadata`` as ``((role, (key, ...)), ...)``."""
    if not value:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError("reads_metadata must map input role names to metadata keys")
    reads = []
    for role, keys in value.items():
        if not isinstance(role, str) or not NAME_PATTERN.match(role):
            raise ValueError(f"Invalid role in reads_metadata: {role!r}")
        if isinstance(keys, str):
            raise ValueError("reads_metadata keys must be a sequence, not a string")
        keys = tuple(keys)
        if not keys or any(not isinstance(k, str) or not k for k in keys):
            raise ValueError(f"Invalid metadata keys for role {role!r}: {keys!r}")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Metadata keys of role {role!r} must be unique")
        reads.append((role, keys))
    return tuple(reads)


def _check_contract_declarations(
    operation_id: str | None,
    qualification: ReplayQualification | None,
    x_alignment: str | None,
    reads_options: Sequence[str],
    roles: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Validate contract declarations; return ``(reads_options, roles)``.

    Raises:
        ValueError: If the declarations are inconsistent or malformed.
    """
    reads_options = _as_names(reads_options, "reads_options")
    roles = _as_names(roles, "roles")
    if operation_id is None and (
        qualification is not None or x_alignment is not None or reads_options or roles
    ):
        raise ValueError(
            "qualification, x_alignment, reads_options, reads_metadata and roles "
            "require an operation_id"
        )
    if qualification is not None and not isinstance(qualification, ReplayQualification):
        raise ValueError("qualification must be a ReplayQualification")
    if x_alignment is not None and (
        not isinstance(x_alignment, str) or not OPERATION_ID_PATTERN.match(x_alignment)
    ):
        raise ValueError(f"Invalid X-alignment policy identifier: {x_alignment!r}")
    if roles is not None and not roles:
        raise ValueError("roles must not be empty")
    return reads_options, roles


def _check_operation_identity(
    operation_id: str | None,
    contract_version: int | None,
    aliases: Sequence[str],
) -> tuple[str, ...]:
    """Validate operation identity arguments and return aliases as a tuple.

    Raises:
        ValueError: If the identity arguments are inconsistent or malformed.
    """
    if isinstance(aliases, str):
        raise ValueError("aliases must be a sequence of identifiers, not a string")
    aliases = tuple(aliases)
    if operation_id is None:
        if contract_version is not None or aliases:
            raise ValueError("contract_version and aliases require an operation_id")
        return aliases
    for identifier in (operation_id,) + aliases:
        if not isinstance(identifier, str) or not OPERATION_ID_PATTERN.match(
            identifier
        ):
            raise ValueError(f"Invalid operation identifier: {identifier!r}")
    if (
        not isinstance(contract_version, int)
        or isinstance(contract_version, bool)
        or contract_version < 1
    ):
        raise ValueError("operation_id requires an integer contract_version >= 1")
    if operation_id in aliases or len(set(aliases)) != len(aliases):
        raise ValueError("aliases must be unique and differ from operation_id")
    return aliases


def _dataset_class(annotation: Any, namespace: dict[str, Any]) -> type | None:
    """Return the parameter class of an annotation, if it designates one.

    ``SomeParam`` and ``SomeParam | None`` (or ``Optional[SomeParam]``) designate
    ``SomeParam``. String annotations are resolved without :func:`eval`
    (see :mod:`sigima.proc.contracts._annotations`).
    """
    if annotation is inspect.Signature.empty:
        return None
    try:
        tp, _optional = split_optional(resolve_annotation(annotation, namespace))
    except AnnotationError:
        return None
    if (
        isinstance(tp, type)
        and issubclass(tp, gds.DataSet)
        and tp.__name__ not in ("SignalObj", "ImageObj")
    ):
        return tp
    return None


def _is_default(value: Any, item: gds.DataItem) -> bool:
    """Return True if *value* is the default value of *item*."""
    default = item.get_default()
    if value is default:
        return True
    try:
        return bool(value == default)
    except (TypeError, ValueError):  # e.g. arrays
        return False


def _make_computation_wrapper(
    f: Callable,
    ds_cls: type,
    ds_param: inspect.Parameter,
    params: list,
    ds_items: list,
    new_sig: inspect.Signature,
    signature_info: str,
    metadata: ComputationMetadata,
) -> Callable:
    """
    Create a computation function wrapper supporting both DataSet and expanded-kwarg
    signatures.

    Args:
        f: The original function.
        ds_cls: The DataSet class type.
        ds_param: The DataSet parameter in the signature.
        params: The full function signature parameters.
        ds_items: The DataSet's items (parameters).
        new_sig: The explicit signature (with kwargs) to expose.
        signature_info: The Sphinx docstring note to append.
        metadata: ComputationMetadata to attach to the wrapper.

    Returns:
        The wrapped function.
    """

    @makefun.with_signature(new_sig)
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        """
        Dispatch function supporting both DataSet parameter and expanded keyword
        arguments.

        Behavior:
            - If a DataSet object is provided, it is always used and keyword arguments
              for DataSet items are ignored.
            - If no DataSet is provided, DataSet items are constructed from keyword
              arguments.

        Returns:
            Result of the original computation function.
        """
        ba = new_sig.bind(*args, **kwargs)
        ba.apply_defaults()
        ds_obj = ba.arguments.get(ds_param.name, None)
        ds_item_names = set(item.get_name() for item in ds_items)

        if isinstance(ds_obj, ds_cls):
            # DataSet object provided: ignore any keyword arguments for its items
            pass
        else:
            # DataSet instance not provided: build from keyword arguments
            ds_kwargs = {
                k: ba.arguments.pop(k)
                for k in list(ba.arguments.keys())
                if k in ds_item_names
            }
            if ds_param.default is None and all(
                _is_default(ds_kwargs.get(item.get_name(), item.get_default()), item)
                for item in ds_items
            ):
                # Optional parameter set, not given (keywords at their defaults):
                # the function applies its own default, e.g. a global option
                # read by the parameter set
                ds_obj = None
            else:
                ds_obj = ds_cls.create(**ds_kwargs)

        # Build the final positional argument list for the original function
        final_args = []
        validation_context = []
        for p in params:
            if p is ds_param:
                final_args.append(ds_obj)
            else:
                value = ba.arguments.get(p.name, None)
                final_args.append(value)
                validation_context.append(value)

        if ds_obj is not None:
            validate_dataset(ds_obj, *validation_context)

        # Call the original function
        token, on_result = _begin_observation(wrapper, final_args)
        try:
            result = f(*final_args)
        finally:
            _end_observation(token)

        # Auto-inject func_name into result objects if they support it
        if (
            isinstance(result, (TableResult, GeometryResult))
            and result.func_name is None
        ):
            # Since results are frozen dataclasses, we need to recreate them
            result = dataclasses.replace(result, func_name=f.__name__)

        if on_result is not None:
            on_result(result)
        return result

    # Attach dynamic Sphinx docstring and signature
    doc = f.__doc__ or ""
    if not doc.endswith("\n"):
        doc += "\n"
    wrapper.__doc__ = doc + signature_info
    wrapper.__signature__ = new_sig
    setattr(wrapper, COMPUTATION_METADATA_ATTR, metadata)
    return wrapper


def computation_function(
    *,
    name: typing.Optional[str] = None,
    description: typing.Optional[str] = None,
    operation_id: typing.Optional[str] = None,
    contract_version: typing.Optional[int] = None,
    aliases: Sequence[str] = (),
    qualification: typing.Optional[ReplayQualification] = None,
    x_alignment: typing.Optional[str] = None,
    reads_options: Sequence[str] = (),
    reads_metadata: typing.Optional[Mapping[str, Sequence[str]]] = None,
    roles: typing.Optional[Sequence[str]] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to mark a function as a Sigima computation function.

    This decorator enables two calling conventions:
      1. With a guidata DataSet object as a parameter (classic style).
      2. With the DataSet items passed as individual keyword arguments (expanded style).

    The decorator ensures:
      - An explicit and informative function signature (including all DataSet items as
        keyword arguments).
      - A Sphinx-friendly docstring documenting both call styles.
      - Pickle-compatibility (crucial for multiprocessing).
      - Conflict detection if both DataSet instance and expanded keyword arguments are
        used simultaneously.

    Args:
        name: Optional custom name for metadata.
        description: Optional custom description or docstring.
        operation_id: Optional stable operation identifier (see
         :mod:`sigima.proc.contracts`). Never derived from the Python name.
        contract_version: Contract version, required with ``operation_id``: 1 at
         the first declaration, increased whenever a change of the function changes
         the result it returns, whatever the releases of the applications using
         Sigima (see the contributing guidelines, "Operation contracts").
        aliases: Earlier operation identifiers, only with ``operation_id``.
        qualification: Replay qualification of the tested contract version
         (see :class:`sigima.proc.contracts.declarations.ReplayQualification`), only
         with ``operation_id``.
        x_alignment: Identifier of the X-alignment policy applied to the inputs
         before the function runs (see :mod:`sigima.proc.contracts.policies`), only with
         ``operation_id``.
        reads_options: Names of the :data:`sigima.config.options` that the
         computation reads, only with ``operation_id``.
        reads_metadata: Metadata keys of input objects that the computation
         reads, by input role name (e.g. ``{"source": ("fit_params",)}``), only
         with ``operation_id``. They are not covered by state fingerprints:
         applications record their values and check them before a replay.
        roles: Input role names, only with ``operation_id``, when they cannot
         be inferred from the signature.

    Returns:
        The decorated, enhanced computation function.

    Raises:
        ValueError: If the operation identity or contract declarations are
         invalid.
    """
    aliases = _check_operation_identity(operation_id, contract_version, aliases)
    reads_options, roles = _check_contract_declarations(
        operation_id, qualification, x_alignment, reads_options, roles
    )
    reads_metadata = _as_metadata_reads(reads_metadata)
    if reads_metadata and operation_id is None:
        raise ValueError("reads_metadata requires an operation_id")

    def _metadata(f: Callable) -> ComputationMetadata:
        return ComputationMetadata(
            name=name or f.__name__,
            description=description or f.__doc__,
            operation_id=operation_id,
            contract_version=contract_version,
            aliases=aliases,
            qualification=qualification,
            x_alignment=x_alignment,
            reads_options=reads_options,
            reads_metadata=reads_metadata,
            roles=roles,
        )

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        # Gather signature information
        sig = inspect.signature(f)
        params = list(sig.parameters.values())

        # Find DataSet parameter if any (optional or not: ``SomeParam | None``)
        ds_param = None
        ds_cls = None
        for p in params:
            annot = _dataset_class(p.annotation, f.__globals__)
            if annot is not None:
                ds_param = p
                ds_cls = annot
                break

        # If a DataSet param is present, expand signature and docstring
        if ds_cls is not None:
            # Build signature exposing all DataSet items as keyword-only parameters
            ds_items: list[gds.DataItem] = [
                item
                for item in ds_cls._items  # pylint: disable=W0212
                if not item.get_prop("data", "transient", False)
            ]
            item_names = [item.get_name() for item in ds_items]
            items = []
            for item in ds_items:
                if item.get_name() not in [p.name for p in params]:
                    # Support ChoiceItem as Literal if available
                    if hasattr(gds, "ChoiceItem") and isinstance(item, gds.ChoiceItem):
                        choice_data = item.get_prop("data", "choices")
                        choices = [v[0] for v in choice_data]
                        item_type = Literal[tuple(choices)]
                    else:
                        item_type = item.type
                    items.append(
                        inspect.Parameter(
                            item.get_name(),
                            inspect.Parameter.KEYWORD_ONLY,
                            annotation=item_type,
                            default=item.get_default(),
                        )
                    )
            # DataSet parameter remains positional-or-keyword, but optional
            # (default=None)
            base_params = []
            for p in params:
                if p is ds_param:
                    base_params.append(
                        inspect.Parameter(
                            p.name,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=p.annotation,
                            default=None,
                        )
                    )
                else:
                    base_params.append(p)
            new_params = base_params + items
            new_sig = sig.replace(parameters=new_params)
            param_class_name = ds_cls.__name__
            kwarg_example = ", ".join(f"{name}=..." for name in item_names)
            # Sphinx-style docstring describing both call conventions
            signature_info = (
                f".. note::\n\n"
                f"   This computation function can be called in two ways:\n\n"
                f"   1. With a parameter ``{param_class_name}`` object:\n\n"
                f"      .. code-block:: python\n\n"
                f"         param = {param_class_name}.create({kwarg_example})\n"
                f"         func(obj, param)\n\n"
                f"   2. Or, with keyword arguments directly:\n\n"
                f"      .. code-block:: python\n\n"
                f"         func(obj, {kwarg_example})\n\n"
                f"   Both styles are fully supported and equivalent.\n\n"
            )
            metadata = _metadata(f)
            return _make_computation_wrapper(
                f, ds_cls, ds_param, params, ds_items, new_sig, signature_info, metadata
            )

        # No DataSet parameter: simple passthrough with func_name injection
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if _CALL_OBSERVER.get() is not None:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                token, on_result = _begin_observation(
                    wrapper, tuple(bound.arguments.values())
                )
            else:
                token = on_result = None
            try:
                result = f(*args, **kwargs)
            finally:
                _end_observation(token)

            # Auto-inject func_name into result objects if they support it
            if (
                isinstance(result, (TableResult, GeometryResult))
                and result.func_name is None
            ):
                # Since results are frozen dataclasses, we need to recreate them
                result = dataclasses.replace(result, func_name=f.__name__)

            if on_result is not None:
                on_result(result)
            return result

        metadata = _metadata(f)
        setattr(wrapper, COMPUTATION_METADATA_ATTR, metadata)
        return wrapper

    return decorator


def is_computation_function(function: Callable) -> bool:
    """Check if a function is a Sigima computation function.

    Args:
        function: The function to check.

    Returns:
        True if the function is a Sigima computation function, False otherwise.
    """
    return getattr(function, COMPUTATION_METADATA_ATTR, None) is not None


def get_computation_metadata(function: Callable) -> ComputationMetadata:
    """Get the metadata of a Sigima computation function.

    Args:
        function: The function to get metadata from.

    Returns:
        Computation function metadata.

    Raises:
        ValueError: If the function is not a Sigima computation function.
    """
    metadata = getattr(function, COMPUTATION_METADATA_ATTR, None)
    if not isinstance(metadata, ComputationMetadata):
        raise ValueError(
            f"The function {function.__name__} is not a Sigima computation function."
        )
    return metadata


def find_computation_functions() -> list[tuple[str, str, str | None]]:
    """Find all computation functions in the `sigima.proc` package.

    This function uses introspection to locate all functions decorated with
    `@computation_function` in the `sigima.proc` package and its subpackages.

    Returns:
        A list of ``(module name, function name, docstring)`` tuples.
    """
    functions = []
    objs = []
    for _, modname, _ in pkgutil.walk_packages(
        path=[osp.dirname(__file__)], prefix=".".join(__name__.split(".")[:-1]) + "."
    ):
        module = importlib.import_module(modname)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if is_computation_function(obj):
                if obj in objs:  # Avoid double entries for the same function
                    continue
                objs.append(obj)
                functions.append((modname, name, obj.__doc__))
    return functions
