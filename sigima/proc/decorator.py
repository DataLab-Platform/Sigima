# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Computation function decorator and utilities
(see parent package :mod:`sigima.computation`)
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import os.path as osp
import pkgutil
import sys
import typing
from typing import Callable, Literal, TypeVar

import guidata.dataset as gds
import makefun

from sigima.objects.scalar.geometry import GeometryResult
from sigima.objects.scalar.table import TableResult
from sigima.validation import validate_dataset

if sys.version_info >= (3, 10):
    # Use ParamSpec from typing module in Python 3.10+
    from typing import ParamSpec
else:
    # Use ParamSpec from typing_extensions module in Python < 3.10
    from typing_extensions import ParamSpec

# NOTE: Parameter classes should NOT be included in __all__ to avoid Sphinx
# cross-reference conflicts. All parameter classes are re-exported through
# sigima.params module which serves as the single source of truth for the
# public API. Only utility functions should be exported from this module.
__all__ = [
    "computation_function",
    "find_computation_functions",
    "get_computation_metadata",
    "is_computation_function",
]

# Marker attribute used by @computation_function and introspection
COMPUTATION_METADATA_ATTR = "__computation_function_metadata"

P = ParamSpec("P")
R = TypeVar("R")


@dataclasses.dataclass(frozen=True)
class ComputationMetadata:
    """Metadata for a computation function.

    Attributes:
        name: The name of the computation function.
        description: A description or docstring for the computation function.
    """

    name: str
    description: str


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

        validate_dataset(ds_obj, *validation_context)

        # Call the original function
        result = f(*final_args)

        # Auto-inject func_name into result objects if they support it
        if (
            isinstance(result, (TableResult, GeometryResult))
            and result.func_name is None
        ):
            # Since results are frozen dataclasses, we need to recreate them
            result = dataclasses.replace(result, func_name=f.__name__)

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

    Returns:
        The decorated, enhanced computation function.
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        # Gather signature and typing information
        sig = inspect.signature(f)
        params = list(sig.parameters.values())
        try:
            type_hints = typing.get_type_hints(f)
        except Exception:  # pylint: disable=broad-except
            # Python 3.9 cannot evaluate stringized PEP 604 return annotations.
            # Preserve direct parameter-class references resolved from module globals.
            type_hints = {
                p.name: f.__globals__.get(p.annotation, p.annotation)
                if isinstance(p.annotation, str)
                else p.annotation
                for p in params
            }

        # Find DataSet parameter if any
        ds_param = None
        ds_cls = None
        for p in params:
            annot = type_hints.get(p.name, p.annotation)
            if (
                annot is not inspect.Signature.empty
                and isinstance(annot, type)
                and issubclass(annot, gds.DataSet)
                and annot.__name__ not in ("SignalObj", "ImageObj")
            ):
                ds_param = p
                ds_cls = annot
                break

        # If a DataSet param is present, expand signature and docstring
        if ds_cls is not None:
            # Build signature exposing all DataSet items as keyword-only parameters
            ds_items: list[gds.DataItem] = ds_cls._items  # pylint: disable=W0212
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
            metadata = ComputationMetadata(
                name=name or f.__name__,
                description=description or f.__doc__,
            )
            return _make_computation_wrapper(
                f, ds_cls, ds_param, params, ds_items, new_sig, signature_info, metadata
            )

        # No DataSet parameter: simple passthrough with func_name injection
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)

            # Auto-inject func_name into result objects if they support it
            if (
                isinstance(result, (TableResult, GeometryResult))
                and result.func_name is None
            ):
                # Since results are frozen dataclasses, we need to recreate them
                result = dataclasses.replace(result, func_name=f.__name__)

            return result

        metadata = ComputationMetadata(
            name=name or f.__name__,
            description=description or f.__doc__,
        )
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


def find_computation_functions() -> list[tuple[str, Callable]]:
    """Find all computation functions in the `sigima.proc` package.

    This function uses introspection to locate all functions decorated with
    `@computation_function` in the `sigima.proc` package and its subpackages.

    Args:
        module: Optional module to search in. If None, the current module is used.

    Returns:
        A list of tuples, each containing the function name and the function object.
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
