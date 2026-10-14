# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Annotation analysis for operation contracts
===========================================

Computation modules use ``from __future__ import annotations``: their
annotations are strings. On Python 3.9, :func:`typing.get_type_hints` cannot
evaluate ``X | None``. This module resolves such strings without :func:`eval`:
the string is parsed with :mod:`ast` and only names, attribute accesses,
subscripts and ``|`` unions are resolved, against the namespace of the module
that defines the function.

Annotations come from Sigima source code, never from recorded data.
"""

from __future__ import annotations

import ast
import builtins
import collections.abc
import types
import typing
from typing import Any, Union

__all__ = [
    "AnnotationError",
    "resolve_annotation",
    "sequence_item",
    "split_optional",
    "union_members",
]

NoneType = type(None)

# ``X | Y`` evaluated at runtime (Python >= 3.10) is a ``types.UnionType``.
_UNION_TYPE = getattr(types, "UnionType", None)

_SEQUENCE_ORIGINS = (list, tuple, collections.abc.Sequence)


class AnnotationError(TypeError):
    """Raised when an annotation cannot be resolved."""


def _resolve_node(node: ast.AST, namespace: dict[str, Any]) -> Any:
    # pylint: disable=too-many-return-statements
    if isinstance(node, ast.Constant):
        if node.value is None:
            return NoneType
        if node.value is Ellipsis:
            return Ellipsis
        if isinstance(node.value, str):
            return resolve_annotation(node.value, namespace)
        raise AnnotationError(f"Unsupported constant in annotation: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in namespace:
            return namespace[node.id]
        if hasattr(builtins, node.id):
            return getattr(builtins, node.id)
        raise AnnotationError(f"Unknown name in annotation: {node.id!r}")
    if isinstance(node, ast.Attribute):
        base = _resolve_node(node.value, namespace)
        try:
            return getattr(base, node.attr)
        except AttributeError as exc:
            raise AnnotationError(
                f"Unknown attribute in annotation: {node.attr}"
            ) from exc
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _resolve_node(node.left, namespace)
        right = _resolve_node(node.right, namespace)
        return Union[left, right]
    if isinstance(node, ast.Subscript):
        origin = _resolve_node(node.value, namespace)
        index = node.slice
        elts = index.elts if isinstance(index, ast.Tuple) else [index]
        args = tuple(_resolve_node(elt, namespace) for elt in elts)
        try:
            return origin[args if len(args) > 1 else args[0]]
        except TypeError as exc:
            raise AnnotationError(
                f"Unsupported subscript in annotation: {exc}"
            ) from exc
    raise AnnotationError(f"Unsupported annotation syntax: {ast.dump(node)}")


def resolve_annotation(annotation: Any, namespace: dict[str, Any]) -> Any:
    """Return the type designated by *annotation*.

    Args:
        annotation: Annotation object or string.
        namespace: Globals of the module defining the annotated function.

    Returns:
        A class or a :mod:`typing` construct (``Union``, ``list[...]``...).

    Raises:
        AnnotationError: If the annotation is missing or cannot be resolved.
    """
    if annotation is None:
        return NoneType
    if not isinstance(annotation, str):
        return annotation
    try:
        node = ast.parse(annotation, mode="eval").body
    except SyntaxError as exc:
        raise AnnotationError(f"Invalid annotation: {annotation!r}") from exc
    return _resolve_node(node, namespace)


def union_members(tp: Any) -> tuple[Any, ...]:
    """Return the members of a union, or ``(tp,)``."""
    origin = typing.get_origin(tp)
    if origin is Union or (_UNION_TYPE is not None and origin is _UNION_TYPE):
        return typing.get_args(tp)
    return (tp,)


def split_optional(tp: Any) -> tuple[Any, bool]:
    """Return ``(type without None, optional)``."""
    members = union_members(tp)
    if NoneType not in members:
        return tp, False
    rest = tuple(member for member in members if member is not NoneType)
    if not rest:
        raise AnnotationError("An annotation cannot be only None")
    return (rest[0] if len(rest) == 1 else Union[rest]), True


def sequence_item(tp: Any) -> Any | None:
    """Return the item type of ``list[X]``, ``tuple[X, ...]`` or ``Sequence[X]``.

    Returns:
        The item type, or None if *tp* is not a homogeneous sequence type.
    """
    origin = typing.get_origin(tp)
    if origin not in _SEQUENCE_ORIGINS:
        return None
    args = typing.get_args(tp)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        return None
    return args[0] if len(args) == 1 else None
