# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Execution policies of operation contracts
=========================================

A computation function may declare an X-alignment policy by identifier
(``@computation_function(x_alignment=...)``). The algorithm of each policy is
implemented once and registered here under its identifier and version; this
module never associates a policy with an operation.

Applications record the policy (identifier, version and outcome) with each
execution; a replay accepts only registered policies.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from sigima.proc.alignment import (
    SMALLEST_GRID_LINEAR_HOLD,
    SOURCE_GRID_LINEAR,
    SOURCE_GRID_LINEAR_HOLD,
    XAlignmentError,
    align_operand,
    align_operand_hold,
    align_signals_hold,
)

__all__ = [
    "XAlignmentPolicy",
    "check_x_alignment_record",
    "get_x_alignment_policy",
    "register_x_alignment_policy",
    "x_alignment_policies",
]


@dataclasses.dataclass(frozen=True)
class XAlignmentPolicy:
    """X-alignment policy applied to the inputs of an operation.

    Attributes:
        identifier: Stable policy identifier.
        version: Policy version.
        patterns: Contract patterns the policy applies to (e.g. ``("2_to_1",)``).
        apply: ``(inputs) -> (prepared inputs, record)``; never modifies the
         inputs; *record* holds ``rule``, ``version`` and the outcome.
    """

    identifier: str
    version: int
    patterns: tuple[str, ...]
    apply: Callable[[list[Any]], tuple[list[Any], dict[str, Any]]]


_POLICIES: dict[str, XAlignmentPolicy] = {}


def register_x_alignment_policy(policy: XAlignmentPolicy) -> None:
    """Register an X-alignment policy.

    Raises:
        ValueError: If a policy with the same identifier is already registered.
    """
    if policy.identifier in _POLICIES:
        raise ValueError(f"X-alignment policy already registered: {policy.identifier}")
    _POLICIES[policy.identifier] = policy


def x_alignment_policies() -> tuple[XAlignmentPolicy, ...]:
    """Return the registered X-alignment policies."""
    return tuple(_POLICIES.values())


def get_x_alignment_policy(identifier: str) -> XAlignmentPolicy:
    """Return the registered X-alignment policy *identifier*.

    Raises:
        XAlignmentError: ``unknown_rule`` if no such policy is registered.
    """
    policy = _POLICIES.get(identifier)
    if policy is None:
        raise XAlignmentError(
            "unknown_rule", f"Unsupported X alignment rule: {identifier!r}"
        )
    return policy


def check_x_alignment_record(record: Any, identifier: str) -> None:
    """Check that a recorded alignment names the expected, supported policy.

    Args:
        record: Recorded ``x_alignment`` context.
        identifier: Policy declared by the contract.

    Raises:
        XAlignmentError: ``unknown_rule`` if the record is missing, malformed,
         names another policy or an unsupported version.
    """
    if not isinstance(record, dict):
        raise XAlignmentError("unknown_rule", "No X alignment rule was recorded")
    policy = _POLICIES.get(identifier)
    key = (record.get("rule"), record.get("version"))
    if policy is None or key != (policy.identifier, policy.version):
        raise XAlignmentError(
            "unknown_rule", f"Unsupported X alignment rule: {key[0]!r} v{key[1]!r}"
        )


def _source_grid_linear(inputs: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    source, operand = inputs
    aligned, record = align_operand(source, operand)
    return [source, aligned], record


def _source_grid_linear_hold(inputs: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    source, operand = inputs
    aligned, record = align_operand_hold(source, operand)
    return [source, aligned], record


register_x_alignment_policy(
    XAlignmentPolicy(SOURCE_GRID_LINEAR, 1, ("2_to_1",), _source_grid_linear)
)
register_x_alignment_policy(
    XAlignmentPolicy(SOURCE_GRID_LINEAR_HOLD, 1, ("2_to_1",), _source_grid_linear_hold)
)
register_x_alignment_policy(
    XAlignmentPolicy(SMALLEST_GRID_LINEAR_HOLD, 1, ("n_to_1",), align_signals_hold)
)
