# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Operation contract descriptors
==============================

Plain, immutable descriptors used to declare and describe operation contracts:
input and output roles, parameter arguments, comparison rules and replay
qualifications, and the errors raised when using contracts.

This module depends on no other part of :mod:`sigima.proc`, so that the
computation decorator, the parameter codec and the contract registry can all use
it without import cycles. Every descriptor is picklable.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

__all__ = [
    "ARTIFACT_KINDS",
    "EXACT",
    "OBJECT_KINDS",
    "ComparisonRule",
    "ContractDeclarationError",
    "ContractError",
    "IncompatibleContractError",
    "InvalidParametersError",
    "OutputTolerance",
    "ParameterEncodingError",
    "ParameterSpec",
    "ReplayQualification",
    "Role",
    "UnknownOperationError",
    "tolerance",
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


#: Kinds of scientific objects bound to roles.
OBJECT_KINDS = ("signal", "image")
#: Kinds of analysis results recorded as artifacts.
ARTIFACT_KINDS = ("table", "geometry")

#: Role and argument names.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

#: Parameter argument containers.
PARAMETER_CONTAINERS = ("single", "list", "single_or_list")

#: Comparison criteria applicable within the same environment.
SAME_ENVIRONMENT_CRITERIA = ("exact", "tolerance")


@dataclasses.dataclass(frozen=True)
class Role:
    """Typed input or output role of an operation.

    Attributes:
        name: Role name (e.g. ``"source"``, ``"operand"``, ``"sources"``,
         ``"result"``).
        kind: ``"signal"`` or ``"image"`` for objects, ``"table"`` or
         ``"geometry"`` for analysis results.
        cardinality: Number of objects bound to the role, or None for a variable
         number (at least one), bound in order.
        optional: True if the role may be empty (e.g. an analysis without result).
    """

    name: str
    kind: str
    cardinality: int | None = 1
    optional: bool = False

    @property
    def variadic(self) -> bool:
        """True if the role binds a variable number of objects."""
        return self.cardinality is None

    @property
    def is_artifact(self) -> bool:
        """True if the role is an analysis result (table or geometry)."""
        return self.kind in ARTIFACT_KINDS


@dataclasses.dataclass(frozen=True)
class ParameterSpec:
    """Parameter argument of an operation.

    Attributes:
        name: Python argument name.
        paramclass: Parameter class (a :class:`guidata.dataset.DataSet` subclass).
        container: ``"single"`` for one instance, ``"list"`` for a list of
         instances, ``"single_or_list"`` when both are accepted.
        optional: True if None is accepted; None then stands for a default
         instance of *paramclass*.
    """

    name: str
    paramclass: type
    container: str = "single"
    optional: bool = False

    def __post_init__(self) -> None:
        if self.container not in PARAMETER_CONTAINERS:
            raise ValueError(f"Invalid parameter container: {self.container!r}")


@dataclasses.dataclass(frozen=True)
class OutputTolerance:
    """Tolerance accepted on one output.

    A value ``c`` of the candidate matches the reference value ``r`` when
    ``|c - r| <= atol + rtol * |r|``.

    Attributes:
        role: Output role name.
        atol: Absolute tolerance (>= 0).
        rtol: Relative tolerance (>= 0).
    """

    role: str
    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not NAME_PATTERN.match(self.role):
            raise ValueError(f"Invalid output role: {self.role!r}")
        for name in ("atol", "rtol"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite number >= 0")
        if self.atol == 0 and self.rtol == 0:
            raise ValueError("A tolerance needs atol > 0 or rtol > 0")


@dataclasses.dataclass(frozen=True)
class ComparisonRule:
    """Rule used to compare a replayed result with its reference.

    The default rule is exact equality. A tolerance rule is declared only when
    the nature of the computation justifies it (transforms, interpolations,
    iterative fits...), with that justification.

    Attributes:
        tolerances: Per-output tolerances; empty for the exact rule.
        same_environment: Criterion within an identical environment:
         ``"exact"`` (the tolerance only applies across different environments)
         or ``"tolerance"``.
        justification: Why a tolerance is needed (required for a tolerance).
    """

    tolerances: tuple[OutputTolerance, ...] = ()
    same_environment: str = "exact"
    justification: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.tolerances, tuple) or not all(
            isinstance(tol, OutputTolerance) for tol in self.tolerances
        ):
            raise ValueError("tolerances must be a tuple of OutputTolerance")
        roles = [tol.role for tol in self.tolerances]
        if len(set(roles)) != len(roles):
            raise ValueError("A tolerance is declared twice for the same output")
        if self.same_environment not in SAME_ENVIRONMENT_CRITERIA:
            raise ValueError(f"Invalid same_environment: {self.same_environment!r}")
        if self.tolerances:
            if not isinstance(self.justification, str) or not self.justification:
                raise ValueError("A tolerance rule requires a justification")
        elif self.same_environment != "exact" or self.justification:
            raise ValueError("The exact rule has no criterion nor justification")

    @property
    def exact(self) -> bool:
        """True for the exact rule."""
        return not self.tolerances

    def to_dict(self) -> dict[str, Any]:
        """Return the rule as plain JSON values."""
        if self.exact:
            return {"rule": "exact"}
        return {
            "rule": "tolerance",
            "outputs": {
                tol.role: {"atol": float(tol.atol), "rtol": float(tol.rtol)}
                for tol in self.tolerances
            },
            "same_environment": self.same_environment,
            "justification": self.justification,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ComparisonRule:
        """Rebuild a rule from :meth:`to_dict` values.

        Raises:
            ValueError: If *value* is not a valid rule.
        """
        if not isinstance(value, dict):
            raise ValueError("A comparison rule must be a JSON object")
        if value == {"rule": "exact"}:
            return EXACT
        if value.get("rule") != "tolerance" or set(value) != {
            "rule",
            "outputs",
            "same_environment",
            "justification",
        }:
            raise ValueError(f"Invalid comparison rule: {value!r}")
        outputs = value["outputs"]
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError("A tolerance rule needs per-output tolerances")
        tolerances = []
        for role, tol in outputs.items():
            if not isinstance(tol, dict) or set(tol) != {"atol", "rtol"}:
                raise ValueError(f"Invalid tolerance for output {role!r}")
            tolerances.append(OutputTolerance(role, tol["atol"], tol["rtol"]))
        return cls(tuple(tolerances), value["same_environment"], value["justification"])


#: Exact comparison rule (the default).
EXACT = ComparisonRule()


def tolerance(
    justification: str,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    outputs: tuple[str, ...] = ("result",),
    same_environment: str = "exact",
) -> ComparisonRule:
    """Return a tolerance rule applying the same tolerance to *outputs*.

    Args:
        justification: Why the computation cannot be compared exactly.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        outputs: Output roles the tolerance applies to.
        same_environment: ``"exact"`` if results must be identical within an
         identical environment, ``"tolerance"`` otherwise.
    """
    return ComparisonRule(
        tuple(OutputTolerance(role, atol, rtol) for role in outputs),
        same_environment,
        justification,
    )


@dataclasses.dataclass(frozen=True)
class ReplayQualification:
    """Declaration that replay of a contract version is qualified.

    The declaration is backed by tests: the validation test of the function and
    the generic replay test of its contract shape. It names the contract version
    actually tested; increasing the contract version does not qualify the new
    version.

    Attributes:
        contract_version: Contract version that was tested.
        comparison: Rule used to compare a replayed result with its reference.
    """

    contract_version: int
    comparison: ComparisonRule = EXACT

    def __post_init__(self) -> None:
        if (
            isinstance(self.contract_version, bool)
            or not isinstance(self.contract_version, int)
            or self.contract_version < 1
        ):
            raise ValueError("contract_version must be an integer >= 1")
        if not isinstance(self.comparison, ComparisonRule):
            raise ValueError("comparison must be a ComparisonRule")
