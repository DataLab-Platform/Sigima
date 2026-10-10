# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
X alignment of two-input signal operations
==========================================

Rule ``sigima.signal.x_alignment.source_grid_linear`` (version 1): the result is
computed on the grid of the ``source`` signal.

- If both grids are exactly equal (same length, same values), nothing changes.
- Otherwise the ``operand`` is linearly interpolated on the source grid. Both
  grids must be finite and strictly increasing, the operand grid must cover the
  source grid (no extrapolation, no constant extension at the ends) and the
  operand must have no uncertainty rows.
- Units are checked separately for X and Y, as exact strings: identical units
  are accepted, including two empty units; an empty unit never matches a
  non-empty one, and different units (e.g. ``V`` and ``mV``) are refused. No
  conversion is ever applied.

The inputs are never modified. Applications record the rule (identifier and
version) with each execution; a replay accepts only the rules listed in
:data:`X_ALIGNMENT_RULES`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sigima.objects import SignalObj

__all__ = [
    "SOURCE_GRID_LINEAR",
    "X_ALIGNMENT_RULES",
    "XAlignmentError",
    "align_operand",
    "check_alignment_record",
]

#: Identifier of the source-grid rule.
SOURCE_GRID_LINEAR = "sigima.signal.x_alignment.source_grid_linear"
#: Supported ``(rule, version)`` pairs.
X_ALIGNMENT_RULES = frozenset({(SOURCE_GRID_LINEAR, 1)})


class XAlignmentError(ValueError):
    """Raised when two signals cannot be aligned by the rule.

    Args:
        code: Reason code (``incompatible_units``, ``invalid_grid``,
         ``insufficient_coverage``, ``uncertainty``, ``unknown_rule``).
        message: Human-readable detail.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _check_grid(role: str, x: np.ndarray) -> None:
    if x.ndim != 1 or x.size < 2:
        raise XAlignmentError(
            "invalid_grid", f"The {role} grid has fewer than 2 points"
        )
    if not np.all(np.isfinite(x)):
        raise XAlignmentError("invalid_grid", f"The {role} grid is not finite")
    if not np.all(np.diff(x) > 0):
        raise XAlignmentError(
            "invalid_grid", f"The {role} grid is not strictly increasing"
        )


def align_operand(
    source: SignalObj, operand: SignalObj
) -> tuple[SignalObj, dict[str, Any]]:
    """Return the operand on the source grid, and the alignment record.

    Args:
        source: Signal whose grid is kept.
        operand: Signal to align; it is returned unchanged when the grids are
         exactly equal, otherwise a new interpolated signal is returned.

    Returns:
        ``(operand, record)``; *record* holds ``rule``, ``version`` and
        ``interpolated``.

    Raises:
        XAlignmentError: If the units differ or the grids cannot be aligned.
    """
    for axis in ("x", "y"):
        src_unit = getattr(source, f"{axis}unit")
        op_unit = getattr(operand, f"{axis}unit")
        if src_unit != op_unit:
            raise XAlignmentError(
                "incompatible_units",
                f"{axis.upper()} units differ: {src_unit!r} and {op_unit!r}",
            )
    record = {"rule": SOURCE_GRID_LINEAR, "version": 1, "interpolated": False}
    xs, xo = np.asarray(source.x), np.asarray(operand.x)
    if xs.shape == xo.shape and np.array_equal(xs, xo):
        return operand, record
    _check_grid("source", xs)
    _check_grid("operand", xo)
    if operand.dx is not None or operand.dy is not None:
        raise XAlignmentError(
            "uncertainty", "The operand has uncertainties, which cannot be interpolated"
        )
    if xo[0] > xs[0] or xo[-1] < xs[-1]:
        raise XAlignmentError(
            "insufficient_coverage",
            f"The operand grid [{xo[0]}, {xo[-1]}] does not cover "
            f"the source grid [{xs[0]}, {xs[-1]}]",
        )
    aligned = operand.copy(title=f"{operand.title} (interpolated)")
    aligned.set_xydata(xs.copy(), np.interp(xs, xo, np.asarray(operand.y)))
    return aligned, {**record, "interpolated": True}


def check_alignment_record(record: Any) -> None:
    """Check that a recorded alignment names a supported rule.

    Raises:
        XAlignmentError: ``unknown_rule`` if the record is missing, malformed or
         names an unsupported rule or version.
    """
    if not isinstance(record, dict):
        raise XAlignmentError("unknown_rule", "No X alignment rule was recorded")
    key = (record.get("rule"), record.get("version"))
    if key not in X_ALIGNMENT_RULES:
        raise XAlignmentError(
            "unknown_rule", f"Unsupported X alignment rule: {key[0]!r} v{key[1]!r}"
        )
