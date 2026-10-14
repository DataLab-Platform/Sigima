# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
X alignment of multi-input signal operations
============================================

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

Rules ``sigima.signal.x_alignment.source_grid_linear_hold`` and
``sigima.signal.x_alignment.smallest_grid_linear_hold`` (version 1) state the
historical behavior of multi-input signal operations in DataLab:

- ``source_grid_linear_hold`` (two inputs): the result is computed on the grid of
  the ``source`` signal; when the grids differ, the operand is linearly
  interpolated on it.
- ``smallest_grid_linear_hold`` (list of inputs): when the grids are not all
  equal, the target is the grid with the fewest points (the first one on ties);
  every signal on another grid is linearly interpolated on it.
- In both rules, an interpolated signal keeps its first and last values beyond
  its own grid (constant extension at the ends), loses its uncertainty rows, and
  no grid or unit check is made. Grids are equal only when they have the same
  length and the same values.

The inputs are never modified. Applications record the rule (identifier and
version) with each execution; a replay accepts only the rules listed in
:data:`X_ALIGNMENT_RULES`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sigima.objects import SignalObj

__all__ = [
    "SMALLEST_GRID_LINEAR_HOLD",
    "SOURCE_GRID_LINEAR",
    "SOURCE_GRID_LINEAR_HOLD",
    "X_ALIGNMENT_RULES",
    "XAlignmentError",
    "align_operand",
    "align_operand_hold",
    "align_signals_hold",
    "check_alignment_record",
]

#: Identifier of the source-grid rule.
SOURCE_GRID_LINEAR = "sigima.signal.x_alignment.source_grid_linear"
#: Identifier of the historical source-grid rule (constant extension).
SOURCE_GRID_LINEAR_HOLD = "sigima.signal.x_alignment.source_grid_linear_hold"
#: Identifier of the historical smallest-grid rule (constant extension).
SMALLEST_GRID_LINEAR_HOLD = "sigima.signal.x_alignment.smallest_grid_linear_hold"
#: Supported ``(rule, version)`` pairs.
X_ALIGNMENT_RULES = frozenset(
    {
        (SOURCE_GRID_LINEAR, 1),
        (SOURCE_GRID_LINEAR_HOLD, 1),
        (SMALLEST_GRID_LINEAR_HOLD, 1),
    }
)


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


def _same_grid(first: SignalObj, second: SignalObj) -> bool:
    x1, x2 = np.asarray(first.x), np.asarray(second.x)
    return x1.shape == x2.shape and bool(np.array_equal(x1, x2))


def _interpolate_hold(signal: SignalObj, target_x: np.ndarray) -> SignalObj:
    """Return a copy of *signal* linearly interpolated on *target_x*.

    Beyond its own grid, the copy keeps the first and last values of *signal*
    (constant extension); its uncertainty rows are dropped.
    """
    # pylint: disable=import-outside-toplevel
    from sigima.enums import Interpolation1DMethod
    from sigima.tools.signal.interpolation import interpolate

    y_new = interpolate(
        np.asarray(signal.x),
        np.asarray(signal.y),
        target_x,
        Interpolation1DMethod.LINEAR,
        fill_value=None,
    )
    aligned = signal.copy(title=f"{signal.title} (interpolated)", all_metadata=True)
    aligned.set_xydata(np.array(target_x, copy=True), y_new)
    return aligned


def align_operand_hold(
    source: SignalObj, operand: SignalObj
) -> tuple[SignalObj, dict[str, Any]]:
    """Return the operand on the source grid (historical rule), and the record.

    See :data:`SOURCE_GRID_LINEAR_HOLD`: constant extension at the ends,
    uncertainty rows of the interpolated operand dropped, no grid or unit check.

    Args:
        source: Signal whose grid is kept.
        operand: Signal to align; returned unchanged when the grids are equal.

    Returns:
        ``(operand, record)``; *record* holds ``rule``, ``version`` and
        ``interpolated``.
    """
    record = {"rule": SOURCE_GRID_LINEAR_HOLD, "version": 1, "interpolated": False}
    if _same_grid(source, operand):
        return operand, record
    aligned = _interpolate_hold(operand, np.asarray(source.x))
    return aligned, {**record, "interpolated": True}


def align_signals_hold(
    signals: list[SignalObj],
) -> tuple[list[SignalObj], dict[str, Any]]:
    """Return the signals on the smallest grid (historical rule), and the record.

    See :data:`SMALLEST_GRID_LINEAR_HOLD`: the target is the grid with the fewest
    points (the first one on ties); constant extension at the ends, uncertainty
    rows of interpolated signals dropped, no grid or unit check.

    Args:
        signals: Signals, in role order; returned unchanged when all grids are
         equal.

    Returns:
        ``(signals, record)``; *record* holds ``rule``, ``version``,
        ``interpolated`` (indexes of the interpolated signals) and ``target``
        (index of the signal whose grid is kept, or None).
    """
    record: dict[str, Any] = {
        "rule": SMALLEST_GRID_LINEAR_HOLD,
        "version": 1,
        "interpolated": [],
        "target": None,
    }
    if len(signals) < 2 or all(_same_grid(signals[0], s) for s in signals[1:]):
        return list(signals), record
    sizes = [np.asarray(signal.x).size for signal in signals]
    target = sizes.index(min(sizes))
    target_x = np.asarray(signals[target].x)
    aligned, interpolated = [], []
    for index, signal in enumerate(signals):
        if index == target or _same_grid(signals[target], signal):
            aligned.append(signal)
        else:
            aligned.append(_interpolate_hold(signal, target_x))
            interpolated.append(index)
    return aligned, {**record, "interpolated": interpolated, "target": target}


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
