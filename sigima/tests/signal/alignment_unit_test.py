# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the X alignment of two-input signal operations."""

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.proc.signal
from sigima.proc import contracts
from sigima.proc.contracts.alignment import (
    SOURCE_GRID_LINEAR,
    XAlignmentError,
    align_operand,
    check_alignment_record,
)
from sigima.proc.contracts.execution import prepare_execution


def _signal(x, y, title: str = "s", units=("s", "V")) -> sigima.objects.SignalObj:
    return sigima.objects.create_signal(
        title, np.array(x, dtype=float), np.array(y, dtype=float), units=units
    )


def _difference(source, operand):
    """Run the difference contract as an application does: prepare, then call."""
    contract = contracts.contract_for_function(sigima.proc.signal.difference)
    prepared = prepare_execution(contract, [source, operand], param=None)
    return prepared.run(), prepared.context


def test_equal_grids_are_not_interpolated() -> None:
    """Exactly equal grids keep the operand object itself."""
    source = _signal([0, 1, 2, 3], [1, 4, 9, 16])
    operand = _signal([0, 1, 2, 3], [1, 1, 2, 3])
    aligned, record = align_operand(source, operand)
    assert aligned is operand
    assert record == {"rule": SOURCE_GRID_LINEAR, "version": 1, "interpolated": False}
    result, _context = _difference(source, operand)
    assert np.array_equal(result.y, [0.0, 3.0, 7.0, 13.0])


def test_same_size_and_ends_but_different_grid() -> None:
    """[0, 1, 2] and [0, 0.5, 2] are different grids: no index-by-index result."""
    source = _signal([0, 1, 2], [0, 1, 2])
    operand = _signal([0, 0.5, 2], [0, 0.5, 2])
    result, context = _difference(source, operand)
    assert context["x_alignment"]["interpolated"] is True
    assert np.array_equal(result.x, [0.0, 1.0, 2.0])
    assert np.array_equal(result.y, [0.0, 0.0, 0.0])


def test_role_inversion_on_an_asymmetric_case() -> None:
    """Both orders are valid and give different results on different grids."""
    a = _signal([0, 1, 2, 3], [0, 1, 4, 9], "a")
    b = _signal([0, 1.5, 3], [0, 3, 6], "b")
    a_minus_b, _ = _difference(a, b)
    b_minus_a, _ = _difference(b, a)
    assert np.array_equal(a_minus_b.x, [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(a_minus_b.y, [0.0, -1.0, 0.0, 3.0])
    assert np.array_equal(b_minus_a.x, [0.0, 1.5, 3.0])
    assert np.array_equal(b_minus_a.y, [0.0, 0.5, -3.0])


def test_inputs_are_not_modified() -> None:
    """Alignment never changes the original signals."""
    source = _signal([0, 1, 2, 3], [0, 1, 4, 9])
    operand = _signal([0, 1.5, 3], [0, 3, 6])
    before = [(s.x.copy(), s.y.copy(), s.title) for s in (source, operand)]
    _difference(source, operand)
    for sig, (x, y, title) in zip((source, operand), before):
        assert np.array_equal(sig.x, x) and np.array_equal(sig.y, y)
        assert sig.title == title


@pytest.mark.parametrize(
    "source, operand, code",
    [
        # Operand starts after the source: no extrapolation.
        (([0, 1, 2], [0, 1, 2]), ([0.5, 1, 2], [0, 1, 2]), "insufficient_coverage"),
        # Operand stops before the source: no constant extension.
        (([0, 1, 2], [0, 1, 2]), ([0, 1, 1.5], [0, 1, 2]), "insufficient_coverage"),
        (([0, 2, 1], [0, 1, 2]), ([0, 1, 2], [0, 1, 2]), "invalid_grid"),
        (([0, 1, 2], [0, 1, 2]), ([0, 1, 1, 2], [0, 1, 1, 2]), "invalid_grid"),
        (([0, 1, np.nan], [0, 1, 2]), ([0, 1, 2], [0, 1, 2]), "invalid_grid"),
        (([0, 1, 2], [0, 1, 2]), ([0, np.inf, 3], [0, 1, 2]), "invalid_grid"),
    ],
)
def test_alignment_refusals(source, operand, code) -> None:
    """Uncovered, unordered and non-finite grids are refused."""
    with pytest.raises(XAlignmentError) as info:
        align_operand(_signal(*source), _signal(*operand))
    assert info.value.code == code


@pytest.mark.parametrize(
    "source_units, operand_units, accepted",
    [
        (("s", "V"), ("s", "V"), True),
        (("", ""), ("", ""), True),
        (("s", ""), ("s", ""), True),
        (("", "V"), ("", "V"), True),
        (("", "V"), ("s", "V"), False),
        (("s", "V"), ("", "V"), False),
        (("s", ""), ("s", "V"), False),
        (("s", "V"), ("s", ""), False),
        (("s", "V"), ("ms", "V"), False),
        (("s", "V"), ("s", "mV"), False),
    ],
)
def test_units_are_compared_exactly(source_units, operand_units, accepted) -> None:
    """X and Y units must be identical; an empty unit only matches an empty one."""
    source = _signal([0, 1, 2], [0, 1, 2], units=source_units)
    for operand_x in ([0, 1, 2], [0, 0.5, 2]):
        operand = _signal(operand_x, [0, 1, 2], units=operand_units)
        if accepted:
            align_operand(source, operand)
        else:
            with pytest.raises(XAlignmentError) as info:
                align_operand(source, operand)
            assert info.value.code == "incompatible_units"


def test_uncertainties_are_never_interpolated() -> None:
    """An operand with uncertainty rows is refused when it must be interpolated."""
    source = _signal([0, 1, 2], [0, 1, 2])
    operand = _signal([0, 0.5, 2], [0, 1, 2])
    operand.dy = np.full(3, 0.1)
    with pytest.raises(XAlignmentError) as info:
        align_operand(source, operand)
    assert info.value.code == "uncertainty"


def test_recorded_rule_is_checked_on_replay() -> None:
    """A replay accepts only a recorded, supported rule and version."""
    contract = contracts.get_operation_contract("sigima.signal.difference", 1)
    source = _signal([0, 1, 2], [0, 1, 2])
    operand = _signal([0, 0.5, 2], [0, 0.5, 2])
    context = prepare_execution(contract, [source, operand], param=None).context
    replay = prepare_execution(contract, [source, operand], values={}, context=context)
    assert np.array_equal(replay.args[1].y, [0.0, 1.0, 2.0])
    for recorded in (
        {},
        {"x_alignment": None},
        {"x_alignment": {"rule": SOURCE_GRID_LINEAR, "version": 2}},
        {"x_alignment": {"rule": "other.rule", "version": 1}},
    ):
        with pytest.raises(XAlignmentError) as info:
            prepare_execution(contract, [source, operand], values={}, context=recorded)
        assert info.value.code == "unknown_rule"
    check_alignment_record(context["x_alignment"])
