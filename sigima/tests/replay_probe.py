# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Replay probe for the qualification of operation contracts (test helper)
=======================================================================

While the validation tests run, the probe captures the outermost calls of
computation functions (:func:`sigima.proc.decorator.observe_calls`): copies of
the inputs and parameters before the call, and a copy of the result. The generic
replay test then replays each captured call exactly as an application does:

1. :func:`sigima.proc.contracts.describe_call` encodes the effective parameters;
2. the description and the execution context go through JSON;
3. :func:`sigima.proc.contracts.prepare_execution` decodes and prepares the call;
4. the prepared execution runs, :func:`~sigima.proc.contracts.normalize_outputs`
   checks its outputs, and the result is compared with the captured one under the
   comparison rule declared by the qualification (exact for unqualified
   contracts).

Calls whose inputs the contract's X-alignment policy changes are not used as
proof (a validation test calls the function directly, an application prepares
it first).

Cross-environment replays: with ``SIGIMA_REPLAY_DUMP=<file>``, the captured calls
are saved (inputs, parameters and results); with ``SIGIMA_REPLAY_REFERENCE=<file>``
and ``SIGIMA_REPLAY_CROSS_REPORT=<file>``, the calls saved in another environment
(other NumPy or SciPy versions) are replayed here and their results compared.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import pickle
from collections import defaultdict
from typing import Any, Callable

import guidata.dataset as gds
import numpy as np

from sigima.objects import ImageObj, SignalObj
from sigima.objects.scalar.geometry import GeometryResult
from sigima.objects.scalar.table import TableResult
from sigima.proc import contracts
from sigima.proc.contracts.codec import encode_dataset
from sigima.proc.contracts.declarations import EXACT, ComparisonRule
from sigima.proc.contracts.execution import (
    describe_call,
    normalize_outputs,
    prepare_execution,
)

#: Largest number of calls captured per operation.
MAX_SAMPLES = 3
#: Number of array elements (inputs and result) above which a call is captured
#: only if it is the first one of its operation.
LARGE_ELEMENTS = 4_000_000
#: Largest number of array elements of a captured call.
MAX_ELEMENTS = 20_000_000


@dataclasses.dataclass
class Sample:
    """A call captured during a validation test."""

    test: str
    index: int
    args: tuple
    result: Any
    digest: str = ""

    @property
    def key(self) -> str:
        """Key of the sample, stable across environments (content of the inputs)."""
        return f"{self.test}#{self.digest or self.index}"


@dataclasses.dataclass
class Outcome:
    """Outcome of the replay of one captured call."""

    key: str
    status: str
    verdict: str | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    detail: str = ""
    result: Any = None


#: Captured calls by operation identifier.
SAMPLES: dict[str, list[Sample]] = defaultdict(list)
#: Replay outcomes by operation identifier (filled by the generic replay test).
OUTCOMES: dict[str, list[Outcome]] = {}
#: Outcome of the validation tests and of the replay test of the session, by
#: function name (``passed``, ``failed``, ``skipped``), filled by ``conftest.py``.
TEST_OUTCOMES: dict[str, str] = {}
_COUNTS: dict[tuple[str, str], int] = defaultdict(int)


def _size(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return value.size
    if isinstance(value, SignalObj):
        return 0 if value.xydata is None else value.xydata.size
    if isinstance(value, ImageObj):
        return 0 if value.data is None else value.data.size
    if isinstance(value, (list, tuple)):
        return sum(_size(v) for v in value)
    return 0


def _feed(hasher: Any, value: Any) -> None:
    """Feed the scientific content of *value* to *hasher* (inputs digest)."""
    if isinstance(value, SignalObj):
        _feed(hasher, value.xydata)
    elif isinstance(value, ImageObj):
        _feed(hasher, value.data)
    elif isinstance(value, np.ndarray):
        hasher.update(f"{value.dtype.str}{value.shape}".encode())
        hasher.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        for item in value:
            _feed(hasher, item)
    elif isinstance(value, gds.DataSet):
        hasher.update(
            json.dumps(encode_dataset(value), sort_keys=True, allow_nan=False).encode()
        )
    else:
        hasher.update(repr(value).encode())


def _digest(args: tuple) -> str:
    hasher = hashlib.sha256()
    try:
        _feed(hasher, args)
    except Exception:  # pylint: disable=broad-exception-caught
        return ""
    return hasher.hexdigest()[:16]


def make_observer(test: str) -> Callable[[Callable, tuple], Any]:
    """Return the call observer of the validation test *test*."""

    def observe(function: Callable, args: tuple) -> Callable[[Any], None] | None:
        contract = contracts.contract_for_function(function)
        if contract is None:
            return None
        op_id = contract.operation_id
        index = _COUNTS[(op_id, test)]
        _COUNTS[(op_id, test)] += 1
        size = _size(args)
        count = len(SAMPLES[op_id])
        if (
            count >= MAX_SAMPLES
            or size > MAX_ELEMENTS
            or (size > LARGE_ELEMENTS and count > 0)
        ):
            return None
        snapshot = copy.deepcopy(args)
        digest = _digest(snapshot)

        def on_result(result: Any) -> None:
            if _size(result) <= MAX_ELEMENTS:
                SAMPLES[op_id].append(
                    Sample(test, index, snapshot, copy.deepcopy(result), digest)
                )

        return on_result

    return observe


# --- Comparison --------------------------------------------------------------


@dataclasses.dataclass
class _Comparison:
    equal: bool = True
    within: bool = True
    max_abs: float = 0.0
    max_rel: float = 0.0
    undefined: bool = False
    detail: str = ""

    def mismatch(self, detail: str) -> None:
        """Record a structural mismatch (no error measure is defined)."""
        self.equal = self.within = False
        self.undefined = True
        self.detail = self.detail or detail


def _compare_arrays(
    name: str, cand: Any, ref: Any, tol: tuple[float, float] | None, out: _Comparison
) -> None:
    cand, ref = np.asarray(cand), np.asarray(ref)
    if cand.shape != ref.shape or cand.dtype != ref.dtype:
        out.mismatch(f"{name}: {cand.dtype}{cand.shape} != {ref.dtype}{ref.shape}")
        return
    if cand.dtype.kind not in "biufc":
        if not np.array_equal(cand, ref):
            out.mismatch(f"{name}: values differ")
        return
    if cand.dtype.kind in "fc":
        work = complex if cand.dtype.kind == "c" else float
        c, r = cand.astype(work), ref.astype(work)
        parts = (c.real, r.real), (c.imag, r.imag)
        same = np.ones(c.shape, dtype=bool)
        for cp, rp in parts:
            same &= (cp == rp) | (np.isnan(cp) & np.isnan(rp))
    else:
        c, r = cand.astype(float), ref.astype(float)
        same = c == r
    if bool(np.all(same)):
        return
    out.equal = False
    with np.errstate(all="ignore"):
        abs_err = np.abs(c[~same] - r[~same])
        mag = np.abs(r[~same])
        rel_err = abs_err / mag
    if not np.all(np.isfinite(abs_err)):
        out.undefined = True
        out.within = False
        out.detail = out.detail or f"{name}: undefined error"
        return
    out.max_abs = max(out.max_abs, float(abs_err.max()))
    finite_rel = rel_err[np.isfinite(rel_err)]
    if finite_rel.size:
        out.max_rel = max(out.max_rel, float(finite_rel.max()))
    if tol is None or not bool(np.all(abs_err <= tol[0] + tol[1] * mag)):
        out.within = False
        out.detail = out.detail or f"{name}: max abs error {float(abs_err.max()):g}"


def _units(obj: Any) -> list[Any]:
    return [getattr(obj, f"{axis}unit", None) for axis in "xyz"]


def _roi_dict(obj: Any) -> Any:
    return None if obj.roi is None else obj.roi.to_dict()


def _compare_values(
    name: str, cand: Any, ref: Any, tol: tuple[float, float] | None, out: _Comparison
) -> None:
    """Compare two values of the same type, recursively."""
    # pylint: disable=too-many-branches
    if type(cand) is not type(ref):
        out.mismatch(f"{name}: {type(cand).__name__} != {type(ref).__name__}")
    elif cand is None:
        return
    elif isinstance(cand, SignalObj):
        for row in ("x", "y", "dx", "dy"):
            a, b = getattr(cand, row), getattr(ref, row)
            if (a is None) != (b is None):
                out.mismatch(f"{name}.{row}: presence differs")
            elif a is not None:
                _compare_arrays(f"{name}.{row}", a, b, tol, out)
        _compare_values(f"{name}.units", _units(cand), _units(ref), tol, out)
        _compare_values(f"{name}.roi", _roi_dict(cand), _roi_dict(ref), tol, out)
    elif isinstance(cand, ImageObj):
        _compare_arrays(f"{name}.data", cand.data, ref.data, tol, out)
        names = ("x0", "y0", "dx", "dy") if cand.is_uniform_coords else ()
        for attr in names:
            _compare_arrays(
                f"{name}.{attr}", getattr(cand, attr), getattr(ref, attr), tol, out
            )
        if cand.is_uniform_coords != ref.is_uniform_coords:
            out.mismatch(f"{name}: coordinates differ")
        elif not cand.is_uniform_coords:
            _compare_arrays(f"{name}.xcoords", cand.xcoords, ref.xcoords, tol, out)
            _compare_arrays(f"{name}.ycoords", cand.ycoords, ref.ycoords, tol, out)
        _compare_values(f"{name}.units", _units(cand), _units(ref), tol, out)
        _compare_values(f"{name}.roi", _roi_dict(cand), _roi_dict(ref), tol, out)
    elif isinstance(cand, (TableResult, GeometryResult)):
        for field in dataclasses.fields(cand):
            _compare_values(
                f"{name}.{field.name}",
                getattr(cand, field.name),
                getattr(ref, field.name),
                tol,
                out,
            )
    elif isinstance(cand, np.ndarray):
        _compare_arrays(name, cand, ref, tol, out)
    elif isinstance(cand, (list, tuple)):
        if len(cand) != len(ref):
            out.mismatch(f"{name}: lengths differ")
            return
        if all(isinstance(v, (int, float, complex, np.number)) for v in [*cand, *ref]):
            _compare_arrays(name, np.asarray(cand), np.asarray(ref), tol, out)
            return
        for index, (a, b) in enumerate(zip(cand, ref)):
            _compare_values(f"{name}[{index}]", a, b, tol, out)
    elif isinstance(cand, dict):
        if set(cand) != set(ref):
            out.mismatch(f"{name}: keys differ")
            return
        for key in cand:
            _compare_values(f"{name}.{key}", cand[key], ref[key], tol, out)
    elif isinstance(cand, float) and isinstance(ref, float):
        if not (cand == ref or (math.isnan(cand) and math.isnan(ref))):
            _compare_arrays(name, np.asarray(cand), np.asarray(ref), tol, out)
    elif isinstance(cand, np.generic):
        _compare_arrays(name, np.asarray(cand), np.asarray(ref), tol, out)
    elif cand != ref:
        out.mismatch(f"{name}: {cand!r} != {ref!r}")


def compare_results(
    candidate: Any,
    reference: Any,
    rule: ComparisonRule,
    same_environment: bool,
) -> tuple[str, _Comparison]:
    """Compare a replayed result with its reference under *rule*.

    Returns:
        ``(verdict, comparison)``; verdict ``exact``, ``within_tolerance`` or
        ``different``.
    """
    tol = None
    if not rule.exact and not (same_environment and rule.same_environment == "exact"):
        tolerance = {t.role: t for t in rule.tolerances}.get("result")
        if tolerance is not None:
            tol = (tolerance.atol, tolerance.rtol)
    out = _Comparison()
    _compare_values("result", candidate, reference, tol, out)
    if out.equal:
        return "exact", out
    if tol is not None and out.within:
        return "within_tolerance", out
    return "different", out


# --- Replay ------------------------------------------------------------------


def _split(contract: contracts.OperationContract, args: tuple) -> tuple[list, Any]:
    """Return the input objects (flattened) and the parameter argument."""
    count = len(contract.inputs)
    inputs: list[Any] = []
    for role, arg in zip(contract.inputs, args[:count]):
        if role.variadic:
            inputs.extend(arg)
        else:
            inputs.append(arg)
    params = list(args[count:])
    params += [None] * (len(contract.parameters) - len(params))
    if not contract.parameters:
        return inputs, None
    return inputs, params[0] if len(params) == 1 else tuple(params)


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def replay(contract: contracts.OperationContract, sample: Sample) -> Outcome:
    """Replay one captured call as an application does and compare its result."""
    inputs, param = _split(contract, copy.deepcopy(sample.args))
    description = describe_call(contract.function, inputs, param, contract=contract)
    if description["parameters"] is None or description["limits"]:
        return Outcome(sample.key, "not_described", detail=str(description["limits"]))
    context = prepare_execution(contract, inputs, param=param).context
    record = context.get("x_alignment") or {}
    if record.get("interpolated"):
        return Outcome(sample.key, "aligned", detail="inputs aligned by the policy")
    prepared = prepare_execution(
        contract,
        inputs,
        values=_json(description["parameters"]),
        context=_json(context),
    )
    candidate = prepared.run()
    normalize_outputs(contract, candidate)
    rule = contract.comparison or EXACT
    verdict, comparison = compare_results(candidate, sample.result, rule, True)
    return Outcome(
        sample.key,
        "replayed",
        verdict,
        comparison.max_abs if not comparison.undefined else None,
        comparison.max_rel if not comparison.undefined else None,
        comparison.detail,
        candidate,
    )


def safe_replay(contract: contracts.OperationContract, sample: Sample) -> Outcome:
    """Replay one captured call; an error is reported as an outcome."""
    try:
        return replay(contract, sample)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return Outcome(sample.key, "error", detail=f"{type(exc).__name__}: {exc}")


def dump_samples(path: str) -> None:
    """Save the captured calls (inputs, parameters and results), for a cross
    replay in another environment (:func:`cross_replay`)."""
    data = {
        op: [(sample.key, sample.args, sample.result) for sample in samples]
        for op, samples in SAMPLES.items()
    }
    with open(path, "wb") as file:
        pickle.dump(data, file)


def cross_replay(path: str) -> dict[str, dict[str, Any]]:
    """Replay here the calls captured in another environment, and compare.

    Each call saved by :func:`dump_samples` is replayed with the local library,
    as an application replays a recorded activity, and its result compared with
    the result of the other environment: exactly (observed errors), and under
    the declared comparison rule (*verdict*, the worst of the samples).

    Returns:
        ``{operation: {"samples", "exact", "verdict", "max_abs_error",
        "max_rel_error", "detail"}}``; errors are None when undefined (e.g. NaN
        moved).
    """
    with open(path, "rb") as file:
        reference = pickle.load(file)  # noqa: S301 - file written by this module
    registry = {
        c.operation_id: c for c in contracts.build_contract_registry().contracts
    }
    order = ("exact", "within_tolerance", "different")
    report: dict[str, dict[str, Any]] = {}
    for op, entries in sorted(reference.items()):
        contract = registry.get(op)
        if contract is None:
            continue
        rule = contract.comparison or EXACT
        entry = {
            "samples": 0,
            "exact": True,
            "verdict": "exact",
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "detail": "",
        }
        for key, args, result in entries:
            sample = Sample(key, 0, args, result)
            outcome = safe_replay(contract, sample)
            if outcome.status != "replayed":
                continue
            entry["samples"] += 1
            verdict, comparison = compare_results(outcome.result, result, EXACT, False)
            if verdict == "exact":
                continue
            ruled, _comparison = compare_results(outcome.result, result, rule, False)
            if order.index(ruled) > order.index(entry["verdict"]):
                entry["verdict"] = ruled
            entry["exact"] = False
            entry["detail"] = entry["detail"] or comparison.detail
            if comparison.undefined:
                entry["max_abs_error"] = entry["max_rel_error"] = None
            elif entry["max_abs_error"] is not None:
                entry["max_abs_error"] = max(entry["max_abs_error"], comparison.max_abs)
                entry["max_rel_error"] = max(entry["max_rel_error"], comparison.max_rel)
        if entry["samples"]:
            report[op] = entry
    return report


def environment_paths() -> tuple[str | None, str | None, str | None]:
    """Return the dump, reference and cross-report paths from the environment."""
    return (
        os.environ.get("SIGIMA_REPLAY_DUMP"),
        os.environ.get("SIGIMA_REPLAY_REFERENCE"),
        os.environ.get("SIGIMA_REPLAY_CROSS_REPORT"),
    )
