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
from collections.abc import Iterator, Sequence
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


def _canonical(value: Any) -> Any:  # pylint: disable=R0911
    """Return *value* as JSON-ready data independent of library versions."""
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        data = np.ascontiguousarray(value)
        digest = hashlib.sha256(data.tobytes()).hexdigest()
        return {"$ndarray": [data.dtype.str, list(data.shape), digest]}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _feed_json(hasher: Any, value: Any) -> None:
    hasher.update(json.dumps(_canonical(value), sort_keys=True).encode())


def _feed(hasher: Any, value: Any, metadata_keys: Sequence[str] = ()) -> None:
    """Feed the scientific content of *value* to *hasher* (inputs digest).

    Signals and images contribute their data, coordinates, region of interest
    and the metadata *metadata_keys* read by the computation, so that two calls
    on the same data with another ROI or other read metadata get other keys.
    """
    if isinstance(value, (SignalObj, ImageObj)):
        if isinstance(value, SignalObj):
            _feed(hasher, value.xydata)
        else:
            _feed(hasher, value.data)
            if value.is_uniform_coords:
                coords = [value.x0, value.y0, value.dx, value.dy]
            else:
                coords = [value.xcoords, value.ycoords]
            _feed_json(hasher, coords)
        roi = getattr(value, "roi", None)
        _feed_json(hasher, None if roi is None else roi.to_dict())
        _feed_json(hasher, {k: value.metadata.get(k) for k in metadata_keys})
    elif isinstance(value, np.ndarray):
        hasher.update(f"{value.dtype.str}{value.shape}".encode())
        hasher.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        for item in value:
            _feed(hasher, item, metadata_keys)
    elif isinstance(value, gds.DataSet):
        hasher.update(
            json.dumps(encode_dataset(value), sort_keys=True, allow_nan=False).encode()
        )
    else:
        hasher.update(repr(value).encode())


def input_digest(args: tuple, metadata_keys: Sequence[str] = ()) -> str:
    """Return the digest of the inputs of a call (key of a captured call)."""
    hasher = hashlib.sha256()
    try:
        _feed(hasher, args, metadata_keys)
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
        keys = sorted({key for _role, keys in contract.reads_metadata for key in keys})
        digest = input_digest(snapshot, keys)

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


def _is_directory(path: str) -> bool:
    return path.endswith(("/", "\\")) or os.path.isdir(path)


def _file_name(operation_id: str) -> str:
    return f"{operation_id}.pkl"


def dump_samples(path: str) -> None:
    """Save the captured calls (inputs, parameters and results), for a cross
    replay in another environment (:func:`cross_replay`, :func:`replay_dump`).

    Args:
        path: A file, or a directory (existing, or ending with a separator)
         receiving one file per operation, so that an environment with little
         memory (Pyodide) loads one operation at a time.
    """
    data = {
        op: [(sample.key, sample.args, sample.result) for sample in samples]
        for op, samples in SAMPLES.items()
    }
    if not _is_directory(path):
        with open(path, "wb") as file:
            pickle.dump(data, file)
        return
    os.makedirs(path, exist_ok=True)
    for op, entries in data.items():
        with open(os.path.join(path, _file_name(op)), "wb") as file:
            pickle.dump(entries, file)


def load_dump(path: str) -> Iterator[tuple[str, list[tuple[str, Any, Any]]]]:
    """Yield ``(operation, [(key, args, result)])`` from :func:`dump_samples`.

    A directory is read one operation at a time.
    """
    if not _is_directory(path):
        with open(path, "rb") as file:
            data = pickle.load(file)  # noqa: S301 - file written by this module
        yield from sorted(data.items())
        return
    for name in sorted(os.listdir(path)):
        if name.endswith(".pkl"):
            with open(os.path.join(path, name), "rb") as file:
                entries = pickle.load(file)  # noqa: S301 - written by this module
            yield name[: -len(".pkl")], entries


_VERDICT_ORDER = ("exact", "within_tolerance", "different")


def _new_entry() -> dict[str, Any]:
    return {
        "samples": 0,
        "exact": True,
        "verdict": "exact",
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "detail": "",
    }


def _add_comparison(
    entry: dict[str, Any], candidate: Any, reference: Any, rule: ComparisonRule
) -> None:
    """Add one compared sample to a report entry (environments differ)."""
    entry["samples"] += 1
    verdict, comparison = compare_results(candidate, reference, EXACT, False)
    if verdict == "exact":
        return
    ruled, _comparison = compare_results(candidate, reference, rule, False)
    if _VERDICT_ORDER.index(ruled) > _VERDICT_ORDER.index(entry["verdict"]):
        entry["verdict"] = ruled
    entry["exact"] = False
    entry["detail"] = entry["detail"] or comparison.detail
    if comparison.undefined:
        entry["max_abs_error"] = entry["max_rel_error"] = None
    elif entry["max_abs_error"] is not None:
        entry["max_abs_error"] = max(entry["max_abs_error"], comparison.max_abs)
        entry["max_rel_error"] = max(entry["max_rel_error"], comparison.max_rel)


def _registry() -> dict[str, contracts.OperationContract]:
    return {c.operation_id: c for c in contracts.build_contract_registry().contracts}


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
    registry = _registry()
    report: dict[str, dict[str, Any]] = {}
    for op, entries in load_dump(path):
        contract = registry.get(op)
        if contract is None:
            continue
        rule = contract.comparison or EXACT
        entry = _new_entry()
        for key, args, result in entries:
            outcome = safe_replay(contract, Sample(key, 0, args, result))
            if outcome.status == "replayed":
                _add_comparison(entry, outcome.result, result, rule)
        if entry["samples"]:
            report[op] = entry
    return report


def replay_dump(
    path: str, out_dir: str, operations: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Replay captured calls here and save the results, one file per operation.

    Used where the test suite does not run (e.g. Pyodide): each call of
    :func:`dump_samples` is replayed as an application replays a recorded
    activity, and ``(key, status, result, detail)`` is saved for
    :func:`compare_replays`, run in another environment.

    Args:
        path: Captured calls (file or directory).
        out_dir: Directory receiving one result file per operation.
        operations: Operations to replay (default: all).

    Returns:
        Report against the results captured with the calls (see
        :func:`cross_replay`), plus ``"statuses"`` (count of replay statuses).
    """
    registry = _registry()
    os.makedirs(out_dir, exist_ok=True)
    report: dict[str, dict[str, Any]] = {}
    for op, entries in load_dump(path):
        contract = registry.get(op)
        if contract is None or (operations is not None and op not in operations):
            continue
        rule = contract.comparison or EXACT
        entry = _new_entry()
        statuses: dict[str, int] = defaultdict(int)
        saved = []
        for key, args, result in entries:
            outcome = safe_replay(contract, Sample(key, 0, args, result))
            statuses[outcome.status] += 1
            saved.append((key, outcome.status, outcome.result, outcome.detail))
            if outcome.status == "replayed":
                _add_comparison(entry, outcome.result, result, rule)
        entry["statuses"] = dict(statuses)
        with open(os.path.join(out_dir, _file_name(op)), "wb") as file:
            pickle.dump(saved, file)
        report[op] = entry
    return report


def compare_replays(results_dir: str, reference: str) -> dict[str, dict[str, Any]]:
    """Compare results saved by :func:`replay_dump` with captured results.

    Args:
        results_dir: Directory written by :func:`replay_dump`.
        reference: Captured calls of another environment (file or directory),
         whose results are the references; samples are matched by key (content
         of the inputs).

    Returns:
        Report as :func:`cross_replay`, plus ``"unmatched"``: samples replayed
        without a reference with the same key (e.g. inputs computed upstream,
        which differ between environments); ``"exact"`` and ``"verdict"`` are
        None when no sample was compared.
    """
    registry = _registry()
    report: dict[str, dict[str, Any]] = {}
    for op, entries in load_dump(reference):
        contract = registry.get(op)
        result_file = os.path.join(results_dir, _file_name(op))
        if contract is None or not os.path.exists(result_file):
            continue
        with open(result_file, "rb") as file:
            replayed = pickle.load(file)  # noqa: S301 - written by this module
        references: dict[str, list[Any]] = defaultdict(list)
        for key, _args, result in entries:
            references[key].append(result)
        rule = contract.comparison or EXACT
        entry = _new_entry()
        unmatched = 0
        for key, status, result, _detail in replayed:
            # Calls with the same inputs are matched in capture order.
            candidates = references.get(key)
            if not candidates:
                unmatched += status == "replayed"
                continue
            reference_result = candidates.pop(0)
            if status == "replayed":
                _add_comparison(entry, result, reference_result, rule)
        entry["unmatched"] = unmatched
        if not entry["samples"]:
            entry["exact"] = entry["verdict"] = None
        if entry["samples"] or unmatched:
            report[op] = entry
    return report


def environment_paths() -> tuple[str | None, str | None, str | None]:
    """Return the dump, reference and cross-report paths from the environment."""
    return (
        os.environ.get("SIGIMA_REPLAY_DUMP"),
        os.environ.get("SIGIMA_REPLAY_REFERENCE"),
        os.environ.get("SIGIMA_REPLAY_CROSS_REPORT"),
    )
