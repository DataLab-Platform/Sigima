# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Generic replay test of the operation contracts (replay qualification).

The calls made by the validation tests are captured (see
:mod:`sigima.tests.replay_probe`) and replayed here, as an application replays a
recorded activity: parameters encoded by the contract and decoded through JSON,
execution context recorded and applied again, outputs checked by role, result
compared with the captured one under the declared comparison rule.

This test runs after the validation tests (see ``conftest.py``). It is not a
validation test: it proves the transport and replay of each contract, while the
validation test of each function proves the computation. A declared
qualification needs both (``--qualification-report``).
"""

from __future__ import annotations

import pytest

from sigima.proc import contracts
from sigima.proc.validation import generate_valid_test_names_for_function
from sigima.tests import replay_probe

REGISTRY = contracts.build_contract_registry()
CONTRACTS = {contract.operation_id: contract for contract in REGISTRY.contracts}


@pytest.mark.parametrize("operation_id", sorted(CONTRACTS))
def test_replay_qualification(operation_id: str) -> None:
    """Replay the calls captured for one contract and compare the results."""
    contract = CONTRACTS[operation_id]
    samples = replay_probe.SAMPLES.get(operation_id, [])
    outcomes = [replay_probe.safe_replay(contract, sample) for sample in samples]
    replay_probe.OUTCOMES[operation_id] = outcomes
    if contract.qualification is None:
        pytest.skip("not qualified: replay outcomes are reported only")
    family = contract.operation_id.split(".")[1]
    names = generate_valid_test_names_for_function(
        f"sigima.proc.{family}", contract.function.__name__
    )
    ran = [
        name
        for name in names
        if replay_probe.TEST_OUTCOMES.get(name) in ("passed", "failed")
    ]
    if not ran:
        pytest.skip("the validation test of the function did not run in this session")
    assert samples, "no call captured from the validation tests"
    proofs = [o for o in outcomes if o.status == "replayed"]
    assert proofs, f"no replayable captured call: {[o.status for o in outcomes]}"
    accepted = {"exact", "within_tolerance"}
    failures = [
        f"{o.key}: {o.verdict} ({o.detail})"
        for o in proofs
        if o.verdict not in accepted
    ]
    assert not failures, failures
