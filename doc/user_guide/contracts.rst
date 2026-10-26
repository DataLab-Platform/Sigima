.. _contracts:

Operation contracts and versions
================================

Every public computation function of Sigima declares an **operation contract**:
a stable identifier (e.g. ``sigima.signal.normalize``) and a version, the roles
of its inputs, its parameters and outputs, and the rules its result depends on
besides them. Applications such as DataLab record the identifier and the
version of each computation with the processed data (provenance), so that the
computation can be replayed later, possibly with another Sigima version, and
compared with the recorded result.

.. code-block:: python

    from sigima.proc import contracts
    import sigima.proc.signal as sips

    contract = contracts.contract_for_function(sips.normalize)
    print(contract.operation_id, contract.contract_version)
    # sigima.signal.normalize 1

What a version guarantees
-------------------------

For a given identifier and version, the same inputs, parameters and execution
context give the same result, compared under the rule declared by the replay
qualification of the contract: exact by default, or within a justified
tolerance for some computations (transforms, fits, reductions...) whose results
differ slightly between numerical libraries or platforms. Contracts that are not
qualified keep their identity but are never replayed.

When a version is increased
---------------------------

Sigima increases the version of a contract whenever a change of the computation
function changes the result it returns: new, removed or renamed parameters, a
different default behavior, different outputs or a different result for the
same inputs (even for unusual inputs only), or inputs that were accepted and
are now refused. This is done in the Sigima release that brings the change,
whatever the releases of the applications using Sigima: Sigima is a reference
of its own.

The version is not increased when the result cannot change:

- inputs that were refused (with an error) are now accepted;
- the change has no effect on the result (refactoring, performance, messages);
- the change is entirely covered by a versioned rule of the execution context
  (see below), whose version is increased instead.

The release notes list the contracts whose version was increased.

Rules of the execution context
------------------------------

Some rules apply to every operation, or to a family of operations, and are
recorded with each call rather than in each contract, with their own identifier
and version:

- the selection of image pixels by regions of interest
  (``sigima.image.roi.pixel_centers``,
  :data:`sigima.proc.contracts.IMAGE_ROI_CONVENTION`), recorded when an input
  image carries regions of interest or when the operation takes image regions of
  interest as parameters (ROI extraction, erasing, offset correction);
- the alignment of signals on a common X grid, for operations combining several
  signals (``sigima.signal.x_alignment.*`` policies);
- the global options read by a computation and the metadata of its inputs it
  reads.

When such a rule changes, its version is increased and the contracts keep
theirs.

Consequences for replay
-----------------------

An application replays a recorded call only with a local contract of the same
identifier and version, qualified for that version, and with the recorded
execution context. A call recorded with another version of the contract, or
with a rule of the context that Sigima no longer supports, is refused
explicitly (unsupported contract or unsupported context): it is never replayed
to give a silently different result. To obtain a record with the current
version, compute the result again.

.. note::

    The contracts were published with Sigima 1.3.0 and kept version 1 in
    Sigima 1.4.0 although several results changed, because they had just been
    published and were not used yet. This one-off exception does not apply to
    later releases.

See also the rules followed by contributors
(:doc:`/contributing/guidelines`, section "Operation contracts") and the API of
:mod:`sigima.proc.contracts`.
