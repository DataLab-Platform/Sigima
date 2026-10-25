.. _guidelines:

Coding guidelines
=================

Generic coding guidelines
-------------------------

We follow the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ coding style.

In particular, we are especially strict about the following guidelines:

- Limit all lines to a maximum of 88 characters.
- Respect the naming conventions (classes, functions, variables, etc.).
- Use specific exceptions instead of the generic :class:`Exception`.

To enforce these guidelines, the following tools are mandatory:

- `ruff <https://pypi.org/project/ruff/>`_ for code formatting and static code analysis.
- `pylint <https://pypi.org/project/pylint/>`_ for static code analysis.

ruff
^^^^

If you are using `Visual Studio Code <https://code.visualstudio.com/>`_,
the project settings will automatically format your code with `ruff` on save.
You may also run the Ruff tasks on the project.

To format and check the code with `ruff`, run the following commands from the
repository root::

  python scripts/run_with_env.py python -m ruff format
  python scripts/run_with_env.py python -m ruff check

pylint
^^^^^^

To run `pylint`, run the following command from the repository root::

  python scripts/run_with_env.py python -m pylint sigima

If you are using `Visual Studio Code <https://code.visualstudio.com/>`_
on Windows, you may run the "Pylint" task on the project. The enabled checks
and accepted exclusions are defined in the repository's ``.pylintrc`` file.

.. note::

  The Ruff and Pylint checks must both exit successfully before a pull request
  may be merged. Pylint's numeric rating is informative only: the merge gate is
  based on the absence of diagnostics from the enabled checks.

Specific coding guidelines
--------------------------

In addition to the generic coding guidelines, we have the following specific
guidelines:

- Write docstrings for all classes, methods and functions. The docstrings
  should follow the `Google style <https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings>`_.

- Add typing annotations for all functions and methods. The annotations should
  use the future syntax (``from __future__ import annotations``)

- Try to keep the code as simple as possible. If you have to write a complex
  piece of code, try to split it into several functions or classes.

- Add as many comments as possible. The code should be self-explanatory, but
  it is always useful to add some comments to explain the general idea of the
  code, or to explain some tricky parts.

- Do not use ``from module import *`` statements, even in the ``__init__``
  module of a package.

- Avoid using mixins (multiple inheritance) when possible. It is often
  possible to use composition instead of inheritance.

- Avoid using ``__getattr__`` and ``__setattr__`` methods. They are often used
  to implement lazy initialization, but this can be done in a more explicit
  way.

Operation contracts
-------------------

Every public computation function declares an operation contract with
``@computation_function(operation_id=..., contract_version=...)`` (see
:mod:`sigima.proc.contracts`). Applications record these identifiers in
workspace provenance and use them to replay a computation later, possibly with
another Sigima version. Everything about an operation is declared next to its
function; :mod:`sigima.proc.contracts` holds no data naming an operation.
Follow these rules when you add or change a computation function:

- Write the identifier in clear next to the function: ``sigima.signal.<name>``
  or ``sigima.image.<name>``, ``contract_version=1`` at the first declaration.
  Never derive an identifier from a Python name. Keep the identifier when you
  rename or move the function.

- If the identifier itself must change, keep the previous one in ``aliases``.

- Increase ``contract_version`` whenever a change of the computation function
  changes the result it returns: new, removed or renamed parameters, a
  different default behavior, different outputs or a different scientific
  result for the same inputs, even for unusual inputs only (e.g. mixed real and
  complex, or ``float32`` and ``float64``, data), and inputs that were accepted
  and are now refused. Do it in the change itself, whatever the releases of
  DataLab or of any other application using Sigima: Sigima is a reference of
  its own, and a replay of an activity recorded with the earlier version is
  then refused explicitly instead of giving another result.

- Do not increase it when the result cannot change: only inputs that were
  refused (with an error) are now accepted, the change has no effect on the
  result (refactoring, performance, messages), or the change is entirely
  covered by a versioned rule of the execution context (see below), whose
  version is increased instead.

- Checklist for any change of an existing computation function: decide whether
  the result can change; if so, increase ``contract_version`` and the version
  of its ``ReplayQualification`` (after proving it, see below), update the
  reference cases deliberately, run ``pytest --qualification-report``, list the
  increased contracts in the release notes, and update the provenance tests of
  the applications that check a contract version (e.g. DataLab).

- Historical exception: the contracts were published with Sigima 1.3.0 and
  kept version 1 in Sigima 1.4.0 although several results changed, because they
  had just been published and were not used yet. This one-off decision does not
  apply to later changes.

- Rules that apply to every operation are recorded in the execution context
  rather than in each contract: the selection of image pixels by regions of
  interest (:data:`sigima.proc.contracts.IMAGE_ROI_CONVENTION`, recorded when an
  input image carries regions of interest) and the X-alignment policies (see
  below). Increase the version of such a rule, not the contract versions, when
  it changes: a replay refuses the calls recorded with another version.

- The structure of the contract (input roles, parameter arguments, outputs) is
  inferred from the annotations. Annotate every argument and the return value
  with names imported at runtime (not only under ``TYPE_CHECKING``). When roles
  cannot be inferred (more than two input objects), declare them with
  ``roles=(...)``.

- Declare the :data:`sigima.config.options` read by the computation itself with
  ``reads_options=(...)``: applications record their values and restore them
  for a replay. Random generators take their seed as a parameter.

- Declare the metadata keys of input objects read by the computation itself
  with ``reads_metadata={role: (key, ...)}`` (e.g. the fit parameters read by
  ``evaluate_fit``): state fingerprints do not cover metadata, so applications
  record their values and refuse a replay when they changed.

- Never change a reference case to make a test pass. Update the reference
  cases deliberately, and explain the scientific reason, when a numerical
  correction changes the result.

- Declaring a contract does not qualify it for replay. A qualification is
  declared next to the function with
  ``qualification=ReplayQualification(contract_version=..., comparison=...)``
  (see :mod:`sigima.proc.contracts.declarations`). It names the contract
  version that was actually tested: increasing ``contract_version`` does not qualify the new
  version. It requires the validation test of the function (marked
  ``validation``) and the generic replay test
  (``sigima/tests/common/replay_qualification_test.py``), which replays the
  calls captured during the validation tests as an application replays a
  recorded activity. Run ``pytest --qualification-report=report.json``: the
  run fails if a declared qualification is not proven.

- The comparison rule of a qualification is exact by default (``EXACT``).
  Declare a tolerance (``tolerance(...)``) only when the nature of the
  computation justifies it (transforms, interpolations, iterative fits...),
  with that justification, and state whether results must still be identical
  within an identical environment. Never widen a tolerance to hide an error:
  changing a rule changes the qualification.

- Set a tolerance from differences measured between environments (NumPy,
  SciPy, platforms), not from guesses: run the test suite in a first
  environment with ``SIGIMA_REPLAY_DUMP=<calls file>``, then in a second one
  with ``SIGIMA_REPLAY_REFERENCE=<calls file>`` and
  ``SIGIMA_REPLAY_CROSS_REPORT=<report.json>``. The second run replays the
  calls captured by the first one and reports the differences by operation. A
  function whose results differ beyond any justified tolerance stays
  unqualified. To include Pyodide (WebAssembly, the runtime of DataLab-Web),
  capture the calls into a directory with NumPy 1.26
  (``SIGIMA_REPLAY_DUMP=<directory>/``), replay them with
  ``node scripts/pyodide_replay/run.mjs`` (see its header) and compare the
  results with ``sigima.tests.replay_probe.compare_replays``.

- Some two-input contracts align the operand on the source grid before the
  call: declare the policy with ``x_alignment=...`` (see
  :mod:`sigima.proc.contracts.policies` and
  :mod:`sigima.proc.contracts.alignment`). Applications
  record the policy and its version, and a replay refuses unknown policies.
  Never change a policy in place: add a new version and keep the previous one
  supported.
