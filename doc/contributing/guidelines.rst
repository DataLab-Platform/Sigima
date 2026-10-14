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

- Increase ``contract_version`` for any change of meaning: new, removed or
  renamed parameters, a different default behavior, different outputs or a
  different scientific result for the same inputs.

- The structure of the contract (input roles, parameter arguments, outputs) is
  inferred from the annotations. Annotate every argument and the return value
  with names imported at runtime (not only under ``TYPE_CHECKING``). When roles
  cannot be inferred (more than two input objects), declare them with
  ``roles=(...)``.

- Declare the :data:`sigima.config.options` read by the computation itself with
  ``reads_options=(...)``: applications record their values and restore them
  for a replay. Random generators take their seed as a parameter.

- Never change a reference case to make a test pass. Update the reference
  cases deliberately, and explain the scientific reason, when a numerical
  correction changes the result.

- Declaring a contract does not qualify it for replay. A qualification is
  declared next to the function with
  ``qualification=ReplayQualification(contract_version=..., comparison=...)``
  (see :mod:`sigima.proc.contract_types`). It names the contract version that
  was actually tested: increasing ``contract_version`` does not qualify the new
  version. It requires the validation test of the function and the replay
  test of its contract shape.

- The comparison rule of a qualification is exact by default (``EXACT``).
  Declare a tolerance (``tolerance(...)``) only when the nature of the
  computation justifies it (transforms, interpolations, iterative fits...),
  with that justification, and state whether results must still be identical
  within an identical environment. Never widen a tolerance to hide an error:
  changing a rule changes the qualification.

- Some two-input contracts align the operand on the source grid before the
  call: declare the policy with ``x_alignment=...`` (see
  :mod:`sigima.proc.policies` and :mod:`sigima.proc.alignment`). Applications
  record the policy and its version, and a replay refuses unknown policies.
  Never change a policy in place: add a new version and keep the previous one
  supported.
