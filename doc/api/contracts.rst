.. automodule:: sigima.proc.contracts
    :no-members:

.. automodule:: sigima.proc.contracts.declarations
    :no-members:

.. currentmodule:: sigima.proc.contracts

.. autoclass:: Role
.. autoclass:: ParameterSpec
.. autoclass:: OutputTolerance
.. autoclass:: ComparisonRule
.. autoclass:: ReplayQualification
.. autofunction:: tolerance
.. py:data:: EXACT
    :type: ComparisonRule

    Exact comparison rule (the default).

.. py:data:: OBJECT_KINDS
    :type: tuple[str, ...]

    Kinds of scientific objects bound to roles.

.. py:data:: ARTIFACT_KINDS
    :type: tuple[str, ...]

    Kinds of analysis results recorded as artifacts.

.. autoexception:: ContractError
.. autoexception:: ContractDeclarationError
.. autoexception:: UnknownOperationError
.. autoexception:: IncompatibleContractError
.. autoexception:: InvalidParametersError
.. autoexception:: ParameterEncodingError

.. automodule:: sigima.proc.contracts.registry
    :no-members:

.. currentmodule:: sigima.proc.contracts

.. autoclass:: OperationContract
.. autoclass:: ContractRegistry
.. autofunction:: build_contract_registry
.. autofunction:: contract_for_function
.. autofunction:: get_operation_contract
.. autofunction:: parameters_from_values
.. autofunction:: parameters_to_values

.. automodule:: sigima.proc.contracts.codec
    :no-members:

.. currentmodule:: sigima.proc.contracts

.. py:data:: PARAMETER_CODEC_VERSION
    :type: int

    Version of the parameter encoding produced by the parameter codec.

.. autofunction:: encode_parameters
.. autofunction:: decode_parameters
.. autofunction:: encode_dataset
.. autofunction:: decode_dataset

.. automodule:: sigima.proc.contracts.execution
    :no-members:

.. currentmodule:: sigima.proc.contracts

.. autoclass:: PreparedExecution
.. autofunction:: describe_call
.. autofunction:: prepare_execution
.. autofunction:: execute
.. autofunction:: normalize_outputs
.. autofunction:: with_roi
.. autofunction:: default_roles
.. autofunction:: implementation_of
.. autofunction:: image_roi_convention
.. py:data:: IMAGE_ROI_CONVENTION
    :type: types.MappingProxyType

    Selection of image pixels by regions of interest: pixels whose centers lie
    in their shape, bounds included (recorded under ``roi_convention``).

.. autoexception:: ExecutionError
.. py:data:: EXECUTION_ERROR_CODES
    :type: tuple[str, ...]

    Stable codes of :class:`ExecutionError`.

.. automodule:: sigima.proc.contracts.policies
    :no-members:

.. currentmodule:: sigima.proc.contracts

.. autoclass:: XAlignmentPolicy
.. autofunction:: get_x_alignment_policy
.. autofunction:: register_x_alignment_policy
.. autofunction:: x_alignment_policies
.. autofunction:: check_x_alignment_record

.. automodule:: sigima.proc.contracts.alignment
    :no-members:

.. currentmodule:: sigima.proc.contracts

.. py:data:: SOURCE_GRID_LINEAR
    :type: str

    Identifier of the source-grid rule.

.. py:data:: SOURCE_GRID_LINEAR_HOLD
    :type: str

    Identifier of the historical source-grid rule (constant extension).

.. py:data:: SMALLEST_GRID_LINEAR_HOLD
    :type: str

    Identifier of the historical smallest-grid rule (constant extension).

.. py:data:: X_ALIGNMENT_RULES
    :type: frozenset[tuple[str, int]]

    Supported ``(rule, version)`` pairs.

.. autoexception:: XAlignmentError
