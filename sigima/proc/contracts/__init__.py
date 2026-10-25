# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Operation contracts
===================

Every public computation function of Sigima declares an operation contract next
to it (``@computation_function(operation_id=..., contract_version=...)``):
stable identifier and version, X-alignment policy, global options read and
replay qualification. The structure of the contract (input roles, parameter
arguments, outputs) is inferred from the annotations of the function.
Applications use contracts to record what was computed and to replay it later,
possibly with another Sigima version, with the same meaning.

This package is the public API of contracts; import from it rather than from
its modules:

- :mod:`~sigima.proc.contracts.declarations`: what a function declares (roles,
  parameter specifications, comparison rules, replay qualification, errors);
- :mod:`~sigima.proc.contracts.registry`: contract inference and registry
  (:func:`build_contract_registry`, :func:`contract_for_function`,
  :func:`get_operation_contract`);
- :mod:`~sigima.proc.contracts.codec`: parameter codec (JSON values of the
  parameters actually used, rebuilt against the local classes only);
- :mod:`~sigima.proc.contracts.execution`: description and preparation of calls,
  shared by ordinary processing and replay (:func:`describe_call`,
  :func:`prepare_execution`, :func:`normalize_outputs`, :func:`with_roi`), and
  the conventions recorded in the execution context
  (:func:`image_roi_convention`);
- :mod:`~sigima.proc.contracts.policies` and
  :mod:`~sigima.proc.contracts.alignment`: X-alignment policies of signal
  operations and their algorithms.

Names are loaded on first access, so that :mod:`sigima.proc.decorator`, which the
registry itself uses, can import the declarations without an import cycle.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sigima.proc.contracts.alignment import (
        SMALLEST_GRID_LINEAR_HOLD,
        SOURCE_GRID_LINEAR,
        SOURCE_GRID_LINEAR_HOLD,
        X_ALIGNMENT_RULES,
        XAlignmentError,
    )
    from sigima.proc.contracts.codec import (
        PARAMETER_CODEC_VERSION,
        decode_dataset,
        decode_parameters,
        encode_dataset,
        encode_parameters,
    )
    from sigima.proc.contracts.declarations import (
        ARTIFACT_KINDS,
        EXACT,
        OBJECT_KINDS,
        ComparisonRule,
        ContractDeclarationError,
        ContractError,
        IncompatibleContractError,
        InvalidParametersError,
        OutputTolerance,
        ParameterEncodingError,
        ParameterSpec,
        ReplayQualification,
        Role,
        UnknownOperationError,
        tolerance,
    )
    from sigima.proc.contracts.execution import (
        EXECUTION_ERROR_CODES,
        IMAGE_ROI_CONVENTION,
        ExecutionError,
        PreparedExecution,
        default_roles,
        describe_call,
        execute,
        image_roi_convention,
        implementation_of,
        normalize_outputs,
        prepare_execution,
        with_roi,
    )
    from sigima.proc.contracts.policies import (
        XAlignmentPolicy,
        check_x_alignment_record,
        get_x_alignment_policy,
        register_x_alignment_policy,
        x_alignment_policies,
    )
    from sigima.proc.contracts.registry import (
        ContractRegistry,
        OperationContract,
        build_contract_registry,
        contract_for_function,
        get_operation_contract,
        parameters_from_values,
        parameters_to_values,
    )

#: Public name -> module of this package defining it.
_NAMES = {
    "declarations": (
        "ARTIFACT_KINDS",
        "EXACT",
        "OBJECT_KINDS",
        "ComparisonRule",
        "ContractDeclarationError",
        "ContractError",
        "IncompatibleContractError",
        "InvalidParametersError",
        "OutputTolerance",
        "ParameterEncodingError",
        "ParameterSpec",
        "ReplayQualification",
        "Role",
        "UnknownOperationError",
        "tolerance",
    ),
    "registry": (
        "ContractRegistry",
        "OperationContract",
        "build_contract_registry",
        "contract_for_function",
        "get_operation_contract",
        "parameters_from_values",
        "parameters_to_values",
    ),
    "codec": (
        "PARAMETER_CODEC_VERSION",
        "decode_dataset",
        "decode_parameters",
        "encode_dataset",
        "encode_parameters",
    ),
    "execution": (
        "EXECUTION_ERROR_CODES",
        "IMAGE_ROI_CONVENTION",
        "ExecutionError",
        "PreparedExecution",
        "default_roles",
        "describe_call",
        "execute",
        "image_roi_convention",
        "implementation_of",
        "normalize_outputs",
        "prepare_execution",
        "with_roi",
    ),
    "policies": (
        "XAlignmentPolicy",
        "check_x_alignment_record",
        "get_x_alignment_policy",
        "register_x_alignment_policy",
        "x_alignment_policies",
    ),
    "alignment": (
        "SMALLEST_GRID_LINEAR_HOLD",
        "SOURCE_GRID_LINEAR",
        "SOURCE_GRID_LINEAR_HOLD",
        "X_ALIGNMENT_RULES",
        "XAlignmentError",
    ),
}
_MODULE_OF = {name: module for module, names in _NAMES.items() for name in names}
_SUBMODULES = ("_annotations", *_NAMES)

__all__ = [
    "ARTIFACT_KINDS",
    "EXACT",
    "EXECUTION_ERROR_CODES",
    "IMAGE_ROI_CONVENTION",
    "OBJECT_KINDS",
    "PARAMETER_CODEC_VERSION",
    "SMALLEST_GRID_LINEAR_HOLD",
    "SOURCE_GRID_LINEAR",
    "SOURCE_GRID_LINEAR_HOLD",
    "X_ALIGNMENT_RULES",
    "ComparisonRule",
    "ContractDeclarationError",
    "ContractError",
    "ContractRegistry",
    "ExecutionError",
    "IncompatibleContractError",
    "InvalidParametersError",
    "OperationContract",
    "OutputTolerance",
    "ParameterEncodingError",
    "ParameterSpec",
    "PreparedExecution",
    "ReplayQualification",
    "Role",
    "UnknownOperationError",
    "XAlignmentError",
    "XAlignmentPolicy",
    "build_contract_registry",
    "check_x_alignment_record",
    "contract_for_function",
    "decode_dataset",
    "decode_parameters",
    "default_roles",
    "describe_call",
    "encode_dataset",
    "encode_parameters",
    "execute",
    "get_operation_contract",
    "get_x_alignment_policy",
    "image_roi_convention",
    "implementation_of",
    "normalize_outputs",
    "parameters_from_values",
    "parameters_to_values",
    "prepare_execution",
    "register_x_alignment_policy",
    "tolerance",
    "with_roi",
    "x_alignment_policies",
]


def __getattr__(name: str) -> Any:
    """Load a public name, or a module of this package, on first access."""
    module = _MODULE_OF.get(name)
    if module is not None:
        value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
