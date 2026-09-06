# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Validation helpers for parameter DataSets."""

from __future__ import annotations

from typing import Protocol

__all__ = ["validate_dataset"]


class ParameterValidator(Protocol):
    """Structural contract for DataSets with relational validation."""

    def validate_parameters(self, *context: object) -> None:
        """Validate parameters, optionally using execution context."""


def validate_dataset(dataset: object, *context: object) -> None:
    """Run a DataSet's optional relational validation hook.

    Args:
        dataset: DataSet-like object to validate.
        *context: Execution objects required by contextual validation.
    """
    validator = getattr(dataset, "validate_parameters", None)
    if callable(validator):
        validator(*context)
