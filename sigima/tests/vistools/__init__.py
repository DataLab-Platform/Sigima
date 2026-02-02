# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Backward compatibility shim for sigima.tests.vistools
=====================================================

.. deprecated:: 1.1.0
    The ``sigima.tests.vistools`` module has been moved to :mod:`sigima.viz`.
    Please update your imports to use ``from sigima import viz`` instead.

This module re-exports all functions from :mod:`sigima.viz` for backward
compatibility with code that imports from ``sigima.tests.vistools``.

Example migration::

    # Old (deprecated):
    from sigima.tests import vistools
    vistools.view_curves([signal])

    # New (recommended):
    from sigima import viz
    viz.view_curves([signal])
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "sigima.tests.vistools is deprecated and will be removed in a future version. "
    "Use 'from sigima import viz' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from sigima.viz
from sigima.viz import *  # noqa: F401, F403, E402  # pylint: disable=C0413

# Note: BACKEND_NAME and BACKEND_SOURCE are included via star import from __all__
# They are lazy-loaded attributes that will be available when accessed, but we
# don't import them explicitly to avoid triggering backend initialization during
# module import (which would fail in environments without visualization backends)
