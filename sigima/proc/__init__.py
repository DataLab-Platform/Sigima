"""
Computation (:mod:`sigima.proc`)
--------------------------------

This package contains the Sigima computation functions that implement processing
features for signal and image objects. These functions are designed to operate directly
on :class:`sigima.objects.SignalObj` and :class:`sigima.objects.ImageObj` objects,
which are defined in the :mod:`sigima.objects` package.

Even though these functions are primarily designed to be used in the Sigima pipeline,
they can also be used independently.

.. seealso::

    See the :mod:`sigima.tools` package for some algorithms that operate directly on
    NumPy arrays.

Each computation module defines a set of computation objects, that is, functions
that implement processing features and classes that implement the corresponding
parameters (in the form of :py:class:`guidata.dataset.datatypes.Dataset` subclasses).
The computation functions takes a signal or image object
(e.g. :py:class:`sigima.objects.SignalObj`)
and a parameter object (e.g. :py:class:`sigima.params.MovingAverageParam`) as input
and return a signal or image object as output (the result of the computation).
The parameter object is used to configure the computation function
(e.g. the size of the moving average window).

In Sigima overall architecture, the purpose of this package is to provide the
computation functions that are used by the :mod:`sigima.core.gui.processor` module,
based on the algorithms defined in the :mod:`sigima.tools` module and on the
data model defined in the :mod:`sigima.objects` (or :mod:`sigima.core.model`) module.

The computation modules are organized in subpackages according to their purpose.
The following subpackages are available:

- :mod:`sigima.proc.base`: Common processing features
- :mod:`sigima.proc.signal`: Signal processing features
- :mod:`sigima.proc.image`: Image processing features

Common processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.proc.base
   :members:

Signal processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.proc.signal
   :members:

Image processing features
^^^^^^^^^^^^^^^^^^^^^^^^^

Basic image processing
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image
   :members:

Thresholding
~~~~~~~~~~~~

.. automodule:: sigima.proc.image.threshold
    :members:

Exposure correction
~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image.exposure
    :members:

Restoration
~~~~~~~~~~~

.. automodule:: sigima.proc.image.restoration
    :members:

Morphology
~~~~~~~~~~

.. automodule:: sigima.proc.image.morphology
    :members:

Edge detection
~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image.edges

Detection
~~~~~~~~~

.. automodule:: sigima.proc.image.detection
    :members:

Utilities
^^^^^^^^^

This package also provides some utility functions to help with the creation and
introspection of computation functions:

.. autofunction:: sigima.proc.computation_function
.. autofunction:: sigima.proc.is_computation_function
.. autofunction:: sigima.proc.get_computation_metadata
.. autofunction:: sigima.proc.find_computation_functions
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import os.path as osp
import pkgutil
import sys
from types import ModuleType
from typing import Callable, Optional, TypeVar

if sys.version_info >= (3, 10):
    # Use ParamSpec from typing module in Python 3.10+
    from typing import ParamSpec
else:
    # Use ParamSpec from typing_extensions module in Python < 3.10
    from typing_extensions import ParamSpec

# Marker attribute used by @computation_function and introspection
COMPUTATION_METADATA_ATTR = "__computation_function_metadata"

P = ParamSpec("P")
R = TypeVar("R")


@dataclasses.dataclass(frozen=True)
class ComputationMetadata:
    """Metadata for a computation function.

    Attributes:
        name: The name of the computation function.
        description: A description or docstring for the computation function.
    """

    name: str
    description: str


def computation_function(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to mark a function as a Sigima computation function.

    Args:
        name: Optional name to override the function name.
        description: Optional docstring override or additional description.

    Returns:
        The wrapped function, tagged with a marker attribute.
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        """Decorator to mark a function as a Sigima computation function.
        This decorator adds a marker attribute to the function, allowing
        it to be identified as a computation function.
        It also allows for optional name and description overrides.
        The function can be used as a decorator or as a standalone function.
        """

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return f(*args, **kwargs)

        metadata = ComputationMetadata(
            name=name or f.__name__, description=description or f.__doc__
        )
        setattr(wrapper, COMPUTATION_METADATA_ATTR, metadata)
        return wrapper

    return decorator


def is_computation_function(function: Callable) -> bool:
    """Check if a function is a Sigima computation function.

    Args:
        function: The function to check.

    Returns:
        True if the function is a Sigima computation function, False otherwise.
    """
    return getattr(function, COMPUTATION_METADATA_ATTR, None) is not None


def get_computation_metadata(function: Callable) -> ComputationMetadata:
    """Get the metadata of a Sigima computation function.

    Args:
        function: The function to get metadata from.

    Returns:
        Computation function metadata.

    Raises:
        ValueError: If the function is not a Sigima computation function.
    """
    metadata = getattr(function, COMPUTATION_METADATA_ATTR, None)
    if not isinstance(metadata, ComputationMetadata):
        raise ValueError(
            f"The function {function.__name__} is not a Sigima computation function."
        )
    return metadata


def find_computation_functions(
    module: ModuleType | None = None,
) -> list[tuple[str, Callable]]:
    """Find all computation functions in the `sigima.proc` package.

    This function uses introspection to locate all functions decorated with
    `@computation_function` in the `sigima.proc` package and its subpackages.

    Args:
        module: Optional module to search in. If None, the current module is used.

    Returns:
        A list of tuples, each containing the function name and the function object.
    """
    functions = []
    if module is None:
        path = [osp.dirname(__file__)]
    else:
        path = module.__path__
    objs = []
    for _, modname, _ in pkgutil.walk_packages(path=path, prefix=__name__ + "."):
        try:
            module = importlib.import_module(modname)
        except Exception:  # pylint: disable=broad-except
            continue
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if is_computation_function(obj):
                if obj in objs:  # Avoid double entries for the same function
                    continue
                objs.append(obj)
                functions.append((modname, name, obj.__doc__))
    return functions
