"""
Computation (:mod:`sigima.proc`)
--------------------------------

This package contains the Sigima computation functions that implement processing
features for signal, image and scalar objects. These functions are designed to operate
directly on :class:`sigima.objects.SignalObj`, :class:`sigima.objects.ImageObj`,
:class:`sigima.objects.GeometryResult` and :class:`sigima.objects.TableResult` objects,
which are defined in the :mod:`sigima.objects` package.

Even though these functions are primarily designed to be used in the Sigima pipeline,
they can also be used independently.

.. seealso::

    See the :mod:`sigima.tools` package for some algorithms that operate directly on
    NumPy arrays.

Each computation module defines a set of computation objects, that is, functions
that implement processing features and classes that implement the corresponding
parameters (in the form of :py:class:`guidata.dataset.datatypes.DataSet` subclasses).
The computation functions takes a signal or image object
(e.g. :py:class:`sigima.objects.SignalObj`)
and a parameter object (e.g. :py:class:`sigima.params.MovingAverageParam`) as input
and return a signal or image object as output (the result of the computation).
The parameter object is used to configure the computation function
(e.g. the size of the moving average window).

In Sigima overall architecture, the purpose of this package is to provide the
computation functions that are used by DataLab processor modules,
based on the algorithms defined in the :mod:`sigima.tools` module and on the
data model defined in the :mod:`sigima.objects` module.

The computation modules are organized in subpackages according to their purpose.
The following subpackages are available:

- :mod:`sigima.proc.base`: Common processing features
- :mod:`sigima.proc.signal`: Signal processing features
- :mod:`sigima.proc.image`: Image processing features (including transformations)

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

.. automodule:: sigima.proc.image
   :members:

Decorators and utilities
^^^^^^^^^^^^^^^^^^^^^^^^

This package also provides some utility functions to help with the creation and
introspection of computation functions:

.. autofunction:: sigima.proc.decorator.computation_function
.. autofunction:: sigima.proc.decorator.is_computation_function
.. autofunction:: sigima.proc.decorator.get_computation_metadata
.. autofunction:: sigima.proc.decorator.find_computation_functions

Title Formatting System
^^^^^^^^^^^^^^^^^^^^^^^

The title formatting system provides configurable title generation for computation
results, enabling different applications (Sigima standalone vs DataLab integration)
to use different title formatting strategies:

.. automodule:: sigima.proc.title_formatting
   :members:
"""

# Import title formatting components for easy access
from sigima.proc.title_formatting import (
    PlaceholderTitleFormatter,
    SimpleTitleFormatter,
    TitleFormatter,
    get_default_title_formatter,
    set_default_title_formatter,
)

__all__ = [
    "PlaceholderTitleFormatter",
    "SimpleTitleFormatter",
    "TitleFormatter",
    "get_default_title_formatter",
    "set_default_title_formatter",
]
