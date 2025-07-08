"""
Algorithms (:mod:`sigima.tools`)
--------------------------------------

This package contains the algorithms used by the DataLab project. Those algorithms
operate directly on NumPy arrays and are designed to be used in the DataLab pipeline,
but can be used independently as well.

.. seealso::

    The :mod:`sigima.tools` package is the main entry point for the DataLab
    algorithms when manipulating NumPy arrays. See the :mod:`sigima.proc`
    package for algorithms that operate directly on DataLab objects (i.e.
    :class:`sigima.objects.SignalObj` and :class:`sigima.objects.ImageObj`).

The algorithms are organized in subpackages according to their purpose. The following
subpackages are available:

- :mod:`sigima.tools.signal`: Signal processing algorithms
- :mod:`sigima.tools.image`: Image processing algorithms
- :mod:`sigima.tools.datatypes`: Data type conversion algorithms
- :mod:`sigima.tools.coordinates`: Coordinate conversion algorithms

Signal Processing Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.tools.signal
   :members:

Image Processing Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.tools.image
   :members:

Data Type Conversion Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.tools.datatypes
   :members:

Coordinate Conversion Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.tools.coordinates
   :members:

"""
