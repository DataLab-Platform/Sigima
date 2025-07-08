"""
Tools (:mod:`sigima.tools`)
---------------------------

This package contains functions operating on NumPy arrays that are intended to be
used in Sigima computation functions. These functions are complementary to the
algorithms provided by external libraries such as SciPy, NumPy, and scikit-image.

Even though these functions are primarily designed to be used in the Sigima pipeline,
they can also be used independently. They provide a wide range of algorithms but are
not exhaustive due to the vast number of algorithms already available in the
scientific Python ecosystem.

.. seealso::

    The :mod:`sigima.proc` contains the Sigima computation functions that operate on
    signal and image objects (i.e. :class:`sigima.objects.SignalObj` and
    :class:`sigima.objects.ImageObj`, defined in the :mod:`sigima.objects` package).

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
