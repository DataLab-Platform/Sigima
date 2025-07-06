# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Model classes for signals and images (:mod:`sigima.obj`)
---------------------------------------------------------

The :mod:`sigima.obj` module aims at providing all the necessary classes and functions
to create and manipulate DataLab signal and image objects.

Those classes and functions are defined in submodules:
    - :mod:`sigima.obj.base`
    - :mod:`sigima.obj.image`
    - :mod:`sigima.obj.signal`

.. code-block:: python

    # Full import statement
    from sigima.obj.signal import SignalObj
    from sigima.obj.image import ImageObj

    # Short import statement
    from sigima.obj import SignalObj, ImageObj

Common objects
^^^^^^^^^^^^^^

.. autoclass:: sigima.obj.ResultProperties
    :members:
.. autoclass:: sigima.obj.ResultShape
    :members:
.. autoclass:: sigima.obj.ShapeTypes
    :members:
.. autoclass:: sigima.obj.UniformRandomParam
.. autoclass:: sigima.obj.NormalRandomParam
.. autoclass:: sigima.obj.TypeObj
.. autoclass:: sigima.obj.TypeROI
.. autoclass:: sigima.obj.TypeROIParam
.. autoclass:: sigima.obj.TypeSingleROI


Signal model
^^^^^^^^^^^^

.. autodataset:: sigima.obj.SignalObj
    :members:
    :inherited-members:
.. autofunction:: sigima.obj.read_signal
.. autofunction:: sigima.obj.read_signals
.. autofunction:: sigima.obj.create_signal_roi
.. autofunction:: sigima.obj.create_signal
.. autofunction:: sigima.obj.create_signal_from_param
.. autofunction:: sigima.obj.new_signal_param
.. autoclass:: sigima.obj.SignalTypes
.. autodataset:: sigima.obj.NewSignalParam
.. autodataset:: sigima.obj.GaussLorentzVoigtParam
.. autodataset:: sigima.obj.StepParam
.. autodataset:: sigima.obj.PeriodicParam
.. autodataset:: sigima.obj.ROI1DParam
.. autoclass:: sigima.obj.SignalROI

Image model
^^^^^^^^^^^

.. autodataset:: sigima.obj.ImageObj
    :members:
    :inherited-members:
.. autofunction:: sigima.obj.read_image
.. autofunction:: sigima.obj.read_images
.. autofunction:: sigima.obj.create_image_roi
.. autofunction:: sigima.obj.create_image
.. autofunction:: sigima.obj.create_image_from_param
.. autofunction:: sigima.obj.new_image_param
.. autoclass:: sigima.obj.ImageTypes
.. autodataset:: sigima.obj.NewImageParam
.. autodataset:: sigima.obj.Gauss2DParam
.. autodataset:: sigima.obj.ROI2DParam
.. autoclass:: sigima.obj.ImageROI
.. autoclass:: sigima.obj.ImageDatatypes
"""

# pylint:disable=unused-import
# flake8: noqa

from sigima.obj.base import (
    UniformRandomParam,
    NormalRandomParam,
    ResultProperties,
    ResultShape,
    TypeObj,
    TypeROI,
    TypeROIParam,
    TypeSingleROI,
    ShapeTypes,
)
from sigima.obj.image import (
    ImageObj,
    ImageROI,
    create_image_roi,
    create_image,
    create_image_from_param,
    Gauss2DParam,
    ROI2DParam,
    RectangularROI,
    ImageTypes,
    CircularROI,
    PolygonalROI,
    ImageDatatypes,
    NewImageParam,
)
from sigima.obj.signal import (
    SignalObj,
    ROI1DParam,
    SegmentROI,
    SignalTypes,
    SignalROI,
    create_signal_roi,
    create_signal,
    create_signal_from_param,
    ExponentialParam,
    ExperimentalSignalParam,
    PulseParam,
    PolyParam,
    StepParam,
    PeriodicParam,
    GaussLorentzVoigtParam,
    NewSignalParam,
)
