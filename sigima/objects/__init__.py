# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Model classes for signals and images (:mod:`sigima.objects`)
---------------------------------------------------------

The :mod:`sigima.objects` module aims at providing all the necessary classes and functions
to create and manipulate DataLab signal and image objects.

Those classes and functions are defined in submodules:
    - :mod:`sigima.objects.base`
    - :mod:`sigima.objects.image`
    - :mod:`sigima.objects.signal`

.. code-block:: python

    # Full import statement
    from sigima.objects.signal import SignalObj
    from sigima.objects.image import ImageObj

    # Short import statement
    from sigima.objects import SignalObj, ImageObj

Common objects
^^^^^^^^^^^^^^

.. autoclass:: sigima.objects.ResultProperties
    :members:
.. autoclass:: sigima.objects.ResultShape
    :members:
.. autoclass:: sigima.objects.ShapeTypes
    :members:
.. autoclass:: sigima.objects.UniformRandomParam
.. autoclass:: sigima.objects.NormalRandomParam
.. autoclass:: sigima.objects.TypeObj
.. autoclass:: sigima.objects.TypeROI
.. autoclass:: sigima.objects.TypeROIParam
.. autoclass:: sigima.objects.TypeSingleROI


Signal model
^^^^^^^^^^^^

.. autodataset:: sigima.objects.SignalObj
    :members:
    :inherited-members:
.. autofunction:: sigima.objects.create_signal_roi
.. autofunction:: sigima.objects.create_signal
.. autofunction:: sigima.objects.create_signal_from_param
.. autoclass:: sigima.objects.SignalTypes
.. autodataset:: sigima.objects.NewSignalParam
.. autodataset:: sigima.objects.GaussLorentzVoigtParam
.. autodataset:: sigima.objects.StepParam
.. autodataset:: sigima.objects.PeriodicParam
.. autodataset:: sigima.objects.ROI1DParam
.. autoclass:: sigima.objects.SignalROI

Image model
^^^^^^^^^^^

.. autodataset:: sigima.objects.ImageObj
    :members:
    :inherited-members:
.. autofunction:: sigima.objects.create_image_roi
.. autofunction:: sigima.objects.create_image
.. autofunction:: sigima.objects.create_image_from_param
.. autoclass:: sigima.objects.ImageTypes
.. autodataset:: sigima.objects.NewImageParam
.. autodataset:: sigima.objects.Gauss2DParam
.. autodataset:: sigima.objects.ROI2DParam
.. autoclass:: sigima.objects.ImageROI
.. autoclass:: sigima.objects.ImageDatatypes
"""

# pylint:disable=unused-import
# flake8: noqa

from sigima.objects.base import (
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
from sigima.objects.image import (
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
from sigima.objects.signal import (
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
