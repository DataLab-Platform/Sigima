# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Parameters (:mod:`sigima.params`)
---------------------------------

The :mod:`sigima.params` module aims at providing all the dataset parameters that are
used by the :mod:`sigima.proc` and DataLab's processors.

Those datasets are defined in other modules:

    - :mod:`sigima.proc.base`
    - :mod:`sigima.proc.image`
    - :mod:`sigima.proc.signal`

The :mod:`sigima.params` module is thus a convenient way to import all the sets of
parameters at once.

As a matter of fact, the following import statement is equivalent to the previous one:

.. code-block:: python

    # Original import statement
    from sigima.proc.base import MovingAverageParam
    from sigima.proc.signal import PolynomialFitParam
    from sigima.proc.image.exposure import EqualizeHistParam

    # Equivalent import statement
    from sigima.params import MovingAverageParam, PolynomialFitParam, EqualizeHistParam

Introduction to `DataSet` parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The datasets listed in the following sections are used to define the parameters
necessary for the various computations and processing operations available in DataLab.

Each dataset is a subclass of :py:class:`guidata.dataset.datatypes.DataSet` and thus
needs to be instantiated before being used.

Here is a complete example of how to instantiate a dataset and access its parameters
with the :py:class:`sigima.params.BinningParam` dataset:

    .. autodataset:: sigima.params.BinningParam
        :no-index:
        :shownote:

Common parameters
^^^^^^^^^^^^^^^^^

.. autodataset:: sigima.params.ArithmeticParam
    :no-index:
.. autodataset:: sigima.params.ClipParam
    :no-index:
.. autodataset:: sigima.params.ConstantParam
    :no-index:
.. autodataset:: sigima.params.FFTParam
    :no-index:
.. autodataset:: sigima.params.GaussianParam
    :no-index:
.. autodataset:: sigima.params.HistogramParam
    :no-index:
.. autodataset:: sigima.params.MovingAverageParam
    :no-index:
.. autodataset:: sigima.params.MovingMedianParam
    :no-index:
.. autodataset:: sigima.params.NormalizeParam
    :no-index:
.. autodataset:: sigima.params.SpectrumParam
    :no-index:

Signal parameters
^^^^^^^^^^^^^^^^^

.. autodataset:: sigima.params.AllanVarianceParam
    :no-index:
.. autodataset:: sigima.params.AngleUnitParam
    :no-index:
.. autodataset:: sigima.params.BandPassFilterParam
    :no-index:
.. autodataset:: sigima.params.BandStopFilterParam
    :no-index:
.. autodataset:: sigima.params.DataTypeSParam
    :no-index:
.. autodataset:: sigima.params.DetrendingParam
    :no-index:
.. autodataset:: sigima.params.DynamicParam
    :no-index:
.. autodataset:: sigima.params.AbscissaParam
    :no-index:
.. autodataset:: sigima.params.OrdinateParam
    :no-index:
.. autodataset:: sigima.params.FWHMParam
    :no-index:
.. autodataset:: sigima.params.HighPassFilterParam
    :no-index:
.. autodataset:: sigima.params.InterpolationParam
    :no-index:
.. autodataset:: sigima.params.LowPassFilterParam
    :no-index:
.. autodataset:: sigima.params.PeakDetectionParam
    :no-index:
.. autodataset:: sigima.params.PolynomialFitParam
    :no-index:
.. autodataset:: sigima.params.PowerParam
    :no-index:
.. autodataset:: sigima.params.ResamplingParam
    :no-index:
.. autodataset:: sigima.params.WindowingParam
    :no-index:
.. autodataset:: sigima.params.XYCalibrateParam
    :no-index:
.. autodataset:: sigima.params.ZeroPadding1DParam
    :no-index:

Image parameters
^^^^^^^^^^^^^^^^

Base image parameters
~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.AverageProfileParam
    :no-index:
.. autodataset:: sigima.params.BinningParam
    :no-index:
.. autodataset:: sigima.params.ButterworthParam
    :no-index:
.. autodataset:: sigima.params.DataTypeIParam
    :no-index:
.. autodataset:: sigima.params.FlatFieldParam
    :no-index:
.. autodataset:: sigima.params.GridParam
    :no-index:
.. autodataset:: sigima.params.HoughCircleParam
    :no-index:
.. autodataset:: sigima.params.LineProfileParam
    :no-index:
.. autodataset:: sigima.params.LogP1Param
    :no-index:
.. autodataset:: sigima.params.RadialProfileParam
    :no-index:
.. autodataset:: sigima.params.ResizeParam
    :no-index:
.. autodataset:: sigima.params.RotateParam
    :no-index:
.. autodataset:: sigima.params.SegmentProfileParam
    :no-index:
.. autodataset:: sigima.params.ZCalibrateParam
    :no-index:
.. autodataset:: sigima.params.ZeroPadding2DParam
    :no-index:

Detection parameters
~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.BlobDOGParam
    :no-index:
.. autodataset:: sigima.params.BlobDOHParam
    :no-index:
.. autodataset:: sigima.params.BlobLOGParam
    :no-index:
.. autodataset:: sigima.params.BlobOpenCVParam
    :no-index:
.. autodataset:: sigima.params.ContourShapeParam
    :no-index:
.. autodataset:: sigima.params.Peak2DDetectionParam
    :no-index:

Edge detection parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.CannyParam
    :no-index:

Exposure correction parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.AdjustGammaParam
    :no-index:
.. autodataset:: sigima.params.AdjustLogParam
    :no-index:
.. autodataset:: sigima.params.AdjustSigmoidParam
    :no-index:
.. autodataset:: sigima.params.EqualizeAdaptHistParam
    :no-index:
.. autodataset:: sigima.params.EqualizeHistParam
    :no-index:
.. autodataset:: sigima.params.RescaleIntensityParam
    :no-index:

Morphological parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.MorphologyParam
    :no-index:

Restoration parameters
~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.DenoiseBilateralParam
    :no-index:
.. autodataset:: sigima.params.DenoiseTVParam
    :no-index:
.. autodataset:: sigima.params.DenoiseWaveletParam
    :no-index:

Threshold parameters
~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.ThresholdParam
    :no-index:
"""

# pylint:disable=unused-import
# flake8: noqa

from sigima.proc.base import (
    ArithmeticParam,
    ClipParam,
    ConstantParam,
    FFTParam,
    GaussianParam,
    HistogramParam,
    MovingAverageParam,
    MovingMedianParam,
    NormalizeParam,
    SpectrumParam,
)
from sigima.proc.signal import (
    AllanVarianceParam,
    AbscissaParam,
    AngleUnitParam,
    BandPassFilterParam,
    BandStopFilterParam,
    DataTypeSParam,
    DetrendingParam,
    DynamicParam,
    OrdinateParam,
    FWHMParam,
    HighPassFilterParam,
    InterpolationParam,
    LowPassFilterParam,
    PeakDetectionParam,
    PolynomialFitParam,
    PowerParam,
    ResamplingParam,
    WindowingParam,
    XYCalibrateParam,
    ZeroPadding1DParam,
)

from sigima.proc.image import (
    GridParam,
)
from sigima.proc.image import (
    BlobDOGParam,
    BlobDOHParam,
    BlobLOGParam,
    BlobOpenCVParam,
    ContourShapeParam,
    HoughCircleParam,
    Peak2DDetectionParam,
)
from sigima.proc.image import CannyParam
from sigima.proc.image import (
    AdjustGammaParam,
    AdjustLogParam,
    AdjustSigmoidParam,
    EqualizeAdaptHistParam,
    EqualizeHistParam,
    FlatFieldParam,
    RescaleIntensityParam,
    ZCalibrateParam,
)
from sigima.proc.image import (
    AverageProfileParam,
    LineProfileParam,
    RadialProfileParam,
    SegmentProfileParam,
)
from sigima.proc.image import ButterworthParam
from sigima.proc.image import ZeroPadding2DParam
from sigima.proc.image import BinningParam, ResizeParam, RotateParam
from sigima.proc.image import DataTypeIParam, LogP1Param
from sigima.proc.image import MorphologyParam
from sigima.proc.image import (
    DenoiseBilateralParam,
    DenoiseTVParam,
    DenoiseWaveletParam,
)
from sigima.proc.image import ThresholdParam
