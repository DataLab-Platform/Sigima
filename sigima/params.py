# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Parameters (:mod:`sigima.params`)
---------------------------------

The :mod:`sigima.params` module provides all the dataset parameter classes used by
:mod:`sigima.proc` processing functions and DataLab's GUI.

.. tip::

    **Always import parameters from** :mod:`sigima.params`. While parameter classes
    are defined in various submodules (e.g., ``sigima.proc.signal.fourier``), they are
    all re-exported here for convenience. This avoids confusion about where to import
    from.

    .. code-block:: python

        # ✅ Recommended: import from sigima.params
        import sigima.params
        param = sigima.params.ZeroPadding1DParam.create(strategy="next_pow2")

        # ❌ Avoid: importing from internal modules (works but less clear)
        from sigima.proc.signal.fourier import ZeroPadding1DParam

Introduction to ``DataSet`` Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The datasets listed in the following sections define the parameters necessary for
computation and processing operations in Sigima. Each dataset is a subclass of
:py:class:`guidata.dataset.datatypes.DataSet` and needs to be instantiated before use.

Creating Parameter Instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter classes provide a ``create()`` class method for easy instantiation:

.. code-block:: python

    import sigima.params

    # Create with default values
    param = sigima.params.NormalizeParam.create()

    # Create with custom values
    param = sigima.params.NormalizeParam.create(method="maximum")

Parameters Requiring Signal/Image Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some parameters need to know about the signal or image they will process in order
to compute their values. For example, :class:`ZeroPadding1DParam` needs to know
the signal size to calculate how many points to add for the "next_pow2" strategy.

These parameters provide an ``update_from_obj()`` method that **must be called**
before using the parameters:

.. code-block:: python

    import sigima.params
    import sigima.proc.signal as sips

    # Create the parameter object
    param = sigima.params.ZeroPadding1DParam.create(strategy="next_pow2")

    # ⚠️ At this point, param.n is still the default value (1)
    # because the parameter doesn't know the signal size yet

    # IMPORTANT: Update parameters from the signal
    param.update_from_obj(signal)

    # ✅ Now param.n is computed (e.g., 24 for a 1000-point signal)
    result = sips.zero_padding(signal, param)

Parameter classes that require ``update_from_obj()``:

- :class:`ZeroPadding1DParam`: Computes ``n`` based on strategy and signal size
- :class:`ZeroPadding2DParam`: Computes padding based on strategy and image size
- :class:`Resampling1DParam`: Updates bounds based on signal range
- :class:`Resampling2DParam`: Updates bounds based on signal range
- :class:`ResizeParam`: Updates bounds based on image dimensions
- :class:`TranslateParam`: Updates bounds based on image dimensions
- :class:`LineProfileParam`: Updates line coordinates based on image dimensions
- :class:`BandPassFilterParam`, :class:`BandStopFilterParam`,
  :class:`HighPassFilterParam`, :class:`LowPassFilterParam`: Update frequency bounds

.. note::

    Not all parameters require ``update_from_obj()``. Simple parameters like
    :class:`NormalizeParam` or :class:`GaussianParam` work with fixed values and
    don't need signal/image context.

Complete Example
~~~~~~~~~~~~~~~~

Here is a complete example of how to instantiate a dataset and access its parameters
with the :py:class:`sigima.params.BinningParam` dataset:

    .. autodataset:: sigima.params.BinningParam
        :no-index:
        :shownote:

I/O parameters
^^^^^^^^^^^^^^

.. autodataset:: sigima.io.convenience.SaveToDirectoryParam
    :no-index:

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
.. autodataset:: sigima.params.PulseFeaturesParam
    :no-index:
.. autodataset:: sigima.params.Resampling1DParam
    :no-index:
.. autodataset:: sigima.params.Resampling2DParam
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

.. autodataset:: sigima.params.GridParam
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
.. autodataset:: sigima.params.HoughCircleParam
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
.. autodataset:: sigima.params.FlatFieldParam
    :no-index:
.. autodataset:: sigima.params.XYZCalibrateParam
    :no-index:

Extraction parameters
~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.AverageProfileParam
    :no-index:
.. autodataset:: sigima.params.LineProfileParam
    :no-index:
.. autodataset:: sigima.params.RadialProfileParam
    :no-index:
.. autodataset:: sigima.params.SegmentProfileParam
    :no-index:
.. autoclass:: sigima.params.Direction
    :no-index:
.. autodataset:: sigima.params.ROIGridParam
    :no-index:

Filtering parameters
~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.ButterworthParam
    :no-index:
.. autofunction:: sigima.params.GaussianFreqFilterParam
    :no-index:

Fourier analysis parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.ZeroPadding2DParam
    :no-index:

Geometry parameters
~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.BinningParam
    :no-index:
.. autodataset:: sigima.params.ResizeParam
    :no-index:
.. autodataset:: sigima.params.RotateParam
    :no-index:
.. autodataset:: sigima.params.TranslateParam
    :no-index:

Mathematical operation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.params.DataTypeIParam
    :no-index:
.. autodataset:: sigima.params.Log10ZPlusNParam
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

__all__ = [
    "AbscissaParam",
    "AdjustGammaParam",
    "AdjustLogParam",
    "AdjustSigmoidParam",
    "AllanVarianceParam",
    "AngleUnitParam",
    "ArithmeticParam",
    "AverageProfileParam",
    "BandPassFilterParam",
    "BandStopFilterParam",
    "BinningParam",
    "BlobDOGParam",
    "BlobDOHParam",
    "BlobLOGParam",
    "BlobOpenCVParam",
    "ButterworthParam",
    "CannyParam",
    "ClipParam",
    "ConstantParam",
    "ContourShapeParam",
    "DataTypeIParam",
    "DataTypeSParam",
    "DenoiseBilateralParam",
    "DenoiseTVParam",
    "DenoiseWaveletParam",
    "DetrendingParam",
    "Direction",
    "DynamicParam",
    "EqualizeAdaptHistParam",
    "EqualizeHistParam",
    "FFTParam",
    "FWHMParam",
    "FlatFieldParam",
    "GaussianFreqFilterParam",
    "GaussianParam",
    "GridParam",
    "HighPassFilterParam",
    "HistogramParam",
    "HoughCircleParam",
    "InterpolationParam",
    "LineProfileParam",
    "Log10ZPlusNParam",
    "LowPassFilterParam",
    "MorphologyParam",
    "MovingAverageParam",
    "MovingMedianParam",
    "NormalizeParam",
    "OrdinateParam",
    "Peak2DDetectionParam",
    "PeakDetectionParam",
    "PhaseParam",
    "PolynomialFitParam",
    "PowerParam",
    "PulseFeaturesParam",
    "ROIGridParam",
    "RadialProfileParam",
    "Resampling1DParam",
    "Resampling2DParam",
    "RescaleIntensityParam",
    "ResizeParam",
    "RotateParam",
    "SaveToDirectoryParam",
    "SegmentProfileParam",
    "SignalsToImageParam",
    "SpectrumParam",
    "ThresholdParam",
    "TranslateParam",
    "UniformCoordsParam",
    "WindowingParam",
    "XYCalibrateParam",
    "XYZCalibrateParam",
    "ZeroPadding1DParam",
    "ZeroPadding2DParam",
]

from sigima.io.convenience import SaveToDirectoryParam
from sigima.proc.base import (
    AngleUnitParam,
    ArithmeticParam,
    ClipParam,
    ConstantParam,
    FFTParam,
    GaussianParam,
    HistogramParam,
    MovingAverageParam,
    MovingMedianParam,
    NormalizeParam,
    PhaseParam,
    SignalsToImageParam,
    SpectrumParam,
)
from sigima.proc.image import (
    AdjustGammaParam,
    AdjustLogParam,
    AdjustSigmoidParam,
    AverageProfileParam,
    BinningParam,
    BlobDOGParam,
    BlobDOHParam,
    BlobLOGParam,
    BlobOpenCVParam,
    ButterworthParam,
    CannyParam,
    ContourShapeParam,
    DataTypeIParam,
    DenoiseBilateralParam,
    DenoiseTVParam,
    DenoiseWaveletParam,
    Direction,
    EqualizeAdaptHistParam,
    EqualizeHistParam,
    FlatFieldParam,
    GaussianFreqFilterParam,
    GridParam,
    HoughCircleParam,
    LineProfileParam,
    Log10ZPlusNParam,
    MorphologyParam,
    Peak2DDetectionParam,
    RadialProfileParam,
    Resampling2DParam,
    RescaleIntensityParam,
    ResizeParam,
    ROIGridParam,
    RotateParam,
    SegmentProfileParam,
    ThresholdParam,
    TranslateParam,
    UniformCoordsParam,
    XYZCalibrateParam,
    ZeroPadding2DParam,
)
from sigima.proc.signal import (
    AbscissaParam,
    AllanVarianceParam,
    BandPassFilterParam,
    BandStopFilterParam,
    DataTypeSParam,
    DetrendingParam,
    DynamicParam,
    FWHMParam,
    HighPassFilterParam,
    InterpolationParam,
    LowPassFilterParam,
    OrdinateParam,
    PeakDetectionParam,
    PolynomialFitParam,
    PowerParam,
    PulseFeaturesParam,
    Resampling1DParam,
    WindowingParam,
    XYCalibrateParam,
    ZeroPadding1DParam,
)
