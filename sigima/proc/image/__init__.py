# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Basic image processing
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image.base
    :members:
    :no-index:

Arithmetic
~~~~~~~~~~

.. automodule:: sigima.proc.image.arithmetic
    :members:
    :no-index:

Mathematical Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image.mathops
    :members:
    :no-index:

Measurements
~~~~~~~~~~~~

.. automodule:: sigima.proc.image.measurement
    :members:
    :no-index:

Extraction
~~~~~~~~~~

.. automodule:: sigima.proc.image.extraction
    :members:
    :no-index:

Filtering
~~~~~~~~~

.. automodule:: sigima.proc.image.filtering
    :members:
    :no-index:

Fourier
~~~~~~~

.. automodule:: sigima.proc.image.fourier
    :members:
    :no-index:

Thresholding
~~~~~~~~~~~~

.. automodule:: sigima.proc.image.threshold
    :members:
    :no-index:

Exposure correction
~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image.exposure
    :members:
    :no-index:

Preprocessing
~~~~~~~~~~~~~

.. automodule:: sigima.proc.image.preprocessing
    :members:
    :no-index:

Restoration
~~~~~~~~~~~

.. automodule:: sigima.proc.image.restoration
    :members:
    :no-index:

Noise
~~~~~

.. automodule:: sigima.proc.image.noise
    :members:
    :no-index:

Morphology
~~~~~~~~~~

.. automodule:: sigima.proc.image.morphology
    :members:
    :no-index:

Edge detection
~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image.edges
    :members:
    :no-index:

Detection
~~~~~~~~~

.. automodule:: sigima.proc.image.detection
    :members:
    :no-index:

Geometry
~~~~~~~~

.. automodule:: sigima.proc.image.geometry
    :members:
    :no-index:

Transformation features
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.proc.image.transformations
    :members:
    :no-index:
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# MARK: Important notes
# ---------------------
# - All `guidata.dataset.DataSet` classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` defined in the other modules
#   of this package must be imported right here.

from __future__ import annotations

import guidata.dataset as gds

from sigima.config import _
from sigima.enums import PadLocation2D
from sigima.proc.image.arithmetic import (
    addition,
    addition_constant,
    arithmetic,
    average,
    difference,
    difference_constant,
    division,
    division_constant,
    product,
    product_constant,
    quadratic_difference,
    standard_deviation,
)
from sigima.proc.image.base import (
    Wrap1to1Func,
    compute_geometry_from_obj,
    dst_1_to_1_signal,
    restore_data_outside_roi,
)
from sigima.proc.image.detection import (
    BlobDOGParam,
    BlobDOHParam,
    BlobLOGParam,
    BlobOpenCVParam,
    ContourShapeParam,
    HoughCircleParam,
    Peak2DDetectionParam,
    apply_detection_rois,
    blob_dog,
    blob_doh,
    blob_log,
    blob_opencv,
    contour_shape,
    hough_circle_peaks,
    peak_detection,
    store_contour_roi_metadata,
)
from sigima.proc.image.edges import (
    CannyParam,
    canny,
    farid,
    farid_h,
    farid_v,
    laplace,
    prewitt,
    prewitt_h,
    prewitt_v,
    roberts,
    scharr,
    scharr_h,
    scharr_v,
    sobel,
    sobel_h,
    sobel_v,
)
from sigima.proc.image.exposure import (
    AdjustGammaParam,
    AdjustLogParam,
    AdjustSigmoidParam,
    EqualizeAdaptHistParam,
    EqualizeHistParam,
    FlatFieldParam,
    NormalizeParam,
    RescaleIntensityParam,
    adjust_gamma,
    adjust_log,
    adjust_sigmoid,
    clip,
    equalize_adapthist,
    equalize_hist,
    flatfield,
    histogram,
    normalize,
    offset_correction,
    replace_special_values,
    rescale_intensity,
)
from sigima.proc.image.extraction import (
    AverageProfileParam,
    Direction,
    LineProfileParam,
    RadialProfileParam,
    ROIGridParam,
    SegmentProfileParam,
    average_profile,
    extract_roi,
    extract_rois,
    generate_image_grid_roi,
    line_profile,
    radial_profile,
    segment_profile,
)
from sigima.proc.image.filtering import (
    ButterworthParam,
    GaussianFreqFilterParam,
    butterworth,
    gaussian_filter,
    gaussian_freq_filter,
    moving_average,
    moving_median,
    wiener,
)
from sigima.proc.image.fourier import (
    fft,
    ifft,
    magnitude_spectrum,
    phase_spectrum,
    psd,
)
from sigima.proc.image.geometry import (
    Resampling2DParam,
    ResizeParam,
    RotateParam,
    TranslateParam,
    UniformCoordsParam,
    XYZCalibrateParam,
    calibration,
    fliph,
    flipv,
    resampling,
    resize,
    rotate,
    rotate90,
    rotate270,
    set_uniform_coords,
    translate,
    transpose,
)
from sigima.proc.image.mathops import (
    DataTypeIParam,
    Log10ZPlusNParam,
    absolute,
    astype,
    complex_from_magnitude_phase,
    complex_from_real_imag,
    convolution,
    deconvolution,
    exp,
    imag,
    inverse,
    log10,
    log10_z_plus_n,
    phase,
    real,
)
from sigima.proc.image.measurement import (
    centroid,
    enclosing_circle,
    horizontal_projection,
    stats,
    vertical_projection,
)
from sigima.proc.image.morphology import (
    MorphologyParam,
    black_tophat,
    closing,
    dilation,
    erosion,
    opening,
    white_tophat,
)
from sigima.proc.image.noise import (
    add_gaussian_noise,
    add_poisson_noise,
    add_uniform_noise,
)
from sigima.proc.image.preprocessing import (
    BinningParam,
    ZeroPadding2DParam,
    binning,
    zero_padding,
)
from sigima.proc.image.restoration import (
    DenoiseBilateralParam,
    DenoiseTVParam,
    DenoiseWaveletParam,
    denoise_bilateral,
    denoise_tophat,
    denoise_tv,
    denoise_wavelet,
    erase,
)
from sigima.proc.image.threshold import (
    ThresholdParam,
    threshold,
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from sigima.proc.image.transformations import transformer

__all__ = [
    "AdjustGammaParam",
    "AdjustLogParam",
    "AdjustSigmoidParam",
    "AverageProfileParam",
    "BinningParam",
    "BlobDOGParam",
    "BlobDOHParam",
    "BlobLOGParam",
    "BlobOpenCVParam",
    "ButterworthParam",
    "CannyParam",
    "ContourShapeParam",
    "DataTypeIParam",
    "DenoiseBilateralParam",
    "DenoiseTVParam",
    "DenoiseWaveletParam",
    "Direction",
    "EqualizeAdaptHistParam",
    "EqualizeHistParam",
    "FlatFieldParam",
    "GaussianFreqFilterParam",
    "GridParam",
    "HoughCircleParam",
    "LineProfileParam",
    "Log10ZPlusNParam",
    "MorphologyParam",
    "NormalizeParam",
    "PadLocation2D",
    "Peak2DDetectionParam",
    "ROIGridParam",
    "RadialProfileParam",
    "Resampling2DParam",
    "RescaleIntensityParam",
    "ResizeParam",
    "RotateParam",
    "SegmentProfileParam",
    "ThresholdParam",
    "TranslateParam",
    "UniformCoordsParam",
    "Wrap1to1Func",
    "XYZCalibrateParam",
    "ZeroPadding2DParam",
    "absolute",
    "add_gaussian_noise",
    "add_poisson_noise",
    "add_uniform_noise",
    "addition",
    "addition_constant",
    "adjust_gamma",
    "adjust_log",
    "adjust_sigmoid",
    "apply_detection_rois",
    "arithmetic",
    "astype",
    "average",
    "average_profile",
    "binning",
    "black_tophat",
    "blob_dog",
    "blob_doh",
    "blob_log",
    "blob_opencv",
    "butterworth",
    "calibration",
    "canny",
    "centroid",
    "clip",
    "closing",
    "complex_from_magnitude_phase",
    "complex_from_real_imag",
    "compute_geometry_from_obj",
    "contour_shape",
    "convolution",
    "deconvolution",
    "denoise_bilateral",
    "denoise_tophat",
    "denoise_tv",
    "denoise_wavelet",
    "difference",
    "difference_constant",
    "dilation",
    "division",
    "division_constant",
    "dst_1_to_1_signal",
    "enclosing_circle",
    "equalize_adapthist",
    "equalize_hist",
    "erase",
    "erosion",
    "exp",
    "extract_roi",
    "extract_rois",
    "farid",
    "farid_h",
    "farid_v",
    "fft",
    "flatfield",
    "fliph",
    "flipv",
    "gaussian_filter",
    "gaussian_freq_filter",
    "generate_image_grid_roi",
    "histogram",
    "horizontal_projection",
    "hough_circle_peaks",
    "ifft",
    "imag",
    "inverse",
    "laplace",
    "line_profile",
    "log10",
    "log10_z_plus_n",
    "magnitude_spectrum",
    "moving_average",
    "moving_median",
    "normalize",
    "offset_correction",
    "opening",
    "peak_detection",
    "phase",
    "phase_spectrum",
    "prewitt",
    "prewitt_h",
    "prewitt_v",
    "product",
    "product_constant",
    "psd",
    "quadratic_difference",
    "radial_profile",
    "real",
    "replace_special_values",
    "resampling",
    "rescale_intensity",
    "resize",
    "restore_data_outside_roi",
    "roberts",
    "rotate",
    "rotate90",
    "rotate270",
    "scharr",
    "scharr_h",
    "scharr_v",
    "segment_profile",
    "set_uniform_coords",
    "sobel",
    "sobel_h",
    "sobel_v",
    "standard_deviation",
    "stats",
    "store_contour_roi_metadata",
    "threshold",
    "threshold_isodata",
    "threshold_li",
    "threshold_mean",
    "threshold_minimum",
    "threshold_otsu",
    "threshold_triangle",
    "threshold_yen",
    "transformer",
    "translate",
    "transpose",
    "vertical_projection",
    "white_tophat",
    "wiener",
    "zero_padding",
]


class GridParam(gds.DataSet):
    """Grid parameters"""

    _prop = gds.GetAttrProp("direction")
    _directions = (("col", _("columns")), ("row", _("rows")))
    direction = gds.ChoiceItem(_("Distribute over"), _directions, radio=True).set_prop(
        "display", store=_prop
    )
    cols = gds.IntItem(_("Columns"), default=1, nonzero=True).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "col")
    )
    rows = gds.IntItem(_("Rows"), default=1, nonzero=True).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "row")
    )
    colspac = gds.FloatItem(_("Column spacing"), default=0.0, min=0.0)
    rowspac = gds.FloatItem(_("Row spacing"), default=0.0, min=0.0)
