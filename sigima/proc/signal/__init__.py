# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Basic signal processing
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.signal.base
   :members:
   :no-index:

Arithmetic
~~~~~~~~~~

.. automodule:: sigima.proc.signal.arithmetic
    :members:
    :no-index:

Mathematical Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.signal.mathops
    :members:
    :no-index:

Extraction
~~~~~~~~~~

.. automodule:: sigima.proc.signal.extraction
    :members:
    :no-index:

Filtering
~~~~~~~~~

.. automodule:: sigima.proc.signal.filtering
    :members:
    :no-index:

Processing
~~~~~~~~~~

.. automodule:: sigima.proc.signal.processing
    :members:
    :no-index:

Fourier
~~~~~~~

.. automodule:: sigima.proc.signal.fourier
    :members:
    :no-index:

Fitting
~~~~~~~

.. automodule:: sigima.proc.signal.fitting
    :members:
    :no-index:

Features
~~~~~~~~

.. automodule:: sigima.proc.signal.features
    :members:
    :no-index:

Stability
~~~~~~~~~

.. automodule:: sigima.proc.signal.stability
    :members:
    :no-index:

Analysis
~~~~~~~~

.. automodule:: sigima.proc.signal.analysis
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

# Import parameter classes from the main base module
from sigima.proc.signal.analysis import (
    PulseFeaturesParam,
    contrast,
    extract_pulse_features,
    histogram,
    sampling_rate_period,
    x_at_minmax,
)
from sigima.proc.signal.arithmetic import (
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
    signals_to_image,
    standard_deviation,
)
from sigima.proc.signal.base import (
    Wrap1to1Func,
    compute_geometry_from_obj,
    is_uncertainty_data_available,
    restore_data_outside_roi,
)
from sigima.proc.signal.extraction import (
    extract_roi,
    extract_rois,
)
from sigima.proc.signal.features import (
    AbscissaParam,
    DynamicParam,
    FWHMParam,
    OrdinateParam,
    PeakDetectionParam,
    bandwidth_3db,
    dynamic_parameters,
    full_width_at_y,
    fw1e2,
    fwhm,
    peak_detection,
    stats,
    x_at_y,
    y_at_x,
)
from sigima.proc.signal.filtering import (
    BandPassFilterParam,
    BandStopFilterParam,
    BaseHighLowBandParam,
    FrequencyFilterMethod,
    HighPassFilterParam,
    LowPassFilterParam,
    PadLocation1D,
    add_gaussian_noise,
    add_poisson_noise,
    add_uniform_noise,
    bandpass,
    bandstop,
    frequency_filter,
    gaussian_filter,
    get_nyquist_frequency,
    highpass,
    lowpass,
    moving_average,
    moving_median,
    wiener,
)
from sigima.proc.signal.fitting import (
    PolynomialFitParam,
    cdf_fit,
    evaluate_fit,
    exponential_fit,
    extract_fit_params,
    gaussian_fit,
    linear_fit,
    lorentzian_fit,
    piecewiseexponential_fit,
    planckian_fit,
    polynomial_fit,
    sigmoid_fit,
    sinusoidal_fit,
    twohalfgaussian_fit,
    voigt_fit,
)
from sigima.proc.signal.fourier import (
    ZeroPadding1DParam,
    fft,
    ifft,
    magnitude_spectrum,
    phase_spectrum,
    psd,
    zero_padding,
)
from sigima.proc.signal.mathops import (
    DataTypeSParam,
    PowerParam,
    absolute,
    astype,
    complex_from_magnitude_phase,
    complex_from_real_imag,
    exp,
    imag,
    inverse,
    log10,
    phase,
    power,
    real,
    sqrt,
    to_cartesian,
    to_polar,
    transpose,
)
from sigima.proc.signal.processing import (
    DetrendingParam,
    InterpolationParam,
    Resampling1DParam,
    WindowingParam,
    XYCalibrateParam,
    apply_window,
    calibration,
    check_same_sample_rate,
    clip,
    convolution,
    deconvolution,
    derivative,
    detrending,
    integral,
    interpolate,
    normalize,
    offset_correction,
    replace_x_by_other_y,
    resampling,
    reverse_x,
    xy_mode,
)
from sigima.proc.signal.stability import (
    AllanVarianceParam,
    allan_deviation,
    allan_variance,
    hadamard_variance,
    modified_allan_variance,
    overlapping_allan_variance,
    time_deviation,
    total_variance,
)

__all__ = [
    "AbscissaParam",
    "AllanVarianceParam",
    "BandPassFilterParam",
    "BandStopFilterParam",
    "BaseHighLowBandParam",
    "DataTypeSParam",
    "DetrendingParam",
    "DynamicParam",
    "FWHMParam",
    "FrequencyFilterMethod",
    "HighPassFilterParam",
    "InterpolationParam",
    "LowPassFilterParam",
    "OrdinateParam",
    "PadLocation1D",
    "PeakDetectionParam",
    "PolynomialFitParam",
    "PowerParam",
    "PulseFeaturesParam",
    "Resampling1DParam",
    "WindowingParam",
    "Wrap1to1Func",
    "XYCalibrateParam",
    "ZeroPadding1DParam",
    "absolute",
    "add_gaussian_noise",
    "add_poisson_noise",
    "add_uniform_noise",
    "addition",
    "addition_constant",
    "allan_deviation",
    "allan_variance",
    "apply_window",
    "arithmetic",
    "astype",
    "average",
    "bandpass",
    "bandstop",
    "bandwidth_3db",
    "calibration",
    "cdf_fit",
    "check_same_sample_rate",
    "clip",
    "complex_from_magnitude_phase",
    "complex_from_real_imag",
    "compute_geometry_from_obj",
    "contrast",
    "convolution",
    "deconvolution",
    "derivative",
    "detrending",
    "difference",
    "difference_constant",
    "division",
    "division_constant",
    "dynamic_parameters",
    "evaluate_fit",
    "exp",
    "exponential_fit",
    "extract_fit_params",
    "extract_pulse_features",
    "extract_roi",
    "extract_rois",
    "fft",
    "frequency_filter",
    "full_width_at_y",
    "fw1e2",
    "fwhm",
    "gaussian_filter",
    "gaussian_fit",
    "get_nyquist_frequency",
    "hadamard_variance",
    "highpass",
    "histogram",
    "ifft",
    "imag",
    "integral",
    "interpolate",
    "inverse",
    "is_uncertainty_data_available",
    "linear_fit",
    "log10",
    "lorentzian_fit",
    "lowpass",
    "magnitude_spectrum",
    "modified_allan_variance",
    "moving_average",
    "moving_median",
    "normalize",
    "offset_correction",
    "overlapping_allan_variance",
    "peak_detection",
    "phase",
    "phase_spectrum",
    "piecewiseexponential_fit",
    "planckian_fit",
    "polynomial_fit",
    "power",
    "product",
    "product_constant",
    "psd",
    "quadratic_difference",
    "real",
    "replace_x_by_other_y",
    "resampling",
    "restore_data_outside_roi",
    "reverse_x",
    "sampling_rate_period",
    "sigmoid_fit",
    "signals_to_image",
    "sinusoidal_fit",
    "sqrt",
    "standard_deviation",
    "stats",
    "time_deviation",
    "to_cartesian",
    "to_polar",
    "total_variance",
    "transpose",
    "twohalfgaussian_fit",
    "voigt_fit",
    "wiener",
    "x_at_minmax",
    "x_at_y",
    "xy_mode",
    "y_at_x",
    "zero_padding",
]
