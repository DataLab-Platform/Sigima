# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal objects subpackage
=========================

This subpackage provides signal data structures and utilities.

The subpackage is organized into the following modules:

- `roi`: Region of Interest (ROI) classes and parameters for 1D signals
- `object`: Main SignalObj class for handling 1D signal data
- `creation`: Signal creation utilities and parameter classes

All classes and functions are re-exported at the subpackage level for backward
compatibility. Existing imports like `from sigima.objects.signal import SignalObj`
will continue to work.
"""

# Import all public classes and functions from submodules
from .creation import (
    # Constants
    CREATION_PARAMS_VERSION,
    DEFAULT_TITLE,
    PEAK_PARAMETERIZATION,
    # Mathematical function parameter classes
    BaseGaussLorentzVoigtParam,
    BasePeriodicParam,
    BasePulseParam,
    CosineParam,
    CustomSignalParam,
    # Pulse signal classes
    ExpectedFeatures,
    ExponentialParam,
    FeatureTolerances,
    # Periodic function parameter classes
    FreqUnits,
    GaussParam,
    # Other signal types
    LinearChirpParam,
    LogisticParam,
    LorentzParam,
    # Base parameter classes
    NewSignalParam,
    NormalDistribution1DParam,
    PlanckParam,
    PoissonDistribution1DParam,
    # Polynomial and custom signals
    PolyParam,
    PulseParam,
    SawtoothParam,
    # Enums
    SignalTypes,
    SincParam,
    SineParam,
    SquareParam,
    SquarePulseParam,
    StepParam,
    StepPulseParam,
    TriangleParam,
    UniformDistribution1DParam,
    VoigtParam,
    # Distribution parameter classes
    ZeroParam,
    check_all_signal_parameters_classes,
    convert_legacy_peak_creation_params,
    # Core creation functions
    create_signal,
    create_signal_from_param,
    # Factory and utility functions
    create_signal_parameters,
    get_next_signal_number,
    # Registration functions
    register_signal_parameters_class,
    triangle_func,
    validate_peak_creation_params,
)
from .object import (
    # Main signal class
    SignalObj,
)
from .roi import (
    # ROI classes
    ROI1DParam,
    SegmentROI,
    SignalROI,
    # ROI functions
    create_signal_roi,
)

# Define __all__ for explicit public API
__all__ = [
    "CREATION_PARAMS_VERSION",
    "DEFAULT_TITLE",
    "PEAK_PARAMETERIZATION",
    "BaseGaussLorentzVoigtParam",
    "BasePeriodicParam",
    "BasePulseParam",
    "CosineParam",
    "CustomSignalParam",
    "ExpectedFeatures",
    "ExponentialParam",
    "FeatureTolerances",
    "FreqUnits",
    "GaussParam",
    "LinearChirpParam",
    "LogisticParam",
    "LorentzParam",
    "NewSignalParam",
    "NormalDistribution1DParam",
    "PlanckParam",
    "PoissonDistribution1DParam",
    "PolyParam",
    "PulseParam",
    "ROI1DParam",
    "SawtoothParam",
    "SegmentROI",
    "SignalObj",
    "SignalROI",
    "SignalTypes",
    "SincParam",
    "SineParam",
    "SquareParam",
    "SquarePulseParam",
    "StepParam",
    "StepPulseParam",
    "TriangleParam",
    "UniformDistribution1DParam",
    "VoigtParam",
    "ZeroParam",
    "check_all_signal_parameters_classes",
    "convert_legacy_peak_creation_params",
    "create_signal",
    "create_signal_from_param",
    "create_signal_parameters",
    "create_signal_roi",
    "get_next_signal_number",
    "register_signal_parameters_class",
    "triangle_func",
    "validate_peak_creation_params",
]
