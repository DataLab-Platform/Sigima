# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima
======

Sigima is a scientific computing engine for 1D signals and 2D images.

It provides a set of tools for image and signal processing, including
denoising, segmentation, and restoration. It is designed to be used in
scientific and research applications.

It is a part of the DataLab Platform, which aims at providing a
comprehensive set of tools for data analysis and visualization, around
the DataLab application.
"""

# TODO: Use `numpy.typing.NDArray` for more precise type annotations once NumPy >= 1.21
# can be safely required (e.g. after raising the minimum required version of
# scikit-image to >= 0.19).

__all__ = [
    "NO_ROI",
    "Checkerboard2DParam",
    "CircularROI",
    "ExponentialParam",
    "Gauss2DParam",
    "GaussParam",
    "GeometryResult",
    "ImageDatatypes",
    "ImageObj",
    "ImageROI",
    "ImageTypes",
    "KindShape",
    "LinearChirpParam",
    "LogisticParam",
    "LorentzParam",
    "NormalDistribution1DParam",
    "NormalDistribution2DParam",
    "PlanckParam",
    "PolygonalROI",
    "ROI1DParam",
    "ROI2DParam",
    "Ramp2DParam",
    "RectangularROI",
    "Ring2DParam",
    "SegmentROI",
    "SiemensStar2DParam",
    "SignalObj",
    "SignalROI",
    "SignalTypes",
    "SimpleBaseProxy",
    "SimpleRemoteProxy",
    "Sinc2DParam",
    "SinusoidalGrating2DParam",
    "StepParam",
    "TableResult",
    "TypeObj",
    "TypeROI",
    "UniformDistribution1DParam",
    "UniformDistribution2DParam",
    "VoigtParam",
    "calc_table_from_data",
    "create_image",
    "create_image_from_param",
    "create_image_parameters",
    "create_image_roi",
    "create_image_roi_around_points",
    "create_signal",
    "create_signal_from_param",
    "create_signal_parameters",
    "create_signal_roi",
    "read_image",
    "read_images",
    "read_signal",
    "read_signals",
    "write_image",
    "write_signal",
]


from guidata.config import ValidationMode, set_validation_mode

from sigima.client import SimpleBaseProxy, SimpleRemoteProxy
from sigima.io import (
    read_image,
    read_images,
    read_signal,
    read_signals,
    write_image,
    write_signal,
)
from sigima.objects import (
    NO_ROI,
    Checkerboard2DParam,
    CircularROI,
    ExponentialParam,
    Gauss2DParam,
    GaussParam,
    GeometryResult,
    ImageDatatypes,
    ImageObj,
    ImageROI,
    ImageTypes,
    KindShape,
    LinearChirpParam,
    LogisticParam,
    LorentzParam,
    NormalDistribution1DParam,
    NormalDistribution2DParam,
    PlanckParam,
    PolygonalROI,
    Ramp2DParam,
    RectangularROI,
    Ring2DParam,
    ROI1DParam,
    ROI2DParam,
    SegmentROI,
    SiemensStar2DParam,
    SignalObj,
    SignalROI,
    SignalTypes,
    Sinc2DParam,
    SinusoidalGrating2DParam,
    StepParam,
    TableResult,
    TypeObj,
    TypeROI,
    UniformDistribution1DParam,
    UniformDistribution2DParam,
    VoigtParam,
    calc_table_from_data,
    create_image,
    create_image_from_param,
    create_image_parameters,
    create_image_roi,
    create_image_roi_around_points,
    create_signal,
    create_signal_from_param,
    create_signal_parameters,
    create_signal_roi,
)

# Set validation mode to ENABLED by default (issue warnings for invalid inputs)
set_validation_mode(ValidationMode.ENABLED)

__version__ = "1.0.3"
__docurl__ = "https://sigima.readthedocs.io/"
__homeurl__ = "https://github.com/DataLab-Platform/Sigima"
__supporturl__ = "https://github.com/DataLab-Platform/sigima/issues/new/choose"

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
