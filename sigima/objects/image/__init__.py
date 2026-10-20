# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image objects subpackage
========================

This subpackage provides image data structures and utilities.

The subpackage is organized into the following modules:

- `roi`: Region of Interest (ROI) classes and parameters
- `object`: Main ImageObj class for handling 2D image data
- `creation`: Image creation utilities and parameter classes

All classes and functions are re-exported at the subpackage level for backward
compatibility. Existing imports like `from sigima.objects.image import ImageObj`
will continue to work.
"""

# Import all public classes and functions from submodules
from .creation import (
    # Constants
    DEFAULT_TITLE,
    Checkerboard2DParam,
    DomainImageParam,
    Gauss2DParam,
    # Enums
    ImageDatatypes,
    ImageTypes,
    # Base parameter classes
    NewImageParam,
    NormalDistribution2DParam,
    PoissonDistribution2DParam,
    Ramp2DParam,
    Ring2DParam,
    SiemensStar2DParam,
    Sinc2DParam,
    SinusoidalGrating2DParam,
    UniformDistribution2DParam,
    # Specific parameter classes
    Zero2DParam,
    check_all_image_parameters_classes,
    # Factory and utility functions
    create_image,
    create_image_from_param,
    create_image_parameters,
    get_next_image_number,
    # Registration functions
    register_image_parameters_class,
)
from .object import (
    ImageObj,
)
from .roi import (
    # ROI classes
    BaseSingleImageROI,
    CircularROI,
    # Specific ROI types
    ImageROI,
    PolygonalROI,
    RectangularROI,
    ROI2DParam,
    # ROI utility function
    create_image_roi,
    create_image_roi_around_points,
)

# Define __all__ for explicit public API
__all__ = [
    "DEFAULT_TITLE",
    "BaseSingleImageROI",
    "Checkerboard2DParam",
    "CircularROI",
    "DomainImageParam",
    "Gauss2DParam",
    "ImageDatatypes",
    "ImageObj",
    "ImageROI",
    "ImageTypes",
    "NewImageParam",
    "NormalDistribution2DParam",
    "PoissonDistribution2DParam",
    "PolygonalROI",
    "ROI2DParam",
    "Ramp2DParam",
    "RectangularROI",
    "Ring2DParam",
    "SiemensStar2DParam",
    "Sinc2DParam",
    "SinusoidalGrating2DParam",
    "UniformDistribution2DParam",
    "Zero2DParam",
    "check_all_image_parameters_classes",
    "create_image",
    "create_image_from_param",
    "create_image_parameters",
    "create_image_roi",
    "create_image_roi_around_points",
    "get_next_image_number",
    "register_image_parameters_class",
]
