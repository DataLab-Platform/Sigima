# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image Processing Tools (:mod:`sigima.tools.image`)
--------------------------------------------------

This package contains image processing tools, which are organized into subpackages
according to their purpose:

- :mod:`sigima.tools.image.detection`: Object detection algorithms
  (blob detection, peak detection, contour fitting)
- :mod:`sigima.tools.image.exposure`: Exposure and level adjustment functions,
  including flat-field correction
- :mod:`sigima.tools.image.extraction`: Data extraction and analysis functions
  (radial profiles, feature extraction)
- :mod:`sigima.tools.image.fourier`: 2D Fourier transform operations and
  spectral analysis
- :mod:`sigima.tools.image.geometry`: Geometric analysis and transformations
- :mod:`sigima.tools.image.preprocessing`: Data preprocessing and transformation
  utilities (scaling, normalization, binning, padding)

All functions are re-exported at the subpackage level for backward
compatibility. Existing imports like ``from sigima.tools.image import fft2d``
will continue to work.
"""

from __future__ import annotations

# Import all functions from submodules for backward compatibility
from sigima.tools.image.detection import (
    find_blobs_dog,
    find_blobs_doh,
    find_blobs_log,
    find_blobs_opencv,
    get_2d_peaks_coords,
    get_contour_shapes,
    get_hough_circle_peaks,
    remove_overlapping_disks,
)
from sigima.tools.image.exposure import (
    adjust_brightness_contrast,
    brightness_contrast_context,
    brightness_contrast_ranges,
    flatfield,
    normalize,
)
from sigima.tools.image.extraction import get_radial_profile
from sigima.tools.image.fourier import (
    convolve,
    deconvolve,
    fft2d,
    gaussian_freq_filter,
    ifft2d,
    magnitude_spectrum,
    phase_spectrum,
    psd,
)
from sigima.tools.image.geometry import (
    get_centroid_auto,
    get_centroid_fourier,
    get_enclosing_circle,
    get_projected_profile_centroid,
)
from sigima.tools.image.preprocessing import (
    binning,
    distance_matrix,
    get_absolute_level,
    scale_data_to_min_max,
    zero_padding,
)

# Define __all__ to specify what gets imported with
# "from sigima.tools.image import *"
__all__ = [
    "adjust_brightness_contrast",
    "binning",
    "brightness_contrast_context",
    "brightness_contrast_ranges",
    "convolve",
    "deconvolve",
    "distance_matrix",
    "fft2d",
    "find_blobs_dog",
    "find_blobs_doh",
    "find_blobs_log",
    "find_blobs_opencv",
    "flatfield",
    "gaussian_freq_filter",
    "get_2d_peaks_coords",
    "get_absolute_level",
    "get_centroid_auto",
    "get_centroid_fourier",
    "get_contour_shapes",
    "get_enclosing_circle",
    "get_hough_circle_peaks",
    "get_projected_profile_centroid",
    "get_radial_profile",
    "ifft2d",
    "magnitude_spectrum",
    "normalize",
    "phase_spectrum",
    "psd",
    "remove_overlapping_disks",
    "scale_data_to_min_max",
    "zero_padding",
]
