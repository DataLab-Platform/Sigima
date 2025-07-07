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

# The following comments are used to track the migration process of the `sigima`
# package, in the context of the DataLab Core Architecture Redesign project funded by
# the NLnet Foundation.

# -------- Point of no return after creating an independent `sigima` package ----------
# TODO: In `cdl` Python package, remove modifications related to the inclusion of the
#       `sigima` module within the `cdl` package (e.g., see TODOs in pyproject.toml,
#       VSCode tasks, Pylint configuration, etc.)
# TODO: Move `cdl.tests.sigima_tests` to external `sigima.tests` module
# TODO: Fix TODO related to `OPTIONS_RST` in 'sigima\config.py'
# ** Task 1. Core Architecture Redesign **
# **   Milestone 1.b. Decouple I/O features (including I/O plugins) **
# TODO: Implement a I/O plugin system similar to the `cdl.plugins` module
# **   Milestone 1.c. Redesign the API for the new core library **
#
# ** Task 2. Technical Validation and Testing **
# TODO: Add `pytest` infrastructure. Step 2: migrate `cdl/tests/sigima_tests`
#       to `sigima/tests` directory and create a `conftest.py` file using the
#       `cdl/tests/conftest.py` file as a template (see TODOs in that file).
# TODO: Handle the `CDL_DATA` environment variable in the `sigima` package
#       and its documentation (at the moment, it has been replaced by `SIGIMA_DATA`
#       in sigima\tests\helpers.py)
#
# ** Task 3. Documentation and Training Materials **
# TODO: Add documentation. Step 1: initiate `sigima` package documentation
# TODO: Add documentation. Step 2: migrate parts of `cdl` package documentation
#
# TODO: Migrate `cdlclient` features to a subpackage of `sigima` (e.g., `sigima.client`)
# --------------------------------------------------------------------------------------

# pylint:disable=unused-import
# flake8: noqa

from sigima.io import (
    read_images,
    read_signals,
    write_image,
    write_signal,
    read_image,
    read_signal,
)
from sigima.objects import (
    CircularROI,
    ExponentialParam,
    Gauss2DParam,
    GaussLorentzVoigtParam,
    ImageDatatypes,
    ImageROI,
    ImageObj,
    ImageTypes,
    NewImageParam,
    NewSignalParam,
    NormalRandomParam,
    PeriodicParam,
    PolygonalROI,
    RectangularROI,
    ResultProperties,
    ResultShape,
    ROI1DParam,
    ROI2DParam,
    SegmentROI,
    ShapeTypes,
    SignalObj,
    SignalROI,
    SignalTypes,
    StepParam,
    TypeObj,
    TypeROI,
    UniformRandomParam,
    create_image,
    create_image_from_param,
    create_image_roi,
    create_signal,
    create_signal_from_param,
    create_signal_roi,
)

__version__ = "0.2.0"
__docurl__ = __homeurl__ = "https://datalab-platform.com/"
__supporturl__ = "https://github.com/DataLab-Platform/sigima/issues/new/choose"

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
