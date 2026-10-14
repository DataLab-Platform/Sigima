# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O (:mod:`sigima.io`)
-----------------------

This package provides input/output functionality for reading and writing
signals and images in various formats. It includes a registry for managing
the available formats and their associated read/write functions.

General purpose I/O functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package provides functions to read and write signals and images, allowing users
to easily handle different file formats without needing to know the specifics
of each format.

It includes the following main functions:

- :py:func:`read_signals`: Read a list of signals from a file.
- :py:func:`read_signal`: Read a single signal from a file.
- :py:func:`write_signal`: Write a single signal to a file.
- :py:func:`write_signals`: Write a list of signals to a directory.
- :py:func:`read_images`: Read a list of images from a file.
- :py:func:`read_image`: Read a single image from a file.
- :py:func:`write_image`: Write a single image to a file.
- :py:func:`write_images`: Write a list of images to a directory.

.. autofunction:: read_signals
.. autofunction:: read_signal
.. autofunction:: write_signal
.. autofunction:: write_signals
.. autofunction:: read_images
.. autofunction:: read_image
.. autofunction:: write_image
.. autofunction:: write_images

RAW image import
^^^^^^^^^^^^^^^^

Headerless RAW files (``*.raw``) do not describe their binary layout. They are read
by passing a :py:class:`RawImageImportParam` instance to :py:func:`read_images` or
:py:func:`read_image`. The parameters define the pixel data type, the image width
and height, the offset to the first image, the number of images, the gap between
consecutive images and the byte order. When ``white_is_zero`` is set, pixel values
are left unchanged and the flag is stored in the image metadata, so that
applications can invert the display. Writing RAW files is not supported.

.. code-block:: python

    from sigima.io import RawImageImportParam, read_images

    param = RawImageImportParam.create(
        dtype="uint16", width=640, height=480, count=4, gap=128
    )
    images = read_images("acquisition.raw", param=param)

Format-aware image export
^^^^^^^^^^^^^^^^^^^^^^^^^

Passing an :py:class:`ImageExportParam` instance to :py:func:`write_image` prepares
the image data for the destination format before writing it: normalization (min-max,
percentile or manual bounds), clipping or rescaling of out-of-range values, handling
of non-finite values, gamma correction, inversion and conversion to a data type
supported by the format. The source image is never modified.

Format-specific writer options (compression, quality, tiling, resolution, delimiter,
precision, etc.) are given in the ``format_options`` dictionary. They are validated
against the capabilities of the destination format, which are described by
:py:func:`get_image_export_capabilities`. :py:func:`prepare_image_export_preview`
returns the data as it will be read back from the exported file (encoded and decoded
in memory for BMP, PNG, JPEG, JPEG 2000 and TIFF files), which is useful to check
the effect of quantization or lossy compression before writing.

.. code-block:: python

    from sigima.io import ImageExportParam, write_image

    param = ImageExportParam()
    param.normalization = "percentile"
    param.target_dtype = "uint8"
    param.format_options = {"compress_level": 9}
    write_image("result.png", image, param)

.. autofunction:: get_image_export_capabilities
.. autofunction:: get_supported_export_dtypes
.. autofunction:: prepare_image_for_export
.. autofunction:: prepare_image_export_preview
.. autofunction:: validate_image_export_options
.. autofunction:: validate_image_export_configuration
.. autoclass:: ImageExportCapabilities
.. autoclass:: ImageExportOptionSpec

Errors
^^^^^^

.. autoexception:: UnsupportedFormatOperationError

Supported formats
^^^^^^^^^^^^^^^^^

.. autodata:: SIGNAL_FORMAT_INFO

.. autodata:: IMAGE_FORMAT_INFO

Adding new formats
^^^^^^^^^^^^^^^^^^

To add new formats, you can create a new class that inherits from
:py:class:`sigima.io.image.base.ImageFormatBase` or
:py:class:`sigima.io.signal.base.SignalFormatBase` and implement the required methods.

.. note::

    Thanks to the plugin system, you can add new formats simply by defining a new class
    in a separate module, and it will be automatically discovered and registered, as
    long as it is imported in your application or library.

Example of a new image format plugin:

.. code-block:: python

    from sigima.io.image.base import ImageFormatBase
    from sigima.io.base import FormatInfo

    class MyImageFormat(ImageFormatBase):
        \"\"\"Object representing MyImageFormat image file type\"\"\"

        FORMAT_INFO = FormatInfo(
            name="MyImageFormat",
            extensions="*.myimg",
            readable=True,
            writeable=False,
        )

        @staticmethod
        def read_data(filename: str) -> np.ndarray:
            \"\"\"Read data and return it

            Args:
                filename (str): path to MyImageFormat file

            Returns:
                np.ndarray: image data
            \"\"\"
            # Implement reading logic here
            pass
"""

from __future__ import annotations

from sigima.io.base import UnsupportedFormatOperationError
from sigima.io.common.objmeta import (
    read_annotations,
    read_graphical_annotations,
    read_metadata,
    read_roi,
    read_roi_grid,
    write_annotations,
    write_graphical_annotations,
    write_metadata,
    write_roi,
    write_roi_grid,
)
from sigima.io.convenience import (
    read_image,
    read_images,
    read_signal,
    read_signals,
    write_image,
    write_images,
    write_signal,
    write_signals,
)
from sigima.io.image import (
    IMAGE_EXPORT_CAPABILITIES,
    ImageExportCapabilities,
    ImageExportOptionKind,
    ImageExportOptionSpec,
    ImageExportParam,
    ImageIORegistry,
    RawImageImportParam,
    encode_image_export_data,
    get_image_export_capabilities,
    get_image_export_writer_kwargs,
    get_supported_export_dtypes,
    prepare_image_export_preview,
    prepare_image_for_export,
    validate_image_export_configuration,
    validate_image_export_options,
    write_image_export_data,
)
from sigima.io.signal.base import SignalIORegistry

__all__ = [
    "IMAGE_EXPORT_CAPABILITIES",
    "IMAGE_FORMAT_INFO",
    "SIGNAL_FORMAT_INFO",
    "ImageExportCapabilities",
    "ImageExportOptionKind",
    "ImageExportOptionSpec",
    "ImageExportParam",
    "ImageIORegistry",
    "RawImageImportParam",
    "SignalIORegistry",
    "UnsupportedFormatOperationError",
    "encode_image_export_data",
    "get_image_export_capabilities",
    "get_image_export_writer_kwargs",
    "get_supported_export_dtypes",
    "prepare_image_export_preview",
    "prepare_image_for_export",
    "read_annotations",
    "read_graphical_annotations",
    "read_image",
    "read_images",
    "read_metadata",
    "read_roi",
    "read_roi_grid",
    "read_signal",
    "read_signals",
    "validate_image_export_configuration",
    "validate_image_export_options",
    "write_annotations",
    "write_graphical_annotations",
    "write_image",
    "write_image_export_data",
    "write_images",
    "write_metadata",
    "write_roi",
    "write_roi_grid",
    "write_signal",
    "write_signals",
]

SIGNAL_FORMAT_INFO = SignalIORegistry.get_format_info(mode="rst")
IMAGE_FORMAT_INFO = ImageIORegistry.get_format_info(mode="rst")
