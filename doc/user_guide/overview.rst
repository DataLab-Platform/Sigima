.. _overview:

Overview
========

What is Sigima?
---------------

**Sigima** is an open-source Python library for scientific signal and image processing. It provides a unified, object-oriented approach to working with 1D signals and 2D images, combining the power of NumPy arrays with rich metadata and processing capabilities.

Sigima was created from the externalization of `DataLab <https://datalab-platform.com/>`_'s computing engine and is now an essential building block of the DataLab ecosystem, but is designed to be GUI-independent (and independent of DataLab itself) and can be used in a variety of scientific computing contexts.

Quick Start Example
-------------------

Here's a simple example demonstrating image processing with Regions of Interest (ROIs) using Sigima. This example uses the matplotlib backend for visualization, making it web-friendly and easy to run in Jupyter notebooks.

See the complete interactive notebook in the :ref:`getting_started` section.

Why Sigima?
-----------

Scientific data processing often requires more than just raw numerical arrays. Real-world applications demand:

* **Context-aware data**: physical units, coordinate systems, and measurement uncertainties
* **Reproducible workflows**: clear separation between data and processing logic
* **Testable components**: modular functions that can be validated independently
* **Flexible integration**: ability to work both with high-level objects and low-level arrays

Sigima addresses these needs by providing a structured framework that bridges the gap between raw numerical operations and scientific data analysis. It is designed to be the processing backend for scientific applications, particularly those requiring headless execution or remote processing capabilities.

Signals in Sigima
-----------------

What are Signals?
^^^^^^^^^^^^^^^^^

In Sigima, a **signal** is a 1D data object represented by the :py:class:`~sigima.objects.signal.object.SignalObj` class. Unlike a simple NumPy array, a signal carries complete information about your measurement or calculation.

Core Components
^^^^^^^^^^^^^^^

A signal consists of:

**X and Y Data**
    The fundamental coordinate-value pairs that define your signal. The x-axis typically     represents the independent variable (time, frequency, position, etc.) while the y-axis     contains the measured or calculated values.

**Error Bars (dx, dy)**
    Sigima natively supports uncertainty quantification through optional error bars:

    * ``dx``: uncertainties in the x-coordinate (e.g., timing jitter, position uncertainty)
    * ``dy``: uncertainties in the y-values (e.g., measurement noise, statistical errors)

    These error bars propagate through many processing operations, allowing you to track how uncertainties evolve in your analysis pipeline.

**Metadata**
    A flexible dictionary for storing any additional information about your signal: acquisition parameters, processing history, physical constants, or any custom data your application needs.

**Units and Labels**
    Clear labeling of physical quantities:

    * ``xlabel`` / ``ylabel``: descriptive names (e.g., "Time", "Temperature")
    * ``xunit`` / ``yunit``: physical units (e.g., "s", "°C")

**Annotations**
    User-defined notes or comments about specific features or regions of the signal.

Why This Matters
^^^^^^^^^^^^^^^^

Consider a temperature measurement over time. With a simple NumPy array, you just have numbers. With a Sigima signal, you have:

.. code-block:: python

    import numpy as np
    from sigima.objects import create_signal

    # Time points with some uncertainty in timing
    time = np.linspace(0, 10, 100)
    time_uncertainty = np.full_like(time, 0.01)  # ±10 ms timing jitter

    # Temperature measurements with measurement error
    temperature = 20 + 5 * np.sin(time) + np.random.normal(0, 0.1, 100)
    temp_uncertainty = np.full_like(temperature, 0.1)  # ±0.1°C sensor noise

    # Create a signal with full context
    signal = create_signal(
        title="Temperature oscillation",
        x=time,
        y=temperature,
        dx=time_uncertainty,
        dy=temp_uncertainty,
        units=("s", "°C"),
        labels=("Time", "Temperature"),
    )

    # Metadata is automatically preserved through processing
    signal.metadata["sensor_id"] = "TC-01"
    signal.metadata["location"] = "Lab A"

This signal object can now be processed, analyzed, saved, and loaded while preserving all its context. When you apply operations like filtering, peak detection, or statistics, the units, labels, and error bars remain attached to your data.

Specialized Features
^^^^^^^^^^^^^^^^^^^^

Sigima signals support advanced use cases:

**Datetime Coordinates**
    For time-series data, x-coordinates can represent actual datetime values, with automatic handling of time units and formatting.

**ROI Operations**
    Define regions of interest (ROIs) to process or analyze specific portions of your signal independently.

**Scale Management**
    Control visualization aspects like logarithmic scales and axis bounds.

Images in Sigima
----------------

What are Images?
^^^^^^^^^^^^^^^^

An **image** in Sigima is a 2D data object represented by the :py:class:`~sigima.objects.image.object.ImageObj` class. Like signals, images extend beyond simple 2D NumPy arrays by providing comprehensive metadata and coordinate system support.

Core Components
^^^^^^^^^^^^^^^

An image consists of:

**2D Data Array**
    The pixel values forming your image. Sigima supports various data types including unsigned integers (uint8, uint16), signed integers (int16, int32), floating-point (float32, float64), and complex numbers (complex128).

**Coordinate Systems**
    Two types of coordinate systems are supported:

    * **Uniform coordinates**: regular pixel spacing, common for most imaging systems

      - ``dx``, ``dy``: pixel spacing in physical units
      - ``x0``, ``y0``: origin coordinates

    * **Non-uniform coordinates**: variable pixel spacing for adaptive sampling or irregular grids

      - ``xcoords``, ``ycoords``: explicit coordinate arrays

**Metadata**
    Rich metadata support including:

    * DICOM template integration for medical imaging
    * Custom key-value pairs for any application-specific data
    * Automatic extraction from common file formats

**Units and Labels**
    Physical quantities for all three dimensions:

    * ``xlabel``, ``ylabel``, ``zlabel``: axis descriptions
    * ``xunit``, ``yunit``, ``zunit``: physical units

    The z-axis typically represents the pixel values (intensity, height, temperature, etc.)

**Annotations**
    Structured annotations for documenting features, regions, or processing steps.

**Data Type Flexibility**
    Unlike signals (which are typically float64), images support multiple data types optimized for different use cases:

    * ``uint8``, ``uint16``: efficient storage for camera data, medical images
    * ``int16``, ``int32``: signed data, difference images
    * ``float32``, ``float64``: scientific measurements, calculated results
    * ``complex128``: Fourier transforms, wave simulations

Why This Matters
^^^^^^^^^^^^^^^^

Consider microscopy data. A raw 2D array gives you pixel values, but an ``ImageObj`` provides the complete picture:

.. code-block:: python

    import numpy as np
    from sigima.objects import create_image

    # Create a 512x512 microscopy image
    data = np.random.poisson(1000, (512, 512)).astype(np.uint16)

    image = create_image(
        title="Cell sample - 40x objective",
        data=data,
        units=("μm", "μm", "counts"),
        labels=("X position", "Y position", "Photon counts"),
    )

    # Set physical coordinate system
    # 0.16 μm per pixel, field of view starts at origin
    image.set_uniform_coords(dx=0.16, dy=0.16, x0=0.0, y0=0.0)

    # Add acquisition metadata
    image.metadata["objective"] = "40x"
    image.metadata["exposure_time"] = 100  # ms
    image.metadata["binning"] = "1x1"
    image.metadata["camera_gain"] = 2.5

Processing operations automatically work in physical units. Measurements return results in micrometers, not pixels. The metadata travels with your data through the entire analysis pipeline.

Specialized Features
^^^^^^^^^^^^^^^^^^^^

**Native Support for Multiple Image Formats**
    Sigima can directly read and write images from various scientific and common file formats, making it easy to integrate into existing workflows:

    * **Common formats**: BMP, JPEG, PNG, TIFF, JPEG 2000
    * **Scientific formats**: DICOM (medical imaging), Andor SIF (spectroscopy cameras)
    * **Data exchange**: NumPy (.npy), MATLAB (.mat), HDF5 (.h5img)
    * **Text-based**: CSV, TXT, ASC (with coordinate metadata support)
    * **Specialized**: Spiricon (.scor-data), Dürr NDT (.xyz), FT-Lab (.ima)

    Each format preserves relevant metadata when available, such as DICOM medical imaging headers or acquisition parameters from scientific instruments.

**ROI Operations**
    Define rectangular, circular, or polygonal regions of interest for localized analysis or processing.

**Masked Arrays**
    Support for NumPy masked arrays to handle invalid pixels, saturated regions, or excluded areas.

**Coordinate Conversion**
    Seamless conversion between pixel indices and physical coordinates, essential for multi-scale or multi-modal imaging.

Key Differences from Raw Arrays
--------------------------------

Here's why Sigima objects are more powerful than plain NumPy arrays in physical data processing:

.. list-table::
    :header-rows: 1
    :widths: 30 35 35

    * - Feature
      - NumPy Array
      - Sigima Object
    * - Data storage
      - Numbers only
      - Numbers + context
    * - Units
      - Implicit/external
      - Explicit and tracked
    * - Error propagation
      - Manual
      - Automatic (where supported)
    * - Metadata
      - Separate variables
      - Integrated dictionary
    * - Coordinate systems
      - Index-based only
      - Physical coordinates
    * - Serialization
      - Just data
      - Complete context
    * - Processing history
      - Lost
      - Can be tracked

Common Use Cases
----------------

Sigima excels in scenarios requiring:

**Scientific Instrumentation**
    Process data from sensors, cameras, spectrometers, or any measurement device while preserving calibration, units, and acquisition parameters.

**Automated Analysis Pipelines**
    Build reproducible workflows that maintain data provenance and context through multiple processing steps.

**Remote Processing**
    Execute signal and image processing on remote servers or in cloud environments without GUI dependencies.

**Application Backends**
    Provide the computation engine for scientific software, keeping processing logic independent from user interface concerns.

**Research and Development**
    Develop and validate new processing algorithms in a testable, modular framework.

GUI Connection
--------------
As the core computing engine externalized from `DataLab <https://datalab-platform.com/>`_, Sigima seamlessly integrates with it as an essential building block of the DataLab stack.
You can leverage Sigima's powerful processing features within DataLab's graphical interface, using for example Sigima to handle your signal and image acquisition and sending the results back to DataLab for visualization and further interactive analysis.

Getting Started
---------------

To start using Sigima, check out:

* :doc:`installation` - Set up Sigima in your environment
* :doc:`features` - Explore Sigima's key features and capabilities
* :doc:`../auto_examples/index` - Practical examples showing signals and images in action
* :doc:`../api/index` - Complete API reference

.. note::

   Sigima is designed to be **GUI-independent**. If you need interactive visualization and analysis tools, check out `DataLab <https://datalab-platform.com/>`_, which uses Sigima as its processing backend and adds a complete graphical interface.
