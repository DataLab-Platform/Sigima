.. _features:

Features
========

Sigima provides a comprehensive suite of signal and image processing computational features organized into logical categories. This page provides an organized overview of all available computation functions with direct links to their API documentation.

.. note::
   All computation functions are available in the :mod:`sigima.proc` module. The :mod:`sigima.tools` modules provide lower-level utility functions used internally by the proc functions.

Common Operations
-----------------

These operations are available for both signals and images with similar functionality.

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`addition <sigima.proc.signal.addition>`
     - :func:`addition <sigima.proc.image.addition>`
     - Add two signals/images
   * - :func:`difference <sigima.proc.signal.difference>`
     - :func:`difference <sigima.proc.image.difference>`
     - Subtract one signal/image from another
   * - :func:`product <sigima.proc.signal.product>`
     - :func:`product <sigima.proc.image.product>`
     - Multiply two signals/images
   * - :func:`division <sigima.proc.signal.division>`
     - :func:`division <sigima.proc.image.division>`
     - Divide one signal/image by another
   * - :func:`average <sigima.proc.signal.average>`
     - :func:`average <sigima.proc.image.average>`
     - Compute average of multiple signals/images
   * - :func:`standard_deviation <sigima.proc.signal.standard_deviation>`
     - :func:`standard_deviation <sigima.proc.image.standard_deviation>`
     - Compute standard deviation of multiple signals/images
   * - :func:`quadratic_difference <sigima.proc.signal.quadratic_difference>`
     - :func:`quadratic_difference <sigima.proc.image.quadratic_difference>`
     - Compute quadratic difference between signals/images
   * - :func:`arithmetic <sigima.proc.signal.arithmetic>`
     - :func:`arithmetic <sigima.proc.image.arithmetic>`
     - Generic arithmetic operations with parameters

Constant Operations
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`addition_constant <sigima.proc.signal.addition_constant>`
     - :func:`addition_constant <sigima.proc.image.addition_constant>`
     - Add a constant value to signal/image
   * - :func:`difference_constant <sigima.proc.signal.difference_constant>`
     - :func:`difference_constant <sigima.proc.image.difference_constant>`
     - Subtract a constant from signal/image
   * - :func:`product_constant <sigima.proc.signal.product_constant>`
     - :func:`product_constant <sigima.proc.image.product_constant>`
     - Multiply signal/image by a constant
   * - :func:`division_constant <sigima.proc.signal.division_constant>`
     - :func:`division_constant <sigima.proc.image.division_constant>`
     - Divide signal/image by a constant

Mathematical Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`absolute <sigima.proc.signal.absolute>`
     - :func:`absolute <sigima.proc.image.absolute>`
     - Compute absolute value
   * - :func:`exp <sigima.proc.signal.exp>`
     - :func:`exp <sigima.proc.image.exp>`
     - Exponential function
   * - :func:`log10 <sigima.proc.signal.log10>`
     - :func:`log10 <sigima.proc.image.log10>`
     - Base-10 logarithm
   * - :func:`inverse <sigima.proc.signal.inverse>`
     - :func:`inverse <sigima.proc.image.inverse>`
     - Compute reciprocal
   * - :func:`real <sigima.proc.signal.real>`
     - :func:`real <sigima.proc.image.real>`
     - Extract real part of complex data
   * - :func:`imag <sigima.proc.signal.imag>`
     - :func:`imag <sigima.proc.image.imag>`
     - Extract imaginary part of complex data
   * - :func:`phase <sigima.proc.signal.phase>`
     - :func:`phase <sigima.proc.image.phase>`
     - Extract phase of complex data
   * - :func:`astype <sigima.proc.signal.astype>`
     - :func:`astype <sigima.proc.image.astype>`
     - Convert data type
   * - :func:`transpose <sigima.proc.signal.transpose>`
     - :func:`transpose <sigima.proc.image.transpose>`
     - Transpose coordinates/axes
   * - N/A
     - :func:`log10_z_plus_n <sigima.proc.image.log10_z_plus_n>`
     - Log10 with offset for zero handling

Signal-Specific Mathematical Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Function
     - Description
   * - :func:`sqrt <sigima.proc.signal.sqrt>`
     - Square root
   * - :func:`power <sigima.proc.signal.power>`
     - Raise to power
   * - :func:`to_cartesian <sigima.proc.signal.to_cartesian>`
     - Convert to Cartesian coordinates
   * - :func:`to_polar <sigima.proc.signal.to_polar>`
     - Convert to polar coordinates

Complex Number Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`complex_from_real_imag <sigima.proc.signal.complex_from_real_imag>`
     - :func:`complex_from_real_imag <sigima.proc.image.complex_from_real_imag>`
     - Create complex data from real and imaginary parts
   * - :func:`complex_from_magnitude_phase <sigima.proc.signal.complex_from_magnitude_phase>`
     - :func:`complex_from_magnitude_phase <sigima.proc.image.complex_from_magnitude_phase>`
     - Create complex data from magnitude and phase

Fourier Analysis
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`fft <sigima.proc.signal.fft>`
     - :func:`fft <sigima.proc.image.fft>`
     - Fast Fourier Transform (1D/2D)
   * - :func:`ifft <sigima.proc.signal.ifft>`
     - :func:`ifft <sigima.proc.image.ifft>`
     - Inverse Fast Fourier Transform (1D/2D)
   * - :func:`magnitude_spectrum <sigima.proc.signal.magnitude_spectrum>`
     - :func:`magnitude_spectrum <sigima.proc.image.magnitude_spectrum>`
     - Magnitude spectrum
   * - :func:`phase_spectrum <sigima.proc.signal.phase_spectrum>`
     - :func:`phase_spectrum <sigima.proc.image.phase_spectrum>`
     - Phase spectrum
   * - :func:`psd <sigima.proc.signal.psd>`
     - :func:`psd <sigima.proc.image.psd>`
     - Power spectral density

Convolution Operations
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`convolution <sigima.proc.signal.convolution>`
     - :func:`convolution <sigima.proc.image.convolution>`
     - Convolution operation
   * - :func:`deconvolution <sigima.proc.signal.deconvolution>`
     - :func:`deconvolution <sigima.proc.image.deconvolution>`
     - Deconvolution operation

Filtering
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`gaussian_filter <sigima.proc.signal.gaussian_filter>`
     - :func:`gaussian_filter <sigima.proc.image.gaussian_filter>`
     - Gaussian smoothing filter
   * - :func:`moving_average <sigima.proc.signal.moving_average>`
     - :func:`moving_average <sigima.proc.image.moving_average>`
     - Moving average filter
   * - :func:`moving_median <sigima.proc.signal.moving_median>`
     - :func:`moving_median <sigima.proc.image.moving_median>`
     - Moving median filter
   * - :func:`wiener <sigima.proc.signal.wiener>`
     - :func:`wiener <sigima.proc.image.wiener>`
     - Wiener filter

Noise Addition
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`add_gaussian_noise <sigima.proc.signal.add_gaussian_noise>`
     - :func:`add_gaussian_noise <sigima.proc.image.add_gaussian_noise>`
     - Add Gaussian noise for testing
   * - :func:`add_poisson_noise <sigima.proc.signal.add_poisson_noise>`
     - :func:`add_poisson_noise <sigima.proc.image.add_poisson_noise>`
     - Add Poisson noise for testing
   * - :func:`add_uniform_noise <sigima.proc.signal.add_uniform_noise>`
     - :func:`add_uniform_noise <sigima.proc.image.add_uniform_noise>`
     - Add uniform noise for testing

Signal Conditioning
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`normalize <sigima.proc.signal.normalize>`
     - :func:`normalize <sigima.proc.image.normalize>`
     - Normalize amplitude/intensity values
   * - :func:`clip <sigima.proc.signal.clip>`
     - :func:`clip <sigima.proc.image.clip>`
     - Clip values to specified range
   * - :func:`offset_correction <sigima.proc.signal.offset_correction>`
     - :func:`offset_correction <sigima.proc.image.offset_correction>`
     - Remove DC offset/background

Region of Interest (ROI)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`extract_roi <sigima.proc.signal.extract_roi>`
     - :func:`extract_roi <sigima.proc.image.extract_roi>`
     - Extract single ROI from data
   * - :func:`extract_rois <sigima.proc.signal.extract_rois>`
     - :func:`extract_rois <sigima.proc.image.extract_rois>`
     - Extract multiple ROIs from data

Calibration
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Signal Functions
     - Image Functions
     - Description
   * - :func:`calibration <sigima.proc.signal.calibration>`
     - :func:`calibration <sigima.proc.image.calibration>`
     - Apply coordinate system calibration

Signal Processing
-----------------

Array Operations
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`signals_to_image <sigima.proc.signal.signals_to_image>`
     - Convert signals to image representation

Frequency Filtering
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`lowpass <sigima.proc.signal.lowpass>`
     - Low-pass filter
   * - :func:`highpass <sigima.proc.signal.highpass>`
     - High-pass filter
   * - :func:`bandpass <sigima.proc.signal.bandpass>`
     - Band-pass filter
   * - :func:`bandstop <sigima.proc.signal.bandstop>`
     - Band-stop filter
   * - :func:`frequency_filter <sigima.proc.signal.frequency_filter>`
     - Generic frequency filtering

Signal Conditioning & Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`detrending <sigima.proc.signal.detrending>`
     - Remove linear trend
   * - :func:`apply_window <sigima.proc.signal.apply_window>`
     - Apply windowing function
   * - :func:`resampling <sigima.proc.signal.resampling>`
     - Resample signal
   * - :func:`interpolate <sigima.proc.signal.interpolate>`
     - Interpolate signal
   * - :func:`reverse_x <sigima.proc.signal.reverse_x>`
     - Reverse x-axis order
   * - :func:`xy_mode <sigima.proc.signal.xy_mode>`
     - Handle XY coordinate modes
   * - :func:`replace_x_by_other_y <sigima.proc.signal.replace_x_by_other_y>`
     - Replace X axis using another signal's Y values
   * - :func:`zero_padding <sigima.proc.signal.zero_padding>`
     - Zero-padding for signals

Signal Analysis
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`stats <sigima.proc.signal.stats>`
     - Statistical measurements
   * - :func:`histogram <sigima.proc.signal.histogram>`
     - Compute signal histogram
   * - :func:`contrast <sigima.proc.signal.contrast>`
     - Compute signal contrast
   * - :func:`derivative <sigima.proc.signal.derivative>`
     - Compute signal derivative
   * - :func:`integral <sigima.proc.signal.integral>`
     - Compute signal integral
   * - :func:`sampling_rate_period <sigima.proc.signal.sampling_rate_period>`
     - Analyze sampling rate and period

Peak and Feature Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`peak_detection <sigima.proc.signal.peak_detection>`
     - Peak detection in signals
   * - :func:`bandwidth_3db <sigima.proc.signal.bandwidth_3db>`
     - 3dB bandwidth measurement
   * - :func:`fwhm <sigima.proc.signal.fwhm>`
     - Full Width at Half Maximum
   * - :func:`fw1e2 <sigima.proc.signal.fw1e2>`
     - Full Width at 1/e²
   * - :func:`full_width_at_y <sigima.proc.signal.full_width_at_y>`
     - Full width at custom level
   * - :func:`dynamic_parameters <sigima.proc.signal.dynamic_parameters>`
     - Dynamic range parameters
   * - :func:`x_at_y <sigima.proc.signal.x_at_y>`
     - Find x-coordinates at given y-value
   * - :func:`y_at_x <sigima.proc.signal.y_at_x>`
     - Find y-values at given x-coordinate
   * - :func:`x_at_minmax <sigima.proc.signal.x_at_minmax>`
     - Find x-coordinates of extrema

Curve Fitting
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`linear_fit <sigima.proc.signal.linear_fit>`
     - Linear regression
   * - :func:`polynomial_fit <sigima.proc.signal.polynomial_fit>`
     - Polynomial fitting
   * - :func:`gaussian_fit <sigima.proc.signal.gaussian_fit>`
     - Gaussian curve fitting
   * - :func:`lorentzian_fit <sigima.proc.signal.lorentzian_fit>`
     - Lorentzian curve fitting
   * - :func:`voigt_fit <sigima.proc.signal.voigt_fit>`
     - Voigt profile fitting
   * - :func:`twohalfgaussian_fit <sigima.proc.signal.twohalfgaussian_fit>`
     - Two-half Gaussian fitting
   * - :func:`exponential_fit <sigima.proc.signal.exponential_fit>`
     - Exponential decay fitting
   * - :func:`piecewiseexponential_fit <sigima.proc.signal.piecewiseexponential_fit>`
     - Piecewise exponential fitting
   * - :func:`sigmoid_fit <sigima.proc.signal.sigmoid_fit>`
     - Sigmoid curve fitting
   * - :func:`sinusoidal_fit <sigima.proc.signal.sinusoidal_fit>`
     - Sinusoidal fitting
   * - :func:`planckian_fit <sigima.proc.signal.planckian_fit>`
     - Planckian (blackbody) fitting
   * - :func:`cdf_fit <sigima.proc.signal.cdf_fit>`
     - Cumulative distribution function fitting
   * - :func:`evaluate_fit <sigima.proc.signal.evaluate_fit>`
     - Evaluate fitted function
   * - :func:`extract_fit_params <sigima.proc.signal.extract_fit_params>`
     - Extract fitting parameters

Stability Analysis
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`allan_variance <sigima.proc.signal.allan_variance>`
     - Allan variance
   * - :func:`allan_deviation <sigima.proc.signal.allan_deviation>`
     - Allan deviation
   * - :func:`modified_allan_variance <sigima.proc.signal.modified_allan_variance>`
     - Modified Allan variance
   * - :func:`overlapping_allan_variance <sigima.proc.signal.overlapping_allan_variance>`
     - Overlapping Allan variance
   * - :func:`hadamard_variance <sigima.proc.signal.hadamard_variance>`
     - Hadamard variance
   * - :func:`time_deviation <sigima.proc.signal.time_deviation>`
     - Time deviation
   * - :func:`total_variance <sigima.proc.signal.total_variance>`
     - Total variance

Pulse Analysis
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`extract_pulse_features <sigima.proc.signal.extract_pulse_features>`
     - Extract comprehensive pulse characteristics

Utility Functions
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`check_same_sample_rate <sigima.proc.signal.check_same_sample_rate>`
     - Verify consistent sampling rates
   * - :func:`get_nyquist_frequency <sigima.proc.signal.get_nyquist_frequency>`
     - Calculate Nyquist frequency

Image Processing
----------------

Geometry and Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`rotate <sigima.proc.image.rotate>`
     - Rotate image by arbitrary angle
   * - :func:`rotate90 <sigima.proc.image.rotate90>`
     - Rotate image 90° clockwise
   * - :func:`rotate270 <sigima.proc.image.rotate270>`
     - Rotate image 270° clockwise
   * - :func:`fliph <sigima.proc.image.fliph>`
     - Flip image horizontally
   * - :func:`flipv <sigima.proc.image.flipv>`
     - Flip image vertically
   * - :func:`translate <sigima.proc.image.translate>`
     - Translate image by specified offset
   * - :func:`resize <sigima.proc.image.resize>`
     - Resize image to specified dimensions
   * - :func:`resampling <sigima.proc.image.resampling>`
     - Resample image with different methods
   * - :func:`binning <sigima.proc.image.binning>`
     - Reduce image size by binning pixels
   * - :func:`set_uniform_coords <sigima.proc.image.set_uniform_coords>`
     - Set uniform coordinate system
   * - :data:`transformer <sigima.proc.image.transformer>`
     - Apply custom geometric transformations

Frequency Domain Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`butterworth <sigima.proc.image.butterworth>`
     - Butterworth frequency filter
   * - :func:`gaussian_freq_filter <sigima.proc.image.gaussian_freq_filter>`
     - Gaussian frequency filter

Edge Detection
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`sobel <sigima.proc.image.sobel>`
     - Sobel edge detector
   * - :func:`sobel_h <sigima.proc.image.sobel_h>`
     - Sobel horizontal edges
   * - :func:`sobel_v <sigima.proc.image.sobel_v>`
     - Sobel vertical edges
   * - :func:`scharr <sigima.proc.image.scharr>`
     - Scharr edge detector
   * - :func:`scharr_h <sigima.proc.image.scharr_h>`
     - Scharr horizontal edges
   * - :func:`scharr_v <sigima.proc.image.scharr_v>`
     - Scharr vertical edges
   * - :func:`prewitt <sigima.proc.image.prewitt>`
     - Prewitt edge detector
   * - :func:`prewitt_h <sigima.proc.image.prewitt_h>`
     - Prewitt horizontal edges
   * - :func:`prewitt_v <sigima.proc.image.prewitt_v>`
     - Prewitt vertical edges
   * - :func:`farid <sigima.proc.image.farid>`
     - Farid edge detector
   * - :func:`farid_h <sigima.proc.image.farid_h>`
     - Farid horizontal edges
   * - :func:`farid_v <sigima.proc.image.farid_v>`
     - Farid vertical edges
   * - :func:`roberts <sigima.proc.image.roberts>`
     - Roberts cross-gradient
   * - :func:`laplace <sigima.proc.image.laplace>`
     - Laplacian edge detector
   * - :func:`canny <sigima.proc.image.canny>`
     - Canny edge detector

Feature Detection
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`blob_dog <sigima.proc.image.blob_dog>`
     - Blob detection using Difference of Gaussians
   * - :func:`blob_doh <sigima.proc.image.blob_doh>`
     - Blob detection using Determinant of Hessian
   * - :func:`blob_log <sigima.proc.image.blob_log>`
     - Blob detection using Laplacian of Gaussians
   * - :func:`blob_opencv <sigima.proc.image.blob_opencv>`
     - OpenCV-based blob detection
   * - :func:`peak_detection <sigima.proc.image.peak_detection>`
     - 2D peak detection
   * - :func:`contour_shape <sigima.proc.image.contour_shape>`
     - Contour shape analysis
   * - :func:`hough_circle_peaks <sigima.proc.image.hough_circle_peaks>`
     - Circular Hough transform

Morphological Operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`erosion <sigima.proc.image.erosion>`
     - Morphological erosion
   * - :func:`dilation <sigima.proc.image.dilation>`
     - Morphological dilation
   * - :func:`opening <sigima.proc.image.opening>`
     - Morphological opening
   * - :func:`closing <sigima.proc.image.closing>`
     - Morphological closing
   * - :func:`white_tophat <sigima.proc.image.white_tophat>`
     - White top-hat transform
   * - :func:`black_tophat <sigima.proc.image.black_tophat>`
     - Black top-hat transform

Thresholding
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`threshold <sigima.proc.image.threshold>`
     - Manual threshold with custom value
   * - :func:`threshold_otsu <sigima.proc.image.threshold_otsu>`
     - Otsu's thresholding
   * - :func:`threshold_li <sigima.proc.image.threshold_li>`
     - Li's thresholding
   * - :func:`threshold_yen <sigima.proc.image.threshold_yen>`
     - Yen's thresholding
   * - :func:`threshold_triangle <sigima.proc.image.threshold_triangle>`
     - Triangle thresholding
   * - :func:`threshold_isodata <sigima.proc.image.threshold_isodata>`
     - Isodata thresholding
   * - :func:`threshold_mean <sigima.proc.image.threshold_mean>`
     - Mean thresholding
   * - :func:`threshold_minimum <sigima.proc.image.threshold_minimum>`
     - Minimum thresholding

Exposure and Intensity Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`histogram <sigima.proc.image.histogram>`
     - Compute image histogram
   * - :func:`equalize_hist <sigima.proc.image.equalize_hist>`
     - Histogram equalization
   * - :func:`equalize_adapthist <sigima.proc.image.equalize_adapthist>`
     - Adaptive histogram equalization
   * - :func:`adjust_gamma <sigima.proc.image.adjust_gamma>`
     - Gamma correction
   * - :func:`adjust_log <sigima.proc.image.adjust_log>`
     - Logarithmic adjustment
   * - :func:`adjust_sigmoid <sigima.proc.image.adjust_sigmoid>`
     - Sigmoid adjustment
   * - :func:`rescale_intensity <sigima.proc.image.rescale_intensity>`
     - Rescale intensity range
   * - :func:`flatfield <sigima.proc.image.flatfield>`
     - Flat-field correction

Image Restoration
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`denoise_bilateral <sigima.proc.image.denoise_bilateral>`
     - Bilateral denoising
   * - :func:`denoise_tv <sigima.proc.image.denoise_tv>`
     - Total variation denoising
   * - :func:`denoise_wavelet <sigima.proc.image.denoise_wavelet>`
     - Wavelet denoising
   * - :func:`denoise_tophat <sigima.proc.image.denoise_tophat>`
     - Top-Hat denoising
   * - :func:`erase <sigima.proc.image.erase>`
     - Erase image regions

Preprocessing
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`zero_padding <sigima.proc.image.zero_padding>`
     - Zero-padding for images

Measurements and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Function
      - Description
    * - :func:`centroid <sigima.proc.image.centroid>`
      - Compute image centroid
    * - :func:`enclosing_circle <sigima.proc.image.enclosing_circle>`
      - Find minimum enclosing circle
    * - :func:`stats <sigima.proc.image.stats>`
      - Statistical measurements
    * - :func:`line_profile <sigima.proc.image.line_profile>`
      - Extract profile along a line
    * - :func:`segment_profile <sigima.proc.image.segment_profile>`
      - Extract profile along a multi-segment line
    * - :func:`radial_profile <sigima.proc.image.radial_profile>`
      - Extract radial profile from center
    * - :func:`average_profile <sigima.proc.image.average_profile>`
      - Average profile over region
    * - :func:`horizontal_projection <sigima.proc.image.horizontal_projection>`
      - Project image onto horizontal axis
    * - :func:`vertical_projection <sigima.proc.image.vertical_projection>`
      - Project image onto vertical axis
    * - :func:`generate_image_grid_roi <sigima.proc.image.generate_image_grid_roi>`
      - Generate grid of ROIs

See Also
--------

- :doc:`../api/proc` - Complete API reference for all processing functions
- :doc:`../api/objects` - Data objects (SignalObj, ImageObj) used by processing functions
- :doc:`../auto_examples/index` - Examples demonstrating various computation features
