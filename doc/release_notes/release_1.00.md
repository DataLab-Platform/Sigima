# Version 1.0 #

## Sigima Version 1.0.3 (2025-12-03) ##

🛠️ Bug fixes:

* **Signal data type validation**: Fixed integer arrays not being automatically converted to float64
  * Integer input arrays are now automatically converted to float64 instead of raising errors
  * Validation applied consistently across all signal data setters: `set_xydata()`, `x`, `y`, `dx`, `dy`
  * Improves usability by accepting integer inputs (common in test data and calibration values) while maintaining computational precision
  * Raises clear `ValueError` for truly invalid dtypes with helpful error message listing valid types

* **Signal axis calibration**: Added `replace_x_by_other_y()` function to replace signal X coordinates with Y values from another signal
  * Addresses missing functionality for wavelength calibration and similar use cases where calibration data is stored in a separate signal's Y values
  * This operation was previously impossible, even if the ambiguous X-Y mode feature existed and seemed related to this use case (but this feature performs resampling/interpolation, which is not desired here)
  * The new function directly uses Y arrays from both signals without interpolation, requiring signals to have the same number of points
  * Takes two signals: first provides Y data for output, second provides Y data to become X coordinates
  * Automatically transfers metadata: X label/unit from second signal's Y label/unit, Y label/unit preserved from first signal
  * Typical use case: spectroscopy wavelength calibration (combine absorption measurements with wavelength scale)
  * This closes [Issue #4](https://github.com/DataLab-Platform/Sigima/issues/4) - Missing functionality: Replace X coordinates with Y values from another signal for calibration

* Signal title formatting:
  * **Polynomial signal titles**: Fixed polynomial signal title generation to display mathematical notation (e.g., `1+2x-3x²+4x³`) instead of verbose parameter listing (e.g., `polynomial(a0=1,a1=2,a2=-3,a3=4,a4=0,a5=0)`)
  * The `PolyParam.generate_title()` method now constructs proper mathematical expressions with correct sign handling, coefficient simplification (e.g., `x` instead of `1x`, `-x` instead of `-1x`), and exponent notation using `^` symbol
  * Improves readability in DataLab GUI and signal listings by presenting polynomials in standard mathematical form
  * Zero coefficients are automatically omitted from the expression (e.g., `1+x+x³` when a2=0)
  * Handles edge cases including all-zero polynomials (returns `"0"`), single terms, and negative coefficients
  * This closes [Issue #3](https://github.com/DataLab-Platform/Sigima/issues/3) - Polynomial signal titles should use mathematical notation instead of parameter listing

* ROI data extraction:
  * Fixed `ValueError: zero-size array to reduction operation minimum which has no identity` error when computing statistics on images with ROI extending beyond canvas boundaries
  * The `ImageObj.get_data()` method now properly clips ROI bounding boxes to image boundaries
  * When a ROI is completely outside the image bounds, returns a fully masked 1x1 array (containing NaN) to avoid zero-size array errors in statistics computations
  * Partial overlap ROIs are correctly handled by clipping coordinates to valid image ranges
  * This fix ensures robust statistics computation regardless of ROI position relative to image boundaries
  * This closes [Issue #1](https://github.com/DataLab-Platform/Sigima/issues/1) - `ValueError` when computing statistics on ROI extending beyond image boundaries

## Sigima Version 1.0.2 (2025-11-12) ##

✨ New features and enhancements:

* **New parametric image types**: Added five new parametric image generation types for testing and calibration
  * **Checkerboard pattern**: Alternating squares for camera calibration and spatial frequency analysis. Parameters include square size, offset, and min/max values
  * **Sinusoidal grating**: Frequency response testing with configurable spatial frequencies (fx, fy), phase, amplitude, and DC offset
  * **Ring pattern**: Concentric circular rings for radial analysis. Configurable period, width, center position, and amplitude range
  * **Siemens star**: Resolution testing pattern with radial spokes. Parameters include number of spokes, inner/outer radius, center position, and value range
  * **2D sinc function**: PSF/diffraction modeling with cardinal sine function. Configurable amplitude, center, scale factor (sigma), and DC offset

* **GeometryResult.value property**: New convenience property for easy script access to computed geometry values
  * Supports POINT, MARKER, and SEGMENT shapes
  * Returns `(x, y)` tuple for POINT and MARKER shapes (both coordinates accessible)
  * Returns `float` length for SEGMENT shapes (calculated via `segments_lengths()`)
  * Return type annotation: `float | tuple[float, float]`
  * Provides intuitive API: unpack coordinates with `x, y = result.value` or get length with `length = result.value`
  * Comprehensive error handling for unsupported shapes and multi-row results

* **Signal analysis functions return GeometryResult**: Changed `x_at_y()` and `y_at_x()` to return geometry results for better visualization
  * `x_at_y()` now returns `GeometryResult` with `MARKER` kind (previously returned `TableResult`)
  * `y_at_x()` now returns `GeometryResult` with `MARKER` kind (previously returned `TableResult`)
  * Both functions return coordinates as `[x, y]` in N×2 array format for cross marker display
  * Enables proper marker visualization in DataLab GUI (displayed as cross markers on plots)
  * Script-friendly API: use `.value` property to easily extract coordinates as `(x, y)` tuple
  * Example: `x, y = proxy.compute_x_at_y(params).value` provides direct access to both coordinates
  * Breaking change: Scripts accessing results as tables need to update to use `.value` property or `.coords` array

🛠️ Bug fixes:

* Detection functions:
  * **Contour detection**: Removed ROI creation support from `ContourShapeParam` as it doesn't make sense for contour detection use cases. The `ContourShapeParam` class no longer inherits from `DetectionROIParam`, and the `contour_shape()` function no longer calls `store_roi_creation_metadata()`. ROI creation remains available for other detection methods (blob detection, 2D peak detection) where it is appropriate.

  * **ROI creation error handling**: Enhanced error handling in `create_image_roi_around_points()` function to provide clearer error messages:
    * Now raises `ValueError` when calculated ROI size is too small (points too close together)
    * Improved error messages to help users understand the cause of failures
    * Validates ROI geometry parameter more explicitly
    * Better handling of edge cases in automatic ROI sizing

* Public API:
  * Made `BaseObj.roi_has_changed` method private (by renaming to `BaseObj.__roi_has_changed`) to avoid accidental external usage. This would interfere with the internal mask refresh mechanism that relies on controlled access to this method. The method is not part of the public API and should not be called directly by applications.

## Sigima Version 1.0.1 (2025-11-05) ##

✨ New features and enhancements:

* **Detection ROI creation**: Generic mechanism for ROI creation across all detection functions
  * New `DetectionROIParam` parameter class providing standardized ROI creation fields
    * `create_rois`: Boolean flag to enable/disable ROI creation (default: False)
    * `roi_geometry`: Enum selecting ROI shape (RECTANGLE or CIRCLE, default: RECTANGLE)
  * New `DetectionROIGeometry` enum in `sigima.enums` with RECTANGLE and CIRCLE options
  * All detection parameter classes now inherit from `DetectionROIParam`:
    * `Peak2DDetectionParam`: 2D peak detection
    * `ContourShapeParam`: Contour shape fitting
    * `BlobDOGParam`, `BlobDOHParam`, `BlobLOGParam`, `BlobOpenCVParam`: Blob detection methods
    * `HoughCircleParam`: Hough circle detection
  * New `store_roi_creation_metadata()` helper function:
    * Stores ROI creation intent in `GeometryResult.attrs` dictionary
    * Called within computation functions to communicate ROI preferences
    * Does not violate function purity (no object modification)
  * New `apply_detection_rois()` helper function:
    * Creates ROIs on image objects based on `GeometryResult.attrs` metadata
    * Returns `True` if ROIs were created, `False` otherwise
    * Handles both rectangle and circle geometries
    * Automatically calculates optimal ROI size based on feature spacing
    * Can be called by applications outside computation functions
  * Metadata-based architecture maintains separation of concerns:
    * Computation functions remain pure (no side effects)
    * Applications control when/how ROIs are created
    * Works seamlessly with multiprocessing engines (e.g., DataLab processors)
  * Comprehensive test coverage with `validate_detection_rois()` helper in test suite

* **Automatic `func_name` injection for result objects**
  * The `@computation_function` decorator now automatically injects the function name into `TableResult` and `GeometryResult` objects
  * When a computation function returns a result object with `func_name=None`, the decorator sets it to the function's name using `dataclasses.replace()`
  * Ensures systematic assignment of `func_name` for proper result tracking and display
  * Implementation uses direct `isinstance()` type checking for `TableResult` and `GeometryResult`
  * Applies to both main decorator wrapper (with DataSet parameters) and simple passthrough wrapper
  * Eliminates need for manual `func_name` assignment in computation functions

* **Image ROI creation utility**: New `create_image_roi_around_points()` function in `sigima.objects.image.roi`
  * Creates rectangular or circular ROIs around a set of point coordinates
  * Automatically calculates optimal ROI size based on minimum distance between points
  * Handles boundary conditions to keep ROIs within valid image coordinates
  * Supports both "rectangle" and "circle" geometry types
  * Designed for creating ROIs around detected features (peaks, blobs, etc.)
  * Centralizes ROI creation logic previously duplicated across applications

* **Annotations API**: New public API for managing annotations on Signal and Image objects
  * Added `get_annotations()` method: Returns a list of annotations in versioned JSON format
  * Added `set_annotations(annotations)` method: Sets annotations from a list (replaces existing annotations)
  * Added `add_annotation(annotation)` method: Adds a single annotation to the object
  * Added `clear_annotations()` method: Removes all annotations from the object
  * Added `has_annotations()` method: Returns True if the object has any annotations
  * Annotations are stored in object metadata with versioning support (currently version "1.0")
  * Each annotation is a dictionary with keys such as `type`, `item_class`, and `item_json` (for example)
  * Provides clean separation between generic annotation storage and visualization-specific details
  * Enables applications to manage plot annotations (shapes, labels, etc.) independently of ROIs
  * Fully compatible with DataLab's PlotPy adapter pattern for visualization

🛠️ Bug fixes:

* **2D peak detection**: Fixed architectural violation in `peak_detection()` computation function
  * Removed direct ROI creation from computation function (was modifying input objects)
  * Computation functions decorated with `@computation_function()` must be pure (no side effects)
  * Removed line 128: `obj.roi = create_image_roi(...)` which violated this principle
  * ROI creation now handled by applications in their presentation layer
  * DataLab uses new `create_image_roi_around_points()` utility for this purpose
  * Maintains separation of concerns: Sigima computes results, applications create visual representations
  * Fixes regression where ROIs were not appearing in DataLab's processor-based workflow

* **Parameter classes**: Removed default titles from generic `OrdinateParam` and `AbscissaParam` classes
  * These parameter classes are reused across multiple computation functions (e.g., `full_width_at_y`, `x_at_y`)
  * Default titles like "Ordinate" created redundancy when displayed with function names in analysis results
  * Titles are now empty by default, allowing applications to provide context-specific titles when needed
  * Improves clarity when the same parameter class is used by different functions

* **Result HTML representation**: Improved color contrast for dark mode
  * Changed result title color in `to_html()` methods from standard blue (#0000FF) to a lighter shade (#5294e2)
  * Affects TableResult and GeometryResult HTML output
  * Provides better visibility in dark mode while maintaining good appearance in light mode
  * Improves readability when results are displayed in applications with dark themes

* Fixed pulse features extraction with ROI signals. When extracting pulse features from signals with ROIs, the start/end range parameters (which apply to the full signal) were being used on ROI-extracted data, causing incorrect results. Now `extract_pulse_features()` detects when the parameter ranges are outside the ROI's x-range and automatically switches to auto-detection mode. Additionally, `extract_pulse_features()` in `sigima.tools.signal.pulse` now properly initializes `None` ranges using `get_start_range()` and `get_end_range()` with the `fraction` parameter. This ensures pulse features extracted from a signal with ROIs match the features extracted from individually extracted ROI signals.

* Fixed ROI extraction for signals: ROIs are no longer incorrectly copied to destination signals when extracting ROIs. When using `extract_roi()` or `extract_rois()`, the extracted signals now have no ROI defined, which is the expected behavior since the extracted data already represents the ROI itself. This fixes the issue where extracted signals would inherit the source signal's ROI definitions.

* Fixed pulse features computation to be ROI-exclusive when ROIs are defined. Previously, `TableKind.PULSE_FEATURES` incorrectly computed results for both the whole object and each ROI. This made no sense for pulse analysis, where defining ROIs indicates the presence of multiple pulses, making whole-object features irrelevant. Now `PULSE_FEATURES` correctly computes only on ROIs when they exist, otherwise on the whole object. `TableKind.STATISTICS` and `TableKind.CUSTOM` maintain the expected behavior (whole object + ROIs).

* Fixed `ValueError` in `choose_savgol_window_auto()` when processing small data arrays (e.g., ROI segments). The function now properly constrains the Savitzky-Golay window length to be strictly less than the array size, as required by scipy's `mode='interp'` option. This fixes the issue when extracting pulse features from small ROI segments in signals.

* Modified `RadialProfileParam` to allow initialization of the dataset even when the associated image object is not yet set (call to `update_from_obj`). This is useful when creating the parameter object before assigning the image, enabling more flexible workflows.

* Removed unused `signals_to_array()` function from `sigima.proc.signal.arithmetic` module. This function was not used anywhere in the codebase and has been replaced by direct NumPy array construction in `__signals_y_to_array()` and `__signals_dy_to_array()` functions, for internal use only.

* **ROI coordinate setters**: Fixed bugs in `set_physical_coords()` and `set_indices_coords()` methods
  * Fixed `RectangularROI.set_physical_coords()`: Now correctly stores coordinates in delta format `[x0, y0, dx, dy]` instead of corner format `[x0, y0, x1, y1]` when `indices=True`
  * Fixed `BaseSingleROI.set_indices_coords()`: Now correctly converts the input `coords` parameter instead of `self.coords` when `indices=False`
  * These methods were implemented for API completeness but were not used in the Sigima/DataLab codebase
  * Added comprehensive unit tests covering all ROI types (rectangular, circular, polygonal) and edge cases

## Sigima Version 1.0.0 (2025-10-28) ##

✨ New features and enhancements:

* **Signals to image conversion**: New feature to combine multiple signals into a 2D image
  * New computation function `signals_to_image()` in `sigima.proc.signal.arithmetic`
  * Takes a list of signals and combines them into an image by stacking Y-arrays
  * Two orientation modes:
    * **Rows**: Each signal becomes a row in the image (default)
    * **Columns**: Each signal becomes a column in the image
  * Optional normalization:
    * Supports multiple normalization methods (Z-score, Min-Max, Maximum)
    * Normalizes each signal independently before stacking
    * Useful for visualizing signals with different amplitude ranges
  * Validates that all signals have the same size before combining
  * New parameter class `SignalsToImageParam` with orientation and normalization settings
  * New enum `SignalsToImageOrientation` for specifying row/column orientation
  * Comprehensive validation tests for all combinations of parameters
  * Ideal for creating spectrograms, heatmaps, or waterfall displays from signal collections

* **Non-uniform coordinate support for images**: Added comprehensive support for non-uniform pixel coordinates
  * `ImageObj` now supports both uniform and non-uniform coordinate systems:
    * Uniform coordinates: defined by origin (`x0`, `y0`) and pixel spacing (`dx`, `dy`)
    * Non-uniform coordinates: defined by coordinate arrays (`xcoords`, `ycoords`)
  * New methods for coordinate manipulation:
    * `set_coords()`: Set non-uniform X and Y coordinate arrays
    * `is_uniform_coords`: Property to check if coordinates are uniform
  * New computation function `set_uniform_coords()`: Convert non-uniform to uniform coordinates
    * Automatically extracts uniform spacing from non-uniform arrays
    * Handles numerical precision issues from linspace-generated arrays
    * Preserves image data while transforming coordinate system
  * Enhanced `calibration()` function with polynomial support:
    * Now supports polynomial calibration up to cubic order: `dst = a0 + a1*src + a2*src² + a3*src³`
    * Parameter class changed from `a, b` (linear) to `a0, a1, a2, a3` (polynomial)
    * Works on X-axis, Y-axis (creating non-uniform coordinates), and Z-axis (data values)
    * Linear calibration is a special case with `a2=0, a3=0`
    * Automatically handles conversion between uniform and non-uniform coordinate systems
  * Enhanced I/O support:
    * HDF5 format now serializes/deserializes non-uniform coordinates
    * Coordinated text files support non-uniform coordinate arrays
  * All geometric operations updated to handle both coordinate types:
    * Coordinate transformations preserve or create appropriate coordinate system
    * ROI operations work seamlessly with both uniform and non-uniform coordinates

* **DateTime support for signal data**: Added comprehensive datetime handling for signal X-axis data
  * Automatic detection and conversion of datetime columns when reading CSV files
    * Detects datetime values in the first or second column (handling index columns)
    * Validates datetime format and ensures reasonable date ranges (post-1900)
    * Converts datetime strings to float timestamps for efficient computation
    * Preserves datetime metadata for proper display and export
  * New `SignalObj` methods for datetime manipulation:
    * `set_x_from_datetime()`: Convert datetime objects/strings to signal X data with configurable time units (s, ms, μs, ns, min, h)
    * `get_x_as_datetime()`: Retrieve X values as datetime objects for display or export
    * `is_x_datetime()`: Check if signal contains datetime data
  * Enhanced CSV export to preserve datetime format when writing signals with datetime X-axis
  * New constants module (`sigima.objects.signal.constants`) defining datetime metadata keys and time unit conversion factors
  * Comprehensive unit tests covering datetime conversion, I/O roundtrip, and edge cases
  * Example test data file with real-world temperature/humidity logger data (`datetime.txt`)

* **New client subpackage**: Migrated DataLab client functionality to `sigima.client`
  * Added `sigima.client.remote.SimpleRemoteProxy` for XML-RPC communication with DataLab
  * Added `sigima.client.base.SimpleBaseProxy` as abstract base class for DataLab proxies
  * Included comprehensive unit tests and API documentation
  * Maintains headless design principle (GUI components excluded)
  * Enables remote control of DataLab application from Python scripts and Jupyter notebooks
  * Client functionality is now directly accessible: `from sigima import SimpleRemoteProxy`

* **New image ROI feature**: Added inverse ROI functionality for image ROIs
  * Added `inside` parameter to `BaseSingleImageROI` base class, inherited by all image ROI types (`PolygonalROI`, `RectangularROI`, `CircularROI`)
  * When `inside=True`, ROI represents the region inside the shape (inverted behavior)
  * When `inside=False` (default), ROI represents the region outside the shape (original behavior)
  * Fully integrated with serialization (`to_dict`/`from_dict`) and parameter conversion (`to_param`/`from_param`)
  * Signal ROIs (`SegmentROI`) are unaffected as the concept doesn't apply to 1D intervals
  * Optimal architecture with zero code duplication - all `inside` functionality implemented once in the base class
  * Individual ROI classes no longer need custom constructors, inheriting directly from base class

* New image operation:
  * Convolution.

* New image format support:
  * **Coordinated text image files**: Added support for reading coordinated text files (`.txt` extension), similar to the Matris image format.
    * Supports both real and complex-valued image data with optional error images.
    * Automatically handles NaN values in the data.
    * Reads metadata including units (X, Y, Z) and labels from file headers.

* New image analysis features:
  * Horizontal and vertical projections
    * Compute the horizontal projection profile by summing values along the y-axis (`sigima.proc.image.measurement.horizontal_projection`).
    * Compute the vertical projection profile by summing values along the x-axis (`sigima.proc.image.measurement.vertical_projection`).

* **New curve fitting algorithms**: Complete curve fitting framework with `sigima.tools.signal.fitting` module:
  * **Core fitting functions**: Comprehensive set of curve fitting algorithms for scientific data analysis:
    * `linear_fit`: Linear regression fitting
    * `polynomial_fit`: Polynomial fitting with configurable degree
    * `gaussian_fit`: Gaussian profile fitting for peak analysis
    * `lorentzian_fit`: Lorentzian profile fitting for spectroscopy
    * `voigt_fit`: Voigt profile fitting (convolution of Gaussian and Lorentzian profiles)
    * `exponential_fit`: Single exponential fitting with overflow protection
    * `piecewiseexponential_fit`: Piecewise exponential (raise-decay) fitting with advanced parameter estimation
    * `planckian_fit`: Planckian (blackbody radiation) fitting with correct physics implementation
    * `twohalfgaussian_fit`: Asymmetric peak fitting with separate left/right parameters
    * `multilorentzian_fit`: Multi-peak Lorentzian fitting for complex spectra
    * `sinusoidal_fit`: Sinusoidal fitting with FFT-based frequency estimation
    * `cdf_fit`: Cumulative Distribution Function fitting using error function
    * `sigmoid_fit`: Sigmoid (logistic) function fitting for S-shaped curves
  * **Advanced piecewise exponential (raise-decay) fitting**: Enhanced algorithm with:
    * Standard piecewise exponential model: `y = a_left*exp(b_left*x) + a_right*exp(b_right*x) + y0`
    * Multi-start optimization strategy for robust convergence to global minimum
    * Support for both positive and negative exponential rates (growth and decay components)
    * Comprehensive parameter bounds validation to prevent optimization errors
  * **Enhanced asymmetric peak fitting**: Advanced `twohalfgaussian_fit` with:
    * Separate baseline offsets for left and right sides (`y0_left`, `y0_right`)
    * Independent amplitude parameters (`amp_left`, `amp_right`) for better asymmetric modeling
    * Robust baseline estimation using percentile-based methods
  * **Technical features**: All fitting functions include:
    * Automatic initial parameter estimation from data characteristics
    * Proper bounds enforcement ensuring optimization stability
    * Comprehensive error handling and parameter validation
    * Consistent dataclass-based parameter structures
    * Full test coverage with synthetic and experimental data validation

* New common signal/image feature:
  * Added `phase` (argument) feature to extract the phase information from complex signals or images.
  * Added operation to create complex-valued signal/image from real and imaginary parts.
  * Added operation to create complex-valued signal/image from magnitude and phase.
  * Standard deviation of the selected signals or images (this complements the "Average" feature).
  * Generate new signal or image: Poisson noise.
  * Add noise to the selected signals or images.
    * Gaussian, Poisson or uniform noise can be added.
  * New utility functions to generate file basenames.
  * Deconvolution in the frequency domain.

* New ROI features:
  * Improved single ROI title handling, using default title based on the index of the ROI when no title is provided.
  * Added `combine_with` method to ROI objects (`SignalROI` and `ImageROI`) to return a new ROI that combines the current ROI with another one (union) and handling duplicate ROIs.
  * Image ROI transformations:
    * Before this change, image ROI were removed after applying each single computation function.
    * Now, the geometry computation functions preserve the ROI information across transformations: the transformed ROIs are automatically updated in the image object.
  * Image ROI coordinates:
    * Before this change, image ROI coordinates were defined using indices by default.
    * Now, `ROI2DParam` uses physical coordinates by default.
    * Note that ROI may still be defined using indices instead (using `create_image_roi` function).
  * Image ROI grid:
    * New `generate_image_grid_roi` function: create a grid of ROIs from an image, with customizable parameters for grid size, spacing, and naming.
    * This function allows for easy extraction of multiple ROIs from an image in a structured manner.
    * Parameters are handled via the `ROIGridParam` class, which provides a convenient way to specify grid properties:
      * `nx` / `ny`: Number of grid cells in the X/Y direction.
      * `xsize` / `ysize`: Size of each grid cell in pixels.
      * `xtranslation` / `ytranslation`: Translation of the grid in pixels.
      * `xdirection` / `ydirection`: Direction of the grid (increasing/decreasing).

* New image processing features:
  * New "2D resampling" feature:
    * This feature allows to resample 2D images to a new coordinate grid using interpolation.
    * It supports two resampling modes: pixel size and output shape.
    * Multiple interpolation methods are available: linear, cubic, and nearest neighbor.
    * The `fill_value` parameter controls how out-of-bounds pixels are handled, with support for numeric values or NaN.
    * Automatic data type conversion ensures proper NaN handling for integer images.
    * It is implemented in the `sigima.proc.image.resampling` function with parameters defined in `Resampling2DParam`.
  * New "Frequency domain Gaussian filter" feature:
    * This feature allows to filter an image in the frequency domain using a Gaussian filter.
    * It is implemented in the `sigima.proc.image.frequency_domain_gaussian_filter` function.
  * New "Erase" feature:
    * This feature allows to erase an area of the image using the mean value of the image.
    * It is implemented in the `sigima.proc.image.erase` function.
    * The erased area is defined by a region of interest (ROI) parameter set.
    * Example usage:

      ```python
      import numpy as np
      import sigima.objects as sio
      import sigima.proc.image as sipi

      obj = sio.create_image("test_image", data=np.random.rand(1024, 1024))
      p = sio.ROI2DParam.create(x0=600, y0=800, width=300, height=200)
      dst = sipi.erase(obj, p)
      ```

  * By default, pixel binning changes the pixel size.

  * Improved centroid estimation:
    * New `get_centroid_auto` method implements an adaptive strategy that chooses between the Fourier-based centroid and a more robust fallback (scikit-image), based on agreement with a projected profile-based reference.
    * Introduced `get_projected_profile_centroid` function for robust estimation via 1D projections (median or barycentric), offering high accuracy even with truncated or noisy images.
    * These changes improve centroid accuracy and stability in edge cases (e.g. truncated disks or off-center spots), while preserving noise robustness.
    * See [DataLab issue #251](https://github.com/DataLab-Platform/DataLab/issues/251) for more details.

* New signal processing features:
  * New "Brick wall filter" feature:
    * This feature allows to filter a signal in the frequency domain using an ideal ("brick wall") filter.
    * It is implemented in `sigima.proc.signal.frequency_filter`, along the other frequency domain filtering features (`Bessel`, `Butterworth`, etc.).
  * Enhanced zero padding to support prepend and append. Change default strategy to next power of 2.
  * **Pulse analysis algorithms**: Comprehensive pulse feature extraction framework in `sigima.tools.signal.pulse` module:
    * **Core pulse analysis functions**: Complete set of algorithms for step and square pulse characterization:
      * `extract_pulse_features`: Main function for automated pulse feature extraction
      * `heuristically_recognize_shape`: Intelligent signal type detection (step, square, or other)
      * `detect_polarity`: Robust polarity detection using baseline analysis
    * **Advanced timing parameter extraction**: Precise measurement algorithms for:
      * Rise and fall time calculations with configurable start/stop ratios (e.g., 10%-90%)
      * Timing parameters at specific fractions (x10, x50, x90, x100) of signal amplitude
      * Full width at half maximum (FWHM) computation for square pulses
      * Foot duration measurement for pulse characterization
    * **Baseline analysis capabilities**: Statistical methods for:
      * Automatic baseline range detection from signal extremes
      * Robust baseline level estimation using mean values within ranges
      * Start and end baseline characterization for differential analysis
    * **Signal validation and error handling**: Comprehensive input validation with:
      * Data array consistency checks and NaN/infinity detection
      * Signal length validation and range boundary verification
      * Graceful error handling with descriptive exception messages
    * **PulseFeatures dataclass**: Structured result container with all extracted parameters:
      * Amplitude, polarity, and offset measurements
      * Timing parameters (rise_time, fall_time, fwhm, x10, x50, x90, x100)
      * Baseline ranges (xstartmin, xstartmax, xendmin, xendmax)
      * Signal shape classification and foot duration
    * Implementation leverages robust statistical methods and provides both high-level convenience functions and low-level building blocks for custom pulse analysis workflows.
  * Comprehensive uncertainty propagation implementation:
    * Added mathematically correct uncertainty propagation to ~15 core signal processing functions.
    * Enhanced `Wrap1to1Func` class to handle uncertainty propagation for mathematical functions (`sqrt`, `log10`, `exp`, `clip`, `absolute`, `real`, `imag`).
    * Implemented uncertainty propagation for arithmetic operations (`product_constant`, `division_constant`).
    * Added uncertainty propagation for advanced processing functions (`power`, `normalize`, `derivative`, `integral`, `calibration`).
    * All implementations use proper error propagation formulas with numerical stability handling (NaN/infinity protection).
    * Optimized for memory efficiency by leveraging `dst_1_to_1` automatic uncertainty copying and in-place modifications.
    * Maintains backward compatibility with existing signal processing workflows.

* New 2D ramp image generator:
  * This feature allows to generate a 2D ramp image: z = a(x − x₀) + b(y − y₀) + c
  * It is implemented in the `sigima.objects.Ramp2DParam` parameter class.
  * Example usage:

    ```python
    import sigima.objects as sio
    param = sio.Ramp2DParam.create(width=100, height=100, a=1.0, b=2.0)
    image = sio.create_image_from_param(param)
    ```

* New signal generators: linear chirp, logistic function, Planck function.

* New image "Extent" computed parameters:
  * Added computed parameters for image extent: `xmin`, `xmax`, `ymin`, and `ymax`.
  * These parameters are automatically calculated based on the image origin, pixel spacing, and dimensions.
  * They provide the physical coordinate boundaries of the image for enhanced spatial analysis.

* New I/O features:
  * Added HDF5 format for signal and image objects (extensions `.h5sig` and `.h5ima`) that may be opened with any HDF5 viewer.
  * Added support for MCA (Multi-Channel Analyzer) spectrum file format:
    * Reading MCA files (`.mca` extension) commonly used in spectroscopy and radiation detection
    * Automatically extracts spectrum data and calibration information
    * Supports energy calibration with interpolation for accurate X-axis values
    * Parses metadata from multiple sections (PMCA SPECTRUM, DPP STATUS, CALIBRATION)
    * Handles various encoding formats (UTF-8, Latin-1, CP1252) for maximum compatibility
  * Added support for FT-Lab signal and image format.
  * Added functions to read and write metadata and ROIs in JSON format:
    * `sigima.io.read_metadata` and `sigima.io.write_metadata` for metadata.
    * `sigima.io.read_roi` and `sigima.io.write_roi` for ROIs.
  * Added convenience I/O functions `write_signals` and `write_images` with `SaveToDirectoryParam` support:
    * These functions enable batch saving of multiple signal or image objects to a directory with configurable naming patterns.
    * `SaveToDirectoryParam` provides control over file basenames (with Python format string support), extensions, directory paths, and overwrite behavior.
    * Automatic filename conflict resolution ensures unique filenames when duplicates would occur.
    * Enhanced workflow efficiency for processing and saving multiple objects in batch operations.

✨ Core architecture update: scalar result types

* Introduced two new immutable result types: `TableResult` and `GeometryResult`, replacing the legacy `ResultProperties` and `ResultShape` objects.
  * These new result types are computation-oriented and free of application-specific logic (e.g., Qt, metadata), enabling better separation of concerns and future reuse.
  * Added a `TableResultBuilder` utility to incrementally define tabular computations (e.g., statistics on signals or images) and generate a `TableResult` object.
  * All metadata-related behaviors of former result types have been migrated to the DataLab application layer.
  * Removed obsolete or tightly coupled features such as `from_metadata_entry()` and `transform_shapes()` from the Sigima core.
* This refactoring greatly improves modularity, testability, and the clarity of the scalar computation API.

🛠️ Bug fixes:

* Fix how data is managed in signal objects (`SignalObj`):
  * Signal data is stored internally as a 2D array with shape `(2, n)`, where the first row is the x data and the second row is the y data: that is the `xydata` attribute.
  * Because of this, when storing complex Y data, the data type is propagated to the x data, which is not always desired.
  * As a workaround, the `x` property now returns the real part of the x data.
  * Furthermore, the `get_data` method now returns a tuple of numpy arrays instead of a single array, allowing to access both x and y data separately, keeping the original data type.
* Fix ROI conversion between physical and indices coordinates:
  * The conversion between physical coordinates and indices has been corrected (half pixel error was removed).
  * The `indices_to_physical` and `physical_to_indices` methods now raise a `ValueError` if the input does not contain an even number of elements (x, y pairs).

🔒 Security fixes:

* **Dependency vulnerability fix**: Fixed CVE-2023-4863 vulnerability in opencv-python-headless
  * Updated minimum requirement from 4.5.4.60 to 4.8.1.78
  * Addresses libwebp binaries vulnerability in bundled OpenCV wheels
  * See [DataLab security advisory](https://github.com/DataLab-Platform/DataLab/security/dependabot/1) for details
