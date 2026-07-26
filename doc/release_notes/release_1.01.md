# Version 1.1 #

## Sigima Version 1.1.6 ##

### 🛠️ Bug Fixes since version 1.1.5 ###

* **Gaussian, Lorentzian and Voigt peak amplitude**: Corrected the `amplitude` parameter so it consistently represents signed peak height above the baseline, in the same Y units as the signal, instead of an integrated area whose scale changed with peak width. Fit and signal-creation parameters now carry explicit schema and parameterization markers; historical unversioned area-based parameters are rejected until explicitly converted, preventing silent reinterpretation while preserving the generated or fitted X/Y data.
* **Multi-peak fit metadata**: Multi-Gaussian and multi-Lorentzian results now persist each detected `x0_i` center alongside the corresponding `amplitude_i` and `sigma_i`, allowing the complete versioned model to be exported and evaluated on any X axis.
* **Color image import (JPEG, PNG, BMP, TIFF)**: Fixed pixel values being rescaled to the `[0, 1]` range when opening a color (RGB/RGBA) image, which made the imported data unusable for quantitative analysis. Color images are now converted to grayscale while keeping their original value range (e.g. `0`–`255` for 8-bit images), matching the behavior of tools such as ImageJ. Grayscale images stored as RGB are read back with their exact original values. This closes [Issue #37](https://github.com/DataLab-Platform/Sigima/issues/37).
* **Stability analysis (⚠️ results change)**: Fixed five of the six frequency-stability estimators, which returned values that were wrong by a factor growing with the averaging time τ — so the *shape* of the plotted curve was wrong, not merely its scale. **Any stability figure produced with an earlier version must be recomputed.** This closes [Issue #49](https://github.com/DataLab-Platform/Sigima/issues/49).
  * The overlapping Allan variance and the Hadamard variance compared averages one sample apart instead of one *averaging interval* apart, under-estimating the result by a factor of about τ
  * The modified Allan variance was not the modified Allan variance at all, and was further divided by τ², under-estimating the result by a factor of about τ²
  * The total variance was missing its factor of 1/2, over-estimating the result by a factor of 2; it is now the genuine TOTVAR estimator computed on the reflected phase data, instead of a duplicate of the Allan variance
  * The time deviation is now derived from the modified Allan deviation, as its definition requires, instead of the Allan deviation — a factor of √3 at the shortest τ
  * All estimators are now defined at the shortest averaging time τ = dt, where they previously returned `NaN`, and they consistently reject τ shorter than the sampling interval
  * Estimator definitions are documented with their reference (IEEE Std 1139, NIST SP 1065), and validated against exact analytical expectations rather than order-of-magnitude bounds

## Sigima Version 1.1.5 ##

### 🛠️ Bug Fixes since version 1.1.4 ###

* **EllipseModel TypeError**: Added defensive fallback for `EllipseModel` raising `TypeError` under scikit-image 0.26 edge cases
* **Deprecated `collections.abc.ByteString`**: Replaced usage removed in Python 3.14 with `bytes` and `bytearray` checks

### 🔧 Other changes since version 1.1.4 ###

* **Test infrastructure**: Made `WorkdirRestoringTempDir` cleanup robust on Windows (avoids `PermissionError` on temporary directories)

## Sigima Version 1.1.4 ##

### 🛠️ Bug Fixes since version 1.1.3 ###

* **Ellipse/circle contour detection**: Fixed regression in axis orientation after the v1.1.3 coordinate swap fix. This closes [Issue #27](https://github.com/DataLab-Platform/Sigima/issues/27).
  * The sign correction applied in v1.1.3 for the `(row, col)` → `(x, y)` conversion introduced a regression in ellipse axis orientation, causing the semi-major and semi-minor axes to be transposed
  * Added ground-truth unit tests for `fit_circle_model` and `fit_ellipse_model` verifying all output parameters (center coordinates, radius/semi-axes, and rotation angle)

## Sigima Version 1.1.3 ##

### 🛠️ Bug Fixes since version 1.1.2 ###

* **Ellipse/circle contour detection**: Fixed swapped X/Y coordinates when using scikit-image ≥ 0.26.0. This closes [Issue #21](https://github.com/DataLab-Platform/Sigima/issues/21).
  * The new `EllipseModel`/`CircleModel` API returns center coordinates as `(row, col)` but the code was treating them as `(x, y)`, causing detected ellipses and circles to appear at wrong positions on the image
  * Semi-axis lengths were also swapped for ellipses, resulting in incorrect aspect ratios
  * This bug only affected scikit-image ≥ 0.26.0; older versions were handled correctly
* **Ellipse visualization**: Fixed incorrect minor axis direction in `ellipse_to_diameters` coordinate conversion
  * The minor axis endpoints were computed with a wrong sign, making the minor axis non-perpendicular to the major axis for rotated ellipses (e.g., at θ=45° both axes pointed in the same direction). This caused distorted ellipse overlays in DataLab when displaying detected or annotated ellipses at non-trivial rotation angles
* **Circle/ellipse detection with calibrated images**: Fixed radius and semi-axes not being converted from pixel units to physical units
  * When an image had non-default pixel calibration (e.g., dx=2 mm/pixel), center coordinates were correctly converted to physical units but the radius and semi-axes remained in pixels, causing values to be wrong by a factor equal to the pixel size
* **Custom signal XY preservation**: Fixed user-edited XY values being silently discarded when regenerating a custom signal from its creation parameters. `generate_1d_data()` now initializes the XY array only on first use and preserves user-edited contents on subsequent calls. This closes [Issue #25](https://github.com/DataLab-Platform/Sigima/issues/25).

### 🔧 Other changes since version 1.1.2 ###

* **Debug environment variable**: Renamed the test debug environment variable from `DEBUG` to `SIGIMA_DEBUG` to avoid collisions with unrelated tools and frameworks that commonly export a bare `DEBUG` variable. This closes [Issue #19](https://github.com/DataLab-Platform/Sigima/issues/19).
* **Remote control client**: Added `get_current_object_uuid()` method to `SimpleRemoteProxy` and `SimpleAbstractDLControl`, allowing users to retrieve the UUID of the currently selected object in DataLab

### 🧪 Tests ###

* **Non-uniform image coordinates**: Strengthened test coverage for images with non-uniform (irregularly spaced) X/Y coordinates, closing the remaining gaps in coordinate-mode switching, geometry operations and coordinated-text reading. This closes [Issue #24](https://github.com/DataLab-Platform/Sigima/issues/24).

## Sigima Version 1.1.2 (2026-04-20) ##

### 💥 Breaking changes since version 1.1.1 ###

* **Removed deprecated `sigima.tests.vistools` module**: The backward-compatibility shim introduced in version 1.1.0 has been removed
  * Update imports from `from sigima.tests import vistools` to `from sigima import viz`
  * All visualization helpers (`view_curves`, `view_images`, `create_curve`, `create_image`, etc.) remain available under their new home in `sigima.viz`

### 🛠️ Bug Fixes since version 1.1.1 ###

* **CSV header parsing**: Fixed whitespace and unit extraction issues when reading CSV column headers
  * Leading and trailing whitespace in column titles is now properly trimmed (e.g. `"  Padded NoUnit  "` → `"Padded NoUnit"`)
  * Nested parentheses in units are now correctly handled (e.g. `"Signal (a.u. (norm))"` → label `"Signal"`, unit `"a.u.(norm)"` instead of incorrectly splitting at the last parenthesis)
  * Exotic whitespace characters (tabs, non-breaking spaces) in headers are normalized before parsing

* **Pandas 3.0 compatibility**: Fixed datetime CSV parsing issue with pandas 3.0+
  * Replaced `iloc` assignment with column name assignment to handle dtype conversion correctly
  * Ensures compatibility with both pandas < 3.0 and pandas 3.0+ versions
  * Affected functionality: CSV file reading with datetime columns and signal data import from CSV files
  * All related tests pass after the fix
  * This closes [Issue #12](https://github.com/DataLab-Platform/Sigima/issues/12) - Fix pandas 3.0 compatibility issue in datetime CSV parsing

* **HDF5 serialization of TableResult**: Fixed DataLab workspace save failure when table results contain enum values or callable formatters
  * Enum subclasses (e.g. `SignalShape`) stored in table data are now converted to plain strings before serialization, preventing `dtype('<U4')` errors
  * Non-serializable entries (e.g. callable column formatters) in `TableResult.attrs` are now sanitized before HDF5/JSON export

* **Datetime signal compatibility**: Fixed datetime column handling for compatibility with various NumPy and pandas versions
  * Allowed any `datetime64` resolution (not just `ns`) to support newer NumPy/pandas combinations

* **Signal uncertainty (dx/dy)**: Fixed phantom error bars displayed when only one of `dx` or `dy` was set on a fresh signal
  * When `dy` was set first (or `dx` first), the deferred uncertainty row for the other dimension was previously filled with zeros, which downstream tools interpreted as a real all-zero uncertainty array (e.g. PlotPy drew zero-width error bars)
  * Deferred rows are now initialized with NaN, so a missing uncertainty is properly reported as absent — no more spurious error bars

* **ROI extraction with uncertainty data**: Fixed `extract_roi` and `extract_rois` losing the input signal's `dx` / `dy` uncertainty arrays
  * Uncertainty samples are now sliced and propagated to the extracted signal exactly like the X/Y data
  * Affects both single-ROI and multi-ROI extraction workflows

* **Zero-crossing detection**: Fixed `find_zero_crossings` over-reporting on signals containing exact-zero samples
  * Samples exactly equal to zero used to count as two crossings (e.g. `[-1, 0, 1]` reported two crossings instead of one) and "touch-and-bounce" patterns like `[1, 0, 1]` were wrongly flagged as crossings
  * Sign changes are now detected between consecutive non-zero samples, with contiguous zero runs collapsed to a single crossing only when they actually bridge opposite signs
  * Downstream features benefit from the fix: `find_x_axis_crossings`, `find_x_values_at_y`, `find_bandwidth_coordinates`, and pulse FWHM / rise / fall measurements

* **Contrast computation**: Fixed `contrast` returning `inf` / `nan` warnings on signals with negative or all-zero values
  * Denominator now uses absolute values (`(max - min) / (|max| + |min|)`), keeping the result bounded in `[0, 1]` for signals with negative samples while matching the standard Michelson contrast for non-negative signals
  * Returns `nan` cleanly (without a division-by-zero warning) when both `max` and `min` are zero

* **Datetime conversion accuracy**: Fixed `datetime64_to_seconds` returning wrong Unix timestamps for coarse `datetime64` resolutions
  * The previous implementation only recognized `ns`, `us` and `ms` and silently used a nanosecond divisor for any other resolution, producing incorrect timestamps for `datetime64[s]`, `datetime64[m]`, `datetime64[h]` or `datetime64[D]`
  * Inputs are now normalized to `datetime64[ns]` before conversion, ensuring correct results for any source resolution and for pandas `DatetimeIndex` objects

* **DICOM image origin and spacing**: Fixed `ImageObj.dicom_template` falling back to inconsistent default coordinates when DICOM tags were missing
  * When `ImagePositionPatient` or `PixelSpacing` was absent from the DICOM template, only the Y origin and Y pixel spacing received the fallback while the X origin and X pixel spacing were left as bare scalars, due to operator precedence
  * Both X and Y origin (`x0`, `y0`) and pixel spacing (`dx`, `dy`) are now consistently filled with the intended `(0.0, 0.0)` and `(1.0, 1.0)` defaults

### ✨ Enhancements since version 1.1.1 ###

* **Legend value formatting**: Improved numeric formatting in analysis result legends
  * Short values are displayed as plain numbers, while long values use scientific notation adapted to fit within the display width
  * Pulse features table now uses adaptive scientific notation showing only the significant digits needed for exact round-trip representation

* **Table column display formats**: Added per-column format control for `TableResult`
  * New `get_column_formats()` / `set_column_formats()` methods on `TableResult`
  * `TableResultBuilder` supports `set_column_formats()` for specifying format strings (e.g. `{"x0": ".2e"}`) or callable formatters
  * Supports `"*"` wildcard key as a per-table default formatter

### 📦 Dependencies changes since version 1.1.1 ###

* **Pandas 3.0 support**: Officially support pandas 3.0.x after validation — upper bound raised from `< 3.0` to `< 3.1`
  * This closes [Issue #13](https://github.com/DataLab-Platform/Sigima/issues/13) - Support pandas 3.0

* **Python 3.14 support**: Added Python 3.14 classifier

### 🔧 Other changes since version 1.1.1 ###

* **Remote control client**: Added `set_object` method to `SimpleRemoteProxy` and `SimpleAbstractDLControl`, allowing users to push modified signal/image objects back to DataLab after retrieving them with `get_object` (see [DataLab Issue #305](https://github.com/DataLab-Platform/DataLab/issues/305))
* **Stub server**: Added `set_object` support to `DataLabStubServer` with proper UUID-based object lookup, and registered `add_object`/`set_object` as XML-RPC functions
* **Stub server**: Signal and image objects now carry their UUID in metadata, enabling round-trip `get_object` → modify → `set_object` workflows

* Improved development environment setup: new `run_with_env.py` script supporting multiple Python environment contexts (WinPython, venv, etc.)
* CI: gated PyPI deployment on test suite passing
* **Test coverage**: Substantially expanded the unit-test suite to exercise previously untested branches across `sigima.objects`, `sigima.io`, `sigima.tools`, `sigima.proc` and `sigima.viz` (26 new test modules), and excluded the GUI visualization layer from coverage reports for more meaningful metrics

## Sigima Version 1.1.1 (2026-02-02) ##

### 🛠️ Bug Fixes since version 1.1.0 ###

* **Stub server**: Added missing Web API control methods to `DataLabStubServer` for testing DataLab's REST API integration
  * Added `start_webapi_server()` stub (returns dummy URL and token)
  * Added `stop_webapi_server()` stub
  * Added `get_webapi_status()` stub (returns server status dictionary)

### 🔧 Other changes since version 1.1.0 ###

* Updated development status classifier to "Production/Stable"
* Added "Try it Online" section with [notebook.link](https://notebook.link/) integration in documentation

## Sigima Version 1.1.0 (2026-01-31) ##

✨ New features and enhancements:

* **Stub server**: Added missing methods to `DataLabStubServer` to support new DataLab features
  * Added `remove_object()` method (removes an object from DataLab)
  * Added `call_method()` method (simulates calling a method on DataLab's main interface or panels)

* **Test infrastructure**: Refactored visualization code in test suite for better maintainability
  * Centralized PlotPy visualization utilities in `sigima.viz` module
  * Added standardized helper functions: `create_curve()`, `create_image()`, `create_circle()`, `create_segment()`, `create_cursor()`, `create_marker()`, and `create_contour_shapes()`
  * Updated all interactive test functions to use the new unified API instead of direct PlotPy builder calls
  * Improved code consistency across signal and image test modules

* **Visualization backend flexibility**: Added support for Matplotlib as an alternative to PlotPy for test visualizations
  * New configuration option `viz_backend` to select visualization library ("auto", "plotpy", or "matplotlib")
  * Environment variable `SIGIMA_VIZ_BACKEND` can override configuration
  * Automatic fallback from PlotPy to Matplotlib when PlotPy is not available
  * Backend information exposed via `sigima.viz.BACKEND_NAME` and `BACKEND_SOURCE`
  * Matplotlib backend provides stubs for unsupported functions (raises `NotImplementedError`)
  * Unit test ensures API compatibility between backends

* **Jupyter notebook HTML representation**: Added rich HTML display for Sigima objects in Jupyter notebooks
  * `SignalObj` displays shape, dtype, X/Y ranges, axis labels, and title in a formatted table
  * `ImageObj` displays shape, dtype, value range, origin, pixel spacing, extent, axis labels, and title
  * ROI objects (`BaseSingleROI`, `BaseROI`) display their geometric properties
  * `BaseCoordinates` and derived classes display point coordinates
  * `TableResult` and `GeometryResult` display result data with source object information
  * Centralized CSS styling in `HTML_TABLE_CSS` constant for consistent appearance
