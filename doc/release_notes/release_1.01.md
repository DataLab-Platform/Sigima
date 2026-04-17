# Version 1.1 #

## Sigima Version 1.1.2 (2026-04-20) ##

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

### ✨ Enhancements since version 1.1.1 ###

* **Legend value formatting**: Improved numeric formatting in analysis result legends
  * Short values are displayed as plain numbers, while long values use scientific notation adapted to fit within the display width
  * Pulse features table now uses adaptive scientific notation showing only the significant digits needed for exact round-trip representation

* **Table column display formats**: Added per-column format control for `TableResult`
  * New `get_column_formats()` / `set_column_formats()` methods on `TableResult`
  * `TableResultBuilder` supports `set_column_formats()` for specifying format strings (e.g. `{"x0": ".2e"}`) or callable formatters
  * Supports `"*"` wildcard key as a per-table default formatter

### 📦 Dependencies ###

* **Pandas 3.0 support**: Officially support pandas 3.0.x after validation — upper bound raised from `< 3.0` to `< 3.1`
  * This closes [Issue #13](https://github.com/DataLab-Platform/Sigima/issues/13) - Support pandas 3.0

* **Python 3.14 support**: Added Python 3.14 classifier

### 🔧 Other changes ###

* Improved development environment setup: new `run_with_env.py` script supporting multiple Python environment contexts (WinPython, venv, etc.)
* CI: gated PyPI deployment on test suite passing

## Sigima Version 1.1.1 (2026-02-02) ##

### 🛠️ Bug Fixes since version 1.1.0 ###

* **Stub server**: Added missing Web API control methods to `DataLabStubServer` for testing DataLab's REST API integration
  * Added `start_webapi_server()` stub (returns dummy URL and token)
  * Added `stop_webapi_server()` stub
  * Added `get_webapi_status()` stub (returns server status dictionary)

### 🔧 Other changes ###

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
