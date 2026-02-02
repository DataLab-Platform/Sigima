# Version 1.1 #

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
