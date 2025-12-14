# Release notes #

## Sigima Version 1.1.0 ##

✨ New features and enhancements:

* **Stub server**: Added missing methods to `DataLabStubServer` to support new DataLab features
  * Added `remove_object()` method (removes an object from DataLab)
  * Added `call_method()` method (simulates calling a method on DataLab's main interface or panels)

* **Test infrastructure**: Refactored visualization code in test suite for better maintainability
  * Centralized PlotPy visualization utilities in `sigima.tests.vistools` module
  * Added standardized helper functions: `create_curve()`, `create_image()`, `create_circle()`, `create_segment()`, `create_cursor()`, `create_marker()`, and `create_contour_shapes()`
  * Updated all interactive test functions to use the new unified API instead of direct PlotPy builder calls
  * Improved code consistency across signal and image test modules
