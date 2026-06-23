# Version 1.2 #

## Sigima Version 1.2.0 ##

### ✨ New features since version 1.1.5 ###

* **Contour-to-ROI creation**: Contour detection (`contour_shape`) can now automatically generate ROIs from detected shapes. This closes [Issue #20](https://github.com/DataLab-Platform/Sigima/issues/20).
  * New `create_rois` parameter on `ContourShapeParam` enables ROI creation from detected contours (circles, ellipses, polygons)
  * Detected contour geometries are converted to `CircularROI` or `PolygonalROI` objects that downstream applications (DataLab) can display and edit
  * Ellipse contours are approximated as polygonal ROIs using a dedicated ellipse-to-polygon conversion that correctly handles rotation angles and semi-axes in physical coordinates

* **ROI creation with physical coordinates**: `create_image_roi_around_points` now generates ROIs in physical coordinates instead of pixel indices, ensuring correct placement on images with non-unit pixel spacing or non-zero origin

* **Replace special values**: Added `replace_special_values` processing function for images, allowing replacement of NaN/Inf values with configurable substitution (zero, mean, median, or interpolation). This closes [Issue #30](https://github.com/DataLab-Platform/Sigima/issues/30).

* **Extract peak positions**: Added `extract_peak_positions` function to extract XY marker coordinates from peak detection results as a new signal object

* **Signal markers to signal**: Added `markers_table_to_signal` function to convert marker tables (from peak detection or other analyses) into signal objects

* **Curve-clipped ROI rendering**: Signal ROI visualization now supports curve-clipped fill rendering with a consistent color palette

### 🛠️ Bug Fixes since version 1.1.5 ###

* **ROI physical coordinates**: Fixed `create_image_roi_around_points` using pixel indices instead of physical coordinates (`indices=False`), causing misplaced ROIs on calibrated images
* **Ellipse-to-polygon conversion**: Fixed shearing in polygon approximation of ellipse ROIs caused by incorrect trigonometric decomposition of rotation and semi-axes
* **Custom signal XY preservation**: Fixed user-edited XY values being silently discarded when regenerating a custom signal from its creation parameters. This closes [Issue #25](https://github.com/DataLab-Platform/Sigima/issues/25).

### 🔧 Other changes since version 1.1.5 ###

* **Remote control client**: Added `get_current_object_uuid()` method to `SimpleRemoteProxy` and `SimpleAbstractDLControl`
* **API cleanup**: `store_contour_roi_metadata` made private (`_store_contour_roi_metadata`) — internal helper only used by `contour_shape()`
* **`peak_detection` deprecated**: Use `extract_peak_positions` + `markers_table_to_signal` for new code
* **Test coverage**: Added unit tests for contour ROI metadata handling, detection with non-unit pixel spacing, and non-uniform coordinate geometry operations
