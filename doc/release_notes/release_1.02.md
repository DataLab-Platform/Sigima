# Version 1.2 #

## Sigima Version 1.2.0 ##

### ✨ New features since version 1.1.6 ###

* **Contour-to-ROI creation**: Contour detection (`contour_shape`) can now automatically generate ROIs from detected shapes. This closes [Issue #20](https://github.com/DataLab-Platform/Sigima/issues/20).
  * New `create_rois` parameter on `ContourShapeParam` enables ROI creation from detected contours (circles, ellipses, polygons)
  * Detected contour geometries are converted to `CircularROI` or `PolygonalROI` objects that downstream applications (DataLab) can display and edit
  * Ellipse contours are approximated as polygonal ROIs using a dedicated ellipse-to-polygon conversion that correctly handles rotation angles and semi-axes in physical coordinates

* **ROI creation with physical coordinates**: `create_image_roi_around_points` now generates ROIs in physical coordinates instead of pixel indices, ensuring correct placement on images with non-unit pixel spacing or non-zero origin

* **Replace special values**: Added the `replace_special_values` processing function for both signals and images, allowing NaN and infinite values to be replaced with a configurable strategy (zero, constant, minimum, maximum, mean, median, neighborhood statistics, and — for signals only — point deletion, forward/backward fill and linear, spline, quadratic, cubic or PCHIP interpolation). Strategies are exposed as the `ReplacementStrategySignal` and `ReplacementStrategyImage` enumerations, with the `ReplaceSpecialValuesSignalParam` and `ReplaceSpecialValuesImageParam` parameter classes. This closes [Issue #30](https://github.com/DataLab-Platform/Sigima/issues/30).

* **Extract peak positions**: Added `extract_peak_positions` function to extract XY marker coordinates from peak detection results as a new signal object

* **Signal markers to signal**: Added `markers_table_to_signal` function to convert marker tables (from peak detection or other analyses) into signal objects

* **Marker result tables**: `TableKind` gained the `XY_MARKERS`, `X_MARKERS` and `Y_MARKERS` kinds, along with the matching `TableResult.is_xy_markers()`, `is_x_markers()` and `is_y_markers()` predicates. Marker tables describe remarkable points to be displayed as crosses or as vertical/horizontal cursors, and — like pulse features — are computed on the regions of interest only when the object defines any. Setting `show_row_index` in the result attributes numbers the rows `#0`, `#1`, … instead of labelling them by ROI. These building blocks support spectral line analysis, as requested in [Issue #17](https://github.com/DataLab-Platform/Sigima/issues/17).

* **Curve-clipped ROI rendering**: Signal ROI visualization now supports curve-clipped fill rendering with a consistent color palette. This closes [Issue #18](https://github.com/DataLab-Platform/Sigima/issues/18).

* **Option field categories**: `OptionField` (and its subclasses) now accept an optional `category` attribute, letting client applications (such as SigimaX) group configuration options, for example into settings-dialog tabs or INI sections

### 🛠️ Bug Fixes since version 1.1.6 ###

* **Inverse regions of interest**: Fixed inverse ROIs being processed through the excluded shape's bounding box instead of the whole image. Extracting an inverse ROI returned an image cropped to the very area that was meant to be excluded, coordinates reported by detection functions run on an inverse ROI were shifted by the shape origin, and an inverse rectangle was drawn covering the entire image instead of its own outline. Data read through a non-rectangular or inverse ROI now has the pixels lying outside the region set to `NaN` — which implies a conversion to floating point, so an extracted region no longer keeps the integer type of the source image. This closes [Issue #34](https://github.com/DataLab-Platform/Sigima/issues/34).
* **Masked array warning in detection functions**: Detection functions (`get_2d_peaks_coords`, `get_hough_circle_peaks`, `find_blobs_dog`, `find_blobs_doh`, `find_blobs_log`, `find_blobs_opencv`) now emit a warning when called with a masked array (e.g. when using a non-rectangular ROI): the underlying libraries (scikit-image, OpenCV, SciPy) do not support masked arrays, so the mask is ignored and results may be unexpected inside or near masked areas. This closes [Issue #35](https://github.com/DataLab-Platform/Sigima/issues/35).
* **Detected ROIs preserve existing ones**: Detection algorithms (peak, blob, Hough and contour detection) with `create_rois=True` now append the newly detected ROIs to any ROIs already defined on the image, instead of replacing them. Previously defined regions of interest are no longer lost. This closes [Issue #36](https://github.com/DataLab-Platform/Sigima/issues/36).
* **Region of interest comparison**: Fixed two regions of interest of different geometries being considered equal as soon as their coordinate arrays matched — a circle and a rectangle sharing the same numbers, for instance. Merging regions of interest also compared the incoming ones against the original list instead of the list being built, so duplicates within a single merge slipped through.
* **ROI physical coordinates**: Fixed `create_image_roi_around_points` using pixel indices instead of physical coordinates (`indices=False`), causing misplaced ROIs on calibrated images
* **Ellipse-to-polygon conversion**: Fixed shearing in polygon approximation of ellipse ROIs caused by incorrect trigonometric decomposition of rotation and semi-axes
* **Custom signal XY preservation**: Fixed user-edited XY values being silently discarded when regenerating a custom signal from its creation parameters. This closes [Issue #25](https://github.com/DataLab-Platform/Sigima/issues/25).
* **Negative pulse features**: Fixed square pulses with negative polarity using a baseline sample as their peak, which could make full pulse-feature extraction fail instead of returning timing and amplitude measurements.

### 🔧 Other changes since version 1.1.6 ###

* **Object title moved next to the data**: The `title` field of `SignalObj` and `ImageObj` now belongs to the *Data and metadata* group instead of *Titles / Units*, so both object types present their parameters consistently. Applications displaying these objects in a parameter dialog will see the title in its new location.
* **New `DetectionROIParam` parameter class**: Publicly exported from `sigima.proc.image` and `sigima.params`, it groups the ROI creation options shared by the detection functions
* **`ROI2DParam.get_bounding_box_physical()` renamed to `get_shape_circumscribed_rect()`**: The former name suggested an extraction bounding box, whereas the method returns the rectangle circumscribing the ROI shape — which, unlike the bounding box, does not depend on the inverse flag
* **Remote control client**: Added `get_current_object_uuid()` method to `SimpleRemoteProxy` and `SimpleAbstractDLControl`
* **New `BaseObj.add_roi()` method**: Non-destructive helper that merges a ROI object into an object's existing ROIs (keeping existing regions and deduplicating identical single ROIs); used by the detection ROI creation path
* **API cleanup**: `store_contour_roi_metadata` made private (`_store_contour_roi_metadata`) — internal helper only used by `contour_shape()`
* **`peak_detection` deprecated**: Use `extract_peak_positions` + `markers_table_to_signal` for new code
* **Test coverage**: Added unit tests for contour ROI metadata handling, detection with non-unit pixel spacing, and non-uniform coordinate geometry operations
* **`OptionField.get()`/`.set()` `sync_env` made keyword-only**: Calling either method with `sync_env` as a positional argument now raises `TypeError` instead of silently being misinterpreted, preventing accidental misuse
