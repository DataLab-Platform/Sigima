# Version 1.3 #

## Sigima Version 1.3.0 ##

### ✨ New features since version 1.2.0 ###

* **Brightness and contrast adjustment**: A new ROI-aware image processing performs numerically stable clipped linear remapping from persistent minimum and maximum bounds, preserves the source dtype and non-finite values, and exposes portable histogram context for native and browser interfaces. Tiny floating-point ranges and finite IEEE-754 extremes are supported without coupling headless computation to histogram rendering. The low-level helpers support floating-point arrays and integers up to 32 bits; they explicitly reject `int64` and `uint64` instead of risking precision loss at integer endpoints.
* **Portable graphical annotations**: Signals and images may now carry versioned, renderer-independent points, shapes, text, cursors and axis ranges. Annotations survive Sigima file round trips and supported image transformations, and are displayed consistently by the PlotPy and Matplotlib visualization backends. Existing PlotPy annotations remain readable and can be migrated explicitly while unknown application data is preserved. This implements [Issue #53](https://github.com/DataLab-Platform/Sigima/issues/53).
* **Interactive Plotly visualization**: Signals and images can now be inspected in a browser with zoom, pan and hover through the optional Plotly backend. Sigima also exposes dependency-free Plotly JSON builders for applications and notebooks, including portable annotation, ROI and geometry-result overlays.
* **Parameter validation**: Numeric creation and processing parameters now expose precise unconditional bounds, while mode-dependent and source-dependent constraints are checked before computation. Signed coordinate and grid conventions, as well as point-like or descending signal domains, remain supported.
* **RAW image import**: Headerless RAW files (`*.raw`) can now be read with `RawImageImportParam`, which describes the pixel data type, image size, offset to the first image, number of images, gap between images and byte order. Multi-image files return one image per frame, and the "white is zero" convention is stored in the image metadata without altering pixel values. Writing RAW files is not supported.
* **Format-aware image export**: `write_image` now accepts an optional `ImageExportParam` to prepare data for the destination format: min-max, percentile or manual normalization, clipping or rescaling of out-of-range values, handling of non-finite values, gamma correction, inversion and conversion to a supported data type. Format-specific writer options (compression, quality, tiling, resolution, delimiter, precision, etc.) are validated against per-format capabilities for BMP, PNG, JPEG, JPEG 2000, TIFF, NumPy, MAT-File, text/CSV and HDF5 files, and an exact round-trip preview is available for BMP, PNG, JPEG, JPEG 2000 and TIFF files. Images with non-uniform coordinates keep their coordinates when exported to text formats.

### ??? Bug Fixes since version 1.2.0 ###

* **Compressed TIFF files**: TIFF files compressed with LZW, JPEG, WebP or CCITT (Group 3, Group 4, RLE) could not be opened and failed with the error "requires the 'imagecodecs' package". LZW is the default compression of many acquisition and image-editing tools. The same limitation prevented format-aware TIFF export with LZW or JPEG compression. These compression schemes are now supported out of the box. This closes [DataLab issue #381](https://github.com/DataLab-Platform/DataLab/issues/381).

### ?? Dependencies changes since version 1.2.0 ###

* **imagecodecs**: Added `imagecodecs >= 2024.12.30` as a required dependency. It provides the TIFF decoders and encoders used by `tifffile` for compression schemes that `tifffile` does not implement itself. The minimum version is the latest release with wheels for every supported Python version (3.9 and later); pip automatically selects the most recent release compatible with the installed Python and NumPy versions.
