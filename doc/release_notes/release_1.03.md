# Version 1.3 #

## Sigima Version 1.3.0 ##

### ✨ New features since version 1.2.0 ###

* **Brightness and contrast adjustment**: A new ROI-aware image processing performs numerically stable clipped linear remapping from persistent minimum and maximum bounds, preserves the source dtype and non-finite values, and exposes portable histogram context for native and browser interfaces. Tiny floating-point ranges and finite IEEE-754 extremes are supported without coupling headless computation to histogram rendering. The low-level helpers support floating-point arrays and integers up to 32 bits; they explicitly reject `int64` and `uint64` instead of risking precision loss at integer endpoints.
* **Portable graphical annotations**: Signals and images may now carry versioned, renderer-independent points, shapes, text, cursors and axis ranges. Annotations survive Sigima file round trips and supported image transformations, and are displayed consistently by the PlotPy and Matplotlib visualization backends. Existing PlotPy annotations remain readable and can be migrated explicitly while unknown application data is preserved. This implements [Issue #53](https://github.com/DataLab-Platform/Sigima/issues/53).
* **Interactive Plotly visualization**: Signals and images can now be inspected in a browser with zoom, pan and hover through the optional Plotly backend. Sigima also exposes dependency-free Plotly JSON builders for applications and notebooks, including portable annotation, ROI and geometry-result overlays.
* **Parameter validation**: Numeric creation and processing parameters now expose precise unconditional bounds, while mode-dependent and source-dependent constraints are checked before computation. Signed coordinate and grid conventions, as well as point-like or descending signal domains, remain supported.
