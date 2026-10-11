# Version 1.3 #

## Sigima Version 1.3.0 ##

### ✨ New features since version 1.2.0 ###

* **Brightness and contrast adjustment**: A new ROI-aware image processing performs numerically stable clipped linear remapping from persistent minimum and maximum bounds, preserves the source dtype and non-finite values, and exposes portable histogram context for native and browser interfaces. Tiny floating-point ranges and finite IEEE-754 extremes are supported without coupling headless computation to histogram rendering. The low-level helpers support floating-point arrays and integers up to 32 bits; they explicitly reject `int64` and `uint64` instead of risking precision loss at integer endpoints.
* **Portable graphical annotations**: Signals and images may now carry versioned, renderer-independent points, shapes, text, cursors and axis ranges. Annotations survive Sigima file round trips and supported image transformations, and are displayed consistently by the PlotPy and Matplotlib visualization backends. Existing PlotPy annotations remain readable and can be migrated explicitly while unknown application data is preserved. This implements [Issue #53](https://github.com/DataLab-Platform/Sigima/issues/53).
* **Interactive Plotly visualization**: Signals and images can now be inspected in a browser with zoom, pan and hover through the optional Plotly backend. Sigima also exposes dependency-free Plotly JSON builders for applications and notebooks, including portable annotation, ROI and geometry-result overlays.
* **Parameter validation**: Numeric creation and processing parameters now expose precise unconditional bounds, while mode-dependent and source-dependent constraints are checked before computation. Signed coordinate and grid conventions, as well as point-like or descending signal domains, remain supported.
* **Operation contracts for verifiable replay**: A computation function can now declare a versioned contract (stable identifier, version, roles of its inputs and parameter schema), so that an application can record what was computed and later recompute it with exactly the same meaning, even across DataLab Desktop and DataLab-Web. Normalization (every method) and signal difference are the first declared operations. Functions without a contract keep working as before; applications simply do not replay them.
* **Explicit X alignment for signal difference**: When the two signals of a difference have different X grids, the second one is interpolated linearly on the grid of the first, under a named and versioned rule. The second grid must cover the first one (no extrapolation), both must be strictly increasing, and X and Y units must match exactly (no conversion): otherwise the operation is refused with a clear reason instead of producing a misleading result. The inputs are never modified.

### 🔧 Dependencies ###

* guidata 3.16.0 or later is now required (histogram-backed range items used by brightness and contrast adjustment).
