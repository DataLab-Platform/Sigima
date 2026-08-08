.. _api_viz:

:mod:`sigima.viz` --- Visualization Tools
=========================================

.. module:: sigima.viz

This module provides visualization utilities for Sigima objects, useful for:

- Interactive testing and debugging
- Data analysis in Jupyter notebooks
- Quick visual inspection of processing results

Backend Selection
-----------------

The module supports **PlotPy**, **Matplotlib**, and **Plotly** backends.  The
first two participate in automatic selection; Plotly is selected explicitly.

The backend selection follows this priority:

1. Environment variable ``SIGIMA_VIZ_BACKEND`` (if set)
2. Configuration option :attr:`sigima.config.options.viz_backend`
3. Auto-detection (PlotPy preferred, Matplotlib as fallback)

Backend selection logic:

- ``"auto"``: Try PlotPy first, fall back to Matplotlib
- ``"plotpy"``: Use PlotPy (raise :class:`ImportError` if not available)
- ``"matplotlib"``: Use Matplotlib (raise :class:`ImportError` if not available)
- ``"plotly"``: Use browser-based Plotly (raise :class:`ImportError` if not available)

Selecting Plotly does not change the ``"auto"`` priority.  Install the optional
dependency with ``pip install "sigima[plotly]"``.

.. rubric:: Configuring the Backend

Using environment variable:

.. code-block:: python

    import os
    os.environ["SIGIMA_VIZ_BACKEND"] = "matplotlib"  # Before importing sigima.viz

    from sigima import viz
    # Now uses Matplotlib backend

Using configuration option:

.. code-block:: python

    from sigima.config import options
    options.viz_backend.set("plotpy")

    from sigima import viz
    # Now uses PlotPy backend

Module Attributes
-----------------

.. py:data:: BACKEND_NAME
   :type: str

  Name of the currently selected backend: ``"plotpy"``, ``"matplotlib"``, or
  ``"plotly"``.

.. py:data:: BACKEND_SOURCE
   :type: str

   How the backend was selected: ``"env"``, ``"config"``, or ``"auto"``.

Quick Start
-----------

.. code-block:: python

    from sigima import viz
    import sigima.proc.signal as sips
    from sigima.tests.data import get_test_signal

    # Load a test signal
    signal = get_test_signal("paracetamol.txt")

    # Apply some processing
    filtered = sips.moving_average(signal, n=10)

    # View the results
    viz.view_curves([signal, filtered], title="Signal Processing")

Viewing Functions
-----------------

These functions display Sigima objects (:class:`~sigima.objects.SignalObj` and :class:`~sigima.objects.ImageObj`) in interactive dialogs or plots.

.. autofunction:: view_curves

.. autofunction:: view_images

.. autofunction:: view_images_side_by_side

.. autofunction:: view_curves_and_images

Canonical annotations
---------------------

Object viewing functions render canonical graphical annotations by default.
Pass ``show_annotations=False`` to hide them independently from regions of
interest.  Both backends support all canonical primitives, styles, attached
labels, and layer order.  Text may be positioned in data coordinates or in
normalized axes coordinates.

PlotPy also displays valid historical ``plotpy_json`` payloads without changing
the object.  Matplotlib and Plotly ignore those opaque renderer-specific payloads. Use
the explicit migration described in :ref:`api_annotations` to make historical
annotations portable.

Plotly JSON specifications
--------------------------

The :mod:`sigima.viz.plotly_spec` module builds plain JSON-compatible
``dict``/``list`` structures without importing the Plotly Python package.  These
specifications may be consumed directly by Plotly.js or materialized as
``plotly.graph_objects.Figure`` objects by the Plotly backend.  Overlay builders
are independent from the signal and image arrays so browser applications may
reuse annotations, ROIs, and geometry results without copying large datasets.

.. autofunction:: sigima.viz.plotly_spec.build_curve_figure_spec

.. autofunction:: sigima.viz.plotly_spec.build_image_figure_spec

.. autofunction:: sigima.viz.plotly_spec.build_signal_roi_overlay

.. autofunction:: sigima.viz.plotly_spec.build_image_roi_overlay

.. autofunction:: sigima.viz.plotly_spec.build_geometry_overlay

The autonomous interactive gallery is available from the repository with:

.. code-block:: powershell

  python scripts/run_with_env.py python -m pytest sigima/tests/viz/plotly_gallery_gui_test.py --gui -v

Low-Level Viewing Functions
---------------------------

These functions display plot items (curves, images) rather than Sigima objects.

.. autofunction:: view_curve_items

.. autofunction:: view_image_items

Creation Functions
------------------

These functions create plot items that can be passed to the low-level viewing functions.

Curve and Image Items
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: create_curve

.. autofunction:: create_image

Annotation Items
~~~~~~~~~~~~~~~~

.. autofunction:: create_contour_shapes

.. autofunction:: create_circle

.. autofunction:: create_segment

.. autofunction:: create_cursor

.. autofunction:: create_range

.. autofunction:: create_label

.. autofunction:: create_marker

Backend Differences
-------------------

The three backends have different capabilities:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Feature
     - PlotPy
     - Matplotlib
     - Plotly
   * - Interactive zoom/pan
     - Full Qt tools
     - Basic toolbar
     - Browser tools
   * - ROI display
     - Native support
     - Patches overlay
     - JSON overlays
   * - Geometry results
     - Shape annotations
     - Markers/lines
     - Shapes/traces
   * - Canonical annotations
     - Native interactive items
     - Read-only artists
     - Read-only overlays
   * - Historical PlotPy annotations
     - View-only compatibility
     - Opaque payload ignored
     - Opaque payload ignored
   * - Linked axes
     - Native
     - via ``sharex``/``sharey``
     - Plotly subplots
   * - Qt integration
     - Native
     - Requires Qt backend
     - Not required
   * - Headless/CI
     - Needs display
     - ``Agg`` backend
     - JSON validation

For automated testing and CI environments, Matplotlib with the ``Agg`` backend
or the dependency-free Plotly specifications may be used.  PlotPy provides Qt
editing tools; Plotly provides an interactive browser view with zoom, pan, and
hover but does not edit canonical annotations.
