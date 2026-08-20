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

The module automatically selects between **PlotPy** and **Matplotlib** backends based on availability and configuration settings.

The backend selection follows this priority:

1. Environment variable ``SIGIMA_VIZ_BACKEND`` (if set)
2. Configuration option :attr:`sigima.config.options.viz_backend`
3. Auto-detection (PlotPy preferred, Matplotlib as fallback)

Backend selection logic:

- ``"auto"``: Try PlotPy first, fall back to Matplotlib
- ``"plotpy"``: Use PlotPy (raise :class:`ImportError` if not available)
- ``"matplotlib"``: Use Matplotlib (raise :class:`ImportError` if not available)

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

   Name of the currently selected backend: ``"plotpy"`` or ``"matplotlib"``.

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
the object.  Matplotlib ignores those opaque renderer-specific payloads.  Use
the explicit migration described in :ref:`api_annotations` to make historical
annotations portable.

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

The two backends have different capabilities:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Feature
     - PlotPy
     - Matplotlib
   * - Interactive zoom/pan
     - ✅ Full Qt tools
     - ✅ Basic toolbar
   * - ROI display
     - ✅ Native support
     - ✅ Patches overlay
   * - Geometry results
     - ✅ Shape annotations
     - ✅ Markers/lines
   * - Canonical annotations
     - ✅ Native interactive items
     - ✅ Read-only artists
   * - Historical PlotPy annotations
     - ✅ View-only compatibility
     - ❌ Opaque payload ignored
   * - Linked axes
     - ✅ Native
     - ✅ via ``sharex``/``sharey``
   * - Qt integration
     - ✅ Native
     - ⚠️ Requires Qt backend
   * - Headless/CI
     - ⚠️ Needs display
     - ✅ ``Agg`` backend

For automated testing and CI environments, Matplotlib with the ``Agg`` backend is recommended. For interactive data exploration, PlotPy provides a richer experience.
