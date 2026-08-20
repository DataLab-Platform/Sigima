:orphan:

.. _api_annotations:

Graphical annotations
=====================

Sigima provides a renderer-independent model for editorial graphics attached to
signals and images.  Graphical annotations are distinct from two other concepts:

- a region of interest selects samples or pixels for computation;
- a :class:`~sigima.objects.GeometryResult` stores an analysis result;
- a graphical annotation communicates information to a reader and may be edited
  by a consuming application.

Model
-----

The canonical model supports points, segments, oriented rectangles, circles,
oriented ellipses, polylines, polygons, text, axis or crosshair cursors, and
axis ranges.  All geometry is expressed in calibrated data coordinates.  Text
may instead use normalized axes coordinates, from ``(0, 0)`` at the bottom left
to ``(1, 1)`` at the top right.

Annotations are immutable dataclasses.  Their common fields include a stable
UUID, visibility, locking, layer order, title, structured style, optional label,
metadata, and namespaced extensions.  Metadata and extensions accept only
JSON-compatible values and are copied into immutable containers.

.. autoclass:: sigima.objects.GraphicalAnnotation
.. autoclass:: sigima.objects.PointAnnotation
.. autoclass:: sigima.objects.SegmentAnnotation
.. autoclass:: sigima.objects.RectangleAnnotation
.. autoclass:: sigima.objects.CircleAnnotation
.. autoclass:: sigima.objects.EllipseAnnotation
.. autoclass:: sigima.objects.PolylineAnnotation
.. autoclass:: sigima.objects.PolygonAnnotation
.. autoclass:: sigima.objects.TextAnnotation
.. autoclass:: sigima.objects.CursorAnnotation
.. autoclass:: sigima.objects.RangeAnnotation

Object API
----------

The typed API is parallel to the historical free-form JSON API.  This preserves
application-specific entries while allowing portable annotations to coexist in
the same ``annotations`` field.

.. code-block:: python

    from sigima.objects import PointAnnotation, create_signal

    signal = create_signal("Annotated signal", [0, 1], [2, 3])
    signal.add_graphical_annotation(
        PointAnnotation(x=1.0, y=3.0, title="Maximum")
    )

    annotations = signal.get_graphical_annotations()
    signal.set_graphical_annotations(annotations, preserve_opaque=True)

The methods ``get_annotations()`` and ``set_annotations()`` retain their
existing free-form behavior.  ``set_graphical_annotations()`` replaces only
canonical entries by default.  PlotPy payloads and unknown consumer data remain
unchanged.  An entry declaring the canonical format but using an unsupported
version raises an error instead of being silently treated as opaque.

Serialization and files
-----------------------

Each canonical dictionary is marked with ``format: "sigima.annotation"`` and
``version: "1.0"``.  The versioned JSON Schema is distributed as
``sigima/objects/annotations/schema-v1.json``.  It is independent from the
historical object wrapper version and from the ``.dlabann`` container version.

.. autofunction:: sigima.objects.annotation_to_dict
.. autofunction:: sigima.objects.annotation_from_dict
.. autofunction:: sigima.io.write_graphical_annotations
.. autofunction:: sigima.io.read_graphical_annotations

Canonical annotations survive object copies and the normal ``.h5sig``,
``.h5ima``, and ``.dlabann`` round trips without a renderer dependency.

Transformations
---------------

Pure transformation functions return a new annotation and preserve its UUID
and non-geometric fields.  Translation, quarter turns, flips, transposition,
and scaling are also applied by the corresponding image operations.  Resizing
does not move annotations because their coordinates are calibrated data values.
Arbitrary image rotation clears canonical annotations, like regions of interest,
when the output coordinate mapping is not reliable; opaque payloads are kept.

.. autofunction:: sigima.objects.translate_annotation
.. autofunction:: sigima.objects.rotate_annotation
.. autofunction:: sigima.objects.flip_annotation_horizontally
.. autofunction:: sigima.objects.flip_annotation_vertically
.. autofunction:: sigima.objects.transpose_annotation
.. autofunction:: sigima.objects.scale_annotation

An exact transform may change the primitive type, for example from a circle to
an ellipse under anisotropic scaling.  A transform that cannot be represented
exactly raises :class:`~sigima.objects.AnnotationTransformError`.

PlotPy migration
----------------

The PlotPy backend can display historical ``plotpy_json`` payloads without
rewriting them.  Migration to the canonical model is always explicit:

.. code-block:: python

    from sigima.viz.annotation_plotpy import migrate_legacy_plotpy_annotations

    preview = migrate_legacy_plotpy_annotations(signal, dry_run=True)
    if not preview.diagnostics:
        report = migrate_legacy_plotpy_annotations(signal)

Known PlotPy annotation types are converted.  A malformed payload, an unknown
item class, or a payload containing a partially unsupported item is preserved
and reported.  Running migration again is idempotent.
