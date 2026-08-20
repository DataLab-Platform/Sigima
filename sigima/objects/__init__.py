# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Model classes for signals and images (:mod:`sigima.objects`)
------------------------------------------------------------

The :mod:`sigima.objects` module aims at providing all the necessary classes and
functions to create and manipulate Sigima scalar, signal and image objects.

Those classes and functions are defined in submodules:

- :mod:`sigima.objects.base`
- :mod:`sigima.objects.scalar`
- :mod:`sigima.objects.image`
- :mod:`sigima.objects.signal`

.. code-block:: python

    # Full import statement
    from sigima.objects.scalar import GeometryResult, TableResult
    from sigima.objects.signal import SignalObj
    from sigima.objects.image import ImageObj

    # Short import statement
    from sigima.objects import SignalObj, ImageObj, GeometryResult, TableResult

In Sigima, computation functions take signal or image objects as input and produce
signal, image or scalar objects as output. Scalar objects are represented by the
`GeometryResult` and `TableResult` classes.

.. note::

    The scalar results are not rigorously scalar as they can also represent vector of
    coordinates for example, but the name 'scalar' is retained for simplicity and by
    opposition to the more general 'signal' and 'image' terms).

Scalar results
^^^^^^^^^^^^^^

.. autoclass:: sigima.objects.GeometryResult
.. autoclass:: sigima.objects.TableResult

Common features of signals and images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sigima.objects.TypeObj
.. autoclass:: sigima.objects.TypeROI
.. autoclass:: sigima.objects.TypeROIParam
.. autoclass:: sigima.objects.TypeSingleROI
.. autodataset:: sigima.objects.NormalDistributionParam
.. autodataset:: sigima.objects.PoissonDistributionParam
.. autodataset:: sigima.objects.UniformDistributionParam

Signals
^^^^^^^

.. autodataset:: sigima.objects.SignalObj
    :members:
    :inherited-members:
.. autofunction:: sigima.objects.create_signal_roi
.. autofunction:: sigima.objects.create_signal
.. autofunction:: sigima.objects.create_signal_parameters
.. autofunction:: sigima.objects.create_signal_from_param
.. autoclass:: sigima.objects.SignalTypes
.. autoclass:: sigima.objects.NewSignalParam
.. autodataset:: sigima.objects.ZeroParam
.. autodataset:: sigima.objects.UniformDistribution1DParam
.. autodataset:: sigima.objects.NormalDistribution1DParam
.. autodataset:: sigima.objects.PoissonDistribution1DParam
.. autodataset:: sigima.objects.GaussParam
.. autodataset:: sigima.objects.LorentzParam
.. autodataset:: sigima.objects.VoigtParam
.. autodataset:: sigima.objects.SineParam
.. autodataset:: sigima.objects.CosineParam
.. autodataset:: sigima.objects.SawtoothParam
.. autodataset:: sigima.objects.TriangleParam
.. autodataset:: sigima.objects.SquareParam
.. autodataset:: sigima.objects.SincParam
.. autodataset:: sigima.objects.StepParam
.. autodataset:: sigima.objects.ExponentialParam
.. autodataset:: sigima.objects.PulseParam
.. autodataset:: sigima.objects.PolyParam
.. autodataset:: sigima.objects.CustomSignalParam
.. autodataset:: sigima.objects.ROI1DParam
.. autoclass:: sigima.objects.SignalROI

Images
^^^^^^

.. autodataset:: sigima.objects.ImageObj
    :members:
    :inherited-members:
.. autofunction:: sigima.objects.create_image_roi
.. autofunction:: sigima.objects.create_image
.. autofunction:: sigima.objects.create_image_parameters
.. autofunction:: sigima.objects.create_image_from_param
.. autoclass:: sigima.objects.ImageTypes
.. autoclass:: sigima.objects.NewImageParam
.. autodataset:: sigima.objects.Zero2DParam
.. autodataset:: sigima.objects.UniformDistribution2DParam
.. autodataset:: sigima.objects.NormalDistribution2DParam
.. autodataset:: sigima.objects.PoissonDistribution2DParam
.. autodataset:: sigima.objects.Gauss2DParam
.. autodataset:: sigima.objects.Ramp2DParam
.. autodataset:: sigima.objects.Checkerboard2DParam
.. autodataset:: sigima.objects.SinusoidalGrating2DParam
.. autodataset:: sigima.objects.Ring2DParam
.. autodataset:: sigima.objects.SiemensStar2DParam
.. autodataset:: sigima.objects.Sinc2DParam
.. autodataset:: sigima.objects.ROI2DParam
.. autoclass:: sigima.objects.ImageROI
.. autoclass:: sigima.objects.ImageDatatypes
"""

__all__ = [
    "ANNOTATION_FORMAT",
    "ANNOTATION_VERSION",
    "CREATION_PARAMS_VERSION",
    "NO_ROI",
    "PEAK_PARAMETERIZATION",
    "AnnotationKind",
    "AnnotationLabel",
    "AnnotationStyle",
    "AnnotationTransformError",
    "Axis",
    "Checkerboard2DParam",
    "CircleAnnotation",
    "CircularROI",
    "CoordinateSpace",
    "CosineParam",
    "CursorAnnotation",
    "CursorOrientation",
    "CustomSignalParam",
    "EllipseAnnotation",
    "ExponentialParam",
    "FillStyle",
    "Gauss2DParam",
    "GaussParam",
    "GeometryResult",
    "GraphicalAnnotation",
    "ImageDatatypes",
    "ImageObj",
    "ImageROI",
    "ImageTypes",
    "KindShape",
    "LegacyPlotPyMigrationReport",
    "LinearChirpParam",
    "LogisticParam",
    "LorentzParam",
    "MarkerStyle",
    "NewImageParam",
    "NewSignalParam",
    "NormalDistribution1DParam",
    "NormalDistribution2DParam",
    "NormalDistributionParam",
    "PlanckParam",
    "PointAnnotation",
    "PoissonDistribution1DParam",
    "PoissonDistribution2DParam",
    "PoissonDistributionParam",
    "PolyParam",
    "PolygonAnnotation",
    "PolygonalROI",
    "PolylineAnnotation",
    "PulseParam",
    "ROI1DParam",
    "ROI2DParam",
    "Ramp2DParam",
    "RangeAnnotation",
    "RectangleAnnotation",
    "RectangularROI",
    "Ring2DParam",
    "SawtoothParam",
    "SegmentAnnotation",
    "SegmentROI",
    "SiemensStar2DParam",
    "SignalObj",
    "SignalROI",
    "SignalTypes",
    "Sinc2DParam",
    "SincParam",
    "SineParam",
    "SinusoidalGrating2DParam",
    "SquareParam",
    "SquarePulseParam",
    "StepParam",
    "StepPulseParam",
    "StrokeStyle",
    "TableKind",
    "TableResult",
    "TableResultBuilder",
    "TextAnchor",
    "TextAnnotation",
    "TextStyle",
    "TriangleParam",
    "TypeObj",
    "TypeROI",
    "TypeROIParam",
    "TypeSingleROI",
    "UniformDistribution1DParam",
    "UniformDistribution2DParam",
    "UniformDistributionParam",
    "VoigtParam",
    "Zero2DParam",
    "ZeroParam",
    "annotation_from_dict",
    "annotation_to_dict",
    "calc_table_from_data",
    "concat_geometries",
    "concat_tables",
    "convert_legacy_peak_creation_params",
    "create_image",
    "create_image_from_param",
    "create_image_parameters",
    "create_image_roi",
    "create_image_roi_around_points",
    "create_signal",
    "create_signal_from_param",
    "create_signal_parameters",
    "create_signal_roi",
    "filter_geometry_by_roi",
    "filter_table_by_roi",
    "flip_annotation_horizontally",
    "flip_annotation_vertically",
    "is_graphical_annotation_dict",
    "legacy_plotpy_payload_to_annotations",
    "migrate_legacy_plotpy_annotations",
    "rotate_annotation",
    "scale_annotation",
    "transform_annotation",
    "translate_annotation",
    "transpose_annotation",
    "validate_peak_creation_params",
]

from sigima.objects.annotations import (
    ANNOTATION_FORMAT,
    ANNOTATION_VERSION,
    AnnotationKind,
    AnnotationLabel,
    AnnotationStyle,
    AnnotationTransformError,
    Axis,
    CircleAnnotation,
    CoordinateSpace,
    CursorAnnotation,
    CursorOrientation,
    EllipseAnnotation,
    FillStyle,
    GraphicalAnnotation,
    LegacyPlotPyMigrationReport,
    MarkerStyle,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    StrokeStyle,
    TextAnchor,
    TextAnnotation,
    TextStyle,
    annotation_from_dict,
    annotation_to_dict,
    flip_annotation_horizontally,
    flip_annotation_vertically,
    is_graphical_annotation_dict,
    legacy_plotpy_payload_to_annotations,
    migrate_legacy_plotpy_annotations,
    rotate_annotation,
    scale_annotation,
    transform_annotation,
    translate_annotation,
    transpose_annotation,
)
from sigima.objects.base import (
    NormalDistributionParam,
    PoissonDistributionParam,
    TypeObj,
    TypeROI,
    TypeROIParam,
    TypeSingleROI,
    UniformDistributionParam,
)
from sigima.objects.image import (
    Checkerboard2DParam,
    CircularROI,
    Gauss2DParam,
    ImageDatatypes,
    ImageObj,
    ImageROI,
    ImageTypes,
    NewImageParam,
    NormalDistribution2DParam,
    PoissonDistribution2DParam,
    PolygonalROI,
    Ramp2DParam,
    RectangularROI,
    Ring2DParam,
    ROI2DParam,
    SiemensStar2DParam,
    Sinc2DParam,
    SinusoidalGrating2DParam,
    UniformDistribution2DParam,
    Zero2DParam,
    create_image,
    create_image_from_param,
    create_image_parameters,
    create_image_roi,
    create_image_roi_around_points,
)
from sigima.objects.scalar import (
    NO_ROI,
    GeometryResult,
    KindShape,
    TableKind,
    TableResult,
    TableResultBuilder,
    calc_table_from_data,
    concat_geometries,
    concat_tables,
    filter_geometry_by_roi,
    filter_table_by_roi,
)
from sigima.objects.signal import (
    CREATION_PARAMS_VERSION,
    PEAK_PARAMETERIZATION,
    CosineParam,
    CustomSignalParam,
    ExponentialParam,
    GaussParam,
    LinearChirpParam,
    LogisticParam,
    LorentzParam,
    NewSignalParam,
    NormalDistribution1DParam,
    PlanckParam,
    PoissonDistribution1DParam,
    PolyParam,
    PulseParam,
    ROI1DParam,
    SawtoothParam,
    SegmentROI,
    SignalObj,
    SignalROI,
    SignalTypes,
    SincParam,
    SineParam,
    SquareParam,
    SquarePulseParam,
    StepParam,
    StepPulseParam,
    TriangleParam,
    UniformDistribution1DParam,
    VoigtParam,
    ZeroParam,
    convert_legacy_peak_creation_params,
    create_signal,
    create_signal_from_param,
    create_signal_parameters,
    create_signal_roi,
    validate_peak_creation_params,
)
