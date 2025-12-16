# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Base computation module
-----------------------

This module provides core classes and utility functions that serve as building blocks
for the other computation modules.

Main features include:

- Generic helper functions used across image processing modules
- Core wrappers and infrastructure for computation functions

Intended primarily for internal use, these tools support consistent API design
and code reuse.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from sigima.objects import NO_ROI, GeometryResult, ImageObj, KindShape, SignalObj
from sigima.proc.base import dst_1_to_1 as _dst_1_to_1_base
from sigima.proc.base import dst_2_to_1 as _dst_2_to_1_base
from sigima.proc.base import dst_n_to_1 as _dst_n_to_1_base
from sigima.proc.base import new_signal_result

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "Wrap1to1Func",
    "compute_geometry_from_obj",
    "dst_1_to_1",
    "dst_1_to_1_signal",
    "dst_2_to_1",
    "dst_n_to_1",
    "restore_data_outside_roi",
]


def _reset_lut_range(dst: ImageObj) -> None:
    """Reset LUT range so the image auto-scales on display."""
    dst.zscalemin = None
    dst.zscalemax = None


def dst_1_to_1(src: ImageObj, name: str, suffix: str | None = None) -> ImageObj:
    """Create a result image object for 1-to-1 processing.

    This is the image-specific version that resets the LUT range after copying,
    so the result image auto-scales to fit its new data range.

    Args:
        src: source image object
        name: name of the function
        suffix: suffix to add to the title

    Returns:
        Result image object with LUT range reset
    """
    dst = _dst_1_to_1_base(src, name, suffix)
    _reset_lut_range(dst)
    return dst


def dst_2_to_1(
    src1: ImageObj, src2: ImageObj, name: str, suffix: str | None = None
) -> ImageObj:
    """Create a result image object for 2-to-1 processing.

    This is the image-specific version that resets the LUT range after copying,
    so the result image auto-scales to fit its new data range.

    Args:
        src1: first source image object
        src2: second source image object
        name: name of the function
        suffix: suffix to add to the title

    Returns:
        Result image object with LUT range reset
    """
    dst = _dst_2_to_1_base(src1, src2, name, suffix)
    _reset_lut_range(dst)
    return dst


def dst_n_to_1(
    src_list: list[ImageObj], name: str, suffix: str | None = None
) -> ImageObj:
    """Create a result image object for n-to-1 processing.

    This is the image-specific version that resets the LUT range after copying,
    so the result image auto-scales to fit its new data range.

    Args:
        src_list: list of source image objects
        name: name of the function
        suffix: suffix to add to the title

    Returns:
        Result image object with LUT range reset
    """
    dst = _dst_n_to_1_base(src_list, name, suffix)
    _reset_lut_range(dst)
    return dst


def restore_data_outside_roi(dst: ImageObj, src: ImageObj) -> None:
    """Restore data outside the Region Of Interest (ROI) of the input image
    after a computation, only if the input image has a ROI,
    and if the output image has the same ROI as the input image,
    and if the data types are compatible,
    and if the shapes are the same.
    Otherwise, do nothing.

    Args:
        dst: output image object
        src: input image object
    """
    if src.maskdata is not None and dst.maskdata is not None:
        if (
            np.array_equal(src.maskdata, dst.maskdata)
            and (
                dst.data.dtype == src.data.dtype
                or not np.issubdtype(dst.data.dtype, np.integer)
            )
            and dst.data.shape == src.data.shape
        ):
            dst.data[src.maskdata] = src.data[src.maskdata]


class Wrap1to1Func:
    """Wrap a 1 array → 1 array function to produce a 1 image → 1 image function,
    which can be used as a Sigima computation function and inside DataLab's
    infrastructure to perform computations with the Image Processor object.

    This wrapping mechanism using a class is necessary for the resulted function to be
    pickable by the ``multiprocessing`` module.

    The instance of this wrapper is callable and returns a
    :class:`sigima.objects.ImageObj` object.

    Example:

        >>> import numpy as np
        >>> from sigima.proc.image import Wrap1to1Func
        >>> import sigima.objects
        >>> def add_noise(data):
        ...     return data + np.random.random(data.shape)
        >>> compute_add_noise = Wrap1to1Func(add_noise)
        >>> data= np.ones((100, 100))
        >>> ima0 = sigima.objects.create_image("Example", data)
        >>> ima1 = compute_add_noise(ima0)

    Args:
        func: 1 array → 1 array function
        *args: Additional positional arguments to pass to the function
        **kwargs: Additional keyword arguments to pass to the function

    .. note::

        If `func_name` is provided in the keyword arguments, it will be used as the
        function name instead of the default name derived from the function itself.
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.__name__ = self.kwargs.pop("func_name", func.__name__)
        self.__doc__ = func.__doc__
        self.__call__.__func__.__doc__ = self.func.__doc__

    def __call__(self, src: ImageObj) -> ImageObj:
        """Compute the function on the input image and return the result image

        Args:
            src: input image object

        Returns:
            Output image object
        """
        suffix = ", ".join(
            [str(arg) for arg in self.args]
            + [f"{k}={v}" for k, v in self.kwargs.items() if v is not None]
        )
        dst = dst_1_to_1(src, self.__name__, suffix)
        dst.data = self.func(src.data, *self.args, **self.kwargs)
        restore_data_outside_roi(dst, src)
        return dst


def dst_1_to_1_signal(src: ImageObj, name: str, suffix: str | None = None) -> SignalObj:
    """Create a result signal object, for processing functions that take a single
    image object as input and return a single signal object (1-to-1-signal).

    Args:
        src: input image object
        name: name of the processing function

    Returns:
        Output signal object
    """
    return new_signal_result(
        src, name, suffix, (src.xunit, src.zunit), (src.xlabel, src.zlabel)
    )


def compute_geometry_from_obj(
    title: str,
    shape: KindShape,
    obj: ImageObj,
    func: Callable,
    *args: Any,
) -> GeometryResult | None:
    """Compute a geometry shape from an image object by executing a computation function
    on the data of the image object, for each ROI (Region Of Interest) in the image.

    Args:
        title: result title
        shape: result shape kind
        obj: input image object
        func: computation function
        *args: computation function arguments

    Returns:
        A geometry result object or None if no result is found.

    .. important::
        **Coordinate Conversion**: This function automatically converts coordinates
        from pixel units (image indices) to physical units using the image object's
        calibration information.

        - **Input**: Computation function returns coordinates in pixel units
        - **Output**: GeometryResult with coordinates in physical units (e.g., mm, µm)

        The conversion is performed using the image's calibration parameters:
        ``physical_x = obj.dx * pixel_x + obj.x0`` and
        ``physical_y = obj.dy * pixel_y + obj.y0``

    .. warning::

        The computation function must take either a single argument (the data) or
        multiple arguments (the data followed by the computation parameters).

        Moreover, the computation function must return a single value or a NumPy array
        containing the result of the computation. This array contains the coordinates
        of points, polygons, circles or ellipses in the form [[x, y], ...], or
        [[x0, y0, x1, y1, ...], ...], or [[x0, y0, r], ...], or
        [[x0, y0, a, b, theta], ...].

    Example:
        >>> # func returns pixel coordinates like [[10, 20], [30, 40]]
        >>> result = compute_geometry_from_obj(
        ...     "Points", KindShape.POINT, image_obj, func
        ... )
        >>> # result.coords now contains physical coordinates like [[0.5, 1.0],
        >>> # [1.5, 2.0]]

    See Also:
        :class:`~sigima.objects.scalar.GeometryResult`: The result object that stores
        physical coordinates.
    """
    rows: list[np.ndarray] = []
    num_cols: list[int] = []
    roi_idx: list[int] = []
    for i_roi in obj.iterate_roi_indices():
        data_roi = obj.get_data(i_roi)
        if args is None:
            coords: np.ndarray = func(data_roi)
        else:
            coords: np.ndarray = func(data_roi, *args)

        # This is a very long condition, but it's still quite readable, so we keep it
        # as is and disable the pylint warning.
        #
        # pylint: disable=too-many-boolean-expressions
        if not isinstance(coords, np.ndarray) or (
            (
                coords.ndim != 2
                or coords.shape[1] < 2
                or (coords.shape[1] > 5 and coords.shape[1] % 2 != 0)
            )
            and coords.size > 0
        ):
            raise ValueError(
                f"Computation function {func.__name__} must return a NumPy array "
                f"containing coordinates of points, polygons, circles or ellipses "
                f"(in the form [[x, y], ...], or [[x0, y0, x1, y1, ...], ...], or "
                f"[[x0, y0, r], ...], or [[x0, y0, a, b, theta], ...]), or an empty "
                f"array."
            )

        if coords.size:
            coords = np.array(coords, dtype=float)
            if coords.shape[1] % 2 == 0:
                # Coordinates are in the form [x0, y0, x1, y1, ...]
                colx, coly = slice(None, None, 2), slice(1, None, 2)
            else:
                # Circle [x0, y0, r] or ellipse coordinates [x0, y0, a, b, theta]
                colx, coly = 0, 1
            coords[:, colx] = obj.dx * coords[:, colx] + obj.x0
            coords[:, coly] = obj.dy * coords[:, coly] + obj.y0
            if obj.roi is not None:
                x0, y0, _x1, _y1 = obj.roi.get_single_roi(i_roi).get_bounding_box(obj)
                coords[:, colx] += x0 - obj.x0
                coords[:, coly] += y0 - obj.y0

            rows.append(coords)
            num_cols.append(coords.shape[1])
            roi_idx.extend([NO_ROI if i_roi is None else int(i_roi)] * coords.shape[0])
    if rows:
        if len(set(num_cols)) != 1:
            # This happens when the number of columns is not the same for all ROIs.
            # As of now, this happens only for polygon contours.
            # We need to pad the arrays with NaNs.
            max_cols = max(num_cols)
            num_rows = sum(coords.shape[0] for coords in rows)
            array = np.full((num_rows, max_cols), np.nan)
            start = 0
            for row in rows:
                array[start : start + row.shape[0], : row.shape[1]] = row
                start += row.shape[0]
        else:
            array = np.vstack(rows)
        return GeometryResult(title, shape, array, np.asarray(roi_idx, dtype=int))
    return None
