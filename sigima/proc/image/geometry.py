# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry computation module
---------------------------

This module implements geometric transformations and manipulations for images,
such as rotations, flips, resizing, axis swapping, binning, and padding.

Main features include:

- Rotation by arbitrary or fixed angles
- Horizontal and vertical flipping
- Resizing and binning of images
- Axis swapping and zero padding

These functions are useful for preparing and augmenting image data.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
import scipy.ndimage as spi

from sigima.config import _
from sigima.enums import BorderMode, Interpolation2DMethod
from sigima.objects.image import ImageObj
from sigima.proc.contracts import EXACT, ReplayQualification
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import dst_1_to_1, restore_data_outside_roi
from sigima.proc.image.transformations import transformer

# NOTE: Only parameter classes DEFINED in this module should be included in __all__.
# Parameter classes imported from other modules (like sigima.proc.base) should NOT
# be re-exported to avoid Sphinx cross-reference conflicts. The sigima.params module
# serves as the central API point that imports and re-exports all parameter classes.
__all__ = [
    "Resampling2DParam",
    "ResizeParam",
    "RotateParam",
    "TranslateParam",
    "UniformCoordsParam",
    "XYZCalibrateParam",
    "calibration",
    "fliph",
    "flipv",
    "resampling",
    "resize",
    "rotate",
    "rotate90",
    "rotate270",
    "set_uniform_coords",
    "translate",
    "transpose",
]


class TranslateParam(gds.DataSet):
    """Translate parameters"""

    dx = gds.FloatItem(_("X translation"), default=0.0)
    dy = gds.FloatItem(_("Y translation"), default=0.0)


@computation_function(
    operation_id="sigima.image.translate",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def translate(src: ImageObj, p: TranslateParam) -> ImageObj:
    """Translate data with :py:func:`scipy.ndimage.shift`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "translate", f"dx={p.dx}, dy={p.dy}")
    if src.is_uniform_coords:
        dst.set_uniform_coords(dst.dx, dst.dy, dst.x0 + p.dx, dst.y0 + p.dy)
    else:
        dst.set_coords(src.xcoords + p.dx, src.ycoords + p.dy)
    transformer.transform_roi(dst, "translate", dx=p.dx, dy=p.dy)
    transformer.transform_annotations(dst, "translate", dx=p.dx, dy=p.dy)
    return dst


class RotateParam(gds.DataSet):
    """Rotate parameters"""

    prop = gds.ValueProp(False)

    angle = gds.FloatItem(f"{_('Angle')} (°)", default=0.0)
    mode = gds.ChoiceItem(_("Mode"), BorderMode, default=BorderMode.CONSTANT)
    cval = gds.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    reshape = gds.BoolItem(
        _("Reshape the output array"),
        default=False,
        help=_(
            "Reshape the output array "
            "so that the input array is "
            "contained completely in the output"
        ),
    )
    prefilter = gds.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gds.IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


@computation_function(
    operation_id="sigima.image.rotate",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def rotate(src: ImageObj, p: RotateParam) -> ImageObj:
    """Rotate data with :py:func:`scipy.ndimage.rotate`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object

    Raises:
        ValueError: if source image has non-uniform coordinates
    """
    if not src.is_uniform_coords:
        # An arbitrary rotation of the pixels does not map a non-uniform grid onto
        # a grid: the image must be given uniform coordinates first
        raise ValueError("Source image must have uniform coordinates for rotation")
    dst = dst_1_to_1(src, "rotate", f"α={p.angle:.3f}°, mode='{p.mode}'")
    dst.data = spi.rotate(
        src.data,
        p.angle,
        reshape=p.reshape,
        order=p.order,
        mode=p.mode,
        cval=p.cval,
        prefilter=p.prefilter,
    )
    dst.roi = None  # Reset ROI as it may change after rotation
    dst.clear_graphical_annotations()
    return dst


def __swap_axes(src: ImageObj, dst: ImageObj) -> None:
    """Exchange the X and Y axes of ``dst``, which holds transposed ``src`` data.

    Pixel sizes, origins, labels and units all belong to a given axis: when the data
    array is transposed, they must follow, otherwise an anisotropic image is silently
    rescaled and the axis annotations describe the wrong direction.

    Args:
        src: input image object
        dst: output image object, whose data is already transposed
    """
    dst.xlabel = src.ylabel
    dst.ylabel = src.xlabel
    dst.xunit = src.yunit
    dst.yunit = src.xunit
    if src.is_uniform_coords:
        dst.set_uniform_coords(src.dy, src.dx, src.y0, src.x0)
    else:
        dst.set_coords(src.ycoords.copy(), src.xcoords.copy())


@computation_function(
    operation_id="sigima.image.rotate90",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def rotate90(src: ImageObj) -> ImageObj:
    """Rotate data 90° with :py:func:`numpy.rot90`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "rotate90")
    dst.data = np.rot90(src.data).copy()
    __swap_axes(src, dst)
    # ``numpy.rot90(a)`` is ``numpy.flipud(a.T)``: composing the ROI mapping the same
    # way keeps it aligned with the pixels, which an arbitrary rotation of the ROI
    # shapes cannot do (a rotated rectangle is no longer a rectangle).
    transformer.apply_image_operation(dst, src, "rotate90")
    return dst


@computation_function(
    operation_id="sigima.image.rotate270",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def rotate270(src: ImageObj) -> ImageObj:
    """Rotate data 270° with :py:func:`numpy.rot90`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "rotate270")
    dst.data = np.rot90(src.data, 3).copy()
    __swap_axes(src, dst)
    # ``numpy.rot90(a, 3)`` is ``numpy.fliplr(a.T)`` (see :func:`rotate90`).
    transformer.apply_image_operation(dst, src, "rotate270")
    return dst


@computation_function(
    operation_id="sigima.image.fliph",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def fliph(src: ImageObj) -> ImageObj:
    """Flip data horizontally with :py:func:`numpy.fliplr`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "fliph")
    dst.data = np.fliplr(src.data).copy()
    transformer.apply_image_operation(dst, src, "fliph")
    return dst


@computation_function(
    operation_id="sigima.image.flipv",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def flipv(src: ImageObj) -> ImageObj:
    """Flip data vertically with :py:func:`numpy.flipud`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "flipv")
    dst.data = np.flipud(src.data).copy()
    transformer.apply_image_operation(dst, src, "flipv")
    return dst


class ResizeParam(gds.DataSet):
    """Resize parameters"""

    prop = gds.ValueProp(False)

    zoom = gds.FloatItem(_("Zoom"), default=1.0, min=0.0, nonzero=True)
    mode = gds.ChoiceItem(_("Mode"), BorderMode, default=BorderMode.CONSTANT)
    cval = gds.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    prefilter = gds.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gds.IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


@computation_function(
    operation_id="sigima.image.resize",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def resize(src: ImageObj, p: ResizeParam) -> ImageObj:
    """Zooming function with :py:func:`scipy.ndimage.zoom`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object

    Raises:
        ValueError: if source image has non-uniform coordinates
    """
    if not src.is_uniform_coords:
        raise ValueError("Source image must have uniform coordinates for resampling")
    mode = p.mode
    dst = dst_1_to_1(src, "resize", f"zoom={p.zoom:.3f}")
    dst.data = spi.zoom(
        src.data,
        p.zoom,
        order=p.order,
        mode=mode,
        cval=p.cval,
        prefilter=p.prefilter,
    )
    if not np.isnan(dst.dx) and not np.isnan(dst.dy):
        dst.set_uniform_coords(dst.dx / p.zoom, dst.dy / p.zoom, dst.x0, dst.y0)
    return dst


@computation_function(
    operation_id="sigima.image.transpose",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def transpose(src: ImageObj) -> ImageObj:
    """Transpose image with :py:func:`numpy.transpose`.

    Args:
        src: Input image object.

    Returns:
        Output image object.
    """
    dst = dst_1_to_1(src, "transpose")
    dst.data = np.transpose(src.data).copy()
    __swap_axes(src, dst)
    transformer.apply_image_operation(dst, src, "transpose")
    return dst


class Resampling2DParam(gds.DataSet):
    """Resample parameters for 2D images

    ``xmin``/``xmax`` and ``ymin``/``ymax`` are the coordinates of the centers of
    the first and last output pixels, like :attr:`ImageObj.xmin` and
    :attr:`ImageObj.xmax`: with the default values, the output grid is the source
    grid.
    """

    # Output coordinate system
    xmin = gds.FloatItem(
        "X<sub>min</sub>",
        default=None,
        allow_none=True,
        help=_("X-coordinate of the center of the first output pixel"),
    )
    xmax = gds.FloatItem(
        "X<sub>max</sub>",
        default=None,
        allow_none=True,
        help=_("X-coordinate of the center of the last output pixel"),
    )
    ymin = gds.FloatItem(
        "Y<sub>min</sub>",
        default=None,
        allow_none=True,
        help=_("Y-coordinate of the center of the first output pixel"),
    )
    ymax = gds.FloatItem(
        "Y<sub>max</sub>",
        default=None,
        allow_none=True,
        help=_("Y-coordinate of the center of the last output pixel"),
    )

    # Mode selection
    _prop = gds.GetAttrProp("mode")
    _modes = (("dxy", _("Pixel size")), ("shape", _("Output shape")))
    mode = gds.ChoiceItem(_("Mode"), _modes, default="shape", radio=True).set_prop(
        "display", store=_prop
    )

    # Pixel size mode parameters
    dx = gds.FloatItem(
        "ΔX",
        default=None,
        allow_none=True,
        help=_("Pixel size in X direction"),
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "dxy"))
    dy = gds.FloatItem(
        "ΔY",
        default=None,
        allow_none=True,
        help=_("Pixel size in Y direction"),
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "dxy"))

    # Shape mode parameters
    width = gds.IntItem(
        _("Width"),
        default=None,
        allow_none=True,
        help=_("Output image width in pixels"),
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "shape"))
    height = gds.IntItem(
        _("Height"),
        default=None,
        allow_none=True,
        help=_("Output image height in pixels"),
    ).set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "shape"))

    # Interpolation parameters
    method = gds.ChoiceItem(
        _("Interpolation method"),
        Interpolation2DMethod,
        default=Interpolation2DMethod.LINEAR,
    )
    fill_value = gds.FloatItem(
        _("Fill value"),
        default=None,
        help=_(
            "Value to use for points outside the input image domain. "
            "If None, uses NaN for extrapolation."
        ),
        check=False,
    )

    def validate_parameters(self, *context: object) -> None:
        """Validate output bounds and the active sampling mode."""
        if not context or not isinstance(context[0], ImageObj):
            raise ValueError("resampling validation requires a source ImageObj")
        xmin, xmax, ymin, ymax = _resampling_bounds(context[0], self)
        x_extent = xmax - xmin
        y_extent = ymax - ymin

        if self.mode == "dxy":
            if self.dx is None or self.dy is None:
                raise ValueError("dx and dy must be specified in pixel size mode")
            if self.dx == 0.0 or self.dy == 0.0:
                raise ValueError("dx and dy must be nonzero in pixel size mode")
            x_sign_mismatch = x_extent != 0.0 and (x_extent < 0.0) != (self.dx < 0.0)
            y_sign_mismatch = y_extent != 0.0 and (y_extent < 0.0) != (self.dy < 0.0)
            if x_sign_mismatch or y_sign_mismatch:
                raise ValueError("pixel sizes must have the same sign as their extents")
        elif self.width is None or self.height is None:
            raise ValueError("width and height must be specified in shape mode")
        elif self.width < 1 or self.height < 1:
            raise ValueError("width and height must be at least 1 in shape mode")
        elif (x_extent == 0.0 and self.width > 1) or (
            y_extent == 0.0 and self.height > 1
        ):
            raise ValueError(
                "resampling coordinate extents must be nonzero "
                "for more than one output pixel"
            )

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from an image object."""
        xmin, xmax, ymin, ymax = _resampling_bounds(obj, self)
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        if self.dx is None:
            self.dx = obj.dx
        if self.dy is None:
            self.dy = obj.dy
        if self.width is None:
            self.width = obj.data.shape[1]
        if self.height is None:
            self.height = obj.data.shape[0]


def _resampling_bounds(
    src: ImageObj, p: Resampling2DParam
) -> tuple[float, float, float, float]:
    """Return the output bounds, the missing ones being those of the source.

    Bounds are the coordinates of the first and last pixel centers: by default,
    the output grid is the source grid.

    Args:
        src: source image
        p: resampling parameters

    Returns:
        Tuple (xmin, xmax, ymin, ymax)
    """
    nrows, ncols = src.data.shape
    xmin = p.xmin if p.xmin is not None else src.x0
    xmax = p.xmax if p.xmax is not None else src.x0 + (ncols - 1) * src.dx
    ymin = p.ymin if p.ymin is not None else src.y0
    ymax = p.ymax if p.ymax is not None else src.y0 + (nrows - 1) * src.dy
    return xmin, xmax, ymin, ymax


def _resampling_axis(
    vmin: float,
    vmax: float,
    count: int | None,
    step: float | None,
    default_step: float,
) -> tuple[np.ndarray, float]:
    """Return the pixel center coordinates and pixel size of one output axis.

    Args:
        vmin: coordinate of the first pixel center
        vmax: coordinate of the last pixel center
        count: number of pixels (output shape mode), or None
        step: pixel size (pixel size mode), or None
        default_step: pixel size of a single-pixel output over an empty extent

    Returns:
        Tuple (coordinates, pixel size)
    """
    if step is not None:
        # Pixel size mode: the pixel size is exact, the last pixel center is the
        # last one not beyond ``vmax`` (rounding absorbs floating-point noise)
        count = int(np.floor(np.round((vmax - vmin) / step, 9))) + 1
        return vmin + np.arange(count) * step, step
    assert count is not None
    if count == 1:
        size = vmax - vmin if vmax != vmin else default_step
        return np.array([(vmin + vmax) / 2.0]), size
    size = (vmax - vmin) / (count - 1)
    return vmin + np.arange(count) * size, size


@computation_function(
    operation_id="sigima.image.resampling",
    contract_version=2,
    qualification=ReplayQualification(contract_version=2, comparison=EXACT),
)
def resampling(src: ImageObj, p: Resampling2DParam) -> ImageObj:
    """Resample image to new coordinate grid using interpolation

    Args:
        src: source image
        p: resampling parameters

    Returns:
        Resampled image object

    Raises:
        ValueError: if source image has non-uniform coordinates
    """
    if not src.is_uniform_coords:
        raise ValueError("Source image must have uniform coordinates for resampling")

    # Output bounds are the coordinates of the first and last pixel centers
    output_xmin, output_xmax, output_ymin, output_ymax = _resampling_bounds(src, p)

    # Determine output grid parameters
    method: Interpolation2DMethod = p.method
    fill_suffix = f", fill_value={p.fill_value}" if p.fill_value is not None else ""
    if p.mode == "dxy":
        if p.dx is None or p.dy is None:
            raise ValueError("dx and dy must be specified in pixel size mode")
        out_x, output_dx = _resampling_axis(
            output_xmin, output_xmax, None, p.dx, src.dx
        )
        out_y, output_dy = _resampling_axis(
            output_ymin, output_ymax, None, p.dy, src.dy
        )
        suffix = f"method={method.value}, dx={p.dx:.3f}, dy={p.dy:.3f}{fill_suffix}"
    else:
        if p.width is None or p.height is None:
            raise ValueError("width and height must be specified in shape mode")
        out_x, output_dx = _resampling_axis(
            output_xmin, output_xmax, p.width, None, src.dx
        )
        out_y, output_dy = _resampling_axis(
            output_ymin, output_ymax, p.height, None, src.dy
        )
        suffix = f"method={method.value}, size=({p.width}x{p.height}){fill_suffix}"

    # Create destination image
    dst = dst_1_to_1(src, "resample", suffix)

    # Create meshgrids
    out_X, out_Y = np.meshgrid(out_x, out_y, indexing="xy")

    # Convert interpolation method to scipy parameter
    if method == Interpolation2DMethod.LINEAR:
        order = 1
    elif method == Interpolation2DMethod.CUBIC:
        order = 3
    elif method == Interpolation2DMethod.NEAREST:
        order = 0
    else:
        order = 1  # fallback to linear

    # Convert physical coordinates to source image indices (pixel ``i`` is
    # centered on ``x0 + i * dx``); indices within rounding noise of an integer
    # are snapped to it, so that grid points coinciding with source pixels are
    # neither interpolated nor considered out of bounds on the edges
    src_i = (out_X - src.x0) / src.dx
    src_j = (out_Y - src.y0) / src.dy
    for indices in (src_i, src_j):
        rounded = np.round(indices)
        snap = np.isclose(indices, rounded, rtol=0.0, atol=1e-9)
        indices[snap] = rounded[snap]

    # Perform interpolation using map_coordinates
    # Note: map_coordinates expects (j, i) order (row, col)
    coordinates = np.array([src_j.ravel(), src_i.ravel()])

    # Determine fill value for interpolation
    cval = p.fill_value if p.fill_value is not None else np.nan

    # For NaN fill values, the data must be able to hold NaN: integer data is
    # converted to float (floating-point and complex data are kept as they are)
    if np.isnan(cval) and not np.issubdtype(src.data.dtype, np.inexact):
        input_data = src.data.astype(np.float64)
    else:
        input_data = src.data

    # Interpolate
    resampled_data = spi.map_coordinates(
        input_data, coordinates, order=order, mode="constant", cval=cval, prefilter=True
    ).reshape(out_y.size, out_x.size)

    # Set output data and coordinate system
    dst.data = resampled_data
    dst.set_uniform_coords(output_dx, output_dy, out_x[0], out_y[0])

    return dst


class UniformCoordsParam(gds.DataSet):
    """Uniform coordinates parameters"""

    x0 = gds.FloatItem("X<sub>0</sub>", default=0.0, help=_("Origin X-axis coordinate"))
    y0 = gds.FloatItem("Y<sub>0</sub>", default=0.0, help=_("Origin Y-axis coordinate"))
    dx = gds.FloatItem(
        "Δx", default=1.0, nonzero=True, help=_("Pixel size along X-axis")
    )
    dy = gds.FloatItem(
        "Δy", default=1.0, nonzero=True, help=_("Pixel size along Y-axis")
    )

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update default values from image object's non-uniform coordinates.

        This method extracts uniform coordinate approximations from non-uniform
        coordinate arrays, handling numerical precision issues that may arise
        from arrays created using linspace.

        Args:
            obj: Image object with non-uniform coordinates
        """
        if obj.is_uniform_coords:
            # Already uniform, just copy the values
            self.x0 = obj.x0
            self.y0 = obj.y0
            self.dx = obj.dx
            self.dy = obj.dy
        else:
            # Extract from non-uniform coordinates
            if obj.xcoords is not None and len(obj.xcoords) >= 2:
                self.x0 = float(obj.xcoords[0])
                # Calculate dx with rounding to handle numerical precision
                dx_raw = (obj.xcoords[-1] - obj.xcoords[0]) / (len(obj.xcoords) - 1)
                # Round to reasonable precision (12 decimal places)
                self.dx = float(np.round(dx_raw, 12))
            else:
                self.x0 = 0.0
                self.dx = 1.0

            if obj.ycoords is not None and len(obj.ycoords) >= 2:
                self.y0 = float(obj.ycoords[0])
                # Calculate dy with rounding to handle numerical precision
                dy_raw = (obj.ycoords[-1] - obj.ycoords[0]) / (len(obj.ycoords) - 1)
                # Round to reasonable precision (12 decimal places)
                self.dy = float(np.round(dy_raw, 12))
            else:
                self.y0 = 0.0
                self.dy = 1.0


@computation_function(
    operation_id="sigima.image.set_uniform_coords",
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def set_uniform_coords(src: ImageObj, p: UniformCoordsParam) -> ImageObj:
    """Convert image to uniform coordinate system

    Args:
        src: input image object
        p: uniform coordinates parameters

    Returns:
        Output image object with uniform coordinates
    """
    dst = dst_1_to_1(src, "uniform_coords", f"dx={p.dx}, dy={p.dy}")
    dst.set_uniform_coords(p.dx, p.dy, p.x0, p.y0)
    return dst


class XYZCalibrateParam(gds.DataSet):
    """Image polynomial calibration parameters"""

    axes = (("x", _("X-axis")), ("y", _("Y-axis")), ("z", _("Z-axis")))
    axis = gds.ChoiceItem(_("Calibrate"), axes, default="z")
    a0 = gds.FloatItem("a<sub>0</sub>", default=0.0, help=_("Constant term"))
    a1 = gds.FloatItem("a<sub>1</sub>", default=1.0, help=_("Linear term"))
    a2 = gds.FloatItem("a<sub>2</sub>", default=0.0, help=_("Quadratic term"))
    a3 = gds.FloatItem("a<sub>3</sub>", default=0.0, help=_("Cubic term"))


@computation_function(
    operation_id="sigima.image.calibration",
    contract_version=2,
    qualification=ReplayQualification(contract_version=2, comparison=EXACT),
)
def calibration(src: ImageObj, p: XYZCalibrateParam) -> ImageObj:
    """Compute polynomial calibration

    Applies polynomial transformation: dst = a0 + a1*src + a2*src² + a3*src³

    Args:
        src: input image object
        p: calibration parameters

    Returns:
        Output image object
    """
    # Build polynomial description for metadata
    terms = []
    if p.a0 != 0.0:
        terms.append(f"{p.a0}")
    if p.a1 != 0.0:
        terms.append(f"{p.a1}*{p.axis}" if p.a1 != 1.0 else p.axis)
    if p.a2 != 0.0:
        terms.append(f"{p.a2}*{p.axis}²")
    if p.a3 != 0.0:
        terms.append(f"{p.a3}*{p.axis}³")
    poly_str = "+".join(terms) if terms else "0"

    dst = dst_1_to_1(src, "calibration", f"{p.axis}={poly_str}")

    shape = src.data.shape

    if p.axis == "z":
        # Apply polynomial to data values (complex data is kept complex)
        data = src.data.astype(np.result_type(src.data.dtype, np.float64))
        dst.data = p.a0 + p.a1 * data + p.a2 * data**2 + p.a3 * data**3
        restore_data_outside_roi(dst, src)
    elif p.axis == "x":
        # For X-axis, polynomial calibration requires non-uniform coordinates
        # (unless it's linear but we don't special case that here)
        if src.is_uniform_coords:
            # Generate uniform coordinates array
            x_uniform = src.x0 + np.arange(src.data.shape[1]) * src.dx
            # Apply polynomial transformation
            x_new = p.a0 + p.a1 * x_uniform + p.a2 * x_uniform**2 + p.a3 * x_uniform**3
            # Set non-uniform coordinates
            ycoords = np.linspace(src.y0, src.y0 + src.dy * (shape[0] - 1), shape[0])
            dst.set_coords(x_new, ycoords)
        else:
            # Apply polynomial to existing non-uniform coordinates
            x_new = (
                p.a0
                + p.a1 * src.xcoords
                + p.a2 * src.xcoords**2
                + p.a3 * src.xcoords**3
            )
            dst.set_coords(x_new, dst.ycoords)
    elif p.axis == "y":
        # For Y-axis, polynomial calibration requires non-uniform coordinates
        if src.is_uniform_coords:
            # Generate uniform coordinates array
            y_uniform = src.y0 + np.arange(src.data.shape[0]) * src.dy
            # Apply polynomial transformation
            y_new = p.a0 + p.a1 * y_uniform + p.a2 * y_uniform**2 + p.a3 * y_uniform**3
            # Set non-uniform coordinates
            xcoords = np.linspace(src.x0, src.x0 + src.dx * (shape[1] - 1), shape[1])
            dst.set_coords(xcoords, y_new)
        else:
            # Apply polynomial to existing non-uniform coordinates
            y_new = (
                p.a0
                + p.a1 * src.ycoords
                + p.a2 * src.ycoords**2
                + p.a3 * src.ycoords**3
            )
            dst.set_coords(dst.xcoords, y_new)
    else:  # Should not happen
        raise ValueError(f"Unknown axis: {p.axis}")  # pragma: no cover
    return dst
