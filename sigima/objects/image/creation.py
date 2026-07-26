# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image creation utilities
========================

This module provides functions and parameter classes for creating new images.

The module includes:

- `create_image`: Factory function for creating ImageObj instances
- `ImageDatatypes`: Enumeration of supported image data types
- `ImageTypes`: Enumeration of supported image generation types
- `NewImageParam` and subclasses: Parameter classes for image generation
- Factory functions and registration utilities

These utilities support creating images from various sources:
- Raw NumPy arrays
- Synthetic data (zeros, random distributions, analytical functions)
- Parameterized image generation
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Type

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.objects import base
from sigima.objects.image.object import ImageObj
from sigima.tools.image import scale_data_to_min_max


def create_image(
    title: str,
    data: np.ndarray | None = None,
    metadata: dict | None = None,
    units: tuple | None = None,
    labels: tuple | None = None,
) -> ImageObj:
    """Create a new Image object

    Args:
        title: image title
        data: image data
        metadata: image metadata
        units: X, Y, Z units (tuple of strings)
        labels: X, Y, Z labels (tuple of strings)

    Returns:
        Image object
    """
    assert isinstance(title, str)
    assert data is None or isinstance(data, np.ndarray)
    image = ImageObj(title=title)
    image.title = title
    image.data = data
    if units is not None:
        image.xunit, image.yunit, image.zunit = units
    if labels is not None:
        image.xlabel, image.ylabel, image.zlabel = labels
    if metadata is not None:
        image.metadata.update(metadata)
    return image


class ImageDatatypes(gds.LabeledEnum):
    """Image data types"""

    @classmethod
    def from_numpy_dtype(cls: type[ImageDatatypes], dtype: np.dtype) -> ImageDatatypes:
        """Return ImageDatatypes member from NumPy dtype

        Args:
            dtype: NumPy dtype object

        Returns:
            Corresponding ImageDatatypes member
        """
        dtype_str = str(dtype)
        for member in cls:
            if member.value == dtype_str:
                return member
        return cls.UINT8  # Default fallback

    def to_numpy_dtype(self) -> np.dtype:
        """Return the corresponding NumPy dtype object.

        This is the symmetrical counterpart to from_numpy_dtype().

        Returns:
            NumPy dtype object that can be used directly with numpy functions.
        """
        return np.dtype(self.value)

    @classmethod
    def check(cls: type[ImageDatatypes]) -> None:
        """Check if data types are valid"""
        for member in cls:
            assert hasattr(np, member.value)

    #: Unsigned integer number stored with 8 bits
    UINT8 = "uint8"
    #: Unsigned integer number stored with 16 bits
    UINT16 = "uint16"
    #: Signed integer number stored with 16 bits
    INT16 = "int16"
    #: Float number stored with 32 bits
    FLOAT32 = "float32"
    #: Float number stored with 64 bits
    FLOAT64 = "float64"


ImageDatatypes.check()


class ImageTypes(gds.LabeledEnum):
    """Image types."""

    #: Image filled with zero
    ZEROS = "zero", _("Zero")
    #: Image filled with random data (normal distribution)
    NORMAL_DISTRIBUTION = "normal_distribution", _("Normal distribution")
    #: Image filled with random data (Poisson distribution)
    POISSON_DISTRIBUTION = "poisson_distribution", _("Poisson distribution")
    #: Image filled with random data (uniform distribution)
    UNIFORM_DISTRIBUTION = "uniform_distribution", _("Uniform distribution")
    #: 2D Gaussian image
    GAUSS = "gauss", _("Gaussian")
    #: Bilinear form image
    RAMP = "ramp", _("2D ramp")
    #: Checkerboard pattern
    CHECKERBOARD = "checkerboard", _("Checkerboard")
    #: Sinusoidal grating pattern
    SINUSOIDAL_GRATING = "sinusoidal_grating", _("Sinusoidal grating")
    #: Ring/circular pattern
    RING = "ring", _("Ring pattern")
    #: Siemens star pattern
    SIEMENS_STAR = "siemens_star", _("Siemens star")
    #: 2D sinc function
    SINC = "sinc", _("2D sinc")


DEFAULT_TITLE = _("Untitled image")


class NewImageParam(gds.DataSet):
    """New image dataset.

    Subclasses can optionally implement a ``generate_title()`` method to provide
    automatic title generation based on their parameters. This method should return
    a string containing the generated title, or an empty string if no title can be
    generated.

    Example::

        def generate_title(self) -> str:
            '''Generate a title based on current parameters.'''
            return f"MyImage(param1={self.param1},param2={self.param2})"
    """

    hide_height = False
    hide_width = False
    hide_dtype = False
    hide_type = False

    title = gds.StringItem(_("Title"), default=DEFAULT_TITLE)
    height = gds.IntItem(
        _("Height"), default=1024, help=_("Image height: number of rows"), min=1
    ).set_prop("display", hide=gds.GetAttrProp("hide_height"))
    width = gds.IntItem(
        _("Width"), default=1024, help=_("Image width: number of columns"), min=1
    ).set_prop("display", col=1, hide=gds.GetAttrProp("hide_width"))
    dtype = gds.ChoiceItem(
        _("Type"),
        ImageDatatypes,
        default=ImageDatatypes.FLOAT64,
        help=_("Image data type"),
    ).set_prop("display", hide=gds.GetAttrProp("hide_dtype"))
    xlabel = gds.StringItem(_("X label"), default="")
    xunit = gds.StringItem(_("X unit"), default="").set_prop("display", col=1)
    ylabel = gds.StringItem(_("Y label"), default="")
    yunit = gds.StringItem(_("Y unit"), default="").set_prop("display", col=1)
    zlabel = gds.StringItem(_("Z label"), default="")
    zunit = gds.StringItem(_("Z unit"), default="").set_prop("display", col=1)

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        return np.zeros(shape, dtype=self.dtype.to_numpy_dtype())


IMAGE_TYPE_PARAM_CLASSES = {}


def register_image_parameters_class(itype: ImageTypes, param_class) -> None:
    """Register a parameters class for a given image type.

    Args:
        itype: image type
        param_class: parameters class
    """
    IMAGE_TYPE_PARAM_CLASSES[itype] = param_class


def __get_image_parameters_class(itype: ImageTypes) -> Type[NewImageParam]:
    """Get parameters class for a given image type.

    Args:
        itype: image type

    Returns:
        Parameters class

    Raises:
        ValueError: if no parameters class is registered for the given image type
    """
    try:
        return IMAGE_TYPE_PARAM_CLASSES[itype]
    except KeyError as exc:
        raise ValueError(
            f"Image type {itype} has no parameters class registered"
        ) from exc


def check_all_image_parameters_classes() -> None:
    """Check all registered parameters classes."""
    for itype, param_class in IMAGE_TYPE_PARAM_CLASSES.items():
        assert __get_image_parameters_class(itype) is param_class


def create_image_parameters(
    itype: ImageTypes,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    idtype: ImageDatatypes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    **kwargs: dict,
) -> NewImageParam:
    """Create parameters for a given image type.

    Args:
        itype: image type
        title: image title
        height: image height (number of rows)
        width: image width (number of columns)
        idtype: image data type (`ImageDatatypes` member)
        xlabel: X axis label
        ylabel: Y axis label
        zlabel: Z axis label
        xunit: X axis unit
        yunit: Y axis unit
        zunit: Z axis unit
        **kwargs: additional parameters (specific to the image type)

    Returns:
        Parameters object for the given image type
    """
    pclass = __get_image_parameters_class(itype)
    p = pclass.create(**kwargs)
    if title is not None:
        p.title = title
    if height is not None:
        p.height = height
    if width is not None:
        p.width = width
    if idtype is not None:
        assert isinstance(idtype, ImageDatatypes)
        p.dtype = idtype
    if xlabel is not None:
        p.xlabel = xlabel
    if ylabel is not None:
        p.ylabel = ylabel
    if zlabel is not None:
        p.zlabel = zlabel
    if xunit is not None:
        p.xunit = xunit
    if yunit is not None:
        p.yunit = yunit
    if zunit is not None:
        p.zunit = zunit
    return p


class Zero2DParam(NewImageParam, title=_("Zero")):
    """Image parameters for a 2D image filled with zero."""

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        return np.zeros(shape, dtype=self.dtype.to_numpy_dtype())


register_image_parameters_class(ImageTypes.ZEROS, Zero2DParam)


class UniformDistribution2DParam(
    NewImageParam, base.UniformDistributionParam, title=_("Uniform distribution")
):
    """Uniform-distribution image parameters."""

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        rng = np.random.default_rng(self.seed)
        assert self.vmin is not None
        assert self.vmax is not None
        data = scale_data_to_min_max(rng.random(shape), self.vmin, self.vmax)
        return data.astype(self.dtype.to_numpy_dtype())


register_image_parameters_class(
    ImageTypes.UNIFORM_DISTRIBUTION, UniformDistribution2DParam
)


class NormalDistribution2DParam(
    NewImageParam, base.NormalDistributionParam, title=_("Normal distribution")
):
    """Normal-distribution image parameters."""

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array.
        """
        rng = np.random.default_rng(self.seed)
        assert self.mu is not None
        assert self.sigma is not None
        data: np.ndarray = rng.normal(self.mu, self.sigma, shape)
        return data.astype(self.dtype.to_numpy_dtype())


register_image_parameters_class(
    ImageTypes.NORMAL_DISTRIBUTION, NormalDistribution2DParam
)


class PoissonDistribution2DParam(
    NewImageParam, base.PoissonDistributionParam, title=_("Poisson distribution")
):
    """Poisson-distribution image parameters."""

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array.
        """
        rng = np.random.default_rng(self.seed)
        assert self.lam is not None
        data: np.ndarray = rng.poisson(lam=self.lam, size=shape)
        return data.astype(self.dtype.to_numpy_dtype())


register_image_parameters_class(
    ImageTypes.POISSON_DISTRIBUTION, PoissonDistribution2DParam
)


class Gauss2DParam(
    NewImageParam,
    title=_("Gaussian"),
    comment="z = A exp(-((√((x - x<sub>0</sub>)<sup>2</sup> + "
    "(y - y<sub>0</sub>)<sup>2</sup>) - μ)<sup>2</sup>) / (2 σ<sup>2</sup>))",
):
    """2D Gaussian parameters."""

    a = gds.FloatItem("A", default=None, check=False)
    xmin = gds.FloatItem("x<sub>min</sub>", default=-10.0).set_pos(col=1)
    sigma = gds.FloatItem("σ", default=1.0)
    xmax = gds.FloatItem("x<sub>max</sub>", default=10.0).set_pos(col=1)
    mu = gds.FloatItem("μ", default=0.0)
    ymin = gds.FloatItem("y<sub>min</sub>", default=-10.0).set_pos(col=1)
    x0 = gds.FloatItem("x<sub>0</sub>", default=0.0)
    ymax = gds.FloatItem("y<sub>max</sub>", default=10.0).set_pos(col=1)
    y0 = gds.FloatItem("y<sub>0</sub>", default=0.0).set_pos(col=0, colspan=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return (
            f"Gauss(a={self.a:g},μ={self.mu:g},"
            f"σ={self.sigma:g},x0={self.x0:g},y0={self.y0:g})"
        )

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        if self.a is None:
            try:
                self.a = np.iinfo(self.dtype.to_numpy_dtype()).max / 2.0
            except ValueError:
                self.a = 10.0
        x, y = np.meshgrid(
            np.linspace(self.xmin, self.xmax, shape[1]),
            np.linspace(self.ymin, self.ymax, shape[0]),
        )
        data = self.a * np.exp(
            -((np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2) - self.mu) ** 2)
            / (2.0 * self.sigma**2)
        )
        return np.array(data, dtype=self.dtype.to_numpy_dtype())


register_image_parameters_class(ImageTypes.GAUSS, Gauss2DParam)


class Ramp2DParam(
    NewImageParam,
    title=_("2D ramp"),
    comment="z = A (x - x<sub>0</sub>) + B (y - y<sub>0</sub>) + C",
):
    """Define the parameters of a 2D ramp (planar ramp)."""

    _g0_begin = gds.BeginGroup(_("Coefficients"))
    a = gds.FloatItem("A", default=1.0).set_pos(col=0)
    b = gds.FloatItem("B", default=1.0).set_pos(col=1)
    c = gds.FloatItem("C", default=0.0).set_pos(colspan=1)
    x0 = gds.FloatItem("x<sub>0</sub>", default=0.0).set_pos(col=0)
    y0 = gds.FloatItem("y<sub>0</sub>", default=0.0).set_pos(col=1)
    _g0_end = gds.EndGroup("")
    _g1_begin = gds.BeginGroup(_("Domain"))
    xmin = gds.FloatItem("x<sub>min</sub>", default=-1.0).set_pos(col=0)
    xmax = gds.FloatItem("x<sub>max</sub>", default=1.0).set_pos(col=1)
    ymin = gds.FloatItem("y<sub>min</sub>", default=-1.0).set_pos(col=0)
    ymax = gds.FloatItem("y<sub>max</sub>", default=1.0).set_pos(col=1)
    _g1_end = gds.EndGroup("")

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        terms = []  # Build terms list for non-zero coefficients
        if self.a != 0.0:
            if self.x0 == 0.0:
                x_part = f"{self.a:g} x"
            else:
                x_part = f"{self.a:g} (x - {self.x0:g})"
            terms.append(x_part)
        if self.b != 0.0:
            if self.y0 == 0.0:
                y_part = f"{self.b:g} y"
            else:
                y_part = f"{self.b:g} (y - {self.y0:g})"
            terms.append(y_part)
        if self.c != 0.0 or not terms:  # Include c if it's the only term
            terms.append(f"{self.c:g}")
        return f"z = {' + '.join(terms)}"

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        x = np.linspace(self.xmin, self.xmax, shape[1])
        y = np.linspace(self.ymin, self.ymax, shape[0])
        xx, yy = np.meshgrid(x, y)
        data = self.a * (xx - self.x0) + self.b * (yy - self.y0) + self.c
        return np.array(data, dtype=self.dtype.to_numpy_dtype())


register_image_parameters_class(ImageTypes.RAMP, Ramp2DParam)


class Checkerboard2DParam(
    NewImageParam,
    title=_("Checkerboard"),
    comment=_("Checkerboard pattern with alternating squares"),
):
    """Checkerboard pattern parameters."""

    square_size = gds.IntItem(
        _("Square size"), default=64, min=1, help=_("Size of each square in pixels")
    )
    x0 = gds.FloatItem("x<sub>0</sub>", default=0.0, help=_("X offset"))
    y0 = gds.FloatItem("y<sub>0</sub>", default=0.0, help=_("Y offset")).set_pos(col=1)
    vmin = gds.FloatItem(
        _("Minimum value"), default=0.0, help=_("Value for dark squares")
    )
    vmax = gds.FloatItem(
        _("Maximum value"), default=255.0, help=_("Value for light squares")
    ).set_pos(col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"Checkerboard(size={self.square_size})"

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        # Create coordinate arrays
        y = np.arange(shape[0]) - self.y0
        x = np.arange(shape[1]) - self.x0
        xx, yy = np.meshgrid(x, y)

        # Create checkerboard pattern using floor division
        pattern = ((xx // self.square_size) + (yy // self.square_size)) % 2

        # Scale to desired range
        data = np.where(pattern == 0, self.vmin, self.vmax)
        return data.astype(self.dtype.to_numpy_dtype())


register_image_parameters_class(ImageTypes.CHECKERBOARD, Checkerboard2DParam)


class SinusoidalGrating2DParam(
    NewImageParam,
    title=_("Sinusoidal grating"),
    comment="z = A sin(2π(f<sub>x</sub>·x + f<sub>y</sub>·y) + φ) + C",
):
    """Sinusoidal grating parameters."""

    _g0_begin = gds.BeginGroup(_("Amplitude and offset"))
    a = gds.FloatItem("A", default=100.0, help=_("Amplitude")).set_pos(col=0)
    c = gds.FloatItem("C", default=128.0, help=_("DC offset")).set_pos(col=1)
    _g0_end = gds.EndGroup("")

    _g1_begin = gds.BeginGroup(_("Frequency and phase"))
    fx = gds.FloatItem(
        "f<sub>x</sub>", default=0.1, help=_("Spatial frequency in X direction")
    ).set_pos(col=0)
    fy = gds.FloatItem(
        "f<sub>y</sub>", default=0.0, help=_("Spatial frequency in Y direction")
    ).set_pos(col=1)
    phase = gds.FloatItem("φ", default=0.0, help=_("Phase"), unit="rad").set_pos(
        colspan=1
    )
    _g1_end = gds.EndGroup("")

    _g2_begin = gds.BeginGroup(_("Domain"))
    xmin = gds.FloatItem("x<sub>min</sub>", default=0.0).set_pos(col=0)
    xmax = gds.FloatItem("x<sub>max</sub>", default=100.0).set_pos(col=1)
    ymin = gds.FloatItem("y<sub>min</sub>", default=0.0).set_pos(col=0)
    ymax = gds.FloatItem("y<sub>max</sub>", default=100.0).set_pos(col=1)
    _g2_end = gds.EndGroup("")

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"Grating(fx={self.fx:g},fy={self.fy:g},φ={self.phase:g})"

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        x = np.linspace(self.xmin, self.xmax, shape[1])
        y = np.linspace(self.ymin, self.ymax, shape[0])
        xx, yy = np.meshgrid(x, y)

        data = self.a * np.sin(2 * np.pi * (self.fx * xx + self.fy * yy) + self.phase)
        data = data + self.c

        return data.astype(self.dtype.to_numpy_dtype())


register_image_parameters_class(ImageTypes.SINUSOIDAL_GRATING, SinusoidalGrating2DParam)


class Ring2DParam(
    NewImageParam,
    title=_("Ring pattern"),
    comment=_("Concentric ring pattern"),
):
    """Ring pattern parameters."""

    x0 = gds.FloatItem("x<sub>0</sub>", default=0.0, help=_("Center X coordinate"))
    y0 = gds.FloatItem(
        "y<sub>0</sub>", default=0.0, help=_("Center Y coordinate")
    ).set_pos(col=1)

    _g0_begin = gds.BeginGroup(_("Ring parameters"))
    period = gds.FloatItem(
        _("Period"), default=50.0, min=0.1, help=_("Distance between ring centers")
    )
    ring_width = gds.FloatItem(
        _("Ring width"), default=10.0, min=0.1, help=_("Width of each ring")
    ).set_pos(col=1)
    _g0_end = gds.EndGroup("")

    _g1_begin = gds.BeginGroup(_("Amplitude"))
    vmin = gds.FloatItem(_("Minimum value"), default=0.0)
    vmax = gds.FloatItem(_("Maximum value"), default=255.0).set_pos(col=1)
    _g1_end = gds.EndGroup("")

    _g2_begin = gds.BeginGroup(_("Domain"))
    xmin = gds.FloatItem("x<sub>min</sub>", default=-100.0).set_pos(col=0)
    xmax = gds.FloatItem("x<sub>max</sub>", default=100.0).set_pos(col=1)
    ymin = gds.FloatItem("y<sub>min</sub>", default=-100.0).set_pos(col=0)
    ymax = gds.FloatItem("y<sub>max</sub>", default=100.0).set_pos(col=1)
    _g2_end = gds.EndGroup("")

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"Ring(period={self.period:g},width={self.ring_width:g})"

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        x = np.linspace(self.xmin, self.xmax, shape[1])
        y = np.linspace(self.ymin, self.ymax, shape[0])
        xx, yy = np.meshgrid(x, y)

        # Calculate distance from center
        r = np.sqrt((xx - self.x0) ** 2 + (yy - self.y0) ** 2)

        # Create ring pattern: modulo creates repeating pattern
        ring_phase = (r % self.period) / self.period
        # Create rings: value is high when ring_phase is within the ring width
        ring_width_fraction = self.ring_width / self.period
        rings = np.where(ring_phase < ring_width_fraction, 1.0, 0.0)

        # Scale to desired range
        data = self.vmin + rings * (self.vmax - self.vmin)

        return data.astype(self.dtype.to_numpy_dtype())


register_image_parameters_class(ImageTypes.RING, Ring2DParam)


class SiemensStar2DParam(
    NewImageParam,
    title=_("Siemens star"),
    comment=_("Siemens star pattern for resolution testing"),
):
    """Siemens star pattern parameters."""

    x0 = gds.FloatItem("x<sub>0</sub>", default=0.0, help=_("Center X coordinate"))
    y0 = gds.FloatItem(
        "y<sub>0</sub>", default=0.0, help=_("Center Y coordinate")
    ).set_pos(col=1)

    n_spokes = gds.IntItem(
        _("Number of spokes"), default=36, min=2, help=_("Number of spoke pairs")
    )

    _g0_begin = gds.BeginGroup(_("Radial limits"))
    inner_radius = gds.FloatItem(
        _("Inner radius"), default=0.0, min=0.0, help=_("Inner radius (hole in center)")
    )
    outer_radius = gds.FloatItem(
        _("Outer radius"),
        default=100.0,
        min=0.1,
        help=_("Outer radius (edge of pattern)"),
    ).set_pos(col=1)
    _g0_end = gds.EndGroup("")

    _g1_begin = gds.BeginGroup(_("Amplitude"))
    vmin = gds.FloatItem(_("Minimum value"), default=0.0)
    vmax = gds.FloatItem(_("Maximum value"), default=255.0).set_pos(col=1)
    _g1_end = gds.EndGroup("")

    _g2_begin = gds.BeginGroup(_("Domain"))
    xmin = gds.FloatItem("x<sub>min</sub>", default=-100.0).set_pos(col=0)
    xmax = gds.FloatItem("x<sub>max</sub>", default=100.0).set_pos(col=1)
    ymin = gds.FloatItem("y<sub>min</sub>", default=-100.0).set_pos(col=0)
    ymax = gds.FloatItem("y<sub>max</sub>", default=100.0).set_pos(col=1)
    _g2_end = gds.EndGroup("")

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"Siemens(n={self.n_spokes})"

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        x = np.linspace(self.xmin, self.xmax, shape[1])
        y = np.linspace(self.ymin, self.ymax, shape[0])
        xx, yy = np.meshgrid(x, y)

        # Calculate polar coordinates
        r = np.sqrt((xx - self.x0) ** 2 + (yy - self.y0) ** 2)
        theta = np.arctan2(yy - self.y0, xx - self.x0)

        # Create spoke pattern: alternating black and white spokes
        # Normalize angle to [0, 2π] and create pattern
        theta_normalized = (theta + np.pi) / (2 * np.pi)  # Now in [0, 1]
        spoke_pattern = np.floor(theta_normalized * self.n_spokes * 2) % 2

        # Apply radial mask
        radial_mask = (r >= self.inner_radius) & (r <= self.outer_radius)

        # Combine pattern with mask
        data = np.where(radial_mask, spoke_pattern, 0.5)  # 0.5 for outside region

        # Scale to desired range
        data = self.vmin + data * (self.vmax - self.vmin)

        return data.astype(self.dtype.to_numpy_dtype())


register_image_parameters_class(ImageTypes.SIEMENS_STAR, SiemensStar2DParam)


class Sinc2DParam(
    NewImageParam,
    title=_("2D sinc"),
    comment="z = A sinc(√((x - x<sub>0</sub>)<sup>2</sup> + "
    "(y - y<sub>0</sub>)<sup>2</sup>) / σ) + C",
):
    """2D sinc function parameters."""

    _g0_begin = gds.BeginGroup(_("Amplitude and offset"))
    a = gds.FloatItem("A", default=100.0, help=_("Amplitude")).set_pos(col=0)
    c = gds.FloatItem("C", default=0.0, help=_("DC offset")).set_pos(col=1)
    _g0_end = gds.EndGroup("")

    _g1_begin = gds.BeginGroup(_("Center and scale"))
    x0 = gds.FloatItem("x<sub>0</sub>", default=0.0, help=_("Center X coordinate"))
    y0 = gds.FloatItem(
        "y<sub>0</sub>", default=0.0, help=_("Center Y coordinate")
    ).set_pos(col=1)
    sigma = gds.FloatItem("σ", default=10.0, min=0.1, help=_("Scale factor")).set_pos(
        colspan=1
    )
    _g1_end = gds.EndGroup("")

    _g2_begin = gds.BeginGroup(_("Domain"))
    xmin = gds.FloatItem("x<sub>min</sub>", default=-50.0).set_pos(col=0)
    xmax = gds.FloatItem("x<sub>max</sub>", default=50.0).set_pos(col=1)
    ymin = gds.FloatItem("y<sub>min</sub>", default=-50.0).set_pos(col=0)
    ymax = gds.FloatItem("y<sub>max</sub>", default=50.0).set_pos(col=1)
    _g2_end = gds.EndGroup("")

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"Sinc(σ={self.sigma:g},x0={self.x0:g},y0={self.y0:g})"

    def generate_2d_data(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate 2D data based on current parameters.

        Args:
            shape: Tuple (height, width) for the output array.

        Returns:
            2D data array
        """
        x = np.linspace(self.xmin, self.xmax, shape[1])
        y = np.linspace(self.ymin, self.ymax, shape[0])
        xx, yy = np.meshgrid(x, y)

        # Calculate radial distance from center
        r = np.sqrt((xx - self.x0) ** 2 + (yy - self.y0) ** 2)

        # Calculate sinc function: sinc(x) = sin(x)/x, with special case for x=0
        # Scale by sigma
        r_scaled = r / self.sigma

        # Use numpy's sinc which is defined as sin(pi*x)/(pi*x)
        # We want sin(x)/x, so we divide by pi
        data = np.where(r_scaled == 0, 1.0, np.sin(r_scaled) / r_scaled)

        # Apply amplitude and offset
        data = self.a * data + self.c

        return data.astype(self.dtype.to_numpy_dtype())


register_image_parameters_class(ImageTypes.SINC, Sinc2DParam)


check_all_image_parameters_classes()

IMG_NB = 0


def get_next_image_number():
    """Get the next image number.

    This function is used to keep track of the number of signals created.
    It is typically used to generate unique titles for new signals.

    Returns:
        int: new image number
    """
    global IMG_NB  # pylint: disable=global-statement
    IMG_NB += 1
    return IMG_NB


def create_image_from_param(param: NewImageParam) -> ImageObj:
    """Create a new Image object from parameters.

    Args:
        param: new image parameters

    Returns:
        Image object

    Raises:
        NotImplementedError: if the image type is not supported
    """
    if param.height is None:
        param.height = 1024
    if param.width is None:
        param.width = 1024
    if param.dtype is None:
        param.dtype = ImageDatatypes.UINT16
    # Generate data first, as some `generate_title()` methods may depend on it:
    shape = (param.height, param.width)
    data = param.generate_2d_data(shape)
    # Check if user has customized the title or left it as default/empty
    use_generated_title = not param.title or param.title == DEFAULT_TITLE
    if use_generated_title:
        # Try to generate a descriptive title
        gen_title = getattr(param, "generate_title", lambda: "")()
        if gen_title:
            title = gen_title
        else:
            # No generated title available, use default with number
            title = f"{DEFAULT_TITLE} {get_next_image_number()}"
    else:
        # User has set a custom title, use it as-is
        title = param.title
    image = create_image(
        title,
        data,
        units=(param.xunit, param.yunit, param.zunit),
        labels=(param.xlabel, param.ylabel, param.zlabel),
    )
    return image
