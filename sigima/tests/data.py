# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test data functions

Functions creating test data: curves, images, ...
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: skip

from __future__ import annotations

from typing import Any, Callable, Generator

import guidata.dataset as gds
import numpy as np

from sigima.config import _
from sigima.io import read_image, read_signal
from sigima.objects import (
    GaussParam,
    GeometryResult,
    ImageDatatypes,
    ImageObj,
    ImageROI,
    ImageTypes,
    NewImageParam,
    NewSignalParam,
    NormalDistribution1DParam,
    NormalDistribution2DParam,
    SignalObj,
    SignalROI,
    SignalTypes,
    TableResult,
    create_image,
    create_image_from_param,
    create_image_roi,
    create_signal_from_param,
    create_signal_parameters,
    create_signal_roi,
)
from sigima.objects.image import UniformDistribution2DParam, create_image_parameters
from sigima.objects.scalar import KindShape
from sigima.tests.env import execenv
from sigima.tests.helpers import get_test_fnames


def get_test_signal(filename: str) -> SignalObj:
    """Return test signal

    Args:
        filename: Filename

    Returns:
        Signal object
    """
    return read_signal(get_test_fnames(filename)[0])


def get_test_image(filename: str) -> ImageObj:
    """Return test image

    Args:
        filename: Filename

    Returns:
        Image object
    """
    return read_image(get_test_fnames(filename)[0])


def iterate_signal_creation(
    size: int = 500,
    non_zero: bool = False,
    verbose: bool = True,
    preproc: Callable[[NewSignalParam], None] | None = None,
    postproc: Callable[[SignalObj], None] | None = None,
) -> Generator[SignalObj, None, None]:
    """Iterate over all possible signals created from parameters

    Args:
        size: Size of the data. Defaults to 500.
        non_zero: If True, skip zero signals. Defaults to False.
        verbose: If True, print the signal types being created. Defaults to True.
        preproc: Callback function to preprocess the signal parameters set before
         creation. Defaults to None.
        postproc: Callback function to postprocess the signal object after creation.
         Defaults to None.

    Yields:
        Signal object created from parameters.
    """
    if verbose:
        execenv.print(
            f"  Iterating over {len(SignalTypes)} signal types "
            f"(size={size}, non_zero={non_zero}):"
        )
    for stype in SignalTypes:
        if non_zero and stype in (SignalTypes.ZERO,):
            continue
        if verbose:
            execenv.print(f"    {stype.value}")
        param = create_signal_parameters(stype, size=size)
        if preproc is not None:
            preproc(param)
        signal = create_signal_from_param(param)
        if postproc is not None:
            postproc(signal, stype)
        yield signal


def create_paracetamol_signal(
    size: int | None = None, title: str | None = None
) -> SignalObj:
    """Create test signal (Paracetamol molecule spectrum)

    Args:
        size: Size of the data. Defaults to None.
        title: Title of the signal. Defaults to None.

    Returns:
        Signal object
    """
    obj = read_signal(get_test_fnames("paracetamol.txt")[0])
    if title is not None:
        obj.title = title
    if size is not None:
        x0, y0 = obj.xydata
        x1 = np.linspace(x0[0], x0[-1], size)
        y1 = np.interp(x1, x0, y0)
        obj.set_xydata(x1, y1)
    return obj


def add_gaussian_noise_to_signal(
    signal: SignalObj, p: NormalDistribution1DParam | None = None
) -> None:
    """Add Gaussian (Normal-law) random noise to data

    Args:
        signal: Signal object
        p: Gaussian noise parameters.
    """
    if p is None:
        p = NormalDistribution1DParam()
    rng = np.random.default_rng(p.seed)
    signal.data += rng.normal(p.mu, p.sigma, size=signal.data.shape)
    signal.title = f"GaussNoise({signal.title}, µ={p.mu}, σ={p.sigma})"


def create_noisy_signal(
    noiseparam: NormalDistribution1DParam | None = None,
    param: NewSignalParam | None = None,
    title: str | None = None,
    noised: bool | None = None,
) -> SignalObj:
    """Create curve data, optionally noised

    Args:
        noiseparam: Noise parameters. Default: None: No noise
        newparam: New signal parameters.
         Default: Gaussian, size=500, xmin=-10, xmax=10,
         a=1.0, sigma=1.0, mu=0.0, ymin=0.0
        title: Title of the signal. Default: None
         If not None, overrides the title in newparam
        noised: If True, add noise to the signal.
         Default: None (use noiseparam)
         If True, eventually creates a new noiseparam if None

    Returns:
        Signal object
    """
    if param is None:
        param = GaussParam()
    if title is not None:
        param.title = title
    param.title = "Test signal (noisy)" if param.title is None else param.title
    if noised is not None and noised and noiseparam is None:
        noiseparam = NormalDistribution1DParam()
        noiseparam.sigma = 5.0
    sig = create_signal_from_param(param)
    if noiseparam is not None:
        add_gaussian_noise_to_signal(sig, noiseparam)
    return sig


def create_periodic_signal(
    stype: SignalTypes,
    freq: float = 50.0,
    size: int = 10000,
    xmin: float = -10.0,
    xmax: float = 10.0,
    a: float = 1.0,
) -> SignalObj:
    """Create a periodic signal

    Args:
        stype: Type of the signal (shape of the periodic signal).
        freq: Frequency of the signal. Defaults to 50.0.
        size: Size of the signal. Defaults to 10000.
        xmin: Minimum value of the signal. Defaults to None.
        xmax: Maximum value of the signal. Defaults to None.
        a: Amplitude of the signal. Defaults to 1.0.

    Returns:
        Signal object
    """
    p = create_signal_parameters(stype, size=size, xmin=xmin, xmax=xmax, freq=freq, a=a)
    return create_signal_from_param(p)


def create_2d_steps_data(size: int, width: int, dtype: np.dtype) -> np.ndarray:
    """Creating 2D steps data for testing purpose

    Args:
        size: Size of the data
        width: Width of the steps
        dtype: Data type

    Returns:
        2D data
    """
    data = np.zeros((size, size), dtype=dtype)
    value = 1
    for col in range(0, size - width + 1, width):
        data[:, col : col + width] = np.array(value).astype(dtype)
        value *= 10
    data2 = np.zeros_like(data)
    value = 1
    for row in range(0, size - width + 1, width):
        data2[row : row + width, :] = np.array(value).astype(dtype)
        value *= 10
    data += data2
    return data


def create_2d_random(
    size: int, dtype: np.dtype, level: float = 0.1, seed: int = 1
) -> np.ndarray:
    """Creating 2D Uniform-law random image

    Args:
        size: Size of the data
        dtype: Data type
        level: Level of the random noise. Defaults to 0.1.
        seed: Seed for random number generator. Defaults to 1.

    Returns:
        2D data
    """
    rng = np.random.default_rng(seed)
    amp = (np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0) * level
    return np.array(rng.random((size, size)) * amp, dtype=dtype)


def create_2d_gaussian(
    size: int,
    dtype: np.dtype,
    x0: float = 0,
    y0: float = 0,
    mu: float = 0.0,
    sigma: float = 2.0,
    amp: float | None = None,
) -> np.ndarray:
    """Creating 2D Gaussian (-10 <= x <= 10 and -10 <= y <= 10)

    Args:
        size: Size of the data
        dtype: Data type
        x0: x0. Defaults to 0.
        y0: y0. Defaults to 0.
        mu: mu. Defaults to 0.0.
        sigma: sigma. Defaults to 2.0.
        amp: Amplitude. Defaults to None.

    Returns:
        2D data
    """
    xydata = np.linspace(-10, 10, size)
    x, y = np.meshgrid(xydata, xydata)
    if amp is None:
        try:
            amp = np.iinfo(dtype).max * 0.5
        except ValueError:
            # dtype is not integer
            amp = 1.0
    return np.array(
        amp
        * np.exp(
            -((np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - mu) ** 2) / (2.0 * sigma**2)
        ),
        dtype=dtype,
    )


def get_laser_spot_data() -> list[np.ndarray]:
    """Return a list of NumPy arrays containing images which are relevant for
    testing laser spot image processing features

    Returns:
        List of NumPy arrays
    """
    znoise = create_2d_random(2000, np.uint16)
    zgauss = create_2d_gaussian(2000, np.uint16, x0=2.0, y0=-3.0)
    return [zgauss + znoise] + [
        read_image(fname).data for fname in get_test_fnames("*.scor-data")
    ]


class PeakDataParam(gds.DataSet, title=_("Image with peaks")):
    """Peak data test image parameters"""

    size = gds.IntItem(_("Size"), unit="pixels", default=2000, min=1)
    num_peaks = gds.IntItem(
        "N<sub>peaks</sub>", default=4, min=1, help=_("Number of peaks to generate")
    ).set_prop("display", col=1)
    sigma_gauss2d = gds.FloatItem(
        "σ<sub>Gauss2D</sub>", default=0.06, help=_("Sigma of the 2D Gaussian")
    )
    amp_gauss2d = gds.IntItem(
        "A<sub>Gauss2D</sub>", default=1900, help=_("Amplitude of the 2D Gaussian")
    ).set_prop("display", col=1)
    mu_noise = gds.IntItem(
        "μ<sub>noise</sub>", default=845, help=_("Mean of the Gaussian distribution")
    )
    sigma_noise = gds.IntItem(
        "σ<sub>noise</sub>",
        default=25,
        help=_("Standard deviation of the Gaussian distribution"),
    ).set_prop("display", col=1)
    dx0 = gds.FloatItem("dx0", default=0.0)
    dy0 = gds.FloatItem("dy0", default=0.0).set_prop("display", col=1)
    att = gds.FloatItem(_("Attenuation"), default=1.0)


def get_peak2d_data(
    p: PeakDataParam | None = None, seed: int | None = None, multi: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return a list of NumPy arrays containing images which are relevant for
    testing 2D peak detection or similar image processing features

    Args:
        p: Peak data test image parameters. Defaults to None.
        seed: Seed for random number generator. Defaults to None.
        multi: If True, multiple peaks are generated. Defaults to False.

    Returns:
        A tuple containing the image data and coordinates of the peaks.
    """
    if p is None:
        p = PeakDataParam()
    delta = 0.1
    rng = np.random.default_rng(seed)
    coords_phys = (rng.random((p.num_peaks, 2)) - 0.5) * 10 * (1 - delta)
    data = rng.normal(p.mu_noise, p.sigma_noise, size=(p.size, p.size))
    multi_nb = 2 if multi else 1
    for x0, y0 in coords_phys:
        for idx in range(multi_nb):
            if idx != 0:
                p.dx0 = 0.08 + rng.random() * 0.08
                p.dy0 = 0.08 + rng.random() * 0.08
                p.att = 0.2 + rng.random() * 0.8
            data += create_2d_gaussian(
                p.size,
                np.uint16,
                x0=x0 + p.dx0,
                y0=y0 + p.dy0,
                sigma=p.sigma_gauss2d,
                amp=p.amp_gauss2d / multi_nb * p.att,
            )
    # Convert coordinates to indices
    coords = []
    for x0, y0 in coords_phys:
        x = (x0 + 10) / 20 * p.size
        y = (y0 + 10) / 20 * p.size
        if 0 <= x < p.size and 0 <= y < p.size:
            coords.append((x, y))
    return data, np.array(coords)


CLASS_NAME = "class_name"


def create_test_signal_rois(
    obj: SignalObj,
) -> Generator[SignalROI, None, None]:
    """Create test signal ROIs (sigima.objects.SignalROI test object)

    Yields:
        SignalROI object
    """
    # ROI coordinates: for each ROI type, the coordinates are given for indices=True
    # and indices=False (physical coordinates)
    roi_coords = {
        "segment": {
            CLASS_NAME: "SegmentROI",
            True: [50, 100],  # indices [x0, dx]
            False: [7.5, 10.0],  # physical
        },
    }
    for indices in (True, False):
        execenv.print("indices:", indices)

        for geometry, coords in roi_coords.items():
            execenv.print("  geometry:", geometry)

            roi = create_signal_roi(coords[indices], indices=indices)

            sroi = roi.get_single_roi(0)
            assert sroi.__class__.__name__ == coords[CLASS_NAME]

            cds_ind = [int(val) for val in sroi.get_indices_coords(obj)]
            assert cds_ind == coords[True]

            cds_phys = [float(val) for val in sroi.get_physical_coords(obj)]
            assert cds_phys == coords[False]

            execenv.print("    get_physical_coords:", cds_phys)
            execenv.print("    get_indices_coords: ", cds_ind)

            yield roi


def __idx_to_phys(obj: ImageObj, idx_coords: list[int]) -> list[float]:
    """Convert index coordinates to physical coordinates.

    Args:
        obj: Image object
        idx_coords: List of index coordinates [x0, y0, dx, dy].

    Returns:
        List of physical coordinates [x0, y0, dx, dy].
    """
    coords_array = np.array(idx_coords, dtype=float)
    coords_array[::2] = coords_array[::2] * obj.dx + obj.x0
    coords_array[1::2] = coords_array[1::2] * obj.dy + obj.y0
    return coords_array.tolist()


def create_test_image_rois(obj: ImageObj) -> Generator[ImageROI, None, None]:
    """Create test image ROIs (sigima.objects.ImageROI test object)

    Yields:
        ImageROI object
    """
    # ROI coordinates: for each ROI type, the coordinates are given for indices=True
    # and indices=False (physical coordinates)
    rect_idx = [500, 750, 1000, 1250]  # [x0, y0, dx, dy]
    circ_idx = [1500, 1500, 500]  # [x0, y0, radius]
    poly_idx = [450, 150, 1300, 350, 1250, 950, 400, 1350]  # [x0, y0, ...]
    roi_coords = {
        "rectangle": {
            CLASS_NAME: "RectangularROI",
            True: rect_idx,  # indices [x0, y0, dx, dy]
            False: __idx_to_phys(obj, rect_idx),  # physical
        },
        "circle": {
            CLASS_NAME: "CircularROI",
            True: circ_idx,  # indices [x0, y0, radius]
            False: __idx_to_phys(obj, circ_idx),  # physical
        },
        "polygon": {
            CLASS_NAME: "PolygonalROI",
            True: poly_idx,  # indices [x0, y0, ...]
            False: __idx_to_phys(obj, poly_idx),  # physical
        },
    }
    for indices in (True, False):
        execenv.print("indices:", indices)

        for geometry, coords in roi_coords.items():
            execenv.print("  geometry:", geometry)

            roi = create_image_roi(geometry, coords[indices], indices=indices)

            sroi = roi.get_single_roi(0)
            assert sroi.__class__.__name__ == coords[CLASS_NAME]

            bbox_phys = [float(val) for val in sroi.get_bounding_box(obj)]
            if geometry in ("rectangle", "circle"):
                # pylint: disable=unbalanced-tuple-unpacking
                x0, y0, x1, y1 = obj.physical_to_indices(bbox_phys)
                if geometry == "rectangle":
                    coords_from_bbox = [int(xy) for xy in [x0, y0, x1 - x0, y1 - y0]]
                else:
                    coords_from_bbox = [
                        int(xy) for xy in [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2]
                    ]
                assert coords_from_bbox == coords[True]

            cds_phys = np.array(sroi.get_physical_coords(obj), float)
            assert all(np.isclose(cds_phys, coords[False]))
            cds_ind = np.rint(sroi.get_indices_coords(obj))
            assert all(np.isclose(cds_ind, coords[True]))

            execenv.print("    get_bounding_box:   ", bbox_phys)
            execenv.print("    get_physical_coords:", cds_phys)
            execenv.print("    get_indices_coords: ", cds_ind)

            yield roi


def __iterate_image_datatypes(
    itype: ImageTypes,
    data_size: int,
    verbose: bool,
    preproc: Callable[[NewImageParam], None] | None = None,
    postproc: Callable[[ImageObj, ImageTypes], None] | None = None,
) -> Generator[ImageObj | None, None, None]:
    """Iterate over all datatypes for a given image type

    Args:
        itype: Image type
        data_size: Size of the data
        verbose: If True, print the image types being created
        preproc: Callback function to preprocess the image parameters set before
         creation. Defaults to None.
        postproc: Callback function to postprocess the image object after creation.
         Defaults to None.

    Yields:
        Image object created from parameters
    """
    for idtype in ImageDatatypes:
        if verbose:
            execenv.print(f"      {idtype.value}")
        param = create_image_parameters(
            itype, idtype=idtype, width=data_size, height=data_size
        )
        if itype == ImageTypes.RAMP and idtype != ImageDatatypes.FLOAT64:
            continue  # Testing only float64 for ramp
        if itype == ImageTypes.UNIFORM_DISTRIBUTION:
            assert isinstance(param, UniformDistribution2DParam)
            param.set_from_datatype(idtype.value)
        elif itype == ImageTypes.NORMAL_DISTRIBUTION:
            assert isinstance(param, NormalDistribution2DParam)
            param.set_from_datatype(idtype.value)
        if preproc is not None:
            preproc(param)
        image = create_image_from_param(param)
        if postproc is not None:
            postproc(image, itype)
        yield image


def iterate_image_creation(
    size: int = 500,
    non_zero: bool = False,
    verbose: bool = True,
    preproc: Callable[[NewImageParam], None] | None = None,
    postproc: Callable[[ImageObj, ImageTypes], None] | None = None,
) -> Generator[ImageObj, None, None]:
    """Iterate over all possible images created from parameters

    Args:
        size: Size of the data. Defaults to 500.
        non_zero: If True, skip empty and zero images. Defaults to False.
        verbose: If True, print the image types being created. Defaults to True.
        preproc: Callback function to preprocess the image parameters set before
         creation. Defaults to None.
        postproc: Callback function to postprocess the image object after creation.

    Yields:
        Image object created from parameters.
    """
    if verbose:
        execenv.print(
            f"  Iterating over {len(ImageTypes)} image types "
            f"(size={size}, non_zero={non_zero}):"
        )
    for itype in ImageTypes:
        if non_zero and itype == ImageTypes.ZEROS:
            continue
        if verbose:
            execenv.print(f"    {itype.value}")
        yield from __iterate_image_datatypes(itype, size, verbose, preproc, postproc)


def __set_default_size_dtype(
    p: NewImageParam | None = None,
) -> NewImageParam:
    """Set default shape and dtype

    Args:
        p: Image parameters. Defaults to None. If None, a new object is created.

    Returns:
        Image parameters
    """
    if p is None:
        p = NewImageParam()
    p.height = 2000 if p.height is None else p.height
    p.width = 2000 if p.width is None else p.width
    p.dtype = ImageDatatypes.UINT16 if p.dtype is None else p.dtype
    return p


def create_checkerboard(p: NewImageParam | None = None, num_checkers=8) -> ImageObj:
    """Generate a checkerboard pattern

    Args:
        p: Image parameters. Defaults to None.
        num_checkers: Number of checkers. Defaults to 8.
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (checkerboard)" if p.title is None else p.title
    obj = create_image_from_param(p)
    re = np.r_[num_checkers * [0, 1]]  # one row of the checkerboard
    board = np.vstack(num_checkers * (re, re ^ 1))  # build the checkerboard
    board = np.kron(
        board, np.ones((p.height // num_checkers, p.height // num_checkers))
    )  # scale up the board
    obj.data = board
    return obj


def create_2dstep_image(p: NewImageParam | None = None) -> ImageObj:
    """Creating 2D step image

    Args:
        p: Image parameters. Defaults to None.

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (2D step)" if p.title is None else p.title
    obj = create_image_from_param(p)
    obj.data = create_2d_steps_data(p.height, p.height // 10, p.dtype.to_numpy_dtype())
    return obj


class RingParam(gds.DataSet, title=_("Ring image")):
    """Parameters for creating a ring image"""

    image_size = gds.IntItem(_("Size"), unit="pixels", default=1000)
    intensity = gds.IntItem(_("Intensity"), unit="lsb", default=1000).set_prop(
        "display", col=1
    )
    xc = gds.IntItem(_("X<sub>center</sub>"), unit="pixels", default=500)
    yc = gds.IntItem(_("Y<sub>center</sub>"), unit="pixels", default=500).set_prop(
        "display", col=1
    )
    radius = gds.IntItem(_("Radius"), unit="pixels", default=250)
    thickness = gds.IntItem(_("Thickness"), unit="pixels", default=10).set_prop(
        "display", col=1
    )


def create_ring_data(
    image_size: int, xc: int, yc: int, thickness: int, radius: int, intensity: int
) -> np.ndarray:
    """Create 2D ring data

    Args:
        image_size: Size of the image
        xc: Center x coordinate
        yc: Center y coordinate
        thickness: Thickness of the ring
        radius: Radius of the ring
        intensity: Intensity of the ring

    Returns:
        2D data
    """
    data = np.zeros((image_size, image_size), dtype=np.uint16)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            distsq = (x - xc) ** 2 + (y - yc) ** 2
            if (radius - thickness) ** 2 <= distsq <= (radius + thickness) ** 2:
                data[x, y] = intensity
    return data


def create_ring_image(p: RingParam | None = None) -> ImageObj:
    """Creating 2D ring image

    Args:
        p: Ring image parameters. Defaults to None.

    Returns:
        Image object
    """
    if p is None:
        p = RingParam()
    obj = create_image(
        f"Ring(size={p.image_size},xc={p.xc},yc={p.yc},thickness={p.thickness},"
        f"radius={p.radius},intensity={p.intensity})"
    )
    obj.data = create_ring_data(
        p.image_size,
        p.xc,
        p.yc,
        p.thickness,
        p.radius,
        p.intensity,
    )
    return obj


def create_peak_image(p: NewImageParam | None = None) -> ImageObj:
    """Creating image with bright peaks

    Args:
        p: Image parameters. Defaults to None

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (2D peaks)" if p.title is None else p.title
    obj = create_image_from_param(p)
    param = PeakDataParam()
    if p.height is not None and p.width is not None:
        param.size = max(p.height, p.width)
    obj.data, coords = get_peak2d_data(param)
    obj.metadata["peak_coords"] = coords
    return obj


def create_sincos_image(p: NewImageParam | None = None) -> ImageObj:
    """Creating test image (sin(x)+cos(y))

    Args:
        p: Image parameters. Defaults to None

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (sin(x)+cos(y))" if p.title is None else p.title
    x, y = np.meshgrid(np.linspace(0, 10, p.width), np.linspace(0, 10, p.height))
    raw_data = 0.5 * (np.sin(x) + np.cos(y)) + 0.5
    obj = create_image_from_param(p)
    if np.issubdtype(p.dtype.to_numpy_dtype(), np.floating):
        obj.data = raw_data
        return obj
    dmin = np.iinfo(p.dtype.to_numpy_dtype()).min * 0.95
    dmax = np.iinfo(p.dtype.to_numpy_dtype()).max * 0.95
    obj.data = np.array(raw_data * (dmax - dmin) + dmin, dtype=p.dtype.to_numpy_dtype())
    return obj


def add_annotations_from_file(obj: SignalObj | ImageObj, filename: str) -> None:
    """Add annotations from a file to a Signal or Image object

    Args:
        obj: Signal or Image object to which annotations will be added
        filename: Filename containing annotations
    """
    with open(filename, "r", encoding="utf-8") as file:
        json_str = file.read()
    if obj.annotations:
        json_str = obj.annotations[:-1] + "," + json_str[1:]
    obj.annotations = json_str


def create_noisy_gaussian_image(
    p: NewImageParam | None = None,
    center: tuple[float, float] | None = None,
    level: float = 0.1,
    add_annotations: bool = False,
) -> ImageObj:
    """Create test image (2D noisy gaussian)

    Args:
        p: Image parameters. Defaults to None.
        center: Center of the gaussian. Defaults to None.
        level: Level of the random noise. Defaults to 0.1.
        add_annotations: If True, add annotations. Defaults to False.

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (noisy 2D Gaussian)" if p.title is None else p.title
    obj = create_image_from_param(p)
    if center is None:
        # Default center
        x0, y0 = 2.0, 3.0
    else:
        x0, y0 = center
    obj.data = create_2d_gaussian(p.width, dtype=p.dtype.to_numpy_dtype(), x0=x0, y0=y0)
    if level:
        obj.data += create_2d_random(p.width, p.dtype.to_numpy_dtype(), level)
    if add_annotations:
        add_annotations_from_file(obj, get_test_fnames("annotations.json")[0])
    return obj


def iterate_noisy_images(size: int = 128) -> Generator[ImageObj, None, None]:
    """Iterate over all possible noisy Gaussian images in different datatypes.

    Args:
        size: Size of the image. Defaults to 128.
    """
    for dtype in ImageDatatypes:
        param = NewImageParam.create(dtype=dtype, height=size, width=size)
        yield create_noisy_gaussian_image(param, level=0.0)


def iterate_noisy_image_couples(
    size: int = 128,
) -> Generator[tuple[ImageObj, ImageObj], None, None]:
    """Iterate over all possible pairs of noisy Gaussian images in different datatypes.

    Args:
        size: Size of the images. Defaults to 128.
    """
    for dtype1 in ImageDatatypes:
        param1 = NewImageParam.create(dtype=dtype1, height=size, width=size)
        ima1 = create_noisy_gaussian_image(param1, level=0.0)
        for dtype2 in ImageDatatypes:
            param2 = NewImageParam.create(dtype=dtype2, height=size, width=size)
            ima2 = create_noisy_gaussian_image(param2, level=0.0)
            yield ima1, ima2


def create_n_images(n: int = 100) -> list[ImageObj]:
    """Create a list of N different images for testing."""
    images = []
    for i in range(n):
        param = NewImageParam.create(
            dtype=ImageDatatypes.FLOAT32,
            height=128,
            width=128,
        )
        img = create_noisy_gaussian_image(param, level=(i + 1) * 0.1)
        images.append(img)
    return images


class GridOfGaussianImages(gds.DataSet, title=_("Grid of Gaussian images")):
    """Grid of Gaussian images"""

    nrows = gds.IntItem(_("Number of rows"), default=3, min=1)
    ncols = gds.IntItem(_("Number of columns"), default=3, min=1)


def create_grid_of_gaussian_images(p: GridOfGaussianImages | None = None) -> ImageObj:
    """Create a grid image with multiple noisy Gaussian images.

    Args:
        p: Grid of Gaussian images parameters. Defaults to None.

    Returns:
        Image object containing the grid of images.
    """
    p = p or GridOfGaussianImages()
    size = 512
    grid_data = np.zeros((size, size), dtype=np.float32)
    xmin, xmax = -10.0, 10.0
    ymin, ymax = -10.0, 10.0
    xstep = (xmax - xmin) / p.ncols
    ystep = (ymax - ymin) / p.nrows
    sigma = 0.1
    amp = 1.0
    for j in range(p.ncols):
        for i in range(p.nrows):
            grid_data += create_2d_gaussian(
                size,
                dtype=float,
                x0=(i + 0.5) * xstep + xmin,
                y0=(j + 0.5) * ystep + ymin,
                sigma=sigma,
                amp=amp,
            )
            sigma += 0.05
            amp *= 1.1
    return create_image("Grid Image", grid_data)


def create_multigaussian_image(p: NewImageParam | None = None) -> ImageObj:
    """Create test image (multiple 2D-gaussian peaks)

    Args:
        p: Image parameters. Defaults to None.

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (multi-2D-gaussian)" if p.title is None else p.title
    obj = create_image_from_param(p)
    obj.data = (
        create_2d_gaussian(p.width, p.dtype.to_numpy_dtype(), x0=0.5, y0=3.0)
        + create_2d_gaussian(
            p.width, p.dtype.to_numpy_dtype(), x0=-1.0, y0=-1.0, sigma=1.0
        )
        + create_2d_gaussian(p.width, p.dtype.to_numpy_dtype(), x0=7.0, y0=8.0)
    )
    return obj


def create_annotated_image(title: str | None = None) -> ImageObj:
    """Create test image with annotations

    Returns:
        Image object
    """
    data = create_2d_gaussian(600, np.uint16, x0=2.0, y0=3.0)
    title = "Test image (with metadata)" if title is None else title
    image = create_image(title, data)
    add_annotations_from_file(image, get_test_fnames("annotations.json")[0])
    return image


def create_test_metadata() -> dict[str, Any]:
    """Create test metadata for signals or images.

    Returns:
        Metadata dictionary
    """
    metadata = {}
    metadata["tata"] = {
        "lkl": 2,
        "tototo": 3,
        "arrdata": np.array([0, 1, 2, 3, 4, 5]),
        "zzzz": "lklk",
        "bool": True,
        "float": 1.234,
        "list": [1, 2.5, 3, "str", False, 5],
        "d": {
            "lkl": 2,
            "tototo": 3,
            "zzzz": "lklk",
            "bool": True,
            "float": 1.234,
            "list": [
                1,
                2.5,
                3,
                "str",
                False,
                5,
                {"lkl": 2, "l": [1, 2, 3]},
            ],
        },
    }
    metadata["toto"] = [
        np.array([[1, 2], [-3, 0]]),
        np.array([[1, 2], [-3, 0], [99, 241]]),
    ]
    metadata["array"] = np.array([-5, -4, -3, -2, -1])
    return metadata


def create_test_signal_with_metadata() -> SignalObj:
    """Create a test signal with complex metadata for serialization testing.

    Returns:
        Signal object with metadata containing various data types.
    """
    signal = create_paracetamol_signal()
    signal.metadata = create_test_metadata()
    return signal


def create_test_image_with_metadata() -> ImageObj:
    """Create a test image with complex metadata for serialization testing.

    Returns:
        Image object with metadata containing various data types.
    """
    data = get_test_image("flower.npy").data
    image = create_image("Test image with peaks", data)
    image.metadata = create_test_metadata()
    return image


def generate_geometry_results() -> Generator[GeometryResult, None, None]:
    """Create test geometry results.

    Yields:
        GeometryResult object
    """
    for index, (shape, coords, func_name) in enumerate(
        (
            (KindShape.CIRCLE, [[250, 250, 200]], "func_producing_circle"),
            (KindShape.RECTANGLE, [[300, 200, 150, 250]], "func_producing_rectangle"),
            (KindShape.SEGMENT, [[50, 250, 400, 400]], "func_producing_segment"),
            (KindShape.POINT, [[500, 500]], "func_producing_point"),
            (
                KindShape.POLYGON,
                [[100, 100, 150, 100, 150, 150, 200, 100, 250, 50]],
                "func_producing_polygon",
            ),
        )
    ):
        yield GeometryResult(
            f"GeomResult{index}", shape, coords=np.asarray(coords), func_name=func_name
        )


def generate_table_results() -> Generator[TableResult, None, None]:
    """Create test table results.

    Yields:
        TableResult object
    """
    for index, (names, data) in enumerate(
        (
            (["A", "B", "C", "D"], [["banana", 2.5, -30909, 1.0]]),
            (["P1", "P2", "P3", "P4"], [["apple", 1.232325, -9, 0]]),
        )
    ):
        yield TableResult(
            f"TestProperties{index}",
            "test",
            names,
            data=data,
            func_name="func_producing_table",
        )
