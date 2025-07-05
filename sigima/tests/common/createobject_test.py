# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
New signal/image test

Testing parameter-based signal/image creation.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Generator

import sigima.obj
from sigima.tests.env import execenv


def iterate_signal_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[sigima.obj.SignalObj, None, None]:
    """Iterate over all possible signals created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over signal types (size={data_size}, non_zero={non_zero}):"
        )
    for stype in sigima.obj.SignalTypes:
        if non_zero and stype in (sigima.obj.SignalTypes.ZEROS,):
            continue
        if verbose:
            execenv.print(f"    {stype.value}")
        base_param = sigima.obj.NewSignalParam.create(stype=stype, size=data_size)
        if stype == sigima.obj.SignalTypes.UNIFORMRANDOM:
            extra_param = sigima.obj.UniformRandomParam()
        elif stype == sigima.obj.SignalTypes.NORMALRANDOM:
            extra_param = sigima.obj.NormalRandomParam()
        elif stype in (
            sigima.obj.SignalTypes.GAUSS,
            sigima.obj.SignalTypes.LORENTZ,
            sigima.obj.SignalTypes.VOIGT,
        ):
            extra_param = sigima.obj.GaussLorentzVoigtParam()
        elif stype in (
            sigima.obj.SignalTypes.SINUS,
            sigima.obj.SignalTypes.COSINUS,
            sigima.obj.SignalTypes.SAWTOOTH,
            sigima.obj.SignalTypes.TRIANGLE,
            sigima.obj.SignalTypes.SQUARE,
            sigima.obj.SignalTypes.SINC,
        ):
            extra_param = sigima.obj.PeriodicParam()
        elif stype == sigima.obj.SignalTypes.STEP:
            extra_param = sigima.obj.StepParam()
        elif stype == sigima.obj.SignalTypes.EXPONENTIAL:
            extra_param = sigima.obj.ExponentialParam()
        elif stype == sigima.obj.SignalTypes.PULSE:
            extra_param = sigima.obj.PulseParam()
        elif stype == sigima.obj.SignalTypes.POLYNOMIAL:
            extra_param = sigima.obj.PolyParam()
        elif stype == sigima.obj.SignalTypes.EXPERIMENTAL:
            extra_param = sigima.obj.ExperimentalSignalParam()
        else:
            extra_param = None
        signal = sigima.obj.create_signal_from_param(
            base_param, extra_param=extra_param
        )
        if stype == sigima.obj.SignalTypes.ZEROS:
            assert (signal.y == 0).all()
        yield signal


def iterate_image_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[sigima.obj.ImageObj, None, None]:
    """Iterate over all possible images created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over image types (size={data_size}, non_zero={non_zero}):"
        )
    for itype in sigima.obj.ImageTypes:
        if non_zero and itype in (
            sigima.obj.ImageTypes.EMPTY,
            sigima.obj.ImageTypes.ZEROS,
        ):
            continue
        if verbose:
            execenv.print(f"    {itype.value}")
        yield from _iterate_image_datatypes(itype, data_size, verbose)


def _iterate_image_datatypes(
    itype: sigima.obj.ImageTypes, data_size: int, verbose: bool
) -> Generator[sigima.obj.ImageObj | None, None, None]:
    for dtype in sigima.obj.ImageDatatypes:
        if verbose:
            execenv.print(f"      {dtype.value}")
        base_param = sigima.obj.NewImageParam.create(
            itype=itype, dtype=dtype, width=data_size, height=data_size
        )
        extra_param = _get_additional_param(itype, dtype)
        image = sigima.obj.create_image_from_param(base_param, extra_param=extra_param)
        if image is not None:
            _test_image_data(itype, image)
        yield image


def _get_additional_param(
    itype: sigima.obj.ImageTypes, dtype: sigima.obj.ImageDatatypes
) -> (
    sigima.obj.Gauss2DParam
    | sigima.obj.UniformRandomParam
    | sigima.obj.NormalRandomParam
    | None
):
    if itype == sigima.obj.ImageTypes.GAUSS:
        addparam = sigima.obj.Gauss2DParam()
        addparam.x0 = addparam.y0 = 3
        addparam.sigma = 5
    elif itype == sigima.obj.ImageTypes.UNIFORMRANDOM:
        addparam = sigima.obj.UniformRandomParam()
        addparam.set_from_datatype(dtype.value)
    elif itype == sigima.obj.ImageTypes.NORMALRANDOM:
        addparam = sigima.obj.NormalRandomParam()
        addparam.set_from_datatype(dtype.value)
    else:
        addparam = None
    return addparam


def _test_image_data(itype: sigima.obj.ImageTypes, image: sigima.obj.ImageObj) -> None:
    """
    Tests the data of an image based on its type.

    Args:
        itype: The type of the image.
        image: The image object containing the data to be tested.

    Raises:
        AssertionError: If the image data does not match the expected values
         for the given image type.
    """
    if itype == sigima.obj.ImageTypes.ZEROS:
        assert (image.data == 0).all()
    else:
        assert image.data is not None


def test_all_combinations() -> None:
    """Test all combinations for new signal/image feature"""
    execenv.print(f"Testing {test_all_combinations.__name__}:")
    execenv.print(f"  Signal types ({len(sigima.obj.SignalTypes)}):")
    for signal in iterate_signal_creation():
        assert signal.x is not None and signal.y is not None
    execenv.print(f"  Image types ({len(sigima.obj.ImageTypes)}):")
    for image in iterate_image_creation():
        assert image.data is not None
    execenv.print(f"{test_all_combinations.__name__} OK")


if __name__ == "__main__":
    test_all_combinations()
