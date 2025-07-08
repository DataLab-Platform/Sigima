# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
New signal/image test

Testing parameter-based signal/image creation.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Generator

import sigima.objects
from sigima.tests.env import execenv


def iterate_signal_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[sigima.objects.SignalObj, None, None]:
    """Iterate over all possible signals created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over signal types (size={data_size}, non_zero={non_zero}):"
        )
    for stype in sigima.objects.SignalTypes:
        if non_zero and stype in (sigima.objects.SignalTypes.ZEROS,):
            continue
        if verbose:
            execenv.print(f"    {stype.value}")
        base_param = sigima.objects.NewSignalParam.create(stype=stype, size=data_size)
        if stype == sigima.objects.SignalTypes.UNIFORMRANDOM:
            extra_param = sigima.objects.UniformRandomParam()
        elif stype == sigima.objects.SignalTypes.NORMALRANDOM:
            extra_param = sigima.objects.NormalRandomParam()
        elif stype in (
            sigima.objects.SignalTypes.GAUSS,
            sigima.objects.SignalTypes.LORENTZ,
            sigima.objects.SignalTypes.VOIGT,
        ):
            extra_param = sigima.objects.GaussLorentzVoigtParam()
        elif stype in (
            sigima.objects.SignalTypes.SINUS,
            sigima.objects.SignalTypes.COSINUS,
            sigima.objects.SignalTypes.SAWTOOTH,
            sigima.objects.SignalTypes.TRIANGLE,
            sigima.objects.SignalTypes.SQUARE,
            sigima.objects.SignalTypes.SINC,
        ):
            extra_param = sigima.objects.PeriodicParam()
        elif stype == sigima.objects.SignalTypes.STEP:
            extra_param = sigima.objects.StepParam()
        elif stype == sigima.objects.SignalTypes.EXPONENTIAL:
            extra_param = sigima.objects.ExponentialParam()
        elif stype == sigima.objects.SignalTypes.PULSE:
            extra_param = sigima.objects.PulseParam()
        elif stype == sigima.objects.SignalTypes.POLYNOMIAL:
            extra_param = sigima.objects.PolyParam()
        elif stype == sigima.objects.SignalTypes.EXPERIMENTAL:
            extra_param = sigima.objects.ExperimentalSignalParam()
        else:
            extra_param = None
        signal = sigima.objects.create_signal_from_param(
            base_param, extra_param=extra_param
        )
        if stype == sigima.objects.SignalTypes.ZEROS:
            assert (signal.y == 0).all()
        yield signal


def iterate_image_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[sigima.objects.ImageObj, None, None]:
    """Iterate over all possible images created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over image types (size={data_size}, non_zero={non_zero}):"
        )
    for itype in sigima.objects.ImageTypes:
        if non_zero and itype in (
            sigima.objects.ImageTypes.EMPTY,
            sigima.objects.ImageTypes.ZEROS,
        ):
            continue
        if verbose:
            execenv.print(f"    {itype.value}")
        yield from _iterate_image_datatypes(itype, data_size, verbose)


def _iterate_image_datatypes(
    itype: sigima.objects.ImageTypes, data_size: int, verbose: bool
) -> Generator[sigima.objects.ImageObj | None, None, None]:
    for dtype in sigima.objects.ImageDatatypes:
        if verbose:
            execenv.print(f"      {dtype.value}")
        base_param = sigima.objects.NewImageParam.create(
            itype=itype, dtype=dtype, width=data_size, height=data_size
        )
        extra_param = _get_additional_param(itype, dtype)
        image = sigima.objects.create_image_from_param(
            base_param, extra_param=extra_param
        )
        if image is not None:
            _test_image_data(itype, image)
        yield image


def _get_additional_param(
    itype: sigima.objects.ImageTypes, dtype: sigima.objects.ImageDatatypes
) -> (
    sigima.objects.Gauss2DParam
    | sigima.objects.UniformRandomParam
    | sigima.objects.NormalRandomParam
    | None
):
    if itype == sigima.objects.ImageTypes.GAUSS:
        addparam = sigima.objects.Gauss2DParam()
        addparam.x0 = addparam.y0 = 3
        addparam.sigma = 5
    elif itype == sigima.objects.ImageTypes.UNIFORMRANDOM:
        addparam = sigima.objects.UniformRandomParam()
        addparam.set_from_datatype(dtype.value)
    elif itype == sigima.objects.ImageTypes.NORMALRANDOM:
        addparam = sigima.objects.NormalRandomParam()
        addparam.set_from_datatype(dtype.value)
    else:
        addparam = None
    return addparam


def _test_image_data(
    itype: sigima.objects.ImageTypes, image: sigima.objects.ImageObj
) -> None:
    """
    Tests the data of an image based on its type.

    Args:
        itype: The type of the image.
        image: The image object containing the data to be tested.

    Raises:
        AssertionError: If the image data does not match the expected values
         for the given image type.
    """
    if itype == sigima.objects.ImageTypes.ZEROS:
        assert (image.data == 0).all()
    else:
        assert image.data is not None


def test_all_combinations() -> None:
    """Test all combinations for new signal/image feature"""
    execenv.print(f"Testing {test_all_combinations.__name__}:")
    execenv.print(f"  Signal types ({len(sigima.objects.SignalTypes)}):")
    for signal in iterate_signal_creation():
        assert signal.x is not None and signal.y is not None
    execenv.print(f"  Image types ({len(sigima.objects.ImageTypes)}):")
    for image in iterate_image_creation():
        assert image.data is not None
    execenv.print(f"{test_all_combinations.__name__} OK")


if __name__ == "__main__":
    test_all_combinations()
