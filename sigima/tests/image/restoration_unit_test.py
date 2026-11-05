# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for restoration computation functions.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage import morphology, restoration

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests import guiutils
from sigima.tests.data import create_multigaussian_image, get_test_image
from sigima.tests.helpers import check_array_result, rectangle_selection


@pytest.mark.validation
def test_denoise_tv() -> None:
    """Validation test for the image Total Variation denoising processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    src.data = src.data[::8, ::8]
    for weight, eps, mni in ((0.1, 0.0002, 200), (0.5, 0.0001, 100)):
        p = sigima.params.DenoiseTVParam.create(
            weight=weight, eps=eps, max_num_iter=mni
        )
        dst = sigima.proc.image.denoise_tv(src, p)
        exp = restoration.denoise_tv_chambolle(src.data, weight, eps, mni)
        check_array_result(
            f"DenoiseTV[weight={weight},eps={eps},max_num_iter={mni}]",
            dst.data,
            exp,
        )


@pytest.mark.validation
def test_denoise_bilateral() -> None:
    """Validation test for the image bilateral denoising processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    src.data = src.data[::8, ::8]
    for sigma, mode in ((1.0, "constant"), (2.0, "edge")):
        p = sigima.params.DenoiseBilateralParam.create(sigma_spatial=sigma, mode=mode)
        dst = sigima.proc.image.denoise_bilateral(src, p)
        exp = restoration.denoise_bilateral(src.data, sigma_spatial=sigma, mode=mode)
        check_array_result(
            f"DenoiseBilateral[sigma_spatial={sigma},mode={mode}]",
            dst.data,
            exp,
        )


@pytest.mark.validation
def test_denoise_wavelet() -> None:
    """Validation test for the image wavelet denoising processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    src.data = src.data[::8, ::8]
    p = sigima.params.DenoiseWaveletParam()
    for wavelets in ("db1", "db2", "db3"):
        for mode in sigima.enums.ThresholdMethod:
            for method in ("BayesShrink",):
                p.wavelets, p.mode, p.method = wavelets, mode, method
                dst = sigima.proc.image.denoise_wavelet(src, p)
                exp = restoration.denoise_wavelet(
                    src.data, wavelet=wavelets, mode=mode.value, method=method
                )
                check_array_result(
                    f"DenoiseWavelet[wavelets={wavelets},mode={mode},method={method}]",
                    dst.data,
                    exp,
                    atol=0.1,
                )


@pytest.mark.validation
def test_denoise_tophat() -> None:
    """Validation test for the image top-hat denoising processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    p = sigima.params.MorphologyParam.create(radius=10)
    dst = sigima.proc.image.denoise_tophat(src, p)
    footprint = morphology.disk(p.radius)
    exp = src.data - morphology.white_tophat(src.data, footprint=footprint)
    check_array_result(f"DenoiseTophat[radius={p.radius}]", dst.data, exp)


@pytest.mark.validation
def test_erase() -> None:
    """Validation test for the image erase processing."""
    obj = create_multigaussian_image()

    # Single ROI erase
    coords = [600, 800, 300, 200]
    p = sigima.objects.ROI2DParam()
    p.x0, p.y0, p.dx, p.dy = coords
    dst = sigima.proc.image.erase(obj, p)
    exp = obj.data.copy()
    selection = rectangle_selection(obj, *coords)
    exp[selection] = np.ma.mean(obj.data[selection])
    guiutils.view_images_side_by_side_if_gui(
        [obj.data, dst.data, exp], ["Original", "Erased", "Expected"]
    )
    check_array_result("Erase", dst.data, exp)

    # Multiple ROIs erase
    coords = [
        [600, 800, 300, 200],
        [100, 200, 300, 200],
        [400, 500, 300, 200],
    ]
    params: list[sigima.objects.ROI2DParam] = []
    for c in coords:
        p = sigima.objects.ROI2DParam()
        p.x0, p.y0, p.dx, p.dy = c
        params.append(p)
    dst = sigima.proc.image.erase(obj, params)
    exp = obj.data.copy()
    for p in params:
        selection = rectangle_selection(obj, p.x0, p.y0, p.dx, p.dy)
        exp[selection] = np.ma.mean(obj.data[selection])
    guiutils.view_images_side_by_side_if_gui(
        [obj.data, dst.data, exp], ["Original", "Erased", "Expected"]
    )
    check_array_result("Erase", dst.data, exp)


if __name__ == "__main__":
    guiutils.enable_gui()
    test_denoise_tv()
    test_denoise_bilateral()
    test_denoise_wavelet()
    test_denoise_tophat()
    test_erase()
