# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for exposure computation functions.
"""

# `@computation_function` rewrites the decorated signature at import time, making the
# DataSet parameter optional. Pylint only sees the source signature, so it wrongly
# reports a missing argument on the expanded-keyword calls exercised below.
# pylint: disable=no-value-for-parameter

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
from guidata.dataset.jsonschema import dataset_to_schema
from skimage import exposure

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests.data import get_test_image
from sigima.tests.helpers import check_array_result, check_scalar_result
from sigima.tools.image import exposure as image_exposure


@pytest.mark.validation
def test_adjust_brightness_contrast() -> None:
    """The source-derived window drives a clipped dtype-preserving remap."""
    data = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    src = sigima.objects.create_image("uint8", data)
    p = sigima.params.BrightnessContrastParam()
    p.update_from_obj(src)

    assert (
        dataset_to_schema(type(p))["properties"]["histogram"][
            "x-guidata-histogram-presentation"
        ]
        == "brightness_contrast"
    )
    assert (p.minimum, p.maximum) == (0.0, 255.0)
    assert p.histogram["domain"] == [0.0, 255.0]
    assert len(p.histogram["counts"]) == 256
    assert (
        "histogram"
        not in inspect.signature(
            sigima.proc.image.adjust_brightness_contrast
        ).parameters
    )

    p.minimum, p.maximum = 64.0, 192.0
    dst = sigima.proc.image.adjust_brightness_contrast(src, p)
    np.testing.assert_array_equal(dst.data, [[0, 0, 128, 255]])
    assert dst.data.dtype == src.data.dtype
    np.testing.assert_array_equal(src.data, data)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("values", [[], [0, 1, 2]])
@pytest.mark.parametrize("operation", ["ranges", "context", "remap", "identity"])
def test_brightness_contrast_rejects_64bit_integers(dtype, values, operation):
    """All tools reject unsupported types, including early-return paths."""
    data = np.array([values], dtype=dtype)
    original = data.copy()
    with np.errstate(all="raise"), pytest.raises(TypeError, match="32-bit"):
        if operation == "ranges":
            image_exposure.brightness_contrast_ranges(data, data.dtype)
        elif operation == "context":
            image_exposure.brightness_contrast_context(data, data.dtype)
        else:
            output = (0.0, 2.0) if operation == "identity" else (0.0, 10.0)
            image_exposure.adjust_brightness_contrast(data, 0.0, 2.0, output)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]
)
def test_brightness_contrast_integer_endpoints(dtype):
    """Supported integer types retain exact endpoints and nearest-even rounding."""
    data = np.array([[0, 1, 2]], dtype=dtype)
    bounds = np.iinfo(dtype)
    _, output = image_exposure.brightness_contrast_ranges(data, data.dtype)
    with np.errstate(all="raise"):
        result = image_exposure.adjust_brightness_contrast(data, 0.0, 2.0, output)
    assert result.dtype == dtype
    assert result[0, 0] == bounds.min
    assert result[0, 2] == bounds.max
    assert int(result[0, 1]) == round((int(bounds.min) + int(bounds.max)) / 2)
    assert np.all(result[0, 1:] >= result[0, :-1])
    np.testing.assert_array_equal(data, [[0, 1, 2]])


def test_adjust_brightness_contrast_roi() -> None:
    """The ROI controls initialization and pixels outside it are restored."""
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    src = sigima.objects.create_image("roi", data)
    src.roi = sigima.objects.create_image_roi("rectangle", [1, 1, 2, 2], indices=True)
    p = sigima.params.BrightnessContrastParam()
    p.update_from_obj(src)
    assert (p.minimum, p.maximum) == (5.0, 10.0)

    dst = sigima.proc.image.adjust_brightness_contrast(src, p)
    mask = src.maskdata
    np.testing.assert_array_equal(dst.data[mask], src.data[mask])
    np.testing.assert_array_equal(dst.data[~mask], [0, 13107, 52428, 65535])


def test_adjust_brightness_contrast_float_nonfinite() -> None:
    """Non-finite values do not affect statistics and survive the remap."""
    data = np.array([[-1.0, 0.0, 1.0, np.nan, np.inf, -np.inf]])
    src = sigima.objects.create_image("float", data)
    p = sigima.params.BrightnessContrastParam()
    p.update_from_obj(src)
    assert (p.minimum, p.maximum) == (-1.0, 1.0)

    p.minimum, p.maximum = 0.0, 1.0
    dst = sigima.proc.image.adjust_brightness_contrast(src, p)
    np.testing.assert_array_equal(dst.data[0, :3], [-1.0, -1.0, 1.0])
    assert np.isnan(dst.data[0, 3])
    assert np.isposinf(dst.data[0, 4])
    assert np.isneginf(dst.data[0, 5])


def test_brightness_contrast_editor_context_preserves_range() -> None:
    """Refreshing transient context does not overwrite persisted bounds."""
    src = sigima.objects.create_image(
        "source", np.array([[0, 64, 128, 255]], dtype=np.uint8)
    )
    p = sigima.params.BrightnessContrastParam()
    p.minimum, p.maximum = 64.0, 192.0

    p.update_editor_context(src)

    assert (p.minimum, p.maximum) == (64.0, 192.0)
    assert p.histogram["domain"] == [0.0, 255.0]


def test_adjust_brightness_contrast_tiny_float_reset_is_identity() -> None:
    """Reset preserves a non-constant float image at sub-unit scales."""
    data = np.array([[0.0, 1e-9, 2e-9]], dtype=np.float32)
    src = sigima.objects.create_image("tiny", data)
    p = sigima.params.BrightnessContrastParam()
    p.update_from_obj(src)

    assert (p.minimum, p.maximum) == (0.0, float(data.max()))
    assert 0.0 < p.histogram["minimum_width"] < p.maximum
    dst = sigima.proc.image.adjust_brightness_contrast(src, p)
    np.testing.assert_array_equal(dst.data, data)


def test_brightness_contrast_float32_histogram_uses_safe_precision() -> None:
    """Large finite float32 values all contribute to the histogram."""
    data = np.array([[-3e38, 0.0, 3e38]], dtype=np.float32)
    src = sigima.objects.create_image("large-float32", data)
    p = sigima.params.BrightnessContrastParam()

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        p.update_from_obj(src)

    counts = p.histogram["counts"]
    assert sum(counts) == data.size
    assert (counts[0], counts[128], counts[-1]) == (1, 1, 1)
    assert p.histogram["auto_range"] == p.histogram["reset_range"]
    dst = sigima.proc.image.adjust_brightness_contrast(src, p)
    np.testing.assert_array_equal(dst.data, data)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[-1e308, 0.0, 1e308]], dtype=np.float64),
        np.array([[np.finfo(np.float64).max]], dtype=np.float64),
        np.array([[-np.finfo(np.float64).max]], dtype=np.float64),
    ],
)
def test_brightness_contrast_extreme_float_context_is_finite(
    data: np.ndarray,
) -> None:
    """Finite IEEE-754 extremes produce a JSON-safe context and no exception."""
    src = sigima.objects.create_image("extreme", data)
    p = sigima.params.BrightnessContrastParam()
    p.update_from_obj(src)

    json.dumps(p.histogram, allow_nan=False)
    assert np.all(np.isfinite(p.histogram["domain"]))
    assert np.all(np.isfinite(p.histogram["bin_edges"]))
    assert np.isfinite(p.histogram["minimum_width"])
    assert p.histogram["minimum_width"] > 0.0
    assert sum(p.histogram["counts"]) == data.size
    dst = sigima.proc.image.adjust_brightness_contrast(src, p)
    np.testing.assert_array_equal(dst.data, data)


def test_adjust_brightness_contrast_degenerate_inputs() -> None:
    """Constant images are no-ops and complex images are rejected."""
    constant = sigima.objects.create_image(
        "constant", np.full((3, 3), 7.0, dtype=np.float32)
    )
    p = sigima.params.BrightnessContrastParam()
    p.update_from_obj(constant)
    assert p.minimum < p.maximum
    assert not p.histogram["active"]
    dst = sigima.proc.image.adjust_brightness_contrast(constant, p)
    np.testing.assert_array_equal(dst.data, constant.data)

    complex_image = sigima.objects.create_image(
        "complex", np.ones((2, 2), dtype=np.complex128)
    )
    with pytest.raises(ValueError, match="real image"):
        p.update_from_obj(complex_image)
    with pytest.raises(ValueError, match="real image"):
        sigima.proc.image.adjust_brightness_contrast(
            complex_image, minimum=0.0, maximum=1.0
        )


@pytest.mark.validation
def test_adjust_gamma() -> None:
    """Validation test for the image gamma adjustment processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    for gamma, gain in ((0.5, 1.0), (1.0, 2.0), (1.5, 0.5)):
        p = sigima.params.AdjustGammaParam.create(gamma=gamma, gain=gain)
        dst = sigima.proc.image.adjust_gamma(src, p)
        exp = exposure.adjust_gamma(src.data, gamma=gamma, gain=gain)
        check_array_result(f"AdjustGamma[gamma={gamma},gain={gain}]", dst.data, exp)


@pytest.mark.validation
def test_adjust_log() -> None:
    """Validation test for the image logarithmic adjustment processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    for gain, inv in ((1.0, False), (2.0, True)):
        p = sigima.params.AdjustLogParam.create(gain=gain, inv=inv)
        dst = sigima.proc.image.adjust_log(src, p)
        exp = exposure.adjust_log(src.data, gain=gain, inv=inv)
        check_array_result(f"AdjustLog[gain={gain},inv={inv}]", dst.data, exp)


@pytest.mark.validation
def test_adjust_sigmoid() -> None:
    """Validation test for the image sigmoid adjustment processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    for cutoff, gain, inv in ((0.5, 1.0, False), (0.25, 2.0, True)):
        p = sigima.params.AdjustSigmoidParam.create(cutoff=cutoff, gain=gain, inv=inv)
        dst = sigima.proc.image.adjust_sigmoid(src, p)
        exp = exposure.adjust_sigmoid(src.data, cutoff=cutoff, gain=gain, inv=inv)
        check_array_result(
            f"AdjustSigmoid[cutoff={cutoff},gain={gain},inv={inv}]", dst.data, exp
        )


@pytest.mark.validation
def test_rescale_intensity() -> None:
    """Validation test for the image intensity rescaling processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    p = sigima.params.RescaleIntensityParam.create(in_range="dtype", out_range="image")
    dst = sigima.proc.image.rescale_intensity(src, p)
    exp = exposure.rescale_intensity(
        src.data, in_range=p.in_range, out_range=p.out_range
    )
    check_array_result("RescaleIntensity", dst.data, exp)


@pytest.mark.validation
def test_equalize_hist() -> None:
    """Validation test for the image histogram equalization processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    for nbins in (256, 512):
        p = sigima.params.EqualizeHistParam.create(nbins=nbins)
        dst = sigima.proc.image.equalize_hist(src, p)
        exp = exposure.equalize_hist(src.data, nbins=nbins)
        check_array_result(f"EqualizeHist[nbins={nbins}]", dst.data, exp)


@pytest.mark.validation
def test_equalize_adapthist() -> None:
    """Validation test for the image adaptive histogram equalization processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src = get_test_image("flower.npy")
    for clip_limit in (0.01, 0.1):
        p = sigima.params.EqualizeAdaptHistParam.create(clip_limit=clip_limit)
        dst = sigima.proc.image.equalize_adapthist(src, p)
        exp = exposure.equalize_adapthist(src.data, clip_limit=clip_limit)
        check_array_result(f"AdaptiveHist[clip_limit={clip_limit}]", dst.data, exp)


@pytest.mark.validation
def test_flatfield() -> None:
    """Validation test for the image flat-field correction processing."""
    # See [1] in sigima\tests\image\__init__.py for more details about the validation.
    src1 = get_test_image("flower.npy")  # Raw data
    src2 = get_test_image("flower.npy")  # Flat field data (using same image as base)

    # Modify flat field data to create realistic flat field variation
    src2.data = src2.data.astype(float)
    src2.data = src2.data / np.max(src2.data) * 100 + 50  # Scale to reasonable range

    for threshold in (0.0, 10.0, 30.0):
        p = sigima.params.FlatFieldParam.create(threshold=threshold)
        dst = sigima.proc.image.flatfield(src1, src2, p)

        # Compute expected result using the same algorithm as in sigima.tools.image
        dtemp = np.array(src1.data, dtype=float, copy=True) * np.nanmean(src2.data)
        dunif = np.array(src2.data, dtype=float, copy=True)
        dunif[dunif == 0] = 1.0
        dcorr_all = np.array(dtemp / dunif, dtype=src1.data.dtype)
        exp = np.array(src1.data, copy=True)
        exp[src1.data > threshold] = dcorr_all[src1.data > threshold]

        check_array_result(f"FlatField[threshold={threshold}]", dst.data, exp)


@pytest.mark.validation
def test_image_normalize() -> None:
    """Validation test for the image normalization processing."""
    src = get_test_image("flower.npy")
    src.data = np.array(src.data, dtype=float)
    src.data[20:30, 20:30] = np.nan  # Adding NaN values to the image
    p = sigima.params.NormalizeParam()

    # Given the fact that the normalization methods implementations are
    # straightforward, we do not need to compare arrays with each other,
    # we simply need to check if some properties are satisfied.
    for method in sigima.enums.NormalizationMethod:
        p.method = method
        dst = sigima.proc.image.normalize(src, p)
        title = f"Normalize[method='{p.method}']"
        exp_min, exp_max = None, None
        if p.method == sigima.enums.NormalizationMethod.MAXIMUM:
            exp_min, exp_max = np.nanmin(src.data) / np.nanmax(src.data), 1.0
        elif p.method == sigima.enums.NormalizationMethod.AMPLITUDE:
            exp_min, exp_max = 0.0, 1.0
        elif p.method == sigima.enums.NormalizationMethod.AREA:
            area = np.nansum(src.data)
            exp_min, exp_max = np.nanmin(src.data) / area, np.nanmax(src.data) / area
        elif p.method == sigima.enums.NormalizationMethod.ENERGY:
            energy = np.sqrt(np.nansum(np.abs(src.data) ** 2))
            exp_min, exp_max = (
                np.nanmin(src.data) / energy,
                np.nanmax(src.data) / energy,
            )
        elif p.method == sigima.enums.NormalizationMethod.RMS:
            rms = np.sqrt(np.nanmean(np.abs(src.data) ** 2))
            exp_min, exp_max = np.nanmin(src.data) / rms, np.nanmax(src.data) / rms
        check_scalar_result(f"{title}|min", np.nanmin(dst.data), exp_min)
        check_scalar_result(f"{title}|max", np.nanmax(dst.data), exp_max)


@pytest.mark.validation
def test_image_clip() -> None:
    """Validation test for the image clipping processing."""
    src = get_test_image("flower.npy")
    p = sigima.params.ClipParam()

    for lower, upper in ((float("-inf"), float("inf")), (50, 100)):
        p.lower, p.upper = lower, upper
        dst = sigima.proc.image.clip(src, p)
        exp = np.clip(src.data, p.lower, p.upper)
        check_array_result(f"Clip[{lower},{upper}]", dst.data, exp)


@pytest.mark.validation
def test_image_histogram() -> None:
    """Validation test for the image histogram computation function."""
    src = get_test_image("flower.npy")
    for bins in (128, 256, 512):
        for lower, upper in ((None, None), (50.0, 200.0)):
            p = sigima.params.HistogramParam.create(bins=bins, lower=lower, upper=upper)
            dst = sigima.proc.image.histogram(src, p)

            # Get the actual data used for histogram computation
            data = src.get_masked_view().compressed()

            # Determine the range for numpy.histogram
            hist_range = (p.lower, p.upper)
            if p.lower is None:
                hist_range = (np.min(data), hist_range[1])
            if p.upper is None:
                hist_range = (hist_range[0], np.max(data))

            # Compute expected histogram using numpy.histogram
            exp_y, bin_edges = np.histogram(data, bins=p.bins, range=hist_range)
            exp_x = (bin_edges[:-1] + bin_edges[1:]) / 2

            title = f"Histogram[bins={bins},lower={lower},upper={upper}]"
            check_array_result(f"{title}|x", dst.x, exp_x)
            check_array_result(f"{title}|y", dst.y, np.array(exp_y, dtype=float))


@pytest.mark.validation
def test_image_offset_correction() -> None:
    """Validation test for the image offset correction processing."""
    src = get_test_image("flower.npy")
    # Defining the ROI that will be used to estimate the offset
    p = sigima.objects.ROI2DParam.create(x0=0, y0=0, dx=50, dy=20)
    dst = sigima.proc.image.offset_correction(src, p)
    ix0, iy0 = int(p.x0), int(p.y0)
    ix1, iy1 = int(p.x0 + p.dx), int(p.y0 + p.dy)
    exp = src.data - np.mean(src.data[iy0:iy1, ix0:ix1])
    check_array_result("OffsetCorrection", dst.data, exp)


if __name__ == "__main__":
    test_adjust_gamma()
    test_adjust_log()
    test_adjust_sigmoid()
    test_rescale_intensity()
    test_equalize_hist()
    test_equalize_adapthist()
    test_flatfield()
    test_image_normalize()
    test_image_clip()
    test_image_histogram()
    test_image_offset_correction()
