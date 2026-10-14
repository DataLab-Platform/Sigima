# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the parameter codec of operation contracts."""

from __future__ import annotations

import base64
import datetime
import json
import math

import guidata.dataset as gds
import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.image
import sigima.proc.signal
from sigima.config import options
from sigima.enums import FrequencyFilterMethod
from sigima.proc import contracts
from sigima.proc.contract_types import ParameterSpec
from sigima.proc.parameter_codec import (
    decode_dataset,
    decode_parameters,
    decode_value,
    encode_dataset,
    encode_parameters,
    encode_value,
    parameter_items,
)


def _json_round_trip(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _assert_same_dataset(first: gds.DataSet, second: gds.DataSet) -> None:
    assert type(first) is type(second)
    for item in parameter_items(type(first)):
        a, b = item.get_value(first), item.get_value(second)
        if isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray) and a.dtype == b.dtype
            assert a.shape == b.shape
            assert np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        elif isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b)
        else:
            assert type(a) is type(b) and a == b, item.get_name()


def _valid_instance(paramclass: type[gds.DataSet]) -> gds.DataSet:
    """Return a default instance whose required empty items are filled."""
    param = paramclass()
    for item in parameter_items(paramclass):
        if item.get_value(param) is not None or item.get_prop(
            "data", "allow_none", False
        ):
            continue
        if isinstance(item, gds.FloatItem):
            candidates = [1.0, item.get_prop("data", "min"), 0.0]
        elif isinstance(item, gds.IntItem):
            candidates = [1, item.get_prop("data", "min"), 0]
        else:
            candidates = []
        for candidate in candidates:
            if candidate is not None and item.check_value(candidate):
                setattr(param, f"_{item.get_name()}", type(candidates[0])(candidate))
                break
        else:
            raise AssertionError(f"No valid value for {paramclass.__name__}.{item}")
    return param


def _default_argument(spec: ParameterSpec):
    if spec.container == "list":
        return [_valid_instance(spec.paramclass), _valid_instance(spec.paramclass)]
    return _valid_instance(spec.paramclass)


def test_catalogue_round_trip() -> None:
    """Default parameters of every contract survive a JSON round trip."""
    registry = contracts.build_contract_registry()
    checked = 0
    for contract in registry.contracts:
        if not contract.parameters:
            assert contract.encode_parameters([]) == {}
            assert contract.decode_parameters({}) == []
            continue
        arguments = [_default_argument(spec) for spec in contract.parameters]
        values = _json_round_trip(contract.encode_parameters(arguments))
        decoded = contract.decode_parameters(values)
        assert contract.encode_parameters(decoded) == values
        for original, rebuilt in zip(arguments, decoded):
            if isinstance(original, list):
                assert len(original) == len(rebuilt)
                for first, second in zip(original, rebuilt):
                    _assert_same_dataset(first, second)
            else:
                _assert_same_dataset(original, rebuilt)
        checked += 1
    assert checked > 100


def test_roi_lists() -> None:
    """Lists of ROI parameter sets keep their order, values and dtypes."""
    contract = contracts.contract_for_function(sigima.proc.signal.extract_rois)
    rois = [
        sigima.objects.ROI1DParam.create(title="a", xmin=0.5, xmax=1.0),
        sigima.objects.ROI1DParam.create(title="b", xmin=-math.inf, xmax=2.0),
    ]
    values = _json_round_trip(contract.encode_parameters([rois]))
    assert list(values) == ["params"]
    assert [roi["title"] for roi in values["params"]] == ["a", "b"]
    assert values["params"][1]["xmin"] == {"$float": "-Infinity"}
    (decoded,) = contract.decode_parameters(values)
    for first, second in zip(rois, decoded):
        _assert_same_dataset(first, second)
    image_contract = contracts.contract_for_function(sigima.proc.image.extract_rois)
    polygon = sigima.objects.ROI2DParam.create(
        geometry="polygon", points=np.array([0.0, 0.0, 4.0, 0.0, 2.0, 3.5])
    )
    values = _json_round_trip(image_contract.encode_parameters([[polygon]]))
    (decoded,) = image_contract.decode_parameters(values)
    _assert_same_dataset(polygon, decoded[0])


def test_single_or_list_is_preserved() -> None:
    """A single parameter set and a list of one are restored as passed."""
    contract = contracts.contract_for_function(sigima.proc.image.erase)
    roi = sigima.objects.ROI2DParam.create(geometry="rectangle", x0=1, y0=2, dx=3, dy=4)
    single = contract.decode_parameters(contract.encode_parameters([roi]))[0]
    assert isinstance(single, sigima.objects.ROI2DParam)
    listed = contract.decode_parameters(contract.encode_parameters([[roi]]))[0]
    assert isinstance(listed, list) and len(listed) == 1
    with pytest.raises(contracts.ParameterEncodingError):
        contract.encode_parameters([[roi, 3]])
    with pytest.raises(contracts.ParameterEncodingError):
        contract.encode_parameters([None])


def test_declared_choice_keys() -> None:
    """Choices are resolved against the keys declared by the local item."""
    param = sigima.params.PulseFeaturesParam()
    param.reference_levels = (20, 80)
    param.signal_shape = None
    values = _json_round_trip(encode_dataset(param))
    assert values["reference_levels"] == {"$tuple": [20, 80]}
    assert values["signal_shape"] is None
    decoded = decode_dataset(sigima.params.PulseFeaturesParam, values)
    assert decoded.reference_levels == (20, 80) and decoded.signal_shape is None
    with pytest.raises(contracts.InvalidParametersError):
        decode_dataset(
            sigima.params.PulseFeaturesParam,
            dict(values, reference_levels={"$tuple": [1, 2]}),
        )
    lowpass = sigima.params.LowPassFilterParam()
    lowpass.method = FrequencyFilterMethod.BESSEL
    values = _json_round_trip(encode_dataset(lowpass))
    assert values["method"] == FrequencyFilterMethod.BESSEL.value
    decoded = decode_dataset(sigima.params.LowPassFilterParam, values)
    assert decoded.method is FrequencyFilterMethod.BESSEL


def test_optional_parameter_records_effective_values() -> None:
    """None for an optional parameter set records the default values used."""
    contract = contracts.contract_for_function(sigima.proc.signal.fft)
    for shift in (False, True):
        with options.fft_shift_enabled.context(shift):
            values = contract.encode_parameters([None])
        assert values == {"shift": shift}
        (decoded,) = contract.decode_parameters(values)
        assert decoded.shift is shift
    assert contract.encode_parameters([]) == contract.encode_parameters([None])


def test_version_1_values_decode_unchanged() -> None:
    """Flat scalar values written by the first codec are still accepted."""
    contract = contracts.get_operation_contract("sigima.signal.normalize", 1)
    (param,) = contract.decode_parameters({"method": "maximum"})
    assert contracts.parameters_to_values(param) == {"method": "maximum"}
    roi = decode_dataset(
        sigima.objects.ROI1DParam,
        {"title": "", "xmin": {"$float": "-Infinity"}, "xmax": 1},
    )
    assert roi.xmin == -math.inf and roi.xmax == 1.0 and isinstance(roi.xmax, float)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        3,
        -2.5,
        "text",
        math.nan,
        math.inf,
        complex(1.0, -math.inf),
        [1, "a", None],
        (1, (2.0, "b")),
        {"a": [1, {"b": 2.0}]},
        datetime.date(2026, 10, 14),
        datetime.datetime(2026, 10, 14, 12, 30, 5),
        np.float32(1.5),
        np.arange(6, dtype=np.int16).reshape(2, 3),
        np.array([1 + 2j, math.nan], dtype=np.complex128),
        np.array([True, False]),
        np.array([1.0, 2.0], dtype=">f4"),
        np.zeros((0, 3), dtype=np.uint8),
    ],
)
def test_generic_values_round_trip(value) -> None:
    """Generic values survive a JSON round trip with their type."""
    decoded = decode_value("v", _json_round_trip(encode_value("v", value)))
    if isinstance(value, np.ndarray):
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == value.dtype.newbyteorder("=")
        assert decoded.shape == value.shape
        assert np.array_equal(decoded, value, equal_nan=value.dtype.kind in "fc")
    elif isinstance(value, np.generic):
        assert decoded == value.item() and type(decoded) is type(value.item())
    elif isinstance(value, float) and math.isnan(value):
        assert math.isnan(decoded)
    else:
        assert decoded == value and type(decoded) is type(value)


def test_encoding_is_stable() -> None:
    """Typed forms are fixed by the codec version."""
    assert encode_value("v", np.array([1.0, -0.5])) == {
        "$ndarray": {
            "dtype": "<f8",
            "shape": [2],
            "base64": base64.b64encode(
                np.array([1.0, -0.5], dtype="<f8").tobytes()
            ).decode("ascii"),
        }
    }
    assert encode_value("v", complex(1, 2)) == {"$complex": [1.0, 2.0]}
    assert encode_value("v", (1, 2)) == {"$tuple": [1, 2]}
    assert encode_value("v", {"k": math.nan}) == {"$dict": {"k": {"$float": "NaN"}}}
    assert encode_value("v", datetime.date(2026, 1, 2)) == {"$date": "2026-01-02"}


@pytest.mark.parametrize(
    "value",
    [
        np.array([object()]),
        np.array(["a"]),
        {1: 2},
        object(),
        {1, 2},
    ],
)
def test_unsupported_values_are_refused(value) -> None:
    """Values without a typed form cannot be encoded."""
    with pytest.raises(contracts.ParameterEncodingError):
        encode_value("v", value)


@pytest.mark.parametrize(
    "value",
    [
        {"$unknown": 1},
        {"$float": "inf"},
        {"$complex": [1.0]},
        {"$tuple": "ab"},
        {"$dict": [1]},
        {"$date": "14/10/2026"},
        {"$ndarray": {"dtype": "|O8", "shape": [1], "base64": ""}},
        {"$ndarray": {"dtype": ">f8", "shape": [1], "base64": "AAAAAAAAAAA="}},
        {"$ndarray": {"dtype": "<f8", "shape": [2], "base64": "AAAAAAAAAAA="}},
        {"$ndarray": {"dtype": "<f8", "shape": [1], "base64": "not base64!"}},
        {"$ndarray": {"dtype": "<f8", "shape": [-1], "base64": ""}},
        {"$ndarray": {"dtype": "<f8", "shape": [1]}},
        {"a": 1, "b": 2},
    ],
)
def test_malformed_values_are_refused(value) -> None:
    """Malformed typed forms are refused explicitly."""
    with pytest.raises(contracts.InvalidParametersError):
        decode_value("v", value)


def test_envelope_is_checked() -> None:
    """Envelopes must name exactly the declared parameter arguments."""
    specs = (
        ParameterSpec("rois", sigima.objects.ROI1DParam, "list"),
        ParameterSpec("p", sigima.params.NormalizeParam, "single", optional=True),
    )
    roi = sigima.objects.ROI1DParam.create(xmin=0.0, xmax=1.0)
    values = _json_round_trip(encode_parameters(specs, [[roi], None]))
    assert values["p"] == encode_dataset(sigima.params.NormalizeParam())
    decoded = decode_parameters(specs, values)
    assert len(decoded[0]) == 1 and isinstance(decoded[1], sigima.params.NormalizeParam)
    for bad in (
        [],
        {"rois": []},
        dict(values, extra=1),
        dict(values, rois=values["rois"][0]),
        dict(values, rois=None),
        dict(values, p=[]),
    ):
        with pytest.raises(contracts.InvalidParametersError):
            decode_parameters(specs, bad)
    with pytest.raises(contracts.ParameterEncodingError):
        encode_parameters(specs, [roi, None])
    with pytest.raises(contracts.ParameterEncodingError):
        encode_parameters(specs, [[roi], None, None])
    with pytest.raises(contracts.InvalidParametersError):
        decode_parameters((), {"a": 1})
