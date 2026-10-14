# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Parameter codec of operation contracts
======================================

Encodes the parameter arguments actually passed to a computation function as
plain JSON values, and rebuilds them, guided only by the types declared by the
local contract. No class name, Python expression or pickle is ever read from
the encoded values.

Format (codec version 2, a superset of version 1):

- A parameter set (:class:`guidata.dataset.DataSet`) is a JSON object mapping
  each item name to its value; transient, computed, group and button items are
  excluded. Enum choices are encoded by their stable ``.value``, other choices by
  their key; on decoding, a choice is resolved against the keys declared by the
  item.
- Typed forms, all single-key objects whose key starts with ``$``:
  ``{"$float": "NaN" | "Infinity" | "-Infinity"}`` for non-finite floats,
  ``{"$complex": [re, im]}``, ``{"$tuple": [...]}``, ``{"$dict": {...}}``,
  ``{"$date": "YYYY-MM-DD"}``, ``{"$datetime": "ISO 8601"}`` and
  ``{"$ndarray": {"dtype": "<f8", "shape": [...], "base64": "..."}}`` (raw
  little-endian C-order bytes; boolean, integer, float and complex dtypes only).
- The parameters of a call (:func:`encode_parameters`) are the encoded parameter
  set itself when the operation takes a single parameter set (version 1 form),
  and otherwise an envelope object mapping each parameter argument name to a
  parameter set, a list of parameter sets or null.

Version 1 values (flat scalar items) are decoded unchanged.
"""

from __future__ import annotations

import base64
import datetime
import json
import math
import re
from collections.abc import Sequence
from enum import Enum
from typing import Any

import guidata.dataset as gds
import numpy as np
from guidata.dataset.datatypes import BeginGroup, ComputedProp, EndGroup

from sigima.proc.contract_types import (
    InvalidParametersError,
    ParameterEncodingError,
    ParameterSpec,
)

__all__ = [
    "PARAMETER_CODEC_VERSION",
    "decode_dataset",
    "decode_parameters",
    "encode_dataset",
    "encode_parameters",
    "parameter_items",
]

#: Version of the parameter encoding produced by this module.
PARAMETER_CODEC_VERSION = 2

_NON_FINITE = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}
_DTYPE_PATTERN = re.compile(r"^[<|][biufc][0-9]+$")
_TYPED_KEYS = ("$float", "$complex", "$tuple", "$dict", "$date", "$datetime")


def parameter_items(paramclass: type[gds.DataSet]) -> list[gds.DataItem]:
    """Return the items that make up the parameter values of *paramclass*."""
    items = []
    for item in paramclass._items:  # pylint: disable=protected-access
        if isinstance(item, (BeginGroup, EndGroup, gds.ButtonItem)):
            continue
        if item.get_prop("data", "transient", False):
            continue
        if isinstance(item.get_prop("data", "computed", None), ComputedProp):
            continue
        items.append(item)
    return items


# --- Generic values ----------------------------------------------------------


def _encode_float(value: float) -> Any:
    if math.isnan(value):
        return {"$float": "NaN"}
    if math.isinf(value):
        return {"$float": "Infinity" if value > 0 else "-Infinity"}
    return value


def _encode_array(name: str, value: np.ndarray) -> dict[str, Any]:
    dtype = (
        value.dtype.newbyteorder("<") if value.dtype.byteorder == ">" else value.dtype
    )
    if dtype.kind not in "biufc" or not _DTYPE_PATTERN.match(dtype.str):
        raise ParameterEncodingError(
            f"Parameter {name!r}: arrays of dtype {value.dtype} cannot be encoded"
        )
    data = np.ascontiguousarray(value, dtype=dtype)
    return {
        "$ndarray": {
            "dtype": dtype.str,
            "shape": list(data.shape),
            "base64": base64.b64encode(data.tobytes()).decode("ascii"),
        }
    }


def encode_value(name: str, value: Any) -> Any:
    """Encode a generic value as plain JSON values.

    Args:
        name: Parameter name, for error messages.
        value: Value to encode.

    Raises:
        ParameterEncodingError: If the value has an unsupported type.
    """
    # pylint: disable=too-many-return-statements
    if isinstance(value, np.ndarray):
        return _encode_array(name, value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Enum):
        return encode_value(name, value.value)
    if value is None or isinstance(value, (bool, str)):
        return str(value) if isinstance(value, str) else value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _encode_float(float(value))
    if isinstance(value, complex):
        return {"$complex": [_encode_float(value.real), _encode_float(value.imag)]}
    if isinstance(value, list):
        return [encode_value(name, v) for v in value]
    if isinstance(value, tuple):
        return {"$tuple": [encode_value(name, v) for v in value]}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ParameterEncodingError(f"Parameter {name!r}: keys must be strings")
        return {"$dict": {key: encode_value(name, v) for key, v in value.items()}}
    if isinstance(value, datetime.datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, datetime.date):
        return {"$date": value.isoformat()}
    raise ParameterEncodingError(
        f"Parameter {name!r}: values of type {type(value).__name__} cannot be encoded"
    )


def _decode_float(name: str, value: Any) -> float:
    if isinstance(value, dict):
        if set(value) != {"$float"} or value["$float"] not in _NON_FINITE:
            raise InvalidParametersError(f"Parameter {name!r}: invalid float")
        return _NON_FINITE[value["$float"]]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParametersError(f"Parameter {name!r}: a number is expected")
    return float(value)


def _decode_array(name: str, value: Any) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "base64"}:
        raise InvalidParametersError(f"Parameter {name!r}: invalid array")
    dtype_str, shape, data = value["dtype"], value["shape"], value["base64"]
    if not isinstance(dtype_str, str) or not _DTYPE_PATTERN.match(dtype_str):
        raise InvalidParametersError(f"Parameter {name!r}: invalid array dtype")
    if not isinstance(shape, list) or not all(
        isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape
    ):
        raise InvalidParametersError(f"Parameter {name!r}: invalid array shape")
    try:
        dtype = np.dtype(dtype_str)
        raw = base64.b64decode(data, validate=True) if isinstance(data, str) else None
    except (TypeError, ValueError) as exc:
        raise InvalidParametersError(f"Parameter {name!r}: invalid array") from exc
    if raw is None or len(raw) != dtype.itemsize * math.prod(shape):
        raise InvalidParametersError(f"Parameter {name!r}: invalid array data")
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return array.astype(dtype.newbyteorder("="), copy=True)


def decode_value(name: str, value: Any) -> Any:
    """Decode a generic value produced by :func:`encode_value`.

    Raises:
        InvalidParametersError: If the value is malformed.
    """
    # pylint: disable=too-many-return-statements
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return [decode_value(name, v) for v in value]
    if not isinstance(value, dict) or len(value) != 1:
        raise InvalidParametersError(f"Parameter {name!r}: invalid value {value!r}")
    key, content = next(iter(value.items()))
    if key == "$float":
        return _decode_float(name, value)
    if key == "$complex":
        if not isinstance(content, list) or len(content) != 2:
            raise InvalidParametersError(f"Parameter {name!r}: invalid complex")
        return complex(_decode_float(name, content[0]), _decode_float(name, content[1]))
    if key == "$tuple" and isinstance(content, list):
        return tuple(decode_value(name, v) for v in content)
    if key == "$dict" and isinstance(content, dict):
        return {k: decode_value(name, v) for k, v in content.items()}
    if key == "$ndarray":
        return _decode_array(name, content)
    if key in ("$date", "$datetime") and isinstance(content, str):
        cls = datetime.datetime if key == "$datetime" else datetime.date
        try:
            return cls.fromisoformat(content)
        except ValueError as exc:
            raise InvalidParametersError(f"Parameter {name!r}: invalid date") from exc
    raise InvalidParametersError(f"Parameter {name!r}: invalid value {value!r}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


# --- Parameter sets ----------------------------------------------------------


def _choice_keys(item: gds.ChoiceItem) -> list[Any] | None:
    choices = item.get_prop("data", "choices")
    if callable(choices):
        return None
    return [choice[0] for choice in choices]


def _encode_item(item: gds.DataItem, raw: Any) -> Any:
    name = item.get_name()
    enum_cls = getattr(item, "_enum_cls", None)
    if isinstance(item, gds.ChoiceItem) and enum_cls is not None:
        if isinstance(raw, (list, tuple)):
            return [enum_cls[v].value for v in raw]
        return None if raw is None else enum_cls[raw].value
    if isinstance(item, gds.MultipleChoiceItem) and isinstance(raw, (list, tuple)):
        return [encode_value(name, v) for v in raw]
    return encode_value(name, raw)


def encode_dataset(
    param: gds.DataSet, paramclass: type[gds.DataSet] | None = None
) -> dict[str, Any]:
    """Encode a parameter set as plain JSON values.

    Args:
        param: Parameter set instance actually used.
        paramclass: Declared parameter class; only its items are encoded (an
         instance of a subclass may carry other items, which the computation
         does not use). Default: the class of *param*.

    Returns:
        Mapping of item names to JSON values.

    Raises:
        ParameterEncodingError: If an item value cannot be encoded.
    """
    values: dict[str, Any] = {}
    for item in parameter_items(paramclass or type(param)):
        values[item.get_name()] = _encode_item(item, item.get_value(param))
    json.dumps(values, allow_nan=False)
    return values


def _decode_choice(item: gds.ChoiceItem, value: Any) -> Any:
    name = item.get_name()
    enum_cls: type[Enum] | None = getattr(item, "_enum_cls", None)
    if enum_cls is not None:
        if isinstance(item, gds.MultipleChoiceItem) and isinstance(value, list):
            return [_decode_enum(name, enum_cls, v) for v in value]
        return _decode_enum(name, enum_cls, value)
    keys = _choice_keys(item)
    if keys is None:
        return decode_value(name, value)
    by_value: dict[str, Any] = {}
    for key in keys:
        try:
            by_value.setdefault(_canonical(encode_value(name, key)), key)
        except (ParameterEncodingError, ValueError):
            continue
    values = (
        value
        if isinstance(item, gds.MultipleChoiceItem) and isinstance(value, list)
        else [value]
    )
    resolved = []
    for val in values:
        try:
            resolved.append(by_value[_canonical(val)])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidParametersError(
                f"Parameter {name!r}: {val!r} is not a declared choice"
            ) from exc
    return resolved if values is value else resolved[0]


def _decode_enum(name: str, enum_cls: type[Enum], value: Any) -> str:
    for member in enum_cls:
        if isinstance(value, str) and value == member.value:
            return member.name
    raise InvalidParametersError(
        f"Parameter {name!r}: {value!r} is not a value of {enum_cls.__name__}"
    )


def _accepts_none(item: gds.DataItem) -> bool:
    """True if *item* accepts None (``allow_none``, or validation disabled)."""
    if item.get_prop("data", "allow_none", False):
        return True
    try:
        return bool(item.check_value(None))
    except Exception:  # pylint: disable=broad-except
        return False


def _decode_item(item: gds.DataItem, value: Any) -> Any:
    """Return the raw stored value of *item* for the JSON *value*."""
    # pylint: disable=too-many-branches
    name = item.get_name()
    if value is None and not (
        isinstance(item, gds.ChoiceItem) and None in (_choice_keys(item) or ())
    ):
        if not _accepts_none(item):
            raise InvalidParametersError(f"Parameter {name!r} cannot be null")
        return None
    if isinstance(item, gds.ChoiceItem):
        raw = _decode_choice(item, value)
    elif isinstance(item, gds.FloatArrayItem):
        if not isinstance(value, dict) or set(value) != {"$ndarray"}:
            raise InvalidParametersError(f"Parameter {name!r}: an array is expected")
        raw = _decode_array(name, value["$ndarray"])
    elif isinstance(item, gds.FloatItem):
        raw = _decode_float(name, value)
    elif isinstance(item, gds.IntItem):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidParametersError(f"Parameter {name!r}: an integer is expected")
        raw = value
    elif isinstance(item, gds.BoolItem):
        if not isinstance(value, bool):
            raise InvalidParametersError(f"Parameter {name!r}: a boolean is expected")
        raw = value
    elif isinstance(item, gds.StringItem):
        if not isinstance(value, str):
            raise InvalidParametersError(f"Parameter {name!r}: a string is expected")
        raw = value
    elif isinstance(item, (gds.DictItem, gds.DateItem)):
        raw = decode_value(name, value)
    else:
        raise InvalidParametersError(
            f"Parameter {name!r}: items of type {type(item).__name__} are not supported"
        )
    try:
        valid = item.check_value(raw, raise_exception=True)
    except NotImplementedError:
        valid = True
    except Exception as exc:  # pylint: disable=broad-except
        raise InvalidParametersError(f"Parameter {name!r}: {exc}") from exc
    if valid is False:
        raise InvalidParametersError(f"Parameter {name!r}: invalid value {value!r}")
    return raw


def decode_dataset(paramclass: type[gds.DataSet], values: Any) -> gds.DataSet:
    """Rebuild a parameter set of the declared *paramclass*.

    Every item is required; unknown items are refused; enum choices are accepted
    only by ``.value`` and other choices only among the declared keys.

    Args:
        paramclass: Parameter class declared by the local contract.
        values: Mapping produced by :func:`encode_dataset`.

    Returns:
        A new parameter instance.

    Raises:
        InvalidParametersError: If *values* do not match the parameter class.
    """
    if not isinstance(values, dict) or not all(isinstance(k, str) for k in values):
        raise InvalidParametersError("Parameters must be a JSON object")
    items = parameter_items(paramclass)
    names = [item.get_name() for item in items]
    unknown = sorted(set(values) - set(names))
    missing = [name for name in names if name not in values]
    if unknown or missing:
        raise InvalidParametersError(
            f"{paramclass.__name__}: unknown parameters {unknown}, "
            f"missing parameters {missing}"
        )
    param = paramclass()
    for item in items:
        raw = _decode_item(item, values[item.get_name()])
        # Already checked: bypass the global guidata validation mode.
        setattr(param, f"_{item.get_name()}", raw)
    return param


# --- Parameter arguments of a call -------------------------------------------


def _is_flat(specs: Sequence[ParameterSpec]) -> bool:
    return len(specs) == 1 and specs[0].container == "single"


def _encode_argument(spec: ParameterSpec, value: Any) -> Any:
    if value is None:
        if not spec.optional:
            raise ParameterEncodingError(f"Parameter {spec.name!r} is required")
        if spec.container == "single":
            # None stands for a default instance: record the values actually used.
            return encode_dataset(spec.paramclass())
        return None
    multiple = isinstance(value, (list, tuple))
    if multiple and spec.container in ("list", "single_or_list"):
        if not all(isinstance(v, spec.paramclass) for v in value):
            raise ParameterEncodingError(
                f"Parameter {spec.name!r}: {spec.paramclass.__name__} items expected"
            )
        return [encode_dataset(v, spec.paramclass) for v in value]
    if not multiple and spec.container in ("single", "single_or_list"):
        if not isinstance(value, spec.paramclass):
            raise ParameterEncodingError(
                f"Parameter {spec.name!r}: {spec.paramclass.__name__} expected"
            )
        return encode_dataset(value, spec.paramclass)
    raise ParameterEncodingError(
        f"Parameter {spec.name!r}: unexpected container {type(value).__name__}"
    )


def encode_parameters(
    specs: Sequence[ParameterSpec], arguments: Sequence[Any]
) -> dict[str, Any]:
    """Encode the parameter arguments of a call.

    Args:
        specs: Parameter arguments declared by the contract.
        arguments: Values actually passed, in the same order (missing trailing
         arguments stand for None).

    Returns:
        The encoded parameters (see the module documentation).

    Raises:
        ParameterEncodingError: If an argument cannot be encoded.
    """
    if len(arguments) > len(specs):
        raise ParameterEncodingError(
            f"{len(arguments)} parameter arguments for {len(specs)} declared"
        )
    arguments = list(arguments) + [None] * (len(specs) - len(arguments))
    if _is_flat(specs):
        return _encode_argument(specs[0], arguments[0])
    return {
        spec.name: _encode_argument(spec, value)
        for spec, value in zip(specs, arguments)
    }


def _decode_argument(spec: ParameterSpec, value: Any) -> Any:
    if value is None:
        if not spec.optional:
            raise InvalidParametersError(f"Parameter {spec.name!r} cannot be null")
        return None
    if isinstance(value, list) and spec.container in ("list", "single_or_list"):
        return [decode_dataset(spec.paramclass, v) for v in value]
    if isinstance(value, dict) and spec.container in ("single", "single_or_list"):
        return decode_dataset(spec.paramclass, value)
    raise InvalidParametersError(f"Parameter {spec.name!r}: unexpected value")


def decode_parameters(specs: Sequence[ParameterSpec], values: Any) -> list[Any]:
    """Rebuild the parameter arguments of a call from :func:`encode_parameters`.

    Args:
        specs: Parameter arguments declared by the local contract.
        values: Encoded parameters.

    Returns:
        The parameter arguments, in call order.

    Raises:
        InvalidParametersError: If *values* do not match the declarations.
    """
    if not isinstance(values, dict) or not all(isinstance(k, str) for k in values):
        raise InvalidParametersError("Parameters must be a JSON object")
    if not specs:
        if values:
            raise InvalidParametersError("The operation has no parameters")
        return []
    if _is_flat(specs):
        return [decode_dataset(specs[0].paramclass, values)]
    names = [spec.name for spec in specs]
    if sorted(values) != sorted(names):
        raise InvalidParametersError(
            f"Expected parameter arguments {names}, got {sorted(values)}"
        )
    return [_decode_argument(spec, values[spec.name]) for spec in specs]
