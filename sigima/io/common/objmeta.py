# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Sigima I/O module for handling object metadata and ROIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from guidata.io import JSONHandler, JSONReader, JSONWriter

from sigima.objects import (
    GraphicalAnnotation,
    ImageROI,
    SignalROI,
    annotation_from_dict,
    annotation_to_dict,
    is_graphical_annotation_dict,
)

if TYPE_CHECKING:
    from sigima.params import ROIGridParam

FORMAT_TAG = "sigima"
FORMAT_VERSION = "1.0"
ROI_TYPE_FIELD = "roi_type"


def _check_tag(data: dict, expected_format: str) -> None:
    """Validate the presence and type of sigima tag.

    Args:
        data: The data dictionary to check.
        expected_format: The expected format string for the tag.

    Raises:
        ValueError: If the tag is missing or does not match the expected format.
    """
    tag: dict = data.get(FORMAT_TAG, {})
    if tag.get("format") != expected_format:
        raise ValueError(f"Unexpected or missing format: {tag}")


def write_dict(filepath: str, data: dict) -> None:
    """Write a dictionary to a file in JSON format.

    Args:
        filepath: The file path to write the data to.
        data: The dictionary to serialize.

    Raises:
        ValueError: If the data is not a dictionary.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary, got {type(data)}")
    handler = JSONHandler(filepath)
    handler.set_json_dict(data)
    handler.save()


def read_dict(filepath: str) -> dict:
    """Read a dictionary from a file and return it.

    Args:
        filepath: The file path to read the data from.

    Returns:
        The dictionary read from the file.
    """
    handler = JSONHandler(filepath)
    handler.load()
    data = handler.get_json_dict()
    return data


def write_roi(filepath: str, roi: SignalROI | ImageROI) -> None:
    """Write a signal or image ROI to a file in JSON format.

    Args:
        filepath: The file path to write the ROI data to.
        roi: The signal or image ROI object to serialize.

    Raises:
        ValueError: If the ROI object is not of type SignalROI or ImageROI.
    """
    if isinstance(roi, SignalROI):
        roi_type: Literal["signal", "image"] = "signal"
    elif isinstance(roi, ImageROI):
        roi_type = "image"
    else:
        raise ValueError(
            f"Unsupported ROI type: {type(roi)}. Expected SignalROI or ImageROI."
        )
    roi_dict = roi.to_dict()
    roi_dict[ROI_TYPE_FIELD] = roi_type
    data = {
        FORMAT_TAG: {"format": "roi", "version": FORMAT_VERSION},
        "roi": roi_dict,
    }
    write_dict(filepath, data)


def read_roi(filepath: str) -> SignalROI | ImageROI:
    """Read ROI data from a file and return the corresponding ROI object.

    Args:
        filepath: The file path to read the ROI data from.

    Returns:
        The corresponding ROI object (SignalROI or ImageROI).

    Raises:
        ValueError: If the file does not contain the expected format.
    """
    json_dict = read_dict(filepath)
    _check_tag(json_dict, expected_format="roi")
    roi_dict = json_dict["roi"]
    assert isinstance(roi_dict, dict), "ROI data must be a dictionary"
    roi_type = roi_dict.pop(ROI_TYPE_FIELD, None)
    if roi_type == "signal":
        return SignalROI.from_dict(roi_dict)
    if roi_type == "image":
        return ImageROI.from_dict(roi_dict)
    raise ValueError(f"Unsupported or missing ROI type: {roi_type}")


def write_roi_grid(filepath: str, param: ROIGridParam) -> None:
    """Write ROI grid parameters to a file in JSON format.

    Args:
        filepath: The file path to write the ROI grid parameters to.
        param: The ROI grid parameters to serialize.
    """
    writer = JSONWriter(filepath)
    param.serialize(writer)
    print(writer.jsondata)
    writer.save()


def read_roi_grid(filepath: str) -> ROIGridParam:
    """Read ROI grid parameters from a file in JSON format.

    Args:
        filepath: The file path to read the ROI grid parameters from.

    Returns:
        The ROI grid parameters read from the file.
    """
    from sigima.params import ROIGridParam  # pylint: disable=import-outside-toplevel

    handler = JSONReader(filepath)
    handler.load()
    param = ROIGridParam()
    param.deserialize(handler)
    return param


def write_metadata(filepath: str, metadata: dict[str, Any]) -> None:
    """Write metadata to a file in JSON format.

    Args:
        filepath: The file path to write the metadata to.
        metadata: The metadata dictionary to serialize.

    Raises:
        ValueError: If the object does not have a metadata attribute.
    """
    data = {
        FORMAT_TAG: {"format": "metadata", "version": FORMAT_VERSION},
        "metadata": metadata.copy(),
    }
    write_dict(filepath, data)


def read_metadata(filepath: str) -> dict[str, Any]:
    """Read metadata from a file and return the metadata dictionary.

    Args:
        filepath: The file path to read the metadata from.

    Returns:
        The metadata dictionary.

    Raises:
        ValueError: If the file does not contain the expected format.
    """
    json_dict = read_dict(filepath)
    _check_tag(json_dict, expected_format="metadata")
    return json_dict["metadata"]


def write_annotations(filepath: str, annotations: list[dict[str, Any]]) -> None:
    """Write annotations to a file in JSON format.

    Args:
        filepath: The file path to write the annotations to.
        annotations: The annotations list to serialize.
    """
    data = {
        FORMAT_TAG: {"format": "annotations", "version": FORMAT_VERSION},
        "annotations": annotations.copy(),
    }
    write_dict(filepath, data)


def read_annotations(filepath: str) -> list[dict[str, Any]]:
    """Read annotations from a file and return the annotations list.

    Args:
        filepath: The file path to read the annotations from.

    Returns:
        The annotations list.

    Raises:
        ValueError: If the file does not contain the expected format.
    """
    json_dict = read_dict(filepath)
    _check_tag(json_dict, expected_format="annotations")
    return json_dict["annotations"]


def write_graphical_annotations(
    filepath: str, annotations: list[GraphicalAnnotation]
) -> None:
    """Write canonical graphical annotations to a ``.dlabann`` JSON file.

    Args:
        filepath: The file path to write the annotations to.
        annotations: Canonical graphical annotations to serialize.

    Raises:
        TypeError: If annotations is not a list of GraphicalAnnotation objects.
    """
    if not isinstance(annotations, list) or not all(
        isinstance(item, GraphicalAnnotation) for item in annotations
    ):
        raise TypeError("annotations must be a list of GraphicalAnnotation objects")
    write_annotations(filepath, [annotation_to_dict(item) for item in annotations])


def read_graphical_annotations(filepath: str) -> list[GraphicalAnnotation]:
    """Read canonical graphical annotations from a ``.dlabann`` JSON file.

    Opaque entries remain available through :func:`read_annotations` and are ignored
    by this typed convenience function.

    Args:
        filepath: The file path to read the annotations from.

    Returns:
        Canonical graphical annotations in storage order.
    """
    return [
        annotation_from_dict(item)
        for item in read_annotations(filepath)
        if is_graphical_annotation_dict(item)
    ]
