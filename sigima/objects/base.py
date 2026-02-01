# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Base model classes for signals and images.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import json
import re
import sys
from collections.abc import Generator
from copy import deepcopy
from typing import Any, Generic, Iterator, Type, TypeVar

import guidata.dataset as gds
import numpy as np
from numpy import ma

from sigima.config import _

if sys.version_info >= (3, 11):
    # Use Self from typing module in Python 3.11+
    from typing import Self
else:
    # Use Self from typing_extensions module in Python < 3.11
    from typing_extensions import Self

# CSS styling for HTML tables in Jupyter notebooks
HTML_TABLE_CSS = """
<style>
    .sigima-html-table {
        border-collapse: collapse;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     'Helvetica Neue', Arial, sans-serif;
        font-size: 13px;
        margin: 10px 0;
    }
    .sigima-html-table th {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
    }
    .sigima-html-table td {
        border: 1px solid #dee2e6;
        padding: 8px 12px;
        text-align: right;
    }
    .sigima-html-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    .sigima-html-table tr:hover {
        background-color: #e9ecef;
    }
</style>
"""

ROI_KEY = "_roi_"


def deepcopy_metadata(
    metadata: dict[str, Any],
    special_keys: set[str] | None = None,
    all_metadata: bool = False,
) -> dict[str, Any]:
    """Deepcopy metadata, except keys starting with "_" (private keys)
    with the exception of "_roi_" and "_ann_" keys.

    Args:
        metadata: Metadata dictionary to deepcopy.
        special_keys: Set of keys that should not be removed even if they
         start with "_".
        all_metadata: if True, copy all metadata, including private keys

    Returns:
        A new dictionary with deepcopied metadata, excluding private keys
        except those in `special_keys`.
    """
    if special_keys is None:
        special_keys = set([ROI_KEY])
    mdcopy = {}
    for key, value in metadata.items():
        if not key.startswith("_") or key in special_keys or all_metadata:
            mdcopy[key] = deepcopy(value)
    return mdcopy


class BaseProcParam(gds.DataSet):
    """Base class for processing parameters."""

    def apply_integer_range(self, vmin, vmax):  # pylint: disable=unused-argument
        """Do something in case of integer min-max range."""

    def apply_float_range(self, vmin, vmax):  # pylint: disable=unused-argument
        """Do something in case of float min-max range."""

    def set_from_datatype(self, dtype):
        """Set min/max range from NumPy datatype."""
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            self.apply_integer_range(info.min, info.max)
        else:
            info = np.finfo(dtype)
            self.apply_float_range(info.min, info.max)


class BaseRandomParam(BaseProcParam):
    """Random signal/image parameters."""

    seed = gds.IntItem(_("Seed"), default=1)


class UniformDistributionParam(BaseRandomParam):
    """Uniform-distribution signal/image parameters."""

    def apply_integer_range(self, vmin, vmax):
        """Do something in case of integer min-max range."""
        self.vmin, self.vmax = float(vmin), float(vmax)

    vmin = gds.FloatItem(
        "V<sub>min</sub>", default=-0.5, help=_("Uniform distribution lower bound")
    )
    vmax = gds.FloatItem(
        "V<sub>max</sub>", default=0.5, help=_("Uniform distribution higher bound")
    ).set_pos(col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"UniformRandom(vmin={self.vmin:g},vmax={self.vmax:g},seed={self.seed})"


class NormalDistributionParam(BaseRandomParam):
    """Normal-distribution signal/image parameters."""

    DEFAULT_RELATIVE_MU = 0.1
    DEFAULT_RELATIVE_SIGMA = 0.02

    def apply_integer_range(self, vmin, vmax):
        """Do something in case of integer min-max range."""
        delta = vmax - vmin
        self.mu = float(vmin + self.DEFAULT_RELATIVE_MU * delta)
        self.sigma = float(self.DEFAULT_RELATIVE_SIGMA * delta)

    mu = gds.FloatItem(
        "μ", default=DEFAULT_RELATIVE_MU, help=_("Normal distribution mean")
    )
    sigma = gds.FloatItem(
        "σ",
        default=DEFAULT_RELATIVE_SIGMA,
        min=0.0,
        help=_("Normal distribution standard deviation"),
    ).set_pos(col=1)

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"NormalRandom(μ={self.mu:g},σ={self.sigma:g},seed={self.seed})"


class PoissonDistributionParam(BaseRandomParam):
    """Base Poisson-distribution signal/image parameters."""

    DEFAULT_RELATIVE_LAMBDA = 0.1

    def apply_integer_range(self, vmin, vmax):
        """Adjust default λ based on integer min-max range."""
        positive_span = max(0.0, float(vmax) - max(0.0, float(vmin)))
        self.lam = float(max(self.DEFAULT_RELATIVE_LAMBDA * positive_span, 1.0))

    lam = gds.FloatItem(
        "λ",
        default=DEFAULT_RELATIVE_LAMBDA,
        min=0.0,
        help=_("Poisson distribution mean"),
    )

    def generate_title(self) -> str:
        """Generate a title based on current parameters."""
        return f"PoissonRandom(λ={self.lam:g},seed={self.seed})"


TypeObj = TypeVar("TypeObj", bound="BaseObj")
TypeROIParam = TypeVar("TypeROIParam", bound="BaseROIParam")
TypeSingleROI = TypeVar("TypeSingleROI", bound="BaseSingleROI")
TypeROI = TypeVar("TypeROI", bound="BaseROI")


class BaseObjMeta(abc.ABCMeta, gds.DataSetMeta):
    """Mixed metaclass to avoid conflicts"""


class NoDefaultOption:
    """Marker class for metadata option without default value"""


class BaseObj(Generic[TypeROI], metaclass=BaseObjMeta):
    """Object (signal/image) interface"""

    #: Class attribute that defines a string prefix used to uniquely identify instances
    #: of this class in metadata serialization. Each subclass should override this with
    #: a unique identifier (e.g., "s" for signals, "i" for images).
    #: This prefix is used as part of the key for storing and retrieving object-specific
    #: metadata, supporting type-based serialization and deserialization.
    PREFIX = ""  # This is overriden in children classes

    # This is overriden in children classes with a gds.DictItem instance:
    metadata: dict[str, Any] = {}
    annotations: str = ""

    #: Class attribute that defines a tuple of valid NumPy data types supported by this
    #: class. This is used to validate the data type of the object when it is set or
    #: modified and to ensure that the object can handle the data correctly.
    #: Subclasses should override this with a specific set of valid data types.
    VALID_DTYPES = (np.float64,)  # To be overriden in children classes

    def __init__(self):
        self.__roi_changed: bool | None = None
        self._maskdata_cache: np.ndarray | None = None
        self.__metadata_options_defaults: dict[str, Any] = {}
        self.__roi_cache: TypeROI | None = None

    @staticmethod
    @abc.abstractmethod
    def get_roi_class() -> Type[TypeROI]:
        """Return ROI class"""

    @property
    @abc.abstractmethod
    def data(self) -> np.ndarray | None:
        """Data"""

    @classmethod
    def get_valid_dtypenames(cls) -> list[str]:
        """Get valid data type names

        Returns:
            Valid data type names supported by this class
        """
        return [
            dtname
            for dtname in np.sctypeDict
            if isinstance(dtname, str)
            and dtname in (dtype.__name__ for dtype in cls.VALID_DTYPES)
        ]

    def check_data(self):
        """Check if data is valid, raise an exception if that's not the case

        Raises:
            TypeError: if data type is not supported
        """
        if self.data is not None:
            if self.data.dtype not in self.VALID_DTYPES:
                raise TypeError(f"Unsupported data type: {self.data.dtype}")

    def iterate_roi_indices(self) -> Generator[int | None, None, None]:
        """Iterate over object ROI indices (if there is no ROI, yield None)"""
        if self.roi is None:
            yield None
        else:
            yield from range(len(self.roi))

    @abc.abstractmethod
    def get_data(self, roi_index: int | None = None) -> np.ndarray:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).

        Args:
            roi_index: ROI index

        Returns:
            Data
        """

    @abc.abstractmethod
    def copy(
        self,
        title: str | None = None,
        dtype: np.dtype | None = None,
        all_metadata: bool = False,
    ) -> Self:
        """Copy object.

        Args:
            title: title
            dtype: data type
            all_metadata: if True, copy all metadata, otherwise only basic metadata

        Returns:
            Copied object
        """

    @abc.abstractmethod
    def set_data_type(self, dtype):
        """Change data type.

        Args:
            dtype: data type
        """

    @abc.abstractmethod
    def physical_to_indices(self, coords: list[float]) -> list[int]:
        """Convert coordinates from physical (real world) to indices

        Args:
            coords: coordinates

        Returns:
            Indices
        """

    @abc.abstractmethod
    def indices_to_physical(self, indices: list[int]) -> list[float]:
        """Convert coordinates from indices to physical (real world)

        Args:
            indices: indices

        Returns:
            Coordinates
        """

    def __roi_has_changed(self) -> bool:
        """Return True if ROI has changed since last call to this method.

        The first call to this method will return True if ROI has not yet been set,
        or if ROI has been set and has changed since the last call to this method.
        The next call to this method will always return False if ROI has not changed
        in the meantime.

        Returns:
            True if ROI has changed
        """
        if self.__roi_changed is None:
            self.__roi_changed = True
        returned_value = self.__roi_changed
        self.__roi_changed = False
        return returned_value

    @property
    def roi(self) -> TypeROI | None:
        """Return object regions of interest object.

        Returns:
            Regions of interest object
        """
        # If we have a cached ROI, return it
        if self.__roi_cache is not None:
            return self.__roi_cache

        # Otherwise, try to load from metadata
        roidata = self.metadata.get(ROI_KEY)
        if roidata is None:
            return None
        if not isinstance(roidata, dict):
            # Old or unsupported format: remove it
            self.metadata.pop(ROI_KEY)
            return None

        # Create ROI from metadata and cache it
        self.__roi_cache = self.get_roi_class().from_dict(roidata)
        return self.__roi_cache

    @roi.setter
    def roi(self, roi: TypeROI | None) -> None:
        """Set object regions of interest.

        Args:
            roi: regions of interest object
        """
        # Cache the ROI object
        self.__roi_cache = roi

        # Update metadata
        if roi is None:
            if ROI_KEY in self.metadata:
                self.metadata.pop(ROI_KEY)
        else:
            self.metadata[ROI_KEY] = roi.to_dict()
        self.__roi_changed = True

    @property
    def maskdata(self) -> np.ndarray | None:
        """Return masked data (areas outside defined regions of interest)

        Returns:
            Masked data
        """
        roi_changed = self.__roi_has_changed()
        if self.roi is None:
            if roi_changed:
                self._maskdata_cache = None
        elif roi_changed or self._maskdata_cache is None:
            self._maskdata_cache = self.roi.to_mask(self)
        return self._maskdata_cache

    def get_masked_view(self) -> ma.MaskedArray:
        """Return masked view for data

        Returns:
            Masked view
        """
        assert isinstance(self.data, np.ndarray)
        view = self.data.view(ma.MaskedArray)
        if self.maskdata is None:
            view.mask = np.isnan(self.data)
        else:
            view.mask = self.maskdata | np.isnan(self.data)
        return view

    def invalidate_maskdata_cache(self) -> None:
        """Invalidate mask data cache: force to rebuild it"""
        self._maskdata_cache = None

    def invalidate_roi_cache(self) -> None:
        """Invalidate ROI cache: force to reload it from metadata"""
        self.__roi_cache = None
        # Also invalidate mask data cache since ROI data might have changed
        self.invalidate_maskdata_cache()

    def sync_roi_to_metadata(self) -> None:
        """Synchronize the current ROI cache to metadata.

        This should be called after modifying the ROI object directly
        to ensure the changes are persisted in metadata.
        """
        if self.__roi_cache is not None:
            self.metadata[ROI_KEY] = self.__roi_cache.to_dict()
            self.__roi_changed = True
            # Also invalidate mask data cache since ROI has changed
            self.invalidate_maskdata_cache()

    def mark_roi_as_changed(self) -> None:
        """Mark the ROI as changed and invalidate dependent caches.

        This should be called after modifying the ROI object directly
        to ensure all dependent data (like mask cache) is properly invalidated.
        """
        self.__roi_changed = True
        self.invalidate_maskdata_cache()
        # Optionally sync to metadata immediately
        self.sync_roi_to_metadata()

    def update_metadata_from(self, other_metadata: dict[str, Any]) -> None:
        """Update metadata from another object's metadata (merge result shapes and
        annotations, and update the rest of the metadata).

        Args:
            other_metadata: other object metadata
        """
        self.metadata.update(other_metadata)
        # Invalidate ROI cache since metadata might have changed ROI data
        self.invalidate_roi_cache()

    # Method to set the default values of metadata options:
    def set_metadata_options_defaults(
        self, defaults: dict[str, Any], overwrite: bool = False
    ) -> None:
        """Set default values for metadata options

        A metadata option is a metadata entry starting with a double underscore.
        It is a way to store application-specific options in object metadata.

        .. note::

            This will not overwrite existing metadata options
            (unless `overwrite` is True).
            It will only set the default values for options that are not already set.*
            Use `reset_metadata_to_defaults` method to reset all metadata options
            to their default values.

        Args:
            defaults: dictionary of default values for metadata options
            overwrite: whether to overwrite existing metadata options (default: False)
        """
        self.__metadata_options_defaults.update(defaults)
        for key, value in defaults.items():
            self.set_metadata_option(key, value, overwrite)

    def get_metadata_options_defaults(self) -> dict[str, Any]:
        """Return default values for metadata options

        A metadata option is a metadata entry starting with a double underscore.
        It is a way to store application-specific options in object metadata.

        Returns:
            Dictionary of default values for metadata options
        """
        return self.__metadata_options_defaults

    def get_metadata_option(self, name: str, default: Any = NoDefaultOption) -> Any:
        """Return metadata option value

        A metadata option is a metadata entry starting with a double underscore.
        It is a way to store application-specific options in object metadata.

        Args:
            name: option name
            default: default value if option is not set (optional)

        Returns:
            Option value

        Raises:
            ValueError: if option name is invalid
        """
        if (
            default is not NoDefaultOption
            and name not in self.__metadata_options_defaults
        ):
            # If default is provided, store it in defaults
            # and set it as the option value
            self.__metadata_options_defaults[name] = default
            self.set_metadata_option(name, default, overwrite=False)
        try:
            value = self.metadata[f"__{name}"]
        except KeyError as exc:
            defaults = self.get_metadata_options_defaults()
            if name in defaults:
                value = defaults[name]
            else:
                raise ValueError(
                    f"Invalid metadata option name `{name}` "
                    f"(valid names: {', '.join(defaults.keys())})"
                ) from exc
        return value

    def set_metadata_option(
        self, name: str, value: Any, overwrite: bool = True
    ) -> None:
        """Set metadata option value

        A metadata option is a metadata entry starting with a double underscore.
        It is a way to store application-specific options in object metadata.

        Args:
            name: option name
            value: option value
            overwrite: whether to overwrite existing metadata options (default: True)

        Raises:
            ValueError: if option name is invalid
        """
        if overwrite or f"__{name}" not in self.metadata:
            self.metadata[f"__{name}"] = value

    def get_metadata_options(self) -> dict[str, Any]:
        """Return metadata options
        A metadata option is a metadata entry starting with a double underscore.

        Returns:
            Dictionary of metadata options (name: value)
        """
        options = {}
        for name, value in self.metadata.items():
            if name.startswith("__"):
                options[name[2:]] = value
        return options

    def reset_metadata_to_defaults(self) -> None:
        """Reset metadata to default values"""
        self.metadata = {}
        self.invalidate_roi_cache()
        defaults = self.get_metadata_options_defaults()
        for name, value in defaults.items():
            self.set_metadata_option(name, value)

    def save_attr_to_metadata(self, attrname: str, new_value: Any) -> None:
        """Save attribute to metadata

        Args:
            attrname: attribute name
            new_value: new value
        """
        value = getattr(self, attrname)
        if value:
            self.metadata[f"orig_{attrname}"] = value
        setattr(self, attrname, new_value)

    def restore_attr_from_metadata(self, attrname: str, default: Any) -> None:
        """Restore attribute from metadata

        Args:
            attrname: attribute name
            default: default value
        """
        value = self.metadata.pop(f"orig_{attrname}", default)
        setattr(self, attrname, value)

    # ------Annotation management methods

    def get_annotations(self) -> list[dict[str, Any]]:
        """Get annotations as a list of dictionaries.

        Returns:
            List of annotation dictionaries. Each dict contains application-specific
            annotation data. Returns empty list if no annotations exist.

        Notes:
            The annotation format is defined by the application layer. Sigima only
            provides storage and basic validation (valid JSON structure).

        Example:
            >>> obj.set_annotations([
            ...     {"type": "label", "x": 10, "y": 20, "text": "Peak"},
            ...     {"type": "rectangle", "x0": 0, "y0": 0, "x1": 100, "y1": 100}
            ... ])
            >>> annotations = obj.get_annotations()
            >>> len(annotations)
            2
        """
        if not self.annotations:
            return []
        try:
            data = json.loads(self.annotations)
            if isinstance(data, dict) and "annotations" in data:
                return data["annotations"]
            return []
        except (json.JSONDecodeError, TypeError):
            # Invalid JSON - return empty list
            return []

    def set_annotations(self, annotations: list[dict[str, Any]]) -> None:
        """Set annotations from a list of dictionaries.

        Args:
            annotations: List of annotation dictionaries

        Raises:
            TypeError: If annotations is not a list
            ValueError: If annotation items are not JSON-serializable

        Notes:
            Each annotation dictionary should be JSON-serializable.
            The internal storage format includes a version field for future migration.

        Example:
            >>> obj.set_annotations([{"type": "label", "text": "Test"}])
        """
        if not isinstance(annotations, list):
            raise TypeError(f"Annotations must be a list, got {type(annotations)}")

        # Validate JSON serializability
        try:
            # Store with version for future-proofing
            data = {"version": "1.0", "annotations": annotations}
            self.annotations = json.dumps(data, indent=2)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Annotations must be JSON-serializable: {exc}") from exc

    def add_annotation(self, annotation: dict[str, Any]) -> None:
        """Add a single annotation.

        Args:
            annotation: Annotation dictionary to add

        Example:
            >>> obj.add_annotation({"type": "circle", "x": 50, "y": 50, "r": 10})
        """
        current = self.get_annotations()
        current.append(annotation)
        self.set_annotations(current)

    def clear_annotations(self) -> None:
        """Remove all annotations.

        Example:
            >>> obj.clear_annotations()
            >>> obj.get_annotations()
            []
        """
        self.annotations = ""

    def has_annotations(self) -> bool:
        """Check if object has any annotations.

        Returns:
            True if annotations exist, False otherwise

        Example:
            >>> obj.has_annotations()
            False
            >>> obj.add_annotation({"type": "label"})
            >>> obj.has_annotations()
            True
        """
        return bool(self.get_annotations())


class BaseROIParamMeta(abc.ABCMeta, gds.DataSetMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseROIParam(
    gds.DataSet,
    Generic[TypeObj, TypeSingleROI],  # type: ignore
    metaclass=BaseROIParamMeta,
):
    """Base class for ROI parameters"""

    @abc.abstractmethod
    def to_single_roi(self, obj: TypeObj) -> TypeSingleROI:
        """Convert parameters to single ROI

        Args:
            obj: object (signal/image)

        Returns:
            Single ROI
        """


class BaseSingleROI(Generic[TypeObj, TypeROIParam], abc.ABC):  # type: ignore
    """Base class for single ROI

    Args:
        coords: ROI edge (physical or pixel coordinates)
        indices: if True, coords are indices (pixels) instead of physical coordinates
        title: ROI title
    """

    def __init__(
        self,
        coords: np.ndarray | list[int] | list[float],
        indices: bool,
        title: str = "ROI",
    ) -> None:
        self.coords = np.array(coords, int if indices else float)
        self.indices = indices
        self.title = title
        self.check_coords()

    def __eq__(self, other: BaseSingleROI | None) -> bool:
        """Test equality with another single ROI"""
        if other is None:
            return False
        if not isinstance(other, BaseSingleROI):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        return (
            np.array_equal(self.coords, other.coords) and self.indices == other.indices
        )

    def get_physical_coords(self, obj: TypeObj) -> list[float]:
        """Return physical coords

        Args:
            obj: object (signal/image)

        Returns:
            Physical coords
        """
        if self.indices:
            return obj.indices_to_physical(self.coords.tolist())
        return self.coords.tolist()

    def set_physical_coords(self, obj: TypeObj, coords: np.ndarray) -> None:
        """Set physical coords

        Args:
            obj: object (signal/image)
            coords: physical coords
        """
        if self.indices:
            self.coords = np.array(obj.physical_to_indices(coords.tolist()))
        else:
            self.coords = np.array(coords, float)

    def get_indices_coords(self, obj: TypeObj) -> list[int]:
        """Return indices coords

        Args:
            obj: object (signal/image)

        Returns:
            Indices coords
        """
        if self.indices:
            return self.coords.tolist()
        return obj.physical_to_indices(self.coords.tolist())

    def set_indices_coords(self, obj: TypeObj, coords: np.ndarray) -> None:
        """Set indices coords

        Args:
            obj: object (signal/image)
            coords: indices coords
        """
        if self.indices:
            self.coords = coords
        else:
            self.coords = np.array(obj.indices_to_physical(coords.tolist()))

    @abc.abstractmethod
    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """

    @abc.abstractmethod
    def to_mask(self, obj: TypeObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal or image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """

    @abc.abstractmethod
    def to_param(self, obj: TypeObj, index: int) -> TypeROIParam:
        """Convert ROI to parameters

        Args:
            obj: object (signal/image), for physical-indices coordinates conversion
            index: ROI index
        """

    def to_dict(self) -> dict:
        """Convert ROI to dictionary

        Returns:
            Dictionary
        """
        return {
            "coords": self.coords,
            "indices": self.indices,
            "title": self.title,
            "type": type(self).__name__,
        }

    @classmethod
    def from_dict(cls: Type[TypeSingleROI], dictdata: dict) -> TypeSingleROI:
        """Convert dictionary to ROI

        Args:
            dictdata: dictionary

        Returns:
            ROI
        """
        return cls(dictdata["coords"], dictdata["indices"], dictdata["title"])

    def get_coords_html_rows(self) -> list[tuple[str, str]]:
        """Return HTML table rows describing the ROI coordinates.

        Override this method in subclasses to provide more detailed
        coordinate descriptions (e.g., "Center", "Radius" for circular ROIs).

        Returns:
            List of (label, value) tuples for HTML table rows.
        """
        coord_type = "indices" if self.indices else "physical"
        coords_str = ", ".join(f"{c:.4g}" for c in self.coords)
        return [(f"Coordinates ({coord_type})", f"[{coords_str}]")]

    def get_coords_summary(self) -> str:
        """Return a short summary of the ROI coordinates for table display.

        Override this method in subclasses to provide a more meaningful summary.

        Returns:
            Short string summarizing the ROI coordinates.
        """
        coords_str = ", ".join(f"{c:.4g}" for c in self.coords)
        return f"[{coords_str}]"

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display.

        This method is automatically called by Jupyter when displaying the object
        as a cell output, providing a rich HTML rendering of the ROI.

        Returns:
            HTML representation of the ROI with coordinates.
        """
        roi_type = type(self).__name__
        html = f'<u><b style="color: #5294e2">{roi_type}:</b></u>'
        if self.title:
            html += f" <b>{self.title}</b>"
        html += '<table border="0">'
        for label, value in self.get_coords_html_rows():
            html += (
                f"<tr><td style='text-align:right; padding-right:10px;'>{label}:</td>"
                f"<td>{value}</td></tr>"
            )
        html += "</table>"
        return html


class BaseROI(Generic[TypeObj, TypeSingleROI, TypeROIParam], abc.ABC):  # type: ignore
    """Abstract base class for ROIs (Regions of Interest)

    Args:
        inverse: if True, ROI is outside the region of interest
    """

    #: Class attribute that defines a string prefix used for identifying ROI types
    #: in object metadata. This prefix is used when serializing and deserializing ROIs,
    #: allowing the system to determine the appropriate ROI class for reconstruction.
    #: Each ROI subclass should override this with a unique string identifier.
    PREFIX = ""  # This is overriden in children classes

    def __init__(self) -> None:
        self.single_rois: list[TypeSingleROI] = []

    @staticmethod
    @abc.abstractmethod
    def get_compatible_single_roi_classes() -> list[Type[BaseSingleROI]]:
        """Return compatible single ROI classes"""

    def __len__(self) -> int:
        """Return number of ROIs"""
        return len(self.single_rois)

    def __iter__(self) -> Iterator[TypeSingleROI]:
        """Iterate over single ROIs"""
        return iter(self.single_rois)

    def __eq__(self, other: BaseROI | None) -> bool:
        """Test equality with another ROI"""
        if other is None:
            return False
        if not isinstance(other, BaseROI):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        return self.single_rois == other.single_rois

    def get_single_roi(self, index: int) -> TypeSingleROI:
        """Return single ROI at index

        Args:
            index: ROI index
        """
        return self.single_rois[index]

    def set_single_roi(self, index: int, roi: TypeSingleROI) -> None:
        """Set single ROI at index

        Args:
            index: ROI index
            roi: ROI to set
        """
        self.single_rois[index] = roi

    def get_single_roi_title(self, index: int) -> str:
        """Generate title for single ROI, based on its index, using either the
        ROI title or a default generic title as fallback.

        Args:
            index: ROI index
        """
        single_roi = self.get_single_roi(index)
        title = single_roi.title or get_generic_roi_title(index)
        return title

    def is_empty(self) -> bool:
        """Return True if no ROI is defined"""
        return len(self) == 0

    @classmethod
    def create(
        cls: Type[BaseROI], single_roi: TypeSingleROI
    ) -> BaseROI[TypeObj, TypeSingleROI, TypeROIParam]:
        """Create Regions of Interest object from a single ROI.

        Args:
            single_roi: single ROI

        Returns:
            Regions of Interest object
        """
        roi = cls()
        roi.add_roi(single_roi)
        return roi

    def copy(self) -> BaseROI[TypeObj, TypeSingleROI, TypeROIParam]:
        """Return a copy of ROIs"""
        return deepcopy(self)

    def empty(self) -> None:
        """Empty ROIs"""
        self.single_rois.clear()

    def combine_with(
        self, other: BaseROI[TypeObj, TypeSingleROI, TypeROIParam]
    ) -> BaseROI[TypeObj, TypeSingleROI, TypeROIParam]:
        """Combine ROIs with another ROI object, by merging single ROIs (and ignoring
        duplicate single ROIs) and returning a new combined ROI object.

        Args:
            other: other ROI object

        Returns:
            Combined ROIs object
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot combine {type(self)} with {type(other)}")
        combined_roi = self.copy()
        for roi in other.single_rois:
            if all(s_roi != roi for s_roi in self.single_rois):
                combined_roi.single_rois.append(roi)
        return combined_roi

    def add_roi(
        self, roi: TypeSingleROI | BaseROI[TypeObj, TypeSingleROI, TypeROIParam]
    ) -> None:
        """Add ROI.

        Args:
            roi: ROI

        Raises:
            TypeError: if roi type is not supported (not a single ROI or a ROI)
            ValueError: if `inverse` values are incompatible
        """
        if isinstance(roi, BaseSingleROI):
            self.single_rois.append(roi)
        elif isinstance(roi, BaseROI):
            self.single_rois.extend(roi.single_rois)
        else:
            raise TypeError(f"Unsupported ROI type: {type(roi)}")

    @abc.abstractmethod
    def to_mask(self, obj: TypeObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal or image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """

    def to_params(self, obj: TypeObj) -> list[TypeROIParam]:
        """Convert ROIs to a list of parameters

        Args:
            obj: object (signal/image), for physical to pixel conversion

        Returns:
            ROI parameters
        """
        return [iroi.to_param(obj, index=idx) for idx, iroi in enumerate(self)]

    @classmethod
    def from_params(
        cls: Type[BaseROI],
        obj: TypeObj,
        params: list[TypeROIParam],
    ) -> BaseROI[TypeObj, TypeSingleROI, TypeROIParam]:
        """Create ROIs from parameters

        Args:
            obj: object (signal/image)
            params: ROI parameters

        Returns:
            ROIs
        """
        roi = cls()
        for param in params:
            assert isinstance(param, BaseROIParam), "Invalid ROI parameter type"
            roi.add_roi(param.to_single_roi(obj))
        return roi

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks."""
        roi_type = type(self).__name__
        count = len(self.single_rois)

        rows = []
        for idx, single_roi in enumerate(self.single_rois):
            title = single_roi.title or get_generic_roi_title(idx)
            stype = type(single_roi).__name__
            coords_summary = single_roi.get_coords_summary()
            rows.append(
                f"<tr><td>{idx}</td><td>{title}</td><td>{stype}</td>"
                f"<td>{coords_summary}</td></tr>"
            )

        table_rows = (
            "\n".join(rows) if rows else "<tr><td colspan='4'>No ROIs</td></tr>"
        )

        html = f"""
        {HTML_TABLE_CSS}
        <div class="sigima-html-container">
            <div class="sigima-html-title">{roi_type}</div>
            <table class="sigima-html-table">
                <tr><th>Index</th><th>Title</th><th>Type</th><th>Coordinates</th></tr>
                {table_rows}
            </table>
            <div class="sigima-html-footer">{count} ROI(s)</div>
        </div>
        """
        return html

    def to_dict(self) -> dict:
        """Convert ROIs to dictionary

        Returns:
            Dictionary
        """
        return {
            "single_rois": [roi.to_dict() for roi in self.single_rois],
        }

    @classmethod
    def from_dict(cls: Type[TypeROI], dictdata: dict) -> TypeROI:
        """Convert dictionary to ROIs

        Args:
            dictdata: dictionary

        Returns:
            ROIs
        """
        instance = cls()
        if not all(key in dictdata for key in ["single_rois"]):
            raise ValueError("Invalid ROI: dictionary must contain 'single_rois' key")
        instance.single_rois = []
        for single_roi in dictdata["single_rois"]:
            for single_roi_class in instance.get_compatible_single_roi_classes():
                if single_roi["type"] == single_roi_class.__name__:
                    instance.single_rois.append(single_roi_class.from_dict(single_roi))
                    break
            else:
                raise ValueError(f"Unsupported single ROI type: {single_roi['type']}")
        return instance


GENERIC_ROI_TITLE_REGEXP = r"ROI(\d+)"


def get_generic_roi_title(index: int) -> None:
    """Return a generic title for the ROI"""
    title = f"ROI{index:02d}"
    assert re.match(GENERIC_ROI_TITLE_REGEXP, title)
    return title
