# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry results
================

Geometry results are compute-friendly result containers for geometric outputs.

This module defines the `GeometryResult` class and related utilities:

- `GeometryResult`: geometric outputs (points, segments, circles, ...)
- `KindShape`: enumeration of geometric shape types
- Utility functions for geometry operations (concatenation, filtering, etc.)

Each result object is a simple data container with no behavior or methods:

- It contains the result of a 1-to-0 processing function
  (e.g. `sigima.proc.image.contour_shape()`), i.e. a computation function that takes a
  signal or image object (`SignalObj` or `ImageObj`) as input and produces a geometric
  output (`GeometryResult`).

- The result may consist of multiple rows, each corresponding to a different ROI.

.. note::

    No UI/HTML, no DataLab-specific metadata here. Adapters/formatters live in
    DataLab. These classes are JSON-friendly via `to_dict()`/`from_dict()`.

Conventions
-----------

Conventions regarding ROI and geometry are as follows:

- ROI indexing:

  - `NO_ROI = -1` sentinel is used for "full image / no ROI" rows.
  - Per-ROI rows use non-negative indices (0-based).

- Geometry coordinates (physical units):

  - `"point"` / `"marker"`: `[x, y]`
  - `"segment"`: `[x0, y0, x1, y1]`
  - `"rectangle"`: `[x0, y0, width, height]`
  - `"circle"`: `[x0, y0, radius]`
  - `"ellipse"`: `[x0, y0, a, b, theta]`   # theta in radians
  - `"polygon"`: `[x0, y0, x1, y1, ..., xn, yn]`  (rows may be NaN-padded)
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from sigima.objects.base import HTML_TABLE_CSS
from sigima.objects.scalar.common import (
    NO_ROI,
    DataFrameManager,
    DisplayPreferencesManager,
    ResultHtmlGenerator,
)

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj


class KindShape(str, enum.Enum):
    """Geometric shape types."""

    POINT = "point"
    SEGMENT = "segment"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    MARKER = "marker"

    @classmethod
    def values(cls) -> list[str]:
        """Return all shape type values."""
        return [e.value for e in cls]


@dataclasses.dataclass(frozen=True)
class GeometryResult:
    """Geometric outputs, optionally per-ROI.

    Args:
        title: Human-readable title for this geometric output set.
        kind: Shape kind (`KindShape` member or its string value).
        coords: 2-D array (N, K) with coordinates per row. K depends on `kind`
         and may be NaN-padded (e.g., for polygons).
        roi_indices: Optional 1-D array (N,) mapping rows to ROI indices.
         Use NO_ROI (-1) for the "full signal/image / no ROI" row.
        func_name: Optional name of the computation function that produced this result.
        attrs: Optional algorithmic context (e.g. thresholds, method variant).

    Raises:
        ValueError: If dimensions are inconsistent or fields are invalid.

    .. important::
        **Coordinate System**: GeometryResult coordinates are stored in **physical
        units** (e.g., mm, µm), not pixel coordinates. The conversion from pixel to
        physical coordinates is performed automatically when creating GeometryResult
        objects from image measurements using
        :func:`~sigima.proc.image.base.compute_geometry_from_obj`.

        This ensures that geometric measurements are:

        * **Scale-independent**: Results remain valid when images are resized
        * **Physically meaningful**: Measurements have real-world significance
        * **Consistent**: Same geometric features yield same results across different
          images

    .. note::

        Coordinate conventions are as follows:

        - `KindShape.POINT`: `[x, y]`
        - `KindShape.SEGMENT`: `[x0, y0, x1, y1]`
        - `KindShape.RECTANGLE`: `[x0, y0, width, height]`
        - `KindShape.CIRCLE`: `[x0, y0, radius]`
        - `KindShape.ELLIPSE`: `[x0, y0, a, b, theta]`   # theta in radians
        - `KindShape.POLYGON`: `[x0, y0, x1, y1, ..., xn, yn]`  (rows may be NaN-padded)

        All coordinate values and dimensions (width, height, radius, semi-axes) are
        expressed in the image's physical units as defined by the image calibration.

    See Also:
        :func:`~sigima.proc.image.base.compute_geometry_from_obj`: Function that
        creates GeometryResult objects with automatic coordinate conversion from
        pixel to physical units.
    """

    title: str
    kind: KindShape
    coords: np.ndarray
    roi_indices: np.ndarray | None = None
    func_name: str | None = None
    attrs: dict[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        # --- kind validation/coercion (smooth migration) ---
        k = object.__getattribute__(self, "kind")
        if isinstance(k, str):
            try:
                k = KindShape(k)  # coerce "ellipse" -> KindShape.ELLIPSE
            except ValueError as exc:
                raise ValueError(f"Unsupported geometry kind: {k!r}") from exc
            object.__setattr__(self, "kind", k)
        elif not isinstance(k, KindShape):
            raise ValueError("kind must be a KindShape or its string value")
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("title must be a non-empty string")
        if not isinstance(self.coords, np.ndarray) or self.coords.ndim != 2:
            raise ValueError("coords must be a 2-D numpy array")
        if k == KindShape.POINT and self.coords.shape[1] != 2:
            raise ValueError("coords for 'point' must be (N,2)")
        if k == KindShape.MARKER and self.coords.shape[1] != 2:
            raise ValueError("coords for 'marker' must be (N,2)")
        if k == KindShape.SEGMENT and self.coords.shape[1] != 4:
            raise ValueError("coords for 'segment' must be (N,4)")
        if k == KindShape.CIRCLE and self.coords.shape[1] != 3:
            raise ValueError("coords for 'circle' must be (N,3)")
        if k == KindShape.ELLIPSE and self.coords.shape[1] != 5:
            raise ValueError("coords for 'ellipse' must be (N,5)")
        if k == KindShape.RECTANGLE and self.coords.shape[1] != 4:
            raise ValueError("coords for 'rectangle' must be (N,4)")
        if k == KindShape.POLYGON and self.coords.shape[1] % 2 != 0:
            raise ValueError("coords for 'polygon' must be (N,2M) for M vertices")
        if self.roi_indices is not None:
            if (
                not isinstance(self.roi_indices, np.ndarray)
                or self.roi_indices.ndim != 1
            ):
                raise ValueError("roi_indices must be a 1-D numpy array if provided")
            if len(self.roi_indices) != len(self.coords):
                raise ValueError("roi_indices length must match number of coord rows")

    @property
    def name(self) -> str:
        """Get the unique identifier name for this geometry result.

        Returns:
            The string value of the kind attribute, which serves as a unique
             name identifier for this geometry result type.
        """
        return self.kind.value

    @property
    def value(self) -> float | tuple[float, float]:
        """Get the value from a single-row POINT, MARKER, or SEGMENT geometry result.

        This property provides convenient access to computed values:

        - For POINT: returns (x, y) coordinates as a tuple
        - For MARKER: returns (x, y) coordinates as a tuple
        - For SEGMENT: returns the length of the segment as a float

        Returns:
            For POINT/MARKER: tuple of (x, y) coordinates
            For SEGMENT: float length of the segment

        Raises:
            ValueError: If the result has multiple rows or is not a POINT, MARKER,
                       or SEGMENT kind

        Examples:
            >>> # Get coordinates from x_at_y result (MARKER)
            >>> result = proxy.compute_x_at_y(p)
            >>> x, y = result.value  # Get both coordinates
            >>>
            >>> # Get coordinates from peak detection (POINT)
            >>> result = proxy.compute_peak_detection(p)
            >>> x, y = result.value  # Get peak coordinates
            >>>
            >>> # Get segment length (SEGMENT)
            >>> result = proxy.compute_fwhm(p)
            >>> length = result.value  # Get FWHM length
        """
        if self.kind not in (KindShape.POINT, KindShape.MARKER, KindShape.SEGMENT):
            raise ValueError(
                f"value property only valid for POINT, MARKER, or SEGMENT kinds, "
                f"got {self.kind}"
            )
        if len(self.coords) != 1:
            raise ValueError(
                f"value property only valid for single-row results, "
                f"got {len(self.coords)} rows"
            )

        if self.kind == KindShape.SEGMENT:
            return float(self.segments_lengths()[0])

        # POINT or MARKER: return (x, y) tuple
        x, y = self.coords[0]
        return (float(x), float(y))

    # -------- Factory methods --------

    @classmethod
    def from_coords(
        cls,
        title: str,
        kind: KindShape,
        coords: np.ndarray,
        roi_indices: np.ndarray | None = None,
        *,
        func_name: str | None = None,
        attrs: dict[str, object] | None = None,
    ) -> GeometryResult:
        """Create a GeometryResult from raw data.

        Args:
            title: Human-readable title for this geometric output.
            kind: Shape kind (e.g. "point", "segment").
            coords: 2-D array (N, K) with coordinates per row.
            roi_indices: Optional 1-D array (N,) mapping rows to ROI indices.
            func_name: Optional name of the computation function.
            attrs: Optional algorithmic context (e.g. thresholds, method variant).

        Returns:
            A GeometryResult instance.
        """
        return cls(
            title=title,
            kind=kind,
            coords=np.asarray(coords, float),
            roi_indices=None if roi_indices is None else np.asarray(roi_indices, int),
            func_name=func_name,
            attrs={} if attrs is None else dict(attrs),
        )

    # -------- JSON-friendly (de)serialization (no DataLab metadata coupling) -----

    def to_dict(self) -> dict:
        """Convert the GeometryResult to a dictionary."""
        return {
            "schema": 1,
            "title": self.title,
            "kind": self.kind.value,
            "coords": self.coords.tolist(),
            "roi_indices": None
            if self.roi_indices is None
            else self.roi_indices.tolist(),
            "func_name": self.func_name,
            "attrs": dict(self.attrs) if self.attrs else {},
        }

    @staticmethod
    def from_dict(d: dict) -> GeometryResult:
        """Convert a dictionary to a GeometryResult."""
        return GeometryResult(
            title=d["title"],
            kind=KindShape(d["kind"]),
            coords=np.asarray(d["coords"], dtype=float),
            roi_indices=None
            if d.get("roi_indices") is None
            else np.asarray(d["roi_indices"], dtype=int),
            func_name=d.get("func_name"),
            attrs=dict(d.get("attrs", {})),
        )

    # -------- Pandas DataFrame interop --------

    @property
    def headers(self) -> list[str]:
        """Get column headers for the coordinates.

        Returns:
            List of column headers
        """
        # Create headers based on the shape type
        kind = self.kind.value

        # Define headers based on shape type
        headers_map = {
            "point": ["x", "y"],
            "marker": ["x", "y"],
            "segment": ["x0", "y0", "x1", "y1"],
            "rectangle": ["x", "y", "width", "height"],
            "circle": ["x", "y", "r"],
            "ellipse": ["x", "y", "a", "b", "θ"],
        }

        if kind in headers_map:
            return headers_map[kind]

        num_coords = self.coords.shape[1]

        if kind == "polygon":
            headers = []
            for i in range(0, num_coords, 2):
                headers.extend([f"x{i // 2}", f"y{i // 2}"])
            return headers[:num_coords]

        # Generic headers for unknown shapes
        return [f"coord_{i}" for i in range(num_coords)]

    def to_dataframe(self, visible_only: bool = False):
        """Convert the result to a pandas DataFrame.

        Args:
            visible_only: If True, include only visible headers based on display
             preferences. Default is False.

        Returns:
            DataFrame with an optional 'roi_index' column.
             If visible_only is True, only columns with visible headers are included.
        """
        df = pd.DataFrame(self.coords, columns=self.headers)
        visible_headers = self.get_visible_headers()

        # For segments, add a length column
        if self.kind == KindShape.SEGMENT:
            lengths = self.segments_lengths()
            # Name the length column "Δx" if y0 == y1 for all rows,
            # "Δy" if x0 == x1 for all rows, else "length"
            if np.allclose(self.coords[:, 1], self.coords[:, 3]):
                length_name = "Δx"
            elif np.allclose(self.coords[:, 0], self.coords[:, 2]):
                length_name = "Δy"
            else:
                length_name = "length"
            df[length_name] = lengths
            visible_headers = [length_name]  # always show length for segments

        if self.roi_indices is not None:
            df.insert(0, "roi_index", self.roi_indices)

        # Filter to visible columns if requested
        if visible_only:
            df = DataFrameManager.apply_visible_only_filter(df, visible_headers)

        return df

    def get_display_preferences(self) -> dict[str, bool]:
        """Get display preferences for coordinate headers.

        Returns:
            Dictionary mapping header names to visibility (True=visible, False=hidden).
            By default, all coordinates are visible unless specified in attrs.
        """
        return DisplayPreferencesManager.get_display_preferences(
            self, self.headers, "hidden_coords"
        )

    def set_display_preferences(self, preferences: dict[str, bool]) -> None:
        """Set display preferences for coordinate headers.

        Args:
            preferences: Dictionary mapping header names to visibility
                        (True=visible, False=hidden)
        """
        DisplayPreferencesManager.set_display_preferences(
            self, preferences, self.headers, "hidden_coords"
        )

    def get_visible_headers(self) -> list[str]:
        """Get list of currently visible headers based on display preferences.

        Returns:
            List of header names that should be displayed
        """
        return DisplayPreferencesManager.get_visible_headers(
            self, self.headers, "hidden_coords"
        )

    # -------- User-oriented methods --------

    def __len__(self) -> int:
        """Return the number of coordinates (rows) in the result."""
        return self.coords.shape[0]

    def rows(self, roi: int | None = None) -> np.ndarray:
        """Return coords for all rows (this ROI or full-image row).

        Args:
            roi: Optional ROI index to filter rows.

        Returns:
            2-D array of shape (M, K) with coordinates for the selected rows.
        """
        if self.roi_indices is None:
            return self.coords
        target = NO_ROI if roi is None else int(roi)
        return self.coords[self.roi_indices == target]

    def bounding_boxes(self) -> np.ndarray:
        """Return bounding boxes for each shape in the result.

        Returns:
            2-D array of shape (N, 4) with bounding boxes [x_min, y_min, x_max, y_max]
            for each shape.
        """
        bboxes = []
        for row in self.coords:
            if self.kind in (KindShape.POINT, KindShape.MARKER):
                x, y = row
                bbox = [x, y, x, y]
            elif self.kind == KindShape.SEGMENT:
                x0, y0, x1, y1 = row
                bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            elif self.kind == KindShape.RECTANGLE:
                x, y, width, height = row
                bbox = [x, y, x + width, y + height]
            elif self.kind == KindShape.CIRCLE:
                x, y, r = row
                bbox = [x - r, y - r, x + r, y + r]
            elif self.kind == KindShape.ELLIPSE:
                x, y, a, b, _ = row
                bbox = [x - a, y - b, x + a, y + b]
            elif self.kind == KindShape.POLYGON:
                xs = row[0::2]
                ys = row[1::2]
                xs = xs[~np.isnan(xs)]
                ys = ys[~np.isnan(ys)]
                bbox = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
            else:
                raise ValueError(f"Unsupported kind for bounding box: {self.kind}")
            bboxes.append(bbox)
        return np.array(bboxes)

    def centers(self) -> np.ndarray:
        """Return center points for each shape in the result.

        Returns:
            2-D array of shape (N, 2) with center coordinates [x_center, y_center]
            for each shape.
        """
        # To compute the centers, the most elegant and compact solution is to use the
        # bounding boxes.
        bboxes = self.bounding_boxes()
        x_centers = (bboxes[:, 0] + bboxes[:, 2]) / 2
        y_centers = (bboxes[:, 1] + bboxes[:, 3]) / 2
        return np.column_stack((x_centers, y_centers))

    # Optional convenience for common kinds:
    def segments_lengths(self) -> np.ndarray:
        """For kind='segment': return vector of segment lengths."""
        if self.kind != KindShape.SEGMENT:
            raise ValueError("segments_lengths requires kind='segment'")
        dx = self.coords[:, 2] - self.coords[:, 0]
        dy = self.coords[:, 3] - self.coords[:, 1]
        return np.sqrt(dx * dx + dy * dy)

    def circles_radii(self) -> np.ndarray:
        """For kind='circle': return radii."""
        if self.kind != KindShape.CIRCLE:
            raise ValueError("circles_radii requires kind='circle'")
        return self.coords[:, 2]

    def ellipse_axes_angles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For kind='ellipse': return (a, b, theta)."""
        if self.kind != KindShape.ELLIPSE:
            raise ValueError("ellipse_axes_angles requires kind='ellipse'")
        return self.coords[:, 2], self.coords[:, 3], self.coords[:, 4]

    def to_html(
        self,
        obj: SignalObj | ImageObj | None = None,
        visible_only: bool = True,
        transpose_single_row: bool = True,
        **kwargs,
    ) -> str:
        """Convert the result to HTML format.

        Args:
            obj: Optional SignalObj or ImageObj for ROI title extraction
            visible_only: If True, include only visible headers based on display
             preferences.
            transpose_single_row: If True, transpose when there's only one row
            **kwargs: Additional arguments passed to DataFrame.to_html()

        Returns:
            HTML representation of the result
        """
        return ResultHtmlGenerator.generate_html(
            self, obj, visible_only, transpose_single_row, **kwargs
        )

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display.

        This method is automatically called by Jupyter when displaying the object
        as a cell output, providing a rich HTML rendering of the geometry result.

        Returns:
            HTML representation of the geometry result with styling.
        """
        return HTML_TABLE_CSS + self.to_html()


# ===========================
# Geometry utility functions
# ===========================


def concat_geometries(
    title: str,
    geometries: Iterable[GeometryResult],
    *,
    kind: KindShape | None = None,
) -> GeometryResult:
    """Concatenate multiple GeometryResult objects of the same kind.

    Args:
        title: Title for the concatenated result.
        geometries: Iterable of GeometryResult objects to concatenate.
        kind: Optional kind label for the concatenated result.

    Returns:
        GeometryResult with concatenated data and updated metadata.
    """
    geometries = list(geometries)
    if not geometries:
        raise ValueError("Cannot concatenate empty sequence of GeometryResult objects")
    k = kind if kind is not None else geometries[0].kind
    if any(geom.kind != k for geom in geometries):
        raise ValueError(
            "All GeometryResult objects must share the same kind to concatenate"
        )
    fn = geometries[0].func_name
    if fn is None:
        raise ValueError(
            "All GeometryResult objects must have a func_name to concatenate"
        )
    if any(geom.func_name != fn for geom in geometries):
        raise ValueError(
            "All GeometryResult objects must share the same func_name to concatenate"
        )
    max_k = max(geom.coords.shape[1] for geom in geometries) if geometries else 0
    # right-pad with NaNs to match width
    padded = []
    for geometry in geometries:
        c = geometry.coords
        if c.shape[1] < max_k:
            pad = np.full((c.shape[0], max_k - c.shape[1]), np.nan, dtype=float)
            c = np.hstack([c, pad])
        padded.append(c)
    coords = np.vstack(padded) if padded else np.zeros((0, max_k))
    if any(it.roi_indices is not None for it in geometries):
        parts = [
            (
                it.roi_indices
                if it.roi_indices is not None
                else np.full((len(it.coords),), NO_ROI, int)
            )
            for it in geometries
        ]
        roi = np.concatenate(parts) if len(parts) else None
    else:
        roi = None
    return GeometryResult(
        title=title, kind=k, coords=coords, roi_indices=roi, func_name=fn
    )


def filter_geometry_by_roi(res: GeometryResult, roi: int | None) -> GeometryResult:
    """Filter shapes by ROI index. If roi is None, keeps NO_ROI rows.

    Args:
        res: The GeometryResult to filter.
        roi: The ROI index to filter by, or None to keep all.

    Returns:
        A filtered GeometryResult.
    """
    if res.roi_indices is None:
        keep_all = roi in (None, NO_ROI)
        coords = res.coords if keep_all else np.zeros((0, res.coords.shape[1]))
        indices = None if keep_all else np.zeros((0,), int)
        return GeometryResult(
            title=res.title,
            kind=res.kind,
            coords=coords,
            roi_indices=indices,
            attrs=dict(res.attrs),
        )
    target = NO_ROI if roi is None else int(roi)
    mask = res.roi_indices == target
    return GeometryResult(
        title=res.title,
        kind=res.kind,
        coords=res.coords[mask],
        roi_indices=res.roi_indices[mask],
        attrs=dict(res.attrs),
    )
