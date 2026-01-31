# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Table results
=============

Table results are compute-friendly result containers for scalar table outputs.

This module defines the `TableResult` class and related utilities:

- `TableResult`: table of scalar metrics
- `TableResultBuilder`: builder for TableResult with fluent interface
- Utility functions for table operations (concatenation, filtering, etc.)

Each result object is a simple data container with no behavior or methods:

- It contains the result of a 1-to-0 processing function
  (e.g. `sigima.proc.signal.fwhm()`), i.e. a computation function that takes a signal
  or image object (`SignalObj` or `ImageObj`) as input and produces a scalar output.

- The result may consist of multiple rows, each corresponding to a different ROI.

.. note::

    No UI/HTML, no DataLab-specific metadata here. Adapters/formatters live in
    DataLab. These classes are JSON-friendly via `to_dict()`/`from_dict()`.

Conventions
-----------

Conventions regarding ROI indexing:

- `NO_ROI = -1` sentinel is used for "full image / no ROI" rows.
- Per-ROI rows use non-negative indices (0-based).
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import os
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

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


class TableKind(str, enum.Enum):
    """Types of table results."""

    STATISTICS = "statistics"
    PULSE_FEATURES = "pulse_features"
    CUSTOM = "results"

    @classmethod
    def values(cls) -> list[str]:
        """Return all table kind values."""
        return [e.value for e in cls]


@dataclasses.dataclass(frozen=True)
class TableResult:
    """Table of scalar results, optionally per-ROI.

    Args:
        title: Human-readable title for this table of results.
        kind: Type of table result (e.g., TableKind.PULSE_FEATURES,
         TableKind.STATISTICS). Default is TableKind.CUSTOM.
        headers: Column names (one per metric).
        data: 2-D list of shape (N, len(headers)) with scalar values.
        roi_indices: Optional list (N,) mapping rows to ROI indices.
         Use NO_ROI (-1) for the "full image / no ROI" row.
        func_name: Optional name of the computation function that produced this result.
        attrs: Optional algorithmic context (e.g. thresholds, method variant).

    Raises:
        ValueError: If dimensions are inconsistent or fields are invalid.

    Notes:
        - No UI/presentation concerns, no persistence schema here.
        - Use DataLab-side adapters to store results in metadata if needed.
    """

    title: str
    kind: TableKind | str = TableKind.CUSTOM
    headers: Sequence[str] = dataclasses.field(default_factory=list)
    data: list[list] = dataclasses.field(default_factory=list)
    roi_indices: list[int] | None = None
    func_name: str | None = None
    attrs: dict[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if isinstance(self.kind, str):
            try:
                object.__setattr__(self, "kind", TableKind(self.kind))
            except ValueError:
                pass  # Allow custom string values that are not in the enum
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("title must be a non-empty string")
        if not isinstance(self.headers, (list, tuple)) or not all(
            isinstance(c, str) for c in self.headers
        ):
            raise ValueError("names must be a sequence of strings")
        if not isinstance(self.data, list):
            raise ValueError("data must be a list of lists")
        if self.data and not isinstance(self.data[0], list):
            raise ValueError("data must be a list of lists")
        if self.data and len(self.data[0]) != len(self.headers):
            raise ValueError("data columns must match names length")
        if self.roi_indices is not None:
            if not isinstance(self.roi_indices, list):
                raise ValueError("roi_indices must be a list if provided")
            if self.roi_indices and isinstance(self.roi_indices[0], list):
                raise ValueError("roi_indices must be a list if provided")
            if len(self.roi_indices) != len(self.data):
                raise ValueError("roi_indices length must match number of data rows")

    @property
    def name(self) -> str:
        """Get the unique identifier name for this scalar table result.

        Returns:
            The string value of the kind attribute, which serves as a unique
             name identifier for this scalar table result type.
        """
        if isinstance(self.kind, TableKind):
            return self.kind.value
        return self.kind

    def __str__(self) -> str:
        """Return a string representation of the TableResult."""
        df = self.to_dataframe()
        text = f"TableResult(title={self.title}, kind={self.kind}, shape={df.shape})"
        text += os.linesep * 2
        text += str(df)
        return text

    # -------- Factory methods --------

    @classmethod
    def from_rows(
        cls,
        title: str,
        headers: Sequence[str],
        rows: list[list],
        roi_indices: list[int] | None = None,
        *,
        kind: TableKind | str = TableKind.CUSTOM,
        func_name: str | None = None,
        attrs: dict[str, object] | None = None,
    ) -> TableResult:
        """Create a TableResult from raw data.

        Args:
            title: Human-readable title for this table of results.
            headers: Column names (one per metric).
            rows: 2-D list of lists of shape (N, len(headers)) with values.
            roi_indices: Optional list (N,) mapping rows to ROI indices.
             Use NO_ROI (-1) for the "full image / no ROI" row.
            kind: Type of table result (e.g., TableKind.PULSE_FEATURES).
            func_name: Optional name of the computation function.
            attrs: Optional algorithmic context (e.g. thresholds, method variant).

        Returns:
            A TableResult instance.
        """
        return cls(
            title=title,
            kind=kind,
            headers=headers,
            data=rows,
            roi_indices=roi_indices,
            func_name=func_name,
            attrs={} if attrs is None else dict(attrs),
        )

    # -------- JSON-friendly (de)serialization (no DataLab metadata coupling) -----

    def to_dict(self) -> dict:
        """Convert the TableResult to a dictionary."""
        return {
            "schema": 1,
            "title": self.title,
            "kind": self.kind.value if isinstance(self.kind, TableKind) else self.kind,
            "names": list(self.headers),
            "data": self.data,
            "roi_indices": self.roi_indices,
            "func_name": self.func_name,
            "attrs": dict(self.attrs) if self.attrs else {},
        }

    @staticmethod
    def from_dict(d: dict) -> TableResult:
        """Convert a dictionary to a TableResult."""
        return TableResult(
            title=d["title"],
            kind=d.get("kind", TableKind.CUSTOM),
            headers=list(d["names"]),
            data=d["data"],
            roi_indices=d.get("roi_indices"),
            func_name=d.get("func_name"),
            attrs=dict(d.get("attrs", {})),
        )

    # -------- Pandas DataFrame interop --------

    def to_dataframe(self, visible_only: bool = False):
        """Convert the result to a pandas DataFrame.

        Args:
            visible_only: If True, include only visible headers based on display
             preferences. Default is False.

        Returns:
            DataFrame with an optional 'roi_index' column.
             If visible_only is True, only columns with visible headers are included.
        """
        df = pd.DataFrame(self.data, columns=self.headers)

        # Add roi_index column if present
        if self.roi_indices is not None:
            df.insert(0, "roi_index", self.roi_indices)

        # Filter to visible columns if requested
        if visible_only:
            visible_headers = self.get_visible_headers()
            df = DataFrameManager.apply_visible_only_filter(df, visible_headers)

        return df

    def get_display_preferences(self) -> dict[str, bool]:
        """Get display preferences for metrics.

        Returns:
            Dictionary mapping header names to visibility (True=visible, False=hidden).
            By default, all metrics are visible unless specified in attrs.
        """
        return DisplayPreferencesManager.get_display_preferences(
            self, self.headers, "hidden_metrics"
        )

    def set_display_preferences(self, preferences: dict[str, bool]) -> None:
        """Set display preferences for metrics.

        Args:
            preferences: Dictionary mapping header names to visibility
                        (True=visible, False=hidden)
        """
        DisplayPreferencesManager.set_display_preferences(
            self, preferences, self.headers, "hidden_metrics"
        )

    def get_visible_headers(self) -> list[str]:
        """Get list of currently visible headers based on display preferences.

        Returns:
            List of header names that should be displayed
        """
        return DisplayPreferencesManager.get_visible_headers(
            self, self.headers, "hidden_metrics"
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        title: str,
        kind: TableKind | str = TableKind.CUSTOM,
        attrs: dict = None,
    ) -> TableResult:
        """Create a TableResult from a pandas DataFrame.

        Args:
            df: pandas DataFrame. If 'roi_index' column is present, it is used
                for roi_indices.
            title: Title for the TableResult.
            kind: Type of table result (e.g., TableKind.PULSE_FEATURES).
            attrs: Optional dictionary of attributes.

        Returns:
            TableResult instance.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        cols = list(df.columns)
        if "roi_index" in cols:
            roi_indices = df["roi_index"].tolist()
            names = [c for c in cols if c != "roi_index"]
            data = df[names].values.tolist()
        else:
            roi_indices = None
            names = cols
            data = df.values.tolist()
        if attrs is None:
            attrs = {}
        return cls(
            title=title,
            kind=kind,
            headers=names,
            data=data,
            roi_indices=roi_indices,
            attrs=attrs,
        )

    # -------- User-oriented methods --------

    def col(self, name: str) -> list:
        """Return the column vector by name (raises KeyError if missing).

        Args:
            name: The name of the column to retrieve.

        Returns:
            A list containing the column data.
        """
        try:
            j = list(self.headers).index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return [row[j] for row in self.data]

    def __getitem__(self, name: str) -> list:
        """Shorthand for col(name)."""
        return self.col(name)

    def __contains__(self, name: str) -> bool:
        """Check if a column name exists in the table.

        Args:
            name: The name of the column to check.

        Returns:
            True if the column exists, False otherwise.
        """
        return name in self.headers

    def __len__(self) -> int:
        """Return the number of names in the table."""
        return len(self.headers)

    def value(self, name: str, roi: int | None = None) -> float:
        """Return a single scalar by column name and ROI.

        Args:
            name: The name of the column to retrieve.
            roi: The region of interest (ROI) to filter by (optional).
             Use None for NO_ROI row.

        Returns:
            A single scalar value from the specified column and ROI.
        """
        vec = self.col(name)
        if self.roi_indices is None:
            # single row (common in 'full image' stats)
            if len(vec) != 1:
                raise ValueError(
                    "Ambiguous selection: multiple rows but no ROI indices"
                )
            return vec[0]
        target = NO_ROI if roi is None else int(roi)
        matching_indices = [
            i for i, roi_idx in enumerate(self.roi_indices) if roi_idx == target
        ]
        if not matching_indices:
            raise KeyError(f"No row for ROI={target}")
        if len(matching_indices) != 1:
            raise ValueError(
                f"Ambiguous selection: {len(matching_indices)} rows for ROI={target}"
            )
        return vec[matching_indices[0]]

    def as_dict(self, roi: int | None = None) -> dict[str, Any]:
        """Return a {column -> value} mapping for one row (ROI or full image).

        Args:
            roi: The region of interest (ROI) to filter by (optional).
             Use None for NO_ROI row.

        Returns:
            A dictionary mapping column names to their corresponding values.
        """
        if self.roi_indices is None:
            if len(self.data) != 1:
                raise ValueError(
                    "Ambiguous selection: multiple rows but no ROI indices"
                )
            row = self.data[0]
        else:
            target = NO_ROI if roi is None else int(roi)
            matching_indices = [
                i for i, roi_idx in enumerate(self.roi_indices) if roi_idx == target
            ]
            if not matching_indices:
                raise KeyError(f"No row for ROI={target}")
            if len(matching_indices) != 1:
                raise ValueError(
                    f"Ambiguous selection: {len(matching_indices)} rows for "
                    f"ROI={target}"
                )
            row = self.data[matching_indices[0]]
        return {name: row[j] for j, name in enumerate(self.headers)}

    def to_html(
        self,
        obj: SignalObj | ImageObj | None = None,
        visible_only: bool = True,
        transpose_single_row: bool = True,
        **kwargs,
    ) -> str:
        """Convert the result to HTML format.

        Args:
            obj: SignalObj or ImageObj for ROI title extraction
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
        as a cell output, providing a rich HTML rendering of the table result.

        Returns:
            HTML representation of the table result with styling.
        """
        return HTML_TABLE_CSS + self.to_html()

    # -------- Convenience methods for table type identification --------

    def is_statistics(self) -> bool:
        """Check if this is a statistics table."""
        return self.kind == TableKind.STATISTICS

    def is_pulse_features(self) -> bool:
        """Check if this is a pulse features table."""
        return self.kind == TableKind.PULSE_FEATURES

    def is_custom(self) -> bool:
        """Check if this is a custom table."""
        return self.kind == TableKind.CUSTOM


class TableResultBuilder:
    """Builder for TableResult with fluent interface.

    Args:
        title: The title of the table.
        kind: The type of table result.
    """

    def __init__(self, title: str, kind: TableKind | str = TableKind.CUSTOM) -> None:
        self.title = title
        self.kind = kind

        # We define either a list of column functions, or a single global function
        # that returns a dataclass instance with float/int fields.
        self.column_funcs: list[tuple[Callable, str]] = []
        self.global_func: Callable | None = None

        self._hidden_columns: set[str] = set()

    def set_global_function(self, func: Callable) -> None:
        """Set a global function that returns a dataclass with float/int fields.

        Args:
            func: The function to compute the dataclass instance.
        """
        assert not self.column_funcs, "Cannot mix global and per-column functions"
        assert isinstance(func, Callable), "Global function must be callable"
        # Check function signature:
        sig = inspect.signature(func)
        if len(sig.parameters) < 1:
            raise ValueError(
                "Global function must accept at least one argument (xydata tuple)"
            )
        firstparam = list(sig.parameters.values())[0]
        if (
            firstparam.annotation is not sig.empty
            and firstparam.annotation != "tuple[np.ndarray, np.ndarray]"
        ):
            raise ValueError(
                "Global function must accept a (np.ndarray, np.ndarray) tuple"
            )
        # Check return type
        if sig.return_annotation is not sig.empty:
            ret_type = sig.return_annotation
            if not dataclasses.is_dataclass(ret_type):
                raise ValueError("Global function must return a dataclass")
        self.global_func = func

    def add(self, func: Callable, name: str) -> None:
        """Add a column function to the table.

        Args:
            func: The function to compute the column values.
            name: The name of the column.
        """
        assert self.global_func is None, "Cannot mix global and per-column functions"
        assert isinstance(name, str) and name, "Column name must be a non-empty string"
        assert isinstance(func, Callable), "Column function must be callable"
        # Check function signature:
        sig = inspect.signature(func)
        if len(sig.parameters) < 1:
            raise ValueError(
                f"Column function '{name}' must accept at least one argument"
            )
        first_param = list(sig.parameters.values())[0]
        if (
            first_param.annotation is not sig.empty
            and first_param.annotation != "np.ndarray"
        ):
            raise ValueError(f"Column function '{name}' must accept a np.ndarray")
        # Check return type
        if sig.return_annotation is not sig.empty and sig.return_annotation not in (
            "float",
            "int",
        ):
            raise ValueError(f"Column function '{name}' must return a float or int")
        self.column_funcs.append((name, func))

    def hide_columns(self, names: list[str]) -> TableResultBuilder:
        """Mark multiple columns as hidden in the display.

        Args:
            names: List of column names to hide.

        Returns:
            Self for method chaining.
        """
        self._hidden_columns.update(names)
        return self

    @staticmethod
    def __check_value(value) -> float | str:
        """Check and convert a value to float or str.

        Args:
            value: The value to check.

        Returns:
            The value converted to float or str.

        Raises:
            ValueError: If the value is not convertible to float or str.
        """
        try:
            value = float(value)
        except ValueError as exc:
            if not isinstance(value, str):
                raise ValueError(f"Unexpected non-numeric value: {value!r}") from exc
        return value

    def __compute_row_from_column_funcs(self, data: np.ndarray) -> list:
        """Compute a single row using the column functions.

        Args:
            data: The input data array.

        Returns:
            A list of computed values for the row.
        """
        row_data = []
        for _name, func in self.column_funcs:
            value = func(data)
            value = self.__check_value(value)
            row_data.append(value)
        return row_data

    def __compute_row_from_dataclass(self, result) -> tuple[list, list]:
        """Compute a single row using the global function's dataclass result.

        Args:
            result: The dataclass instance returned by the global function.

        Returns:
            A tuple of (row_data, names).
        """
        row_data = []
        names = []
        if not dataclasses.is_dataclass(result):
            raise ValueError("Global function must return a dataclass instance")
        for field in dataclasses.fields(result):
            value = getattr(result, field.name)
            if isinstance(value, (int, float, np.floating, np.integer, enum.Enum, str)):
                value = self.__check_value(value)
            else:
                value = None
            row_data.append(value)
            names.append(field.name)
        return row_data, names

    def compute(self, obj: SignalObj | ImageObj) -> TableResult:
        """Extract data from the image or signal object and compute the table.

        The ROI computation behavior depends on the TableKind:

        - STATISTICS: Computes results for both the whole object (NO_ROI) and each
          defined ROI.
        - PULSE_FEATURES: Computes results ONLY for ROIs if any are defined; otherwise
          computes for the whole object. This is because pulse features are meaningful
          only within specific ROI regions when multiple pulses are present.
        - CUSTOM: Default behavior is same as STATISTICS (whole object + ROIs).

        Args:
            obj: The image or signal object to extract data from.

        Returns:
            A TableResult object containing the extracted data.
        """
        names = [name for name, _ in self.column_funcs]
        roi_indices = list(obj.iterate_roi_indices())

        # Determine whether to include whole object computation based on TableKind
        # Convert kind to TableKind enum if it's a string
        if isinstance(self.kind, str):
            try:
                kind_enum = TableKind(self.kind)
            except ValueError:
                # If string doesn't match any TableKind, default to CUSTOM behavior
                kind_enum = TableKind.CUSTOM
        else:
            kind_enum = self.kind

        # Add whole object (None ROI) if:
        # 1. No ROIs exist, OR
        # 2. ROIs exist AND kind is not PULSE_FEATURES (which computes only on ROIs)
        has_rois = roi_indices and roi_indices[0] is not None
        if not has_rois or kind_enum != TableKind.PULSE_FEATURES:
            if has_rois:
                roi_indices.insert(0, None)

        rows = []
        roi_idx = []
        for i_roi in roi_indices:
            data = obj.get_data(i_roi)
            row_data = []
            if self.column_funcs:
                row_data = self.__compute_row_from_column_funcs(data)
            elif self.global_func:
                result = self.global_func(data)
                row_data, names = self.__compute_row_from_dataclass(result)
            rows.append(row_data)
            roi_idx.append(NO_ROI if i_roi is None else int(i_roi))

        # Remove columns with all None and/or NaN values
        if rows and names:
            valid_cols = []
            for j, name in enumerate(names):
                col_values = [row[j] for row in rows]
                if any(
                    v is not None and not (isinstance(v, float) and np.isnan(v))
                    for v in col_values
                ):
                    valid_cols.append(j)
            if len(valid_cols) < len(names):
                names = [names[j] for j in valid_cols]
                rows = [[row[j] for j in valid_cols] for row in rows]

        result = TableResult.from_rows(
            title=self.title,
            headers=names,
            rows=rows,
            roi_indices=roi_idx,
            kind=self.kind,
        )

        # Apply display preferences
        if self._hidden_columns:
            hidden_prefs = {name: name not in self._hidden_columns for name in names}
            result.set_display_preferences(hidden_prefs)

        return result


# ===========================
# Table utility functions
# ===========================


def calc_table_from_data(
    title: str,
    data: np.ndarray,
    labeledfuncs: Mapping[str, Callable[[np.ndarray], float]],
    roi_masks: list[np.ndarray] | None = None,
    kind: TableKind | str = TableKind.CUSTOM,
    attrs: dict[str, object] | None = None,
) -> TableResult:
    """Run scalar metrics on a full array or per-ROI masks and return a TableResult.

    Args:
        title: Result title.
        data: N-D array consumed by metric functions.
        labeledfuncs: Mapping of {label: func}, where func(data_or_masked) -> float.
        roi_masks: Optional list of boolean masks (same shape as data). If provided,
         results are computed per mask; otherwise a single full-image row is returned.
        kind: Type of table result (e.g., TableKind.PULSE_FEATURES).
        attrs: Optional algorithmic context.

    Returns:
        TableResult with rows per ROI mask (or one row if `roi_masks` is None).
        `roi_indices` will be the mask indices (0..M-1) or NO_ROI for the
        single row.
    """
    names = list(labeledfuncs.keys())
    funcs = list(labeledfuncs.values())

    if roi_masks:
        rows = []
        roi_idx = []
        for i, m in enumerate(roi_masks):
            sub = data[m] if (isinstance(m, np.ndarray) and m.dtype == bool) else data
            rows.append([float(f(sub)) for f in funcs])
            roi_idx.append(i)
        return TableResult(
            title=title,
            kind=kind,
            headers=names,
            data=rows,
            roi_indices=roi_idx,
            attrs={} if attrs is None else dict(attrs),
        )

    # No ROI: single row with NO_ROI sentinel
    row = [float(f(data)) for f in funcs]
    return TableResult(
        title=title,
        kind=kind,
        headers=names,
        data=[row],
        roi_indices=[NO_ROI],
        attrs={} if attrs is None else dict(attrs),
    )


def concat_tables(title: str, items: Iterable[TableResult]) -> TableResult:
    """Concatenate multiple TableResult objects with identical names.

    Args:
        title: Title for the concatenated result.
        items: Iterable of TableResult objects to concatenate.

    Returns:
        TableResult with concatenated data and updated metadata.
    """
    items = list(items)
    if not items:
        return TableResult(title=title, headers=[], data=[])
    first = items[0]
    cols = list(first.headers)
    kind = first.kind
    for it in items[1:]:
        if list(it.headers) != cols:
            raise ValueError(
                "All TableResult objects must share the same names to concatenate"
            )
        if it.kind != kind:
            kind = TableKind.CUSTOM  # Default to CUSTOM if kinds don't match
    data = []
    for it in items:
        data.extend(it.data)
    if any(it.roi_indices is not None for it in items):
        roi = []
        for it in items:
            if it.roi_indices is not None:
                roi.extend(it.roi_indices)
            else:
                roi.extend([NO_ROI] * len(it.data))
    else:
        roi = None
    return TableResult(title=title, kind=kind, headers=cols, data=data, roi_indices=roi)


def filter_table_by_roi(res: TableResult, roi: int | None) -> TableResult:
    """Filter rows by ROI index. If roi is None, keeps NO_ROI rows.

    Args:
        res: The TableResult to filter.
        roi: The ROI index to filter by, or None to keep all.

    Returns:
        A filtered TableResult.
    """
    if res.roi_indices is None:
        # No ROI info: either keep all or none depending on request
        keep_all = roi in (None, NO_ROI)
        data = res.data if keep_all else []
        indices = None if keep_all else []
        return TableResult(
            title=res.title,
            headers=list(res.headers),
            data=data,
            roi_indices=indices,
            attrs=dict(res.attrs),
        )
    target = NO_ROI if roi is None else int(roi)
    filtered_data = []
    filtered_indices = []
    for i, roi_idx in enumerate(res.roi_indices):
        if roi_idx == target:
            filtered_data.append(res.data[i])
            filtered_indices.append(roi_idx)
    return TableResult(
        title=res.title,
        headers=list(res.headers),
        data=filtered_data,
        roi_indices=filtered_indices,
        attrs=dict(res.attrs),
    )
