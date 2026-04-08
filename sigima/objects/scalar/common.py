# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Common utilities for scalar result objects
==========================================

This module provides shared functionality for TableResult and GeometryResult classes
without using inheritance or mixins, maintaining their dataclass integrity.
"""

from __future__ import annotations

from math import isinf, isnan
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sigima.objects import GeometryResult, ImageObj, SignalObj, TableResult

# Sentinel value for "full signal/image / no ROI" rows in result tables
NO_ROI: int = -1

NUM_DISPLAY_INFO_MAX_PLAIN = 6
NUM_DISPLAY_INFO_MAX_SCI = 12


def _exact_scientific(x: float) -> str:
    """Format a float in scientific notation with the minimum number of significant
    digits required for an exact round-trip representation.

    Args:
        x: The float value to format.

    Returns:
        Scientific notation string with the minimum significant digits.
    """
    s = repr(abs(float(x)))
    if "e" in s or "E" in s:
        mantissa = s.split("e")[0]
    else:
        mantissa = s
    # Remove dot, leading zeros, trailing zeros to count significant digits
    digits = mantissa.replace(".", "").lstrip("0").rstrip("0")
    n_sig = len(digits) or 1
    n_dec = max(0, n_sig - 1)  # decimal places in scientific notation
    return format(x, f".{n_dec}e")


def format_legend_value(x: float | int) -> str:
    """Format a numeric value for display in the legend area of a plot.

    Display strategy:

    1. If the plain representation is **≤ 6 characters** (``"."`` or ``","``
       included): display the value as-is (int or float).
    2. Otherwise, switch to **scientific notation**:

       a. If the exact scientific string is **≤ 12 characters** (``"."``,
          ``"e-"``, ``"e+"`` included): display the exact scientific value.
       b. If it exceeds 12 characters: round the mantissa so that the
          scientific string fits within 12 characters.

    Args:
        x: The numeric value to format.

    Returns:
        Formatted string suitable for plot legend display.
    """
    # Plain representation
    if isinstance(x, int):
        plain = str(x)
    else:
        xf = float(x)
        if isnan(xf) or isinf(xf):  # NaN or infinity should be displayed as-is
            return str(xf)
        if xf == int(xf):
            plain = str(int(xf))
        else:
            plain = str(xf)

    if len(plain) <= NUM_DISPLAY_INFO_MAX_PLAIN:
        return plain

    # Exact scientific notation
    sci = _exact_scientific(x)
    if len(sci) <= NUM_DISPLAY_INFO_MAX_SCI:
        return sci

    # Rounded scientific notation
    for n_dec in range(NUM_DISPLAY_INFO_MAX_SCI, -1, -1):
        s = format(float(x), f".{n_dec}e")
        if len(s) <= NUM_DISPLAY_INFO_MAX_SCI:
            return s
    return format(float(x), ".0e")


class DisplayPreferencesManager:
    """Manages display preferences for result objects."""

    @staticmethod
    def get_display_preferences(
        result: GeometryResult | TableResult,
        headers: list[str],
        attr_name: str = "hidden_headers",
    ) -> dict[str, bool]:
        """Get display preferences for headers.

        Args:
            result: The result object containing attrs
            headers: List of header names
            attr_name: Name of the attribute storing hidden headers

        Returns:
            Dictionary mapping header names to visibility (True=visible, False=hidden)
        """
        prefs = {}
        hidden_headers = result.attrs.get(attr_name, set())
        if isinstance(hidden_headers, (list, tuple)):
            hidden_headers = set(hidden_headers)

        for header in headers:
            prefs[header] = header not in hidden_headers
        return prefs

    @staticmethod
    def set_display_preferences(
        result: GeometryResult | TableResult,
        preferences: dict[str, bool],
        headers: list[str],
        attr_name: str = "hidden_headers",
    ) -> None:
        """Set display preferences for headers.

        Args:
            result: The result object to modify
            preferences: Dictionary mapping header names to visibility
            headers: List of valid header names
            attr_name: Name of the attribute to store hidden headers
        """
        hidden_headers = {
            header
            for header, visible in preferences.items()
            if not visible and header in headers
        }
        if hidden_headers:
            result.attrs[attr_name] = list(hidden_headers)
        elif attr_name in result.attrs:
            del result.attrs[attr_name]

    @staticmethod
    def get_visible_headers(
        result: GeometryResult | TableResult,
        headers: list[str],
        attr_name: str = "hidden_headers",
    ) -> list[str]:
        """Get list of currently visible headers.

        Args:
            result: The result object
            headers: List of all header names
            attr_name: Name of the attribute storing hidden headers

        Returns:
            List of header names that should be displayed
        """
        prefs = DisplayPreferencesManager.get_display_preferences(
            result, headers, attr_name
        )
        return [header for header in headers if prefs.get(header, True)]


class DataFrameManager:
    """Manages DataFrame operations for result objects."""

    @staticmethod
    def apply_visible_only_filter(
        df: pd.DataFrame, visible_headers: list[str]
    ) -> pd.DataFrame:
        """Apply visible-only filter to a DataFrame.

        Args:
            df: DataFrame to filter
            visible_headers: List of headers that should be visible

        Returns:
            Filtered DataFrame with only visible columns
        """
        # Keep roi_index column if present
        if "roi_index" in df.columns:
            visible_headers = ["roi_index"] + visible_headers

        # Filter to only available visible columns
        available_headers = [col for col in visible_headers if col in df.columns]
        if available_headers:
            return df[available_headers]
        return df


class ResultHtmlGenerator:
    """Utility class for generating HTML from result objects using composition."""

    @staticmethod
    def generate_html(
        result: GeometryResult | TableResult,
        obj: SignalObj | ImageObj | None = None,
        visible_only: bool = True,
        transpose_single_row: bool = True,
        **kwargs,
    ) -> str:
        """Generate HTML from a result object.

        Args:
            result: The result object (TableResult or GeometryResult)
            obj: SignalObj or ImageObj for ROI title extraction
            visible_only: If True, include only visible headers based on display
             preferences. Default is False.
            transpose_single_row: If True, transpose the table when there's only one row
            **kwargs: Additional arguments passed to DataFrame.to_html()

        Returns:
            HTML representation of the result
        """
        df = result.to_dataframe(visible_only=visible_only)

        # Remove roi_index column for display
        if "roi_index" in df.columns:
            roi_indices = df["roi_index"].tolist()
            df = df.drop(columns=["roi_index"])
        else:
            roi_indices = None

        # Create row headers
        row_headers = ResultHtmlGenerator._get_row_headers(result, roi_indices, obj)

        # Apply per-column formatting on the original df (before any transpose)
        # so that column names are still available for format lookup.
        # We iterate over ALL columns (not just numeric dtype) because columns with
        # Optional[float] fields may have object dtype in pandas
        # when they contain None values, and would be missed by select_dtypes.
        column_formats = result.attrs.get("column_formats", {})
        global_default_fmt = ".4g"
        default_fmt = column_formats.get("*", global_default_fmt)
        for col in df.columns:
            fmt = column_formats.get(col, default_fmt)
            if callable(fmt):
                df[col] = df[col].map(
                    lambda x, f=fmt: (
                        f(float(x))
                        if isinstance(x, (int, float)) and pd.notna(x)
                        else x
                    )
                )
            else:
                df[col] = df[col].map(
                    lambda x, f=fmt: (
                        format(float(x), f)
                        if isinstance(x, (int, float)) and pd.notna(x)
                        else x
                    )
                )

        # Transpose if single row and flag is set
        if transpose_single_row and len(df) == 1:
            # Transpose the dataframe (values already formatted)
            df_t = df.T
            df_t.columns = [row_headers[0] if row_headers[0] else "Value"]
            df_t.index.name = "Item"
            # Get labels for the transposed view
            display_labels = list(df.columns)
            df_t.index = display_labels
            text = f'<u><b style="color: #5294e2">{result.title}</b></u>:'
            html_kwargs = {"border": 0}
            html_kwargs.update(kwargs)
            text += df_t.to_html(**html_kwargs)
        else:
            # Standard horizontal layout
            df.index = row_headers
            text = f'<u><b style="color: #5294e2">{result.title}</b></u>:'
            html_kwargs = {"border": 0}
            html_kwargs.update(kwargs)
            text += df.to_html(**html_kwargs)

        return text

    @staticmethod
    def _get_row_headers(
        result: TableResult | GeometryResult,
        roi_indices: list[int] | None,
        obj: SignalObj | ImageObj | None,
    ) -> list[str]:
        """Create row headers from ROI indices.

        .. note::

           Handles gracefully the case where:
           - `obj` is None: uses generic "ROI N" headers instead of ROI titles
           - `roi_indices` reference ROIs that no longer exist in `obj.roi`
             (e.g., if HTML rendering happens before result recomputation after
             ROI deletion)
        """
        row_headers = []
        if roi_indices is not None:
            for roi_idx in roi_indices:
                if roi_idx == NO_ROI:
                    header = ""
                else:
                    header = f"ROI {roi_idx}"
                    # Try to get ROI title from object if available
                    if obj is not None and obj.roi is not None:
                        # Check if roi_idx is valid (defensive against stale indices)
                        if 0 <= roi_idx < len(obj.roi.single_rois):
                            header = obj.roi.get_single_roi_title(roi_idx)
                        # else: keep default "ROI {roi_idx}" for out-of-bounds indices
                row_headers.append(header)
        else:
            # Need to get DataFrame to know the number of rows
            df = result.to_dataframe()
            if "roi_index" in df.columns:
                df = df.drop(columns=["roi_index"])
            row_headers = [""] * len(df)
        return row_headers
