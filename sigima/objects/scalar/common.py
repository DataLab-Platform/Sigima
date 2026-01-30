# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Common utilities for scalar result objects
==========================================

This module provides shared functionality for TableResult and GeometryResult classes
without using inheritance or mixins, maintaining their dataclass integrity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sigima.objects import GeometryResult, ImageObj, SignalObj, TableResult

# Sentinel value for "full signal/image / no ROI" rows in result tables
NO_ROI: int = -1

# CSS styling for HTML tables in Jupyter notebooks
RESULT_TABLE_CSS = """
<style>
    .sigima-result-table {
        border-collapse: collapse;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     'Helvetica Neue', Arial, sans-serif;
        font-size: 13px;
        margin: 10px 0;
    }
    .sigima-result-table th {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
    }
    .sigima-result-table td {
        border: 1px solid #dee2e6;
        padding: 8px 12px;
        text-align: right;
    }
    .sigima-result-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    .sigima-result-table tr:hover {
        background-color: #e9ecef;
    }
</style>
"""


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

        # Transpose if single row and flag is set
        if transpose_single_row and len(df) == 1:
            # Transpose the dataframe
            df_t = df.T
            df_t.columns = [row_headers[0] if row_headers[0] else "Value"]
            df_t.index.name = "Item"
            # Get labels for the transposed view
            display_labels = list(df.columns)
            df_t.index = display_labels
            text = f'<u><b style="color: #5294e2">{result.title}</b></u>:'
            html_kwargs = {"border": 0}
            html_kwargs.update(kwargs)
            # Format numeric columns only, avoiding float_format on mixed data types
            for col in df_t.select_dtypes(include=["number"]).columns:
                df_t[col] = df_t[col].map(lambda x: f"{x:.3g}" if pd.notna(x) else x)
            text += df_t.to_html(**html_kwargs)
        else:
            # Standard horizontal layout
            df.index = row_headers
            text = f'<u><b style="color: #5294e2">{result.title}</b></u>:'
            html_kwargs = {"border": 0}
            html_kwargs.update(kwargs)
            # Format numeric columns only, avoiding float_format on mixed data types
            for col in df.select_dtypes(include=["number"]).columns:
                df[col] = df[col].map(lambda x: f"{x:.3g}" if pd.notna(x) else x)
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
