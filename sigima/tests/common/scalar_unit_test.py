# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for scalar computation functions (GeometryResult transformations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sigima.objects import (
    NO_ROI,
    GeometryResult,
    KindShape,
    TableResult,
    calc_table_from_data,
    concat_geometries,
    concat_tables,
    filter_geometry_by_roi,
    filter_table_by_roi,
)
from sigima.proc.image import transformer


def create_rectangle(x0=0.0, y0=0.0, w=1.0, h=1.0) -> GeometryResult:
    """Create a simple rectangle GeometryResult."""
    coords = np.array([[x0, y0, w, h]], dtype=float)
    return GeometryResult("rect", "rectangle", coords)


class TestGeometryTransformations:
    """Test class for geometry transformation functions."""

    def test_rotate(self) -> None:
        """Test rotation of a rectangle geometry result."""
        rect = create_rectangle(0.0, 0.0, 1.0, 2.0)
        rotated = transformer.rotate(rect, np.pi / 2, center=(0.5, 1.0))
        expected_coords = np.array([[-0.5, 0.5, 2.0, 1.0]])
        assert rotated.coords.shape == rect.coords.shape
        assert np.allclose(rotated.coords, expected_coords)

    def test_fliph(self) -> None:
        """Test horizontal flip and its reversibility."""
        rect = create_rectangle(1.0, 2.0, 2.0, 3.0)
        flipped = transformer.fliph(rect, cx=2.0)
        flipped_back = transformer.fliph(flipped, cx=2.0)
        np.testing.assert_allclose(flipped_back.coords, rect.coords, rtol=1e-12)

    def test_flipv(self) -> None:
        """Test vertical flip and its reversibility."""
        rect = create_rectangle(1.0, 2.0, 2.0, 3.0)
        flipped = transformer.flipv(rect, cy=3.5)
        flipped_back = transformer.flipv(flipped, cy=3.5)
        np.testing.assert_allclose(flipped_back.coords, rect.coords, rtol=1e-12)

    def test_translate(self) -> None:
        """Test translation of a geometry result."""
        rect = create_rectangle()
        translated = transformer.translate(rect, 1.5, -2.0)
        expected = rect.coords + np.array([1.5, -2.0, 0.0, 0.0])
        np.testing.assert_allclose(translated.coords, expected, rtol=1e-12)

    def test_scale(self) -> None:
        """Test scaling and inverse scaling of a geometry result."""
        rect = create_rectangle(1.0, 1.0, 2.0, 2.0)
        scaled = transformer.scale(rect, 2.0, 0.5, center=(2.0, 2.0))
        unscaled = transformer.scale(scaled, 0.5, 2.0, center=(2.0, 2.0))
        np.testing.assert_allclose(unscaled.coords, rect.coords, rtol=1e-12)

    def test_transpose(self) -> None:
        """Test transpose and double-transpose (should restore original)."""
        rect = create_rectangle(1.0, 2.0, 3.0, 4.0)
        transposed = transformer.transpose(rect)
        transposed_back = transformer.transpose(transposed)
        np.testing.assert_allclose(transposed_back.coords, rect.coords, rtol=1e-12)


class TestTableResultInitialization:
    """Test class for TableResult initialization and validation."""

    def test_init_valid(self) -> None:
        """Test TableResult initialization with valid data."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        roi_indices = [0, 1]
        table = TableResult(
            title="Test Table",
            headers=["col1", "col2"],
            data=data,
            roi_indices=roi_indices,
            attrs={"method": "test"},
        )
        assert table.title == "Test Table"
        assert list(table.headers) == ["col1", "col2"]
        assert table.data == data
        assert table.roi_indices == roi_indices
        assert table.attrs == {"method": "test"}

    def test_init_invalid_title(self) -> None:
        """Test TableResult initialization with invalid title."""
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            TableResult(title="", headers=["col"], data=[[1]])

    def test_init_invalid_names(self) -> None:
        """Test TableResult initialization with invalid headers."""
        with pytest.raises(ValueError, match="names must be a sequence of strings"):
            TableResult(title="Test", headers=[1, 2], data=[[1, 2]])

    def test_init_invalid_data_shape(self) -> None:
        """Test TableResult initialization with invalid data shape."""
        with pytest.raises(ValueError, match="data must be a list of lists"):
            TableResult(title="Test", headers=["col"], data=[1, 2, 3])

    def test_init_invalid_data_columns(self) -> None:
        """Test TableResult initialization with mismatched data columns."""
        with pytest.raises(ValueError, match="data columns must match names length"):
            TableResult(title="Test", headers=["col1", "col2"], data=[[1, 2, 3]])

    def test_init_invalid_roi_indices(self) -> None:
        """Test TableResult initialization with mismatched ROI indices length."""
        with pytest.raises(
            ValueError, match="roi_indices length must match number of data rows"
        ):
            TableResult(
                title="Test",
                headers=["col1", "col2"],
                data=[[1, 2], [3, 4]],
                roi_indices=[0],
            )

    def test_init_invalid_roi_indices_type(self) -> None:
        """Test TableResult initialization with invalid roi_indices type."""
        with pytest.raises(ValueError, match="roi_indices must be a list if provided"):
            TableResult(
                title="Test",
                headers=["col1", "col2"],
                data=[[1, 2], [3, 4]],
                roi_indices=(0, 1),
            )

    def test_init_invalid_roi_indices_2d(self) -> None:
        """Test TableResult initialization with 2D roi_indices."""
        with pytest.raises(ValueError, match="roi_indices must be a list if provided"):
            TableResult(
                title="Test",
                headers=["col1", "col2"],
                data=[[1, 2], [3, 4]],
                roi_indices=[[0], [1]],
            )

    def test_from_dataframe(self) -> None:
        """Test TableResult.from_dataframe factory method."""
        # Test without roi_index column
        df = pd.DataFrame({"col1": [1.0, 2.0], "col2": [3.0, 4.0]})
        result = TableResult.from_dataframe(df, title="Test DataFrame")
        assert result.title == "Test DataFrame"
        assert list(result.headers) == ["col1", "col2"]
        # DataFrame converts row-wise: row 0 has [1.0, 3.0], row 1 has [2.0, 4.0]
        assert result.data == [[1.0, 3.0], [2.0, 4.0]]
        assert result.roi_indices is None

        # Test with roi_index column
        df_with_roi = pd.DataFrame(
            {"col1": [1.0, 2.0], "col2": [3.0, 4.0], "roi_index": [0, 1]}
        )
        result_with_roi = TableResult.from_dataframe(
            df_with_roi, title="Test DataFrame with ROI"
        )
        assert result_with_roi.title == "Test DataFrame with ROI"
        assert list(result_with_roi.headers) == ["col1", "col2"]
        assert result_with_roi.data == [[1.0, 3.0], [2.0, 4.0]]
        assert result_with_roi.roi_indices == [0, 1]

        # Test with attrs
        result_with_attrs = TableResult.from_dataframe(
            df, title="Test with Attrs", attrs={"method": "test"}
        )
        assert result_with_attrs.attrs == {"method": "test"}

        # Test with invalid input (not a DataFrame)
        with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
            TableResult.from_dataframe([[1, 2]], title="Invalid")


class TestTableResultFactory:
    """Test class for TableResult factory methods."""

    def test_from_rows(self) -> None:
        """Test TableResult.from_rows factory method."""
        result = TableResult.from_rows(
            title="Test",
            headers=["col1", "col2"],
            rows=[[1.0, 2.0], [3.0, 4.0]],
            roi_indices=[0, 1],
            attrs={"method": "test"},
        )
        assert result.title == "Test"
        assert list(result.headers) == ["col1", "col2"]
        assert result.data == [[1.0, 2.0], [3.0, 4.0]]
        assert result.roi_indices == [0, 1]
        assert result.attrs == {"method": "test"}


class TestTableResultSerialization:
    """Test class for TableResult serialization methods."""

    def test_to_dict(self) -> None:
        """Test TableResult.to_dict serialization."""
        table = TableResult(
            title="Test Table",
            headers=["col1", "col2"],
            data=[[1.0, 2.0], [3.0, 4.0]],
            roi_indices=[0, 1],
            attrs={"method": "test"},
        )
        expected = {
            "schema": 1,
            "title": "Test Table",
            "kind": "results",
            "names": ["col1", "col2"],
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "roi_indices": [0, 1],
            "func_name": None,
            "attrs": {"method": "test"},
        }
        assert table.to_dict() == expected

    def test_from_dict(self) -> None:
        """Test TableResult.from_dict deserialization."""
        data = {
            "title": "Test Table",
            "kind": "results",
            "names": ["col1", "col2"],
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "roi_indices": [0, 1],
            "attrs": {"method": "test"},
        }
        table = TableResult.from_dict(data)
        assert table.title == "Test Table"
        assert list(table.headers) == ["col1", "col2"]
        assert table.data == [[1.0, 2.0], [3.0, 4.0]]
        assert table.roi_indices == [0, 1]
        assert table.attrs == {"method": "test"}

    def test_to_html(self) -> None:
        """Test TableResult.to_html serialization."""
        table = TableResult(
            title="Test Table",
            headers=["col1", "col2"],
            data=[[1.0, 2.0], [3.0, 4.0]],
        )
        html = table.to_html()
        assert "<table" in html
        assert "col1" in html
        assert "col2" in html

    def test_repr_html(self) -> None:
        """Test TableResult._repr_html_ for Jupyter notebook display."""
        table = TableResult(
            title="Test Table",
            headers=["col1", "col2"],
            data=[[1.0, 2.0], [3.0, 4.0]],
        )
        html = table._repr_html_()  # pylint: disable=protected-access
        # Check that CSS styling is included
        assert "<style>" in html
        assert "sigima-html-table" in html
        # Check that table content is included
        assert "<table" in html
        assert "col1" in html
        assert "col2" in html


class TestTableResultDataAccess:
    """Test class for TableResult data access methods."""

    def setup_method(self) -> None:
        """Set up test data for each test method."""
        # pylint: disable=attribute-defined-outside-init
        self.table = TableResult(
            title="Test Table",
            headers=["col1", "col2"],
            data=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            roi_indices=[NO_ROI, 0, 1],
        )

    def test_col(self) -> None:
        """Test TableResult.col method."""
        assert self.table.col("col1") == [1.0, 3.0, 5.0]
        assert self.table.col("col2") == [2.0, 4.0, 6.0]

        # Test missing column
        with pytest.raises(KeyError, match="missing_col"):
            self.table.col("missing_col")

    def test_getitem(self) -> None:
        """Test TableResult.__getitem__ method (shorthand for col)."""
        assert self.table["col1"] == [1.0, 3.0, 5.0]
        assert self.table["col2"] == [2.0, 4.0, 6.0]

    def test_contains(self) -> None:
        """Test TableResult.__contains__ method."""
        assert "col1" in self.table
        assert "col2" in self.table
        assert "missing_col" not in self.table

    def test_len(self) -> None:
        """Test TableResult.__len__ method."""
        assert len(self.table) == 2  # Number of columns

    def test_value_single_row(self) -> None:
        """Test TableResult.value with single row."""
        single_row_table = TableResult(
            title="Single", headers=["col1", "col2"], data=[[1.0, 2.0]]
        )
        assert single_row_table.value("col1") == 1.0
        assert single_row_table.value("col2") == 2.0

    def test_value_with_roi(self) -> None:
        """Test TableResult.value with ROI filtering."""
        # NO_ROI row
        assert self.table.value("col1", roi=None) == 1.0
        assert self.table.value("col2", roi=None) == 2.0

        # ROI 0
        assert self.table.value("col1", roi=0) == 3.0
        assert self.table.value("col2", roi=0) == 4.0

        # ROI 1
        assert self.table.value("col1", roi=1) == 5.0
        assert self.table.value("col2", roi=1) == 6.0

    def test_value_ambiguous_selection(self) -> None:
        """Test TableResult.value with ambiguous selection."""
        multi_row_no_roi = TableResult(
            title="Multi", headers=["col1"], data=[[1.0], [2.0]]
        )
        with pytest.raises(ValueError, match="Ambiguous selection"):
            multi_row_no_roi.value("col1")

    def test_value_duplicate_roi(self) -> None:
        """Test TableResult.value with duplicate ROI indices."""
        duplicate_roi_table = TableResult(
            title="Dup",
            headers=["col1"],
            data=[[1.0], [2.0]],
            roi_indices=[0, 0],
        )
        with pytest.raises(ValueError, match="Ambiguous selection"):
            duplicate_roi_table.value("col1", roi=0)

    def test_as_dict_single_row(self) -> None:
        """Test TableResult.as_dict with single row."""
        single_row_table = TableResult(
            title="Single", headers=["col1", "col2"], data=[[1.0, 2.0]]
        )
        expected = {"col1": 1.0, "col2": 2.0}
        assert single_row_table.as_dict() == expected

    def test_as_dict_with_roi(self) -> None:
        """Test TableResult.as_dict with ROI filtering."""
        # NO_ROI row
        expected_no_roi = {"col1": 1.0, "col2": 2.0}
        assert self.table.as_dict(roi=None) == expected_no_roi

        # ROI 0
        expected_roi_0 = {"col1": 3.0, "col2": 4.0}
        assert self.table.as_dict(roi=0) == expected_roi_0


class TestTableResultDisplayPreferences:
    """Test class for TableResult display preferences functionality."""

    def setup_method(self) -> None:
        """Set up test data for each test method."""
        # pylint: disable=attribute-defined-outside-init
        self.table = TableResult(
            title="Test Table",
            headers=["col1", "col2", "col3"],
            data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )

    def test_display_preferences_default(self) -> None:
        """Test default display preferences (all visible)."""
        prefs = self.table.get_display_preferences()
        expected = {"col1": True, "col2": True, "col3": True}
        assert prefs == expected

        visible = self.table.get_visible_headers()
        assert visible == ["col1", "col2", "col3"]

    def test_set_display_preferences(self) -> None:
        """Test setting display preferences."""
        self.table.set_display_preferences({"col1": True, "col2": False, "col3": True})

        prefs = self.table.get_display_preferences()
        expected = {"col1": True, "col2": False, "col3": True}
        assert prefs == expected

        visible = self.table.get_visible_headers()
        assert visible == ["col1", "col3"]

    def test_display_preferences_invalid_columns(self) -> None:
        """Test setting display preferences with invalid column names."""
        # Should ignore invalid columns
        self.table.set_display_preferences(
            {"col1": False, "invalid_col": False, "col3": True}
        )

        prefs = self.table.get_display_preferences()
        expected = {"col1": False, "col2": True, "col3": True}
        assert prefs == expected

    def test_display_preferences_all_hidden(self) -> None:
        """Test hiding all columns."""
        self.table.set_display_preferences(
            {"col1": False, "col2": False, "col3": False}
        )

        visible = self.table.get_visible_headers()
        assert visible == []

    def test_to_dataframe_visible_only_default(self) -> None:
        """Test to_dataframe with visible_only=False (default)."""
        df = self.table.to_dataframe(visible_only=False)
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_to_dataframe_visible_only_true(self) -> None:
        """Test to_dataframe with visible_only=True."""
        self.table.set_display_preferences({"col1": True, "col2": False, "col3": True})

        df = self.table.to_dataframe(visible_only=True)
        assert list(df.columns) == ["col1", "col3"]

    def test_to_dataframe_visible_only_all_hidden(self) -> None:
        """Test to_dataframe with all columns hidden."""
        self.table.set_display_preferences(
            {"col1": False, "col2": False, "col3": False}
        )

        df = self.table.to_dataframe(visible_only=True)
        # When all columns are hidden, should return original dataframe
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_display_preferences_persistence(self) -> None:
        """Test that display preferences persist through serialization."""
        self.table.set_display_preferences({"col1": True, "col2": False, "col3": True})

        # Serialize and deserialize
        serialized = self.table.to_dict()
        restored = TableResult.from_dict(serialized)

        # Check preferences are preserved
        prefs = restored.get_display_preferences()
        expected = {"col1": True, "col2": False, "col3": True}
        assert prefs == expected


class TestGeometryResultInitialization:
    """Test class for GeometryResult initialization and validation."""

    def test_init_valid_point(self) -> None:
        """Test GeometryResult initialization with valid point data."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        roi_indices = np.array([0, 1])
        geom = GeometryResult(
            title="Test Points",
            kind=KindShape.POINT,
            coords=coords,
            roi_indices=roi_indices,
            attrs={"method": "test"},
        )
        assert geom.title == "Test Points"
        assert geom.kind == KindShape.POINT
        np.testing.assert_array_equal(geom.coords, coords)
        np.testing.assert_array_equal(geom.roi_indices, roi_indices)
        assert geom.attrs == {"method": "test"}

    def test_init_string_kind(self) -> None:
        """Test GeometryResult initialization with string kind."""
        coords = np.array([[1.0, 2.0]])
        geom = GeometryResult("Test", "point", coords)
        assert geom.kind == KindShape.POINT

    def test_init_invalid_kind(self) -> None:
        """Test GeometryResult initialization with invalid kind."""
        coords = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="Unsupported geometry kind"):
            GeometryResult("Test", "invalid_shape", coords)

    def test_init_invalid_title(self) -> None:
        """Test GeometryResult initialization with invalid title."""
        coords = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            GeometryResult("", KindShape.POINT, coords)

    def test_init_invalid_coords_shape(self) -> None:
        """Test GeometryResult initialization with invalid coords shape."""
        coords = np.array([1.0, 2.0])  # 1D instead of 2D
        with pytest.raises(ValueError, match="coords must be a 2-D numpy array"):
            GeometryResult("Test", KindShape.POINT, coords)

    def test_init_point_wrong_columns(self) -> None:
        """Test GeometryResult point initialization with wrong number of columns."""
        coords = np.array([[1.0, 2.0, 3.0]])  # 3 columns instead of 2
        with pytest.raises(ValueError, match="coords for 'point' must be \\(N,2\\)"):
            GeometryResult("Test", KindShape.POINT, coords)

    def test_init_segment_wrong_columns(self) -> None:
        """Test GeometryResult segment initialization with wrong number of columns."""
        coords = np.array([[1.0, 2.0, 3.0]])  # 3 columns instead of 4
        with pytest.raises(ValueError, match="coords for 'segment' must be \\(N,4\\)"):
            GeometryResult("Test", KindShape.SEGMENT, coords)

    def test_init_circle_wrong_columns(self) -> None:
        """Test GeometryResult circle initialization with wrong number of columns."""
        coords = np.array([[1.0, 2.0]])  # 2 columns instead of 3
        with pytest.raises(ValueError, match="coords for 'circle' must be \\(N,3\\)"):
            GeometryResult("Test", KindShape.CIRCLE, coords)

    def test_init_ellipse_wrong_columns(self) -> None:
        """Test GeometryResult ellipse initialization with wrong number of columns."""
        coords = np.array([[1.0, 2.0, 3.0, 4.0]])  # 4 columns instead of 5
        with pytest.raises(ValueError, match="coords for 'ellipse' must be \\(N,5\\)"):
            GeometryResult("Test", KindShape.ELLIPSE, coords)

    def test_init_rectangle_wrong_columns(self) -> None:
        """Test GeometryResult rectangle initialization with wrong columns."""
        coords = np.array([[1.0, 2.0, 3.0]])  # 3 columns instead of 4
        with pytest.raises(
            ValueError, match="coords for 'rectangle' must be \\(N,4\\)"
        ):
            GeometryResult("Test", KindShape.RECTANGLE, coords)

    def test_init_polygon_odd_columns(self) -> None:
        """Test GeometryResult polygon initialization with odd number of columns."""
        coords = np.array([[1.0, 2.0, 3.0]])  # 3 columns (odd)
        with pytest.raises(
            ValueError, match="coords for 'polygon' must be \\(N,2M\\) for M vertices"
        ):
            GeometryResult("Test", KindShape.POLYGON, coords)

    def test_init_mismatched_roi_indices(self) -> None:
        """Test GeometryResult initialization with mismatched ROI indices."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        roi_indices = np.array([0])  # Only 1 element for 2 coords
        with pytest.raises(
            ValueError, match="roi_indices length must match number of coord rows"
        ):
            GeometryResult("Test", KindShape.POINT, coords, roi_indices)


class TestGeometryResultFactory:
    """Test class for GeometryResult factory methods."""

    def test_from_coords(self) -> None:
        """Test GeometryResult.from_coords factory method."""
        coords = [[1.0, 2.0], [3.0, 4.0]]
        roi_indices = [0, 1]
        geom = GeometryResult.from_coords(
            title="Test Points",
            kind=KindShape.POINT,
            coords=coords,
            roi_indices=roi_indices,
            attrs={"method": "test"},
        )
        assert geom.title == "Test Points"
        assert geom.kind == KindShape.POINT
        np.testing.assert_array_equal(geom.coords, np.array(coords, dtype=float))
        np.testing.assert_array_equal(
            geom.roi_indices, np.array(roi_indices, dtype=int)
        )
        assert geom.attrs == {"method": "test"}


class TestGeometryResultSerialization:
    """Test class for GeometryResult serialization methods."""

    def test_to_dict(self) -> None:
        """Test GeometryResult.to_dict serialization."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        roi_indices = np.array([0, 1])
        geom = GeometryResult(
            title="Test Points",
            kind=KindShape.POINT,
            coords=coords,
            roi_indices=roi_indices,
            attrs={"method": "test"},
        )
        expected = {
            "schema": 1,
            "title": "Test Points",
            "kind": "point",
            "coords": [[1.0, 2.0], [3.0, 4.0]],
            "roi_indices": [0, 1],
            "func_name": None,
            "attrs": {"method": "test"},
        }
        assert geom.to_dict() == expected

    def test_from_dict(self) -> None:
        """Test GeometryResult.from_dict deserialization."""
        data = {
            "title": "Test Points",
            "kind": "point",
            "coords": [[1.0, 2.0], [3.0, 4.0]],
            "roi_indices": [0, 1],
            "attrs": {"method": "test"},
        }
        geom = GeometryResult.from_dict(data)
        assert geom.title == "Test Points"
        assert geom.kind == KindShape.POINT
        np.testing.assert_array_equal(geom.coords, np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(geom.roi_indices, np.array([0, 1]))
        assert geom.attrs == {"method": "test"}


class TestGeometryResultDataAccess:
    """Test class for GeometryResult data access methods."""

    def test_len(self) -> None:
        """Test GeometryResult.__len__ method."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        geom = GeometryResult("Test", KindShape.POINT, coords)
        assert len(geom) == 3

    def test_rows_no_roi(self) -> None:
        """Test GeometryResult.rows with no ROI indices."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        geom = GeometryResult("Test", KindShape.POINT, coords)
        np.testing.assert_array_equal(geom.rows(), coords)
        np.testing.assert_array_equal(geom.rows(roi=0), coords)

    def test_rows_with_roi(self) -> None:
        """Test GeometryResult.rows with ROI filtering."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        roi_indices = np.array([NO_ROI, 0, 1])
        geom = GeometryResult("Test", KindShape.POINT, coords, roi_indices)

        # Test NO_ROI
        expected_no_roi = np.array([[1.0, 2.0]])
        np.testing.assert_array_equal(geom.rows(roi=None), expected_no_roi)

        # Test ROI 0
        expected_roi_0 = np.array([[3.0, 4.0]])
        np.testing.assert_array_equal(geom.rows(roi=0), expected_roi_0)

        # Test ROI 1
        expected_roi_1 = np.array([[5.0, 6.0]])
        np.testing.assert_array_equal(geom.rows(roi=1), expected_roi_1)


class TestGeometryResultShapeSpecific:
    """Test class for shape-specific GeometryResult methods."""

    def test_segments_to_dataframe(self) -> None:
        """Test GeometryResult.to_dataframe for segments."""
        for coords, add_col in [
            (
                np.array([[0.0, 0.0, 3.0, 4.0], [1.0, 1.0, 4.0, 5.0]]),
                "length",  # General segments
            ),
            (
                np.array([[0.0, 0.0, 3.0, 0.0], [1.0, 1.0, 4.0, 1.0]]),
                "Δx",  # Horizontal segments
            ),
            (
                np.array([[0.0, 0.0, 0.0, 4.0], [1.0, 1.0, 1.0, 5.0]]),
                "Δy",  # Vertical segments
            ),
        ]:
            geom = GeometryResult("Test", KindShape.SEGMENT, coords)
            df = geom.to_dataframe()
            assert list(df.columns) == ["x0", "y0", "x1", "y1", add_col]
            np.testing.assert_array_equal(df.values[:, :-1], coords)

    def test_segments_to_html(self) -> None:
        """Test GeometryResult.to_html for segments."""
        coords = np.array([[0.0, 0.0, 3.0, 4.0], [1.0, 1.0, 4.0, 5.0]])
        geom = GeometryResult("Test", KindShape.SEGMENT, coords)
        html_visible_false = geom.to_html(visible_only=False)
        assert "<table" in html_visible_false
        assert "x0" in html_visible_false
        assert "y0" in html_visible_false
        assert "x1" in html_visible_false
        assert "y1" in html_visible_false
        assert "length" in html_visible_false
        html_visible_true = geom.to_html(visible_only=True)
        assert "<table" in html_visible_true
        assert "x0" not in html_visible_true
        assert "y0" not in html_visible_true
        assert "x1" not in html_visible_true
        assert "y1" not in html_visible_true
        assert "length" in html_visible_true

    def test_segments_repr_html(self) -> None:
        """Test GeometryResult._repr_html_ for Jupyter notebook display."""
        coords = np.array([[0.0, 0.0, 3.0, 4.0], [1.0, 1.0, 4.0, 5.0]])
        geom = GeometryResult("Test", KindShape.SEGMENT, coords)
        html = geom._repr_html_()  # pylint: disable=protected-access
        # Check that CSS styling is included
        assert "<style>" in html
        assert "sigima-html-table" in html
        # Check that table content is included
        assert "<table" in html
        assert "length" in html

    def test_segments_lengths(self) -> None:
        """Test GeometryResult.segments_lengths method."""
        # Create segments: (0,0)-(3,4) and (1,1)-(4,5)
        coords = np.array([[0.0, 0.0, 3.0, 4.0], [1.0, 1.0, 4.0, 5.0]])
        geom = GeometryResult("Test", KindShape.SEGMENT, coords)
        lengths = geom.segments_lengths()
        expected = np.array([5.0, 5.0])  # Both segments have length 5
        np.testing.assert_array_equal(lengths, expected)

    def test_segments_lengths_wrong_kind(self) -> None:
        """Test GeometryResult.segments_lengths with wrong kind."""
        coords = np.array([[1.0, 2.0]])
        geom = GeometryResult("Test", KindShape.POINT, coords)
        with pytest.raises(
            ValueError, match="segments_lengths requires kind='segment'"
        ):
            geom.segments_lengths()

    def test_circles_radii(self) -> None:
        """Test GeometryResult.circles_radii method."""
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        geom = GeometryResult("Test", KindShape.CIRCLE, coords)
        radii = geom.circles_radii()
        expected = np.array([3.0, 6.0])
        np.testing.assert_array_equal(radii, expected)

    def test_circles_radii_wrong_kind(self) -> None:
        """Test GeometryResult.circles_radii with wrong kind."""
        coords = np.array([[1.0, 2.0]])
        geom = GeometryResult("Test", KindShape.POINT, coords)
        with pytest.raises(ValueError, match="circles_radii requires kind='circle'"):
            geom.circles_radii()

    def test_ellipse_axes_angles(self) -> None:
        """Test GeometryResult.ellipse_axes_angles method."""
        coords = np.array([[1.0, 2.0, 3.0, 4.0, 0.5], [5.0, 6.0, 7.0, 8.0, 1.0]])
        geom = GeometryResult("Test", KindShape.ELLIPSE, coords)
        a, b, theta = geom.ellipse_axes_angles()
        np.testing.assert_array_equal(a, np.array([3.0, 7.0]))
        np.testing.assert_array_equal(b, np.array([4.0, 8.0]))
        np.testing.assert_array_equal(theta, np.array([0.5, 1.0]))

    def test_ellipse_axes_angles_wrong_kind(self) -> None:
        """Test GeometryResult.ellipse_axes_angles with wrong kind."""
        coords = np.array([[1.0, 2.0]])
        geom = GeometryResult("Test", KindShape.POINT, coords)
        with pytest.raises(
            ValueError, match="ellipse_axes_angles requires kind='ellipse'"
        ):
            geom.ellipse_axes_angles()


class TestGeometryResultDisplayPreferences:
    """Test class for GeometryResult display preferences functionality."""

    def test_display_preferences_default(self) -> None:
        """Test default display preferences for rectangle (all visible)."""
        coords = np.array([[0.0, 0.0, 10.0, 5.0]])
        geom = GeometryResult("Test Rectangle", KindShape.RECTANGLE, coords)

        prefs = geom.get_display_preferences()
        expected = {"x": True, "y": True, "width": True, "height": True}
        assert prefs == expected

        visible = geom.get_visible_headers()
        assert visible == ["x", "y", "width", "height"]

    def test_set_display_preferences(self) -> None:
        """Test setting display preferences for rectangle."""
        coords = np.array([[0.0, 0.0, 10.0, 5.0]])
        geom = GeometryResult("Test Rectangle", KindShape.RECTANGLE, coords)

        geom.set_display_preferences(
            {"x": True, "y": True, "width": False, "height": True}
        )

        prefs = geom.get_display_preferences()
        expected = {"x": True, "y": True, "width": False, "height": True}
        assert prefs == expected

        visible = geom.get_visible_headers()
        assert visible == ["x", "y", "height"]

    def test_display_preferences_point(self) -> None:
        """Test display preferences for point geometry."""
        coords = np.array([[1.0, 2.0]])
        geom = GeometryResult("Test Point", KindShape.POINT, coords)

        # Test default
        prefs = geom.get_display_preferences()
        expected = {"x": True, "y": True}
        assert prefs == expected

        # Test setting preferences
        geom.set_display_preferences({"x": False, "y": True})
        prefs = geom.get_display_preferences()
        expected = {"x": False, "y": True}
        assert prefs == expected

    def test_display_preferences_circle(self) -> None:
        """Test display preferences for circle geometry."""
        coords = np.array([[1.0, 2.0, 3.0]])
        geom = GeometryResult("Test Circle", KindShape.CIRCLE, coords)

        # Test default
        prefs = geom.get_display_preferences()
        expected = {"x": True, "y": True, "r": True}
        assert prefs == expected

        # Test setting preferences
        geom.set_display_preferences({"x": True, "y": False, "r": True})
        prefs = geom.get_display_preferences()
        expected = {"x": True, "y": False, "r": True}
        assert prefs == expected

    def test_display_preferences_invalid_coords(self) -> None:
        """Test setting display preferences with invalid coordinate names."""
        coords = np.array([[0.0, 0.0, 10.0, 5.0]])
        geom = GeometryResult("Test Rectangle", KindShape.RECTANGLE, coords)

        # Should ignore invalid coordinates
        geom.set_display_preferences(
            {"x": False, "invalid_coord": False, "height": True}
        )

        prefs = geom.get_display_preferences()
        expected = {"x": False, "y": True, "width": True, "height": True}
        assert prefs == expected

    def test_display_preferences_all_hidden(self) -> None:
        """Test hiding all coordinates."""
        coords = np.array([[0.0, 0.0, 10.0, 5.0]])
        geom = GeometryResult("Test Rectangle", KindShape.RECTANGLE, coords)

        geom.set_display_preferences(
            {"x": False, "y": False, "width": False, "height": False}
        )

        visible = geom.get_visible_headers()
        assert visible == []

    def test_to_dataframe_visible_only_default(self) -> None:
        """Test to_dataframe with visible_only=False (default)."""
        coords = np.array([[0.0, 0.0, 10.0, 5.0]])
        geom = GeometryResult("Test Rectangle", KindShape.RECTANGLE, coords)

        df = geom.to_dataframe(visible_only=False)
        assert list(df.columns) == ["x", "y", "width", "height"]

    def test_to_dataframe_visible_only_true(self) -> None:
        """Test to_dataframe with visible_only=True."""
        coords = np.array([[0.0, 0.0, 10.0, 5.0]])
        geom = GeometryResult("Test Rectangle", KindShape.RECTANGLE, coords)

        geom.set_display_preferences(
            {"x": True, "y": False, "width": True, "height": False}
        )

        df = geom.to_dataframe(visible_only=True)
        assert list(df.columns) == ["x", "width"]

    def test_to_dataframe_visible_only_all_hidden(self) -> None:
        """Test to_dataframe with all coordinates hidden."""
        coords = np.array([[0.0, 0.0, 10.0, 5.0]])
        geom = GeometryResult("Test Rectangle", KindShape.RECTANGLE, coords)

        geom.set_display_preferences(
            {"x": False, "y": False, "width": False, "height": False}
        )

        df = geom.to_dataframe(visible_only=True)
        # When all coordinates are hidden, should return original dataframe
        assert list(df.columns) == ["x", "y", "width", "height"]

    def test_display_preferences_persistence(self) -> None:
        """Test that display preferences persist through serialization."""
        coords = np.array([[0.0, 0.0, 10.0, 5.0]])
        geom = GeometryResult("Test Rectangle", KindShape.RECTANGLE, coords)

        geom.set_display_preferences(
            {"x": True, "y": False, "width": True, "height": False}
        )

        # Serialize and deserialize
        serialized = geom.to_dict()
        restored = GeometryResult.from_dict(serialized)

        # Check preferences are preserved
        prefs = restored.get_display_preferences()
        expected = {"x": True, "y": False, "width": True, "height": False}
        assert prefs == expected


class TestUtilityFunctions:
    """Test class for utility functions (calc_table_from_data, concat_*, filter_*)."""

    def test_calc_table_from_data_no_roi(self) -> None:
        """Test calc_table_from_data without ROI masks."""
        data = np.array([1, 2, 3, 4, 5])
        labeledfuncs = {"mean": np.mean, "std": np.std}
        result = calc_table_from_data("Stats", data, labeledfuncs)

        assert result.title == "Stats"
        assert list(result.headers) == ["mean", "std"]
        assert len(result.data) == 1
        assert result.roi_indices == [NO_ROI]
        assert result.data[0][0] == 3.0  # mean
        assert abs(result.data[0][1] - np.std(data)) < 1e-10  # std

    def test_calc_table_from_data_with_roi(self) -> None:
        """Test calc_table_from_data with ROI masks."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        roi_masks = [
            np.array(
                [[True, True, False], [False, False, False], [False, False, False]]
            ),
            np.array(
                [[False, False, False], [False, True, True], [False, False, False]]
            ),
        ]
        labeledfuncs = {"mean": np.mean, "sum": np.sum}
        result = calc_table_from_data("ROI Stats", data, labeledfuncs, roi_masks)

        assert result.title == "ROI Stats"
        assert list(result.headers) == ["mean", "sum"]
        assert len(result.data) == 2
        assert result.roi_indices == [0, 1]
        # ROI 0: mean of [1, 2] = 1.5, sum = 3
        assert result.data[0][0] == 1.5
        assert result.data[0][1] == 3.0
        # ROI 1: mean of [5, 6] = 5.5, sum = 11
        assert result.data[1][0] == 5.5
        assert result.data[1][1] == 11.0

    def test_concat_tables_empty(self) -> None:
        """Test concat_tables with empty list."""
        result = concat_tables("Empty", [])
        assert result.title == "Empty"
        assert result.headers == []
        assert not result.data

    def test_concat_tables_single(self) -> None:
        """Test concat_tables with single table."""
        table = TableResult("Single", headers=["col1"], data=[[1.0]])
        result = concat_tables("Concat", [table])
        assert result.title == "Concat"
        assert list(result.headers) == ["col1"]
        assert result.data == [[1.0]]

    def test_concat_tables_multiple(self) -> None:
        """Test concat_tables with multiple tables."""
        table1 = TableResult(
            "Table1",
            headers=["col1", "col2"],
            data=[[1.0, 2.0]],
            roi_indices=[0],
        )
        table2 = TableResult(
            "Table2",
            headers=["col1", "col2"],
            data=[[3.0, 4.0], [5.0, 6.0]],
            roi_indices=[1, 2],
        )
        result = concat_tables("Combined", [table1, table2])

        assert result.title == "Combined"
        assert list(result.headers) == ["col1", "col2"]
        assert result.data == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        assert result.roi_indices == [0, 1, 2]

    def test_concat_tables_mismatched_names(self) -> None:
        """Test concat_tables with mismatched column names."""
        table1 = TableResult("Table1", headers=["col1"], data=[[1.0]])
        table2 = TableResult("Table2", headers=["col2"], data=[[2.0]])
        with pytest.raises(
            ValueError, match="All TableResult objects must share the same names"
        ):
            concat_tables("Mismatched", [table1, table2])

    def test_filter_table_by_roi_no_roi_indices(self) -> None:
        """Test filter_table_by_roi with no ROI indices."""
        table = TableResult("Test", headers=["col1"], data=[[1.0], [2.0]])

        # Filter for None should keep all
        result_none = filter_table_by_roi(table, None)
        assert result_none.data == [[1.0], [2.0]]

        # Filter for specific ROI should return empty
        result_roi = filter_table_by_roi(table, 0)
        assert not result_roi.data

    def test_filter_table_by_roi_with_roi_indices(self) -> None:
        """Test filter_table_by_roi with ROI indices."""
        table = TableResult(
            "Test",
            headers=["col1"],
            data=[[1.0], [2.0], [3.0]],
            roi_indices=[NO_ROI, 0, 1],
        )

        # Filter for NO_ROI
        result_none = filter_table_by_roi(table, None)
        assert result_none.data == [[1.0]]
        assert result_none.roi_indices == [NO_ROI]

        # Filter for ROI 0
        result_roi_0 = filter_table_by_roi(table, 0)
        assert result_roi_0.data == [[2.0]]
        assert result_roi_0.roi_indices == [0]

    def test_concat_geometries_empty(self) -> None:
        """Test concat_geometries with empty list raises ValueError."""
        with pytest.raises(
            ValueError, match="Cannot concatenate empty sequence of GeometryResult"
        ):
            concat_geometries("Empty", [])

    def test_concat_geometries_single(self) -> None:
        """Test concat_geometries with single geometry."""
        coords = np.array([[1.0, 2.0]])
        geom = GeometryResult("Single", KindShape.POINT, coords, func_name="test_func")
        result = concat_geometries("Concat", [geom])
        assert result.title == "Concat"
        assert result.kind == KindShape.POINT
        np.testing.assert_array_equal(result.coords, coords)

    def test_concat_geometries_multiple(self) -> None:
        """Test concat_geometries with multiple geometries."""
        coords1 = np.array([[1.0, 2.0]])
        coords2 = np.array([[3.0, 4.0], [5.0, 6.0]])
        geom1 = GeometryResult(
            "Geom1", KindShape.POINT, coords1, np.array([0]), func_name="test_func"
        )
        geom2 = GeometryResult(
            "Geom2", KindShape.POINT, coords2, np.array([1, 2]), func_name="test_func"
        )
        result = concat_geometries("Combined", [geom1, geom2])

        assert result.title == "Combined"
        assert result.kind == KindShape.POINT
        expected_coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(result.coords, expected_coords)
        np.testing.assert_array_equal(result.roi_indices, np.array([0, 1, 2]))

    def test_concat_geometries_different_widths(self) -> None:
        """Test concat_geometries with different coordinate widths."""
        # Two polygons with different number of vertices
        coords1 = np.array([[1.0, 2.0, 3.0, 4.0]])  # 2 vertices
        coords2 = np.array([[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])  # 3 vertices
        geom1 = GeometryResult(
            "Poly1", KindShape.POLYGON, coords1, func_name="test_func"
        )
        geom2 = GeometryResult(
            "Poly2", KindShape.POLYGON, coords2, func_name="test_func"
        )

        result = concat_geometries("Mixed", [geom1, geom2])
        expected_coords = np.array(
            [[1.0, 2.0, 3.0, 4.0, np.nan, np.nan], [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
        )
        np.testing.assert_array_equal(result.coords[:, :4], expected_coords[:, :4])
        assert np.isnan(result.coords[0, 4])
        assert np.isnan(result.coords[0, 5])
        assert result.coords[1, 4] == 9.0
        assert result.coords[1, 5] == 10.0

    def test_concat_geometries_mismatched_kinds(self) -> None:
        """Test concat_geometries with mismatched kinds."""
        coords1 = np.array([[1.0, 2.0]])
        coords2 = np.array([[3.0, 4.0, 5.0]])
        geom1 = GeometryResult("Point", KindShape.POINT, coords1)
        geom2 = GeometryResult("Circle", KindShape.CIRCLE, coords2)
        with pytest.raises(
            ValueError, match="All GeometryResult objects must share the same kind"
        ):
            concat_geometries("Mismatched", [geom1, geom2])

    def test_filter_geometry_by_roi_no_roi_indices(self) -> None:
        """Test filter_geometry_by_roi with no ROI indices."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        geom = GeometryResult("Test", KindShape.POINT, coords)

        # Filter for None should keep all
        result_none = filter_geometry_by_roi(geom, None)
        np.testing.assert_array_equal(result_none.coords, coords)

        # Filter for specific ROI should return empty
        result_roi = filter_geometry_by_roi(geom, 0)
        assert result_roi.coords.shape == (0, 2)

    def test_filter_geometry_by_roi_with_roi_indices(self) -> None:
        """Test filter_geometry_by_roi with ROI indices."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        roi_indices = np.array([NO_ROI, 0, 1])
        geom = GeometryResult("Test", KindShape.POINT, coords, roi_indices)

        # Filter for NO_ROI
        result_none = filter_geometry_by_roi(geom, None)
        expected_coords = np.array([[1.0, 2.0]])
        np.testing.assert_array_equal(result_none.coords, expected_coords)
        np.testing.assert_array_equal(result_none.roi_indices, np.array([NO_ROI]))

        # Filter for ROI 0
        result_roi_0 = filter_geometry_by_roi(geom, 0)
        expected_coords = np.array([[3.0, 4.0]])
        np.testing.assert_array_equal(result_roi_0.coords, expected_coords)
        np.testing.assert_array_equal(result_roi_0.roi_indices, np.array([0]))


class TestGeometryResultValueProperty:
    """Test class for GeometryResult.value property."""

    def test_value_property_point(self) -> None:
        """Test .value property for POINT shape returns (x, y) tuple."""
        coords = np.array([[3.5, 5.0]])
        result = GeometryResult("Test Point", KindShape.POINT, coords)
        x, y = result.value
        assert isinstance(x, float), "X should be a float"
        assert isinstance(y, float), "Y should be a float"
        assert abs(x - 3.5) < 1e-10, f"X should be 3.5, got {x}"
        assert abs(y - 5.0) < 1e-10, f"Y should be 5.0, got {y}"

    def test_value_property_marker(self) -> None:
        """Test .value property for MARKER shape returns (x, y) tuple."""
        coords = np.array([[2.0, 7.0]])
        result = GeometryResult("Test Marker", KindShape.MARKER, coords)
        x, y = result.value
        assert isinstance(x, float), "X should be a float"
        assert isinstance(y, float), "Y should be a float"
        assert abs(x - 2.0) < 1e-10, f"X should be 2.0, got {x}"
        assert abs(y - 7.0) < 1e-10, f"Y should be 7.0, got {y}"

    def test_value_property_segment(self) -> None:
        """Test .value property for SEGMENT shape returns length."""
        # 3-4-5 triangle: segment from (0,0) to (3,4) has length 5
        coords = np.array([[0.0, 0.0, 3.0, 4.0]])
        result = GeometryResult("Test Segment", KindShape.SEGMENT, coords)
        length = result.value
        assert isinstance(length, float), "Length should be a float"
        assert abs(length - 5.0) < 1e-10, f"Length should be 5.0, got {length}"

    def test_value_property_segment_horizontal(self) -> None:
        """Test .value property for horizontal SEGMENT."""
        coords = np.array([[1.0, 2.0, 6.0, 2.0]])
        result = GeometryResult("Horizontal Segment", KindShape.SEGMENT, coords)
        length = result.value
        assert abs(length - 5.0) < 1e-10, f"Length should be 5.0, got {length}"

    def test_value_property_segment_vertical(self) -> None:
        """Test .value property for vertical SEGMENT."""
        coords = np.array([[3.0, 1.0, 3.0, 8.0]])
        result = GeometryResult("Vertical Segment", KindShape.SEGMENT, coords)
        length = result.value
        assert abs(length - 7.0) < 1e-10, f"Length should be 7.0, got {length}"

    def test_value_property_unsupported_shape(self) -> None:
        """Test .value property raises error for unsupported shapes."""
        # Test CIRCLE
        coords = np.array([[0.0, 0.0, 1.0]])
        result = GeometryResult("Test Circle", KindShape.CIRCLE, coords)
        with pytest.raises(ValueError, match="value property only valid for"):
            _ = result.value

        # Test ELLIPSE
        coords = np.array([[0.0, 0.0, 2.0, 1.0, 0.0]])
        result = GeometryResult("Test Ellipse", KindShape.ELLIPSE, coords)
        with pytest.raises(ValueError, match="value property only valid for"):
            _ = result.value

        # Test RECTANGLE
        coords = np.array([[0.0, 0.0, 2.0, 3.0]])
        result = GeometryResult("Test Rectangle", KindShape.RECTANGLE, coords)
        with pytest.raises(ValueError, match="value property only valid for"):
            _ = result.value

    def test_value_property_multiple_rows(self) -> None:
        """Test .value property raises error for multiple rows."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = GeometryResult("Multiple Points", KindShape.POINT, coords)
        with pytest.raises(ValueError, match="single-row results"):
            _ = result.value

    def test_value_property_consistency(self) -> None:
        """Test .value property is consistent with direct coord access."""
        # For POINT
        coords = np.array([[1.5, 2.5]])
        result = GeometryResult("Point", KindShape.POINT, coords)
        x_val, y_val = result.value
        assert x_val == result.coords[0, 0]
        assert y_val == result.coords[0, 1]

        # For MARKER
        coords = np.array([[3.5, 4.5]])
        result = GeometryResult("Marker", KindShape.MARKER, coords)
        x_val, y_val = result.value
        assert x_val == result.coords[0, 0]
        assert y_val == result.coords[0, 1]

        # For SEGMENT
        coords = np.array([[0.0, 0.0, 3.0, 4.0]])
        result = GeometryResult("Segment", KindShape.SEGMENT, coords)
        length = result.value
        expected_length = result.segments_lengths()[0]
        assert abs(length - expected_length) < 1e-10
