# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for :mod:`sigima.io.signal.funcs`.

Exercises label/unit parsing helpers, CSV reading branches (delimiters,
header detection, comment handling, datetime, metadata columns) and CSV
writing variants.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import datetime
import io
import os
import tempfile

import numpy as np
import pytest

from sigima.io.signal import funcs as iofuncs

# ===========================================================================
# Helpers: normalize_units, get_labels_units_from_dataframe
# ===========================================================================


def test_normalize_units_middle_dot() -> None:
    """Unicode middle-dot characters used as multiplication separators in
    physical units are normalised to plain ASCII ``*``."""
    assert iofuncs.normalize_units("kg·m") == "kg*m"
    assert iofuncs.normalize_units("kg⋅m") == "kg*m"


def test_normalize_units_spaces_around_operators() -> None:
    """Whitespace surrounding ``*`` and ``/`` operators is collapsed so
    units are stored in canonical form."""
    assert iofuncs.normalize_units("kg * m / s") == "kg*m/s"


def test_normalize_units_spaces_between_units() -> None:
    """Whitespace between adjacent unit symbols is interpreted as implicit
    multiplication and replaced by ``*``."""
    assert iofuncs.normalize_units("kg m") == "kg*m"


def test_normalize_units_strip_extra_spaces() -> None:
    """Leading/trailing whitespace and runs of inner spaces are all
    collapsed in the normalised representation."""
    assert iofuncs.normalize_units("  kg   m  ") == "kg*m"


# ===========================================================================
# CSV reading: cover branches via temp files
# ===========================================================================


def _write_temp(content: str, suffix: str = ".csv") -> str:
    """Write ``content`` to a fresh UTF-8 temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def test_read_csv_with_data_header_marker() -> None:
    """The ``##Start Data`` marker line lets the parser skip arbitrary
    free-text preamble and start consuming numeric rows from the next line."""
    content = "instrument: foo\nuser: bar\n##Start Data\n1,10\n2,20\n3,30\n"
    path = _write_temp(content)
    try:
        data = iofuncs.read_csv(path)
        assert data.xydata.shape[0] >= 3
    finally:
        os.unlink(path)


def test_read_csv_with_header_and_units_in_columns() -> None:
    """Header cells of the form ``Name (unit)`` are split into a label and
    a unit, populating ``xlabel``/``xunit``/``ylabels``/``yunits``."""
    content = "Time (s),Voltage (V)\n0,1.0\n1,2.0\n2,3.0\n"
    path = _write_temp(content)
    try:
        data = iofuncs.read_csv(path)
        assert data.xlabel == "Time"
        assert data.xunit == "s"
        assert data.ylabels and "Voltage" in data.ylabels[0]
        assert data.yunits and data.yunits[0] == "V"
    finally:
        os.unlink(path)


def test_read_csv_with_comment_lines() -> None:
    """Lines starting with ``#`` are treated as comments and skipped
    while still allowing a regular header row to follow."""
    content = "# This is a comment\n# Another comment\nTime,Value\n0,1\n1,2\n2,3\n"
    path = _write_temp(content)
    try:
        data = iofuncs.read_csv(path)
        assert data.xydata.shape[0] >= 3
    finally:
        os.unlink(path)


def test_read_csv_with_constant_metadata_column() -> None:
    """A column whose value is constant across all rows (e.g. a serial
    number) is detected and stored in ``column_metadata`` rather than as
    a Y channel."""
    content = (
        "Time,Voltage,Serial\n0,1.0,12345\n1,2.0,12345\n2,3.0,12345\n3,4.0,12345\n"
    )
    path = _write_temp(content)
    try:
        data = iofuncs.read_csv(path)
        # Serial column should be detected as metadata
        assert data.column_metadata is not None
        assert "Serial" in data.column_metadata
        assert data.column_metadata["Serial"] == 12345
    finally:
        os.unlink(path)


def test_read_csv_semicolon_delimiter() -> None:
    """The CSV sniffer correctly detects ``;`` as the delimiter when no
    commas are present."""
    content = "Time;Voltage\n0;1.5\n1;2.5\n2;3.5\n3;4.5\n"
    path = _write_temp(content)
    try:
        data = iofuncs.read_csv(path)
        assert data.xydata.shape[0] >= 3
    finally:
        os.unlink(path)


def test_read_csv_no_header_only_numeric() -> None:
    """Header-less files (purely numeric content) are loaded directly as
    data without losing any row."""
    content = "0,1.0\n1,2.0\n2,3.0\n3,4.0\n"
    path = _write_temp(content)
    try:
        data = iofuncs.read_csv(path)
        assert data.xydata.shape[0] == 4
    finally:
        os.unlink(path)


def test_read_csv_with_datetime_column() -> None:
    """A first column that parses as date/time is converted to numeric
    seconds and the absolute datetime is preserved in ``datetime_metadata``."""
    content = (
        "Time,Value\n"
        "2024-01-01 10:00:00,1.0\n"
        "2024-01-01 10:00:01,2.0\n"
        "2024-01-01 10:00:02,3.0\n"
        "2024-01-01 10:00:03,4.0\n"
        "2024-01-01 10:00:04,5.0\n"
    )
    path = _write_temp(content)
    try:
        data = iofuncs.read_csv(path)
        assert data.datetime_metadata is not None
    finally:
        os.unlink(path)


def test_read_csv_unparseable_file_raises() -> None:
    """Files containing no parseable numeric rows raise ``ValueError``
    rather than returning empty data."""
    content = "this is not csv data at all\nthis really isn't\nreally\n"
    path = _write_temp(content)
    try:
        with pytest.raises(ValueError):
            iofuncs.read_csv(path)
    finally:
        os.unlink(path)


# ===========================================================================
# read_csv_by_chunks - text stream variant
# ===========================================================================


def test_read_csv_by_chunks_text_stream_requires_nlines() -> None:
    """When called on a text stream the chunked reader cannot infer the
    chunk size, so an explicit ``nlines`` argument is mandatory."""
    text = "1,2\n3,4\n5,6\n"
    stream = io.StringIO(text)
    with pytest.raises(ValueError):
        iofuncs.read_csv_by_chunks(stream)


def test_read_csv_by_chunks_text_stream_with_nlines() -> None:
    """With explicit ``nlines`` the chunked reader returns a dataframe of
    the requested length from a text stream."""
    text = "1,2\n3,4\n5,6\n"
    stream = io.StringIO(text)
    df = iofuncs.read_csv_by_chunks(stream, nlines=3, header=None)
    assert df.shape[0] == 3


# ===========================================================================
# write_csv - cover ylabels/header branches
# ===========================================================================


def test_write_csv_with_single_y_no_label() -> None:
    """When ``ylabels`` is empty, ``write_csv`` falls back to a generic
    ``Y`` header so the output is still self-describing."""
    path = _write_temp("", suffix=".csv")
    try:
        xydata = np.array([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]])
        iofuncs.write_csv(
            path,
            xydata,
            xlabel="Time",
            xunit="s",
            ylabels=[""],
            yunits=None,
            header=None,
        )
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Y" in content
    finally:
        os.unlink(path)


def test_write_csv_with_multiple_y_and_units() -> None:
    """With several Y columns plus matching units, ``write_csv`` emits
    headers of the form ``label (unit)`` for each Y column."""
    path = _write_temp("", suffix=".csv")
    try:
        xydata = np.array([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
        iofuncs.write_csv(
            path,
            xydata,
            xlabel="Time",
            xunit="s",
            ylabels=["A", "B"],
            yunits=["V", "mA"],
            header=None,
        )
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "A (V)" in content
        assert "B (mA)" in content
    finally:
        os.unlink(path)


def test_write_csv_with_header_text() -> None:
    """A user-supplied ``header`` string is written verbatim at the top
    of the file before any column header."""
    path = _write_temp("", suffix=".csv")
    try:
        xydata = np.array([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]])
        iofuncs.write_csv(
            path,
            xydata,
            xlabel="X",
            xunit=None,
            ylabels=["Y"],
            yunits=None,
            header="# Custom header line\n",
        )
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("# Custom header line")
    finally:
        os.unlink(path)


# ===========================================================================
# MCAFile - basic decode and section parsing
# ===========================================================================


def test_mca_infer_string_value_int() -> None:
    """Strings that look like integers are returned as Python ``int``
    (rather than ``str`` or ``float``) by the MCA value sniffer."""
    out = iofuncs.MCAFile._infer_string_value("42")  # pylint: disable=protected-access
    assert out == 42 and isinstance(out, int)


def test_mca_infer_string_value_float() -> None:
    """Strings with a decimal point are parsed as ``float`` by the MCA
    value sniffer."""
    out = iofuncs.MCAFile._infer_string_value("3.14")  # pylint: disable=protected-access
    assert out == pytest.approx(3.14)


def test_mca_infer_string_value_datetime() -> None:
    """Strings matching the MCA date format are parsed to a real
    ``datetime`` object."""
    out = iofuncs.MCAFile._infer_string_value(  # pylint: disable=protected-access
        "01/02/2024 12:34:56"
    )
    assert isinstance(out, datetime.datetime)


def test_mca_infer_string_value_string() -> None:
    """Plain text values that match no other type are returned unchanged
    as ``str``."""
    out = iofuncs.MCAFile._infer_string_value("hello")  # pylint: disable=protected-access
    assert out == "hello"


def test_mca_read_section_missing_returns_none() -> None:
    """Querying a missing section returns ``None`` rather than raising,
    so callers can use simple truthiness checks."""
    mca = iofuncs.MCAFile("dummy.mca")
    mca.raw_data = "no sections here"
    assert mca._read_section("MISSING") is None  # pylint: disable=protected-access


def test_mca_extract_metadata_empty_section() -> None:
    """Extracting metadata from a non-existent section yields an empty
    dict (parallel to ``_read_section`` returning ``None``)."""
    mca = iofuncs.MCAFile("dummy.mca")
    mca.raw_data = "no sections here"
    # pylint: disable=protected-access
    assert not mca._extract_metadata_from_section("MISSING")
