# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O signal functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import datetime
import re
import warnings
from dataclasses import dataclass
from typing import TextIO

import numpy as np
import pandas as pd
import scipy.interpolate

from sigima.io.common.textreader import count_lines, read_first_n_lines
from sigima.objects.signal.constants import (
    DATETIME_X_FORMAT_KEY,
    DATETIME_X_KEY,
    DEFAULT_DATETIME_FORMAT,
)
from sigima.worker import CallbackWorkerProtocol


def get_labels_units_from_dataframe(
    df: pd.DataFrame,
) -> tuple[str, list[str], str, list[str]]:
    """Get labels and units from a DataFrame.

    Args:
        df: DataFrame

    Returns:
        Tuple (xlabel, ylabels, xunit, yunits)
    """
    # Reading X,Y labels
    xlabel = str(df.columns[0])
    ylabels = [str(col) for col in df.columns[1:]]

    # Retrieving units from labels
    xunit = ""
    yunits = [""] * len(ylabels)
    pattern = r"([\S ]*) \(([\S]*)\)"
    match = re.match(pattern, xlabel)
    if match is not None:
        xlabel, xunit = match.groups()
    for i, ylabel in enumerate(ylabels):
        match = re.match(pattern, ylabel)
        if match is not None:
            ylabels[i], yunits[i] = match.groups()

    return xlabel, ylabels, xunit, yunits


def read_csv_by_chunks(
    fname_or_fileobj: str | TextIO,
    nlines: int | None = None,
    worker: CallbackWorkerProtocol | None = None,
    decimal: str = ".",
    delimiter: str | None = None,
    header: int | None = "infer",
    skiprows: int | None = None,
    nrows: int | None = None,
    comment: str | None = None,
    chunksize: int = 1000,
) -> pd.DataFrame:
    """Read CSV data with primitive options, using pandas read_csv function defaults,
    and reading data in chunks, using the iterator interface.

    Args:
        fname_or_fileobj: CSV file name or text stream object
        nlines: Number of lines contained in file (this argument is mandatory if
         `fname_or_fileobj` is a text stream object: counting line numbers from a
         text stream is not efficient, especially if one already has access to the
         initial text content from which the text stream was made)
        worker: Callback worker object
        decimal: Decimal character
        delimiter: Delimiter
        header: Header line
        skiprows: Skip rows
        nrows: Number of rows to read
        comment: Comment character
        chunksize: Chunk size

    Returns:
        DataFrame
    """
    if isinstance(fname_or_fileobj, str):
        nlines = count_lines(fname_or_fileobj)
    elif nlines is None:
        raise ValueError("Argument `nlines` must be passed for text streams")
    # Read data in chunks, and concatenate them at the end, thus allowing to call the
    # progress callback function at each chunk read and to return an intermediate result
    # if the operation is canceled.
    chunks = []
    for chunk in pd.read_csv(
        fname_or_fileobj,
        decimal=decimal,
        delimiter=delimiter,
        header=header,
        skiprows=skiprows,
        nrows=nrows,
        comment=comment,
        chunksize=chunksize,
        encoding_errors="ignore",
    ):
        chunks.append(chunk)
        # Compute the progression based on the number of lines read so far
        if worker is not None:
            worker.set_progress(sum(len(chunk) for chunk in chunks) / nlines)
            if worker.was_canceled():
                break
    return pd.concat(chunks)


DATA_HEADERS = [
    "#DATA",  # Generic
    "START_OF_DATA",  # Various logging devices
    ">>>>>Begin Spectral Data<<<<<",  # Ocean Optics
    ">>>Begin Data<<<",  # Ocean Optics (alternative)
    ">>>Begin Spectrum Data<<<",  # Avantes
    "# Data Start",  # Andor, Horiba, Mass Spectrometry (Agilent, Thermo Fisher, ...)
    ">DATA START<",  # Mass Spectrometry, Chromatography
    "BEGIN DATA",  # Mass Spectrometry, Chromatography
    "<Data>",  # Mass Spectrometry (XML-based)
    "##Start Data",  # Bruker (X-ray, Raman, FTIR)
    "[DataStart]",  # PerkinElmer (FTIR, UV-Vis)
    "BEGIN SPECTRUM",  # PerkinElmer
    "%% Data Start %%",  # LabVIEW, MATLAB
    "---Begin Data---",  # General scientific instruments
    "===DATA START===",  # Industrial/scientific devices
]


def _read_df_without_header(
    filename: str, skiprows: int | None = None
) -> tuple[pd.DataFrame | None, str, str]:
    """Try to read a CSV file without header, testing various delimiters and decimal.

    Args:
        filename: CSV file name
        skiprows: Number of rows to skip at the beginning of the file

    Returns:
        A tuple (DataFrame if successful, None otherwise, decimal used, delimiter used)
    """
    for decimal in (".", ","):
        for delimiter in (",", ";", r"\s+"):
            try:
                df = pd.read_csv(
                    filename,
                    decimal=decimal,
                    delimiter=delimiter,
                    header=None,
                    comment="#",
                    nrows=1000,  # Read only the first 1000 lines
                    encoding_errors="ignore",
                    skiprows=skiprows,
                    dtype=float,  # Keep dtype to validate delimiter detection
                )
                break
            except (pd.errors.ParserError, ValueError):
                df = None
        if df is not None:
            break
    return df, decimal, delimiter


def _read_df_with_header(filename: str) -> tuple[pd.DataFrame | None, str, str]:
    """Try to read a CSV file with header, testing various delimiters and decimal.

    Args:
        filename: CSV file name

    Returns:
        A tuple (DataFrame if successful, None otherwise, decimal used, delimiter used)
    """
    for decimal in (".", ","):
        for delimiter in (",", ";", r"\s+"):
            # Headers are generally in the first 10 lines, so we try to skip the
            # minimum number of lines before reading the data:
            for skiprows in range(20):
                try:
                    df = pd.read_csv(
                        filename,
                        decimal=decimal,
                        delimiter=delimiter,
                        skiprows=skiprows,
                        comment="#",
                        nrows=1000,  # Read only the first 1000 lines
                        encoding_errors="ignore",
                    )
                    # Validate: CSV should have at least 2 columns (x and y)
                    # If only 1 column, likely wrong delimiter
                    if df.shape[1] >= 2:
                        break  # Good delimiter found
                    df = None  # Try next delimiter
                except (pd.errors.ParserError, ValueError):
                    df = None
            if df is not None:
                break
        if df is not None:
            break
    return df, decimal, delimiter


def _detect_metadata_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Detect columns containing constant/single-value metadata.

    Columns with a single unique value (excluding NaN) across all rows are treated
    as metadata rather than data columns. These are typically instrument serial numbers,
    experiment IDs, or other constant identifiers.

    Args:
        df: Input DataFrame

    Returns:
        A tuple (DataFrame with metadata columns removed,
        dict of metadata key-value pairs)
    """
    metadata = {}
    cols_to_drop = []

    # Start from column 1 (skip X column) and check for constant-value columns
    for col_idx in range(1, df.shape[1]):
        col_data = df.iloc[:, col_idx]
        col_name = df.columns[col_idx]

        # Get unique non-NaN values
        unique_values = col_data.dropna().unique()

        # If column has exactly one unique value (excluding NaN), it's metadata
        if len(unique_values) == 1:
            # Store the metadata
            value = unique_values[0]
            # Try to convert to appropriate type (keep as string if necessary)
            try:
                # Try int first
                if float(value).is_integer():
                    value = int(float(value))
                else:
                    value = float(value)
            except (ValueError, TypeError):
                # Keep as string
                value = str(value)

            metadata[str(col_name)] = value
            cols_to_drop.append(col_name)  # Store column name, not index

    # Drop metadata columns from DataFrame
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df, metadata


def _detect_datetime_col(df: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    """Try to detect the presence of a datetime column in a DataFrame.

    Detect if the first or second column contains datetime values, and convert it to
    float timestamps if so.

    Args:
        df: Input DataFrame

    Returns:
        A tuple (DataFrame with datetime column converted, datetime metadata dict)
    """
    datetime_col_idx = None

    for col_idx in [0, 1]:  # Check first two columns
        col_data = df.iloc[:, col_idx]

        # Skip numeric columns - datetime columns in CSV are loaded as object (string)
        # dtype, not as float/int. Numeric values (like frequencies in Hz) would be
        # wrongly interpreted as nanoseconds since Unix epoch by pd.to_datetime.
        if pd.api.types.is_numeric_dtype(col_data):
            continue

        # Try to convert to datetime
        try:
            # Attempt to parse as datetime
            # Note: format="mixed" was causing failures in some pandas versions,
            # so we use warnings filter to suppress the UserWarning instead
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format",
                    category=UserWarning,
                )
                datetime_series = pd.to_datetime(col_data, errors="coerce")
            # Check if most values were successfully converted (>90%)
            valid_ratio = datetime_series.notna().sum() / len(datetime_series)

            # Skip if conversion ratio is too low
            if valid_ratio <= 0.9:
                continue

            # Check if values have reasonable variation and are actual dates
            unique_dates = datetime_series.dropna().nunique()
            if unique_dates <= 1:
                continue

            # Check date range - should be reasonable dates, not epoch times
            min_date = datetime_series.min()
            max_date = datetime_series.max()
            # Dates should be after 1900 and the range should be > 1 sec
            valid_datetime = (
                min_date.year >= 1900 and (max_date - min_date).total_seconds() > 1.0
            )

            if valid_datetime:
                # This is a datetime column!
                datetime_col_idx = col_idx
                break
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            # Not a datetime column, continue checking
            pass

    datetime_metadata = None

    if datetime_col_idx is not None:
        # Convert datetime column to float timestamps
        col_data = df.iloc[:, datetime_col_idx]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Could not infer format", category=UserWarning
            )
            datetime_series = pd.to_datetime(col_data, errors="coerce")
        x_float = datetime_series.astype(np.int64) / 1e9
        # Store datetime metadata (unit will be stored in xunit attribute)
        datetime_metadata = {
            DATETIME_X_KEY: True,
            DATETIME_X_FORMAT_KEY: DEFAULT_DATETIME_FORMAT,
        }

        # If datetime is in column 1 and column 0 looks like an index, drop column 0
        if datetime_col_idx == 1:
            try:
                # Try to convert first column to int - if sequential,
                # it's likely an index column
                first_col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                if first_col.notna().all():
                    # Check if it's a sequential index (1, 2, 3, ...)
                    diffs = first_col.diff().dropna()
                    if (diffs == 1).sum() / len(diffs) > 0.9:
                        # Drop the index column
                        df = df.iloc[:, 1:].copy()
                        datetime_col_idx = 0  # Now datetime is in position 0
            except (ValueError, TypeError):
                pass

        # Replace datetime column with float timestamps
        df.iloc[:, datetime_col_idx] = x_float

    return df, datetime_metadata


@dataclass
class CSVData:
    """Data structure for CSV file contents.

    This dataclass encapsulates all the data extracted from a CSV file,
    including the actual XY data, labels, units, and metadata.

    Attributes:
        xydata: Numpy array containing X and Y data columns
        xlabel: Label for the X axis
        xunit: Unit for the X axis
        ylabels: List of labels for Y columns
        yunits: List of units for Y columns
        header: Optional header text from the CSV file
        datetime_metadata: Optional dict with datetime conversion info
        column_metadata: Optional dict with constant-value column metadata
    """

    xydata: np.ndarray
    xlabel: str | None = None
    xunit: str | None = None
    ylabels: list[str] | None = None
    yunits: list[str] | None = None
    header: str | None = None
    datetime_metadata: dict | None = None
    column_metadata: dict | None = None


def read_csv(
    filename: str,
    worker: CallbackWorkerProtocol | None = None,
) -> CSVData:
    """Read CSV data and return parsed components including datetime metadata.

    Args:
        filename: CSV file name
        worker: Callback worker object

    Returns:
        CSVData object containing all parsed CSV components
    """
    xydata, xlabel, xunit, ylabels, yunits = None, None, None, None, None
    header, datetime_metadata, column_metadata = None, None, {}

    # The first attempt is to read the CSV file assuming it has no header because it
    # won't raise an error if the first line is data. If it fails, we try to read it
    # with a header, and if it fails again, we try to skip some lines before reading
    # the data.

    skiprows = None

    # Begin by reading the first 100 lines to search for a line that could mark the
    # beginning of the data after it (e.g., a line '#DATA' or other).
    first_100_lines = read_first_n_lines(filename, n=100).splitlines()
    for data_header in DATA_HEADERS:
        if data_header in first_100_lines:
            # Skip the lines before the data header
            skiprows = first_100_lines.index(data_header) + 1
            break

    # First attempt: no header (try to read with different delimiters)
    read_without_header = True
    df, decimal, delimiter = _read_df_without_header(filename, skiprows=skiprows)

    # Second attempt: with header
    if df is None:
        df, decimal, delimiter = _read_df_with_header(filename)

        if df is None:
            raise ValueError("Unable to read CSV file (format not supported)")

        # At this stage, we have a DataFrame with column names, but we don't know
        # if the first line is a header or data. We try to read the first line as
        # a header, and if it fails, we read it as data.
        try:
            # Try to convert columns to float - if first column is datetime, this will
            # fail and we know we have a header
            first_col_numeric = pd.to_numeric(df.columns[0], errors="coerce")
            if pd.notna(first_col_numeric):
                # First column name is numeric, might be data
                df.columns.astype(float)
                # This means the first line is data, so we re-read it, but
            # without the header:
            read_without_header = True
        except (ValueError, TypeError):  # TypeError can occur with pandas >= 2.2
            read_without_header = False
            # This means that the first line is a header, so we already have the data
            # without missing values.
            # However, it also means that there could be text information preceding
            # the header. Let's try to read it and put it in `header` variable.

            # 1. We read only the first 1000 lines to avoid reading the whole file
            # 2. We keep only the lines beginning with a comment character
            # 3. We join the lines to create a single string
            header = ""
            with open(filename, "r", encoding="utf-8") as file:
                for _ in range(1000):
                    line = file.readline()
                    if line.startswith("#"):
                        header += line
                    else:
                        break
            # Remove the last line if it contains the column names:
            last_line = header.splitlines()[-1] if header.splitlines() else ""
            if str(df.columns[0]) in last_line:
                header = "\n".join(header.splitlines()[:-1])

    # Now we read the whole file with the correct options
    try:
        df = read_csv_by_chunks(
            filename,
            worker=worker,
            decimal=decimal,
            delimiter=delimiter,
            header=None if read_without_header else "infer",
            skiprows=skiprows,
            comment="#",
        )
    except pd.errors.ParserError:
        # If chunked reading fails (e.g., ragged CSV), try different approaches
        df = None
        # Try with python engine (more flexible)
        for skip in [skiprows, 0, 9, 10, 15, 20]:  # Try different skiprows values
            if df is not None:
                break
            try:
                df = pd.read_csv(
                    filename,
                    decimal=decimal,
                    delimiter=delimiter,
                    header=None if read_without_header else "infer",
                    skiprows=skip,
                    comment="#",
                    engine="python",
                    encoding_errors="ignore",
                )
                break  # Success!
            except (pd.errors.ParserError, ValueError):
                continue

        # If still failing, try auto-detect
        if df is None:
            try:
                df = pd.read_csv(
                    filename,
                    engine="python",
                    encoding_errors="ignore",
                    comment="#",
                )
            except (pd.errors.ParserError, ValueError) as e:
                raise ValueError(f"Unable to parse CSV file: {e}") from e

    # Remove rows and columns where all values are NaN in the DataFrame:
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Check if first row contains header strings (non-numeric values in all columns)
    # This happens when header="infer" fails to detect the header
    if not df.empty and isinstance(df.columns[0], (int, np.integer)):
        # Columns are integers, not strings - header wasn't properly parsed
        first_row = df.iloc[0]
        # Count how many values in first row are non-numeric strings
        non_numeric_count = 0
        for val in first_row:
            try:
                float(val)
            except (ValueError, TypeError):
                if isinstance(val, str):
                    non_numeric_count += 1
        # If most of first row is non-numeric strings, it's likely a header row
        if non_numeric_count / len(first_row) > 0.5:
            # Use first row as column names
            df.columns = first_row.values
            # Drop the first row (header)
            df = df.iloc[1:].reset_index(drop=True)

    # Try to detect datetime columns - check first two columns
    # Often CSV files have an index column, then a datetime column
    if not df.empty and df.shape[1] >= 2:
        df, datetime_metadata = _detect_datetime_col(df)

    # Try to detect metadata columns (constant-value columns like serial numbers)
    # This must be done after datetime detection but before converting to numpy
    if not df.empty and df.shape[1] >= 2:
        df, column_metadata = _detect_metadata_cols(df)

    # Converting to NumPy array
    try:
        xydata = df.to_numpy(float)
    except (ValueError, TypeError):
        # If conversion fails, try converting each column individually
        # and dropping columns that can't be converted
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(axis=1, how="all")
        xydata = df.to_numpy(float)

    if xydata.size == 0:
        raise ValueError(
            f"Unable to read CSV file (no supported data after cleaning): {filename}"
        )

    xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)

    return CSVData(
        xydata=xydata,
        xlabel=xlabel,
        xunit=xunit,
        ylabels=ylabels,
        yunits=yunits,
        header=header,
        datetime_metadata=datetime_metadata,
        column_metadata=column_metadata,
    )


def write_csv(
    filename: str,
    xydata: np.ndarray,
    xlabel: str | None,
    xunit: str | None,
    ylabels: list[str] | None,
    yunits: list[str] | None,
    header: str | None,
) -> None:
    """Write CSV data.

    Args:
        filename: CSV file name
        xydata: XY data
        xlabel: X label
        xunit: X unit
        ylabels: Y labels
        yunits: Y units
        header: Header
    """
    labels = ""
    delimiter = ","
    if len(ylabels) == 1:
        ylabels = ["Y"] if not ylabels[0] else ylabels
    elif ylabels:
        ylabels = [
            f"Y{i + 1}" if not label else label for i, label in enumerate(ylabels)
        ]
        if yunits:
            ylabels = [
                f"{label} ({unit})" if unit else label
                for label, unit in zip(ylabels, yunits)
            ]
    if ylabels:
        xlabel = xlabel or "X"
        if xunit:
            xlabel += f" ({xunit})"
        labels = delimiter.join([xlabel] + ylabels)
    df = pd.DataFrame(xydata.T, columns=[xlabel] + ylabels)
    df.to_csv(filename, index=False, header=labels, sep=delimiter)
    # Add header if present
    if header:
        with open(filename, "r+", encoding="utf-8") as file:
            content = file.read()
            file.seek(0, 0)
            file.write(header + "\n" + content)


class MCAFile:
    """Class to handle MCA files."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.raw_data: str = ""
        self.xlabel: str | None = None
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.metadata: dict[str, str] = {}

    def __try_decode(self, raw_bytes: bytes) -> str:
        """Try to decode raw bytes with the specified encoding."""
        encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings_to_try:
            try:
                return raw_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        # If all attempts fail, use 'utf-8' with replacement
        warnings.warn("All decoding attempts failed. Used 'utf-8' with replacement.")
        return raw_bytes.decode("utf-8", errors="replace")

    def _read_raw_data(self) -> str:
        """Read the raw data from the MCA file, trying multiple encodings."""
        with open(self.filename, "rb") as file:
            raw_bytes = file.read()
        raw_data = self.__try_decode(raw_bytes)
        self.raw_data = raw_data.replace("\r\n", "\n").replace("\r", "\n")

    def _read_section(self, section: str) -> str | None:
        """Read a section from the raw data."""
        pattern = f"(?:.*)(^<<{section}>>$)(.*?)(?:<<.*>>)"
        match = re.search(pattern, self.raw_data, re.DOTALL + re.MULTILINE)
        if match:
            return match.group(2).strip()
        return None

    @staticmethod
    def _infer_string_value(value_str: str) -> str | float | int | datetime.datetime:
        """Infer the type of a string value and convert it accordingly."""
        # Try to convert the value to a number or datetime
        try:
            if value_str.isdigit():
                value = int(value_str)
            else:
                try:
                    value = float(value_str)
                except ValueError:
                    # Try to parse as datetime
                    try:
                        value = datetime.datetime.strptime(
                            value_str, "%m/%d/%Y %H:%M:%S"
                        )
                    except ValueError:
                        value = value_str  # Keep as string
        except ValueError:
            value = value_str
        return value

    def _extract_metadata_from_section(
        self, section: str
    ) -> dict[str, str | float | int | datetime.datetime]:
        """Extract metadata from a specific section."""
        section_contents = self._read_section(section)
        if section_contents is None:
            return {}
        metadata = {}
        patterns = (r"(.*?) - (.*?)$", r"(.*?)\s*: \s*(.*)$", r"(.*?)\s*=\s*(.*);")
        for line in section_contents.splitlines():
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    key, value_str = match.groups()
                    metadata[key.strip()] = self._infer_string_value(value_str.strip())
                    break
        return metadata

    def read(self) -> None:
        """Read the MCA file and extract data and metadata."""
        self._read_raw_data()
        self.metadata = self._extract_metadata_from_section("PMCA SPECTRUM")
        additional_metadata = self._extract_metadata_from_section("DPP STATUS")
        self.metadata.update(additional_metadata)
        data_section = self._read_section("DATA")
        self.y = np.fromstring(data_section, sep=" ") if data_section else None
        if self.y is not None:
            self.x = np.arange(len(self.y))
            cal_section = self._read_section("CALIBRATION")
            if cal_section:
                cal_metadata = self._extract_metadata_from_section(cal_section)
                self.xlabel = cal_metadata.get("LABEL")
                cal_data = np.array(
                    [
                        [float(v) for v in val.split(" ")]
                        for val in cal_section.splitlines()[1:]
                    ]
                )
                self.x = scipy.interpolate.interp1d(
                    cal_data[:, 0],
                    cal_data[:, 1],
                    bounds_error=False,
                    fill_value="extrapolate",
                )(self.x)
