# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Common functions for file name handling."""

from __future__ import annotations

import re
import string
import sys
import unicodedata
from typing import Any, Iterable

from sigima.objects.image import ImageObj
from sigima.objects.signal import SignalObj


class CustomFormatter(string.Formatter):
    """Custom string formatter to handle uppercase and lowercase strings."""

    def format_field(self, value, format_spec):
        """Format the given `value` according to the specified `format_spec`.

        If the value is a string and the format_spec ends with 'upper' or 'lower',
        convert the value to uppercase or lowercase, respectively, and remove the
        suffix from `format_spec` before formatting.

        Args:
            value: Value to format.
            format_spec: Format specification, may end with 'upper' or 'lower'.

        Returns:
            The formatted value.

        Raises:
            ValueError: If `format_spec` is invalid.
        """
        # Ignore dict objects silently (metadata should only be accessed via keys)
        if isinstance(value, dict):
            return ""
        if isinstance(value, str):
            if format_spec.endswith("upper"):
                value = value.upper()
                format_spec = format_spec[:-5]
            elif format_spec.endswith("lower"):
                value = value.lower()
                format_spec = format_spec[:-5]
        return super().format_field(value, format_spec)


def format_basenames(
    objects: Iterable[SignalObj | ImageObj],
    fmt: str,
    replacement: str = "_",
) -> list[str]:
    """Generate sanitized filenames for SignalObj or ImageObj instances.

    Format each object's name using the provided Python format string, then sanitize
    the result for safe use as a filename. The format string may reference any of:
        - {title}: object title
        - {index}: 1-based index
        - {count}: total number of objects
        - {xlabel}, {xunit}, {ylabel}, {yunit}: axis labels/units (if present)
        - {metadata[key]}: specific metadata value
          (direct {metadata} use is silently ignored)

    Args:
        objects: Objects to name.
        fmt: Python format string for naming.
        replacement: Replacement for invalid filename characters.

    Returns:
        Sanitized filenames for each object.

    Raises:
        KeyError: If the format string references an unknown placeholder.
    """
    result: list[str] = []
    formatter = CustomFormatter()
    for i, obj in enumerate(objects):
        # Note: We provide metadata dict only for {metadata[key]} access,
        # not for direct {metadata} use (which would create overly long filenames)
        metadata = obj.metadata
        context: dict[str, Any] = {
            "title": obj.title,
            "index": i + 1,
            "count": len(list(objects)),
            "xlabel": obj.xlabel,
            "xunit": obj.xunit,
            "ylabel": obj.ylabel,
            "yunit": obj.yunit,
            "metadata": metadata,
        }
        try:
            formatted = formatter.format(fmt, **context)
        except KeyError as exc:
            missing = str(exc.args[0]) if exc.args else str(exc)
            raise KeyError(f"Unknown format key in fmt: {missing!r}") from exc
        except ValueError as exc:
            # Re-raise with more context about which object failed
            raise ValueError(
                f"Invalid format string '{fmt}' for object '{context['title']}': {exc}"
            ) from exc
        # Sanitize final result to ensure it's a safe basename.
        result.append(sanitize_basename(formatted, replacement=replacement))
    return result


def sanitize_basename(basename: str, replacement: str = "_") -> str:
    """Sanitize a string to create a valid basename for the current operating system.

    This function removes or replaces characters that are invalid in basenames,
    depending on the underlying OS (Windows, macOS, Linux). It also strips trailing dots
    and spaces on Windows and normalizes unicode characters.

    Args:
        basename: Input string.
        replacement: Replacement string for invalid characters (default: "_").

    Returns:
        A sanitized string that can safely be used as a basename.
    """
    # Normalize unicode characters (NFKD form for decomposing accents and the like).
    basename = unicodedata.normalize("NFKD", basename)
    basename = basename.encode("ascii", "ignore").decode("ascii")

    # Characters not allowed in filenames (platform-dependent).
    if sys.platform.startswith("win"):
        # Reserved characters on Windows.
        invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
    else:
        # Only '/' is disallowed on Unix-based systems.
        invalid_chars = r"/"
        reserved_names = set()

    # Replace invalid characters.
    sanitized = re.sub(invalid_chars, replacement, basename)

    # Strip leading/trailing whitespace.
    sanitized = sanitized.strip()
    # On Windows, also strip trailing dots and spaces.
    if sys.platform.startswith("win"):
        sanitized = sanitized.rstrip(" .")

    # Truncate to a reasonable length to avoid OS path issues.
    sanitized = sanitized[:255]

    # Avoid reserved basenames.
    if sanitized.upper() in reserved_names:
        sanitized += "_"

    # If result is empty, fallback to a default name.
    if not sanitized:
        sanitized = "unnamed"

    return sanitized
