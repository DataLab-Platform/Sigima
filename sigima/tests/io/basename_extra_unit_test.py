# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Extra unit tests for :mod:`sigima.io.common.basename`.

These complement ``basename_unit_test.py`` by directly exercising
``sanitize_basename`` and the ``CustomFormatter`` upper/lower format specs as
well as the error path of ``format_basenames`` on invalid format strings.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from sigima.io.common.basename import (
    CustomFormatter,
    format_basenames,
    sanitize_basename,
)
from sigima.objects import create_signal

# ===========================================================================
# CustomFormatter: upper / lower suffixes
# ===========================================================================


def test_custom_formatter_upper_suffix() -> None:
    """The ``upper`` format suffix uppercases the substituted value."""
    fmt = CustomFormatter()
    assert fmt.format("{x:upper}", x="hello") == "HELLO"


def test_custom_formatter_lower_suffix() -> None:
    """The ``lower`` format suffix lowercases the substituted value."""
    fmt = CustomFormatter()
    assert fmt.format("{x:lower}", x="WORLD") == "world"


def test_custom_formatter_upper_with_extra_spec() -> None:
    """Standard alignment specs (e.g. ``>5``) compose with the custom
    ``upper`` suffix without conflict."""
    fmt = CustomFormatter()
    # Combine alignment with upper suffix.
    assert fmt.format("{x:>5upper}", x="ab") == "   AB"


def test_custom_formatter_dict_silently_ignored() -> None:
    """Mapping arguments cannot be safely embedded in a filename and are
    silently rendered as an empty string."""
    fmt = CustomFormatter()
    assert fmt.format("{m}", m={"a": 1}) == ""


def test_custom_formatter_passthrough_for_non_string() -> None:
    """Standard numeric format specs (e.g. ``.2f``) on numeric inputs are
    forwarded unchanged to the underlying ``str.format`` machinery."""
    fmt = CustomFormatter()
    assert fmt.format("{n:.2f}", n=3.14159) == "3.14"


# ===========================================================================
# format_basenames: error paths
# ===========================================================================


def test_format_basenames_unknown_key_raises() -> None:
    """Referencing a key that is not part of the documented placeholder
    set raises ``KeyError`` with an explicit ``Unknown format key`` message."""
    sig = create_signal("hello", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(KeyError, match="Unknown format key"):
        format_basenames([sig], fmt="{not_a_key}")


def test_format_basenames_invalid_format_raises() -> None:
    """A format spec incompatible with the value type (e.g. ``.2f`` on a
    string title) surfaces as a ``ValueError`` rather than crashing later."""
    sig = create_signal("hello", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    # ``.2f`` applied to a string title triggers a ``ValueError`` re-raised by
    # ``format_basenames``.
    with pytest.raises(ValueError, match="Invalid format string"):
        format_basenames([sig], fmt="{title:.2f}")


def test_format_basenames_with_upper_suffix() -> None:
    """End-to-end check that the ``upper`` suffix works through
    ``format_basenames`` (not only at the formatter level)."""
    sig = create_signal("hello", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    out = format_basenames([sig], fmt="{title:upper}")
    assert out == ["HELLO"]


# ===========================================================================
# sanitize_basename
# ===========================================================================


def test_sanitize_basename_empty_returns_unnamed() -> None:
    """An empty title is replaced by the literal ``"unnamed"`` so the
    generated filename remains valid."""
    assert sanitize_basename("") == "unnamed"


def test_sanitize_basename_normalizes_unicode() -> None:
    # Accented characters should be stripped to ASCII.
    """Unicode is normalised to plain ASCII so the filename is safe across
    file systems with different default encodings."""
    assert sanitize_basename("café") == "cafe"


def test_sanitize_basename_replaces_slash() -> None:
    # '/' is forbidden on every platform.
    """Forward slash is forbidden in filenames on every OS, so it must be
    replaced (default replacement is ``_``)."""
    out = sanitize_basename("a/b")
    assert "/" not in out
    assert "_" in out


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="Windows-specific reserved name handling",
)
def test_sanitize_basename_windows_reserved_names() -> None:
    """Windows reserved device names (``CON``, ``PRN``, etc.) must be
    altered (e.g. by appending a suffix) so the resulting filename is
    actually creatable on Windows."""
    for name in ("CON", "PRN", "AUX", "NUL", "COM1", "LPT9"):
        out = sanitize_basename(name)
        # Must not equal the reserved name as-is.
        assert out.upper() != name
        assert out.upper().startswith(name)


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="Windows-specific invalid characters",
)
def test_sanitize_basename_windows_invalid_chars() -> None:
    """All Windows-forbidden characters (``<>:"|?*``) are stripped from
    the sanitized output."""
    out = sanitize_basename('a<b>:"c|?*d')
    for bad in '<>:"|?*':
        assert bad not in out


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="Windows strips trailing dots/spaces",
)
def test_sanitize_basename_windows_strips_trailing() -> None:
    """Trailing dots and spaces (which Windows silently strips when
    creating files) are removed up-front to keep behaviour predictable."""
    assert sanitize_basename("hello. .") == "hello"


def test_sanitize_basename_truncates_to_255_chars() -> None:
    """Names longer than the typical 255-byte file system limit are
    truncated to exactly 255 characters."""
    long_name = "a" * 1000
    assert len(sanitize_basename(long_name)) == 255


def test_sanitize_basename_custom_replacement() -> None:
    # On non-Windows, only '/' is replaced; use it to verify replacement param.
    """The ``replacement`` argument overrides the default ``_`` substitute
    used for forbidden characters."""
    out = sanitize_basename("a/b", replacement="-")
    assert out == "a-b"


if __name__ == "__main__":
    pytest.main([__file__])
