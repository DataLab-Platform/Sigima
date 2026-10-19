# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for :py:func:`sigima.io.common.basename.format_basenames`."""

from __future__ import annotations

import pytest

from sigima.io.common.basename import format_basenames
from sigima.objects.image import ImageObj
from sigima.objects.signal import SignalObj


def make_signal(
    title: str = "",
    xlabel: str = "",
    xunit: str = "",
    ylabel: str = "",
    yunit: str = "",
    metadata: dict | None = None,
) -> SignalObj:
    """Create a SignalObj with specified attributes for testing.

    Args:
        title: Title of the signal.
        xlabel: Label for the x-axis.
        xunit: Unit for the x-axis.
        ylabel: Label for the y-axis.
        yunit: Unit for the y-axis.
        metadata: Metadata dictionary.

    Returns:
        Configured SignalObj instance.
    """
    sig = SignalObj()
    sig.title = title
    sig.xlabel = xlabel
    sig.xunit = xunit
    sig.ylabel = ylabel
    sig.yunit = yunit
    sig.metadata = {} if metadata is None else metadata
    return sig


def make_image(title: str = "", metadata: dict | None = None) -> ImageObj:
    """Create an ImageObj with specified attributes for testing.

    Args:
        title: Title of the image.
        metadata: Metadata dictionary.

    Returns:
        Configured ImageObj instance.
    """
    img = ImageObj()
    img.title = title
    img.metadata = {} if metadata is None else metadata
    return img


def test_format_basenames_with_indices_and_total_count():
    """Test with indexing and total count placeholders."""
    objs = [make_signal("sig1"), make_signal("sig2"), make_signal("sig3")]
    names = format_basenames(objs, fmt="{index:02d}_of_{count:02d}")
    assert names == ["01_of_03", "02_of_03", "03_of_03"]
    gen_names = format_basenames((o for o in objs), fmt="{index:02d}_of_{count:02d}")
    assert gen_names == ["01_of_03", "02_of_03", "03_of_03"]


def test_format_basenames_with_metadata_and_axes_placeholders():
    """Test with metadata and axis placeholders."""
    sig = make_signal(
        title="My/Signal",
        xlabel="Time",
        xunit="s",
        ylabel="Amp",
        yunit="V",
        metadata={"id": 42},
    )
    names = format_basenames(
        [sig], fmt="{title}_{xlabel}[{xunit}]_{ylabel}[{yunit}]_{metadata[id]}"
    )
    assert names == ["My_Signal_Time[s]_Amp[V]_42"]


def test_format_basenames_sanitization():
    """Test with sanitization for titles."""
    objs = [make_signal("A/B"), make_image("C/D")]  # '/' must be sanitized on all OSes
    names = format_basenames(objs, fmt="{title}")
    assert names == ["A_B", "C_D"]


def test_format_basenames_sanitization_with_custom_replacement():
    """Test with custom replacement character."""
    objs = [make_signal("a/b"), make_image("c/d")]
    names = format_basenames(objs, fmt="{title}", replacement="-")
    assert names == ["a-b", "c-d"]


def test_format_basenames_with_unknown_metadata_key():
    """Test that requesting a missing metadata key raises a KeyError."""
    sig = make_signal(title="T", metadata={"other": 1})
    with pytest.raises(KeyError):
        format_basenames([sig], fmt="{metadata[id]}")


def test_format_basenames_with_unknown_placeholder():
    """Test with unknown placeholder should raise a KeyError."""
    with pytest.raises(KeyError):
        format_basenames([make_signal("x")], fmt="{unknown}")


def test_format_basenames_with_direct_metadata_use():
    """Test that direct {metadata} use returns empty string (silently ignored)."""
    sig = make_signal(title="Test", metadata={"key1": "value1", "key2": "value2"})
    names = format_basenames([sig], fmt="{title} {metadata}")
    # The {metadata} placeholder should be replaced with empty string
    # The trailing space gets sanitized away
    assert names == ["Test"]


if __name__ == "__main__":
    test_format_basenames_with_indices_and_total_count()
    test_format_basenames_with_metadata_and_axes_placeholders()
    test_format_basenames_sanitization()
    test_format_basenames_sanitization_with_custom_replacement()
    test_format_basenames_with_unknown_placeholder()
    test_format_basenames_with_direct_metadata_use()
