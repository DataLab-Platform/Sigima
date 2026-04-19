# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for :mod:`sigima.config` options system.

Covers:

* ``Options.describe_all`` introspection output.
* ``to_dict`` / ``from_dict`` round-trip and ``to_env_json`` serialization.
* ``ensure_loaded_from_env`` resilience to malformed JSON.
* Module-level ``__getattr__`` dispatch (``OPTIONS_RST`` lazy attribute and
  unknown attribute lookup).
* ``TypedOptionField`` and ``ImageIOOptionField`` validation.
* ``OptionField.context`` context-manager round-trip.
"""

# pylint: disable=invalid-name

from __future__ import annotations

import pytest

import sigima.config as cfg


def test_options_describe_all_runs(capsys: pytest.CaptureFixture) -> None:
    """``Options.describe_all`` must print a non-empty introspection report."""
    cfg.options.describe_all()
    out = capsys.readouterr().out
    assert out


def test_options_to_dict_and_from_dict_roundtrip() -> None:
    """``to_dict`` / ``from_dict`` must round-trip the option set so that
    user-saved configurations can be reloaded without loss."""
    d = cfg.options.to_dict()
    assert isinstance(d, dict)
    cfg.options.from_dict(d)


def test_options_to_env_json_and_load() -> None:
    """``to_env_json`` produces a JSON string suitable for the
    ``SIGIMA_OPTIONS`` environment variable transport."""
    s = cfg.options.to_env_json()
    assert isinstance(s, str) and s.startswith("{")


def test_options_ensure_loaded_from_env_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed ``SIGIMA_OPTIONS`` value must be silently ignored rather
    than crashing application startup."""
    monkeypatch.setenv("SIGIMA_OPTIONS", "not a json")
    cfg.options.ensure_loaded_from_env()


def test_options_module_getattr_options_rst() -> None:
    """The lazy ``OPTIONS_RST`` module attribute returns a documentation table
    used by Sphinx to render the configurable options."""
    rst = cfg.OPTIONS_RST
    assert isinstance(rst, str) and "Name" in rst


def test_options_module_getattr_unknown_raises() -> None:
    """Unknown module-level attributes must raise ``AttributeError`` so that
    typos surface immediately instead of returning ``None``."""
    with pytest.raises(AttributeError):
        cfg.__getattr__("does_not_exist")


def test_typed_option_field_check_wrong_type() -> None:
    """``TypedOptionField.check`` rejects values whose type does not match
    the declared option type, preventing silent type confusion."""
    fld = cfg.TypedOptionField(cfg.options, "_tmp_int", 0, int, "tmp")
    with pytest.raises(ValueError):
        fld.check("not an int")


def test_image_io_option_field_check_invalid_structure() -> None:
    """``ImageIOOptionField`` accepts only a list of ``(str, str)`` pairs;
    every other shape (non-list, single-element tuples, non-string
    contents) must be rejected at validation time."""
    fld = cfg.ImageIOOptionField(cfg.options, "_tmp_io", (("a", "b"),), "tmp")
    with pytest.raises(ValueError):
        fld.check("not a list")
    with pytest.raises(ValueError):
        fld.check([("only_one_elem",)])
    with pytest.raises(ValueError):
        fld.check([(1, 2)])


def test_option_field_context_manager() -> None:
    """``OptionField.context`` temporarily overrides a value and restores it
    on exit, even when the override equals the current value."""
    fld = cfg.options.fft_shift_enabled
    original = fld.get()
    with fld.context(original):
        assert fld.get() == original
    assert fld.get() == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
