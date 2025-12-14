# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Configuration (:mod:`sigima.config`)
-------------------------------------

The :mod:`sigima.config` module provides a way to manage configuration options for the
`sigima` library, as well as to handle translations and data paths, and other
configuration-related tasks.

It allows users to set and retrieve options that affect the behavior of the library,
such as whether to keep results of computations or not. The options are handled as
in-memory objects with default values provided, and can be temporarily overridden using
a context manager.

Typical usage:

.. code-block:: python

    from sigima.config import options

    # Get an option
    value = options.fft_shift_enabled.get(default=True)

    # Set an option
    options.fft_shift_enabled.set(False)

    # Temporarily override an option
    with options.fft_shift_enabled.context(True):
        ...

The following table lists the available options:

.. options-table::

.. note::

    The options are stored in an environment variable in JSON format, allowing for
    synchronization with external configurations or other processes that may need to
    read or modify the options. The environment variable name is defined by
    :attr:`sigima.config.OptionsContainer.ENV_VAR`. This is especially useful for
    applications such as DataLab (where the `sigima` library is used as a core
    component) as computations may be run in separate processes.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Generator

from guidata import configtools

# Translation and data path configuration
MOD_NAME = "sigima"
_ = configtools.get_translation(MOD_NAME)
DATAPATH = configtools.get_module_data_path(MOD_NAME, "data")


class OptionField:
    """A configurable option field with get/set/context interface.

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option (used for introspection or errors).
        default: Default value of the option.
    """

    def __init__(
        self,
        container: OptionsContainer,
        name: str,
        default: Any,
        description: str = "",
    ) -> None:
        self._container = container
        self.name = name
        self.check(default)  # Validate the default value
        self._value = default
        self.description = description

    def check(self, value: Any) -> None:  # pylint: disable=unused-argument
        """Check if the value is valid for this option.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is not valid.
        """
        # This method can be overridden in subclasses for specific validation

    def get(self, sync_env: bool = True) -> Any:
        """Return the current value of the option.

        Args:
            sync_env: Whether to ensure the environment variable is synchronized
             with the current value.

        Returns:
            The current value of the option.
        """
        if sync_env:
            self._container.ensure_loaded_from_env()
        return self._value

    def set(self, value: Any, sync_env: bool = True) -> None:
        """Set the value of the option.

        Args:
            value: The new value to assign.
            sync_env: Whether to synchronize the environment variable.
        """
        self.check(value)  # Validate the new value
        self._value = value
        if sync_env:
            self._container.sync_env()

    def context(self, temp_value: Any) -> Generator[None, None, None]:
        """Temporarily override the option within a context.

        Args:
            temp_value: Temporary value to use within the context.

        Yields:
            None. Restores the original value upon exit.
        """

        @contextmanager
        def _ctx():
            old_value = self._value
            self.set(temp_value)
            try:
                yield
            finally:
                self.set(old_value)

        return _ctx()


class TypedOptionField(OptionField):
    """A configurable option field with type checking.

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option (used for introspection or errors).
        default: Default value of the option.
        expected_type: Expected type of the option value.
        description: Description of the option.
    """

    def __init__(
        self,
        container: OptionsContainer,
        name: str,
        default: Any,
        expected_type: type,
        description: str = "",
    ) -> None:
        self.expected_type = expected_type
        super().__init__(container, name, default, description)

    def check(self, value: Any) -> None:
        """Check if the value is of the expected type.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is not of the expected type.
        """
        if not isinstance(value, self.expected_type):
            raise ValueError(
                f"Expected {self.expected_type.__name__}, got {type(value).__name__}"
            )


class ImageIOOptionField(OptionField):
    """A configurable option field for image I/O formats.

    .. note::

        This option is specifically for image I/O formats and expects a tuple of
        tuples (or list of lists) of strings representing the formats,
        similar to the following:

        ... code-block:: python

            imageio_formats = (
                ("*.gel", "Opticks GEL"),
                ("*.spe", "Princeton Instruments SPE"),
                ("*.ndpi", "Hamamatsu Slide Scanner NDPI"),
                ("*.rec", "PCO Camera REC"),
            )

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option (used for introspection or errors).
        default: Default value of the option.
        description: Description of the option.
    """

    def check(
        self,
        value: list[list[str, str]]
        | list[tuple[str, str]]
        | tuple[tuple[str, str]]
        | tuple[list[str, str]],
    ) -> None:
        """Check if the value is a valid image I/O format.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is not a valid image I/O format.
        """
        if not isinstance(value, (tuple, list)) or not all(
            isinstance(item, (tuple, list)) and len(item) == 2 for item in value
        ):
            raise ValueError(
                "Expected a tuple of tuples with two elements each "
                "(format, description)"
            )
        for item in value:
            if not isinstance(item[0], str) or not isinstance(item[1], str):
                raise ValueError(
                    "Each item must be a tuple of (format, description) as strings"
                )

    def set(self, value: Any, sync_env: bool = True) -> None:
        """Set the value of the option.

        Args:
            value: The new value to assign.
            sync_env: Whether to synchronize the environment variable.
        """
        super().set(value, sync_env)
        # pylint: disable=cyclic-import
        # pylint: disable=import-outside-toplevel
        from sigima.io.image import formats

        # Generate image I/O format classes based on the new value
        # This allows dynamic loading of formats based on the configuration
        formats.generate_imageio_format_classes(value)


IMAGEIO_FORMATS = (
    ("*.gel", "Opticks GEL"),
    ("*.spe", "Princeton Instruments SPE"),
    ("*.ndpi", "Hamamatsu Slide Scanner NDPI"),
    ("*.rec", "PCO Camera REC"),
)  # Default image I/O formats


class OptionsContainer:
    """Container for all configurable options in the `sigima` library.

    Options are exposed as attributes with `.get()`, `.set()` and `.context()` methods.
    """

    #: Environment variable name for options in JSON format
    # This is used to synchronize options with external configurations or with
    # separate processes that may need to read or modify the options.
    ENV_VAR = "SIGIMA_OPTIONS_JSON"

    @classmethod
    def set_env(cls, value: str) -> None:
        """Set the environment variable with the given JSON string.

        Args:
            value: A JSON string representation of the options to set.
        """
        os.environ[cls.ENV_VAR] = value

    @classmethod
    def get_env(cls) -> str:
        """Get the current value of the environment variable.

        Returns:
            The JSON string representation of the options from the environment variable.
        """
        return os.environ.get(cls.ENV_VAR, "{}")

    def __init__(self) -> None:
        self.fft_shift_enabled = TypedOptionField(
            self,
            "fft_shift_enabled",
            default=True,
            expected_type=bool,
            description=_(
                "If True, the FFT operations will apply a shift to the zero frequency "
                "component to the center of the spectrum. This is useful for "
                "visualizing frequency components in a more intuitive way."
            ),
        )
        self.auto_normalize_kernel = TypedOptionField(
            self,
            "auto_normalize_kernel",
            default=False,
            expected_type=bool,
            description=_(
                "If True, convolution kernels will be automatically normalized to "
                "sum to 1.0 before convolution. This ensures that the output signal "
                "or image has the same overall magnitude as the input when using "
                "smoothing kernels. Set to False to preserve the mathematical "
                "properties of the original kernel."
            ),
        )
        self.imageio_formats = ImageIOOptionField(
            self,
            "imageio_formats",
            default=IMAGEIO_FORMATS,
            description=_(
                """List of supported image I/O formats. Each format is a tuple of
``(file_extension, description)``.

The ``sigima`` library supports any image format that can be read by the ``imageio``
library, provided that the associated plugin(s) are installed (see `imageio
documentation <https://imageio.readthedocs.io/en/stable/formats/index.html>`_)
and that the output NumPy array data type and shape are supported by ``sigima``.

To add a new file format, you may use the ``imageio_formats`` option to specify
additional formats. Each entry should be a tuple of (file extension, description).
"""
            ),
        )
        self.vistools_backend = TypedOptionField(
            self,
            "vistools_backend",
            default="auto",
            expected_type=str,
            description=_(
                """Backend library for test visualization (interactive tests only).

Valid values: ``"auto"``, ``"plotpy"``, ``"matplotlib"``.

- ``"auto"`` (default): Automatically select PlotPy if available, otherwise Matplotlib
- ``"plotpy"``: Use PlotPy for interactive visualizations (requires PlotPy and Qt)
- ``"matplotlib"``: Use Matplotlib for visualizations (simpler, view-only)

This setting can also be overridden using the ``SIGIMA_VISTOOLS_BACKEND`` environment
variable. Note that Matplotlib backend does not support all features of PlotPy
(e.g., ``create_segment()``, ``create_cursor()``, etc. will raise NotImplementedError).
"""
            ),
        )
        # Add new options here

    def describe_all(self) -> None:
        """Print the name, value, and description of all options."""
        for name in vars(self):
            opt = getattr(self, name)
            if isinstance(opt, OptionField):
                print(f"{name} = {opt.get()}  # {opt.description}")

    def generate_rst_doc(self) -> str:
        """Generate reStructuredText documentation for all options.

        Returns:
            A string containing the reStructuredText documentation.
        """
        doc = """.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Default Value
      - Description
"""
        for name in vars(self):
            opt = getattr(self, name)
            if isinstance(opt, OptionField):
                # Process description to work within table cells
                description = opt.description.strip()
                # For table cells, we need to indent continuation lines properly
                # and handle multi-line content correctly
                description_lines = description.split("\n")
                if len(description_lines) > 1:
                    # Multi-line descriptions need special handling in RST tables
                    processed_lines = [description_lines[0]]  # First line
                    for line in description_lines[1:]:
                        if line.strip():  # Non-empty lines
                            processed_lines.append("        " + line.strip())
                        else:  # Empty lines
                            processed_lines.append("")
                    description = "\n".join(processed_lines)

                # Get the value and format it nicely
                value = repr(opt.get(sync_env=False))
                if len(value) > 200:  # Truncate very long values
                    value = value[:197] + "..."

                doc += f"    * - ``{name}``\n"
                doc += f"      - ``{value}``\n"
                doc += f"      - {description}\n"
        return doc

    def ensure_loaded_from_env(self) -> None:
        """Lazy-load from JSON env var on first access."""
        value = self.get_env()
        try:
            values = json.loads(value)
            self.from_dict(values)
        except Exception as exc:  # pylint: disable=broad-except
            # If loading fails, we just log a warning and continue with defaults
            print(f"[sigima] Warning: failed to load options from env: {exc}")

    def to_env_json(self) -> str:
        """Return the current options as a JSON string for environment variable.

        Returns:
            A JSON string representation of the current options.
        """
        return json.dumps(self.to_dict())

    def sync_env(self) -> None:
        """Update env var with current option values."""
        self.set_env(self.to_env_json())

    def to_dict(self) -> dict[str, Any]:
        """Return the current option values as a dictionary.

        Returns:
            A dictionary with option names as keys and their current values.
        """
        return {
            name: getattr(self, name).get(sync_env=False)
            for name in vars(self)
            if isinstance(getattr(self, name), OptionField)
        }

    def from_dict(self, values: dict[str, Any]) -> None:
        """Set option values from a dictionary.

        Args:
            values: A dictionary with option names as keys and their new values.
        """
        for name, value in values.items():
            if hasattr(self, name):
                opt = getattr(self, name)
                if isinstance(opt, OptionField):
                    opt.set(value, sync_env=False)
        self.sync_env()


#: Global instance of the options container
options = OptionsContainer()

# Generate OPTIONS_RST at module load time after options is created
# This avoids circular import issues since everything is already loaded
OPTIONS_RST = options.generate_rst_doc()


def __getattr__(name: str):
    """Handle lazy evaluation of module-level attributes.

    This provides backward compatibility for any code that might access OPTIONS_RST.
    """
    if name == "OPTIONS_RST":
        # Return the global variable if it exists, otherwise generate it
        return globals().get("OPTIONS_RST", options.generate_rst_doc())
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
