# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima Common tools for signal and image io support
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import dataclasses
import enum
import os
import os.path as osp
import re

from sigima.config import _
from sigima.objects.base import BaseObj
from sigima.worker import CallbackWorkerProtocol


class IOAction(enum.Enum):
    """I/O action type"""

    LOAD = enum.auto()
    SAVE = enum.auto()


@dataclasses.dataclass
class FormatInfo:
    """Format info"""

    name: str = None  # e.g. "Foobar camera image files"
    extensions: str = None  # e.g. "*.foobar *.fb"
    readable: bool = False  # True if format can be read
    writeable: bool = False  # True if format can be written
    requires: list[str] = None  # e.g. ["foobar"] if format requires foobar package

    def __str__(self) -> str:
        """Return string representation of format info"""
        return f"""{self.name}:
    Extensions: {self.extensions}
    Readable:   {"Yes" if self.readable else "No"}
    Writeable:  {"Yes" if self.writeable else "No"}
    Requires:   {", ".join(self.requires) if self.requires else "None"}"""

    def to_rst_table_row(self) -> str:
        """Return reStructuredText table row for format info
        (table `.. list-table::` format, with 5 columns)"""
        return f"""    * - {self.name}
      - {self.extensions}
      - {"•" if self.readable else ""}
      - {"•" if self.writeable else ""}
      - {", ".join(self.requires) if self.requires else "-"}"""


class FormatBase:
    """Object representing a data file io"""

    FORMAT_INFO: FormatInfo = None

    def __init__(self):
        self.info = self.FORMAT_INFO
        if self.info is None:
            raise ValueError(f"Format info not set for {self.__class__.__name__}")
        if self.info.name is None:
            raise ValueError(f"Format name not set for {self.__class__.__name__}")
        if self.info.extensions is None:
            raise ValueError(f"Format extensions not set for {self.__class__.__name__}")
        if not self.info.readable and not self.info.writeable:
            raise ValueError(f"Format {self.info.name} is not readable nor writeable")
        self.extlist = get_file_extensions(self.info.extensions)
        if not self.extlist:
            raise ValueError(f"Invalid format extensions for {self.__class__.__name__}")
        if self.info.requires:
            for package in self.info.requires:
                try:
                    __import__(package)
                except ImportError as exc:
                    raise ImportError(
                        f"Format {self.info.name} requires {package} package"
                    ) from exc

    def get_filter(self, action: IOAction) -> str:
        """Return file filter for Qt file dialog

        Args:
            action: I/O action type

        Returns:
            File filter string
        """
        assert action in (IOAction.LOAD, IOAction.SAVE)
        if action == IOAction.LOAD and not self.info.readable:
            return ""
        if action == IOAction.SAVE and not self.info.writeable:
            return ""
        return f"{self.info.name} ({self.info.extensions})"

    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[BaseObj]:
        """Read list of native objects (signal or image) from file.
        For single object, return a list with one object.

        Args:
            filename: file name
            worker: Callback worker object

        Raises:
            NotImplementedError: if format is not supported

        Returns:
            List of native objects (signal or image)
        """
        raise NotImplementedError(f"Reading from {self.info.name} is not supported")

    def write(self, filename: str, obj: BaseObj) -> None:
        """Write data to file

        Args:
            filename: file name
            obj: native object (signal or image)

        Raises:
            NotImplementedError: if format is not supported
        """
        raise NotImplementedError(f"Writing to {self.info.name} is not supported")


#  pylint: disable=bad-mcs-classmethod-argument
class BaseIORegistry(type):
    """Metaclass for registering I/O handler classes"""

    REGISTRY_INFO: str = ""  # Registry info, override in subclasses

    _io_format_instances: list[FormatBase] = []

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if not name.endswith("FormatBase"):
            try:
                #  pylint: disable=no-value-for-parameter
                cls._io_format_instances.append(cls())
            except ImportError:
                # This format is not supported
                pass

    @classmethod
    def get_formats(cls) -> list[FormatBase]:
        """Return I/O format handlers

        Returns:
            List of I/O format handlers
        """
        return cls._io_format_instances

    @classmethod
    def get_format_info(cls, rst: bool = True) -> str:
        """Return I/O format info

        Args:
            rst: if True, return in reStructuredText format (default: True)

        Returns:
            Text description for all I/O formats
        """
        if rst:
            txt = f"{cls.REGISTRY_INFO}:\n\n.. list-table::\n    :header-rows: 1\n\n"
            txt += "    * - Name\n      - Extensions\n      "
            txt += "- Readable\n      - Writeable\n      - Requires\n"
            txt += "\n".join([fmt.info.to_rst_table_row() for fmt in cls.get_formats()])
        else:
            txt = f"{cls.REGISTRY_INFO}:{os.linesep}"
            indent = " " * 4
            finfo = "\n".join([str(fmt.info) for fmt in cls.get_formats()])
            txt += "\n".join([indent + line for line in finfo.splitlines()])
        return txt

    @classmethod
    def get_all_filters(cls, action: IOAction) -> str:
        """Return all file filters for Qt file dialog

        Args:
            action: I/O action type

        Returns:
            File filters
        """
        extlist = []  # file extension list
        for fmt in cls.get_formats():
            fmt: FormatBase
            if not fmt.info.readable and action == IOAction.LOAD:
                continue
            if not fmt.info.writeable and action == IOAction.SAVE:
                continue
            extlist.extend(fmt.extlist)
        allsupported = _("All supported files")
        return f"{allsupported} ({'*.' + ' *.'.join(extlist)})"

    @classmethod
    def get_filters(cls, action: IOAction) -> str:
        """Return file filters for Qt file dialog

        Args:
            action: I/O action type

        Returns:
            File filters
        """
        flist = []  # file filter list
        flist.append(cls.get_all_filters(action))
        for fmt in cls.get_formats():
            fmt: FormatBase
            flist.append(fmt.get_filter(action))
        return "\n".join(flist)

    @classmethod
    def get_read_filters(cls) -> str:
        """Return file filters for Qt open file dialog

        Returns:
            File filters
        """
        return cls.get_filters(IOAction.LOAD)

    @classmethod
    def get_write_filters(cls) -> str:
        """Return file filters for Qt save file dialog

        Returns:
            File filters
        """
        return cls.get_filters(IOAction.SAVE)

    @classmethod
    def get_format(cls, filename: str, action: IOAction) -> FormatBase:
        """Return format handler for filename

        Args:
            filename: file name
            action: I/O action type

        Raises:
            NotImplementedError: if file data type is not supported

        Returns:
            Format handler
        """
        for fmt in cls.get_formats():
            fmt: FormatBase
            if osp.splitext(filename)[1][1:].lower() in fmt.extlist:
                if not fmt.info.readable and action == IOAction.LOAD:
                    continue
                if not fmt.info.writeable and action == IOAction.SAVE:
                    continue
                return fmt
        raise NotImplementedError(
            f"{filename} is not supported for {action.name.lower()}"
        )

    @classmethod
    def read(
        cls, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[BaseObj]:
        """Read data from file, return native object (signal or image) list.
        For single object, return a list with one object.

        Args:
            filename: file name
            worker: Callback worker object

        Raises:
            NotImplementedError: if file data type is not supported

        Returns:
            List of native objects (signal or image)
        """
        fmt = cls.get_format(filename, IOAction.LOAD)
        return fmt.read(filename, worker)

    @classmethod
    def write(cls, filename: str, obj: BaseObj) -> None:
        """Write data to file from native object (signal or image).

        Args:
            filename: file name
            obj: native object (signal or image)

        Raises:
            NotImplementedError: if file data type is not supported
        """
        fmt = cls.get_format(filename, IOAction.SAVE)
        fmt.write(filename, obj)


def get_file_extensions(string: str) -> list[str]:
    """Return a list of file extensions in a string

    Args:
        string: string containing file extensions

    Returns:
        List of file extensions
    """
    pattern = r"\S+\.[\w-]+"
    matches = re.findall(pattern, string)
    return [match.split(".")[-1].lower() for match in matches]
