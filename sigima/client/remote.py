# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima Client Remote Control
----------------------------

This module provides utilities to control DataLab from a Python script (e.g. with
Spyder) or from a Jupyter notebook.

The :class:`SimpleRemoteProxy` class provides the main interface
to DataLab XML-RPC server.
"""

from __future__ import annotations

import configparser as cp
import json
import os
import os.path as osp
import sys
import time
import warnings
from xmlrpc.client import ServerProxy

import guidata.dataset as gds
import numpy as np
from guidata.env import execenv
from guidata.io import JSONReader, JSONWriter
from guidata.userconfig import get_config_basedir
from packaging.version import Version

from sigima.client.base import SimpleBaseProxy
from sigima.client.utils import (
    array_to_rpcbinary,
    dataset_to_rpcjson,
    rpcjson_to_dataset,
)
from sigima.objects import ImageObj, SignalObj

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

__required_server_version__ = "1.0.0"  # Minimum required DataLab server version

XMLRPCPORT_ENV = "DATALAB_XMLRPCPORT"


def get_xmlrpcport_from_env() -> int | None:
    """Get XML-RPC port number from environment variable."""
    try:
        return int(os.environ.get(XMLRPCPORT_ENV))
    except (TypeError, ValueError):
        return None


def get_cdl_xmlrpc_port():
    """Return DataLab current XML-RPC port.

    This function attempts to read the XML-RPC port from DataLab's configuration file.
    It first tries the versioned configuration folders (starting with the latest
    version), and falls back to the legacy .DataLab folder for backward compatibility
    with DataLab v0.x.

    Returns:
        str: XML-RPC port number

    Raises:
        ConnectionRefusedError: DataLab has not yet been executed or configuration
            file is not accessible
    """
    if sys.platform == "win32" and "HOME" in os.environ:
        os.environ.pop("HOME")  # Avoid getting old WinPython settings dir

    config_basedir = get_config_basedir()

    # List of configuration folders to try, in order of preference
    # Try versioned folders first (v3, v2, v1), then legacy .DataLab
    config_folders = [
        (".DataLab_v3", "DataLab_v3.ini"),  # Future versions
        (".DataLab_v2", "DataLab_v2.ini"),
        (".DataLab_v1", "DataLab_v1.ini"),  # Current stable (v1.x)
        (".DataLab", "DataLab.ini"),  # Legacy (v0.x)
    ]

    # Try each configuration folder in order
    for folder_name, ini_name in config_folders:
        fname = osp.join(config_basedir, folder_name, ini_name)
        if osp.exists(fname):
            ini = cp.ConfigParser()
            try:
                ini.read(fname)
                port = ini.get("main", "rpc_server_port")
                # Successfully read port from this version's config
                return port
            except (cp.NoSectionError, cp.NoOptionError):
                # Config file exists but doesn't have the port yet, try next version
                continue

    # No valid configuration found
    raise ConnectionRefusedError(
        "DataLab has not yet been executed or no valid configuration found"
    )


def items_to_json(items: list) -> str | None:
    """Convert plot items to JSON string.

    Args:
        items (list): list of plot items

    Returns:
        str: JSON string or None if items is empty
    """
    from plotpy.io import save_items  # pylint: disable=import-outside-toplevel

    if items:
        writer = JSONWriter(None)
        save_items(writer, items)
        return writer.get_json(indent=4)
    return None


def json_to_items(json_str: str | None) -> list:
    """Convert JSON string to plot items.

    Args:
        json_str (str): JSON string or None

    Returns:
        list: list of plot items
    """
    from plotpy.io import load_items  # pylint: disable=import-outside-toplevel

    items = []
    if json_str:
        try:
            for item in load_items(JSONReader(json_str)):
                items.append(item)
        except json.decoder.JSONDecodeError:
            pass
    return items


class SimpleRemoteProxy(SimpleBaseProxy):
    """Object representing a proxy/client to DataLab XML-RPC server.
    This object is used to call DataLab functions from a Python script.

    This is a subset of DataLab's `RemoteClient` class, with only the methods
    that do not require DataLab object model to be implemented.

    Args:
        autoconnect: If True, automatically connect to DataLab XML-RPC server.
         Defaults to True.

    Raises:
        ConnectionRefusedError: DataLab is currently not running

    Examples:
        Here is a simple example of how to use SimpleRemoteProxy in a Python script
        or in a Jupyter notebook:

        >>> from sigima.client import SimpleRemoteProxy
        >>> proxy = SimpleRemoteProxy()  # autoconnect is on by default
        Connecting to DataLab XML-RPC server...OK (port: 28867)
        >>> proxy.get_version()
        '1.0.0'
        >>> proxy.add_signal("toto", np.array([1., 2., 3.]), np.array([4., 5., -1.]))
        True
        >>> proxy.get_object_titles()
        ['toto']
        >>> proxy["toto"]
        <sigima.objects.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> "toto" in proxy
        True
        >>> proxy[1]
        <sigima.objects.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy[1].data
        array([1., 2., 3.])
    """

    def __init__(self, autoconnect: bool = True) -> None:
        super().__init__()
        self.port: str = None
        self._datalab: ServerProxy
        if autoconnect:
            self.connect()

    def __connect_to_server(self, port: str | None = None) -> None:
        """Connect to DataLab XML-RPC server.

        Args:
            port (str | None): XML-RPC port to connect to. If not specified,
                the port is automatically retrieved from DataLab configuration.

        Raises:
            ConnectionRefusedError: DataLab is currently not running
        """
        if port is None:
            port = get_xmlrpcport_from_env()
            if port is None:
                port = get_cdl_xmlrpc_port()
        self.port = port
        self._datalab = ServerProxy(f"http://127.0.0.1:{port}", allow_none=True)
        try:
            version = self.get_version()
        except ConnectionRefusedError as exc:
            raise ConnectionRefusedError("DataLab is currently not running") from exc

        # If DataLab version is not compatible with this client, show a warning
        # pylint: disable=cyclic-import
        from sigima import __version__  # pylint: disable=import-outside-toplevel

        server_ver = Version(version)
        client_ver = Version(__version__)
        required_ver = Version(__required_server_version__)

        # Compare base versions (ignore pre-release/dev identifiers)
        # This allows 1.0.0b1 to satisfy >= 1.0.0 requirement
        if server_ver.base_version < required_ver.base_version:
            warnings.warn(
                f"DataLab server version ({server_ver}) may not be fully compatible "
                f"with Sigima client version {client_ver} "
                f"(requires DataLab >= {__required_server_version__}).\n"
                f"Please upgrade DataLab to {__required_server_version__} or higher.",
                UserWarning,
                stacklevel=2,
            )

    def connect(
        self,
        port: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        """Try to connect to DataLab XML-RPC server.

        Args:
            port (str | None): XML-RPC port to connect to. If not specified,
                the port is automatically retrieved from DataLab configuration.
            timeout (float | None): Maximum time to wait for connection in seconds.
                Defaults to 5.0. This is the total maximum wait time, not per retry.
            retries (int | None): Number of retries. Defaults to 10. This parameter
                is deprecated and will be removed in a future version (kept for
                backward compatibility).

        Raises:
            ConnectionRefusedError: Unable to connect to DataLab
            ValueError: Invalid timeout (must be >= 0.0)
            ValueError: Invalid number of retries (must be >= 1)
        """
        timeout = 5.0 if timeout is None else timeout
        retries = 10 if retries is None else retries  # Kept for backward compatibility
        if timeout < 0.0:
            raise ValueError("timeout must be >= 0.0")
        if retries < 1:
            raise ValueError("retries must be >= 1")

        execenv.print("Connecting to DataLab XML-RPC server...", end="")

        # Use exponential backoff for more efficient retrying
        start_time = time.time()
        poll_interval = 0.1  # Start with 100ms
        max_poll_interval = 1.0  # Cap at 1 second

        while True:
            try:
                # Try to connect - this may fail if DataLab hasn't written its
                # config file yet, so we retry it in the loop
                self.__connect_to_server(port=port)
                elapsed = time.time() - start_time
                execenv.print(f"OK (port: {self.port}, connected in {elapsed:.1f}s)")
                return
            except (ConnectionRefusedError, OSError) as exc:
                # Catch both ConnectionRefusedError and OSError (includes socket errors)
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    execenv.print("KO")
                    raise ConnectionRefusedError(
                        f"Unable to connect to DataLab after {elapsed:.1f}s"
                    ) from exc
                # Wait before next retry with exponential backoff
                time.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, max_poll_interval)

    def disconnect(self) -> None:
        """Disconnect from DataLab XML-RPC server."""
        # This is not mandatory with XML-RPC, but if we change protocol in the
        # future, it may be useful to have a disconnect method.
        self._datalab = None

    def is_connected(self) -> bool:
        """Return True if connected to DataLab XML-RPC server."""
        if self._datalab is not None:
            try:
                self.get_version()
                return True
            except ConnectionRefusedError:
                self._datalab = None
        return False

    def get_method_list(self) -> list[str]:
        """Return list of available methods."""
        return self._datalab.system.listMethods()

    # === Following methods should match the register functions in XML-RPC server

    def add_signal(
        self,
        title: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        xunit: str | None = None,
        yunit: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add signal data to DataLab.

        Args:
            title (str): Signal title
            xdata (numpy.ndarray): X data
            ydata (numpy.ndarray): Y data
            xunit (str | None): X unit. Defaults to None.
            yunit (str | None): Y unit. Defaults to None.
            xlabel (str | None): X label. Defaults to None.
            ylabel (str | None): Y label. Defaults to None.

        Returns:
            bool: True if signal was added successfully, False otherwise

        Raises:
            ValueError: Invalid xdata dtype
            ValueError: Invalid ydata dtype
        """
        xbinary = array_to_rpcbinary(xdata)
        ybinary = array_to_rpcbinary(ydata)
        return self._datalab.add_signal(
            title, xbinary, ybinary, xunit, yunit, xlabel, ylabel
        )

    # pylint: disable=too-many-arguments
    def add_image(
        self,
        title: str,
        data: np.ndarray,
        xunit: str | None = None,
        yunit: str | None = None,
        zunit: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
    ) -> bool:
        """Add image data to DataLab.

        Args:
            title (str): Image title
            data (numpy.ndarray): Image data
            xunit (str | None): X unit. Defaults to None.
            yunit (str | None): Y unit. Defaults to None.
            zunit (str | None): Z unit. Defaults to None.
            xlabel (str | None): X label. Defaults to None.
            ylabel (str | None): Y label. Defaults to None.
            zlabel (str | None): Z label. Defaults to None.

        Returns:
            bool: True if image was added successfully, False otherwise

        Raises:
            ValueError: Invalid data dtype
        """
        zbinary = array_to_rpcbinary(data)
        return self._datalab.add_image(
            title, zbinary, xunit, yunit, zunit, xlabel, ylabel, zlabel
        )

    def add_object(
        self, obj: SignalObj | ImageObj, group_id: str = "", set_current: bool = True
    ) -> None:
        """Add object to DataLab.

        Args:
            obj: Signal or image object
            group_id: group id in which to add the object. Defaults to ""
            set_current: if True, set the added object as current
        """
        self._datalab.add_object(dataset_to_rpcjson(obj), group_id, set_current)

    def load_from_directory(self, path: str) -> None:
        """Open objects from directory in current panel (signals/images).

        Args:
            path: directory path
        """
        self._datalab.load_from_directory(path)

    def calc(self, name: str, param: gds.DataSet | None = None) -> None:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name: Compute function name
            param: Compute function parameter. Defaults to None.

        Raises:
            ValueError: unknown function
        """
        if param is None:
            return self._datalab.calc(name)
        return self._datalab.calc(name, dataset_to_rpcjson(param))

    def get_object(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from index.

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            Object

        Raises:
            KeyError: if object not found
        """
        param_data = self._datalab.get_object(nb_id_title, panel)
        if param_data is None:
            return None
        return rpcjson_to_dataset(param_data)

    def get_object_shapes(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> list:
        """Get plot item shapes associated to object (signal/image).

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            List of plot item shapes
        """
        items_json = self._datalab.get_object_shapes(nb_id_title, panel)
        return json_to_items(items_json)

    def add_annotations_from_items(
        self, items: list, refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items).

        .. note:: This method is only available if PlotPy is installed.

        Args:
            items (list): annotation plot items
            refresh_plot (bool | None): refresh plot. Defaults to True.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.
        """
        try:
            items_json = items_to_json(items)
        except ImportError as exc:
            raise ImportError("PlotPy is not installed") from exc
        if items_json is not None:
            self._datalab.add_annotations_from_items(items_json, refresh_plot, panel)
