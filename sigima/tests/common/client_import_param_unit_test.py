# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Remote client import parameters unit tests

Format-specific import parameters (e.g. RAW image layout) are sent to DataLab by
``load_from_files`` and ``load_from_directory`` like computation parameters with
``calc``: serialized as RPC JSON, and only when provided, so that older DataLab
servers keep receiving the original call signature.
"""

from __future__ import annotations

import xmlrpc.client
from unittest.mock import Mock

from sigima.client.remote import SimpleRemoteProxy
from sigima.client.stub import datalab_stub_server
from sigima.client.utils import rpcjson_to_dataset
from sigima.io import RawImageImportParam


def create_proxy_with_mock_server() -> tuple[SimpleRemoteProxy, Mock]:
    """Return a disconnected proxy whose server calls are recorded."""
    proxy = SimpleRemoteProxy(autoconnect=False)
    server = Mock()
    proxy._datalab = server  # pylint: disable=protected-access
    return proxy, server


def test_load_from_files_without_param_keeps_legacy_call() -> None:
    """Do not send any import parameter argument when none is given."""
    proxy, server = create_proxy_with_mock_server()

    proxy.load_from_files(["image.png"])

    server.load_from_files.assert_called_once_with(["image.png"])


def test_load_from_files_serializes_import_param() -> None:
    """Send import parameters as RPC JSON, like ``calc`` parameters."""
    proxy, server = create_proxy_with_mock_server()
    param = RawImageImportParam.create(dtype="uint8", width=4, height=3)

    proxy.load_from_files(["image.raw"], param=param)

    filenames, param_data = server.load_from_files.call_args.args
    assert filenames == ["image.raw"]
    received = rpcjson_to_dataset(param_data)
    assert isinstance(received, RawImageImportParam)
    assert (received.dtype, received.width, received.height) == ("uint8", 4, 3)


def test_load_from_directory_without_param_keeps_legacy_call() -> None:
    """Do not send any import parameter argument when none is given."""
    proxy, server = create_proxy_with_mock_server()

    proxy.load_from_directory("data")

    server.load_from_directory.assert_called_once_with("data")


def test_load_from_directory_serializes_import_param() -> None:
    """Send directory import parameters as RPC JSON."""
    proxy, server = create_proxy_with_mock_server()
    param = RawImageImportParam.create(width=2, height=5, little_endian=False)

    proxy.load_from_directory("data", param=param)

    path, param_data = server.load_from_directory.call_args.args
    assert path == "data"
    received = rpcjson_to_dataset(param_data)
    assert isinstance(received, RawImageImportParam)
    assert (received.width, received.height) == (2, 5)
    assert received.little_endian is False


def test_stub_server_accepts_import_param() -> None:
    """The stub server accepts calls with and without import parameters."""
    param = RawImageImportParam.create(width=2, height=2)
    with datalab_stub_server() as port:
        proxy = SimpleRemoteProxy(autoconnect=False)
        proxy.connect(port=str(port))
        try:
            proxy.load_from_files(["image.raw"])
            proxy.load_from_files(["image.raw"], param=param)
            proxy.load_from_directory("data")
            proxy.load_from_directory("data", param=param)
        finally:
            proxy.disconnect()


def test_stub_server_exports_and_opens_capsules(tmp_path) -> None:
    """The stub server answers the capsule calls of the DataLab control API."""
    capsule = tmp_path / "workspace.dlcapsule"
    with datalab_stub_server() as port:
        server = xmlrpc.client.ServerProxy(f"http://127.0.0.1:{port}")
        fingerprint = server.export_capsule(str(capsule), "Name")
        assert fingerprint.startswith("sha256:")
        assert capsule.exists()
        server.open_capsule(str(capsule))
