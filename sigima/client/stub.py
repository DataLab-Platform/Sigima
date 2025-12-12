# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima Client Stub/Mock Server (Real Objects)
---------------------------------------------

This module provides a stub XML-RPC server that emulates DataLab's XML-RPC interface
for testing purposes using real Sigima objects. The stub server allows tests to run
without requiring a real DataLab instance.
"""

from __future__ import annotations

import os
import threading
import uuid
from contextlib import contextmanager
from socketserver import ThreadingMixIn
from typing import TYPE_CHECKING
from xmlrpc.client import Binary
from xmlrpc.server import SimpleXMLRPCServer

from guidata.env import execenv

from sigima.client import utils
from sigima.objects import ImageObj, SignalObj, create_image, create_signal

if TYPE_CHECKING:
    from collections.abc import Generator

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code


class MockGroup:
    """Mock group object."""

    def __init__(self, title: str, uuid_str: str | None = None):
        self.title = title
        self.uuid = uuid_str or str(uuid.uuid4())
        self.objects: list[str] = []  # List of object UUIDs


class ThreadingXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    """Threading XML-RPC server to handle multiple requests."""

    daemon_threads = True
    allow_reuse_address = True


class DataLabStubServer:
    """Stub XML-RPC server emulating DataLab XML-RPC interface with real objects.

    This server provides mock implementations of all DataLab XML-RPC methods
    using real SignalObj and ImageObj instances for maximum compatibility.

    Args:
        port: Port to bind to. If 0, uses a random available port.
        verbose: If True, print verbose debug information.
    """

    def __init__(self, port: int = 0, verbose: bool = True) -> None:
        """Initialize the stub server.

        Args:
            port: Port to bind to. If 0, uses a random available port.
        """
        self.port = port
        self.verbose = verbose
        self.server: ThreadingXMLRPCServer | None = None
        self.server_thread: threading.Thread | None = None

        # Real Sigima object storage
        self.signals: dict[str, SignalObj] = {}  # uuid -> SignalObj
        self.images: dict[str, ImageObj] = {}  # uuid -> ImageObj
        self.signal_groups: dict[str, MockGroup] = {}  # uuid -> group
        self.image_groups: dict[str, MockGroup] = {}  # uuid -> group

        # Current state
        self.current_panel = "signal"  # "signal", "image", or "macro"
        self.selected_objects: list[str] = []  # list of UUIDs
        self.selected_groups: list[str] = []  # list of group UUIDs
        self.auto_refresh = True
        self.show_titles = True

        # Add default groups
        self._add_default_group("signal")
        self._add_default_group("image")

    def _add_default_group(self, panel: str) -> None:
        """Add default group for a panel."""
        group = MockGroup("Group 1")
        if panel == "signal":
            self.signal_groups[group.uuid] = group
        elif panel == "image":
            self.image_groups[group.uuid] = group

    def start(self) -> int:
        """Start the XML-RPC server.

        Returns:
            Port number the server is listening on
        """
        self.server = ThreadingXMLRPCServer(
            ("127.0.0.1", self.port), allow_none=True, logRequests=False
        )

        # Register all methods
        self._register_functions()

        self.port = self.server.server_address[1]

        # Start server in a separate thread
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()

        execenv.print(f"DataLab stub server started on port {self.port}")
        return self.port

    def stop(self) -> None:
        """Stop the XML-RPC server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.server_thread:
                self.server_thread.join(timeout=1.0)
            execenv.print("DataLab stub server stopped")

    def _register_functions(self) -> None:
        """Register all XML-RPC functions."""
        # System introspection methods
        self.server.register_introspection_functions()

        # Basic server methods
        self.server.register_function(self.get_version, "get_version")
        self.server.register_function(self.close_application, "close_application")
        self.server.register_function(self.raise_window, "raise_window")

        # Panel management
        self.server.register_function(self.get_current_panel, "get_current_panel")
        self.server.register_function(self.set_current_panel, "set_current_panel")

        # Application control
        self.server.register_function(self.reset_all, "reset_all")
        self.server.register_function(self.toggle_auto_refresh, "toggle_auto_refresh")
        self.server.register_function(self.toggle_show_titles, "toggle_show_titles")

        # File operations
        self.server.register_function(self.save_to_h5_file, "save_to_h5_file")
        self.server.register_function(self.open_h5_files, "open_h5_files")
        self.server.register_function(self.import_h5_file, "import_h5_file")
        self.server.register_function(self.load_h5_workspace, "load_h5_workspace")
        self.server.register_function(self.save_h5_workspace, "save_h5_workspace")

        # Object operations
        self.server.register_function(self.add_signal, "add_signal")
        self.server.register_function(self.add_image, "add_image")
        self.server.register_function(self.get_object_titles, "get_object_titles")
        self.server.register_function(self.get_object_uuids, "get_object_uuids")
        self.server.register_function(self.get_object, "get_object")
        self.server.register_function(self.get_object_shapes, "get_object_shapes")
        self.server.register_function(self.delete_metadata, "delete_metadata")

        # Selection operations
        self.server.register_function(self.select_objects, "select_objects")
        self.server.register_function(self.select_groups, "select_groups")
        self.server.register_function(self.get_sel_object_uuids, "get_sel_object_uuids")

        # Group operations
        self.server.register_function(self.add_group, "add_group")
        self.server.register_function(
            self.get_group_titles_with_object_info, "get_group_titles_with_object_info"
        )

        # Calculation operations
        self.server.register_function(self.calc, "calc")

        # Annotation operations
        self.server.register_function(
            self.add_annotations_from_items, "add_annotations_from_items"
        )
        self.server.register_function(self.add_label_with_title, "add_label_with_title")

    # Basic server methods
    def get_version(self) -> str:
        """Get DataLab version.

        Returns a valid PEP 440 version string for testing purposes.
        Since this is a stub server, we return "1.0.0" which is compliant
        with both PEP 440 and the minimum version requirement.
        """
        return "1.0.0"

    def close_application(self) -> None:
        """Close DataLab application."""
        # In stub mode, do nothing

    def raise_window(self) -> None:
        """Raise DataLab window."""
        # In stub mode, do nothing

    # Panel management
    def get_current_panel(self) -> str:
        """Get current panel name."""
        return self.current_panel

    def set_current_panel(self, panel: str) -> None:
        """Set current panel."""
        if panel in ("signal", "image", "macro"):
            self.current_panel = panel

    # Application control
    def reset_all(self) -> None:
        """Reset all data."""
        self.signals.clear()
        self.images.clear()
        self.selected_objects.clear()
        self.selected_groups.clear()

    def toggle_auto_refresh(self, state: bool) -> None:
        """Toggle auto refresh mode."""
        self.auto_refresh = state
        if self.verbose:
            execenv.print(f"[STUB] Auto-refresh set to: {state}")

    def toggle_show_titles(self, state: bool) -> None:
        """Toggle show titles mode."""
        self.show_titles = state
        if self.verbose:
            execenv.print(f"[STUB] Show titles set to: {state}")

    # File operations
    def save_to_h5_file(self, filename: str) -> None:
        """Save to a DataLab HDF5 file."""
        if self.verbose:
            execenv.print(f"[STUB] Simulating H5 file save to: {filename}")
        # In stub mode, just create a dummy text file to simulate the save operation
        # This avoids HDF5 dependencies and potential test failures
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# DataLab stub file (for testing)\n")
                f.write(f"# Signals: {len(self.signals)}\n")
                f.write(f"# Images: {len(self.images)}\n")
                f.write("# This is a dummy file created by the stub server\n")
            if self.verbose:
                execenv.print(
                    f"[STUB] Successfully created dummy file with {len(self.signals)} "
                    f"signals and {len(self.images)} images"
                )
        except Exception as exc:  # pylint: disable=broad-except
            if self.verbose:
                execenv.print(f"[STUB] Failed to create dummy file: {exc}")
            # Ignore errors in stub mode

    # pylint: disable=unused-argument
    def open_h5_files(
        self,
        h5files: list[str] | None = None,
        import_all: bool | None = None,
        reset_all: bool | None = None,
    ) -> None:
        """Open a DataLab HDF5 file or import from any other HDF5 file."""
        if h5files is None:
            return

        if self.verbose:
            execenv.print(f"[STUB] Simulating H5 file loading: {h5files}")

        if reset_all:
            if self.verbose:
                execenv.print("[STUB] Resetting all data before loading")
            self.reset_all()

        # In stub mode, just simulate loading by creating dummy objects
        # This avoids complex HDF5 file parsing and potential test failures
        for _i, filename in enumerate(h5files):
            # Create a dummy signal for each file
            signal = create_signal(f"Loaded Signal from {os.path.basename(filename)}")
            self.signals[str(uuid.uuid4())] = signal
            if self.verbose:
                execenv.print(f"[STUB] Created dummy signal: {signal.title}")

            # Create a dummy image for each file
            image = create_image(f"Loaded Image from {os.path.basename(filename)}")
            self.images[str(uuid.uuid4())] = image
            if self.verbose:
                execenv.print(f"[STUB] Created dummy image: {image.title}")

    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file."""
        self.open_h5_files([filename], import_all=True, reset_all=reset_all)

    def load_h5_workspace(self, h5files: list[str], reset_all: bool = True) -> None:
        """Load HDF5 workspace files without showing file dialog.

        This is a headless version that bypasses Qt file dialogs.

        Args:
            h5files: List of HDF5 file paths to load
            reset_all: If True (default), reset workspace before loading
        """
        if self.verbose:
            execenv.print(f"[STUB] load_h5_workspace: {h5files}, reset_all={reset_all}")

        if reset_all:
            self.reset_all()

        # Simulate loading by creating dummy objects for each file
        for filename in h5files:
            signal = create_signal(f"Loaded Signal from {os.path.basename(filename)}")
            self.signals[str(uuid.uuid4())] = signal
            image = create_image(f"Loaded Image from {os.path.basename(filename)}")
            self.images[str(uuid.uuid4())] = image
            if self.verbose:
                execenv.print(f"[STUB] Created dummy signal and image for: {filename}")

    def save_h5_workspace(self, filename: str) -> None:
        """Save workspace to HDF5 file without showing file dialog.

        This is a headless version that bypasses Qt file dialogs.

        Args:
            filename: Path to the output HDF5 file
        """
        if self.verbose:
            execenv.print(f"[STUB] save_h5_workspace: {filename}")

        # Create a dummy file to simulate save
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# DataLab workspace stub file (for testing)\n")
                f.write(f"# Signals: {len(self.signals)}\n")
                f.write(f"# Images: {len(self.images)}\n")
            if self.verbose:
                execenv.print(
                    f"[STUB] Successfully saved workspace with {len(self.signals)} "
                    f"signals and {len(self.images)} images"
                )
        except Exception as exc:  # pylint: disable=broad-except
            if self.verbose:
                execenv.print(f"[STUB] Failed to save workspace: {exc}")

    # Object operations
    # pylint: disable=unused-argument
    def add_signal(
        self,
        title: str,
        xbinary: Binary,
        ybinary: Binary,
        xunit: str = "",
        yunit: str = "",
        xlabel: str = "",
        ylabel: str = "",
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:
        """Add signal data to DataLab."""
        xdata = utils.rpcbinary_to_array(xbinary)
        ydata = utils.rpcbinary_to_array(ybinary)

        # Create real SignalObj using factory function
        signal = create_signal(title, x=xdata, y=ydata)
        signal.xunit = xunit or ""
        signal.yunit = yunit or ""
        signal.xlabel = xlabel or ""
        signal.ylabel = ylabel or ""

        # Store signal
        obj_uuid = str(uuid.uuid4())
        self.signals[obj_uuid] = signal

        # Add to group if specified
        if group_id and group_id in self.signal_groups:
            self.signal_groups[group_id].objects.append(obj_uuid)

        return True

    # pylint: disable=unused-argument
    def add_image(
        self,
        title: str,
        zbinary: Binary,
        xunit: str = "",
        yunit: str = "",
        zunit: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:
        """Add image data to DataLab."""
        data = utils.rpcbinary_to_array(zbinary)

        # Create real ImageObj using factory function
        image = create_image(title, data=data)
        image.xunit = xunit or ""
        image.yunit = yunit or ""
        image.zunit = zunit or ""
        image.xlabel = xlabel or ""
        image.ylabel = ylabel or ""
        image.zlabel = zlabel or ""

        # Store image
        obj_uuid = str(uuid.uuid4())
        self.images[obj_uuid] = image

        # Add to group if specified
        if group_id and group_id in self.image_groups:
            self.image_groups[group_id].objects.append(obj_uuid)

        return True

    def add_object(
        self, obj_data: list[str], group_id: str = "", set_current: bool = True
    ) -> bool:
        """Add object to stub server.

        Args:
            obj_serialized: Serialized signal or image object
            group_id: group id in which to add the object. Defaults to ""
            set_current: if True, set the added object as current
        """
        obj: SignalObj | ImageObj = utils.rpcjson_to_dataset(obj_data)
        if self.verbose:
            obj_str = "signal" if isinstance(obj, SignalObj) else "image"
            obj_uuid = str(uuid.uuid4())
            print(f"Added {obj_str} {obj.title} with UUID {obj_uuid}")
            if isinstance(obj, SignalObj):
                self.signals[obj_uuid] = obj
                if group_id and group_id in self.signal_groups:
                    self.signal_groups[group_id].objects.append(obj_uuid)
            else:
                self.images[obj_uuid] = obj
                if group_id and group_id in self.image_groups:
                    self.image_groups[group_id].objects.append(obj_uuid)

    def load_from_files(self, filenames: list[str]) -> None:
        """Load objects from files (stub implementation).

        Args:
            filenames: list of file names
        """
        if self.verbose:
            print(
                f"load_from_files called with {len(filenames)} files "
                "(stub - not implemented)"
            )

    def load_from_directory(self, path: str) -> None:
        """Load objects from directory (stub implementation).

        Args:
            path: directory path
        """
        if self.verbose:
            print(
                f"load_from_directory called with path: {path} (stub - not implemented)"
            )

    def get_object_titles(self, panel: str | None = None) -> list[str]:
        """Get object titles for panel."""
        panel = panel or self.current_panel
        if panel == "signal":
            return [signal.title for signal in self.signals.values()]
        if panel == "image":
            return [image.title for image in self.images.values()]
        return []

    # pylint: disable=unused-argument
    def get_object_uuids(
        self, panel: str | None = None, group: int | str | None = None
    ) -> list[str]:
        """Get object UUIDs for panel."""
        panel = panel or self.current_panel
        if panel == "signal":
            return list(self.signals.keys())
        if panel == "image":
            return list(self.images.keys())
        return []

    def get_object(self, uuid_str: str, panel: str | None = None) -> list[str] | None:
        """Get object by UUID, index, or title."""
        panel = panel or self.current_panel

        # Get the appropriate objects dictionary
        if panel == "signal":
            objects = self.signals
            object_list = list(self.signals.keys())
        elif panel == "image":
            objects = self.images
            object_list = list(self.images.keys())
        else:
            return None

        # Try to resolve uuid_str as UUID first
        if uuid_str in objects:
            obj = objects[uuid_str]
        else:
            # Try to resolve as 1-based index
            try:
                index = int(uuid_str) - 1  # Convert to 0-based
                if 0 <= index < len(object_list):
                    uuid_key = object_list[index]
                    obj = objects[uuid_key]
                else:
                    return None
            except (ValueError, TypeError):
                # Try to find by title
                obj = None
                for object_instance in objects.values():
                    if object_instance.title == uuid_str:
                        obj = object_instance
                        break
                if obj is None:
                    return None

        # Use standard serialization with real objects
        return utils.dataset_to_rpcjson(obj)

    # pylint: disable=unused-argument
    def get_object_shapes(
        self, uuid_str: str, panel: str | None = None
    ) -> list[dict] | None:
        """Get object shapes."""
        obj = self.signals.get(uuid_str) or self.images.get(uuid_str)
        if obj is None:
            return None
        return [roi.to_dict() for roi in obj.roi]

    def delete_metadata(self, uuid_str: str, key: str) -> bool:
        """Delete metadata entry for object."""
        obj = self.signals.get(uuid_str) or self.images.get(uuid_str)
        if obj is None:
            return False
        if key in obj.metadata:
            del obj.metadata[key]
            return True
        return False

    # Selection operations
    def select_objects(
        self, selection: list[int | str], panel: str | None = None
    ) -> None:
        """Select objects by indices or UUIDs."""
        panel = panel or self.current_panel
        if panel == "signal":
            uuids = list(self.signals.keys())
        elif panel == "image":
            uuids = list(self.images.keys())
        else:
            return

        selected_uuids = []
        for item in selection:
            if isinstance(item, str):
                # Item is a UUID
                if item in uuids:
                    selected_uuids.append(item)
            elif isinstance(item, int):
                # Item is a 1-based index
                index = item - 1  # Convert to 0-based
                if 0 <= index < len(uuids):
                    selected_uuids.append(uuids[index])

        self.selected_objects = selected_uuids

    # pylint: disable=unused-argument
    def get_sel_object_uuids(self, include_groups: bool = False) -> list[str]:
        """Return selected objects uuids."""
        return self.selected_objects.copy()

    def select_groups(self, selection: list[int], panel: str | None = None) -> None:
        """Select groups by indices."""
        panel = panel or self.current_panel
        if panel == "signal":
            group_uuids = list(self.signal_groups.keys())
        elif panel == "image":
            group_uuids = list(self.image_groups.keys())
        else:
            return

        self.selected_groups = [
            group_uuids[i] for i in selection if 0 <= i < len(group_uuids)
        ]

    # Group operations
    # pylint: disable=unused-argument
    def add_group(
        self, title: str, panel: str | None = None, select: bool = False
    ) -> str:
        """Add group and return UUID."""
        panel = panel or self.current_panel
        group = MockGroup(title)

        if panel == "signal":
            self.signal_groups[group.uuid] = group
        elif panel == "image":
            self.image_groups[group.uuid] = group

        return group.uuid

    def get_group_titles_with_object_info(
        self,
    ) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """Return groups titles and lists of inner objects uuids and titles."""
        panel = self.current_panel
        if panel == "signal":
            groups = self.signal_groups
            objects = self.signals
        elif panel == "image":
            groups = self.image_groups
            objects = self.images
        else:
            return ([], [], [])

        group_titles = []
        group_uuids_lists = []
        group_titles_lists = []

        for group in groups.values():
            group_titles.append(group.title)

            # Get objects in this group
            object_uuids = [uuid for uuid in group.objects if uuid in objects]
            object_titles = [
                objects[uuid].title for uuid in object_uuids if uuid in objects
            ]

            group_uuids_lists.append(object_uuids)
            group_titles_lists.append(object_titles)

        return (group_titles, group_uuids_lists, group_titles_lists)

    # Macro operations (stub implementations)
    def import_macro_from_file(self, filename: str) -> None:
        """Import macro from file (stub implementation).

        Args:
            filename: Filename
        """
        if self.verbose:
            print(f"import_macro_from_file called: {filename} (stub - not implemented)")

    def run_macro(self, number_or_title: int | str | None = None) -> None:
        """Run macro (stub implementation).

        Args:
            number_or_title: Number or title of the macro. Defaults to None.
        """
        if self.verbose:
            print(f"run_macro called: {number_or_title} (stub - not implemented)")

    def stop_macro(self, number_or_title: int | str | None = None) -> None:
        """Stop macro (stub implementation).

        Args:
            number_or_title: Number or title of the macro. Defaults to None.
        """
        if self.verbose:
            print(f"stop_macro called: {number_or_title} (stub - not implemented)")

    # Calculation operations
    def calc(self, name: str, param: list[str] | None = None) -> str | None:
        """Execute calculation and return result object UUID."""
        if self.verbose:
            execenv.print(
                f"[STUB] Simulating calculation '{name}' with params: {param}"
            )

        # In stub mode, just simulate by creating a dummy result object
        if not self.selected_objects:
            if self.verbose:
                execenv.print("[STUB] No objects selected for calculation")
            return None

        src_uuid = self.selected_objects[0]
        src_obj = self.signals.get(src_uuid) or self.images.get(src_uuid)

        if src_obj is None:
            if self.verbose:
                execenv.print(f"[STUB] Source object {src_uuid} not found")
            return None

        # Create a dummy result object based on source type
        if isinstance(src_obj, SignalObj):
            result = create_signal(f"{name}({src_obj.title})")
            obj_uuid = str(uuid.uuid4())
            self.signals[obj_uuid] = result
            if self.verbose:
                execenv.print(f"[STUB] Created dummy signal result: {result.title}")
            return obj_uuid
        if isinstance(src_obj, ImageObj):
            result = create_image(f"{name}({src_obj.title})")
            obj_uuid = str(uuid.uuid4())
            self.images[obj_uuid] = result
            if self.verbose:
                execenv.print(f"[STUB] Created dummy image result: {result.title}")
            return obj_uuid

        if self.verbose:
            execenv.print("[STUB] Unsupported object type for calculation")
        return None

    # Annotation operations
    # pylint: disable=unused-argument
    def add_annotations_from_items(
        self, items: list[str], refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items)."""
        # In stub mode, just acknowledge the annotations
        # Real implementation would deserialize items and add to objects

    # pylint: disable=unused-argument
    def add_label_with_title(
        self, title: str | None = None, panel: str | None = None
    ) -> None:
        """Add label with title."""
        if self.verbose:
            execenv.print(f"[STUB] Simulating add label with title: {title}")
        # In stub mode, just acknowledge the label


@contextmanager
def datalab_stub_server(
    port: int = 0, verbose: bool = True
) -> Generator[int, None, None]:
    """Context manager for DataLab stub server.

    Args:
        port: Port to bind to. If 0, uses a random available port.
        verbose: If True, print verbose debug information.

    Yields:
        Port number the server is listening on
    """
    server = DataLabStubServer(port, verbose=verbose)
    try:
        actual_port = server.start()
        yield actual_port
    finally:
        server.stop()


def patch_simpleremoteproxy_for_stub() -> DataLabStubServer:
    """Patch SimpleRemoteProxy to connect to a stub server instead of real DataLab.

    This utility function:
    1. Creates and starts a DataLabStubServer instance
    2. Patches SimpleRemoteProxy.__connect_to_server to use the stub server port
    3. Returns the stub server instance for later cleanup

    Returns:
        The running DataLabStubServer instance that needs to be stopped later

    Example:
        >>> stub_server = patch_simpleremoteproxy_for_stub()
        >>> try:
        ...     # Your code using SimpleRemoteProxy here
        ...     proxy = SimpleRemoteProxy()
        ...     # proxy will connect to stub server automatically
        ... finally:
        ...     stub_server.stop()
    """
    # pylint: disable=import-outside-toplevel
    # Import directly from remote module to avoid circular dependency
    from sigima.client.remote import SimpleRemoteProxy

    # Store original method
    # pylint: disable=protected-access
    original_connect_to_server = SimpleRemoteProxy._SimpleRemoteProxy__connect_to_server

    # Start stub server
    stub_server_instance = DataLabStubServer(verbose=False)
    stub_port = stub_server_instance.start()

    # pylint: disable=unused-argument
    def patched_connect_to_server(self, port=None):
        """Patched connect that uses stub server port."""
        # Always use the stub server port, ignore the requested port
        original_connect_to_server(self, port=str(stub_port))

    # Apply the patch
    # pylint: disable=protected-access
    SimpleRemoteProxy._SimpleRemoteProxy__connect_to_server = patched_connect_to_server

    return stub_server_instance
