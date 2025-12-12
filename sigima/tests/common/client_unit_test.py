# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima Client Comprehensive Headless Test

This test uses a stub XML-RPC server to emulate DataLab, allowing the test
to run without requiring a real DataLab instance.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: skip

from __future__ import annotations

import os.path as osp
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import pytest
from guidata.env import execenv

from sigima.client.remote import SimpleRemoteProxy
from sigima.client.stub import datalab_stub_server


@contextmanager
def temporary_directory() -> Generator[str, None, None]:
    """Create a temporary directory and clean-up afterwards"""
    tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    try:
        yield tmp.name
    finally:
        try:
            tmp.cleanup()
        except (PermissionError, RecursionError):
            pass


class RemoteClientTester:
    """Headless remote client tester class"""

    SIG_TITLES = ("Oscilloscope", "Digitizer", "Radiometer", "Voltmeter", "Sensor")
    IMA_TITLES = (
        "Camera",
        "Streak Camera",
        "Image Scanner",
        "Laser Beam Profiler",
        "Gated Imaging Camera",
    )

    def __init__(self):
        """Initialize the tester"""
        self.datalab = None
        self.log_messages = []
        self.stub_server_port = None

    def log(self, message: str) -> None:
        """Log message for debugging"""
        self.log_messages.append(message)
        execenv.print(f"[CLIENT] {message}")

    def init_cdl_with_stub(self, port: int) -> bool:
        """Initialize DataLab connection with stub server"""
        try:
            self.stub_server_port = port
            self.datalab = SimpleRemoteProxy(autoconnect=False)
            self.datalab.connect(port=str(port), timeout=1.0, retries=1)
            self.log("✨ Initialized DataLab connection with stub server ✨")
            self.log(f"  Communication port: {self.datalab.port}")

            # Test getting method list
            methods = self.datalab.get_method_list()
            self.log(f"  Available methods: {len(methods)} found")
            return True

        except ConnectionRefusedError:
            self.log("🔥 Connection refused 🔥 (Stub server is not ready)")
            return False

    def init_cdl(self, port: str | None = None) -> bool:
        """Initialize DataLab connection

        Args:
            port: Port to connect to (if None, uses default)
        """
        try:
            self.datalab = SimpleRemoteProxy(autoconnect=False)
            self.datalab.connect(port=port, timeout=1.0, retries=1)
            self.log("✨ Initialized DataLab connection ✨")
            self.log(f"  Communication port: {self.datalab.port}")

            # Test getting method list
            methods = self.datalab.get_method_list()
            self.log(f"  Available methods: {len(methods)} found")
            return True

        except ConnectionRefusedError:
            self.log("🔥 Connection refused 🔥 (DataLab server is not ready)")
            return False

    def close_datalab(self) -> None:
        """Close DataLab connection"""
        if self.datalab is not None:
            try:
                self.datalab.close_application()
                self.log("🎬 Closed DataLab!")
            except ConnectionRefusedError:
                self.log("Connection lost while closing DataLab")
            finally:
                self.datalab = None

    def test_connection_management(self) -> None:
        """Test connection initialization and method listing"""
        # If we already have a connection (from stub server), skip init
        if self.datalab is None:
            assert self.init_cdl(), "Failed to initialize DataLab connection"

        # Test method listing
        methods = self.datalab.get_method_list()
        assert isinstance(methods, list), "Method list should be a list"
        assert len(methods) > 0, "Should have at least some methods available"

        # Test basic server info
        version = self.datalab.get_version()
        assert isinstance(version, str), "Version should be a string"
        self.log(f"DataLab version: {version}")

    def add_test_signals(self) -> None:
        """Add test signals to DataLab"""
        if self.datalab is None:
            return

        x = np.linspace(0, 10, 1000)
        signals_data = [
            ("Sine", np.sin(x)),
            ("Cosine", np.cos(x)),
            ("Exponential", np.exp(-x / 5)),
        ]

        for title, y in signals_data:
            success = self.datalab.add_signal(title, x, y)
            assert success, f"Failed to add signal: {title}"
            self.log(f"Added signal: {title}")

    def add_test_images(self) -> None:
        """Add test images to DataLab"""
        if self.datalab is None:
            return

        images_data = [
            ("Zeros", np.zeros((100, 100))),
            ("Ones", np.ones((100, 100))),
            ("Random", np.random.random((100, 100))),
        ]

        for title, z in images_data:
            success = self.datalab.add_image(title, z)
            assert success, f"Failed to add image: {title}"
            self.log(f"Added image: {title}")

    def test_object_management(self) -> None:
        """Test object listing, retrieval, and manipulation"""
        if self.datalab is None:
            return

        # Test with signals
        self.datalab.set_current_panel("signal")
        assert self.datalab.get_current_panel() == "signal"

        titles = self.datalab.get_object_titles()
        self.log(f"Signal titles: {titles}")
        assert isinstance(titles, list), "Titles should be a list"

        uuids = self.datalab.get_object_uuids()
        self.log(f"Signal UUIDs: {uuids}")
        assert isinstance(uuids, list), "UUIDs should be a list"
        assert len(titles) == len(uuids), "Should have equal number of titles and UUIDs"

        if titles:
            # Test getting first object
            obj = self.datalab.get_object(1)  # Get first object
            assert obj is not None, "Should be able to retrieve first object"
            assert hasattr(obj, "title"), "Object should have title attribute"
            self.log(f"Retrieved object: {obj.title}")

            # Test getting object by UUID
            first_uuid = uuids[0]
            obj_by_uuid = self.datalab.get_object(first_uuid)
            assert obj_by_uuid is not None, "Should retrieve object by UUID"
            assert obj_by_uuid.title == obj.title, "Objects should be the same"

        # Test with images
        self.datalab.set_current_panel("image")
        assert self.datalab.get_current_panel() == "image"

        img_titles = self.datalab.get_object_titles()
        self.log(f"Image titles: {img_titles}")

        if img_titles:
            img_obj = self.datalab.get_object(1)
            assert img_obj is not None, "Should retrieve first image"
            self.log(f"Retrieved image: {img_obj.title}")

    def test_selection_operations(self) -> None:
        """Test object selection operations"""
        if self.datalab is None:
            return

        # Test selecting objects
        self.datalab.set_current_panel("signal")
        uuids = self.datalab.get_object_uuids()

        if uuids:
            # Select first object
            self.datalab.select_objects([uuids[0]])
            selected = self.datalab.get_sel_object_uuids()
            assert uuids[0] in selected, "Should have selected the object"
            self.log(f"Selected object: {uuids[0]}")

            # Test selecting multiple objects if available
            if len(uuids) > 1:
                self.datalab.select_objects([uuids[0], uuids[1]])
                selected = self.datalab.get_sel_object_uuids()
                assert len(selected) == 2, "Should have selected 2 objects"

    def test_annotations_and_shapes(self) -> None:
        """Test annotation and shape operations"""
        if self.datalab is None:
            return

        # pylint: disable=import-outside-toplevel
        from plotpy.builder import make

        # Test with images (annotations are more meaningful for images)
        self.datalab.set_current_panel("image")
        uuids = self.datalab.get_object_uuids()

        if uuids:
            # Add an annotation
            rect = make.annotated_rectangle(10, 10, 50, 50, title="Test Rectangle")
            self.datalab.add_annotations_from_items([rect])
            self.log("Added annotation rectangle")

            # Retrieve shapes
            shapes = self.datalab.get_object_shapes()
            assert isinstance(shapes, list), "Shapes should be a list"
            self.log(f"Retrieved {len(shapes)} shapes")

            # Add label
            self.datalab.add_label_with_title("Test Label")
            self.log("Added label with title")

    def test_file_operations(self) -> None:
        """Test file save/load operations"""
        if self.datalab is None:
            return

        with temporary_directory() as tmpdir:
            # Save to HDF5 file
            fname = osp.join(tmpdir, "test_remote.h5")
            self.datalab.save_to_h5_file(fname)
            self.log(f"Saved data to: {fname}")
            assert osp.exists(fname), "HDF5 file should exist"

            # Clear all data
            self.datalab.reset_all()
            self.log("Reset all data")

            # Verify data is cleared
            titles = self.datalab.get_object_titles("signal")
            assert len(titles) == 0, "Signal panel should be empty after reset"

            # Reload from file
            self.datalab.open_h5_files([fname], import_all=True, reset_all=False)
            self.log("Reloaded data from HDF5 file")

            # Verify data is restored
            titles = self.datalab.get_object_titles("signal")
            assert len(titles) > 0, "Should have signals after reload"

    def test_workspace_headless_api(self) -> None:
        """Test load_h5_workspace and save_h5_workspace headless API (Issue #275)"""
        if self.datalab is None:
            return

        with temporary_directory() as tmpdir:
            # First add some data
            self.add_test_signals()
            self.add_test_images()

            # Save workspace using headless API
            fname = osp.join(tmpdir, "test_workspace.h5")
            self.datalab.save_h5_workspace(fname)
            self.log(f"Saved workspace to: {fname}")
            assert osp.exists(fname), "Workspace file should exist"

            # Clear all data
            self.datalab.reset_all()
            titles = self.datalab.get_object_titles("signal")
            assert len(titles) == 0, "Signal panel should be empty after reset"

            # Load workspace using headless API
            self.datalab.load_h5_workspace([fname], reset_all=True)
            self.log("Loaded workspace from file")

            # Verify data is restored
            titles = self.datalab.get_object_titles("signal")
            assert len(titles) > 0, "Should have signals after load_h5_workspace"
            self.log(f"Workspace loaded with {len(titles)} signals")

            # Test with single file path (string instead of list)
            self.datalab.reset_all()
            self.datalab.load_h5_workspace(fname, reset_all=True)
            titles = self.datalab.get_object_titles("signal")
            assert len(titles) > 0, "Should work with single file path string"
            self.log("load_h5_workspace works with single file path")

    def test_computation_operations(self) -> None:
        """Test computation operations"""
        if self.datalab is None:
            return

        # Test signal computations
        self.datalab.set_current_panel("signal")
        uuids = self.datalab.get_object_uuids()

        if uuids:
            # Select first signal
            self.datalab.select_objects([uuids[0]])

            # Test some basic computations
            try:
                self.datalab.calc("log10")
                self.log("Applied log10 computation")
            except Exception as exc:  # pylint: disable=broad-except
                self.log(f"log10 computation failed: {exc}")

            try:
                self.datalab.calc("fft")
                self.log("Applied FFT computation")
            except Exception as exc:  # pylint: disable=broad-except
                self.log(f"FFT computation failed: {exc}")

    def test_group_operations(self) -> None:
        """Test group operations"""
        if self.datalab is None:
            return

        # Add a group
        self.datalab.add_group("Test Group", panel="signal")
        self.log("Added test group")

        # Get group information
        group_info = self.datalab.get_group_titles_with_object_info()
        assert isinstance(group_info, (list, tuple)), (
            "Group info should be a list or tuple"
        )
        assert len(group_info) == 3, "Should have 3 elements (titles, uuids, titles)"
        self.log(f"Groups: {group_info[0]}")

    def test_metadata_operations(self) -> None:
        """Test metadata operations"""
        if self.datalab is None:
            return

        uuids = self.datalab.get_object_uuids()
        if uuids:
            # Select an object
            self.datalab.select_objects([uuids[0]])

            # Delete metadata
            self.datalab.delete_metadata(refresh_plot=False, keep_roi=False)
            self.log("Deleted metadata")

    def run_comprehensive_test(self) -> None:
        """Run all tests in sequence"""
        self.log("Starting comprehensive remote client test")

        try:
            # Basic connection test
            self.test_connection_management()

            # Add test data
            self.add_test_signals()
            self.add_test_images()

            # Test object management
            self.test_object_management()

            # Test selection operations
            self.test_selection_operations()

            # Test annotations and shapes
            try:
                self.test_annotations_and_shapes()
            except ImportError:
                self.log("PlotPy not available, skipping annotations and shapes test")

            # Test computations
            self.test_computation_operations()

            # Test group operations
            self.test_group_operations()

            # Test metadata operations
            self.test_metadata_operations()

            # Test file operations (this will reset data)
            self.test_file_operations()

            # Test workspace headless API (Issue #275)
            self.test_workspace_headless_api()

            self.log("✅ All tests completed successfully!")

        finally:
            # Always try to clean up
            try:
                self.datalab.reset_all()
                self.log("Final cleanup: reset all data")
            except Exception:  # pylint: disable=broad-except
                pass


def test_comprehensive_remote_client():
    """Comprehensive remote client test (pytest version)"""
    # First try with stub server (always available)
    with datalab_stub_server() as port:
        tester = RemoteClientTester()
        if tester.init_cdl_with_stub(port):
            try:
                tester.run_comprehensive_test()
                return  # Test passed with stub server
            finally:
                tester.close_datalab()

    # If stub server test failed, try with real DataLab
    tester = RemoteClientTester()
    if not tester.init_cdl():
        pytest.skip("Neither stub server nor real DataLab server is available")

    try:
        tester.run_comprehensive_test()
    finally:
        tester.close_datalab()


if __name__ == "__main__":
    test_comprehensive_remote_client()
