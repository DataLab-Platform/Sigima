# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Client base proxy module
--------------------------------
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING

import guidata.dataset as gds
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sigima.client.remote import ServerProxy
    from sigima.objects import ImageObj, SignalObj


class SimpleAbstractDLControl(abc.ABC):
    """Simple abstract base class for controlling DataLab

    This is a subset of DataLab's AbstractDLControl, with only the methods that do not
    require DataLab object model to be implemented."""

    def __len__(self) -> int:
        """Return number of objects"""
        return len(self.get_object_uuids())

    def __getitem__(
        self,
        nb_id_title: int | str | None = None,
    ) -> SignalObj | ImageObj:
        """Return object"""
        return self.get_object(nb_id_title)

    def __iter__(self) -> Iterator[SignalObj | ImageObj]:
        """Iterate over objects"""
        uuids = self.get_object_uuids()
        for uuid in uuids:
            yield self.get_object(uuid)

    def __str__(self) -> str:
        """Return object string representation"""
        return super().__repr__()

    def __repr__(self) -> str:
        """Return object representation"""
        titles = self.get_object_titles()
        uuids = self.get_object_uuids()
        text = f"{str(self)} (DataLab, {len(titles)} items):\n"
        for uuid, title in zip(uuids, titles):
            text += f"  {uuid}: {title}\n"
        return text

    def __bool__(self) -> bool:
        """Return True if model is not empty"""
        return bool(self.get_object_uuids())

    def __contains__(self, id_title: str) -> bool:
        """Return True if object (UUID or title) is in model"""
        return id_title in (self.get_object_titles() + self.get_object_uuids())

    @classmethod
    def get_public_methods(cls) -> list[str]:
        """Return all public methods of the class, except itself.

        Returns:
            list[str]: List of public methods
        """
        return [
            method
            for method in dir(cls)
            if not method.startswith("_") and method != "get_public_methods"
        ]

    @abc.abstractmethod
    def get_version(self) -> str:
        """Return DataLab version.

        Returns:
            str: DataLab version
        """

    @abc.abstractmethod
    def close_application(self) -> None:
        """Close DataLab application"""

    @abc.abstractmethod
    def raise_window(self) -> None:
        """Raise DataLab window"""

    @abc.abstractmethod
    def get_current_panel(self) -> str:
        """Return current panel name.

        Returns:
            str: Panel name (valid values: "signal", "image", "macro"))
        """

    @abc.abstractmethod
    def set_current_panel(self, panel: str) -> None:
        """Switch to panel.

        Args:
            panel (str): Panel name (valid values: "signal", "image", "macro"))
        """

    @abc.abstractmethod
    def reset_all(self) -> None:
        """Reset all application data"""

    @abc.abstractmethod
    def toggle_auto_refresh(self, state: bool) -> None:
        """Toggle auto refresh state.

        Args:
            state (bool): Auto refresh state
        """

    @abc.abstractmethod
    def toggle_show_titles(self, state: bool) -> None:
        """Toggle show titles state.

        Args:
            state (bool): Show titles state
        """

    @abc.abstractmethod
    def save_to_h5_file(self, filename: str) -> None:
        """Save to a DataLab HDF5 file.

        Args:
            filename (str): HDF5 file name
        """

    @abc.abstractmethod
    def open_h5_files(
        self,
        h5files: list[str] | None = None,
        import_all: bool | None = None,
        reset_all: bool | None = None,
    ) -> None:
        """Open a DataLab HDF5 file or import from any other HDF5 file.

        Args:
            h5files (list[str] | None): List of HDF5 files to open. Defaults to None.
            import_all (bool | None): Import all objects from HDF5 files.
                Defaults to None.
            reset_all (bool | None): Reset all application data. Defaults to None.
        """

    @abc.abstractmethod
    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file.

        Args:
            filename (str): HDF5 file name
            reset_all (bool | None): Reset all application data. Defaults to None.
        """

    @abc.abstractmethod
    def load_h5_workspace(
        self, h5files: list[str] | str, reset_all: bool = True
    ) -> None:
        """Load HDF5 workspace files without showing file dialog.

        This method loads one or more DataLab native HDF5 files directly, bypassing
        the file dialog. It is safe to call from the internal console or any context
        where Qt dialogs would cause threading issues.

        Args:
            h5files: Path(s) to HDF5 file(s). Can be a single path string or a list
             of paths.
            reset_all: If True (default), reset workspace before loading.
             If False, append to existing workspace.

        Raises:
            ValueError: if file is not a DataLab native HDF5 file
        """

    @abc.abstractmethod
    def save_h5_workspace(self, filename: str) -> None:
        """Save workspace to HDF5 file without showing file dialog.

        This method saves the current workspace to a DataLab native HDF5 file
        directly, bypassing the file dialog. It is safe to call from the internal
        console or any context where Qt dialogs would cause threading issues.

        Args:
            filename: Path to the output HDF5 file
        """

    @abc.abstractmethod
    def load_from_files(self, filenames: list[str]) -> None:
        """Open objects from files in current panel (signals/images).

        Args:
            filenames: list of file names
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
    def add_object(
        self, obj: SignalObj | ImageObj, group_id: str = "", set_current: bool = True
    ) -> None:
        """Add object to DataLab.

        Args:
            obj: Signal or image object
            group_id: group id in which to add the object. Defaults to ""
            set_current: if True, set the added object as current

        Returns:
            True if object was added successfully, False otherwise
        """

    @abc.abstractmethod
    def load_from_directory(self, path: str) -> None:
        """Open objects from directory in current panel (signals/images).

        Args:
            path: directory path
        """

    @abc.abstractmethod
    def add_group(
        self, title: str, panel: str | None = None, select: bool = False
    ) -> None:
        """Add group to DataLab.

        Args:
            title: Group title
            panel: Panel name (valid values: "signal", "image"). Defaults to None.
            select: Select the group after creation. Defaults to False.
        """

    @abc.abstractmethod
    def get_sel_object_uuids(self, include_groups: bool = False) -> list[str]:
        """Return selected objects uuids.

        Args:
            include_groups: If True, also return objects from selected groups.

        Returns:
            List of selected objects uuids.
        """

    @abc.abstractmethod
    def select_objects(
        self,
        selection: list[int | str],
        panel: str | None = None,
    ) -> None:
        """Select objects in current panel.

        Args:
            selection: List of object numbers (1 to N) or uuids to select
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used. Defaults to None.
        """

    @abc.abstractmethod
    def select_groups(
        self, selection: list[int | str] | None = None, panel: str | None = None
    ) -> None:
        """Select groups in current panel.

        Args:
            selection: List of group numbers (1 to N), or list of group uuids,
             or None to select all groups. Defaults to None.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used. Defaults to None.
        """

    @abc.abstractmethod
    def delete_metadata(
        self, refresh_plot: bool = True, keep_roi: bool = False
    ) -> None:
        """Delete metadata of selected objects

        Args:
            refresh_plot: Refresh plot. Defaults to True.
            keep_roi: Keep ROI. Defaults to False.
        """

    @abc.abstractmethod
    def get_group_titles_with_object_info(
        self,
    ) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """Return groups titles and lists of inner objects uuids and titles.

        Returns:
            Tuple: groups titles, lists of inner objects uuids and titles
        """

    @abc.abstractmethod
    def get_object_titles(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image", "macro").
             If None, current data panel is used (i.e. signal or image panel).

        Returns:
            List of object titles

        Raises:
            ValueError: if panel not found
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def get_object_uuids(
        self, panel: str | None = None, group: int | str | None = None
    ) -> list[str]:
        """Get object (signal/image) uuid list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
            group: Group number, or group id, or group title.
             Defaults to None (all groups).

        Returns:
            List of object uuids

        Raises:
            ValueError: if panel not found
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def add_annotations_from_items(
        self, items: list, refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items (list): annotation plot items
            refresh_plot (bool | None): refresh plot. Defaults to True.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.
        """

    @abc.abstractmethod
    def add_label_with_title(
        self, title: str | None = None, panel: str | None = None
    ) -> None:
        """Add a label with object title on the associated plot

        Args:
            title (str | None): Label title. Defaults to None.
                If None, the title is the object title.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.
        """

    @abc.abstractmethod
    def run_macro(self, number_or_title: int | str | None = None) -> None:
        """Run macro.

        Args:
             number: Number of the macro (starting at 1). Defaults to None (run
              current macro, or does nothing if there is no macro).
        """

    @abc.abstractmethod
    def stop_macro(self, number_or_title: int | str | None = None) -> None:
        """Stop macro.

        Args:
            number: Number of the macro (starting at 1). Defaults to None (stop
             current macro, or does nothing if there is no macro).
        """

    @abc.abstractmethod
    def import_macro_from_file(self, filename: str) -> None:
        """Import macro from file

        Args:
            filename: Filename.
        """

    @abc.abstractmethod
    def calc(self, name: str, param: gds.DataSet | None = None) -> None:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name: Compute function name
            param: Compute function parameter. Defaults to None.

        Raises:
            ValueError: unknown function
        """

    def __getattr__(self, name: str) -> Callable:
        """Return compute function ``name`` in current panel's processor.

        Args:
            name (str): Compute function name

        Returns:
            Callable: Compute function

        Raises:
            AttributeError: If compute function ``name`` does not exist
        """

        def compute_func(param: gds.DataSet | None = None) -> gds.DataSet:
            """Compute function.

            Args:
                param (guidata.dataset.DataSet | None): Compute function
                 parameter. Defaults to None.

            Returns:
                guidata.dataset.DataSet: Compute function result
            """
            return self.calc(name, param)

        if name.startswith("compute_"):
            return compute_func
        raise AttributeError(f"DataLab has no compute function '{name}'")


class SimpleBaseProxy(SimpleAbstractDLControl, metaclass=abc.ABCMeta):
    """Simple common base class for DataLab proxies

    This is a subset of DataLab's BaseProxy, with only the methods that do not require
    DataLab object model to be implemented.

    Args:
        datalab (DLMainWindow | ServerProxy | None): DLMainWindow instance or
         ServerProxy instance. If None, then the proxy implementation will
         have to set it later (e.g. see SimpleRemoteProxy).
    """

    def __init__(self, datalab: ServerProxy | None = None) -> None:
        self._datalab = datalab

    def get_version(self) -> str:
        """Return DataLab version.

        Returns:
            str: DataLab version
        """
        return self._datalab.get_version()

    def close_application(self) -> None:
        """Close DataLab application"""
        self._datalab.close_application()

    def raise_window(self) -> None:
        """Raise DataLab window"""
        self._datalab.raise_window()

    def get_current_panel(self) -> str:
        """Return current panel name.

        Returns:
            str: Panel name (valid values: "signal", "image", "macro"))
        """
        return self._datalab.get_current_panel()

    def set_current_panel(self, panel: str) -> None:
        """Switch to panel.

        Args:
            panel (str): Panel name (valid values: "signal", "image", "macro"))
        """
        self._datalab.set_current_panel(panel)

    def reset_all(self) -> None:
        """Reset all application data"""
        self._datalab.reset_all()

    def toggle_auto_refresh(self, state: bool) -> None:
        """Toggle auto refresh state.

        Args:
            state (bool): Auto refresh state
        """
        self._datalab.toggle_auto_refresh(state)

    # Returns a context manager to temporarily disable autorefresh
    def context_no_refresh(self) -> Callable:
        """Return a context manager to temporarily disable auto refresh.

        Returns:
            Context manager

        Example:

            >>> with proxy.context_no_refresh():
            ...     proxy.add_image("image1", data1)
            ...     proxy.compute_fft()
            ...     proxy.compute_wiener()
            ...     proxy.compute_ifft()
            ...     # Auto refresh is disabled during the above operations
        """

        class NoRefreshContextManager:
            """Context manager to temporarily disable auto refresh"""

            def __init__(self, datalab: SimpleAbstractDLControl) -> None:
                self._datalab = datalab

            def __enter__(self) -> None:
                self._datalab.toggle_auto_refresh(False)

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                self._datalab.toggle_auto_refresh(True)

        return NoRefreshContextManager(self)

    def toggle_show_titles(self, state: bool) -> None:
        """Toggle show titles state.

        Args:
            state (bool): Show titles state
        """
        self._datalab.toggle_show_titles(state)

    def save_to_h5_file(self, filename: str) -> None:
        """Save to a DataLab HDF5 file.

        Args:
            filename (str): HDF5 file name
        """
        self._datalab.save_to_h5_file(filename)

    def open_h5_files(
        self,
        h5files: list[str] | None = None,
        import_all: bool | None = None,
        reset_all: bool | None = None,
    ) -> None:
        """Open a DataLab HDF5 file or import from any other HDF5 file.

        Args:
            h5files (list[str] | None): List of HDF5 files to open. Defaults to None.
            import_all (bool | None): Import all objects from HDF5 files.
                Defaults to None.
            reset_all (bool | None): Reset all application data. Defaults to None.
        """
        self._datalab.open_h5_files(h5files, import_all, reset_all)

    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file.

        Args:
            filename (str): HDF5 file name
            reset_all (bool | None): Reset all application data. Defaults to None.
        """
        self._datalab.import_h5_file(filename, reset_all)

    def load_h5_workspace(
        self, h5files: list[str] | str, reset_all: bool = True
    ) -> None:
        """Load HDF5 workspace files without showing file dialog.

        This method loads one or more DataLab native HDF5 files directly, bypassing
        the file dialog. It is safe to call from the internal console or any context
        where Qt dialogs would cause threading issues.

        Args:
            h5files: Path(s) to HDF5 file(s). Can be a single path string or a list
             of paths.
            reset_all: If True (default), reset workspace before loading.
             If False, append to existing workspace.

        Raises:
            ValueError: if file is not a DataLab native HDF5 file
        """
        if isinstance(h5files, str):
            h5files = [h5files]
        self._datalab.load_h5_workspace(h5files, reset_all)

    def save_h5_workspace(self, filename: str) -> None:
        """Save workspace to HDF5 file without showing file dialog.

        This method saves the current workspace to a DataLab native HDF5 file
        directly, bypassing the file dialog. It is safe to call from the internal
        console or any context where Qt dialogs would cause threading issues.

        Args:
            filename: Path to the output HDF5 file
        """
        self._datalab.save_h5_workspace(filename)

    def load_from_files(self, filenames: list[str]) -> None:
        """Open objects from files in current panel (signals/images).

        Args:
            filenames: list of file names
        """
        self._datalab.load_from_files(filenames)

    def add_group(
        self, title: str, panel: str | None = None, select: bool = False
    ) -> None:
        """Add group to DataLab.

        Args:
            title: Group title
            panel: Panel name (valid values: "signal", "image"). Defaults to None.
            select: Select the group after creation. Defaults to False.
        """
        self._datalab.add_group(title, panel, select)

    def get_sel_object_uuids(self, include_groups: bool = False) -> list[str]:
        """Return selected objects uuids.

        Args:
            include_groups: If True, also return objects from selected groups.

        Returns:
            List of selected objects uuids.
        """
        return self._datalab.get_sel_object_uuids(include_groups)

    def select_objects(
        self,
        selection: list[int | str],
        panel: str | None = None,
    ) -> None:
        """Select objects in current panel.

        Args:
            selection: List of object numbers (1 to N) or uuids to select
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used. Defaults to None.
        """
        self._datalab.select_objects(selection, panel)

    def select_groups(
        self, selection: list[int | str] | None = None, panel: str | None = None
    ) -> None:
        """Select groups in current panel.

        Args:
            selection: List of group numbers (1 to N), or list of group uuids,
             or None to select all groups. Defaults to None.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used. Defaults to None.
        """
        self._datalab.select_groups(selection, panel)

    def delete_metadata(
        self, refresh_plot: bool = True, keep_roi: bool = False
    ) -> None:
        """Delete metadata of selected objects

        Args:
            refresh_plot: Refresh plot. Defaults to True.
            keep_roi: Keep ROI. Defaults to False.
        """
        self._datalab.delete_metadata(refresh_plot, keep_roi)

    def get_group_titles_with_object_info(
        self,
    ) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """Return groups titles and lists of inner objects uuids and titles.

        Returns:
            Tuple: groups titles, lists of inner objects uuids and titles
        """
        return self._datalab.get_group_titles_with_object_info()

    def get_object_titles(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image", "macro").
             If None, current data panel is used (i.e. signal or image panel).

        Returns:
            List of object titles

        Raises:
            ValueError: if panel not found
        """
        return self._datalab.get_object_titles(panel)

    def get_object_uuids(
        self, panel: str | None = None, group: int | str | None = None
    ) -> list[str]:
        """Get object (signal/image) uuid list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
            group: Group number, or group id, or group title.
             Defaults to None (all groups).

        Returns:
            List of object uuids

        Raises:
            ValueError: if panel not found
        """
        return self._datalab.get_object_uuids(panel, group)

    def add_label_with_title(
        self, title: str | None = None, panel: str | None = None
    ) -> None:
        """Add a label with object title on the associated plot

        Args:
            title (str | None): Label title. Defaults to None.
                If None, the title is the object title.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.
        """
        self._datalab.add_label_with_title(title, panel)

    def run_macro(self, number_or_title: int | str | None = None) -> None:
        """Run macro.

        Args:
             number: Number of the macro (starting at 1). Defaults to None (run
              current macro, or does nothing if there is no macro).
        """
        self._datalab.run_macro(number_or_title)

    def stop_macro(self, number_or_title: int | str | None = None) -> None:
        """Stop macro.

        Args:
            number: Number of the macro (starting at 1). Defaults to None (stop
             current macro, or does nothing if there is no macro).
        """
        self._datalab.stop_macro(number_or_title)

    def import_macro_from_file(self, filename: str) -> None:
        """Import macro from file

        Args:
            filename: Filename.
        """
        return self._datalab.import_macro_from_file(filename)
