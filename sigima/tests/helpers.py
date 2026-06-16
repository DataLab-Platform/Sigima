# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Module providing test utilities
"""

from __future__ import annotations

import atexit
import functools
import os
import os.path as osp
import pathlib
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable
from typing import Any, Generator

import numpy as np
from guidata.configtools import get_module_data_path

import sigima.enums
import sigima.objects
from sigima.config import MOD_NAME
from sigima.io.image import ImageIORegistry
from sigima.io.signal import SignalIORegistry
from sigima.objects.image import ImageObj
from sigima.objects.signal import SignalObj
from sigima.tests.env import execenv

TST_PATH = []


def _patch_python39_tempdir_cleanup() -> None:
    """Patch TemporaryDirectory.cleanup to prevent stack overflow on Python 3.9.

    On Python 3.9 (Windows), ``TemporaryDirectory.__del__`` during interpreter
    shutdown causes recursive ``shutil.rmtree`` calls, leading to a stack overflow.
    Python 3.10+ is not affected and is left untouched.
    """
    if sys.version_info >= (3, 10):
        return

    def _safe_cleanup(self: tempfile.TemporaryDirectory) -> None:
        self._finalizer.detach()
        shutil.rmtree(self.name, ignore_errors=True)

    tempfile.TemporaryDirectory.cleanup = _safe_cleanup  # type: ignore[assignment]


_patch_python39_tempdir_cleanup()


def get_test_paths() -> list[str]:
    """Return the list of test data paths"""
    return TST_PATH


def add_test_path(path: str) -> None:
    """Appends test data path, after normalizing it and making it absolute.
    Do nothing if the path is already in the list.

    Args:
        Path to add to the list of test data paths

    Raises:
        FileNotFoundError: if the path does not exist
    """
    path = osp.abspath(osp.normpath(path))
    if path not in TST_PATH:
        if not osp.exists(path):
            raise FileNotFoundError(f"Test data path does not exist: {path}")
        TST_PATH.append(path)


def add_test_path_from_env(envvar: str) -> None:
    """Appends test data path from environment variable (fails silently)"""
    # Note: this function is used in third-party plugins
    path = os.environ.get(envvar)
    if path:
        add_test_path(path)


# Add test data files and folders pointed by `SIGIMA_DATA` environment variable:
add_test_path_from_env("SIGIMA_DATA")


def add_test_module_path(modname: str, relpath: str) -> None:
    """
    Appends test data path relative to a module name.
    Used to add module local data that resides in a module directory
    but will be shipped under sys.prefix / share/ ...

    modname must be the name of an already imported module as found in
    sys.modules
    """
    add_test_path(get_module_data_path(modname, relpath=relpath))


# Add test data files and folders for the Sigima module:
add_test_module_path(MOD_NAME, osp.join("data", "tests"))


def get_test_fnames(pattern: str, in_folder: str | None = None) -> list[str]:
    """
    Return the absolute path list to test files with specified pattern

    Pattern may be a file name (basename), a wildcard (e.g. *.txt)...

    Args:
        pattern: pattern to match
        in_folder: folder to search in, in test data path (default: None,
         search in all test data paths)
    """
    pathlist = []
    for pth in [osp.join(TST_PATH[0], in_folder)] if in_folder else TST_PATH:
        pathlist += sorted(pathlib.Path(pth).rglob(pattern))
    if not pathlist:
        raise FileNotFoundError(f"Test file(s) {pattern} not found")
    return [str(path) for path in pathlist]


def read_test_objects(
    registry: SignalIORegistry | ImageIORegistry,
    pattern: str = "*.*",
    in_folder: str | None = None,
) -> Generator[tuple[str, ImageObj | None] | tuple[str, SignalObj | None], None, None]:
    """Read test images and yield their file names and objects

    Args:
        registry: I/O registry to use
        pattern: File name pattern to match
        in_folder: Folder to search for test files

    Yields:
        Tuple of file name and object (or None if not implemented)
    """
    if registry is ImageIORegistry:
        in_folder = in_folder or "image_formats"
    elif registry is SignalIORegistry:
        in_folder = in_folder or "curve_formats"
    else:
        raise ValueError(f"Unsupported registry type: {registry}")
    fnames = get_test_fnames(pattern, in_folder)
    for fname in fnames:
        try:
            obj = registry.read(fname)[0]
            yield fname, obj
        except NotImplementedError:
            yield fname, None


def try_open_test_data(title: str, pattern: str) -> Callable:
    """Decorator handling test data opening"""

    def try_open_test_data_decorator(func: Callable) -> Callable:
        """Decorator handling test data opening"""

        @functools.wraps(func)
        def func_wrapper(*args, **kwargs) -> None:
            """Decorator wrapper function"""
            execenv.print(title + ":")
            execenv.print("-" * len(title))
            try:
                for fname in get_test_fnames(pattern):
                    execenv.print(f"=> Opening: {fname}")
                    func(fname, title, *args, **kwargs)
            except FileNotFoundError:
                execenv.print(f"  No test data available for {pattern}")
            finally:
                execenv.print(os.linesep)

        return func_wrapper

    return try_open_test_data_decorator


def get_default_test_name(suffix: str | None = None) -> str:
    """Return default test name based on script name"""
    name = osp.splitext(osp.basename(sys.argv[0]))[0]
    if suffix is not None:
        name += "_" + suffix
    return name


def get_output_data_path(extension: str, suffix: str | None = None) -> str:
    """Return full path for data file with extension, generated by a test script"""
    name = get_default_test_name(suffix)
    return osp.join(TST_PATH[0], f"{name}.{extension}")


def reduce_path(filename: str) -> str:
    """Reduce a file path to a relative path

    Args:
        filename: path to reduce

    Returns:
        Relative path to the file, relative to its parent directory
    """
    return osp.relpath(filename, osp.join(osp.dirname(filename), osp.pardir))


class WorkdirRestoringTempDir(tempfile.TemporaryDirectory):
    """Enhanced temporary directory with working directory preservation.

    A subclass of :py:class:`tempfile.TemporaryDirectory` that:

    * Preserves and automatically restores the working directory during cleanup
    * Handles common cleanup errors silently (PermissionError, RecursionError)

    Example::

        with WorkdirRestoringTempDir() as tmpdir:
            os.chdir(tmpdir)  # Directory change is automatically reverted at exit
    """

    def __init__(self) -> None:
        super().__init__()
        self.__cwd = os.getcwd()

    def cleanup(self) -> None:
        """Restore the working directory before cleaning up."""
        os.chdir(self.__cwd)
        super().cleanup()


def get_temporary_directory() -> str:
    """Return path to a temporary directory, and clean-up at exit"""
    tmp = WorkdirRestoringTempDir()
    atexit.register(tmp.cleanup)
    return tmp.name


def exec_script(
    path: str,
    wait: bool = True,
    args: list[str] = None,
    env: dict[str, str] | None = None,
    verbose: bool = False,
) -> subprocess.Popen | None:
    """Run test script.

    Args:
        path: path to script
        wait: wait for script to finish
        args: arguments to pass to script
        env: environment variables to pass to script
        verbose: if True, print command and output

    Returns:
        subprocess.Popen object if wait is False, None otherwise
    """
    stderr = subprocess.DEVNULL if execenv.unattended else None
    # pylint: disable=consider-using-with
    if verbose:
        command = [sys.executable, path] + ([] if args is None else args)
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
    else:
        command = [sys.executable, '"' + path + '"'] + ([] if args is None else args)
        proc = subprocess.Popen(" ".join(command), shell=True, stderr=stderr, env=env)
    if wait:
        if verbose:
            stdout, stderr = proc.communicate()
            print("Command:", " ".join(command))
            print("Return code:", proc.returncode)
            print("---- STDOUT ----\n", stdout)
            print("---- STDERR ----\n", stderr)
            return None
        proc.wait()
    return proc


def get_script_output(
    path: str, args: list[str] = None, env: dict[str, str] | None = None
) -> str:
    """Run test script and return its output.

    Args:
        path (str): path to script
        args (list): arguments to pass to script
        env (dict): environment variables to pass to script

    Returns:
        str: script output
    """
    command = [sys.executable, '"' + path + '"'] + ([] if args is None else args)
    result = subprocess.run(
        " ".join(command), capture_output=True, text=True, env=env, check=False
    )
    return result.stdout.strip()


def compare_lists(
    list1: list, list2: list, level: int = 1, raise_on_diff: bool = False
) -> tuple[bool, list[str]]:
    """Compare two lists

    Args:
        list1: first list
        list2: second list
        level: recursion level
        raise_on_diff: if True, raise an AssertionError on difference (default: False)

    Returns:
        A tuple (same, diff) where `same` is True if lists are the same,
        False otherwise, and `diff` is a list of differences found

    Raises:
        AssertionError: if raise_on_diff is True and lists are different
    """
    same = True
    prefix = "  " * level
    diff = []
    # Check for length mismatch
    if len(list1) != len(list2):
        same = False
        diff += [f"{prefix}Lists have different lengths: {len(list1)} != {len(list2)}"]
    for idx, (elem1, elem2) in enumerate(zip(list1, list2)):
        execenv.print(f"{prefix}Checking element {idx}...", end=" ")
        if isinstance(elem1, (list, tuple)):
            execenv.print("")
            cl_same, cl_diff = compare_lists(elem1, elem2, level + 1)
            diff += cl_diff
            same = same and cl_same
        elif isinstance(elem1, dict):
            execenv.print("")
            cm_same, cm_diff = compare_metadata(elem1, elem2, level + 1)
            diff += cm_diff
            same = same and cm_same
        else:
            same_value = str(elem1) == str(elem2)
            if not same_value:
                diff += [
                    f"{prefix}Different values for element {idx}: {elem1} != {elem2}"
                ]
            same = same and same_value
            execenv.print("OK" if same_value else "KO")
    if diff:
        all_diff = os.linesep.join(diff)
        if raise_on_diff:
            raise AssertionError(all_diff)
        execenv.print("Lists are different:")
        execenv.print(all_diff)
    return same, diff


def compare_metadata(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    level: int = 1,
    raise_on_diff: bool = False,
) -> tuple[bool, list[str]]:
    """Compare metadata dictionaries without private elements

    Args:
        dict1: first dictionary, exclusively with string keys
        dict2: second dictionary, exclusively with string keys
        level: recursion level
        raise_on_diff: if True, raise an AssertionError on difference (default: False)

    Returns:
        A tuple (same, diff) where `same` is True if dictionaries are the same,
        False otherwise, and `diff` is a list of differences found

    Raises:
        AssertionError: if raise_on_diff is True and metadata is different
    """
    dict_a, dict_b = dict1.copy(), dict2.copy()
    for dict_ in (dict_a, dict_b):
        for key in list(dict_.keys()):
            if key.startswith("__"):
                dict_.pop(key)
    same = True
    prefix = "  " * level
    diff = []
    # Check for keys only in dict_a
    for key in dict_a:
        if key not in dict_b:
            same = False
            diff += [f"{prefix}Key {key} found in first dict but not in second"]
            continue
        val_a, val_b = dict_a[key], dict_b[key]
        execenv.print(f"{prefix}Checking key {key}...", end=" ")
        if isinstance(val_a, dict):
            execenv.print("")
            cm_same, cm_diff = compare_metadata(val_a, val_b, level + 1)
            diff += cm_diff
            same = same and cm_same
        elif isinstance(val_a, (list, tuple)):
            execenv.print("")
            cl_same, cl_diff = compare_lists(val_a, val_b, level + 1)
            diff += cl_diff
            same = same and cl_same
        else:
            same_value = str(val_a) == str(val_b)
            if not same_value:
                diff += [f"{prefix}Different values for key {key}: {val_a} != {val_b}"]
            same = same and same_value
            execenv.print("OK" if same_value else "KO")
    # Check for keys only in dict_b
    for key in dict_b:
        if key not in dict_a:
            same = False
            diff += [f"{prefix}Key {key} found in second dict but not in first"]
    if diff:
        all_diff = os.linesep.join(diff)
        if raise_on_diff:
            raise AssertionError(all_diff)
        execenv.print("Dictionaries are different:")
        execenv.print(all_diff)
    return same, diff


def __evaluate_func_safely(func: Callable, fallback: float | int = np.nan) -> Any:
    """Evaluate function, ignore warnings and exceptions.

    Args:
        func: function to evaluate
        fallback: value to return if function raises an exception (default: np.nan)

    Returns:
        Function result, or fallback value if function raises an exception
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return func()
        except Exception:  # pylint: disable=broad-except
            return fallback


def __array_to_str(data: np.ndarray) -> str:
    """Return a compact description of the array properties.

    Args:
        data: input array

    Returns:
        String describing array dimensions, dtype, min/max, mean, std, sum
    """
    dims = "×".join(str(dim) for dim in data.shape)
    efs = __evaluate_func_safely
    return (
        f"{dims},{data.dtype},"
        f"{efs(data.min):.2g}→{efs(data.max):.2g},"
        f"µ={efs(data.mean):.2g},σ={efs(data.std):.2g},∑={efs(data.sum):.2g}"
    )


def check_array_result(
    title: str,
    res: np.ndarray,
    exp: np.ndarray,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    similar: bool = False,
    sort: bool = False,
    verbose: bool = True,
) -> None:
    """Assert that two arrays are almost equal.

    Args:
        title: title of the test
        res: result array
        exp: expected array
        rtol: relative tolerance for comparison
        atol: absolute tolerance for comparison
        similar: if True, arrays are compared exclusively using their textual
         global representation (e.g. '824,float64,-0.00012→0.036,µ=0.018')
        sort: if True, sort arrays before comparison (default: False)
        verbose: if True, print detailed result (default: True)

    Raises:
        AssertionError: if arrays are not almost equal or have different dtypes
    """
    if sort:
        res = np.sort(np.array(res, copy=True), axis=None)
        exp = np.sort(np.array(exp, copy=True), axis=None)
    restxt = f"{title}: {__array_to_str(res)} (expected: {__array_to_str(exp)})"
    if verbose:
        execenv.print(restxt)
    assert res.shape == exp.shape, f"{restxt} - Different shapes"
    try:
        if similar:
            assert __array_to_str(res) == __array_to_str(exp), restxt
        else:
            assert np.allclose(res, exp, rtol=rtol, atol=atol, equal_nan=True), restxt
    except AssertionError as exc:
        raise AssertionError(restxt) from exc
    assert res.dtype == exp.dtype, f"{restxt} - Different dtypes"


def check_scalar_result(
    title: str,
    res: float,
    exp: float | tuple[float, ...],
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    verbose: bool = True,
) -> None:
    """Assert that two scalars are almost equal.

    Args:
        title: title of the test
        res: result value
        exp: expected value or tuple of expected values
        rtol: relative tolerance for comparison
        atol: absolute tolerance for comparison
        verbose: if True, print detailed result (default: True)

    Raises:
        AssertionError: if values are not almost equal or if expected is not a scalar
         or tuple
    """
    restxt = f"{title}: {res} (expected: {exp}) ± {rtol * abs(exp) + atol:.2g}"
    if verbose:
        execenv.print(restxt)
    if isinstance(exp, tuple):
        assert any(np.isclose(res, exp_val, rtol=rtol, atol=atol) for exp_val in exp), (
            restxt
        )
    else:
        assert np.isclose(res, exp, rtol=rtol, atol=atol), restxt


def print_obj_data_dimensions(obj: SignalObj | ImageObj, indent: int = 0) -> None:
    """Print data array shape for the given signal or image object,
    including ROI data if available.

    Args:
        obj: Signal or image object to print data dimensions for.
        indent: Indentation level for printing (default: 0)
    """
    indent_str = "  " * indent
    execenv.print(f"{indent_str}Accessing object '{obj.title}':")
    execenv.print(f"{indent_str}  data: {__array_to_str(obj.data)}")
    if obj.roi is not None:
        for idx in range(len(obj.roi)):
            roi_data = obj.get_data(idx)
            if isinstance(obj, SignalObj):
                roi_data = roi_data[1]  # y data
            execenv.print(f"{indent_str}  ROI[{idx}]: {__array_to_str(roi_data)}")


def validate_detection_rois(
    obj: ImageObj,
    coords: np.ndarray,
    create_rois: bool,
    roi_geometry: sigima.enums.DetectionROIGeometry,
) -> None:
    """Validate that ROIs were created correctly from detection results.

    Args:
        obj: Image object that should contain ROIs
        coords: Detection coordinates array
        create_rois: Whether ROI creation was requested
        roi_geometry: Expected ROI geometry type

    Raises:
        AssertionError: if ROI validation fails
    """
    if create_rois and len(coords) > 1:
        assert obj.roi is not None, "ROI should be created when create_rois=True"
        assert len(obj.roi) == coords.shape[0], (
            f"Expected {coords.shape[0]} ROIs, got {len(obj.roi)}"
        )
        execenv.print(f"✓ Created {len(obj.roi)} ROIs")

        # Validate ROI type based on geometry
        for i, roi in enumerate(obj.roi):
            if roi_geometry == sigima.enums.DetectionROIGeometry.CIRCLE:
                assert isinstance(roi, sigima.objects.CircularROI), (
                    f"Expected CircularROI, got {type(roi)}"
                )
            else:  # RECTANGLE
                assert isinstance(roi, sigima.objects.RectangularROI), (
                    f"Expected RectangularROI, got {type(roi)}"
                )

            # Check that ROIs are correctly positioned around detection points
            x0, y0, x1, y1 = roi.get_bounding_box(obj)
            # For point detections, coords has 2 columns [x, y]
            # For blob detections, coords has 3 columns [x, y, radius]
            x, y = coords[i, 0], coords[i, 1]
            assert x0 <= x < x1, f"ROI {i} x0={x0}, x={x}, x1={x1} does not match"
            assert y0 <= y < y1, f"ROI {i} y0={y0}, y={y}, y1={y1} does not match"
    else:
        assert obj.roi is None or len(obj.roi) == 0, (
            "ROI should not be created when create_rois=False or too few detections"
        )
