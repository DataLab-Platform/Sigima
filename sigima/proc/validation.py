# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Validation statistics module
(see parent package :mod:`sigima.computation`)
"""

from __future__ import annotations

import csv
import dataclasses
import importlib
import inspect
import os.path as osp
import pkgutil
import re

from _pytest.mark import Mark

import sigima.tests as tests_pkg
from sigima import __version__
from sigima.proc.decorator import find_computation_functions


def generate_valid_test_names_for_function(
    module_name: str, func_name: str
) -> list[str]:
    """Generate all valid test names for a computation function.

    Args:
        module_name: Module name containing the computation function
            (e.g., "sigima.proc.image")
        func_name: Function name (e.g., "compute_add_gaussian_noise")

    Returns:
        List of valid test names that could test this computation function
    """
    family = module_name.split(".")[-1]  # "signal" or "image"
    shortname = func_name.removeprefix("compute_")
    endings = [shortname, shortname + "_unit", shortname + "_validation"]
    beginnings = ["test", f"test_{family}", f"test_{family[:3]}", f"test_{family[0]}"]
    names = [f"{beginning}_{ending}" for beginning in beginnings for ending in endings]
    return names


def check_for_validation_test(
    full_function_name: str, validation_tests: list[tuple[str, str]]
) -> str:
    """Check if a validation test exists for a compute function

    Args:
        full_function_name: Compute function name
        validation_tests: List of validation tests

    Returns:
        Text to be included in the CSV file or None if it doesn't exist
    """
    # Extract module name and function name from full function name
    module_parts = full_function_name.split(".")
    module_name = ".".join(module_parts[:-1])  # e.g., "sigima.proc.image"
    func_name = module_parts[-1]  # e.g., "compute_add_gaussian_noise"

    # Generate all valid test names for this computation function
    names = generate_valid_test_names_for_function(module_name, func_name)

    stable_version = re.sub(r"\.?(post|dev|rc|b|a)\S*", "", __version__)
    for test, path, line_number in validation_tests:
        if test in names:
            # Path relative to the `datalab` package:
            path = osp.relpath(path, start=osp.dirname(osp.join(tests_pkg.__file__)))
            name = "/".join(path.split(osp.sep))
            link = (
                f"https://github.com/DataLab-Platform/Sigima/blob/"
                f"v{stable_version}/sigima/tests/{name}#L{line_number}"
            )
            return f"`{test} <{link}>`_"
    return None


def get_validation_tests(package: str) -> list:
    """Retrieve list of validation tests from a package and its submodules

    Args:
        package: Python package

    Returns:
        List of tuples containing the test name, module path and line number
    """
    validation_tests = []
    package_path = package.__path__
    for _, module_name, _ in pkgutil.walk_packages(
        package_path, package.__name__ + "."
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            if "viz" in module_name:
                # This is expected as viz requires a GUI
                continue
            raise ImportError(
                f"Failed to import module {module_name}. "
                "Ensure the module is correctly installed and accessible."
            ) from exc
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if hasattr(obj, "pytestmark"):
                for mark in obj.pytestmark:
                    if isinstance(mark, Mark) and mark.name == "validation":
                        module_path = inspect.getfile(obj)
                        try:
                            line_number = inspect.getsourcelines(obj)[1]
                        except OSError as exc:
                            raise RuntimeError(
                                f"Failed to get source line for {name} in {module_name}"
                            ) from exc
                        validation_tests.append((name, module_path, line_number))
    return validation_tests


def shorten_docstring(docstring: str) -> str:
    """Shorten a docstring to a single line

    Args:
        docstring: Docstring

    Returns:
        Shortened docstring
    """
    shorter = docstring.split("\n")[0].strip() if docstring else "-"
    for suffix in (".", ":", ",", "using", "with"):
        shorter = shorter.removesuffix(suffix)
    shorter = shorter.split(" with :py:func:")[0]  # Remove function references
    return shorter


@dataclasses.dataclass
class ValidationStatus:
    """Data class to hold validation status of a compute function"""

    module_name: str
    function_name: str
    description: str
    test_script: str

    def get_pyfunc_link(self) -> str:
        """Get the reStructuredText link to the compute function"""
        return (
            f":py:func:`{self.function_name} <{self.module_name}.{self.function_name}>`"
        )

    def __str__(self) -> str:
        """String representation of the validation status"""
        return f"{self.get_pyfunc_link()} - {self.description} - {self.test_script}"

    def to_csv_row(self) -> list[str]:
        """Convert the validation status to a CSV row"""
        return [self.get_pyfunc_link(), self.description, self.test_script]


class ValidationStatistics:
    """Data class to hold validation statistics of compute functions"""

    def __init__(self):
        self.submodules: dict[str, list[tuple[str, str, str]]] = {}
        self.t_count: dict[str, int] = {}
        self.v_count: dict[str, int] = {}
        self.signal_pct: int = 0
        self.image_pct: int = 0
        self.total_pct: int = 0
        self.validations: dict[str, list[ValidationStatus]] = {}
        self.validation_info_list: list[str] = []

    def collect_validation_status(self, verbose: bool = False) -> None:
        """Populate the statistics from the validation status"""
        compute_functions = find_computation_functions()
        validation_tests = get_validation_tests(tests_pkg)

        self.submodules = {"signal": [], "image": []}
        for modname, funcname, docstring in compute_functions:
            if "signal" in modname:
                self.submodules["signal"].append((modname, funcname, docstring))
            elif "image" in modname:
                self.submodules["image"].append((modname, funcname, docstring))

        self.t_count = t_count = {"signal": 0, "image": 0, "total": 0}
        self.v_count = v_count = {"signal": 0, "image": 0, "total": 0}

        self.validations = {"signal": [], "image": []}
        for submodule, functions in self.submodules.items():
            for modname, funcname, docstring in functions:
                full_funcname = f"{modname}.{funcname}"
                test_link = check_for_validation_test(full_funcname, validation_tests)
                if test_link:
                    v_count[submodule] += 1
                    v_count["total"] += 1
                t_count[submodule] += 1
                t_count["total"] += 1
                status = ValidationStatus(
                    module_name=modname,
                    function_name=funcname,
                    description=shorten_docstring(docstring),
                    test_script=test_link if test_link else "N/A",
                )
                self.validations[submodule].append(status)

            self.signal_pct = signal_pct = (
                int((v_count["signal"] / t_count["signal"]) * 100)
                if t_count["signal"] > 0
                else 0
            )
            self.image_pct = image_pct = (
                int((v_count["image"] / t_count["image"]) * 100)
                if t_count["image"] > 0
                else 0
            )
            self.total_pct = total_pct = (
                int((v_count["total"] / t_count["total"]) * 100)
                if t_count["total"] > 0
                else 0
            )

        self.validation_info_list = [
            f"Validation statistics for Sigima {__version__}:",
            f"  Signal: {v_count['signal']}/{t_count['signal']} ({signal_pct}%)",
            f"  Image : {v_count['image']}/{t_count['image']} ({image_pct}%)",
            f"  Total : {v_count['total']}/{t_count['total']} ({total_pct}%)",
        ]

        if verbose:
            print("\n".join(self.validation_info_list))
            print()

    def get_validation_info(self) -> list[str]:
        """Get the validation information as a list of strings"""
        if not self.validation_info_list:
            self.collect_validation_status(verbose=False)
        return self.validation_info_list

    def generate_csv_files(self, path: str) -> None:
        """Generate CSV files for each submodule's validation status"""
        for submodule, validation_status_list in self.validations.items():
            rows = [status.to_csv_row() for status in validation_status_list]
            fname = osp.join(path, f"validation_status_{submodule}.csv")
            with open(fname, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rows)

    def generate_statistics_csv(self, path: str) -> None:
        """Generate a CSV file with the validation statistics"""
        statistics_rows = [
            [
                "Number of compute functions",
                self.t_count["signal"],
                self.t_count["image"],
                self.t_count["total"],
            ],
            [
                "Number of validated compute functions",
                self.v_count["signal"],
                self.v_count["image"],
                self.v_count["total"],
            ],
            [
                "Percentage of validated compute functions",
                f"{self.signal_pct}%",
                f"{self.image_pct}%",
                f"{self.total_pct}%",
            ],
        ]
        fname = osp.join(path, "validation_statistics.csv")
        with open(fname, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(statistics_rows)
