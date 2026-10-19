# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
`sigima` pytest configuration
-----------------------------

This file contains the configuration for running pytest in `sigima`. It is
executed before running any tests.
"""

import json
import os
import os.path as osp

import guidata
import h5py
import numpy
import pytest
import scipy
import skimage
from guidata.config import ValidationMode, set_validation_mode
from guidata.utils.gitreport import format_git_info_for_pytest, get_git_info_for_modules

import sigima
from sigima.proc.contracts import build_contract_registry
from sigima.proc.decorator import observe_calls
from sigima.proc.validation import ValidationStatistics
from sigima.tests import SIGIMA_TESTS_GUI_ENV, env, helpers
from sigima.tests import replay_probe as replay_probe_module

# Set validation mode to STRICT for all tests
set_validation_mode(ValidationMode.STRICT)

# Turn on unattended mode for executing tests without user interaction
env.execenv.unattended = True
env.execenv.verbose = "quiet"

INITIAL_CWD = os.getcwd()
REPLAY_TEST_MODULE = "replay_qualification_test.py"


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--show-windows",
        action="store_true",
        default=False,
        help="Display Qt windows during tests (disables QT_QPA_PLATFORM=offscreen)",
    )
    parser.addoption(
        "--gui", action="store_true", default=False, help="Run tests that require a GUI"
    )
    parser.addoption(
        "--qualification-report",
        default=None,
        help="Write the replay qualification report (JSON) to this file and fail "
        "if a declared qualification is not backed by passing tests",
    )


def pytest_report_header(config):  # pylint: disable=unused-argument
    """Add additional information to the pytest report header."""
    infolist = [
        f"sigima {sigima.__version__}",
        f"  guidata {guidata.__version__},",
        f"  NumPy {numpy.__version__}, SciPy {scipy.__version__}, "
        f"h5py {h5py.__version__}, scikit-image {skimage.__version__}",
    ]
    try:
        import cv2  # pylint: disable=import-outside-toplevel

        infolist[-1] += f", OpenCV {cv2.__version__}"
    except ImportError:
        pass
    envlist = []
    for vname in ("SIGIMA_DATA", "PYTHONPATH", "DEBUG", "QT_API", "QT_QPA_PLATFORM"):
        value = os.environ.get(vname, "")
        if value:
            if vname == "PYTHONPATH":
                pathlist = value.split(os.pathsep)
                envlist.append(f"  {vname}:")
                envlist.extend(f"    {p}" for p in pathlist if p)
            else:
                envlist.append(f"  {vname}: {value}")
    if envlist:
        infolist.append("Environment variables:")
        infolist.extend(envlist)
    infolist.append("Test paths:")
    for test_path in helpers.get_test_paths():
        test_path = osp.abspath(test_path)
        infolist.append(f"  {test_path}")

    # Git information for all modules using the new gitreport module
    modules_config = [
        ("Sigima", sigima, "."),  # Sigima uses current directory
        ("guidata", guidata, None),
    ]
    git_repos = get_git_info_for_modules(modules_config)
    git_info_lines = format_git_info_for_pytest(git_repos, "Sigima")
    if git_info_lines:
        infolist.extend(git_info_lines)

    infolist.extend(ValidationStatistics().get_validation_info())
    return infolist


def pytest_configure(config):
    """Add custom markers to pytest."""
    if config.option.durations is None:
        config.option.durations = 10  # Default to showing 10 slowest tests
    config.addinivalue_line(
        "markers",
        "validation: mark a test as a validation test (ground truth or analytical)",
    )
    if not config.getoption("--show-windows"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    config.addinivalue_line("markers", "gui: mark test as requiring GUI")
    if config.getoption("--gui"):
        os.environ[SIGIMA_TESTS_GUI_ENV] = "1"
    else:
        os.environ.pop(SIGIMA_TESTS_GUI_ENV, None)


def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on command line options."""
    # The generic replay test replays the calls captured during the validation
    # tests: it runs last.
    replay = [item for item in items if REPLAY_TEST_MODULE in item.nodeid]
    items[:] = [item for item in items if REPLAY_TEST_MODULE not in item.nodeid]
    items.extend(replay)
    if config.getoption("--gui"):
        return  # User requested GUI tests

    skip_gui = pytest.mark.skip(reason="GUI test: run with --gui")

    for item in items:
        if "gui" in item.keywords:
            item.add_marker(skip_gui)


@pytest.fixture(autouse=True)
def reset_cwd(request):  # pylint: disable=unused-argument
    """Reset the current working directory to the initial one after each test."""
    yield
    os.chdir(INITIAL_CWD)


@pytest.fixture(autouse=True)
def replay_probe(request):
    """Capture the computation calls of validation tests, for the replay test."""
    if request.node.get_closest_marker("validation") is None:
        yield
        return
    with observe_calls(replay_probe_module.make_observer(request.node.nodeid)):
        yield


def pytest_runtest_logreport(report):
    """Record the outcome of validation and replay tests (qualification report).

    Only the validation tests (marked ``validation``, as for the validation
    statistics) and the generic replay test are recorded, by function name:
    other tests may share the name of a computation function. Parametrized
    tests are recorded under ``name[parameter]`` and under their base name,
    which fails if any of its cases failed.
    """
    if report.when != "call" and (report.when != "setup" or report.passed):
        return
    if "validation" not in report.keywords and REPLAY_TEST_MODULE not in report.nodeid:
        return
    outcomes = replay_probe_module.TEST_OUTCOMES
    name = report.nodeid.split("::")[-1]
    # Test names are not unique across modules: a failure is never overwritten
    if outcomes.get(name) != "failed":
        outcomes[name] = report.outcome
    base = name.split("[", 1)[0]
    if base != name and outcomes.get(base) != "failed":
        if outcomes.get(base) != "passed" or report.outcome == "failed":
            outcomes[base] = report.outcome


def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    """Write the qualification report and the cross-environment replay files."""
    dump, reference, cross_report = replay_probe_module.environment_paths()
    if dump:
        replay_probe_module.dump_samples(dump)
    if reference and cross_report:
        report = replay_probe_module.cross_replay(reference)
        with open(cross_report, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=1, sort_keys=True)
        qualified = {
            c.operation_id
            for c in build_contract_registry().contracts
            if c.qualification is not None
        }
        different = sorted(
            op
            for op, entry in report.items()
            if op in qualified and entry["verdict"] in ("different", "error")
        )
        if different:
            print(
                "\nQualified operations differing from the reference environment"
                " beyond their comparison rule:\n  " + "\n  ".join(different)
            )
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    path = session.config.getoption("--qualification-report")
    if not path:
        return
    stats = ValidationStatistics()
    statuses = stats.collect_qualification_status(
        replay_probe_module.TEST_OUTCOMES, replay_probe_module.OUTCOMES
    )
    problems = [p for status in statuses for p in status.problems]
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "environment": stats.environment_info(),
                "problems": problems,
                "operations": [status.to_dict() for status in statuses],
            },
            file,
            indent=1,
        )
    if problems:
        print("\nQualification problems:\n  " + "\n  ".join(problems))
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
