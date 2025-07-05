# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
`sigima` pytest configuration
-----------------------------

This file contains the configuration for running pytest in `sigima`. It is
executed before running any tests.
"""

import os
import os.path as osp
import subprocess

import guidata
import h5py
import numpy
import pytest
import scipy
import skimage

import sigima
from sigima.tests import env, helpers

# Turn on unattended mode for executing tests without user interaction
env.execenv.unattended = True
env.execenv.verbose = "quiet"

INITIAL_CWD = os.getcwd()


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
    for vname in ("CDL_DATA", "PYTHONPATH", "DEBUG", "QT_API", "QT_QPA_PLATFORM"):
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
    sco = subprocess.check_output
    try:
        gitlist = ["Git information:"]
        branch = sco(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        commit = sco(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        message = sco(["git", "log", "-1", "--pretty=%B"], text=True).strip()
        if len(message.splitlines()) > 1:
            message = message.splitlines()[0]
        message = message[:60] + "[…]" if len(message) > 60 else message
        gitlist.append(f"  Branch: {branch}")
        gitlist.append(f"  Commit: {commit}")
        gitlist.append(f"  Message: {message}")
        infolist.extend(gitlist)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return infolist


def pytest_configure(config):
    """Add custom markers to pytest."""
    config.addinivalue_line(
        "markers",
        "validation: mark a test as a validation test (ground truth or analytical)",
    )
    if not config.getoption("--show-windows"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    config.addinivalue_line("markers", "gui: mark test as requiring GUI")


def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on command line options."""
    if config.getoption("--gui"):
        return  # L'utilisateur a demandé d'exécuter les tests GUI

    skip_gui = pytest.mark.skip(reason="GUI test: run with --gui")

    for item in items:
        if "gui" in item.keywords:
            item.add_marker(skip_gui)


@pytest.fixture(autouse=True)
def reset_cwd(request):  # pylint: disable=unused-argument
    """Reset the current working directory to the initial one after each test."""
    yield
    os.chdir(INITIAL_CWD)
