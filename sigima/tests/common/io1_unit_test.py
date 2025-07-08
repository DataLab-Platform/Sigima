# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O test

Testing `sigima` specific formats.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sigima.io import read_images, read_signals
from sigima.io.ftlab import FTLabImageFile, imread_ftlabima, sigread_ftlabsig
from sigima.io.image import funcs as image_funcs
from sigima.objects import ImageObj, SignalObj
from sigima.tests import guiutils, helpers
from sigima.tests.env import execenv


def __read_objs(fname: str) -> list[ImageObj] | list[SignalObj]:
    """Read objects from a file"""
    if "curve_formats" in fname:
        objs = read_signals(fname)
    else:
        objs = read_images(fname)
    for obj in objs:
        if np.all(np.isnan(obj.data)):
            raise ValueError("Data is all NaNs")
    for obj in objs:
        # Ignore warnings for complex numbers (workaround for guidata)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            execenv.print(obj)
    return objs


@helpers.try_open_test_data("Testing TXT file reader", "*.txt")
def open_txt(fname: str | None = None) -> None:
    """Testing TXT files"""
    objs = __read_objs(fname)
    if guiutils.is_gui_enabled():
        from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

        with vistools.qt_app_context():
            vistools.view_curves_and_images(objs, title="TXT file")


@helpers.try_open_test_data("Testing CSV file reader", "*.csv")
def open_csv(fname: str | None = None) -> None:
    """Testing CSV files"""
    objs = __read_objs(fname)
    if guiutils.is_gui_enabled():
        from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

        with vistools.qt_app_context():
            vistools.view_curves_and_images(objs, title="CSV file")


@helpers.try_open_test_data("Testing FTLab signal file reader", "*.sig")
def open_sigdata(fname: str | None = None) -> None:
    """Testing FTLab signal files"""
    objs = __read_objs(fname)
    if guiutils.is_gui_enabled():
        from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

        vistools.view_curves_and_images(objs, title="FTLab signal file")
    data = sigread_ftlabsig(fname)
    ref = read_signals(fname.replace(".sig", ".npy"))[0]
    helpers.check_array_result(f"{fname}", data, ref.xydata)


@helpers.try_open_test_data("Testing MAT-File reader", "*.mat")
def open_mat(fname: str | None = None) -> None:
    """Testing MAT files"""
    objs = __read_objs(fname)
    if guiutils.is_gui_enabled():
        from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

        with vistools.qt_app_context():
            vistools.view_curves_and_images(objs, title="MAT file")


@helpers.try_open_test_data("Testing SIF file handler", "*.sif")
def open_sif(fname: str | None = None) -> None:
    """Testing SIF files"""
    execenv.print(image_funcs.SIFFile(fname))
    datalist = image_funcs.imread_sif(fname)
    if guiutils.is_gui_enabled():
        from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

        with vistools.qt_app_context():
            vistools.view_images(datalist)


@helpers.try_open_test_data("Testing SCOR-DATA file handler", "*.scor-data")
def open_scordata(fname: str | None = None) -> None:
    """Testing SCOR-DATA files"""
    execenv.print(image_funcs.SCORFile(fname))
    data = image_funcs.imread_scor(fname)
    if guiutils.is_gui_enabled():
        from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

        with vistools.qt_app_context():
            vistools.view_images(data, title="SCOR-DATA file")


@helpers.try_open_test_data("Testing FTLab image file handler", "*.ima")
def open_imadata(fname: str | None = None) -> None:
    """Testing FTLab image files.

    Args:
        fname: Name of the file to open.
    """
    objs = __read_objs(fname)
    if guiutils.is_gui_enabled():
        from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

        vistools.view_curves_and_images(objs, title="FTLab image file")
    ftlab_file = FTLabImageFile(fname)
    _ = ftlab_file.read_data()
    execenv.print(ftlab_file)
    data = imread_ftlabima(fname)
    ref = read_images(fname.replace(".ima", ".npy"))[0]
    helpers.check_array_result(f"{fname}", data, ref.data)


def test_io1(request: pytest.FixtureRequest | None = None) -> None:
    """I/O test"""
    guiutils.set_current_request(request)
    open_txt()
    open_csv()
    open_sigdata()
    open_mat()
    open_sif()
    open_scordata()
    open_imadata()


if __name__ == "__main__":
    test_io1(request=guiutils.DummyRequest(gui=True))
