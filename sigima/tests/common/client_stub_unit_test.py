# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests of the DataLab stub server used by the remote client tests."""

from __future__ import annotations

import numpy as np
import pytest

from sigima.client import utils
from sigima.client.stub import DataLabStubServer
from sigima.objects import create_image, create_signal


@pytest.mark.parametrize("verbose", [False, True])
def test_add_object_registers_objects_quietly_or_not(verbose: bool) -> None:
    """Objects are added whatever the verbosity of the stub server."""
    server = DataLabStubServer(verbose=verbose)
    signal = create_signal("S", np.linspace(0.0, 1.0, 5), np.zeros(5))
    image = create_image("I", np.zeros((3, 4)))
    assert server.add_object(utils.dataset_to_rpcjson(signal)) is True
    assert server.add_object(utils.dataset_to_rpcjson(image)) is True
    assert server.get_object_titles("signal")[-1] == "S"
    assert server.get_object_titles("image")[-1] == "I"


if __name__ == "__main__":
    test_add_object_registers_objects_quietly_or_not(False)
