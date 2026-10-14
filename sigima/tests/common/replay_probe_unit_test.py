# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the cross-environment tools of the replay probe.

The probe captures the calls of the validation tests (see ``conftest.py``); these
tests check how the captured calls are saved, replayed in another environment
(e.g. Pyodide, without pytest) and compared with the results of a third one.
"""

# pylint: disable=redefined-outer-name  # pytest fixtures

from __future__ import annotations

import os

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
from sigima.tests import replay_probe

NORMALIZE = "sigima.signal.normalize"


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> replay_probe.Sample:
    """One captured call of the signal normalization."""
    src = sigima.objects.create_signal(
        "s", np.linspace(0.0, 1.0, 5), np.array([-2.0, 0.0, 1.0, 4.0, 3.0])
    )
    param = sigima.params.NormalizeParam.create(method="maximum")
    result = sigima.proc.signal.normalize(src, param)
    sample = replay_probe.Sample("test_x", 0, (src, param), result, "abc")
    monkeypatch.setattr(replay_probe, "SAMPLES", {NORMALIZE: [sample]})
    return sample


def test_dump_to_a_directory_and_replay(captured, tmp_path) -> None:
    """Calls saved per operation are replayed here and the results saved."""
    capture = str(tmp_path / "capture") + os.sep
    replay_probe.dump_samples(capture)
    assert os.listdir(capture) == [f"{NORMALIZE}.pkl"]
    ((op, entries),) = list(replay_probe.load_dump(capture))
    assert op == NORMALIZE and entries[0][0] == captured.key
    results = str(tmp_path / "results")
    report = replay_probe.replay_dump(capture, results)
    assert report[NORMALIZE]["verdict"] == "exact"
    assert report[NORMALIZE]["statuses"] == {"replayed": 1}
    assert not replay_probe.replay_dump(capture, results, operations=["x"])
    compared = replay_probe.compare_replays(results, capture)
    assert compared[NORMALIZE]["verdict"] == "exact"
    assert compared[NORMALIZE]["samples"] == 1
    assert compared[NORMALIZE]["unmatched"] == 0


def test_compare_with_another_environment(captured, tmp_path, monkeypatch) -> None:
    """Results differing from the reference are reported under the rule."""
    capture = str(tmp_path / "capture") + os.sep
    replay_probe.dump_samples(capture)
    results = str(tmp_path / "results")
    replay_probe.replay_dump(capture, results)
    other = captured.result.copy()
    other.y = other.y + 1e-9
    shifted = replay_probe.Sample("test_x", 0, captured.args, other, "abc")
    monkeypatch.setattr(replay_probe, "SAMPLES", {NORMALIZE: [shifted]})
    reference = str(tmp_path / "reference.pkl")
    replay_probe.dump_samples(reference)
    compared = replay_probe.compare_replays(results, reference)
    entry = compared[NORMALIZE]
    assert entry["verdict"] == "different" and not entry["exact"]
    assert entry["max_abs_error"] == pytest.approx(1e-9)
    renamed = replay_probe.Sample("test_y", 0, captured.args, other, "def")
    monkeypatch.setattr(replay_probe, "SAMPLES", {NORMALIZE: [renamed]})
    replay_probe.dump_samples(reference)
    unmatched = replay_probe.compare_replays(results, reference)[NORMALIZE]
    assert (unmatched["samples"], unmatched["unmatched"]) == (0, 1)
    assert unmatched["verdict"] is None and unmatched["exact"] is None
    cross = replay_probe.cross_replay(reference)
    assert cross[NORMALIZE]["verdict"] == "different"


def test_input_digest_covers_roi_coordinates_and_metadata_read() -> None:
    """Calls on the same data with another ROI, grid or read metadata differ."""
    data = np.arange(16.0).reshape(4, 4)
    image = sigima.objects.create_image("i", data)
    digest = replay_probe.input_digest((image,))
    assert digest == replay_probe.input_digest((image.copy(),))
    with_roi = image.copy()
    with_roi.roi = sigima.objects.create_image_roi("rectangle", [0, 0, 2, 2])
    moved = image.copy()
    moved.x0 = 1.0
    signal = sigima.objects.create_signal("s", np.arange(3.0), np.arange(3.0))
    fitted = signal.copy()
    fitted.metadata["fit_params"] = {"fit_type": "linear"}
    digests = {
        replay_probe.input_digest(args, keys)
        for args, keys in [
            ((image,), ()),
            ((with_roi,), ()),
            ((moved,), ()),
            ((signal,), ("fit_params",)),
            ((fitted,), ("fit_params",)),
        ]
    }
    assert len(digests) == 5
    # Metadata not read by the computation do not change the key.
    unread = replay_probe.input_digest((fitted,))
    assert unread == replay_probe.input_digest((signal,))
