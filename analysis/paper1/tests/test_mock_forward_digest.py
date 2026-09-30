# SPDX-License-Identifier: BSD-3-Clause
"""Comparing two source trees is only meaningful for the same model.

``mock_forward_digest --compare`` exists to answer whether moving the pin
forward changes the paper's figure. The failure mode it has to refuse is the
one that would make the answer look reassuring: two trees that declare
different free parameters are two different models, and a per-band diff
between them reports a difference in *configuration* as though it were a
difference in *code*. Shapes can match while the models do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

pytestmark = pytest.mark.unit

BANDS = ["galex_nuv", "sdss_r"]


def _digest(tmp_path, name, phot, free_params, spectrum=None):
    path = tmp_path / name
    np.savez(
        path,
        photometry=np.asarray(phot, dtype=float),
        spectrum=np.zeros(4) if spectrum is None else np.asarray(spectrum, dtype=float),
        filters=np.array(BANDS),
        free_params=np.array(free_params),
        tengri_path=np.array(f"/fake/{name}"),
    )
    return path


@pytest.fixture
def truth(tmp_path, monkeypatch):
    """A stand-in for the mock's saved truth, for its two sigma arrays."""
    from paper1 import mock_forward_digest as M

    path = tmp_path / "mock_joint_truth.npz"
    np.savez(path, phot_sig=np.array([1.0, 1.0]), spec_sig=np.ones(4))
    monkeypatch.setattr(M, "TRUTH", path)
    return path


def test_different_free_parameters_are_refused(tmp_path, truth):
    """The defect this guard exists for: same shapes, different models."""
    from paper1.mock_forward_digest import compare

    a = _digest(tmp_path, "a.npz", [1.0, 2.0], ["sfh_tau", "met_logzsol"])
    b = _digest(tmp_path, "b.npz", [1.0, 2.0], ["sfh_tau", "neb_logU"])

    with pytest.raises(SystemExit) as excinfo:
        compare(a, b)
    assert "different free parameters" in str(excinfo.value)


def test_identical_trees_compare_cleanly(tmp_path, truth, capsys):
    from paper1.mock_forward_digest import compare

    a = _digest(tmp_path, "a.npz", [1.0, 2.0], ["sfh_tau"])
    b = _digest(tmp_path, "b.npz", [1.0, 2.0], ["sfh_tau"])

    assert compare(a, b) == 0
    out = capsys.readouterr().out
    assert "2/2 bit-identical" in out
    assert "worst shift across both channels: 0.0000 sigma" in out


def test_a_moved_band_is_named_with_its_shift(tmp_path, truth, capsys):
    """A summary that only reported the worst number would hide which band."""
    from paper1.mock_forward_digest import compare

    a = _digest(tmp_path, "a.npz", [1.0, 2.0], ["sfh_tau"])
    b = _digest(tmp_path, "b.npz", [1.5, 2.0], ["sfh_tau"])

    compare(a, b)
    out = capsys.readouterr().out
    assert "galex_nuv" in out
    assert "sdss_r" not in out, "an unmoved band should not be listed"
    assert "0.5000 sigma" in out


def test_a_shape_mismatch_is_refused_rather_than_broadcast(tmp_path, truth):
    """numpy would happily subtract a length-1 array from a length-2 one."""
    from paper1.mock_forward_digest import compare

    a = _digest(tmp_path, "a.npz", [1.0, 2.0], ["sfh_tau"])
    b = _digest(tmp_path, "b.npz", [1.0], ["sfh_tau"])

    with pytest.raises(SystemExit) as excinfo:
        compare(a, b)
    assert "shapes differ" in str(excinfo.value)
