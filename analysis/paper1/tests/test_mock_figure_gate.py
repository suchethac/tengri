# SPDX-License-Identifier: BSD-3-Clause
"""The mock figure may not carry the paper's filename from a failed fit.

``fig01_mock_joint_infer.py`` chose its output name on ``posterior is not
None`` -- an existence check. It published ``fig01_mock_joint_infer.pdf``, the
name the paper includes, from a posterior with ``ess_min`` 53.5, four
divergences and no recorded R-hat at all.

The case these tests exist for is the unrecorded diagnostic. ``rhat_max`` is
``None`` on the posterior that provoked this, because it predates
``fit_mock_joint.py`` calling the R-hat accessor by its real name. Written as
``if rhat > 1.01: refuse``, ``None`` compares false and the figure publishes
itself: absence and success are the same value to that comparison. Recomputed
from its own draws that posterior's R-hat is 1.0550, so the file a
presence-blind check waves through is exactly the one that had to be stopped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _posterior_gate import (
    GATE_ESS_MIN,
    GATE_MAX_DIVERGENCES,
    GATE_RHAT_MAX,
    posterior_gate,
)

pytestmark = pytest.mark.contract

#: A posterior that clears every clause. Every failing case below is this
#: dictionary with exactly one field spoiled, so a failure names one cause.
CONVERGED = {
    "rhat_max": 1.004,
    "ess_min": 900.0,
    "divergences": 0,
    "method": "mcmc_nuts",
}


def _write(tmp_path: Path, diagnostics: dict | None, stem: str = "mock_joint_mcmc_nuts") -> Path:
    """Lay out draws and their sidecar the way the sampler does."""
    npz = tmp_path / f"{stem}.npz"
    npz.write_bytes(b"not read by the gate")
    if diagnostics is not None:
        npz.with_suffix(".json").write_text(json.dumps(diagnostics))
    return npz


def test_a_converged_posterior_passes(tmp_path):
    """The gate must not refuse everything, or it says nothing when it refuses."""
    passed, reasons, _ = posterior_gate(_write(tmp_path, CONVERGED))
    assert passed, reasons
    assert reasons == []


def test_an_unrecorded_rhat_fails_rather_than_passing(tmp_path):
    """The defect this gate was written for: absence read as success."""
    passed, reasons, _ = posterior_gate(_write(tmp_path, {**CONVERGED, "rhat_max": None}))
    assert not passed
    assert any("never recorded" in r for r in reasons)


def test_the_real_failed_posterior_is_refused(tmp_path):
    """The actual numbers on disk, which published under the paper's name."""
    passed, reasons, _ = posterior_gate(
        _write(tmp_path, {"rhat_max": None, "ess_min": 53.49303100356316, "divergences": 4})
    )
    assert not passed
    # All three clauses fire; a one-clause gate would still have caught it, but
    # not for every reason it is wrong.
    assert len(reasons) == 3


def test_a_high_rhat_fails(tmp_path):
    passed, reasons, _ = posterior_gate(
        _write(tmp_path, {**CONVERGED, "rhat_max": GATE_RHAT_MAX + 0.001})
    )
    assert not passed
    assert any("rhat_max" in r for r in reasons)


def test_too_few_effective_samples_fail(tmp_path):
    passed, reasons, _ = posterior_gate(
        _write(tmp_path, {**CONVERGED, "ess_min": GATE_ESS_MIN - 1.0})
    )
    assert not passed
    assert any("ess_min" in r for r in reasons)


def test_any_divergence_fails(tmp_path):
    passed, reasons, _ = posterior_gate(
        _write(tmp_path, {**CONVERGED, "divergences": GATE_MAX_DIVERGENCES + 1})
    )
    assert not passed
    assert any("divergence" in r for r in reasons)


def test_a_missing_sidecar_fails(tmp_path):
    """Draws with no diagnostics beside them are unverified, not fine."""
    passed, reasons, _ = posterior_gate(_write(tmp_path, None))
    assert not passed
    assert any("missing" in r for r in reasons)


def test_an_unreadable_sidecar_fails(tmp_path):
    npz = _write(tmp_path, CONVERGED)
    npz.with_suffix(".json").write_text("{ truncated")
    passed, reasons, _ = posterior_gate(npz)
    assert not passed
    assert any("cannot be read" in r for r in reasons)
