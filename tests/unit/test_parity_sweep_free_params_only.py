# SPDX-License-Identifier: BSD-3-Clause
"""The parity sweep hands the model its free parameters and nothing else.

A reference file records the full ``spec.sample`` truth, pinned parameters included, so
the file says what the float64 run saw. Since #2296 a call-time value for a ``Fixed``
parameter raises ``ParameterError`` instead of overriding the pin, and a sweep that
passes the whole truth back in cannot measure a single seam: the float64 self-check on
``e05c13078`` reported all six unmeasured. The pins come from the model; a changed pin
shows up in the self-check as float64 drift, which is where it belongs.
"""

import sys
from pathlib import Path
from typing import ClassVar

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[2] / "bench" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import benchmark_float32_mps_parity as sweep

_FREE = ["dust_tau_diff", "sfh_delayed_log_total_mass"]
_TRUTH = {"dust_tau_diff": 0.3, "sfh_delayed_log_total_mass": 10.0, "redshift": 0.1}


class _Spec:
    free_params = tuple(_FREE)


class _PinnedModel:
    """Behaves like an ``SEDModel`` since #2296: a value for a pinned name raises."""

    spec = _Spec()

    def predict_photometry(self, params):
        pinned = sorted(set(params) - set(_FREE))
        if pinned:
            raise ValueError(f"params overrides Fixed parameter(s): {pinned}")
        return jnp.stack([params["dust_tau_diff"], params["sfh_delayed_log_total_mass"]])


def test_gradient_stage_measures_a_seam_whose_truth_records_pinned_values(monkeypatch):
    import tengri

    monkeypatch.setattr(tengri.SEDModel, "build", classmethod(lambda cls, **kw: _PinnedModel()))
    ref_row = {
        "free_names": _FREE,
        "truth": dict(_TRUTH),
        "photometry": [0.3, 10.0],
        "grad": [0.0, 0.0],
        "mock_flux": [0.3, 10.0],
        "mock_noise": [0.1, 0.1],
    }
    rec = sweep._run_seam(
        "stub", {}, None, None, 0.1, 10, False, ref_row, stage="grad", dtype=jnp.float64
    )
    assert "error" not in rec, rec.get("error")
    assert rec["rel_forward"] == 0.0


class _Fit:
    """Stands in for ``ForwardModel``; records the ``init_from`` it was handed."""

    inits: ClassVar[list[dict]] = []

    @classmethod
    def build(cls, *, sed, observation):
        return cls()

    def fit(self, flux, noise, *, init_from, **kwargs):
        from tengri import Posterior

        type(self).inits.append(dict(init_from.params))
        return Posterior(
            samples=None,
            params={k: jnp.asarray(_TRUTH[k]) for k in _FREE},
            method="MAP (stub)",
            wall_time_s=0.0,
            diagnostics={"n_steps": 1, "converged": True},
            loss_history=jnp.asarray([0.5]),
            _model=None,
        )


def test_map_stage_starts_from_the_free_parameters_only():
    _Fit.inits.clear()
    sweep._map_loss(_Fit, None, None, [1.0], [0.1], dict(_TRUTH), _FREE, jnp.float64, 10)
    assert [sorted(p) for p in _Fit.inits] == [sorted(_FREE)]
