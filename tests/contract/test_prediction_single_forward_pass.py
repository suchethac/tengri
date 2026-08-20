# SPDX-License-Identifier: BSD-3-Clause
"""``model.predict(params)`` runs exactly one forward pass across groups.

The Prediction wrapper's documented contract is one cached ForwardState
shared by the ``sfh`` / ``sed`` / ``lines`` property groups. Before this
fix, ``pred.lines`` re-ran ``predict_state`` after ``_ensure_sfh`` had
already cached the state, doubling the forward-model cost (and its
transient memory peak) of the *recommended* interactive path — while the
deprecated ``predict_emission_lines()`` ran it once, so memory-constrained
users saw the recommended API die where the deprecated one survived.
"""

import os
import warnings

import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel

_WEIGHTS = os.path.join("data", "cue_weights.npz")
pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(
        not os.path.exists(_WEIGHTS), reason="data/cue_weights.npz absent (data-gated)"
    ),
]


@pytest.fixture(scope="module")
def cue_model(synthetic_ssp_wide):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            synthetic_ssp_wide,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(0.0),
                "*": FIXED,
            },
            dust_attenuation={"law": "power_law", "type": "two_component", "*": FIXED},
            neb={"type": "cue", "*": FIXED},
            redshift=Fixed(0.0),
        )


def test_sed_quantities_use_state_wave_grid(cue_model):
    """Integrated SED quantities must use the pipeline's own wave grid.

    The ForwardState evaluates on its own grid (auto-extended for dust
    emission, trimmed to the modeling range) — before this fix,
    ``pred.sed.l_bol`` and the luminosity-weighted age/metallicity
    integrated state-grid SED arrays against the raw ``ssp_wave`` axis
    and crashed with a broadcasting TypeError whenever the two differed
    (the README-quickstart recipe model among them).
    """
    pred = cue_model.predict({})
    assert np.isfinite(float(pred.sed.l_bol))
    assert np.isfinite(float(pred.sed.luminosity_weighted_age_gyr))


def test_lines_access_is_single_forward_pass(cue_model, monkeypatch):
    calls = []
    orig_predict_state = type(cue_model).predict_state

    def counting_predict_state(self, *args, **kwargs):
        calls.append(1)
        return orig_predict_state(self, *args, **kwargs)

    monkeypatch.setattr(type(cue_model), "predict_state", counting_predict_state)

    pred = cue_model.predict({})
    halpha = pred.lines.halpha
    _ = pred.sfh.stellar_mass

    assert np.isfinite(float(halpha)), "Cue line catalog was not exercised"
    assert len(calls) == 1, (
        f"predict(params) ran the forward model {len(calls)} times across "
        "lines/sfh/sed accesses — the Prediction wrapper must reuse the "
        "one ForwardState cached by _ensure_sfh (see prediction.py:_ensure_lines)."
    )
