# SPDX-License-Identifier: BSD-3-Clause
"""The stochastic field SFH is reachable via the nested-dict ``field`` sub-block.

Fresh-user audit (2026-07): the IFT correlated-field burstiness (a headline
feature) was only reachable via the list form ``sfh=['dpl', 'field']`` /
``sfh={'type': ['dpl', 'field']}``. A user mirroring the ``dust={'emission':
{...}}`` idiom reached for ``sfh={'type': 'dpl', 'field': {...}}`` and hit a
bare "Unknown key 'field'". The grammar now accepts the ``field`` modulator
sub-block (normalized to the list-``type`` composition), with a ``'*'`` wildcard
scoped to the field/PSD params only.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import FIXED, FREE, Fixed, SEDModel, Uniform

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_FIELD = {"sfh_field_psd_sigma", "sfh_field_psd_tau_myr"}


def _build(ssp, obs, sfh):
    return SEDModel.build(ssp_data=ssp, observation=obs, redshift=Fixed(0.05), sfh=sfh)


def _free(model):
    return set(model.spec.free_params)


def test_field_subblock_enables_modulator(synthetic_ssp_wide, synthetic_tophat_obs):
    model = _build(
        synthetic_ssp_wide, synthetic_tophat_obs, {"type": "dpl", "field": {"all_params": FREE}}
    )
    assert _free(model) >= _FIELD, "field PSD params should be free"


def test_field_subblock_true(synthetic_ssp_wide, synthetic_tophat_obs):
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, {"type": "dpl", "field": True})
    # 'field' modulator is composed in; with no '*' the PSD params sit at defaults.
    assert "field" in _as_list(model.spec.mean_sfh_type)


def test_field_wildcard_scoped_to_field_params(synthetic_ssp_wide, synthetic_tophat_obs):
    """A field '*' frees ONLY the field params; the smooth SFH keeps its own."""
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        {"type": "dpl", "all_params": FIXED, "field": {"all_params": FREE}},
    )
    free = _free(model)
    assert free >= _FIELD
    assert not any(p.startswith("sfh_dpl_") for p in free), "dpl params must stay fixed"


def test_field_explicit_prior(synthetic_ssp_wide, synthetic_tophat_obs):
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        {"type": "dpl", "field": {"psd_sigma": Uniform(0.1, 4.0)}},
    )
    assert "sfh_field_psd_sigma" in _free(model)
    assert "sfh_field_psd_tau_myr" not in _free(model)


def test_list_form_still_works(synthetic_ssp_wide, synthetic_tophat_obs):
    model = _build(
        synthetic_ssp_wide, synthetic_tophat_obs, {"type": ["dpl", "field"], "all_params": FREE}
    )
    assert _free(model) >= _FIELD


def test_plain_dpl_unaffected(synthetic_ssp_wide, synthetic_tophat_obs):
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, {"type": "dpl", "all_params": FREE})
    assert not (_FIELD & _free(model)), "plain dpl must not gain field params"


def test_field_model_predicts_finite(synthetic_ssp_wide, synthetic_tophat_obs):
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        {"type": "dpl", "all_params": FIXED, "field": {"all_params": FIXED}},
    )
    phot = np.asarray(model.predict_photometry(model.spec.sample(jax.random.PRNGKey(0))))
    assert np.all(np.isfinite(phot))


def _as_list(t):
    return list(t) if isinstance(t, (list, tuple)) else [t]
