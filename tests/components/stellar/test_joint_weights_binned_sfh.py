# SPDX-License-Identifier: BSD-3-Clause
"""``compute_joint_weights`` serves the binned SFH families (#3505, row B15).

The fast path used to refuse the six binned (non-parametric) SFH families with a
``ValueError``. It already integrates them on the same bin-edge knots that
``apply`` uses (``bin_edges_sfh_kwarg`` and the shared ``_cic_integrand``), so the
refusal guarded nothing. These tests require the fast path to return the same joint
weights and formed mass as ``apply`` publishes, at a non-trivial parameter point
drawn from each family's own prior.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import Fixed, SEDModel
from tengri.components.stellar.component import StellarSEDComponent
from tengri.parameters.resolve import merge_fixed_params

pytestmark = pytest.mark.regression_bug

#: Registry keys of the six binned families the fast path used to refuse.
BINNED_SFH = (
    "continuity",
    "continuity_young_fraction",
    "continuity_flex",
    "dirichlet",
    "psb_suess2022",
    "psb_flex",
)


def _build(sfh_name, ssp, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": sfh_name},
            dust_attenuation=None,
            neb={"type": "none"},
            redshift=Fixed(0.5),
        )


def _stellar_of(model):
    chain = model._build_component_chain()
    return next(c for c in chain if isinstance(c, StellarSEDComponent))


@pytest.mark.parametrize("sfh_name", BINNED_SFH)
def test_joint_weights_match_apply_for_binned_sfh(
    sfh_name, synthetic_ssp_wide, synthetic_tophat_obs
):
    model = _build(sfh_name, synthetic_ssp_wide, synthetic_tophat_obs)
    stellar = _stellar_of(model)
    spec = model.spec

    # A non-trivial point: one draw from the family's own prior, with the Fixed
    # entries (redshift among them) merged in for the fast path.
    params = merge_fixed_params(spec, dict(spec.sample(jax.random.PRNGKey(7))))
    jw_fast, mass_fast, _ages = stellar.compute_joint_weights(params)

    fixed_names = set(spec.fixed_params)
    state = model.predict_state({k: v for k, v in params.items() if k not in fixed_names})
    jw_exact = np.asarray(state.derived["joint_weights"])
    mass_exact = float(10.0 ** np.asarray(state.derived["log_mstar_formed"]))

    assert np.all(np.isfinite(np.asarray(jw_fast))), "weights must be finite"
    np.testing.assert_allclose(np.asarray(jw_fast), jw_exact, rtol=1e-10, atol=0.0)
    assert float(mass_fast) > 0.0
    np.testing.assert_allclose(float(mass_fast), mass_exact, rtol=1e-10)
