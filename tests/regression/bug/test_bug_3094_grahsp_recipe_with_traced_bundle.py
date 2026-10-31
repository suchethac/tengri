# SPDX-License-Identifier: BSD-3-Clause
"""The GRAHSP recipe normalization accepts a template bundle passed through ``jax.jit`` (#3094).

``disc/grahsp_netzer`` declares ``load_grahsp_templates`` as its ``template_loader``, so the
bundle reaches the block as a traced argument. The normalization integrals need the template
wavelength nodes on the host (they pick which nodes lie above the lumBolBBB cut), so those
axes are read from the cached bundle when the argument is traced.
"""

import warnings

import jax
import numpy as np
import pytest

from tengri.components.agn.blocks._grahsp_recipe import grahsp_recipe_log_l5100
from tengri.components.agn.grahsp.templates import load_grahsp_templates

pytestmark = pytest.mark.regression_bug

_LOG_LBOL = 45.0


def _log_l5100(templates, feii_block):
    return grahsp_recipe_log_l5100(
        _LOG_LBOL,
        {"agn_type": 1},
        disc_block="grahsp_netzer",
        nlr_block="grahsp",
        blr_block="grahsp",
        feii_block=feii_block,
        templates=templates,
    )


@pytest.mark.parametrize("feii_block", ["grahsp", "grahsp_veroncetty"])
def test_recipe_normalization_under_jit_with_a_traced_bundle(feii_block):
    """A jitted recipe with the bundle as an argument equals the eager recipe."""
    templates = load_grahsp_templates()
    eager = float(_log_l5100(templates, feii_block))
    traced = float(jax.jit(lambda t: _log_l5100(t, feii_block))(templates))
    assert np.isfinite(traced)
    np.testing.assert_allclose(traced, eager, rtol=1e-6)


def test_composable_model_with_unset_l5100_predicts_with_the_bundle_threaded(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The composable model leaves ``agn_grahsp_log_l5100`` unset and predicts finite fluxes."""
    from tengri import DEFAULT, Fixed, SEDModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "grahsp_netzer"},
                "blr": {"type": "grahsp"},
                "feii": {"type": "grahsp_veroncetty"},
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(2.0),
        )
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        assert params.get("agn_grahsp_log_l5100") is None
        photometry = np.asarray(model.predict_photometry(params))
    assert np.all(np.isfinite(photometry))
