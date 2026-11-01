# SPDX-License-Identifier: BSD-3-Clause
"""The GRAHSP recipe normalization accepts a template bundle passed through ``jax.jit`` (#3094).

``disc/grahsp_netzer`` declares ``load_grahsp_templates`` as its ``template_loader``, so the
bundle reaches the block as a traced argument. The normalization integrals need the template
wavelength nodes on the host (they pick which nodes lie above the lumBolBBB cut), so the disc
and FeII wavelength axes are static aux data of the bundle pytree and stay concrete in ``jit``.
"""

import dataclasses
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


def test_a_custom_axis_is_used_under_jit_not_the_default_one():
    """A bundle with a shifted disc axis normalizes on its own nodes, jitted or eager."""
    default = load_grahsp_templates()
    custom = dataclasses.replace(default, disc_wave_nm=np.asarray(default.disc_wave_nm) * 1.07)
    eager_default = float(_log_l5100(default, "grahsp"))
    eager_custom = float(_log_l5100(custom, "grahsp"))
    assert abs(eager_custom - eager_default) > 1e-4  # the axis matters to the result
    traced_custom = float(jax.jit(lambda t: _log_l5100(t, "grahsp"))(custom))
    np.testing.assert_allclose(traced_custom, eager_custom, rtol=1e-9)


@pytest.mark.parametrize(
    ("feii_block", "axis"),
    [("grahsp", "feii_wave_nm"), ("grahsp_veroncetty", "feii_vc04_wave_nm")],
)
def test_a_custom_feii_axis_is_used_under_jit(feii_block, axis):
    """A shifted FeII axis changes the normalization, and jit agrees with eager on it."""
    default = load_grahsp_templates()
    custom = dataclasses.replace(default, **{axis: np.asarray(getattr(default, axis)) * 1.07})
    eager_default = float(_log_l5100(default, feii_block))
    eager_custom = float(_log_l5100(custom, feii_block))
    assert abs(eager_custom - eager_default) > 1e-6
    traced_custom = float(jax.jit(lambda t: _log_l5100(t, feii_block))(custom))
    np.testing.assert_allclose(traced_custom, eager_custom, rtol=1e-9)


def test_equal_bundles_share_a_trace_and_a_different_axis_retraces():
    """The static axes hash by value: equal bundles hit the jit cache, a shifted axis misses."""
    default = load_grahsp_templates()
    same = dataclasses.replace(default, disc_wave_nm=np.array(default.disc_wave_nm))
    shifted = dataclasses.replace(default, disc_wave_nm=np.asarray(default.disc_wave_nm) * 1.07)
    traces = []

    @jax.jit
    def run(t):
        traces.append(1)
        return _log_l5100(t, "grahsp")

    run(default)
    run(same)
    assert len(traces) == 1
    run(shifted)
    assert len(traces) == 2


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
