# SPDX-License-Identifier: BSD-3-Clause
"""Pin: qsogen monolithic ≡ sum of qsogen blocks.

Verifies that ``compute_qsogen_sed`` (the registered AGN_MODELS["qsogen"]
entry) is bit-for-bit reproduced by composing the 5 qsogen blocks via
``composable_agn_l_nu``. Tolerance < 1e-12 because both paths share the
same internal helper :func:`_qsogen_components`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

from tengri.components.agn.blocks import composable_agn_l_nu
from tengri.components.agn.qsogen import compute_qsogen_sed


@pytest.mark.parametrize(
    "params",
    [
        # Default qsogen settings.
        dict(),
        # Strong dust reddening + Balmer continuum on.
        dict(agn_ebv=0.3, agn_bcnorm=1.0),
        # Bluer continuum, hotter dust.
        dict(agn_plslp1=-0.7, agn_plslp2=0.4, agn_tbb=1500.0),
        # Heavy emission-line scaling, suppressed hot dust.
        dict(agn_emline_scale=2.5, agn_bbnorm=1.0),
        # Off-default break.
        dict(agn_plbrk=4500.0, agn_plslp1=-0.2),
    ],
)
def test_qsogen_blocks_sum_to_monolithic(params):
    """Composable qsogen recipe == compute_qsogen_sed for all parameter sets."""
    wave_aa = jnp.logspace(2.96, 5.0, 600)
    agn_log_lbol = 45.5

    out_monolithic = compute_qsogen_sed(wave_aa, agn_log_lbol=agn_log_lbol, **params)
    out_composable = composable_agn_l_nu(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_disc_block="qsogen",
        agn_nlr_block="none",
        agn_blr_block="qsogen",
        agn_feii_block="qsogen_balmer",
        agn_torus_block="qsogen",
        agn_attenuation_block="qsogen_smc",
        **params,
    )
    np.testing.assert_allclose(
        np.asarray(out_composable),
        np.asarray(out_monolithic),
        rtol=1e-12,
        atol=0.0,
    )


def test_qsogen_components_helper_keys():
    """Internal helper exports all five expected components."""
    from tengri.components.agn.qsogen import _qsogen_components

    out = _qsogen_components(
        jnp.logspace(2.96, 5.0, 100),
        agn_plslp1=-0.349,
        agn_plslp2=0.593,
        agn_plbrk=3880.0,
        agn_tbb=1240.0,
        agn_bbnorm=3.96,
        agn_emline_scale=1.0,
        agn_ebv=0.0,
        agn_log_lbol=11.42,
        agn_bcnorm=0.0,
    )
    assert set(out.keys()) == {
        "continuum",
        "hot_dust",
        "emission_lines",
        "balmer_continuum",
        "smc_factor",
    }
    for k, arr in out.items():
        assert jnp.all(jnp.isfinite(arr)), f"{k} has non-finite values"
