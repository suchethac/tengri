# SPDX-License-Identifier: BSD-3-Clause
"""Direct call to composable runner in float32 (#2321).

The composable runner is a public entry point (`tengri.components.agn.blocks.runner`).
When called directly, it must apply float32 reference evaluation: evaluate blocks at
_AGN_LBOL_REF (low enough to stay finite), then rescale in log space.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from tengri.components.agn.blocks.runner import compose_l_nu

pytestmark = pytest.mark.regression_bug


def test_powerlaw_disc_float32_finite():
    """Wave/inputs built in float32 must stay finite when guard is active."""
    with jax.enable_x64(False):
        wave = jnp.logspace(1.5, 4.5, 256)
        assert wave.dtype == jnp.float32, f"wave must be float32, got {wave.dtype}"

        L_nu = compose_l_nu(
            wave,
            agn_log_lbol=12.0,
            agn_disc_block="powerlaw",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
        )

    inf_count = jnp.sum(~jnp.isfinite(L_nu))
    assert jnp.all(jnp.isfinite(L_nu)), f"Expected finite; got inf count {inf_count}"


def test_skirtor_torus_float32_finite():
    """SKIRTOR torus with inputs in float32 must stay finite."""
    with jax.enable_x64(False):
        wave = jnp.logspace(1.5, 4.5, 256)
        assert wave.dtype == jnp.float32

        L_nu = compose_l_nu(
            wave,
            agn_log_lbol=12.0,
            agn_disc_block="none",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="skirtor",
            agn_attenuation_block="none",
        )

    inf_count = jnp.sum(~jnp.isfinite(L_nu))
    assert jnp.all(jnp.isfinite(L_nu)), f"Expected finite; got inf count {inf_count}"


def test_powerlaw_disc_float32_red_mutation():
    """RED mutation: bypass reference guard implies inf (proves guard active)."""
    with jax.enable_x64(False):
        wave = jnp.logspace(1.5, 4.5, 256)

        # Monkeypatch: disable the reference evaluation guard
        def fake_eval(agn_log_lbol, wave_dtype):
            return agn_log_lbol, False, 0.0  # use_ref=False disables guard

        module_path = "tengri.components.agn.blocks.runner"
        eval_func = "_apply_float32_reference_evaluation"
        with patch(f"{module_path}.{eval_func}", fake_eval):
            L_nu_red = compose_l_nu(
                wave,
                agn_log_lbol=12.0,
                agn_disc_block="powerlaw",
                agn_nlr_block="none",
                agn_blr_block="none",
                agn_feii_block="none",
                agn_torus_block="none",
                agn_attenuation_block="none",
            )

    # RED: without guard, should overflow to inf
    inf_count = jnp.sum(~jnp.isfinite(L_nu_red))
    assert inf_count > 0, f"Mutation must produce inf; got {inf_count} infs"
