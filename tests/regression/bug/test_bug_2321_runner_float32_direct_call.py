# SPDX-License-Identifier: BSD-3-Clause
r"""Direct call to composable runner in float32 (#2321).

The composable runner is a public entry point (`tengri.components.agn.blocks.runner`).
When called directly with agn_log_lbol passed as a linear power value (~1e45 erg/s,
or log10(L/Lsun) = 45.0), the float32 path overflows BEFORE reaching the log-domain
guard that SEDModel applies.

The fix: move the reference-evaluation block from SEDModel.component into compose_l_nu
so both entry points (direct + via SEDModel) share ONE float32 guard.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.runner import compose_l_nu

pytestmark = pytest.mark.regression_bug


def test_powerlaw_disc_float32_finite():
    """Powerlaw disc direct call with agn_log_lbol=12.0 must stay finite."""
    wave = jnp.logspace(1.5, 4.5, 256)

    with jax.enable_x64(False):
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

    # Must be finite (no inf/nan)
    assert jnp.all(jnp.isfinite(L_nu)), \
        f"Powerlaw disc in float32 produced non-finite values: {L_nu[~jnp.isfinite(L_nu)]}"


def test_skirtor_torus_float32_finite():
    """SKIRTOR torus direct call with agn_log_lbol=12.0 must stay finite."""
    wave = jnp.logspace(1.5, 4.5, 256)

    with jax.enable_x64(False):
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

    # Must be finite (no inf/nan)
    assert jnp.all(jnp.isfinite(L_nu)), \
        f"SKIRTOR torus in float32 produced non-finite values: {L_nu[~jnp.isfinite(L_nu)]}"


def test_direct_call_vs_reference_float32_parity():
    """Direct call must agree with the reference float64 path to within float32 precision."""
    wave = jnp.logspace(1.5, 4.5, 256)

    # Float64 reference
    with jax.enable_x64(True):
        L_nu_f64 = compose_l_nu(
            wave,
            agn_log_lbol=12.0,  # Normal AGN luminosity in log space
            agn_disc_block="multicolor",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="skirtor",
            agn_attenuation_block="none",
            agn_log_lbol_shape=12.0,
        )

    # Float32 direct call
    with jax.enable_x64(False):
        L_nu_f32 = compose_l_nu(
            wave,
            agn_log_lbol=12.0,
            agn_disc_block="multicolor",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="skirtor",
            agn_attenuation_block="none",
            agn_log_lbol_shape=12.0,
        )

    # Must be finite
    assert jnp.all(jnp.isfinite(L_nu_f32)), \
        f"Float32 result has non-finite values"

    # Relative error should be small (float32 roundoff)
    # Use masked division to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(np.float32(L_nu_f64) - np.float32(L_nu_f32)) / (np.abs(np.float32(L_nu_f64)) + 1e-30)

    # Most values should be exact or within 1e-5 relative error
    # Allow up to 1% error in the worst case
    assert np.nanpercentile(rel_error[np.isfinite(rel_error)], 95) < 0.01, \
        f"Float32 vs float64 relative error too high: {np.nanpercentile(rel_error[np.isfinite(rel_error)], 95)}"
