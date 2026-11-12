# SPDX-License-Identifier: BSD-3-Clause
r"""A vmapped float32 AGN disc must match the single evaluation (#3105).

``jax.vmap`` of the composable AGN over ``agn_cos_inc`` changed the multicolor disc behind
the ``smc_prevot`` attenuation by up to 1e8 at 30-75 A, the EUV tail behind the screen. The
cause: the disc's closed-form bolometric power ``sum(B * T^4 * ring_area) / L_sun`` has the
constant ``B / L_sun`` ~ 4.7e-39, a subnormal float32 value. Under the vmapped graph XLA
folds that constant and flushes it to zero, so the bolometric power was 0, the EUV tail
amplitude was 0 and only the Wien core (exp-suppressed at 30-75 A) survived. The single
evaluation keeps the constant unfolded and is correct.

Errors are measured peak-relative against the single evaluation, in pure float32.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.components.agn.disc import multicolor_disc

pytestmark = pytest.mark.regression_bug

_WAVE = np.geomspace(20.0, 1.0e4, 3001)
_TOL = 1.0e-4  # peak-relative, float32 vs float32


def _peak_rel(batched: np.ndarray, single: np.ndarray) -> float:
    peak = np.abs(single).max()
    return float(np.abs(batched - single).max() / peak)


def test_multicolor_disc_vmap_over_cos_inc_matches_single_in_float32():
    """The bare multicolor disc, vmapped over the inclination, equals the single call."""
    with jax.enable_x64(False):
        w = jnp.asarray(_WAVE, dtype=jnp.float32)
        mu = jnp.asarray([0.35, 0.7], dtype=jnp.float32)

        def disc(cos_inc):
            return multicolor_disc(
                w, agn_log_lbol=jnp.asarray(10.5, jnp.float32), agn_cos_inc=cos_inc
            )

        single = np.asarray(jax.jit(disc)(mu[0]))
        batched = np.asarray(jax.jit(jax.vmap(disc))(mu))[0]

    assert single.dtype == np.float32
    assert np.all(np.isfinite(batched))
    assert np.max(np.abs(batched)) > 0.0, "vmapped multicolor disc collapsed to zero"
    rel = _peak_rel(batched, single)
    assert rel < _TOL, (
        f"vmap over agn_cos_inc changes the float32 multicolor disc: peak-relative error "
        f"{rel:.2e} (tolerance {_TOL:.0e}); the 30-75 A EUV tail is zeroed in the batched graph"
    )


def test_composable_agn_disc_vmap_over_cos_inc_matches_single_in_float32():
    """The composable AGN disc behind ``smc_prevot`` (the #3105 recipe), vmap vs single."""

    def disc(cos_inc):
        return composable_agn_l_nu(
            jnp.asarray(_WAVE, dtype=jnp.float32),
            10.5,
            return_components=True,
            agn_disc_block="multicolor",
            agn_torus_block="two_temperature",
            agn_attenuation_block="smc_prevot",
            agn_norm="conserving",
            agn_torus_frac=0.4,
            agn_ebv=0.3,
            agn_cos_inc=cos_inc,
        )[1]["disc"]

    with jax.enable_x64(False):
        mu = jnp.asarray([0.35, 0.7], dtype=jnp.float32)
        single = np.asarray(jax.jit(disc)(mu[0]))
        batched = np.asarray(jax.jit(jax.vmap(disc))(mu))[0]

    assert np.all(np.isfinite(batched))
    assert np.max(np.abs(batched)) > 0.0, "vmapped composable disc collapsed to zero"
    rel = _peak_rel(batched, single)
    assert rel < _TOL, (
        f"vmap over agn_cos_inc changes the float32 composable disc behind smc_prevot: "
        f"peak-relative error {rel:.2e} (tolerance {_TOL:.0e})"
    )
