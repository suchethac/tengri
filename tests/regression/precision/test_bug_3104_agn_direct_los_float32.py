# SPDX-License-Identifier: BSD-3-Clause
r"""#3104: the composable AGN line-of-sight luminosity is finite in pure float32 under jit.

``log_L_agn_direct_los`` (``log_L_agn_los`` before #3076) was NaN for the ``kubota_done`` and
``multicolor`` discs in pure float32 once the runner was traced under ``jax.jit``, with a finite
disc SED and no error. The eager float32 path was finite.

The cause is the Planck prefactor ``2 h nu (nu/c)**2``. Under jit XLA re-associates the constant
products inside the vmapped ring sum of the multicolor Wien disc, and at every frequency where
``exp(-x)`` underflows to exactly zero (the Wien tail, lambda below about 0.05 A for a 2e5 K ring)
the forward value is ``inf * 0 = NaN``. The same class as #2767. The float32 grouping is changed
in :func:`tengri.utils.blackbody.planck_bnu_nu`; the float64 expression is unchanged, so float64
is bit-identical.

The reproducer requires the jitted float32 line-of-sight power to be finite and to agree with
float64 to 1e-4 dex, and the jitted float32 disc SED to match float64 to 1e-4 of its peak
(``max|Delta| / max|L_nu|``, the measure of #3166).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.utils.blackbody import planck_bnu_nu
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug

_WAVE = np.geomspace(1.0e2, 1.0e7, 120)  # Angstrom
_LOG_LBOL = 11.0
_RECIPE = dict(
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="simple",
    agn_attenuation_block="none",
    agn_norm="conserving",
)
_DEX = 1.0e-4
_RTOL_SED = 1.0e-4


def _direct_key(comps):
    return comps["log_L_agn_direct_los"]


def _run_f64_eager(disc):
    with jax.enable_x64(True):
        lnu, comps = composable_agn_l_nu(
            jnp.asarray(_WAVE, dtype=jnp.float64),
            _LOG_LBOL,
            return_components=True,
            agn_disc_block=disc,
            agn_cos_inc=0.4,
            **_RECIPE,
        )
        return np.asarray(lnu, np.float64), float(_direct_key(comps))


def _run_f32_jit(disc):
    """The composable runner in pure float32, traced under jit with the luminosity as an input."""
    with jax.enable_x64(False):

        @jax.jit
        def fn(wave, log_lbol):
            lnu, comps = composable_agn_l_nu(
                wave,
                log_lbol,
                return_components=True,
                agn_disc_block=disc,
                agn_cos_inc=0.4,
                **_RECIPE,
            )
            return lnu, _direct_key(comps)

        lnu, key = fn(
            jnp.asarray(_WAVE, dtype=jnp.float32), jnp.asarray(_LOG_LBOL, dtype=jnp.float32)
        )
        assert lnu.dtype == jnp.float32, "the float32 run did not stay in float32"
        assert key.dtype == jnp.float32
        return np.asarray(lnu, np.float64), float(key)


@pytest.mark.parametrize("disc", ["kubota_done", "multicolor"])
def test_direct_los_is_finite_and_float64_accurate_under_jit(disc):
    lnu64, key64 = _run_f64_eager(disc)
    lnu32, key32 = _run_f32_jit(disc)
    assert np.all(np.isfinite(lnu32)), f"{disc} float32 SED is non-finite under jit"
    assert np.isfinite(key32), f"{disc} log_L_agn_direct_los is {key32} in float32 under jit"
    assert abs(key32 - key64) <= _DEX, f"{disc}: float32 {key32} vs float64 {key64} dex"
    peak_rel = np.max(np.abs(lnu32 - lnu64)) / np.max(np.abs(lnu64))
    assert peak_rel <= _RTOL_SED, f"{disc} float32 SED {peak_rel:.3e} of peak"


def test_planck_ring_sum_is_finite_under_jit():
    """Minimal reproducer: a vmapped sum of Planck rings, as the multicolor disc forms it."""
    wave = np.geomspace(1.0e-3, 1.0e10, 13001)
    temps = np.geomspace(5.0e3, 2.0e5, 60)

    def ring_sum(nu, t_rings):
        return jnp.sum(jax.vmap(lambda t: planck_bnu_nu(nu, t))(t_rings), axis=0)

    with jax.enable_x64(True):
        nu64 = C_AA / jnp.asarray(wave, dtype=jnp.float64)
        ref = np.asarray(ring_sum(nu64, jnp.asarray(temps, jnp.float64)), np.float64)
    with jax.enable_x64(False):
        nu32 = C_AA / jnp.asarray(wave, dtype=jnp.float32)
        t32 = jnp.asarray(temps, jnp.float32)
        out = np.asarray(jax.jit(ring_sum)(nu32, t32), np.float64)
    assert np.all(np.isfinite(out)), f"{int((~np.isfinite(out)).sum())} non-finite points under jit"
    assert np.max(np.abs(out - ref)) / np.max(np.abs(ref)) <= _RTOL_SED
