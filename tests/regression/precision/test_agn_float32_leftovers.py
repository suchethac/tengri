# SPDX-License-Identifier: BSD-3-Clause
"""float32 toy-torus luminosities stay finite and match float64 (#3401 leftovers).

``simple_torus`` and ``two_temperature_torus`` formed ``10**agn_log_lbol * L_sun``
as a linear erg/s scale before the float32 product. Above ``agn_log_lbol`` ~ 4.9
that power is past float32's 3.4e38 limit, so the emitted SED was ``nan`` at every
luminosity the prior covers. The float32 arm now carries the power as a log10
offset through ``_template_grid.scale_by_torus_power``, the helper the template
tori already use (#3084).

The float32 arm draws its inputs in float64 first and then evaluates under
``jax.enable_x64(False)``. Every gradient assertion checks finite AND nonzero.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.torus import simple_torus, two_temperature_torus

pytestmark = pytest.mark.regression_bug

LOG_LBOLS = (10.0, 11.0, 12.0, 13.0)
#: Wavelength span [Angstrom] where both tori emit: the Wien tail below 1e3 A
#: is negligible and underflows in float32 by design.
WAVE_LO, WAVE_HI, N_WAVE = 1.0e3, 3.0e4, 64
#: Element-wise float32 vs float64 tolerance, relative to max|float64|.
RTOL_PEAK = 1e-4

SITES = {
    "simple_torus": simple_torus,
    "two_temperature_torus": two_temperature_torus,
}


def _wave64() -> np.ndarray:
    return np.logspace(np.log10(WAVE_LO), np.log10(WAVE_HI), N_WAVE)


def _sed_f64(site: str, log_lbol: float) -> np.ndarray:
    wave = jnp.asarray(_wave64(), dtype=jnp.float64)
    lbol = jnp.asarray(log_lbol, dtype=jnp.float64)
    return np.asarray(SITES[site](wave, lbol), dtype=np.float64)


def _sed_f32(site: str, log_lbol: float) -> np.ndarray:
    with jax.enable_x64(False):
        wave = jnp.asarray(_wave64(), dtype=jnp.float32)
        lbol = jnp.asarray(log_lbol, dtype=jnp.float32)
        return np.asarray(SITES[site](wave, lbol), dtype=np.float64)


def _grad_f32(site: str, log_lbol: float) -> float:
    fn = SITES[site]

    def total(lbol):
        wave = jnp.asarray(_wave64(), dtype=jnp.float32)
        return jnp.sum(fn(wave, lbol))

    with jax.enable_x64(False):
        return float(jax.grad(total)(jnp.asarray(log_lbol, dtype=jnp.float32)))


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("log_lbol", LOG_LBOLS)
def test_sed_float32_is_finite_and_nonzero(site, log_lbol):
    sed = _sed_f32(site, log_lbol)
    assert np.all(np.isfinite(sed)), f"{site} float32 SED has non-finite nodes at {log_lbol}"
    assert np.max(sed) > 0.0


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("log_lbol", LOG_LBOLS)
def test_sed_float32_matches_float64(site, log_lbol):
    ref = _sed_f64(site, log_lbol)
    sed = _sed_f32(site, log_lbol)
    peak = np.max(np.abs(ref))
    assert np.max(np.abs(sed - ref)) <= RTOL_PEAK * peak


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("log_lbol", LOG_LBOLS)
def test_gradient_float32_wrt_agn_log_lbol_is_finite_and_nonzero(site, log_lbol):
    grad = _grad_f32(site, log_lbol)
    assert np.isfinite(grad), f"{site} d/d(agn_log_lbol) is {grad} at {log_lbol}"
    assert grad != 0.0
