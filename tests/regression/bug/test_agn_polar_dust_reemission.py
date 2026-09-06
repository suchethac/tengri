# SPDX-License-Identifier: BSD-3-Clause
"""Regression: AGN polar-dust re-emission wiring (CIGALE skirtor2016 polar dust).

The standalone ``atten={'type': 'polar_dust'}`` composable AGN block declares
three polar-dust re-emission knobs (``agn_polar_T``, ``agn_polar_beta``,
``agn_polar_oa``, all owned by :mod:`tengri.components.agn._params`), but the
re-emission helper that reads them,
:func:`tengri.components.agn.blocks.atten.polar_dust_reemission_lnu`, took a
keyword named ``agn_polar_temperature`` instead of the declared
``agn_polar_T``. Since the composable runner forwards every ``agn_*`` key
through ``**params``, a caller passing ``agn_polar_T`` (the only name the
grammar and every other polar-dust-aware block, e.g. the SKIRTOR torus block,
recognizes) silently fell through to ``**_params`` and was discarded: the
re-emission graybody was always evaluated at its hardcoded 100 K default,
regardless of the requested temperature.

The energy-balance test below additionally pins the CIGALE/Yang+2020
normalization this block already implements: the luminosity the polar screen
removes from the pre-attenuation SED must equal the luminosity re-emitted as
the graybody, to within numerical-integration noise.

References
----------
.. [1] Yang, A., et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust, section
   2.2.2). https://doi.org/10.1093/mnras/stz3001
.. [2] Boquien, M. et al. 2019, A&A, 622, A103, CIGALE ``skirtor2016`` module.
   arXiv:1811.03094.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug

#: 500 A to 1 mm, wide enough to catch the UV/optical disc AND the FIR
#: graybody bump the polar screen re-emits into.
_WAVE = jnp.logspace(jnp.log10(500.0), jnp.log10(1e8), 400)

#: Common composable-AGN recipe: disc-only (no torus/nlr/blr/feii), the
#: standalone polar_dust attenuation block, 'independent' normalization
#: (AGN norm-policy rule: always explicit in tests) so no cross-block energy
#: coupling confounds the polar-dust-only physics under test. Face-on
#: (agn_cos_inc=1.0) with a small opening angle so the smooth Type-1/2
#: sigmoid mask saturates close to 1 (deep Type 1): the observed
#: before/after-screen difference then equals the geometry-independent
#: absorbed luminosity the reemission normalizes to (Yang+2020 sec 2.2.2),
#: making the energy-balance check clean.
_BASE_PARAMS = dict(
    agn_log_lbol=12.0,
    agn_lum_ratio=1.0,
    agn_disc_block="multicolor",
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="none",
    agn_attenuation_block="polar_dust",
    agn_norm="independent",
    agn_cos_inc=1.0,
    agn_polar_ebv=0.3,
    agn_polar_oa=10.0,
    agn_polar_law="smc",
    agn_polar_T=100.0,
    agn_polar_beta=1.6,
)


def _integrate_lnu_bolometric(l_nu, wave=_WAVE):
    """Bolometric luminosity [erg/s] of an L_nu array: ``integral(L_lambda, dlambda)``.

    Converts to L_lambda and integrates over WAVELENGTH (ascending), matching
    the quadrature :func:`polar_dust_reemission_lnu` uses internally for
    ``l_absorbed_total``. Comparing quantities integrated this same way keeps
    the energy-balance check a physics assertion rather than a coarse-grid
    quadrature-scheme mismatch: integrating one curve over wavelength and
    another over frequency on the same finite log-spaced grid disagrees by
    ~4e-3 at 400 points here (shrinking only as the grid is refined), while
    integrating both consistently over wavelength agrees to ~1e-7 at any
    grid density (both are discretizations of the same continuous integral,
    L_bol = integral(L_lambda dlambda) = integral(L_nu dnu), but a finite
    trapezoidal rule is not scheme-invariant).
    """
    l_lambda = l_nu * C_AA / wave**2
    idx_w = jnp.argsort(wave)
    return jnp.trapezoid(l_lambda[idx_w], wave[idx_w])


class TestPolarDustTemperatureIsLive:
    """agn_polar_T must move the AGN SED (name-mismatch regression)."""

    def test_temperature_changes_sed(self):
        sed_cold = composable_agn_l_nu(_WAVE, **{**_BASE_PARAMS, "agn_polar_T": 50.0})
        sed_hot = composable_agn_l_nu(_WAVE, **{**_BASE_PARAMS, "agn_polar_T": 300.0})
        max_rel_diff = float(jnp.max(jnp.abs(sed_hot - sed_cold)) / jnp.max(jnp.abs(sed_cold)))
        assert max_rel_diff > 1e-3, (
            f"agn_polar_T=50K vs 300K changed the SED by only {max_rel_diff:.3e} "
            "relative -- the re-emission temperature has no effect."
        )


class TestPolarDustEnergyConservation:
    """Absorbed-by-screen luminosity must equal re-emitted graybody luminosity."""

    def test_absorbed_equals_reemitted_at_type1(self):
        p_before = {**_BASE_PARAMS, "agn_attenuation_block": "none"}
        sed_before = composable_agn_l_nu(_WAVE, **p_before)

        sed_after_total, components = composable_agn_l_nu(
            _WAVE, return_components=True, **_BASE_PARAMS
        )
        sed_agn_polar = components["polar"]
        sed_after_no_reemit = sed_after_total - sed_agn_polar

        absorbed = _integrate_lnu_bolometric(sed_before) - _integrate_lnu_bolometric(
            sed_after_no_reemit
        )
        reemitted = _integrate_lnu_bolometric(sed_agn_polar)

        rel_diff = abs(float(absorbed) - float(reemitted)) / float(reemitted)
        assert rel_diff < 1e-3, (
            f"absorbed={float(absorbed):.6e} erg/s, reemitted={float(reemitted):.6e} "
            f"erg/s, relative difference {rel_diff:.3e} (expected < 1e-3)."
        )


class TestPolarDustOpeningAngleAndBetaMoveSED:
    def test_opening_angle_changes_sed(self):
        sed_narrow = composable_agn_l_nu(_WAVE, **{**_BASE_PARAMS, "agn_polar_oa": 15.0})
        sed_wide = composable_agn_l_nu(_WAVE, **{**_BASE_PARAMS, "agn_polar_oa": 75.0})
        max_rel_diff = float(
            jnp.max(jnp.abs(sed_wide - sed_narrow)) / jnp.max(jnp.abs(sed_narrow))
        )
        assert max_rel_diff > 1e-3, (
            f"agn_polar_oa=15deg vs 75deg changed the SED by only {max_rel_diff:.3e} relative."
        )

    def test_beta_changes_sed(self):
        sed_lo = composable_agn_l_nu(_WAVE, **{**_BASE_PARAMS, "agn_polar_beta": 1.0})
        sed_hi = composable_agn_l_nu(_WAVE, **{**_BASE_PARAMS, "agn_polar_beta": 2.0})
        max_rel_diff = float(jnp.max(jnp.abs(sed_hi - sed_lo)) / jnp.max(jnp.abs(sed_lo)))
        assert max_rel_diff > 1e-3, (
            f"agn_polar_beta=1.0 vs 2.0 changed the SED by only {max_rel_diff:.3e} relative."
        )


class TestPolarDustGradients:
    """agn_polar_T, agn_polar_oa, agn_polar_beta must all have nonzero grad."""

    @staticmethod
    def _objective(key, value):
        def obj(v):
            p = {**_BASE_PARAMS, key: v}
            return jnp.sum(composable_agn_l_nu(_WAVE, **p))

        return float(jax.grad(obj)(value))

    def test_temperature_gradient_nonzero(self):
        g = self._objective("agn_polar_T", 100.0)
        assert g != 0.0, "d(sum(sed))/d(agn_polar_T) is exactly zero."

    def test_opening_angle_gradient_nonzero(self):
        # 45 deg (not the energy-balance test's saturated 10 deg): the Type-1/2
        # sigmoid mask is unsaturated there, so its gradient w.r.t. oa is
        # resolvably nonzero.
        g = self._objective("agn_polar_oa", 45.0)
        assert g != 0.0, "d(sum(sed))/d(agn_polar_oa) is exactly zero."

    def test_beta_gradient_nonzero(self):
        g = self._objective("agn_polar_beta", 1.6)
        assert g != 0.0, "d(sum(sed))/d(agn_polar_beta) is exactly zero."
