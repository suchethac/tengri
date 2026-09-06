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

``TestPolarDustCoveringFactor`` additionally pins the CIGALE/Yang+2020
normalization this block implements, including the cone-covering factor
(task13 fix-round-1 item 1): the disc's anisotropic emission integrated over
the polar cone's solid angle, times the absorbed fraction, equals the
re-emitted graybody luminosity, to numerical-integration noise.

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

from tengri.components.agn.blocks.atten import polar_dust_reemission_lnu
from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.components.agn.polar_dust import polar_cone_covering_fraction, polar_dust_extinction
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


class TestPolarDustCoveringFactor:
    """Yang et al. 2020 X-CIGALE section 2.2.2 cone-covering factor (task13
    fix-round-1 item 1): the absorbed (and hence re-emitted) luminosity is
    the disc's anisotropic emission integrated over the polar cone's solid
    angle -- a function of ``agn_polar_oa`` alone -- times the absorbed
    fraction. Supersedes the earlier "deep-Type-1 coincidence" energy test
    (``TestPolarDustEnergyConservation``, removed): once the reemission is
    weighted by :func:`polar_cone_covering_fraction`, comparing it against
    the LOS-only "before minus after the Stage-5 screen" difference no
    longer coincides (that quantity never carried the cone weight to begin
    with) -- the identity below is the correct one, and holds for EVERY
    ``(oa, cos_inc)``, not just a deep-Type-1 special case, because the
    absorbed luminosity :func:`polar_dust_extinction` returns is
    geometry-independent (Yang+2020 §2.2.2): cos_inc has no effect on it at
    all, so this identity is exact at every inclination by construction.
    """

    #: Synthetic disc-like spectrum: an arbitrary smooth power law, exactly
    #: like ``test_polar_dust.py``'s ``L_NU_DISC`` fixture. The identity
    #: under test does not depend on which physical disc produced this
    #: shape.
    _L_LAMBDA_DISC = 1e10 * (_WAVE / 5000.0) ** (-1.5)
    _EBV = 0.3

    @pytest.mark.parametrize("cos_inc", [1.0, 0.5, 0.0])
    @pytest.mark.parametrize("oa", [10.0, 45.0, 80.0])
    def test_reemitted_equals_cone_absorbed(self, oa, cos_inc):
        reemitted = polar_dust_reemission_lnu(
            _WAVE,
            self._L_LAMBDA_DISC,
            agn_polar_ebv=self._EBV,
            agn_cos_inc=cos_inc,
            agn_polar_oa=oa,
            agn_polar_T=100.0,
            agn_polar_beta=1.6,
            agn_polar_law="smc",
        )
        reemitted_bol = _integrate_lnu_bolometric(reemitted)

        # Independent reconstruction from the two public physics functions:
        # the existing (unchanged) geometry-independent absorbed fraction,
        # times the NEW cone-covering fraction.
        _, l_absorbed_per_bin = polar_dust_extinction(
            self._L_LAMBDA_DISC,
            _WAVE,
            cos_inc=cos_inc,
            opening_angle_deg=oa,
            ebv=self._EBV,
            law="smc",
        )
        idx_w = jnp.argsort(_WAVE)
        l_absorbed_raw = jnp.trapezoid(l_absorbed_per_bin[idx_w], _WAVE[idx_w])
        cone_absorbed = polar_cone_covering_fraction(oa) * l_absorbed_raw

        assert float(cone_absorbed) != 0.0, (
            "cone-absorbed luminosity is exactly zero -- the energy-balance "
            "ratio below is undefined, not merely small."
        )
        rel_diff = abs(float(reemitted_bol) - float(cone_absorbed)) / float(cone_absorbed)
        assert rel_diff < 1e-6, (
            f"oa={oa}, cos_inc={cos_inc}: reemitted={float(reemitted_bol):.6e} erg/s, "
            f"cone_absorbed={float(cone_absorbed):.6e} erg/s, relative difference "
            f"{rel_diff:.3e} (expected < 1e-6)."
        )

    def test_covering_fraction_bounds(self):
        """f_cone(0) = 1 (whole hemisphere is polar cone), f_cone(90) = 0
        (torus edge reaches the pole, no escape cone left), monotonically
        decreasing in between -- the physical range the derivation predicts."""
        f0 = float(polar_cone_covering_fraction(0.0))
        f90 = float(polar_cone_covering_fraction(90.0))
        f45 = float(polar_cone_covering_fraction(45.0))
        assert f0 == pytest.approx(1.0, abs=1e-9)
        assert f90 == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < f45 < 1.0
        assert f0 > f45 > f90

    def test_sed_agn_polar_changes_with_oa(self):
        """The published sed_agn_polar component itself (not just the total
        SED, which already moved via the Stage-5 attenuation factor) must
        differ across agn_polar_oa -- the direct regression this item fixes.
        """
        p_narrow = {**_BASE_PARAMS, "agn_polar_oa": 10.0}
        p_wide = {**_BASE_PARAMS, "agn_polar_oa": 80.0}
        _, comp_narrow = composable_agn_l_nu(_WAVE, return_components=True, **p_narrow)
        _, comp_wide = composable_agn_l_nu(_WAVE, return_components=True, **p_wide)
        polar_narrow = comp_narrow["polar"]
        polar_wide = comp_wide["polar"]
        max_rel_diff = float(
            jnp.max(jnp.abs(polar_wide - polar_narrow)) / jnp.max(jnp.abs(polar_narrow))
        )
        assert max_rel_diff > 1e-3, (
            f"sed_agn_polar at agn_polar_oa=10deg vs 80deg differs by only "
            f"{max_rel_diff:.3e} relative -- the cone-covering factor has no effect."
        )


class TestPolarDustOpeningAngleAndBetaMoveSED:
    def test_opening_angle_changes_sed(self):
        # cos_inc=0.5 (not the energy-balance test's saturated 1.0): the
        # Type-1/2 sigmoid mask is unsaturated at this inclination, so
        # sweeping oa across its full declared range gives a clean,
        # comfortably-detectable signal through the Stage-5 attenuation
        # factor alone (independent of the cone-covering-factor fix, R22
        # fix-round-1 item 1, which additionally makes sed_agn_polar itself
        # oa-dependent -- see TestPolarDustCoveringFactor below).
        p = {**_BASE_PARAMS, "agn_cos_inc": 0.5}
        sed_narrow = composable_agn_l_nu(_WAVE, **{**p, "agn_polar_oa": 10.0})
        sed_wide = composable_agn_l_nu(_WAVE, **{**p, "agn_polar_oa": 80.0})
        max_rel_diff = float(
            jnp.max(jnp.abs(sed_wide - sed_narrow)) / jnp.max(jnp.abs(sed_narrow))
        )
        assert max_rel_diff > 1e-3, (
            f"agn_polar_oa=10deg vs 80deg changed the SED by only {max_rel_diff:.3e} relative."
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
