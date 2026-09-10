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
import numpy as np
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

    **Which policy this measures (R60).** The factor asserted here is
    ``f_cone``, the cone's share of the disc's HEMISPHERE-INTEGRATED
    bolometric luminosity -- the reference that
    ``agn_norm='independent'`` and ``'conserving'`` put on the disc
    (``10**agn_log_lbol``). It is called through
    :func:`polar_dust_reemission_lnu`'s default
    ``agn_polar_reference='bolometric'``, so that is the policy these cases
    run under, and the identity is correct there. Under
    ``agn_norm='cigale_joint'`` with the SKIRTOR torus the disc is instead
    tied to CIGALE's inclination-specific ``disk`` template, whose factor is
    ``g = (7/18) f_cone`` -- exactly 18/7 smaller. That twin is
    :class:`TestPolarDustCoveringFactorFaceOnReference` below; neither pin
    replaces the other, they pin two different reference frames.
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


class TestPolarDustCoveringFactorFaceOnReference:
    """The ``'face_on'`` twin of the pin above (R60).

    The polar-cone geometry is one thing measured against two disc reference
    luminosities, and the two differ by **exactly 18/7**:

    * ``f_cone(oa) = 1 - (3/7)sin^2 oa - (4/7)sin^3 oa`` against the
      hemisphere-integrated bolometric ``L_bol = (7 pi/3) I_0``;
    * ``g(oa) = 7/18 - sin^2(oa)/6 - (2/9) sin^3(oa)`` against
      ``int L(theta=0) dlambda``, the face-on value a SKIRTOR flux table
      carries (already multiplied by ``4 pi d^2``, hence the compensating
      ``1/4 pi`` folded into ``g``) -- CIGALE's ``skirtor2016`` convention.

    Reproduced from pcigale at ``oa=40``: ``g = 0.261007`` against
    ``f_cone = 0.671162``, ratio 2.571429 = 18/7.

    Picking a reference silently would move the polar re-emission by that
    2.571x, so :func:`polar_cone_covering_factor` refuses an unknown one
    rather than defaulting.
    """

    _L_LAMBDA_DISC = 1e10 * (_WAVE / 5000.0) ** (-1.5)
    _EBV = 0.3

    @pytest.mark.parametrize("oa", [10.0, 40.0, 45.0, 80.0])
    def test_face_on_factor_is_exactly_seven_eighteenths_of_bolometric(self, oa):
        from tengri.components.agn.polar_dust import polar_cone_covering_factor

        bol = float(polar_cone_covering_factor(oa, reference="bolometric"))
        face = float(polar_cone_covering_factor(oa, reference="face_on"))
        assert face == pytest.approx((7.0 / 18.0) * bol, rel=1e-12, abs=0.0)

    def test_matches_pcigale_g_at_the_skirtor_fiducial(self):
        """The face-on branch reproduces CIGALE's own ``l_ext`` coefficient."""
        from tengri.components.agn.polar_dust import polar_cone_covering_factor

        assert float(polar_cone_covering_factor(40.0, reference="face_on")) == pytest.approx(
            0.261007, rel=1e-5, abs=0.0
        )
        assert float(polar_cone_covering_factor(40.0, reference="bolometric")) == pytest.approx(
            0.671162, rel=1e-5, abs=0.0
        )

    @pytest.mark.parametrize("oa", [10.0, 45.0, 80.0])
    def test_reemitted_tracks_the_face_on_factor(self, oa):
        """The same absorbed-equals-reemitted identity, in the face-on frame."""
        from tengri.components.agn.polar_dust import (
            polar_cone_covering_factor,
            polar_dust_extinction,
        )

        reemitted = polar_dust_reemission_lnu(
            _WAVE,
            self._L_LAMBDA_DISC,
            agn_polar_ebv=self._EBV,
            agn_cos_inc=1.0,
            agn_polar_oa=oa,
            agn_polar_reference="face_on",
        )
        _att, absorbed_per_bin = polar_dust_extinction(
            self._L_LAMBDA_DISC, _WAVE, cos_inc=1.0, opening_angle_deg=oa, ebv=self._EBV
        )
        idx = jnp.argsort(_WAVE)
        absorbed = jnp.trapezoid(absorbed_per_bin[idx], _WAVE[idx])
        cone_absorbed = polar_cone_covering_factor(oa, reference="face_on") * absorbed
        assert float(cone_absorbed) != 0.0
        rel = abs(_integrate_lnu_bolometric(reemitted) - float(cone_absorbed)) / float(
            cone_absorbed
        )
        assert rel < 1e-6, f"oa={oa}: face-on reemission is not the cone-absorbed power"

    def test_unknown_reference_raises(self):
        """No defaulting: an unknown frame is a 2.571x error."""
        from tengri.components.agn.polar_dust import polar_cone_covering_factor

        with pytest.raises(ValueError, match=r"reference"):
            polar_cone_covering_factor(40.0, reference="whatever")


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


#: Wide grid for the cigale_joint frame tests: the polar-cone budget is read off
#: the FIR graybody and the AGN dust budget off the SKIRTOR torus, so the grid
#: has to span the disc UV (500 A) and the whole torus/graybody IR (1e8 A).
_WAVE_JOINT = jnp.asarray(np.geomspace(500.0, 1.0e8, 3000))

#: The SKIRTOR fiducial the CIGALE reproduction uses (t=7, pl=1, q=1, oa=40,
#: i=30, disk_type=1 -> schartmann2005, E(B-V)=0.03, T=100 K, beta=1.6).
_JOINT_BASE = dict(
    agn_disc_block="schartmann2005",
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="skirtor",
    agn_attenuation_block="polar_dust",
    agn_norm="cigale_joint",
    agn_tau_skirtor=7.0,
    agn_p_skirtor=1.0,
    agn_q_skirtor=1.0,
    agn_oa_skirtor=40.0,
    agn_polar_ebv=0.03,
    agn_polar_oa=40.0,
    agn_polar_T=100.0,
    agn_polar_beta=1.6,
)


def _joint_integrals(*, ir_frac, torus_frac, wave=_WAVE_JOINT, **over):
    """Bolometric ``polar``/``torus``/``disc`` of one cigale_joint build [erg/s]."""
    from tengri.components.agn.blocks.runner import compose_l_nu

    nu = C_AA / wave
    order = jnp.argsort(nu)
    _sed, comps = compose_l_nu(
        wave,
        12.0,
        agn_ir_frac=ir_frac,
        agn_torus_frac=torus_frac,
        return_components=True,
        **{**_JOINT_BASE, **over},
    )
    return {
        key: float(jnp.abs(jnp.trapezoid(jnp.asarray(comps[key])[order], nu[order])))
        for key in ("polar", "torus", "disc")
    }


def _frame_invariant(integrals):
    """``q = (P/B) / (c D/B)``: cone-absorbed power per unit disc reference.

    Constant of the geometry when the cone factor is referenced to the disc
    the SED carries (see :class:`TestPolarReemissionFollowsTheDiscsActualFrame`).
    """
    budget = integrals["polar"] + integrals["torus"]
    return (integrals["polar"] / integrals["torus"]) * budget / integrals["disc"]


class TestPolarReemissionFollowsTheDiscsActualFrame:
    """R60, finished: ONE traced predicate picks the disc's frame AND the
    covering-factor reference.

    Stage 4 of the composable runner selects the disc's normalization frame
    with a **traced** ``jnp.where(agn_ir_frac > 0, ...)``: at ``agn_ir_frac >
    0`` the disc is tied to CIGALE's face-on ``disk`` template
    (``agn_power x R``), and at ``agn_ir_frac = 0`` -- the registry default,
    and what the shipped gallery example and any composable
    skirtor+polar_dust build without an explicit ``ir_frac`` carry -- it is
    the bolometric-frame disc debited by ``(1 - agn_torus_frac)``.

    The polar-cone reference used to be chosen by a *static* Python branch on
    ``(agn_norm, agn_torus_block)`` alone, so in the ``agn_ir_frac = 0``
    regime a face-on ``g`` factor was applied to a rebuilt face-on array
    while the disc in the SED was the bolometric-frame one. Measured at the
    fiducial: ``sed_agn_polar`` was **bit-identical** across the two regimes
    (3.731930e+44 erg/s at both) although the disc it reprocesses differs by
    2.8357x, and the polar-to-torus ratio was **exactly** 0.243324 at
    ``agn_torus_frac`` = 0.2 and 0.6, where the disc-to-torus ratio moves 6x
    (3.942284 -> 0.657047). Frame-consistent, the ``agn_ir_frac = 0`` polar
    is 1.58x smaller than what shipped.

    The observable both tests use is ``r = int(polar) / int(torus)``. Under
    the joint budget (R59) ``polar = B s`` and ``torus = B (1 - s)`` with
    ``s = P/(B + P)``, so ``r = P/B`` exactly: the raw cone-absorbed power
    over the AGN dust budget, with the budget's own normalization divided
    out. The disc's own frame is read off the same build as
    ``d = int(disc)/(int(polar) + int(torus)) = c D/B``, where ``D`` is the
    disc reference luminosity and ``c`` the mask/attenuation weighting the
    two builds share, so

    ``q = r/d = f_cone(oa) x absorbed_fraction / c``

    is a constant of the geometry -- invariant under anything that moves the
    disc and the budget together, ``agn_torus_frac`` included -- whenever the
    cone factor is referenced to the disc the SED carries.
    """

    @staticmethod
    def _skip_without_grid():
        from tengri.components.agn.skirtor import _load_raw_disk_dust_grid

        if _load_raw_disk_dust_grid() is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")

    def test_bolometric_regime_polar_tracks_the_disc(self):
        """agn_ir_frac=0: the polar re-emission moves with the disc it reprocesses.

        ``agn_torus_frac`` moves the debited disc as ``(1 - f)`` and the torus
        as ``f``, so a frame-consistent polar moves ``r`` exactly as ``d``
        moves. A face-on reference applied here instead reads the rebuilt
        ``agn_power x R_faceon`` array, whose ratio to the budget is
        independent of ``agn_torus_frac``: ``r`` then does not move at all.
        """
        self._skip_without_grid()
        lo = _joint_integrals(ir_frac=0.0, torus_frac=0.2)
        hi = _joint_integrals(ir_frac=0.0, torus_frac=0.6)
        q_lo = _frame_invariant(lo)
        q_hi = _frame_invariant(hi)
        d_ratio = (hi["disc"] / (hi["polar"] + hi["torus"])) / (
            lo["disc"] / (lo["polar"] + lo["torus"])
        )
        assert not (0.9 < d_ratio < 1.1), (
            f"probe setup failed: agn_torus_frac barely moved the disc's share of the "
            f"AGN dust budget ({d_ratio:.6f}), so this configuration cannot tell the "
            "two frames apart"
        )
        assert q_hi == pytest.approx(q_lo, rel=1e-6, abs=0.0), (
            f"at agn_ir_frac=0 the cone-absorbed power per unit disc moved from "
            f"{q_lo:.6f} to {q_hi:.6f} ({q_hi / q_lo:.4f}x) when agn_torus_frac went "
            f"0.2 -> 0.6, which moved the disc's share of the budget by {d_ratio:.6f}: "
            "the covering factor is referenced to a disc frame the SED does not "
            "carry (R60)."
        )

    def test_face_on_regime_reemits_g_times_the_face_on_disc(self):
        """agn_ir_frac>0: the reference is CIGALE's face-on disk and ``g``.

        Independent reconstruction from the public physics functions:
        ``r = P/B = g(oa) x int(faceon_disc (1 - ext)) / agn_power`` with the
        face-on disc built the way Stage 4's R-tie builds the disc,
        ``agn_power x R_faceon`` on the unit-normalized intrinsic disc shape.
        A bolometric reference here would read ``f_cone x R`` instead --
        0.671162 x 2.2 against 0.261007 x 4.42, a 28% error.

        The shape is normalized, and the absorbed integral taken, on the
        SKIRTOR templates' NATIVE grid -- the grid ``R_faceon`` itself was
        derived on, and the grid CIGALE integrates its own polar proxy over
        (``x=AGN1.wl``). On the native grid the reconstruction reproduces the
        shipped ratio to every digit printed (0.242631734 at
        ``agn_torus_frac=0.5`` on ``_WAVE_JOINT``); reconstructing the same
        expression on ``_WAVE_JOINT`` instead reads 0.243324134, +0.29%, which
        this test's 1e-4 tolerance rejects -- and it would make the
        expectation move with whatever grid the test happens to use
        (:class:`TestFaceOnReferenceUsesTheNativeSkirtorGrid`).
        """
        self._skip_without_grid()
        from tengri.components.agn.blocks import resolve_agn_block
        from tengri.components.agn.polar_dust import polar_cone_covering_factor
        from tengri.components.agn.skirtor import skirtor_disc_dust_ratio
        from tengri.utils.grid_interp import resample_template

        wave = _WAVE_JOINT
        cos_inc = 0.86602540378443864
        disc = jnp.asarray(
            resolve_agn_block("disc", _JOINT_BASE["agn_disc_block"])(
                wave, agn_log_lbol=12.0, templates=None
            )
        )
        tie = skirtor_disc_dust_ratio(
            wave,
            disc,
            jnp.ones_like(wave),
            agn_tau_skirtor=_JOINT_BASE["agn_tau_skirtor"],
            agn_p_skirtor=_JOINT_BASE["agn_p_skirtor"],
            agn_q_skirtor=_JOINT_BASE["agn_q_skirtor"],
            agn_oa_skirtor=_JOINT_BASE["agn_oa_skirtor"],
            agn_cos_inc=cos_inc,
        )
        r_faceon = tie.R_faceon
        # The grid the face-on reference is normalized and absorbed on: the
        # native template grid. Rebuilt here from the raw disc rather than
        # read off ``tie.faceon_shape_native``, so the normalization step
        # itself is reconstructed and not just echoed.
        ref_wave = tie.wave_native
        ref_disc = resample_template(ref_wave, wave, disc, left=0.0, right=0.0)
        idx = jnp.argsort(ref_wave)
        shape_unit = ref_disc / jnp.trapezoid(ref_disc[idx], ref_wave[idx])
        _att, absorbed_per_bin = polar_dust_extinction(
            shape_unit * float(r_faceon),
            ref_wave,
            cos_inc=cos_inc,
            opening_angle_deg=_JOINT_BASE["agn_polar_oa"],
            ebv=_JOINT_BASE["agn_polar_ebv"],
            law="smc",
        )
        absorbed_over_power = float(jnp.trapezoid(absorbed_per_bin[idx], ref_wave[idx]))
        expected = (
            float(polar_cone_covering_factor(_JOINT_BASE["agn_polar_oa"], reference="face_on"))
            * absorbed_over_power
        )

        got = _joint_integrals(ir_frac=0.3, torus_frac=0.5)
        r = got["polar"] / got["torus"]
        assert r == pytest.approx(expected, rel=1e-4, abs=0.0), (
            f"agn_ir_frac=0.3: polar/torus = {r:.6f}, but g x absorbed(face-on disc) / "
            f"agn_power = {expected:.6f}; the face-on frame is not the one applied."
        )

    def test_face_on_regime_is_tied_to_agn_power(self):
        """The R-tie's own signature: at agn_ir_frac>0 the disc scales WITH the
        budget, so ``r`` does not move with agn_torus_frac. The negative
        control for the test above: it is the ``agn_ir_frac=0`` regime, and
        only that one, whose polar must follow ``agn_torus_frac``."""
        self._skip_without_grid()
        lo = _joint_integrals(ir_frac=0.3, torus_frac=0.2)
        hi = _joint_integrals(ir_frac=0.3, torus_frac=0.6)
        assert _frame_invariant(hi) == pytest.approx(_frame_invariant(lo), rel=1e-6, abs=0.0)
        assert (hi["polar"] / hi["torus"]) == pytest.approx(
            lo["polar"] / lo["torus"], rel=1e-9, abs=0.0
        )
        assert (hi["disc"] / (hi["polar"] + hi["torus"])) == pytest.approx(
            lo["disc"] / (lo["polar"] + lo["torus"]), rel=1e-9, abs=0.0
        )


#: Three user grids of different extent AND resolution, each carrying the
#: whole of the analytic disc's 8 A - 1e6 A support and the whole of the
#: SKIRTOR templates' 10 A - 1e8 A support. Nothing physical distinguishes
#: them, so nothing in the SED's component split may.
_GRID_SKIRTOR_SPAN = jnp.asarray(np.geomspace(8.0, 1.0e8, 3000))
_GRID_PANCHROMATIC = jnp.asarray(np.geomspace(0.0413, 3.0e11, 4000))
_GRID_SKIRTOR_SPAN_FINE = jnp.asarray(np.geomspace(1.0, 1.0e9, 6000))


class TestFaceOnReferenceUsesTheNativeSkirtorGrid:
    """R60/Item B: the face-on reference is normalized on the grid that
    produced ``R_faceon``, not on the caller's wavelength grid.

    ``skirtor_disc_dust_ratio`` derives ``R_faceon = int_disk0/int_dust`` on
    the SKIRTOR templates' NATIVE grid, with its own comment that resampling
    those templates onto a user grid distorts the integrals by ~10%. CIGALE
    does the same: ``skirtor2016.py`` normalizes the analytic disc shape with
    ``disk(SKIRTOR2016.wl) * trapezoid(AGN1.disk, x=AGN1.wl)`` and integrates
    ``AGN1.disk * (1 - ext_fac)`` over ``AGN1.wl`` -- the template grid, never
    the model's output grid. Unit-normalizing that shape on the caller's grid
    instead pairs a native-grid ratio with a caller-grid integral, and the
    polar share then moves with the caller's wavelength extent, which is not
    a physical parameter.

    The observable is ``r = int(polar)/int(torus)``. Under the joint budget
    (R59) ``polar = B s`` and ``torus = B (1 - s)`` with ``s = P/(B + P)``, so
    ``r = P/B``: the raw cone-absorbed power over the AGN dust budget. ``P``
    is built from ``agn_power x R_faceon`` and ``agn_power`` is the torus
    integral on the user grid, so ``r`` divides the user grid's effect on the
    torus normalization back out and isolates the grid the reference shape is
    normalized on.

    Measured at ``agn_ir_frac=0.3`` and the SKIRTOR fiducial, on the caller
    grid: ``r`` = 0.255908936 (8 A - 1e8 A, n=3000) vs 0.253838113
    (0.0413 A - 3e11 A, n=4000) for the ``skirtor`` disc block and
    0.249880352 vs 0.249874295 for ``schartmann2005``. On the native grid all
    three grids below agree bit-for-bit (0.2565718334 / 0.2500544570).

    One caveat this does NOT fix, and cannot: a caller grid that truncates the
    disc block's own support carries no disc light there, so the resampled
    shape really is different and ``r`` really does move (0.2827450165 on
    500 A - 1e8 A for the ``skirtor`` block, +10.2%). The disc's own share of
    the budget moves with it (2.20985 -> 2.14773), so the frames stay
    consistent -- but a model whose grid starts redward of its disc's peak is
    reprocessing a disc it never emitted. Span the disc's support.
    """

    @staticmethod
    def _skip_without_grid():
        from tengri.components.agn.skirtor import _load_raw_disk_dust_grid

        if _load_raw_disk_dust_grid() is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")

    @pytest.mark.parametrize("disc_block", ("skirtor", "schartmann2005"))
    def test_polar_share_invariant_to_user_grid(self, disc_block):
        self._skip_without_grid()
        rs = {}
        for tag, grid in (
            ("8 A - 1e8 A n=3000", _GRID_SKIRTOR_SPAN),
            ("0.0413 A - 3e11 A n=4000", _GRID_PANCHROMATIC),
            ("1 A - 1e9 A n=6000", _GRID_SKIRTOR_SPAN_FINE),
        ):
            got = _joint_integrals(
                ir_frac=0.3, torus_frac=0.5, wave=grid, agn_disc_block=disc_block
            )
            rs[tag] = got["polar"] / got["torus"]
        ref_tag = "8 A - 1e8 A n=3000"
        ref = rs[ref_tag]
        for tag, r in rs.items():
            assert r == pytest.approx(ref, rel=1e-6, abs=0.0), (
                f"{disc_block!r}: polar/torus = {ref:.10f} on {ref_tag} and {r:.10f} on "
                f"{tag} ({r / ref:.6f}x). Every grid here spans the whole disc AND the "
                "whole SKIRTOR template support, so no physical quantity differs "
                "between them; the face-on reference shape is being unit-normalized on "
                "the caller's grid instead of the native grid that produced R_faceon "
                "(R60). Measured spread before the fix: 1.008158x for 'skirtor'."
            )
