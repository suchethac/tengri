# SPDX-License-Identifier: BSD-3-Clause
"""Conservation: L_absorbed excludes the Lyman continuum — #922.

LyC photons ionize hydrogen rather than heat dust, so the canonical
energy-balance integral (:func:`tengri.forward.energy_balance.
bolometric_absorbed`) masks the ionizing side of
:data:`tengri.components.lyc.LYMAN_LIMIT_AA` (911.76 Å; the one Lyman edge,
see that module's docstring -- edge moved from a bare 912 Å literal). This
matches CIGALE (attenuation zeroed at λ ≤ 91.2 nm) and Bagpipes (``fesc``
masking of the ionizing continuum); FSPS by contrast includes LyC absorption
in its dust heating.

The synthetic wide SSP is UV-bright (a ``(5000 Å/λ)²`` continuum down to
100 Å), so the unmasked integral exceeds the masked one by a large factor —
exactly the case where an accidentally unmasked ``L_absorbed`` variant
(the pre-#922 ``nonstell``/single-screen behavior) shows up. These tests
pin the masked convention on every dust attenuation path so a refactor
cannot silently reintroduce the unmasked integral.

CI-runnable on the synthetic wide SSP (WG00 is data-gated and skips
without ``data/wg00_attenuation_grid.h5``).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp, builders
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.forward.energy_balance import bolometric_absorbed
from tengri.observation.photometry import FilterCurve
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.conservation

_WG00_GRID = Path(__file__).resolve().parents[3] / "data" / "wg00_attenuation_grid.h5"

TWO_COMPONENT = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_bc": 0.5,
    "tau_diff": 0.3,
}
SINGLE_SCREEN = {
    "law": "power_law",
    "type": "single_component",
    "all_params": Fixed(DEFAULT),
    "tau_v": 0.5,
}
WG00 = {"type": "wg00", "all_params": Fixed(DEFAULT), "tau_v": 0.5}

_DUST_CASES = [
    pytest.param(TWO_COMPONENT, id="two_component"),
    pytest.param(SINGLE_SCREEN, id="single_screen"),
    pytest.param(
        WG00,
        id="wg00",
        marks=pytest.mark.skipif(not _WG00_GRID.is_file(), reason="WG00 grid data not present"),
    ),
]


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _obs() -> Observation:
    # Optical bands probe the absorbed light; the far-IR band (100 um) is
    # where the re-emitted dust luminosity lands.
    centers = (3500.0, 4800.0, 6200.0, 9000.0, 1.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(ssp, dust, approx=None):
    return SEDModel.build(
        ssp_data=ssp,
        observation=_obs(),
        approx=approx,
        # Pinned to the pre-#1007 prior-midpoint fallbacks the golden values
        # were captured under — the registry's curated defaults would
        # otherwise shift the SFH (and every golden) silently.
        sfh=builders.sfh.tsnorm(
            all_params=Fixed(DEFAULT),
            log_total_mass=9.75,
            peak_lbt_gyr=6.25,
            width_gyr=2.6,
            skew=0.0,
            trunc=5.5,
        ),
        dust_attenuation=dust,
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


def _params(model):
    # Free-only (#2296): every predict_* surface below self-merges the
    # spec's Fixed values internally and refuses a params key the spec
    # declared Fixed. Spreading get_fixed_values() here used to be
    # harmless (the old blanket-merge machinery matched it); now it hands
    # every one of those names back as a refused override.
    return dict(model.spec.sample(jax.random.PRNGKey(0)))


def _zero_lyc(ssp):
    """Copy of the SSP with all ionizing-side flux zeroed.

    Uses :func:`tengri.components.lyc.ionizing_mask` (edge at
    ``LYMAN_LIMIT_AA`` = 911.76 Å), not a bare ``912.0`` literal: on
    ``synthetic_ssp_wide``'s 1600-node logspace grid the node at 911.9583 Å
    sits strictly between 911.76 and 912.0, so a ``912.0``-based zeroing
    would also zero the node ``edge_trapezoid`` uses as the "nonionizing"
    anchor of the bracket cell straddling the true edge -- breaking the
    invariance this fixture exists to test (measured ~1.3% spurious
    difference) for a reason that has nothing to do with the physics.
    """
    from tengri.components.lyc import ionizing_mask

    return SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * (~ionizing_mask(ssp.ssp_wave)),
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )


class TestLycMaskedLAbsorbed:
    @pytest.mark.parametrize("dust", _DUST_CASES)
    def test_derived_l_absorbed_is_lyc_masked_integral(self, synthetic_ssp_wide, dust):
        """``state.derived['L_absorbed']`` equals the independent λ ≥ 912 Å integral.

        The independent reference integrates intrinsic-minus-attenuated rest
        SEDs (dust off vs. on) through the canonical helper; the unmasked
        variant of the same integral must NOT match — the premise guard that
        makes this regression test non-vacuous on the UV-bright fixture.
        """
        model = _build(synthetic_ssp_wide, dict(dust))
        p = _params(model)
        state = model.predict_state(p)
        l_absorbed = float(state.derived["L_absorbed"])

        # Intrinsic reference from a transparent build: two-component dust with
        # both taus pinned to 0 (exp(0)=1 for any law). ``dust=None`` would NOT
        # work — build() auto-fills a default dust group with *free* taus that
        # ``_params`` then samples; and WG00 has no tau=0 grid node, so zeroing
        # the dusty model's own taus is wrong for the wg00 case.
        transparent = {
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.0,
            "tau_diff": 0.0,
        }
        model_nodust = _build(synthetic_ssp_wide, transparent)
        attenuated = model.predict_rest_sed(p)
        intrinsic = model_nodust.predict_rest_sed(_params(model_nodust))
        wave = attenuated.wavelength
        nu = C_AA / wave

        masked = float(jnp.abs(bolometric_absorbed(intrinsic.sed, attenuated.sed, nu, wave=wave)))
        unmasked = float(
            jnp.abs(
                bolometric_absorbed(
                    intrinsic.sed, attenuated.sed, nu, wave=wave, lyman_cutoff_aa=None
                )
            )
        )

        assert masked > 0.0
        # Premise guard: the UV-bright fixture must make the mask matter. The
        # difference can go either way — Calzetti's extrapolated k(λ) turns
        # negative in the far-UV, so the unmasked integral picks up *negative*
        # absorbed energy below 912 Å (amplification), another failure mode the
        # mask protects against.
        assert abs(unmasked - masked) > 0.05 * masked, (
            "premise guard: the UV-bright fixture must make the LyC mask matter "
            f"(masked={masked:.6e}, unmasked={unmasked:.6e})"
        )
        np.testing.assert_allclose(
            l_absorbed,
            masked,
            rtol=1e-9,
            err_msg="published L_absorbed is not the LyC-masked integral",
        )

    @pytest.mark.parametrize("dust", _DUST_CASES)
    def test_l_absorbed_invariant_to_ssp_lyc_flux(self, synthetic_ssp_wide, dust):
        """Zeroing all SSP flux below 912 Å must not change L_absorbed.

        With the mask active, LyC photons carry no weight in the energy
        balance, so an SSP with its ionizing continuum removed publishes the
        identical absorbed luminosity. An unmasked integral fails this
        immediately on the UV-bright fixture.
        """
        model_full = _build(synthetic_ssp_wide, dict(dust))
        model_nolyc = _build(_zero_lyc(synthetic_ssp_wide), dict(dust))
        l_full = float(model_full.predict_state(_params(model_full)).derived["L_absorbed"])
        l_nolyc = float(model_nolyc.predict_state(_params(model_nolyc)).derived["L_absorbed"])
        np.testing.assert_allclose(l_full, l_nolyc, rtol=1e-12)


class TestGoldenValues:
    """Golden L_absorbed / L_ir on the synthetic wide SSP (#922).

    Pinned so the LyC-masked convention cannot silently drift. Regenerate
    only for a deliberate physics change, and record why in the commit.
    """

    # Captured 2026-07-08 on the synthetic wide SSP (float64, CPU), after
    # the #964 CIC age-weight kernel replaced the DSPS histogram handoff —
    # a deliberate physics change (~+2.4 % L_absorbed here: the old kernel
    # zeroed the SSP node bracketing the SFH's maximum age and pushed its
    # mass onto younger nodes). Previous capture (2026-07-05, pre-#964):
    # two_component 5.697121948709991e59, single_screen 7.149162290344824e59,
    # wg00 6.745997586687046e59.
    # Re-pinned after #1731: dust laws renormalized to k(5500)=1; rescale ~+0.031%.
    # Re-pinned after #2521: the CIC age weights already bounded the tsnorm
    # shape's SF-onset support to [0, age(z)] and renormalized within it, but
    # the *total* formed mass fed to the SED scale was still the pre-clamp
    # trapezoid integral over the full (unbounded) shape -- silently below
    # 10**log_total_mass whenever any of the tsnorm Gaussian's tail (here
    # peak_lbt_gyr=6.25, width_gyr=2.6 at z=0.05) fell past age(z). Formed
    # mass is now pinned to 10**log_total_mass by construction; ~+0.187%
    # L_absorbed here (the physical fix, not a rescale of an unrelated law).
    # Re-pinned (L2, one-Lyman-edge): the physical edge moved from a bare
    # 912.0 A literal to LYMAN_LIMIT_AA = 911.76 A, and the grid cell
    # straddling it is now integrated with the step model
    # (tengri.components.lyc.edge_trapezoid: two rectangles, not a linear
    # ramp) instead of a plain trapezoid over a node-level boolean mask.
    # Moving the edge down widens the non-ionizing (dust-heating) side by
    # one SSP node on this 1600-node logspace grid, and the step-model
    # rectangle for that bracket cell no longer underweights it -- both
    # push L_absorbed up by ~1.3-1.4%, in the same direction as the #964
    # and #1731 re-pins above. This re-pin combines the #2521 formed-mass
    # fix and the L2 edge move, which landed independently on two merge
    # parents: two_component 5.849679302923598e59 -> 5.932047709369588e59
    # (+1.408%), single_screen 7.338318188268545e59 -> 7.4319301202253715e59
    # (+1.276%), wg00 6.924486363284743e59 -> 7.022833393235165e59 (+1.420%).
    # Re-pinned (exact young/old split): two_component's default birth-cloud
    # dispersal moved from a logistic of width 0.3 dex evaluated AT each SSP
    # node age to the hard step at 10 Myr applied to the exact share of every
    # node's formed mass younger than 10 Myr (the node straddling 10 Myr holds
    # 26 % young mass, not 0 or 1). L_absorbed 5.932047709369588e59 ->
    # 5.93036676326139e59 (-0.0283 %). Proof the new value is the step answer
    # (``test_two_component_golden_is_the_dense_step_reference``): the same
    # fixture integrated against an independent dense-parcel SFH reference
    # (3e6 uniform-in-lookback parcels, same cloud-in-cell split, S = 1[t < 10 Myr])
    # gives 5.930366796495171e59, new/ref - 1 = 5.6e-9; evaluating the hard step
    # at the SSP node ages instead (the grid-dependent answer) gives
    # 5.930341625031011e59, 4.2e-6 low. single_screen and wg00 have no age split
    # and did not move.
    GOLDEN_L_ABSORBED: ClassVar[dict[str, float]] = {
        "two_component": 5.93036676326139e59,
        "single_screen": 7.4319301202253715e59,
        "wg00": 7.022833393235165e59,
    }

    @pytest.mark.parametrize(
        "key,dust",
        [
            pytest.param("two_component", TWO_COMPONENT, id="two_component"),
            pytest.param("single_screen", SINGLE_SCREEN, id="single_screen"),
            pytest.param(
                "wg00",
                WG00,
                id="wg00",
                marks=pytest.mark.skipif(
                    not _WG00_GRID.is_file(), reason="WG00 grid data not present"
                ),
            ),
        ],
    )
    def test_golden_l_absorbed_and_l_ir(self, synthetic_ssp_wide, key, dust):
        model = _build(synthetic_ssp_wide, dict(dust))
        state = model.predict_state(_params(model))
        l_absorbed = float(state.derived["L_absorbed"])
        l_ir = float(state.derived["L_ir"])
        np.testing.assert_allclose(l_absorbed, self.GOLDEN_L_ABSORBED[key], rtol=1e-7)
        # Default eta_balance = 1 → strict conservation.
        np.testing.assert_allclose(l_ir, l_absorbed, rtol=1e-12)

    def test_two_component_golden_is_the_dense_step_reference(self, synthetic_ssp_wide):
        """The re-pinned golden is the hard-step answer, against a dense reference.

        Independent of the age kernel: the SFH is read back as published
        (``sfr_history`` on the lookback grid), densified to 3e6 uniform
        lookback parcels, split between the bracketing SSP nodes with the
        same log-age weights, and each parcel is young iff t < 10 Myr. The
        absorbed energy then follows from the closed-form population mixture.
        """
        from tengri.components.dust.laws._registry import resolve_dust_law
        from tengri.components.lyc import LYMAN_LIMIT_AA
        from tengri.forward.energy_balance import bolometric_absorbed_log10

        model = _build(synthetic_ssp_wide, dict(TWO_COMPONENT))
        state = model.predict_state(_params(model))
        ages = np.asarray(state.derived["ssp_ages_yr"])
        wave = np.asarray(state.wave)
        lnu_age = np.asarray(state.derived["lnu_age"])
        lbt = np.asarray(state.derived["sfh_grid_lbt_yr"])
        sfr = np.asarray(state.derived["sfr_history"])

        t = np.linspace(0.0, lbt.max(), 3_000_001)[1:]
        sfr_dense = np.interp(np.log10(t), np.log10(lbt), sfr)
        lg_nodes, lg_t = np.log10(ages), np.log10(t)
        lo = np.clip(np.searchsorted(lg_nodes, lg_t) - 1, 0, ages.size - 2)
        up_share = np.clip((lg_t - lg_nodes[lo]) / (lg_nodes[lo + 1] - lg_nodes[lo]), 0.0, 1.0)
        young_parcel = (t < 1.0e7).astype(float)
        total = np.zeros(ages.size)
        young = np.zeros(ages.size)
        for weight, node in ((1.0 - up_share, lo), (up_share, lo + 1)):
            np.add.at(total, node, sfr_dense * weight)
            np.add.at(young, node, sfr_dense * young_parcel * weight)
        y_ref = np.where(total > 0.0, young / np.maximum(total, 1e-300), 0.0)

        k = np.asarray(resolve_dust_law("calzetti")(jnp.asarray(wave)))
        t_young = np.exp(-(0.5 + 0.3) * k)
        t_old = np.exp(-0.3 * k)
        transmission = y_ref[:, None] * t_young[None, :] + (1.0 - y_ref[:, None]) * t_old[None, :]
        log_ref, _ = bolometric_absorbed_log10(
            jnp.asarray(lnu_age.sum(0)),
            jnp.asarray((lnu_age * transmission).sum(0)),
            jnp.asarray(C_AA / wave),
            wave=jnp.asarray(wave),
            lyman_cutoff_aa=LYMAN_LIMIT_AA,
        )
        np.testing.assert_allclose(
            float(state.derived["L_absorbed"]), 10.0 ** float(log_ref), rtol=1e-7
        )

    def test_lut_tracks_exact_on_lyc_bright_fixture(self, synthetic_ssp_wide):
        """WavePrecomp energy-balance LUT tracks the exact path on this fixture.

        The far-IR band is L_ir-dominated, so parity there pins agreement of
        the LUT contraction with the exact LyC-masked integral for the
        UV-bright case specifically.
        """
        dust_attenuation = dict(TWO_COMPONENT)
        dust_emission = {"type": "modified_blackbody", "all_params": Fixed(DEFAULT)}
        m_exact = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=_obs(),
            sfh=builders.sfh.tsnorm(
                all_params=Fixed(DEFAULT),
                log_total_mass=9.75,
                peak_lbt_gyr=6.25,
                width_gyr=2.6,
                skew=0.0,
                trunc=5.5,
            ),
            dust_attenuation=dust_attenuation,
            dust_emission=dust_emission,
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )
        m_lut = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=_obs(),
            approx=WavePrecomp(),
            sfh=builders.sfh.tsnorm(
                all_params=Fixed(DEFAULT),
                log_total_mass=9.75,
                peak_lbt_gyr=6.25,
                width_gyr=2.6,
                skew=0.0,
                trunc=5.5,
            ),
            dust_attenuation=dust_attenuation,
            dust_emission=dust_emission,
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )
        phot_exact = np.asarray(m_exact.predict_photometry(_params(m_exact)))
        phot_lut = np.asarray(m_lut.predict_photometry(_params(m_lut)))
        far_ir_rel = abs(phot_lut[-1] - phot_exact[-1]) / abs(phot_exact[-1])
        assert far_ir_rel < 0.02, f"far-IR LUT-vs-exact drift {far_ir_rel:.3%} (> 2%)"


@pytest.mark.bounds
class TestFiniteGuard:
    """Non-finite integrals clamp to zero (guard carried over from the
    retired compositional kernel, BUG-NSS-02 era — see #922)."""

    def test_inf_sed_clamps_to_zero(self):
        wave = jnp.logspace(2.0, 5.0, 50)
        nu = C_AA / wave
        # Index 30 sits well above the Lyman edge, so the Inf survives the
        # LyC mask and must be caught by the finiteness guard instead.
        assert float(wave[30]) > 912.0
        sed_intr = jnp.ones_like(wave).at[30].set(jnp.inf)
        sed_att = jnp.zeros_like(wave)
        out = bolometric_absorbed(sed_intr, sed_att, nu, wave=wave)
        assert jnp.isfinite(out)
        assert float(out) == 0.0

    def test_finite_inputs_unaffected(self):
        """Finite inputs pass through the ordinary (edge-aware) integral unclamped.

        The oracle here is an INDEPENDENT step-model integral (numpy, not
        calling ``tengri.components.lyc``): an ordinary trapezoid over every
        panel entirely on the non-ionizing side of ``LYMAN_LIMIT_AA``, plus
        the ONE bracket panel's non-ionizing rectangle (the first
        non-ionizing node's value times the non-ionizing portion of that
        panel's width) -- never a linear ramp through the bracket cell.

        Re-derived (L2, one-Lyman-edge) from a bare ``jnp.trapezoid(where(wave
        >= 912.0, ...), nu)`` oracle: that formula reproduced the exact
        partial-bin ramp bug (#537/#2447) this task removes -- it zeroed the
        whole ionizing side at NODE level and then ran a plain trapezoid
        through the bracket cell, so it no longer describes what
        ``bolometric_absorbed`` (now edge-aware) computes. ``bolometric_absorbed``
        is also positively oriented now (module docstring's "Sign convention"
        note), so this oracle is not negated, unlike the retired formula.
        """
        wave = jnp.logspace(2.0, 5.0, 50)
        nu = C_AA / wave
        sed_intr = jnp.ones_like(wave)
        sed_att = 0.5 * sed_intr
        out = bolometric_absorbed(sed_intr, sed_att, nu, wave=wave)

        integrand = np.asarray(sed_intr - sed_att)
        wave_np = np.asarray(wave)
        nu_np = np.asarray(nu)
        edge = 911.76
        nu_edge = float(C_AA) / edge
        total = 0.0
        for i in range(len(wave_np) - 1):
            w_lo, w_hi = wave_np[i], wave_np[i + 1]
            y_lo, y_hi = integrand[i], integrand[i + 1]
            nu_lo, nu_hi = nu_np[i], nu_np[i + 1]
            if w_hi < edge:
                continue  # fully ionizing panel: excluded from the non-ionizing side
            if w_lo >= edge:
                total += 0.5 * (y_lo + y_hi) * abs(nu_hi - nu_lo)  # ordinary trapezoid
                continue
            # Bracket panel: edge falls inside [w_lo, w_hi). Step model holds
            # the non-ionizing portion at y_hi across |nu_hi - nu_edge|.
            total += y_hi * abs(nu_hi - nu_edge)

        np.testing.assert_allclose(np.asarray(out), total, rtol=1e-12)
