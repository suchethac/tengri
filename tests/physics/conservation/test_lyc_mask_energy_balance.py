# SPDX-License-Identifier: BSD-3-Clause
"""Conservation: L_absorbed excludes the Lyman continuum (λ < 912 Å) — #922.

LyC photons ionize hydrogen rather than heat dust, so the canonical
energy-balance integral (:func:`tengri.forward.energy_balance.
bolometric_absorbed`) masks λ < 912 Å. This matches CIGALE (attenuation
zeroed at λ ≤ 91.2 nm) and Bagpipes (``fesc`` masking of the ionizing
continuum); FSPS by contrast includes LyC absorption in its dust heating.

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

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp, builders
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.forward.energy_balance import bolometric_absorbed
from tengri.observation.photometry import FilterCurve
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.conservation

_WG00_GRID = Path(__file__).resolve().parents[3] / "data" / "wg00_attenuation_grid.h5"

TWO_COMPONENT = {"law_diff": 'calzetti', 
    "type": "two_component",
    "law_bc": "calzetti",
    "*": FIXED,
    "tau_bc": 0.5,
    "tau_diff": 0.3,
}
SINGLE_SCREEN = {"law": "power_law", "type": "single_component", "*": FIXED, "tau_v": 0.5}
WG00 = {"type": "wg00", "*": FIXED, "tau_v": 0.5}

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
            defaults=FIXED,
            log_total_mass=9.75,
            peak_lbt_gyr=6.25,
            width_gyr=2.6,
            skew=0.0,
            trunc=5.5,
        ),
        dust=dust,
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


def _params(model):
    return {**model.spec.get_fixed_values(), **model.spec.sample(jax.random.PRNGKey(0))}


def _zero_lyc(ssp):
    """Copy of the SSP with all flux below 912 Å zeroed."""
    return SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * (ssp.ssp_wave >= 912.0),
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
        transparent = {"law_diff": 'calzetti', 
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
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
    GOLDEN_L_ABSORBED: ClassVar[dict[str, float]] = {
        "two_component": 5.838762e59,
        "single_screen": 7.32462266124103e59,
        "wg00": 6.911563171933806e59,
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

    def test_lut_tracks_exact_on_lyc_bright_fixture(self, synthetic_ssp_wide):
        """WavePrecomp energy-balance LUT tracks the exact path on this fixture.

        The far-IR band is L_ir-dominated, so parity there pins agreement of
        the LUT contraction with the exact LyC-masked integral for the
        UV-bright case specifically.
        """
        dust = dict(
            TWO_COMPONENT,
            emission={"type": "modified_blackbody", "*": FIXED},
        )
        m_exact = _build(synthetic_ssp_wide, dict(dust))
        m_lut = _build(synthetic_ssp_wide, dict(dust), approx=WavePrecomp())
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
        # Index 30 sits well above the 912 Å cutoff, so the Inf survives the
        # LyC mask and must be caught by the finiteness guard instead.
        assert float(wave[30]) > 912.0
        sed_intr = jnp.ones_like(wave).at[30].set(jnp.inf)
        sed_att = jnp.zeros_like(wave)
        out = bolometric_absorbed(sed_intr, sed_att, nu, wave=wave)
        assert jnp.isfinite(out)
        assert float(out) == 0.0

    def test_finite_inputs_unaffected(self):
        wave = jnp.logspace(2.0, 5.0, 50)
        nu = C_AA / wave
        sed_intr = jnp.ones_like(wave)
        sed_att = 0.5 * sed_intr
        out = bolometric_absorbed(sed_intr, sed_att, nu, wave=wave)
        expected = jnp.trapezoid(jnp.where(wave >= 912.0, sed_intr - sed_att, 0.0), nu)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
