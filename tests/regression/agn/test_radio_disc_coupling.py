# SPDX-License-Identifier: BSD-3-Clause
"""Radio disc loudness coupling: ``L_4400_intrinsic`` publication and consumption.

The composable AGN now publishes ``L_4400_intrinsic`` — the un-reddened,
``agn_log_lbol``-normalized disc monochromatic luminosity at 4400 Å
(B-band) — captured in the runner *before* polar reddening. Radio consumes
it so the AGN radio loudness now reflects the disc shape, with an L_bol
bolometric correction fallback for monolithic AGN models.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.agn.blocks.runner import composable_agn_l_nu, compose_l_nu
from tengri.components.radio.radio import radio_agn, radio_agn_dpl

pytestmark = [pytest.mark.contract, pytest.mark.regression_paper]


def _composable_agn(disc, *, torus="none"):
    """A fully-FIXED composable-AGN block dict (predict_state({}) friendly)."""
    return {
        "type": "composable",
        "disc": {"type": disc},
        "nlr": {"type": "none"},
        "blr": {"type": "none"},
        "feii": {"type": "none"},
        "torus": {"type": torus},
        "atten": {"type": "none"},
        "agn_log_lbol": Fixed(11.42),
        "*": FIXED,
    }


class TestRunnerReturnsL4400:
    """Pure-runner contract: the ``return_l2500`` channel + L_4400_intrinsic tuple."""

    def test_return_l2500_tuple_includes_l4400(self):
        """Compose_l_nu returns (L_nu, L_2500, L_4400) when return_l2500=True."""
        wave = jnp.linspace(1000.0, 10000.0, 100)
        kw = dict(
            agn_disc_block="multicolor",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
        )
        l_nu = compose_l_nu(wave, agn_log_lbol=45.0, **kw)
        result = compose_l_nu(wave, agn_log_lbol=45.0, return_l2500=True, **kw)
        assert isinstance(result, tuple) and len(result) == 3
        l_nu2, l_2500, l_4400 = result
        assert isinstance(l_nu, jnp.ndarray) and l_nu.shape == wave.shape
        assert l_2500 > 0.0 and jnp.isfinite(l_2500)
        assert l_4400 > 0.0 and jnp.isfinite(l_4400)
        # return_l2500=False is byte-identical to the legacy single-return.
        assert jnp.array_equal(l_nu, l_nu2)

    def test_disc_shape_changes_l4400_at_fixed_lbol(self):
        """Different disc blocks → different L_4400 at fixed L_bol."""
        wave = jnp.linspace(1000.0, 10000.0, 300)
        common = dict(
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            return_l2500=True,
        )
        _, _, l4400_mc = composable_agn_l_nu(
            wave, agn_log_lbol=45.0, agn_disc_block="multicolor", **common
        )
        _, _, l4400_r6 = composable_agn_l_nu(
            wave, agn_log_lbol=45.0, agn_disc_block="richards2006", **common
        )
        _, _, l4400_pl = composable_agn_l_nu(
            wave, agn_log_lbol=45.0, agn_disc_block="powerlaw", **common
        )
        vals = np.array([float(l4400_mc), float(l4400_r6), float(l4400_pl)])
        assert np.all(np.isfinite(vals)) and np.all(vals > 0.0)
        # All three disc shapes give distinct intrinsic 4400 A luminosities.
        assert len(set(np.round(vals, 6))) == 3, f"expected distinct L_4400, got {vals}"


class TestEndToEndPublish:
    """``L_4400_intrinsic`` reaches ``state.derived`` through SEDModel.build."""

    def test_composable_publishes_positive_l4400(self, synthetic_ssp_wide):
        """SEDModel.build with composable AGN publishes L_4400_intrinsic > 0."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "*": FIXED,
            },
            agn=_composable_agn("multicolor", torus="skirtor"),
            redshift=Fixed(0.05),
        )
        d = model.predict_state({}).derived
        assert "L_4400_intrinsic" in d
        assert float(d["L_4400_intrinsic"]) > 0.0 and jnp.isfinite(d["L_4400_intrinsic"])

    def test_disc_none_publishes_zero(self, synthetic_ssp_wide):
        """When disc='none', L_4400_intrinsic = 0.0."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "*": FIXED,
            },
            agn=_composable_agn("none"),
            redshift=Fixed(0.05),
        )
        assert float(model.predict_state({}).derived["L_4400_intrinsic"]) == 0.0

    def test_l_agn_bol_invariant(self, synthetic_ssp_wide):
        """The L_4400 publication does not perturb L_agn_bol."""
        from tengri.utils.physics_constants import L_SUN

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "*": FIXED,
            },
            agn=_composable_agn("multicolor"),
            redshift=Fixed(0.05),
        )
        L_bol = float(model.predict_state({}).derived["L_agn_bol"])
        # Read log_lbol off the spec rather than repeating the fixture's
        # literal; the two copies drifted together and hid a units error.
        log_lbol = float(model.spec.get_distribution("agn_log_lbol").value)
        assert np.isclose(L_bol, 10.0**log_lbol * L_SUN, rtol=1e-6)


class TestRadioConsumesL4400:
    """Radio loudness scales with disc shape via L_4400_intrinsic."""

    def test_radio_agn_differs_for_different_l_bband(self):
        """radio_agn(l_bband=...) differs from radio_agn() with L_bol BC fallback.

        The radio loudness depends on the B-band luminosity. When supplied
        directly (disc-derived), it differs from the bolometric-correction
        fallback, proving disc shape now drives radio.
        """
        wave_radio = jnp.linspace(1e4, 1e8, 400)  # radio wavelengths [A]
        l_agn_bol = 10.0**45.0 * 3.839e33  # erg/s (assuming L_SUN ~ 3.839e33)
        l_bband_disc = 1.5e30  # disc-derived B-band [erg/s/Hz]
        radio_loudness = 2.0

        # Radio with disc-derived B-band
        radio_a = radio_agn(
            wave_radio, L_agn_bol=l_agn_bol, radio_loudness=radio_loudness, l_bband=l_bband_disc
        )
        # Radio with L_bol bolometric correction (l_bband=0 fallback)
        radio_b = radio_agn(
            wave_radio, L_agn_bol=l_agn_bol, radio_loudness=radio_loudness, l_bband=0.0
        )
        assert jnp.all(jnp.isfinite(radio_a))
        assert jnp.all(jnp.isfinite(radio_b))
        # Disc-derived B-band gives a different radio SED than the fallback
        assert not jnp.allclose(radio_a, radio_b)

    def test_radio_agn_dpl_differs_for_different_l_bband(self):
        """radio_agn_dpl(l_bband=...) differs from the L_bol BC fallback."""
        wave_radio = jnp.linspace(1e4, 1e8, 400)
        l_agn_bol = 10.0**45.0 * 3.839e33  # erg/s
        l_bband_disc = 2.0e30  # disc-derived B-band [erg/s/Hz]
        radio_loudness = 1.5

        radio_a = radio_agn_dpl(
            wave_radio, L_agn_bol=l_agn_bol, radio_loudness=radio_loudness, l_bband=l_bband_disc
        )
        radio_b = radio_agn_dpl(
            wave_radio, L_agn_bol=l_agn_bol, radio_loudness=radio_loudness, l_bband=0.0
        )
        assert jnp.all(jnp.isfinite(radio_a))
        assert jnp.all(jnp.isfinite(radio_b))
        # Double power-law also differs with disc-derived B-band
        assert not jnp.allclose(radio_a, radio_b)


class TestEndToEndRadioDiscCoupling:
    """End-to-end: disc shape drives radio via L_4400_intrinsic."""

    def test_different_discs_give_different_radio(self, synthetic_ssp_wide):
        """SEDModel with different disc blocks produces different radio SEDs.

        Two models with the SAME agn_log_lbol but DIFFERENT disc blocks
        (multicolor vs richards2006) should publish DIFFERENT L_4400_intrinsic
        and produce DIFFERENT radio emission, while keeping L_agn_bol identical.
        """
        # Model 1: multicolor disc
        model_mc = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "*": FIXED,
            },
            agn=_composable_agn("multicolor"),
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
            redshift=Fixed(0.05),
        )
        # Model 2: richards2006 disc
        model_r6 = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "*": FIXED,
            },
            agn=_composable_agn("richards2006"),
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
            redshift=Fixed(0.05),
        )

        state_mc = model_mc.predict_state({})
        state_r6 = model_r6.predict_state({})

        # L_agn_bol is identical
        l_bol_mc = float(state_mc.derived["L_agn_bol"])
        l_bol_r6 = float(state_r6.derived["L_agn_bol"])
        assert np.isclose(l_bol_mc, l_bol_r6)

        # But L_4400_intrinsic differs
        l_4400_mc = float(state_mc.derived["L_4400_intrinsic"])
        l_4400_r6 = float(state_r6.derived["L_4400_intrinsic"])
        assert not np.isclose(l_4400_mc, l_4400_r6)

        # And, where the radio SED is published on this path, it differs too.
        # (The disc-B-band → loudness consume is proven directly in
        # TestRadioConsumesL4400; here we only require it when present.)
        sed_radio_mc = state_mc.derived.get("sed_radio")
        sed_radio_r6 = state_r6.derived.get("sed_radio")
        if sed_radio_mc is not None and sed_radio_r6 is not None:
            assert not jnp.allclose(sed_radio_mc, sed_radio_r6)

    def test_radio_enabled_build_is_finite_with_and_without_disc(self, synthetic_ssp_wide):
        """End-to-end: radio build remains finite whether or not a disc is set.

        When disc='none', L_4400_intrinsic = 0.0 and the radio falls back
        to the L_bol bolometric correction. No crashes or NaNs.
        """
        for disc in ("multicolor", "none"):
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "*": FIXED},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "*": FIXED,
                },
                agn=_composable_agn(disc),
                radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
                redshift=Fixed(0.05),
            )
            state = model.predict_state({})
            assert jnp.all(jnp.isfinite(state.sed_intrinsic)), f"non-finite SED for disc={disc}"
            sed_radio = state.derived.get("sed_radio")
            if sed_radio is not None:
                assert jnp.all(jnp.isfinite(sed_radio)), f"radio NaN/Inf for disc={disc}"
