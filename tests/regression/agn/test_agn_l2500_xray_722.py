# SPDX-License-Identifier: BSD-3-Clause
"""Issue #722: ``L_2500_intrinsic`` publication + disc-shape-dependent X-ray.

The composable AGN now publishes ``L_2500_intrinsic`` — the un-reddened,
``agn_log_lbol``-normalized disc monochromatic luminosity at 2500 Å — captured
in the runner *before* polar reddening. X-ray consumes it (fallback chain:
``L_2500_intrinsic`` → ``L_2500_30deg`` → L_bol bolometric correction) so the
AGN corona's :math:`\\alpha_{\\rm ox}` (Just+2007) now reflects the disc shape.
Radio deliberately stays on the B-band (4400 Å) L_bol correction.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.agn.blocks.runner import composable_agn_l_nu, compose_l_nu
from tengri.components.xray.xray import xray_total

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


class TestRunnerReturnsL2500:
    """Pure-runner contract: the ``return_l2500`` channel + disc-shape spread."""

    def test_return_l2500_tuple_and_backcompat(self):
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
        # return_l2500=True returns (L_nu, L_2500_intrinsic, L_4400_intrinsic).
        l_nu2, l_2500, l_4400 = compose_l_nu(wave, agn_log_lbol=45.0, return_l2500=True, **kw)
        assert isinstance(l_nu, jnp.ndarray) and l_nu.shape == wave.shape
        assert l_2500 > 0.0 and jnp.isfinite(l_2500)
        assert l_4400 > 0.0 and jnp.isfinite(l_4400)
        # return_l2500=False is byte-identical to the legacy single-return.
        assert jnp.array_equal(l_nu, l_nu2)

    def test_disc_shape_changes_l2500_at_fixed_lbol(self):
        """Headline #722: different disc blocks → different L_2500 at fixed L_bol."""
        wave = jnp.linspace(1000.0, 10000.0, 300)
        common = dict(
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            return_l2500=True,
        )
        _, l2500_mc, _ = composable_agn_l_nu(
            wave, agn_log_lbol=45.0, agn_disc_block="multicolor", **common
        )
        _, l2500_r6, _ = composable_agn_l_nu(
            wave, agn_log_lbol=45.0, agn_disc_block="richards2006", **common
        )
        _, l2500_pl, _ = composable_agn_l_nu(
            wave, agn_log_lbol=45.0, agn_disc_block="powerlaw", **common
        )
        vals = np.array([float(l2500_mc), float(l2500_r6), float(l2500_pl)])
        assert np.all(np.isfinite(vals)) and np.all(vals > 0.0)
        # All three disc shapes give distinct intrinsic 2500 A luminosities.
        assert len(set(np.round(vals, 6))) == 3, f"expected distinct L_2500, got {vals}"

    def test_l2500_independent_of_free_inclination(self):
        """CIGALE intrin_Lnu_2500A_30deg: L_2500/L_4400 are evaluated at a FIXED

        30 deg reference, so they do NOT change with the (free) viewing angle
        agn_cos_inc — only the observed SED foreshortens. This keeps alpha_ox and
        radio loudness anchored to the intrinsic accretion luminosity.
        """
        wave = jnp.linspace(1000.0, 10000.0, 300)
        common = dict(
            agn_disc_block="multicolor",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            return_l2500=True,
        )
        # Edge-on vs face-on viewing — same disc, same L_bol.
        _, l2500_edge, l4400_edge = composable_agn_l_nu(
            wave, agn_log_lbol=45.0, agn_cos_inc=0.2, **common
        )
        _, l2500_face, l4400_face = composable_agn_l_nu(
            wave, agn_log_lbol=45.0, agn_cos_inc=0.98, **common
        )
        assert np.isclose(float(l2500_edge), float(l2500_face), rtol=1e-9)
        assert np.isclose(float(l4400_edge), float(l4400_face), rtol=1e-9)


class TestEndToEndPublish:
    """``L_2500_intrinsic`` reaches ``state.derived`` through SEDModel.build."""

    def test_composable_publishes_positive_l2500(self, synthetic_ssp_wide):
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust={
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
        assert "L_2500_intrinsic" in d
        assert float(d["L_2500_intrinsic"]) > 0.0 and jnp.isfinite(d["L_2500_intrinsic"])

    def test_l_agn_bol_invariant(self, synthetic_ssp_wide):
        """The new key does not perturb L_agn_bol (= 10**log_lbol * L_sun)."""
        from tengri.utils.physics_constants import L_SUN

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust={
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
        # literal. The two copies previously drifted together and hid a units
        # error: both said 45, which is a log10(erg/s) magnitude, so the test
        # confirmed a 3.8e78 erg/s AGN against itself and passed.
        log_lbol = float(model.spec.get_distribution("agn_log_lbol").value)
        assert np.isclose(L_bol, 10.0**log_lbol * L_SUN, rtol=1e-6)

    def test_disc_none_publishes_zero(self, synthetic_ssp_wide):
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "*": FIXED,
            },
            agn=_composable_agn("none"),
            redshift=Fixed(0.05),
        )
        assert float(model.predict_state({}).derived["L_2500_intrinsic"]) == 0.0


class TestXRayConsumesL2500:
    """The X-ray corona is now non-zero and L_2500-dependent (consume side)."""

    def test_xray_corona_scales_with_l2500(self):
        """xray_total AGN corona differs for different l_2500_30deg (#722 fix:

        previously the live component passed L_agn_bol where l_2500_30deg was
        expected, so the AGN corona was silently zero).
        """
        wave_xray = jnp.linspace(1.0, 100.0, 400)  # ~0.12-12 keV band [A]
        l2500_a = 1.0e30
        l2500_b = 4.0e30
        agn_a = xray_total(wave_xray, sfr=0.0, stellar_mass=1.0, l_2500_30deg=l2500_a)
        agn_b = xray_total(wave_xray, sfr=0.0, stellar_mass=1.0, l_2500_30deg=l2500_b)
        agn_zero = xray_total(wave_xray, sfr=0.0, stellar_mass=1.0, l_2500_30deg=0.0)
        assert jnp.all(jnp.isfinite(agn_a))
        # Non-zero AGN corona when L_2500 > 0 ...
        assert float(jnp.trapezoid(agn_a, wave_xray)) != 0.0
        # ... and it scales with L_2500 (brighter disc → different corona) ...
        assert not jnp.allclose(agn_a, agn_b)
        # ... while L_2500=0 (no-AGN fallback floor) gives a strictly smaller corona.
        assert float(jnp.sum(jnp.abs(agn_zero))) < float(jnp.sum(jnp.abs(agn_a)))

    def test_xray_enabled_build_is_finite_with_and_without_agn(self, synthetic_ssp_wide):
        """End-to-end: an X-ray-enabled build predicts finite output whether or

        not a disc is set (the L_2500 fallback path must not crash/NaN).
        """
        for disc in ("multicolor", "none"):
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "*": FIXED},
                dust={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "*": FIXED,
                },
                agn=_composable_agn(disc),
                xray={"type": "simple"},
                redshift=Fixed(0.05),
            )
            state = model.predict_state({})
            assert jnp.all(jnp.isfinite(state.sed_intrinsic)), f"non-finite SED for disc={disc}"
            sed_xray = state.derived.get("sed_xray")
            if sed_xray is not None:
                assert jnp.all(jnp.isfinite(sed_xray)), f"X-ray NaN/Inf for disc={disc}"
