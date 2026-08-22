# SPDX-License-Identifier: BSD-3-Clause
"""Bounds tests for xray_log_nh photoelectric absorption parameter.

Verifies that the fittable xray_log_nh parameter (hydrogen column density)
correctly attenuates soft X-ray flux below ~2 keV, with higher N_H giving
lower flux. Tests the Morrison & McCammon (1983) photoelectric absorption
cross-section via tbabs_transmission.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


class TestXrayLogNh:
    def test_higher_nh_suppresses_soft_xray(self):
        r"""Higher log_nh suppresses flux below ~2 keV (photoelectric absorption).

        Photoelectric absorption cross-section σ(E) ∝ E^-3 below the K-edge
        (~0.14 keV for neutral hydrogen). Line-of-sight transmission
        T(E) = exp(−σ(E)·N_H) is energy-dependent: at low E, high N_H gives
        very low transmission. At high E (>~ 10 keV), σ → 0 and T → 1
        regardless of N_H.

        Physical bound: 0.5 keV flux @ log_nh=20 > 0.5 keV flux @ log_nh=23.
        """
        from tengri.components.xray.xray import xray_total

        # Rest-frame wavelength grid spanning soft to hard X-ray
        wave = jnp.array([12.4, 6.2, 1.24, 0.62])  # ~1, 2, 10, 20 keV

        # Minimal configuration: AGN-only, no XRB/hot-gas
        l_2500_30deg = 1e25  # erg/s/Hz
        sfr = 0.0  # No star formation
        stellar_mass = 1e10  # Required for contract, but SFR=0 → no XRB contribution

        # Predict at two column densities
        L_thin = xray_total(
            wave,
            sfr=sfr,
            stellar_mass=stellar_mass,
            l_2500_30deg=l_2500_30deg,
            log_nh=20.0,  # Optically thin
        )
        L_thick = xray_total(
            wave,
            sfr=sfr,
            stellar_mass=stellar_mass,
            l_2500_30deg=l_2500_30deg,
            log_nh=23.0,  # Compton-thick
        )

        # Soft X-ray (0.5–2 keV): index 0 and 1 (12.4 and 6.2 Å)
        soft_thin = float(np.mean(np.asarray(L_thin[:2])))
        soft_thick = float(np.mean(np.asarray(L_thick[:2])))

        # Hard X-ray (10–20 keV): indices 2 and 3 (1.24 and 0.62 Å)
        hard_thin = float(np.mean(np.asarray(L_thin[2:])))
        hard_thick = float(np.mean(np.asarray(L_thick[2:])))

        # Higher N_H must suppress soft X-ray strongly
        assert soft_thick < soft_thin, (
            f"Soft X-ray not suppressed by absorption: "
            f"log_nh=20 → {soft_thin:.2e}, log_nh=23 → {soft_thick:.2e}"
        )

        # Hard X-ray suppression is mild (mostly via Compton scattering at edge)
        hard_suppression_factor = hard_thick / hard_thin if hard_thin > 0 else 1.0
        assert 0.5 < hard_suppression_factor < 1.0, (
            f"Hard X-ray suppression factor = {hard_suppression_factor:.3f}, "
            f"expected ~0.5–1.0 (Compton dominates at N_H=23)"
        )

        # Soft attenuation factor should be much larger than hard
        soft_attenuation = soft_thick / soft_thin if soft_thin > 0 else 1.0
        hard_attenuation = hard_suppression_factor
        assert soft_attenuation < hard_attenuation, (
            f"Soft band attenuation ({soft_attenuation:.3f}) should exceed "
            f"hard band ({hard_attenuation:.3f}) due to photoelectric edge"
        )

    def test_xray_log_nh_suppresses_agn_corona_e2e(self, synthetic_ssp_wide):
        r"""End-to-end: xray_log_nh attenuates the AGN X-ray corona.

        Exercises all wiring layers through the full forward model — parameter
        declaration (``_params.py``), component read (``params['xray_log_nh']``
        in ``component.py``), and threading into ``xray_total`` →
        ``tbabs_transmission``. ``tbabs`` absorbs the AGN corona (``l_intr``
        derived from L_2500), so a high column strongly suppresses the soft
        X-ray SED. Runs on the synthetic SSP (the X-ray component extends the
        rest grid into the X-ray), so it is CI-runnable without real data.
        """
        import jax

        from tengri import FIXED, Fixed, SEDModel

        def _soft_xray_luminosity(log_nh: float) -> float:
            # Cool, sub-Eddington disc (log_lbol=10.5, log10 L_sun) so the disc's
            # own Wien tail does NOT reach the 0.2-2 keV band — leaving that band
            # dominated by the AGN corona, the component xray_log_nh absorbs.
            # (Under the luminosity-first disc, ADR-0020, a *hot* disc emits soft
            # X-rays into the band that are not attenuated by the corona column,
            # which would dilute the suppression; a cool disc isolates the corona
            # so N_H attenuation reads cleanly.)
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "all_params": FIXED, "log_total_mass": 10.0},
                agn={"type": "multicolor_agn", "all_params": FIXED, "log_lbol": Fixed(10.5)},
                xray={"type": "yang20", "all_params": FIXED, "log_nh": Fixed(log_nh)},
                redshift=Fixed(0.05),
            )
            params = dict(model.spec.sample(jax.random.PRNGKey(0)))
            # The param must actually reach the spec (guards the silent-no-op
            # class: declared-but-unthreaded params).
            assert "xray_log_nh" in params
            assert float(params["xray_log_nh"]) == pytest.approx(log_nh)
            state = model.predict_rest_sed(params)
            w = np.asarray(state.wavelength)
            sed = np.asarray(state.sed)
            band = (w > 6.0) & (w < 60.0)  # ~0.2-2 keV soft X-ray
            return float(np.trapezoid(sed[band], w[band]))

        soft_thin = _soft_xray_luminosity(20.0)  # optically thin
        soft_thick = _soft_xray_luminosity(24.5)  # Compton-thick

        assert soft_thick < 0.5 * soft_thin, (
            f"xray_log_nh did not attenuate the soft X-ray AGN corona "
            f"(silent no-op?): N_H=20 → {soft_thin:.3e}, N_H=24.5 → {soft_thick:.3e}"
        )
