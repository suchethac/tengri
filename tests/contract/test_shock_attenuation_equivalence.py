# SPDX-License-Identifier: BSD-3-Clause
"""
Shock dust attenuation equivalence: exact vs precomp paths (#1434).

Regression test for #1434: shock dust attenuation was inconsistent between paths.
- Exact path (predict): applied NO attenuation (shock was in unattenuated non_stellar_other)
- Precomp path (predict_via_precomp): applied a_diff·a_bc
- Measurement: 37.7% photometric disagreement at tau_bc=2/z=1 on FSPS SSP + SDSS gri

Fix (#1434): Unified the seam at two_component.py lines 819-820. Both paths apply
the SAME dust attenuation -- both derive from ``sed_shock_attenuated``, computed
once in ``DustSEDComponent.apply()`` step 2d and reused by both the exact SED
sum and the precomp band-projection, so they cannot drift relative to each
other regardless of which screen is selected.

#2234 (a replacement) made the screen a per-source choice
(``dust_attenuation={'shock_screen': ...}``) and changed the DEFAULT from the
unconditional young-limit form (``tau_bc·k_bc + tau_diff·k_diff``, this test's
original fixture) to diffuse-only (``tau_diff·k_diff``): an AGN-outflow shock is
not, in general, still inside the birth cloud the young-star screen models.
Measured on this fixture (tau_bc=2, tau_diff=1, z=0.5, SDSS gri): the exact-vs-
precomp gap is unaffected by the default flip (both screens agree to well
under this test's 5% bound), but the shock contribution to photometry itself
grows by roughly 8x-33x band to band, because dropping the tau_bc·k_bc term
removes the dominant attenuation factor. Dust component publishes
shock_phot_lnu_attenuated_precomp; precomp path reads it. Exact path applies
attenuation in two_component.apply(), step 2d.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.contract


@pytest.mark.regression_bug
class TestShockAttenuationEquivalence:
    """Shock emission receives consistent dust attenuation on both surfaces (#1434)."""

    @pytest.mark.unit
    def test_shock_attenuation_exact_vs_precomp(self):
        """
        Photometry with shock agrees between exact and precomp paths (#1434).

        Assertion: both paths measure the same shock photometric effect
        (delta = phot_with_shock - phot_without_shock) within the precomp
        dust-channel accuracy (stellar sub-band quadrature floor ~0.6%).

        Measured pre-#1434-fix disagreement: 37.7% at tau_bc=2/z=1. Both
        paths derive from the same ``sed_shock_attenuated`` value regardless
        of which ``shock_screen`` choice is active (#2234), so re-measuring on
        this exact fixture today gives agreement to float precision
        (~1e-12) under BOTH the new default (``"diffuse"``) and the old one
        (``"birth_cloud"``) -- the 5% bound below is generous headroom, not
        a tight measurement.

        Precondition: shock contribution > noise floor (non-vacuity check—
        verifies shock is resolved and contributes measurably to photometry).
        """
        from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.data import download_ssp

        # Use a real SSP for realistic shock contribution (bare stellar only)
        # wNE versions aren't in the downloadable catalog; use bare + nebular backend
        ssp_path = download_ssp("fsps_prsc_miles_chabrier")
        ssp = load_ssp_data(str(ssp_path))

        # Build a simple model with shock + dust
        model_exact = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(
                photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])
            ),
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "tau_gyr": 1.0,
                "log_total_mass": 10.0,
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_bc": 2.0,
                "tau_diff": 1.0,
            },
            neb={"type": "none"},
            shock={"frac": 1.0, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.5),
        )

        # Same model but precomp path
        from tengri import WavePrecomp

        model_precomp = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(
                photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])
            ),
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "tau_gyr": 1.0,
                "log_total_mass": 10.0,
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_bc": 2.0,
                "tau_diff": 1.0,
            },
            neb={"type": "none"},
            shock={"frac": 1.0, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.5),
            approx=WavePrecomp(),  # Enables WavePrecomp for photometry
        )

        params = {}  # Use defaults (all Fixed in this model)

        # Compute deltas: with shock minus without shock
        # Exact path
        phot_exact_with_shock = model_exact.predict_photometry(params)
        params_no_shock = {"shock_frac": 0.0}
        phot_exact_without_shock = model_exact.predict_photometry(params_no_shock)
        delta_exact = phot_exact_with_shock - phot_exact_without_shock

        # Precomp path
        phot_precomp_with_shock = model_precomp.predict_photometry(params)
        phot_precomp_without_shock = model_precomp.predict_photometry(params_no_shock)
        delta_precomp = phot_precomp_with_shock - phot_precomp_without_shock

        # Precondition: shock contribution must be above noise floor
        # (otherwise test is vacuous — it passes trivially for zero difference)
        delta_magnitude = np.abs(delta_exact)
        noise_floor = 1e-32  # Approximate photometry noise floor
        max_delta = delta_magnitude.max()
        msg = f"Shock contribution ({max_delta:.2e}) below noise floor ({noise_floor:.2e})"
        assert max_delta > noise_floor, msg

        # Assert: both paths measure the same shock effect within dust-channel accuracy.
        # Measured stellar sub-band floor: ~0.6% on FSPS through SDSS gri.
        # Bound set at 5% to account for shock's line-dominated spectrum (worse
        # than stellar continuum under band-averaging). Pre-#1434-fix: 37.7%
        # disagreement. Post-fix, re-measured under the #2234 diffuse-only default:
        # ~1.7e-12 (float precision) on this fixture -- both screen choices give
        # the same near-exact agreement since both paths share one computed value.
        rel_error = np.abs(
            (delta_exact - delta_precomp) / np.maximum(np.abs(delta_exact), noise_floor)
        )
        max_rel_error = 0.05  # Conservative margin for shock line sampling
        assert np.all(rel_error < max_rel_error), (
            f"Shock attenuation disagreement: {rel_error.max():.3%} exceeds {max_rel_error:.1%}\n"
            f"delta_exact = {delta_exact}\n"
            f"delta_precomp = {delta_precomp}\n"
            f"rel_error = {rel_error}"
        )
