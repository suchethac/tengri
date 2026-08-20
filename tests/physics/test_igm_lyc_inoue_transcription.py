# SPDX-License-Identifier: BSD-3-Clause
"""Transcription guard for Inoue+2014 Lyman-continuum optical depth formulas.

Verifies that the LAF and DLA Lyman-continuum implementations are faithful
transcriptions of Inoue et al. 2014, MNRAS 442, 1805, Eqs. 25–29:

- Branch continuity at regime boundaries (|Δtau| < 1e-3)
- Optical depth at source Lyman limit is small (< 0.02)
- Pinned probe values match measured transmissions from the paper
- Monotonicity of transmission with wavelength (LyC path accumulates opacity)
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.igm import igm_transmission
from tengri.components.igm.igm import _LAMBDA_LIMIT, _tau_lc_dla, _tau_lc_laf

pytestmark = pytest.mark.regression_paper


class TestInoueTranscriptionGuard:
    """Transcription guard for Inoue+2014 LAF and DLA Lyman-continuum."""

    def test_laf_branch_continuity_mid_z_z3(self):
        """LAF mid-z seam (z1=1.2) is continuous at z=3.

        Inoue+2014 Eqs. 25–27: LAF opacity switches at wave_obs = lambda_L*(1+z1)
        with z1=1.2 for 1.2 <= z < 4.7. Continuity requires |Δtau| < 1e-3.
        """
        lam_L = _LAMBDA_LIMIT
        z_seam = 1.2
        z_source = 3.0
        wave_seam = lam_L * (1 + z_seam)

        wave_below = jnp.array([wave_seam - 0.01])
        wave_above = jnp.array([wave_seam + 0.01])

        tau_below = float(np.asarray(_tau_lc_laf(wave_below, z_source))[0])
        tau_above = float(np.asarray(_tau_lc_laf(wave_above, z_source))[0])
        delta_tau = abs(tau_above - tau_below)

        assert delta_tau < 1e-3, (
            f"LAF mid-z seam discontinuity at z=3: |Δtau| = {delta_tau:.8f} "
            "(Inoue+2014 requires continuous)"
        )

    def test_laf_branch_continuity_high_z_seams_z55(self):
        """LAF high-z seams (z1=1.2, z2=4.7) are continuous at z=5.5.

        Inoue+2014 Eqs. 25–27: For z >= 4.7, LAF opacity has three regimes
        split at wave_obs = lambda_L*(1+z1) and lambda_L*(1+z2).
        """
        lam_L = _LAMBDA_LIMIT
        z_source = 5.5

        for z_break, name in [(1.2, "mid-z"), (4.7, "high-z")]:
            wave_seam = lam_L * (1 + z_break)
            wave_below = jnp.array([wave_seam - 0.01])
            wave_above = jnp.array([wave_seam + 0.01])

            tau_below = float(np.asarray(_tau_lc_laf(wave_below, z_source))[0])
            tau_above = float(np.asarray(_tau_lc_laf(wave_above, z_source))[0])
            delta_tau = abs(tau_above - tau_below)

            assert delta_tau < 1e-3, (
                f"LAF {name} seam discontinuity at z=5.5: |Δtau| = {delta_tau:.8f}"
            )

    def test_dla_branch_continuity_z1_seam(self):
        """DLA seam (z1=2.0) is continuous.

        Inoue+2014 Eqs. 28–29: DLA opacity switches at wave_obs = lambda_L*(1+z1)
        with z1=2.0 for z >= 2.0. Continuity required.
        """
        lam_L = _LAMBDA_LIMIT
        z_seam = 2.0
        wave_seam = lam_L * (1 + z_seam)

        for z_source in [3.0, 5.5]:
            wave_below = jnp.array([wave_seam - 0.01])
            wave_above = jnp.array([wave_seam + 0.01])

            tau_below = float(np.asarray(_tau_lc_dla(wave_below, z_source))[0])
            tau_above = float(np.asarray(_tau_lc_dla(wave_above, z_source))[0])
            delta_tau = abs(tau_above - tau_below)

            assert delta_tau < 1e-3, (
                f"DLA seam discontinuity at z={z_source}: |Δtau| = {delta_tau:.8f}"
            )

    def test_source_lyman_limit_opacity_z3(self):
        """At rest Lyman limit, tau_lc_laf + tau_lc_dla is small at z=3.

        Inoue+2014: The Lyman-continuum opacity is defined as an integral
        over (0, z_source]. At wave_obs = lambda_L*(1+z_source) - 1.0 Å
        (just short of the source's Lyman limit), the path is nearly complete,
        but the opacity is small (< 0.02) because the integral is cut off.
        """
        lam_L = _LAMBDA_LIMIT
        z_source = 3.0
        wave_at_limit = lam_L * (1 + z_source) - 1.0

        tau_laf = float(np.asarray(_tau_lc_laf(jnp.array([wave_at_limit]), z_source))[0])
        tau_dla = float(np.asarray(_tau_lc_dla(jnp.array([wave_at_limit]), z_source))[0])
        total_tau = tau_laf + tau_dla

        assert total_tau < 0.02, (
            f"Source-limit opacity at z=3 too large: tau = {total_tau:.6f} (expected < 0.02)"
        )

    def test_source_lyman_limit_opacity_z55(self):
        """At rest Lyman limit, tau_lc_laf + tau_lc_dla is small at z=5.5."""
        lam_L = _LAMBDA_LIMIT
        z_source = 5.5
        wave_at_limit = lam_L * (1 + z_source) - 1.0

        tau_laf = float(np.asarray(_tau_lc_laf(jnp.array([wave_at_limit]), z_source))[0])
        tau_dla = float(np.asarray(_tau_lc_dla(jnp.array([wave_at_limit]), z_source))[0])
        total_tau = tau_laf + tau_dla

        assert total_tau < 0.02, (
            f"Source-limit opacity at z=5.5 too large: tau = {total_tau:.6f} (expected < 0.02)"
        )

    def test_pinned_t_obs_3000a_z3(self):
        """T at obs 3000 Å (rest 750 Å), z=3 matches Inoue+2014.

        Rest 750 Å is ~160 Å below the Lyman limit, with an absorbing path
        Δz ≈ 0.7 (about 1.5 mean free paths). Inoue+2014 predicts
        T ≈ 0.172 (tau_LAF^LC ≈ 0.77 + tau_DLA^LC ≈ 0.76 + Lyman-series
        ≈ 0.25 → T ≈ e^-1.78). Measured value ±3% tolerance.
        """
        wave_obs = jnp.array([3000.0])
        T = float(np.asarray(igm_transmission(wave_obs, 3.0))[0])

        np.testing.assert_allclose(
            T, 0.171848, rtol=0.03, err_msg="T(obs 3000 A, z=3) from Inoue+2014 Eqs. 25–29"
        )

    def test_pinned_mean_t_rest_800_900a_z4(self):
        """Mean T over rest 800–900 Å at z=4 matches Inoue+2014.

        Rest 800–900 Å represents a path Δz ≈ 0.3–0.6 below the Lyman limit
        (about 1.5 mean free paths), not the asymptotic tau_LL >> 1 regime.
        Inoue+2014 predicts mean T ≈ 0.114. Measured value ±3% tolerance.
        """
        z = 4.0
        wave_obs = jnp.linspace(800.0, 900.0, 32) * (1 + z)
        T = float(jnp.mean(igm_transmission(wave_obs, z)))

        np.testing.assert_allclose(
            T,
            0.114284,
            rtol=0.03,
            err_msg="Mean T(rest 800–900 A, z=4) from Inoue+2014 Eqs. 25–29",
        )

    def test_monotonic_transmission_blueward_at_z3(self):
        """IGM transmission is non-increasing going blueward (Δz accumulates).

        At z=3, going blueward from obs 3600 Å to obs 2000 Å, the rest-frame
        wavelength decreases while the path length *into the IGM* increases
        (the Δz integral extends further to higher redshifts), accumulating
        opacity. Transmission should be non-increasing with decreasing wave_obs.
        """
        z = 3.0
        # Going from longer to shorter observed-frame wavelengths (blueward)
        wave_obs = jnp.array([3600.0, 3500.0, 3000.0, 2500.0, 2000.0])
        T_igm = np.asarray(igm_transmission(wave_obs, z))

        # As wave_obs decreases (blueward), T should be non-increasing (diffs <= 0)
        diffs = np.diff(T_igm)
        assert np.all(diffs <= 1e-8), (
            f"Transmission should be non-increasing blueward at z=3; found diffs = {diffs}"
        )
