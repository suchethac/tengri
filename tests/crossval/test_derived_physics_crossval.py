# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate derived physical quantities against published relations.

Tests SSP color sequences, mass-to-light ratios, and IGM transmission
against published references and analytic expectations.

References
----------
- Balogh et al. 1999, ApJ, 527, 54 — Dn4000 age sequence
- Conroy, Gunn & White 2009, ApJ, 699, 486 — M/L ratios
- Inoue et al. 2014, MNRAS, 442, 1805 — IGM transmission
- Kewley et al. 2001, ApJ, 556, 121 — BPT diagram
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.forward.sed_model import SEDModel
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform
from tests._bounds import assert_non_negative
from tests.crossval.conftest import SSP_EXISTS

pytestmark = [
    pytest.mark.crossval,
    pytest.mark.skipif(not SSP_EXISTS, reason="SSP data not found"),
]


# ── 1. SSP color sequence with age ────────────────────────────────


class TestSSPColorSequence:
    """SSP colors must evolve monotonically with age: young=blue, old=red."""

    def test_dn4000_increases_with_age(self, ssp_data):
        """Dn4000 must increase monotonically with SSP age.

        The 4000A break deepens as massive stars die and metal
        absorption strengthens in older populations.
        """
        from tengri.utils.sed_quantities import compute_dn4000

        wave = ssp_data.ssp_wave
        ssp_flux = ssp_data.ssp_flux  # shape (n_met, n_age, n_wave)
        ages_yr = 10**ssp_data.ssp_lg_age_gyr * 1e9

        # Use solar metallicity (middle of grid)
        n_met = ssp_flux.shape[0]
        met_idx = n_met // 2

        # Sample ages: 10 Myr, 100 Myr, 1 Gyr, 5 Gyr, 10 Gyr
        target_ages = [1e7, 1e8, 1e9, 5e9, 1e10]
        dn4000_values = []

        for target in target_ages:
            age_idx = int(np.argmin(np.abs(np.array(ages_yr) - target)))
            sed = ssp_flux[met_idx, age_idx, :]
            dn = float(compute_dn4000(jnp.array(sed), jnp.array(wave)))
            dn4000_values.append(dn)

        # Dn4000 should increase monotonically with age
        for i in range(len(dn4000_values) - 1):
            assert dn4000_values[i] < dn4000_values[i + 1], (
                f"Dn4000 not increasing: age={target_ages[i]:.0e} yr → "
                f"Dn4000={dn4000_values[i]:.3f}, age={target_ages[i + 1]:.0e} yr → "
                f"Dn4000={dn4000_values[i + 1]:.3f}"
            )

    def test_uv_slope_reddens_with_age(self, ssp_data):
        """UV slope beta should redden (increase) with SSP age."""
        from tengri.utils.sed_quantities import compute_uv_slope_beta

        wave = ssp_data.ssp_wave
        ssp_flux = ssp_data.ssp_flux
        ages_yr = 10**ssp_data.ssp_lg_age_gyr * 1e9
        n_met = ssp_flux.shape[0]
        met_idx = n_met // 2

        # Young SSP (10 Myr) should be blue (beta < -1.5)
        age_young_idx = int(np.argmin(np.abs(np.array(ages_yr) - 1e7)))
        sed_young = jnp.array(ssp_flux[met_idx, age_young_idx, :])
        beta_young = float(compute_uv_slope_beta(sed_young, jnp.array(wave)))

        # Old SSP (5 Gyr) should be red (beta > -1.0)
        age_old_idx = int(np.argmin(np.abs(np.array(ages_yr) - 5e9)))
        sed_old = jnp.array(ssp_flux[met_idx, age_old_idx, :])
        beta_old = float(compute_uv_slope_beta(sed_old, jnp.array(wave)))

        assert beta_young < beta_old, (
            f"UV slope not reddening with age: young={beta_young:.2f}, old={beta_old:.2f}"
        )

    def test_young_ssp_blue_beta(self, ssp_data):
        """10 Myr SSP should have beta < -1.0 (dominated by O/B stars)."""
        from tengri.utils.sed_quantities import compute_uv_slope_beta

        wave = ssp_data.ssp_wave
        ssp_flux = ssp_data.ssp_flux
        ages_yr = 10**ssp_data.ssp_lg_age_gyr * 1e9
        n_met = ssp_flux.shape[0]
        met_idx = n_met // 2

        age_idx = int(np.argmin(np.abs(np.array(ages_yr) - 1e7)))
        sed = jnp.array(ssp_flux[met_idx, age_idx, :])
        beta = float(compute_uv_slope_beta(sed, jnp.array(wave)))

        assert beta < -1.0, f"10 Myr SSP beta = {beta:.2f}, expected < -1.0"


# ── 2. SSP mass-to-light ratio ────────────────────────────────────


class TestSSPMassLightRatio:
    """M/L must increase with age (more mass in dim evolved stars).

    Reference: Conroy, Gunn & White 2009, Table 2.
    """

    def test_ml_increases_with_age(self, ssp_data):
        """M/L in V-band should increase monotonically with SSP age."""
        from tengri.utils.sed_quantities import _mean_flux_in_band

        wave = jnp.array(ssp_data.ssp_wave)
        ssp_flux = ssp_data.ssp_flux
        ages_yr = 10**ssp_data.ssp_lg_age_gyr * 1e9
        n_met = ssp_flux.shape[0]
        met_idx = n_met // 2

        # M/L = 1 Msun / L_V  (SSP flux is per Msun)
        target_ages = [1e8, 1e9, 5e9, 1e10]
        ml_values = []

        for target in target_ages:
            age_idx = int(np.argmin(np.abs(np.array(ages_yr) - target)))
            sed = jnp.array(ssp_flux[met_idx, age_idx, :])
            l_v = float(_mean_flux_in_band(sed, wave, 5000.0, 5800.0))
            if l_v > 0:
                ml_values.append(1.0 / l_v)
            else:
                ml_values.append(float("inf"))

        # M/L should increase with age
        for i in range(len(ml_values) - 1):
            assert ml_values[i] < ml_values[i + 1], (
                f"M/L not increasing: age={target_ages[i]:.0e} M/L={ml_values[i]:.3f}, "
                f"age={target_ages[i + 1]:.0e} M/L={ml_values[i + 1]:.3f}"
            )

    def test_ml_increases_with_metallicity(self, ssp_data):
        """M/L should increase with metallicity (more line blanketing)."""
        from tengri.utils.sed_quantities import _mean_flux_in_band

        wave = jnp.array(ssp_data.ssp_wave)
        ssp_flux = ssp_data.ssp_flux
        ages_yr = 10**ssp_data.ssp_lg_age_gyr * 1e9

        # Fix age at 5 Gyr
        age_idx = int(np.argmin(np.abs(np.array(ages_yr) - 5e9)))

        # Compare lowest and highest metallicity
        n_met = ssp_flux.shape[0]
        sed_lo = jnp.array(ssp_flux[0, age_idx, :])
        sed_hi = jnp.array(ssp_flux[n_met - 1, age_idx, :])

        l_v_lo = float(_mean_flux_in_band(sed_lo, wave, 5000.0, 5800.0))
        l_v_hi = float(_mean_flux_in_band(sed_hi, wave, 5000.0, 5800.0))

        # Higher metallicity → more absorption → lower L_V → higher M/L
        assert l_v_lo > l_v_hi, (
            f"Low-Z should be brighter in V: L_V(low)={l_v_lo:.4e}, L_V(high)={l_v_hi:.4e}"
        )


# ── 3. IGM transmission published values ──────────────────────────


class TestIGMTransmissionPhysics:
    """IGM transmission should match Inoue+2014 physics."""

    def test_no_igm_below_lyman_limit_at_z0(self):
        """At z=0, no IGM absorption (T=1 everywhere)."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.linspace(800.0, 10000.0, 500)
        t_igm = np.asarray(igm_transmission(wave_obs, 0.0))
        np.testing.assert_allclose(t_igm, 1.0, atol=1e-10, err_msg="IGM at z=0 should be unity")

    def test_lyman_limit_absorption_at_z3(self):
        """T at obs 3000 Å (rest 750 Å) at z=3 matches Inoue+2014 predictions.

        Issue #1992 adjudication: observed 3000 Å at z=3 corresponds to rest
        750 Å, which is only ~160 Å below the Lyman limit. The absorbing path
        Δz ≈ 0.7 carries ~1.5 mean free paths of tau, not the asymptotic
        tau_LL >> 1 regime. Inoue+2014 predicts T ≈ 0.172 here
        (tau_LAF^LC ≈ 0.77 + tau_DLA^LC ≈ 0.76 + Lyman-series ≈ 0.25 → T ≈ e^-1.78).
        See Issue #1992 for the full adjudication.
        """
        from tengri.components.igm import igm_transmission

        # Lyman limit at z=3: 912 * (1+3) = 3648 A observed
        wave_obs = jnp.array([3000.0, 3648.0, 5000.0, 8000.0])
        t_igm = np.asarray(igm_transmission(wave_obs, 3.0))

        # Below Lyman limit: pin to measured Inoue+2014 value (±3% tolerance)
        np.testing.assert_allclose(
            t_igm[0],
            0.171848,
            rtol=0.03,
            err_msg="T_IGM at obs 3000 A (rest 750 A) at z=3 from Inoue+2014",
        )

        # Well above Lyman-alpha: T ~ 1
        assert t_igm[3] > 0.95, f"T_IGM at 8000A at z=3 = {t_igm[3]:.3f}"

    def test_igm_transmission_bounded(self):
        """IGM transmission must be in [0, 1] at all wavelengths."""
        from tengri.components.igm import igm_transmission

        for z in [0.5, 1.0, 2.0, 3.0, 5.0]:
            wave_obs = jnp.linspace(500.0, 20000.0, 1000)
            t_igm = np.asarray(igm_transmission(wave_obs, z))
            assert_non_negative(t_igm, name="t_igm", msg=f"Negative IGM at z={z}")
            assert np.all(t_igm <= 1.0 + 1e-10), f"IGM > 1 at z={z}"

    def test_igm_increases_with_redshift(self):
        """More IGM absorption at higher redshift (average T decreases)."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.linspace(1000.0, 5000.0, 500)
        mean_t = []
        for z in [1.0, 2.0, 3.0, 5.0]:
            t = np.asarray(igm_transmission(wave_obs, z))
            mean_t.append(np.mean(t))

        # Mean transmission should decrease with z
        for i in range(len(mean_t) - 1):
            assert mean_t[i] > mean_t[i + 1], (
                f"Mean T_IGM not decreasing: z step {i} → {i + 1}: "
                f"{mean_t[i]:.3f} → {mean_t[i + 1]:.3f}"
            )


# ── 4. Cosmology full suite (tighter than existing) ───────────────


class TestCosmologyFullSuite:
    """Comprehensive cosmology validation against astropy Planck18."""

    def test_dl_tighter_tolerance(self):
        """dL should match astropy to < 0.5% with increased n_quad."""
        astropy = pytest.importorskip("astropy")
        from astropy.cosmology import Planck18

        from tengri.utils.cosmology import luminosity_distance

        for z in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            dl_astropy = Planck18.luminosity_distance(z).to("cm").value
            dl_tengri = float(luminosity_distance(z))
            ratio = dl_tengri / dl_astropy
            np.testing.assert_allclose(
                ratio,
                1.0,
                atol=0.005,
                err_msg=f"dL at z={z}: ratio={ratio:.6f}",
            )

    def test_age_at_z_full_range(self):
        """Age of universe at z=0,0.5,1 should match astropy to <2%."""
        astropy = pytest.importorskip("astropy")
        from astropy.cosmology import Planck18

        from tengri.utils.cosmology import age_at_z

        for z in [0.0, 0.5, 1.0]:
            # tengri's age_at_z returns Gyr; converting astropy to yr made the
            # ratio 1e-9 and the test unpassable (#1728).
            age_ap = Planck18.age(z).to("Gyr").value
            age_tg = float(age_at_z(z))
            ratio = age_tg / age_ap
            np.testing.assert_allclose(
                ratio,
                1.0,
                atol=0.02,
                err_msg=f"Age at z={z}: ratio={ratio:.6f}",
            )


# ── 5. Physical quantity ranges over random parameter draws ───────


class TestRandomParameterPhysics:
    """Random parameter draws should all produce physical quantities."""

    def test_random_draws_physical(self, ssp_data):
        """10 random parameter sets should give physical derived quantities."""
        from tengri.observation.filters import load_filter_set

        filters = load_filter_set(["sdss_r"])
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            met_logzsol=Uniform(-1.5, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=-0.7,
            redshift=0.1,
        )
        model = SEDModel(spec, ssp_data, filters=filters)

        for i in range(10):
            key = jax.random.PRNGKey(i * 7 + 13)
            params = spec.sample(key)

            # SED must be finite and positive
            sed = model.predict_rest_sed(params).sed
            assert jnp.all(jnp.isfinite(sed)), f"NaN in SED for draw {i}"
            assert jnp.all(sed >= 0), f"Negative SED for draw {i}"

            # Derived quantities must be physical
            d = model.predict_derived(params)
            mass = float(d["stellar_mass"])
            assert mass > 0, f"Negative mass for draw {i}"
            assert mass < 1e15, f"Unreasonable mass {mass:.3e} for draw {i}"
            assert np.isfinite(float(d["sfr_100myr"])), f"Non-finite SFR for draw {i}"
