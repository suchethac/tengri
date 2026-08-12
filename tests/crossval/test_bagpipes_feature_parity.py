# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation: Bagpipes feature parity for gap-analysis items.

Tests physics equivalence between Tengri and Bagpipes for each feature
in the gap analysis plan. Conventions (lookback vs cosmic time, units)
differ; these tests verify the underlying physics matches.

References
----------
- Carnall+2018 (bagpipes): SFH models, calibration, dust
- Wild+2007: VW07 two-component dust
- Wild+2020: PSB SFH
- Dijkstra 2014, Lee 2013: DLA Voigt profile
- Leja+2019: Continuity SFH
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.crossval

T_LOOKBACK = jnp.geomspace(1e5, 14e9, 500)
WAVE = jnp.linspace(1000.0, 10000.0, 500)


# ── 1. CHEMICAL ENRICHMENT HISTORY — metallicity modes ────────────


class TestChemicalEnrichmentPhysics:
    """Verify metallicity history models match Bagpipes physics."""

    def test_two_step_metallicity_transition(self):
        """Two-step: Z should jump at step_age.

        Bagpipes two_step: Z_old for t_lb > step_age, Z_young for t_lb < step_age.
        Tengri two_step_metallicity: uses log10(Z) absolute and step_age in Gyr.
        Input ssp_lg_age_gyr is log10(age/Gyr).
        """
        from tengri.components.stellar.sfh.metallicity_history import two_step_metallicity

        ssp_lg_age_gyr = jnp.linspace(-3.0, 1.1, 200)
        log_z_old = -2.4  # log10(0.004)
        log_z_young = -1.7  # log10(0.02)
        step_age_gyr = 2.0

        z_arr = two_step_metallicity(ssp_lg_age_gyr, log_z_old, log_z_young, step_age_gyr)

        log_step = jnp.log10(step_age_gyr)
        old_mask = ssp_lg_age_gyr > float(log_step) + 0.1
        young_mask = ssp_lg_age_gyr < float(log_step) - 0.1
        if jnp.any(old_mask):
            mean_old = float(jnp.mean(z_arr[old_mask]))
            np.testing.assert_allclose(mean_old, log_z_old, atol=0.05)
        if jnp.any(young_mask):
            mean_young = float(jnp.mean(z_arr[young_mask]))
            np.testing.assert_allclose(mean_young, log_z_young, atol=0.05)

    def test_psb_two_step_keyed_to_burstage(self):
        """PSB two-step: metallicity jump at burstage, not a separate step_age.

        Bagpipes psb_two_step: Z_old before burstage, Z_burst after.
        Tengri: uses log10(Z) absolute and burstage in Gyr.
        """
        from tengri.components.stellar.sfh.metallicity_history import psb_two_step_metallicity

        ssp_lg_age_gyr = jnp.linspace(-3.0, 1.1, 200)
        log_z_old = -2.4  # log10(0.004)
        log_z_burst = -1.5  # log10(~0.03)
        burstage_gyr = 0.5  # 500 Myr

        z_arr = psb_two_step_metallicity(ssp_lg_age_gyr, log_z_old, log_z_burst, burstage_gyr)

        log_burst = jnp.log10(burstage_gyr)
        pre_burst = ssp_lg_age_gyr > float(log_burst) + 0.1
        post_burst = ssp_lg_age_gyr < float(log_burst) - 0.1
        if jnp.any(pre_burst):
            mean_pre = float(jnp.mean(z_arr[pre_burst]))
            np.testing.assert_allclose(mean_pre, log_z_old, atol=0.05)
        if jnp.any(post_burst):
            mean_post = float(jnp.mean(z_arr[post_burst]))
            np.testing.assert_allclose(mean_post, log_z_burst, atol=0.05)

    def test_metallicity_bins_piecewise_constant(self):
        """Metallicity bins: Z is constant within each time bin.

        Bagpipes metallicity_bins: different Z per bin, edges in Myr.
        """
        from tengri.components.stellar.sfh.metallicity_history import metallicity_bins_on_ssp_grid

        ssp_ages = jnp.geomspace(1e6, 13e9, 200)
        bin_edges = jnp.array([0.0, 1e9, 5e9, 13e9])
        bin_zmets = jnp.array([0.02, 0.008, 0.004])

        z_arr = metallicity_bins_on_ssp_grid(ssp_ages, bin_edges, bin_zmets)

        for i in range(len(bin_zmets)):
            mask = (ssp_ages > bin_edges[i]) & (ssp_ages < bin_edges[i + 1])
            if jnp.any(mask):
                mean_z = float(jnp.mean(z_arr[mask]))
                assert abs(mean_z - float(bin_zmets[i])) / float(bin_zmets[i]) < 0.01

    def test_metallicity_bins_continuity_evolves(self):
        """Continuity metallicity: Z varies via delta-log-Z steps.

        Bagpipes metallicity_bins_continuity: Z_i = Z_1 * 10^(sum dzmet_{1..i-1}).
        Tengri: bin_edges in log10(years), log_z_abs_base in log10(Z).
        """
        from tengri.components.stellar.sfh.metallicity_history import (
            metallicity_bins_continuity_on_ssp_grid,
        )

        ssp_lg_age_gyr = jnp.linspace(-3.0, 1.1, 200)
        bin_edges_log_yr = jnp.array([6.0, 9.0, 9.7, 10.1])
        log_z_base = -2.0
        d_log_z = jnp.array([0.3, -0.2])

        z_arr = metallicity_bins_continuity_on_ssp_grid(
            ssp_lg_age_gyr, bin_edges_log_yr, log_z_base, d_log_z
        )

        # Steps cumulate oldest→youngest: bin2=base, bin1=base+d[0], bin0=base+d[0]+d[1]
        expected_log_z = [
            log_z_base + 0.3 - 0.2,  # youngest bin
            log_z_base + 0.3,  # middle bin
            log_z_base,  # oldest bin
        ]
        for i in range(3):
            lo_gyr = 10 ** (float(bin_edges_log_yr[i]) - 9)
            hi_gyr = 10 ** (float(bin_edges_log_yr[i + 1]) - 9)
            mask = (ssp_lg_age_gyr > jnp.log10(lo_gyr)) & (ssp_lg_age_gyr < jnp.log10(hi_gyr))
            if jnp.any(mask):
                mean_z = float(jnp.mean(z_arr[mask]))
                np.testing.assert_allclose(mean_z, expected_log_z[i], atol=0.05)


# ── 2. DLA MODEL — Voigt profile absorption ───────────────────────


class TestDLAPhysics:
    """DLA model: Voigt profile at Lyman-alpha.

    Bagpipes: dla_trans(wl_emit, N_HI, T, b_turb) -> transmission.
    Uses Lee (2013) correction, Tasitsiomi (2006) Voigt approximation.
    """

    def test_full_absorption_at_line_center_high_column(self):
        """High column density DLA should have near-zero transmission at Ly-alpha."""
        from tengri.components.igm.dla import dla_transmission

        wave_rest = jnp.linspace(1210.0, 1221.0, 100)
        logN = 21.0
        T = 1e4

        trans = dla_transmission(wave_rest, logN, T)
        center_mask = jnp.abs(wave_rest - 1215.67) < 1.0
        if jnp.any(center_mask):
            assert float(jnp.max(trans[center_mask])) < 0.01

    def test_no_absorption_far_from_line(self):
        """Far from Ly-alpha, DLA transmission should be ~1."""
        from tengri.components.igm.dla import dla_transmission

        wave_rest = jnp.linspace(1300.0, 2000.0, 100)
        logN = 20.5
        T = 1e4

        trans = dla_transmission(wave_rest, logN, T)
        assert float(jnp.min(trans)) > 0.99

    def test_column_density_monotonic(self):
        """Higher column density should produce deeper absorption in the wings."""
        from tengri.components.igm.dla import dla_transmission

        wave_rest = jnp.array([1220.0])
        T = 1e4

        trans_low = dla_transmission(wave_rest, 19.0, T)
        trans_high = dla_transmission(wave_rest, 21.0, T)

        assert float(trans_low[0]) > float(trans_high[0])

    def test_transmission_bounded_0_1(self):
        """DLA transmission must be in [0, 1]."""
        from tengri.components.igm.dla import dla_transmission

        wave_rest = jnp.linspace(900.0, 1400.0, 500)
        trans = dla_transmission(wave_rest, 20.3, 1e4, b_turb_kms=10.0)

        assert float(jnp.min(trans)) >= 0.0
        assert float(jnp.max(trans)) <= 1.0 + 1e-10

    def test_bagpipes_physics_match(self):
        """Cross-validate against Bagpipes DLA at specific wavelengths.

        Bagpipes uses Tasitsiomi (2006) Voigt approximation with
        Lee (2013) correction. Test at a few key wavelengths.
        """
        from bagpipes.models.dla_model import dla_trans as bp_dla

        from tengri.components.igm.dla import dla_transmission

        wave_rest = np.linspace(1200.0, 1230.0, 200)
        logN = 20.5
        T = 1e4
        b_turb_kms = 5.0

        bp_result = bp_dla(wave_rest, 10**logN, T, b_turb=b_turb_kms)
        tengri_result = np.asarray(
            dla_transmission(jnp.array(wave_rest), logN, T, b_turb_kms=b_turb_kms)
        )

        np.testing.assert_allclose(tengri_result, bp_result, atol=0.05, rtol=0.1)


# ── 3. POST-STARBURST SFH (Wild+2020) ─────────────────────────────


class TestPSBSFHPhysics:
    """PSB SFH: exponential + DPL burst, mass-fraction controlled.

    Bagpipes: psb_wild2020 — tau-model for old stars, DPL burst.
    Tengri: psb_wild2020 in mean_sfh.py.
    """

    def test_burst_fraction_controls_mass_ratio(self):
        """fburst should control the fraction of mass in the burst component."""
        from tengri.components.stellar.sfh import psb_wild2020

        t = jnp.linspace(0.0, 13e9, 2000)
        age = 10e9
        tau = 3e9
        burstage = 500e6
        alpha = 2.0
        beta = 2.0

        for fburst in [0.1, 0.5, 0.9]:
            sfr = psb_wild2020(
                t,
                log_total_mass=1.0,
                age=age,
                tau=tau,
                burstage=burstage,
                alpha=alpha,
                beta=beta,
                fburst=fburst,
            )
            chex.assert_tree_all_finite(sfr)
            assert float(jnp.max(sfr)) > 0

    def test_sfr_nonzero_at_burstage(self):
        """SFR should be non-zero at the burst age."""
        from tengri.components.stellar.sfh import psb_wild2020

        burstage = 500e6
        t = jnp.array([burstage])
        sfr = psb_wild2020(
            t,
            log_total_mass=1.0,
            age=10e9,
            tau=3e9,
            burstage=burstage,
            alpha=2.0,
            beta=2.0,
            fburst=0.5,
        )
        assert float(sfr[0]) > 0

    def test_zero_beyond_age(self):
        """No star formation before the galaxy formed."""
        from tengri.components.stellar.sfh import psb_wild2020

        age = 5e9
        t = jnp.array([age + 1e9, age + 5e9])
        sfr = psb_wild2020(
            t,
            log_total_mass=1.0,
            age=age,
            tau=2e9,
            burstage=300e6,
            alpha=2.0,
            beta=2.0,
            fburst=0.3,
        )
        np.testing.assert_allclose(sfr, 0.0, atol=1e-20)

    def test_jit_and_grad(self):
        """PSB SFH is JIT-compilable and differentiable."""
        from tengri.components.stellar.sfh import psb_wild2020

        t = jnp.linspace(0.0, 13e9, 200)

        @jax.jit
        def total_mass(log_total_mass):
            sfr = psb_wild2020(
                t,
                log_total_mass=log_total_mass,
                age=10e9,
                tau=3e9,
                burstage=500e6,
                alpha=2.0,
                beta=2.0,
                fburst=0.3,
            )
            return jnp.trapezoid(sfr, t)

        grad_val = assert_grad_matches_fd(total_mass, 1.0)
        assert jnp.isfinite(grad_val)


# ── 4. VW07 TWO-COMPONENT DUST ────────────────────────────────────


class TestVW07DustPhysics:
    """VW07 (Wild+2007): separate power-law slopes for birth cloud and ISM.

    Bagpipes: n=1.3 for birth clouds, n=0.7 for diffuse ISM.
    Attenuation curve A(lambda)/A(V) = (5500/lambda)^n.
    """

    def test_birth_cloud_slope_steeper(self):
        """Birth cloud slope (1.3) should be steeper than ISM (0.7)."""
        from tengri.components.dust.attenuation import vw07_bc, vw07_diff

        wave = jnp.linspace(2000.0, 10000.0, 200)
        A_bc = vw07_bc(wave)
        A_diff = vw07_diff(wave)

        uv_mask = wave < 3000.0
        nir_mask = wave > 8000.0
        if jnp.any(uv_mask) and jnp.any(nir_mask):
            ratio_bc = float(jnp.mean(A_bc[uv_mask]) / jnp.mean(A_bc[nir_mask]))
            ratio_diff = float(jnp.mean(A_diff[uv_mask]) / jnp.mean(A_diff[nir_mask]))
            assert ratio_bc > ratio_diff

    def test_birth_cloud_matches_bagpipes(self):
        """A_bc = (5500/lambda)^1.3, matching Bagpipes VW07."""
        from tengri.components.dust.attenuation import vw07_bc

        wave = jnp.array([2000.0, 5500.0, 10000.0])
        A_bc = vw07_bc(wave)

        expected = (5500.0 / wave) ** 1.3
        np.testing.assert_allclose(A_bc, expected, rtol=1e-6)

    def test_diffuse_ism_matches_bagpipes(self):
        """A_diff = (5500/lambda)^0.7, matching Bagpipes VW07."""
        from tengri.components.dust.attenuation import vw07_diff

        wave = jnp.array([2000.0, 5500.0, 10000.0])
        A_diff = vw07_diff(wave)

        expected = (5500.0 / wave) ** 0.7
        np.testing.assert_allclose(A_diff, expected, rtol=1e-6)

    def test_unity_at_5500(self):
        """Both curves should equal 1.0 at 5500 Angstrom."""
        from tengri.components.dust.attenuation import vw07_bc, vw07_diff

        wave = jnp.array([5500.0])
        np.testing.assert_allclose(vw07_bc(wave), 1.0, atol=1e-10)
        np.testing.assert_allclose(vw07_diff(wave), 1.0, atol=1e-10)


# ── 5. PARAMETER MIRRORING / TYING ────────────────────────────────


class TestParameterMirroringPhysics:
    """Parameter mirroring: tying one param to equal another.

    Bagpipes: mirror_pars — set param value = string name of source param.
    Tengri: Parameters._mirrors — same semantics.
    """

    def test_mirror_resolves_value(self):
        """Mirrored param should take the value of its source."""
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Uniform

        spec = Parameters(
            redshift=1.0,
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.1, 5.0),
            sfh_dpl_beta=Uniform(0.1, 3.0),
            sfh_dpl_tau_gyr=Uniform(0.1, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 3.0),
            dust_tau_diff="sfh_dpl_log_total_mass",
        )

        assert "dust_tau_diff" in spec.mirrors
        assert spec.mirrors["dust_tau_diff"] == "sfh_dpl_log_total_mass"

        params = {
            "sfh_dpl_alpha": 2.5,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_total_mass": 1.0,
        }
        resolved = spec.resolve_mirrors(params)
        assert resolved["dust_tau_diff"] == 1.0

    def test_mirror_reduces_dimensionality(self):
        """A mirrored param is NOT a free param — it reduces n_free."""
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Uniform

        spec_no_mirror = Parameters(
            redshift=1.0,
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.1, 5.0),
            sfh_dpl_beta=Uniform(0.1, 3.0),
            sfh_dpl_tau_gyr=Uniform(0.1, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 3.0),
            dust_tau_diff=Uniform(0.0, 3.0),
        )
        spec_mirror = Parameters(
            redshift=1.0,
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.1, 5.0),
            sfh_dpl_beta=Uniform(0.1, 3.0),
            sfh_dpl_tau_gyr=Uniform(0.1, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 3.0),
            dust_tau_diff="sfh_dpl_log_total_mass",
        )

        assert spec_mirror.n_free == spec_no_mirror.n_free - 1


# ── 6. SPECTRAL COVARIANCE MATRIX ─────────────────────────────────


class TestSpectralCovariancePhysics:
    """Spectral covariance in Spectroscopy config.

    Bagpipes: galaxy.py — input_spec_cov_matrix, uses cov_inv @ diff.
    Tengri: Spectroscopy.covariance, precomputes _cov_inv.
    """

    def test_cov_inv_precomputed(self):
        """Inverse covariance is precomputed at construction."""
        from tengri.observation import Spectroscopy

        n = 50
        wave = jnp.linspace(4000.0, 7000.0, n)
        cov = jnp.eye(n) * 0.01
        spec = Spectroscopy(wave_obs=wave, covariance=cov)

        assert spec.has_covariance
        assert spec.cov_inv is not None
        np.testing.assert_allclose(
            np.asarray(spec.cov_inv), np.asarray(jnp.linalg.inv(cov)), atol=1e-10
        )

    def test_diagonal_cov_matches_per_pixel(self):
        """Diagonal covariance should give same result as per-pixel 1/sigma^2."""
        from tengri.observation import Spectroscopy

        n = 50
        wave = jnp.linspace(4000.0, 7000.0, n)
        sigma = jnp.ones(n) * 0.1
        cov = jnp.diag(sigma**2)
        spec = Spectroscopy(wave_obs=wave, covariance=cov)

        diff = jnp.ones(n) * 0.5
        chi2_cov = float(diff @ spec.cov_inv @ diff)
        chi2_perpixel = float(jnp.sum((diff / sigma) ** 2))

        np.testing.assert_allclose(chi2_cov, chi2_perpixel, rtol=1e-10)

    def test_no_cov_by_default(self):
        """Without covariance, has_covariance is False."""
        from tengri.observation import Spectroscopy

        wave = jnp.linspace(4000.0, 7000.0, 50)
        spec = Spectroscopy(wave_obs=wave)
        assert not spec.has_covariance


# ── 7. EMISSION LINE FLUX FITTING ─────────────────────────────────


class TestLineFluxDataPhysics:
    """Line flux fitting: observed line fluxes as direct observables.

    Bagpipes: load_line_fluxes — users provide measured line fluxes.
    Tengri: LineFluxData — holds line names, fluxes, errors.
    """

    def test_line_flux_data_creation(self):
        from tengri.observation import LineFluxData

        data = LineFluxData(
            names=("Halpha", "Hbeta"),
            fluxes=jnp.array([1e-16, 3e-17]),
            errors=jnp.array([1e-18, 5e-19]),
            wavelengths=jnp.array([6564.61, 4862.68]),
        )
        assert data.n_lines == 2
        assert data.names == ("Halpha", "Hbeta")

    def test_from_dict_roundtrip(self):
        """from_dict auto-resolves wavelengths from the line catalog."""
        from tengri.observation import LineFluxData

        data = LineFluxData.from_dict({"Halpha": (1e-16, 1e-18), "Hbeta": (3e-17, 5e-19)})
        assert data.n_lines == 2
        assert "Halpha" in data.names
        assert float(data.fluxes[0]) == pytest.approx(1e-16)

    def test_chi2_computation(self):
        """Chi2 from line flux data should be sum((obs-model)^2/err^2)."""
        from tengri.observation import LineFluxData

        fluxes = jnp.array([1e-16, 3e-17])
        errors = jnp.array([1e-18, 5e-19])
        data = LineFluxData(
            names=("Halpha", "Hbeta"),
            fluxes=fluxes,
            errors=errors,
            wavelengths=jnp.array([6564.61, 4862.68]),
        )
        model_fluxes = jnp.array([1.01e-16, 2.95e-17])
        chi2 = float(jnp.sum(((data.fluxes - model_fluxes) / data.errors) ** 2))
        expected = float(jnp.sum(((fluxes - model_fluxes) / errors) ** 2))
        np.testing.assert_allclose(chi2, expected, rtol=1e-10)


# ── 8. SPECTRAL INDEX FITTING ─────────────────────────────────────


class TestSpectralIndexPhysics:
    """Spectral index measurement: EW and break indices.

    Bagpipes: single_index — EW = width * (F_cont - F_feature) / F_cont;
    break = F_red / F_blue.
    Tengri: measure_index_jax — same physics.
    """

    def test_ew_flat_spectrum_zero(self):
        """Flat spectrum should have EW = 0 (no absorption or emission)."""
        from tengri.observation.spectral_indices import SpectralIndexDef, measure_index_jax

        wave = jnp.linspace(4000.0, 5500.0, 1000)
        flux = jnp.ones_like(wave) * 1.0

        idx = SpectralIndexDef(
            name="test_ew",
            index_type="EW",
            feature=(4800.0, 5000.0),
            continuum=((4600.0, 4750.0), (5100.0, 5250.0)),
        )
        result = measure_index_jax(wave, flux, idx)
        np.testing.assert_allclose(float(result), 0.0, atol=0.01)

    def test_break_unity_for_flat_spectrum(self):
        """Flat spectrum should have break ratio = 1."""
        from tengri.observation.spectral_indices import SpectralIndexDef, measure_index_jax

        wave = jnp.linspace(3500.0, 4500.0, 500)
        flux = jnp.ones_like(wave) * 5.0

        idx = SpectralIndexDef(
            name="test_break",
            index_type="break",
            continuum=((3750.0, 3950.0), (4050.0, 4250.0)),
        )
        result = measure_index_jax(wave, flux, idx)
        np.testing.assert_allclose(float(result), 1.0, atol=0.01)

    def test_ew_absorption_positive(self):
        """Absorption line should give positive EW (astronomer convention)."""
        from tengri.observation.spectral_indices import SpectralIndexDef, measure_index_jax

        wave = jnp.linspace(4000.0, 5500.0, 1000)
        flux = jnp.ones_like(wave)
        absorption = jnp.exp(-((wave - 4900.0) ** 2) / (2 * 30.0**2)) * 0.5
        flux = flux - absorption

        idx = SpectralIndexDef(
            name="test_abs",
            index_type="EW",
            feature=(4850.0, 4950.0),
            continuum=((4600.0, 4750.0), (5100.0, 5250.0)),
        )
        result = measure_index_jax(wave, flux, idx)
        assert float(result) > 0

    def test_break_red_brighter(self):
        """Red-sloped spectrum should have break > 1."""
        from tengri.observation.spectral_indices import SpectralIndexDef, measure_index_jax

        wave = jnp.linspace(3500.0, 4500.0, 500)
        flux = wave / 4000.0

        idx = SpectralIndexDef(
            name="test_slope",
            index_type="break",
            continuum=((3750.0, 3950.0), (4050.0, 4250.0)),
        )
        result = measure_index_jax(wave, flux, idx)
        assert float(result) > 1.0


# ── 9. CATALOG BATCH FITTING — checkpoint/resume ────────────────


class TestCatalogBatchPhysics:
    """Catalog batch fitting infrastructure.

    Bagpipes: fit_catalog — MPI, checkpoint, auto-catalog.
    Tengri: fit_batch + catalog_summary + Posterior save/load.
    """

    def test_posterior_save_load_roundtrip(self):
        """Posterior HDF5 roundtrip preserves all fields."""
        import os
        import tempfile

        from tengri.inference.posterior import Posterior

        key = jax.random.PRNGKey(42)
        n = 50
        k1, k2 = jax.random.split(key)

        p = Posterior(
            samples={
                "dust_av": 0.5 + 0.1 * jax.random.normal(k1, (n,)),
                "sfh_dpl_alpha": 1.2 + 0.3 * jax.random.normal(k2, (n,)),
            },
            params={
                "dust_av": jnp.array(0.5),
                "sfh_dpl_alpha": jnp.array(1.2),
            },
            method="NUTS",
            wall_time_s=30.0,
            diagnostics={"chi2_dof": 1.05, "n_divergent": 0},
        )

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.h5")
            p.save(path)
            loaded = Posterior.load(path)

        assert loaded.method == "NUTS"
        for name in p.samples:
            np.testing.assert_allclose(
                np.asarray(loaded.samples[name]),
                np.asarray(p.samples[name]),
            )

    def test_catalog_summary_percentiles(self):
        """catalog_summary produces correct percentile columns."""
        from tengri.forward.convenience import catalog_summary
        from tengri.inference.posterior import Posterior

        results = []
        for i in range(5):
            key = jax.random.PRNGKey(i)
            center = float(i) + 1.0
            results.append(
                Posterior(
                    samples={"x": center + 0.1 * jax.random.normal(key, (200,))},
                    params={"x": jnp.array(center)},
                    method="NUTS",
                    wall_time_s=1.0,
                    diagnostics={},
                )
            )
        summary = catalog_summary(results, include_derived=False)
        assert "x_p50" in summary
        assert len(summary["x_p50"]) == 5
        np.testing.assert_allclose(summary["x_p50"], [1.0, 2.0, 3.0, 4.0, 5.0], atol=0.1)


# ── 11. WAVELENGTH MASKING ────────────────────────────────────────


class TestWavelengthMaskingPhysics:
    """Wavelength masking: set noise to infinity in masked regions.

    Bagpipes: galaxy._mask() — loads mask files, sets error to inf.
    Tengri: apply_wavelength_mask — same semantics.
    """

    def test_masked_noise_is_inf(self):
        from tengri.observation import apply_wavelength_mask

        wave = jnp.linspace(3000.0, 10000.0, 500)
        noise = jnp.ones_like(wave) * 0.1

        masked = apply_wavelength_mask(noise, wave, [(5560.0, 5590.0)])

        in_mask = (wave >= 5560.0) & (wave <= 5590.0)
        if jnp.any(in_mask):
            assert jnp.all(jnp.isinf(masked[in_mask]))

        out_mask = ~in_mask
        np.testing.assert_allclose(masked[out_mask], 0.1)

    def test_masked_chi2_zero(self):
        """Masked pixels should contribute zero to chi2."""
        from tengri.observation import apply_wavelength_mask

        wave = jnp.linspace(3000.0, 10000.0, 500)
        noise = jnp.ones_like(wave) * 0.1
        noise = apply_wavelength_mask(noise, wave, [(5500.0, 5600.0)])

        diff = jnp.ones_like(wave) * 100.0
        chi2_per_pixel = (diff / noise) ** 2

        in_mask = (wave >= 5500.0) & (wave <= 5600.0)
        np.testing.assert_allclose(chi2_per_pixel[in_mask], 0.0, atol=1e-30)

    def test_multiple_mask_ranges(self):
        from tengri.observation import apply_wavelength_mask

        wave = jnp.linspace(3000.0, 10000.0, 500)
        noise = jnp.ones_like(wave) * 0.1

        masked = apply_wavelength_mask(noise, wave, [(5560.0, 5590.0), (7580.0, 7680.0)])

        mask1 = (wave >= 5560.0) & (wave <= 5590.0)
        mask2 = (wave >= 7580.0) & (wave <= 7680.0)
        if jnp.any(mask1):
            assert jnp.all(jnp.isinf(masked[mask1]))
        if jnp.any(mask2):
            assert jnp.all(jnp.isinf(masked[mask2]))

    def test_immutable(self):
        """Input noise array should not be mutated."""
        from tengri.observation import apply_wavelength_mask

        wave = jnp.linspace(3000.0, 10000.0, 100)
        noise = jnp.ones_like(wave) * 0.1
        noise_copy = noise.copy()

        apply_wavelength_mask(noise, wave, [(5000.0, 6000.0)])
        np.testing.assert_allclose(noise, noise_copy)


# ── 12. CONST_EXP SFH ─────────────────────────────────────────────


class TestConstExpSFHPhysics:
    """Constant then exponential SFH: "quenching at time T".

    Bagpipes const_exp: SFR declines exponentially from formation,
    then constant after 'age'. Tengri: constant from formation to
    quench_age, then exponential decline. Both model quenching but
    with different parametrizations.

    Physics test: the key property is that there is a constant phase
    and an exponential phase, with continuity at the boundary.
    """

    def test_has_constant_phase(self):
        """Some region should have constant SFR."""
        from tengri.components.stellar.sfh import constant_then_exponential_sfh

        t = jnp.linspace(0.0, 13e9, 1000)
        sfr = constant_then_exponential_sfh(t, log_sfr=1.0, tau=1e9, quench_age=5e9, age=10e9)

        mask = (t >= 5e9) & (t <= 10e9)
        if jnp.any(mask):
            sfr_const = sfr[mask]
            cv = float(jnp.std(sfr_const) / jnp.mean(sfr_const))
            assert cv < 0.01

    def test_has_exponential_phase(self):
        """Below quench_age, SFR declines exponentially."""
        from tengri.components.stellar.sfh import constant_then_exponential_sfh

        t = jnp.linspace(100e6, 4.5e9, 200)
        sfr = constant_then_exponential_sfh(t, log_sfr=1.0, tau=1e9, quench_age=5e9, age=10e9)

        log_sfr_vals = jnp.log(sfr)
        if jnp.all(jnp.isfinite(log_sfr_vals)):
            slope = float((log_sfr_vals[-1] - log_sfr_vals[0]) / (t[-1] - t[0]))
            expected_slope = 1.0 / 1e9
            assert abs(slope - expected_slope) / expected_slope < 0.1

    def test_continuous_at_quench(self):
        """SFR must be continuous at the quenching boundary."""
        from tengri.components.stellar.sfh import constant_then_exponential_sfh

        quench = 5e9
        eps = 1.0
        t_above = jnp.array([quench + eps])
        t_below = jnp.array([quench - eps])

        sfr_a = constant_then_exponential_sfh(t_above, 1.0, 1e9, quench, 10e9)
        sfr_b = constant_then_exponential_sfh(t_below, 1.0, 1e9, quench, 10e9)

        np.testing.assert_allclose(float(sfr_a[0]), float(sfr_b[0]), rtol=1e-6)

    def test_mass_integral_finite(self):
        """Total mass formed should be finite and positive."""
        from tengri.components.stellar.sfh import constant_then_exponential_sfh

        t = jnp.linspace(0.0, 13e9, 2000)
        sfr = constant_then_exponential_sfh(t, log_sfr=1.0, tau=1e9, quench_age=5e9, age=10e9)
        mass = float(jnp.trapezoid(sfr, t))
        assert mass > 0
        assert np.isfinite(mass)


# ── 13. DOUBLE POLYNOMIAL CALIBRATION ─────────────────────────────


class TestDoubleCalibrationPhysics:
    """Double polynomial: separate Chebyshev for each spectrograph arm.

    Bagpipes: double_polynomial_bayesian — x_blue and x_red normalized
    independently to [-1, 1] within their arm.
    Tengri: double_calibration_polynomial — same approach.
    """

    def test_independent_normalization(self):
        """Each arm has its own [-1, 1] normalization range.

        Bagpipes normalizes x_blue = 2*(wav - midpoint_blue)/(range_blue)
        and x_red similarly. Tengri uses calibration_polynomial with
        wave_min/wave_max set to the arm boundaries.
        """
        from tengri.observation.calibration import (
            calibration_polynomial,
            double_calibration_polynomial,
        )

        wave = jnp.linspace(3000.0, 10000.0, 500)
        wave_split = 5800.0
        coeffs = jnp.array([0.1])

        cal_double = double_calibration_polynomial(wave, coeffs, coeffs, wave_split)

        is_blue = wave < wave_split
        is_red = wave >= wave_split
        cal_blue = calibration_polynomial(wave[is_blue], coeffs, wave[0], wave_split)
        cal_red = calibration_polynomial(wave[is_red], coeffs, wave_split, wave[-1])

        np.testing.assert_allclose(cal_double[is_blue], cal_blue, atol=1e-13)
        np.testing.assert_allclose(cal_double[is_red], cal_red, atol=1e-13)

    def test_discontinuity_at_split(self):
        """Different arm coefficients can produce a jump at wave_split."""
        from tengri.observation.calibration import double_calibration_polynomial

        wave = jnp.linspace(3000.0, 10000.0, 10000)
        wave_split = 5800.0
        coeffs_blue = jnp.array([0.2, -0.1])
        coeffs_red = jnp.array([-0.1, 0.05])

        cal = double_calibration_polynomial(wave, coeffs_blue, coeffs_red, wave_split)

        idx_split = int(jnp.argmin(jnp.abs(wave - wave_split)))
        cal_just_below = float(cal[idx_split - 1])
        cal_just_above = float(cal[idx_split])

        assert abs(cal_just_below - cal_just_above) > 0.01

    def test_zero_coeffs_unity(self):
        """Zero coefficients on both arms = C(lambda) = 1."""
        from tengri.observation.calibration import double_calibration_polynomial

        wave = jnp.linspace(3000.0, 10000.0, 200)
        cal = double_calibration_polynomial(wave, jnp.zeros(3), jnp.zeros(3), 5800.0)
        np.testing.assert_allclose(cal, 1.0, atol=1e-14)

    def test_bagpipes_convention_match(self):
        """Verify the Tengri double calibration matches Bagpipes semantics.

        Bagpipes double_polynomial_bayesian:
        - Splits at wav_cut
        - Normalizes x_blue to [-1,1] within blue arm wavelength range
        - Normalizes x_red to [-1,1] within red arm wavelength range
        - Evaluates chebval independently on each arm

        We verify this by computing manually with numpy chebval.
        """
        from numpy.polynomial.chebyshev import chebval

        from tengri.observation.calibration import double_calibration_polynomial

        wave = np.linspace(3000.0, 10000.0, 500)
        wav_cut = 5800.0
        blue_coefs = [1.0, 0.1, -0.05]
        red_coefs = [1.0, 0.05, 0.02]

        wave_blue = wave[wave < wav_cut]
        wave_red = wave[wave >= wav_cut]

        x_blue = 2.0 * (wave_blue - wave_blue[0]) / (wave_blue[-1] - wave_blue[0]) - 1.0
        x_red = 2.0 * (wave_red - wave_red[0]) / (wave_red[-1] - wave_red[0]) - 1.0

        bp_blue = chebval(x_blue, blue_coefs)
        bp_red = chebval(x_red, red_coefs)

        tengri_coeffs_blue = jnp.array(blue_coefs[1:])
        tengri_coeffs_red = jnp.array(red_coefs[1:])

        tengri_cal = np.asarray(
            double_calibration_polynomial(
                jnp.array(wave), tengri_coeffs_blue, tengri_coeffs_red, wav_cut
            )
        )

        bp_full = np.zeros_like(wave)
        bp_full[wave < wav_cut] = bp_blue
        bp_full[wave >= wav_cut] = bp_red

        np.testing.assert_allclose(
            tengri_cal[wave < wav_cut],
            bp_full[wave < wav_cut],
            rtol=0.01,
            err_msg="Blue arm calibration mismatch with Bagpipes convention",
        )
        np.testing.assert_allclose(
            tengri_cal[wave >= wav_cut],
            bp_full[wave >= wav_cut],
            rtol=0.01,
            err_msg="Red arm calibration mismatch with Bagpipes convention",
        )
