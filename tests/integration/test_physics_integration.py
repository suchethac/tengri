# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for physical consistency of tengri forward model.

These tests use full SEDModel predictions with real SSP data to verify that
the forward model produces physically consistent SEDs — checking that
old galaxies are red, dusty galaxies have strong IR, metallicity affects
colors correctly, etc.

References
----------
- Balogh et al. 1999, ApJ, 527, 54 — Dn4000 definition
- Conroy, Gunn & White 2009, ApJ, 699, 486 — M/L ratios
- Lehmer et al. 2010, ApJ, 724, 559 — XRB scaling
"""

from pathlib import Path

import jax
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel
from tengri.observation.filters import load_filter_set
from tengri.parameters.parameters import Parameters

# ── Skip if SSP data not available ────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(
    not _SSP_EXISTS,
    reason="SSP data file not found",
)


# ── Shared fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp_data():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def filters():
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


def _make_model(ssp_data, filters, **spec_kwargs):
    """Helper to create SEDModel with given Parameters overrides."""
    defaults = dict(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_total_mass=1.0,
        sfh_tsnorm_peak_lbt_gyr=5.0,
        sfh_tsnorm_width_gyr=2.0,
        sfh_tsnorm_skew=0.0,
        sfh_tsnorm_trunc=5.0,
        met_logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
        redshift=0.1,
    )
    defaults.update(spec_kwargs)
    spec = Parameters(**defaults)
    return SEDModel(spec, ssp_data, filters=filters), spec.sample(jax.random.PRNGKey(0))


# ── 1. Stellar mass consistency ───────────────────────────────────


class TestStellarMassConsistency:
    """Surviving mass must be less than formed mass, and in a physical range."""

    def test_surviving_less_than_formed(self, ssp_data, filters):
        """Surviving mass < formed mass (stellar evolution returns mass)."""
        model, params = _make_model(ssp_data, filters)
        d = model.predict_derived(params)
        mass_formed = float(d["stellar_mass"])
        mass_surv = d["stellar_mass_surviving"]

        # Unconditional: an absent surviving mass is the failure this test
        # exists to catch, not a reason to skip it. Guarded as
        # ``if mass_surv is not None:`` the whole claim vanished whenever the
        # SSP carried no mass-remaining grid -- the same shape that left the
        # #29 regression test asserting nothing (fixed in #2156).
        assert mass_surv is not None, (
            "stellar_mass_surviving was not published, so the surviving-vs-formed "
            "claim cannot be checked; the SSP grid must carry mass-remaining data"
        )
        assert float(mass_surv) < mass_formed, (
            f"Surviving mass ({float(mass_surv):.3e}) >= formed mass ({mass_formed:.3e})"
        )

    def test_return_fraction_physical(self, ssp_data, filters):
        """Surviving/formed mass ratio in [0.3, 0.9] for Chabrier IMF."""
        model, params = _make_model(ssp_data, filters)
        d = model.predict_derived(params)
        mass_formed = float(d["stellar_mass"])
        mass_surv = d["stellar_mass_surviving"]

        if mass_surv is not None:
            ratio = float(mass_surv) / mass_formed
            assert 0.2 < ratio < 0.95, f"Return fraction = {ratio:.3f}, expected 0.2-0.95"

    def test_doubling_sfr_doubles_mass(self, ssp_data, filters):
        """Doubling peak SFR should approximately double stellar mass."""
        model1, params1 = _make_model(ssp_data, filters, sfh_tsnorm_log_total_mass=1.0)
        model2, params2 = _make_model(ssp_data, filters, sfh_tsnorm_log_total_mass=1.301)

        d1 = model1.predict_derived(params1)
        d2 = model2.predict_derived(params2)
        ratio = float(d2["stellar_mass"]) / float(d1["stellar_mass"])
        np.testing.assert_allclose(ratio, 2.0, rtol=0.1, err_msg=f"Mass ratio = {ratio:.3f}")


# ── 2. Old vs young galaxy physics ────────────────────────────────


class TestOldVsYoungGalaxyPhysics:
    """Old galaxies must be redder, have higher Dn4000, higher M/L."""

    @pytest.fixture(scope="class")
    def old_galaxy(self, ssp_data, filters):
        return _make_model(
            ssp_data,
            filters,
            sfh_tsnorm_peak_lbt_gyr=10.0,
            sfh_tsnorm_width_gyr=1.0,
        )

    @pytest.fixture(scope="class")
    def young_galaxy(self, ssp_data, filters):
        return _make_model(
            ssp_data,
            filters,
            sfh_tsnorm_peak_lbt_gyr=0.5,
            sfh_tsnorm_width_gyr=0.3,
        )

    def test_dn4000_higher_for_old(self, old_galaxy, young_galaxy):
        """Old galaxy Dn4000 > young galaxy Dn4000."""
        model_old, params_old = old_galaxy
        model_young, params_young = young_galaxy

        dn_old = float(model_old.predict_sed_quantities(params_old).dn4000)
        dn_young = float(model_young.predict_sed_quantities(params_young).dn4000)

        assert dn_old > dn_young, f"Dn4000: old={dn_old:.3f}, young={dn_young:.3f}"

    def test_old_galaxy_dn4000_above_threshold(self, real_ssp_only, old_galaxy):
        """Old (10 Gyr peak) galaxy should have Dn4000 > 1.3."""
        model, params = old_galaxy
        dn = float(model.predict_sed_quantities(params).dn4000)
        assert dn > 1.3, f"Old galaxy Dn4000 = {dn:.3f}, expected > 1.3"

    def test_young_galaxy_bluer_uv_slope(self, real_ssp_only, old_galaxy, young_galaxy):
        """Young galaxy should have bluer UV slope beta."""
        model_old, params_old = old_galaxy
        model_young, params_young = young_galaxy

        beta_old = float(model_old.predict_sed_quantities(params_old).uv_slope_beta)
        beta_young = float(model_young.predict_sed_quantities(params_young).uv_slope_beta)

        assert beta_young < beta_old, f"UV slope: young={beta_young:.2f}, old={beta_old:.2f}"

    def test_mass_weighted_age_ordering(self, old_galaxy, young_galaxy):
        """Mass-weighted age of old galaxy > young galaxy."""
        model_old, params_old = old_galaxy
        model_young, params_young = young_galaxy

        age_old = float(model_old.predict_sfh_quantities(params_old).mass_weighted_age_gyr)
        age_young = float(model_young.predict_sfh_quantities(params_young).mass_weighted_age_gyr)

        assert age_old > age_young, f"MW age: old={age_old:.2f} Gyr, young={age_young:.2f} Gyr"

    def test_old_galaxy_redder_uv_color(self, old_galaxy, young_galaxy):
        """Old galaxy should have redder rest-frame UV color (higher U-V)."""
        model_old, params_old = old_galaxy
        model_young, params_young = young_galaxy

        uv_old = float(model_old.predict_sed_quantities(params_old).rest_uv_color)
        uv_young = float(model_young.predict_sed_quantities(params_young).rest_uv_color)

        assert uv_old > uv_young, f"UV color: old={uv_old:.2f}, young={uv_young:.2f}"


# ── 3. Dust effects on SED ────────────────────────────────────────


class TestDustEffectsOnSED:
    """Dust must redden the SED and suppress UV flux."""

    @pytest.fixture(scope="class")
    def dustfree(self, ssp_data, filters):
        return _make_model(
            ssp_data,
            filters,
            sfh_tsnorm_peak_lbt_gyr=2.0,
            dust_tau_bc=0.0,
            dust_tau_diff=0.0,
        )

    @pytest.fixture(scope="class")
    def dusty(self, ssp_data, filters):
        return _make_model(
            ssp_data,
            filters,
            sfh_tsnorm_peak_lbt_gyr=2.0,
            dust_tau_bc=1.0,
            dust_tau_diff=2.0,
        )

    def test_dust_suppresses_uv(self, dustfree, dusty):
        """Dust should suppress UV flux by >5x for tau_diff=2.0."""
        model_df, params_df = dustfree
        model_dy, params_dy = dusty

        fuv_clean = float(model_df.predict_sed_quantities(params_df).fuv_flux)
        fuv_dusty = float(model_dy.predict_sed_quantities(params_dy).fuv_flux)

        assert fuv_dusty < fuv_clean / 5.0, (
            f"FUV: clean={fuv_clean:.3e}, dusty={fuv_dusty:.3e}, "
            f"ratio={fuv_clean / fuv_dusty:.1f}x"
        )

    def test_dust_reddens_uv_slope(self, dustfree, dusty):
        """Dusty SED should have redder UV slope beta."""
        model_df, params_df = dustfree
        model_dy, params_dy = dusty

        beta_clean = float(model_df.predict_sed_quantities(params_df).uv_slope_beta)
        beta_dusty = float(model_dy.predict_sed_quantities(params_dy).uv_slope_beta)

        assert beta_dusty > beta_clean, f"Beta: clean={beta_clean:.2f}, dusty={beta_dusty:.2f}"

    def test_dust_reddens_rest_uv_color(self, dustfree, dusty):
        """Dusty SED should have redder U-V color."""
        model_df, params_df = dustfree
        model_dy, params_dy = dusty

        uv_clean = float(model_df.predict_sed_quantities(params_df).rest_uv_color)
        uv_dusty = float(model_dy.predict_sed_quantities(params_dy).rest_uv_color)

        assert uv_dusty > uv_clean, f"U-V color: clean={uv_clean:.2f}, dusty={uv_dusty:.2f}"

    def test_zero_dust_no_absorption(self, dustfree):
        """Zero dust should give zero absorbed luminosity."""
        model, params = dustfree
        sed_q = model.predict_sed_quantities(params)
        l_dust = float(sed_q.l_dust_absorbed)
        # Should be ~0 (within numerical precision of SED integration)
        assert abs(l_dust) < 1e-3 * float(sed_q.l_bol), (
            f"L_dust_absorbed = {l_dust:.3e} at zero dust"
        )


# ── 4. Metallicity effects on SED ─────────────────────────────────


class TestMetallicityEffectsOnSED:
    """Higher metallicity should produce redder SEDs."""

    @pytest.fixture(scope="class")
    def low_z(self, ssp_data, filters):
        return _make_model(ssp_data, filters, met_logzsol=-1.0)

    @pytest.fixture(scope="class")
    def solar_z(self, ssp_data, filters):
        return _make_model(ssp_data, filters, met_logzsol=0.0)

    @pytest.fixture(scope="class")
    def high_z(self, ssp_data, filters):
        return _make_model(ssp_data, filters, met_logzsol=0.2)

    def test_dn4000_increases_with_metallicity(self, real_ssp_only, low_z, solar_z, high_z):
        """Higher metallicity → higher Dn4000 (more metal absorption)."""
        model_lo, params_lo = low_z
        model_sol, params_sol = solar_z
        model_hi, params_hi = high_z

        dn_lo = float(model_lo.predict_sed_quantities(params_lo).dn4000)
        dn_sol = float(model_sol.predict_sed_quantities(params_sol).dn4000)
        dn_hi = float(model_hi.predict_sed_quantities(params_hi).dn4000)

        assert dn_lo < dn_sol < dn_hi, (
            f"Dn4000 not increasing with Z: low={dn_lo:.3f}, solar={dn_sol:.3f}, high={dn_hi:.3f}"
        )

    def test_higher_metallicity_redder_uv_color(self, real_ssp_only, low_z, high_z):
        """Higher metallicity → redder U-V color (more line blanketing)."""
        model_lo, params_lo = low_z
        model_hi, params_hi = high_z

        uv_lo = float(model_lo.predict_sed_quantities(params_lo).rest_uv_color)
        uv_hi = float(model_hi.predict_sed_quantities(params_hi).rest_uv_color)

        assert uv_hi > uv_lo, f"U-V color: low Z={uv_lo:.2f}, high Z={uv_hi:.2f}"


# ── 5. Passive galaxy properties ──────────────────────────────────


class TestPassiveGalaxyProperties:
    """Passive (quenched) galaxies must have old-population diagnostics."""

    @pytest.fixture(scope="class")
    def passive_galaxy(self, ssp_data, filters):
        """Old, narrow SFH with peak at 10 Gyr lookback."""
        return _make_model(
            ssp_data,
            filters,
            sfh_tsnorm_peak_lbt_gyr=10.0,
            sfh_tsnorm_width_gyr=0.5,
            sfh_tsnorm_skew=0.0,
            sfh_tsnorm_trunc=3.0,
            sfh_tsnorm_log_total_mass=1.0,
        )

    def test_low_ssfr(self, passive_galaxy):
        """Passive galaxy should have sSFR < 1e-10 yr^-1."""
        model, params = passive_galaxy
        sfh_q = model.predict_sfh_quantities(params)
        ssfr = float(sfh_q.ssfr)
        assert ssfr < 1e-10, f"Passive galaxy sSFR = {ssfr:.3e}, expected < 1e-10"

    def test_high_dn4000(self, real_ssp_only, passive_galaxy):
        """Passive galaxy should have Dn4000 > 1.3."""
        model, params = passive_galaxy
        dn = float(model.predict_sed_quantities(params).dn4000)
        assert dn > 1.3, f"Passive galaxy Dn4000 = {dn:.3f}, expected > 1.3"

    def test_high_mass_weighted_age(self, passive_galaxy):
        """Passive galaxy mass-weighted age > 5 Gyr."""
        model, params = passive_galaxy
        age = float(model.predict_sfh_quantities(params).mass_weighted_age_gyr)
        assert age > 5.0, f"Passive galaxy MW age = {age:.2f} Gyr, expected > 5"


# ── 6. Photometric color ordering ─────────────────────────────────


class TestPhotometricColorOrdering:
    """SDSS magnitudes must follow physical color ordering."""

    def test_old_population_red(self, real_ssp_only, ssp_data, filters):
        """Old stellar population: u > g > r (fainter at blue, in mags)."""
        model, params = _make_model(
            ssp_data,
            filters,
            sfh_tsnorm_peak_lbt_gyr=10.0,
            sfh_tsnorm_width_gyr=0.5,
        )
        mags = np.asarray(model.predict_magnitudes(params))
        # u, g, r, i, z bands: old population should be u > g > r
        assert mags[0] > mags[1], f"u ({mags[0]:.2f}) should be fainter than g ({mags[1]:.2f})"
        assert mags[1] > mags[2], f"g ({mags[1]:.2f}) should be fainter than r ({mags[2]:.2f})"

    def test_dust_reddens_photometry(self, ssp_data, filters):
        """Dust should increase u-r color (make galaxy redder)."""
        model_clean, params_clean = _make_model(
            ssp_data,
            filters,
            sfh_tsnorm_peak_lbt_gyr=2.0,
            dust_tau_diff=0.0,
        )
        model_dusty, params_dusty = _make_model(
            ssp_data,
            filters,
            sfh_tsnorm_peak_lbt_gyr=2.0,
            dust_tau_diff=2.0,
        )

        mags_clean = np.asarray(model_clean.predict_magnitudes(params_clean))
        mags_dusty = np.asarray(model_dusty.predict_magnitudes(params_dusty))

        ur_clean = mags_clean[0] - mags_clean[2]  # u - r
        ur_dusty = mags_dusty[0] - mags_dusty[2]

        assert ur_dusty > ur_clean, f"u-r: clean={ur_clean:.2f}, dusty={ur_dusty:.2f}"
