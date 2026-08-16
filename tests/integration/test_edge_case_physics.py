# SPDX-License-Identifier: BSD-3-Clause
"""Edge case physics tests for extreme parameter values.

These tests verify that the forward model remains physically sensible
at the boundaries of parameter space — extreme dust, zero dust, extreme
redshift, high burstiness, and metallicity extremes.
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel
from tengri.observation.filters import load_filter_set
from tengri.parameters.parameters import Parameters
from tests._bounds import assert_non_negative

# ── Skip if SSP data not available ────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(
    not _SSP_EXISTS,
    reason="SSP data file not found",
)


@pytest.fixture(scope="module")
def ssp_data():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def filters():
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


def _make_model(ssp_data, filters, **spec_kwargs):
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


# ── 1. Extreme dust ───────────────────────────────────────────────


class TestExtremeDust:
    """A_V ~ 10 should nearly extinguish UV but leave NIR mostly intact."""

    def test_extreme_dust_uv_suppression(self, ssp_data, filters):
        """UV flux at tau_diff=5 should be < 1% of dust-free."""
        model_clean, params_clean = _make_model(ssp_data, filters)
        model_dusty, params_dusty = _make_model(
            ssp_data,
            filters,
            dust_tau_diff=5.0,
            dust_tau_bc=3.0,
        )

        fuv_clean = float(model_clean.predict_sed_quantities(params_clean).fuv_flux)
        fuv_dusty = float(model_dusty.predict_sed_quantities(params_dusty).fuv_flux)

        assert fuv_dusty < fuv_clean * 0.01, (
            f"Extreme dust UV suppression: {fuv_dusty / fuv_clean:.4f}, expected < 0.01"
        )

    def test_extreme_dust_sed_finite(self, ssp_data, filters):
        """SED should remain finite and positive even with extreme dust."""
        model, params = _make_model(
            ssp_data,
            filters,
            dust_tau_diff=5.0,
            dust_tau_bc=3.0,
        )
        sed = model.predict_rest_sed(params).sed
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed", msg="SED has negative values at extreme dust")

    def test_extreme_dust_photometry_physical(self, ssp_data, filters):
        """Photometry should still be physical (positive, finite)."""
        model, params = _make_model(
            ssp_data,
            filters,
            dust_tau_diff=5.0,
            dust_tau_bc=3.0,
        )
        phot = model.predict_photometry(params)
        chex.assert_tree_all_finite(phot)
        assert jnp.all(phot > 0)


# ── 2. Zero dust ──────────────────────────────────────────────────


class TestZeroDust:
    """Zero optical depth should give zero attenuation."""

    def test_zero_dust_uv_matches_intrinsic(self, ssp_data, filters):
        """With zero dust, attenuated and intrinsic UV flux should match."""
        model, params = _make_model(
            ssp_data,
            filters,
            dust_tau_bc=0.0,
            dust_tau_diff=0.0,
        )
        sed_q = model.predict_sed_quantities(params)

        # Intrinsic FUV should equal observed FUV at zero dust
        fuv = float(sed_q.fuv_flux)
        fuv_intr = float(sed_q.fuv_flux_intrinsic)

        if np.isfinite(fuv_intr):
            np.testing.assert_allclose(
                fuv, fuv_intr, rtol=1e-6, err_msg="FUV mismatch at zero dust"
            )


# ── 3. Extreme redshift ───────────────────────────────────────────


class TestExtremeRedshift:
    """SED should be physical at very low and moderately high redshift."""

    def test_very_low_redshift(self, ssp_data, filters):
        """z = 0.001: SED finite, positive, physical photometry."""
        model, params = _make_model(ssp_data, filters, redshift=0.001)
        sed = model.predict_rest_sed(params).sed
        chex.assert_tree_all_finite(sed)
        assert jnp.all(sed >= 0)

        phot = model.predict_photometry(params)
        chex.assert_tree_all_finite(phot)
        assert jnp.all(phot > 0)

    def test_high_redshift(self, ssp_data, filters):
        """z = 3.0: SED finite, positive."""
        model, params = _make_model(ssp_data, filters, redshift=3.0)
        sed = model.predict_rest_sed(params).sed
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed")

    def test_higher_z_fainter(self, ssp_data, filters):
        """Higher redshift galaxy should be fainter (more distant)."""
        model_lo, params_lo = _make_model(ssp_data, filters, redshift=0.1)
        model_hi, params_hi = _make_model(ssp_data, filters, redshift=1.0)

        phot_lo = np.asarray(model_lo.predict_photometry(params_lo))
        phot_hi = np.asarray(model_hi.predict_photometry(params_hi))

        # r-band flux should be much lower at z=1 than z=0.1
        assert phot_hi[2] < phot_lo[2], (
            f"r-band: z=0.1 flux={phot_lo[2]:.3e}, z=1 flux={phot_hi[2]:.3e}"
        )


# ── 4. High burstiness (stochastic SFH) ───────────────────────────


class TestHighBurstiness:
    """Extreme GP burstiness should still produce physical SEDs."""

    def test_extreme_psd_sigma_finite(self, ssp_data, filters):
        """psd_sigma=5.0 (very bursty): SED should be finite and positive."""
        spec = Parameters(
            mean_sfh_type=["tsnorm", "field"],
            sfh_tsnorm_log_total_mass=1.0,
            sfh_tsnorm_peak_lbt_gyr=5.0,
            sfh_tsnorm_width_gyr=2.0,
            sfh_tsnorm_skew=0.0,
            sfh_tsnorm_trunc=5.0,
            sfh_field_psd_sigma=5.0,
            sfh_field_psd_tau_myr=10.0,
            met_logzsol=0.0,
            dust_tau_bc=0.0,
            dust_tau_diff=0.0,
            dust_slope=-0.7,
            redshift=0.1,
            n_grid=64,
        )
        model = SEDModel(spec, ssp_data, filters=filters)
        params = spec.sample(jax.random.PRNGKey(42))

        sed = model.predict_rest_sed(params).sed
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed", msg="Negative SED values at psd_sigma=5")

    def test_moderate_burstiness_physical(self, ssp_data, filters):
        """psd_sigma=2.0 (moderately bursty): derived quantities physical."""
        spec = Parameters(
            mean_sfh_type=["tsnorm", "field"],
            sfh_tsnorm_log_total_mass=1.0,
            sfh_tsnorm_peak_lbt_gyr=5.0,
            sfh_tsnorm_width_gyr=2.0,
            sfh_tsnorm_skew=0.0,
            sfh_tsnorm_trunc=5.0,
            sfh_field_psd_sigma=2.0,
            sfh_field_psd_tau_myr=50.0,
            met_logzsol=0.0,
            dust_tau_bc=0.0,
            dust_tau_diff=0.0,
            dust_slope=-0.7,
            redshift=0.1,
            n_grid=64,
        )
        model = SEDModel(spec, ssp_data, filters=filters)
        params = spec.sample(jax.random.PRNGKey(42))

        d = model.predict_derived(params)
        assert float(d["stellar_mass"]) > 0, "Negative stellar mass"
        assert np.isfinite(float(d["sfr_100myr"])), "Non-finite SFR"


# ── 5. Metallicity extremes ───────────────────────────────────────


class TestMetallicityExtremes:
    """SED at metallicity grid boundaries should be physical."""

    def test_low_metallicity_bluer(self, real_ssp_only, ssp_data, filters):
        """Sub-solar metallicity should produce bluer SED than solar."""
        model_lo, params_lo = _make_model(ssp_data, filters, met_logzsol=-1.5)
        model_sol, params_sol = _make_model(ssp_data, filters, met_logzsol=0.0)

        uv_lo = float(model_lo.predict_sed_quantities(params_lo).rest_uv_color)
        uv_sol = float(model_sol.predict_sed_quantities(params_sol).rest_uv_color)

        assert uv_lo < uv_sol, f"Low-Z U-V={uv_lo:.2f} should be bluer than solar U-V={uv_sol:.2f}"

    def test_high_metallicity_redder(self, real_ssp_only, ssp_data, filters):
        """Super-solar metallicity should produce redder SED than solar."""
        model_sol, params_sol = _make_model(ssp_data, filters, met_logzsol=0.0)
        model_hi, params_hi = _make_model(ssp_data, filters, met_logzsol=0.2)

        uv_sol = float(model_sol.predict_sed_quantities(params_sol).rest_uv_color)
        uv_hi = float(model_hi.predict_sed_quantities(params_hi).rest_uv_color)

        assert uv_hi > uv_sol, (
            f"High-Z U-V={uv_hi:.2f} should be redder than solar U-V={uv_sol:.2f}"
        )

    def test_extreme_low_z_finite(self, ssp_data, filters):
        """Lowest metallicity: SED finite and positive."""
        model, params = _make_model(ssp_data, filters, met_logzsol=-1.5)
        sed = model.predict_rest_sed(params).sed
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed")

    def test_extreme_high_z_finite(self, ssp_data, filters):
        """Highest metallicity: SED finite and positive."""
        model, params = _make_model(ssp_data, filters, met_logzsol=0.2)
        sed = model.predict_rest_sed(params).sed
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed")
