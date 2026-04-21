"""Regression tests for the DL07 hybrid kernel energy-balance fix.

Root cause of the 43% hybrid error (now fixed):
    L_absorbed_stellar was computed from a Voronoi-bandwidth-weighted sum over
    SDSS filter bands only.  SDSS ugriz at z=0.1 covers rest-frame ~2600–8800 Å,
    missing all UV absorption where dust attenuation peaks.  The fix replaces the
    Voronoi sum with a 200-point coarse-wavelength trapz (same formula as the
    exact/compositional path).

These tests verify:
1. DL07 + SDSS ugriz at z=0.1: hybrid error vs exact < 2% per band (was 43%).
2. Dale 2014 hybrid error is not regressed (must remain < 1%).
3. THEMIS hybrid error is not regressed (must remain < 1%).
4. Stellar-only hybrid error is not regressed (must remain < 1%).

All tests require SSP data on disk; they are skipped gracefully when missing.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.forward.sed_model import SEDModel
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

# ── Skip guards — require SSP data ────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_DL07_FILE = _DATA_DIR / "dl07_templates.npz"
_DALE_FILE = _DATA_DIR / "dale2014_templates.npz"
_THEMIS_FILE = _DATA_DIR / "themis_templates.npz"

_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(
    not _SSP_EXISTS,
    reason="SSP data file not found — integration test requires data/ssp_*.h5",
)

_FILTER_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]

# Random seed for reproducible parameter samples
_KEY = jax.random.PRNGKey(42)


# ── Session-scoped fixtures ───────────────────────────────────────


@pytest.fixture(scope="session")
def ssp_data(ssp_data_wne):
    return ssp_data_wne


@pytest.fixture(scope="session")
def filters(sdss_filters):
    return sdss_filters


def _base_spec_kwargs():
    """Common DPL SFH + dust attenuation kwargs for all spec fixtures."""
    return dict(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
        met_logzsol=Fixed(-1.5),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.1),
        dust_slope=Fixed(-0.7),
        redshift=0.1,
    )


def _photometry_error(model, params, key_a="hybrid", key_b="exact"):
    """Return per-band fractional error |hybrid - exact| / |exact|."""
    flux_a = model.predict_photometry(params, mode=key_a)
    flux_b = model.predict_photometry(params, mode=key_b)
    return jnp.abs(flux_a - flux_b) / (jnp.abs(flux_b) + 1e-50)


# ── DL07 energy-balance regression ────────────────────────────────


class TestDL07EnergyBalance:
    """DL07 hybrid error must be < 2% per band (was ~43% before fix)."""

    @pytest.fixture(scope="class")
    def dl07_spec(self):
        if not _DL07_FILE.is_file():
            pytest.skip("DL07 template file not found")
        return Parameters(**_base_spec_kwargs(), dust_emission="draine_li2007")

    @pytest.fixture(scope="class")
    def dl07_model(self, ssp_data, filters, dl07_spec):
        return SEDModel(dl07_spec, ssp_data, filters=filters)

    @pytest.fixture(scope="class")
    def dl07_params(self, dl07_spec):
        return dl07_spec.sample(_KEY)

    def test_dl07_hybrid_error_below_2pct(self, dl07_model, dl07_params):
        """DL07 hybrid error < 2% per SDSS band at z=0.1 (regression: was ~43%)."""
        err = _photometry_error(dl07_model, dl07_params)
        max_err = float(jnp.max(err)) * 100.0
        assert max_err < 2.0, (
            f"DL07 hybrid max per-band error {max_err:.1f}% exceeds 2% threshold. "
            "Energy-balance coarse-grid fix may have been reverted."
        )

    def test_dl07_hybrid_photometry_is_finite(self, dl07_model, dl07_params):
        """DL07 hybrid photometry must be finite in all bands."""
        flux = dl07_model.predict_photometry(dl07_params, mode="hybrid")
        assert jnp.all(jnp.isfinite(flux)), "DL07 hybrid photometry contains NaN/Inf"

    def test_dl07_hybrid_photometry_is_positive(self, dl07_model, dl07_params):
        """DL07 hybrid photometry must be positive in all bands."""
        flux = dl07_model.predict_photometry(dl07_params, mode="hybrid")
        assert jnp.all(flux > 0.0), "DL07 hybrid photometry has non-positive values"


# ── Non-regression: Dale 2014 ─────────────────────────────────────


class TestDale2014NonRegression:
    """Dale 2014 hybrid error must not regress — must stay < 1%."""

    @pytest.fixture(scope="class")
    def dale_spec(self):
        if not _DALE_FILE.is_file():
            pytest.skip("Dale 2014 template file not found")
        return Parameters(**_base_spec_kwargs(), dust_emission="dale2014")

    @pytest.fixture(scope="class")
    def dale_model(self, ssp_data, filters, dale_spec):
        return SEDModel(dale_spec, ssp_data, filters=filters)

    @pytest.fixture(scope="class")
    def dale_params(self, dale_spec):
        return dale_spec.sample(_KEY)

    def test_dale_hybrid_error_below_1pct(self, dale_model, dale_params):
        """Dale 2014 hybrid error < 1% per band (non-regression guard)."""
        err = _photometry_error(dale_model, dale_params)
        max_err = float(jnp.max(err)) * 100.0
        assert max_err < 1.0, (
            f"Dale 2014 hybrid max per-band error {max_err:.2f}% regressed past 1%."
        )


# ── Non-regression: THEMIS ────────────────────────────────────────


class TestTHEMISNonRegression:
    """THEMIS hybrid error must not regress — must stay < 1%."""

    @pytest.fixture(scope="class")
    def themis_spec(self):
        if not _THEMIS_FILE.is_file():
            pytest.skip("THEMIS template file not found")
        return Parameters(**_base_spec_kwargs(), dust_emission="themis")

    @pytest.fixture(scope="class")
    def themis_model(self, ssp_data, filters, themis_spec):
        return SEDModel(themis_spec, ssp_data, filters=filters)

    @pytest.fixture(scope="class")
    def themis_params(self, themis_spec):
        return themis_spec.sample(_KEY)

    def test_themis_hybrid_error_below_1pct(self, themis_model, themis_params):
        """THEMIS hybrid error < 1% per band (non-regression guard)."""
        err = _photometry_error(themis_model, themis_params)
        max_err = float(jnp.max(err)) * 100.0
        assert max_err < 1.0, f"THEMIS hybrid max per-band error {max_err:.2f}% regressed past 1%."


# ── Non-regression: stellar only ──────────────────────────────────


class TestStellarOnlyNonRegression:
    """Stellar-only hybrid error must not regress — must stay < 1%."""

    @pytest.fixture(scope="class")
    def stellar_spec(self):
        return Parameters(**_base_spec_kwargs())

    @pytest.fixture(scope="class")
    def stellar_model(self, ssp_data, filters, stellar_spec):
        return SEDModel(stellar_spec, ssp_data, filters=filters)

    @pytest.fixture(scope="class")
    def stellar_params(self, stellar_spec):
        return stellar_spec.sample(_KEY)

    def test_stellar_hybrid_error_below_1pct(self, stellar_model, stellar_params):
        """Stellar-only hybrid error < 1% per band (non-regression guard)."""
        err = _photometry_error(stellar_model, stellar_params)
        max_err = float(jnp.max(err)) * 100.0
        assert max_err < 1.0, (
            f"Stellar-only hybrid max per-band error {max_err:.2f}% regressed past 1%."
        )


# ── DL07 worst-case: young, heavily-dusty galaxy ──────────────────
# UV absorption peaks at <2000 Å — the original Voronoi-sum bug missed this
# band entirely (SDSS ugriz at z=0.1 covers rest ~2600–8800 Å).


class TestDL07EnergyBalanceWorstCase:
    """DL07 hybrid error < 2% for young high-dust galaxy (max UV absorption gap).

    The original bug (Voronoi sum over SDSS bands → missed UV) produces the
    largest error for galaxies with:
    - Rapid recent star formation (high UV output to attenuate)
    - High dust optical depth (τ_bc >> 1 → strong UV absorption)
    A fixed test at a single prior draw may happen to be a low-UV-absorption
    galaxy and pass even if the fix is reverted.  This class pins the
    worst-case regime explicitly.
    """

    @pytest.fixture(scope="class")
    def dl07_spec_young_dusty(self):
        if not _DL07_FILE.is_file():
            pytest.skip("DL07 template file not found")
        return Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(3.0),  # steep rise — young burst
            sfh_dpl_beta=Fixed(0.3),  # slow decline — sustained recent SFR
            sfh_dpl_tau_gyr=Fixed(0.5),  # peak at 500 Myr → strong UV
            sfh_dpl_log_peak_sfr=Fixed(2.0),  # 100 Msun/yr peak
            met_logzsol=Fixed(-1.5),
            dust_tau_bc=Fixed(1.5),  # heavy birth-cloud dust → large UV attenuation
            dust_tau_diff=Fixed(0.5),
            dust_slope=Fixed(-0.7),
            dust_emission="draine_li2007",
            redshift=0.1,
        )

    @pytest.fixture(scope="class")
    def dl07_model_young_dusty(self, ssp_data, filters, dl07_spec_young_dusty):
        return SEDModel(dl07_spec_young_dusty, ssp_data, filters=filters)

    @pytest.fixture(scope="class")
    def dl07_params_young_dusty(self, dl07_spec_young_dusty):
        return dl07_spec_young_dusty.sample(_KEY)

    def test_dl07_worst_case_hybrid_error_below_2pct(
        self, dl07_model_young_dusty, dl07_params_young_dusty
    ):
        """Young high-dust DL07: hybrid error < 2% per band.

        This is the worst-case for the Voronoi-sum UV absorption bug.
        τ_bc=1.5 → exp(-1.5) ≈ 22% UV transmission, large L_absorbed_stellar.
        If the coarse-grid trapz fix is reverted the error will reappear here first.
        """
        err = _photometry_error(dl07_model_young_dusty, dl07_params_young_dusty)
        max_err = float(jnp.max(err)) * 100.0
        assert max_err < 2.0, (
            f"Young/dusty DL07 hybrid max per-band error {max_err:.1f}% exceeds 2%. "
            "This probes the UV-absorption gap that caused the original 43% error — "
            "the coarse-grid trapz fix for L_absorbed_stellar may have been reverted."
        )
