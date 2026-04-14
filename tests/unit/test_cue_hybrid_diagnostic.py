"""Diagnostic test for the Cue hybrid photometry discrepancy.

The Cue nebular emulator shows ~23% hybrid error vs exact in some configurations.
Root cause is currently unidentified.  Ruled out:
- SFR propagation: identical in both paths (sfr_on_ssp[-1])
- Wavelength grid: same ssp_wave_f64 in both
- Unit conventions: same erg/s/Hz
- Filter integration: identical trapz + interp in both compute_flux_density paths

This test documents the known discrepancy and will be promoted to a passing test
once the root cause is found and fixed.

See docs/dev/optimization-architecture.md footnote ① for context.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.core.model import Model
from tengri.core.parameters import ParamSpec
from tengri.distributions import Fixed, Uniform
from tengri.models.observation.filters import load_filter_set
from tengri.models.sps.dsps_wrapper import load_ssp_data

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(
    not _SSP_EXISTS,
    reason="SSP data file not found — integration test requires data/ssp_*.h5",
)

_FILTER_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ssp_data():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def filters():
    return load_filter_set(_FILTER_NAMES)


@pytest.fixture(scope="module")
def cue_spec():
    """DPL SFH + Cue nebular emulator, no dust emission, fixed z=0.1."""
    return ParamSpec(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
        met_logzsol=Fixed(-1.5),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.1),
        dust_slope=Fixed(-0.7),
        nebular="cue",
        redshift=0.1,
    )


@pytest.fixture(scope="module")
def cue_model(ssp_data, filters, cue_spec):
    """Model with Cue nebular emulator, no dust IR emission."""
    try:
        return Model(cue_spec, ssp_data, filters=filters)
    except (ImportError, FileNotFoundError) as exc:
        pytest.skip(f"Cue emulator not available: {exc}")


_KEY = jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def cue_params(cue_spec):
    return cue_spec.sample(_KEY)


# ---------------------------------------------------------------------------
# Diagnostic: compare exact vs hybrid, assert discrepancy is documented
# ---------------------------------------------------------------------------


class TestCueHybridDiagnostic:
    """Document and bound the known Cue hybrid error.

    These tests are marked xfail because the root cause of the ~23% Cue
    hybrid error is unidentified.  They become passing assertions once fixed.
    """

    @pytest.mark.xfail(
        reason=(
            "Cue hybrid error ~23%: root cause unidentified after ruling out "
            "SFR, wavelength grid, units, and filter integration differences. "
            "Promote to passing once fixed."
        ),
        strict=False,
    )
    def test_cue_hybrid_error_below_5pct(self, cue_model, cue_params):
        """Cue hybrid photometry should agree with exact within 5% per band.

        Currently fails with ~23% error in some configurations.
        This test documents the known failure mode so it does not go unnoticed.
        """
        flux_hybrid = cue_model.predict_photometry(cue_params, mode="hybrid")
        flux_exact = cue_model.predict_photometry(cue_params, mode="exact")

        err = jnp.abs(flux_hybrid - flux_exact) / (jnp.abs(flux_exact) + 1e-50)
        max_err_pct = float(jnp.max(err)) * 100.0

        # Diagnostic: print per-band errors to help future debugging
        band_errs = [float(e) * 100.0 for e in err]
        bands = _FILTER_NAMES
        print("\nCue hybrid vs exact per-band error:")
        for b, e in zip(bands, band_errs):
            print(f"  {b}: {e:.2f}%")

        assert max_err_pct < 5.0, (
            f"Cue hybrid max per-band error {max_err_pct:.1f}% exceeds 5%. "
            "See test_cue_hybrid_diagnostic.py for investigation notes."
        )

    def test_cue_hybrid_photometry_is_finite(self, cue_model, cue_params):
        """Cue hybrid photometry must be finite regardless of the magnitude error."""
        flux = cue_model.predict_photometry(cue_params, mode="hybrid")
        assert jnp.all(jnp.isfinite(flux)), "Cue hybrid photometry contains NaN/Inf"

    def test_cue_hybrid_photometry_is_positive(self, cue_model, cue_params):
        """Cue hybrid photometry must be positive regardless of the magnitude error."""
        flux = cue_model.predict_photometry(cue_params, mode="hybrid")
        assert jnp.all(flux > 0.0), "Cue hybrid photometry has non-positive values"

    def test_cue_exact_photometry_is_finite(self, cue_model, cue_params):
        """Cue exact photometry must be finite (sanity check)."""
        flux = cue_model.predict_photometry(cue_params, mode="exact")
        assert jnp.all(jnp.isfinite(flux)), "Cue exact photometry contains NaN/Inf"
