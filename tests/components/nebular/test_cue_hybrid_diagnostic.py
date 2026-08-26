# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Cue hybrid photometry energy-balance fix.

Previously showed ~23% hybrid error vs exact.  Root cause was the same as the DL07
hybrid bug: L_absorbed_stellar was computed from a Voronoi-bandwidth-weighted sum over
SDSS filter bands only, missing all UV absorption where dust attenuation peaks.  Fixed
by replacing the Voronoi sum with a 200-point coarse-wavelength trapz.

These tests verify:
1. Cue hybrid error < 5% per SDSS band at z=0.1 (was ~23%).
2. Hybrid photometry is finite and positive.
3. Exact photometry is finite (sanity check).

All tests require SSP data on disk; they are skipped gracefully when missing.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri.forward.sed_model import SEDModel, WavePrecomp
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

# ── Skip guards ───────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SSP_FILE = _DATA_DIR / "bc03_pdva_stelib_chabrier.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

# One assignment holding both. Assigning `pytestmark` twice rebinds the name,
# which silently dropped the `bounds` taxonomy marker.
pytestmark = [
    pytest.mark.bounds,
    pytest.mark.skipif(
        not _SSP_EXISTS,
        reason=(
            "BC03 bare-stellar SSP not found — required for Cue diagnostics "
            "(wNE SSPs raise CueWNESSPError). "
            "Run `tengri.download_ssp('bc03_pdva_stelib_chabrier')`."
        ),
    ),
]

_FILTER_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp_data(ssp_data_bc03):
    """Cue needs bare-stellar SSPs (see ``ssp_data_bc03`` in conftest)."""
    return ssp_data_bc03


@pytest.fixture(scope="module")
def filters(sdss_filters):
    return sdss_filters


@pytest.fixture(scope="module")
def cue_spec():
    """DPL SFH + Cue nebular emulator, no dust emission, fixed z=0.1."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Fixed(-1.5),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.1),
        dust_slope=Fixed(-0.7),
        nebular="cue",
        redshift=0.1,
    )


@pytest.fixture(scope="module")
def cue_model(ssp_data, filters, cue_spec):
    """Cue model on the WavePrecomp LUT path (the former ``mode="hybrid"``).

    The fast path is selected at build time via ``approx=WavePrecomp()``; the
    removed call-time ``predict_photometry(..., mode="hybrid")`` kwarg no longer
    exists.
    """
    try:
        return SEDModel(cue_spec, ssp_data, filters=filters, approx=WavePrecomp())
    except (ImportError, FileNotFoundError) as exc:
        pytest.skip(f"Cue emulator not available: {exc}")


@pytest.fixture(scope="module")
def cue_model_exact(ssp_data, filters, cue_spec):
    """Cue model on the exact wave-grid path (the former ``mode="exact"``)."""
    try:
        return SEDModel(cue_spec, ssp_data, filters=filters, approx=None)
    except (ImportError, FileNotFoundError) as exc:
        pytest.skip(f"Cue emulator not available: {exc}")


_KEY = jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def cue_params(cue_spec):
    return cue_spec.sample(_KEY)


# ── Diagnostic: compare exact vs hybrid, assert discrepancy is documented


class TestCueHybridDiagnostic:
    """Cue hybrid photometry error-bound and sanity checks."""

    def test_cue_hybrid_error_below_5pct(self, cue_model, cue_model_exact, cue_params):
        """Cue hybrid photometry agrees with exact within 5% per band.

        Previously failed with ~23% error (root cause: L_absorbed_stellar computed
        from Voronoi-bandwidth-weighted sum over SDSS filter bands, missing UV absorption).
        Fixed alongside DL07 hybrid: replaced Voronoi sum with 200-point coarse-wavelength
        trapz, matching the exact/compositional path.
        """
        flux_hybrid = cue_model.predict_photometry(cue_params)
        flux_exact = cue_model_exact.predict_photometry(cue_params)

        err = jnp.abs(flux_hybrid - flux_exact) / (jnp.abs(flux_exact) + 1e-50)
        max_err_pct = float(jnp.max(err)) * 100.0
        band_errs = {b: float(e) * 100.0 for b, e in zip(_FILTER_NAMES, err)}
        per_band = "  ".join(f"{b}:{v:.1f}%" for b, v in band_errs.items())

        assert max_err_pct < 5.0, (
            f"Cue hybrid max per-band error {max_err_pct:.1f}% exceeds 5%. "
            f"Per-band: {per_band}. "
            "See test_cue_hybrid_diagnostic.py for investigation notes."
        )

    def test_cue_hybrid_photometry_is_finite(self, cue_model, cue_params):
        """Cue hybrid photometry must be finite regardless of the magnitude error."""
        flux = cue_model.predict_photometry(cue_params)
        chex.assert_tree_all_finite(flux)

    def test_cue_hybrid_photometry_is_positive(self, cue_model, cue_params):
        """Cue hybrid photometry must be positive regardless of the magnitude error."""
        flux = cue_model.predict_photometry(cue_params)
        assert jnp.all(flux > 0.0), "Cue hybrid photometry has non-positive values"

    def test_cue_exact_photometry_is_finite(self, cue_model_exact, cue_params):
        """Cue exact photometry must be finite (sanity check)."""
        flux = cue_model_exact.predict_photometry(cue_params)
        chex.assert_tree_all_finite(flux)
