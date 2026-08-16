# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end correctness tests for ``CatalogFitter`` ``n_pad``.

The core safety property: padding a catalog with dummy galaxies must
not change real galaxies' posteriors. The catalog VI engine is
per-galaxy with no cross-galaxy reduction, so this should be true by
construction — these tests are the regression fence.

Skipped gracefully when SSP data is not on disk.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# ── Skip guard ─────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(not _SSP_EXISTS, reason=f"SSP file not found: {_SSP_FILE}")


def _build_catalog(n_gal: int, base_key):
    """Construct a small mock photometric catalog plus a model factory."""
    from tengri import (
        Fixed,
        Observation,
        Parameters,
        Photometry,
        SEDModel,
        Uniform,
    )
    from tengri.sps.dsps_wrapper import load_ssp_data

    ssp_data = load_ssp_data(str(_SSP_FILE))
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    spec = Parameters(
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_beta=Uniform(0.5, 5.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        sfh_dpl_tau_gyr=Uniform(0.1, 12.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )

    def model_factory():
        return SEDModel(spec, ssp_data, observation=obs)

    template = model_factory()
    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(base_key, i)
        true_params = template.spec.sample(k)
        flux = template.predict_photometry(true_params)
        noise = jnp.abs(flux) * 0.1 + 1e-3
        flux_obs = flux + noise * jax.random.normal(k, shape=flux.shape)
        galaxies.append({"flux_obs": flux_obs, "noise": noise})
    return galaxies, model_factory


def test_n_pad_does_not_change_posteriors():
    """7-galaxy catalog: default vs n_pad=16 must agree per-galaxy."""
    from tengri import CatalogFitter

    n_gal = 7
    galaxies, factory = _build_catalog(n_gal, jax.random.PRNGKey(13))

    cat = CatalogFitter(factory(), galaxies, data_type="photometry")
    # These tests are about PADDING SEMANTICS on the native path, not about the
    # backend's science quality, so the tier gate added in #1394 is opted out of
    # rather than worked around: exercising a broken-tier backend deliberately is
    # exactly the "benchmarking or backend development" case the hatch exists for.
    common = dict(
        n_iterations=8,
        n_samples=3,
        n_posterior_samples=24,
        forward_chunk_size=4,
        kl_rtol=1e-2,
        verbose=False,
        allow_unvalidated=True,
    )

    cp_default = cat.run("native_vi_linear", key=jax.random.PRNGKey(7), **common)
    cp_padded = cat.run("native_vi_linear", key=jax.random.PRNGKey(7), n_pad=16, **common)

    assert cp_default.n_galaxies == n_gal == cp_padded.n_galaxies
    assert len(cp_default.posteriors) == n_gal
    assert len(cp_padded.posteriors) == n_gal

    for i in range(n_gal):
        p_def = cp_default.posteriors[i].params
        p_pad = cp_padded.posteriors[i].params
        for name in p_def:
            if name == "psd_xi":
                continue
            v_def = float(np.asarray(p_def[name]).mean())
            v_pad = float(np.asarray(p_pad[name]).mean())
            assert v_def == pytest.approx(v_pad, rel=1e-5, abs=1e-5), (
                f"galaxy {i} param {name}: default={v_def}, padded={v_pad}"
            )


def test_n_pad_auto_safe_for_small_catalog():
    """n_pad='auto' on a tiny catalog must produce sane output (no crash, n_real preserved)."""
    from tengri import CatalogFitter

    n_gal = 3
    galaxies, factory = _build_catalog(n_gal, jax.random.PRNGKey(21))

    cat = CatalogFitter(factory(), galaxies, data_type="photometry")
    cp = cat.run(
        "native_vi_linear",
        key=jax.random.PRNGKey(0),
        forward_chunk_size=2,
        n_pad="auto",
        n_iterations=6,
        n_samples=3,
        n_posterior_samples=12,
        kl_rtol=1e-2,
        verbose=False,
        allow_unvalidated=True,
    )
    assert cp.n_galaxies == n_gal
    assert len(cp.posteriors) == n_gal


def test_n_pad_below_n_gal_raises():
    """n_pad < n_galaxies must raise from run()."""
    from tengri import CatalogFitter

    n_gal = 5
    galaxies, factory = _build_catalog(n_gal, jax.random.PRNGKey(0))
    cat = CatalogFitter(factory(), galaxies, data_type="photometry")

    with pytest.raises(ValueError, match="must be >= n_galaxies"):
        cat.run(
            "native_vi_linear",
            key=jax.random.PRNGKey(0),
            forward_chunk_size=1,
            n_pad=3,
            n_iterations=2,
            n_samples=3,
            n_posterior_samples=4,
            verbose=False,
            allow_unvalidated=True,
        )
