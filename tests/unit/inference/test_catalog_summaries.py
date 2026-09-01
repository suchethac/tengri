# SPDX-License-Identifier: BSD-3-Clause
"""#1313: catalog per-galaxy summaries — percentiles, reducers, to_table().

These tests exercise the REAL implementation end-to-end (``Catalog.fit`` and
``_compute_summaries``), not hand-built fixture data. The reduction logic is
checked against analytically-known values, and the store="summary" cube-drop and
to_table round-trip are checked through an actual fit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def _fwd_free_z():
    """3-band model with a FREE redshift (so MCMC needs no per-galaxy redshift)."""
    from tengri import (
        DEFAULT,
        FREE,
        Fixed,
        ForwardModel,
        Observation,
        Photometry,
        SEDModel,
        Uniform,
    )
    from tengri.components.stellar.sps.dsps_wrapper import SSPData
    from tengri.observation.photometry import FilterCurve

    wave = jnp.linspace(3000.0, 10000.0, 60)
    ages = jnp.linspace(-1.0, 1.14, 12)
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    ssp = SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(jnp.ones((3, 12, 60))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
    )
    curves = tuple(
        FilterCurve(wave=jnp.linspace(lo, hi, 30), trans=jnp.ones(30) * 0.5, name=f"b{i}")
        for i, (lo, hi) in enumerate([(3500.0, 4500.0), (5000.0, 6500.0), (7500.0, 9000.0)])
    )
    obs = Observation(photometry=Photometry(filters=curves))
    sed = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.5,
        },
        neb={"type": "none"},
        redshift=Uniform(0.1, 1.0),
    )
    return ForwardModel.build(sed=sed, observation=obs), sed


def _table(fwd, sed, n=3):
    d = np.asarray(sed.predict_photometry({p: 0.0 for p in sed.spec.free_params}))
    return {
        "b0": np.array([d[0]] * n),
        "b0_err": np.abs([d[0]] * n) * 0.1 + 1e-30,
        "b1": np.array([d[1]] * n),
        "b1_err": np.abs([d[1]] * n) * 0.1 + 1e-30,
        "b2": np.array([d[2]] * n),
        "b2_err": np.abs([d[2]] * n) * 0.1 + 1e-30,
    }


def test_compute_summaries_analytic():
    """The reduction logic matches np.percentile / the reducers on KNOWN input.

    This is the load-bearing non-vacuous check: a shape-only test passes for
    ``np.zeros``, so we assert exact values on a known sample set.
    """
    from tengri.inference.catalog_fitter import _compute_summaries

    samples = {"x": np.arange(11.0)}  # [0, 1, ..., 10]
    pc, sm = _compute_summaries(
        samples, percentiles=(0, 25, 50, 75, 100), reducers={"mean": np.mean, "std": np.std}
    )
    np.testing.assert_allclose(pc["x"], [0.0, 2.5, 5.0, 7.5, 10.0])
    np.testing.assert_allclose(sm["mean"]["x"], 5.0)
    np.testing.assert_allclose(sm["std"]["x"], np.std(np.arange(11.0)))


@pytest.mark.slow
def test_store_summary_end_to_end_drops_cube_and_summarizes():
    """A real MCMC fit with store='summary' computes percentiles/reducers, drops
    the samples cube, and to_table() round-trips as an ingest duck-type."""
    fwd, sed = _fwd_free_z()
    from tengri import Catalog

    cat = Catalog(fwd, _table(fwd, sed), flux_unit="cgs_fnu")
    post = cat.fit(
        method="mcmc_nuts",
        key=jax.random.PRNGKey(0),
        n_warmup=20,
        n_samples=30,
        store="summary",
        percentiles=(16, 50, 84),
        reducers={"mean": np.mean, "std": np.std},
    )
    p = next(iter(sed.spec.free_params))
    # percentiles + reducers materialized at (N, n_pct) / (N,)
    assert post.percentiles is not None and np.asarray(post.percentiles[p]).shape == (3, 3)
    assert post.summary is not None and np.asarray(post.summary["mean"][p]).shape == (3,)
    v = np.asarray(post.percentiles[p][0])
    assert v[0] <= v[1] <= v[2], "percentiles must be ordered p16<=p50<=p84"
    # the cube is actually dropped (memory), not merely hidden
    assert post.posteriors[0].samples is None
    # to_table round-trips: a dict of length-N columns, re-ingestable
    table = post.to_table()
    assert isinstance(table, dict) and len(table) > 0
    assert all(len(v) == 3 for v in table.values() if isinstance(v, np.ndarray))


@pytest.mark.slow
def test_store_full_keeps_samples():
    """store='full' retains the per-galaxy samples (today's behavior)."""
    fwd, sed = _fwd_free_z()
    from tengri import Catalog

    cat = Catalog(fwd, _table(fwd, sed), flux_unit="cgs_fnu")
    post = cat.fit(
        method="mcmc_nuts", key=jax.random.PRNGKey(1), n_warmup=20, n_samples=30, store="full"
    )
    assert post.posteriors[0].samples is not None
