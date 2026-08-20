# SPDX-License-Identifier: BSD-3-Clause
"""#1317: one noun, action verbs. Wraps the existing engine; ingestion
and validation happen at construction (fail fast, before any compile)."""

import jax
import numpy as np
import pytest


@pytest.fixture
def fwd_3band(synthetic_ssp_wide, simple_observation):
    """Synthetic 3-band ForwardModel for testing."""
    from tengri import FIXED, FREE, ForwardModel, SEDModel, WavePrecomp

    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_bc": 0.5,
        },
        neb={"type": "none"},
        redshift=FIXED,
        approx=WavePrecomp(catalog_z_range=(0.01, 2.0)),
    )
    fwd = ForwardModel.build(sed=sed, observation=simple_observation)
    return fwd


@pytest.fixture
def fwd_3band_no_zrange(synthetic_ssp_wide, simple_observation):
    """3-band ForwardModel WITHOUT a catalog_z_range — no runtime-z LUT."""
    from tengri import FIXED, FREE, ForwardModel, SEDModel

    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_bc": 0.5,
        },
        neb={"type": "none"},
        redshift=FIXED,
    )
    return ForwardModel.build(sed=sed, observation=simple_observation)


@pytest.fixture
def fwd_3band_zrange(synthetic_ssp_wide, simple_observation):
    """3-band ForwardModel with WavePrecomp catalog_z_range."""
    from tengri import FIXED, FREE, ForwardModel, SEDModel, WavePrecomp

    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_bc": 0.5,
        },
        neb={"type": "none"},
        redshift=FIXED,
        approx=WavePrecomp(catalog_z_range=(0.05, 1.5)),
    )
    fwd = ForwardModel.build(sed=sed, observation=simple_observation)
    return fwd


@pytest.fixture
def table_3band():
    """3-row table with 3 bands and redshifts."""
    return {
        "band_0": np.array([1.0, 2.0, 1.5]),
        "band_0_err": np.array([0.1, 0.1, 0.1]),
        "band_1": np.array([3.0, 4.0, 3.5]),
        "band_1_err": np.array([0.2, 0.2, 0.2]),
        "band_2": np.array([5.0, 6.0, 5.5]),
        "band_2_err": np.array([0.3, 0.3, 0.3]),
        "z": np.array([0.1, 0.5, 0.3]),
    }


@pytest.fixture
def table_3band_bad_missing_col():
    """3-row table missing an error column."""
    return {
        "band_0": np.array([1.0, 2.0, 1.5]),
        "band_0_err": np.array([0.1, 0.1, 0.1]),
        "band_1": np.array([3.0, 4.0, 3.5]),
        # Missing band_1_err!
        "band_2": np.array([5.0, 6.0, 5.5]),
        "band_2_err": np.array([0.3, 0.3, 0.3]),
        "z": np.array([0.1, 0.5, 0.3]),
    }


@pytest.fixture
def table_z_outside():
    """3-row table with redshifts outside the catalog_z_range (0.05, 1.5)."""
    return {
        "band_0": np.array([1.0, 2.0, 1.5]),
        "band_0_err": np.array([0.1, 0.1, 0.1]),
        "band_1": np.array([3.0, 4.0, 3.5]),
        "band_1_err": np.array([0.2, 0.2, 0.2]),
        "band_2": np.array([5.0, 6.0, 5.5]),
        "band_2_err": np.array([0.3, 0.3, 0.3]),
        "z": np.array([0.01, 2.0, 0.3]),  # 0.01 < 0.05 and 2.0 > 1.5
    }


@pytest.fixture
def param_table_3rows(fwd_3band):
    """Parameter table with 3 rows and all free params at prior midpoints."""
    from tengri import Uniform

    free_params = fwd_3band.spec.free_params
    param_table = {}
    for name in free_params:
        # Get the prior from the spec
        prior = fwd_3band.spec.get_distribution(name)
        # Use prior midpoint for initialization
        if isinstance(prior, Uniform):
            midpoint = (prior.lo + prior.hi) / 2.0
        else:
            # For other distributions, use a reasonable default
            midpoint = 0.5
        param_table[name] = np.full(3, midpoint, dtype=np.float64)
    return param_table


def test_construction_validates_eagerly(fwd_3band, table_3band_bad_missing_col):
    from tengri import Catalog

    with pytest.raises(ValueError):  # missing err column found at __init__
        Catalog(fwd_3band, table_3band_bad_missing_col, flux_unit="cgs_fnu")


def test_fit_default_is_map_and_returns_catalog_posterior(fwd_3band, table_3band):
    from tengri import Catalog

    cat = Catalog(fwd_3band, table_3band, flux_unit="cgs_fnu", redshift_col="z")
    post = cat.fit(key=jax.random.PRNGKey(0))  # no method= -> "map"
    assert post.n_galaxies == 3
    assert np.asarray(post.properties["stellar_mass"]).shape == (3,)


def test_redshift_span_validated_against_catalog_z_range(fwd_3band_zrange, table_z_outside):
    from tengri import Catalog

    with pytest.raises(ValueError, match="catalog_z_range"):
        Catalog(fwd_3band_zrange, table_z_outside, flux_unit="cgs_fnu", redshift_col="z")


def test_per_galaxy_redshift_reaches_forward_pass(fwd_3band):
    """HARD GATE: per-galaxy redshift must change the FIT, not just the echoed z.

    Rigour requires three things a naive version gets wrong:
      1. measure a FREE parameter (a fixed one, e.g. dust_tau_bc pinned at 0.5,
         cannot diverge — vacuous);
      2. use MODEL-SCALE data (arbitrary order-1 data against this synthetic
         model's tiny flux makes chi^2 blind to the model, hence blind to z —
         a *correct* fix then looks broken);
      3. ISOLATE z from the per-galaxy key confound: CatalogPosterior splits a
         fresh key per galaxy, so two galaxies differ by their keys even at the
         same z. We therefore fit galaxy 0 in two runs with the SAME key and
         SAME data, changing only its redshift.
    """
    from tengri import Catalog

    free = list(fwd_3band.spec.free_params)
    # Model-scale data generated from the model itself (matched flux scale) at
    # prior midpoints — param=0.0 gives a degenerate zero-flux SFH for this model.
    truth = {}
    for p in free:
        dist = fwd_3band.spec.get_distribution(p)
        truth[p] = float((dist.lo + dist.hi) / 2.0) if hasattr(dist, "lo") else 0.5
    flux = np.asarray(fwd_3band.predict_photometry(truth))
    err = np.abs(flux) * 0.05 + 1e-30

    def table_with_g0_z(z0):
        # Two galaxies with IDENTICAL data; galaxy 0's redshift is z0, galaxy 1
        # is a fixed anchor. Only galaxy 0's z changes between the two runs.
        return {
            "band_0": np.array([flux[0], flux[0]]),
            "band_0_err": np.array([err[0], err[0]]),
            "band_1": np.array([flux[1], flux[1]]),
            "band_1_err": np.array([err[1], err[1]]),
            "band_2": np.array([flux[2], flux[2]]),
            "band_2_err": np.array([err[2], err[2]]),
            "z": np.array([z0, 0.5]),
        }

    key = jax.random.PRNGKey(0)
    post_lo = Catalog(fwd_3band, table_with_g0_z(0.1), flux_unit="cgs_fnu", redshift_col="z").fit(
        method="map", key=key
    )
    post_hi = Catalog(fwd_3band, table_with_g0_z(1.5), flux_unit="cgs_fnu", redshift_col="z").fit(
        method="map", key=key
    )
    # Galaxy 0 uses the SAME split key and SAME data in both runs — only its
    # redshift differs (0.1 vs 1.5). Any MAP difference is the redshift effect.
    g_lo, g_hi = post_lo.posteriors[0].params, post_hi.posteriors[0].params
    max_delta = max(abs(float(g_lo[p]) - float(g_hi[p])) for p in free)
    assert max_delta > 1e-3, (
        f"galaxy 0 at z=0.1 vs z=1.5 (same key, same data) gave identical free-param "
        f"MAPs (max Δ={max_delta:.2e}) — per-galaxy redshift did NOT reach the forward "
        f"pass (silent relabel)."
    )


def test_per_galaxy_redshift_batched_guards(fwd_3band, fwd_3band_no_zrange, table_3band):
    """The batched paths that cannot take per-galaxy z must fail loudly.

    Two guarded combinations remain after #1349 (which made batched MCMC *with*
    a ``catalog_z_range`` work — this test originally expected that case to
    raise, and went stale unnoticed because ``tests/unit`` runs in no CI job):

    * batched native VI never threads per-galaxy z → ``NotImplementedError``;
    * batched MCMC without a ``catalog_z_range`` has no runtime-z LUT →
      ``ValueError`` teaching the ``WavePrecomp(catalog_z_range=...)`` build.
    """
    from tengri import Catalog

    cat = Catalog(fwd_3band, table_3band, flux_unit="cgs_fnu", redshift_col="z")
    # `allow_unvalidated=True` is required since #1394 put the tier gate ahead of
    # these checks: `native_vi_linear` is tier="broken", so without the opt-in
    # this raises BackendError and the per-galaxy-z guard below never runs. The
    # opt-in keeps THIS test measuring what it is named for.
    with pytest.raises(NotImplementedError, match="native VI"):
        cat.fit(method="native_vi_linear", key=jax.random.PRNGKey(0), allow_unvalidated=True)

    with pytest.warns(UserWarning, match="catalog_z_range"):
        cat_nz = Catalog(fwd_3band_no_zrange, table_3band, flux_unit="cgs_fnu", redshift_col="z")
    with pytest.raises(ValueError, match="catalog_z_range"):
        cat_nz.fit(method="mcmc_nuts", key=jax.random.PRNGKey(0), n_warmup=5, n_samples=5)


def test_predict_mock_shapes(fwd_3band, param_table_3rows):
    from tengri import Catalog

    cat = Catalog(fwd_3band, None, flux_unit="cgs_fnu")  # prediction-only: no data table
    mock = cat.predict(param_table_3rows)
    assert mock.shape == (3, fwd_3band.observation.photometry.n_filters)


def test_no_vi_default_anywhere(fwd_3band, table_3band):
    import inspect

    from tengri import Catalog

    sig = inspect.signature(Catalog.fit)
    assert sig.parameters["method"].default == "map"
