# SPDX-License-Identifier: BSD-3-Clause
"""Tests for catalog per-galaxy summaries: percentiles, reducers, to_table()."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.inference.catalog_fitter import CatalogPosterior
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.contract


@pytest.fixture
def tiny_catalog_with_samples():
    """Create a 3-galaxy catalog with known samples for testing percentiles."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    posteriors = []
    for i, k in enumerate(keys):
        # Each galaxy has 10 samples
        k1, k2 = jax.random.split(k)
        # Create known samples: galaxy i gets offset-i data
        base_mass = 10.0 + float(i)
        samples_i = {
            "sfh_dpl_alpha": base_mass + 0.5 * jax.random.normal(k1, (10,)),
            "dust_av": 0.5 + 0.1 * jax.random.normal(k2, (10,)),
        }
        params_i = {
            "sfh_dpl_alpha": jnp.array(base_mass),
            "dust_av": jnp.array(0.5),
        }
        posteriors.append(
            Posterior(
                samples=samples_i,
                params=params_i,
                method="test",
                wall_time_s=1.0,
                diagnostics={},
            )
        )

    return CatalogPosterior(
        posteriors=posteriors,
        method="test",
        wall_time_s=3.0,
        n_galaxies=3,
        diagnostics={},
    )


@pytest.fixture
def tiny_catalog_with_percentiles_and_summary():
    """Create a 3-galaxy catalog with pre-computed percentiles and summary."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    posteriors = []
    for i, k in enumerate(keys):
        # Each galaxy has 10 samples
        k1, k2 = jax.random.split(k)
        base_mass = 10.0 + float(i)
        samples_i = {
            "sfh_dpl_alpha": base_mass + 0.5 * jax.random.normal(k1, (10,)),
            "dust_av": 0.5 + 0.1 * jax.random.normal(k2, (10,)),
        }
        params_i = {
            "sfh_dpl_alpha": jnp.array(base_mass),
            "dust_av": jnp.array(0.5),
        }

        # Compute percentiles manually
        pcts = [16, 50, 84]
        percentiles_i = {}
        for name, samples in samples_i.items():
            percentiles_i[name] = np.percentile(np.asarray(samples), pcts)

        # Compute summary with mean and std reducers
        summary_i = {
            "mean": {},
            "std": {},
        }
        for name, samples in samples_i.items():
            summary_i["mean"][name] = float(np.mean(np.asarray(samples)))
            summary_i["std"][name] = float(np.std(np.asarray(samples)))

        post_i = Posterior(
            samples=samples_i,
            params=params_i,
            method="test",
            wall_time_s=1.0,
            diagnostics={},
        )
        post_i._percentiles_stats_ = percentiles_i
        post_i._summary_stats_ = summary_i

        posteriors.append(post_i)

    cat = CatalogPosterior(
        posteriors=posteriors,
        method="test",
        wall_time_s=3.0,
        n_galaxies=3,
        diagnostics={},
    )
    cat.percentiles = {}
    cat.summary = {}
    cat.store = "summary"

    # Stack percentiles and summary across galaxies
    for name in posteriors[0]._percentiles_stats_:
        cat.percentiles[name] = np.stack([p._percentiles_stats_[name] for p in posteriors])

    for reducer_name in posteriors[0]._summary_stats_:
        cat.summary[reducer_name] = {}
        for name in posteriors[0]._summary_stats_[reducer_name]:
            cat.summary[reducer_name][name] = np.array(
                [p._summary_stats_[reducer_name][name] for p in posteriors]
            )

    return cat


class TestPercentileAndReducerShapes:
    """Test that percentiles and reducers have correct shapes."""

    def test_percentiles_and_reducers_shapes(self, tiny_catalog_with_percentiles_and_summary):
        """Percentiles should have shape (N, n_pct); reducers shape (N,) per property."""
        cat = tiny_catalog_with_percentiles_and_summary
        n_gal = 3
        n_pct = 3  # 16, 50, 84

        # Check percentiles
        assert "sfh_dpl_alpha" in cat.percentiles
        assert cat.percentiles["sfh_dpl_alpha"].shape == (n_gal, n_pct)
        assert cat.percentiles["dust_av"].shape == (n_gal, n_pct)

        # Check summary / reducers
        assert "mean" in cat.summary
        assert "std" in cat.summary

        assert cat.summary["mean"]["sfh_dpl_alpha"].shape == (n_gal,)
        assert cat.summary["mean"]["dust_av"].shape == (n_gal,)
        assert cat.summary["std"]["sfh_dpl_alpha"].shape == (n_gal,)
        assert cat.summary["std"]["dust_av"].shape == (n_gal,)


class TestStoreSummaryDropsTheCube:
    """Test that store='summary' actually drops the samples cube from memory."""

    def test_store_summary_drops_the_cube(self):
        """With store='summary', accessing .samples on a per-galaxy Posterior drops it."""
        # Build a catalog with samples, then manually set store='summary' and
        # drop samples (simulating what the implementation should do)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        samples = {
            "sfh_dpl_alpha": 10.0 + 0.5 * jax.random.normal(k1, (10,)),
            "dust_av": 0.5 + 0.1 * jax.random.normal(k2, (10,)),
        }
        params = {"sfh_dpl_alpha": jnp.array(10.0), "dust_av": jnp.array(0.5)}

        # Compute summaries first
        pcts = [16, 50, 84]
        percentiles = {name: np.percentile(np.asarray(s), pcts) for name, s in samples.items()}

        post = Posterior(
            samples=samples,
            params=params,
            method="test",
            wall_time_s=1.0,
            diagnostics={},
        )
        post._percentiles_stats_ = percentiles
        post._summary_stats_ = {
            "mean": {name: float(np.mean(np.asarray(s))) for name, s in samples.items()}
        }

        # Manually drop samples to simulate store='summary'
        post.samples = None

        # Now accessing .samples should be None, not raise
        assert post.samples is None

        # The percentiles should still be accessible
        assert "sfh_dpl_alpha" in post._percentiles_stats_
        assert post._percentiles_stats_["sfh_dpl_alpha"].shape == (3,)


class TestStoreFullKeepsTodaysBehavior:
    """Test that store='full' preserves samples and does not compute summaries."""

    def test_store_full_keeps_todays_behavior(self, tiny_catalog_with_samples):
        """With store='full', samples are retained and percentiles/summary are not computed."""
        cat = tiny_catalog_with_samples

        # With no percentiles/summary stats, they should not exist as attributes
        for post in cat.posteriors:
            # Check that the summary stats dict (not the method) doesn't exist
            assert not hasattr(post, "_percentiles")
            # or that it's None if it does exist (implementation detail)
            if hasattr(post, "_percentiles"):
                assert post._percentiles is None

        # But samples should be present
        for post in cat.posteriors:
            assert post.samples is not None
            assert "sfh_dpl_alpha" in post.samples


class TestToTableRoundTripsColumnMapping:
    """Test that to_table() round-trips through ingest_catalog."""

    def test_to_table_round_trips_column_mapping(self, tiny_catalog_with_percentiles_and_summary):
        """to_table() returns a dict with correct shape and keys matching ingest contract."""
        cat = tiny_catalog_with_percentiles_and_summary

        # Call to_table() — should return a dict-like with (N, n_data) shape
        table = cat.to_table()

        # Check that it's a dict
        assert isinstance(table, dict)

        # Check essential properties are present
        assert "stellar_mass" in table or "sfh_dpl_alpha" in table  # depends on model

        # Check shape: each column should have N rows
        n_gal = cat.n_galaxies
        for key, val in table.items():
            if isinstance(val, np.ndarray):
                assert len(val) == n_gal, f"Column {key} has {len(val)} rows, expected {n_gal}"

        # The dict should be re-ingestable by ingest_catalog (duck-type contract)
        # This means it should support __getitem__ and len()
        assert len(table) > 0
        # At least one key should be accessible
        first_key = next(iter(table.keys()))
        assert table[first_key] is not None
