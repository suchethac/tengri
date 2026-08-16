# SPDX-License-Identifier: BSD-3-Clause
"""Tests for catalog batch fitting: Posterior save/load, checkpoint resume, catalog_summary."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.forward.convenience import catalog_summary
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.contract

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def map_posterior():
    return Posterior(
        samples=None,
        params={
            "sfh_dpl_alpha": jnp.array(1.2),
            "dust_av": jnp.array(0.5),
        },
        method="MAP (Adam)",
        wall_time_s=1.5,
        diagnostics={"n_steps": 100, "final_loss": 5.2},
        loss_history=jnp.array([10.0, 5.0, 2.0]),
    )


@pytest.fixture
def sampling_posterior():
    key = jax.random.PRNGKey(42)
    n = 50
    k1, k2 = jax.random.split(key)
    return Posterior(
        samples={
            "sfh_dpl_alpha": 1.2 + 0.3 * jax.random.normal(k1, (n,)),
            "dust_av": 0.5 + 0.1 * jax.random.normal(k2, (n,)),
        },
        params={
            "sfh_dpl_alpha": jnp.array(1.2),
            "dust_av": jnp.array(0.5),
        },
        method="NUTS (BlackJAX)",
        wall_time_s=30.0,
        diagnostics={"n_divergent": 0, "chi2_dof": 1.05},
    )


@pytest.fixture
def sampling_posterior_with_elines():
    key = jax.random.PRNGKey(99)
    n = 30
    k1, _k2 = jax.random.split(key)
    return Posterior(
        samples={
            "sfh_dpl_alpha": 1.0 + 0.2 * jax.random.normal(k1, (n,)),
        },
        params={"sfh_dpl_alpha": jnp.array(1.0)},
        method="VI",
        wall_time_s=5.0,
        diagnostics={},
        eline_fluxes=jnp.ones((n, 3)),
        eline_names=("Halpha", "Hbeta", "OIII_5007"),
        eline_wavelengths=jnp.array([6564.61, 4862.68, 5008.24]),
    )


# ── Posterior.save / Posterior.load ───────────────────────────────


class TestPosteriorSaveLoad:
    def test_map_roundtrip(self, map_posterior):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.h5")
            map_posterior.save(path)
            loaded = Posterior.load(path)

        assert loaded.method == map_posterior.method
        assert loaded.wall_time_s == map_posterior.wall_time_s
        assert loaded.samples is None
        np.testing.assert_allclose(
            float(loaded.params["sfh_dpl_alpha"]),
            float(map_posterior.params["sfh_dpl_alpha"]),
        )
        np.testing.assert_allclose(
            np.asarray(loaded.loss_history),
            np.asarray(map_posterior.loss_history),
        )

    def test_sampling_roundtrip(self, sampling_posterior):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.h5")
            sampling_posterior.save(path)
            loaded = Posterior.load(path)

        assert loaded.method == sampling_posterior.method
        assert loaded.samples is not None
        for name in sampling_posterior.samples:
            np.testing.assert_allclose(
                np.asarray(loaded.samples[name]),
                np.asarray(sampling_posterior.samples[name]),
            )

    def test_diagnostics_roundtrip(self, sampling_posterior):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.h5")
            sampling_posterior.save(path)
            loaded = Posterior.load(path)

        assert loaded.diagnostics["n_divergent"] == 0
        np.testing.assert_allclose(loaded.diagnostics["chi2_dof"], 1.05)

    def test_nested_diagnostics(self):
        p = Posterior(
            samples=None,
            params={"x": jnp.array(1.0)},
            method="test",
            wall_time_s=0.1,
            diagnostics={
                "ess_bulk": {"x": 500.0, "y": 300.0},
                "final_loss": 2.5,
            },
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "nested.h5")
            p.save(path)
            loaded = Posterior.load(path)

        assert loaded.diagnostics["ess_bulk"]["x"] == 500.0
        assert loaded.diagnostics["ess_bulk"]["y"] == 300.0
        assert loaded.diagnostics["final_loss"] == 2.5

    def test_eline_roundtrip(self, sampling_posterior_with_elines):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "eline.h5")
            sampling_posterior_with_elines.save(path)
            loaded = Posterior.load(path)

        assert loaded.eline_names == ("Halpha", "Hbeta", "OIII_5007")
        np.testing.assert_allclose(
            np.asarray(loaded.eline_wavelengths),
            np.asarray(sampling_posterior_with_elines.eline_wavelengths),
        )
        np.testing.assert_allclose(
            np.asarray(loaded.eline_fluxes),
            np.asarray(sampling_posterior_with_elines.eline_fluxes),
        )

    def test_log_evidence_roundtrip(self):
        p = Posterior(
            samples=None,
            params={"x": jnp.array(1.0)},
            method="NSS",
            wall_time_s=60.0,
            diagnostics={},
            log_evidence=-42.5,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "evidence.h5")
            p.save(path)
            loaded = Posterior.load(path)

        np.testing.assert_allclose(loaded.log_evidence, -42.5)

    def test_no_log_evidence(self, map_posterior):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "no_ev.h5")
            map_posterior.save(path)
            loaded = Posterior.load(path)

        assert loaded.log_evidence is None

    def test_psd_xi_roundtrip(self):
        """2D psd_xi arrays survive save/load."""
        n, n_grid = 20, 64
        key = jax.random.PRNGKey(7)
        p = Posterior(
            samples={
                "sfh_dpl_alpha": jax.random.normal(key, (n,)),
                "psd_xi": jax.random.normal(key, (n, n_grid)),
            },
            params={"sfh_dpl_alpha": jnp.array(1.0)},
            method="NUTS",
            wall_time_s=10.0,
            diagnostics={},
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "xi.h5")
            p.save(path)
            loaded = Posterior.load(path)

        assert loaded.samples["psd_xi"].shape == (n, n_grid)
        np.testing.assert_allclose(
            np.asarray(loaded.samples["psd_xi"]),
            np.asarray(p.samples["psd_xi"]),
        )


# ── catalog_summary ───────────────────────────────────────────────


class TestCatalogSummary:
    def test_empty_results(self):
        assert catalog_summary([]) == {}

    def test_map_results(self):
        results = [
            Posterior(
                samples=None,
                params={"dust_av": jnp.array(float(v))},
                method="MAP",
                wall_time_s=1.0,
                diagnostics={"chi2_dof": 1.0 + 0.1 * i},
            )
            for i, v in enumerate([0.3, 0.5, 0.7])
        ]
        summary = catalog_summary(results, include_derived=False)
        np.testing.assert_allclose(summary["dust_av_p50"], [0.3, 0.5, 0.7])
        np.testing.assert_allclose(summary["dust_av_p16"], [0.3, 0.5, 0.7])
        np.testing.assert_allclose(summary["chi2_dof"], [1.0, 1.1, 1.2], atol=1e-6)

    def test_sampling_results(self):
        results = []
        for i in range(3):
            key = jax.random.PRNGKey(i)
            center = float(i) + 1.0
            samples = {"dust_av": center + 0.1 * jax.random.normal(key, (100,))}
            results.append(
                Posterior(
                    samples=samples,
                    params={"dust_av": jnp.array(center)},
                    method="NUTS",
                    wall_time_s=10.0,
                    diagnostics={},
                )
            )
        summary = catalog_summary(results, include_derived=False)
        assert "dust_av_p50" in summary
        assert "dust_av_p16" in summary
        assert "dust_av_p84" in summary
        assert len(summary["dust_av_p50"]) == 3
        np.testing.assert_allclose(summary["dust_av_p50"], [1.0, 2.0, 3.0], atol=0.1)

    def test_custom_percentiles(self):
        results = [
            Posterior(
                samples=None,
                params={"x": jnp.array(1.0)},
                method="MAP",
                wall_time_s=1.0,
                diagnostics={},
            )
        ]
        summary = catalog_summary(results, percentiles=(5.0, 50.0, 95.0), include_derived=False)
        assert "x_p5" in summary
        assert "x_p50" in summary
        assert "x_p95" in summary

    def test_chi2_dof_extracted(self):
        results = [
            Posterior(
                samples=None,
                params={"x": jnp.array(1.0)},
                method="MAP",
                wall_time_s=1.0,
                diagnostics={"chi2_dof": 2.5},
            )
        ]
        summary = catalog_summary(results, include_derived=False)
        np.testing.assert_allclose(summary["chi2_dof"], [2.5])

    def test_missing_chi2_is_nan(self):
        results = [
            Posterior(
                samples=None,
                params={"x": jnp.array(1.0)},
                method="MAP",
                wall_time_s=1.0,
                diagnostics={},
            )
        ]
        summary = catalog_summary(results, include_derived=False)
        assert np.isnan(summary["chi2_dof"][0])
