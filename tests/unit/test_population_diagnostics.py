"""Tests for PopulationPosterior.population_diagnostics()."""

from __future__ import annotations

import numpy as np

from tengri.inference.hierarchical import PopulationPosterior


def _fake_chain(n: int = 200, mean: float = 1.0, scale: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(0)
    return mean + scale * rng.standard_normal(n)


def test_shared_only_returns_rhat_ess() -> None:
    post = PopulationPosterior(
        shared_samples={
            "sfh_field_psd_sigma": _fake_chain(),
            "sfh_field_psd_tau_myr": _fake_chain(mean=50.0, scale=2.0),
        },
        shared_params={},
        individual_samples=None,
        method="test",
    )
    diag = post.population_diagnostics()
    assert "shared" in diag
    assert "per_galaxy" not in diag
    for name in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
        assert name in diag["shared"]
        assert "rhat" in diag["shared"][name]
        assert "ess" in diag["shared"][name]
        # Single chain on a stationary signal ⇒ rhat ≈ 1, ess > 0.
        assert 0.9 < diag["shared"][name]["rhat"] < 1.5
        assert diag["shared"][name]["ess"] > 0


def test_per_galaxy_aggregation() -> None:
    individual = [
        {"met_logzsol": _fake_chain(mean=0.0, scale=0.05)},
        {"met_logzsol": _fake_chain(mean=-0.1, scale=0.04)},
        {"met_logzsol": _fake_chain(mean=0.05, scale=0.06)},
    ]
    post = PopulationPosterior(
        shared_samples={"sfh_field_psd_sigma": _fake_chain()},
        shared_params={},
        individual_samples=individual,
        method="test",
    )
    diag = post.population_diagnostics()
    assert "per_galaxy" in diag
    pg = diag["per_galaxy"]["met_logzsol"]
    for key in ("rhat_p50", "rhat_p90", "rhat_max", "ess_p50", "ess_min"):
        assert key in pg
    assert pg["n_galaxies"] == 3
    assert pg["rhat_p50"] <= pg["rhat_p90"] <= pg["rhat_max"]
    assert pg["ess_min"] <= pg["ess_p50"]


def test_psd_xi_excluded_by_default() -> None:
    """The default exclude_prefixes drops psd_xi (GP latent fields)."""
    individual = [
        {
            "met_logzsol": _fake_chain(),
            "psd_xi": _fake_chain(),
        }
    ]
    post = PopulationPosterior(
        shared_samples={"sfh_field_psd_sigma": _fake_chain()},
        shared_params={},
        individual_samples=individual,
    )
    diag = post.population_diagnostics()
    assert "met_logzsol" in diag["per_galaxy"]
    assert "psd_xi" not in diag["per_galaxy"]


def test_empty_individual_samples_drops_per_galaxy_block() -> None:
    post = PopulationPosterior(
        shared_samples={"sfh_field_psd_sigma": _fake_chain()},
        shared_params={},
        individual_samples=[],
    )
    diag = post.population_diagnostics()
    assert "per_galaxy" not in diag
