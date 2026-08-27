# SPDX-License-Identifier: BSD-3-Clause
"""Tests for split-Rhat (Gelman-Rubin) chain convergence diagnostic.

References
----------
- Gelman, A., Rubin, D. B., 1992, Statistical Science, 7, 457.
- Vehtari, A. et al., 2021, Bayesian Analysis, 16, 667 (rank-normalized split-Rhat).
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.analysis.diagnostics.autocorrelation import (
    rank_normalized_rhat,
    rhat,
    split_rhat,
)

# One assignment, not two: Python rebinds the name, so a second
# `pytestmark = ...` silently discarded the taxonomy marker.
pytestmark = [pytest.mark.bounds, pytest.mark.unit]


# ── split_rhat (single chain) ────────────────────────────────────────


class TestSplitRhatScalar:
    def test_white_noise_rhat_close_to_one(self):
        """A long well-mixed chain should yield R̂ ≈ 1."""
        rng = np.random.default_rng(0)
        chain = rng.normal(size=4000)
        r = split_rhat(chain)
        assert abs(r - 1.0) < 0.05

    def test_drift_between_halves_increases_rhat(self):
        """A chain whose two halves have different means yields R̂ > 1."""
        rng = np.random.default_rng(1)
        first = rng.normal(loc=0.0, size=1000)
        second = rng.normal(loc=1.5, size=1000)
        chain = np.concatenate([first, second])
        r = split_rhat(chain)
        assert r > 1.1

    def test_perfect_constant_returns_nan_or_one(self):
        """Zero-variance chain has no convergence info — should be NaN."""
        chain = np.full(1000, 3.14)
        r = split_rhat(chain)
        assert np.isnan(r)

    def test_short_chain_returns_nan(self):
        """Chain of length < 4 cannot be split + variance-estimated; return NaN."""
        chain = np.array([1.0, 2.0, 3.0])
        r = split_rhat(chain)
        assert np.isnan(r)

    def test_two_chains_input(self):
        """Passing a (m, n) array uses the m chains directly (no split)."""
        rng = np.random.default_rng(2)
        chains = rng.normal(size=(4, 1000))
        r = split_rhat(chains)
        assert abs(r - 1.0) < 0.05

    def test_two_chains_with_offsets(self):
        """Chains with different means should give R̂ > 1."""
        rng = np.random.default_rng(3)
        offsets = np.array([-1.0, 0.0, 1.0, 2.0]).reshape(-1, 1)
        chains = rng.normal(size=(4, 1000)) + offsets
        r = split_rhat(chains)
        assert r > 1.5


# ── rhat (dict API across parameters) ────────────────────────────────


class TestRhatDict:
    def test_dict_in_dict_out(self):
        rng = np.random.default_rng(4)
        chains = {
            "alpha": rng.normal(size=2000),
            "beta": rng.normal(size=2000),
        }
        out = rhat(chains)
        assert set(out.keys()) == {"alpha", "beta"}
        for v in out.values():
            assert abs(v - 1.0) < 0.05

    def test_excludes_default_psd_xi(self):
        """High-D latent fields are skipped by default."""
        rng = np.random.default_rng(5)
        chains = {
            "alpha": rng.normal(size=2000),
            "psd_xi": rng.normal(size=(2000, 100)),
        }
        out = rhat(chains)
        assert "psd_xi" not in out
        assert "alpha" in out

    def test_excludes_static_parameter(self):
        """Zero-variance parameters are skipped (not converged-vs-not informative)."""
        chains = {
            "frozen": np.full(1000, 0.5),
            "live": np.random.default_rng(6).normal(size=1000),
        }
        out = rhat(chains)
        assert "frozen" not in out
        assert "live" in out


# ── Posterior.rhat() integration ─────────────────────────────────────


class TestPosteriorRhatMethod:
    def test_map_raises_value_error(self):
        import jax.numpy as jnp

        from tengri.inference.posterior import Posterior

        p = Posterior(
            samples=None,
            params={"x": jnp.array(1.0)},
            method="MAP",
            wall_time_s=0.1,
            diagnostics={},
        )
        with pytest.raises(ValueError, match="samples"):
            p.rhat()

    def test_sampling_returns_dict_with_rhat_close_to_one(self):
        import jax.numpy as jnp

        from tengri.inference.posterior import Posterior

        rng = np.random.default_rng(7)
        n = 2000
        p = Posterior(
            samples={"x": jnp.asarray(rng.normal(size=n))},
            params={"x": jnp.array(0.0)},
            method="mcmc_nuts",
            wall_time_s=10.0,
            diagnostics={},
        )
        out = p.rhat()
        assert "x" in out
        assert abs(out["x"] - 1.0) < 0.05


# ── Vehtari+2021 rank-normalized folded split-Rhat ────────────────────


class TestRankNormalizedRhat:
    def test_white_noise_close_to_one(self):
        rng = np.random.default_rng(100)
        chain = rng.normal(size=4000)
        r = rank_normalized_rhat(chain)
        assert abs(r - 1.0) < 0.05

    def test_drift_between_halves_flagged(self):
        rng = np.random.default_rng(101)
        chain = np.concatenate([rng.normal(size=1000), rng.normal(loc=2.0, size=1000)])
        r = rank_normalized_rhat(chain)
        assert r > 1.1

    def test_variance_only_drift_flagged_by_folded(self):
        """Chain whose two halves have identical mean but different variance.
        Classical R̂ misses this; the folded variant catches it."""
        rng = np.random.default_rng(102)
        chain = np.concatenate(
            [rng.normal(scale=1.0, size=2000), rng.normal(scale=4.0, size=2000)]
        )
        r_classical = split_rhat(chain)
        r_rank = rank_normalized_rhat(chain)
        # Classical may or may not flag; rank-normalized folded must flag.
        assert r_rank > 1.05, (
            f"rank-normalized R̂={r_rank:.3f} should flag scale drift; classical={r_classical:.3f}"
        )

    def test_heavy_tailed_well_mixed_robust(self):
        """Cauchy-distributed well-mixed chain — classical R̂ noisy, rank version stable."""
        rng = np.random.default_rng(103)
        chain = rng.standard_cauchy(size=4000)
        # Filter wild outliers so classical R̂ doesn't blow up; just sanity check
        # that rank version stays near 1.
        r_rank = rank_normalized_rhat(chain)
        assert abs(r_rank - 1.0) < 0.10

    def test_returns_nan_for_short_or_constant(self):
        assert np.isnan(rank_normalized_rhat(np.full(100, 0.5)))
        assert np.isnan(rank_normalized_rhat(np.array([1.0, 2.0])))

    def test_two_chains_input(self):
        rng = np.random.default_rng(104)
        chains = rng.normal(size=(4, 1000))
        r = rank_normalized_rhat(chains)
        assert abs(r - 1.0) < 0.05
