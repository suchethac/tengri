# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Sokal/Behroozi autocorrelation time estimator."""

import numpy as np
import pytest

from tengri.analysis.diagnostics.autocorrelation import (
    autocorrelation_at_lag,
    autocorrelation_time,
    autocorrelation_time_combined,
    check_chain_length,
    effective_sample_size,
)

pytestmark = pytest.mark.bounds

# ── Helpers ───────────────────────────────────────────────────────


def _make_ar1(n, phi, seed=42):
    """Generate AR(1) process: x[t] = phi * x[t-1] + eps."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _make_iid(n, seed=42):
    """Generate IID N(0,1) samples."""
    return np.random.default_rng(seed).standard_normal(n)


# ── autocorrelation_at_lag ────────────────────────────────────────


class TestAutocorrelationAtLag:
    def test_lag1_ar1_positive(self):
        """AR(1) with phi=0.9 should have high correlation at lag 1."""
        x = _make_ar1(10_000, phi=0.9)
        rho = autocorrelation_at_lag(x, lag=1)
        assert 0.85 < rho < 0.95

    def test_lag1_iid_near_zero(self):
        """IID samples should have near-zero autocorrelation."""
        x = _make_iid(10_000)
        rho = autocorrelation_at_lag(x, lag=1)
        assert rho < 0.05

    def test_absolute_mode(self):
        """Absolute mode should also detect correlations."""
        x = _make_ar1(10_000, phi=0.9)
        rho_abs = autocorrelation_at_lag(x, lag=1, absolute=True)
        # Absolute mode captures magnitude correlations
        assert rho_abs > 0.5

    def test_constant_chain_returns_zero(self):
        """Constant chain should return 0 (zero variance)."""
        x = np.ones(1000)
        rho = autocorrelation_at_lag(x, lag=1)
        assert rho == 0.0

    def test_lag_clamped(self):
        """Lag larger than chain should be clamped."""
        x = _make_iid(100)
        # Should not raise, just clamp
        rho = autocorrelation_at_lag(x, lag=200)
        assert isinstance(rho, float)


# ── autocorrelation_time (Sokal's method) ─────────────────────────


class TestAutocorrelationTime:
    def test_iid_tau_near_one(self):
        """IID samples should have τ ≈ 1."""
        x = _make_iid(10_000)
        tau = autocorrelation_time(x)
        assert 0.8 < tau < 2.0

    def test_ar1_tau_matches_analytic(self):
        """AR(1) with phi=0.9: analytic τ = (1+phi)/(1-phi) = 19.

        The Sokal estimator should get within ~30% for N=50000.
        """
        x = _make_ar1(50_000, phi=0.9)
        tau = autocorrelation_time(x)
        tau_analytic = (1 + 0.9) / (1 - 0.9)  # = 19
        assert tau_analytic * 0.6 < tau < tau_analytic * 1.5

    def test_ar1_absolute_mode(self):
        """Absolute-mode τ should also be > 1 for correlated chain."""
        x = _make_ar1(10_000, phi=0.9)
        tau_abs = autocorrelation_time(x, absolute=True)
        assert tau_abs > 3.0

    def test_short_chain(self):
        """Very short chain should not crash."""
        x = np.array([1.0, 2.0, 3.0])
        tau = autocorrelation_time(x)
        assert tau >= 1.0


# ── autocorrelation_time_combined ─────────────────────────────────


class TestAutocorrelationTimeCombined:
    def test_keys_present(self):
        x = _make_ar1(5000, phi=0.5)
        info = autocorrelation_time_combined(x)
        assert "tau_standard" in info
        assert "tau_absolute" in info
        assert "tau_max" in info
        assert "ess" in info
        assert "chain_converged" in info

    def test_tau_max_is_max(self):
        x = _make_ar1(5000, phi=0.9)
        info = autocorrelation_time_combined(x)
        assert info["tau_max"] == max(info["tau_standard"], info["tau_absolute"])

    def test_ess_equals_n_over_tau(self):
        x = _make_iid(5000)
        info = autocorrelation_time_combined(x)
        expected_ess = 5000 / info["tau_max"]
        assert abs(info["ess"] - expected_ess) < 0.01

    def test_converged_for_long_chain(self):
        """Long IID chain (τ≈1) should be converged (N >> 5τ)."""
        x = _make_iid(10_000)
        info = autocorrelation_time_combined(x)
        assert info["chain_converged"] is True

    def test_not_converged_for_short_correlated_chain(self):
        """Short highly-correlated chain should warn."""
        x = _make_ar1(50, phi=0.99)  # τ ≈ 199, way more than N/5=10
        info = autocorrelation_time_combined(x)
        assert info["chain_converged"] is False


# ── effective_sample_size (dict interface) ────────────────────────


class TestEffectiveSampleSize:
    def test_returns_dict(self):
        chains = {"a": _make_iid(5000), "b": _make_ar1(5000, phi=0.5)}
        result = effective_sample_size(chains)
        assert "a" in result
        assert "b" in result

    def test_excludes_psd_xi(self):
        chains = {"a": _make_iid(100), "psd_xi": _make_iid(100)}
        result = effective_sample_size(chains)
        assert "a" in result
        assert "psd_xi" not in result

    def test_skips_multidim(self):
        chains = {"a": _make_iid(100), "b": np.ones((100, 5))}
        result = effective_sample_size(chains)
        assert "a" in result
        assert "b" not in result

    def test_iid_ess_near_n(self):
        chains = {"a": _make_iid(10_000)}
        result = effective_sample_size(chains)
        assert result["a"]["ess"] > 5000  # τ ≈ 1 → ESS ≈ N


# ── check_chain_length ────────────────────────────────────────────


class TestCheckChainLength:
    def test_converged_chain(self):
        chains = {"a": _make_iid(10_000)}
        info = check_chain_length(chains, verbose=False)
        assert info["all_converged"] is True
        assert len(info["warnings"]) == 0

    def test_short_correlated_warns(self):
        chains = {"a": _make_ar1(50, phi=0.99)}
        info = check_chain_length(chains, verbose=False)
        assert info["all_converged"] is False
        assert len(info["warnings"]) > 0
        assert "a" in info["warnings"][0]

    def test_verbose_prints(self, capsys):
        chains = {"a": _make_iid(5000)}
        check_chain_length(chains, verbose=True)
        captured = capsys.readouterr()
        assert "Parameter" in captured.out
        assert "τ (std)" in captured.out
