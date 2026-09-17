# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end posterior identity test for profile_mass: PIT uniformity check.

This test verifies that the profiled mass posterior (analytically marginalized
amplitude, reinserted via inverse-CDF per sample in finalize_profile_mass) is
statistically equivalent to the unprofiled mass posterior (sampled jointly).

The PRIMARY TEST uses a Probability Integral Transform (PIT) approach:
1. Run ONE unprofiled fit (profile_mass=False) to get posterior samples.
2. For each sample i with mass ell_i and parameters theta_i:
   - Evaluate the profiled arm's exact conditional CDF at theta_i.
   - Compute F(ell_i | theta_i, d) using the same quadrature as _sample_log_mass.
3. If the conditional p(ell | theta, d) is correct, the u_i = F(ell_i|...) are
   Uniform(0, 1).
4. Test with one-sample KS against scipy.stats.uniform.

This directly validates the mechanism that finalize_profile_mass uses for
reinsertion. It requires only one fit (no Monte Carlo error in the CDF),
exercises the exact conditional, and is fast.

The SECONDARY (SMOKE) TEST compares profiled vs unprofiled arms on the same
data with matched seeds and budgets, using two-sample KS and percentile
agreement. This is weaker (smoother result) but catches plumbing errors.

NOTE: MCMC draws are autocorrelated. Before KS testing, draws are thinned to
approximate independence using effective sample size (ESS). Both the raw draw
count and the thinned count are reported.

Reference: Talts et al. (2018) "Validating Bayesian inference algorithms with
simulation-based calibration"; Cook-Gelman-Rubin (2006) "Validation of software
for Bayesian models using posterior quantiles".

See tests/inference/test_profile_mass.py::TestMarginalLikelihoodCorrectness
for validation that the ell-MARGINAL likelihood is correct. This test validates
the ell-CONDITIONAL. Together they establish the joint posterior is correct.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.diagnostics import effective_sample_size
from scipy import stats

from tengri import (
    ForwardModel,
    SEDModel,
    recipes,
)
from tengri.inference.mass_profile import (
    _REINSERT_QUAD_NODES,
    _log_quadrature_terms,
    _profile_stats,
)
from tengri.observation import Observation, Photometry

pytestmark = pytest.mark.contract

_FILTERS_MINIMAL = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
_FILTERS_SPARSE = ["sdss_g", "sdss_r", "des_i"]
_MASS_NAME = "sfh_tsnorm_log_total_mass"


def _minimal_model(ssp_data):
    """Photometry-only model over 8 filters (high S/N, weak mass constraint)."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS_MINIMAL))
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())


def _sparse_model(ssp_data):
    """Photometry-only model over 3 filters (low S/N, skewed mass posterior)."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS_SPARSE))
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())


def _mock(model, *, seed: int, snr: float = 30.0):
    """Generate mock data and ground truth at a given seed and SNR."""
    from tengri import generate_mock

    key_truth, key_mock = jax.random.split(jax.random.PRNGKey(seed))
    truth = model.spec.sample(key_truth)
    mock = generate_mock(model, truth, key=key_mock, snr=snr)
    return truth, jnp.asarray(mock["flux_obs"]), jnp.asarray(mock["noise"])


def _thin_by_ess(samples, max_thinned_count=500):
    """Thin MCMC samples to approximate independence using ESS.

    Parameters
    ----------
    samples : ndarray, shape (n_draws,)
        MCMC samples, potentially autocorrelated.
    max_thinned_count : int, optional
        Maximum number of independent samples to keep.

    Returns
    -------
    thinned : ndarray
        Approximately independent samples, thinned to roughly min(ess,
        max_thinned_count) samples.
    n_kept : int
        Number of samples kept.
    ess_per_draw : float
        ESS / n_draws ratio (autocorrelation penalty).

    Notes
    -----
    The step size is computed as ceil(n / min(ess, max_thinned_count)) to
    ensure that the thinned array contains approximately the effective
    sample size worth of independent samples (capped at max_thinned_count).
    """
    # samples are shape (n_chains * n_samples,) = (n,)
    # Reshape for ESS computation if needed (ESS expects (n_chains, n_per_chain))
    n = samples.shape[0]
    if n < 10:
        return samples, n, 1.0

    # Try to reshape: assume roughly equal-sized chains
    # If not possible, treat as a single chain
    try:
        # Try 1 chain
        samples_2d = np.asarray(samples).reshape(1, -1)
    except ValueError:
        samples_2d = np.asarray(samples).reshape(1, -1)

    ess = float(effective_sample_size(jnp.asarray(samples_2d)))
    ess_per_draw = ess / float(n)
    # Thin to roughly min(ess, max_thinned_count) independent samples.
    # Use ceiling division to ensure we actually thin: e.g., if n=200 and
    # ess=150, step = ceil(200/150) = 2, keeping 100 truly independent samples.
    target_count = min(ess, max_thinned_count)
    step = max(1, -(-int(float(n)) // int(target_count)))  # Ceiling division
    thinned = samples[::step]
    return thinned, len(thinned), ess_per_draw


def _evaluate_cdf_at_sample(sample_mass, sample_params, model, forward, flux, noise, data_type):
    """Evaluate the profiled conditional CDF at a single sample's mass.

    Computes F(ell | theta, d) using the exact quadrature from the profiled arm.
    Returns u = F(sample_mass | sample_params, data).

    Parameters
    ----------
    sample_mass : float
        The unprofiled sample's log10(mass) value.
    sample_params : dict
        Physical parameter dict (NOT standardized, complete with all free and fixed params).
    model : SEDModel
    forward : ForwardModel
    flux, noise : arrays
        Data.
    data_type : str
        "photometry", "spectroscopy", or "joint".

    Returns
    -------
    u : float
        The CDF value at sample_mass. Should be uniform [0, 1] if conditional
        is correct.
    """
    # Call _profile_stats to get the quadrature parameters A_ref, a_star
    A_ref, a_star, _chi2_min, ell_ref = _profile_stats(
        model, _MASS_NAME, sample_params, flux, noise, data_type=data_type
    )

    # Get mass prior and bounds
    mass_prior = model.spec.get_distribution(_MASS_NAME)
    ell_lo, ell_hi = mass_prior.bounds

    # Build quadrature terms using the same finer grid as _sample_log_mass
    u_nodes, log_terms = _log_quadrature_terms(
        A_ref, a_star, ell_lo, ell_hi, mass_prior, ell_ref, quad_nodes=_REINSERT_QUAD_NODES
    )

    # Reconstruct absolute ell nodes from offset and reference
    ell_nodes = u_nodes + ell_ref

    # Form normalized CDF from softmax + cumsum (same as _sample_log_mass)
    probs = jax.nn.softmax(log_terms)
    cdf = jnp.cumsum(probs)
    cdf_normalized = cdf / cdf[-1]

    # Interpolate CDF at the sample's mass to get u
    u = float(jnp.interp(sample_mass, ell_nodes, cdf_normalized))

    return u


class TestProfileMassPosteriorIdentityPIT:
    """PIT uniformity test: the profiled conditional CDF is correct."""

    @pytest.fixture(params=[0, 1, 2, 3, 4, 5])
    def seed(self, request):
        """All six seeds must pass (no seed selection)."""
        return request.param

    def test_pit_uniformity_minimal_model(self, ssp_data_fsps, seed):
        """PIT test on the minimal (8-filter) photometry model.

        Procedure:
        1. Run ONE unprofiled fit to get posterior samples.
        2. For each sample, evaluate the profiled conditional CDF at that sample's mass.
        3. If the conditional is correct, the u_i are Uniform(0, 1).
        4. Thin by ESS and test with one-sample KS.

        The model has many photometry bands so the mass amplitude is well
        determined (A is large). This tests that the quadrature is accurate
        in the well-determined regime.
        """
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _truth, flux, noise = _mock(model, seed=seed, snr=30.0)

        # One unprofiled fit
        fit_kw = dict(
            method="mcmc_nuts",
            dense_mass_matrix=False,
            n_warmup=100,
            n_samples=200,
            n_chains=1,
        )

        t0 = time.time()
        key_fit = jax.random.PRNGKey(seed + 10000)
        post = forward.fit(flux, noise, profile_mass=False, key=key_fit, **fit_kw)
        elapsed_fit = time.time() - t0

        # Extract mass samples and other parameters
        mass_samples = np.asarray(post.samples[_MASS_NAME]).flatten()
        n_raw = len(mass_samples)

        # For each sample, evaluate the profiled conditional CDF
        t0 = time.time()
        u_samples = np.empty(n_raw)

        for i, m_i in enumerate(mass_samples):
            # Reconstruct physical params from the sample
            # Get free params (excluding mass)
            sample_params = {}
            for name in post.samples:
                if name != _MASS_NAME:
                    sample_params[name] = float(post.samples[name].flatten()[i])

            # Get fixed values from the model
            fixed_vals = model.spec.get_fixed_values()
            for name, val in fixed_vals.items():
                sample_params[name] = float(val) if hasattr(val, "__float__") else val

            # Add dummy mass value (required by model.predict but not used by _profile_stats)
            sample_params[_MASS_NAME] = 0.0

            # Evaluate CDF at this sample
            u_samples[i] = _evaluate_cdf_at_sample(
                float(m_i), sample_params, model, forward, flux, noise, data_type="photometry"
            )

        elapsed_cdf = time.time() - t0

        # Thin by ESS to approximate independence
        u_thinned, n_thinned, ess_ratio = _thin_by_ess(u_samples, max_thinned_count=500)

        # KS test
        ks_stat, ks_pvalue = stats.kstest(u_thinned, stats.uniform(0, 1).cdf)

        print(
            f"\nSeed {seed}: Minimal model (8 filters, SNR=30).\n"
            f"  Fit time: {elapsed_fit:.1f}s. CDF eval time: {elapsed_cdf:.1f}s.\n"
            f"  Raw draws: {n_raw}. ESS ratio: {ess_ratio:.3f}. Thinned: {n_thinned}.\n"
            f"  KS stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}.\n"
            f"  u_samples: mean={np.mean(u_thinned):.3f}, std={np.std(u_thinned):.3f}.\n"
        )

        assert ks_pvalue >= 0.01, (
            f"Seed {seed}: KS p-value {ks_pvalue:.4f} on minimal model. "
            f"PIT u samples are not uniform — conditional may be incorrect."
        )

    def test_pit_uniformity_sparse_model(self, ssp_data_fsps, seed):
        """PIT test on the sparse (3-filter) model with low SNR.

        This fixture has a skewed, poorly-constrained mass posterior (few bands,
        low SNR = low A). Tests that the quadrature is accurate even when the
        posterior is broad and non-Gaussian.
        """
        model = _sparse_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _truth, flux, noise = _mock(model, seed=seed, snr=10.0)

        fit_kw = dict(
            method="mcmc_nuts",
            dense_mass_matrix=False,
            n_warmup=100,
            n_samples=200,
            n_chains=1,
        )

        t0 = time.time()
        key_fit = jax.random.PRNGKey(seed + 20000)
        post = forward.fit(flux, noise, profile_mass=False, key=key_fit, **fit_kw)
        elapsed_fit = time.time() - t0

        mass_samples = np.asarray(post.samples[_MASS_NAME]).flatten()
        n_raw = len(mass_samples)

        t0 = time.time()
        u_samples = np.empty(n_raw)

        for i, m_i in enumerate(mass_samples):
            # Reconstruct physical params from the sample
            sample_params = {}
            for name in post.samples:
                if name != _MASS_NAME:
                    sample_params[name] = float(post.samples[name].flatten()[i])

            # Get fixed values from the model
            fixed_vals = model.spec.get_fixed_values()
            for name, val in fixed_vals.items():
                sample_params[name] = float(val) if hasattr(val, "__float__") else val

            # Add dummy mass value (required by model.predict but not used by _profile_stats)
            sample_params[_MASS_NAME] = 0.0

            u_samples[i] = _evaluate_cdf_at_sample(
                float(m_i), sample_params, model, forward, flux, noise, data_type="photometry"
            )

        elapsed_cdf = time.time() - t0

        u_thinned, n_thinned, ess_ratio = _thin_by_ess(u_samples, max_thinned_count=500)

        ks_stat, ks_pvalue = stats.kstest(u_thinned, stats.uniform(0, 1).cdf)

        print(
            f"\nSeed {seed}: Sparse model (3 filters, SNR=10).\n"
            f"  Fit time: {elapsed_fit:.1f}s. CDF eval time: {elapsed_cdf:.1f}s.\n"
            f"  Raw draws: {n_raw}. ESS ratio: {ess_ratio:.3f}. Thinned: {n_thinned}.\n"
            f"  KS stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}.\n"
            f"  u_samples: mean={np.mean(u_thinned):.3f}, std={np.std(u_thinned):.3f}.\n"
        )

        assert ks_pvalue >= 0.01, (
            f"Seed {seed}: KS p-value {ks_pvalue:.4f} on sparse model. "
            f"PIT u samples are not uniform — conditional may be incorrect."
        )


class TestProfileMassPosteriorIdentitySmoke:
    """Smoke test: profiled vs unprofiled mass posteriors agree."""

    @pytest.fixture(params=[0, 1, 2, 3, 4, 5])
    def seed(self, request):
        """All six seeds must pass."""
        return request.param

    def test_profiled_unprofiled_agreement(self, ssp_data_fsps, seed):
        """Profiled and unprofiled mass posteriors agree on percentiles.

        Procedure:
        1. Generate one mock dataset.
        2. Fit twice: once with profile_mass=False, once with True.
        3. Use matched seeds and sampler budgets.
        4. Compare mass posteriors: two-sample KS test + percentile agreement.

        This is a smoke test with looser tolerance than the PIT: it exercises
        the code path and catches gross plumbing errors, but is not as stringent.
        """
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _truth, flux, noise = _mock(model, seed=seed, snr=30.0)

        fit_kw = dict(
            method="mcmc_nuts",
            dense_mass_matrix=False,
            n_warmup=100,
            n_samples=200,
            n_chains=1,
        )

        key_base = jax.random.PRNGKey(seed + 30000)
        t0 = time.time()
        post_off = forward.fit(flux, noise, profile_mass=False, key=key_base, **fit_kw)
        post_on = forward.fit(flux, noise, profile_mass=True, key=key_base, **fit_kw)
        elapsed = time.time() - t0

        mass_off = np.asarray(post_off.samples[_MASS_NAME]).flatten()
        mass_on = np.asarray(post_on.samples[_MASS_NAME]).flatten()

        # Thin by ESS to approximate independence. MCMC draws are autocorrelated;
        # the two-sample KS test assumes independent samples. Thinning to roughly
        # the effective sample size validates the statistic.
        mass_off_thinned, n_off_thinned, ess_off_ratio = _thin_by_ess(
            mass_off, max_thinned_count=500
        )
        mass_on_thinned, n_on_thinned, ess_on_ratio = _thin_by_ess(mass_on, max_thinned_count=500)

        # Two-sample KS test
        _ks_stat, ks_pvalue = stats.ks_2samp(mass_off_thinned, mass_on_thinned)

        # Percentile agreement (16, 50, 84)
        perc_off = np.percentile(mass_off, [16, 50, 84])
        perc_on = np.percentile(mass_on, [16, 50, 84])
        perc_diff = np.abs(perc_off - perc_on)

        print(
            f"\nSeed {seed}: Smoke test, both fits in {elapsed:.1f}s.\n"
            f"  Unprofiled: raw={len(mass_off)}, "
            f"ESS ratio={ess_off_ratio:.3f}, thinned={n_off_thinned}.\n"
            f"  Profiled:   raw={len(mass_on)}, "
            f"ESS ratio={ess_on_ratio:.3f}, thinned={n_on_thinned}.\n"
            f"  Unprofiled percentiles [16, 50, 84]: {perc_off}.\n"
            f"  Profiled percentiles:                {perc_on}.\n"
            f"  Difference:                          {perc_diff}.\n"
            f"  Two-sample KS p-value: {ks_pvalue:.4f}.\n"
        )

        # Loose smoke-test thresholds
        assert np.all(perc_diff < 0.5), (
            f"Seed {seed}: percentile diff {perc_diff} exceeds 0.5 dex. "
            f"Mass posteriors may differ significantly."
        )
        assert ks_pvalue >= 0.001, (
            f"Seed {seed}: KS p-value {ks_pvalue:.4f} suggests distributions differ."
        )
