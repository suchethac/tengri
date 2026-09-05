# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Ray Tracing Sampler integration."""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri.forward.sed_model import SEDModel
from tengri.inference.backends.mcmc.raytrace import sample_hamiltonian, sample_raytrace
from tengri.inference.fitter import Fitter
from tengri.inference.posterior import Posterior
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

# ── Helpers ───────────────────────────────────────────────────────


def _gaussian_log_prob(mean, cov_inv):
    """Return a log-probability function for a multivariate Gaussian."""

    def log_prob(x):
        diff = x - mean
        return -0.5 * diff @ cov_inv @ diff

    return log_prob


# ── Pure sampler tests (no SEDModel/Fitter dependency) ───────────────


class TestSampleRaytraceGaussian:
    """Sample from a 5D Gaussian, verify mean and std are close to truth."""

    def test_sample_raytrace_gaussian(self):
        D = 5
        key = jax.random.PRNGKey(0)
        true_mean = jnp.array([1.0, -0.5, 0.3, 2.0, -1.0])
        cov_inv = jnp.eye(D) * 4.0  # variance = 0.25, std = 0.5

        log_prob_fn = _gaussian_log_prob(true_mean, cov_inv)
        step_size = 0.03 * jnp.sqrt(float(D))

        chain, _log_likelihood, _accept_prob, _n_nonfinite = sample_raytrace(
            key=key,
            params_init=true_mean + 0.1,
            log_prob_fn=log_prob_fn,
            n_steps=100,
            n_leapfrog_steps=5,
            step_size=float(step_size),
        )

        # Discard burn-in
        chain_post = chain[20:]
        n_post = chain_post.shape[0]  # 80 samples

        # Mean recovery: tolerance = 5 × SE_mean = 5 × (std / sqrt(N))
        # true std = 0.5, N = 80 → SE_mean ≈ 0.056 → atol ≈ 0.28
        # atol=0.5 (= 1 full std) was far too loose (~9σ) — this catches broken samplers.
        recovered_mean = jnp.mean(chain_post, axis=0)
        se_mean = 0.5 / float(jnp.sqrt(n_post))  # std / sqrt(N)
        atol_mean = 5.0 * se_mean  # 5σ guard — robust to random variation
        assert jnp.allclose(recovered_mean, true_mean, atol=atol_mean), (
            f"Recovered mean {recovered_mean} too far from truth {true_mean} "
            f"(atol={atol_mean:.3f} = 5×SE_mean, true std=0.5, N={n_post})"
        )

        # Std recovery: true std = 0.5; accept [0.25, 1.0] (factor-of-2 band)
        # Previous bounds (0.1, 2.0) allowed 4× error each direction.
        recovered_std = jnp.std(chain_post, axis=0)
        assert jnp.all(recovered_std > 0.25), (
            f"Recovered std {recovered_std} < 0.25 — sampler may be stuck (true std=0.5)"
        )
        assert jnp.all(recovered_std < 1.0), (
            f"Recovered std {recovered_std} > 1.0 — sampler may not be converging (true std=0.5)"
        )


class TestSampleRaytraceAcceptance:
    """Verify acceptance rate is reasonable."""

    def test_sample_raytrace_acceptance(self):
        D = 5
        key = jax.random.PRNGKey(1)
        mean = jnp.zeros(D)
        cov_inv = jnp.eye(D)

        log_prob_fn = _gaussian_log_prob(mean, cov_inv)
        step_size = 0.03 * jnp.sqrt(float(D))

        _chain, _log_likelihood, accept_prob, _n_nonfinite = sample_raytrace(
            key=key,
            params_init=jnp.zeros(D),
            log_prob_fn=log_prob_fn,
            n_steps=80,
            n_leapfrog_steps=5,
            step_size=float(step_size),
        )

        mean_accept = float(jnp.mean(accept_prob))
        assert mean_accept > 0.3, f"Acceptance rate {mean_accept:.2%} is too low (expected >30%)"


class TestSampleRaytraceShapes:
    """Check chain shape is (n_steps, D)."""

    def test_sample_raytrace_returns_correct_shapes(self):
        D = 5
        n_steps = 50
        key = jax.random.PRNGKey(2)
        mean = jnp.zeros(D)
        cov_inv = jnp.eye(D)

        log_prob_fn = _gaussian_log_prob(mean, cov_inv)
        step_size = 0.03 * jnp.sqrt(float(D))

        chain, log_likelihood, accept_prob, _n_nonfinite = sample_raytrace(
            key=key,
            params_init=jnp.zeros(D),
            log_prob_fn=log_prob_fn,
            n_steps=n_steps,
            n_leapfrog_steps=5,
            step_size=float(step_size),
        )

        assert chain.shape == (n_steps, D), (
            f"Chain shape {chain.shape} != expected ({n_steps}, {D})"
        )
        assert log_likelihood.shape == (n_steps,), (
            f"Log-likelihood shape {log_likelihood.shape} != ({n_steps},)"
        )
        assert accept_prob.shape == (n_steps,), (
            f"Accept prob shape {accept_prob.shape} != ({n_steps},)"
        )


class TestRaytraceVsHmcGaussian:
    """Both samplers should recover similar means on a simple Gaussian."""

    def test_raytrace_vs_hmc_gaussian(self):
        D = 3
        true_mean = jnp.array([1.0, 0.0, -1.0])
        cov_inv = jnp.eye(D) * 2.0  # std = 1/sqrt(2) ~ 0.71

        log_prob_fn = _gaussian_log_prob(true_mean, cov_inv)
        step_size = 0.03 * jnp.sqrt(float(D))
        n_steps = 100
        n_leapfrog = 5
        burnin = 30

        # Ray tracing
        key_rt = jax.random.PRNGKey(10)
        chain_rt, _, _, _ = sample_raytrace(
            key=key_rt,
            params_init=true_mean + 0.05,
            log_prob_fn=log_prob_fn,
            n_steps=n_steps,
            n_leapfrog_steps=n_leapfrog,
            step_size=float(step_size),
            sample_hmc=False,
        )
        mean_rt = jnp.mean(chain_rt[burnin:], axis=0)

        # HMC
        key_hmc = jax.random.PRNGKey(20)
        chain_hmc, _, _ = sample_hamiltonian(
            key=key_hmc,
            params_init=true_mean + 0.05,
            log_prob_fn=log_prob_fn,
            n_steps=n_steps,
            n_leapfrog_steps=n_leapfrog,
            step_size=float(step_size),
        )
        mean_hmc = jnp.mean(chain_hmc[burnin:], axis=0)

        # Both should be within 1.0 of the true mean
        assert jnp.allclose(mean_rt, true_mean, atol=1.0), (
            f"Ray tracing mean {mean_rt} too far from truth {true_mean}"
        )
        assert jnp.allclose(mean_hmc, true_mean, atol=1.0), (
            f"HMC mean {mean_hmc} too far from truth {true_mean}"
        )

        # And they should agree with each other within ~1.5
        assert jnp.allclose(mean_rt, mean_hmc, atol=1.5), (
            f"Ray tracing mean {mean_rt} and HMC mean {mean_hmc} differ by more than expected"
        )


# ── Fitter integration tests (require SSP data) ───────────────────

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()


@pytest.fixture(scope="module")
def fitter_setup(ssp_data_wne, sdss_filters):
    """Create a simple SEDModel/Fitter setup for integration tests."""
    ssp = ssp_data_wne
    filters = sdss_filters

    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=-0.3,
        dust_tau_bc=0.3,
        dust_tau_diff=0.2,
        dust_slope=-0.7,
        redshift=0.1,
    )
    model = SEDModel(spec, ssp, filters=filters)

    true_params = {
        "sfh_dpl_alpha": 1.5,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 5.0,
        # 1e10 Msun, mid-prior. The comment here used to read "log10(10 Msun/yr)",
        # which is the pre-#369 meaning of this name: that rename turned
        # log10(SFR) into log10(M*), and c66c0aff0 (#1839) converted the prior
        # above to the declared Uniform(7.0, 12.5) without converting this truth.
        # A truth outside its own prior gives MAP nothing to find, so the chain
        # started at a boundary and ray tracing reported 0% acceptance.
        "sfh_dpl_log_total_mass": 10.0,
        # free (it carries a prior) but never given a truth value — the forward
        # used to substitute the spec default silently. Say it out loud (#1021).
        "sfh_dpl_age_gyr": float(spec.get_distribution("sfh_dpl_age_gyr").default),
        "met_logzsol": -0.3,
        "dust_tau_bc": 0.3,
        "dust_tau_diff": 0.2,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }
    mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(0))
    fitter = Fitter(model, mock.flux_obs, mock.noise)
    return fitter, model, mock, true_params


@pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")
class TestFitterRaytraceMethod:
    """Integration test: run fitter.run('raytrace') and verify Posterior."""

    def test_fitter_raytrace_method(self, fitter_setup):
        fitter, _model, _mock, _true_params = fitter_setup

        result = fitter.run(
            "mcmc_raytrace",
            n_steps=60,
            n_leapfrog_steps=5,
            n_burnin=10,
            verbose=False,
            key=jax.random.PRNGKey(99),
        )

        # Returns a Posterior
        assert isinstance(result, Posterior)

        # Method string is set
        assert "Ray Tracing" in result.method

        # Has samples (not MAP)
        assert result.samples is not None
        n_expected = 60  # n_steps (burn-in is separate)
        for name in fitter._free_names:
            assert name in result.samples, f"Missing samples for {name}"
            assert result.samples[name].shape[0] == n_expected, (
                f"Expected {n_expected} samples for {name}, got {result.samples[name].shape[0]}"
            )

        # Has params (posterior mean)
        assert result.params is not None
        for name in fitter._free_names:
            assert name in result.params

        # Diagnostics present
        assert "accept_rate" in result.diagnostics
        assert "n_samples" in result.diagnostics
        assert result.diagnostics["n_samples"] == n_expected

        # Wall time recorded
        assert result.wall_time_s > 0

        # Samples are in physical (bounded) space
        for name in fitter._free_names:
            lo, hi = fitter._bounds[name]
            samples = result.samples[name]
            assert jnp.all(samples >= lo), f"{name} has samples below lower bound {lo}"
            assert jnp.all(samples <= hi), f"{name} has samples above upper bound {hi}"


@pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")
class TestFitterRaytraceInitFromMap:
    """Test init_from chaining: MAP -> Ray Tracing."""

    def test_fitter_raytrace_init_from_map(self, fitter_setup):
        fitter, _model, _mock, _true_params = fitter_setup

        # First run MAP
        map_result = fitter.run(
            "map",
            n_steps=200,
            verbose=False,
            key=jax.random.PRNGKey(10),
        )
        assert isinstance(map_result, Posterior)
        assert map_result.samples is None  # MAP has no samples

        # Chain into Ray Tracing
        rt_result = fitter.run(
            "mcmc_raytrace",
            init_from=map_result,
            n_steps=60,
            n_leapfrog_steps=5,
            n_burnin=10,
            verbose=False,
            key=jax.random.PRNGKey(11),
        )
        assert isinstance(rt_result, Posterior)
        assert rt_result.samples is not None
        assert "Ray Tracing" in rt_result.method

        n_expected = 60  # n_steps (burn-in is separate)
        assert rt_result.diagnostics["n_samples"] == n_expected

        # Acceptance rate should be non-zero when initialized from MAP
        # (threshold lowered because short chains with few leapfrog steps
        # can have variable acceptance rates)
        assert rt_result.diagnostics["accept_rate"] > 0.01, (
            f"Accept rate {rt_result.diagnostics['accept_rate']:.2%} "
            f"unexpectedly low when initialized from MAP"
        )


# ── KDK integrator tests ──────────────────────────────────────────


class TestKDKIntegrator:
    """Verify KDK integrator produces valid samples."""

    def test_kdk_raytrace_gaussian(self):
        """KDK should recover Gaussian mean, matching DKD."""
        D = 5
        key = jax.random.PRNGKey(42)
        true_mean = jnp.zeros(D)
        cov_inv = jnp.eye(D)

        log_prob_fn = _gaussian_log_prob(true_mean, cov_inv)
        step_size = 0.03 * jnp.sqrt(float(D))

        chain, _lnl, accept_prob, _n_nonfinite = sample_raytrace(
            key=key,
            params_init=jnp.zeros(D),
            log_prob_fn=log_prob_fn,
            n_steps=200,
            n_leapfrog_steps=10,
            step_size=float(step_size),
            integrator="kdk",
        )

        # Acceptance should be reasonable
        mean_accept = float(jnp.mean(accept_prob))
        assert mean_accept > 0.3, f"KDK accept rate {mean_accept:.2%} too low"

        # Mean should be near zero
        chain_mean = jnp.mean(chain[50:], axis=0)
        assert jnp.allclose(chain_mean, true_mean, atol=0.8)

    def test_kdk_hmc_gaussian(self):
        """KDK HMC should also work."""
        D = 3
        key = jax.random.PRNGKey(7)
        true_mean = jnp.ones(D)
        cov_inv = jnp.eye(D) * 2.0

        log_prob_fn = _gaussian_log_prob(true_mean, cov_inv)

        chain, _lnl, accept_prob, _n_nonfinite = sample_raytrace(
            key=key,
            params_init=true_mean,
            log_prob_fn=log_prob_fn,
            n_steps=100,
            n_leapfrog_steps=10,
            step_size=0.05,
            sample_hmc=True,
            integrator="kdk",
        )

        mean_accept = float(jnp.mean(accept_prob))
        assert mean_accept > 0.3
        chex.assert_shape(chain, (100, D))

    def test_invalid_integrator_raises(self):
        """Unknown integrator should raise ValueError."""
        D = 3
        key = jax.random.PRNGKey(0)

        with pytest.raises(ValueError, match="Unknown integrator"):
            sample_raytrace(
                key=key,
                params_init=jnp.zeros(D),
                log_prob_fn=lambda x: -0.5 * jnp.sum(x**2),
                n_steps=10,
                n_leapfrog_steps=5,
                step_size=0.1,
                integrator="invalid",
            )


# ── Posterior autocorrelation integration ─────────────────────────


class TestPosteriorAutocorrelation:
    """Verify Posterior.autocorrelation_time() and check_convergence()."""

    def test_autocorrelation_time_on_gaussian_chain(self):
        """Run RT on Gaussian, check autocorrelation_time() returns sane values."""
        D = 5
        key = jax.random.PRNGKey(0)
        log_prob_fn = _gaussian_log_prob(jnp.zeros(D), jnp.eye(D))

        chain, _log_lik, _accept_prob, _n_nonfinite = sample_raytrace(
            key=key,
            params_init=jnp.zeros(D),
            log_prob_fn=log_prob_fn,
            n_steps=500,
            n_leapfrog_steps=10,
            step_size=0.05,
        )

        # Build a minimal Posterior
        samples = {f"param_{i}": chain[50:, i] for i in range(D)}

        posterior = Posterior(
            samples=samples,
            params={f"param_{i}": jnp.mean(chain[50:, i]) for i in range(D)},
            method="Ray Tracing test",
            wall_time_s=1.0,
            diagnostics={},
            loss_history=None,
            _model=None,
        )

        # autocorrelation_time should return per-param info
        act = posterior.autocorrelation_time()
        assert len(act) == D
        for _name, info in act.items():
            assert "tau_max" in info
            assert "ess" in info
            assert info["tau_max"] >= 1.0

        # effective_sample_size should return ESS values
        ess = posterior.effective_sample_size()
        assert len(ess) == D
        for _name, val in ess.items():
            assert val > 0

        # check_convergence should not crash
        conv = posterior.check_convergence(verbose=False)
        assert "all_converged" in conv
