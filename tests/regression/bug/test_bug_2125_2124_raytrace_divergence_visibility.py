# SPDX-License-Identifier: BSD-3-Clause
"""Tests for #2125 (non-finite proposal visibility with auto-backoff) and #2124 (step reduction).

#2125: sample_raytrace counts non-finite proposal energies instead of silently
       converting them to rejections; run_raytrace backs the step size off
       (up to 8 halvings, via _backoff_step_on_divergence) and raises
       DeadFitError only when backoff is exhausted.

#2124: The x0.3 mode-start step reduction (_resolve_initial_step) keys on how
       the start point was produced -- MAP/Laplace posteriors are mode starts;
       sampler warm starts (NUTS draws) are not -- never on which argument
       carried it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import DeadFitError
from tengri.inference.backends.mcmc.raytrace import (
    _backoff_step_on_divergence,
    _resolve_initial_step,
    sample_raytrace,
)
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.regression_bug


def _posterior_with_method(method):
    """A minimal Posterior carrying only what _resolve_initial_step reads."""
    return Posterior(
        samples=None,
        params={"p": 0.5},
        method=method,
        wall_time_s=1.0,
        diagnostics={},
        loss_history=None,
    )


class TestNonFiniteProposalCounting:
    """#2125: sample_raytrace exposes the non-finite proposal count."""

    def test_sample_raytrace_returns_nonfinite_count(self):
        """sample_raytrace returns a count of non-finite proposals."""

        def log_prob_fn(params):
            return -0.5 * jnp.sum(params**2)

        key = jax.random.PRNGKey(42)
        params_init = jnp.array([0.0, 0.0])

        _chain, _log_lik, _accept, n_nonfinite = sample_raytrace(
            key=key,
            params_init=params_init,
            log_prob_fn=log_prob_fn,
            n_steps=20,
            n_leapfrog_steps=3,
            step_size=0.1,
        )

        assert isinstance(n_nonfinite, (int, np.integer))
        assert n_nonfinite < 3, f"Expected <3 non-finite proposals, got {n_nonfinite}"
        assert _chain.shape[0] == 20, f"Expected 20 steps, got {_chain.shape[0]}"

    def test_divergent_run_detects_nonfinite_proposals(self):
        """A too-large step against a support edge yields non-finite proposals."""

        def log_prob_fn(params):
            x = params[0]
            y = params[1]
            safe_log = -0.5 * (x**2 + y**2)
            return jnp.where(x > -5.0, safe_log, -jnp.inf)

        key = jax.random.PRNGKey(42)
        params_init = jnp.array([0.0, 0.0])

        _chain, _log_lik, _accept, n_nonfinite = sample_raytrace(
            key=key,
            params_init=params_init,
            log_prob_fn=log_prob_fn,
            n_steps=20,
            n_leapfrog_steps=5,
            step_size=5.0,
        )

        assert n_nonfinite > 0, "Expected >0 non-finite proposals with divergent step"

    def test_everywhere_nan_counts_every_proposal(self):
        """A NaN-everywhere target marks every proposal non-finite."""

        def everywhere_nan(params):
            return jnp.array(jnp.nan)

        key = jax.random.PRNGKey(42)
        params_init = jnp.array([0.0, 0.0])

        _chain, _log_lik, _accept, n_nonfinite = sample_raytrace(
            key=key,
            params_init=params_init,
            log_prob_fn=everywhere_nan,
            n_steps=10,
            n_leapfrog_steps=2,
            step_size=0.1,
        )

        assert n_nonfinite == 10, f"Expected 10 non-finite proposals, got {n_nonfinite}"


class TestBackoffLoop:
    """#2125: the production backoff loop run_raytrace delegates to."""

    def test_healthy_fraction_never_probes(self):
        """A live burnin returns the step unchanged with zero backoffs."""
        probed = []

        def probe(step):
            probed.append(step)
            return 0

        step, n_backoffs, frac = _backoff_step_on_divergence(
            probe,
            step_size=0.1,
            initial_fraction=0.02,
            n_probe_proposals=100,
            verbose=False,
        )
        assert step == 0.1
        assert n_backoffs == 0
        assert frac == 0.02
        assert probed == [], "Healthy runs must not pay any probe cost"

    def test_recoverable_divergence_halves_until_live(self):
        """The loop halves past the stability cliff and stops there."""

        def probe(step):
            return 100 if step > 0.03 else 3

        step, n_backoffs, frac = _backoff_step_on_divergence(
            probe,
            step_size=0.1,
            initial_fraction=1.0,
            n_probe_proposals=100,
            verbose=False,
        )
        # 0.1 -> 0.05 (still diverged) -> 0.025 (live)
        assert n_backoffs == 2
        assert step == pytest.approx(0.025)
        assert frac == pytest.approx(0.03)

    def test_exhaustion_raises_dead_fit_error(self):
        """Eight halvings without recovery raise loudly with diagnostics."""
        with pytest.raises(DeadFitError) as excinfo:
            _backoff_step_on_divergence(
                lambda step: 100,
                step_size=0.1,
                initial_fraction=1.0,
                n_probe_proposals=100,
                verbose=False,
            )
        assert "diverged" in str(excinfo.value).lower()
        assert excinfo.value.step_size == pytest.approx(0.1 * 0.5**8)
        assert excinfo.value.warmup_divergence_frac == pytest.approx(1.0)


class TestModeStartStepReduction:
    """#2124: the production step resolver, with real Posterior method strings."""

    def test_map_posterior_gets_the_reduction(self):
        """A real MAP method string ("MAP (L-BFGS-B)") is a mode start."""
        step = _resolve_initial_step(5, None, _posterior_with_method("MAP (L-BFGS-B)"))
        assert float(step) == pytest.approx(0.03 * np.sqrt(5.0) * 0.3)

    def test_laplace_posterior_gets_the_reduction(self):
        """Laplace point estimates sit at the mode too."""
        step = _resolve_initial_step(5, None, _posterior_with_method("Laplace"))
        assert float(step) == pytest.approx(0.03 * np.sqrt(5.0) * 0.3)

    def test_nuts_warm_start_keeps_the_full_step(self):
        """A sampler warm start hands over a mean, not a mode: no reduction."""
        step = _resolve_initial_step(5, None, _posterior_with_method("mcmc_nuts"))
        assert float(step) == pytest.approx(0.03 * np.sqrt(5.0))

    def test_random_start_keeps_the_full_step(self):
        step = _resolve_initial_step(5, None, None)
        assert float(step) == pytest.approx(0.03 * np.sqrt(5.0))

    def test_high_dimension_default(self):
        step = _resolve_initial_step(20, None, None)
        assert float(step) == pytest.approx(0.01)

    def test_explicit_step_is_never_touched(self):
        """A user-provided step bypasses defaults and the reduction alike."""
        step = _resolve_initial_step(5, 0.42, _posterior_with_method("MAP (adam)"))
        assert step == 0.42
