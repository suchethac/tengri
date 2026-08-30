# SPDX-License-Identifier: BSD-3-Clause
"""GHMC adapts with MEADS, not with HMC's window adaptation.

``mcmc_ghmc`` sat at ``tier="broken"`` with R-hat 2.5-3.1 and ESS ~ 1, and the
diagnosis was in the adaptation rather than in the kernel: ``run_ghmc`` called
``blackjax.window_adaptation``, which dual-averages a step size against a target
*acceptance rate*. Generalized HMC has no Metropolis acceptance to target -- it
uses a non-reversible slice update -- and window adaptation cannot see the
damping ``alpha`` at all, so the parameter that actually governs GHMC's mixing
was left at a hand-set ``0.8`` while a knob the kernel does not have was tuned.

BlackJAX ships ``meads_adaptation`` for exactly this kernel. These tests pin the
seam: the ensemble resolver's arithmetic and refusals, that the runner no longer
reaches for window adaptation, that the hand-set constants are gone from the
defaults, and that the adapted path actually samples a target whose answer is
known analytically.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.contract

blackjax = pytest.importorskip("blackjax")

from tengri.inference.backends.mcmc._shared import (
    _MEADS_DEFAULT_ENSEMBLE,
    _MEADS_JITTER_SCALE,
    _MEADS_MIN_CHAINS_PER_FOLD,
    _resolve_meads_ensemble,
)
from tengri.inference.backends.mcmc.ghmc import run_ghmc


class TestTheEnsembleResolver:
    """The ensemble is a *superset* of the sampling chains, never the same axis."""

    def test_auto_gives_the_documented_default(self):
        assert _resolve_meads_ensemble("auto", n_chains=1, n_folds=4) == _MEADS_DEFAULT_ENSEMBLE

    def test_one_chain_still_gets_a_full_ensemble(self):
        """The whole point of decoupling the axes.

        ``n_chains=1`` is ``run_ghmc``'s default and what every catalog fit uses.
        Had the ensemble been tied to ``n_chains``, the default configuration
        would be the one case where MEADS's cross-chain statistics degenerate to
        a single sample -- adapted in name only.
        """
        assert _resolve_meads_ensemble("auto", n_chains=1, n_folds=4) >= 4 * 4

    def test_the_ensemble_is_never_smaller_than_the_sampling_chains(self):
        """Sampling chains are seeded from the ensemble's final states."""
        assert _resolve_meads_ensemble("auto", n_chains=200, n_folds=4) >= 200
        assert _resolve_meads_ensemble(64, n_chains=200, n_folds=4) >= 200

    @pytest.mark.parametrize("n_folds", [1, 2, 3, 4, 5, 8])
    def test_the_result_is_always_divisible_by_the_fold_count(self, n_folds):
        """BlackJAX raises otherwise, and it raises *inside* a jitted trace."""
        for n_chains in (1, 3, 7, 33):
            got = _resolve_meads_ensemble("auto", n_chains=n_chains, n_folds=n_folds)
            assert got % n_folds == 0

    def test_an_explicit_size_is_rounded_up_not_down(self):
        assert _resolve_meads_ensemble(33, n_chains=1, n_folds=4) == 36

    def test_too_few_chains_per_fold_is_refused_loudly(self):
        """LOAD-BEARING. Neuter: clamp instead of raising.

        MEADS is meaningless on a small ensemble and the failure is silent: the
        maximum-eigenvalue estimator divides by ``n * (n - 1)`` over a fold's
        chains, so an undersized fold returns noise, and a step size drawn from
        noise is indistinguishable from the hand-set constant this change exists
        to remove. Refusing is the only signal a caller gets.
        """
        with pytest.raises(ValueError) as exc:
            _resolve_meads_ensemble(4, n_chains=1, n_folds=4)
        msg = str(exc.value)
        assert "n_ensemble" in msg
        assert "n_chains" in msg, "the refusal must say which knob it is *not* talking about"
        assert str(4 * _MEADS_MIN_CHAINS_PER_FOLD) in msg, "must name a size that would work"

    def test_a_single_chain_ensemble_is_refused(self):
        with pytest.raises(ValueError):
            _resolve_meads_ensemble(1, n_chains=1, n_folds=4)

    def test_an_unknown_string_is_refused(self):
        with pytest.raises(ValueError, match="auto"):
            _resolve_meads_ensemble("automatic", n_chains=1, n_folds=4)

    def test_a_nonsense_fold_count_is_refused(self):
        with pytest.raises(ValueError, match="n_folds"):
            _resolve_meads_ensemble("auto", n_chains=1, n_folds=0)


class TestTheRunnerNoLongerBorrowsHMCsAdaptation:
    def test_run_ghmc_does_not_call_window_adaptation(self):
        """Anti-drift: the defect was a *call site*, so assert on the call site."""
        source = inspect.getsource(run_ghmc)
        assert "window_adaptation" not in source, (
            "run_ghmc is back on HMC's window adaptation, which tunes a target "
            "acceptance rate GHMC does not have and cannot see alpha at all"
        )
        assert "_ghmc_meads_scan" in source

    def test_alpha_and_delta_default_to_adapted(self):
        """``0.8`` / ``0.65`` were guesses; MEADS derives both."""
        sig = inspect.signature(run_ghmc)
        assert sig.parameters["alpha"].default is None
        assert sig.parameters["delta"].default is None

    def test_the_ensemble_knobs_are_on_the_public_signature(self):
        sig = inspect.signature(run_ghmc)
        for name in ("n_ensemble", "n_folds", "ensemble_jitter"):
            assert name in sig.parameters, f"{name} must be tunable without editing source"

    def test_the_low_rank_metric_is_available_but_not_the_default(self):
        """MEADS-LRD is an experiment with a measured answer, not an improvement.

        The 1e5-1e8 latent condition numbers in
        :mod:`tengri.inference.preconditioning` make a low-rank momentum metric
        the obvious lever, so it must be reachable without editing source. It
        must equally not be a default: measured at ``rank = D`` it reaches
        split-R-hat 1.81 on nb05 and 1.46 on nb01 against a 1.01 bar, i.e. it
        does not help. ``None`` is also BlackJAX's own default, so the diagonal
        path stays bit-for-bit the original behavior.
        """
        sig = inspect.signature(run_ghmc)
        assert sig.parameters["low_rank_rank"].default is None
        assert sig.parameters["low_rank_window_fraction"].default == 0.5

    def test_target_accept_rate_defaults_to_unset(self):
        """A knob MEADS ignores must not carry a value that looks honored."""
        assert inspect.signature(run_ghmc).parameters["target_accept_rate"].default is None

    def test_the_ensemble_dispersion_is_not_the_chain_jitter(self):
        """Two different jobs, so two different numbers -- see _MEADS_JITTER_SCALE.

        ``_vmap_chains`` jitters to decorrelate chains that already have a tuned
        step size. MEADS reads the ensemble spread *as* the posterior scale. A
        1e-3 ball saturates its step-size clamp on the first step.
        """
        vmap_default = (
            inspect.signature(
                __import__(
                    "tengri.inference.backends.mcmc._shared", fromlist=["_vmap_chains"]
                )._vmap_chains
            )
            .parameters["jitter_scale"]
            .default
        )
        assert 100 * vmap_default < _MEADS_JITTER_SCALE


class TestTheAdaptedPathSamplesAKnownTarget:
    """A signature test proves nothing about whether the sampler works.

    This runs the real jitted entry point on a correlated Gaussian whose mean and
    covariance are known exactly, so a wrong ``alpha``/step-size handoff (or a
    wrong ``in_axes`` on the ensemble slice) shows up as a wrong moment rather
    than as a shape error.
    """

    @staticmethod
    def _target(dim=4, seed=0):
        import jax.numpy as jnp

        rng = np.random.default_rng(seed)
        a = rng.normal(size=(dim, dim))
        cov = a @ a.T / dim + np.eye(dim)
        prec = jnp.asarray(np.linalg.inv(cov))
        mean = jnp.asarray(rng.normal(size=dim))

        def logdensity(x, _data):
            d = x - mean
            return -0.5 * d @ prec @ d

        return logdensity, np.asarray(mean), cov

    def test_it_recovers_the_mean_and_the_marginal_scales(self):
        import jax

        from tengri.inference.backends.mcmc._shared import _ghmc_meads_scan

        dim = 4
        logdensity, mean, cov = self._target(dim)
        n_chains, n_iter = 4, 3000
        keys = jax.random.split(jax.random.PRNGKey(1), n_chains * n_iter)
        keys = keys.reshape(n_chains, n_iter, 2)

        positions, divergent, step_size, mis, alpha, delta = _ghmc_meads_scan(
            jax.numpy.zeros(dim),
            jax.random.PRNGKey(0),
            keys,
            logdensity,
            (),
            600,
            32,
            4,
            None,
            None,
            1.0,
            None,
            0.5,
        )
        draws = np.asarray(positions[:, 1000:, :]).reshape(-1, dim)

        assert np.all(np.isfinite(draws))
        assert float(step_size) > 0.0
        assert 0.0 < float(alpha) <= 1.0
        assert abs(float(delta) - 0.5 * float(alpha)) < 1e-6, (
            "MEADS Algorithm 3 sets delta = alpha / 2; a mismatch means the "
            "parameters dict was unpacked in the wrong order"
        )
        assert np.allclose(draws.mean(axis=0), mean, atol=0.35), (
            f"mean off: {draws.mean(axis=0)} vs {mean}"
        )
        got = draws.std(axis=0)
        want = np.sqrt(np.diag(cov))
        assert np.all(got > 0.4 * want) and np.all(got < 1.8 * want), (
            f"marginal scales off: {got} vs {want}"
        )
        assert int(np.sum(np.asarray(divergent))) == 0
        assert np.asarray(mis).shape == (dim,)

    def test_pinning_alpha_overrides_the_adapted_value(self):
        """The override is what the old hand-set default did unconditionally."""
        import jax

        from tengri.inference.backends.mcmc._shared import _ghmc_meads_scan

        dim = 4
        logdensity, _, _ = self._target(dim)
        keys = jax.random.split(jax.random.PRNGKey(1), 2 * 50).reshape(2, 50, 2)
        common = (jax.numpy.zeros(dim), jax.random.PRNGKey(0), keys, logdensity, (), 100, 16, 4)

        _, _, _, _, adapted_alpha, adapted_delta = _ghmc_meads_scan(
            *common, None, None, 1.0, None, 0.5
        )
        _, _, _, _, pinned_alpha, pinned_delta = _ghmc_meads_scan(
            *common, 0.8, 0.65, 1.0, None, 0.5
        )

        assert float(pinned_alpha) == pytest.approx(0.8)
        assert float(pinned_delta) == pytest.approx(0.65)
        assert float(adapted_alpha) != pytest.approx(0.8), (
            "the adapted alpha landed on the old hand-set constant, which would "
            "make this whole change unobservable"
        )
        assert float(adapted_delta) != pytest.approx(0.65)
