# SPDX-License-Identifier: BSD-3-Clause
"""ChEES-HMC learns the trajectory length and never the metric.

``mcmc_chees`` exists to answer one question: does cross-chain adaptive
trajectory length converge where a fixed global ``L`` did not? Three reports
(``bench/reports/2026-08-17_*``) measured that no single ``L`` serves the three
notebook posteriors, and NUTS's per-chain answer is exactly what breaks lock-step
on an accelerator. ChEES adapts one ``L`` for the whole ensemble, so every chain
still takes the same number of leapfrogs.

The load-bearing constraint is the one Phase 1a paid for. ``mcmc_ghmc``'s MEADS
adaptation derives its momentum metric from the adapting ensemble's own per-fold
standard deviation, and the measured result
(``bench/reports/2026-08-30_ghmc_meads_adaptation.md``) is an unopposed feedback
loop -- wider ensemble, larger momentum, longer excursions, wider ensemble -- that
reached split-R-hat 1.1e10 while acceptance sat at 0.989, because energy really is
conserved under the same inflated metric that produced the excursions. ChEES can
be configured the same way (``mass_matrix_estimation="diagonal"``). It is not,
and these tests are what keeps it that way: the metric comes from
:mod:`tengri.inference.preconditioning`'s analytic ``J^T N^-1 J + I``, and the
flag stays off by default.
"""

from __future__ import annotations

import inspect
from unittest import mock

import numpy as np
import pytest

pytestmark = pytest.mark.contract

blackjax = pytest.importorskip("blackjax")
pytest.importorskip("optax")

from tengri.inference._backend_registry import get_backend
from tengri.inference.backends.mcmc._shared import (
    _CHEES_CHAIN_JITTER_SCALE,
    _CHEES_DEFAULT_ENSEMBLE,
    _CHEES_JITTER_SCALE,
    _CHEES_MIN_ENSEMBLE,
    _chees_scan,
    _resolve_chees_ensemble,
)
from tengri.inference.backends.mcmc.chees import CHEES_TARGET_ACCEPT_RATE, run_chees


class TestTheEnsembleResolver:
    """The ensemble is a *superset* of the sampling chains, never the same axis."""

    def test_auto_gives_the_documented_default(self):
        assert _resolve_chees_ensemble("auto", n_chains=1) == _CHEES_DEFAULT_ENSEMBLE

    def test_one_chain_still_gets_a_full_ensemble(self):
        """The whole point of decoupling the axes.

        ``n_chains=1`` is ``run_chees``'s default and what every catalog fit
        uses. ChEES's trajectory-length gradient is built from positions centered
        *across* chains, so a one-chain ensemble centers to exactly zero: the
        length would never move off its initial value, and nothing would say so.
        """
        assert _resolve_chees_ensemble("auto", n_chains=1) >= _CHEES_MIN_ENSEMBLE

    def test_the_ensemble_is_never_smaller_than_the_sampling_chains(self):
        """Sampling chains are seeded from the ensemble's warmed final states."""
        assert _resolve_chees_ensemble("auto", n_chains=200) >= 200
        assert _resolve_chees_ensemble(8, n_chains=200) >= 200

    def test_an_undersized_ensemble_is_refused_loudly(self):
        """LOAD-BEARING. Neuter: clamp to the floor instead of raising.

        A silently clamped ensemble and a silently un-adapted one are
        indistinguishable from the outside, and the entire claim of this backend
        is that ``L`` is learned rather than set. Refusing is the only signal a
        caller gets.
        """
        with pytest.raises(ValueError) as exc:
            _resolve_chees_ensemble(1, n_chains=1)
        msg = str(exc.value)
        assert "n_ensemble" in msg
        assert "n_chains" in msg, "the refusal must say which knob it is *not* talking about"
        assert str(_CHEES_DEFAULT_ENSEMBLE) in msg, "must name a size that would work"

    def test_an_unknown_string_is_refused(self):
        with pytest.raises(ValueError, match="auto"):
            _resolve_chees_ensemble("automatic", n_chains=1)

    def test_the_ensemble_axis_is_chains_within_galaxy(self):
        """The phase's blocking design decision, pinned as prose in the resolver.

        Only chains-within-galaxy keeps per-galaxy posteriors independent: an
        ensemble spanning *galaxies* would tune one ``L`` against a mixture of
        different posteriors and make each galaxy's draws depend on which other
        galaxies shared its batch. The decision lives in the docstring because
        there is no other place a future reader would look before wiring the
        batched catalog path.
        """
        doc = _resolve_chees_ensemble.__doc__
        assert "chains-within-galaxy" in doc
        assert "galaxies-within-batch" in doc


class TestTheMetricIsNeverEstimatedFromTheEnsemble:
    """The Phase 1a lesson, held as a default rather than as a comment."""

    def test_mass_matrix_estimation_is_off_by_default(self):
        """LOAD-BEARING. Neuter: flip the default to ``"diagonal"``.

        This is the exact shape that broke ``mcmc_ghmc``. ChEES's Metropolis step
        and dual-averaged step size make the loop weaker, not absent, and tengri
        can compute the metric analytically -- so guessing it from 32 chains is
        strictly worse information at strictly higher risk.
        """
        assert inspect.signature(run_chees).parameters["mass_matrix_estimation"].default is None

    def test_the_identity_metric_is_what_the_default_actually_produces(self):
        """The default must be checked at the output, not only at the signature.

        A ``None`` default that BlackJAX then overrode would satisfy the
        signature test and fail the claim. ``inverse_mass_matrix`` coming back as
        exactly ones is the observable form of "the ensemble did not set the
        metric".
        """
        import jax
        import jax.numpy as jnp

        dim = 4
        logdensity, _, _ = _correlated_gaussian(dim)
        keys = jax.random.split(jax.random.PRNGKey(1), 2 * 40).reshape(2, 40, 2)
        *_, imm, _ = _chees_scan(
            jnp.zeros(dim),
            jax.random.PRNGKey(0),
            keys,
            logdensity,
            (),
            60,
            16,
            2,
            40,
            _CHEES_JITTER_SCALE,
            1.0,
            CHEES_TARGET_ACCEPT_RATE,
            64,
            0.05,
            None,
        )
        assert np.array_equal(np.asarray(imm), np.ones(dim)), (
            f"mass_matrix_estimation=None must leave the metric at ones, got {imm}"
        )

    def test_the_adaptation_receives_the_ENSEMBLE_width_not_the_chain_count(self):
        """LOAD-BEARING. Neuter: pass ``num_chains=n_chains`` instead.

        ``chees_adaptation``'s trajectory-length gradient is a cross-chain
        statistic, so the number it is constructed with is the size of the
        sample that estimates it. Handing it ``n_chains`` (1 by default, 2 in
        every notebook row) would estimate the ChEES criterion from two points
        and call the result adapted. This spies on the actual call rather than
        reading the source, because the source is what a refactor changes.

        BlackJAX also asserts ``positions.shape[0] == num_chains`` inside
        ``run``, so a mismatch between the two would raise -- but that assert
        cannot distinguish "both are 32" from "both are 2", which is the failure
        this test exists for.
        """
        import blackjax
        import jax
        import jax.numpy as jnp

        seen = {}
        real = blackjax.chees_adaptation

        def _spy(logdensity_fn, num_chains, **kw):
            seen["num_chains"] = num_chains
            return real(logdensity_fn, num_chains, **kw)

        logdensity, _, _ = _correlated_gaussian(4)
        keys = jax.random.split(jax.random.PRNGKey(1), 2 * 30).reshape(2, 30, 2)
        with mock.patch.object(blackjax, "chees_adaptation", _spy):
            _chees_scan(
                jnp.zeros(4),
                jax.random.PRNGKey(0),
                keys,
                logdensity,
                (),
                40,
                16,  # n_ensemble
                2,  # n_chains -- deliberately different
                30,
                _CHEES_JITTER_SCALE,
                1.0,
                CHEES_TARGET_ACCEPT_RATE,
                64,
                0.05,
                None,
            )
        assert seen["num_chains"] == 16, (
            f"chees_adaptation was constructed with num_chains={seen.get('num_chains')}, "
            "but the adaptation ensemble is 16. The criterion would be estimated "
            "from the sampling chains instead of from the ensemble."
        )

    def test_an_unknown_estimation_mode_is_refused(self):
        with pytest.raises(ValueError, match="mass_matrix_estimation"):
            run_chees(object(), key=None, mass_matrix_estimation="dense")


class TestTheTwoJitterDialsAreSeparate:
    """The ensemble estimates a criterion; the sampling chains feed R-hat.

    Collapsing them into one dial makes "tight enough to adapt a long
    trajectory" and "wide enough for split R-hat to mean anything" look like a
    trade-off. They are two different chain sets, and only one of them is what
    R-hat is computed over, so there is no trade-off to make -- but only if the
    two are actually separable at the call site.
    """

    def test_both_dials_are_on_the_public_signature(self):
        sig = inspect.signature(run_chees)
        assert "ensemble_jitter" in sig.parameters
        assert "chain_jitter" in sig.parameters

    def test_the_chain_dial_is_wider_than_the_ensemble_dial(self):
        """They point in opposite directions, so the constants must differ.

        Dispersion inflates ChEES's cross-chain jump-distance criterion and
        drives the adapted trajectory length down, so the ensemble wants to be
        tight. Split R-hat only detects non-convergence when its chains start
        overdispersed, so the sampling chains want to be wide. Equal constants
        would mean one of the two jobs was not thought about.
        """
        assert _CHEES_CHAIN_JITTER_SCALE > _CHEES_JITTER_SCALE

    def test_the_chain_dial_actually_moves_the_starting_positions(self):
        """LOAD-BEARING. Neuter: ignore ``chain_jitter`` and always slice the ensemble.

        A diagnostic knob that is accepted and dropped is worse than one that
        does not exist: the R-hat it produces would look like an independent
        test and be a consistency check. Measured at the first draw, where the
        two seeding routes are furthest apart.
        """
        import jax
        import jax.numpy as jnp

        dim = 4
        logdensity, _, _ = _correlated_gaussian(dim)
        keys = jax.random.split(jax.random.PRNGKey(1), 4 * 40).reshape(4, 40, 2)
        common = (
            jnp.zeros(dim),
            jax.random.PRNGKey(0),
            keys,
            logdensity,
            (),
            60,
            16,
            4,
            40,
            0.01,
            1.0,
            CHEES_TARGET_ACCEPT_RATE,
            64,
            0.05,
            None,
        )
        from_ensemble = np.asarray(_chees_scan(*common, None)[0][:, 0, :])
        overdispersed = np.asarray(_chees_scan(*common, 1.0)[0][:, 0, :])

        # The overdispersed chains must start genuinely further apart, not merely
        # elsewhere: it is the SPREAD that R-hat reads, so a shifted-but-equally-
        # tight set of starts would satisfy a naive "the arrays differ" check
        # while changing nothing about the diagnostic.
        assert overdispersed.std(axis=0).mean() > from_ensemble.std(axis=0).mean(), (
            f"chain_jitter=1.0 gave starting spread {overdispersed.std(axis=0).mean():.3g}, "
            f"no wider than the ensemble's {from_ensemble.std(axis=0).mean():.3g}"
        )


class TestTheTuningKnobsAreChEESsOwn:
    def test_the_target_acceptance_rate_is_not_nuts(self):
        """LOAD-BEARING. Neuter: set the default to 0.8.

        Each ChEES step is a *fixed*-length HMC proposal, whose optimal
        acceptance rate is 0.651 -- not the 0.8 NUTS is tuned to and not the 0.9
        the notebook rows pass to ``mcmc_hmc``. Carrying the NUTS value across
        would still run: it would just dual-average the step size toward a target
        chosen for a different proposal, and nothing would report it.
        """
        assert pytest.approx(0.651) == CHEES_TARGET_ACCEPT_RATE
        assert (
            inspect.signature(run_chees).parameters["target_accept_rate"].default
            == CHEES_TARGET_ACCEPT_RATE
        )
        blackjax_default = (
            inspect.signature(blackjax.chees_adaptation)
            .parameters["target_acceptance_rate"]
            .default
        )
        assert pytest.approx(blackjax_default) == CHEES_TARGET_ACCEPT_RATE, (
            "BlackJAX moved its own default; ours was pinned to it deliberately"
        )

    def test_the_ensemble_knobs_are_on_the_public_signature(self):
        sig = inspect.signature(run_chees)
        for name in ("n_ensemble", "ensemble_jitter", "jitter_amount", "max_leapfrog_steps"):
            assert name in sig.parameters, f"{name} must be tunable without editing source"

    def test_the_leapfrog_cap_is_bounded_below_blackjaxs_default(self):
        """The cap is what bounds a warmup step's cost.

        At BlackJAX's 1000, one adaptation step over a 32-chain ensemble is
        32,000 gradient evaluations -- a budget nothing in this repo's D <= 8
        photometry fits needs, and one that would make the warmup unprofilable.
        """
        assert inspect.signature(run_chees).parameters["max_leapfrog_steps"].default < 1000


class TestTheRegistryDeclarationMatchesTheRunner:
    def test_the_backend_is_registered_experimental(self):
        entry = get_backend("mcmc_chees")
        assert entry.tier == "experimental"
        assert entry.requires == ("blackjax",)
        assert entry.legacy_fitter is False

    def test_it_declares_the_preconditioning_capability(self):
        """The analytic metric is the whole reason the ensemble does not need one."""
        assert get_backend("mcmc_chees").accepts_precondition is True
        assert "precondition" in inspect.signature(run_chees).parameters

    def test_the_short_doc_does_not_self_flag_as_broken(self):
        """``test_broken_backends_quarantined`` derives the broken tier from prose.

        A backend at ``experimental`` whose short_doc carries ``[UNSTABLE]`` /
        ``[POOR MIXING]`` / ``Do not use`` reddens that contract. Pinned here so
        a later edit to the short_doc fails in the file that owns the wording.
        """
        doc = get_backend("mcmc_chees").short_doc
        for flag in ("[UNSTABLE]", "[POOR MIXING]", "Do not use"):
            assert flag not in doc


def _correlated_gaussian(dim=4, seed=0):
    """A target whose mean and covariance are known exactly."""
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


class TestTheAdaptedPathSamplesAKnownTarget:
    """A signature test proves nothing about whether the sampler works.

    This runs the real jitted entry point on a correlated Gaussian whose moments
    are known exactly, so a wrong parameter handoff -- an ``L`` read out of the
    wrong tuple slot, a kernel rebuilt without the adaptation's own Halton
    sequence, an ensemble sliced on the wrong axis -- shows up as a wrong moment
    rather than as a shape error.
    """

    def test_it_recovers_the_mean_and_the_marginal_scales(self):
        import jax
        import jax.numpy as jnp

        dim = 4
        logdensity, mean, cov = _correlated_gaussian(dim)
        n_chains, n_iter = 4, 2500
        keys = jax.random.split(jax.random.PRNGKey(1), n_chains * n_iter)
        keys = keys.reshape(n_chains, n_iter, 2)

        positions, divergent, step_size, _imm, _n_leapfrog = _chees_scan(
            jnp.zeros(dim),
            jax.random.PRNGKey(0),
            keys,
            logdensity,
            (),
            400,
            32,
            n_chains,
            n_iter,
            _CHEES_JITTER_SCALE,
            1.0,
            CHEES_TARGET_ACCEPT_RATE,
            200,
            0.05,
            None,
        )
        draws = np.asarray(positions[:, 500:, :]).reshape(-1, dim)

        assert np.all(np.isfinite(draws))
        assert float(step_size) > 0.0
        assert np.allclose(draws.mean(axis=0), mean, atol=0.25), (
            f"mean off: {draws.mean(axis=0)} vs {mean}"
        )
        got = draws.std(axis=0)
        want = np.sqrt(np.diag(cov))
        assert np.all(got > 0.7 * want) and np.all(got < 1.4 * want), (
            f"marginal scales off: {got} vs {want}"
        )
        assert int(np.sum(np.asarray(divergent))) == 0

    def test_the_trajectory_length_is_actually_learned(self):
        """LOAD-BEARING. Neuter: return the initial length instead of the adapted one.

        The claim of this backend is that ``L`` comes from the posterior. Two
        targets whose correlation lengths differ by construction must therefore
        get different ``L``. A backend that returned a constant would pass every
        other test in this file.
        """
        import jax
        import jax.numpy as jnp

        def _adapted_length(scale):
            def logdensity(x, _data):
                return -0.5 * jnp.sum((x / scale) ** 2)

            keys = jax.random.split(jax.random.PRNGKey(3), 2 * 60).reshape(2, 60, 2)
            *_, n_leapfrog = _chees_scan(
                jnp.zeros(4),
                jax.random.PRNGKey(2),
                keys,
                logdensity,
                (),
                400,
                32,
                2,
                60,
                _CHEES_JITTER_SCALE,
                1.0,
                CHEES_TARGET_ACCEPT_RATE,
                200,
                0.05,
                None,
            )
            return float(n_leapfrog)

        tight = _adapted_length(jnp.asarray([1.0, 1.0, 1.0, 1.0]))
        # One direction 30x wider: the trajectory has to run further to cross it,
        # so the learned length must rise.
        wide = _adapted_length(jnp.asarray([1.0, 1.0, 1.0, 30.0]))
        assert tight > 0.0 and wide > 0.0
        assert wide > 1.5 * tight, (
            f"L did not respond to the posterior's correlation length: "
            f"{tight:.2f} (isotropic) vs {wide:.2f} (one 30x direction)"
        )
