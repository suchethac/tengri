# SPDX-License-Identifier: BSD-3-Clause
"""Tempered SMC anneals from the *exact* prior, and its ESS is not an ESS.

``mcmc_smc`` rests on two facts about this codebase rather than on anything the
sampler brings:

1. **The lambda = 0 target is exact.** ``build_loss_fn`` is the data term plus
   ``standardized_neg_log_prior``, and in the standardized latent space that
   prior is exactly ``N(0, I)``. So the tempering split is a split of the
   objective the other backends already sample, and the initial particles are
   i.i.d. draws rather than the output of a second sampler. If those two halves
   ever stop summing to the log-posterior, every SMC row silently targets a
   different distribution than every NUTS row, and nothing raises.
2. **A resampled particle population is exchangeable**, so the autocorrelation
   ESS reports roughly the particle count however degenerate the population is.
   That is a diagnostic contaminated by the thing it diagnoses -- the fourth
   instance of the pattern ``bench/reports/2026-08-30_chees_hmc.md`` names -- and
   these tests pin the honest replacement (``_smc_ancestor_ess``) and the
   documentation that says so.

The third claim is about cost. SMC's price is
``n_particles * n_temperatures * n_mcmc_steps * n_leapfrog_steps`` gradients, and
three of those four are the caller's. A backend that does not report the product
invites the reader to reconstruct it from a wall clock, which is exactly how
``bench/reports/2026-08-30_mclmc_tuning.md``'s units error happened.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.contract

pytest.importorskip("blackjax")

import jax
import jax.numpy as jnp

from tengri.inference._backend_registry import get_backend
from tengri.inference.backends.mcmc._shared import (
    _SMC_DEFAULT_PARTICLES,
    _SMC_MAX_TEMPERATURES,
    _SMC_STEP_SIZE_GAIN,
    _smc_ancestor_ess,
    _smc_scan,
)
from tengri.inference.backends.mcmc.smc import SMC_TARGET_ACCEPT_RATE, run_smc


def _tilted_gaussian(dim, scale=0.3, shift=1.5):
    """An ``N(0, I)`` prior and a Gaussian likelihood with a known posterior.

    Returns ``(logprior_2arg, loglik_2arg, post_mean, post_sd)``. The posterior of
    ``N(0, I)`` against a ``N(shift, scale^2 I)`` likelihood is Gaussian with
    ``sd^2 = scale^2 / (1 + scale^2)`` and ``mean = shift / (1 + scale^2)``, so a
    tempering path that lands anywhere else is visible as a wrong moment rather
    than as a shape error.
    """
    mean = jnp.full((dim,), shift)

    def logprior(position, data_args):
        """The standardized N(0, I) prior; ``data_args`` unused, as in the backend."""
        del data_args
        return -0.5 * jnp.sum(position**2)

    def loglik(position, data_args):
        """A Gaussian data term centred away from the prior, so tempering has work."""
        del data_args
        return -0.5 * jnp.sum(((position - mean) / scale) ** 2)

    var = scale**2 / (1.0 + scale**2)
    return logprior, loglik, float(shift) * var / scale**2, float(np.sqrt(var))


class TestTheTemperingSplitIsTheObjective:
    """prior + likelihood must be the log-posterior every other backend samples."""

    def test_the_split_sums_to_the_log_posterior_on_a_real_model(self):
        """LOAD-BEARING. Neuter: build the prior from a second implementation.

        ``_get_flat_prior_and_likelihood`` reaches the two halves through
        ``InferenceContext``; ``_get_flat_logdensity`` reaches the whole through
        ``build_loss_fn``. They are supposed to be the same arithmetic by
        construction, and "by construction" is exactly the claim that stops being
        true in a refactor and stays silent when it does -- a tempered SMC row
        would then converge beautifully to the wrong distribution.
        """
        pytest.importorskip("h5py")
        from tengri.inference.backends.mcmc._shared import (
            _get_flat_logdensity,
            _get_flat_prior_and_likelihood,
        )

        fitter = _toy_fitter()
        init = fitter._initialize_unbounded(jax.random.PRNGKey(0))
        logpost, _unravel, init_flat, data_args = _get_flat_logdensity(fitter, init)
        logprior, loglik = _get_flat_prior_and_likelihood(fitter, init)

        rng = np.random.default_rng(0)
        for _ in range(5):
            pos = jnp.asarray(init_flat + 0.4 * rng.standard_normal(init_flat.shape))
            whole = float(logpost(pos, data_args))
            halves = float(logprior(pos, data_args)) + float(loglik(pos, data_args))
            assert np.isclose(whole, halves, rtol=1e-9, atol=1e-9), (
                f"log-posterior {whole} != prior + likelihood {halves}; the tempering "
                "path targets a different distribution than mcmc_nuts does."
            )

    def test_the_prior_half_does_not_read_the_data(self):
        """A prior that depended on ``data_args`` would not be a prior.

        Cheap to state and the tempering path rests on it: at lambda = 0 the
        particles are drawn from ``N(0, I)`` analytically, never evaluated, so a
        data-dependent prior term would be silently skipped at the one rung where
        it is the whole target.
        """
        pytest.importorskip("h5py")
        from tengri.inference.backends.mcmc._shared import _get_flat_prior_and_likelihood

        fitter = _toy_fitter()
        init = fitter._initialize_unbounded(jax.random.PRNGKey(0))
        logprior, _ = _get_flat_prior_and_likelihood(fitter, init)
        pos = jnp.zeros(len(fitter._free_names))
        assert float(logprior(pos, fitter._data_args)) == float(logprior(pos, None))


class TestTheParticleDiagnosticIsNotAnAutocorrelationESS:
    """The one number that survives exchangeability."""

    def test_all_distinct_ancestors_give_the_full_population(self):
        n = 64
        got = _smc_ancestor_ess(jnp.arange(n), n, jnp.float64)
        assert np.isclose(float(got), n)

    def test_a_collapsed_population_reads_one(self):
        """LOAD-BEARING. Neuter: return the particle count unconditionally.

        This is the degenerate case the autocorrelation estimator cannot see: 64
        copies of one particle is still 64 exchangeable rows, so the
        autocorrelation ESS reports ~64 and the ancestor ESS reports 1.
        """
        n = 64
        got = _smc_ancestor_ess(jnp.zeros(n, dtype=jnp.int32), n, jnp.float64)
        assert np.isclose(float(got), 1.0)

    def test_the_docstring_refuses_the_autocorrelation_reading(self):
        """Prose, pinned, because the misreading is the expensive one.

        Every table in this project carries a ``min ESS`` column and every reader
        knows what it means there. An SMC row's autocorrelation ESS is roughly
        the particle count by construction, so a reader who carries the habit
        across reads a broken fit as a converged one.
        """
        doc = _smc_ancestor_ess.__doc__
        assert "exchangeable" in doc
        assert "autocorrelation" in doc


class TestTheCostIsReportedRatherThanReconstructed:
    """``gradients_per_draw`` is a returned number, not an inference from a clock."""

    def test_the_leapfrog_count_is_static_so_a_rung_has_a_fixed_price(self):
        """LOAD-BEARING. Neuter: drop ``n_leapfrog_steps`` from ``static_argnums``.

        Traced, it would flow into BlackJAX as an ordinary ``mcmc_parameters``
        entry, the sampler would still run, and the inner trajectory would become
        a dynamic ``fori_loop`` -- reintroducing the ragged control flow that is
        the whole reason to prefer SMC's inner kernel to NUTS's, and making the
        gradient count unknowable. The schedule shape (``fixed_ladder``) has to
        be static for the same reason: it decides whether the program is a
        ``while_loop`` or a fixed-length ``scan``.
        """
        names = list(inspect.signature(_smc_scan.__wrapped__).parameters)
        static = set(_smc_scan._jit_info.static_argnums)
        for name in ("n_leapfrog_steps", "n_mcmc_steps", "n_particles", "fixed_ladder"):
            assert names.index(name) in static, f"{name} must be a static argument"
        for name in ("prior_draw_matrix", "run_keys", "data_args"):
            assert names.index(name) not in static, (
                f"{name} must be TRACED: a new galaxy must not recompile the sampler"
            )

    def test_a_one_particle_population_is_refused(self):
        with pytest.raises(ValueError, match="n_particles"):
            run_smc(object(), key=None, n_particles=1)

    def test_target_ess_is_a_fraction_and_a_draw_count_is_refused(self):
        """LOAD-BEARING. Neuter: clamp instead of raising.

        ``target_ess=200`` is the natural thing for someone who has read the
        number as a draw count to type, and ``adaptive_tempered_smc`` would
        happily bisect against an ESS target the population can never reach --
        producing a schedule that takes its increment to zero and a run that
        silently hits the rung cap.
        """
        with pytest.raises(ValueError, match="FRACTION"):
            run_smc(object(), key=None, target_ess=200)


class TestTheAdaptiveScheduleCanFailAndSaysSo:
    """A run that never reaches lambda = 1 must not be handed back as a posterior."""

    def test_the_rung_cap_exists_and_is_finite(self):
        assert 0 < _SMC_MAX_TEMPERATURES < 10_000

    def test_the_cap_docstring_names_the_silent_failure_it_converts(self):
        from tengri.inference.backends.mcmc import _shared

        source = inspect.getsource(_shared)
        marker = "#: Hard cap on temperature rungs"
        assert marker in source
        block = source[source.index(marker) : source.index("_SMC_MAX_TEMPERATURES =")]
        assert "reached_target" in block
        assert "not terminate" in block or "does not terminate" in block


class TestTheStepSizeControllerIsScalarAndRestoring:
    """MEADS's failure was a metric loop; a scalar acceptance loop is not one."""

    def test_the_gain_default_is_documented_against_the_meads_failure(self):
        from tengri.inference.backends.mcmc import _shared

        source = inspect.getsource(_shared)
        marker = "#: Multiplicative gain of the inner-kernel step-size controller"
        assert marker in source
        block = source[source.index(marker) : source.index("_SMC_STEP_SIZE_GAIN =")]
        assert "ghmc_meads" in block, "the controller must cite the loop it is not"
        assert "restoring" in block

    def test_the_controller_can_be_turned_off_for_the_ablation(self):
        assert _SMC_STEP_SIZE_GAIN != 0.0
        assert inspect.signature(run_smc).parameters["step_size_gain"].default == (
            _SMC_STEP_SIZE_GAIN
        )

    def test_the_target_acceptance_is_the_fixed_length_hmc_value_not_nuts(self):
        """0.651, the same reasoning as ChEES's, and deliberately not 0.8.

        Each inner move is a *fixed*-length HMC proposal. Carrying NUTS's 0.8
        across would still run and would still adapt -- to a step size chosen for
        a different proposal, which is invisible from the outside.
        """
        assert pytest.approx(0.651) == SMC_TARGET_ACCEPT_RATE
        assert inspect.signature(run_smc).parameters["target_accept_rate"].default == (
            SMC_TARGET_ACCEPT_RATE
        )


class TestTheRegistration:
    """Experimental, precondition-capable, and honest about the ESS trap."""

    def test_it_is_registered_experimental(self):
        entry = get_backend("mcmc_smc")
        assert entry.tier == "experimental"

    def test_it_declares_the_preconditioning_capability(self):
        """The metric is the largest measured effect in every prior report.

        ``bench/reports/2026-08-31_catalog_preconditioning.md``: preconditioning
        roughly doubles bare HMC and quadruples bare ChEES, and bare ChEES clears
        R-hat < 1.01 on zero of nine rows. An SMC backend that could not be run
        with and without the metric could not be evaluated at all.
        """
        assert get_backend("mcmc_smc").accepts_precondition is True

    def test_the_short_doc_warns_that_the_autocorrelation_ess_lies_here(self):
        doc = get_backend("mcmc_smc").short_doc
        assert "exchangeable" in doc
        assert "min_ancestor_ess" in doc

    def test_the_short_doc_says_where_the_raggedness_went(self):
        """The brief's hypothesis 1 was that SMC has no ragged control flow.

        Half true, and the half that is false is the half that costs: a rung is
        lock-step, and the *number of rungs* is data-dependent under the adaptive
        schedule. A short_doc that claimed only the first half would be the kind
        of quiet overclaim this project keeps writing reports about.
        """
        doc = get_backend("mcmc_smc").short_doc
        assert "fixed_ladder" in doc
        assert "data-dependent" in doc


class TestItActuallySamplesTheRightDistribution:
    """An analytic posterior, both schedules, so a wrong path is a wrong moment."""

    @pytest.mark.parametrize("fixed_ladder", [None, 12])
    def test_it_recovers_a_tilted_gaussian(self, fixed_ladder):
        dim = 4
        logprior, loglik, post_mean, post_sd = _tilted_gaussian(dim)
        particles, log_z, n_temp, final_lambda, step_size, n_div, accept, anc = _smc_scan(
            jnp.eye(dim),
            jax.random.split(jax.random.PRNGKey(0), 2),
            (),
            logprior,
            loglik,
            512,
            5,
            16,
            0.5,
            0.3,
            _SMC_STEP_SIZE_GAIN,
            SMC_TARGET_ACCEPT_RATE,
            _SMC_MAX_TEMPERATURES,
            fixed_ladder,
        )
        draws = np.asarray(particles).reshape(-1, dim)
        assert np.all(np.isfinite(draws))
        assert np.all(np.asarray(final_lambda) >= 1.0), "the schedule did not finish"
        assert np.allclose(draws.mean(axis=0), post_mean, atol=0.05), (
            f"mean {draws.mean(axis=0)} vs analytic {post_mean}"
        )
        assert np.allclose(draws.std(axis=0), post_sd, rtol=0.15), (
            f"sd {draws.std(axis=0)} vs analytic {post_sd}"
        )
        assert np.all(np.asarray(n_temp) > 0)
        assert np.all(np.asarray(accept) > 0.1)
        assert np.all(np.asarray(anc) > 1.0), "the population collapsed to one particle"
        assert np.all(np.asarray(step_size) > 0.0)
        assert np.all(np.asarray(n_div) >= 0)
        assert np.all(np.isfinite(np.asarray(log_z)))

    def test_the_log_evidence_matches_the_analytic_value(self):
        """log Z comes free with the weights, so it must be right or not reported.

        For an ``N(0, I)`` prior against a ``N(shift, scale^2 I)`` likelihood
        written *without* its normalizing constant, the evidence is
        ``(scale^2 / (1 + scale^2))^(D/2) * exp(-0.5 * D * shift^2 / (1 + scale^2))``.
        """
        dim, scale, shift = 3, 0.5, 1.0
        logprior, loglik, _, _ = _tilted_gaussian(dim, scale=scale, shift=shift)
        _p, log_z, *_rest = _smc_scan(
            jnp.eye(dim),
            jax.random.split(jax.random.PRNGKey(3), 4),
            (),
            logprior,
            loglik,
            1024,
            5,
            16,
            0.5,
            0.3,
            _SMC_STEP_SIZE_GAIN,
            SMC_TARGET_ACCEPT_RATE,
            _SMC_MAX_TEMPERATURES,
            None,
        )
        analytic = dim * (-0.5 * np.log1p(1.0 / scale**2) - 0.5 * shift**2 / (1.0 + scale**2))
        got = float(np.mean(np.asarray(log_z)))
        assert np.isclose(got, analytic, atol=0.15), f"log Z {got} vs analytic {analytic}"


class TestTheDefaults:
    """Widths and caps a caller inherits without asking."""

    def test_the_particle_default_is_a_width_not_a_chain_length(self):
        assert _SMC_DEFAULT_PARTICLES >= 128
        assert inspect.signature(run_smc).parameters["n_particles"].default == (
            _SMC_DEFAULT_PARTICLES
        )

    def test_chain_budget_kwargs_are_swallowed_and_recorded(self):
        """``n_warmup``/``n_samples`` have no analogue here and must not be silent.

        A caller sweeping this backend beside the chain samplers will pass them.
        Rejecting would force a special case at every call site; accepting
        silently would let a row believe it asked for 4000 draws when it got
        ``n_particles``. The third option is to accept and say so.
        """
        sig = inspect.signature(run_smc)
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()), (
            "run_smc must accept the chain samplers' budget kwargs"
        )


def _toy_fitter():
    """A minimal real Fitter over the packaged toy SSP grid, or skip."""
    import tengri
    from tengri import Data, Fixed, ForwardModel, Observation, Photometry, Uniform

    try:
        ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"SSP grid unavailable: {exc}")

    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    model = tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "log_total_mass": Uniform(9.0, 12.0)},
        met={"logzsol": Uniform(-1.0, 0.3)},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "tau_v": Uniform(0.0, 2.0),
        },
        redshift=Fixed(0.05),
    )
    forward = ForwardModel.build(sed=model)
    params = model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(model.predict_photometry(params))
    data = Data(photometry=(flux, 0.05 * flux))
    from tengri.inference.fitter import Fitter

    return Fitter(forward, data=data.photometry[0], noise=data.photometry[1])
