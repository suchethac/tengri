# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2087 (defect 2): the frozen-parameter dead-fit
check flagged every ``Fixed`` parameter.

Bug: ``Posterior.samples`` carries every parameter the forward model consumes.
``Fitter._to_physical`` merges the ``Fixed`` values into the params dict, so
the vmap over draws broadcasts each into a constant array. The construction
guard looped over ``self.samples.items()`` and called any zero-variance column
a frozen parameter, so every model with at least one ``Fixed`` parameter and
100+ draws warned "dead fit", naming the pinned parameters and never the free
ones (41 names on the fit that filed the issue, none of them free).

Guard: ``Posterior.free_names`` reads the free names off the model's spec; the
frozen check restricts itself to those. A posterior built without a model
cannot tell and keeps checking every column.

``Parameters.free_params`` excludes the stochastic-SFH field latent by design
(it rides under the sampler's key ``psd_xi``, not as a named distribution —
see ``Parameters.n_latent``), so ``free_names`` appends ``"psd_xi"`` itself
when the spec is stochastic and the latent is present in ``samples`` (R12).
Otherwise a frozen field latent would ride along as silently as a frozen
``Fixed`` column once did.

Mutation checks:
1. ``test_fixed_parameters_do_not_trigger_the_frozen_warning`` and
   ``test_a_frozen_free_parameter_is_named_alone``: make the loop use
   ``list(self.samples)`` regardless of ``free_names``.
2. ``test_free_names_lists_the_spec_free_params``: return
   ``tuple(self._model.spec.fixed_params)`` instead.
3. ``test_without_a_model_every_column_is_checked``: make ``free_names``
   return ``()`` when ``_model`` is None.
4. ``test_free_names_includes_the_stochastic_field_latent`` and
   ``test_a_frozen_field_latent_is_flagged_dead``: revert ``free_names`` to
   ``tuple(spec.free_params)`` (drop the ``psd_xi`` append).
5. ``test_convergence_check_ignores_fixed_parameters``: revert
   ``convergence_check``'s frozen loop to ``result.samples.items()``.
6. ``test_a_saved_posterior_reloads_without_a_false_dead_fit_warning``: make
   ``Posterior.load()`` stop passing ``_free_names`` (or ``save()`` stop
   writing the attribute).
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri.components.nebular import BakedInNebularWarning
from tengri.config.exceptions import DeadFitWarning
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.regression_bug

_N = 200
_FREE = "sfh_dpl_log_total_mass"
_N_GRID = 4


@pytest.fixture
def model(synthetic_ssp, simple_observation):
    """One free parameter (mass), eight pinned. Same object type ``fit`` attaches as ``_model``."""
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(1.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BakedInNebularWarning)
        return SEDModel(spec, synthetic_ssp, observation=simple_observation)


@pytest.fixture
def stochastic_model(synthetic_ssp, simple_observation):
    """Same one-free-parameter model, plus a stochastic-SFH field latent.

    ``n_grid=4`` keeps the field small (build stays well under a second);
    the two field GP hyperparameters are pinned so ``sfh_dpl_log_total_mass``
    is still the only *named* free parameter — the field latent moves
    through ``psd_xi``, not through a named distribution (#2087, R12).
    """
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(1.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        sfh_field_psd_sigma=Fixed(0.3),
        sfh_field_psd_tau_myr=Fixed(100.0),
        mean_sfh_type=["dpl", "field"],
        n_grid=_N_GRID,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BakedInNebularWarning)
        return SEDModel(spec, synthetic_ssp, observation=simple_observation)


def _samples(model, free_draws):
    fixed = {k: jnp.full((_N,), float(v)) for k, v in model.spec.get_fixed_values().items()}
    return {**fixed, _FREE: free_draws}


def _posterior(model, free_draws):
    return Posterior(
        samples=_samples(model, free_draws),
        params={_FREE: jnp.array(10.0)},
        method="mcmc_nuts",
        wall_time_s=1.0,
        diagnostics={"n_divergent": 0, "n_samples": _N, "n_chains": 1},
        _model=model,
    )


def _stochastic_posterior(model, free_draws, psd_xi_draws):
    samples = _samples(model, free_draws)
    samples["psd_xi"] = psd_xi_draws
    return Posterior(
        samples=samples,
        params={_FREE: jnp.array(10.0)},
        method="mcmc_nuts",
        wall_time_s=1.0,
        diagnostics={"n_divergent": 0, "n_samples": _N, "n_chains": 1},
        _model=model,
    )


def test_free_names_lists_the_spec_free_params(model):
    post = _posterior(model, 10.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (_N,)))
    assert post.free_names == (_FREE,)


def test_fixed_parameters_do_not_trigger_the_frozen_warning(model):
    free_draws = 10.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (_N,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _posterior(model, free_draws)


def test_a_frozen_free_parameter_is_named_alone(model):
    with pytest.warns(DeadFitWarning, match=r"dead fit") as record:
        _posterior(model, jnp.full((_N,), 10.0))
    message = str(record[0].message)
    assert f"'{_FREE}'" in message
    for pinned in model.spec.fixed_params:
        assert f"'{pinned}'" not in message, pinned


def test_without_a_model_every_column_is_checked():
    post_kwargs = dict(
        params={"x": jnp.array(5.0)},
        method="mcmc_hmc",
        wall_time_s=1.0,
        diagnostics={"n_divergent": 0, "n_samples": _N},
    )
    with pytest.warns(DeadFitWarning, match=r"'pinned_looking'"):
        post = Posterior(samples={"pinned_looking": jnp.full((_N,), 3.0)}, **post_kwargs)
    assert post.free_names is None


def test_free_names_includes_the_stochastic_field_latent(stochastic_model):
    post = _stochastic_posterior(
        stochastic_model,
        10.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(10), (_N,)),
        jax.random.normal(jax.random.PRNGKey(11), (_N, _N_GRID)),
    )
    assert post.free_names == (_FREE, "psd_xi")


def test_a_frozen_field_latent_is_flagged_dead(stochastic_model):
    free_draws = 10.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(12), (_N,))
    frozen_xi = jnp.full((_N, _N_GRID), 5.0)
    with pytest.warns(DeadFitWarning, match=r"dead fit") as record:
        _stochastic_posterior(stochastic_model, free_draws, frozen_xi)
    message = str(record[0].message)
    assert "'psd_xi'" in message
    assert f"'{_FREE}'" not in message


def test_a_moving_field_latent_does_not_warn(stochastic_model):
    free_draws = 10.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(13), (_N,))
    moving_xi = jax.random.normal(jax.random.PRNGKey(14), (_N, _N_GRID))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _stochastic_posterior(stochastic_model, free_draws, moving_xi)


def test_convergence_check_ignores_fixed_parameters(model):
    """The report reads the same free names the construction guard does (#2087, R18)."""
    from tengri.analysis.plotting.convergence import convergence_check

    post = _posterior(model, 10.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(20), (_N,)))
    info = convergence_check(post, verbose=False)
    assert "frozen_params" not in info, info.get("frozen_params")
    assert not [w for w in info["warnings"] if "FROZEN" in w], info["warnings"]


def test_convergence_check_names_only_the_frozen_free_parameter(model):
    from tengri.analysis.plotting.convergence import convergence_check

    with pytest.warns(DeadFitWarning):
        post = _posterior(model, jnp.full((_N,), 10.0))
    info = convergence_check(post, verbose=False)
    assert info["frozen_params"] == [_FREE]
    frozen_warning = " ".join(w for w in info["warnings"] if "FROZEN" in w)
    assert _FREE in frozen_warning
    for pinned in model.spec.fixed_params:
        assert pinned not in frozen_warning, pinned


def test_a_saved_posterior_reloads_without_a_false_dead_fit_warning(model, tmp_path):
    """The free names ride in the file, so a model-less reload judges the same columns.

    ``Posterior.load(path)`` with no ``model=`` used to leave ``free_names``
    ``None``, which put every ``Fixed`` column back under the frozen check --
    the #2087 false positive, re-created on every reload of a saved real fit.
    """
    post = _posterior(model, 10.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(21), (_N,)))
    path = tmp_path / "post.h5"
    post.save(str(path))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = Posterior.load(str(path))
    assert loaded.free_names == (_FREE,)
    assert loaded._model is None
