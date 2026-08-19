# SPDX-License-Identifier: BSD-3-Clause
"""Metric preconditioning is a coordinate change, not a model change (#1301).

``precondition=True`` samples ``H(A zeta)`` instead of ``H(xi)`` and maps the draws
back with ``xi = A zeta``. A linear reparametrization has a constant Jacobian, so
the posterior is untouched — only the sampler's geometry changes.

The failure this guards against is silent and severe: if the backend forgets to map
positions back (in ``nuts.py``, ``positions = problem.restore(positions)``), every
returned "posterior" is expressed in the whitened coordinates. The numbers still
look like plausible parameter values, the fit still reports a step size and a
divergence count, and nothing raises.

**Bounds cannot see that failure, and this file used to claim they could.** Every
prior standardizes through a constrained bijection —
``Uniform.unstandardize(xi) = lo + (hi - lo) * Phi(xi)`` — and the Gaussian CDF maps
all of the reals into ``(lo, hi)``. So draws left in the whitened basis remain *in
range*: they do not overflow the prior box, they **collapse against its edges**
(measured on this model, ``dust_tau_bc`` goes from a healthy ``[0.21, 3.72]`` spread
to exactly ``[0.0, 0.0]``, because ``Phi(-7.4) ~ 1e-13``). A bounds check is blind to
a basis error by construction, not by accident.

What does see it is asking whether the draws **explain the data**. Draws in the wrong
basis were never evaluated against the photometry in that basis, so their
log-posterior is catastrophically low — and that stays true however badly or well the
chain mixed. The measured separation is ~2500x (:data:`_DEFICIT_GATE`). Unlike a mean
comparison it needs neither a second chain nor a usable effective sample size, which
matters here: 250 draws buy an ESS of about 2.5.

The tests divide the work by what each statistic can actually resolve:

* :func:`test_preconditioned_samples_respect_the_prior_support` — the primary guard,
  unconditional, run against **every** backend that declares
  ``accepts_precondition``, read from the live registry so a sampler added later
  inherits it. Checks the deficit (a basis error) *and* the bounds (a missing
  standardization), and carries its own negative control: it undoes the inverse map
  on the same draws it just passed and requires the bar to notice. A guard whose
  threshold has never been shown to be reachable is not a guard.
* A second, weaker angle — comparing the two arms' posterior means — lives in
  ``tests/integration/test_preconditioning_posterior_invariance.py``. It is
  sensitive to a *partial* mapping error that still lands in the typical set, which
  the deficit would not notice, but it is gated on convergence and costs 155 s
  against this file's 53 s, so it stays in the nightly tier.

**Why this file is in ``tests/contract/``.** The heavy trees are auto-marked
``slow`` by directory (``_SLOW_TREES`` in ``tests/conftest.py``), so while all of
this sat in ``tests/integration/`` it ran *only* on the nightly schedule —
including through both pull requests that rewrote it (#1525, #1526), whose green
boards showed ``slow: skipped`` beside twelve passing jobs. A guard against a
silent failure that no pull request runs is a guard in name only.

Measured 2026-08-04: this file is ~53 s of sampling at ~4.8 GB peak. The mean
comparison that stayed behind was 71 % of the old file's runtime on its own,
because its ``precondition=False`` arm samples an unwhitened ``cond ~ 1e5``
posterior and NUTS pays for that in tree depth — the very effect under test. That
split is why gating this file costs the ``contract`` shard minutes rather than
tens of minutes.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, Fixed, ForwardModel, NoiseModel, Observation, Photometry, SEDModel

pytestmark = pytest.mark.contract

BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_j", "2mass_h"]
OVERRIDES = {
    "sfh_dpl_alpha": 2.0,
    "sfh_dpl_beta": 1.5,
    "sfh_dpl_log_total_mass": 10.5,
    "dust_tau_diff": 0.2,
}


def _model(ssp):
    obs = Observation(
        photometry=Photometry.from_names(BANDS),
        noise=NoiseModel(calibration_floor=0.02, student_t_dof=None),
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        met={"logzsol": Fixed(-0.3)},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "all_params": FREE},
        neb={"type": "none"},
        redshift=Fixed(0.1),
        apply_igm=False,
    )


def _mock(model):
    """A valid full parameter set: a prior draw with a few values pinned."""
    drawn = model.spec.sample(jax.random.PRNGKey(0))
    params = {
        **model.spec.get_fixed_values(),
        **drawn,
        **{k: jnp.array(v) for k, v in OVERRIDES.items() if k in drawn},
    }
    mock = model.mock(params, snr=30.0, key=jax.random.PRNGKey(1))
    return np.asarray(mock.flux_obs), np.asarray(mock.noise)


def _capable_backends() -> list[str]:
    """Every backend the registry says can whiten its metric.

    Read at collection time so a newly registered Hamiltonian sampler is covered
    without editing this file — the point of making the capability a declaration.
    """
    import tengri  # noqa: F401  (registers the backends)
    from tengri.inference._backend_registry import all_backends

    return sorted(e.name for e in all_backends() if e.accepts_precondition)


def _fit(model, data, noise, *, precondition, method="mcmc_nuts"):
    return ForwardModel.build(sed=model).fit(
        data,
        noise,
        method=method,
        key=jax.random.PRNGKey(7),
        n_warmup=250,
        n_samples=250,
        n_chains=1,
        dense_mass_matrix=False,
        precondition=precondition,
        verbose=False,
    )


# ── Thresholds. Every one of these was measured, not chosen. ─────────────────

#: Largest median log-posterior deficit a healthy posterior may show, as a multiple
#: of the ``D/2`` a correct sampler produces by construction.
#:
#: Where ``D/2`` comes from: near the mode the log-density is locally quadratic, so a
#: draw's shortfall below it is distributed as ``0.5 * chi2_D`` — median ``~D/2``, and
#: a 99th percentile of about ``9.3`` at ``D = 7``. Measured on this model: a real
#: preconditioned fit sits at **1.06x D/2**, and the *same draws* with the inverse map
#: undone sit at **2527x**. Any bar from 20 to 1000 separates them. 20 is ~2x above
#: the theoretical p99 and still 126x below the failure.
_DEFICIT_GATE = 20.0


@pytest.fixture(scope="module")
def objective(ssp_data_wne):
    """Model, data, and the standardized objective a backend actually sees.

    Module-scoped because the MAP expansion point costs a few seconds and several
    tests need it. Sharing it also pins them to the *same* point, which is what
    makes their numbers comparable to one another.
    """
    from jax.flatten_util import ravel_pytree

    from tengri.inference.context import InferenceContext
    from tengri.inference.fitter import Fitter
    from tengri.inference.preconditioning import (
        metric_preconditioner,
        negative_hessian_metric,
        prepare_preconditioning,
    )

    model = _model(ssp_data_wne)
    data, noise = _mock(model)
    forward = ForwardModel.build(sed=model)
    rmap = forward.fit(
        data,
        noise,
        method="map",
        n_steps=3000,
        n_restarts=2,
        verbose=False,
        key=jax.random.PRNGKey(5),
    )

    ctx = InferenceContext.from_target(Fitter(forward, data, noise, approx="auto"))
    nlp, data_args = ctx.neg_log_posterior_fn, ctx.data_args
    flat0, unravel = ravel_pytree(ctx.initial_params(jax.random.PRNGKey(0), init_from=rmap))

    def log_p(flat, args):
        return -nlp(unravel(flat), args)

    return SimpleNamespace(
        model=model,
        data=data,
        noise=noise,
        log_p=log_p,
        data_args=data_args,
        flat0=flat0,
        unravel=unravel,
        # The objective's own parameter order — not spec.free_params, which need not
        # agree with how ravel_pytree laid the vector out.
        names=list(unravel(flat0).keys()),
        # log-density at the mode: the reference every deficit is measured from.
        lp_mode=float(log_p(flat0, data_args)),
        # As deployed: partial whitening at DEFAULT_WHITENING_STRENGTH, which is what
        # a backend actually applies and therefore what a bug would fail to undo.
        preconditioner=prepare_preconditioning(
            log_p, flat0, data_args, precondition=True
        ).preconditioner,
        # Full strength, so ``A A^T = G^-1`` exactly — the Laplace covariance. Used to
        # *generate* typical-set draws, not to whiten anything.
        laplace=metric_preconditioner(negative_hessian_metric(log_p, flat0, data_args)),
    )


def _standardize(model, samples, names) -> jnp.ndarray:
    """Physical draws → the standardized latents the objective is written in.

    ``Distribution.standardize`` is the documented inverse of the ``unstandardize``
    the sampler's output already went through, so this reconstructs the ``xi`` the
    backend was holding — the vector the log-posterior can be evaluated at.
    """
    missing = [n for n in names if not hasattr(model.spec.get_distribution(n), "standardize")]
    assert not missing, (
        f"priors for {missing} do not implement standardize(), so their draws cannot "
        "be mapped back into the objective's coordinates — the deficit guard would "
        "silently cover fewer parameters than it claims"
    )
    return jnp.stack(
        [
            model.spec.get_distribution(n).standardize(jnp.asarray(np.asarray(samples[n])))
            for n in names
        ],
        axis=1,
    )


def _median_deficit_ratio(obj, xi_stack) -> float:
    """Median log-posterior shortfall below the mode, in units of ``D / 2``.

    ``~1.0`` is what a correct sampler produces by construction; see
    :data:`_DEFICIT_GATE`.
    """
    lp = jax.vmap(lambda flat: obj.log_p(flat, obj.data_args))
    deficit = obj.lp_mode - np.asarray(lp(jnp.asarray(xi_stack)))
    finite = deficit[np.isfinite(deficit)]
    assert finite.size, "every draw scored a non-finite log-posterior"
    return float(np.median(finite)) / (len(obj.names) / 2.0)


def test_at_least_one_backend_declares_the_capability():
    """Anti-vacuity: an empty parametrization would silently check nothing."""
    assert _capable_backends(), "no backend declares accepts_precondition"


@pytest.mark.parametrize("method", _capable_backends())
def test_preconditioned_samples_respect_the_prior_support(method, objective):
    """Whitened draws mapped back must explain the data *and* lie inside the priors.

    The primary guard, and the only unconditional one. Two assertions, because the
    transform can be dropped at two different places and the two failures do not
    look alike:

    * **deficit** — catches a forgotten ``restore()``, i.e. draws reported in the
      whitened basis. Bounds are blind to this (see the module docstring): the
      constrained bijection keeps such draws in range and merely collapses them onto
      the prior edges. Asking whether they explain the photometry resolves it, with
      no reference to a second chain or to how well anything mixed.
    * **bounds** — catches a missing *standardization*, i.e. raw ``xi`` returned as
      though it were physical. ``xi`` is an unconstrained real, so it does overflow.

    Parametrized over the live registry so all four Hamiltonian backends carry the
    same guarantee, rather than only the one this file happened to name.
    """
    model = objective.model
    result = _fit(model, objective.data, objective.noise, precondition=True, method=method)

    xi = _standardize(model, result.samples, objective.names)
    ratio = _median_deficit_ratio(objective, xi)
    assert ratio < _DEFICIT_GATE, (
        f"{method}: posterior draws sit {ratio:.0f}x D/2 below the mode (healthy is "
        f"~1x, bar is {_DEFICIT_GATE:g}x) — they do not explain the data, which is "
        "what a posterior still expressed in whitened coordinates looks like"
    )

    # Negative control, on the very draws just checked. Undo the inverse map the
    # backend applied and confirm the bar above notices. Without this the gate could
    # be any number no posterior could reach and the assertion would still be green
    # — which is exactly how the bounds check below spent this file's whole history
    # looking like a mapping test while being unable to fail.
    leaked = _median_deficit_ratio(objective, jax.vmap(objective.preconditioner.to_latent)(xi))
    assert leaked > _DEFICIT_GATE, (
        f"{method}: undoing the inverse map left the deficit at {leaked:.1f}x D/2, "
        f"under the {_DEFICIT_GATE:g}x bar — the guard cannot see a "
        "whitened-coordinate leak, so passing it means nothing"
    )

    checked = 0
    for name in model.spec.free_params:
        if name not in result.samples:
            continue
        dist = model.spec.get_distribution(name)
        low, high = getattr(dist, "lo", None), getattr(dist, "hi", None)
        if low is None or high is None:
            continue
        draws = np.asarray(result.samples[name])
        assert draws.min() >= float(low) - 1e-8, f"{name} below prior support"
        assert draws.max() <= float(high) + 1e-8, f"{name} above prior support"
        checked += 1
    assert checked > 0, "no bounded priors were checked — the guard would pass vacuously"


def test_gradients_through_the_real_model_are_exact_and_finite(objective):
    """The chain rule and finiteness, on the actual SED likelihood.

    The synthetic-Gaussian unit tests prove the algebra; this proves it survives the
    real forward model, where the Jacobian comes from SSP interpolation, dust
    attenuation and filter projection rather than a matrix multiply. A wrong-but-finite
    gradient raises nothing and would simply sample the wrong distribution.

    Reaches the objective the way a backend does (``InferenceContext`` over a
    ``Fitter``) because that *is* the internal seam under test.
    """
    log_p, data_args, flat0 = objective.log_p, objective.data_args, objective.flat0

    # The full-strength preconditioner from the fixture. Any invertible A exercises
    # the chain rule; reusing this one avoids a second Hessian at the same point.
    pc = objective.laplace
    wrapped = pc.wrap(log_p)

    # Chain rule on the real likelihood.
    zeta0 = pc.to_latent(flat0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        zeta = zeta0 + jnp.asarray(rng.standard_normal(flat0.shape[0]))
        got = np.asarray(jax.grad(wrapped)(zeta, data_args))
        want = np.asarray(pc.matrix.T @ jax.grad(log_p)(pc.to_xi(zeta), data_args))
        scale = max(float(np.max(np.abs(want))), 1e-12)
        assert np.max(np.abs(got - want)) / scale < 1e-8, "chain rule violated on the SED model"
        assert np.all(np.isfinite(got)), "non-finite gradient on the SED model"
        assert np.max(np.abs(got)) > 0.0, "gradient identically zero — the objective is flat"
