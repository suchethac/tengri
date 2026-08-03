# SPDX-License-Identifier: BSD-3-Clause
"""Metric preconditioning is a coordinate change, not a model change (#1301).

``precondition=True`` samples ``H(A zeta)`` instead of ``H(xi)`` and maps the draws
back with ``xi = A zeta``. A linear reparametrization has a constant Jacobian, so
the posterior is untouched — only the sampler's geometry changes.

The failure this guards against is silent and severe: if the backend forgets to map
positions back, every returned "posterior" is expressed in the whitened
coordinates. The numbers still look like plausible parameter values, the fit still
reports a step size and a divergence count, and nothing raises.

The support check is parametrized over **every backend that declares
``accepts_precondition``**, read from the live registry, so a sampler added later
inherits the guard without anyone remembering to extend this file. It asserts bounds
rather than comparing posteriors because bounds are deterministic: whitened draws
ignore the priors' physical limits regardless of how well any particular sampler
mixed, so the check does not go flaky when a new backend has different tuning needs.
"""

from __future__ import annotations

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
        stellar={"met_logzsol": Fixed(-0.3)},
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FREE},
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


#: Split-R-hat above which a chain is not converged enough for its posterior mean
#: to mean anything. The usual 1.05.
_RHAT_GATE = 1.05


def test_preconditioning_leaves_the_posterior_unchanged(ssp_data_wne):
    """Same posterior, different coordinates — the invariant a mapping bug breaks.

    Gated on convergence (#1498). Comparing the means of two independent 250/250
    NUTS runs only probes the inverse map when **both** chains have converged. If
    one has not, the means differ because that chain is somewhere else, not
    because the draws are still whitened — and preconditioning is documented to
    mix *worse* on exactly this configuration (D=8 dust, median ESS/s ratio 0.62,
    range 0.10–1.13). Ungated, this failed deterministically on the CI runner at
    2.13 sd while passing on macOS, and the difference was mixing, not a bug.

    The invariant itself is guarded unconditionally by
    :func:`test_preconditioned_samples_respect_the_prior_support`, which checks
    bounds rather than means and so does not depend on how well anything mixed.
    """
    model = _model(ssp_data_wne)
    data, noise = _mock(model)

    plain = _fit(model, data, noise, precondition=False)
    preconditioned = _fit(model, data, noise, precondition=True)

    unconverged = {
        arm: {k: round(v, 3) for k, v in fit.rhat().items() if v > _RHAT_GATE}
        for arm, fit in (("plain", plain), ("preconditioned", preconditioned))
    }
    if any(unconverged.values()):
        pytest.skip(
            "posterior means are not comparable — split-R-hat above "
            f"{_RHAT_GATE} in: { {k: v for k, v in unconverged.items() if v} }. "
            "A mean shift here would be mixing, not a whitening-mapping bug; the "
            "mapping itself is covered by the prior-support test."
        )

    for name in plain.samples:
        a = np.asarray(plain.samples[name])
        b = np.asarray(preconditioned.samples[name])
        if a.ndim != 1:
            continue
        spread = max(float(np.std(a)), 1e-3)
        shift = abs(float(np.mean(a) - np.mean(b))) / spread
        assert shift < 1.0, (
            f"{name}: posterior mean moved {shift:.2f} sd between the plain and "
            f"preconditioned runs ({np.mean(a):.4f} vs {np.mean(b):.4f}) — "
            "the draws are probably still in whitened coordinates"
        )


def test_at_least_one_backend_declares_the_capability():
    """Anti-vacuity: an empty parametrization would silently check nothing."""
    assert _capable_backends(), "no backend declares accepts_precondition"


@pytest.mark.parametrize("method", _capable_backends())
def test_preconditioned_samples_respect_the_prior_support(method, ssp_data_wne):
    """Whitened draws mapped back must land inside the priors' physical bounds.

    A missing inverse map shows up here even when the means happen to agree.
    """
    model = _model(ssp_data_wne)
    data, noise = _mock(model)
    result = _fit(model, data, noise, precondition=True, method=method)

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


def test_gradients_through_the_real_model_are_exact_and_finite(ssp_data_wne):
    """The chain rule and finiteness, on the actual SED likelihood.

    The synthetic-Gaussian unit tests prove the algebra; this proves it survives the
    real forward model, where the Jacobian comes from SSP interpolation, dust
    attenuation and filter projection rather than a matrix multiply. A wrong-but-finite
    gradient raises nothing and would simply sample the wrong distribution.

    Reaches the objective the way a backend does (``InferenceContext`` over a
    ``Fitter``) because that *is* the internal seam under test.
    """
    from jax.flatten_util import ravel_pytree

    from tengri.inference.context import InferenceContext
    from tengri.inference.fitter import Fitter
    from tengri.inference.preconditioning import metric_preconditioner, negative_hessian_metric

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

    pc = metric_preconditioner(negative_hessian_metric(log_p, flat0, data_args))
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
