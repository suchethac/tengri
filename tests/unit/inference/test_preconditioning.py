# SPDX-License-Identifier: BSD-3-Clause
"""Metric preconditioning of the standardized latent space (#1301).

Every tengri parameter is standardized, so the sampled objective is
``H(xi) = 1/2 chi^2 + 1/2 xi^T xi`` and the *prior* contributes exactly the
identity to the metric. That is NIFTy's setup, and NIFTy exploits it by handing
geoVI/MGVI the **position-dependent** Fisher metric ``I + J^T N^-1 J`` at every
iteration. NUTS instead freezes a mass matrix estimated from warmup draws, which
cannot cover the measured ``cond(grad^2 H) ~ 1e5`` on the field posterior.

These tests pin the linear change of variables that closes that gap:
``xi = A zeta`` with ``A A^T = G^-1`` for a metric ``G``, so the curvature in
``zeta`` is ``A^T G A = I``. A linear reparametrization changes the density only
by a constant Jacobian, so the sampled distribution is unchanged -- draws are
mapped back with ``xi = A zeta``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Uniform
from tengri.inference.preconditioning import (
    metric_preconditioner,
    negative_hessian_metric,
    preconditioned_logdensity,
)


def _ill_conditioned_metric(n=6, cond=1e5, seed=0):
    """A PD matrix with a prescribed condition number and a non-trivial rotation."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = np.geomspace(1.0, cond, n)
    return q @ np.diag(eig) @ q.T


def _gaussian_logdensity(metric):
    """``log p(xi) = -1/2 xi^T M xi`` as a 2-arg (position, data_args) callable."""

    def log_p(flat, data_args):
        return -0.5 * flat @ (jnp.asarray(metric) @ flat) + 0.0 * data_args

    return log_p


class TestNegativeHessianMetric:
    def test_recovers_the_metric_of_an_exact_gaussian(self):
        M = _ill_conditioned_metric()
        G = negative_hessian_metric(_gaussian_logdensity(M), jnp.zeros(M.shape[0]), 0.0)
        assert np.max(np.abs(np.asarray(G) - M)) / np.max(M) < 1e-10

    def test_floors_eigenvalues_at_the_prior_contribution(self):
        # A standardized prior contributes exactly I, and the Gauss-Newton
        # likelihood term is PSD, so no eigenvalue of the true metric is below 1.
        # Anything below is residual curvature; flooring keeps the metric PD.
        M = np.diag([0.01, 0.5, 4.0])
        G = np.asarray(negative_hessian_metric(_gaussian_logdensity(M), jnp.zeros(3), 0.0))
        assert np.allclose(np.linalg.eigvalsh(G), [1.0, 1.0, 4.0])

    def test_metric_is_positive_definite_even_at_a_saddle(self):
        # An indefinite Hessian (negative curvature) must still yield a usable metric.
        M = np.diag([-3.0, 2.0])
        G = np.asarray(negative_hessian_metric(_gaussian_logdensity(M), jnp.zeros(2), 0.0))
        assert np.all(np.linalg.eigvalsh(G) > 0.0)


class TestLinearPreconditioner:
    def test_roundtrip_is_the_identity(self):
        pc = metric_preconditioner(_ill_conditioned_metric())
        xi = jnp.asarray(np.random.default_rng(1).standard_normal(6))
        assert np.max(np.abs(np.asarray(pc.to_xi(pc.to_latent(xi)) - xi))) < 1e-8

    def test_preconditioned_curvature_is_the_identity(self):
        """The whole point: cond 1e5 -> 1 in the sampled coordinates."""
        M = _ill_conditioned_metric(cond=1e5)
        log_p = _gaussian_logdensity(M)
        pc = metric_preconditioner(negative_hessian_metric(log_p, jnp.zeros(6), 0.0))
        H_zeta = np.asarray(negative_hessian_metric(pc.wrap(log_p), jnp.zeros(6), 0.0, floor=0.0))
        assert np.linalg.cond(np.asarray(M)) > 1e4  # the problem is real
        assert abs(np.linalg.cond(H_zeta) - 1.0) < 1e-6

    def test_wrapped_density_equals_the_original_at_the_mapped_point(self):
        """Validity of the draws: sampling zeta and mapping back samples p(xi)."""
        M = _ill_conditioned_metric()
        log_p = _gaussian_logdensity(M)
        pc = metric_preconditioner(negative_hessian_metric(log_p, jnp.zeros(6), 0.0))
        zeta = jnp.asarray(np.random.default_rng(2).standard_normal(6))
        assert float(pc.wrap(log_p)(zeta, 0.0)) == pytest.approx(
            float(log_p(pc.to_xi(zeta), 0.0)), rel=1e-10
        )

    def test_wrapped_density_is_jittable_and_differentiable(self):
        log_p = _gaussian_logdensity(_ill_conditioned_metric())
        pc = metric_preconditioner(negative_hessian_metric(log_p, jnp.zeros(6), 0.0))
        wrapped = pc.wrap(log_p)
        g = jax.jit(jax.grad(wrapped))(jnp.ones(6), 0.0)
        assert g.shape == (6,)
        assert bool(jnp.all(jnp.isfinite(g)))
        assert jnp.any(g != 0.0), (
            "`g` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

    def test_identity_metric_is_a_no_op(self):
        """An already-white posterior must not be perturbed."""
        pc = metric_preconditioner(jnp.eye(4))
        xi = jnp.asarray([1.0, -2.0, 0.5, 3.0])
        assert np.max(np.abs(np.asarray(pc.to_xi(xi) - xi))) < 1e-12

    def test_rejects_a_non_positive_definite_metric(self):
        with pytest.raises(ValueError, match="positive definite"):
            metric_preconditioner(jnp.asarray([[1.0, 2.0], [2.0, 1.0]]))


class TestPreconditionedLogdensity:
    """The one-call seam a sampler backend uses."""

    def test_initial_position_maps_back_to_the_original(self):
        log_p = _gaussian_logdensity(_ill_conditioned_metric())
        xi0 = jnp.asarray(np.random.default_rng(3).standard_normal(6))
        _, pc, zeta0 = preconditioned_logdensity(log_p, xi0, 0.0)
        assert np.max(np.abs(np.asarray(pc.to_xi(zeta0) - xi0))) < 1e-8

    def test_full_strength_whitens_the_curvature_at_the_expansion_point(self):
        """``strength=1.0`` is exact whitening — the property the transform is named for."""
        M = _ill_conditioned_metric(cond=1e6)
        log_p = _gaussian_logdensity(M)
        xi0 = jnp.zeros(6)
        wrapped, _, zeta0 = preconditioned_logdensity(log_p, xi0, 0.0, strength=1.0)
        H = np.asarray(negative_hessian_metric(wrapped, zeta0, 0.0, floor=0.0))
        assert np.linalg.cond(np.asarray(M)) > 1e5
        assert abs(np.linalg.cond(H) - 1.0) < 1e-6

    def test_the_default_strength_deliberately_leaves_residual_curvature(self):
        """The default is partial (#1442): it must improve conditioning, not erase it.

        Erasing it is what full whitening does, and full whitening is what amplifies a
        misspecified metric without bound. The residual is the price of that bound, and
        for an exact metric it is exactly ``sqrt(cond)``.
        """
        from tengri.inference.preconditioning import DEFAULT_WHITENING_STRENGTH

        M = _ill_conditioned_metric(cond=1e6)
        log_p = _gaussian_logdensity(M)
        wrapped, _, zeta0 = preconditioned_logdensity(log_p, jnp.zeros(6), 0.0)
        got = np.linalg.cond(np.asarray(negative_hessian_metric(wrapped, zeta0, 0.0, floor=0.0)))
        assert 1.0 < got < np.linalg.cond(np.asarray(M)), (
            f"default strength left cond={got:.3e}; expected a strict improvement short of 1"
        )
        want = np.linalg.cond(np.asarray(M)) ** (1.0 - DEFAULT_WHITENING_STRENGTH)
        assert got == pytest.approx(want, rel=1e-4)


def _run_nuts(logdensity, init, *, seed, n_warmup=400, n_samples=600):
    """Minimal blackjax NUTS run, mirroring the backend's use.

    Returns
    -------
    draws : ndarray, shape (n_samples, D)
    leapfrog_per_draw : float
        Mean integrator steps per draw — the true cost of bad conditioning, since
        NUTS answers a stiff target by building deeper trees rather than by failing.
    step_size : float
    """
    import blackjax

    def ld(x):
        return logdensity(x, 0.0)

    warmup = blackjax.window_adaptation(blackjax.nuts, ld, is_mass_matrix_diagonal=True)
    (state, params), _ = warmup.run(jax.random.PRNGKey(seed), init, num_steps=n_warmup)
    kernel = blackjax.nuts.build_kernel()

    def step(s, k):
        s, info = kernel(k, s, ld, params["step_size"], params["inverse_mass_matrix"])
        return s, (s.position, info.num_integration_steps)

    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_samples)
    _, (draws, n_steps) = jax.lax.scan(step, state, keys)
    return (
        np.asarray(draws),
        float(np.mean(np.asarray(n_steps))),
        float(params["step_size"]),
    )


@pytest.mark.slow
def test_preconditioning_makes_an_ill_conditioned_gaussian_cheap_to_sample():
    """The payoff, measured where it actually shows up: integrator cost.

    A rotated cond-1e6 Gaussian does not make NUTS *fail* — NUTS compensates by
    shrinking the step size and building enormous trajectories, so both arms land
    on a similar answer. The cost is what differs, by nearly two orders of
    magnitude. Both facts are asserted: the mapped-back draws must still recover
    the true covariance (a linear reparametrization cannot change the target), and
    the trajectory cost must collapse.
    """
    n = 6
    M = _ill_conditioned_metric(n=n, cond=1e6, seed=5)
    true_var = np.diag(np.linalg.inv(M))
    log_p = _gaussian_logdensity(M)
    init = jnp.zeros(n)

    _, raw_leapfrog, raw_step = _run_nuts(log_p, init, seed=0)
    wrapped, pc, zeta0 = preconditioned_logdensity(log_p, init, 0.0)
    zeta_draws, pre_leapfrog, pre_step = _run_nuts(wrapped, zeta0, seed=0)
    pre_draws = np.asarray(jax.vmap(pc.to_xi)(jnp.asarray(zeta_draws)))

    def var_err(draws):
        return float(np.max(np.abs(np.var(draws, axis=0) - true_var) / true_var))

    # Correctness: sampling the reparametrized target and mapping back is unbiased.
    assert var_err(pre_draws) < 0.30, (
        f"preconditioned run did not recover the target ({var_err(pre_draws):.2f})"
    )
    # Efficiency: measured 77x fewer leapfrog steps and a 185x larger step size;
    # asserted at 10x so the test tracks the effect, not the exact arithmetic.
    assert raw_leapfrog / pre_leapfrog > 10.0, (
        f"no trajectory-cost gain: {raw_leapfrog:.1f} -> {pre_leapfrog:.1f} steps/draw"
    )
    assert pre_step / raw_step > 10.0, f"no step-size gain: {raw_step:.2e} -> {pre_step:.2e}"


# Backends whose runner takes ``precondition``. Hamiltonian samplers only: the flag
# whitens the metric the integrator sees, which is meaningless for the gradient-free
# and self-tuning kernels (raytrace, ess, mclmc).
PRECONDITIONABLE_BACKENDS = ("mcmc_nuts", "mcmc_hmc", "mcmc_dynamic_hmc")


@pytest.mark.parametrize("name", PRECONDITIONABLE_BACKENDS)
def test_registered_backend_accepts_precondition(name):
    """Regression: ``mcmc_hmc`` dispatches to ``run_hmc``, not ``run_dynamic_hmc``.

    Wiring the flag into ``dynamic_hmc.py`` alone left ``method="mcmc_hmc"`` raising
    ``TypeError: run_hmc() got an unexpected keyword argument 'precondition'`` at
    runtime — after the MAP had already been paid for. A signature check is cheap and
    catches it before any fit starts.
    """
    import inspect

    from tengri.inference._backend_registry import _BACKENDS

    params = inspect.signature(_BACKENDS[name].runner).parameters
    assert "precondition" in params, f"{name} runner does not accept `precondition`"
    assert params["precondition"].default is None, f"{name}: default must be the auto policy"


def test_the_signature_check_can_actually_fail():
    """Non-vacuity: the check above must reject a runner that lacks the kwarg."""
    import inspect

    def runner_without_it(context, *, key):
        return None

    assert "precondition" not in inspect.signature(runner_without_it).parameters


def test_building_the_preconditioner_is_documented_as_not_jit_safe():
    """The docstring says it raises under trace; pin that so the claim cannot drift.

    ``metric_preconditioner`` reads a concrete boolean to reject a non-PD metric — a
    guard that fails loudly is worth more than one that silently returns NaNs, but it
    does mean this must be called outside any transform.
    """
    metric = np.diag([1.0, 4.0])
    log_p = _gaussian_logdensity(metric)
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(lambda x: preconditioned_logdensity(log_p, x, 0.0)[2])(jnp.zeros(2))


class TestGradients:
    """Gradients through the whitening map must be exact, finite, and optimizable.

    Everything else rests on this: if ``grad`` is wrong the sampler is wrong, and if it
    is merely *finite but wrong* nothing raises — the chain just explores the wrong
    distribution. So the chain rule is checked as an identity, not a smoke test.
    """

    def test_gradient_obeys_the_chain_rule(self):
        """``grad_zeta H(A zeta) == A^T grad_xi H(xi)`` exactly, by construction."""
        M = _ill_conditioned_metric(cond=1e6)
        log_p = _gaussian_logdensity(M)
        pc = metric_preconditioner(negative_hessian_metric(log_p, jnp.zeros(6), 0.0))
        zeta = jnp.asarray(np.random.default_rng(4).standard_normal(6))

        got = np.asarray(jax.grad(pc.wrap(log_p))(zeta, 0.0))
        want = np.asarray(pc.matrix.T @ jax.grad(log_p)(pc.to_xi(zeta), 0.0))
        scale = max(float(np.max(np.abs(want))), 1e-12)
        assert np.max(np.abs(got - want)) / scale < 1e-10

    def test_gradient_matches_finite_differences(self):
        """Independent check that does not reuse the analytic chain rule."""
        M = _ill_conditioned_metric(n=4, cond=1e4)
        log_p = _gaussian_logdensity(M)
        pc = metric_preconditioner(negative_hessian_metric(log_p, jnp.zeros(4), 0.0))
        wrapped = pc.wrap(log_p)
        zeta = jnp.asarray([0.3, -0.7, 0.2, 0.9])

        analytic = np.asarray(jax.grad(wrapped)(zeta, 0.0))
        eps = 1e-6
        numeric = np.array(
            [
                float(wrapped(zeta.at[i].add(eps), 0.0) - wrapped(zeta.at[i].add(-eps), 0.0))
                / (2 * eps)
                for i in range(4)
            ]
        )
        assert np.max(np.abs(analytic - numeric)) / max(np.max(np.abs(numeric)), 1e-12) < 1e-6

    @pytest.mark.parametrize("radius", [1.0, 10.0, 100.0])
    def test_gradient_is_finite_far_from_the_expansion_point(self, radius):
        """The metric is built at one point; the gradient must survive leaving it."""
        M = _ill_conditioned_metric(cond=1e6)
        log_p = _gaussian_logdensity(M)
        pc = metric_preconditioner(negative_hessian_metric(log_p, jnp.zeros(6), 0.0))
        wrapped = pc.wrap(log_p)
        keys = jax.random.split(jax.random.PRNGKey(21), 8)
        for k in keys:
            g = jax.grad(wrapped)(jax.random.normal(k, (6,)) * radius, 0.0)
            assert bool(jnp.all(jnp.isfinite(g))), f"non-finite gradient at radius {radius}"
            assert jnp.any(g != 0.0), (
                "`g` is identically zero — finite is not enough, "
                "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
            )

    def test_gradient_descent_converges_where_the_raw_coordinates_stall(self):
        """ "Can we at least optimize?" — yes, and that is exactly what whitening buys.

        Plain gradient descent with a fixed step size is limited by the largest
        curvature (or it diverges) while progress along the flattest direction goes as
        ``1 - step * lambda_min``. At cond 1e6 that is hopeless. After whitening every
        direction has curvature 1, so one well-scaled step size fits all of them.
        """
        n = 6
        M = _ill_conditioned_metric(n=n, cond=1e6, seed=8)
        log_p = _gaussian_logdensity(M)
        start = jnp.ones(n)

        def descend(objective, x0, *, n_steps, step):
            grad_fn = jax.jit(jax.grad(objective))

            def body(x, _):
                return x + step * grad_fn(x, 0.0), None  # ascend the log-density

            x, _ = jax.lax.scan(body, x0, None, length=n_steps)
            return x

        # Largest stable step for the raw problem is set by its stiffest direction.
        raw_step = 1.0 / float(np.linalg.eigvalsh(M).max())
        raw = descend(log_p, start, n_steps=2000, step=raw_step)

        pc = metric_preconditioner(negative_hessian_metric(log_p, jnp.zeros(n), 0.0))
        pre_zeta = descend(pc.wrap(log_p), pc.to_latent(start), n_steps=2000, step=0.5)
        pre = pc.to_xi(pre_zeta)

        # The optimum is the origin; measure how far each run got, in Mahalanobis
        # distance so the comparison is scale-free.
        def distance(x):
            v = np.asarray(x)
            return float(np.sqrt(v @ np.asarray(M) @ v))

        assert distance(pre) < 1e-6, (
            f"preconditioned descent did not converge ({distance(pre):.2e})"
        )
        assert distance(raw) > 1e-3, (
            "raw descent unexpectedly converged — the test no longer discriminates "
            f"(distance {distance(raw):.2e})"
        )


class TestAutoPolicy:
    """``precondition=None`` resolves by dimension, mirroring `dense_mass_matrix` (#319).

    Preconditioning is a strict geometry win but costs one dense ``(D, D)`` Hessian
    (``O(D)`` backward passes through the forward model), so it cannot be unconditionally
    on. These pin the *semantics*; the threshold itself is set from measurement and
    referenced through the constant so the tests do not encode a guess.
    """

    def test_explicit_true_and_false_round_trip(self):
        from tengri.inference.preconditioning import (
            DEFAULT_WHITENING_STRENGTH,
            _resolve_whitening_strength,
        )

        assert _resolve_whitening_strength(True, 10_000) == DEFAULT_WHITENING_STRENGTH
        assert _resolve_whitening_strength(False, 2) is None

    @pytest.mark.parametrize("n_dim", [2, 7, 8, 30, 137, 1024, 10_000])
    def test_the_default_is_off_at_every_dimension(self, n_dim):
        """Opt-in (#1397). Default-on shipped, and it broke fits that had worked.

        Two failures, both on published notebooks:

        * nb01 could not run at all — a NaN MAP init makes the metric non-finite, and
          the Cholesky guard turned a working fit into a hard ``ValueError``.
        * nb07's photometry fit went from max R-hat 1.014 to **1.839** while reporting
          **zero** divergences. The usual health signal is inverted, so a divergence
          check scores the broken arm as the healthiest of the three.

        Against that, no throughput win was ever demonstrated: 5 seeds per config gave
        a median 1.87x ESS/s at D=7 but 0.84x — a loss — at D=8. A feature that can
        turn a converging fit into a non-converging one has to be asked for.
        """
        from tengri.inference.preconditioning import _resolve_whitening_strength

        assert _resolve_whitening_strength(None, n_dim) is None

    def test_explicit_true_is_honored_above_the_cost_threshold(self):
        """The cap advises on O(D^3) cost; it must not veto an explicit request."""
        from tengri.inference.preconditioning import (
            DEFAULT_WHITENING_STRENGTH,
            PRECONDITION_MAX_DIM,
            _resolve_whitening_strength,
        )

        got = _resolve_whitening_strength(True, PRECONDITION_MAX_DIM + 1)
        assert got == DEFAULT_WHITENING_STRENGTH

    def test_threshold_is_a_measured_dimension_not_a_sentinel(self):
        from tengri.inference.preconditioning import PRECONDITION_MAX_DIM

        assert isinstance(PRECONDITION_MAX_DIM, int)
        assert 8 <= PRECONDITION_MAX_DIM <= 2048, "threshold outside any measured regime"


class TestMcmcAutoDispatch:
    """``method="mcmc"`` picks raytrace above D=20, which has no metric to whiten.

    Passing the flag through unchanged raised
    ``TypeError: run_raytrace() got an unexpected keyword argument`` — the same class
    of wiring bug as ``mcmc_hmc`` dispatching to ``run_hmc``. Silently dropping it
    would be worse: the user asks for preconditioning, gets none, and nothing says so.
    """

    class _Spec:
        def __init__(self, n_free):
            self.n_free = n_free
            # The dispatcher compares n_latent since #1408; scalar-only stubs
            # have no vector latents, so the two counts coincide.
            self.n_latent = n_free

    class _Ctx:
        def __init__(self, n_free):
            self.spec = TestMcmcAutoDispatch._Spec(n_free)

    def test_explicit_request_that_cannot_be_honored_raises(self):
        from tengri.inference._registration import _mcmc_auto_pick

        with pytest.raises(ValueError, match="ray"):
            _mcmc_auto_pick(self._Ctx(200), key=None, precondition=True)

    def test_auto_and_disabled_are_dropped_quietly_for_raytrace(self):
        """No request was made, so there is nothing to warn about — but it must not crash."""
        import tengri.inference._registration as reg

        seen = {}

        def fake_raytrace(context, **kw):
            seen.update(kw)
            return "ok"

        original = reg._ctx_run_raytrace
        reg._ctx_run_raytrace = fake_raytrace
        try:
            for value in (None, False):
                seen.clear()
                assert reg._mcmc_auto_pick(self._Ctx(200), key=None, precondition=value) == "ok"
                assert "precondition" not in seen, "raytrace must not receive the flag"
        finally:
            reg._ctx_run_raytrace = original


def test_saddle_curvature_is_whitened_by_magnitude_not_floored_to_one():
    """At a non-stationary point, scale by |curvature| — flooring mis-scales badly.

    Found in the breadth sweep: a field fit whose MAP had not converged carried five
    negative-curvature directions with min eigenvalue -51. Flooring those to +1 left
    them mis-scaled by 51x, which is exactly the ``stiff@MAP = 51.04`` observed. Using
    the magnitude instead is the saddle-free Newton choice (Dauphin et al. 2014) and
    whitens them properly: a direction with curvature -51 is *steep*, not flat.
    """
    metric = np.diag([-51.0, 2.0, 0.5])
    log_p = _gaussian_logdensity(metric)
    pc = metric_preconditioner(negative_hessian_metric(log_p, jnp.zeros(3), 0.0))

    whitened = np.asarray(pc.matrix.T @ metric @ pc.matrix)
    stiffness = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (whitened + whitened.T)))))
    assert stiffness == pytest.approx(1.0, abs=1e-8), (
        f"steep negative-curvature direction left at stiffness {stiffness:.2f}"
    )


class TestPreconditionedProblem:
    """The one seam a sampler backend touches (#1359 follow-up).

    Before this, each of ``run_nuts`` / ``run_hmc`` / ``run_dynamic_hmc`` carried its
    own copy of resolve -> wrap -> ``if preconditioner is not None: positions @ A.T``.
    The last line is the dangerous one: omitting it returns draws in the *whitened*
    coordinates, which are finite, correctly shaped, and wrong. Nothing downstream can
    tell. Folding it into an object that is the identity when disabled removes the
    branch a fourth backend could forget.
    """

    @staticmethod
    def _problem(enabled, n=5, cond=1e4):
        from tengri.inference.preconditioning import prepare_preconditioning

        metric = _ill_conditioned_metric(n, cond)
        log_p = _gaussian_logdensity(metric)
        return (
            prepare_preconditioning(log_p, jnp.zeros(n), 0.0, precondition=enabled),
            log_p,
        )

    def test_disabled_problem_is_the_identity(self):
        problem, log_p = self._problem(False)
        assert problem.enabled is False
        assert problem.preconditioner is None
        assert problem.logdensity is log_p, "disabled must hand back the original density"
        draws = jnp.asarray(np.random.default_rng(0).standard_normal((7, 5)))
        np.testing.assert_array_equal(np.asarray(problem.restore(draws)), np.asarray(draws))

    def test_enabled_problem_reports_itself_enabled(self):
        problem, _ = self._problem(True)
        assert problem.enabled is True
        assert problem.preconditioner is not None

    def test_restore_maps_a_stack_of_draws_row_wise(self):
        problem, _ = self._problem(True)
        draws = jnp.asarray(np.random.default_rng(1).standard_normal((11, 5)))
        got = np.asarray(problem.restore(draws))
        want = np.stack([np.asarray(problem.preconditioner.to_xi(d)) for d in draws])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_restore_maps_a_single_draw(self):
        problem, _ = self._problem(True)
        draw = jnp.asarray(np.random.default_rng(2).standard_normal(5))
        np.testing.assert_allclose(
            np.asarray(problem.restore(draw)),
            np.asarray(problem.preconditioner.to_xi(draw)),
            rtol=0,
            atol=1e-12,
        )

    def test_restore_round_trips_the_transported_start(self):
        """``init_flat`` is handed to the sampler in zeta; restoring must undo it."""
        from tengri.inference.preconditioning import prepare_preconditioning

        metric = _ill_conditioned_metric(6, 1e5)
        log_p = _gaussian_logdensity(metric)
        xi0 = jnp.asarray(np.random.default_rng(3).standard_normal(6))
        problem = prepare_preconditioning(log_p, xi0, 0.0, precondition=True)
        np.testing.assert_allclose(
            np.asarray(problem.restore(problem.init_flat)), np.asarray(xi0), rtol=0, atol=1e-9
        )

    def test_wrapped_density_equals_the_original_at_the_mapped_point(self):
        problem, log_p = self._problem(True)
        zeta = jnp.asarray(np.random.default_rng(4).standard_normal(5))
        assert float(problem.logdensity(zeta, 0.0)) == pytest.approx(
            float(log_p(problem.preconditioner.to_xi(zeta), 0.0)), rel=1e-12
        )

    def test_restoring_without_the_transpose_would_give_a_different_answer(self):
        """Neuter check: the round-trip assertions must be sensitive to the transpose.

        ``A`` is triangular and non-symmetric, so ``zeta @ A`` and ``zeta @ A.T`` differ.
        If they did not, the tests above would pass on a broken ``restore``.
        """
        problem, _ = self._problem(True)
        draws = jnp.asarray(np.random.default_rng(5).standard_normal((4, 5)))
        right = np.asarray(problem.restore(draws))
        wrong = np.asarray(draws @ problem.preconditioner.matrix)
        assert np.max(np.abs(right - wrong)) > 1e-6, (
            "transposed and untransposed restore agree — this guard is vacuous"
        )

    def test_the_default_builds_the_identity_not_a_metric(self):
        """``precondition=None`` is off (#1397), so no Hessian is even built."""
        problem, log_p = self._problem(None, n=4)
        assert problem.enabled is False
        assert problem.preconditioner is None
        assert problem.logdensity is log_p


class TestNonFiniteExpansionPoint:
    """#1397: a NaN MAP init produced advice that was already in force.

    ``notebooks/01_why_jax.py`` died with::

        ValueError: metric is not positive definite — Cholesky failed. Build it with
        `negative_hessian_metric`, whose eigenvalue floor guarantees this.

    but ``preconditioned_logdensity`` builds the metric with ``negative_hessian_metric``
    on the line above the raise. The remedy was already applied. The real defect was
    upstream — ``MAP init done (loss=nan)`` — and no eigenvalue floor can make a matrix
    positive definite when its entries are NaN.

    The guard was right to refuse and wrong about why. These pin the distinction.
    """

    #: A well-behaved density. The defect in #1397 is the *point*, not the function:
    #: ``MAP init done (loss=nan)`` means the expansion point itself is NaN, and the
    #: Hessian evaluated there is NaN however well-conditioned the density is.
    _NAN_POINT = None  # set in _fail_at, below

    @staticmethod
    def _quartic(xi, data_args):
        """Curvature that DEPENDS on the point — ``-6 diag(xi^2)``.

        A plain quadratic will not do: its Hessian is a constant ``-I`` evaluated
        anywhere, so a NaN point still yields a finite metric and the failure never
        reproduces. Real likelihoods have point-dependent curvature, which is why the
        NaN MAP in #1397 poisoned the metric.
        """
        return -0.5 * jnp.sum(xi**4)

    @classmethod
    def _fail_at_a_nan_point(cls):
        from tengri.inference.preconditioning import prepare_preconditioning

        bad_init = jnp.asarray([0.0, jnp.nan, 0.0, 0.0])
        return prepare_preconditioning(cls._quartic, bad_init, 0.0, precondition=True)

    def test_a_nonfinite_expansion_point_is_named_as_the_cause(self):
        with pytest.raises(ValueError) as excinfo:
            self._fail_at_a_nan_point()
        msg = str(excinfo.value).lower()
        assert "finite" in msg or "nan" in msg, (
            f"message does not name non-finiteness: {excinfo.value}"
        )

    def test_it_does_not_repeat_advice_that_is_already_in_force(self):
        """The old message sent the caller to the function already being used."""
        with pytest.raises(ValueError) as excinfo:
            self._fail_at_a_nan_point()
        assert "negative_hessian_metric" not in str(excinfo.value), (
            "still telling the caller to do the thing this function just did"
        )

    def test_the_message_offers_a_reachable_next_step(self):
        with pytest.raises(ValueError) as excinfo:
            self._fail_at_a_nan_point()
        msg = str(excinfo.value)
        assert "precondition=False" in msg or "init" in msg.lower(), (
            f"no actionable remedy offered: {msg}"
        )

    def test_a_finite_indefinite_metric_still_reports_positive_definiteness(self):
        """The old diagnosis stays correct for the case it was actually about."""
        from tengri.inference.preconditioning import metric_preconditioner

        indefinite = jnp.asarray(np.diag([1.0, -2.0, 3.0]))
        with pytest.raises(ValueError, match="positive definite"):
            metric_preconditioner(indefinite)

    def test_a_nonfinite_metric_is_not_blamed_on_definiteness(self):
        from tengri.inference.preconditioning import metric_preconditioner

        with pytest.raises(ValueError) as excinfo:
            metric_preconditioner(jnp.full((3, 3), jnp.nan))
        msg = str(excinfo.value).lower()
        # Assert the DISTINCTION, not one phrasing of it. "non-finite" and "not
        # finite" mean the same thing and both are correct; what must never happen
        # is a NaN matrix being reported as a definiteness problem, which sends the
        # reader to fix curvature when the defect is upstream.
        assert "finite" in msg or "nan" in msg, f"non-finiteness not named: {excinfo.value}"
        assert "positive definite" not in msg, f"NaN metric diagnosed as non-PD: {excinfo.value}"


def _powered_metric(metric, gamma):
    """``M^gamma`` through the eigenbasis — a metric misspecified by exponent ``gamma``."""
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(metric))
    return (eigenvectors * eigenvalues**gamma) @ eigenvectors.T


def _condition(matrix):
    """Condition number of the symmetric part [dimensionless]."""
    eigenvalues = np.linalg.eigvalsh(0.5 * (np.asarray(matrix) + np.asarray(matrix).T))
    return float(eigenvalues.max() / eigenvalues.min())


class TestTemperMetric:
    """``G -> G^alpha``, capped — the spectral shaping that bounds whitening (#1442).

    Kept separate from :func:`metric_preconditioner` because it is a pure function of
    the spectrum and is where the entire robustness argument lives. The factorization
    that follows it is unchanged.
    """

    def test_full_strength_returns_the_metric_unchanged(self):
        from tengri.inference.preconditioning import temper_metric

        metric = _ill_conditioned_metric(5, 1e4)
        got = np.asarray(temper_metric(metric, strength=1.0, max_condition=np.inf))
        np.testing.assert_allclose(got, metric, rtol=1e-10, atol=1e-10)

    def test_zero_strength_returns_the_identity(self):
        """``alpha = 0`` must be exactly "no preconditioning", not "nearly"."""
        from tengri.inference.preconditioning import temper_metric

        metric = _ill_conditioned_metric(5, 1e6)
        got = np.asarray(temper_metric(metric, strength=0.0, max_condition=np.inf))
        np.testing.assert_allclose(got, np.eye(5), rtol=0, atol=1e-10)

    def test_half_strength_is_the_matrix_square_root(self):
        from tengri.inference.preconditioning import temper_metric

        metric = _ill_conditioned_metric(4, 1e3)
        root = np.asarray(temper_metric(metric, strength=0.5, max_condition=np.inf))
        np.testing.assert_allclose(root @ root, metric, rtol=1e-8, atol=1e-8)

    def test_preserves_the_eigenvectors(self):
        """Tempering rescales the spectrum; it must not rotate the basis."""
        from tengri.inference.preconditioning import temper_metric

        metric = _ill_conditioned_metric(5, 1e4)
        tempered = np.asarray(temper_metric(metric, strength=0.4, max_condition=np.inf))
        _, want = np.linalg.eigh(metric)
        _, got = np.linalg.eigh(tempered)
        # Eigenvectors are defined up to sign; compare the projectors instead.
        np.testing.assert_allclose(np.abs(got.T @ want), np.eye(5), rtol=0, atol=1e-7)

    def test_caps_the_condition_number_before_exponentiating(self):
        from tengri.inference.preconditioning import temper_metric

        metric = _ill_conditioned_metric(5, 1e12)
        tempered = temper_metric(metric, strength=1.0, max_condition=1e6)
        assert _condition(tempered) <= 1e6 * (1 + 1e-9)

    def test_an_uncapped_metric_keeps_its_full_spread(self):
        """Neuter check: the cap test above must not pass because the input was tame."""
        from tengri.inference.preconditioning import temper_metric

        metric = _ill_conditioned_metric(5, 1e12)
        assert _condition(temper_metric(metric, strength=1.0, max_condition=np.inf)) > 1e11

    @pytest.mark.parametrize("bad", [-0.1, 1.5, np.nan])
    def test_refuses_a_strength_outside_the_unit_interval(self, bad):
        from tengri.inference.preconditioning import temper_metric

        with pytest.raises(ValueError, match="strength"):
            temper_metric(_ill_conditioned_metric(3, 10.0), strength=bad)


class TestWhiteningStrengthBoundsTheDamage:
    """#1442: whitening with a wrong metric amplifies rather than degrading gracefully.

    Write the true precision as ``H`` and the metric actually used as ``G = H^gamma``
    (``gamma = 1`` is a perfect metric). For any ``A`` with ``A A^T = G^-alpha`` the
    eigenvalues of the whitened precision ``A^T H A`` are the generalized eigenvalues of
    the pencil ``(H, G^alpha)``, so

    .. math:: \\kappa_{\\rm whitened} = \\kappa(H)^{|1 - \\alpha\\gamma|}

    Preconditioning is therefore worse than doing nothing exactly when
    ``|1 - alpha*gamma| > 1``, i.e. when ``gamma > 2/alpha``. Full whitening
    (``alpha = 1``) tolerates only ``gamma <= 2``.
    """

    @staticmethod
    def _whitened_precision(true_precision, gamma, strength):
        from tengri.inference.preconditioning import metric_preconditioner, temper_metric

        metric = _powered_metric(true_precision, gamma)
        tempered = temper_metric(metric, strength=strength, max_condition=np.inf)
        a = np.asarray(metric_preconditioner(tempered).matrix)
        return a.T @ np.asarray(true_precision) @ a

    def test_full_whitening_is_exactly_as_bad_as_nothing_at_gamma_two(self):
        """The headline failure: alpha=1 buys *nothing* when the metric is squared."""
        h = _ill_conditioned_metric(6, 1e4)
        got = _condition(self._whitened_precision(h, gamma=2.0, strength=1.0))
        assert got == pytest.approx(_condition(h), rel=1e-4), (
            f"expected full whitening to reproduce kappa(H)={_condition(h):.3e}, got {got:.3e}"
        )

    def test_half_strength_repairs_the_case_full_whitening_ruins(self):
        """Same misspecification, alpha=0.5: perfectly conditioned instead of untouched."""
        h = _ill_conditioned_metric(6, 1e4)
        got = _condition(self._whitened_precision(h, gamma=2.0, strength=0.5))
        assert got == pytest.approx(1.0, abs=1e-6), f"kappa after half-strength = {got:.3e}"

    def test_full_whitening_amplifies_ill_conditioning_beyond_gamma_two(self):
        """Past gamma=2 the transform is worse than the problem it was meant to fix."""
        h = _ill_conditioned_metric(6, 1e4)
        full = _condition(self._whitened_precision(h, gamma=3.0, strength=1.0))
        half = _condition(self._whitened_precision(h, gamma=3.0, strength=0.5))
        assert full > 100 * _condition(h), f"expected amplification, got {full:.3e}"
        assert half < _condition(h), f"half strength should still help, got {half:.3e}"

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("strength", [0.25, 0.5, 1.0])
    def test_whitened_condition_follows_the_exponent_law(self, gamma, strength):
        h = _ill_conditioned_metric(6, 1e4)
        got = _condition(self._whitened_precision(h, gamma=gamma, strength=strength))
        want = _condition(h) ** abs(1.0 - strength * gamma)
        assert got == pytest.approx(want, rel=1e-4), (
            f"gamma={gamma} alpha={strength}: got {got:.4e}, law predicts {want:.4e}"
        )

    def test_the_default_strength_tolerates_a_squared_metric(self):
        """Whatever the default is, it must survive the gamma=2 case that broke fits."""
        from tengri.inference.preconditioning import DEFAULT_WHITENING_STRENGTH

        h = _ill_conditioned_metric(6, 1e4)
        got = _condition(
            self._whitened_precision(h, gamma=2.0, strength=DEFAULT_WHITENING_STRENGTH)
        )
        assert got < _condition(h), (
            f"default strength {DEFAULT_WHITENING_STRENGTH} is no better than nothing at gamma=2"
        )


class TestResolveWhiteningStrength:
    """``precondition`` carries both the switch and the strength (#1442)."""

    def test_true_selects_the_default_strength(self):
        from tengri.inference.preconditioning import (
            DEFAULT_WHITENING_STRENGTH,
            _resolve_whitening_strength,
        )

        assert _resolve_whitening_strength(True, 10) == DEFAULT_WHITENING_STRENGTH

    def test_none_and_false_are_off(self):
        from tengri.inference.preconditioning import _resolve_whitening_strength

        assert _resolve_whitening_strength(None, 10) is None
        assert _resolve_whitening_strength(False, 10) is None

    def test_a_float_selects_that_strength(self):
        from tengri.inference.preconditioning import _resolve_whitening_strength

        assert _resolve_whitening_strength(0.25, 10) == pytest.approx(0.25)
        assert _resolve_whitening_strength(1.0, 10) == pytest.approx(1.0)

    def test_zero_strength_is_off_rather_than_an_identity_transform(self):
        """``alpha=0`` is a no-op, so skip the Hessian entirely instead of building one."""
        from tengri.inference.preconditioning import _resolve_whitening_strength

        assert _resolve_whitening_strength(0.0, 10) is None

    @pytest.mark.parametrize("bad", [-0.5, 1.01, 7.0])
    def test_a_strength_outside_the_unit_interval_is_refused(self, bad):
        from tengri.inference.preconditioning import _resolve_whitening_strength

        with pytest.raises(ValueError, match="precondition"):
            _resolve_whitening_strength(bad, 10)

    def test_the_default_is_a_partial_whitening(self):
        """A default of 1.0 would reinstate the unbounded-damage behavior of #1442."""
        from tengri.inference.preconditioning import DEFAULT_WHITENING_STRENGTH

        assert 0.0 < DEFAULT_WHITENING_STRENGTH < 1.0


class TestPrepareHonorsTheStrength:
    def test_a_float_precondition_whitens_partially(self):
        from tengri.inference.preconditioning import prepare_preconditioning

        metric = _ill_conditioned_metric(5, 1e4)
        problem = prepare_preconditioning(
            _gaussian_logdensity(metric), jnp.zeros(5), 0.0, precondition=0.5
        )
        assert problem.enabled is True
        a = np.asarray(problem.preconditioner.matrix)
        # A A^T = G^-alpha  =>  (A A^T)^-1 = G^0.5, whose square is G.
        g_half = np.linalg.inv(a @ a.T)
        np.testing.assert_allclose(g_half @ g_half, metric, rtol=1e-6, atol=1e-6)

    def test_zero_precondition_is_the_untouched_problem(self):
        from tengri.inference.preconditioning import prepare_preconditioning

        log_p = _gaussian_logdensity(_ill_conditioned_metric(5, 1e4))
        problem = prepare_preconditioning(log_p, jnp.zeros(5), 0.0, precondition=0.0)
        assert problem.enabled is False
        assert problem.logdensity is log_p

    def test_the_strength_is_reported_on_the_problem(self):
        """A fit that cannot say how hard it whitened cannot be compared to another."""
        from tengri.inference.preconditioning import prepare_preconditioning

        metric = _ill_conditioned_metric(5, 1e4)
        problem = prepare_preconditioning(
            _gaussian_logdensity(metric), jnp.zeros(5), 0.0, precondition=0.25
        )
        assert problem.strength == pytest.approx(0.25)


class TestAdaptationCacheKey:
    """A step size tuned in one basis is meaningless in another (#1442).

    ``run_nuts`` / ``run_hmc`` / ``run_dynamic_hmc`` each cache the warmup result on a
    per-fitter key. That key recorded *whether* the coordinates were whitened but not
    *how hard*, so two fits differing only in strength shared a step size — silently,
    since a step size is a plain float and any value "works".

    The knowledge of what makes two adaptations incompatible belongs to the transform,
    not to three copies in three backends.
    """

    @staticmethod
    def _problem(precondition):
        from tengri.inference.preconditioning import prepare_preconditioning

        metric = _ill_conditioned_metric(5, 1e4)
        return prepare_preconditioning(
            _gaussian_logdensity(metric), jnp.zeros(5), 0.0, precondition=precondition
        )

    def test_two_strengths_do_not_share_a_key(self):
        assert self._problem(0.25).cache_key != self._problem(0.75).cache_key

    def test_enabled_and_disabled_do_not_share_a_key(self):
        assert self._problem(0.5).cache_key != self._problem(False).cache_key

    def test_the_same_strength_shares_a_key(self):
        """Otherwise the cache never hits and every fit re-runs warmup."""
        assert self._problem(0.5).cache_key == self._problem(0.5).cache_key

    def test_true_and_the_default_strength_agree(self):
        from tengri.inference.preconditioning import DEFAULT_WHITENING_STRENGTH

        assert self._problem(True).cache_key == self._problem(DEFAULT_WHITENING_STRENGTH).cache_key

    def test_the_key_is_hashable(self):
        assert isinstance(hash(self._problem(0.5).cache_key), int)
        assert isinstance(hash(self._problem(False).cache_key), int)


def test_every_hamiltonian_backend_keys_its_adaptation_on_the_strength():
    """Guard: a fourth backend must not reintroduce the copy-pasted boolean key.

    Grep-based on purpose. The failure it prevents is invisible at runtime — a stale
    step size is a finite float that samples happily and badly.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "src/tengri/inference/backends/mcmc"
    offenders = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text()
        if "prepare_preconditioning" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "adapt_key" in line and "problem.enabled" in line:
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "adaptation key ignores whitening strength:\n" + "\n".join(offenders)


def test_each_mcmc_backend_owns_its_adaptation_cache_namespace():
    """Two samplers must not share a warmup cache entry.

    Found while wiring the strength into the key: ``hmc.py`` and ``dynamic_hmc.py``
    both opened their tuple with the literal ``"hmc"`` and carried the same arity, so a
    process running both handed the second whatever the first had tuned. Dynamic HMC
    randomizes trajectory length, so it is a different sampler with a different optimal
    step size — the collision was silent and the draws were merely worse.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3] / "src/tengri/inference/backends/mcmc"
    prefixes = {}
    for path in sorted(root.glob("*.py")):
        for line in path.read_text().splitlines():
            match = re.search(r'adapt_key\s*=\s*\(\s*"([^"]+)"', line)
            if match:
                prefixes.setdefault(match.group(1), []).append(path.name)
    shared = {k: v for k, v in prefixes.items() if len(set(v)) > 1}
    assert not shared, f"backends sharing an adaptation cache namespace: {shared}"


class TestReportedConditioning:
    """A fit that cannot say what the geometry was cannot be diagnosed (#1442).

    The whole 0.10x-5.76x spread was invisible from inside a run: the log said
    "metric whitened at the initial point" whether the metric was excellent or useless.
    """

    @staticmethod
    def _problem(cond, precondition):
        from tengri.inference.preconditioning import prepare_preconditioning

        metric = _ill_conditioned_metric(6, cond)
        return prepare_preconditioning(
            _gaussian_logdensity(metric), jnp.zeros(6), 0.0, precondition=precondition
        )

    def test_reports_the_raw_metric_condition(self):
        problem = self._problem(1e5, True)
        assert problem.metric_condition == pytest.approx(1e5, rel=1e-3)

    def test_reports_the_residual_after_tempering(self):
        """``cond ** (1 - alpha)`` — what the sampler actually faces at the MAP."""
        problem = self._problem(1e4, 0.5)
        assert problem.whitened_condition == pytest.approx(1e2, rel=1e-3)

    def test_full_strength_reports_a_whitened_condition_of_one(self):
        problem = self._problem(1e6, 1.0)
        assert problem.whitened_condition == pytest.approx(1.0, rel=1e-6)

    def test_a_disabled_problem_reports_nothing_rather_than_a_wrong_number(self):
        problem = self._problem(1e5, False)
        assert problem.metric_condition is None
        assert problem.whitened_condition is None


#: Standardized coordinates far outside anything a converged posterior visits.
#:
#: ``xi`` is ``~N(0,1)`` by construction, so these stand in for a whitened
#: coordinate ``zeta`` leaking through in its place. ``-7.4`` is the offset
#: measured on a real tengri fit for ``dust_tau_bc``; the rest bracket it.
_EXTREME_XI = jnp.asarray([-1e3, -50.0, -7.4, 7.4, 50.0, 1e3])


def _instantiable_priors() -> list[tuple[str, object]]:
    """Every concrete prior in ``parameters.priors``, discovered rather than listed.

    Discovered for the same reason ``_capable_backends()`` reads the registry: a
    prior added later must inherit the claim below without anyone remembering to
    extend this file. A prior whose constructor does not match either shape is
    dropped, and the count assertion catches it if that ever hides most of them.
    """
    import inspect

    import tengri.parameters.priors as priors_module
    from tengri.parameters.priors import Distribution

    found = []
    for name, cls in sorted(vars(priors_module).items()):
        if not (inspect.isclass(cls) and issubclass(cls, Distribution)):
            continue
        if cls is Distribution:
            continue
        for args in ((1.0, 4.0), (1.0,)):
            try:
                found.append((name, cls(*args)))
                break
            except Exception:  # wrong constructor arity — try the next shape
                continue
    return found


_PRIORS = _instantiable_priors()


def _why_bounds_cannot_fail(prior) -> str | None:
    """Why a prior-support assertion is unfalsifiable for ``prior``, or None.

    Returns the route, so the taxonomy is visible in the test rather than implied.
    ``None`` means a bounds check *could* fail for this prior — which would be
    news, and is what the assertion is watching for.
    """
    lo, hi = getattr(prior, "lo", None), getattr(prior, "hi", None)

    if lo is None or hi is None:
        return "no lo/hi, so the bounds loop skips it entirely"

    if not (np.isfinite(float(lo)) and np.isfinite(float(hi))):
        return f"bounds are infinite ({lo}, {hi}), so the assertion reads x >= -inf"

    theta = np.asarray(prior.unstandardize(_EXTREME_XI))
    if theta.min() >= float(lo) - 1e-8 and theta.max() <= float(hi) + 1e-8:
        return "unstandardize is a saturating bijection onto [lo, hi]"

    return None


class TestBoundsCannotGuardTheMapping:
    """A bounds check cannot see draws left in the whitened basis (#1498).

    ``tests/contract/test_preconditioning_roundtrip.py`` guards the inverse map
    by asking whether preconditioned draws *explain the data*, which looks
    over-elaborate next to "are they inside the priors?" until you notice that
    bounds are structurally incapable of the job.

    Standardization is unconditional in tengri — every parameter reaches physical
    units through its prior's inverse CDF, whether or not preconditioning is on —
    so this incapacity is universal, not a quirk of one prior or one model. There
    are three routes to it and **every** prior tengri ships takes one of them:

    * **saturating bijection** — ``Uniform``, ``LogUniform``. ``lo + (hi - lo) *
      Phi(xi)`` maps all of the reals into ``[lo, hi]``, so the assertion holds
      for any finite input whatsoever.
    * **infinite bounds** — ``Gaussian`` exposes ``lo=-inf, hi=inf``, so the
      assertion reads ``x >= -inf`` and is true by inspection.
    * **no bounds at all** — ``Laplace``, ``LogNormal``, ``StudentT``, ``Fixed``
      expose no ``lo``/``hi``, so the bounds loop skips them. Note these are the
      heavy-tailed ones, where a leak would be *most* visible: the check declines
      to look precisely where it could have worked.

    Pinned here, in the fast tier, on purpose. The reasoning otherwise lives only
    in a slow-tier docstring that no pull request runs, and "simplify this to a
    bounds check" is exactly the change that would read as correct in review and
    leave the invariant unguarded.
    """

    def test_the_prior_sweep_is_not_empty(self):
        """Anti-vacuity: a discovery that found nothing would assert nothing."""
        assert len(_PRIORS) >= 5, (
            f"only {len(_PRIORS)} priors were instantiable ({[n for n, _ in _PRIORS]}) "
            "— the sweep below is no longer covering the prior surface"
        )

    @pytest.mark.parametrize("name,prior", _PRIORS, ids=[n for n, _ in _PRIORS])
    def test_no_prior_makes_a_bounds_check_falsifiable(self, name, prior):
        """The rule, over every prior tengri ships — not one example of it."""
        assert _why_bounds_cannot_fail(prior) is not None, (
            f"{name} is the first prior for which a prior-support assertion could "
            "actually fail. That would be genuinely new: it would make a bounds "
            "check a viable guard against a whitened-coordinate leak, and the "
            "integration test's deficit statistic could be reconsidered. Until "
            "then the deficit is the only thing that can see one."
        )

    def test_the_sweep_would_notice_a_prior_that_broke_the_pattern(self):
        """Negative control for the classifier itself.

        Every branch of :func:`_why_bounds_cannot_fail` that runs on a real prior
        returns a reason, so the sweep above would be green on *any* input unless
        the classifier can also answer "this one could fail". A stub with finite
        bounds and a non-saturating map is that case, and it must come back
        ``None`` — otherwise the sweep is asserting a tautology about tautologies.
        """

        class _EscapingPrior:
            lo, hi = 0.0, 4.0

            @staticmethod
            def unstandardize(xi):
                return xi  # unbounded: leaves [0, 4] on the first extreme value

        assert _why_bounds_cannot_fail(_EscapingPrior()) is None, (
            "the classifier called an escaping prior unfalsifiable, so the sweep "
            "above cannot distinguish a real prior from a broken one"
        )

    def test_a_leak_collapses_onto_the_bound_instead_of_escaping_it(self):
        """The failure signature, on the prior the measured leak actually used.

        What a basis error does downstream of a saturating map is destroy the
        posterior's *width*, not its range — so if you must detect one on the far
        side of a bijection, test the spread and never the bounds. ``-7.4`` is the
        offset measured on a real tengri fit for ``dust_tau_bc`` when the inverse
        map is dropped, and ``Phi(-7.4) ~ 1e-13``.
        """
        prior = Uniform(0.0, 4.0)
        healthy = jnp.linspace(-2.0, 2.0, 64)

        good = np.asarray(prior.unstandardize(healthy))
        bad = np.asarray(prior.unstandardize(healthy - 7.4))

        assert bad.min() >= prior.lo - 1e-8
        assert bad.max() <= prior.hi + 1e-8, (
            "the leak escaped the prior box, so bounds would have caught it"
        )
        assert np.ptp(bad) < np.ptp(good) / 1e3, (
            f"the leaked posterior kept a spread of {np.ptp(bad):.3g} against the "
            f"healthy {np.ptp(good):.3g} — the collapse-onto-the-bound signature "
            "this documents is not what happens"
        )
