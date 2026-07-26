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

from tengri.inference.preconditioning import (
    metric_preconditioner,
    negative_hessian_metric,
    preconditioned_logdensity,
)

jax.config.update("jax_enable_x64", True)


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

    def test_curvature_at_the_expansion_point_is_whitened(self):
        M = _ill_conditioned_metric(cond=1e6)
        log_p = _gaussian_logdensity(M)
        xi0 = jnp.zeros(6)
        wrapped, _, zeta0 = preconditioned_logdensity(log_p, xi0, 0.0)
        H = np.asarray(negative_hessian_metric(wrapped, zeta0, 0.0, floor=0.0))
        assert np.linalg.cond(np.asarray(M)) > 1e5
        assert abs(np.linalg.cond(H) - 1.0) < 1e-6


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
        from tengri.inference.preconditioning import _resolve_precondition

        assert _resolve_precondition(True, 10_000) is True
        assert _resolve_precondition(False, 2) is False

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
        from tengri.inference.preconditioning import _resolve_precondition

        assert _resolve_precondition(None, n_dim) is False

    def test_explicit_true_is_honored_above_the_cost_threshold(self):
        """The cap advises on O(D^3) cost; it must not veto an explicit request."""
        from tengri.inference.preconditioning import (
            PRECONDITION_MAX_DIM,
            _resolve_precondition,
        )

        assert _resolve_precondition(True, PRECONDITION_MAX_DIM + 1) is True

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
        assert "not finite" in msg or "nan" in msg, (
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
        assert "not finite" in str(excinfo.value).lower(), (
            f"NaN metric diagnosed as non-PD: {excinfo.value}"
        )
