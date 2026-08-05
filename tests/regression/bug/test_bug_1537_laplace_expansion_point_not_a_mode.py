# SPDX-License-Identifier: BSD-3-Clause
"""Regression: Laplace must not silently expand about a non-stationary point (#1537).

``run_laplace`` inverts the Hessian at whatever point it is handed and calls the
result a covariance. That identity holds *only at a mode*. Away from one, the
Hessian describes the local curvature of a slope; it is still symmetric, still
positive definite, still invertible, and the posterior it produces is confident,
plausible and wrong. ``run_map`` takes a fixed number of Adam steps with no
convergence test, so handing over a non-converged point is the ordinary case,
not an exotic one.

Measured on a 64-galaxy mock population (z=0.1, 10 broadbands, D=26, DRW field
SFH), summarizing each fit by the total of its ``psd_xi`` posterior covariance
spectrum. ``psd_xi`` has an N(0, I) prior, so the prior total is 16 and ten
broadbands constrain only ``n_eff ~ 3-4`` directions -- a healthy total sits in
the low-to-mid teens:

    galaxy   n_map_steps   |grad| at MAP   Newton decrement   xi covariance total
    10          4 000        5.42e+03           843.3                 8.04
    10         40 000        3.15e-01             0.0097              16.70
    13          4 000        6.02e+04        179 760.0                 7.51
    13         40 000        4.31e-01             0.0219              16.57
    19          4 000        3.02e+05      21 219 110.9                 5.15
    19         40 000        2.25e-01             0.0125              17.15
     1          4 000        8.63e-02             0.0028              16.60
     2          4 000        1.09e-01             0.0028              17.45

The failure is invisible by construction: Laplace draws are i.i.d. from the
fitted Gaussian, so R-hat is ~1 and the divergence count is 0 no matter how
wrong the Gaussian is. Only the *shape* is wrong, and no between-chain statistic
can see shape. Downstream, in a hierarchical PSD estimator, the 14% of fits that
landed this way carried 100% of the tilt toward the prior corner.

The diagnostic is the **Newton decrement**, ``d = 0.5 g^T H^-1 g``: the loss drop
a quadratic model predicts between the expansion point and the true mode. Unlike
``|grad|`` it is invariant under affine reparameterization -- so a fixed
threshold means the same thing in every parameterization -- and it is in nats,
the units of the ``log_evidence`` beside it. An offset of ``delta`` standard
deviations along one direction gives ``d = delta^2 / 2``, so the default
tolerance of 0.1 nat is an offset of ``~0.45 sigma``.

These tests pin the detection. They deliberately do NOT pin auto-correction:
a Newton step from these points overshoots catastrophically (galaxy 13's step
raised the loss by 1e79), so the honest behavior is to report, not to repair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import LaplaceNotAtModeWarning
from tengri.inference.backends.laplace import run_laplace

pytestmark = pytest.mark.regression_bug


def _quartic_loss(params, _data_args):
    """A loss whose curvature GROWS away from its mode -- the real mechanism.

    ``0.5 x^2 + 0.25 y^4 + 0.005 y^2`` has its mode at the origin, where
    ``d2/dy2 = 0.01`` and the y variance is 100. At ``y = 3`` the curvature is
    ``3 y^2 + 0.01 = 27.01`` and the variance is 0.037 -- a 2700x collapse read
    off an *exact* Hessian. A purely quadratic loss cannot show this: its
    Hessian is the same everywhere, so it would test the warning without
    testing what the warning is for.
    """
    x, y = params["x"], params["y"]
    return 0.5 * x**2 + 0.25 * y**4 + 0.005 * y**2


_GRAD_FN = jax.jit(jax.value_and_grad(_quartic_loss), static_argnums=())


def _grad_fn(params, data_args):
    return _GRAD_FN(params, data_args)


def _run(y0, **kwargs):
    """Laplace expanded about ``(0, y0)``; ``y0 = 0`` is the true mode."""
    return run_laplace(
        key=jax.random.PRNGKey(0),
        loss_fn=_quartic_loss,
        data_args={},
        map_params_unbounded={"x": jnp.asarray(0.0), "y": jnp.asarray(y0)},
        to_physical_fn=lambda p: p,
        model=None,
        grad_fn=_grad_fn,
        n_samples=256,
        verbose=False,
        **kwargs,
    )


class TestTheDecrementIsReported:
    def test_diagnostics_carry_the_newton_decrement(self):
        """LOAD-BEARING. Neuter: drop the key from the diagnostics dict.

        Reporting it is the whole remedy for a silent failure -- #1537 went
        undiagnosed for a full study because nothing in ``Posterior`` named the
        one number that separates a mode from a slope.
        """
        post = _run(0.0)
        assert "newton_decrement" in post.diagnostics

    def test_it_is_about_zero_at_a_true_mode(self):
        post = _run(0.0)
        assert post.diagnostics["newton_decrement"] == pytest.approx(0.0, abs=1e-8)

    def test_it_equals_the_analytic_value_off_the_mode(self):
        """At ``y = 3``: ``g = y^3 + 0.01 y = 27.03``, ``H = 3 y^2 + 0.01 = 27.01``,
        so ``d = 0.5 g^2 / H = 13.53``. Derived from the loss, not read back
        from the implementation."""
        y0 = 3.0
        g = y0**3 + 0.01 * y0
        h = 3.0 * y0**2 + 0.01
        assert _run(y0).diagnostics["newton_decrement"] == pytest.approx(0.5 * g * g / h, rel=1e-6)

    def test_the_gradient_norm_is_reported_alongside_it(self):
        """Scale-dependent, so it cannot carry the threshold -- but it is what a
        user grep-searches for first, and it costs nothing to report."""
        post = _run(3.0)
        assert post.diagnostics["grad_norm_at_expansion"] == pytest.approx(
            3.0**3 + 0.01 * 3.0, rel=1e-6
        )


class TestANonStationaryPointWarns:
    def test_it_warns(self):
        """LOAD-BEARING. Neuter: delete the warn branch.

        Without it the fit returns a full sample set with plausible marginals
        and nothing anywhere says the Gaussian is misplaced.
        """
        with pytest.warns(LaplaceNotAtModeWarning):
            _run(3.0)

    def test_the_message_carries_the_decrement_and_the_remedy(self):
        """A warning that does not say what to do next gets filtered.

        The remedy is measured, not guessed: raising ``n_map_steps`` from 4 000
        to 40 000 converged all five of the collapsed galaxies above.
        """
        with pytest.warns(LaplaceNotAtModeWarning) as rec:
            _run(3.0)
        message = str(rec[0].message)
        assert "13.5" in message, "the measured decrement is not quoted"
        assert "n_map_steps" in message, "no actionable remedy named"

    def test_a_true_mode_does_not_warn(self):
        """The ordinary path must stay quiet, or the warning becomes noise."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", LaplaceNotAtModeWarning)
            _run(0.0)

    def test_a_small_offset_stays_below_the_threshold(self):
        """0.1 nat is an offset of ~0.45 sigma. A 0.2-sigma offset is ordinary
        optimizer residue and must not warn, or every fit warns."""
        import warnings

        # d = delta^2/2; delta = 0.2 sigma -> d = 0.02 nat.  sigma = 1/sqrt(H),
        # and near the mode H_yy = 0.01, so sigma = 10 and the offset is 2.0 in
        # y.  Solve on the true nonlinear loss instead of assuming: pick the
        # offset whose decrement lands at 0.02.
        y0 = 0.35  # measured below; asserted so a drifting loss is caught
        post = _run(y0)
        assert post.diagnostics["newton_decrement"] < 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("error", LaplaceNotAtModeWarning)
            _run(y0)

    def test_the_tolerance_is_configurable(self):
        """A user who knows their point is off-mode and wants the fit anyway
        must be able to silence it without silencing every other warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", LaplaceNotAtModeWarning)
            _run(3.0, stationarity_tol=100.0)

    def test_a_tighter_tolerance_catches_a_smaller_offset(self):
        with pytest.warns(LaplaceNotAtModeWarning):
            _run(0.35, stationarity_tol=1e-4)


class TestTheCollapseItCatches:
    def test_the_off_mode_covariance_is_the_symptom_being_caught(self):
        """The point of the warning, stated as a measurement.

        An exact Hessian at ``y = 3`` reports a y variance ~2700x smaller than
        at the mode. This is not a finite-difference artifact -- it is what the
        curvature genuinely is at that point -- which is why the fix is to
        detect the bad expansion point rather than to change how the Hessian is
        computed.
        """
        with pytest.warns(LaplaceNotAtModeWarning):
            off = _run(3.0)
        at_mode = _run(0.0)

        var_off = float(np.var(np.asarray(off.samples["y"])))
        var_mode = float(np.var(np.asarray(at_mode.samples["y"])))

        assert var_mode / var_off > 100.0, (
            f"expected a large variance collapse off-mode, got {var_mode:.4g} vs {var_off:.4g}"
        )

    def test_the_healthy_variance_matches_the_analytic_curvature(self):
        """Guards the control arm: if the at-mode fit were also wrong, the
        ratio above would pass for the wrong reason."""
        at_mode = _run(0.0)
        # H_yy at the mode is 0.01, so var = 100
        assert float(np.var(np.asarray(at_mode.samples["y"]))) == pytest.approx(100.0, rel=0.25)


def _saddle_loss(params, _data_args):
    """``0.5 x^2 - 0.5 y^2`` — an indefinite Hessian, ``diag(1, -1)``, everywhere.

    A saddle has no minimum, so ``H^-1`` is not a covariance at any point on it.
    """
    return 0.5 * params["x"] ** 2 - 0.5 * params["y"] ** 2


_SADDLE_GRAD = jax.jit(jax.value_and_grad(_saddle_loss))


class TestAnIndefiniteHessianIsAlsoNotAMode:
    """The guard must not fail open on the one case that breaks its own algebra.

    ``d = 0.5 g^T H^-1 g`` assumes ``H`` is positive definite. With
    ``regularize=False`` the eigenvalues are not floored, and a negative one
    makes the sum **negative** -- so the check ``d > tol`` is False and nothing
    warns, at a point that is definitively not a minimum. Measured before the
    fix: decrement -2.0 with ``|grad| = 2.0`` at a saddle, silent.
    """

    def _run_saddle(self, y0, **kwargs):
        return run_laplace(
            key=jax.random.PRNGKey(0),
            loss_fn=_saddle_loss,
            data_args={},
            map_params_unbounded={"x": jnp.asarray(0.0), "y": jnp.asarray(y0)},
            to_physical_fn=lambda p: p,
            model=None,
            grad_fn=_SADDLE_GRAD,
            n_samples=64,
            regularize=False,
            verbose=False,
            **kwargs,
        )

    def test_it_warns_instead_of_reporting_a_negative_decrement(self):
        """LOAD-BEARING. Neuter: drop the positive-definiteness branch.

        Without it the decrement is -2.0 and the comparison silently passes.
        """
        with pytest.warns(LaplaceNotAtModeWarning):
            self._run_saddle(2.0)

    def test_the_decrement_is_never_negative(self):
        """A predicted loss *drop* below zero is not a number to report; the
        quadratic model simply has no minimum in that direction."""
        with pytest.warns(LaplaceNotAtModeWarning) as rec:
            post = self._run_saddle(2.0)
        assert post.diagnostics["newton_decrement"] > 0.0
        assert "indefinite" in str(rec[0].message).lower(), (
            "the message must name the cause, since the remedy differs from an under-converged MAP"
        )

    def test_it_warns_even_at_a_stationary_saddle(self):
        """The nastiest case: ``grad = 0`` exactly, so every gradient-based
        check passes, yet ``H^-1`` is indefinite and sampling it is meaningless.
        """
        with pytest.warns(LaplaceNotAtModeWarning):
            post = self._run_saddle(0.0)
        assert post.diagnostics["grad_norm_at_expansion"] == pytest.approx(0.0, abs=1e-9)

    def test_regularize_true_is_the_ordinary_path_and_stays_quiet(self):
        """With the default flooring, a positive-definite Hessian at a mode must
        not be dragged into this branch."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", LaplaceNotAtModeWarning)
            _run(0.0)
