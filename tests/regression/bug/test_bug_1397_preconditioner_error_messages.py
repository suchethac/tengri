# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the preconditioner must diagnose the failure it actually has (#1397).

``metric_preconditioner`` refused a bad metric with

    ValueError: metric is not positive definite — Cholesky failed. Build it with
    `negative_hessian_metric`, whose eigenvalue floor guarantees this.

Both halves were wrong when the metric was **non-finite** rather than merely
indefinite:

* the caller had *already* built it with ``negative_hessian_metric`` — that is
  the only thing ``preconditioned_logdensity`` does, on the line above — so the
  advice was impossible to follow; and
* the eigenvalue floor cannot guarantee anything on NaN input, because
  ``jnp.maximum(nan, 1.0)`` is ``nan``. The floor is a no-op exactly when it is
  needed.

The real cause in #1397 was upstream: the MAP init returned NaN, so the metric
was formed at a NaN point. A guard must produce enough information to tell
"indefinite curvature" from "non-finite input", because the two have completely
different fixes.

These tests pin the *distinction*, not merely that something raises.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.preconditioning import (
    metric_preconditioner,
    negative_hessian_metric,
    preconditioned_logdensity,
)

pytestmark = pytest.mark.regression_bug


class TestNonFiniteIsDistinguishedFromIndefinite:
    def test_nan_metric_names_non_finite_not_positive_definiteness(self):
        """LOAD-BEARING. Neuter: restore the single 'not positive definite' message.

        A NaN metric is not an indefiniteness problem and must not be reported
        as one — that sends the reader to fix curvature when the real defect is
        upstream.
        """
        metric = jnp.full((3, 3), jnp.nan)
        with pytest.raises(ValueError) as excinfo:
            metric_preconditioner(metric)
        message = str(excinfo.value)
        assert "non-finite" in message.lower()
        assert "positive definite" not in message, (
            "a non-finite metric was misreported as an indefiniteness problem"
        )

    def test_non_finite_message_points_upstream(self):
        """The fix is at the expansion point, so the message must say so."""
        with pytest.raises(ValueError) as excinfo:
            metric_preconditioner(jnp.full((2, 2), jnp.inf))
        message = str(excinfo.value).lower()
        assert any(hint in message for hint in ("expansion point", "initial point", "map")), (
            f"message does not point at where the NaN came from: {message!r}"
        )

    def test_non_finite_message_names_the_escape_hatch(self):
        """`precondition=False` is the one-word workaround; it must be discoverable."""
        with pytest.raises(ValueError) as excinfo:
            metric_preconditioner(jnp.full((2, 2), jnp.nan))
        assert "precondition=False" in str(excinfo.value)

    def test_a_genuinely_indefinite_metric_still_says_so(self):
        """The original diagnosis remains correct for the case it was written for.

        This matrix is finite and symmetric but has a negative eigenvalue, so
        Cholesky legitimately fails on indefiniteness.
        """
        metric = jnp.asarray([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
        with pytest.raises(ValueError) as excinfo:
            metric_preconditioner(metric)
        message = str(excinfo.value)
        assert "positive definite" in message
        assert "non-finite" not in message.lower()

    def test_a_good_metric_is_untouched(self):
        """The ordinary path must not regress."""
        metric = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
        pre = metric_preconditioner(metric)
        # A A^T = G^-1  =>  A^T G A = I
        identity = pre.matrix.T @ metric @ pre.matrix
        np.testing.assert_allclose(np.asarray(identity), np.eye(2), atol=1e-10)


class TestTheFloorCannotRescueNaN:
    def test_preconditioned_logdensity_rejects_a_non_finite_expansion_point(self):
        """The failure is caught by the orchestrator, which knows the point.

        The check lives in ``preconditioned_logdensity`` rather than in
        ``negative_hessian_metric``: the latter is documented JIT/grad/vmap-safe
        and reading a concrete boolean inside it would break that contract.
        ``preconditioned_logdensity`` is already non-JIT (it calls
        ``metric_preconditioner``) and is the only layer that holds
        ``init_flat``, so it is the one that can say *where* the NaN came from.
        """

        def healthy(position, data_args):
            return -0.5 * jnp.sum(position**2)

        with pytest.raises(ValueError) as excinfo:
            preconditioned_logdensity(healthy, jnp.full(3, jnp.nan), None)
        message = str(excinfo.value).lower()
        assert "non-finite" in message
        assert "precondition=false" in message.replace("False", "false")

    def test_a_non_finite_curvature_from_a_finite_point_is_also_caught(self):
        """Finite point, finite value, NaN *curvature* — still must not be blamed
        on indefiniteness.

        ``sqrt(|p|)`` has an infinite second derivative at the origin, so this is
        a genuinely non-finite Hessian reached from a perfectly finite point.
        (A merely NaN-valued linear function would not do: ``jax.hessian`` of
        anything linear is exactly zero, and so finite.)
        """

        def nan_curvature(position, data_args):
            return jnp.sum(jnp.sqrt(jnp.abs(position)))

        assert bool(jnp.isfinite(nan_curvature(jnp.zeros(3), None))), "premise: value is finite"

        with pytest.raises(ValueError) as excinfo:
            preconditioned_logdensity(nan_curvature, jnp.zeros(3), None)
        message = str(excinfo.value).lower()
        assert "non-finite" in message
        assert "not positive definite" not in message

    def test_jnp_maximum_really_does_propagate_nan(self):
        """Pins the premise, so the test above cannot silently become vacuous."""
        assert bool(jnp.isnan(jnp.maximum(jnp.nan, 1.0)))

    def test_a_healthy_logdensity_still_yields_a_positive_definite_metric(self):
        """The documented contract holds where the input is finite."""

        def quadratic(position, data_args):
            return -0.5 * jnp.sum(position**2) * 3.0

        metric = negative_hessian_metric(quadratic, jnp.zeros(4), None)
        eigenvalues = np.linalg.eigvalsh(np.asarray(metric))
        assert np.all(np.isfinite(metric))
        assert np.all(eigenvalues > 0), eigenvalues
