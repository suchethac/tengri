# SPDX-License-Identifier: BSD-3-Clause
"""The traced preconditioner: the same metric, built inside a vmap.

``prepare_preconditioning`` cannot cross the catalog seam, and the reason is
structural rather than incidental. It reads three concrete values --
``bool(jnp.all(jnp.isfinite(metric)))`` in ``metric_preconditioner``, the
expansion-point gate in ``_reject_nonfinite_expansion_point``, and the
``float()`` casts on the condition numbers -- and it returns a **Python closure**
over one concrete matrix. Under ``jax.vmap`` the first three raise
``TracerBoolConversionError`` and the fourth is a single matrix where the problem
needs one per galaxy: the metric is ``J^T N^-1 J + I`` with ``J`` the Jacobian at
*that* galaxy's MAP and ``N`` *its* noise, so it has a galaxy axis.

``traced_preconditioner`` is the counterpart that does. These tests pin the three
properties that make it a faithful substitute rather than a second
implementation:

1. **It agrees with the non-traced path** on a single lane, to float tolerance.
   Two ways of building one metric that disagree is worse than one way.
2. **It vmaps**, and each lane gets its *own* transform -- the whole point.
3. **A bad lane falls back to the identity alone.** In a catalog of 10 000 a
   Python raise is not expressible inside ``lax.map`` and would not be wanted if
   it were: one pathological galaxy must not abort the other 9 999. The fallback
   is reported through ``ok``, never silent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.preconditioning import (
    DEFAULT_WHITENING_STRENGTH,
    prepare_preconditioning,
    traced_metric_conditioning,
    traced_preconditioner,
)

pytestmark = pytest.mark.contract


def _pd_matrix(n=5, cond=1e5, seed=0):
    """A PD matrix with a prescribed condition number and a non-trivial rotation."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q @ np.diag(np.geomspace(1.0, cond, n)) @ q.T


def _gaussian_logdensity(flat, data_args):
    """``log p(xi) = -1/2 xi^T M xi`` with ``M`` supplied as traced data.

    The metric rides in ``data_args`` rather than being closed over, because that
    is the shape the catalog engine uses: the per-galaxy geometry has to be a
    traced argument or it cannot have a galaxy axis.
    """
    return -0.5 * flat @ (data_args @ flat)


class TestItAgreesWithTheNonTracedPath:
    """One metric, two construction paths, no disagreement."""

    def test_the_transform_matches_prepare_preconditioning(self):
        metric = jnp.asarray(_pd_matrix())
        position = jnp.zeros(metric.shape[0])

        reference = prepare_preconditioning(
            _gaussian_logdensity, position, metric, precondition=True
        )
        precond, ok = traced_preconditioner(
            _gaussian_logdensity, position, metric, strength=DEFAULT_WHITENING_STRENGTH
        )

        assert bool(ok)
        np.testing.assert_allclose(
            np.asarray(precond.matrix),
            np.asarray(reference.preconditioner.matrix),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_the_condition_numbers_match_too(self):
        """The diagnostic pair, not just the transform.

        These are what a catalog run reports, and a number that disagreed with
        the single-fit path would make the two incomparable while looking fine.
        """
        metric = jnp.asarray(_pd_matrix(cond=1e6, seed=3))
        position = jnp.zeros(metric.shape[0])

        reference = prepare_preconditioning(
            _gaussian_logdensity, position, metric, precondition=True
        )
        raw, whitened, ok = traced_metric_conditioning(
            _gaussian_logdensity, position, metric, strength=DEFAULT_WHITENING_STRENGTH
        )

        assert bool(ok)
        assert float(raw) == pytest.approx(reference.metric_condition, rel=1e-8)
        assert float(whitened) == pytest.approx(reference.whitened_condition, rel=1e-8)

    def test_half_whitening_leaves_the_square_root(self):
        """``kappa_whitened = kappa ** |1 - alpha|`` for an exact metric (#1442).

        Worth pinning as arithmetic rather than as a measurement: it is the
        identity that makes ``DEFAULT_WHITENING_STRENGTH = 0.5`` a *choice* about
        misspecification rather than about how much whitening one can afford.
        """
        metric = jnp.asarray(_pd_matrix(cond=1e6, seed=7))
        raw, whitened, _ = traced_metric_conditioning(
            _gaussian_logdensity, jnp.zeros(metric.shape[0]), metric, strength=0.5
        )
        assert float(whitened) == pytest.approx(float(raw) ** 0.5, rel=1e-6)


class TestItVmaps:
    """The property ``prepare_preconditioning`` cannot have."""

    def test_prepare_preconditioning_cannot_be_vmapped(self):
        """The blocker, asserted rather than described.

        If this ever stops raising, the traced pair is redundant and should be
        deleted -- so the test is here to notice that, not only to justify the
        code today.
        """
        metrics = jnp.stack([jnp.asarray(_pd_matrix(seed=s)) for s in range(3)])
        position = jnp.zeros(metrics.shape[-1])

        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.vmap(
                lambda m: (
                    prepare_preconditioning(
                        _gaussian_logdensity, position, m, precondition=True
                    ).init_flat
                )
            )(metrics)

    def test_each_lane_gets_its_own_transform(self):
        """Not merely "it runs under vmap": the lanes must actually differ.

        A metric hoisted out of the vmap would broadcast one matrix to every
        lane, run without error, and whiten every galaxy against the geometry of
        whichever one happened to build it.
        """
        metrics = jnp.stack(
            [jnp.asarray(_pd_matrix(cond=c, seed=s)) for s, c in ((0, 1e3), (1, 1e6), (2, 1e2))]
        )
        position = jnp.zeros(metrics.shape[-1])

        matrices, oks = jax.vmap(
            lambda m: (
                traced_preconditioner(_gaussian_logdensity, position, m, strength=0.5)[0].matrix,
                traced_preconditioner(_gaussian_logdensity, position, m, strength=0.5)[1],
            )
        )(metrics)

        assert bool(jnp.all(oks))
        assert matrices.shape == (3, metrics.shape[-1], metrics.shape[-1])
        for i, j in ((0, 1), (1, 2), (0, 2)):
            assert not np.allclose(np.asarray(matrices[i]), np.asarray(matrices[j]))

    def test_a_vmapped_lane_matches_the_same_lane_alone(self):
        """Batching is a performance detail, not a numerical one."""
        metrics = jnp.stack(
            [jnp.asarray(_pd_matrix(cond=c, seed=s)) for s, c in ((0, 1e3), (1, 1e6))]
        )
        position = jnp.zeros(metrics.shape[-1])

        batched = jax.vmap(
            lambda m: (
                traced_preconditioner(_gaussian_logdensity, position, m, strength=0.5)[0].matrix
            )
        )(metrics)
        alone = traced_preconditioner(_gaussian_logdensity, position, metrics[1], strength=0.5)[
            0
        ].matrix

        np.testing.assert_allclose(np.asarray(batched[1]), np.asarray(alone), rtol=1e-10)

    def test_it_survives_jit(self):
        metric = jnp.asarray(_pd_matrix())
        position = jnp.zeros(metric.shape[0])
        jitted = jax.jit(
            lambda m: (
                traced_preconditioner(_gaussian_logdensity, position, m, strength=0.5)[0].matrix
            )
        )
        eager = traced_preconditioner(_gaussian_logdensity, position, metric, strength=0.5)[
            0
        ].matrix
        np.testing.assert_allclose(np.asarray(jitted(metric)), np.asarray(eager), rtol=1e-10)


class TestTheFallbackIsPerLaneAndReported:
    """One bad galaxy must not take the catalog with it, and must not hide."""

    def test_a_nonfinite_metric_falls_back_to_the_identity(self):
        bad = jnp.full((4, 4), jnp.nan)
        precond, ok = traced_preconditioner(_gaussian_logdensity, jnp.zeros(4), bad, strength=0.5)
        assert not bool(ok)
        np.testing.assert_allclose(np.asarray(precond.matrix), np.eye(4), atol=1e-12)
        np.testing.assert_allclose(np.asarray(precond.inverse), np.eye(4), atol=1e-12)

    def test_a_nonfinite_expansion_point_falls_back_too(self):
        """``prepare_preconditioning`` raises here; the traced path cannot."""
        metric = jnp.asarray(_pd_matrix(n=4))
        position = jnp.asarray([0.0, jnp.nan, 0.0, 0.0])
        precond, ok = traced_preconditioner(_gaussian_logdensity, position, metric, strength=0.5)
        assert not bool(ok)
        np.testing.assert_allclose(np.asarray(precond.matrix), np.eye(4), atol=1e-12)

    def test_the_bad_lane_does_not_poison_its_neighbors(self):
        """The property the whole design exists for.

        Under vmap a NaN that escaped its lane would arrive as a NaN row of a
        batched array, which is finite-shaped, correctly typed, and wrong.
        """
        good = jnp.asarray(_pd_matrix(n=4, cond=1e4))
        metrics = jnp.stack([good, jnp.full((4, 4), jnp.nan), good])
        position = jnp.zeros(4)

        matrices, oks = jax.vmap(
            lambda m: (
                traced_preconditioner(_gaussian_logdensity, position, m, strength=0.5)[0].matrix,
                traced_preconditioner(_gaussian_logdensity, position, m, strength=0.5)[1],
            )
        )(metrics)

        assert [bool(o) for o in oks] == [True, False, True]
        assert bool(jnp.all(jnp.isfinite(matrices)))
        np.testing.assert_allclose(np.asarray(matrices[1]), np.eye(4), atol=1e-12)
        np.testing.assert_allclose(np.asarray(matrices[0]), np.asarray(matrices[2]), rtol=1e-12)

    def test_a_fallback_lane_reports_condition_one_and_says_so(self):
        """Its metric was replaced by the identity, so 1.0 is honest, not a win.

        Read ``ok`` before reading the condition numbers -- that is why the
        diagnostic returns all three together.
        """
        raw, whitened, ok = traced_metric_conditioning(
            _gaussian_logdensity, jnp.zeros(4), jnp.full((4, 4), jnp.nan), strength=0.5
        )
        assert not bool(ok)
        assert float(raw) == pytest.approx(1.0)
        assert float(whitened) == pytest.approx(1.0)


class TestTheTransformIsAFaithfulChangeOfVariables:
    def test_a_zeta_round_trips_through_a_inverse(self):
        metric = jnp.asarray(_pd_matrix(cond=1e7, seed=11))
        position = jnp.asarray(np.random.default_rng(0).standard_normal(metric.shape[0]))
        precond, _ = traced_preconditioner(_gaussian_logdensity, position, metric, strength=0.5)
        np.testing.assert_allclose(
            np.asarray(precond.to_xi(precond.to_latent(position))),
            np.asarray(position),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_whitening_actually_whitens_at_full_strength(self):
        """``alpha = 1`` takes an exact metric to condition 1.0 at the point."""
        metric = jnp.asarray(_pd_matrix(cond=1e8, seed=13))
        _raw, whitened, ok = traced_metric_conditioning(
            _gaussian_logdensity, jnp.zeros(metric.shape[0]), metric, strength=1.0
        )
        assert bool(ok)
        assert float(whitened) == pytest.approx(1.0, rel=1e-4)
