# SPDX-License-Identifier: BSD-3-Clause
"""The stellar float32 rules must be differentiable in BOTH modes (#1206).

``_mass_scale_lnu`` and ``_flux_weighted_node`` carry custom differentiation
rules whose only job is to pin an operation order that XLA would otherwise
re-associate into an intermediate outside the float32 range.

They were first written as ``jax.custom_vjp``. That works — and makes the
function **opaque to forward-mode autodiff**::

    TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function.

geoVI builds its metric with forward mode, so the float32 hardening silently
broke a float64 inference backend: ``test_geovi_mode_stable_convergence`` went
red. The float32 gradient tests could not catch it — they only ever call
``jax.grad``.

The rules are now ``jax.custom_jvp`` plus ``jax.lax.optimization_barrier``:
forward mode is served directly, reverse mode by transposition, and the barrier
supplies the reassociation immunity that ``custom_vjp`` was really being used
for. Measured: ``custom_jvp`` *without* the barriers reddened nine float32
gradient tests, because a transposed rule is inlined and XLA re-associates it.

These tests pin the property that was missing — that both modes work at all —
rather than the spelling. The float32 ordering itself stays pinned by the
existing gradient suites, and float64 equivalence by the parity checks below.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.component import _flux_weighted_node, _mass_scale_lnu
from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

pytestmark = pytest.mark.regression_bug


def _plain_mass_scale(per_msun_lsun, total_mass):
    """What origin/main computes — it has no custom rule at all."""
    return total_mass * per_msun_lsun * LSUN_ERG_PER_S


def _plain_node(num, den):
    return num / den


# Realistic magnitudes: per-Msun SSP L_nu ~1e-11 Lsun/Hz/Msun, mass ~1e10 Msun.
_PER_MSUN = np.linspace(1e-12, 5e-11, 64)
_TOTAL_MASS = 1.0e10
# The flux-weighted node's denominator is legitimately tiny for a
# near-zero-weight sub-band; that is the case the rule exists for.
_NUM = np.array([5000.0e-20, 5.0, 5000.0])
_DEN = np.array([1e-20, 1e-3, 1.0])


@pytest.mark.parametrize(
    ("custom", "plain", "args", "bit_exact"),
    [
        (_mass_scale_lnu, _plain_mass_scale, (_PER_MSUN, _TOTAL_MASS), True),
        # The node's JVP is regrouped as ``(d_num - q*d_den)/den`` rather than
        # ``d_num/den - num*d_den/den**2``. Algebraically identical, but it
        # rounds differently -- and measurably BETTER: against an exact rational
        # reference at ``den = 1e-20`` the regrouped form is 2.8e-18 relative
        # and the plain one 1.4e-16, because the plain form cancels 1e20 against
        # 5e23. So this pair is round-off close, not bit-equal, and the
        # difference is in the rule's favor.
        (_flux_weighted_node, _plain_node, (_NUM, _DEN), False),
    ],
    ids=["mass_scale_lnu", "flux_weighted_node"],
)
def test_forward_mode_autodiff_works(custom, plain, args, bit_exact):
    """``jax.jvp`` must not raise. A ``custom_vjp`` here raises TypeError."""
    with jax.enable_x64(True):
        primals = tuple(jnp.asarray(a) for a in args)
        tangents = tuple(jnp.ones_like(p) for p in primals)

        _, tangent_custom = jax.jvp(custom, primals, tangents)
        _, tangent_plain = jax.jvp(plain, primals, tangents)

        assert jnp.all(jnp.isfinite(tangent_custom))
        assert jnp.any(tangent_custom != 0.0), (
            "`tangent_custom` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
        if bit_exact:
            assert jnp.array_equal(tangent_custom, tangent_plain), (
                f"forward-mode tangent differs from the plain expression: "
                f"{tangent_custom} vs {tangent_plain}"
            )
        else:
            np.testing.assert_allclose(tangent_custom, tangent_plain, rtol=1e-15)


@pytest.mark.parametrize(
    ("custom", "plain", "args"),
    [
        (_mass_scale_lnu, _plain_mass_scale, (_PER_MSUN, _TOTAL_MASS)),
        (_flux_weighted_node, _plain_node, (_NUM, _DEN)),
    ],
    ids=["mass_scale_lnu", "flux_weighted_node"],
)
def test_float64_is_bit_identical_to_the_plain_expression(custom, plain, args):
    """The rules pin an order; they must not change the float64 answer.

    ``origin/main`` has no custom rule, so the plain expression *is* the
    original float64 path. Value and both gradients must match it exactly —
    an ordering pin that moved float64 would be a behavior change, not a
    range fix.
    """
    with jax.enable_x64(True):
        primals = tuple(jnp.asarray(a) for a in args)

        assert jnp.array_equal(custom(*primals), plain(*primals))

        grad_custom = jax.grad(lambda *z: jnp.sum(custom(*z)), argnums=(0, 1))(*primals)
        grad_plain = jax.grad(lambda *z: jnp.sum(plain(*z)), argnums=(0, 1))(*primals)

        for i, (gc, gp) in enumerate(zip(grad_custom, grad_plain, strict=True)):
            assert jnp.array_equal(jnp.asarray(gc), jnp.asarray(gp)), (
                f"float64 gradient[{i}] moved: {gc} vs {gp}"
            )


def test_reverse_mode_still_works_after_forward_mode():
    """Both modes on the same function, in one trace — geoVI does exactly this."""
    with jax.enable_x64(True):
        per_msun = jnp.asarray(_PER_MSUN)
        mass = jnp.asarray(_TOTAL_MASS)

        def total(p, m):
            return jnp.sum(_mass_scale_lnu(p, m))

        _, tangent = jax.jvp(
            total, (per_msun, mass), (jnp.ones_like(per_msun), jnp.ones_like(mass))
        )
        grads = jax.grad(total, argnums=(0, 1))(per_msun, mass)
        # A Hessian-vector product needs forward-over-reverse, the geoVI shape.
        hvp = jax.jvp(
            lambda p: jax.grad(total, argnums=0)(p, mass),
            (per_msun,),
            (jnp.ones_like(per_msun),),
        )[1]

        assert jnp.isfinite(tangent)
        assert jnp.any(tangent != 0.0), (
            "`tangent` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
        assert all(jnp.all(jnp.isfinite(jnp.asarray(g))) for g in grads)
        assert any(jnp.any(jnp.asarray(g) != 0.0) for g in grads), (
            "the reverse-mode gradients are identically zero — finite is not enough, a "
            "custom rule that has detached is as unusable as a NaN one (#2100)"
        )
        assert jnp.all(jnp.isfinite(hvp))
        assert jnp.any(hvp != 0.0), (
            "`hvp` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
