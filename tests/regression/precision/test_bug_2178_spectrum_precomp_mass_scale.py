# SPDX-License-Identifier: BSD-3-Clause
r"""``total_mass * L_sun`` is reachable in the FORWARD pass, not only the reverse (#2178).

``_mass_scale_lnu`` computes ``total_mass * per_msun_lsun * L_sun``. The two scalars
multiply to ~3.8e43 for a ~1e10 Msun galaxy, which is ``inf`` in float32 (max 3.4e38),
while both operands and the product with the array are comfortably in range. #1206 and
#1415 recognized that hazard and pinned the *reverse* pass against it with a
``custom_jvp`` plus two ``optimization_barrier`` calls in the tangent. The forward was
left as the plain triple product, on the reasoning that Python's left-to-right grouping
makes it ``(total_mass * per_msun_lsun) * L_sun`` and the emitted HLO records that order.

The HLO does record it. The order is still not honored: a CPU backend is free to emit a
kernel for the fused ``multiply -> multiply -> reduce`` that hoists the two scalar
broadcasts into one scalar factor. Ages beyond the galaxy's age carry an exactly-zero SFH
weight, so ``inf * 0`` is ``nan`` and the reduction over age is ``nan`` at every pixel.

Measured 2026-09-06 on the ``SpectrumPrecomp`` seam, CPU, identical model and identical
optimized HLO (byte-for-byte after stripping metadata) on both:

=========================  ==============  ==============
jaxlib                     float64 forward float32 forward
=========================  ==============  ==============
0.11.0                     finite          finite
0.11.1                     finite          **nan**
=========================  ==============  ==============

So the defect is not visible from the graph tengri emits, and it is not visible in
float64 at all. The fix states the grouping in the graph — one more
``optimization_barrier``, on the primal, matching the ones the tangent already carries —
and the tests below pin all three halves of that: the graph says it, the value obeys it,
and float64 does not move.

**Every assertion here is finite AND non-zero, never either alone.** #2100's guard pinned
this family of gradients *finite* and zero is finite; the guard that replaced it pinned
them *non-zero* and ``nan != 0.0`` is ``True``, which is how a ``nan`` satisfied a
non-zero check and hid #2178 for a full CI run. Each predicate alone admits exactly the
value that defeats it.

**Precision is proven on the dtype of the array that came back**, never on
``jax.config.jax_enable_x64``: ``tengri/__init__.py`` re-enables x64 on import, so the
flag lies (#1840).
"""

from __future__ import annotations

import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.component import _mass_scale_lnu
from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

pytestmark = pytest.mark.regression_bug

#: A ~1e10 Msun galaxy: the mass scale on every seam in the precision suite.
_TOTAL_MASS = 1.0e10

#: Per-Msun SSP L_nu on a spectrum pixel grid [Lsun/Hz/Msun], with the exactly-zero rows
#: a real SFH produces. DSPS zeroes the weight of every age bin older than the galaxy, so
#: the tail of this array is exact zeros on every fit — that is what turns an ``inf`` into
#: a ``nan`` rather than leaving it as a loud ``inf``.
_N_AGE, _N_PIX, _N_LIVE = 93, 256, 60


def _per_msun():
    x = np.zeros((_N_AGE, _N_PIX), dtype=np.float32)
    x[:_N_LIVE] = np.linspace(1e-16, 5e-16, _N_LIVE * _N_PIX).reshape(_N_LIVE, _N_PIX)
    return x


def _barrier_count(jaxpr) -> int:
    """``optimization_barrier`` equations in ``jaxpr``, including nested ones."""
    n = 0
    for eqn in jaxpr.eqns:
        if str(eqn.primitive) == "optimization_barrier":
            n += 1
        for value in eqn.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                n += _barrier_count(inner)
    return n


def _finite_and_nonzero(arr, what):
    """The only admissible predicate on this family (#2100 / #2178)."""
    a = np.asarray(arr)
    assert np.all(np.isfinite(a)), (
        f"{what} is non-finite: {a.ravel()[:4]} (dtype {a.dtype}). "
        f"'non-zero' alone would have passed this — nan != 0.0 is True (#2178)."
    )
    assert np.any(a != 0.0), (
        f"{what} is identically zero (dtype {a.dtype}). 'finite' alone would have passed "
        f"this — zero is finite (#2100)."
    )


# --------------------------------------------------------------------------------------
# Preconditions: the hazard is real, and it is real in this dtype
# --------------------------------------------------------------------------------------


def test_setup_the_scalar_product_really_is_out_of_float32_range():
    """Guard the guard: if ``total_mass * L_sun`` fits, nothing below tests anything."""
    m = np.float32(_TOTAL_MASS)
    lsun = np.float32(LSUN_ERG_PER_S)
    with np.errstate(over="ignore"):
        product = m * lsun
    assert np.isfinite(m) and np.isfinite(lsun), "the operands must themselves be in range"
    # Non-zero as well as finite: a zero operand would make the product a finite 0.0,
    # which is the #2100 shape — the overflow precondition below would fail for the
    # wrong reason and every assertion in this module would be vacuous.
    assert np.any(m != 0.0) and np.any(lsun != 0.0), (
        f"an operand is zero (total_mass={m}, L_sun={lsun}), so the product cannot "
        f"overflow and this module tests nothing"
    )
    assert not np.isfinite(product), (
        f"total_mass * L_sun = {product} is representable in float32, so the reassociation "
        f"#2178 is about would be harmless and every assertion below is vacuous. "
        f"float32 max is {np.finfo(np.float32).max:.4e}."
    )
    # And the answer it is an intermediate of *is* representable — that is the whole point.
    reference = np.float64(_TOTAL_MASS) * np.float64(5e-16) * np.float64(LSUN_ERG_PER_S)
    assert np.isfinite(np.float32(reference)), (
        f"the scaled result {reference:.4e} is itself out of float32 range, so this is an "
        f"honest overflow rather than the grouping defect #2178 describes"
    )
    assert np.any(reference != 0.0), (
        f"the scaled result is {reference:.4e} — a zero reference is finite and would make "
        f"the representability claim above vacuous (#2100)"
    )


def test_the_dangerous_grouping_still_demonstrates_the_hazard():
    """The control: written the other way round, the same numbers give ``nan``.

    This is the shape a backend is entitled to rewrite the forward into, spelled out in
    Python so the hazard is demonstrated on *every* backend and jaxlib version rather
    than only on one where the emitter happens to take it. Without this control a green
    suite could mean either "the grouping held" or "this build never regrouped anyway".
    """
    with jax.enable_x64(False):
        x = jnp.asarray(_per_msun())
        m = jnp.asarray(np.float32(_TOTAL_MASS))
        hazard = np.asarray(jax.jit(lambda a, b: jnp.sum((b * LSUN_ERG_PER_S) * a, axis=0))(x, m))
    assert str(hazard.dtype) == "float32", f"control ran at {hazard.dtype}, not float32 (#1840)"
    assert np.all(np.isnan(hazard)), (
        f"the reassociated grouping (total_mass * L_sun) * per_msun no longer produces nan "
        f"(max {np.nanmax(hazard):.4e}) — either float32 grew, L_sun changed, or the "
        f"zero-weight ages are gone. Re-derive #2178 before trusting the tests below."
    )


# --------------------------------------------------------------------------------------
# The fix: the grouping is stated in the graph, not left to whichever kernel is emitted
# --------------------------------------------------------------------------------------


def test_the_forward_grouping_is_stated_in_the_graph():
    """An ``optimization_barrier`` must separate the two multiplies in the PRIMAL.

    Reading the value alone cannot distinguish "the grouping held" from "this build did
    not happen to regroup", which is exactly how #2178 survived: on jaxlib 0.11.0 the
    unfixed forward is finite and on 0.11.1 the identical HLO is ``nan``. The graph is
    the one place the claim is testable on every build.
    """
    with jax.enable_x64(False):
        jaxpr = jax.make_jaxpr(_mass_scale_lnu)(
            jnp.asarray(_per_msun()), jnp.asarray(np.float32(_TOTAL_MASS))
        )
    assert _barrier_count(jaxpr.jaxpr) >= 1, (
        f"_mass_scale_lnu's primal has no optimization_barrier, so the "
        f"(total_mass * per_msun) * L_sun grouping is only Python's evaluation order and "
        f"the emitter may re-form total_mass * L_sun (~3.8e43, inf in float32). #2178.\n"
        f"{jaxpr}"
    )


def test_the_differentiated_forward_carries_the_same_barrier():
    """The custom rule re-spells the primal, and a fix applied to one spelling only fails.

    ``_mass_scale_lnu_jvp`` recomputes ``primal_out`` rather than reusing the function
    body, so there are two independent forward expressions. Barrier the body alone and an
    undifferentiated ``predict_spectrum`` is finite while the identical forward *inside*
    ``jax.grad`` is still ``nan``. The tangent already carried two barriers (#1206); with
    the primal's, a differentiated trace carries three.
    """
    with jax.enable_x64(False):
        x = jnp.asarray(_per_msun())
        m = jnp.asarray(np.float32(_TOTAL_MASS))
        jvp = jax.make_jaxpr(
            lambda a, b: jax.jvp(_mass_scale_lnu, (a, b), (jnp.ones_like(a), jnp.float32(1.0)))
        )(x, m)
        grad = jax.make_jaxpr(
            jax.grad(lambda a, b: jnp.sum(_mass_scale_lnu(a, b)), argnums=(0, 1))
        )(x, m)
    for name, jaxpr in (("jvp", jvp), ("grad", grad)):
        assert _barrier_count(jaxpr.jaxpr) >= 3, (
            f"the {name} trace carries {_barrier_count(jaxpr.jaxpr)} optimization_barrier "
            f"equations, not the 3 that mean the tangent's two (#1206) AND the primal's "
            f"(#2178) are all present. The rule's own primal_out is a second spelling of "
            f"the forward product and needs the barrier just as the function body does."
        )


# --------------------------------------------------------------------------------------
# The value obeys it, and float64 does not move
# --------------------------------------------------------------------------------------


def test_the_mass_scaled_luminosity_is_finite_and_nonzero_in_float32():
    """The behavioral claim, undifferentiated and under ``jax.grad``, at both spellings."""
    with jax.enable_x64(False):
        x = jnp.asarray(_per_msun())
        m = jnp.asarray(np.float32(_TOTAL_MASS))
        value = jax.jit(_mass_scale_lnu)(x, m)
        assert str(np.asarray(value).dtype) == "float32", (
            f"the arm ran at {np.asarray(value).dtype}, not float32; the whole claim is "
            f"about float32 range and this is the only admissible proof of it (#1840)"
        )
        _finite_and_nonzero(value, "jit(_mass_scale_lnu) in float32")

        summed = jax.jit(lambda a, b: jnp.sum(_mass_scale_lnu(a, b), axis=0))(x, m)
        _finite_and_nonzero(summed, "the sum over age of _mass_scale_lnu in float32")

        # A realistic downstream cotangent, not 1.0. ``d/d(per_msun)`` is
        # ``cotangent * total_mass * L_sun``, and at a *unit* cotangent that is the
        # ~3.8e43 quantity this module is about: genuinely out of float32 range, and out
        # of range for the true derivative too, which is #1388's open Tier-B boundary
        # rather than #2178's grouping. A likelihood supplies ~1/sigma**2-sized weights;
        # 1e-30 here stands for one, and puts the Jacobian back in range so the grouping
        # is what is being measured.
        d_x, d_m = jax.jit(
            jax.grad(lambda a, b: jnp.sum(_mass_scale_lnu(a, b)) * 1e-30, argnums=(0, 1))
        )(x, m)
        _finite_and_nonzero(d_x, "d/d(per_msun) in float32 at a representative cotangent")
        _finite_and_nonzero(d_m, "d/d(total_mass) in float32 at a representative cotangent")


def test_float64_does_not_move_under_the_barrier():
    """An ordering pin that moved float64 would be a behavior change, not a range fix.

    ``array_equal``, not a tolerance: the barrier is the identity on values, so anything
    other than bit-equality means it is doing something it is not advertised to do.
    """
    with jax.enable_x64(True):
        x = jnp.asarray(_per_msun(), dtype=jnp.float64)
        m = jnp.asarray(_TOTAL_MASS, dtype=jnp.float64)

        def plain(a, b):
            return b * a * LSUN_ERG_PER_S

        pinned, unpinned = _mass_scale_lnu(x, m), plain(x, m)
        assert str(np.asarray(pinned).dtype) == "float64", "the float64 arm is not float64"
        assert jnp.array_equal(pinned, unpinned), "the float64 forward value moved"

        gp = jax.grad(lambda a, b: jnp.sum(_mass_scale_lnu(a, b)), argnums=(0, 1))(x, m)
        gu = jax.grad(lambda a, b: jnp.sum(plain(a, b)), argnums=(0, 1))(x, m)
        for i, (a, b) in enumerate(zip(gp, gu, strict=True)):
            assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b)), (
                f"the float64 gradient[{i}] moved under the barrier: {a} vs {b}"
            )


# --------------------------------------------------------------------------------------
# End to end: the seam the issue was filed from
# --------------------------------------------------------------------------------------


def test_the_spectrum_precomp_forward_is_finite_and_nonzero_in_float32(ssp_bare):
    """``SpectrumPrecomp`` on the fitting path, which is what ``approx="auto"`` resolves to.

    The unit tests above pin the operator; this pins the shipped path. ``Fitter``'s
    default resolves a spectroscopy channel to ``SpectrumPrecomp``, so a ``nan`` here is
    every default spectroscopic float32 fit, not an opt-in corner (``_check_channel_scales``
    (#1495) then raises at construction — loud, but the channel is unavailable).
    """
    from tengri import DEFAULT, Fixed, Observation, SEDModel, SpectrumPrecomp, Uniform
    from tengri.observation.spectroscopy import Spectroscopy

    wave_obs = np.linspace(4000.0, 9000.0, _N_PIX)
    build = dict(
        observation=Observation(spectroscopy=Spectroscopy(wave_obs=jnp.asarray(wave_obs))),
        approx=SpectrumPrecomp(n_z=48, z_min=0.05, z_max=1.0),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        redshift=Fixed(0.1),
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.0,
        },
    )

    arms = {}
    for x64, tag in ((True, "f64"), (False, "f32")):
        with jax.enable_x64(x64):
            sed = SEDModel.build(ssp_data=ssp_bare, **build)
            assert bool(getattr(sed.approx, "spectrum_precomp", False)), (
                f"the model resolved to {sed.approx}, which is not the LUT this test is "
                f"about; a passing result here would be the exact path wearing its label"
            )
            truth = {
                n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
                for n in sed.spec.free_params
            }
            arms[tag] = np.asarray(sed.predict_spectrum(truth))
        jax.clear_caches()
        gc.collect()

    assert str(arms["f64"].dtype) == "float64" and str(arms["f32"].dtype) == "float32", (
        f"the two arms came back as {arms['f64'].dtype} and {arms['f32'].dtype}; a "
        f"precision claim is void without this and the config flag cannot supply it (#1840)"
    )
    _finite_and_nonzero(arms["f64"], "the SpectrumPrecomp float64 forward")
    _finite_and_nonzero(arms["f32"], "the SpectrumPrecomp float32 forward")

    rel = float(
        np.linalg.norm(arms["f32"].astype(np.float64) - arms["f64"]) / np.linalg.norm(arms["f64"])
    )
    assert rel < 1e-5, (
        f"the float32 SpectrumPrecomp forward tracks float64 to {rel:.2e}, which is far "
        f"wider than the ~1e-7 rounding this path shows when the grouping holds — the "
        f"value is finite but wrong, which #2178's predicate alone would not catch"
    )
