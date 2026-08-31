# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 photometry gradients, enumerated by scale seam (#1415, #1388).

``tests/regression/precision/test_float32_gradient_accuracy.py`` measures the same
defect on **one** model — stellar + dust. #1436's lesson is that this is not enough:
*a float32 result established on one model configuration says nothing about a
configuration with a different scale seam*, and that is exactly how a 30% gradient
error hid behind a green suite. This module enumerates the seams:

======================  =========================================================
seam                    largest scale it applies
======================  =========================================================
``stellar_dust``        none positive; only the projection's ~-58 dex
``dust_ir``             ``_restore_l_ir_scale``, ~+44.5 dex
``agn``                 the AGN reference offset, ~+34.6 dex
======================  =========================================================

**What is measured, and why it is not finiteness.** The bare
``jax.grad(lambda p: sum(model.predict_photometry(p)))`` returns *exactly zero* in
pure float32 on **all three** seams — measured here, not assumed — and zero is
finite, which is why ``test_inference_grad_float32.py`` could never have caught it.
Every assertion below is against central finite differences at the *same* precision,
or against the float64 gradient.

**The mechanism, and what fixes it.** Reverse mode must store, at the rest-frame
:math:`L_\nu` node, ``d(F_nu)/d(L_nu) = 10**(-58)`` — the cosmological dimming.
Float32's smallest subnormal is 1.4e-45, so it flushes to zero and takes every
upstream cotangent with it. That value is the true derivative, so no rule at the
seam can return anything else (#1388 measured a peak-factored cotangent, a
``custom_jvp`` regrouping and an ``optimization_barrier``; all three return the same
``0.0``). What *does* fix it is changing the cotangent that arrives: multiply the
scalar by a constant before differentiating and divide it out afterwards, which is
:func:`tengri.utils.scale.loss_scaled_grad`. Measured here per seam.

This is also why **fitting** is unaffected: a likelihood multiplies the residual by
``1/sigma**2`` ~ 1e32, which is the same lift arriving for free. The gradient a fit
descends is pinned by ``test_float32_grad_bolometric_seams.py`` (float64 comparison,
all four seams) and ``test_float32_gradient_accuracy.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.utils.scale import DEFAULT_COTANGENT_BOOST, loss_scaled_grad

pytestmark = pytest.mark.regression_bug

#: Step in dex / optical-depth units, as in ``test_float32_gradient_accuracy.py``.
_H = 1e-3

_TRUTH = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}

_BASE = dict(
    sfh={
        "type": "delayed",
        "all_params": FIXED,
        "log_total_mass": Uniform(9.0, 11.0),
        "tau_gyr": 1.0,
        "age_gyr": 5.0,
    },
    redshift=Fixed(0.1),
)
_DUST = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": FIXED,
    "tau_diff": Uniform(0.0, 1.5),
    "tau_bc": 0.0,
}

#: One seam each, on a shared stellar backbone, so a failure names the seam.
_SEAM_MODELS = {
    "stellar_dust": dict(dust_attenuation=_DUST),
    "dust_ir": dict(
        dust_attenuation=_DUST,
        dust_emission={"type": "dale2014", "all_params": FIXED},
    ),
    "agn": dict(
        dust_attenuation=_DUST,
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Fixed(10.5),  # #2069: pinned to break the flat direction
            "fracAGN": 0.1,
        },
    ),
}


@pytest.fixture(scope="module")
def obs():
    # herschel_250 is load-bearing for the dust-IR seam: without a far-IR band the
    # component barely reaches the photometry at all.
    return Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w1", "herschel_250"])
    )


def _photometry_gradients(ssp, obs, seam, *, x64, dtype):
    """Every gradient of one seam at one precision, keyed by how it was taken.

    Keys: ``names``, ``plain`` (bare ``jax.grad``), ``boosted``, ``boosted_jit``
    (the same under ``jax.jit``) and ``fd`` (central differences). All come from one
    model build at one precision, so a disagreement between them is a statement about
    autodiff, never about the evaluation point.
    """
    with jax.enable_x64(x64):
        model = SEDModel.build(ssp_data=ssp, observation=obs, **_BASE, **_SEAM_MODELS[seam])
        names = sorted(n for n in model.spec.free_params if n in _TRUTH)

        def scalar(values):
            params = {k: values[i] for i, k in enumerate(names)}
            return jnp.sum(model.predict_photometry(params))

        def as_array(grads):
            return np.array([float(np.asarray(g)) for g in grads])

        base = [jnp.asarray(_TRUTH[k], dtype=dtype) for k in names]

        fd = []
        for i, name in enumerate(names):
            plus, minus = list(base), list(base)
            plus[i] = jnp.asarray(_TRUTH[name] + _H, dtype=dtype)
            minus[i] = jnp.asarray(_TRUTH[name] - _H, dtype=dtype)
            fd.append(float((np.asarray(scalar(plus)) - np.asarray(scalar(minus))) / (2 * _H)))

        return {
            "names": names,
            "plain": as_array(jax.grad(scalar)(base)),
            "boosted": as_array(loss_scaled_grad(scalar)(base)),
            "boosted_jit": as_array(jax.jit(loss_scaled_grad(scalar))(base)),
            "fd": np.array(fd),
        }


@pytest.fixture(scope="module")
def measured(ssp_bare, obs, request):
    seam = request.param
    f64 = _photometry_gradients(ssp_bare, obs, seam, x64=True, dtype=jnp.float64)
    f32 = _photometry_gradients(ssp_bare, obs, seam, x64=False, dtype=jnp.float32)
    return seam, f64, f32


def _parametrize(fn):
    return pytest.mark.parametrize("measured", sorted(_SEAM_MODELS), indirect=True)(fn)


@_parametrize
def test_float64_autodiff_agrees_with_finite_differences(measured):
    """The instrument, before anything is concluded from it, on every seam.

    If float64 autodiff did not reproduce float64 finite differences the
    finite-difference reference would be unsound and no float32 verdict below could
    be attributed to precision.
    """
    seam, f64, _ = measured
    names, plain64, fd64 = f64["names"], f64["plain"], f64["fd"]
    rel = np.abs(plain64 - fd64) / np.maximum(np.abs(fd64), 1e-300)
    assert rel.max() < 1e-4, (
        f"float64 autodiff disagrees with float64 finite differences by {rel.max():.2e} "
        f"on the {seam} seam (names={names}, auto={plain64}, fd={fd64}) — the reference "
        "is unsound"
    )


@_parametrize
def test_boosted_float32_gradient_matches_same_precision_finite_differences(measured):
    """The fix, checked against a reference computed at the SAME precision.

    Deliberately not a float64 comparison: float32 central differences come back
    correct here (~1e-26 to ~1e-28, all well inside float32's normal range), which is
    what forecloses the excuse that "float32 cannot represent this gradient". It can;
    the reverse pass was losing it, and the boost is what keeps it.
    """
    seam, _, f32 = measured
    names, boosted32, fd32 = f32["names"], f32["boosted"], f32["fd"]
    assert np.all(np.isfinite(fd32)) and np.abs(fd32).max() > 0.0, (
        f"float32 finite differences are unusable on the {seam} seam ({fd32}), so this "
        "test cannot attribute blame"
    )
    rel = np.abs(boosted32 - fd32) / np.maximum(np.abs(fd32), 1e-300)
    assert rel.max() < 1e-2, (
        f"boosted float32 photometry gradient disagrees with float32 finite differences "
        f"by {rel.max():.2e} on the {seam} seam (names={names}, auto={boosted32}, "
        f"fd={fd32}) — loss_scaled_grad no longer lifts the projection cotangent (#1415)"
    )


@_parametrize
def test_boosted_float32_gradient_matches_float64(measured):
    """...and against float64, which is the tighter of the two references.

    Measured at ~1e-06 relative on all three seams, CPU. The 1e-3 bar is loose enough
    not to be brittle and three orders tighter than the defect (a *complete* loss of
    the gradient) it guards. It is also what makes this test sensitive to the *size*
    of :data:`~tengri.utils.scale.DEFAULT_COTANGENT_BOOST` and not only to its
    presence: a boost of ``2**70``, which the arithmetic says should be ample, is
    wrong here by 0.7--18% on CPU and would fail this bar while passing a
    finiteness check.
    """
    seam, f64, f32 = measured
    names, plain64, boosted32 = f64["names"], f64["plain"], f32["boosted"]
    assert np.all(np.isfinite(boosted32)), (
        f"boosted float32 gradient is non-finite on the {seam} seam: {boosted32}"
    )
    rel = np.abs(boosted32 - plain64) / np.maximum(np.abs(plain64), 1e-300)
    assert rel.max() < 1e-3, (
        f"boosted float32 photometry gradient disagrees with float64 by {rel.max():.2e} "
        f"on the {seam} seam (names={names}, f32={boosted32}, f64={plain64})"
    )


@_parametrize
def test_unboosted_float32_gradient_is_still_exactly_zero(measured):
    """The residual boundary, pinned per seam so it cannot move unnoticed.

    This is the open half of #1415, and it is stated as an equality rather than as a
    tolerance because the failure is total: every component comes back ``-0.0`` or
    ``+0.0``, signs preserved, nothing raised. It is seam-independent — the projection
    is on every model's photometry path — and the point of measuring all three is that
    the earlier one-model coverage could not have said so.

    **If this test fails because the gradient became correct**, #1388's scaled-SED
    contract (or an equivalent change to the magnitude the pipeline carries) has
    landed. Delete this test, drop the strict xfail in
    ``test_float32_gradient_accuracy.py::test_photometry_gradient_is_accurate_in_float32``,
    and say so in ``docs/dev/float32-tier-b-boundary.md``.
    """
    seam, _, f32 = measured
    names, plain32, fd32 = f32["names"], f32["plain"], f32["fd"]
    assert np.abs(fd32).max() > 0.0, (
        f"float32 finite differences vanished on the {seam} seam, so this pin means nothing"
    )
    assert np.all(plain32 == 0.0), (
        f"the unboosted float32 photometry gradient is no longer identically zero on "
        f"the {seam} seam (names={names}, grad={plain32}, fd={fd32}) — see this test's "
        "docstring: if it is now CORRECT this pin should be deleted, not loosened"
    )


@_parametrize
def test_float64_gradients_are_bit_identical_with_and_without_the_boost(measured):
    """The boost may not perturb float64 by one ulp — it is a power of two.

    ``DEFAULT_COTANGENT_BOOST`` shifts the binary exponent and leaves every mantissa
    bit alone, so multiplying the objective by it and dividing the gradient back is
    exact. That is what makes :func:`loss_scaled_grad` safe to reach for without
    thinking about which precision is active, and it is the no-behavioral-change bar
    #1206 works to.
    """
    seam, f64, _ = measured
    names, plain64, boosted64 = f64["names"], f64["plain"], f64["boosted"]
    assert float(np.log2(DEFAULT_COTANGENT_BOOST)).is_integer(), (
        f"DEFAULT_COTANGENT_BOOST = {DEFAULT_COTANGENT_BOOST!r} is not a power of two, "
        "so multiplying and dividing by it is no longer exact"
    )
    assert np.array_equal(boosted64, plain64), (
        f"loss_scaled_grad changed the float64 gradient on the {seam} seam "
        f"(names={names}, boosted={boosted64}, plain={plain64})"
    )


@_parametrize
def test_the_boost_survives_jit(measured):
    """``jax.jit`` must not cancel the boost against the divide.

    The wrapper multiplies by a constant and divides the gradient by the same
    constant, which is exactly the shape a compiler is entitled to fold away — and
    this document's own #1535 records the lesson: *a mitigation expressed as an
    association order in source is not binding on the compiler.* There, XLA
    re-associated ``(d-mu)/sigma`` under `jit` with closure-constant data and folded
    ``1/sigma**2`` to ``inf``. Here the same folding would put the cotangent back below
    float32's subnormals and return the identically-zero gradient this exists to
    prevent, so the guard has to run the compiled arm rather than trust the eager one.

    Measured: identical to the eager result, and a fit is a `jit`-ed graph, so this
    is the arm that matters for anyone using it inside one.
    """
    seam, _, f32 = measured
    names, boosted32, jit32 = f32["names"], f32["boosted"], f32["boosted_jit"]
    assert np.all(jit32 != 0.0), (
        f"the boosted float32 gradient collapsed to zero under jit on the {seam} seam "
        f"(names={names}, eager={boosted32}, jit={jit32}) — XLA folded the boost "
        "against the divide (#1535's lesson, #1415's defect)"
    )
    rel = np.abs(jit32 - boosted32) / np.maximum(np.abs(boosted32), 1e-300)
    assert rel.max() < 1e-3, (
        f"jit and eager boosted gradients differ by {rel.max():.2e} on the {seam} seam "
        f"(names={names}, eager={boosted32}, jit={jit32})"
    )
