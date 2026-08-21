# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 *gradients* must match float64, not merely be finite (#1415).

Every other float32 guard in this directory asserts two things: the forward pass is
finite in pure float32, and it agrees with float64. Both hold while the reverse pass
is broken, because a gradient of exactly ``0.0`` is finite and forward agreement says
nothing about the cotangent chain. This module adds the missing instrument.

**Central finite differences are the arbiter.** They use forward evaluations only, so
they are independent of the reverse-mode path under test, and they can be computed at
the *same* precision as the autodiff they check — which separates "this precision
cannot represent the answer" from "the reverse pass is wrong at this precision".

What is measured (stellar + default dust, no AGN, no dust IR — the defect needs none
of them):

* ``rest_sed()`` — rest-frame :math:`L_\nu` [erg/s/Hz], no flux projection. Float32
  autodiff matches float64 autodiff to ~1e-6 relative. **Pinned here as a regression
  guard**: the float32 reverse pass through the forward model itself is sound.
* ``predict_photometry()`` — :math:`F_\nu` [erg/s/cm^2/Hz], which applies the
  cosmological dimming :math:`(1+z)/(4\pi d_L^2)` at
  ``observation/redshift_kernel.py`` via ``apply_log10_scale``. That is a span of
  ~-58 dex (:math:`L_\nu` ~1e30 -> flux ~1e-28). ``apply_log10_scale``'s Jacobian
  w.r.t. its array argument is exactly ``10**log10_scale``, so the reverse pass
  multiplies the incoming cotangent by ``10**(-58)`` — far below float32's smallest
  subnormal (~1.4e-45) — and it flushes to **exactly zero**. Tracked as a strict
  ``xfail`` until #1388's scaled-SED contract covers the projection seam.

This is #1388's primitive in the *underflow* direction. The overflow direction
(Jacobian -> ``inf`` above ~38.5 dex) at least announces itself as NaN; underflow
returns a finite, silent, wrong number, which is why it needs its own guard.

#1415 turned out to be **two** defects at the same seam, and they are now separated:

* **Wrong value — FIXED.** ``apply_log10_scale`` left its peak differentiable, so ``arr``
  reached the output by two paths whose derivative contributions cancel analytically but
  not in float32. What survived was an uncancelled term the size of the main one, i.e.
  gradients exactly **2x** too large wherever the chain survived at all. ``stop_gradient``
  on the peak leaves the one correct path; float32 now tracks float64 to ~1e-7, and
  float64 moves by at most a few ulp (bit-identical with one scale seam, ``<=1.5e-15``
  relative with several). Pinned by
  :func:`test_likelihood_gradient_is_accurate_in_float32`.
* **Underflow to zero — OPEN, and REVERSE MODE ONLY.** With a small incoming cotangent
  the single correct path still multiplies by ``10**(-58)`` and flushes to zero. Needs
  #1388.

The two are easy to confuse, because both present as "the float32 gradient is wrong".
The discriminator is whether the answer is *zero* (underflow) or a *clean multiple*
(double-counted path).

**The underflow is not a float32 limitation.** Forward mode computes the same
gradient correctly in pure float32 — measured ``[-1.493829e-27, 7.674008e-27]`` against
a float64 ``[-1.493829e-27, 7.673997e-27]``, while reverse mode returns ``[0.0, 0.0]``.
Reverse mode is forced to materialize ``d(F_nu)/d(L_nu) = 10**(-58)``, a number float32
does not have, on the way to an answer it represents fine; forward mode carries
``d(L_nu)/d(param) ~ 1e30`` instead and never forms that ratio. So the answer is
representable, float32 can reach it, and no local change to ``apply_log10_scale`` can
fix reverse mode — the ratio follows from the magnitudes being related, which is what
#1388 changes. Pinned by
:func:`test_photometry_gradient_is_accurate_in_float32_forward_mode`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

#: Step in dex / optical-depth units. Well above float32 resolution (eps ~1.2e-7) so
#: the difference is not itself noise, small enough that curvature is negligible.
_H = 1e-3

#: The point every gradient in this file is evaluated at. Shared so that a comparison
#: between two tests here is a comparison of *modes*, never of evaluation points.
_TRUTH = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}


def _build(ssp, obs):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.0,
        },
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w1"]))


def _autodiff_and_fd(ssp, obs, x64, dtype, observable):
    """(autodiff, finite-difference) gradients of a scalar observable, at one precision.

    Both are computed inside the same precision context, so the finite differences are
    a same-precision reference for the autodiff rather than a float64 yardstick.
    """
    with jax.enable_x64(x64):
        model = _build(ssp, obs)
        truth = _TRUTH
        names = sorted(truth)

        def scalar(values):
            params = {k: values[i] for i, k in enumerate(names)}
            return observable(model, params)

        base = [jnp.asarray(truth[k], dtype=dtype) for k in names]
        grad = jax.grad(scalar)(base)
        auto = np.array([float(np.asarray(g)) for g in grad])

        fd = []
        for i, name in enumerate(names):
            plus, minus = list(base), list(base)
            plus[i] = jnp.asarray(truth[name] + _H, dtype=dtype)
            minus[i] = jnp.asarray(truth[name] - _H, dtype=dtype)
            fd.append(float((np.asarray(scalar(plus)) - np.asarray(scalar(minus))) / (2 * _H)))
        return auto, np.array(fd)


def _rest_sed_sum(model, params):
    return jnp.sum(model.predict(params).rest_sed())


def _photometry_sum(model, params):
    return jnp.sum(model.predict_photometry(params))


def test_float64_autodiff_agrees_with_finite_differences(ssp_bare, obs):
    """Sanity check on the instrument itself before trusting it about float32.

    If float64 autodiff did not match float64 finite differences, a float32
    disagreement would say nothing about precision.
    """
    for observable in (_rest_sed_sum, _photometry_sum):
        auto, fd = _autodiff_and_fd(ssp_bare, obs, True, jnp.float64, observable)
        rel = np.abs(auto - fd) / np.maximum(np.abs(fd), 1e-300)
        assert rel.max() < 1e-4, (
            f"float64 autodiff disagrees with float64 finite differences by "
            f"{rel.max():.2e} for {observable.__name__} — the finite-difference "
            "reference is unsound, so nothing below can be concluded"
        )


def test_rest_frame_gradient_is_accurate_in_float32(ssp_bare, obs):
    """The float32 reverse pass through the forward model is sound — keep it that way.

    Rest-frame L_nu carries no flux projection, so no large log10 offset is applied,
    and float32 autodiff reproduces float64 autodiff to ~1e-6. This is the half of
    #1415 that works; pinning it means a regression here is distinguishable from the
    projection defect below.
    """
    auto64, _ = _autodiff_and_fd(ssp_bare, obs, True, jnp.float64, _rest_sed_sum)
    auto32, _ = _autodiff_and_fd(ssp_bare, obs, False, jnp.float32, _rest_sed_sum)

    assert np.all(np.isfinite(auto32)), "float32 rest-frame gradient is non-finite"
    rel = np.abs(auto32 - auto64) / np.maximum(np.abs(auto64), 1e-300)
    assert rel.max() < 1e-3, (
        f"float32 rest-frame gradient drifted from float64 by {rel.max():.2e}; it "
        "used to agree to ~1e-6 (#1415)"
    )


def test_likelihood_gradient_is_accurate_in_float32(ssp_bare, obs):
    """The gradient a real fit actually uses must agree between float32 and float64.

    This is the path that matters: through the likelihood the incoming cotangent is
    ~1/sigma^2, large enough to survive the projection's ``10**(-58)``, so the reverse
    chain does *not* underflow here — which is exactly why the double-counted peak path
    was so dangerous. It made this gradient finite, plausible, and wrong by a clean
    factor (measured 1.998x on the mass parameter), so a pure-float32 fit would have
    converged confidently to the wrong answer.

    With the peak under ``stop_gradient`` the two precisions agree to ~1e-7. The
    tolerance here is 1e-3, loose enough not to be brittle and still four orders
    tighter than the defect it guards against.
    """
    from tengri import Fitter
    from tengri.inference.context import InferenceContext

    # One mock, built in float64, so both precisions fit identical data.
    with jax.enable_x64(True):
        model = _build(ssp_bare, obs)
        truth = _TRUTH
        mock = model.mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
        flux = np.asarray(mock.flux_obs, dtype=np.float64)
        noise = np.asarray(mock.noise, dtype=np.float64)

    def nlp_gradient(x64, dtype):
        with jax.enable_x64(x64):
            m = _build(ssp_bare, obs)
            ctx = InferenceContext.from_target(Fitter(m, jnp.asarray(flux), jnp.asarray(noise)))
            data_args = ctx.data_args
            names = sorted(ctx.initial_params(jax.random.PRNGKey(1)))
            point = {k: jnp.asarray(0.0, dtype=dtype) for k in names}
            grad = jax.grad(lambda q: ctx.neg_log_posterior_fn(q, data_args))(point)
            return names, np.array([float(np.asarray(grad[k])) for k in names])

    names, g64 = nlp_gradient(True, jnp.float64)
    _, g32 = nlp_gradient(False, jnp.float32)

    assert np.all(np.isfinite(g32)), "float32 likelihood gradient is non-finite"
    rel = np.abs(g32 - g64) / np.maximum(np.abs(g64), 1e-300)
    assert rel.max() < 1e-3, (
        f"float32 likelihood gradient disagrees with float64 by {rel.max():.2e} "
        f"(names={names}, f32={g32}, f64={g64}). A clean ~2x on one component means "
        "apply_log10_scale's peak is differentiable again (#1415)"
    )


@pytest.mark.xfail(
    reason="#1415 residual, and it is REVERSE MODE ONLY — see "
    "test_photometry_gradient_is_accurate_in_float32_forward_mode just below, which "
    "computes this same gradient correctly in pure float32. Reverse mode has to "
    "materialize d(F_nu)/d(L_nu) = 10**(-58) at the flux projection, which is below "
    "float32's smallest subnormal (~1.4e-45), so the cotangent flushes to exactly 0.0 "
    "on the way to an answer (~1e-27) that float32 represents perfectly well. No local "
    "change to apply_log10_scale can help: that ratio is a property of the magnitudes "
    "being related (L_nu ~1e30 -> F_nu ~1e-28), not of how the scale is applied. It "
    "needs #1388's scaled-SED contract — carry the SED already scaled, so no step ever "
    "relates a ~1e30 quantity to a ~1e-28 one.",
    strict=True,
)
def test_photometry_gradient_is_accurate_in_float32(ssp_bare, obs):
    """Float32 photometry gradients must match float32 finite differences.

    Deliberately compared against **same-precision** finite differences: those come
    back correct (~1e-26), which proves float32 can represent this gradient and the
    reverse pass is what fails. A float64 comparison would leave open the excuse that
    the value is out of float32's range.
    """
    auto, fd = _autodiff_and_fd(ssp_bare, obs, False, jnp.float32, _photometry_sum)

    assert np.all(np.isfinite(fd)) and np.abs(fd).max() > 0.0, (
        "float32 finite differences are unusable, so this test cannot attribute blame"
    )
    rel = np.abs(auto - fd) / np.maximum(np.abs(fd), 1e-300)
    assert rel.max() < 1e-2, (
        f"float32 photometry autodiff disagrees with float32 finite differences by "
        f"{rel.max():.2e} (autodiff={auto}, finite differences={fd}) — the reverse "
        "pass underflows at the flux projection (#1415)"
    )


def test_photometry_gradient_is_accurate_in_float32_forward_mode(ssp_bare, obs):
    """Forward mode gets this right in pure float32, where reverse mode returns zero.

    This is what makes the xfail above a statement about **reverse mode** rather than
    about float32. The two modes are not symmetric here:

    * Reverse mode propagates output-to-input, so it must form
      ``d(F_nu)/d(L_nu) = 10**(-58)`` explicitly. That number does not exist in
      float32, and it flushes to zero — taking everything upstream with it.
    * Forward mode propagates input-to-output, carrying
      ``d(L_nu)/d(log_total_mass) ~ 1e30``, which is fine, and then applying the
      projection to that tangent. ``apply_log10_scale`` divides the tangent by the
      same peak it divides the primal by, so the tangent is O(1) when ``pow10(net)``
      hits it. No intermediate is ever out of range.

    Measured (pure float32 vs float64 autodiff): reverse ``[0.0, 0.0]``; forward
    ``[-1.493829e-27, 7.674008e-27]`` against ``[-1.493829e-27, 7.673997e-27]``.

    Two consequences worth keeping straight. The gradient is representable in
    float32, so "float32 cannot do this" is the wrong diagnosis. And ``jacfwd`` is a
    real workaround for differentiating raw flux with respect to a few parameters
    until #1388 lands — it is not a workaround for inference, which needs reverse
    mode over many parameters, and which already works because the likelihood's
    incoming cotangent (~1/sigma^2) is large enough to survive the projection.
    """
    names = sorted(_TRUTH)

    def scalar_at(model, values):
        return _photometry_sum(model, {k: values[i] for i, k in enumerate(names)})

    def forward_mode(x64, dtype):
        with jax.enable_x64(x64):
            model = _build(ssp_bare, obs)
            base = [jnp.asarray(_TRUTH[k], dtype=dtype) for k in names]
            jac = jax.jacfwd(lambda v: scalar_at(model, v))(base)
            return np.array([float(np.asarray(g)) for g in jac])

    f64 = forward_mode(True, jnp.float64)
    f32 = forward_mode(False, jnp.float32)

    assert np.all(f32 != 0.0), (
        f"forward-mode float32 photometry gradient underflowed to zero ({f32}); the "
        "tangent is supposed to stay in range through the projection (#1415)"
    )
    rel = np.abs(f32 - f64) / np.maximum(np.abs(f64), 1e-300)
    assert rel.max() < 1e-3, (
        f"forward-mode float32 gradient disagrees with float64 by {rel.max():.2e} "
        f"(f32={f32}, f64={f64}); it used to agree to ~1e-6 (#1415)"
    )
