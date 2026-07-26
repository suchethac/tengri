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
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
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
        truth = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}
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


@pytest.mark.xfail(
    reason="#1415: apply_log10_scale's reverse pass underflows at the ~-58 dex flux "
    "projection, so the float32 photometry gradient flushes to exactly 0.0. Fixed by "
    "#1388's scaled-SED contract once it covers the projection seam.",
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
