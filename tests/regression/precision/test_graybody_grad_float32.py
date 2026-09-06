# SPDX-License-Identifier: BSD-3-Clause
"""Casey (2012) graybody: float32 gradients must be finite (#1439).

The graybody shape spelled its exponent ``h*c / (lambda*k*T)`` and its
occupation number ``1 / expm1(x)``. Both are correct in real arithmetic and
both were correct in float32 *forward* — the failure was entirely in the
reverse pass, which has to materialize the square of each denominator:

* ``(lambda*k*T)**2`` measured 2.3e-39 at the blue end of a UV-to-far-IR grid,
  below float32's smallest normal 1.18e-38, so the VJP divided by zero;
* ``expm1(x)**2`` passes float32's 3.4e38 at x ~ 44, half the exponent clamp
  that guarded ``expm1``'s own forward overflow.

Both are now regrouped so the squares stay in range. These tests pin the
property that actually broke — a finite, accurate float32 gradient — rather
than the spelling, so a future rewrite is free to change the algebra as long
as it keeps the derivative representable.

References
----------
.. [1] C. M. Casey, "Far-infrared spectral energy distribution fitting for
   galaxies near and far," Monthly Notices of the Royal Astronomical Society,
   Vol. 425, Issue 4, pp. 3094-3103 (2012). Eq. 1.
   arXiv:1206.1595. DOI: 10.1111/j.1365-2966.2012.21455.x
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.emission.analytic._closures import _casey_graybody_nu

pytestmark = pytest.mark.regression_bug

# UV (1e-5 cm = 0.1 um) to far-IR (1e-1 cm = 1000 um) — the range the closure's
# own docstring says it runs on, and the range whose blue end drove the square
# of the old denominator below float32's smallest normal.
_LAM_LO_CM, _LAM_HI_CM = 1e-5, 1e-1
_T_EFF_K, _BETA = 35.0, 1.8


def _grad_and_value(*, x64, optically_thin, T_eff=_T_EFF_K, beta=_BETA):
    """Sum the graybody over the grid and differentiate w.r.t. temperature."""
    with jax.enable_x64(x64):
        dtype = jnp.result_type(float)
        lam_cm = jnp.logspace(np.log10(_LAM_LO_CM), np.log10(_LAM_HI_CM), 400, dtype=dtype)
        beta_a = jnp.asarray(beta, dtype=dtype)

        def total(t):
            return jnp.sum(_casey_graybody_nu(lam_cm, t, beta_a, optically_thin))

        t_a = jnp.asarray(T_eff, dtype=dtype)
        value = float(total(t_a))
        grad = float(jax.grad(total)(t_a))
        assert jnp.result_type(float) == (jnp.float64 if x64 else jnp.float32)
        return value, grad


@pytest.mark.parametrize("optically_thin", [False, True])
def test_graybody_float32_gradient_is_finite(optically_thin):
    """The float32 dT gradient must not be NaN or inf.

    Pre-fix this returned NaN for both opacity branches while the forward value
    was correct to seven digits — a fit would have died with nothing wrong in
    the model output.
    """
    value, grad = _grad_and_value(x64=False, optically_thin=optically_thin)

    assert np.isfinite(value), "forward value must be finite (it always was)"
    assert np.any(value != 0.0), (
        "`value` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert np.isfinite(grad), (
        f"float32 d(sum graybody)/dT is {grad!r}; the reverse pass must not form "
        "a squared denominator outside the float32 range (#1439)"
    )
    assert grad != 0.0, "a silent zero is the other half of this failure mode"


@pytest.mark.parametrize("optically_thin", [False, True])
def test_graybody_float32_gradient_matches_float64(optically_thin):
    """Finite is not enough — it has to be the right number."""
    v32, g32 = _grad_and_value(x64=False, optically_thin=optically_thin)
    v64, g64 = _grad_and_value(x64=True, optically_thin=optically_thin)

    assert abs(v32 - v64) / abs(v64) < 1e-5
    assert abs(g32 - g64) / abs(g64) < 1e-5, f"float32 gradient {g32!r} vs float64 {g64!r}"


def test_graybody_float32_gradient_finite_across_temperatures():
    """The blue end of the grid is where the old denominator underflowed, so
    sweep the temperature that scales it."""
    for T_eff in (10.0, 20.0, 35.0, 80.0, 200.0):
        _, grad = _grad_and_value(x64=False, optically_thin=False, T_eff=T_eff)
        assert np.isfinite(grad), f"NaN/inf gradient at T_eff={T_eff} K"
        assert np.any(grad != 0.0), (
            "`grad` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )


def test_graybody_float64_is_unchanged_by_the_regrouping():
    """Guard the acceptance bar for the rewrite itself.

    Both regroupings are exact identities, so float64 must be untouched. The
    reference values are the pre-fix float64 outputs, recorded to full
    precision; a rewrite that moved float64 would be a behavior change, not a
    range fix.
    """
    value, grad = _grad_and_value(x64=True, optically_thin=False)

    assert value == pytest.approx(9.6993074170e05, rel=1e-11)
    assert grad == pytest.approx(9.9896332668e04, rel=1e-11)
