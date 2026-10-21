# SPDX-License-Identifier: BSD-3-Clause
r"""``powerlaw_disc`` must be exact in pure float32 across the declared AGN prior (#1206).

The pre-fix body formed the erg/s-scale linear intermediate
``l_bol_erg = 10**agn_log_lbol * L_sun`` before any normalization. In float32 that
product passes the 3.403e38 ceiling at ``agn_log_lbol`` ~ 5, so every declared
``agn_log_lbol`` in ``Uniform(8, 14)`` (``tengri.components.agn._params``) returned
``inf``, even though the spectrum it encodes (peak ~1e34 erg/s/Hz at 14) is
representable. The fix carries the luminosity scale as a log10 offset applied last,
on an O(1) peak-normalized shape, through ``utils/scale.py``.

Four guards:

* (a) float32 output is finite over the whole declared prior and matches float64 to
  float32 precision on the nonzero part;
* (b) float64 output equals the pre-change formula, re-implemented here in NumPy
  from the documented definition (shape ``nu**alpha exp(-h nu / k T)`` normalized
  over the fixed 10 A - 1e8 A band by the log-log rule of #2919, the production
  integrator);
* (c) the float32 gradient with respect to ``agn_log_lbol`` is finite and non-zero;
* (d) the normalization equals the closed-form band integral, an upper incomplete
  gamma function, to ``ANALYTIC_RTOL``, independently of any quadrature rule.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import exp1, gamma, gammaincc

from tengri.components.agn._phys import C_LIGHT, H_PLANCK, K_BOLTZ
from tengri.components.agn.disc import (
    _PL_BAND_HI_AA,
    _PL_BAND_LO_AA,
    _PL_BAND_NODES,
    powerlaw_disc,
)
from tengri.utils.grid_interp import loglog_integral
from tengri.utils.physics_constants import AA_TO_CM, L_SUN

pytestmark = [
    pytest.mark.regression_bug,
    pytest.mark.filterwarnings("ignore:powerlaw_disc is deprecated:DeprecationWarning"),
]

#: Declared ``agn_log_lbol`` prior, [log10(L_bol / L_sun)] (``_params.py``).
LOG_LBOL_PRIOR = (8.0, 14.0)
#: Rest-frame grid spanning the ionizing UV to the far IR [Angstrom].
WAVE = np.geomspace(100.0, 1.0e6, 400)
#: Dust-free disc temperature scale [K] shared by every float64 reference in this file.
T_MAX = 1e5
#: Node count of the superseded 2049-node trapezoid, kept only for the sensitivity check.
_LEGACY_TRAPEZOID_NODES = 2049
#: Relative tolerance of the closed-form check. Measured 2026-10-21: the 8001-node log-log
#: normalization is 1.7e-7 from the closed form at its worst alpha, the 2049-node trapezoid
#: is 1.9e-5 off, so 1e-6 sits well above the first and well below the second.
ANALYTIC_RTOL = 1e-6


def _band_nu_hz(nodes):
    """The normalization band in frequency [Hz], log-spaced in wavelength as production."""
    band_aa = np.exp(np.linspace(math.log(_PL_BAND_HI_AA), math.log(_PL_BAND_LO_AA), nodes))
    return C_LIGHT / (band_aa * AA_TO_CM)


def _band_shape(nu, alpha, t_max):
    """Plain linear-space shape ``nu**alpha exp(-h nu / k T)`` with the exponent clipped."""
    x = np.clip(H_PLANCK * nu / (K_BOLTZ * max(t_max, 1.0)), 0.0, 500.0)
    return nu**alpha * np.exp(-x)


def _old_powerlaw_disc(wave, log_lbol, lum_ratio, alpha, t_max):
    """The pre-change formula, NumPy float64, as documented in ``powerlaw_disc``.

    The normalization integral follows #2919's converged log-log rule (was a 2049-node
    trapezoid, ~1.05e-5 off): the same ``loglog_integral`` and the same band that production
    now uses, so only the float32-safe log-space formulation differs from this reference.
    """
    wave = np.asarray(wave, dtype=np.float64)
    nu = C_LIGHT / (wave * AA_TO_CM)
    l_bol_erg = 10.0**log_lbol * L_SUN
    shape = _band_shape(nu, alpha, t_max)

    nu_int = _band_nu_hz(_PL_BAND_NODES)
    shape_int = _band_shape(nu_int, alpha, t_max)
    with jax.enable_x64(True):
        integral = float(loglog_integral(jnp.asarray(nu_int), jnp.asarray(shape_int)))
    integral_safe = max(abs(integral), 1e-100)
    return l_bol_erg * lum_ratio * shape / integral_safe


def _upper_gamma(s, x):
    """Upper incomplete gamma ``Gamma(s, x)`` for real ``s``, from scipy's regularized form.

    ``gammaincc`` needs ``s > 0``, so ``s <= 0`` steps up with the recurrence
    ``Gamma(s, x) = (Gamma(s + 1, x) - x**s exp(-x)) / s`` (``Gamma(0, x) = E1(x)``).
    """
    if s == 0.0:
        return float(exp1(x))
    if s > 0.0:
        return float(gammaincc(s, x) * gamma(s))
    return (_upper_gamma(s + 1.0, x) - x**s * math.exp(-x)) / s


def _closed_form_band_integral(alpha, t_max):
    """``|int nu**alpha exp(-h nu / k T) dnu|`` over the band, in closed form [Hz**(alpha+1)].

    With ``x = h nu / k T`` the integral is ``(k T / h)**s [Gamma(s, x_lo) - Gamma(s, x_hi)]``
    with ``s = alpha + 1``. The band end points come from the disc module, not restated here.
    """
    kt_over_h = K_BOLTZ * t_max / H_PLANCK
    nu_lo = C_LIGHT / (_PL_BAND_HI_AA * AA_TO_CM)
    nu_hi = C_LIGHT / (_PL_BAND_LO_AA * AA_TO_CM)
    s = alpha + 1.0
    x_lo = H_PLANCK * nu_lo / (K_BOLTZ * t_max)
    x_hi = H_PLANCK * nu_hi / (K_BOLTZ * t_max)
    return kt_over_h**s * (_upper_gamma(s, x_lo) - _upper_gamma(s, x_hi))


def _float64_disc(log_lbol, alpha=-1.0, t_max=T_MAX, lum_ratio=1.0):
    with jax.enable_x64(True):
        out = powerlaw_disc(
            jnp.asarray(WAVE, dtype=jnp.float64),
            log_lbol,
            agn_lum_ratio=lum_ratio,
            agn_alpha=alpha,
            agn_T_max=t_max,
        )
        return np.asarray(out, dtype=np.float64)


def _float32_disc(log_lbol, alpha=-1.0, t_max=T_MAX, lum_ratio=1.0):
    with jax.enable_x64(False):
        out = powerlaw_disc(
            jnp.asarray(WAVE, dtype=jnp.float32),
            jnp.asarray(log_lbol, dtype=jnp.float32),
            agn_lum_ratio=lum_ratio,
            agn_alpha=alpha,
            agn_T_max=t_max,
        )
        assert out.dtype == jnp.float32
        return np.asarray(out, dtype=np.float64)


@pytest.mark.parametrize("log_lbol", np.linspace(*LOG_LBOL_PRIOR, 7))
def test_float32_finite_and_matches_float64_across_prior(log_lbol):
    """(a) float32 is finite over the declared prior and tracks float64 (rtol 1e-5)."""
    ref = _float64_disc(log_lbol)
    out = _float32_disc(log_lbol)
    assert np.all(np.isfinite(out)), f"float32 powerlaw_disc non-finite at log_lbol={log_lbol}"
    assert np.any(out != 0.0), f"float32 powerlaw_disc identically zero at log_lbol={log_lbol}"
    # The nonzero part: everything above 200 dB below the peak. Deeper points are float32
    # round-off in the exponent x = h nu / k T (error ~ x eps), not a shortfall of the scale.
    nonzero = ref > 1e-20 * np.max(ref)
    np.testing.assert_allclose(out[nonzero], ref[nonzero], rtol=2e-5, atol=0.0)


@pytest.mark.parametrize("log_lbol", [8.0, 10.0, 12.0, 14.0])
@pytest.mark.parametrize("alpha", [-1.5, -1.0, -0.5])
def test_float64_equals_pre_change_formula(log_lbol, alpha):
    """(b) float64 output equals the old expression to round-off (rtol <= 1e-12)."""
    got = _float64_disc(log_lbol, alpha=alpha, lum_ratio=0.7)
    want = _old_powerlaw_disc(WAVE, log_lbol, 0.7, alpha, T_MAX)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("alpha", [-1.5, -1.0, -0.5])
def test_normalization_matches_closed_form(alpha):
    """(d) the normalization equals the closed-form band integral to ``ANALYTIC_RTOL``.

    The normalization is recovered from the output at its peak: ``L_nu = l_bol * lum * shape
    / |integral|``, so ``|integral| = l_bol * lum * shape / L_nu`` at any node. The reference
    is the upper-incomplete-gamma closed form, which no quadrature rule can share a bias with.
    """
    log_lbol, lum_ratio = 12.0, 1.0
    out = _float64_disc(log_lbol, alpha=alpha, lum_ratio=lum_ratio)
    i = int(np.argmax(out))
    nu = C_LIGHT / (WAVE[i] * AA_TO_CM)
    l_bol_erg = 10.0**log_lbol * L_SUN
    integral_from_output = l_bol_erg * lum_ratio * _band_shape(nu, alpha, T_MAX) / out[i]
    want = _closed_form_band_integral(alpha, T_MAX)
    assert integral_from_output == pytest.approx(want, rel=ANALYTIC_RTOL)


@pytest.mark.parametrize("alpha", [-1.5, -1.0, -0.5])
def test_legacy_trapezoid_is_outside_closed_form_tolerance(alpha):
    """Sensitivity of (d): the superseded 2049-node trapezoid misses the closed form.

    If this stops holding, ``ANALYTIC_RTOL`` is no longer a test of the quadrature rule.
    """
    nu = _band_nu_hz(_LEGACY_TRAPEZOID_NODES)
    trapezoid = abs(np.trapezoid(_band_shape(nu, alpha, T_MAX), nu))
    want = _closed_form_band_integral(alpha, T_MAX)
    assert abs(trapezoid / want - 1.0) > ANALYTIC_RTOL


def test_float32_gradient_wrt_log_lbol_is_finite_and_nonzero():
    """(c) float32 value and d(sum L_nu)/d agn_log_lbol are finite; the gradient is non-zero.

    The value is asserted too: a gradient taken at a non-finite forward value is not
    evidence of anything, and the pre-fix body had exactly that shape.
    """
    wave32 = jnp.asarray(WAVE, dtype=jnp.float32)
    with jax.enable_x64(False):

        def total(log_lbol):
            return jnp.sum(powerlaw_disc(wave32, log_lbol))

        value, grad = jax.value_and_grad(total)(jnp.asarray(13.0, dtype=jnp.float32))
        value, grad = float(value), float(grad)
    assert np.isfinite(value), "float32 sum of powerlaw_disc is not finite at log_lbol=13"
    assert np.isfinite(grad), "float32 d(powerlaw_disc)/d agn_log_lbol is not finite"
    assert grad != 0.0, "float32 d(powerlaw_disc)/d agn_log_lbol is exactly zero"
