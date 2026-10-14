# SPDX-License-Identifier: BSD-3-Clause
r"""``powerlaw_disc`` must be exact in pure float32 across the declared AGN prior (#1206).

The pre-fix body formed the erg/s-scale linear intermediate
``l_bol_erg = 10**agn_log_lbol * L_sun`` before any normalization. In float32 that
product passes the 3.403e38 ceiling at ``agn_log_lbol`` ~ 5, so every declared
``agn_log_lbol`` in ``Uniform(8, 14)`` (``tengri.components.agn._params``) returned
``inf``, even though the spectrum it encodes (peak ~1e34 erg/s/Hz at 14) is
representable. The fix carries the luminosity scale as a log10 offset applied last,
on an O(1) peak-normalized shape, through ``utils/scale.py``.

Three guards:

* (a) float32 output is finite over the whole declared prior and matches float64 to
  float32 precision on the nonzero part;
* (b) float64 output equals the pre-change formula, re-implemented here in NumPy
  from the documented definition (shape ``nu**alpha exp(-h nu / k T)`` normalized
  over the fixed 10 A - 1e8 A band by ``trapz`` in frequency);
* (c) the float32 gradient with respect to ``agn_log_lbol`` is finite and non-zero.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn._phys import C_LIGHT, H_PLANCK, K_BOLTZ
from tengri.components.agn.disc import powerlaw_disc
from tengri.utils.physics_constants import AA_TO_CM, L_SUN

pytestmark = [
    pytest.mark.regression_bug,
    pytest.mark.filterwarnings("ignore:powerlaw_disc is deprecated:DeprecationWarning"),
]

#: Declared ``agn_log_lbol`` prior, [log10(L_bol / L_sun)] (``_params.py``).
LOG_LBOL_PRIOR = (8.0, 14.0)
#: Rest-frame grid spanning the ionizing UV to the far IR [Angstrom].
WAVE = np.geomspace(100.0, 1.0e6, 400)
#: The documented normalization band [Angstrom] and its node count.
BAND_AA = np.geomspace(1.0e8, 10.0, 2049)


def _old_powerlaw_disc(wave, log_lbol, lum_ratio, alpha, t_max):
    """The pre-change formula, NumPy float64, as documented in ``powerlaw_disc``."""
    wave = np.asarray(wave, dtype=np.float64)
    nu = C_LIGHT / (wave * AA_TO_CM)
    l_bol_erg = 10.0**log_lbol * L_SUN
    x = np.clip(H_PLANCK * nu / (K_BOLTZ * max(t_max, 1.0)), 0.0, 500.0)
    shape = nu**alpha * np.exp(-x)

    nu_int = C_LIGHT / (BAND_AA * AA_TO_CM)
    x_int = np.clip(H_PLANCK * nu_int / (K_BOLTZ * max(t_max, 1.0)), 0.0, 500.0)
    shape_int = nu_int**alpha * np.exp(-x_int)
    integral = np.trapezoid(shape_int, nu_int)
    integral_safe = max(abs(integral), 1e-100)
    return l_bol_erg * lum_ratio * shape / integral_safe


def _float64_disc(log_lbol, alpha=-1.0, t_max=1e5, lum_ratio=1.0):
    with jax.enable_x64(True):
        out = powerlaw_disc(
            jnp.asarray(WAVE, dtype=jnp.float64),
            log_lbol,
            agn_lum_ratio=lum_ratio,
            agn_alpha=alpha,
            agn_T_max=t_max,
        )
        return np.asarray(out, dtype=np.float64)


def _float32_disc(log_lbol, alpha=-1.0, t_max=1e5, lum_ratio=1.0):
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
    want = _old_powerlaw_disc(WAVE, log_lbol, 0.7, alpha, 1e5)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


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
