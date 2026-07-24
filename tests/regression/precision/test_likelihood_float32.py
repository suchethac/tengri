# SPDX-License-Identifier: BSD-3-Clause
r"""The photometry likelihood path must survive pure float32 (#1206).

Making the forward SED float32-safe is necessary but not sufficient for a
float32 *fit*: the projection to observed flux and the Gaussian data term each
carried a float32 underflow that a forward-only finiteness check never sees,
because the failure is a silent **zero** (flux) or a ``0/0`` **NaN** (chi2),
not an ``inf``.

Three seams, each with realistic magnitudes (AB fluxes ~1e-28 erg/s/cm^2/Hz,
uncertainties ~1e-30):

1. **Flux projection** — ``lnu_to_fnu`` applied to the actual L_nu (~1e30), not
   extracted as a standalone ``flux_scale = lnu_to_fnu(1.0, ...)`` ~1e-58 which
   underflows float32 to zero. Verified: the projected flux is *nonzero*.
2. **Effective noise** — ``hypot(sigma, floor)`` instead of
   ``sqrt(sigma**2 + floor**2)``; ``sigma**2`` ~1e-60 underflows to zero and
   collapses sigma_eff.
3. **Gaussian chi2** — the standardized residual ``r = (d - mu)/sigma`` is
   formed before squaring; ``(d-mu)**2/sigma**2`` is ``0/0`` in float32.

Each is bit-for-bit (to float32 precision) identical to its float64 value.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2
from tengri.observation.noise import compute_effective_noise
from tengri.utils.conversions import lnu_to_fnu

pytestmark = pytest.mark.regression_bug

#: An AB flux and its uncertainty at the magnitudes a real photometric fit sees.
_FLUX = 1.9e-28  # erg/s/cm^2/Hz
_SIGMA = 3.0e-30  # erg/s/cm^2/Hz


def _both_precisions(fn):
    """Return ``(f64_result, f32_result)`` of ``fn()`` at each precision."""
    with jax.enable_x64(True):
        r64 = np.asarray(fn(), dtype=np.float64)
    with jax.enable_x64(False):
        r32 = np.asarray(fn())
    return r64, r32


def test_lnu_to_fnu_flux_is_nonzero_in_float32():
    r"""The projected flux must not silently underflow to zero.

    ``lnu_to_fnu`` folds the ~1e-58 dimming into the L_nu peak via
    ``apply_log10_scale``; extracting it against a peak of 1 (the old
    ``flux_scale = lnu_to_fnu(1.0, ...)``) gives ~1e-58, which is zero in
    float32.
    """
    lnu = 1.0e30  # erg/s/Hz, a representative filter-integrated L_nu
    dl_cm = 1.5e27  # ~z=0.1 luminosity distance

    def run():
        return lnu_to_fnu(jnp.asarray(lnu), jnp.asarray(dl_cm), jnp.asarray(0.1))

    f64, f32 = _both_precisions(run)
    assert f32.dtype == jnp.float32, "precondition: genuinely float32"
    assert np.isfinite(f32) and f32 > 0.0, (
        f"float32 flux underflowed to {float(f32):.3e} — the dimming was extracted "
        "as a standalone ~1e-58 scale instead of folded into the L_nu peak"
    )
    # 5e-5, not tighter: apply_log10_scale carries a log10->pow10 round-trip
    # whose float32 error scales with the ~28-decade net offset (~1e-5 measured).
    assert abs(float(f32) / float(f64) - 1.0) < 5e-5, "float32 flux departs from float64"


def test_effective_noise_is_nonzero_in_float32():
    """``sigma_eff`` must stay finite and positive for ~1e-30 uncertainties."""
    noise = jnp.full((5,), _SIGMA)
    model = jnp.full((5,), _FLUX)

    def run():
        return compute_effective_noise(noise, model, f_cal=0.05)

    f64, f32 = _both_precisions(run)
    assert f32.dtype == jnp.float32
    assert np.all(np.isfinite(f32)) and np.all(f32 > 0.0), (
        "sigma_eff collapsed to zero in float32 — sqrt(sigma**2+cal**2) underflowed; "
        "hypot(sigma, cal) does not"
    )
    np.testing.assert_allclose(f32.astype(np.float64), f64, rtol=1e-5)


def test_gaussian_chi2_is_finite_in_float32():
    """The Gaussian data term must be finite (not ``0/0``) at real magnitudes."""
    rng = np.random.default_rng(0)
    observed = jnp.asarray(_FLUX * (1.0 + 0.05 * rng.standard_normal(8)))
    predicted = jnp.asarray(_FLUX * (1.0 + 0.05 * rng.standard_normal(8)))
    sigma = jnp.full((8,), _SIGMA)

    def run():
        return diag_gaussian_chi2(predicted, observed, sigma)

    f64, f32 = _both_precisions(run)
    assert f32.dtype == jnp.float32
    assert np.isfinite(f32), (
        "chi2 is NaN in float32 — (d-mu)**2 (~1e-56) and sigma**2 (~1e-60) both "
        "underflow to zero and 0/0 = NaN; standardizing before squaring keeps r O(1)"
    )
    # chi2 ~ O(n_bands); float32 tracks float64 to its ~7-digit precision.
    assert abs(float(f32) / float(f64) - 1.0) < 1e-4, "float32 chi2 departs from float64"


def test_gaussian_chi2_gradient_is_finite_in_float32():
    """The data term must be differentiable in float32, not just finite."""
    observed = jnp.full((6,), _FLUX)
    sigma = jnp.full((6,), _SIGMA)

    def run():
        g = jax.grad(lambda mu: diag_gaussian_chi2(mu, observed, sigma))(jnp.full((6,), _FLUX))
        return g

    _f64, f32 = _both_precisions(run)
    assert np.all(np.isfinite(f32)), "chi2 gradient is non-finite in float32"
