# SPDX-License-Identifier: BSD-3-Clause
"""Float32 guard floors at the remaining reachable sites must keep zero inputs finite (#1492).

A literal floor below float32's smallest subnormal (``1e-60``, ``1e-100``,
``1e-300``) is exactly ``0.0`` in float32, so the guard does nothing: a log of the
floored value returns ``-inf``, and a divide by it returns ``inf`` or ``NaN``. Each
test below drives one migrated site in float32 (``jax.enable_x64(False)``) at an
input that underflows or vanishes there, and asserts a finite result. Float64 is
unchanged by construction (``representable_floor`` / ``representable_denominator``
return the literal there); the float64 bit-identity is checked against a capture
taken before the migration, not restated here.

Follow-up to #2921 (round 2 of the float32 floor pass).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def test_pexrav_soft_band_wavelength_gradient_is_exactly_zero_float32():
    """A soft-band photon whose ``wabs`` transmission underflows has a finite, zero gradient.

    At 500 A the transmission is exactly 0 in float32, so the reflected fraction is
    set by the floored optical depth and the gradient with respect to wavelength is
    exactly zero. This is a regression pin on the floored ``-log`` (the reverse
    pass through ``-log(T)`` at ``T == 0`` is otherwise ``-inf * 0``). It is not a
    before/after RED: the soft-band output was already finite before this change.
    The hard-band gradient in float32 is a separate issue (a squared denominator
    underflows in the branching ratio) and is not asserted here.
    """
    import tengri.components.xray.xray as xray

    with jax.enable_x64(False):
        wave = jnp.array([500.0], dtype=jnp.float32)
        l_primary = jnp.float32(1e30)

        def total(w):
            return jnp.sum(xray.pexrav_reflection(w, l_primary))

        grad_wave = jax.grad(total)(wave)
        assert bool(jnp.all(jnp.isfinite(grad_wave)))
        assert bool(jnp.all(grad_wave == 0.0))


def test_xray_corona_from_disc_zero_l2500_is_finite_float32():
    """A zero UV luminosity (no AGN upstream) gives a finite, zero X-ray corona.

    The ``l_2500 <= 0`` fallback computes ``alpha_ox`` from a floored positive value;
    with the literal ``1e-300`` that floor was ``0.0`` in float32 and ``log10(0)``
    propagated ``NaN``. The spectrum is masked to zero at the end, so the answer is 0.
    """
    from tengri.components.xray.xray import xray_agn_corona_from_disc

    with jax.enable_x64(False):
        wave = jnp.array([5.0, 50.0, 1000.0], dtype=jnp.float32)
        spec = xray_agn_corona_from_disc(wave, jnp.float32(0.0))
        assert bool(jnp.all(jnp.isfinite(spec)))
        assert bool(jnp.all(spec == 0.0))


def test_spectral_slope_zero_flux_pixel_is_finite_float32():
    """One zero-flux pixel inside the fit window gives a finite, non-zero UV slope.

    ``log(max(flux, 1e-50))`` is ``-inf`` in float32 for a zero pixel, and the
    window-weighted sum then carries ``-inf`` into the normal equations. The
    remaining pixels follow a clean power law, so the slope is genuinely non-zero.
    """
    from tengri.observation.spectral_indices import SpectralIndexDef, _measure_slope

    with jax.enable_x64(False):
        wave = jnp.array([1000.0, 1200.0, 1500.0, 1800.0, 2100.0, 2500.0], dtype=jnp.float32)
        flux = jnp.array([1.0, 0.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float32)
        idx = SpectralIndexDef(
            name="uv", index_type="slope", continuum=(), feature=(1000.0, 2500.0)
        )
        beta = _measure_slope(wave, flux, idx, 0.0)
        assert bool(jnp.isfinite(beta))
        assert float(beta) != 0.0


def test_powerlaw_disc_underflowed_normalization_is_finite_float32():
    """A shape that underflows at a 1 K temperature gives a finite (zero) disc.

    Regression pin on the normalized power-law disc: the output is finite and zero at
    a 1 K electron temperature. The output was already finite before this change (the
    band integral keeps its low-frequency nodes), so this is not a before/after case.
    """
    from tengri.components.agn.disc import powerlaw_disc

    with jax.enable_x64(False), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wave = jnp.array([1000.0, 5000.0], dtype=jnp.float32)
        # log10 L_bol = 3 keeps L_bol = 10**3 * L_sun inside float32 (erg/s overflows past 3e38).
        l_nu = powerlaw_disc(wave, 3.0, agn_T_max=1.0)
        assert bool(jnp.all(jnp.isfinite(l_nu)))
        assert bool(jnp.all(l_nu == 0.0))


def test_hot_corona_underflowed_normalization_is_finite_float32():
    """A corona whose normalization grid underflows gives a finite zero, not ``NaN``.

    With ``kT`` far below the photon energy, the Comptonization cutoff ``exp(-500)``
    is zero in float32 across the normalization grid, so the integral is 0 and the
    divisor must be the derivative-safe floor, not ``0``.
    """
    from tengri.components.agn.disc import _hot_corona_lnu

    with jax.enable_x64(False):
        nu = jnp.array([1e15, 1e17], dtype=jnp.float32)
        l_nu = _hot_corona_lnu(nu, jnp.float32(1e30), 1.8, jnp.float32(1e-30), 0.0)
        assert bool(jnp.all(jnp.isfinite(l_nu)))
        assert bool(jnp.all(l_nu == 0.0))


def test_fracagn_zero_agn_luminosity_log_is_finite_float32():
    """A zero AGN fraction maps to a finite ``log10(L_bol)`` in float32.

    ``log10(l_agn + 1e-100)`` at ``l_agn = 0`` was ``log10(0) = -inf`` in float32.
    The floored log is ``log10(finfo(float32).tiny)``, which is finite and far below
    any physical luminosity.
    """
    from tengri.components.agn.fracagn import fracagn_to_log_lbol

    with jax.enable_x64(False):
        log_lbol = fracagn_to_log_lbol(jnp.float32(0.0), jnp.float32(1.0))
        assert bool(jnp.isfinite(log_lbol))
        assert float(log_lbol) != 0.0
        assert float(log_lbol) < -30.0


def test_fracagn_zero_total_luminosity_fraction_is_finite_float32():
    """Zero AGN and dust luminosity give a finite zero fraction, not ``0/0``.

    The denominator floor must be derivative-safe (``sqrt(tiny)``), and that is what
    keeps the quotient finite when both numerator and denominator underflow.
    """
    from tengri.components.agn.fracagn import log_lbol_to_fracagn

    with jax.enable_x64(False):
        frac = log_lbol_to_fracagn(jnp.float32(-200.0), jnp.float32(0.0))
        assert bool(jnp.isfinite(frac))
        assert float(frac) == 0.0


def test_balmer_continuum_zero_optical_depth_is_finite_float32():
    """Zero Balmer optical depth gives a finite (zero) continuum, not ``inf * 0``.

    ``tau = 0`` makes the normalizing Balmer flux ``b_norm * (1 - e^0) = 0``. With the
    literal ``1e-60`` divisor floor that was ``0`` in float32, so the scale was ``inf``
    and the product with the zero flux was ``NaN``.
    """
    from tengri.components.agn.qsogen import _balmer_continuum

    with jax.enable_x64(False):
        wave = jnp.linspace(1000.0, 5000.0, 64, dtype=jnp.float32)
        cont = jnp.full_like(wave, 1e-17)
        bc = _balmer_continuum(wave, cont, taube=0.0)
        assert bool(jnp.all(jnp.isfinite(bc)))
        assert bool(jnp.all(bc == 0.0))


def test_posterior_agn_fraction_zero_total_is_finite_float32():
    """A zero total SED gives a zero AGN fraction, with no divide in the reverse path.

    The forward value was already guarded by the ``jnp.where``; this pins that the
    selected denominator stays finite (the guard now selects the divisor before the
    division rather than flooring it at a literal that is zero in float32).
    """
    from tengri.inference.posterior import Posterior

    with jax.enable_x64(False):
        zeros = jnp.zeros(8, dtype=jnp.float32)
        post = Posterior.__new__(Posterior)
        post.__dict__["samples"] = None
        post.__dict__["sed_components"] = lambda wavelength=None: {
            "sed_agn": zeros,
            "sed_total": zeros,
        }
        frac = post.agn_fraction()
        assert bool(jnp.all(jnp.isfinite(frac)))
        assert bool(jnp.all(frac == 0.0))


def test_representable_floor_float64_is_literal():
    """Float64 keeps the literal floors bit-identical, so the migration is dtype-only."""
    from tengri.utils.scale import representable_denominator, representable_floor

    with jax.enable_x64(True):
        for literal in (1e-50, 1e-60, 1e-100):
            assert representable_floor(literal) == literal
            assert representable_denominator(literal) == literal
