# SPDX-License-Identifier: BSD-3-Clause
"""Bolometric-family reductions must not form the erg/s intermediate (issue #1206).

``compute_bolometric_luminosity``/``compute_l_tir``/``compute_l_dust_absorbed`` all
return **L_sun** (~1e9 for a 1e10 Msun galaxy), which float32 represents perfectly.
The overflow was entirely in the intermediate: they integrated to erg/s (~1e43,
above the float32 ceiling of 3.4e38) and only then divided by ``L_SUN``, so the
result came back ``inf``.

Peak-factoring the integrand and folding ``1/L_SUN`` into the same log combine
keeps every intermediate in range while leaving the float64 answer unchanged.

The luminosity-weighted age/metallicity averages are ratios of per-bin
luminosities, so their common erg/s scale cancels exactly; they must be computed
without ever forming it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.utils.physics_constants import C_AA, L_SUN
from tengri.utils.sed_quantities import (
    compute_bolometric_luminosity,
    compute_l_dust_absorbed,
    compute_l_tir,
    compute_luminosity_weighted_age,
)

pytestmark = pytest.mark.regression_bug


def _grid(n=600):
    """Wavelength grid spanning UV to far-IR [Angstrom]."""
    return jnp.asarray(np.logspace(np.log10(500.0), np.log10(5.0e6), n))


def _sed(wave, scale=1.0e28):
    """A blackbody-ish L_nu [erg/s/Hz] of realistic magnitude."""
    w = jnp.asarray(wave)
    return scale * jnp.exp(-(((jnp.log10(w) - 4.0) / 1.2) ** 2))


# --- frozen pre-change references (verbatim arithmetic) --------------------
def _frozen_l_bol(sed, wave):
    """FROZEN pre-#1206 compute_bolometric_luminosity."""
    nu = C_AA / wave
    return (-jnp.trapezoid(sed, nu)) / L_SUN


def _frozen_l_tir(sed, wave):
    """FROZEN pre-#1206 compute_l_tir."""
    nu = C_AA / wave
    mask = (wave >= 8.0e4) & (wave <= 1.0e7)
    sed_ir = jnp.where(mask, sed, 0.0)
    return jnp.maximum(-jnp.trapezoid(sed_ir, nu), 0.0) / L_SUN


def _frozen_l_abs(sed_i, sed_a, wave):
    """FROZEN pre-#1206 compute_l_dust_absorbed."""
    nu = C_AA / wave
    return jnp.maximum(-jnp.trapezoid(sed_i - sed_a, nu), 0.0) / L_SUN


def test_bolometric_family_f64_exact_vs_frozen():
    """Reformulation must reproduce the pre-change float64 answer to rtol 1e-12."""
    wave = _grid()
    for scale in (1.0e24, 1.0e28, 1.0e32):
        sed = _sed(wave, scale)
        sed_att = 0.35 * sed
        assert_allclose(
            np.float64(compute_bolometric_luminosity(sed, wave)),
            np.float64(_frozen_l_bol(sed, wave)),
            rtol=1e-12,
        )
        assert_allclose(
            np.float64(compute_l_tir(sed, wave)),
            np.float64(_frozen_l_tir(sed, wave)),
            rtol=1e-12,
        )
        assert_allclose(
            np.float64(compute_l_dust_absorbed(sed, sed_att, wave)),
            np.float64(_frozen_l_abs(sed, sed_att, wave)),
            rtol=1e-12,
        )


def test_bolometric_family_pure_float32_finite_and_accurate():
    """Pure float32: results finite and matching the float64 reference.

    The frozen linear form returns ``inf`` here (the erg/s intermediate overflows),
    so this test is load-bearing — it fails before the reformulation.
    """
    wave = _grid()
    sed = _sed(wave)
    sed_att = 0.35 * sed
    ref_bol = float(compute_bolometric_luminosity(sed, wave))
    ref_tir = float(compute_l_tir(sed, wave))
    ref_abs = float(compute_l_dust_absorbed(sed, sed_att, wave))

    with jax.enable_x64(False):
        w32 = jnp.asarray(np.asarray(wave), dtype=jnp.float32)
        s32 = jnp.asarray(np.asarray(sed), dtype=jnp.float32)
        a32 = jnp.asarray(np.asarray(sed_att), dtype=jnp.float32)
        assert s32.dtype == jnp.float32  # precondition: genuinely pure float32
        got_bol = float(compute_bolometric_luminosity(s32, w32))
        got_tir = float(compute_l_tir(s32, w32))
        got_abs = float(compute_l_dust_absorbed(s32, a32, w32))
        # the pre-change linear form overflows here — proves the fix is load-bearing
        naive = float(-jnp.trapezoid(s32, C_AA / w32) / L_SUN)

    assert np.isfinite(got_bol), f"l_bol non-finite in float32: {got_bol}"
    assert np.any(got_bol != 0.0), (
        "`got_bol` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert np.isfinite(got_tir), f"l_tir non-finite in float32: {got_tir}"
    assert np.any(got_tir != 0.0), (
        "`got_tir` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert np.isfinite(got_abs), f"l_dust_absorbed non-finite in float32: {got_abs}"
    assert np.any(got_abs != 0.0), (
        "`got_abs` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert not np.isfinite(naive), "expected the naive erg/s form to overflow float32"
    assert_allclose(got_bol, ref_bol, rtol=5e-3)
    assert_allclose(got_tir, ref_tir, rtol=5e-3)
    assert_allclose(got_abs, ref_abs, rtol=5e-3)


def test_bolometric_zero_and_gradient_safety():
    """All-zero SED returns exactly 0.0 and the gradient stays finite."""
    wave = _grid(200)
    zero = jnp.zeros_like(wave)
    assert float(compute_bolometric_luminosity(zero, wave)) == 0.0
    assert float(compute_l_tir(zero, wave)) == 0.0
    assert float(compute_l_dust_absorbed(zero, zero, wave)) == 0.0

    def loss(scale):
        return compute_bolometric_luminosity(_sed(wave, scale), wave)

    for s in (1.0e20, 1.0e28):
        g = float(jax.grad(loss)(s))
        assert np.isfinite(g), f"gradient non-finite at scale {s}: {g}"
        assert np.any(g != 0.0), (
            "`g` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )


def test_luminosity_weighted_age_is_scale_free_in_float32():
    """The weighted-age ratio cancels its erg/s scale — it must stay finite in f32."""
    wave = _grid(300)
    n_age = 12
    ages = jnp.asarray(np.logspace(6.0, 10.0, n_age))
    weights = jnp.asarray(np.linspace(1.0, 0.1, n_age))
    flux = jnp.stack([_sed(wave, 1.0e-6) for _ in range(n_age)])

    ref = float(compute_luminosity_weighted_age(weights, flux, ages, wave))
    assert np.isfinite(ref)
    assert np.any(ref != 0.0), (
        "`ref` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )

    with jax.enable_x64(False):
        got = float(
            compute_luminosity_weighted_age(
                jnp.asarray(np.asarray(weights), dtype=jnp.float32),
                jnp.asarray(np.asarray(flux), dtype=jnp.float32),
                jnp.asarray(np.asarray(ages), dtype=jnp.float32),
                jnp.asarray(np.asarray(wave), dtype=jnp.float32),
            )
        )
    assert np.isfinite(got), f"weighted age non-finite in float32: {got}"
    assert np.any(got != 0.0), (
        "`got` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert_allclose(got, ref, rtol=5e-3)
