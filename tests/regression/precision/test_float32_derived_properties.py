# SPDX-License-Identifier: BSD-3-Clause
"""Derived properties whose answer is representable but whose path was not (#1837).

Twenty properties returned ``inf``/``nan`` in pure float32 while their float64
values sat between 1e-1 and 1e10 — comfortably inside float32 range. Every one
reached a representable answer through an unrepresentable erg/s intermediate,
and in each case the range-safe route already existed in the same forward pass
(``log_L_ir``, ``log_L_age``, ``log_line_lums``, or ``_trapz_to_lsun``).

Four families, each pinned below:

A. energy balance   -- ``l_dust_absorbed`` read raw ``L_absorbed`` [erg/s]
B. UV diagnostics   -- ``irx``/``irx_fuv`` formed ``L_TIR * L_SUN`` and
                       ``nu * L_nu`` in erg/s; ``m_uv``/``rest_uv_color`` broke
                       on constants *internal* to the magnitude helpers, with
                       representable values on both sides
C. line ratios      -- six diagnostics read raw ``line_lums`` [erg/s]
D. weighted means   -- ``luminosity_weighted_*`` read raw ``L_age`` [erg/s]

The float64 guards compare against the pre-fix formula written inline, so they
fail if a range-safe rewrite moves a float64 number. A speed or range fix that
moves float64 is a physics change, not a refactor.

See #1837; parent epic #1206.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.magnitudes import (
    ab_mag_to_fnu,
    absolute_ab_mag_to_lnu,
    fnu_to_ab_mag,
    lnu_to_absolute_ab_mag,
)
from tengri.utils.physics_constants import MAGGIES_ZP_CGS, TEN_PC_CM

pytestmark = pytest.mark.regression_bug

F32_MAX = float(np.finfo(np.float32).max)

#: L_nu values spanning the UV/optical continuum of a ~1e10 Msun galaxy
#: [erg/s/Hz]. Every one is representable in float32; so is every magnitude
#: they map to. Only the constants in between were not.
LNU_SAMPLES = [2.352914e27, 8.007146e27, 2.889111e28, 1.0e25, 5.0e30]


# ── The constants that overflow, named explicitly ─────────────────


def test_the_overflowing_constants_are_actually_out_of_range():
    """Non-vacuity: pin the constants this module exists to route around.

    Without this, a change to ``TEN_PC_CM`` or the AB zero-point could make the
    float32 tests below pass for a reason unrelated to the fix.
    """
    assert TEN_PC_CM**2 > F32_MAX, "TEN_PC_CM**2 must overflow float32 at the square"
    assert 4.0 * np.pi * TEN_PC_CM**2 > F32_MAX
    # An L_nu of 8e27 erg/s/Hz over the AB zero-point is ~2.2e47.
    assert 8.007146e27 / MAGGIES_ZP_CGS > F32_MAX


def test_guard_floor_1e300_is_inert_in_float32():
    """The old ``jnp.maximum(fnu, 1e-300)`` guard cannot bind in float32 (#1492)."""
    assert np.float32(1e-300) == 0.0
    from tengri.utils.scale import representable_floor

    with jax.enable_x64(False):
        assert representable_floor(1e-300) > 0.0


# ── B2: magnitude helpers ─────────────────────────────────────────


@pytest.mark.parametrize("lnu", LNU_SAMPLES)
def test_lnu_to_absolute_ab_mag_finite_in_float32(lnu):
    """Input ~1e27 and output ~-17 are both representable; so must be the path."""
    with jax.enable_x64(False):
        out = lnu_to_absolute_ab_mag(jnp.asarray(lnu, dtype=jnp.float32))
        assert np.asarray(out).dtype == np.float32
        assert np.isfinite(out), f"M_UV non-finite in float32 for L_nu={lnu:.3e}"


#: Absolute tolerance for a float64 magnitude compared against the pre-fix
#: linear formula. Moving the zero-point from a division into a log-domain
#: subtraction costs one ulp of the log argument: float64 log10 carries ~1e-16
#: relative error on an argument of order 1e47, i.e. ~5e-15 absolute in log10,
#: which the -2.5 prefactor scales to ~1.2e-14. 1e-13 leaves an order of
#: magnitude of headroom. Judged absolutely, not relatively, because a
#: magnitude of exactly 0.0 is a legitimate value at the AB zero-point and no
#: relative tolerance can be met against it.
MAG_F64_ATOL = 1e-13


@pytest.mark.parametrize("lnu", LNU_SAMPLES)
def test_lnu_to_absolute_ab_mag_float64_unchanged(lnu):
    """float64 must match the pre-fix formula to within one ulp of the log."""
    got = lnu_to_absolute_ab_mag(jnp.asarray(lnu))
    old = -2.5 * jnp.log10(
        jnp.maximum(lnu / (4.0 * jnp.pi * TEN_PC_CM**2), 1e-300) / MAGGIES_ZP_CGS
    )
    np.testing.assert_allclose(np.float64(got), np.float64(old), rtol=1e-13, atol=MAG_F64_ATOL)


@pytest.mark.parametrize("fnu", [3.631e-20, 1e-28, 8.007146e27])
def test_fnu_to_ab_mag_finite_in_float32(fnu):
    """Also exercised far above the flux regime: rest_uv_color feeds it an L_nu."""
    with jax.enable_x64(False):
        out = fnu_to_ab_mag(jnp.asarray(fnu, dtype=jnp.float32))
        assert np.isfinite(out), f"AB mag non-finite in float32 for f_nu={fnu:.3e}"


@pytest.mark.parametrize("fnu", [3.631e-20, 1e-28, 8.007146e27])
def test_fnu_to_ab_mag_float64_unchanged(fnu):
    got = fnu_to_ab_mag(jnp.asarray(fnu))
    old = -2.5 * jnp.log10(jnp.maximum(fnu, 1e-300) / MAGGIES_ZP_CGS)
    np.testing.assert_allclose(np.float64(got), np.float64(old), rtol=1e-13, atol=MAG_F64_ATOL)


@pytest.mark.parametrize("mag", [0.0, 25.0, -118.35])
def test_ab_mag_to_fnu_finite_in_float32(mag):
    """The inverse direction carries the same zero-point constant."""
    with jax.enable_x64(False):
        out = ab_mag_to_fnu(jnp.asarray(mag, dtype=jnp.float32))
        assert np.isfinite(out), f"f_nu non-finite in float32 for m_AB={mag}"


@pytest.mark.parametrize("mag", [0.0, 25.0, -118.35])
def test_ab_mag_to_fnu_float64_unchanged(mag):
    got = ab_mag_to_fnu(jnp.asarray(mag))
    old = MAGGIES_ZP_CGS * 10.0 ** (-0.4 * mag)
    np.testing.assert_allclose(np.float64(got), np.float64(old), rtol=1e-13)


@pytest.mark.parametrize("mag", [-16.8, 0.0, 5.0])
def test_absolute_ab_mag_to_lnu_finite_in_float32(mag):
    with jax.enable_x64(False):
        out = absolute_ab_mag_to_lnu(jnp.asarray(mag, dtype=jnp.float32))
        assert np.isfinite(out), f"L_nu non-finite in float32 for M_AB={mag}"


@pytest.mark.parametrize("mag", [-16.8, 0.0, 5.0])
def test_absolute_ab_mag_to_lnu_float64_unchanged(mag):
    got = absolute_ab_mag_to_lnu(jnp.asarray(mag))
    old = (MAGGIES_ZP_CGS * 10.0 ** (-0.4 * mag)) * (4.0 * jnp.pi * TEN_PC_CM**2)
    np.testing.assert_allclose(np.float64(got), np.float64(old), rtol=1e-13)


@pytest.mark.parametrize("lnu", LNU_SAMPLES)
def test_absolute_ab_mag_round_trip_float32(lnu):
    """L_nu -> M_AB -> L_nu holds in float32 once neither leg overflows."""
    with jax.enable_x64(False):
        x = jnp.asarray(lnu, dtype=jnp.float32)
        back = absolute_ab_mag_to_lnu(lnu_to_absolute_ab_mag(x))
        np.testing.assert_allclose(np.float64(back), lnu, rtol=1e-4)


# ── B1: UV luminosity and IRX ─────────────────────────────────────


def _uv_test_sed():
    """A smooth SED with UV and IR flux, on a grid reaching 1000 um."""
    wave = np.geomspace(9.0e2, 5.0e6, 900)
    # nu L_nu ~ 1e43 erg/s in the UV; a warm IR bump for L_TIR.
    sed = 1.0e28 * (wave / 1600.0) ** -0.5
    sed = sed + 3.0e29 * np.exp(-0.5 * ((np.log10(wave) - np.log10(1.0e6)) / 0.25) ** 2)
    return jnp.asarray(sed), jnp.asarray(wave)


def test_log_uv_luminosity_1600_matches_linear_in_float64():
    """The log form must agree with the erg/s form wherever the latter is finite."""
    from tengri.utils.sed_quantities import (
        compute_log_uv_luminosity_1600,
        compute_uv_luminosity_1600,
    )

    sed, wave = _uv_test_sed()
    linear = np.float64(compute_uv_luminosity_1600(sed, wave))
    log_form = np.float64(compute_log_uv_luminosity_1600(sed, wave))
    np.testing.assert_allclose(log_form, np.log10(linear), rtol=1e-13)


def test_log_uv_luminosity_1600_finite_in_float32():
    """nu*L_nu ~ 1e43 erg/s is not representable; its log10 is."""
    sed, wave = _uv_test_sed()
    with jax.enable_x64(False):
        s32 = jnp.asarray(np.asarray(sed), dtype=jnp.float32)
        w32 = jnp.asarray(np.asarray(wave), dtype=jnp.float32)
        from tengri.utils.sed_quantities import compute_log_uv_luminosity_1600

        out = compute_log_uv_luminosity_1600(s32, w32)
        assert np.isfinite(out), "log10(nu L_nu) must be finite in float32"
        assert 40.0 < float(out) < 46.0


def test_irx_finite_in_float32_and_matches_float64():
    """IRX is a dex ratio of order 1 -- neither side may be formed in erg/s."""
    from tengri.utils.sed_quantities import (
        compute_irx,
        compute_l_tir,
        compute_log_uv_luminosity_1600,
    )

    sed, wave = _uv_test_sed()
    ref = np.float64(
        compute_irx(
            compute_l_tir(sed, wave), log_l_uv_erg=compute_log_uv_luminosity_1600(sed, wave)
        )
    )
    assert np.isfinite(ref)

    with jax.enable_x64(False):
        s32 = jnp.asarray(np.asarray(sed), dtype=jnp.float32)
        w32 = jnp.asarray(np.asarray(wave), dtype=jnp.float32)
        got = compute_irx(
            compute_l_tir(s32, w32), log_l_uv_erg=compute_log_uv_luminosity_1600(s32, w32)
        )
        assert np.isfinite(got), "IRX non-finite in float32"
        np.testing.assert_allclose(np.float64(got), ref, rtol=1e-3)


def test_irx_float64_unchanged_for_the_linear_signature():
    """The erg/s signature keeps its float64 answer (callers outside tengri)."""
    from tengri.utils.physics_constants import L_SUN
    from tengri.utils.sed_quantities import (
        _FLOOR,
        compute_irx,
        compute_l_tir,
        compute_uv_luminosity_1600,
    )

    sed, wave = _uv_test_sed()
    l_tir = compute_l_tir(sed, wave)
    l_uv = compute_uv_luminosity_1600(sed, wave)
    got = np.float64(compute_irx(l_tir, l_uv))
    old = np.float64(jnp.log10(jnp.maximum(l_tir * L_SUN, _FLOOR()) / jnp.maximum(l_uv, _FLOOR())))
    np.testing.assert_allclose(got, old, rtol=1e-13, atol=MAG_F64_ATOL)


def test_compute_irx_refuses_ambiguous_uv_arguments():
    """Exactly one UV argument -- a silent preference would hide the f32 route."""
    from tengri.utils.sed_quantities import compute_irx

    with pytest.raises(TypeError, match="exactly one"):
        compute_irx(jnp.asarray(1e8))
    with pytest.raises(TypeError, match="exactly one"):
        compute_irx(jnp.asarray(1e8), jnp.asarray(1e42), log_l_uv_erg=jnp.asarray(42.0))


# ── A / C / D: the consumer seams, end to end ─────────────────────

#: Every property #1837 measured as non-finite in float32 while float64 was
#: representable. The Cue chain covers all thirteen distinct names; the
#: stellar+dust chain produces the seven non-line ones.
BROKEN_PROPERTIES = (
    "irx",
    "irx_fuv",
    "l_dust_absorbed",
    "luminosity_weighted_age_gyr",
    "luminosity_weighted_metallicity",
    "m_uv",
    "rest_uv_color",
    "balmer_decrement",
    "bpt_nii",
    "bpt_sii",
    "o32",
    "o3hb",
    "r23",
)


LINE_RATIO_PROPERTIES = frozenset({"balmer_decrement", "bpt_nii", "bpt_sii", "o32", "o3hb", "r23"})

PARAMS = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}


def _build_chain(ssp, with_cue):
    from tengri import FIXED, Fixed, SEDModel, Uniform
    from tengri.observation import Observation, Photometry

    groups = dict(
        ssp_data=ssp,
        observation=Observation(
            photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
        ),
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.5,
            "age_gyr": 5.0,
        },
        dust={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.3,
        },
        redshift=Fixed(0.1),
        approx=None,
    )
    if with_cue:
        groups["neb"] = {"type": "cue", "all_params": FIXED}
    return SEDModel.build(**groups)


@pytest.fixture(scope="module")
def property_arms(ssp_bare):
    """The two chains #1837 measured, each evaluated in both precisions.

    The float32 arm is gated non-vacuous: without an explicit dtype assertion a
    "float32" run that silently executed in float64 reports every property as
    healthy, which is exactly how the first #1837 probe went wrong.
    """
    out = {}
    for name, with_cue in (("dust", False), ("cue", True)):
        pred64 = _build_chain(ssp_bare, with_cue).predict(PARAMS)
        out[(name, "f64")] = dict(pred64.properties)
        with jax.enable_x64(False):
            pred32 = _build_chain(ssp_bare, with_cue).predict(PARAMS)
            assert np.asarray(pred32.rest_sed()).dtype == np.float32, (
                f"{name}: float32 arm ran in float64 -- the comparison would be vacuous"
            )
            out[(name, "f32")] = dict(pred32.properties)
    return out


@pytest.mark.parametrize("prop", BROKEN_PROPERTIES)
def test_property_finite_in_float32(property_arms, prop):
    """A float64 answer inside float32 range must not become inf/nan there."""
    chain = "cue" if prop in LINE_RATIO_PROPERTIES else "dust"
    ref = np.float64(np.asarray(property_arms[(chain, "f64")][prop]))
    if not np.isfinite(ref) or abs(ref) > F32_MAX:
        pytest.skip(f"{prop}: float64 value {ref!r} is not itself representable in float32")
    got = np.float64(np.asarray(property_arms[(chain, "f32")][prop]))
    assert np.isfinite(got), (
        f"{prop} is {got!r} in float32 though float64 gives {ref:.6e}, "
        f"which is well inside float32 range (#1837)"
    )
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=1e-6)
