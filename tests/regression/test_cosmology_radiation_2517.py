# SPDX-License-Identifier: BSD-3-Clause
"""Test cosmological distances and ages match astropy.cosmology.Planck18.

Regression test for #2517: tengri.cosmology was omitting radiation and
massive-neutrino densities from E(z), causing D_L and age(z) to drift from
astropy's Planck18 by up to +0.21% (D_L) / +0.53% (age) at z=10. The fix adds
the Komatsu et al. (2011) neutrino energy-density formula and the photon
density Ω_γ(Tcmb0, H0) to E(z), and switches the age integral to an
a=1/(1+z') substitution (the z'-linear substitution alone left a ~0.09%
residual at z=10 from under-resolving the rapidly-varying near-z part of the
integrand). Both now agree with astropy's Planck18 to <1e-4 relative at every
z in {0.01, 0.1, 0.5, 1, 2, 3, 6, 10}.

Each named cosmology (PLANCK18, PLANCK15, WMAP5) states its own published
Tcmb0/Neff/m_nu explicitly at construction -- CosmoParams' field defaults
are radiation-free (Tcmb0=0.0), so a user-built cosmology is unaffected by
this fix unless it opts in to radiation. Both are pinned here to rtol 1e-6
against the matching astropy object, not the looser 1e-4 above (which
tracks the issue's original ask; 1e-6 is what the fix actually achieves).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.cosmology import (
    PLANCK15,
    PLANCK18,
    WMAP5,
    CosmoParams,
    _flat_density_params,
    _radiation_fields,
    age_at_z,
    age_at_z0,
    luminosity_distance,
    luminosity_distance_mpc,
)
from tengri.utils.physics_constants import MPC_CM

pytestmark = pytest.mark.regression_bug

_Z_GRID = (0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 6.0, 10.0)


@pytest.mark.parametrize("z", _Z_GRID)
def test_luminosity_distance_matches_astropy_planck18(z):
    """D_L(z) must agree with astropy.cosmology.Planck18 to <1e-4 relative."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    import astropy.units as u

    dl_astropy = astropy_cosmo.Planck18.luminosity_distance(z).to(u.Mpc).value
    dl_tengri = float(np.asarray(luminosity_distance_mpc(z, cosmo=PLANCK18)))

    rel_err = abs(dl_tengri / dl_astropy - 1.0)
    assert rel_err < 1e-4, (
        f"z={z}: D_L relative error = {rel_err:.2e} (limit 1e-4). "
        f"tengri={dl_tengri:.6f}, astropy={dl_astropy:.6f}"
    )


@pytest.mark.parametrize("z", _Z_GRID)
def test_age_at_z_matches_astropy_planck18(z):
    """age(z) must agree with astropy.cosmology.Planck18 to <1e-4 relative."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")

    age_astropy = astropy_cosmo.Planck18.age(z).value
    age_tengri = float(np.asarray(age_at_z(z, cosmo=PLANCK18)))

    rel_err = abs(age_tengri / age_astropy - 1.0)
    assert rel_err < 1e-4, (
        f"z={z}: age relative error = {rel_err:.2e} (limit 1e-4). "
        f"tengri={age_tengri:.6f}, astropy={age_astropy:.6f}"
    )


def test_planck18_field_values_match_astropy():
    """tengri.PLANCK18's declared fields, and the density fractions derived
    from them, must reproduce astropy.cosmology.Planck18 to <1e-4 relative
    (Om0, H0, Tcmb0, Neff, m_nu — the inputs) and the derived Ogamma0 /
    Onu0 / Ode0 (Planck Collaboration 2020, A&A 641, A6)."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    ap = astropy_cosmo.Planck18

    assert abs(PLANCK18.Om0 - float(ap.Om0)) < 1e-4
    assert abs(PLANCK18.h * 100.0 - float(ap.H0.value)) < 1e-4
    assert abs(PLANCK18.Tcmb0 - float(ap.Tcmb0.value)) < 1e-4
    assert abs(PLANCK18.Neff - float(ap.Neff)) < 1e-4
    assert PLANCK18.m_nu_eV == tuple(float(m) for m in ap.m_nu.value)

    ogamma0, ode0 = _flat_density_params(PLANCK18)
    tcmb0, neff, m_nu_eV = _radiation_fields(PLANCK18)
    from tengri.utils.cosmology import _nu_relative_density

    onu0 = ogamma0 * float(_nu_relative_density(0.0, neff, m_nu_eV, tcmb0))

    assert abs(ogamma0 / float(ap.Ogamma0) - 1.0) < 1e-4
    assert abs(onu0 / float(ap.Onu0) - 1.0) < 1e-4
    assert abs(ode0 / float(ap.Ode0) - 1.0) < 1e-4


def test_jit_traced_redshift():
    """luminosity_distance_mpc / age_at_z must jit-compile with a traced z."""

    @jax.jit
    def dl_jit(z):
        return luminosity_distance_mpc(z, cosmo=PLANCK18)

    @jax.jit
    def age_jit(z):
        return age_at_z(z, cosmo=PLANCK18)

    dl = float(dl_jit(1.0))
    age = float(age_jit(1.0))
    assert jnp.isfinite(dl) and dl > 0.0
    assert jnp.isfinite(age) and age > 0.0
    # Must match the un-jitted values exactly (same computation).
    assert dl == pytest.approx(float(luminosity_distance_mpc(1.0, cosmo=PLANCK18)))
    assert age == pytest.approx(float(age_at_z(1.0, cosmo=PLANCK18)))


def test_grad_traced_redshift():
    """luminosity_distance_mpc / age_at_z must be differentiable w.r.t. z."""
    d_dl_dz = jax.grad(lambda z: luminosity_distance_mpc(z, cosmo=PLANCK18))(1.0)
    d_age_dz = jax.grad(lambda z: age_at_z(z, cosmo=PLANCK18))(1.0)

    # Finite and non-zero together (#2100/#2178 shape): a silently-zero or
    # silently-NaN gradient must each fail loudly, not just one half.
    assert jnp.isfinite(d_dl_dz), "d(D_L)/dz is non-finite"
    assert d_dl_dz != 0.0, "d(D_L)/dz is identically zero"
    assert jnp.isfinite(d_age_dz), "d(age)/dz is non-finite"
    assert d_age_dz != 0.0, "d(age)/dz is identically zero"
    # D_L increases with z; age decreases with z.
    assert d_dl_dz > 0.0
    assert d_age_dz < 0.0


@pytest.mark.parametrize(
    "cosmo_name,cosmology",
    [
        ("PLANCK15", PLANCK15),
        ("WMAP5", WMAP5),
    ],
)
@pytest.mark.parametrize("z", _Z_GRID)
def test_luminosity_distance_vs_astropy_other_cosmologies(z, cosmo_name, cosmology):
    """PLANCK15 and WMAP5 each state their own published Tcmb0/Neff/m_nu
    explicitly at construction (see the lazy loader in ``__getattr__``),
    so both reach the same <1e-4 precision as PLANCK18 — no per-cosmology
    tolerance widening."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    import astropy.units as u

    astropy_obj = getattr(astropy_cosmo, cosmo_name.title().replace("Wmap", "WMAP"), None)
    if astropy_obj is None:
        pytest.skip(f"astropy does not have {cosmo_name}")

    dl_astropy = astropy_obj.luminosity_distance(z).to(u.Mpc).value
    dl_tengri = float(np.asarray(luminosity_distance_mpc(z, cosmo=cosmology)))

    rel_err = abs(dl_tengri / dl_astropy - 1.0)
    assert rel_err < 1e-4, (
        f"{cosmo_name} z={z}: D_L relative error = {rel_err:.2e} (limit 1e-4). "
        f"tengri={dl_tengri:.6f}, astropy={dl_astropy:.6f}"
    )


def test_luminosity_distance_z0_behavior():
    """At z=0, the geometric D_L (Mpc) vanishes exactly; the cm-scale primitive
    used directly for flux conversion floors at 10 pc, the optical
    absolute-magnitude convention, so L_nu -> F_nu stays finite."""
    dl_mpc_z0 = luminosity_distance_mpc(0.0, cosmo=PLANCK18)
    assert float(np.asarray(dl_mpc_z0)) == 0.0, (
        f"D_L(z=0) [Mpc] should vanish exactly, got {float(np.asarray(dl_mpc_z0))}"
    )
    dl_cm_z0 = luminosity_distance(0.0, cosmo=PLANCK18)
    expected_cm = 1e-5 * MPC_CM  # 10 pc in cm
    assert abs(float(np.asarray(dl_cm_z0)) - expected_cm) < 1e-10 * expected_cm, (
        f"D_L(z=0) [cm] should be 10 pc, got {float(np.asarray(dl_cm_z0))}"
    )


def test_radiation_fields_default_for_foreign_cosmo():
    """A bare 4-field cosmo object (e.g. dsps.cosmology.flat_wcdm.CosmoParams,
    which has no Tcmb0/Neff/m_nu_eV attributes at all) must fall back to
    CosmoParams' declared radiation defaults rather than raising."""
    from typing import NamedTuple

    class _FourFieldCosmo(NamedTuple):
        Om0: float
        w0: float
        wa: float
        h: float

    foreign = _FourFieldCosmo(Om0=0.3, w0=-1.0, wa=0.0, h=0.7)
    tcmb0, neff, m_nu_eV = _radiation_fields(foreign)
    assert tcmb0 == CosmoParams._field_defaults["Tcmb0"]
    assert neff == CosmoParams._field_defaults["Neff"]
    assert m_nu_eV == CosmoParams._field_defaults["m_nu_eV"]


# ─────────────────────────────────────────────────────────────────────
# Named cosmologies state their own radiation content explicitly, and a
# bare (radiation-free) CosmoParams reproduces a plain astropy
# FlatLambdaCDM exactly -- neither depends on CosmoParams' field defaults.
# ─────────────────────────────────────────────────────────────────────

_TIGHT_Z_GRID = (0.01, 0.5, 1.0, 3.0, 6.0, 10.0)


@pytest.mark.parametrize(
    "cosmo_name,cosmology",
    [("WMAP5", WMAP5), ("PLANCK15", PLANCK15), ("PLANCK18", PLANCK18)],
)
@pytest.mark.parametrize("z", _TIGHT_Z_GRID)
def test_named_cosmologies_match_astropy_tightly(z, cosmo_name, cosmology):
    """D_L and age(z) for WMAP5/PLANCK15/PLANCK18 vs the matching
    astropy object, rtol 1e-6 -- each named cosmology states its own
    published Tcmb0/Neff/m_nu explicitly at construction (#2517),
    not the (now radiation-free) CosmoParams field defaults."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    import astropy.units as u

    astropy_obj = getattr(astropy_cosmo, cosmo_name.title().replace("Wmap", "WMAP"))

    dl_tengri = float(luminosity_distance_mpc(z, cosmo=cosmology))
    dl_astropy = astropy_obj.luminosity_distance(z).to(u.Mpc).value
    assert dl_tengri == pytest.approx(dl_astropy, rel=1e-6)

    age_tengri = float(age_at_z(z, cosmo=cosmology))
    age_astropy = astropy_obj.age(z).value
    assert age_tengri == pytest.approx(age_astropy, rel=1e-6)


@pytest.mark.parametrize("z", _TIGHT_Z_GRID)
def test_bare_cosmoparams_matches_flatlambdacdm_radiation_free(z):
    """a bare CosmoParams(Om0=0.3, w0=-1.0, wa=0.0, h=0.7) --
    no radiation fields given -- must match
    astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3) (also radiation-free
    by default) to rtol 1e-6. This is #2517: CosmoParams'
    field defaults are radiation-free, so a user-built cosmology is
    unaffected by the radiation fix unless it opts in."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    import astropy.units as u

    bare = CosmoParams(Om0=0.3, w0=-1.0, wa=0.0, h=0.7)
    ap = astropy_cosmo.FlatLambdaCDM(H0=70, Om0=0.3)

    dl_tengri = float(luminosity_distance_mpc(z, cosmo=bare))
    dl_astropy = ap.luminosity_distance(z).to(u.Mpc).value
    assert dl_tengri == pytest.approx(dl_astropy, rel=1e-6)

    age_tengri = float(age_at_z(z, cosmo=bare))
    age_astropy = ap.age(z).value
    assert age_tengri == pytest.approx(age_astropy, rel=1e-6)


@pytest.mark.parametrize("z", _TIGHT_Z_GRID)
def test_bare_cosmoparams_matches_flatlambdacdm_with_radiation(z):
    """the same bare cosmology, this time with Planck18's
    radiation fields given explicitly, must match
    astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.7255,
    Neff=3.046, m_nu=[0,0,0.06]*u.eV) to rtol 1e-6 -- opting in to
    radiation on a non-named cosmology works identically to a named one."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    import astropy.units as u

    bare = CosmoParams(
        Om0=0.3, w0=-1.0, wa=0.0, h=0.7, Tcmb0=2.7255, Neff=3.046, m_nu_eV=(0.0, 0.0, 0.06)
    )
    ap = astropy_cosmo.FlatLambdaCDM(
        H0=70, Om0=0.3, Tcmb0=2.7255, Neff=3.046, m_nu=[0.0, 0.0, 0.06] * u.eV
    )

    dl_tengri = float(luminosity_distance_mpc(z, cosmo=bare))
    dl_astropy = ap.luminosity_distance(z).to(u.Mpc).value
    assert dl_tengri == pytest.approx(dl_astropy, rel=1e-6)

    age_tengri = float(age_at_z(z, cosmo=bare))
    age_astropy = ap.age(z).value
    assert age_tengri == pytest.approx(age_astropy, rel=1e-6)


def test_grad_finite_nonzero_for_radiation_free_bare_cosmology():
    """jax.grad of D_L and age w.r.t. z must be finite and
    non-zero for a bare radiation-free cosmology (Tcmb0=0.0) -- exercises
    the ``jnp.where``-on-a-safe-argument guard in
    :func:`tengri.utils.cosmology._nu_relative_density` against a 0/0 from
    dividing by Tcmb0 (#2517)."""
    bare = CosmoParams(Om0=0.3, w0=-1.0, wa=0.0, h=0.7)
    assert bare.Tcmb0 == 0.0

    d_dl_dz = jax.grad(lambda z: luminosity_distance_mpc(z, cosmo=bare))(1.0)
    d_age_dz = jax.grad(lambda z: age_at_z(z, cosmo=bare))(1.0)

    assert jnp.isfinite(d_dl_dz), "d(D_L)/dz is non-finite for Tcmb0=0"
    assert d_dl_dz != 0.0, "d(D_L)/dz is identically zero for Tcmb0=0"
    assert jnp.isfinite(d_age_dz), "d(age)/dz is non-finite for Tcmb0=0"
    assert d_age_dz != 0.0, "d(age)/dz is identically zero for Tcmb0=0"


# ─────────────────────────────────────────────────────────────────────
# Gradients w.r.t. h in float32: the critical density 3H0^2/(8 pi G) is
# ~9e-30 g/cm^3 at its natural cgs magnitude, and reverse-mode autodiff of
# a division by it needs the square of the denominator (~1e-58) as an
# intermediate -- underflowing to exactly 0 in float32, whose reciprocal
# is +inf. Every h-dependent prefactor in cosmology.py (the photon
# density, the Hubble distance, the Hubble time) is written as a plain
# Python float constant (evaluated once, never traced, at h=1) divided by
# the traced h itself, so no traced quantity ever reaches that magnitude.
# ─────────────────────────────────────────────────────────────────────

_BARE_RADIATION_FREE = CosmoParams(Om0=PLANCK18.Om0, w0=-1.0, wa=0.0, h=PLANCK18.h)
_GRAD_COSMOLOGIES = (
    ("PLANCK18", PLANCK18),
    ("WMAP5", WMAP5),
    ("bare_radiation_free", _BARE_RADIATION_FREE),
)
_GRAD_Z_GRID = (0.1, 2.0, 10.0)


def _log_derivative_wrt_h(func, cosmo, z, x64: bool):
    """d ln(func(z, cosmo)) / d ln(h), holding Om0/w0/wa/Tcmb0/Neff/m_nu fixed."""
    with jax.enable_x64(x64):

        def f(h):
            return func(z, cosmo=cosmo._replace(h=h))

        h = jnp.asarray(cosmo.h)
        value = f(h)
        d_dh = jax.grad(f)(h)
        return float(value), float(d_dh), float(d_dh) * float(h) / float(value)


@pytest.mark.parametrize("z", _GRAD_Z_GRID)
@pytest.mark.parametrize("cosmo_name,cosmology", _GRAD_COSMOLOGIES)
@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
def test_grad_wrt_h_om0_z_is_finite(z, cosmo_name, cosmology, x64):
    """jax.grad of D_L and age w.r.t. h, Om0, and z is finite AND non-zero,
    in both float64 and float32 -- finite alone admits an identically-zero
    gradient (a dead parameter), and non-zero alone admits NaN (`nan !=
    0.0` is True), so both are asserted together."""
    with jax.enable_x64(x64):
        h_val = jnp.asarray(cosmology.h)
        om0_val = jnp.asarray(cosmology.Om0)
        z_val = jnp.asarray(z)

        for func in (luminosity_distance_mpc, age_at_z):
            d_dh = jax.grad(lambda h, func=func: func(z, cosmo=cosmology._replace(h=h)))(h_val)
            d_dom0 = jax.grad(lambda om0, func=func: func(z, cosmo=cosmology._replace(Om0=om0)))(
                om0_val
            )
            d_dz = jax.grad(lambda zz, func=func: func(zz, cosmo=cosmology))(z_val)

            assert jnp.isfinite(d_dh), f"{cosmo_name} z={z} {func.__name__}: d/dh non-finite"
            assert d_dh != 0.0, f"{cosmo_name} z={z} {func.__name__}: d/dh identically zero"
            assert jnp.isfinite(d_dom0), f"{cosmo_name} z={z} {func.__name__}: d/dOm0 non-finite"
            assert d_dom0 != 0.0, f"{cosmo_name} z={z} {func.__name__}: d/dOm0 identically zero"
            assert jnp.isfinite(d_dz), f"{cosmo_name} z={z} {func.__name__}: d/dz non-finite"
            assert d_dz != 0.0, f"{cosmo_name} z={z} {func.__name__}: d/dz identically zero"


@pytest.mark.parametrize("z", _GRAD_Z_GRID)
@pytest.mark.parametrize("cosmo_name,cosmology", _GRAD_COSMOLOGIES)
def test_log_derivative_wrt_h_matches_across_precision(z, cosmo_name, cosmology):
    """d ln D_L / d ln h and d ln age / d ln h computed in float32 must
    agree with the float64 value to 1e-3 relative -- the h-dependence is
    close to a pure power law (D_L, age both scale close to 1/h for a
    fixed physical cosmology), so a numerically sound float32 path
    should reproduce it, not merely avoid inf/nan."""
    for func in (luminosity_distance_mpc, age_at_z):
        _, _, logd_64 = _log_derivative_wrt_h(func, cosmology, z, x64=True)
        _, _, logd_32 = _log_derivative_wrt_h(func, cosmology, z, x64=False)
        assert logd_32 == pytest.approx(logd_64, rel=1e-3), (
            f"{cosmo_name} z={z} {func.__name__}: float32 d ln/d ln h = {logd_32:.6f}, "
            f"float64 = {logd_64:.6f}"
        )


@pytest.mark.parametrize("z", _GRAD_Z_GRID)
@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
def test_radiation_free_log_derivative_wrt_h_is_minus_one(z, x64):
    """For a radiation-free cosmology, D_L and age both scale exactly as
    1/h at fixed physical density parameters (Om0, Ode0, and every E(z)
    term are h-independent by construction), so d ln D_L / d ln h = d ln
    age / d ln h = -1 exactly, in both float64 and float32."""
    for func in (luminosity_distance_mpc, age_at_z):
        _, _, logd = _log_derivative_wrt_h(func, _BARE_RADIATION_FREE, z, x64=x64)
        assert logd == pytest.approx(-1.0, abs=1e-4), (
            f"z={z} {func.__name__} ({'float64' if x64 else 'float32'}): "
            f"d ln/d ln h = {logd:.6f}, expected -1"
        )


@pytest.mark.parametrize("z", _GRAD_Z_GRID)
@pytest.mark.parametrize("cosmo_name,cosmology", _GRAD_COSMOLOGIES)
def test_values_match_astropy_in_float32(z, cosmo_name, cosmology):
    """D_L and age(z) themselves (not just their gradients) stay close to
    the float64/astropy reference when evaluated in float32."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    import astropy.units as u

    if cosmo_name == "bare_radiation_free":
        astropy_obj = astropy_cosmo.FlatLambdaCDM(H0=cosmology.h * 100.0, Om0=cosmology.Om0)
    else:
        astropy_obj = getattr(astropy_cosmo, cosmo_name.title().replace("Wmap", "WMAP"))

    dl_astropy = astropy_obj.luminosity_distance(z).to(u.Mpc).value
    age_astropy = astropy_obj.age(z).value

    with jax.enable_x64(False):
        dl32 = float(luminosity_distance_mpc(z, cosmo=cosmology))
        age32 = float(age_at_z(z, cosmo=cosmology))

    assert dl32 == pytest.approx(dl_astropy, rel=1e-5), f"{cosmo_name} z={z}: D_L"
    assert age32 == pytest.approx(age_astropy, rel=1e-5), f"{cosmo_name} z={z}: age"


def test_cm_unit_and_z0_forms_share_the_float32_safe_gradient():
    """luminosity_distance (cm) and age_at_z0 build on the same h-dependent
    prefactors as luminosity_distance_mpc and age_at_z (a constant unit
    conversion, and z=0 in the same quadrature), so their gradient w.r.t.
    h is finite and non-zero in float32 too."""
    with jax.enable_x64(False):
        h = jnp.asarray(PLANCK18.h)
        d_dl_dh = jax.grad(lambda hh: luminosity_distance(2.0, cosmo=PLANCK18._replace(h=hh)))(h)
        d_age0_dh = jax.grad(lambda hh: age_at_z0(cosmo=PLANCK18._replace(h=hh)))(h)
    assert jnp.isfinite(d_dl_dh), "d(luminosity_distance)/dh is non-finite in float32"
    assert d_dl_dh != 0.0, "d(luminosity_distance)/dh is identically zero in float32"
    assert jnp.isfinite(d_age0_dh), "d(age_at_z0)/dh is non-finite in float32"
    assert d_age0_dh != 0.0, "d(age_at_z0)/dh is identically zero in float32"
