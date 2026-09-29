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
    luminosity_distance_mpc,
)

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
    """At z=0, D_L should return 10 pc (1e-5 Mpc) per the optical convention."""
    dl_z0 = luminosity_distance_mpc(0.0, cosmo=PLANCK18)
    expected = 1e-5  # 10 pc in Mpc
    assert abs(float(np.asarray(dl_z0)) - expected) < 1e-10, (
        f"D_L(z=0) should be 1e-5 Mpc, got {float(np.asarray(dl_z0))}"
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
