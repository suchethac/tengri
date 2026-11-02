# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the free-redshift Hessian of the cosmology is finite.

Komatsu et al. (2011), ApJS 192, 18, eq. 26 gives the massive-neutrino
term ``[1 + (0.3173 y)**1.83]**(1/1.83)``. A massless species has
``y = 0`` exactly. Its first z-derivative is finite, but the second
derivative contains ``0**(1.83 - 2) = inf`` multiplied by ``dy/dz = 0``,
which evaluates to NaN. Every free-redshift log-posterior then carried a
NaN row in ``jax.hessian``. The massless species is now selected as its
exact value 1, so no derivative order differentiates the power at a zero
base, and the value is unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.spatial.sersic import Sersic, _b_n
from tengri.protocols.component import ForwardState
from tengri.utils.cosmology import (
    K_BOLTZ_EV,
    KOMATSU_NU_SHAPE_INDEX,
    KOMATSU_NU_SHAPE_SCALE,
    NEUTRINO_FERMI_DIRAC_CORRECTION,
    NU_TO_CMB_TEMP_RATIO,
    PLANCK18,
    _e_of_z,
    _flat_density_params,
    _nu_relative_density,
    _radiation_fields,
    age_at_z,
)

pytestmark = pytest.mark.regression_bug

REDSHIFTS = (0.05, 0.5, 2.0)
FD_STEP = 1e-5


def _old_nu_relative_density(z, neff, m_nu_eV, tcmb0):
    """The pre-fix formula, written out verbatim (no massless select)."""
    z = jnp.asarray(z)
    n_nu = len(m_nu_eV)
    neff_per_nu = neff / n_nu
    tnu0 = NU_TO_CMB_TEMP_RATIO * tcmb0
    kt_nu0_ev = K_BOLTZ_EV * tnu0
    kt_nu0_ev_safe = jnp.where(kt_nu0_ev > 0.0, kt_nu0_ev, 1.0)
    m_nu = jnp.asarray(m_nu_eV)
    nu_y = jnp.where(kt_nu0_ev > 0.0, m_nu / kt_nu0_ev_safe, 0.0)
    curr_nu_y = nu_y / (1.0 + z)[..., None]
    rel_mass_per = (1.0 + (KOMATSU_NU_SHAPE_SCALE * curr_nu_y) ** KOMATSU_NU_SHAPE_INDEX) ** (
        1.0 / KOMATSU_NU_SHAPE_INDEX
    )
    rel_mass = jnp.sum(rel_mass_per, axis=-1)
    return NEUTRINO_FERMI_DIRAC_CORRECTION * neff_per_nu * rel_mass


def _e_of_z_scalar(z):
    ogamma0, ode0 = _flat_density_params(PLANCK18)
    return _e_of_z(z, PLANCK18, ogamma0, ode0)


def _assert_hessian_matches_fd(f, z0):
    z = jnp.asarray(z0, dtype=jnp.float64)
    hess = float(jax.hessian(f)(z))
    grad = jax.grad(f)
    fd = float((grad(z + FD_STEP) - grad(z - FD_STEP)) / (2.0 * FD_STEP))
    assert np.isfinite(hess), f"jax.hessian is not finite at z={z0}: {hess}"
    assert hess != 0.0, (
        f"jax.hessian is identically zero at z={z0}; the redshift curvature cannot vanish"
    )
    np.testing.assert_allclose(hess, fd, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("z0", REDSHIFTS)
def test_nu_relative_density_value_is_bit_identical_to_old_formula(z0):
    tcmb0, neff, m_nu_eV = _radiation_fields(PLANCK18)
    new = _nu_relative_density(jnp.asarray(z0), neff, m_nu_eV, tcmb0)
    old = _old_nu_relative_density(jnp.asarray(z0), neff, m_nu_eV, tcmb0)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


@pytest.mark.parametrize("z0", REDSHIFTS)
def test_e_of_z_hessian_is_finite_and_matches_fd_of_grad(z0):
    _assert_hessian_matches_fd(_e_of_z_scalar, z0)


@pytest.mark.parametrize("z0", REDSHIFTS)
def test_age_at_z_hessian_is_finite_and_matches_fd_of_grad(z0):
    def age(z):
        return age_at_z(z, cosmo=PLANCK18)

    _assert_hessian_matches_fd(age, z0)


def _sersic_center_intensity(n, re_kpc):
    """Center pixel of an odd grid, where r_ell is exactly zero."""
    x = jnp.linspace(-5.0, 5.0, 21)
    xx, yy = jnp.meshgrid(x, x)
    state = ForwardState(wave=jnp.zeros(1))
    state = state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=(xx, yy)))
    params = {
        "spatial_re_kpc": re_kpc,
        "spatial_n": n,
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    out = Sersic().apply(state, params)
    return out.derived["spatial_profile_2d"][10, 10]


@pytest.mark.parametrize("n0", [1.0, 4.0])
def test_sersic_center_value_is_exp_bn(n0):
    center = _sersic_center_intensity(jnp.float64(n0), jnp.float64(1.5))
    np.testing.assert_array_equal(np.asarray(center), np.asarray(jnp.exp(_b_n(jnp.float64(n0)))))


@pytest.mark.parametrize(("n0", "re0"), [(1.0, 2.0), (4.0, 1.5)])
def test_sersic_center_grad_and_hessian_are_finite(n0, re0):
    def f(q):
        return _sersic_center_intensity(q[0], q[1])

    q = jnp.array([n0, re0], dtype=jnp.float64)
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(q))))
    assert np.all(np.isfinite(np.asarray(jax.hessian(f)(q))))
