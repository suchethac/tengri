# SPDX-License-Identifier: BSD-3-Clause
"""Public helpers documented as JIT/vmap-safe must run under jit and vmap (survey B).

A host conversion of a possibly-traced value (``float(x)``, ``bool(x)``, a Python
``if`` on an array) raises ``ConcretizationTypeError`` under ``jax.jit`` and breaks
``jax.vmap``. Each test here calls one fixed site under ``jit`` and ``vmap`` with a
traced input and compares the result with the eager call at rtol 1e-12.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.analysis.diagnostics.fisher import compute_fisher_matrix
from tengri.analysis.simulate import sed_from_sfh
from tengri.observation.noise import apply_zp_floor

pytestmark = pytest.mark.contract

_RTOL = 1e-12


def _assert_close(a, b):
    np.testing.assert_allclose(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), rtol=_RTOL
    )


def _tabulated_sfh():
    t_gyr = jnp.linspace(0.05, 13.0, 80)
    sfr = 1.0 + jnp.sin(t_gyr) ** 2
    return t_gyr, sfr


# -- analysis/simulate.py::sed_from_sfh ------------------------------------------------


def test_sed_from_sfh_sed_under_jit_and_vmap_over_dust_tau(synthetic_ssp_wide_physical):
    ssp = synthetic_ssp_wide_physical
    t_gyr, sfr = _tabulated_sfh()

    def sed_of(tau):
        return sed_from_sfh(t_gyr, sfr, ssp, log_z=-0.3, dust_tau_bc=tau, dust_tau_diff=0.3)["sed"]

    taus = jnp.array([0.0, 0.5, 1.5])
    eager = jnp.stack([sed_of(t) for t in taus])
    _assert_close(jax.jit(sed_of)(taus[1]), eager[1])
    _assert_close(jax.vmap(sed_of)(taus), eager)


def test_sed_from_sfh_stellar_mass_under_jit(synthetic_ssp_wide_physical):
    ssp = synthetic_ssp_wide_physical
    t_gyr, sfr = _tabulated_sfh()

    def mass_of(scale):
        return sed_from_sfh(t_gyr, scale * sfr, ssp)["stellar_mass"]

    _assert_close(jax.jit(mass_of)(2.0), mass_of(2.0))
    _assert_close(
        jax.vmap(mass_of)(jnp.array([1.0, 2.0])), jnp.array([mass_of(1.0), mass_of(2.0)])
    )


def test_sed_from_sfh_traced_scalar_log_z_under_jit(synthetic_ssp_wide_physical):
    ssp = synthetic_ssp_wide_physical
    t_gyr, sfr = _tabulated_sfh()

    def sed_of(log_z):
        return sed_from_sfh(t_gyr, sfr, ssp, log_z=log_z)["sed"]

    _assert_close(jax.jit(sed_of)(-0.3), sed_of(-0.3))


# -- observation/noise.py::apply_zp_floor ----------------------------------------------


def test_apply_zp_floor_traced_floor_under_jit_and_vmap():
    flux = jnp.array([1.0, 2.0, 3.0])
    noise = jnp.array([0.05, 0.10, 0.15])

    def sigma_of(floor):
        return apply_zp_floor(flux, noise, floor)

    floors = jnp.array([0.02, 0.10])
    eager = jnp.stack([sigma_of(f) for f in floors])
    _assert_close(jax.jit(sigma_of)(floors[0]), eager[0])
    _assert_close(jax.vmap(sigma_of)(floors), eager)


# -- analysis/diagnostics/fisher.py::compute_fisher_matrix -----------------------------


def _physical_ssp(ssp):
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    return SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * 1.0e-17,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )


def _fisher_model(ssp):
    return SEDModel.build(
        ssp_data=_physical_ssp(ssp),
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        redshift=Fixed(0.1),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": Uniform(0.5, 3.0),
            "age_gyr": Fixed(1.0),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
    )


def test_compute_fisher_matrix_under_jit_with_traced_params(synthetic_ssp_wide):
    names = ["sfh_delayed_log_total_mass", "sfh_delayed_tau_gyr"]
    base = {"sfh_delayed_log_total_mass": 10.0, "sfh_delayed_tau_gyr": 1.0}
    model = _fisher_model(synthetic_ssp_wide)
    noise = jnp.abs(model.predict_photometry(base)) * 0.05

    def fim_of(theta):
        params = {"sfh_delayed_log_total_mass": theta[0], "sfh_delayed_tau_gyr": theta[1]}
        fim, _ = compute_fisher_matrix(model, params, noise, param_names=names)
        return fim

    theta = jnp.array([10.0, 1.0])
    _assert_close(jax.jit(fim_of)(theta), fim_of(theta))


def test_compute_photometry_sensitivity_under_jit_with_traced_params(synthetic_ssp_wide):
    from tengri.analysis.diagnostics.saliency import compute_photometry_sensitivity

    names = ["sfh_delayed_log_total_mass", "sfh_delayed_tau_gyr"]
    model = _fisher_model(synthetic_ssp_wide)

    def jac_of(theta):
        params = {"sfh_delayed_log_total_mass": theta[0], "sfh_delayed_tau_gyr": theta[1]}
        jac, _ = compute_photometry_sensitivity(model, params, param_names=names)
        return jac

    theta = jnp.array([10.0, 1.0])
    _assert_close(jax.jit(jac_of)(theta), jac_of(theta))
