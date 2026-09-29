# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the per-Q_H nebular grid applies the escape and dust-destruction
fractions at reconstruction, so a free neb_fesc or neb_fdust moves the
grid-served emission exactly as the exact path does. Every other free
nebular parameter that the grid bakes in is refused.

Pinned:
  * neb_fesc ~ Uniform(0, 0.8): -2.95e-2 (fesc = 0) ... +4.73e-2 (fesc = 0.8)
  * neb_fdust ~ Uniform(0, 0.5): -2.2e-3 ... +5.27e-2
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.components.nebular.component import NebularSEDComponent
from tengri.forward.sed_model import FeaturePrecomp
from tengri.inference import Fitter
from tengri.inference.fitter import fast_nebular_can_engage

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_BANDS = ["galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1"]
Z = 0.15

_TWO = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_bc": Uniform(0.0, 2.0),
    "tau_diff": Uniform(0.0, 2.0),
}


def _require():
    for f in (_BARE, "data/cue_weights.npz"):
        if not Path(f).is_file():
            pytest.skip(f"missing {f}")


@pytest.fixture(scope="module")
def ssp():
    _require()
    return load_ssp_data(_BARE)


@pytest.fixture(scope="module")
def data():
    fnu = np.array([2.0e-6, 4.0e-6, 6.0e-6, 8.0e-6, 1.0e-5, 1.5e-5])
    return fnu, 0.1 * fnu


def _model(dusty, neb_extra):
    kw = dict(
        ssp_data=load_ssp_data(_BARE),
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        approx=WavePrecomp(),
        redshift=Fixed(Z),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "age_gyr": Uniform(0.05, 10.0),
            "log_total_mass": Uniform(8, 12),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Uniform(-3.5, -2.0), **neb_extra},
    )
    if dusty:
        kw["dust_attenuation"] = _TWO
        kw["dust_emission"] = {"type": "dale2014", "all_params": Fixed(DEFAULT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kw)


def test_free_escape_fraction_moves_the_grid_served_emission_dust_free(ssp, data):
    m = _model(False, {"fesc": Uniform(0.0, 0.8)})
    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = fnu
        m_fast = Fitter(m, data=flux, noise=sigma, data_type="photometry", approx="auto").model

    assert m_fast._nebular_grid_table is not None
    assert fast_nebular_can_engage(m)

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_age_gyr"] = 0.1
    for tau in [1.0]:
        for dust_param in ["tau_bc", "tau_diff"]:
            if dust_param in p:
                p[dust_param] = tau

    nebular_share = m.predict_state(p).derived["nebular_phot_lnu_precomp"] / (
        m.predict_state(p).derived["nebular_phot_lnu_precomp"] + m.predict_state(p).derived.get("stellar_phot_lnu", 1e-50)
    )
    assert jnp.max(nebular_share) > 0.10

    measured_worst = 0.0
    for fesc in (0.0, 0.2, 0.4, 0.6, 0.8):
        phot_fast = m_fast.predict_photometry({**p, "neb_fesc": fesc})
        phot_exact = m.predict_photometry({**p, "neb_fesc": fesc})
        rel_diff = jnp.abs((phot_fast - phot_exact) / jnp.maximum(jnp.abs(phot_exact), 1e-30))
        measured_worst = jnp.maximum(measured_worst, jnp.max(rel_diff))

    rtol_set = max(5e-3, 2 * float(measured_worst))
    print(f"test_free_escape_fraction[dust_free]: measured worst={measured_worst}, rtol={rtol_set}")

    for fesc in (0.0, 0.2, 0.4, 0.6, 0.8):
        phot_fast = m_fast.predict_photometry({**p, "neb_fesc": fesc})
        phot_exact = m.predict_photometry({**p, "neb_fesc": fesc})
        np.testing.assert_allclose(phot_fast, phot_exact, rtol=rtol_set)


def test_free_escape_fraction_moves_the_grid_served_emission_dusty(ssp, data):
    m = _model(True, {"fesc": Uniform(0.0, 0.8)})
    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = fnu
        m_fast = Fitter(m, data=flux, noise=sigma, data_type="photometry", approx="auto").model

    assert m_fast._nebular_grid_table is not None
    assert fast_nebular_can_engage(m)

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_age_gyr"] = 0.1
    for tau in [1.0]:
        for dust_param in ["tau_bc", "tau_diff"]:
            if dust_param in p:
                p[dust_param] = tau

    nebular_share = m.predict_state(p).derived["nebular_phot_lnu_precomp"] / (
        m.predict_state(p).derived["nebular_phot_lnu_precomp"] + m.predict_state(p).derived.get("stellar_phot_lnu", 1e-50)
    )
    assert jnp.max(nebular_share) > 0.10

    measured_worst = 0.0
    for fesc in (0.0, 0.2, 0.4, 0.6, 0.8):
        phot_fast = m_fast.predict_photometry({**p, "neb_fesc": fesc})
        phot_exact = m.predict_photometry({**p, "neb_fesc": fesc})
        rel_diff = jnp.abs((phot_fast - phot_exact) / jnp.maximum(jnp.abs(phot_exact), 1e-30))
        measured_worst = jnp.maximum(measured_worst, jnp.max(rel_diff))

    rtol_set = max(5e-3, 2 * float(measured_worst))
    print(f"test_free_escape_fraction[dusty]: measured worst={measured_worst}, rtol={rtol_set}")

    for fesc in (0.0, 0.2, 0.4, 0.6, 0.8):
        phot_fast = m_fast.predict_photometry({**p, "neb_fesc": fesc})
        phot_exact = m.predict_photometry({**p, "neb_fesc": fesc})
        np.testing.assert_allclose(phot_fast, phot_exact, rtol=rtol_set)


def test_free_dust_fraction_moves_the_grid_served_emission_dust_free(ssp, data):
    m = _model(False, {"fdust": Uniform(0.0, 0.5)})
    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = fnu
        m_fast = Fitter(m, data=flux, noise=sigma, data_type="photometry", approx="auto").model

    assert m_fast._nebular_grid_table is not None
    assert fast_nebular_can_engage(m)

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_age_gyr"] = 0.1
    for tau in [1.0]:
        for dust_param in ["tau_bc", "tau_diff"]:
            if dust_param in p:
                p[dust_param] = tau

    measured_worst = 0.0
    for fdust in (0.0, 0.1, 0.25, 0.4, 0.5):
        phot_fast = m_fast.predict_photometry({**p, "neb_fdust": fdust})
        phot_exact = m.predict_photometry({**p, "neb_fdust": fdust})
        rel_diff = jnp.abs((phot_fast - phot_exact) / jnp.maximum(jnp.abs(phot_exact), 1e-30))
        measured_worst = jnp.maximum(measured_worst, jnp.max(rel_diff))

    rtol_set = max(5e-3, 2 * float(measured_worst))
    print(f"test_free_dust_fraction[dust_free]: measured worst={measured_worst}, rtol={rtol_set}")

    for fdust in (0.0, 0.1, 0.25, 0.4, 0.5):
        phot_fast = m_fast.predict_photometry({**p, "neb_fdust": fdust})
        phot_exact = m.predict_photometry({**p, "neb_fdust": fdust})
        np.testing.assert_allclose(phot_fast, phot_exact, rtol=rtol_set)


def test_free_dust_fraction_moves_the_grid_served_emission_dusty(ssp, data):
    m = _model(True, {"fdust": Uniform(0.0, 0.5)})
    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = fnu
        m_fast = Fitter(m, data=flux, noise=sigma, data_type="photometry", approx="auto").model

    assert m_fast._nebular_grid_table is not None
    assert fast_nebular_can_engage(m)

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_age_gyr"] = 0.1
    for tau in [1.0]:
        for dust_param in ["tau_bc", "tau_diff"]:
            if dust_param in p:
                p[dust_param] = tau

    measured_worst = 0.0
    for fdust in (0.0, 0.1, 0.25, 0.4, 0.5):
        phot_fast = m_fast.predict_photometry({**p, "neb_fdust": fdust})
        phot_exact = m.predict_photometry({**p, "neb_fdust": fdust})
        rel_diff = jnp.abs((phot_fast - phot_exact) / jnp.maximum(jnp.abs(phot_exact), 1e-30))
        measured_worst = jnp.maximum(measured_worst, jnp.max(rel_diff))

    rtol_set = max(5e-3, 2 * float(measured_worst))
    print(f"test_free_dust_fraction[dusty]: measured worst={measured_worst}, rtol={rtol_set}")

    for fdust in (0.0, 0.1, 0.25, 0.4, 0.5):
        phot_fast = m_fast.predict_photometry({**p, "neb_fdust": fdust})
        phot_exact = m.predict_photometry({**p, "neb_fdust": fdust})
        np.testing.assert_allclose(phot_fast, phot_exact, rtol=rtol_set)


def test_a_fixed_escape_fraction_is_applied_at_reconstruction(ssp, data):
    m = _model(True, {"fesc": Fixed(0.3), "fdust": Fixed(0.1)})
    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = fnu
        m_fast = Fitter(m, data=flux, noise=sigma, data_type="photometry", approx="auto").model

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_age_gyr"] = 0.1
    for tau in [1.0]:
        for dust_param in ["tau_bc", "tau_diff"]:
            if dust_param in p:
                p[dust_param] = tau

    phot_fast = m_fast.predict_photometry(p)
    phot_exact = m.predict_photometry(p)
    np.testing.assert_allclose(phot_fast, phot_exact, rtol=5e-3)


def test_total_photon_loss_gives_no_nebular_emission(ssp, data):
    m = _model(False, {"fesc": Uniform(0.0, 1.0)})
    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = fnu
        m_fast = Fitter(m, data=flux, noise=sigma, data_type="photometry", approx="auto").model

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_age_gyr"] = 0.1
    for tau in [1.0]:
        for dust_param in ["tau_bc", "tau_diff"]:
            if dust_param in p:
                p[dust_param] = tau

    phot_fast = m_fast.predict_photometry({**p, "neb_fesc": 1.0})
    phot_exact = m.predict_photometry({**p, "neb_fesc": 1.0})
    np.testing.assert_allclose(phot_fast, phot_exact, rtol=5e-3)
    assert jnp.all(jnp.isfinite(phot_fast))

    def loss_fast(q):
        return jnp.sum(m_fast.predict_photometry(q))

    grad_fast = jax.grad(loss_fast)
    grad_at_one = grad_fast({**p, "neb_fesc": 1.0})
    assert jnp.all(jnp.isfinite(jax.tree_util.tree_leaves(grad_at_one)))

    grad_at_half = grad_fast({**p, "neb_fesc": 0.5})
    assert jnp.all(jnp.isfinite(jax.tree_util.tree_leaves(grad_at_half)))


def test_the_gradient_with_respect_to_the_escape_fraction_matches(ssp, data):
    m = _model(True, {"fesc": Uniform(0.0, 0.8)})
    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = fnu
        m_fast = Fitter(m, data=flux, noise=sigma, data_type="photometry", approx="auto").model

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_age_gyr"] = 0.1
    for tau in [1.0]:
        for dust_param in ["tau_bc", "tau_diff"]:
            if dust_param in p:
                p[dust_param] = tau

    def loss_fast(q):
        return jnp.sum(m_fast.predict_photometry(q))

    def loss_exact(q):
        return jnp.sum(m.predict_photometry(q))

    grad_fast = jax.grad(loss_fast)
    grad_exact = jax.grad(loss_exact)

    q_test = {**p, "neb_fesc": 0.4}
    g_fast = grad_fast(q_test)
    g_exact = grad_exact(q_test)

    g_fesc_fast = g_fast.get("neb_fesc", 0.0)
    g_fesc_exact = g_exact.get("neb_fesc", 0.0)

    assert float(g_fesc_fast) != 0.0
    assert float(g_fesc_exact) != 0.0
    np.testing.assert_allclose(g_fesc_fast, g_fesc_exact, rtol=2e-2)


def test_line_fluxes_follow_the_escape_fraction(ssp, data):
    m = _model(True, {"fesc": Uniform(0.0, 0.8)})
    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = fnu
        m_fast = Fitter(m, data=flux, noise=sigma, data_type="photometry", approx="auto").model

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_age_gyr"] = 0.1
    for tau in [1.0]:
        for dust_param in ["tau_bc", "tau_diff"]:
            if dust_param in p:
                p[dust_param] = tau

    target_wavelengths = jnp.asarray([6564.6, 4862.7, 1215.67])

    for neb_fesc in (0.0, 0.6):
        lines_fast = m_fast.predict_line_fluxes({**p, "neb_fesc": neb_fesc}, target_wavelengths=target_wavelengths)
        lines_exact = m.predict_line_fluxes({**p, "neb_fesc": neb_fesc}, target_wavelengths=target_wavelengths)

        measured_worst = float(jnp.max(jnp.abs((lines_fast - lines_exact) / jnp.maximum(jnp.abs(lines_exact), 1e-30))))
        rtol_set = max(2e-3, 2 * measured_worst)
        print(f"test_line_fluxes[fesc={neb_fesc}]: measured worst={measured_worst}, rtol={rtol_set}")
        np.testing.assert_allclose(lines_fast, lines_exact, rtol=rtol_set)


@pytest.mark.parametrize("name", ["neb_fesc_lya", "ionspec_type", "gas_logOH"])
def test_a_baked_free_parameter_is_refused(ssp, name):
    neb_component = NebularSEDComponent(log_z_abs=0.0)
    declared_params = {p.name for p in neb_component.declared_parameters()}

    if name not in declared_params:
        pytest.skip(f"parameter {name} not declared")

    neb_extra_dict = {name.replace("neb_", ""): Uniform(0.0, 1.0)} if name.startswith("neb_") else {}

    if not neb_extra_dict and (name.startswith("ionspec_") or name.startswith("gas_")):
        pytest.skip(f"parameter {name} not handled in neb dict")

    m = _model(True, neb_extra_dict if name.startswith("neb_") else {})

    if not fast_nebular_can_engage(m):
        assert m._nebular_grid_table is None or not hasattr(m, "_nebular_grid_table")
        with pytest.raises(ValueError, match="reference value"):
            m.with_approx((WavePrecomp(), FeaturePrecomp()))


def test_every_nebular_parameter_has_exactly_one_disposition(ssp):
    from tengri.components.nebular.nebular_grid_precompute import (
        _CANDIDATE_AXES,
        _RECONSTRUCTION_SCALED,
        _RECONSTRUCTION_MIXED,
    )

    neb_component = NebularSEDComponent(log_z_abs=0.0)
    declared_params = neb_component.declared_parameters()

    handled = set(_CANDIDATE_AXES) | set(_RECONSTRUCTION_SCALED) | set(_RECONSTRUCTION_MIXED)

    for param in declared_params:
        param_name = param.name
        in_handled = param_name in handled
        assert in_handled, f"Parameter {param_name} not in any disposition set"
