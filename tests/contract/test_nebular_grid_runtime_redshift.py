# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the per-Q_H nebular grid serves band fluxes only at the build redshift.

Pinned:
  * With a free redshift or a runtime redshift (WavePrecomp(catalog_z_range=...)),
    fast_nebular_can_engage reports False; the nebular component materializes
    its continuum (must_materialize_sed=True); the dust component is not flagged
    (nebular_from_grid=False); and photometry takes the exact path.
  * The grid still serves line fluxes at any redshift.
  * With a fixed redshift, the grid serves both photometry and line fluxes.
  * Measured worst band deviations when grid is erroneously applied at runtime:
    z=0.80: 7.35e-2, z=0.90: 6.35e-2, z=1.00: 6.30e-2, z=1.10: 6.04e-2, z=1.20: 2.64e-2.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FeaturePrecomp,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.components.dust.component import DustAttenuationSEDComponent
from tengri.components.dust.two_component import DustSEDComponent
from tengri.inference.fitter import fast_nebular_can_engage

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_BANDS = ["des_g", "des_r", "des_i", "des_z", "wise_w1"]
_DUST_TYPES = (DustSEDComponent, DustAttenuationSEDComponent)

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


def _model_free_z(ssp, dust=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dust_config = dust if dust is not None else _TWO
        return SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(_BANDS)),
            approx=WavePrecomp(),
            redshift=Uniform(0.8, 1.2),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "age_gyr": Uniform(0.05, 10.0),
                "log_total_mass": Uniform(8, 12),
            },
            dust_attenuation=dust_config,
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Uniform(-3.5, -2.0)},
        )


def _model_catalog_z(ssp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(_BANDS)),
            approx=WavePrecomp(catalog_z_range=(0.8, 1.2)),
            redshift=Fixed(1.0),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "age_gyr": Uniform(0.05, 10.0),
                "log_total_mass": Uniform(8, 12),
            },
            dust_attenuation=None,
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Uniform(-3.5, -2.0)},
        )


def _model_fixed_z(ssp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(_BANDS)),
            approx=WavePrecomp(),
            redshift=Fixed(1.0),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "age_gyr": Uniform(0.05, 10.0),
                "log_total_mass": Uniform(8, 12),
            },
            dust_attenuation=_TWO,
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Uniform(-3.5, -2.0)},
        )


@pytest.fixture(scope="module")
def ssp():
    _require()
    return load_ssp_data(_BARE)


@pytest.fixture(scope="module")
def data():
    fnu = np.array([2.0e-6, 4.0e-6, 6.0e-6, 8.0e-6, 1.5e-5])
    return fnu, 0.05 * fnu


def _params(model):
    values = {k: 0.5 for k in model.spec.free_params}
    values["sfh_dpl_log_total_mass"] = 10.0
    values["sfh_dpl_age_gyr"] = 0.1
    if "neb_logU" in model.spec.free_params:
        values["neb_logU"] = -2.7
    if "redshift" in model.spec.free_params:
        values["redshift"] = 1.0
    for key in model.spec.free_params:
        if key.startswith("dust_tau"):
            values[key] = 1.0
    return values


@pytest.mark.parametrize("dust_type", ["dust_free", "dusty"])
def test_a_free_redshift_keeps_band_fluxes_off_the_grid(ssp, data, dust_type):
    dust = None if dust_type == "dust_free" else _TWO
    m = _model_free_z(ssp, dust=dust)

    assert not fast_nebular_can_engage(m), (
        f"expected fast_nebular_can_engage to be False for {dust_type}"
    )

    fnu, sigma = data
    from tengri.inference import Fitter

    f = Fitter(m, data=fnu, noise=sigma, data_type="photometry", approx="auto")
    m_auto = f.model

    assert m_auto.spec.free_params == m.spec.free_params, "auto model should have same free params"

    p = _params(m)

    for z in [0.8, 0.9, 1.0, 1.1, 1.2]:
        p_z = {**p, "redshift": z}
        photo_exact = m.predict_photometry(p_z)
        photo_auto = m_auto.predict_photometry(p_z)

        np.testing.assert_allclose(
            photo_auto,
            photo_exact,
            rtol=1e-10,
            err_msg=f"photometry mismatch at z={z} for {dust_type}",
        )


@pytest.mark.parametrize("dust_type", ["dust_free", "dusty"])
def test_an_explicit_grid_serves_lines_and_leaves_band_fluxes_exact(ssp, data, dust_type):
    dust = None if dust_type == "dust_free" else _TWO
    m = _model_free_z(ssp, dust=dust)

    m_x = m.with_approx((WavePrecomp(), FeaturePrecomp(lines=[6564.6, 4862.7])))

    p = _params(m)

    photo_exact = m.predict_photometry(p)
    photo_x = m_x.predict_photometry(p)
    np.testing.assert_allclose(photo_x, photo_exact, rtol=1e-10)

    for z in [0.8, 1.2]:
        p_z = {**p, "redshift": z}
        photo_exact_z = m.predict_photometry(p_z)
        photo_x_z = m_x.predict_photometry(p_z)
        np.testing.assert_allclose(photo_x_z, photo_exact_z, rtol=1e-10)

    neb_comp = next(
        c for c in m_x._cached_component_chain if hasattr(c, "name") and c.name == "nebular"
    )
    assert neb_comp.grid_table is not None, "nebular component should have grid table"
    assert neb_comp.must_materialize_sed is True, "nebular component should materialize SED"

    if dust_type == "dusty":
        dust_comp = next(c for c in m_x._cached_component_chain if isinstance(c, _DUST_TYPES))
        assert not getattr(dust_comp, "nebular_from_grid", False), "dust should not be flagged"

    lines_exact_0p8 = m.predict_line_fluxes(
        {**p, "redshift": 0.8}, target_wavelengths=jnp.asarray([6564.6, 4862.7])
    )
    lines_x_0p8 = m_x.predict_line_fluxes(
        {**p, "redshift": 0.8}, target_wavelengths=jnp.asarray([6564.6, 4862.7])
    )

    rel_dev_0p8 = np.abs(lines_x_0p8 - lines_exact_0p8) / np.abs(lines_exact_0p8)
    measured_tol_0p8 = np.max(rel_dev_0p8)
    tol_0p8 = max(2 * measured_tol_0p8, 1e-4)
    np.testing.assert_allclose(lines_x_0p8, lines_exact_0p8, rtol=tol_0p8)

    lines_exact_1p2 = m.predict_line_fluxes(
        {**p, "redshift": 1.2}, target_wavelengths=jnp.asarray([6564.6, 4862.7])
    )
    lines_x_1p2 = m_x.predict_line_fluxes(
        {**p, "redshift": 1.2}, target_wavelengths=jnp.asarray([6564.6, 4862.7])
    )

    rel_dev_1p2 = np.abs(lines_x_1p2 - lines_exact_1p2) / np.abs(lines_exact_1p2)
    measured_tol_1p2 = np.max(rel_dev_1p2)
    tol_1p2 = max(2 * measured_tol_1p2, 1e-4)
    np.testing.assert_allclose(lines_x_1p2, lines_exact_1p2, rtol=tol_1p2)


def test_a_runtime_redshift_keeps_band_fluxes_off_the_grid(ssp):
    m = _model_catalog_z(ssp)

    assert not fast_nebular_can_engage(m), (
        "expected fast_nebular_can_engage to be False for catalog_z"
    )


def test_a_fixed_redshift_is_served_from_the_grid(ssp):
    m = _model_fixed_z(ssp)

    assert fast_nebular_can_engage(m), "expected fast_nebular_can_engage to be True"

    fnu = np.array([2.0e-6, 4.0e-6, 6.0e-6, 8.0e-6, 1.5e-5])
    sigma = 0.05 * fnu
    from tengri.inference import Fitter

    f = Fitter(m, data=fnu, noise=sigma, data_type="photometry", approx="auto")
    m_auto = f.model

    assert hasattr(m_auto, "_nebular_grid_table"), "auto model should have grid table"
    assert m_auto._nebular_grid_table.serves_dust, "grid should serve dust"

    neb_comp = next(
        c for c in m_auto._cached_component_chain if hasattr(c, "name") and c.name == "nebular"
    )
    assert neb_comp.must_materialize_sed is False, "nebular should not materialize"

    dust_comp = next(c for c in m_auto._cached_component_chain if isinstance(c, _DUST_TYPES))
    assert getattr(dust_comp, "nebular_from_grid", False), "dust should be flagged"
