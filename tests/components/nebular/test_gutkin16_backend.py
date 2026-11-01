# SPDX-License-Identifier: BSD-3-Clause
r"""The Gutkin+2016 nebular line backend reads its table and normalizes it per photon (#983).

``Gutkin16Backend`` looks up :math:`\log_{10}(L_{\rm line}/Q_H)` in the repackaged
NEOGAL grid and multiplies by the stellar ionizing photon rate. These tests pin the
node-exact lookup, the mup refusal, the sensitivity of every grid axis, the Case B
Lyman-alpha channel, the float32 path, and the JIT, vmap and gradient contracts.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.nebular._shared import render_nebular_lines
from tengri.components.nebular.gutkin16 import (
    LYA_OVER_HB_CASE_B,
    LYA_WAVE_AA,
    Gutkin16Backend,
    Gutkin16MupError,
    load_gutkin16_grid,
)
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_paper

_SSP_FILE = tengri.__file__.rsplit("/src/", 1)[0] + "/data/bc03_pdva_stelib_chabrier.h5"

_LOG_Z_NODE = -2.0  # absolute log10(Z), an exact node of the table
_LOG_U_NODE = -3.0  # log10(U_S), an exact node
_XI_NODE = 0.3
_NH_NODE = 2.0  # log10(n_H), an exact node
_CO_NODE = 1.0


@pytest.fixture(scope="module")
def ssp():
    import pathlib

    path = pathlib.Path(_SSP_FILE)
    if not path.is_file():
        pytest.skip(f"bare-stellar SSP not present: {path}")
    return tengri.load_ssp_data(str(path))


@pytest.fixture(scope="module")
def backend(ssp):
    return Gutkin16Backend(mup=100.0, continuum="analytic", ssp_data=ssp)


def _young_weights(backend, log_z):
    """Weights on one young age bin, scaled so that sum(w * Q_H) = 1 photon/s."""
    young = np.asarray(backend._young_idx)
    log_ages = np.asarray(backend._qh_log_age)
    i = int(young[len(young) // 2])
    log_qh = float(backend._get_log_qh_at(log_z, float(log_ages[i])))
    weights = np.zeros(log_ages.shape[0])
    weights[i] = 10.0 ** (-log_qh)
    return jnp.asarray(weights), jnp.asarray(log_ages)


def _mass_weights(backend, mass=1.0e6):
    """Stellar mass [Msun] on one young age bin: a physically scaled population."""
    young = np.asarray(backend._young_idx)
    weights = np.zeros(np.asarray(backend._qh_log_age).shape[0])
    weights[int(young[len(young) // 2])] = mass
    return jnp.asarray(weights)


def _lines(backend, ssp_weights, log_ages, log_z, **kwargs):
    waves, lums = backend.predict_nebular_line_luminosities(ssp_weights, log_ages, log_z, **kwargs)
    return np.asarray(waves), np.asarray(lums)


def _node_kwargs(**overrides):
    kw = {
        "neb_logU": _LOG_U_NODE,
        "neb_logZ_gas": _LOG_Z_NODE,
        "neb_xi_d": _XI_NODE,
        "neb_log_nH": _NH_NODE,
        "neb_co_ratio": _CO_NODE,
    }
    kw.update(overrides)
    return kw


def test_node_query_returns_table_per_qh(backend):
    grid = load_gutkin16_grid(mup=100.0)
    weights, log_ages = _young_weights(backend, _LOG_Z_NODE)
    _, lums_lsun = _lines(backend, weights, log_ages, _LOG_Z_NODE, **_node_kwargs())

    iz = int(np.argmin(np.abs(np.asarray(grid.logZ_axis) - _LOG_Z_NODE)))
    iu = int(np.argmin(np.abs(np.asarray(grid.logU_axis) - _LOG_U_NODE)))
    ixi = int(np.argmin(np.abs(np.asarray(grid.xi_d_axis) - _XI_NODE)))
    inh = int(np.argmin(np.abs(np.asarray(grid.log_nH_axis) - _NH_NODE)))
    ico = int(np.argmin(np.abs(np.asarray(grid.co_ratio_axis) - _CO_NODE)))
    table = np.asarray(grid.log_line_per_qh)[iz, iu, ixi, inh, ico, :]

    log_l_erg = np.log10(lums_lsun[:-1]) + np.log10(L_SUN)
    np.testing.assert_allclose(log_l_erg, table, rtol=0.0, atol=1e-10)


def test_mup_250_refused_naming_valid_set():
    with pytest.raises(Gutkin16MupError, match=r"100.*300"):
        load_gutkin16_grid(mup=250.0)
    with pytest.raises(Gutkin16MupError, match=r"100.*300"):
        Gutkin16Backend(mup=250.0)


def test_each_axis_moves_lines(backend):
    weights, log_ages = _young_weights(backend, _LOG_Z_NODE)
    base_kwargs = _node_kwargs()
    sweeps = {
        "neb_logU": ("neb_logU", -3.5, -2.0),
        "neb_logZ_gas": ("neb_logZ_gas", -3.0, -1.7),
        "neb_xi_d": ("neb_xi_d", 0.1, 0.5),
        "neb_log_nH": ("neb_log_nH", 2.0, 3.0),
        "neb_co_ratio": ("neb_co_ratio", 0.2, 1.0),
    }
    for axis, (name, lo, hi) in sweeps.items():
        _, a = _lines(backend, weights, log_ages, _LOG_Z_NODE, **{**base_kwargs, name: lo})
        _, b = _lines(backend, weights, log_ages, _LOG_Z_NODE, **{**base_kwargs, name: hi})
        change = float(np.max(np.abs(b - a) / np.abs(a)))
        assert change > 1e-3, f"{axis} does not move the lines (max rel change {change:.2e})"


def test_lya_present_and_scaled_by_fesc_lya(backend):
    weights, log_ages = _young_weights(backend, _LOG_Z_NODE)
    waves, lums = _lines(backend, weights, log_ages, _LOG_Z_NODE, **_node_kwargs())
    assert waves.shape == (19,)
    assert lums.shape == (19,)
    assert waves[-1] == pytest.approx(LYA_WAVE_AA)

    hbeta = lums[1]  # Hbeta is the second table line
    assert lums[-1] / hbeta == pytest.approx(LYA_OVER_HB_CASE_B, rel=1e-10)

    _, half = _lines(backend, weights, log_ages, _LOG_Z_NODE, **_node_kwargs(neb_fesc_lya=0.5))
    assert half[-1] == pytest.approx(0.5 * lums[-1], rel=1e-10)
    np.testing.assert_allclose(half[:-1], lums[:-1], rtol=1e-12)


def test_continuum_none_is_lines_only(ssp):
    none_backend = Gutkin16Backend(mup=100.0, continuum="none", ssp_data=ssp)
    analytic_backend = Gutkin16Backend(mup=100.0, continuum="analytic", ssp_data=ssp)
    weights, log_ages = _young_weights(analytic_backend, _LOG_Z_NODE)
    wave = jnp.asarray(ssp.ssp_wave)

    cont_none = none_backend.predict_nebular_continuum(weights, wave, log_ages, _LOG_Z_NODE)
    assert np.all(np.asarray(cont_none) == 0.0)

    sed_none = none_backend.predict_nebular_sed(
        weights, wave, log_ages, _LOG_Z_NODE, **_node_kwargs()
    )
    sed_analytic = analytic_backend.predict_nebular_sed(
        weights, wave, log_ages, _LOG_Z_NODE, **_node_kwargs()
    )
    cont_analytic = analytic_backend.predict_nebular_continuum(
        weights, wave, log_ages, _LOG_Z_NODE
    )
    np.testing.assert_allclose(
        np.asarray(sed_analytic) - np.asarray(cont_analytic),
        np.asarray(sed_none),
        rtol=1e-10,
        atol=0.0,
    )


@pytest.mark.gradient
def test_grad_finite_wrt_five_axes(backend):
    weights, log_ages = _young_weights(backend, -2.2)
    names = ("neb_logU", "neb_logZ_gas", "neb_xi_d", "neb_log_nH", "neb_co_ratio")
    x0 = jnp.array([-2.2, -2.6, 0.25, 2.4, 0.6])

    def total_lum(x):
        kwargs = dict(zip(names, (x[i] for i in range(5)), strict=True))
        _, lums = backend.predict_nebular_line_luminosities(weights, log_ages, -2.2, **kwargs)
        return jnp.sum(lums)

    grad = jax.grad(total_lum)(x0)
    assert grad.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_float32_agrees_with_float64(ssp):
    reference = Gutkin16Backend(mup=100.0, continuum="analytic", ssp_data=ssp)
    weights = _mass_weights(reference)
    log_ages = jnp.asarray(np.asarray(reference._qh_log_age))
    _, ref_lums = _lines(reference, weights, log_ages, _LOG_Z_NODE, **_node_kwargs())

    with jax.enable_x64(False):
        single = Gutkin16Backend(mup=100.0, continuum="analytic", ssp_data=ssp)
        w32 = jnp.asarray(np.asarray(weights), dtype=jnp.float32)
        a32 = jnp.asarray(np.asarray(log_ages), dtype=jnp.float32)
        _, lums32 = single.predict_nebular_line_luminosities(
            w32, a32, jnp.float32(_LOG_Z_NODE), **_node_kwargs()
        )
        assert lums32.dtype == jnp.float32
        lums32 = np.asarray(lums32, dtype=np.float64)

    np.testing.assert_allclose(lums32, ref_lums, rtol=1e-4, atol=0.0)


def test_jit_and_vmap_over_logU(backend):
    weights, log_ages = _young_weights(backend, _LOG_Z_NODE)

    def lum_at(logu):
        _, lums = backend.predict_nebular_line_luminosities(
            weights, log_ages, _LOG_Z_NODE, **_node_kwargs(neb_logU=logu)
        )
        return lums

    log_us = jnp.array([-3.5, -3.0, -2.5])
    batched = jax.jit(jax.vmap(lum_at))(log_us)
    assert batched.shape == (3, 19)
    for k, logu in enumerate(np.asarray(log_us)):
        np.testing.assert_allclose(
            np.asarray(batched[k]), np.asarray(lum_at(float(logu))), rtol=1e-10
        )


def test_both_predict_methods_name_every_optional_axis():
    required = {
        "neb_logU",
        "neb_logZ_gas",
        "neb_fesc",
        "neb_fesc_lya",
        "neb_fdust_frac",
        "neb_log_nH",
        "neb_xi_d",
        "neb_co_ratio",
    }
    for method in (
        Gutkin16Backend.predict_nebular_line_luminosities,
        Gutkin16Backend.predict_nebular_sed,
    ):
        named = set(inspect.signature(method).parameters)
        missing = required - named
        assert not missing, f"{method.__name__} does not name {sorted(missing)}"


def test_render_helper_is_the_sed_path(backend):
    """The SED is the rendered line set plus the continuum, in erg/s/Hz."""
    weights, log_ages = _young_weights(backend, _LOG_Z_NODE)
    wave = jnp.asarray(np.linspace(1000.0, 9000.0, 400))
    waves, lums = _lines(backend, weights, log_ages, _LOG_Z_NODE, **_node_kwargs())
    lines_only = render_nebular_lines(jnp.asarray(waves), jnp.asarray(lums), wave, 0.0, 100.0)
    sed = backend.predict_nebular_sed(
        weights, wave, log_ages, _LOG_Z_NODE, **_node_kwargs(line_sigma_kms=100.0)
    )
    cont = backend.predict_nebular_continuum(weights, wave, log_ages, _LOG_Z_NODE)
    np.testing.assert_allclose(
        np.asarray(sed),
        np.asarray(lines_only) * float(L_SUN) + np.asarray(cont),
        rtol=1e-10,
        atol=0.0,
    )
