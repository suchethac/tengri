# SPDX-License-Identifier: BSD-3-Clause
"""float32 AGN panchromatic l_tir stays finite and matches float64 (#3401).

``agn_panchromatic`` with a float64-built model evaluated under
``jax.enable_x64(False)`` (the float32 arm; params are drawn in float64 first)
gave ``l_tir = nan`` at ``agn_log_lbol = 12`` while float64 gave 9.15e14.

The cause is a dtype gate that keys on the wavelength grid: the radio component
forwards the float32-safe ``log_L_ir`` companion only when ``wave.dtype`` is
float32 (``components/radio/component.py``). The grid was captured as float64 at
build time, so the gate stayed closed and the linear ``L_ir = 10**log_L_ir erg/s``
(~3e48 here) was used. That overflows float32 to ``inf``, and the free-free term
then produces NaN. The gate must follow the active float precision, not the grid.

Every gradient assertion checks finite AND nonzero (check_gradient_assertions).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from .conftest import build_model

pytestmark = pytest.mark.regression_bug

LOG_LBOL = 12.0
REL_TOL = 1e-4


@pytest.fixture(scope="module")
def agn_params():
    """Draw parameters once in float64, before any float32 arm runs."""
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
    model = build_model(ssp, "float64")
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    params["agn_log_lbol"] = LOG_LBOL
    return model, params


def _l_tir_f64(model, params):
    return float(np.asarray(model.predict(params).l_tir, dtype=np.float64))


def _l_tir_f32(model, params):
    with jax.enable_x64(False):
        return float(np.asarray(model.predict(params).l_tir, dtype=np.float64))


def _grad_f32(model, params):
    def l_tir_of_lbol(lbol):
        p = dict(params)
        p["agn_log_lbol"] = lbol
        return model.predict_properties(p, names=("l_tir",))["l_tir"]

    with jax.enable_x64(False):
        return float(jax.grad(l_tir_of_lbol)(jnp.asarray(LOG_LBOL, dtype=jnp.float32)))


def test_l_tir_float32_is_finite_and_nonzero(agn_params):
    model, params = agn_params
    l32 = _l_tir_f32(model, params)
    assert np.isfinite(l32), f"float32 l_tir non-finite: {l32}"
    assert l32 != 0.0, "float32 l_tir flushed to zero"


def test_l_tir_float32_matches_float64(agn_params):
    model, params = agn_params
    l64 = _l_tir_f64(model, params)
    l32 = _l_tir_f32(model, params)
    assert_allclose(l32, l64, rtol=REL_TOL)


def test_l_tir_gradient_wrt_agn_log_lbol_is_finite_and_nonzero(agn_params):
    model, params = agn_params
    g = _grad_f32(model, params)
    assert np.isfinite(g), f"d l_tir / d agn_log_lbol non-finite: {g}"
    assert g != 0.0, "d l_tir / d agn_log_lbol is zero"
