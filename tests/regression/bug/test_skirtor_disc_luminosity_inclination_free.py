# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR published L_agn_disc is the angle-integrated accretion power (owner 2026-10-03, #2678).

``L_agn_disc`` is the bolometric integral of the emitted disc, which is independent of
the line of sight. ``agn_cos_inc`` enters only the polar-dust absorption, so changing the
inclination must leave ``L_agn_disc`` unchanged (#2851).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SKIRTOR_CANDIDATES = [
    _DATA_DIR / "skirtor_templates_v3.h5",
    _DATA_DIR / "skirtor_templates_v2.h5",
]
_SKIRTOR_PATH = next((p for p in _SKIRTOR_CANDIDATES if p.is_file()), None)

_PARAMS = {
    "log_lbol": jnp.array(10.0),
    "tau_skirtor": jnp.array(7.0),
    "p_skirtor": jnp.array(1.0),
    "q_skirtor": jnp.array(1.0),
    "oa_skirtor": jnp.array(40.0),
    "radius_ratio": jnp.array(20.0),
    "cos_inc": jnp.array(0.5),
    "torus_frac": jnp.array(0.5),
    "polar_ebv": jnp.array(0.1),
    "polar_T": jnp.array(100.0),
    "polar_beta": jnp.array(1.5),
    "delta": jnp.array(0.0),
}


@pytest.mark.skipif(_SKIRTOR_PATH is None, reason="SKIRTOR template data not available")
def test_skirtor_l_agn_disc_independent_of_inclination():
    wave = jnp.logspace(3.5, 7, 300)
    component = SKIRTORTorus(config=SKIRTORTorusConfig(grid_path=str(_SKIRTOR_PATH)))
    object.__setattr__(component, "data", component.load(wave))
    if component.data is None:
        pytest.skip("Could not load SKIRTOR templates")

    published = {}
    for cos_inc in (0.1, 0.9):
        params = dict(_PARAMS, cos_inc=jnp.array(cos_inc))
        _, published[cos_inc] = component.predict(params, jnp.zeros_like(wave), wave)

    # L_agn_disc is the normalization's disc power: torus_frac * L_bol, L_bol = 10**log_lbol L_sun.
    expected = 0.5 * 10.0**10.0 * L_SUN
    for cos_inc in (0.1, 0.9):
        np.testing.assert_allclose(float(published[cos_inc]["L_agn_disc"]), expected, rtol=1e-6)
    # The polar-dust absorption does depend on inclination; the test must not be vacuous.
    assert float(published[0.1]["L_agn_polar_dust"]) != float(published[0.9]["L_agn_polar_dust"])
