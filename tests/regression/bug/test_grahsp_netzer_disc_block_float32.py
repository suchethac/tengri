# SPDX-License-Identifier: BSD-3-Clause
"""The composable GRAHSP Netzer disc block agrees between pure float32 and float64.

``grahsp_netzer_disc_block`` reads the (log M_BH, spin, log Mdot) template library through
``netzer_unit_fn`` -> ``netzer_disc_interp`` (8-corner multilinear interpolation, then the
X-ray floor) and scales the unit spectrum by ``agn_log_lbol`` in log10, evaluated at the
runner's float32 reference luminosity (the template shape does not depend on L_bol). Each
float32 value is compared with the float64 value computed in the same test, at an interior
point and at points next to an axis edge, with the tolerance of the disc float32 inventory
(``tests/regression/precision/test_agn_disc_float32_inventory.py``): relative difference below
1e-3 on points above 1e-6 of the float64 peak.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn._lbol_reference import _AGN_LBOL_REF
from tengri.components.agn.blocks.grahsp_blocks import grahsp_netzer_disc_block
from tengri.components.agn.grahsp import load_grahsp_templates

pytestmark = pytest.mark.regression_bug

#: Same tolerance and live-point floor as the disc float32 inventory.
FLOAT32_RTOL = 1.0e-3
PEAK_FLOOR = 1.0e-6

_WAVE_NM = np.geomspace(10.0, 1.0e5, 400)

#: (log M_BH, spin, log Mdot): interior, then next to the lower and upper axis edges.
_POINTS = {
    "interior": (8.0, 0.0, -0.5228787452803376),
    "low_edge": (7.01, 0.0, -1.49),
    "high_edge": (9.99, 0.0, -0.01),
}


def _block(templates, point, dtype):
    log_mbh, spin, log_mdot = point
    wave_aa = jnp.asarray(_WAVE_NM * 10.0, dtype=dtype)
    out = grahsp_netzer_disc_block(
        wave_aa,
        jnp.asarray(_AGN_LBOL_REF, dtype=dtype),
        agn_grahsp_netzer_log_mbh=jnp.asarray(log_mbh, dtype=dtype),
        agn_grahsp_netzer_spin=jnp.asarray(spin, dtype=dtype),
        agn_grahsp_netzer_log_mdot=jnp.asarray(log_mdot, dtype=dtype),
        templates=templates,
    )
    return np.asarray(out)


@pytest.mark.parametrize("name", sorted(_POINTS))
def test_netzer_disc_block_matches_float64_in_float32(name):
    templates = load_grahsp_templates()
    with jax.enable_x64(True):
        ref = _block(templates, _POINTS[name], jnp.float64)
    with jax.enable_x64(False):
        f32 = _block(templates, _POINTS[name], jnp.float32)
    assert f32.dtype == np.float32
    assert np.all(np.isfinite(f32)), f"{name}: non-finite float32 disc"
    live = np.abs(ref) > PEAK_FLOOR * np.abs(ref).max()
    assert live.sum() > 10, f"{name}: too few live points to compare"
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    assert rel.max() < FLOAT32_RTOL, f"{name}: max relative error {rel.max():.2e}"
