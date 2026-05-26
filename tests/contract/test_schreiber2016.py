# SPDX-License-Identifier: BSD-3-Clause
"""Issue #351: Schreiber+2016 dust IR template family.

Acceptance from the issue:

1. ``'schreiber2016' in [e['name'] for e in tengri.list_dust_emission_models()]``
2. Energy-balance residual ``|L_IR_emit - L_absorbed| / L_absorbed < few %``
3. Builds end-to-end through ``SEDModel.build(..., dust={'emission': ...})``

The kernel itself was already implemented in
``components/dust/emission.py``; this test pins the user-visible menu
entry and end-to-end build path so the model is *discoverable*
(addressed the regression that motivated the issue).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri

pytestmark = pytest.mark.contract


def test_schreiber2016_listed():
    """Acceptance #1 from issue: discoverable from the registry menu."""
    names = [e["name"] for e in tengri.list_dust_emission_models()]
    assert "schreiber2016" in names, f"schreiber2016 missing from menu: {names}"


def test_schreiber2016_builds_and_balances():
    """Acceptance #2+#3: end-to-end build + energy-balance."""
    ssp = tengri.load_ssp()

    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.3,
            "tau_bc": 0.3,
            "emission": {"type": "schreiber2016", "*": tengri.FIXED},
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    sed = np.asarray(m.predict_rest_sed(p).sed)
    assert np.isfinite(sed).all() and sed.max() > 0.0

    # Energy-balance against a no-dust baseline.
    m_no = tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
        redshift=tengri.Fixed(0.05),
    )
    p0 = dict(m_no.spec.sample(jax.random.PRNGKey(0)))
    sed_i = np.asarray(m_no.predict_rest_sed(p0).sed)
    wave = np.asarray(m.ssp_data.ssp_wave)

    c_aa = 2.998e18

    def L(L_nu: np.ndarray, wmin: float, wmax: float) -> float:
        mask = (wave >= wmin) & (wave <= wmax)
        w = wave[mask]
        nu = c_aa / w
        order = np.argsort(nu)
        return float(np.trapezoid(L_nu[mask][order], nu[order]))

    ratio = L(sed, 8.0e4, 1.0e7) / L(sed_i - sed, 912.0, 3.0e4)
    assert 0.90 < ratio < 1.10, f"Schreiber2016 energy balance off: ratio={ratio:.3f}"
