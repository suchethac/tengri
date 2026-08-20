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
    try:
        ssp = tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")

    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust={
            "law": "power_law",
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.3,
            "tau_bc": 0.3,
            "emission": {"type": "schreiber2016", "*": tengri.FIXED},
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    pred = m.predict_rest_sed(p)
    sed = np.asarray(pred.sed)
    assert np.isfinite(sed).all() and sed.max() > 0.0

    # Energy balance against the model's own published L_absorbed — the
    # canonical, LyC-masked quantity the dust IR is normalized to (#922/#961).
    #
    # The absorbed side used to be re-derived here as a band proxy
    # (no-dust minus dusty, integrated over 912 A - 3 um). A proxy measures the
    # *test's* choice of band, not the model's energy budget: it silently omits
    # absorption redward of 3 um and counts the dust IR that leaks blueward into
    # the window, and it drifted to a 1.110 false failure the moment the fiducial
    # galaxy changed (the tsnorm registry defaults, #1034) — with the model's
    # true balance still exact. Assert the physics, not a band.
    state = m.predict_state(p)
    wave = np.asarray(state.wave)
    sed_ir = np.asarray(state.derived["sed_dust_ir"])
    l_absorbed = float(np.asarray(state.derived["L_absorbed"]))

    c_aa = 2.99792458e18

    def bolo(wave_aa: np.ndarray, l_nu: np.ndarray, wmin: float = 0.0, wmax: float = np.inf):
        mask = (wave_aa >= wmin) & (wave_aa <= wmax)
        nu = c_aa / wave_aa[mask]
        order = np.argsort(nu)
        return float(np.trapezoid(l_nu[mask][order], nu[order]))

    l_ir = bolo(wave, sed_ir)
    ratio = l_ir / l_absorbed
    assert 0.99 < ratio < 1.01, (
        f"Schreiber2016 breaks energy balance: emitted L_IR / L_absorbed = {ratio:.4f}"
    )

    # ...and the emission must live in the IR, not be smeared across the grid
    # (the #1005 native-grid regression put it partly outside 8-1000 um).
    in_band = bolo(wave, sed_ir, 8.0e4, 1.0e7) / l_ir
    assert in_band > 0.95, f"only {in_band:.1%} of the dust IR falls in 8-1000 um"
