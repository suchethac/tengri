# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP bolometric normalization is grid-independent and cheap (owner ruling 2026-10-09).

The GRAHSP published SED is ``l5100 * shape``, with ``l5100`` set so the intrinsic
bolometric equals the requested power. That bolometric is a sum of per-component
integrals, each taken on the component's own grid (owner ruling 2026-10-12):
the SBPL disc and Balmer continuum on their own log grids, the emission lines in
closed form, the FeII forest and template torus on their native nodes, and the
gaussian torus on its own log grid. Four properties are checked:

1. Grid independence: the published spectrum at the nodes shared by a coarse
   (200-point) and a fine (5000-point) caller grid agrees to 1e-6.
2. Declared power: on a converged grid the emitted power equals the declared
   power to 1e-4.
3. Native sum: the normalization is the sum of the BBB-side and torus-side
   integrals, and each agrees with a dense-grid evaluation of the full model.
4. Cost: one jitted call costs no more FLOPs than the main branch's.

Scope: GRAHSP only. The composable AGN templates are covered by their own tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

C_AA_PER_S = 2.99792458e18
COARSE_N = 200
FINE_N = 5000
DENSE_NM = np.geomspace(1.0, 1.0e10, 200_000)

# Compiled FLOPs of one ``compute_grahsp_sed`` call at 2000 rest-frame nodes
# (912 A to 1e8 A), jitted with ``agn_log_lbol`` traced, measured on origin/main
# f791a1cc6 on 2026-10-12. The branch before this rework evaluated the full model
# on a 20000-node grid per call and measured 8.5160e6 (10.3x main); this rework
# measures 7.7600e5 (0.94x main). Wall time, warm: 0.40 ms main, 0.38 ms branch.
MAIN_FLOPS_N2000 = 8.2402e5
COST_MARGIN = 1.10


def _shared_grids(lo_aa: float, hi_aa: float) -> tuple[np.ndarray, np.ndarray]:
    """Coarse and fine caller grids; the fine grid contains every coarse node."""
    coarse = np.geomspace(lo_aa, hi_aa, COARSE_N)
    fine = np.union1d(coarse, np.geomspace(lo_aa, hi_aa, FINE_N))
    return coarse, fine


def _emitted_power_erg_s(spectrum_lnu: np.ndarray, wave_aa: np.ndarray) -> float:
    """Frequency integral of L_nu over the caller grid [erg/s]."""
    nu = C_AA_PER_S / wave_aa
    return float(np.trapezoid(spectrum_lnu[::-1], nu[::-1]))


def _grahsp_spectrum(wave_aa: np.ndarray, log_lbol: float) -> np.ndarray:
    from tengri.components.agn.grahsp.model import compute_grahsp_sed

    return np.asarray(
        compute_grahsp_sed(
            jnp.asarray(wave_aa),
            agn_log_lbol=log_lbol,
            agn_grahsp_ebv=0.0,
            agn_grahsp_ebv_agn=0.0,
        )
    )


def _check_grid_independence(spectrum_fn, lo_aa: float, hi_aa: float) -> None:
    coarse, fine = _shared_grids(lo_aa, hi_aa)
    at_coarse = spectrum_fn(coarse)
    at_fine = spectrum_fn(fine)[np.searchsorted(fine, coarse)]
    scale = np.max(np.abs(at_coarse))
    assert scale > 0.0
    assert np.max(np.abs(at_coarse - at_fine)) / scale <= 1e-6


def _unit_params(**overrides):
    """Unit-l5100 GRAHSP parameters, as ``compute_grahsp_sed`` builds them."""
    from tengri.components.agn.grahsp.model import GRAHSPParams

    base = dict(
        l5100=1.0,
        uvslope=0.0,
        plslope=-1.7,
        plbendloc_nm=100.0,
        plbendwidth=1.0,
        cutoff_nm=10000.0,
        a_lines=1.0,
        a_feii=1.0,
        linewidth_kms=5000.0,
        agn_type=1,
        fcov=0.4,
        si=0.0,
        cool_lam_um=17.0,
        cool_width=0.45,
        hot_lam_um=2.0,
        hot_width=0.5,
        hot_fcov=1.0,
        ebv=0.0,
        ebv_agn=0.0,
        a_bc=0.0,
        tor_temp=0.0,
        tor_cutoff_um=1.2,
        torus_model="gaussian",
        feii_template="bruhweiler2008",
        disc_model=None,
        disc_m="8.0",
        disc_a="0",
        disc_mdot="0.3",
    )
    return GRAHSPParams(**{**base, **overrides})


def _compiled_flops(fn, *args) -> float:
    cost = jax.jit(fn).lower(*args).compile().cost_analysis()
    if isinstance(cost, list):
        cost = cost[0] if cost else {}
    return float(dict(cost or {}).get("flops", 0.0))


def test_grahsp_normalization_is_grid_independent():
    _check_grid_independence(lambda w: _grahsp_spectrum(w, 12.0), 912.0, 1e8)


def test_grahsp_emits_declared_bolometric_power_on_converged_grid():
    # lumBolBBB is the integral above the Lyman limit (912 A): the declared
    # power is 10**agn_log_lbol L_sun, emitted above 912 A on a grid that
    # reaches the converged tail (1e11 A).
    wave = np.geomspace(912.0, 1e11, 100_000)
    declared = 10.0**12.0 * 3.828e33
    lnu = _grahsp_spectrum(wave, 12.0)
    above_lyman = wave >= 912.0
    power = _emitted_power_erg_s(np.where(above_lyman, lnu, 0.0), wave)
    assert power == pytest.approx(declared, rel=1e-4)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"torus_model": "mn12", "tor_temp": 0.5, "si": 0.2}],
    ids=["gaussian_torus", "mn12_torus"],
)
def test_normalization_is_sum_of_native_integrals(overrides):
    """The normalization is the BBB-side plus torus-side native integrals.

    Each side is checked against a dense-grid evaluation of the full model, so
    the per-component closed forms and quadratures cannot drift apart from the
    SED the model publishes.
    """
    from tengri.components.agn.grahsp.model import evaluate_grahsp_agn
    from tengri.components.agn.grahsp.normalization import (
        bbb_side_unit_power,
        torus_side_unit_power,
        unit_bolometric_power,
    )
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()
    params = _unit_params(**overrides)
    bbb_side = float(bbb_side_unit_power(params, templates))
    torus_side = float(torus_side_unit_power(params, templates))
    total = float(unit_bolometric_power(params, templates))

    assert total == pytest.approx(bbb_side + torus_side, rel=1e-12)
    dense = evaluate_grahsp_agn(jnp.asarray(DENSE_NM), params, templates)
    assert bbb_side == pytest.approx(float(dense.l_bol_bbb), rel=1e-5)
    assert torus_side == pytest.approx(float(dense.l_bol_torus), rel=1e-5)


def test_normalization_cost_not_above_main():
    """One jitted call costs no more compiled FLOPs than main, plus a margin.

    FLOP counts come from the compiled executable, so this holds on a loaded
    machine too. A per-call evaluation on a 20000-node grid is about 10x main
    and fails it.
    """
    from tengri.components.agn.grahsp.model import compute_grahsp_sed

    wave = jnp.asarray(np.geomspace(912.0, 1e8, 2000))
    flops = _compiled_flops(
        lambda log_lbol: compute_grahsp_sed(wave, agn_log_lbol=log_lbol),
        jnp.asarray(12.0),
    )
    assert flops > 0.0
    assert flops <= COST_MARGIN * MAIN_FLOPS_N2000
