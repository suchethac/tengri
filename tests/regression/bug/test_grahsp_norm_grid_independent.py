# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP bolometric normalization is grid-independent and cheap (owner ruling 2026-10-09).

The GRAHSP published SED is ``l5100 * shape``, with ``l5100`` set so the BBB-side
bolometric (paper lumBolBBB: disc, broad and narrow lines, FeII, Balmer continuum
above the Lyman limit) equals the requested power. That unit power is a sum of
per-component integrals, each taken on the component's own grid rather than the
caller's: the SBPL disc on its own log grid, the Netzer disc on its native nodes,
the Gaussian lines in closed form, the FeII forest on its template nodes and the
Balmer continuum on its own log grid. The torus is not part of lumBolBBB.

Checks:

1. Grid independence: the published spectrum at the nodes shared by a coarse and
   a fine caller grid agrees to 1e-6.
2. Declared power: on a converged caller grid the BBB-side power emitted above
   912 A equals the declared power to 1e-4.
3. Native sum: each component's unit power agrees with a dense-grid evaluation of
   the same component to 1e-5, and the BBB-side total is their sum.
4. Cost: one jitted call costs no more compiled FLOPs than the main branch's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

C_AA_PER_S = 2.99792458e18
LYMAN_NM = 91.2
COARSE_N = 200
FINE_N = 5000

# Compiled FLOPs of one ``compute_grahsp_sed`` call at 2000 rest-frame nodes
# (912 A to 1e8 A), jitted with ``agn_log_lbol`` traced, keyed by the jax major.minor
# that compiled it. The 0.11 value was measured on origin/main (2f891d207, #2834) on
# 2026-10-12; the 0.12 value on jax 0.12.0 (arm; CI linux reports 909490). XLA 0.12 changed
# the compiled HLO for this graph (+8.8 %), with no tengri change, so one number cannot serve
# both. The margin below is not widened: it still catches a real regression on either version.
# Wall time, warm: about 0.5 ms on main.
MAIN_FLOPS_N2000_BY_JAX = {"0.11": 8.2402e5, "0.12": 9.0749e5}
COST_MARGIN = 1.10


def _shared_grids(lo_aa: float, hi_aa: float) -> tuple[np.ndarray, np.ndarray]:
    """Coarse and fine caller grids; the fine grid contains every coarse node."""
    coarse = np.geomspace(lo_aa, hi_aa, COARSE_N)
    fine = np.union1d(coarse, np.geomspace(lo_aa, hi_aa, FINE_N))
    return coarse, fine


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
        a_feii=5.0,
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
        a_bc=0.5,
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


def _dense_log_integral(wave_nm: np.ndarray, f: np.ndarray) -> float:
    """Integral of ``f`` d(lambda) on a log grid (trapezoid in ln lambda)."""
    return float(np.trapezoid(np.asarray(f) * wave_nm, np.log(wave_nm)))


def test_grahsp_normalization_is_grid_independent():
    coarse, fine = _shared_grids(912.0, 1e8)
    at_coarse = _grahsp_spectrum(coarse, 12.0)
    at_fine = _grahsp_spectrum(fine, 12.0)[np.searchsorted(fine, coarse)]
    scale = np.max(np.abs(at_coarse))
    assert scale > 0.0
    assert np.max(np.abs(at_coarse - at_fine)) / scale <= 1e-6


@pytest.mark.parametrize("disc_model", [None, "netzer"], ids=["sbpl_disc", "netzer_disc"])
def test_grahsp_emits_declared_bolometric_power_on_converged_grid(disc_model):
    """The BBB-side power on a converged caller grid equals the declared power.

    The unit power is set by the per-component integrals; the check is that the
    caller-grid integral of the published BBB side converges to it.
    """
    from tengri.components.agn.grahsp.model import evaluate_grahsp_agn
    from tengri.components.agn.grahsp.normalization import bbb_side_unit_power
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()
    params = _unit_params(disc_model=disc_model)
    declared = 10.0**12.0 * 3.828e33
    l5100 = declared / float(bbb_side_unit_power(params, templates))
    published = evaluate_grahsp_agn(
        jnp.asarray(np.geomspace(LYMAN_NM, 1.0e6, 400_000)),
        _unit_params(disc_model=disc_model, l5100=l5100),
        templates,
    )
    assert float(published.l_bol_bbb) == pytest.approx(declared, rel=1e-4)


def test_unit_power_matches_dense_per_component_integrals():
    """Each component's unit power agrees with a dense-grid evaluation of it."""
    from tengri.components.agn.grahsp.balmer import balmer_continuum
    from tengri.components.agn.grahsp.bbb import floor_disc_xray, sbpl_bbb
    from tengri.components.agn.grahsp.disc import netzer_disc, select_disc_model
    from tengri.components.agn.grahsp.lines import feii_forest, gaussian_lines
    from tengri.components.agn.grahsp.normalization import (
        balmer_unit_power,
        bbb_side_unit_power,
        disc_unit_power,
        feii_unit_power,
        line_unit_power,
    )
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()
    for disc_model in (None, "netzer"):
        params = _unit_params(disc_model=disc_model)
        if disc_model == "netzer":
            grid = np.geomspace(LYMAN_NM, float(np.max(templates.disc_wave_nm)), 200_000)
            idx = select_disc_model(
                templates.disc_m,
                templates.disc_a,
                templates.disc_mdot,
                m=params.disc_m,
                a=params.disc_a,
                mdot=params.disc_mdot,
            )
            disc_dense = netzer_disc(
                wave_nm=jnp.asarray(grid),
                l5100=1.0,
                disc_wave_nm=templates.disc_wave_nm,
                disc_lumin_model=templates.disc_lumin[idx],
            )
        else:
            grid = np.geomspace(LYMAN_NM, 1.0e10, 200_000)
            disc_dense = floor_disc_xray(
                jnp.asarray(grid),
                sbpl_bbb(
                    wave_nm=jnp.asarray(grid),
                    l5100=1.0,
                    uvslope=params.uvslope,
                    plslope=params.plslope,
                    plbendloc_nm=params.plbendloc_nm,
                    plbendwidth=params.plbendwidth,
                    cutoff_nm=params.cutoff_nm,
                ),
            )
        assert float(disc_unit_power(params, templates)) == pytest.approx(
            _dense_log_integral(grid, disc_dense), rel=1e-5
        )

    params = _unit_params()
    lines_grid = np.geomspace(LYMAN_NM, 1.0e5, 800_000)
    broad, narrow = gaussian_lines(
        wave_nm=jnp.asarray(lines_grid),
        line_wave_nm=templates.line_wave_nm,
        line_broad=templates.line_broad,
        line_narrow_sy2=templates.line_narrow_sy2,
        line_narrow_liner=templates.line_narrow_liner,
        l5100=1.0,
        a_lines=params.a_lines,
        linewidth_kms=params.linewidth_kms,
        agn_type=params.agn_type,
    )
    assert float(line_unit_power(params, templates)) == pytest.approx(
        _dense_log_integral(lines_grid, broad + narrow), rel=1e-5
    )
    feii = feii_forest(
        wave_nm=jnp.asarray(lines_grid),
        template_wave_nm=templates.feii_wave_nm,
        template_lumin=templates.feii_lumin,
        l5100=1.0,
        a_lines=params.a_lines,
        a_feii=params.a_feii,
        agn_type=params.agn_type,
    )
    assert float(feii_unit_power(params, templates)) == pytest.approx(
        _dense_log_integral(lines_grid, feii), rel=1e-5
    )

    balmer_grid = np.geomspace(LYMAN_NM, 1.0e6, 400_000)
    balmer = balmer_continuum(
        wave_nm=jnp.asarray(balmer_grid),
        l5100=1.0,
        a_bc=params.a_bc,
        linewidth_kms=params.linewidth_kms,
    )
    assert float(balmer_unit_power(params)) == pytest.approx(
        _dense_log_integral(balmer_grid, balmer), rel=1e-5
    )
    total = float(bbb_side_unit_power(params, templates))
    assert total == pytest.approx(
        float(disc_unit_power(params, templates))
        + float(line_unit_power(params, templates))
        + float(feii_unit_power(params, templates))
        + float(balmer_unit_power(params)),
        rel=1e-12,
    )


def test_normalization_cost_not_above_main():
    """One jitted call costs no more compiled FLOPs than main, plus a margin.

    FLOP counts come from the compiled executable, so this holds on a loaded
    machine too.
    """
    from tengri.components.agn.grahsp.model import compute_grahsp_sed

    wave = jnp.asarray(np.geomspace(912.0, 1e8, 2000))
    flops = _compiled_flops(
        lambda log_lbol: compute_grahsp_sed(wave, agn_log_lbol=log_lbol),
        jnp.asarray(12.0),
    )
    assert flops > 0.0
    series = ".".join(jax.__version__.split(".")[:2])
    if series not in MAIN_FLOPS_N2000_BY_JAX:
        pytest.fail(
            f"no compiled-FLOP baseline recorded for jax {jax.__version__} (series {series}); "
            f"measured {flops:.0f} FLOPs. Re-capture the baseline on origin/main under this "
            "jax and add it to MAIN_FLOPS_N2000_BY_JAX."
        )
    assert flops <= COST_MARGIN * MAIN_FLOPS_N2000_BY_JAX[series]
