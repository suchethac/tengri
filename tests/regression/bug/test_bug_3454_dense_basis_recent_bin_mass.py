# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #3454: dense_basis honors ``log_sfr_inst`` and conserves M*.

The recent (last 10 Myr) bins of the dense_basis history are overridden to
``10**log_sfr_inst``, but the old bins were scaled to ``M* - M_recent`` before
that override. The raw composed history therefore integrated to more than M*,
and the component mass pin then lowered the recent level silently by up to
1.35 dex.

The recent mass is now taken first, ``M_recent = 10**log_sfr_inst x width``,
and the old bins are scaled to the remainder, so the composed curve integrates
to M* exactly and the pin is a no-op. A declaration whose recent mass exceeds
M* cannot hold (more mass formed in the last 10 Myr than in total), so the
recent level is capped at ``M* / width``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.stellar.sfh.dense_basis import dense_basis
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug

Z = 1.0
N_RES = 1000
N_DRAWS = 100
RECENT_LO_YR = 1e6
RECENT_HI_YR = 1e7


def _age_yr(z: float) -> float:
    return float(age_at_z(z)) * 1e9


def _composed(log_total_mass, log_sfr_inst, tx, age_yr):
    """The family evaluated on its own GP grid, so the raw integral is exact."""
    nodes = age_yr * np.linspace(0.0, 1.0, N_RES)
    kwargs = {f"tx_frac_{i}": float(v) for i, v in enumerate(tx)}
    sfr = np.asarray(
        dense_basis(
            nodes,
            log_total_mass=log_total_mass,
            log_sfr_inst=log_sfr_inst,
            age_universe_yr=age_yr,
            **kwargs,
        ),
        dtype=float,
    )
    return nodes, sfr


def _draws(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        log_m = rng.uniform(8.0, 12.0)
        log_sfr = rng.uniform(-2.0, 3.0)
        tx = np.sort(rng.uniform(0.05, 0.95, size=3))
        out.append((log_m, log_sfr, tx))
    return out


@pytest.mark.parametrize("draw", _draws(N_DRAWS, seed=11))
def test_raw_composed_mass_is_declared_mass(draw):
    """The composed history integrates to M* to 1e-6, for every draw."""
    log_m, log_sfr, tx = draw
    age_yr = _age_yr(Z)
    _, sfr = _composed(log_m, log_sfr, tx, age_yr)
    dt_yr = age_yr / N_RES
    raw = np.sum(sfr) * dt_yr
    assert raw == pytest.approx(10.0**log_m, rel=1e-6)


def test_recent_level_is_capped_when_its_mass_exceeds_the_total():
    """An inconsistent declaration caps the recent level at M*/width, mass still M*."""
    age_yr = _age_yr(Z)
    log_m, log_sfr = 8.0, 2.64
    _, sfr = _composed(log_m, log_sfr, (0.2, 0.5, 0.8), age_yr)
    dt_yr = age_yr / N_RES
    recent = sfr[np.linspace(0.0, age_yr, N_RES) <= 1e7]
    assert recent.size >= 1
    assert np.allclose(recent, 10.0**log_m / (recent.size * dt_yr), rtol=1e-9)
    assert np.sum(sfr) * dt_yr == pytest.approx(10.0**log_m, rel=1e-6)


def _build_model(ssp):
    return SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "dense_basis", "all_params": FREE},
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(Z),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The family conserves M* (test_raw_composed_mass_is_declared_mass), but the "
        "published history is then rescaled by the trapezoid on the model age grid, "
        "which starts at 1 Myr and has a 13.8 Myr spacing at the recent end, so the "
        "published 1-10 Myr level is still moved by that factor (17 of 88 consistent "
        "draws beyond 0.01 dex, max 0.17 dex, real SSP, z=1). Open decision: #3454."
    ),
)
def test_published_recent_sfr_matches_log_sfr_inst_for_consistent_draws(synthetic_ssp_wide):
    """For every consistent prior draw, the published 1-10 Myr SFR is 10**log_sfr_inst."""
    model = _build_model(synthetic_ssp_wide)
    decls = model.declared_parameters()
    key = jax.random.PRNGKey(11)
    checked = 0
    for _ in range(N_DRAWS):
        params = {}
        for d in decls:
            key, sub = jax.random.split(key)
            params[d.name] = float(np.asarray(d.prior.sample(sub)))
        log_m = next(v for k, v in params.items() if k.endswith("log_total_mass"))
        log_sfr = next(v for k, v in params.items() if k.endswith("log_sfr_inst"))
        if 10.0 ** (log_sfr + 7.0) > 10.0**log_m:
            continue
        out = model.predict_sfh(params)
        t_yr = np.asarray(out["t_gyr"], dtype=float).reshape(-1) * 1e9
        sfr = np.asarray(out["sfr_full"], dtype=float).reshape(-1)
        window = (t_yr >= RECENT_LO_YR) & (t_yr <= RECENT_HI_YR)
        offset = np.log10(sfr[window].mean()) - log_sfr
        assert abs(offset) <= 0.01, f"offset {offset:.4f} dex at log_sfr_inst={log_sfr:.3f}"
        checked += 1
    assert checked > 0
