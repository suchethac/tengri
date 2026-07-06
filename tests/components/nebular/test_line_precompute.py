# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the FeaturePrecomp line table reconstructs Cue lines without Cue (#950).

Cue line luminosities are exactly linear in Q_H, and the per-Q_H factor is
SFH-shape-independent (only stellar metallicity + fixed gas conditions matter).
So ``L_line = nion * interp_met(line_per_qh)`` reproduces the exact
``predict_line_fluxes`` — replacing the ~3 ms Cue neural forward with a
metallicity interpolation + a scalar multiply.

This pins the accuracy contract: reconstruction from a 40-point metallicity
table matches the exact forward to < 1e-3 on the strong DESI lines across
arbitrary (SFH shape, metallicity) — three orders of magnitude below the
measurement floor. The stellar metallicity enters Cue *nonlinearly*, so the
test also guards that the grid is dense enough (a coarse grid fails at
1-60 %, verified during the #950 design).

Data-gated: needs the bare-stellar FSPS SSP (Cue requires it); skips in CI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Observation, Photometry, SEDModel, recipes
from tengri.components.nebular.line_precompute import (
    precompute_line_per_qh,
    reconstruct_line_lums,
)
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.contract

_LINES = ["Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717", "OIII_4959", "NII_6548"]
_MET_LO, _MET_HI = -1.8, 0.4


def _model(ssp_data_fsps):
    import warnings

    dummy = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]), line_fluxes=dummy)
    kw = recipes.star_forming_photometry()
    kw.pop("approx", None)
    kw.pop("redshift", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel.build(ssp_data=ssp_data_fsps, observation=obs, redshift=Fixed(0.15), **kw)
    return m, dummy.wavelengths


def _nion(m, p):
    st = m.predict_state(p)
    n = st.derived["nion"]
    return float(np.asarray(jnp.sum(n) if jnp.ndim(n) else n))


def test_line_table_reconstructs_exact_across_sfh_and_metallicity(ssp_data_fsps):
    m, lw = _model(ssp_data_fsps)
    table = precompute_line_per_qh(m, lw, met_lo=_MET_LO, met_hi=_MET_HI, n_met=40)
    assert table.line_per_qh.shape == (40, len(_LINES))

    base = dict(m.spec.sample(jax.random.PRNGKey(0)))
    key = jax.random.PRNGKey(11)
    worst = 0.0
    for _ in range(8):
        k = jax.random.split(key, 6)
        key = k[0]
        p = dict(base)
        p["met_logzsol"] = jnp.asarray(
            float(jax.random.uniform(k[1], minval=_MET_LO + 0.1, maxval=_MET_HI - 0.1))
        )
        p["sfh_dpl_age_gyr"] = jnp.asarray(float(jax.random.uniform(k[2], minval=1.0, maxval=8.0)))
        p["sfh_dpl_tau_gyr"] = jnp.asarray(float(jax.random.uniform(k[3], minval=0.5, maxval=5.0)))
        p["sfh_dpl_log_total_mass"] = jnp.asarray(
            float(jax.random.uniform(k[4], minval=9.0, maxval=11.0))
        )

        exact = np.asarray(m.predict_line_fluxes(p, target_wavelengths=lw))
        lut = np.asarray(reconstruct_line_lums(_nion(m, p), p["met_logzsol"], table))
        strong = np.abs(exact) > 1e-3 * np.max(np.abs(exact))
        rel = np.max(np.abs(lut - exact)[strong] / np.maximum(np.abs(exact)[strong], 1e-40))
        worst = max(worst, rel)
    assert worst < 1e-3, f"line-table reconstruction off by {worst:.2e} (>1e-3) on strong lines"


def test_reconstruct_is_jit_and_linear_in_nion(ssp_data_fsps):
    """The hot path is jit'able and exactly linear in nion (= Q_H)."""
    m, lw = _model(ssp_data_fsps)
    table = precompute_line_per_qh(m, lw, met_lo=_MET_LO, met_hi=_MET_HI, n_met=20)
    fn = jax.jit(lambda nion, mz: reconstruct_line_lums(nion, mz, table))
    a = np.asarray(fn(1.0e56, -0.3))
    b = np.asarray(fn(2.0e56, -0.3))
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-12)  # exact linearity in Q_H
    assert np.all(np.isfinite(a))
