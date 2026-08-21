# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the FeaturePrecomp line table reconstructs Cue lines without Cue (#950).

Cue line luminosities are exactly linear in Q_H (scaling the SFH amplitude scales
every line by the same factor — pinned to rtol=1e-12 below), and the per-Q_H factor
is SFH-shape-independent **to ~0.2 %**. So ``L_line = nion * interp_met(line_per_qh)``
reproduces the exact ``predict_line_fluxes`` — replacing the ~3 ms Cue neural
forward with a metallicity interpolation + a scalar multiply.

.. note::
   The SFH-shape independence is **approximate, not exact** (#1018). The ionizing
   spectrum's shape is the Q_H-weighted combination of the young age bins, so a
   different SFH shape re-weights that mix and shifts the per-Q_H forbidden-line
   emissivity slightly. Before #1018 the shape was picked from a single
   ``argmax`` age bin, making it piecewise-constant in the SFH — so this test
   passed at < 1e-3 as an *artifact of that bug*, not because the physics is exact.
   Q_H **linearity** (amplitude scaling) remains exact, since it does not change
   the age-mix.

This pins the accuracy contract: reconstruction from a 40-point metallicity
table matches the exact forward to < 5e-3 on the strong DESI lines across
arbitrary (SFH shape, metallicity) — still well below the measurement floor. The
stellar metallicity enters Cue *nonlinearly*, so the test also guards that the
grid is dense enough (a coarse grid fails at 1-60 %, verified during #950).

Data-gated: needs the bare-stellar FSPS SSP (Cue requires it); skips in CI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, Observation, Photometry, SEDModel, recipes
from tengri.components.nebular.line_precompute import (
    precompute_line_per_qh,
    reconstruct_line_lums,
)
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.contract

_LINES = ["Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717", "OIII_4959", "NII_6548"]
_MET_LO, _MET_HI = -1.8, 0.4


def _model(ssp_data_fsps, redshift=0.15, neb=None):
    import warnings

    dummy = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]), line_fluxes=dummy)
    kw = recipes.star_forming_photometry()
    kw.pop("approx", None)
    kw.pop("redshift", None)
    if neb is not None:
        kw["neb"] = neb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel.build(ssp_data=ssp_data_fsps, observation=obs, redshift=Fixed(redshift), **kw)
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

        # intrinsic (redden=False): the line-per-Q_H table is un-reddened
        exact = np.asarray(m.predict_line_fluxes(p, target_wavelengths=lw, redden=False))
        lut = np.asarray(reconstruct_line_lums(_nion(m, p), p["met_logzsol"], 0.15, table))
        strong = np.abs(exact) > 1e-3 * np.max(np.abs(exact))
        rel = np.max(np.abs(lut - exact)[strong] / np.maximum(np.abs(exact)[strong], 1e-40))
        worst = max(worst, rel)
    # 5e-3, not 1e-3: the per-Q_H factor carries a genuine ~0.2 % SFH-shape
    # dependence now that the ionizing shape is the Q_H-weighted age mix (#1018).
    assert worst < 5e-3, f"line-table reconstruction off by {worst:.2e} (>5e-3) on strong lines"


def test_table_is_redshift_independent(ssp_data_fsps):
    """Regression: a table built at one redshift reconstructs correctly at another.

    The table stores line LUMINOSITY (distance-independent); the cosmology is
    applied at reconstruct with the EVALUATION redshift. Building at z=0.1 and
    evaluating at z=0.5 must match the exact forward at z=0.5 — the earlier
    design stored observed flux and was ~38x wrong across this redshift gap.
    """
    m_lo, lw = _model(ssp_data_fsps, redshift=0.1)
    m_hi, _ = _model(ssp_data_fsps, redshift=0.5)
    table = precompute_line_per_qh(m_lo, lw, met_lo=_MET_LO, met_hi=_MET_HI, n_met=40)

    p = dict(m_lo.spec.sample(jax.random.PRNGKey(3)))
    p["met_logzsol"] = jnp.asarray(-0.4)
    # nion is a stellar (distance-independent) quantity — same physical galaxy
    exact_hi = np.asarray(
        m_hi.predict_line_fluxes(
            {**p, "redshift": jnp.asarray(0.5)}, target_wavelengths=lw, redden=False
        )
    )
    lut_hi = np.asarray(
        reconstruct_line_lums(
            _nion(m_hi, {**p, "redshift": jnp.asarray(0.5)}), p["met_logzsol"], 0.5, table
        )
    )
    strong = np.abs(exact_hi) > 1e-3 * np.max(np.abs(exact_hi))
    rel = np.max(np.abs(lut_hi - exact_hi)[strong] / np.maximum(np.abs(exact_hi)[strong], 1e-40))
    # 5e-3: see the module note — SFH-shape independence is ~0.2 %, not exact (#1018).
    assert rel < 5e-3, f"cross-redshift reconstruction off by {rel:.2e} (table built at z=0.1)"


def test_free_ionization_is_rejected_at_build(ssp_data_fsps):
    """Regression: building a table with a FREE ionization param must raise.

    A free neb_logU / neb_logZ_gas / neb_fesc changes line ratios, so the
    single-metallicity-axis table would be silently wrong away from its baked
    reference value. The precondition must be a guard, not a docstring.
    """
    from tengri import FIXED, Uniform

    m, lw = _model(ssp_data_fsps, neb={"type": "cue", "logU": Uniform(-3.5, -1.5), "all_params": FIXED})
    assert "neb_logU" in set(m.spec.free_params)
    with pytest.raises(ValueError, match="requires FIXED nebular ionization"):
        precompute_line_per_qh(m, lw, n_met=5)


def test_reconstruct_is_jit_and_linear_in_nion(ssp_data_fsps):
    """The hot path is jit'able and exactly linear in nion (= Q_H)."""
    m, lw = _model(ssp_data_fsps)
    table = precompute_line_per_qh(m, lw, met_lo=_MET_LO, met_hi=_MET_HI, n_met=20)
    fn = jax.jit(lambda nion, mz: reconstruct_line_lums(nion, mz, 0.15, table))
    a = np.asarray(fn(1.0e56, -0.3))
    b = np.asarray(fn(2.0e56, -0.3))
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-12)  # exact linearity in Q_H
    assert np.all(np.isfinite(a))
