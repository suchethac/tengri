# SPDX-License-Identifier: BSD-3-Clause
"""TEMPORARY diagnostic for #1154 — DELETE once the cause is known.

Fast (WavePrecomp) nebular photometry agrees with exact to 7.4e-4 on macOS/arm64
and is off by 2.09e-01 on linux/x86. Ruled out: data (git blob hashes match),
x64 (disabling gives NaN not 21%), test order (XPASSes under -n 2 locally).

The intermediates must come from the failing platform. This test ALWAYS fails, so
pytest emits its captured stdout into the CI log (a passing test's stdout is
discarded). It is self-contained — no sibling-module import — so xdist cannot
change its behaviour.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1", "wise_w2"]
_LINES = ("Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717")
_LD = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
_LW = _LD.wavelengths


def _build():
    import pathlib

    if not pathlib.Path(_BARE).is_file() or not pathlib.Path("data/cue_weights.npz").is_file():
        pytest.skip("SSP or cue_weights missing")
    ssp = load_ssp_data(_BARE)
    obs = Observation(photometry=Photometry.from_names(_BANDS), line_fluxes=_LD)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
                "tau_diff": Fixed(0.25),
                "tau_bc": Fixed(0.4),
            },
            neb={"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)},
            redshift=Fixed(0.15),
            approx=WavePrecomp(),
        )


def test_diag_1154_always_fails_to_dump():
    m_exact = _build()
    m_fast = _build()
    m_fast.enable_fast_nebular(_LW, n_grid=16)

    lines = [
        "",
        "=" * 72,
        f"platform : {jax.default_backend()}  x64={jax.config.jax_enable_x64}  numpy={np.__version__}",
    ]
    g = m_fast._nebular_grid_table
    lines.append(f"axis_names : {g.axis_names}")
    lines.append(f"axis_kinds : {getattr(g, 'axis_kinds', None)}")
    for nm, ax in zip(g.axis_names, g.axes):
        a = np.asarray(ax)
        lines.append(
            f"  {nm:14s} n={a.size:3d}  {np.array2string(a, precision=6, max_line_width=300)}"
        )
    lp = np.asarray(g.log_line_per_qh)
    lines.append(
        f"log_line_per_qh : {lp.shape} finite={np.isfinite(lp).all()} min={np.nanmin(lp):.5f} max={np.nanmax(lp):.5f} sum={np.nansum(lp):.5f}"
    )
    lph = getattr(g, "log_phot_per_qh", None)
    if lph is not None:
        lph = np.asarray(lph)
        lines.append(
            f"log_phot_per_qh : {lph.shape} finite={np.isfinite(lph).all()} min={np.nanmin(lph):.5f} max={np.nanmax(lph):.5f} sum={np.nansum(lph):.5f}"
        )

    worst = 0.0
    lines.append(f"{'i':>2} {'logU':>8} {'metZ':>8} {'maxrel':>11}  band")
    for i in range(8):
        p = dict(m_exact.spec.sample(jax.random.PRNGKey(600 + i)))
        pe = np.asarray(m_exact.predict_photometry(p))
        pf = np.asarray(m_fast.predict_photometry(p))
        rel = np.abs(pf - pe) / (np.abs(pe) + 1e-40)
        j = int(np.argmax(rel))
        worst = max(worst, rel.max())
        lu = float(p.get("neb_logU", p.get("neb_cue_logU", np.nan)))
        mz = float(p.get("met_logzsol", np.nan))
        lines.append(f"{i:2d} {lu:8.4f} {mz:8.4f} {rel.max():11.4e}  {_BANDS[j]}")
    lines.append(f"WORST = {worst:.5e}")
    lines.append("=" * 72)
    pytest.fail("\n".join(lines))
