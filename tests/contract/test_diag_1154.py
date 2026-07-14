# SPDX-License-Identifier: BSD-3-Clause
"""TEMPORARY diagnostic for #1154 — delete once the cause is known.

The fast (WavePrecomp) nebular photometry agrees with the exact path to 7.4e-4 on
macOS/arm64 and is off by 2.09e-01 on linux/x86. Identical committed data (git
blob hashes match), identical PRNG (threefry is bit-deterministic), and it XPASSes
under `-n 2` locally, so it is not data, not x64, and not test-order.

This dumps the intermediates FROM the failing platform, because guessing from the
passing one has run out of road.
"""

from __future__ import annotations

import importlib.util
import warnings

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.contract

_SPEC = importlib.util.spec_from_file_location(
    "_fnw", str(__import__("pathlib").Path(__file__).with_name("test_fast_nebular_wiring.py"))
)


def test_diag_1154_dump():
    t = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(t)
    t._require()

    print("\n" + "=" * 72)
    print(f"x64 enabled          : {jax.config.jax_enable_x64}")
    print(f"jax backend/platform : {jax.default_backend()} {jax.devices()}")
    print(f"numpy               : {np.__version__}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_exact = t._build(t._CUE)
        m_fast = t._build(t._CUE)
        m_fast.enable_fast_nebular(t._LW, n_grid=16)

    g = m_fast._nebular_grid_table
    print(f"grid axis_names      : {getattr(g, 'axis_names', None)}")
    print(f"grid axis_kinds      : {getattr(g, 'axis_kinds', None)}")
    for nm, ax in zip(getattr(g, "axis_names", ()), getattr(g, "axes", ()) or ()):
        a = np.asarray(ax)
        print(f"  axis {nm:16s}: n={a.size:3d}  [{a.min():.6f}, {a.max():.6f}]")
        print(f"      nodes: {np.array2string(a, precision=6, max_line_width=200)}")
    lp = getattr(g, "log_line_per_qh", None)
    if lp is not None:
        lp = np.asarray(lp)
        print(
            f"log_line_per_qh      : shape={lp.shape} finite={np.isfinite(lp).all()} "
            f"min={np.nanmin(lp):.4f} max={np.nanmax(lp):.4f}"
        )
    lph = getattr(g, "log_phot_per_qh", None)
    print(
        f"log_phot_per_qh      : {'present' if lph is not None else 'MISSING (fast photometry has no table!)'}"
    )
    if lph is not None:
        lph = np.asarray(lph)
        print(
            f"  shape={lph.shape} finite={np.isfinite(lph).all()} "
            f"min={np.nanmin(lph):.4f} max={np.nanmax(lph):.4f}"
        )

    print(f"\n{'i':>2} {'logU':>8} {'max rel':>11}  worst band   fast/exact at worst")
    worst = 0.0
    for i in range(8):
        p = dict(m_exact.spec.sample(jax.random.PRNGKey(600 + i)))
        pe = np.asarray(m_exact.predict_photometry(p))
        pf = np.asarray(m_fast.predict_photometry(p))
        rel = np.abs(pf - pe) / (np.abs(pe) + 1e-40)
        j = int(np.argmax(rel))
        worst = max(worst, rel.max())
        lu = float(p.get("neb_logU", p.get("neb_cue_logU", np.nan)))
        print(f"{i:2d} {lu:8.4f} {rel.max():11.4e}  {t._BANDS[j]:11s}  {pf[j]:.6e} / {pe[j]:.6e}")
    print(f"\nWORST = {worst:.4e}   (tolerance 3.0e-2)")
    print("=" * 72)
