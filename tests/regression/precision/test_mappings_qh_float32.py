# SPDX-License-Identifier: BSD-3-Clause
"""The mappings backend carries log Q_H end to end, so it builds in pure float32 (#2853).

``JAX_ENABLE_X64=0`` must be set before Python starts, so each build runs in a
subprocess. The child builds :class:`MappingsPhotoStellarBackend` on a bare SSP with
a synthetic stellar line grid (the on-disk ``flury2024_grids.h5`` is a stale schema
in this checkout, so the real grid cannot be loaded), then reports:

- ``log10 Q_H`` at every SSP (metallicity, age) node,
- the Q_H-driven line luminosities, in Lsun,
- ``log10 q_h_total`` for the age-weighted young population.

The parent compares float32 against float64. Before #2853 the float32 build raised
``QHTableOverflowError``: the linear Q_H table holds ~1e47 photons/s, which float32
cannot represent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.regression_bug]

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"

#: Bare-stellar SSPs under ``data/``: a solar-metallicity-grid FSPS/MILES SSP and a
#: BPASS-C3K SSP (different metallicity and age grids).
_SSPS = (
    pytest.param("data/fsps_prsc_miles_chabrier.h5", id="prsc"),
    pytest.param("data/bpss_stars_c3k_a_chabrier.h5", id="bpss"),
)

#: Bar on the float32-vs-float64 deviation of log10 Q_H, dex (absolute). The bar is
#: the one the issue allows for log quantities; the Cue log bar (#2849) is 5e-6.
_LOG_BAR_DEX = 1e-5

#: Bar on the float32-vs-float64 relative difference of the Q_H-driven line
#: luminosities (linear space).
_LINE_BAR_REL = 1e-5

_CHILD = r"""
import json
import sys
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

import tengri  # noqa: F401  (applies the JAX_ENABLE_X64 policy)
import jax
from tengri.components.nebular import mappings_photo as mp
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

ssp_path = sys.argv[1]
ssp_kw = {"mass_remaining": "dsps_fit"} if "bpss_" in ssp_path else {}
ssp = load_ssp_data(ssp_path, **ssp_kw)


def _synthetic_grid():
    zo = jnp.array([0.1, 0.5, 1.0])
    shape = (3, 4, 2, 4, 2)
    return mp.MappingsStellarGridData(
        line_wavelengths=jnp.array([1215.67, 4862.68, 5008.24, 6564.61, 3728.0]),
        zo_axis=zo,
        logU_axis=jnp.linspace(-4.0, -1.0, 4),
        log_age_yr_axis=jnp.array([6.0, 6.5, 7.0, 7.5]),
        logn_axis=jnp.array([1.0, 3.0]),
        logHB_per_logq=jnp.full(shape, -12.32),
        line_ratios=jnp.ones((*shape, 5)),
    )


labels = ["cont", "inst"]
with patch.object(mp, "_load_stellar_grid", return_value=(_synthetic_grid(), labels, 1, 0)):
    backend = mp.MappingsPhotoStellarBackend(ssp_data=ssp, ionizing_source_warning="suppress")

log_ages = jnp.asarray(np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64) + 9.0)
lgmet = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
log_z = float(lgmet[1])
weights = jnp.full(log_ages.shape, 1e9)

ages = np.asarray(log_ages)
log_tab = np.array(
    [[float(backend._get_log_qh_at(float(lz), float(la))) for la in ages] for lz in lgmet]
)
_, lum = backend.predict_nebular_line_luminosities(weights, log_ages, log_z)

from tengri.components.nebular._shared import log_q_h_total

log_qh_total = float(log_q_h_total(backend, log_ages, weights, log_z, 0.0))
young = [int(i) for i in np.asarray(backend._young_idx)]
out = {
    "x64": bool(jax.config.jax_enable_x64),
    "young_idx": young,
    "log_qh_table": log_tab.tolist(),
    "line_lum": np.asarray(lum, dtype=np.float64).tolist(),
    "log_qh_total": log_qh_total,
}
print("RESULT " + json.dumps(out))
"""


def _require_data(relpath: str) -> None:
    if not (_ROOT / relpath).is_file():
        pytest.skip(f"{relpath} is not present in this checkout")


def _run_child(ssp_path: str, x64: bool, cache_dir: Path) -> dict:
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1" if x64 else "0"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(_SRC)
    env["TENGRI_JAX_CACHE_DIR"] = str(cache_dir)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, ssp_path],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"mappings build (x64={x64}) failed:\n{proc.stderr[-3000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    report = json.loads(line[len("RESULT ") :])
    assert report["x64"] is x64, f"child ran with x64={report['x64']}, expected {x64}"
    return report


@pytest.mark.parametrize("ssp_path", _SSPS)
def test_mappings_float32_matches_float64(ssp_path, tmp_path):
    """A pure-float32 mappings build agrees with float64 on the log Q_H table,
    the Q_H-driven line luminosities and log q_h_total (#2853).

    The float32 build must be finite throughout. Log quantities are compared in dex
    (absolute), the line luminosities in linear space (relative).
    """
    _require_data(ssp_path)
    f32 = _run_child(ssp_path, x64=False, cache_dir=tmp_path / "c32")
    f64 = _run_child(ssp_path, x64=True, cache_dir=tmp_path / "c64")

    # (a) log10 Q_H table, over the entries within 20 dex of the peak.
    l32 = np.asarray(f32["log_qh_table"], dtype=np.float64)
    l64 = np.asarray(f64["log_qh_table"], dtype=np.float64)
    fin64 = np.isfinite(l64)
    assert fin64.any(), "float64 reference has no ionizing bins"
    peak = float(np.max(l64[fin64]))
    window = fin64 & (l64 > peak - 20.0)
    assert np.all(np.isfinite(l32[window])), "float32 log Q_H is non-finite on the peak window"
    dev_tab = float(np.max(np.abs(l32 - l64)[window]))
    assert dev_tab <= _LOG_BAR_DEX, f"log Q_H table differs by {dev_tab:.3e} dex (bar 1e-5)"

    # (a') log10 q_h_total.
    tot32, tot64 = float(f32["log_qh_total"]), float(f64["log_qh_total"])
    assert np.isfinite(tot32), f"float32 log q_h_total is {tot32}"
    dev_tot = abs(tot32 - tot64)
    assert dev_tot <= _LOG_BAR_DEX, f"log q_h_total differs by {dev_tot:.3e} dex (bar 1e-5)"
    # The float64 log-sum must equal the linear sum it replaces, computed in NumPy
    # from the same table at the metallicity node the child used (uniform 1e9 Msun
    # weights, so the sum is over the young bins only).
    young = f64["young_idx"]
    row = l64[1, young]
    manual = float(np.log10(np.sum(1e9 * 10.0 ** row[np.isfinite(row)])))
    assert abs(manual - tot64) <= 1e-9, f"log q_h_total {tot64} != linear sum {manual}"

    # (b) Q_H-driven line luminosities, linear space.
    lum32 = np.asarray(f32["line_lum"], dtype=np.float64)
    lum64 = np.asarray(f64["line_lum"], dtype=np.float64)
    assert np.all(np.isfinite(lum32)), f"float32 line luminosities are non-finite: {lum32}"
    assert np.any(lum32 != 0.0), "float32 line luminosities are identically zero"
    nz = lum64 != 0.0
    rel = float(np.max(np.abs(lum32 - lum64)[nz] / np.abs(lum64[nz])))
    assert rel <= _LINE_BAR_REL, f"line luminosities differ by {rel:.3e} relative (bar 1e-5)"
