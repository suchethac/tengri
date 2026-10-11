# SPDX-License-Identifier: BSD-3-Clause
"""Photoionized nebular builds must run in pure float32 with log-domain Q_H (#2849).

``JAX_ENABLE_X64=0`` must be set before Python starts, so each build runs in a
subprocess. The child loads the bare SSP in float32, builds one photoionized
backend, and reports its ionizing table and forward outputs as JSON. The parent
rebuilds the same model in float64 and compares.

The ionizing-spectrum fit and the mappings photoionization table must not form the
~1e46 photon rate in linear float32 (the table then holds ``inf`` and the Cue build
raises ``CueWNESSPError``), and must stay bit-identical in float64 (#2849).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.regression_bug]

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_SSP = "data/fsps_prsc_miles_chabrier.h5"

_CHILD = r"""
import json
import sys

import numpy as np

import tengri  # noqa: F401  (applies the JAX_ENABLE_X64 policy)
import jax
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.nebular.ionizing_spectrum import precompute_ionizing_params_table
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry

kind = sys.argv[1]
ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
table = precompute_ionizing_params_table(
    np.asarray(ssp.ssp_wave),
    np.asarray(ssp.ssp_flux),
    np.asarray(ssp.ssp_lgmet),
    ssp_log_age_yr=np.asarray(ssp.ssp_lg_age_gyr) + 9.0,
)
neb = {"type": "cue", "all_params": Fixed(DEFAULT)}
if kind == "cloudy":
    neb = {"type": "cloudy"}
elif kind == "cb19":
    neb = {"type": "cb19"}
elif kind == "mappings":
    neb = {"type": "mappings", "ionizing_source_warning": "suppress"}
model = SEDModel.build(
    ssp_data=ssp,
    observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
    sfh={
        "type": "delayed",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Uniform(9.0, 11.0),
        "tau_gyr": 1.0,
        "age_gyr": 5.0,
    },
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_diff": Uniform(0.0, 1.5),
        "tau_bc": 0.0,
    },
    neb=neb,
    redshift=Fixed(0.1),
    approx=None,
)
params = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}
phot = np.asarray(model.predict_photometry(params), dtype=np.float64)
pred = model.predict(params)
rest = np.asarray(pred.rest_sed(), dtype=np.float64)
out = {
    "x64": bool(jax.config.jax_enable_x64),
    "logq_young_max": float(np.max(table["logqion_table"][:, :])),
    "logq_table": np.asarray(table["logqion_table"], dtype=np.float64).tolist(),
    "phot": phot.tolist(),
    "rest_sed_finite": bool(np.all(np.isfinite(rest))),
}
print("RESULT " + json.dumps(out))
"""


def _run_child(kind: str, x64: bool, cache_dir: Path) -> dict:
    """Build ``kind`` in a child process with x64 on or off and return its report."""
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1" if x64 else "0"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(_SRC)
    env["TENGRI_JAX_CACHE_DIR"] = str(cache_dir)
    env.pop("TENGRI_ALLOW_WNE_CUE", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, kind],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"{kind} build (x64={x64}) failed:\n{proc.stderr[-3000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    report = json.loads(line[len("RESULT ") :])
    assert report["x64"] is x64, f"child ran with x64={report['x64']}, expected {x64}"
    return report


def _require_data(relpath: str) -> None:
    """Skip only when the data file is absent from the checkout."""
    if not (_ROOT / relpath).is_file():
        pytest.skip(f"{relpath} is not present in this checkout")


def _ionizing_mask(wave) -> np.ndarray:
    """Samples on the hydrogen-ionizing side of the Lyman limit, ``wave`` [Angstrom]."""
    from tengri.components.lyc import LYMAN_LIMIT_AA

    return np.asarray(wave, dtype=np.float64) < LYMAN_LIMIT_AA


def test_cue_float32_build_matches_float64(tmp_path):
    """A pure-float32 Cue build reproduces the float64 build (#2849).

    Before the fix the float32 log Q_H table is ``inf`` for every young bin and
    the build raises ``CueWNESSPError``; the subprocess then fails here.
    """
    _require_data(_SSP)
    f32 = _run_child("cue", x64=False, cache_dir=tmp_path / "c32")
    f64 = _run_child("cue", x64=True, cache_dir=tmp_path / "c64")

    q32 = np.asarray(f32["logq_table"], dtype=np.float64)
    q64 = np.asarray(f64["logq_table"], dtype=np.float64)
    assert np.all(np.isfinite(q32)), "float32 log Q_H table holds non-finite entries"
    young = q64 > -90.0
    assert young.any(), "no ionizing bins in the float64 reference"
    dev = float(np.max(np.abs(q32 - q64)[young]))
    # Bar derivation. The fit integrates the integrand divided by a power of two
    # (exact), so the float32 quadrature runs on O(1) terms. Measured against
    # float64 over the 765 young bins of fsps_prsc_miles_chabrier: max 1.06e-7 dex
    # (scratch run of this test, #2849). Float32 eps is 1.19e-7, and the 121 panels
    # below 912 A bound the rounding at 121 * 1.19e-7 relative = 6.3e-6 dex in the
    # worst case, so the measured value sits well inside the bound. The bar is
    # 1e-6 dex, about ten times the measured value. A float64 path is not used
    # (float64 device buffers fail on float64-less backends, #2276, #1880).
    assert dev <= 1e-6, f"float32 log Q_H differs from float64 by {dev:.3e} dex (bar 1e-6)"

    p32 = np.asarray(f32["phot"], dtype=np.float64)
    p64 = np.asarray(f64["phot"], dtype=np.float64)
    assert np.all(np.isfinite(p32)) and np.all(p32 > 0), f"float32 photometry: {p32}"
    rel = float(np.max(np.abs(p32 - p64) / np.abs(p64)))
    assert rel <= 1e-4, f"float32 photometry relative difference {rel:.3e} (bar 1e-4)"
    assert f32["rest_sed_finite"], "float32 rest-frame SED is not finite"


@pytest.mark.parametrize("kind", ["cloudy", "cb19"])
def test_photoionized_backend_float32_build(kind, tmp_path):
    """Cloudy and CB19 already build log-domain; they must stay float32-finite.

    Skipped only when the backend's grid file is absent from the checkout.
    """
    grid = {"cloudy": "data/cloudy_grid_mist.h5", "cb19": "data/cb19_templates.h5"}[kind]
    _require_data(grid)
    _require_data(_SSP)
    f32 = _run_child(kind, x64=False, cache_dir=tmp_path / "c32")
    f64 = _run_child(kind, x64=True, cache_dir=tmp_path / "c64")
    p32 = np.asarray(f32["phot"], dtype=np.float64)
    p64 = np.asarray(f64["phot"], dtype=np.float64)
    assert np.all(np.isfinite(p32)), f"{kind} float32 photometry non-finite: {p32}"
    rel = float(np.max(np.abs(p32 - p64) / np.abs(p64)))
    assert rel <= 1e-4, f"{kind} float32 photometry relative difference {rel:.3e}"


@pytest.mark.parametrize("dummy", [None])
def test_mappings_qh_table_float32_matches_float64(dummy):
    """The mappings Q_H table, built in log space, agrees between float32 and float64.

    The end-to-end mappings build is not exercised here: the bundled
    ``data/flury2024_grids.h5`` is 51% NaN in ``logHB_per_logq`` (#2082), a
    pre-existing grid defect that fails float64 as well. This test calls
    ``_precompute_qh`` directly on the real SSP instead, so the photoionization
    Q_H table itself is checked.
    """
    from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    _require_data(_SSP)

    def _table(x64: bool):
        with jax.enable_x64(x64):
            ssp = load_ssp_data(str(_ROOT / _SSP))
            backend = MappingsPhotoStellarBackend.__new__(MappingsPhotoStellarBackend)
            backend._qh_table = None
            backend._precompute_qh(ssp)
            ion_flux = np.asarray(ssp.ssp_flux, dtype=np.float64)[
                ..., _ionizing_mask(ssp.ssp_wave)
            ]
            flushed = np.max(np.abs(ion_flux), axis=-1) == 0.0
            return (
                np.asarray(backend._qh_table, dtype=np.float64),
                float(backend._log_qh_scale),
                flushed,
            )

    t32, s32, flushed32 = _table(False)
    t64, s64, _ = _table(True)

    assert np.all(np.isfinite(t32)), "float32 mappings Q_H table holds non-finite entries"
    live64 = t64 > 0
    assert live64.any(), "float64 reference has no ionizing bins"

    # Entries float32 zeroes are exactly the (age, Z) rows whose float32 SSP has no
    # ionizing flux at all: load_ssp_data casts the ionizing side to float32, which
    # flushes values near 1e-40 L_sun/Hz/Msun to zero before any Q_H arithmetic.
    # Their float64 peaks are 1e-40 to 1e-39, Q_H is 18 to 19 dex, 28 to 29 dex below
    # the table peak, and an honest zero there (#2849). Any other zero is a defect.
    zeroed = live64 & (t32 == 0.0)
    assert np.array_equal(zeroed, live64 & flushed32), (
        "float32 zeroes an entry whose SSP ionizing flux is not flushed"
    )
    # t64 is absolute in float64 (no scale), so compare to the table peak.
    rel_dex = np.log10(t64[zeroed]) - np.log10(np.max(t64))
    assert np.all(rel_dex <= -25.0), (
        "float32 zeroes an entry that is not negligible relative to the peak"
    )

    # Agreement in log space where Q_H is within 20 dex of the peak. Measured max
    # 5.6e-6 dex there. Entries 20 to 40 dex below the peak reach 2.4e-4 dex
    # (float32 rounding of a tiny integral), at a relative size of 1e-20 or less.
    window = (
        live64 & (t32 > 0) & (np.log10(np.where(live64, t64, 1.0)) - np.log10(np.max(t64)) > -20.0)
    )
    assert window.any(), "no entries within 20 dex of the peak"
    lq32 = np.log10(t32[window]) + s32
    lq64 = np.log10(t64[window]) + s64
    dev = float(np.max(np.abs(lq32 - lq64)))
    assert dev <= 2e-5, f"mappings float32 log Q_H differs from float64 by {dev:.3e} dex"


@pytest.mark.parametrize("kind", ["mappings", "cloudy", "cb19"])
def test_log_qh_table_float32_matches_float64(kind):
    """The log10 Q_H table of each photoionized backend agrees between float32 and float64.

    The three backends build their table with the same log-domain core
    (``compute_qh_log10``). Checked on the log table itself, before the
    ``pow10`` that the backends apply. Entries are finite in float32 except on
    the rows whose float32 SSP has no ionizing flux (see the mappings test); those
    rows are the only non-finite entries, and they must be exactly that set.
    Cloudy and CB19 are gated on their grid files, as the other tests are.
    """
    from tengri.components.nebular.cloudy_cb19 import _compute_log_qh_grid as cb19_grid
    from tengri.components.nebular.cloudy_grid import _compute_log_qh_grid as cloudy_grid
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    grid_file = {
        "mappings": None,
        "cloudy": "data/cloudy_grid_mist.h5",
        "cb19": "data/cb19_templates.h5",
    }[kind]
    fn = cb19_grid if kind == "cb19" else cloudy_grid
    if grid_file is not None:
        _require_data(grid_file)
    _require_data(_SSP)

    def _log_table(x64: bool):
        with jax.enable_x64(x64):
            ssp = load_ssp_data(str(_ROOT / _SSP))
            lq = np.asarray(
                fn(jnp.asarray(ssp.ssp_wave), jnp.asarray(ssp.ssp_flux)), dtype=np.float64
            )
            ion = np.asarray(ssp.ssp_flux, dtype=np.float64)[..., _ionizing_mask(ssp.ssp_wave)]
            return lq, np.max(np.abs(ion), axis=-1) == 0.0

    l32, flushed = _log_table(False)
    l64, _ = _log_table(True)

    finite64 = np.isfinite(l64)
    assert finite64.any(), "float64 reference has no ionizing bins"
    nonfinite32 = ~np.isfinite(l32) & finite64
    assert np.array_equal(nonfinite32, flushed & finite64), (
        f"{kind}: float32 non-finite log Q_H outside the flushed SSP rows"
    )

    peak = float(np.max(l64[finite64]))
    window = finite64 & np.isfinite(l32) & (l64 > peak - 20.0)
    assert window.any(), "no entries within 20 dex of the peak"
    dev = float(np.max(np.abs(l32[window] - l64[window])))
    assert dev <= 2e-5, f"{kind}: float32 log Q_H differs from float64 by {dev:.3e} dex"
    # Deep entries (20 to 40 dex below the peak): the integrand's terms there fall
    # into float32's subnormal range before the power-of-two scale is applied,
    # and the error is 2.4e-4 dex at a relative size of 1e-20 or less (#2849).
    both = finite64 & np.isfinite(l32)
    dev_all = float(np.max(np.abs(l32[both] - l64[both])))
    assert dev_all <= 3e-4, (
        f"{kind}: float32 log Q_H differs by {dev_all:.3e} dex over all finite entries"
    )


def test_ionspec_cache_key_separates_computation_dtype(tmp_path, monkeypatch):
    """A float32 and a float64 ionspec table never share a cache entry (#2849).

    The same float64 SSP is tabulated once with x64 on and once with x64 off, in one
    process, with the in-memory and disk caches enabled. The two builds must write
    two entries and return two different tables: the x64-off edge quadrature runs in
    float32 and differs at the 1e-7 to 1e-6 dex level. A key that ignored the
    computation dtype would return the first table for both.
    """
    import tengri.components.nebular.ionizing_spectrum as ion
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    _require_data(_SSP)
    monkeypatch.setenv("TENGRI_JAX_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ion, "_IONSPEC_TABLE_CACHE", {})

    with jax.enable_x64(True):
        ssp = load_ssp_data(str(_ROOT / _SSP))
        n_met, n_age = 2, 8  # youngest ages only: the fits run on these bins
        wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
        flux = np.asarray(ssp.ssp_flux, dtype=np.float64)[:n_met, :n_age, :]
        lgmet = np.asarray(ssp.ssp_lgmet, dtype=np.float64)[:n_met]
        log_age = np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64)[:n_age] + 9.0

    def _build(x64: bool) -> np.ndarray:
        with jax.enable_x64(x64):
            table = ion.precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=log_age)
            return np.asarray(table["logqion_table"], dtype=np.float64)

    lq64 = _build(True)
    lq32 = _build(False)

    assert len(ion._IONSPEC_TABLE_CACHE) == 2, "x64 on and off share one in-memory entry"
    disk_entries = list((tmp_path / "ionspec_tables").glob("*.npz"))
    assert len(disk_entries) == 2, f"expected two disk entries, found {len(disk_entries)}"

    young = lq64 > -90.0
    assert young.any(), "no ionizing bins in the float64 table"
    dev = float(np.max(np.abs(lq32 - lq64)[young]))
    assert 1e-8 < dev <= 1e-6, f"float32 vs float64 ionspec log Q_H differ by {dev:.3e} dex"


#: Generator for the float64 bit-identity reference (#2849). Run as ``python -c``
#: with argv ``SRC OUT ROOT``: it writes every ionizing and Q_H table and the Cue
#: forward output to ``OUT``. The committed fixture was produced by this exact
#: text against the unfixed tree (origin/main, b8d41e2b8).
_BITWISE_CHILD = r"""
import sys
SRC, OUT, ROOT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC)
import numpy as np
import jax
import tengri  # noqa: F401
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.nebular.ionizing_spectrum import precompute_ionizing_params_table
from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend
from tengri.components.nebular.cloudy_grid import CloudyGridBackend
from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry

ssp = load_ssp_data(ROOT + "/data/fsps_prsc_miles_chabrier.h5")
res = {"x64": np.asarray(bool(jax.config.jax_enable_x64))}
wave = np.asarray(ssp.ssp_wave)
flux = np.asarray(ssp.ssp_flux)
lgmet = np.asarray(ssp.ssp_lgmet)
log_age = np.asarray(ssp.ssp_lg_age_gyr) + 9.0
tab = precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=log_age)
for key in ("ionspec_table", "logqion_table", "seglum_table"):
    res["ion_" + key] = np.asarray(tab[key])
backends = (
    ("mappings", MappingsPhotoStellarBackend),
    ("cloudy", CloudyGridBackend),
    ("cb19", CB19Backend),
)
for name, cls in backends:
    b = cls.__new__(cls)
    b._qh_table = None
    b._max_neb_log_age = 8.0
    b._log_qh_scale = 0.0
    b._precompute_qh(ssp)
    res[name + "_qh"] = np.asarray(b._qh_table, dtype=np.float64)
    res[name + "_scale"] = np.asarray(float(getattr(b, "_log_qh_scale", 0.0)))
model = SEDModel.build(
    ssp_data=ssp,
    observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
    sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0),
         "tau_gyr": 1.0, "age_gyr": 5.0},
    dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT),
                      "tau_diff": Uniform(0.0, 1.5), "tau_bc": 0.0},
    neb={"type": "cue", "all_params": Fixed(DEFAULT)},
    redshift=Fixed(0.1),
    approx=None,
)
params = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}
res["cue_phot"] = np.asarray(model.predict_photometry(params), dtype=np.float64)
pred = model.predict(params)
res["cue_rest_sed"] = np.asarray(pred.rest_sed(), dtype=np.float64)
np.savez(OUT, **res)
print("WROTE", OUT, sorted(res))
"""


def test_float64_tables_bit_identical_to_unfixed_build(tmp_path):
    """With x64 on, the ionizing and Q_H tables and the Cue output are bit-identical (#2849).

    The reference is ``fixtures/nebular_qh_float64_unfixed.json``, written by
    ``_BITWISE_CHILD`` against the unfixed tree. Exact float64 agreement holds
    because the power-of-two scale is exact, and the log is taken in the same
    library (NumPy in the fit, the same jitted XLA log in the Q_H routines).
    """
    _require_data(_SSP)
    with open(
        Path(__file__).resolve().parent / "fixtures" / "nebular_qh_float64_unfixed.json"
    ) as fh:
        ref = {k: np.asarray(v) for k, v in json.load(fh).items() if not k.startswith("_")}
    out = tmp_path / "new.npz"
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    env["JAX_PLATFORMS"] = "cpu"
    env["TENGRI_JAX_CACHE_DIR"] = str(tmp_path / "jax_cache")
    env.pop("TENGRI_ALLOW_WNE_CUE", None)
    proc = subprocess.run(
        [sys.executable, "-c", _BITWISE_CHILD, str(_SRC), str(out), str(_ROOT)],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert proc.returncode == 0, f"bit-identity child failed:\n{proc.stderr[-3000:]}"
    new = np.load(out)
    assert bool(new["x64"]) and bool(ref["x64"]), "both builds must run with x64 on"
    for key in ref:
        if key == "x64":
            continue
        assert np.array_equal(new[key], ref[key]), f"{key} differs from the unfixed float64 build"
