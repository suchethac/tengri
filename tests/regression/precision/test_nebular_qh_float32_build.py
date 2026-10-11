# SPDX-License-Identifier: BSD-3-Clause
"""Photoionized nebular builds must run in pure float32 with log-domain Q_H (#2849).

``JAX_ENABLE_X64=0`` must be set before Python starts, so each build runs in a
subprocess. The child loads the bare SSP in float32, builds one photoionized
backend, and reports its ionizing table and forward outputs as JSON. The parent
rebuilds the same model in float64 and compares.

The ionizing-spectrum fit must not form the ~1e46 photon rate in linear float32 (the
table then holds ``inf`` and the Cue build raises ``CueWNESSPError``), and must stay
bit-identical in float64 (#2849). The mappings backend still refuses a float32 build
with ``QHTableOverflowError``, tracked in #2853.
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
# BPASS ships no surviving-mass table yet (#2751/#2800), so it opts in to the DSPS fit
# explicitly, as every in-repo BPASS load does; the other grids use their own table.
ssp_kw = {"mass_remaining": "dsps_fit"} if "bpss_" in sys.argv[2] else {}
ssp = load_ssp_data(sys.argv[2], **ssp_kw)
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


def _run_child(kind: str, x64: bool, cache_dir: Path, ssp: str = _SSP) -> dict:
    """Build ``kind`` on the SSP file ``ssp`` in a child process, x64 on or off."""
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1" if x64 else "0"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(_SRC)
    env["TENGRI_JAX_CACHE_DIR"] = str(cache_dir)
    env.pop("TENGRI_ALLOW_WNE_CUE", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, kind, ssp],
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


#: Bare-stellar SSPs under ``data/`` that the Cue float32 build is checked on. The
#: wNE MIST grid is not bare, and ``ssp_prsc_bc03_chabrier.h5`` fails Cue's own
#: wNE guard in float64 too (max log Q_H 62.4 dex), so neither is here.
_CUE_SSPS = (
    pytest.param("data/fsps_prsc_miles_chabrier.h5", id="prsc"),
    pytest.param("data/bpss_stars_c3k_a_chabrier.h5", id="bpss"),
)

#: Bar on the float32-vs-float64 log Q_H deviation over the young bins (#2849).
#: Measured max deviation, dex, over the young bins: 1.23e-7 (fsps_prsc_miles),
#: 6.9e-8 (bpss), 8.6e-8 (bc03, a bare grid not in ``data/``), 1.87e-6
#: (fsps_mist_c3k_a, a wNE grid this test does not load). The bar is 5e-6, about
#: 2.7 times the largest of the four.
_CUE_QH_BAR = 5e-6


@pytest.mark.parametrize("ssp_path", _CUE_SSPS)
def test_cue_float32_build_matches_float64(ssp_path, tmp_path):
    """A pure-float32 Cue build reproduces the float64 build on each bare SSP (#2849).

    The float32 log Q_H table holds finite values and agrees with float64 to
    ``_CUE_QH_BAR`` dex. The photometry agrees to 1e-4 relative. A float32 build
    that overflows Q_H raises ``CueWNESSPError`` and fails here.
    """
    _require_data(ssp_path)
    f32 = _run_child("cue", x64=False, cache_dir=tmp_path / "c32", ssp=ssp_path)
    f64 = _run_child("cue", x64=True, cache_dir=tmp_path / "c64", ssp=ssp_path)

    q32 = np.asarray(f32["logq_table"], dtype=np.float64)
    q64 = np.asarray(f64["logq_table"], dtype=np.float64)
    assert np.all(np.isfinite(q32)), "float32 log Q_H table holds non-finite entries"
    assert np.any(q32 != 0.0), "float32 log Q_H table is identically zero"
    young = q64 > -90.0
    assert young.any(), "no ionizing bins in the float64 reference"
    dev = float(np.max(np.abs(q32 - q64)[young]))
    assert dev <= _CUE_QH_BAR, f"float32 log Q_H differs from float64 by {dev:.3e} dex"

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
    assert np.any(p32 != 0.0), f"{kind} float32 photometry is identically zero"
    rel = float(np.max(np.abs(p32 - p64) / np.abs(p64)))
    assert rel <= 1e-4, f"{kind} float32 photometry relative difference {rel:.3e}"


@pytest.mark.parametrize("kind", ["cloudy", "cb19"])
def test_log_qh_table_float32_matches_float64(kind):
    """The log10 Q_H table of each photoionized backend agrees between float32 and float64.

    Cloudy and CB19 build their table in log space (``_compute_log_qh_grid``),
    checked on the log table itself, before the ``pow10`` that the backends apply.
    Entries are finite in float32 except on the rows whose float32 SSP has no
    ionizing flux; those rows are the only non-finite entries, and they must be
    exactly that set. Both are gated on their grid files.
    """
    from tengri.components.nebular.cloudy_cb19 import _compute_log_qh_grid as cb19_grid
    from tengri.components.nebular.cloudy_grid import _compute_log_qh_grid as cloudy_grid
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    grid_file = {"cloudy": "data/cloudy_grid_mist.h5", "cb19": "data/cb19_templates.h5"}[kind]
    fn = cb19_grid if kind == "cb19" else cloudy_grid
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
    # into float32's subnormal range, and the error is 2.4e-4 dex at a relative
    # size of 1e-20 or less.
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


def test_nonfinite_beyond_lyman_edge_does_not_defeat_float32_scaling():
    """A NaN or inf beyond 912 A leaves the float32 Q_H unchanged (#2849).

    The power-of-two exponent comes from the finite ionizing pixels only. A NaN or
    inf at 5000 A and above is outside the Q_H integral, so it must not set the
    scale. The float32 log Q_H of the row, with the bad pixel inserted, must equal
    its float64 value to ``_CUE_QH_BAR``.
    """
    import tengri.components.nebular.ionizing_spectrum as ion
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    _require_data(_SSP)
    with jax.enable_x64(True):
        ssp = load_ssp_data(str(_ROOT / _SSP))
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    # Metallicity index 0, age index 3: the largest young log Q_H of this SSP
    # (46.80 dex in float64).
    row = np.asarray(ssp.ssp_flux, dtype=np.float64)[0, 3]
    beyond = int(np.flatnonzero(wave > 5000.0)[0])

    def _gas_logqion(x64: bool, flux: np.ndarray) -> float:
        with jax.enable_x64(x64):
            return float(ion.fit_ionizing_spectrum(wave, flux)["gas_logqion"])

    ref64 = _gas_logqion(True, row)
    assert 46.0 < ref64 < 47.5, f"reference log Q_H {ref64:.3f} is outside the expected range"
    for bad in (np.nan, np.inf):
        flux = row.copy()
        flux[beyond] = bad
        lq32 = _gas_logqion(False, flux)
        assert np.isfinite(lq32), f"float32 log Q_H is {lq32} with {bad} beyond 5000 A"
        assert lq32 != 0.0, f"float32 log Q_H is exactly zero with {bad} beyond 5000 A"
        dev = abs(lq32 - ref64)
        assert dev <= _CUE_QH_BAR, (
            f"float32 log Q_H with {bad} beyond 5000 A differs from float64 by {dev:.3e} dex"
        )


#: Generator for the float64 bit-identity reference (#2849). Run as ``python -c``
#: with argv ``SRC OUT ROOT``: it writes the ionizing and Cue Q_H tables (Cloudy
#: and CB19 too) and the Cue forward output to ``OUT``. The committed fixture was
#: produced by this text on the pre-#2849 tree (origin/main b8d41e2b8).
_BITWISE_CHILD = r"""
import sys
SRC, OUT, ROOT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC)
import numpy as np
import jax
import tengri  # noqa: F401
from tengri.components.nebular.ionizing_spectrum import precompute_ionizing_params_table
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

ssp = load_ssp_data(ROOT + "/data/fsps_prsc_miles_chabrier.h5")
res = {"x64": np.asarray(bool(jax.config.jax_enable_x64))}
wave = np.asarray(ssp.ssp_wave)
flux = np.asarray(ssp.ssp_flux)
lgmet = np.asarray(ssp.ssp_lgmet)
log_age = np.asarray(ssp.ssp_lg_age_gyr) + 9.0
tab = precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=log_age)
for key in ("ionspec_table", "logqion_table", "seglum_table"):
    res["ion_" + key] = np.asarray(tab[key])
np.savez(OUT, **res)
print("WROTE", OUT, sorted(res))
"""


def test_float64_tables_bit_identical_to_reference(tmp_path):
    """With x64 on, the Cue ionizing tables are bit-identical to the pre-#2849 build.

    The reference is ``fixtures/nebular_qh_float64_unfixed.json``, written by
    ``_BITWISE_CHILD`` on the pre-#2849 tree. Exact float64 agreement holds
    because the power-of-two scale is exact and the log is taken in NumPy, as
    before. Only the three tables this change touches are pinned, so unrelated
    changes to the model's spectrum do not move the reference.
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
        assert np.array_equal(new[key], ref[key]), (
            f"{key} differs from the reference float64 build"
        )
