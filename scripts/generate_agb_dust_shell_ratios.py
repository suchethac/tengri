#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Generate the AGB circumstellar dust-shell weighting ratio template.

FSPS bakes the Villaume, Conroy & Johnson (2015) circumstellar dust-shell
reprocessing of TP-AGB stars into every MIST SSP spectrum at its default
weight (``agb_dust=1.0``, which scales tau(1 micron) of each DUSTY shell
template). This script measures how the emergent SSP spectrum changes with
that weight, at FSPS's own metallicity and age nodes, and packages the result
as a resampling template consumed at runtime by
``tengri.components.stellar.agb_dust_shell``.

Physics
-------
For each (Z, age) SSP and each weight w, the ratio

    R(w; Z, age, lambda) = f(agb_dust=w) / f(agb_dust=1)

is formed with a flux guard: wherever the w=1 reference is below
``FLUX_GUARD_EPS`` of that spectrum's own peak (no flux to reprocess), R is
defined as 1 rather than the quotient of two near-zero floats. The ratio is
stored exactly as FSPS gives it (no clamping). The wavelength window is the
set of FSPS nodes where any stored weight deviates from 1 by more than
``WINDOW_THRESHOLD`` at any (Z, age); outside it the loader assumes R = 1.

R(w) is neither linear in w nor smooth at w = 0 (FSPS switches the shell
model off there, so R(0) differs from the w -> 0+ limit by up to a few
percent). The template therefore stores a ladder of weight planes
(``WEIGHTS``) and the loader interpolates linearly in w between them. The
ladder was chosen by measuring the interpolation error against direct FSPS
output at weights between the planes; this script repeats that measurement at
the midpoint of every interval (:func:`_interpolation_check`) and records it.

Storage
-------
``R`` is stored as float16 (relative error at most ``2**-11`` = 0.05 %,
recorded in the attrs). Values within that of unity collapse to exactly 1, so
the plane compresses well, which keeps the full native FSPS wavelength window
under the 10 MB data-file cap without wavelength decimation.

Provenance
----------
Computed with the FSPS build named in the output attrs (MIST isochrones,
MILES spectral library, Chabrier IMF). The ratio is reused for the c3k_a
tengri grids and for the Kroupa and Salpeter IMFs; the IMF dependence is
measured at Z_sun for three ages and two weights, the spectral-library
dependence cannot be measured with this FSPS build.

Output
------
``data/agb_dust_shell_ratios_mist.h5``, consumed by
:mod:`tengri.components.stellar.agb_dust_shell`. Needs ``SPS_HOME``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime

import h5py
import numpy as np

SPS_HOME = os.environ.get("SPS_HOME")
if not SPS_HOME:
    raise RuntimeError("SPS_HOME environment variable not set.")

import fsps

from tengri.utils.physics_constants import LOG10_ZSUN

#: Weight ladder. w = 1 is the reference (stored implicitly as the exact
#: identity plane). The spacing is dense near w = 0, where R(w) rises
#: steeply, and was set so the linear-in-w interpolation error against direct
#: FSPS stays at or below ``INTERP_TOL`` between planes (see
#: :func:`_interpolation_check`).
WEIGHTS = np.array(
    [
        *(0.0, 1 / 1024, 1 / 128, 1 / 8, 7 / 32, 5 / 16, 7 / 16, 9 / 16, 13 / 16),
        *(1.0, 5 / 4, 13 / 8, 2.0, 3.0),
    ],
    dtype=np.float64,
)

#: A (Z, age, lambda) bin counts as "has flux to reprocess" only when the
#: w = 1 reference exceeds this fraction of that spectrum's own peak. Bins
#: below it hold the troughs of the reference spectrum, where the quotient of
#: two near-zero fluxes is dominated by the trough depth and is not a
#: measurement of the shell.
FLUX_GUARD_EPS = 1e-6

#: Wavelength-window threshold on |R - 1| (max over weights, Z, age).
WINDOW_THRESHOLD = 1e-4

#: Largest accepted interpolation error between planes (relative, over
#: 2-30 um, at Z_sun for ``CHECK_AGES_GYR``). The first interval [0, w_1] is
#: reported but excluded: R is discontinuous at w = 0 (shell model on/off).
INTERP_TOL = 0.02
CHECK_AGES_GYR = (0.3, 1.0, 3.0)
CHECK_BAND_ANGSTROM = (2.0e4, 3.0e5)

#: IMF sensitivity variants: FSPS imf_type 2 = Kroupa (2001), 0 = Salpeter (1955).
IMF_VARIANTS = {"kroupa": 2, "salpeter": 0}
IMF_WEIGHTS = (0.0, 3.0)

SIZE_CAP_MB = 10.0
RATIO_DTYPE = np.float16

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_GRID_PATH = os.path.join(REPO_ROOT, "data", "fsps_mist_c3k_a_chabrier.h5")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "agb_dust_shell_ratios_mist.h5")

#: Scratch cache of the raw FSPS spectra, one file per weight (the FSPS calls
#: are the slow step; everything downstream is cheap numpy).
RAW_CACHE_DIR = os.environ.get(
    "AGB_DUST_RAW_CACHE_DIR", os.path.join(tempfile.gettempdir(), "agb_dust_raw")
)


def _git_describe(path: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", path, "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _make_sp(imf_type: int = 1) -> fsps.StellarPopulation:
    return fsps.StellarPopulation(zcontinuous=0, sfh=0, imf_type=imf_type, add_agb_dust_model=True)


def _spec_cache_path(w: float) -> str:
    return os.path.join(RAW_CACHE_DIR, f"spec_w{w:.10f}.npy")


def _raw_cube(sp, w: float, age_gyr: np.ndarray, n_met: int) -> np.ndarray:
    """FSPS L_nu at every (Z node, age node) for weight ``w``, cached on disk.

    Returns
    -------
    ndarray, shape (n_met, n_age, n_wave), float32
    """
    path = _spec_cache_path(w)
    if os.path.exists(path):
        return np.load(path)
    sp.params["agb_dust"] = float(w)
    cube = None
    for z_idx in range(n_met):
        sp.params["zmet"] = z_idx + 1  # FSPS zmet is 1-indexed
        for a_idx, age in enumerate(age_gyr):
            wave, spec = sp.get_spectrum(tage=float(age), peraa=False)
            if cube is None:
                cube = np.empty((n_met, age_gyr.shape[0], wave.shape[0]), dtype=np.float32)
                np.save(os.path.join(RAW_CACHE_DIR, "wave.npy"), np.asarray(wave, np.float64))
            cube[z_idx, a_idx] = spec
    np.save(path, cube)
    return cube


def _guarded_ratio(spec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """R = spec / ref where ref exceeds the flux guard, 1 elsewhere."""
    peak = np.max(ref, axis=-1, keepdims=True)
    has_flux = ref > FLUX_GUARD_EPS * peak
    ratio = np.ones_like(ref)
    np.divide(spec, ref, out=ratio, where=has_flux)
    return ratio


def _report_extremes(ratio, cubes, ref, wave, log_z, age_gyr, weights) -> dict:
    """Print the most extreme bins and return summary values for the attrs."""
    n_hi = int(np.count_nonzero(ratio > 10.0))
    n_lo = int(np.count_nonzero(ratio < 0.1))
    print(
        f"R range [{ratio.min():.4f}, {ratio.max():.4f}]; "
        f"{n_lo} bins < 0.1, {n_hi} bins > 10 (of {ratio.size})"
    )
    flat_abs_log = np.abs(np.log(ratio)).ravel()
    top = np.argpartition(flat_abs_log, -5)[-5:]
    for flat in top[np.argsort(flat_abs_log[top])[::-1]]:
        wi, zi, ai, li = np.unravel_index(flat, ratio.shape)
        peak = ref[zi, ai].max()
        print(
            f"  extreme: w={weights[wi]:.4g} logZ={log_z[zi]:.3f} age={age_gyr[ai]:.3g} Gyr "
            f"lambda={wave[li]:.1f} A  R={ratio[wi, zi, ai, li]:.4g}  "
            f"f(w)={cubes[wi][zi, ai, li]:.3e}  f(w=1)={ref[zi, ai, li]:.3e}  "
            f"f(w=1)/peak={ref[zi, ai, li] / peak:.2e}"
        )
    return {
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "n_below_0p1": n_lo,
        "n_above_10": n_hi,
    }


def _window(ratio_non_ref, wave, age_gyr, log_z, weights_non_ref) -> tuple[np.ndarray, dict]:
    """Wavelength window and the facts the attrs record about its edges."""
    dev = np.abs(ratio_non_ref - 1.0)
    window_mask = np.max(dev, axis=(0, 1, 2)) > WINDOW_THRESHOLD
    if not window_mask.any():
        raise RuntimeError("No wavelength node exceeds the window threshold.")
    idx = np.where(window_mask)[0]
    info = {"window_lo": float(wave[idx[0]]), "window_hi": float(wave[idx[-1]])}
    below = (wave < 912.0) & window_mask
    cells = dev[..., below] > WINDOW_THRESHOLD  # (w, Z, age, lambda<912)
    n_cells = int(np.count_nonzero(np.any(cells, axis=-1)))
    info["n_cells_below_912"] = n_cells
    info["max_dev_below_912"] = float(dev[..., below].max()) if below.any() else 0.0
    if n_cells:
        ages = np.where(np.any(cells, axis=(0, 1, 3)))[0]
        zs = np.where(np.any(cells, axis=(0, 2, 3)))[0]
        info["below_912_age_gyr_min"] = float(age_gyr[ages].min())
        info["below_912_logz_max"] = float(log_z[zs].max())
    print(
        f"Window ({WINDOW_THRESHOLD:.0e}): {window_mask.sum()}/{wave.shape[0]} nodes, "
        f"{info['window_lo']:.1f}-{info['window_hi']:.1f} A; below 912 A: {n_cells} "
        f"(w,Z,age) cells, max |R-1| = {info['max_dev_below_912']:.2e}"
    )
    return window_mask, info


def _pins(ratio, weights, wave, log_z, age_gyr) -> dict:
    """Pinned numbers at the node nearest Z_sun and 1 Gyr (Chabrier)."""
    z_idx = int(np.argmin(np.abs(log_z - LOG10_ZSUN)))
    a_idx = int(np.argmin(np.abs(age_gyr - 1.0)))
    band = (wave >= 2.0e4) & (wave <= 1.0e5)
    inv_r0 = 1.0 / ratio[int(np.argmin(np.abs(weights - 0.0))), z_idx, a_idx, band]
    i10 = int(np.argmin(np.abs(wave - 1.0e5)))
    r3 = float(ratio[int(np.argmin(np.abs(weights - 3.0))), z_idx, a_idx, i10])
    print(
        f"Pins (logZ={log_z[z_idx]:.4f}, {age_gyr[a_idx]:.4f} Gyr): "
        f"median(1/R(0)) 2-10 um = {np.median(inv_r0):.4f}, max = {np.max(inv_r0):.4f}, "
        f"R(3) at 10 um = {r3:.4f}"
    )
    return {
        "pin_median_inv_r0_2_10um": float(np.median(inv_r0)),
        "pin_max_inv_r0_2_10um": float(np.max(inv_r0)),
        "pin_r3_at_10um": r3,
    }


def _check_nodes(log_z, age_gyr) -> tuple[int, list[int]]:
    z_idx = int(np.argmin(np.abs(log_z - LOG10_ZSUN)))
    a_idx = [int(np.argmin(np.abs(age_gyr - a))) for a in CHECK_AGES_GYR]
    return z_idx, a_idx


def _direct_ratio(sp, w, z_idx, a_idx, age_gyr, ref_spec) -> np.ndarray:
    """Direct FSPS ratio at weight ``w``: shape (n_check_ages, n_wave)."""
    sp.params["agb_dust"] = float(w)
    sp.params["zmet"] = z_idx + 1
    spec = np.stack(
        [sp.get_spectrum(tage=float(age_gyr[a]), peraa=False)[1] for a in a_idx]
    ).astype(np.float32)
    return _guarded_ratio(spec, ref_spec)


def _interpolation_check(sp, ratio_planes, weights, wave, log_z, age_gyr, ref) -> dict:
    """Error of linear-in-w interpolation against direct FSPS at every midpoint.

    Parameters
    ----------
    ratio_planes : ndarray, shape (n_w, n_met, n_age, n_wave)
        Stored planes (after float16 quantization), including w = 1.
    weights : ndarray, shape (n_w,)
        Plane weights, ascending.

    Returns
    -------
    dict
        ``midpoints`` (n_int,), ``max_rel_err`` (n_int,) over CHECK_BAND at
        Z_sun for CHECK_AGES_GYR, and the worst error excluding the first
        interval.
    """
    z_idx, a_idx = _check_nodes(log_z, age_gyr)
    band = (wave >= CHECK_BAND_ANGSTROM[0]) & (wave <= CHECK_BAND_ANGSTROM[1])
    ref_spec = ref[z_idx, a_idx]
    mids, errs = [], []
    for lo, hi in zip(range(len(weights) - 1), range(1, len(weights)), strict=True):
        w_mid = 0.5 * (weights[lo] + weights[hi])
        interp = 0.5 * (ratio_planes[lo][z_idx][a_idx] + ratio_planes[hi][z_idx][a_idx])
        direct = _direct_ratio(sp, w_mid, z_idx, a_idx, age_gyr, ref_spec)
        err = float(np.max(np.abs(interp[:, band] / direct[:, band] - 1.0)))
        mids.append(w_mid)
        errs.append(err)
        print(
            f"  interval [{weights[lo]:.6g}, {weights[hi]:.6g}] midpoint {w_mid:.6g}: "
            f"max |R_interp/R_direct - 1| = {err:.4f}"
        )
    worst = max(errs[1:])
    if worst > INTERP_TOL:
        raise RuntimeError(f"interpolation error {worst:.4f} exceeds INTERP_TOL={INTERP_TOL}")
    return {"midpoints": np.array(mids), "max_rel_err": np.array(errs), "worst_excl_first": worst}


def _imf_sensitivity(ratio, weights, wave, log_z, age_gyr, window_mask) -> float:
    """Max |R_IMF - R_Chabrier| over the window, Z_sun, three ages, two weights."""
    z_idx, a_idx = _check_nodes(log_z, age_gyr)
    worst = 0.0
    for name, imf in IMF_VARIANTS.items():
        sp = _make_sp(imf)
        for w in (1.0, *IMF_WEIGHTS):
            sp.params["agb_dust"] = float(w)
            sp.params["zmet"] = z_idx + 1
            cube_w = {
                a: sp.get_spectrum(tage=float(age_gyr[a]), peraa=False)[1].astype(np.float32)
                for a in a_idx
            }
            if w == 1.0:
                ref_k = cube_w
                continue
            for a in a_idx:
                r_k = _guarded_ratio(cube_w[a], ref_k[a])
                r_c = ratio[int(np.argmin(np.abs(weights - w))), z_idx, a]
                worst = max(worst, float(np.max(np.abs(r_k[window_mask] - r_c[window_mask]))))
        print(f"  IMF {name}: running max |dR| (window) = {worst:.3e}")
    return worst


def _write(path, planes, weights, wave_win, log_z, age_gyr, attrs) -> None:
    stored = planes.astype(RATIO_DTYPE)
    with h5py.File(path, "w") as f:
        f.create_dataset("log_z", data=log_z.astype(np.float32))
        f.create_dataset("log_age_yr", data=(np.log10(age_gyr) + 9.0).astype(np.float32))
        f.create_dataset("wave_angstrom", data=wave_win.astype(np.float64))
        f.create_dataset("weights", data=weights.astype(np.float64))
        f.create_dataset(
            "ratio", data=stored, compression="gzip", compression_opts=9, shuffle=True
        )
        for key, value in attrs.items():
            f.attrs[key] = value


def main() -> None:
    t_start = time.time()
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)

    with h5py.File(SHIPPED_GRID_PATH, "r") as f:
        lgmet_shipped = np.asarray(f["ssp_lgmet"][:], dtype=np.float64)
        age_gyr = 10.0 ** np.asarray(f["ssp_lg_age_gyr"][:], dtype=np.float64)

    sp = _make_sp()
    log_z = np.log10(np.asarray(sp.zlegend, dtype=np.float64))
    n_met = log_z.shape[0]
    if n_met != lgmet_shipped.shape[0] or np.max(np.abs(log_z - lgmet_shipped)) > 1e-3:
        raise RuntimeError("FSPS metallicity nodes do not match the shipped grid's.")
    libraries = ", ".join(x.decode() if isinstance(x, bytes) else str(x) for x in sp.libraries)
    print(f"FSPS {fsps.__version__}; libraries: {libraries}; {n_met} Z x {age_gyr.shape[0]} ages")

    cubes = []
    for w in WEIGHTS:
        t0 = time.time()
        cubes.append(_raw_cube(sp, float(w), age_gyr, n_met))
        print(f"w={w:.6g}: {time.time() - t0:.1f} s")
    wave = np.load(os.path.join(RAW_CACHE_DIR, "wave.npy")).astype(np.float64)

    w1 = int(np.argmin(np.abs(WEIGHTS - 1.0)))
    ref = cubes[w1]
    ratio = np.stack([_guarded_ratio(c, ref) for c in cubes])
    assert float(np.max(np.abs(ratio[w1] - 1.0))) == 0.0, "w=1 plane must be exactly 1"
    extremes = _report_extremes(ratio, cubes, ref, wave, log_z, age_gyr, WEIGHTS)

    keep = np.arange(len(WEIGHTS)) != w1
    window_mask, win_info = _window(ratio[keep], wave, age_gyr, log_z, WEIGHTS[keep])

    pins = _pins(ratio, WEIGHTS, wave, log_z, age_gyr)

    planes = ratio[keep][..., window_mask].astype(RATIO_DTYPE).astype(np.float32)
    quant_err = float(np.max(np.abs(planes / ratio[keep][..., window_mask] - 1.0)))
    print(f"R float16 quantization: max relative error of R = {quant_err:.3e}")

    print("Interpolation check against direct FSPS at every interval midpoint:")
    stored = np.insert(planes, int(np.searchsorted(WEIGHTS[keep], 1.0)), 1.0, axis=0)
    window_idx = np.where(window_mask)[0]
    wave_win = wave[window_idx]
    # Evaluate on the full wavelength axis: R = 1 outside the window.
    full = np.ones((stored.shape[0], *ratio.shape[1:]), dtype=np.float32)
    full[..., window_idx] = stored
    interp = _interpolation_check(sp, full, WEIGHTS, wave, log_z, age_gyr, ref)
    print(f"  worst interval excluding the first: {interp['worst_excl_first']:.4f}")

    print("IMF sensitivity (Z_sun; ages 0.3, 1, 3 Gyr; w = 0, 3):")
    imf_diff = _imf_sensitivity(ratio, WEIGHTS, wave, log_z, age_gyr, window_mask)

    wall = time.time() - t_start
    attrs = {
        "layout": "ratio_planes",
        "fsps_python_version": str(fsps.__version__),
        "libfsps_git_describe": _git_describe(SPS_HOME),
        "libraries": f"{libraries} (isochrones, spectral library, dust emission)",
        "isochrone": "mist",
        "imf_type": "1 (Chabrier 2003)",
        "zcontinuous": "0 (native FSPS metallicity nodes)",
        "reuse_note": (
            "Computed with MIST + MILES + Chabrier. Reused unchanged for the c3k_a "
            "tengri grids and for the Kroupa and Salpeter IMFs. The spectral-library "
            "dependence cannot be measured with this FSPS build (MILES only)."
        ),
        "command": " ".join(sys.argv),
        "generation_date": datetime.now(UTC).isoformat(),
        "flux_guard_eps": FLUX_GUARD_EPS,
        "flux_guard_rule": (
            "R = f(w)/f(w=1) where f(w=1) > eps * max_lambda(f(w=1)) per (Z, age) spectrum; "
            "R = 1 elsewhere. No clamping: R is stored as FSPS gives it."
        ),
        "window_rule": (
            f"wavelength kept where max over (w != 1, Z, age) of |R - 1| > {WINDOW_THRESHOLD:.0e}"
        ),
        "window_lo_angstrom": win_info["window_lo"],
        "window_hi_angstrom": win_info["window_hi"],
        "window_cells_below_912A": win_info["n_cells_below_912"],
        "window_max_dev_below_912A": win_info["max_dev_below_912"],
        "window_edge_note": (
            "The lower edge is set by a few old, low-metallicity (w, Z, age) cells at the "
            f"{WINDOW_THRESHOLD:.0e} threshold (ionizing-wavelength flux of those populations), "
            "not by shell absorption, which acts at lambda > ~1 micron."
        ),
        "ratio_min": extremes["ratio_min"],
        "ratio_max": extremes["ratio_max"],
        "ratio_bins_below_0p1": extremes["n_below_0p1"],
        "ratio_bins_above_10": extremes["n_above_10"],
        "ratio_dtype": "float16",
        "quantization_max_rel_err": quant_err,
        "interp_midpoints": interp["midpoints"],
        "interp_max_rel_err": interp["max_rel_err"],
        "interp_worst_excl_first": interp["worst_excl_first"],
        "interp_note": (
            "Linear-in-w interpolation vs direct FSPS at the midpoint of every interval, "
            f"max over 2-30 um at Z_sun, ages {CHECK_AGES_GYR} Gyr. The first interval is "
            "bounded below by the step in R at w = 0 (shell model on/off)."
        ),
        "imf_sensitivity_max_diff": imf_diff,
        "imf_sensitivity_scope": (
            "max |R_IMF - R_Chabrier| over the window, Kroupa and Salpeter, Z_sun only, "
            "ages 0.3/1/3 Gyr, w = 0 and 3."
        ),
        "log_z_convention": "absolute log10(Z), matching ssp_lgmet in the shipped SSP grids",
        "wall_time_s": wall,
        **pins,
    }
    for key in ("below_912_age_gyr_min", "below_912_logz_max"):
        if key in win_info:
            attrs[f"window_{key}"] = win_info[key]

    print(f"\nWriting {OUTPUT_PATH}")
    _write(OUTPUT_PATH, planes, WEIGHTS[keep], wave_win, log_z, age_gyr, attrs)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"Wrote {OUTPUT_PATH}: {size_mb:.2f} MB; total wall time {wall / 60.0:.1f} min")
    if size_mb > SIZE_CAP_MB:
        raise RuntimeError(f"file size {size_mb:.2f} MB exceeds the {SIZE_CAP_MB:.0f} MB cap")


if __name__ == "__main__":
    main()
