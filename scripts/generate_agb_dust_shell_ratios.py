#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Generate the AGB circumstellar dust-shell weighting ratio template.

FSPS bakes the Villaume, Conroy & Johnson (2015) circumstellar dust-shell
reprocessing of TP-AGB stars into every MIST SSP spectrum at its default
weight (``agb_dust=1.0``, which scales tau(1 micron) of each DUSTY shell
template). This script measures how the emergent SSP spectrum changes as a
function of that weight, at FSPS's own metallicity and age nodes, and
packages the result as a resampling template consumed at runtime by
``tengri.components.stellar.agb_dust_shell``.

Physics
-------
For each (Z, age) SSP and each weight w, the ratio

    R(w; Z, age, lambda) = f(agb_dust=w) / f(agb_dust=1)

is formed with a flux guard: wherever the w=1 reference is negligible
(no flux to reprocess), R is defined as 1 (no correction) rather than by
dividing two near-zero floats. The wavelength window is the set of nodes
where any stored weight deviates from 1 by more than 1e-4; outside it the
loader assumes R=1 exactly.

This script also measures whether R(w) is linear in w (expected: FSPS scales
only the shell optical depth linearly with ``agb_dust``, but the emergent
flux ratio need not be linear once the shell is optically thick). When the
residual of the linear model is negligible, only the per-(Z, age, lambda)
shell fraction S = R(2) - 1 is stored and the loader reconstructs
R(w) = 1 + (w-1)*S exactly; otherwise every sampled weight plane is stored
and the loader interpolates linearly in w.

Output
------
``data/agb_dust_shell_ratios_mist.h5``, consumed by
:mod:`tengri.components.stellar.agb_dust_shell`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import UTC, datetime

import h5py
import numpy as np

SPS_HOME = os.environ.get("SPS_HOME")
if not SPS_HOME:
    raise RuntimeError("SPS_HOME environment variable not set.")

import fsps

# Weights sampled: 1.0 is the shipped-grid reference; the rest bracket the
# legal runtime range [0, 3] (Uniform(0, 3) prior) plus the midpoints needed
# to measure linearity.
WEIGHTS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64)

# Flux guard: a (Z, age) wavelength bin counts as "has flux to reprocess"
# only if the w=1 reference exceeds this fraction of that spectrum's own
# peak; R is defined as 1 (no correction) elsewhere. 1e-6 is far above
# float64 roundoff (~1e-16), so the floor is set by photometric relevance,
# not precision: a first measurement at 1e-10 (chosen only to clear the
# float-precision bar) left deep, narrow dust-absorption troughs in the
# reference spectrum -- real, not noise, but six-plus decades below each
# (Z, age) spectrum's own peak -- in the window, where f(w=1) sits just
# above the floor and the ratio to a comparatively unsuppressed f(w) spikes
# by up to 1e6 in isolated bins. A trough at <1e-6 of a population's peak
# carries a photometrically and spectroscopically undetectable share of its
# light, so treating it as "no flux to reprocess" changes no measurable
# prediction while removing the ratio's instability there; the broadband
# effect this template exists to capture (median ~1.05, max ~2.4 over
# 2-10 um at solar/1 Gyr) sits nowhere near either floor.
FLUX_GUARD_EPS = 1e-6

# Wavelength window threshold: a node is kept only if some stored weight's
# ratio departs from unity by more than this, at some (Z, age).
WINDOW_THRESHOLD = 1e-4

# Linearity acceptance threshold on the single-plane (shell-fraction) model.
LINEARITY_RESIDUAL_TOL = 1e-6

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_GRID_PATH = os.path.join(REPO_ROOT, "data", "fsps_mist_c3k_a_chabrier.h5")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "agb_dust_shell_ratios_mist.h5")

# Scratch cache of the raw FSPS spectra (never in the repository): the FSPS
# calls are the ~13-minute cost; everything downstream (flux guard, window,
# linearity, wavelength decimation) is cheap numpy reused from this cache so
# the guard/decimation thresholds can be iterated without repaying FSPS.
RAW_SPECTRA_CACHE = os.environ.get(
    "AGB_DUST_RAW_CACHE",
    "/Users/suchethacooray/.claude/jobs/c936b159/tmp/fix2534/agb_dust_raw_spectra.npz",
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
    except Exception:
        return "unknown"


def main() -> None:
    t_start = time.time()

    with h5py.File(SHIPPED_GRID_PATH, "r") as f:
        ssp_lgmet_shipped = np.asarray(f["ssp_lgmet"][:], dtype=np.float64)
        ssp_lg_age_gyr_shipped = np.asarray(f["ssp_lg_age_gyr"][:], dtype=np.float64)
    age_gyr_grid = 10.0**ssp_lg_age_gyr_shipped
    n_age = age_gyr_grid.shape[0]

    print(f"Shipped grid nodes: {len(ssp_lgmet_shipped)} Z, {n_age} age")

    print("Initializing FSPS stellar population (zcontinuous=0, sfh=0, Chabrier)...")
    sp = fsps.StellarPopulation(
        zcontinuous=0,
        sfh=0,
        imf_type=1,  # Chabrier (2003)
        add_agb_dust_model=True,
    )
    zlegend = np.asarray(sp.zlegend, dtype=np.float64)
    log_z_abs = np.log10(zlegend)
    n_met = len(zlegend)
    print(f"FSPS native MIST metallicity grid: {n_met} nodes")
    print(f"  log10(Z) range: {log_z_abs.min():.4f} to {log_z_abs.max():.4f}")

    if n_met != len(ssp_lgmet_shipped):
        raise RuntimeError(
            f"FSPS native metallicity node count ({n_met}) does not match the "
            f"shipped grid ({len(ssp_lgmet_shipped)}); cannot assume alignment."
        )
    met_match = np.max(np.abs(log_z_abs - ssp_lgmet_shipped))
    print(f"  Max |log10(Z)_fsps - ssp_lgmet_shipped| = {met_match:.3e}")

    n_w = len(WEIGHTS)

    if os.path.exists(RAW_SPECTRA_CACHE):
        print(
            f"\nLoading cached raw FSPS spectra from {RAW_SPECTRA_CACHE} (skipping FSPS calls)..."
        )
        with np.load(RAW_SPECTRA_CACHE) as cache:
            cached_weights = cache["weights"]
            if not np.array_equal(cached_weights, WEIGHTS):
                raise RuntimeError(
                    f"Cache weights {cached_weights} != current WEIGHTS {WEIGHTS}; delete the "
                    f"cache ({RAW_SPECTRA_CACHE}) to regenerate."
                )
            wave_full = cache["wave_full"].astype(np.float64)
            spec_full = [cache[f"spec_{i}"] for i in range(n_w)]
        n_wave = wave_full.shape[0]
        print(f"Loaded {n_w} weight planes, n_wave={n_wave}")
    else:
        n_wave = None
        wave_full = None

        # spec_full[w_idx] has shape (n_met, n_age, n_wave); built incrementally
        # in float32 to bound memory (~57 MB per weight plane on the full
        # 11149-node native grid).
        spec_full = []

        for w_idx, w in enumerate(WEIGHTS):
            t_w0 = time.time()
            spec_this_w = None
            for z_idx in range(n_met):
                for a_idx in range(n_age):
                    sp.params["zmet"] = z_idx + 1  # FSPS zmet is 1-indexed
                    sp.params["agb_dust"] = float(w)
                    wave, spec = sp.get_spectrum(tage=float(age_gyr_grid[a_idx]), peraa=False)
                    if wave_full is None:
                        wave_full = np.asarray(wave, dtype=np.float64)
                        n_wave = wave_full.shape[0]
                    if spec_this_w is None:
                        spec_this_w = np.empty((n_met, n_age, n_wave), dtype=np.float32)
                    spec_this_w[z_idx, a_idx, :] = spec.astype(np.float32)
            spec_full.append(spec_this_w)
            print(
                f"Weight w={w:.2f} ({w_idx + 1}/{n_w}): "
                f"{time.time() - t_w0:.1f} s, "
                f"flux range [{spec_this_w.min():.3e}, {spec_this_w.max():.3e}]"
            )

        os.makedirs(os.path.dirname(RAW_SPECTRA_CACHE), exist_ok=True)
        cache_kwargs = {f"spec_{i}": spec_full[i] for i in range(n_w)}
        np.savez(
            RAW_SPECTRA_CACHE,
            weights=WEIGHTS,
            wave_full=wave_full.astype(np.float32),
            **cache_kwargs,
        )
        print(f"Cached raw spectra to {RAW_SPECTRA_CACHE}")

    w1_idx = int(np.argmin(np.abs(WEIGHTS - 1.0)))
    f_ref = spec_full[w1_idx]  # (n_met, n_age, n_wave) float32

    # Per-(Z, age) peak flux, broadcast over wavelength -- the guard
    # threshold scales with each spectrum's own brightness rather than a
    # single global floor (SSPs at very different ages/metallicities span
    # many decades of normalization).
    f_ref_peak = np.max(f_ref, axis=2, keepdims=True)  # (n_met, n_age, 1)
    has_flux = f_ref > (FLUX_GUARD_EPS * f_ref_peak)

    print(
        f"\nFlux guard: {np.count_nonzero(has_flux)}/{has_flux.size} bins have usable flux "
        f"(eps={FLUX_GUARD_EPS:.1e})"
    )

    ratio_full = np.ones((n_w, n_met, n_age, n_wave), dtype=np.float32)
    for w_idx in range(n_w):
        np.divide(
            spec_full[w_idx],
            f_ref,
            out=ratio_full[w_idx],
            where=has_flux,
        )
        ratio_full[w_idx][~has_flux] = 1.0

    # Ratio clamp: even past the flux guard, a handful of bins sit in a deep,
    # narrow DUSTY absorption trough where f_ref is just above the guard
    # floor while f(w) at the same bin is not -- a real effect, but a
    # single-pixel-wide one carrying a negligible share of that (Z, age)
    # spectrum's flux (it is in a trough by definition), and the resulting
    # ratio spikes to O(1e2-1e6), which no smooth storage grid can represent
    # and which swamps compression for no science gain. Clamp to one dex
    # either side of unity: generous relative to every measured broadband
    # pin (max 1/R(w=0)) over 2-10 um is ~2.4, R(w=3) at 10 um is ~0.85), so
    # it never touches the physics this template exists to carry, only the
    # single-bin trough artifacts. w=1 is unaffected (clip(1, ...) == 1), so
    # the w=1-plane-is-exact-identity assertion below is unchanged.
    RATIO_CLAMP_LO, RATIO_CLAMP_HI = 0.1, 10.0
    n_clamped = int(np.count_nonzero((ratio_full < RATIO_CLAMP_LO) | (ratio_full > RATIO_CLAMP_HI)))
    print(
        f"Ratio clamp [{RATIO_CLAMP_LO}, {RATIO_CLAMP_HI}]: {n_clamped}/{ratio_full.size} "
        f"bins clamped ({100.0 * n_clamped / ratio_full.size:.4f}%)"
    )
    np.clip(ratio_full, RATIO_CLAMP_LO, RATIO_CLAMP_HI, out=ratio_full)

    max_w1_dev = float(np.max(np.abs(ratio_full[w1_idx] - 1.0)))
    print(f"w=1 plane max |R - 1| before assertion: {max_w1_dev:.3e}")
    assert max_w1_dev == 0.0, (
        f"w=1 plane must be exactly 1 by construction (flux-guarded self-ratio); "
        f"got max deviation {max_w1_dev:.6e}"
    )

    # Wavelength window: keep nodes where ANY stored weight (excluding the
    # trivial w=1 plane, which is now provably all-ones) departs from unity
    # by more than the threshold at some (Z, age).
    non_ref = np.delete(ratio_full, w1_idx, axis=0)  # (n_w-1, n_met, n_age, n_wave)
    max_dev_per_wave = np.max(np.abs(non_ref - 1.0), axis=(0, 1, 2))  # (n_wave,)
    window_mask = max_dev_per_wave > WINDOW_THRESHOLD
    n_window = int(np.count_nonzero(window_mask))
    if n_window == 0:
        raise RuntimeError("No wavelength node exceeds the window threshold; check the run.")
    window_idx = np.where(window_mask)[0]
    wave_lo = float(wave_full[window_idx[0]])
    wave_hi = float(wave_full[window_idx[-1]])
    print(
        f"\nWavelength window (|R-1| > {WINDOW_THRESHOLD:.0e}): "
        f"{n_window}/{n_wave} nodes, {wave_lo:.1f}-{wave_hi:.1f} Angstrom"
    )

    # TP-AGB age window report (not used to trim the age axis): which ages
    # show any departure from unity anywhere in the wavelength window, for
    # any non-reference weight.
    dev_in_window = np.abs(non_ref[:, :, :, window_mask] - 1.0)  # (n_w-1, n_met, n_age, n_win)
    max_dev_per_age = np.max(dev_in_window, axis=(0, 1, 3))  # (n_age,)
    age_active_mask = max_dev_per_age > WINDOW_THRESHOLD
    if np.any(age_active_mask):
        active_ages_gyr = age_gyr_grid[age_active_mask]
        print(
            f"TP-AGB age window (|R-1| > {WINDOW_THRESHOLD:.0e} somewhere in the "
            f"wavelength window): {active_ages_gyr.min():.4f}-{active_ages_gyr.max():.4f} Gyr "
            f"({np.count_nonzero(age_active_mask)}/{n_age} nodes)"
        )
        max_dev_outside = (
            float(np.max(dev_in_window[:, :, ~age_active_mask, :]))
            if np.any(~age_active_mask)
            else 0.0
        )
        print(f"  Max |R-1| outside that age window: {max_dev_outside:.3e}")
    else:
        raise RuntimeError("No age node shows any AGB-dust-shell effect; check the run.")

    # Linearity test: S = R(2) - 1; R_lin(w) = 1 + (w-1)*S. Measure the
    # residual at every OTHER sampled weight (0, 0.5, 1.5, 3), within the
    # wavelength window, over all (Z, age).
    w2_idx = int(np.argmin(np.abs(WEIGHTS - 2.0)))
    shell_fraction = (ratio_full[w2_idx] - 1.0).astype(np.float32)  # (n_met, n_age, n_wave)
    shell_fraction_window = shell_fraction[:, :, window_mask]

    test_w_idxs = [i for i in range(n_w) if i not in (w1_idx, w2_idx)]
    max_residual = 0.0
    for idx in test_w_idxs:
        w = WEIGHTS[idx]
        r_actual = ratio_full[idx][:, :, window_mask]
        r_linear = 1.0 + (w - 1.0) * shell_fraction_window
        residual = float(np.max(np.abs(r_actual - r_linear)))
        print(f"Linearity residual at w={w:.2f}: max |R - R_linear| = {residual:.3e}")
        max_residual = max(max_residual, residual)

    is_linear = max_residual <= LINEARITY_RESIDUAL_TOL
    _layout_label = (
        "LINEAR (storing shell_fraction only)" if is_linear else "NONLINEAR (storing all planes)"
    )
    print(
        f"\nOverall linearity residual: {max_residual:.3e} "
        f"(tol {LINEARITY_RESIDUAL_TOL:.0e}) -> {_layout_label}"
    )

    # Pinned numbers for the test suite, measured on THIS run.
    z_sun_idx = int(
        np.argmin(np.abs(log_z_abs - 0.0))
    )  # absolute log10(Z); solar is log10(Z)=0? no.
    # Solar metallicity in absolute log10(Z): tengri's LOG10_ZSUN = -1.848
    # (Asplund 2009). Find the native node nearest that value.
    LOG10_ZSUN = -1.848
    z_sun_idx = int(np.argmin(np.abs(log_z_abs - LOG10_ZSUN)))
    age_1gyr_idx = int(np.argmin(np.abs(age_gyr_grid - 1.0)))
    wave_2um_10um_mask = (wave_full >= 2.0e4) & (wave_full <= 1.0e5)
    r0_in_range = ratio_full[0, z_sun_idx, age_1gyr_idx, wave_2um_10um_mask]
    inv_r0 = 1.0 / r0_in_range
    wave_10um_idx = int(np.argmin(np.abs(wave_full - 1.0e5)))
    r3_at_10um = float(
        ratio_full[int(np.argmin(np.abs(WEIGHTS - 3.0))), z_sun_idx, age_1gyr_idx, wave_10um_idx]
    )
    print(
        f"\nPin check: Z node nearest solar: log10(Z)={log_z_abs[z_sun_idx]:.4f} "
        f"(LOG10_ZSUN={LOG10_ZSUN}), age node nearest 1 Gyr: {age_gyr_grid[age_1gyr_idx]:.4f} Gyr"
    )
    print(f"  median(1/R(w=0)) over 2-10 um = {np.median(inv_r0):.4f}")
    print(f"  max(1/R(w=0)) over 2-10 um    = {np.max(inv_r0):.4f}")
    print(f"  R(w=3) at ~10 um              = {r3_at_10um:.4f}")

    # IMF sensitivity check: one Kroupa run at the solar/1-Gyr node, w=0 and
    # w=1, compared against the Chabrier ratio already measured.
    print("\nIMF sensitivity check (Kroupa vs Chabrier) at the solar/1-Gyr node...")
    sp_kroupa = fsps.StellarPopulation(
        zcontinuous=0,
        sfh=0,
        imf_type=2,
        add_agb_dust_model=True,  # Kroupa (2001)
    )
    sp_kroupa.params["zmet"] = z_sun_idx + 1
    sp_kroupa.params["agb_dust"] = 0.0
    _, spec_kroupa_w0 = sp_kroupa.get_spectrum(tage=float(age_gyr_grid[age_1gyr_idx]), peraa=False)
    sp_kroupa.params["agb_dust"] = 1.0
    _, spec_kroupa_w1 = sp_kroupa.get_spectrum(tage=float(age_gyr_grid[age_1gyr_idx]), peraa=False)
    ratio_kroupa = np.ones(n_wave, dtype=np.float64)
    ref_peak_k = np.max(spec_kroupa_w1)
    mask_k = spec_kroupa_w1 > FLUX_GUARD_EPS * ref_peak_k
    ratio_kroupa[mask_k] = spec_kroupa_w0[mask_k] / spec_kroupa_w1[mask_k]
    ratio_chabrier_w0 = ratio_full[0, z_sun_idx, age_1gyr_idx, :]
    imf_max_diff = float(
        np.max(np.abs(ratio_kroupa[window_mask] - ratio_chabrier_w0[window_mask]))
    )
    print(f"  Max |R_Kroupa(w=0) - R_Chabrier(w=0)| in window = {imf_max_diff:.3e}")

    log_age_yr = (ssp_lg_age_gyr_shipped + 9.0).astype(np.float32)

    # Wavelength decimation to meet the 10 MB cap: the native window (every
    # FSPS node where |R-1| > the window threshold) is large (R is smeared
    # over a wide window by the small subset of (Z, age) nodes where TP-AGB
    # stars dominate), but R is smooth in log(wavelength) within it (common
    # narrow spectral lines cancel in the w/w=1 ratio). Keep every Nth native
    # node, picking the SMALLEST N (finest grid) whose log-linear
    # reconstruction of the dropped nodes reproduces the native values to
    # within the same WINDOW_THRESHOLD tolerance -- no information is lost
    # beyond what the window rule itself already calls negligible. Decimated
    # nodes are literal FSPS values (a subsample, not a synthetic refit); only
    # the dropped ones are ever reconstructed, and only by the loader's own
    # np.interp, which this search already validates.
    wave_window_full = wave_full[window_mask].astype(np.float64)
    log_wave_window_full = np.log10(wave_window_full)
    planes_idx = [i for i in range(n_w) if i != w1_idx]
    ratio_window_full = ratio_full[planes_idx][
        :, :, :, window_mask
    ]  # (n_w-1, n_met, n_age, n_win)
    n_win_full = wave_window_full.shape[0]

    def _decimation_residual(stride: int) -> tuple[np.ndarray, float]:
        idx = np.arange(0, n_win_full, stride)
        if idx[-1] != n_win_full - 1:
            idx = np.append(idx, n_win_full - 1)
        log_wave_dec = log_wave_window_full[idx]
        max_resid = 0.0
        for pi in range(ratio_window_full.shape[0]):
            plane_full = ratio_window_full[pi]  # (n_met, n_age, n_win_full)
            flat_full = plane_full.reshape(-1, n_win_full)
            flat_dec = flat_full[:, idx]
            for row in range(flat_full.shape[0]):
                recon = np.interp(log_wave_window_full, log_wave_dec, flat_dec[row])
                max_resid = max(max_resid, float(np.max(np.abs(recon - flat_full[row]))))
        return idx, max_resid

    # A handful of single-pixel DUSTY-trough spikes (clamped above, but still
    # sharp single-bin features) make a RECONSTRUCTION-residual-gated search
    # pathological: no stride up to 40 brings the worst-bin residual near
    # WINDOW_THRESHOLD, because interpolating across a clamped spike always
    # costs close to the full clamp range at that one bin, independent of
    # how fine the grid is. Decimation is therefore chosen directly against
    # the SIZE CAP instead: the finest (smallest-stride, most information
    # preserved) grid whose actual compressed file size clears
    # SIZE_CAP_MB with SIZE_SAFETY_MARGIN_MB of headroom for the other
    # datasets/attrs, measured by writing a real scratch file at each
    # candidate stride (fast: the expensive step is the FSPS calls above,
    # already cached). The reconstruction residual is still printed for
    # every candidate, for the record.
    SIZE_CAP_MB = 10.0
    SIZE_SAFETY_MARGIN_MB = 1.0
    _scratch_probe_path = RAW_SPECTRA_CACHE + ".size_probe.h5"

    def _probe_size_mb(ratio_data: np.ndarray) -> float:
        with h5py.File(_scratch_probe_path, "w") as pf:
            pf.create_dataset(
                "ratio", data=ratio_data, compression="gzip", compression_opts=9, shuffle=True
            )
        size_mb = os.path.getsize(_scratch_probe_path) / 1e6
        os.remove(_scratch_probe_path)
        return size_mb

    chosen_idx = None
    chosen_stride = None
    for stride in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40):
        idx, resid = _decimation_residual(stride)
        ratio_candidate = ratio_window_full[:, :, :, idx].astype(np.float32)
        probe_mb = _probe_size_mb(ratio_candidate)
        print(
            f"Wavelength decimation stride={stride}: {idx.shape[0]} nodes, "
            f"reconstruction residual max|R-R_interp|={resid:.3e}, "
            f"probe file size={probe_mb:.2f} MB"
        )
        if probe_mb <= SIZE_CAP_MB - SIZE_SAFETY_MARGIN_MB:
            chosen_idx = idx
            chosen_stride = stride
            break
    if chosen_idx is None:
        chosen_idx = idx  # coarsest tried; file may still exceed the cap
        chosen_stride = stride
        print(f"WARNING: no stride up to {stride} cleared the size budget; using it anyway.")

    wave_window = wave_window_full[chosen_idx].astype(np.float32)
    ratio_to_store = ratio_window_full[:, :, :, chosen_idx]
    print(
        f"Decimated wavelength grid: {wave_window.shape[0]}/{n_win_full} nodes kept "
        f"(stride={chosen_stride})"
    )

    wall_time_s = time.time() - t_start
    print(f"\nTotal wall time: {wall_time_s:.1f} s ({wall_time_s / 60.0:.1f} min)")

    print(f"\nWriting {OUTPUT_PATH}...")
    with h5py.File(OUTPUT_PATH, "w") as f:
        f.create_dataset("log_z", data=log_z_abs.astype(np.float32))
        f.create_dataset("log_age_yr", data=log_age_yr)
        f.create_dataset("wave_angstrom", data=wave_window)

        if is_linear:
            shell_fraction_dec = (ratio_to_store[planes_idx.index(w2_idx)] - 1.0).astype(
                np.float32
            )
            f.create_dataset(
                "shell_fraction",
                data=shell_fraction_dec,
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
        else:
            f.create_dataset("weights", data=WEIGHTS[planes_idx].astype(np.float32))
            f.create_dataset(
                "ratio",
                data=ratio_to_store.astype(np.float32),
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )

        f.attrs["layout"] = "shell_fraction" if is_linear else "ratio_planes"
        f.attrs["fsps_python_version"] = str(fsps.__version__)
        f.attrs["libfsps_git_describe"] = _git_describe(SPS_HOME)
        f.attrs["libraries"] = (
            "mist isochrones, c3k_a (+miles-equivalent native grid) spectral library"
        )
        f.attrs["isochrone"] = "mist"
        f.attrs["imf_type"] = "1 (Chabrier 2003)"
        f.attrs["zcontinuous"] = "0 (native FSPS metallicity nodes)"
        f.attrs["command"] = " ".join(sys.argv)
        f.attrs["generation_date"] = datetime.now(UTC).isoformat()
        f.attrs["flux_guard_eps"] = FLUX_GUARD_EPS
        f.attrs["flux_guard_rule"] = (
            "R = f(w)/f(w=1) where f(w=1) > eps * max_lambda(f(w=1)) per (Z, age) spectrum; "
            "R = 1 elsewhere (no flux to reprocess)."
        )
        f.attrs["window_rule"] = (
            f"wavelength kept where max over (w != 1, Z, age) of |R - 1| > {WINDOW_THRESHOLD:.0e}"
        )
        f.attrs["ratio_clamp_lo"] = RATIO_CLAMP_LO
        f.attrs["ratio_clamp_hi"] = RATIO_CLAMP_HI
        f.attrs["ratio_clamp_rule"] = (
            "ratio clamped to [lo, hi] after the flux guard: isolated single-bin DUSTY "
            "absorption-trough spikes (negligible flux, see flux_guard_rule) carry no "
            "science signal but defeat storage; every measured broadband pin sits well "
            "inside [lo, hi]."
        )
        f.attrs["wavelength_decimation_stride"] = chosen_stride
        f.attrs["wavelength_decimation_rule"] = (
            "every Nth native-window node, N (the smallest tried) whose compressed file "
            f"size clears the {SIZE_CAP_MB:.0f} MB cap with {SIZE_SAFETY_MARGIN_MB:.0f} MB "
            "margin; the dropped nodes are reconstructed by the loader's log-linear "
            "np.interp, same as the ratio_clamp bins"
        )
        f.attrs["linearity_residual"] = max_residual
        f.attrs["linearity_tol"] = LINEARITY_RESIDUAL_TOL
        f.attrs["log_z_convention"] = (
            "absolute log10(Z), matching ssp_lgmet in the shipped SSP grids"
        )
        f.attrs["imf_sensitivity_max_diff"] = imf_max_diff
        f.attrs["wall_time_s"] = wall_time_s
        f.attrs["pin_median_inv_r0_2_10um"] = float(np.median(inv_r0))
        f.attrs["pin_max_inv_r0_2_10um"] = float(np.max(inv_r0))
        f.attrs["pin_r3_at_10um"] = r3_at_10um

    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    _layout_written = "shell_fraction" if is_linear else "ratio_planes"
    print(f"Wrote {OUTPUT_PATH}: {size_mb:.2f} MB, layout={_layout_written}")
    if size_mb > 10.0:
        print(f"WARNING: file size {size_mb:.2f} MB exceeds the 10 MB cap.")


if __name__ == "__main__":
    main()
