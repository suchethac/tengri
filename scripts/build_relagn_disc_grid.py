#!/usr/bin/env python3
"""Build precomputed RELAGN outer-disc SED templates for tengri.

Uses the real RELAGN Python class (Hagen & Done 2023) with KYCONV (Dovciak,
Karas & Yaqoob 2004) per-annulus Kerr-metric ray-tracing.

Requires
--------
- conda env 'henv' with xspec + xspec-data installed
- vendor/relagn/relagn.py (RELAGN Python version)

Run
---
    conda run -n henv python scripts/build_relagn_disc_grid.py [options]

Grid axes
---------
    log_mbh  : log10(M / Msun), range [7, 10]
    log_mdot : log10(Mdot / Mdot_Edd), range [-1.5, 0.3]
    astar    : dimensionless spin, non-uniform nodes [-0.998 … 0.998]

Output
------
    data/relagn_disc_grid.h5
    Shape: (n_mass, n_mdot, n_astar, n_wave)
    SED units: erg/s/Hz at cos_inc = 0.5 (inclination-normalized out)
    Wavelengths: log-spaced [50, 1e5] Angstrom

References
----------
Dovciak, M., Karas, V., & Yaqoob, T. (2004).
ApJS, 153, 205. doi:10.1086/421115  [KYCONV]

Hagen, S. & Done, C. (2023).
MNRAS, 521, 251. doi:10.1093/mnras/stad478  [RELAGN]

Laor, A. & Netzer, H. (1989).
MNRAS, 238, 897. doi:10.1093/mnras/238.3.897  [r_sg]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

# ── Vendor path ───────────────────────────────────────────────────────────────
_VENDOR = Path(__file__).parent.parent / "vendor" / "relagn"
sys.path.insert(0, str(_VENDOR))

# ── Grid axes ─────────────────────────────────────────────────────────────────
# Prograde-only (a ≥ 0): KYCONV (idre) rejects retrograde spins.
# Non-uniform: denser near a=0 (ISCO transition) and near a=0.998 (extremal).
_ASTAR_GRID = np.array(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.87, 0.93, 0.96, 0.975, 0.990, 0.998],
    dtype=np.float64,
)

_C_AA_PER_S = 2.99792458e18  # Å/s


# ── Per-point SED computation ─────────────────────────────────────────────────


def _compute_one(args: tuple) -> tuple[tuple[int, int, int], np.ndarray]:
    """Compute one outer-disc SED using real RELAGN + KYCONV.

    Parameters
    ----------
    args : tuple
        (i_m, i_d, i_a, log_mbh, log_mdot, astar, wave_aa, dr_dex)

    Returns
    -------
    idx : tuple of int
        (i_m, i_d, i_a) grid indices
    lnu_cgs : ndarray, shape (n_wave,)
        Outer-disc L_nu in erg/s/Hz at cos_inc = 0.5 on wave_aa grid.
        All-zero if the computation fails.
    """
    i_m, i_d, i_a, log_mbh, log_mdot, astar, wave_aa, dr_dex = args

    from relagn import relagn  # late import: each worker process needs its own XSPEC state

    lnu_zero = np.zeros(len(wave_aa), dtype=np.float64)

    try:
        mod = relagn(
            M=10**log_mbh,
            dist=100.0,  # arbitrary; we store luminosity not flux
            log_mdot=log_mdot,
            a=float(astar),
            cos_inc=0.5,  # reference inclination
            kTe_hot=100.0,
            kTe_warm=0.2,
            gamma_hot=1.7,
            gamma_warm=2.7,
            r_hot=-1,  # no hot corona (set to r_isco)
            r_warm=-1,  # no warm Comptonisation (set to r_isco)
            log_rout=-1,  # self-gravity outer radius
            fcol=-1,  # Done+2012 colour correction
            h_max=10.0,
            z=0.0,
        )
        # Override radial resolution
        mod.change_rBins(dr_dex)

        # Relativistic disc SED: W/Hz on mod.nu_grid (Hz, increasing)
        lnu_w = mod.do_relDiscSpec()

        # Guard against all-zero (sub-Eddington disc below UV cutoff)
        if lnu_w is None or np.all(lnu_w <= 0):
            return (i_m, i_d, i_a), lnu_zero

        # Convert to erg/s/Hz (1 W = 1e7 erg/s)
        lnu_cgs = lnu_w * 1e7

        # Log-linear interpolation onto our wavelength grid
        # nu_grid is increasing in Hz; wave_aa is increasing in Å (nu decreasing)
        nu_target = _C_AA_PER_S / wave_aa  # decreasing

        log_nu = np.log10(mod.nu_grid)
        log_lnu = np.log10(np.clip(lnu_cgs, 1e10, None))  # floor at 1e10 erg/s/Hz

        # np.interp handles arbitrary x-values (nu_target need not be sorted)
        log_lnu_interp = np.interp(
            np.log10(nu_target),
            log_nu,
            log_lnu,
            left=log_lnu[0],
            right=log_lnu[-1],
        )
        return (i_m, i_d, i_a), 10.0**log_lnu_interp

    except Exception:  # broad catch: log failure, fill zeros, continue grid build
        print(
            f"  FAILED (lm={log_mbh:.2f} ld={log_mdot:.2f} a={astar:.3f}): "
            f"{traceback.format_exc().splitlines()[-1]}",
            flush=True,
        )
        return (i_m, i_d, i_a), lnu_zero


# ── Grid builder ──────────────────────────────────────────────────────────────


def build_grid(
    n_mass: int = 20,
    n_mdot: int = 20,
    n_wave: int = 600,
    dr_dex: int = 25,
    n_workers: int = 4,
) -> dict:
    """Compute the full (n_mass, n_mdot, n_astar, n_wave) grid.

    Parameters
    ----------
    n_mass : int
        Number of log_mbh nodes in [7, 10].
    n_mdot : int
        Number of log_mdot nodes in [-1.5, 0.3].
    n_wave : int
        Number of wavelength nodes in [50, 1e5] Å.
    dr_dex : int
        Radial bins per decade. RELAGN default = 50. Lower = faster, coarser.
    n_workers : int
        Number of parallel worker processes.

    Returns
    -------
    dict with keys: grid, log_mbh, log_mdot, astar, wavelength_aa
    """
    log_mbh = np.linspace(7.0, 10.0, n_mass)
    log_mdot = np.linspace(-1.5, 0.3, n_mdot)
    astar = _ASTAR_GRID
    n_astar = len(astar)
    wave_aa = np.logspace(np.log10(50.0), np.log10(1e5), n_wave, dtype=np.float64)

    grid = np.zeros((n_mass, n_mdot, n_astar, n_wave), dtype=np.float64)

    # Build task list
    tasks = [
        (i_m, i_d, i_a, log_mbh[i_m], log_mdot[i_d], astar[i_a], wave_aa, dr_dex)
        for i_m in range(n_mass)
        for i_d in range(n_mdot)
        for i_a in range(n_astar)
    ]
    total = len(tasks)
    print(f"Grid: {n_mass}×{n_mdot}×{n_astar}×{n_wave} = {total} SEDs", flush=True)
    print(f"Workers: {n_workers}, dr_dex: {dr_dex}", flush=True)

    t0 = time.time()
    done = 0

    # Each worker process loads XSPEC once and processes multiple points.
    # We chunk tasks to amortize the XSPEC startup cost.
    ctx = mp.get_context("spawn")  # spawn is required for XSPEC safety
    with ctx.Pool(processes=n_workers) as pool:
        for idx, lnu in pool.imap_unordered(_compute_one, tasks, chunksize=1):
            i_m, i_d, i_a = idx
            grid[i_m, i_d, i_a, :] = lnu
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(
                    f"  {done}/{total} ({100 * done / total:.0f}%) "
                    f"elapsed {elapsed:.0f}s ETA {eta:.0f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"Completed {total} SEDs in {elapsed:.0f}s ({elapsed / total:.1f}s/SED)", flush=True)

    return {
        "grid": grid,
        "log_mbh": log_mbh,
        "log_mdot": log_mdot,
        "astar": astar,
        "wavelength_aa": wave_aa,
    }


# ── HDF5 writer ───────────────────────────────────────────────────────────────


def write_h5(data: dict, path: Path) -> None:
    """Write grid to HDF5.

    Parameters
    ----------
    data : dict
        Output of :func:`build_grid`.
    path : Path
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "lnu_disc",
            data=data["grid"],
            compression="gzip",
            compression_opts=6,
        )
        ds.attrs["units"] = "erg/s/Hz"
        ds.attrs["cos_inc"] = 0.5
        ds.attrs["dims"] = "log_mbh, log_mdot, astar, wavelength_aa"
        ds.attrs["model"] = "RELAGN outer disc (Hagen & Done 2023) + KYCONV"

        f.create_dataset("log_mbh", data=data["log_mbh"], dtype=np.float64)
        f["log_mbh"].attrs["units"] = "log10(M/Msun)"

        f.create_dataset("log_mdot", data=data["log_mdot"], dtype=np.float64)
        f["log_mdot"].attrs["units"] = "log10(Mdot/Mdot_Edd)"

        f.create_dataset("astar", data=data["astar"], dtype=np.float64)
        f["astar"].attrs["units"] = "dimensionless"

        f.create_dataset("wavelength_aa", data=data["wavelength_aa"], dtype=np.float64)
        f["wavelength_aa"].attrs["units"] = "Angstrom"
        f["wavelength_aa"].attrs["note"] = (
            "float64; do not cast to float32 (nu^3 overflows at lambda < 2000 AA)"
        )

    size_mb = path.stat().st_size / 1e6
    print(f"Wrote {path}  ({size_mb:.1f} MB)", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="data/relagn_disc_grid.h5",
        help="Output HDF5 path (default: data/relagn_disc_grid.h5)",
    )
    p.add_argument("--n-mass", type=int, default=20, help="log_mbh grid nodes (default 20)")
    p.add_argument("--n-mdot", type=int, default=20, help="log_mdot grid nodes (default 20)")
    p.add_argument("--n-wave", type=int, default=600, help="Wavelength nodes (default 600)")
    p.add_argument(
        "--dr-dex",
        type=int,
        default=25,
        help="Radial bins per decade (default 25; RELAGN default 50)",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=max(1, mp.cpu_count() - 2),
        help="Parallel worker processes (default: n_cpu - 2)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Verify RELAGN is importable before spawning workers
    try:
        from relagn import relagn as _  # noqa: F401
    except ImportError as e:
        sys.exit(f"Cannot import relagn: {e}\nEnsure vendor/relagn/ is present.")

    out = Path(args.output)
    data = build_grid(
        n_mass=args.n_mass,
        n_mdot=args.n_mdot,
        n_wave=args.n_wave,
        dr_dex=args.dr_dex,
        n_workers=args.n_workers,
    )
    write_h5(data, out)


if __name__ == "__main__":
    main()
