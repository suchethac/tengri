#!/usr/bin/env python3
"""Benchmark triweight z-table redshift interpolation accuracy.

Quantifies the accuracy of `precompute_photometry_ztable` + `interpolate_ztable_smooth`
relative to calling `precompute_photometry` (exact dense trapz) at every z.

Uses triweight kernel smooth interpolation (Horner-form CDF) for C²-continuous
d(flux)/dz gradients — preferred for VI/MAP fitting.

Three test scenarios:
  (A) Power-law spectrum (λ^-2): smooth, best-case for interpolation
  (B) Spectrum with Lyman break: sharp UV cutoff at 912 Å, worst-case for
      high-z filters where the break sweeps through the bandpass
  (C) Spectrum with Balmer/4000Å break: more typical galaxy SED, intermediate

For each scenario and n_z value, reports:
  - Max fractional error in ssp_phot (across all Z, age, filter, test-z)
  - Mean fractional error
  - Worst z
  - On-grid error (nonzero due to kernel spread)

Usage
-----
    python bench/scripts/benchmark_ztable_interp.py

    # custom n_z sweep:
    python bench/scripts/benchmark_ztable_interp.py --n-z 25 50 100 200

    # custom z range:
    python bench/scripts/benchmark_ztable_interp.py --z-max 6.0

    # with real SDSS filters (default uses synthetic box filters):
    python bench/scripts/benchmark_ztable_interp.py --real-filters
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Synthetic SSP + filter helpers
# ---------------------------------------------------------------------------

_WAVE_GRID = np.linspace(500.0, 12000.0, 500)  # Å, rest frame
_N_MET = 3
_N_AGE = 10


def _power_law_ssp() -> np.ndarray:
    """λ^-2 power-law SED; smooth, best-case for interpolation."""
    spec = (_WAVE_GRID / 5500.0) ** (-2.0)
    return np.broadcast_to(spec, (_N_MET, _N_AGE, len(_WAVE_GRID))).copy() * (
        1.0 + 0.1 * np.arange(_N_MET)[:, None, None]
    )


def _lyman_break_ssp() -> np.ndarray:
    """Spectrum with a Lyman break at 912 Å (hard cut below in rest frame).

    Worst case for high-z photometry: the break sweeps through UV filters.
    """
    spec = (_WAVE_GRID / 5500.0) ** (-1.5)
    spec[_WAVE_GRID < 912.0] = 0.0
    # Add a Lyman-α bump
    dlam = _WAVE_GRID - 1216.0
    spec += 0.3 * np.exp(-0.5 * (dlam / 30.0) ** 2)
    return np.broadcast_to(spec, (_N_MET, _N_AGE, len(_WAVE_GRID))).copy() * (
        1.0 + 0.05 * np.arange(_N_AGE)[None, :, None]
    )


def _balmer_break_ssp() -> np.ndarray:
    """Spectrum with a 4000Å Balmer break; typical evolved galaxy SED."""
    blue = 0.3 * (_WAVE_GRID / 4000.0) ** 2
    red = (_WAVE_GRID / 5500.0) ** (-0.5)
    spec = np.where(_WAVE_GRID > 4000.0, red, blue)
    # Add Hα emission line
    spec += 0.05 * np.exp(-0.5 * ((_WAVE_GRID - 6563.0) / 20.0) ** 2)
    return np.broadcast_to(spec, (_N_MET, _N_AGE, len(_WAVE_GRID))).copy() * (
        1.0 + 0.08 * np.arange(_N_MET)[:, None, None]
    )


def _make_ssp(spectrum_type: str):
    """Build a minimal SSPData-like object."""

    class SSPData(NamedTuple):
        ssp_wave: jnp.ndarray
        ssp_flux: jnp.ndarray
        ssp_lg_age_gyr: jnp.ndarray
        ssp_lgmet: jnp.ndarray

    _ssp_builders = {
        "power_law": _power_law_ssp,
        "lyman_break": _lyman_break_ssp,
        "balmer_break": _balmer_break_ssp,
    }
    flux = _ssp_builders[spectrum_type]()
    return SSPData(
        ssp_wave=jnp.array(_WAVE_GRID),
        ssp_flux=jnp.array(flux),
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.0, _N_AGE),
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )


def _make_box_filters(n_filters: int = 5, z_max: float = 3.0) -> tuple[list, list]:
    """Box-top filters spanning UV–NIR, scaled to cover typical observed wavelengths."""
    # Place filter centres at rest-frame wavelengths (observed frame = rest × (1+z))
    # Use fixed observed-frame wavelengths typical of SDSS+2MASS
    centres_obs = np.array([3700.0, 4800.0, 6200.0, 8000.0, 12000.0])[:n_filters]
    widths = centres_obs * 0.18  # ~18% bandwidth each
    fw = [np.linspace(c - w / 2, c + w / 2, 40) for c, w in zip(centres_obs, widths)]
    ft = [np.ones(40) for _ in fw]
    return [jnp.array(f) for f in fw], [jnp.array(f) for f in ft]


def _try_real_filters(filter_names: list[str]) -> tuple[list, list] | None:
    """Attempt to load real filters via tengri; return None if not available."""
    try:
        from tengri.observation.photometry import load_filter

        fw, ft = [], []
        for name in filter_names:
            w, t = load_filter(name)
            fw.append(w)
            ft.append(t)
        return fw, ft
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------


def compute_errors(
    ssp_data,
    filter_waves: list,
    filter_trans: list,
    n_z: int,
    z_min: float,
    z_max: float,
    n_probe: int = 200,
    scatter_factor: float = 1.5,
) -> dict:
    """Run one (n_z, spectrum) configuration and return error statistics.

    Uses triweight kernel smooth interpolation (Horner-form CDF) for
    C²-continuous d(flux)/dz gradients.

    Parameters
    ----------
    ssp_data : SSPData-like
    filter_waves, filter_trans : filter definitions
    n_z : int
        Number of z-table grid points.
    z_min, z_max : float
        Redshift range.
    n_probe : int
        Number of test z values (uniformly distributed, includes midpoints).
    scatter_factor : float
        Triweight kernel bandwidth as a multiple of the grid spacing ``dz``.
        Default: 1.5.

    Returns
    -------
    dict with keys:
        max_err, mean_err, median_err,
        worst_z,
        on_grid_max_err,
        all_max_errs : array of shape (n_probe,) — max over (met, age, filt) per z
        probe_zs : array of shape (n_probe,)
    """
    from tengri.sps.precompute import (
        interpolate_ztable_smooth,
        precompute_photometry,
        precompute_photometry_ztable,
    )
    from tengri.utils.cosmology import luminosity_distance

    # Build z-table
    ztable = precompute_photometry_ztable(
        ssp_data, filter_waves, filter_trans, z_min=z_min, z_max=z_max, n_z=n_z
    )
    z_grid = np.asarray(ztable.z_grid)
    scatter = scatter_factor * float(z_grid[1] - z_grid[0])

    # Probe at midpoints between consecutive grid nodes: true worst case for
    # linear interpolation. Avoids the artifact where linspace(n_probe) == linspace(n_z).
    midpoints = 0.5 * (z_grid[:-1] + z_grid[1:])
    # Supplement with some off-grid random offsets to get a denser picture
    rng = np.random.default_rng(42)
    n_extra = min(n_probe, len(z_grid) - 1)
    offsets = rng.uniform(0.1, 0.9, size=n_extra) * (z_grid[1 : n_extra + 1] - z_grid[:n_extra])
    extra = z_grid[:n_extra] + offsets
    probe_zs = np.unique(np.concatenate([midpoints, extra]))

    # Also record on-grid errors (subsample 10 grid points)
    on_grid_sample = z_grid[:: max(1, n_z // 10)]

    all_max_errs = np.zeros(len(probe_zs))
    all_mean_errs = np.zeros(len(probe_zs))

    for i, z_test in enumerate(probe_zs):
        dl_cm = float(luminosity_distance(z_test))
        exact = precompute_photometry(ssp_data, filter_waves, filter_trans, float(z_test), dl_cm)
        ssp_exact = np.asarray(exact.ssp_phot)
        ssp_interp = np.asarray(
            interpolate_ztable_smooth(
                ztable.ssp_phot_table,
                ztable.eff_waves_rest_table,
                ztable.log10_flux_scale_table,
                ztable.z_grid,
                z_test,
                scatter,
            )[0]
        )

        denom = np.maximum(np.abs(ssp_exact), 1e-30)
        frac_err = np.abs(ssp_interp - ssp_exact) / denom
        all_max_errs[i] = frac_err.max()
        all_mean_errs[i] = frac_err.mean()

    # On-grid error check
    on_grid_errs = []
    for z_test in on_grid_sample:
        dl_cm = float(luminosity_distance(float(z_test)))
        exact = precompute_photometry(ssp_data, filter_waves, filter_trans, float(z_test), dl_cm)
        ssp_exact = np.asarray(exact.ssp_phot)
        ssp_interp = np.asarray(
            interpolate_ztable_smooth(
                ztable.ssp_phot_table,
                ztable.eff_waves_rest_table,
                ztable.log10_flux_scale_table,
                ztable.z_grid,
                float(z_test),
                scatter,
            )[0]
        )
        denom = np.maximum(np.abs(ssp_exact), 1e-30)
        on_grid_errs.append(float((np.abs(ssp_interp - ssp_exact) / denom).max()))

    worst_idx = int(all_max_errs.argmax())
    return {
        "max_err": float(all_max_errs.max()),
        "mean_err": float(all_max_errs.mean()),
        "median_err": float(np.median(all_max_errs)),
        "worst_z": float(probe_zs[worst_idx]),
        "on_grid_max_err": float(max(on_grid_errs)) if on_grid_errs else 0.0,
        "all_max_errs": all_max_errs,
        "all_mean_errs": all_mean_errs,
        "probe_zs": probe_zs,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_COL = 14

_SCENARIOS = [
    ("power_law", "Power-law λ^-2 (smooth; best case)"),
    ("balmer_break", "Balmer/4000Å break (typical galaxy SED)"),
    ("lyman_break", "Lyman break (sharp UV cutoff; worst case)"),
]


def print_n_z_sweep(
    results: dict[tuple, dict],
    n_z_vals: list[int],
) -> None:
    """Print accuracy table for triweight smooth interpolation."""
    for stype, sdesc in _SCENARIOS:
        print(f"\n{'─' * 80}")
        print(f"  {sdesc}")
        print(f"{'─' * 80}")
        header = (
            f"{'n_z':>6}  {'max err':>{_COL}}  {'mean err':>{_COL}}"
            f"  {'median err':>{_COL}}  {'worst z':>8}  {'on-grid err':>{_COL}}"
        )
        print(header)
        print("─" * len(header))
        for nz in n_z_vals:
            r = results[(stype, nz)]
            max_err = r["max_err"] * 100
            mean_err = r["mean_err"] * 100
            median_err = r["median_err"] * 100
            worst_z = r["worst_z"]
            on_grid_err = r["on_grid_max_err"] * 100
            print(
                f"{nz:>6}  {max_err:>{_COL}.2f}%  {mean_err:>{_COL}.2f}%"
                f"  {median_err:>{_COL}.2f}%  {worst_z:>8.3f}  {on_grid_err:>{_COL}.2e}%"
            )


def print_z_profile(
    results: dict[tuple, dict],
    n_z: int,
) -> None:
    """Print z-profile of triweight interpolation error at a fixed n_z."""
    print(f"\n{'─' * 72}")
    print(f"  Error vs z — triweight smooth | n_z={n_z} (max over all met, age, filters)")
    print(f"{'─' * 72}")
    header = f"{'z':>8}" + "".join(f"  {s[0]:>{_COL}}" for s in _SCENARIOS)
    print(header)
    print("─" * len(header))
    probe_zs = results[(_SCENARIOS[0][0], n_z)]["probe_zs"]
    step = max(1, len(probe_zs) // 20)
    for i in range(0, len(probe_zs), step):
        row = f"{probe_zs[i]:>8.3f}"
        for stype, _ in _SCENARIOS:
            err_pct = results[(stype, n_z)]["all_max_errs"][i] * 100
            row += f"  {err_pct:>{_COL}.2f}%"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-z",
        nargs="+",
        type=int,
        default=[25, 50, 100, 200, 500],
        metavar="N",
        help="n_z values to benchmark (default: 25 50 100 200 500)",
    )
    parser.add_argument("--z-min", type=float, default=0.01, help="Min redshift (default: 0.01)")
    parser.add_argument("--z-max", type=float, default=3.0, help="Max redshift (default: 3.0)")
    parser.add_argument("--n-probe", type=int, default=200, help="Test z values (default: 200)")
    parser.add_argument(
        "--real-filters",
        action="store_true",
        help="Try to load real SDSS ugriz filters (falls back to box filters)",
    )
    parser.add_argument(
        "--profile-n-z",
        type=int,
        default=100,
        metavar="N",
        help="n_z to use for the z-profile plot (default: 100)",
    )
    args = parser.parse_args()

    # Ensure profile n_z is in the sweep
    n_z_vals = sorted(set(args.n_z) | {args.profile_n_z})

    print("tengri triweight z-table interpolation accuracy benchmark")
    print(f"z range: {args.z_min} – {args.z_max} | n_probe={args.n_probe}")

    # Filters
    if args.real_filters:
        real = _try_real_filters(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
        if real is not None:
            fw, ft = real
            print("Using real SDSS ugriz filters.")
        else:
            fw, ft = _make_box_filters()
            print("Real filters not found — using synthetic box filters.")
    else:
        fw, ft = _make_box_filters()
        print("Using synthetic box filters (5 bands, UV–NIR).")

    print()

    # Run all (scenario, n_z) combinations
    results: dict[tuple, dict] = {}
    n_combos = len(_SCENARIOS) * len(n_z_vals)
    done = 0
    for stype, _sdesc in _SCENARIOS:
        ssp = _make_ssp(stype)
        for nz in n_z_vals:
            print(
                f"  [{done + 1}/{n_combos}] {stype}, n_z={nz} ...",
                end=" ",
                flush=True,
            )
            r = compute_errors(
                ssp,
                fw,
                ft,
                n_z=nz,
                z_min=args.z_min,
                z_max=args.z_max,
                n_probe=args.n_probe,
            )
            results[(stype, nz)] = r
            print(f"max={r['max_err'] * 100:.2f}%")
            done += 1

    # Tables
    print_n_z_sweep(results, n_z_vals)
    print_z_profile(results, args.profile_n_z)

    print(f"\n{'─' * 80}")
    print("Notes:")
    print("  'max err' = max fractional |smooth_interp - exact| over all z, met, age, filter")
    print("  'mean err' = mean of per-z max errors")
    print("  'median err' = median of per-z max errors")
    print("  'worst z' = redshift with largest error")
    print("  'on-grid err' = smooth error at grid nodes (nonzero due to kernel spread)")
    print(f"  Default tengri n_z = 100 (linspace {args.z_min}–{args.z_max})")
    print("  Smooth interpolation uses triweight kernel (scatter = 1.5 × dz)")
    print("  Gives C²-continuous d(flux)/dz — preferred for VI/MAP fitting.")
    print(f"{'─' * 80}\n")


if __name__ == "__main__":
    main()
