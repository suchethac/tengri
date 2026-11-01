# SPDX-License-Identifier: BSD-3-Clause
"""Matched-physics check of the BEAGLE comparison: the same inputs on both sides.

Companion to ``01_beagle.py``. The notebook changes configuration on purpose, so an
offset there mixes physics with input choices. This script removes the input choices.
Every tengri build takes BEAGLE's own input row for row: the delayed star-formation
history with BEAGLE's 10 Myr current episode, the metallicity, the redshift and, for
section s5, the CF00 screen with its tau_V. The two rest-frame luminosity densities
are then compared on BEAGLE's wavelength grid and through the CANDELS filter curves
stored in ``data/beagle_workshop_catalog.h5``.

Section s3 is the no-dust fiducial with nebular emission off on both sides. Its table
sets the exit status: a band whose BEAGLE/tengri ratio differs from unity by more than
``--tol`` exits 1. Section s5 (CF00 at tau_V 0.3, 1 and 2, mu 0.4) is printed only.

The residual in the blue is the library bracket. BEAGLE reads the MILES-based
``bc2003_hr_xmiless`` family and tengri the STELIB ``bc03_pdva`` grid, so the
difference is between two libraries and not a configuration offset.

Run after the cache is populated (no Docker is started):

    BEAGLE_WORKDIR=~/beagle/BEAGLE-general JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
        .venv/bin/python reproduction/beagle/validate_matched_physics.py [--tol 0.02]

Exit status: 0 when the s3 fiducial agrees within ``--tol``; 1 when it does not;
2 when a prerequisite is missing (the workdir, the SSP grid, the catalog or the s3 cache).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
from reproduction.beagle._drivers import beagle_run as R
from reproduction.beagle._drivers.beagle_driver import LOG10_ZSUN_SHIFT
from reproduction.beagle._drivers.beagle_matrix import by_section

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.utils.cosmology import age_at_z

REPO = Path(__file__).resolve().parents[2]
SSP_NAME = "bc03_pdva_stelib_chabrier"
SSP_PATH = REPO / "data" / f"{SSP_NAME}.h5"
CATALOG_PATH = REPO / "data" / "beagle_workshop_catalog.h5"
FIDUCIAL = "s3_fiducial"
ATTENUATED = "s5_cf00_tauv"
RECENT_MYR = 10.0
REST_RANGE_AA = (1500.0, 30000.0)
# Bands whose rest-frame pivot lies below this sample theoretical atmospheres shared by both
# libraries; the offset there is gated. Redward, the empirical libraries differ and the offset
# is reported as the library bracket.
SHARED_ATMOSPHERE_LIMIT_AA = 3500.0
# (band, pivot obs [um], pivot rest [A], band-mean L_beagle, band-mean L_tengri, ratio)
Row = tuple[str, float, float, float, float, float]


def _beagle_age_gyr(beagle, k: int) -> float:
    """Stellar age of object ``k`` as BEAGLE used it, read from its own SFH output.

    The input table leaves ``max_stellar_age`` unset, and BEAGLE's SFH output ends with
    a bin at the age of the universe at the redshift, after which the SFR is zero.
    """
    sfr = np.asarray(beagle.sfh["SFR"][k])
    lookback = np.asarray(beagle.sfh["lookback_age"][k])
    last_active = int(np.flatnonzero(sfr > 0).max())
    return float(lookback[last_active + 1]) / 1.0e9


def _sfh(row: dict, age_gyr: float) -> dict:
    """Delayed history plus BEAGLE's 10 Myr current episode, in tengri's grammar.

    BEAGLE's ``mass`` includes the episode, so the smooth part carries the formed mass
    less the episode, M - SFR * 10 Myr (docs/model_reference/sfh.md, mass bookkeeping).
    """
    episode_msun = 10.0 ** row["sfr"] * RECENT_MYR * 1e6
    smooth_msun = 10.0 ** row["mass"] - episode_msun
    return {
        "type": ["delayed", "recent"],
        "tau_gyr": Fixed(10.0 ** (row["tau"] - 9.0)),
        "age_gyr": Fixed(age_gyr),
        "log_total_mass": Fixed(math.log10(smooth_msun)),
        "log_sfr": Fixed(row["sfr"]),
        "timescale_myr": Fixed(RECENT_MYR),
        "all_params": Fixed(DEFAULT),
    }


def _dust(row: dict, section: str) -> dict:
    """No dust for s3; CF00 with BEAGLE's tau_V and mu 0.4 and power-law slopes for s5."""
    if section == FIDUCIAL:
        return {"type": "none"}
    return {
        "type": "two_component",
        "parameterization": "tauv_mu",
        "tau_v": Fixed(row["tauV_eff"]),
        "mu": Fixed(0.4),
        "law_bc": "power_law",
        "law_diff": "power_law",
        "slope_bc": Fixed(-1.3),
        "slope_diff": Fixed(-0.7),
        "t_birth_yr": 1e7,
        "other_params": Fixed(DEFAULT),
    }


def _predict(ssp, row: dict, section: str, age_gyr: float) -> tuple[np.ndarray, np.ndarray]:
    """Rest-frame L_nu [erg/s/Hz] of one BEAGLE input row, built as a tengri model."""
    model = SEDModel.build(
        ssp_data=ssp,
        met={
            "logzsol": Fixed(row["metallicity"] + LOG10_ZSUN_SHIFT),
            "all_params": Fixed(DEFAULT),
        },
        sfh=_sfh(row, age_gyr),
        dust_attenuation=_dust(row, section),
        dust_emission={"type": "none"},
        neb={"type": "none"},
        igm={"type": "none"},
        redshift=Fixed(row["redshift"]),
    )
    state = model.predict_state({})
    return np.asarray(state.wave, dtype=float), np.asarray(state.sed_intrinsic, dtype=float)


def _log_interp(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """Linear interpolation in log-log space; NaN outside the positive support of ``yp``."""
    order = np.argsort(xp)
    xp, yp = xp[order], yp[order]
    ok = yp > 0
    out = np.full(x.shape, np.nan)
    lo, hi = xp[ok].min(), xp[ok].max()
    inside = (x >= lo) & (x <= hi)
    out[inside] = 10.0 ** np.interp(np.log10(x[inside]), np.log10(xp[ok]), np.log10(yp[ok]))
    return out


def _bands(z: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(band, curve wavelength [A, observed], response) for the CANDELS bands at redshift ``z``."""
    with h5py.File(CATALOG_PATH, "r") as f:
        group = f[f"catalog_z{z}"]
        bands = json.loads(group.attrs["bands"])
        curve_for = json.loads(group.attrs["curve_for_band"])
        out = []
        for band in bands:
            curve = curve_for[band]
            wave = np.asarray(f[f"filters/{curve}/wave_aa"], dtype=float)
            resp = np.asarray(f[f"filters/{curve}/response"], dtype=float)
            out.append((band, wave, resp))
    return out


def _band_mean(
    wave_obs: np.ndarray, resp: np.ndarray, wave_rest: np.ndarray, lnu: np.ndarray, z: float
) -> tuple[float, float]:
    """Band-mean L_nu and pivot wavelength [um], both on the observed-frame band samples.

    The mean weights L_nu by R(lambda) dlambda/lambda (the AB-convention weight), with
    the trapezoid rule on the filter's own samples. Each SED is read at the rest
    wavelength lambda_obs / (1 + z). A band that leaves either coverage gives NaN.
    """
    keep = resp > 0
    w, r = wave_obs[keep], resp[keep]
    values = _log_interp(w / (1.0 + z), wave_rest, lnu)
    if not np.all(np.isfinite(values)):
        return math.nan, math.nan
    weight = r / w
    mean = np.trapezoid(values * weight, w) / np.trapezoid(weight, w)
    pivot = math.sqrt(np.trapezoid(r * w, w) / np.trapezoid(r / w, w))
    return float(mean), pivot / 1.0e4


def _band_table(z: float, wave_b, lnu_b, wave_t, lnu_t) -> list[Row]:
    rows = []
    for band, wave, resp in _bands(round(z)):
        beagle = _band_mean(wave, resp, wave_b, lnu_b, z)
        tengri = _band_mean(wave, resp, wave_t, lnu_t, z)
        both = np.isfinite(beagle[0]) and np.isfinite(tengri[0])
        ratio = beagle[0] / tengri[0] if both else math.nan
        rest_pivot = beagle[1] * 1.0e4 / (1.0 + z)
        rows.append((band, beagle[1], rest_pivot, beagle[0], tengri[0], ratio))
    return rows


def _is_shared(row: Row, strict: bool) -> bool:
    """True when the band's offset is gated: shared-atmosphere band, or --strict."""
    return strict or row[2] < SHARED_ATMOSPHERE_LIMIT_AA


def _print_band_table(title: str, rows: list[Row], strict: bool) -> tuple[float, list[float]]:
    """Print the band table; return (largest gated |ratio - 1|, library-bracket offsets).

    Offsets are ratios minus one. The library bracket is the set of bands not gated.
    """
    print(title)
    print(f"  {'band':<12s} {'pivot_um':>8s} {'rest_A':>8s} {'B/t':>8s} {'dev_%':>7s}  class")
    worst_gated = 0.0
    bracket = []
    for band, pivot, rest, lb, lt, ratio in rows:
        if not np.isfinite(ratio):
            print(f"  {band:<12s} {pivot:8.3f} {rest:8.0f} {'not covered':>8s}")
            continue
        gated = _is_shared((band, pivot, rest, lb, lt, ratio), strict)
        label = "gated" if gated else "library bracket"
        print(
            f"  {band:<12s} {pivot:8.3f} {rest:8.0f} {ratio:8.4f} "
            f"{100.0 * (ratio - 1.0):+7.2f}  {label}"
        )
        if gated:
            worst_gated = max(worst_gated, abs(ratio - 1.0))
        else:
            bracket.append(ratio - 1.0)
    return worst_gated, bracket


def _sed_percentiles(wave_b, lnu_b, wave_t, lnu_t) -> np.ndarray:
    """16, 50 and 84 percentiles of L_beagle / L_tengri on BEAGLE's grid in REST_RANGE_AA."""
    sel = (wave_b >= REST_RANGE_AA[0]) & (wave_b <= REST_RANGE_AA[1])
    tengri_on_b = _log_interp(wave_b[sel], wave_t, lnu_t)
    ratio = lnu_b[sel] / tengri_on_b
    return np.percentile(ratio[np.isfinite(ratio)], [16, 50, 84])


def _check_prerequisites() -> int:
    try:
        R.workdir()
    except ValueError as exc:
        print(f"validate_matched_physics: {exc}", file=sys.stderr)
        return 2
    if not SSP_PATH.is_file():
        print(
            f"validate_matched_physics: missing SSP grid {SSP_PATH}; fetch it with "
            f"tengri.download_ssp('{SSP_NAME}', dest='{SSP_PATH.parent}')",
            file=sys.stderr,
        )
        return 2
    if not CATALOG_PATH.is_file():
        print(
            f"validate_matched_physics: missing committed catalog {CATALOG_PATH}", file=sys.stderr
        )
        return 2
    return 0


def _run_section(ssp, section: str, tol: float, strict: bool) -> tuple[str, float | None]:
    """Print the band tables for one section; return (status, worst gated deviation)."""
    spec = by_section(section)[0]
    try:
        beagle = R.run(spec)
    except RuntimeError as exc:
        print(f"[{section}] section not cached: {exc}")
        return "missing", None
    worst_all = 0.0
    bracket_all: list[float] = []
    for k, row in enumerate(spec.rows):
        # BEAGLE's grid bin sits 0.3 % older than tengri's age of the universe (Planck18).
        # tengri refuses an onset older than that age, so the age is clipped to it.
        age_gyr = min(_beagle_age_gyr(beagle, k), float(age_at_z(row["redshift"])))
        wave_t, lnu_t = _predict(ssp, row, section, age_gyr)
        wave_b, lnu_b = beagle.wl_rest_aa, beagle.sed_lnu[k]
        pct = _sed_percentiles(wave_b, lnu_b, wave_t, lnu_t)
        title = (
            f"[{section}] row {k}: z={row['redshift']:.1f} tau={10 ** (row['tau'] - 9):.2f} Gyr "
            f"age={age_gyr:.3f} Gyr "
            f"log M={row['mass']:.1f} log Z/Zsun(BEAGLE)={row['metallicity']:+.3f}"
            + (f" tau_V={row['tauV_eff']:.1f} mu=0.4" if "tauV_eff" in row else "")
        )
        table = _band_table(row["redshift"], wave_b, lnu_b, wave_t, lnu_t)
        worst, bracket = _print_band_table(title, table, strict)
        worst_all = max(worst_all, worst)
        bracket_all.extend(bracket)
        print(
            f"  SED ratio L_beagle/L_tengri over "
            f"{REST_RANGE_AA[0]:.0f}-{REST_RANGE_AA[1]:.0f} A rest: "
            f"median {pct[1]:.4f}, 16-84 % [{pct[0]:.4f}, {pct[2]:.4f}]"
        )
    if bracket_all:
        off = 100.0 * np.asarray(bracket_all)
        print(
            f"  optical-NIR offset: median {np.median(off):+.2f} %, "
            f"range [{off.min():+.2f}, {off.max():+.2f}] % (library bracket, not gated)"
        )
    status = "pass" if worst_all <= tol else "fail"
    return status, worst_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tol",
        type=float,
        default=0.02,
        help="largest allowed |BEAGLE/tengri - 1| for a gated band (default 0.02)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="gate every band at --tol, not only rest pivots below 3500 A",
    )
    args = parser.parse_args(argv)
    status = _check_prerequisites()
    if status:
        return status
    ssp = load_ssp_data(str(SSP_PATH))
    fiducial_status, fiducial_worst = _run_section(ssp, FIDUCIAL, args.tol, args.strict)
    _run_section(ssp, ATTENUATED, args.tol, args.strict)
    if fiducial_status == "missing":
        print(
            f"validate_matched_physics: {FIDUCIAL} is required for the exit status",
            file=sys.stderr,
        )
        return 2
    print(
        f"{FIDUCIAL}: largest gated band deviation {100.0 * fiducial_worst:.2f} % "
        f"against tolerance {100.0 * args.tol:.1f} % "
        f"({'all bands' if args.strict else 'rest pivot < 3500 A'}) -> {fiducial_status}"
    )
    return 0 if fiducial_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
