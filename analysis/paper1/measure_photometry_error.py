#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Relative photometry error of each dust-factorization scheme.

Appendix Table ``tab:photometry_error`` reports these numbers and its caption
names this script. The script did not exist: neither it nor
``results/photometry_error_table.json`` was ever committed on any branch, and
the values are not in the bench report cited beside them. So the table's
numbers had no reproducible source. This regenerates them from the
specification the caption itself gives, so they can be checked rather than
trusted.

The test, exactly as the appendix states it: a :math:`\\lambda^{-2}` power-law
SSP, Charlot--Fall dust with :math:`\\tau_{\\rm BC}=1` and :math:`n=-0.7`, and a
dense trapezoidal integral of :math:`{\\rm SSP}\\times A\\times T\\,w` as the
reference. It is a synthetic worst case, not a real SSP: the point is to
stress the factorization where the attenuation varies most across a bandpass.

Three schemes, matching ``WavePrecomp(band_integration=...)``:

``A.Phi``
    One transmission-weighted effective wavelength; the attenuation is
    evaluated there and multiplies the bare band flux.
``Taylor``
    Adds the first-order term in the attenuation's variation across the band.
``K_b``
    Equal-mass sub-band quadrature: the band is split into K sub-bands of
    equal integrated weight and the attenuation is evaluated at each
    sub-band's own effective wavelength.

CLI:
    python analysis/paper1/measure_photometry_error.py [--out JSON]
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "analysis" / "paper1" / "results" / "photometry_error_table.json"

# The worst case the appendix specifies.
TAU_BC = 1.0
DUST_SLOPE = -0.7
PIVOT_ANGSTROM = 5500.0
SSP_POWER = -2.0
SUBBAND_COUNTS = (1, 3, 5, 8)
DENSE_NODES = 20001

FILTER_GROUPS: dict[str, list[tuple[str, str]]] = {
    "SDSS": [
        ("u", "sdss_u"),
        ("g", "sdss_g"),
        ("r", "sdss_r"),
        ("i", "sdss_i"),
        ("z", "sdss_z"),
    ],
    "LSST": [
        ("g", "lsst_g"),
        ("r", "lsst_r"),
        ("i", "lsst_i"),
    ],
    "GALEX": [
        ("FUV", "galex_fuv"),
        ("NUV", "galex_nuv"),
    ],
}


def attenuation(wave: np.ndarray) -> np.ndarray:
    """Charlot & Fall screen, :math:`A=\\exp(-\\tau(\\lambda/5500)^{n})`."""
    return np.exp(-TAU_BC * (wave / PIVOT_ANGSTROM) ** DUST_SLOPE)


def d_attenuation(wave: np.ndarray) -> np.ndarray:
    """dA/dlambda, for the first-order correction."""
    power = (wave / PIVOT_ANGSTROM) ** DUST_SLOPE
    return attenuation(wave) * (-TAU_BC * DUST_SLOPE * power / wave)


def ssp(wave: np.ndarray) -> np.ndarray:
    return wave**SSP_POWER


def _dense(wave: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample a filter curve onto a dense uniform grid over its support."""
    keep = trans > 0.0
    lo, hi = float(wave[keep].min()), float(wave[keep].max())
    fine = np.linspace(lo, hi, DENSE_NODES)
    return fine, np.interp(fine, wave, trans)


def _weight(wave: np.ndarray, trans: np.ndarray, convention: str) -> np.ndarray:
    """The measure the band integral is taken against.

    A photon-counting detector weights by an extra factor of wavelength; an
    energy-integrating one does not. Read from the Photometry object rather
    than assumed, because it decides the effective wavelength and therefore
    every number below.
    """
    if "photon" in convention.lower():
        return trans * wave
    return trans


def measure_band(wave: np.ndarray, trans: np.ndarray, convention: str) -> dict:
    fine, t_fine = _dense(wave, trans)
    w = _weight(fine, t_fine, convention)
    s = ssp(fine)
    a = attenuation(fine)

    norm = np.trapezoid(w, fine)
    exact = np.trapezoid(s * a * w, fine) / norm
    bare = np.trapezoid(s * w, fine) / norm

    # Transmission-weighted effective wavelength, and the flux-weighted one
    # the first-order term needs.
    lam_eff = np.trapezoid(fine * w, fine) / norm
    lam_flux = np.trapezoid(fine * s * w, fine) / np.trapezoid(s * w, fine)

    single = bare * attenuation(np.array([lam_eff]))[0]
    taylor = bare * (
        attenuation(np.array([lam_eff]))[0]
        + d_attenuation(np.array([lam_eff]))[0] * (lam_flux - lam_eff)
    )

    # Equal-mass sub-bands: split on the cumulative SSP-weighted measure, so
    # each sub-band carries the same share of the band's flux.
    mass = s * w
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (mass[1:] + mass[:-1]) * np.diff(fine))])
    total = cum[-1]
    subband: dict[str, float] = {}
    for k in SUBBAND_COUNTS:
        edges = np.interp(np.linspace(0.0, total, k + 1), cum, fine)
        acc = 0.0
        for lo, hi in pairwise(edges):
            # Integrate on a grid that STARTS and ENDS on the sub-band edges.
            # Masking the shared dense grid instead truncates the mass between
            # the edge and the nearest node, and that loss grows with k, which
            # makes the quadrature appear to diverge as it is refined.
            sub = np.linspace(lo, hi, max(64, DENSE_NODES // k))
            t_sub = np.interp(sub, fine, t_fine)
            w_sub = _weight(sub, t_sub, convention)
            mass_sub = ssp(sub) * w_sub
            sub_mass = np.trapezoid(mass_sub, sub)
            if sub_mass <= 0.0:
                continue
            sub_lam = np.trapezoid(sub * mass_sub, sub) / sub_mass
            acc += sub_mass * attenuation(np.array([sub_lam]))[0]
        subband[str(k)] = acc / norm

    def rel(value: float) -> float:
        return 100.0 * abs(value - exact) / abs(exact)

    return {
        "exact": float(exact),
        "A.Phi": rel(single),
        "Taylor": rel(taylor),
        **{f"K_{k}": rel(v) for k, v in subband.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    import tengri  # deferred so --help costs nothing

    columns = ["A.Phi", "Taylor", *[f"K_{k}" for k in SUBBAND_COUNTS]]
    payload: dict = {
        "spec": {
            "ssp": f"lambda^{SSP_POWER:g}",
            "dust": f"Charlot-Fall tau_BC={TAU_BC:g}, n={DUST_SLOPE:g}",
            "reference": f"dense trapezoid, {DENSE_NODES} nodes",
        },
        "groups": {},
    }

    print(f"{'filter':<10} " + " ".join(f"{c:>9}" for c in columns))
    for group, bands in FILTER_GROUPS.items():
        print(f"-- {group}")
        rows: dict[str, dict] = {}
        for label, name in bands:
            try:
                phot = tengri.Photometry.from_names([name])
            except KeyError:
                print(f"{label:<10} (filter {name!r} not registered, skipped)")
                continue
            wave = np.asarray(phot.filter_waves[0], dtype=float)
            trans = np.asarray(phot.filter_trans[0], dtype=float)
            row = measure_band(wave, trans, str(phot.convention))
            rows[label] = row
            print(f"{label:<10} " + " ".join(f"{row[c]:>9.3f}" for c in columns))
        if rows:
            worst = {c: max(r[c] for r in rows.values()) for c in columns}
            rows["Max"] = worst
            print(f"{'Max':<10} " + " ".join(f"{worst[c]:>9.3f}" for c in columns))
        payload["groups"][group] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
