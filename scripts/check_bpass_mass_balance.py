#!/usr/bin/env python3
"""Mass-balance closure test of the BPASS v2.2.1 ``imf_chab300`` binary outputs (#2800).

A surviving-mass table for the grid ``bpss_stars_c3k_a_chabrier`` needs, at every
(Z, age) node, the fraction of the formed mass that is in living stars plus remnants.
BPASS's ``starmass-*`` file holds living stars only (column 2) and an untested remnant
column (column 3). The alternative is mass balance,

    living(t) + remnants(t) = M0 - E(t),     M0 = 1e6 Msun,

where ``E(t)`` is the mass returned to the ISM by winds and supernovae, integrated from
the ``yields-*`` files (columns 2-4 wind H, He, metals; columns 7-9 supernova H, He,
metals; Msun/yr per 1e6 Msun formed). That route is only usable if the two BPASS outputs
close: before the first supernovae remnants are absent, so ``living + E`` must equal
``M0``; and after, ``living + E`` can never exceed ``M0`` (no mass is created).

This script measures both on the extracted files (or on the committed excerpt) and
exits 1 when the closure gate fails. The bin rule is the BPASS v2.2 manual's: 51 bins of
0.1 dex centered on ``log10(age/yr) = 6.0 ... 11.0``, each of width
``dt = 10**(la + 0.05) - 10**(la - 0.05)`` yr, "except for the first bin which is
10**6.05 years". Both readings of the first bin are evaluated, as are three placements
of ``E`` within a bin (see ``EJECTA_RULES``).

Usage::

    python -I scripts/check_bpass_mass_balance.py --dir /path/to/extracted/files
    python -I scripts/check_bpass_mass_balance.py \\
        --fixture tests/fixtures/bpass_chab300/mass_balance.csv

Requires numpy only.

References
----------
.. [1] Stanway, E. R. & Eldridge, J. J. 2018, MNRAS, 479, 75, "Re-evaluating old stellar
       populations", doi:10.1093/mnras/sty1353.
.. [2] Eldridge, J. J. et al. 2017, PASA, 34, e058, "Binary Population and Spectral
       Synthesis Version 2.1: Construction, Observational Verification, and New
       Results", doi:10.1017/pasa.2017.51.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

#: Initial mass of every BPASS simple stellar population [Msun].
M0 = 1.0e6
#: BPASS metallicity file codes in increasing Z, and the absolute metal mass fraction.
Z_CODES = (
    "zem5", "zem4", "z001", "z002", "z003", "z004", "z006",
    "z008", "z010", "z014", "z020", "z030", "z040",
)  # fmt: skip
Z_ABS = (1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 1e-2, 1.4e-2, 2e-2, 3e-2, 4e-2)
#: The row at log10 age 11.0 is not a physical age of this model set: its living mass
#: rises back to 0.95 M0 and its supernova rate to 2.5e3 Msun/yr. ProSpect's repackaging
#: replaces its living-star value; closure is evaluated for the 50 rows below it. (The
#: supernova columns of the yields files at log10 age 10.3-10.6 are also not physical,
#: 30-300 Msun/yr; the gate here uses log10 age <= 7.0 only.)
N_VALID_ROWS = 50
#: Half-width of a log-age bin [dex].
HALF_BIN_DEX = 0.05
#: Placements of the cumulative ejecta within bin k: through its upper edge, through its
#: lower edge (the previous bins only; a strict lower bound on E at the bin's age), or
#: through its midpoint.
EJECTA_RULES = ("incl", "excl", "mid")
#: First bin: from age 0 (the manual's "10**6.05 years") or from 10**6.0 yr.
FIRST_BINS = ("manual_0_to_6.05", "6.0_to_6.05")
#: Closure tolerance before the first supernovae, relative to M0.
GATE_TOL = 1.0e-3
#: Last log age before the first supernova ejecta (rate > 0) in every Z file.
PRE_SN_MAX_LOG_AGE = 6.35


class Track(NamedTuple):
    """One metallicity's rows of the BPASS ``starmass`` and ``yields`` files."""

    code: str
    log_age: np.ndarray  # (n,) [log10 yr]
    living: np.ndarray  # (n,) starmass column 2 [Msun per 1e6 Msun formed]
    remnant_col3: np.ndarray  # (n,) starmass column 3, flagged untested [Msun]
    wind: np.ndarray  # (n,) wind H+He+Z output [Msun/yr per 1e6 Msun formed]
    sn: np.ndarray  # (n,) supernova H+He+Z output [Msun/yr per 1e6 Msun formed]


def _loadtxt_text(text: str) -> np.ndarray:
    return np.loadtxt(io.StringIO(text))


def _read(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return _loadtxt_text(handle.read())


def tracks_from_dir(directory: Path) -> list[Track]:
    """Read all 13 metallicities of ``imf_chab300`` from extracted BPASS files.

    Parameters
    ----------
    directory : Path
        Holds ``starmass-bin-imf_chab300.<z>.dat[.gz]`` and ``yields-bin-...``.

    Returns
    -------
    list of Track
        One per metallicity code in increasing Z, rows below ``N_VALID_ROWS``.
    """
    out = []
    for code in Z_CODES:
        arrays = []
        for kind in ("starmass", "yields"):
            matches = sorted(directory.glob(f"{kind}-bin-imf_chab300.{code}.dat*"))
            if len(matches) != 1:
                raise FileNotFoundError(f"{directory}: {kind} file for {code}: {matches}")
            arrays.append(_read(matches[0]))
        sm, yl = arrays
        if sm.shape != (51, 3) or yl.shape != (51, 9) or not np.allclose(sm[:, 0], yl[:, 0]):
            raise ValueError(f"{code}: unexpected starmass {sm.shape} / yields {yl.shape}")
        n = N_VALID_ROWS
        out.append(
            Track(
                code, sm[:n, 0], sm[:n, 1], sm[:n, 2],
                yl[:n, 1] + yl[:n, 2] + yl[:n, 3], yl[:n, 6] + yl[:n, 7] + yl[:n, 8],
            )
        )  # fmt: skip
    return out


FIXTURE_COLUMNS = (
    "z_code", "log_age", "living_msun", "remnant_col3_msun", "wind_msun_per_yr", "sn_msun_per_yr",
)  # fmt: skip


def tracks_from_fixture(path: Path) -> list[Track]:
    """Read the committed excerpt written by :func:`write_fixture`."""
    rows: dict[str, list[list[float]]] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(line for line in handle if not line.startswith("#"))
        header = next(reader)
        if tuple(header) != FIXTURE_COLUMNS:
            raise ValueError(f"{path}: header {header} != {FIXTURE_COLUMNS}")
        for row in reader:
            rows.setdefault(row[0], []).append([float(v) for v in row[1:]])
    out = []
    for code in Z_CODES:
        a = np.array(rows[code])
        out.append(Track(code, a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]))
    return out


def write_fixture(tracks: list[Track], path: Path) -> None:
    """Write the excerpt: only the columns closure uses, full precision of the source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write(
            "# Excerpt of BPASS v2.2.1 bin-imf_chab300 starmass (columns 2, 3) and yields\n"
            "# (wind H+He+Z, SN H+He+Z), rows log10 age 6.0-10.9; see PROVENANCE.md.\n"
            "# Eldridge et al. 2017 PASA 34 e058; Stanway & Eldridge 2018 MNRAS 479 75.\n"
        )
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIXTURE_COLUMNS)
        for t in tracks:
            for i in range(t.log_age.size):
                writer.writerow(
                    [t.code, f"{t.log_age[i]:.1f}", repr(float(t.living[i])),
                     repr(float(t.remnant_col3[i])), repr(float(t.wind[i])), repr(float(t.sn[i]))]
                )  # fmt: skip


def ejected_mass(track: Track, first_bin: str, rule: str) -> np.ndarray:
    """Cumulative mass returned to the ISM, ``E`` at each age node.

    Parameters
    ----------
    track : Track
        One metallicity.
    first_bin : {"manual_0_to_6.05", "6.0_to_6.05"}
        Width of the first bin: ``10**6.05`` yr (the manual's wording) or
        ``10**6.05 - 10**6.0`` yr.
    rule : {"incl", "excl", "mid"}
        ``E`` through the upper edge of the node's bin, through its lower edge, or
        through half of the node's own bin.

    Returns
    -------
    ndarray, shape (n_age,)
        ``E`` [Msun per 1e6 Msun formed].
    """
    la = track.log_age
    width = 10.0 ** (la + HALF_BIN_DEX) - 10.0 ** (la - HALF_BIN_DEX)
    width[0] = 10.0**6.05 if first_bin == "manual_0_to_6.05" else 10.0**6.05 - 10.0**6.0
    per_bin = (track.wind + track.sn) * width
    cum = np.cumsum(per_bin)
    if rule == "incl":
        return cum
    if rule == "excl":
        return cum - per_bin
    if rule == "mid":
        return cum - 0.5 * per_bin
    raise ValueError(f"unknown rule {rule!r}")


class Closure(NamedTuple):
    """Closure residuals of one (first_bin, rule) variant, per metallicity."""

    first_bin: str
    rule: str
    pre_sn_max_abs: np.ndarray  # (n_z,) max |living + E - M0| / M0, log age < 6.35
    below_10myr_max_excess: np.ndarray  # (n_z,) max (living + E - M0) / M0, log age <= 7.0


def closure(tracks: list[Track]) -> list[Closure]:
    """Evaluate every bin-rule variant on all metallicities."""
    out = []
    for first_bin in FIRST_BINS:
        for rule in EJECTA_RULES:
            pre, exc = [], []
            for t in tracks:
                res = (t.living + ejected_mass(t, first_bin, rule) - M0) / M0
                pre.append(np.max(np.abs(res[t.log_age < PRE_SN_MAX_LOG_AGE])))
                exc.append(np.max(res[t.log_age <= 7.0 + 1e-9]))
            out.append(Closure(first_bin, rule, np.array(pre), np.array(exc)))
    return out


def chabrier_remnant_fraction(
    m_turnoff: float = 0.95, m_max: float = 300.0, m_min: float = 0.1,
    m_ns_min: float = 8.0, m_bh_min: float = 25.0, bh_fraction: float = 0.3,
) -> float:  # fmt: skip
    """Remnant mass fraction of a Chabrier-IMF population older than the turnoff.

    Single-star estimate, independent of any BPASS output. The IMF is Chabrier (2003)
    lognormal below 1 Msun (peak 0.079 Msun, sigma 0.69 dex) joined continuously to
    dN/dlog m proportional to m**-1.3 above. Stars below ``m_turnoff`` are living;
    ``m_turnoff`` to ``m_ns_min`` leave white dwarfs with
    ``M_f = 0.109 M_i + 0.428`` (assumed initial-final relation); ``m_ns_min`` to
    ``m_bh_min`` leave 1.4 Msun neutron stars; above, black holes of ``bh_fraction``
    of the initial mass (assumed).

    Returns
    -------
    float
        Remnant mass per unit formed mass [dimensionless].
    """
    lm = np.linspace(np.log10(m_min), np.log10(m_max), 400001)
    m = 10.0**lm
    low = np.exp(-((lm - np.log10(0.079)) ** 2) / (2.0 * 0.69**2))
    at_one = np.exp(-(np.log10(0.079) ** 2) / (2.0 * 0.69**2))
    xi = np.where(m < 1.0, low, at_one * m**-1.3)
    mass = xi * m  # dM/dlog m
    total = np.trapezoid(mass, lm)
    remnant = np.where(
        m < m_turnoff, 0.0,
        np.where(m < m_ns_min, (0.109 * m + 0.428) / m,
                 np.where(m < m_bh_min, 1.4 / m, bh_fraction)),
    )  # fmt: skip
    return float(np.trapezoid(mass * remnant, lm) / total)


def _report(tracks: list[Track]) -> bool:
    """Print the gate evidence; return True when every condition holds."""
    results = closure(tracks)
    print(f"columns (first bin, E placement): {[(c.first_bin, c.rule) for c in results]}")
    print("(a) pre-supernova closure, max |living + E - M0| / M0 over log age < 6.35 (gate 1e-3)")
    for i, t in enumerate(tracks):
        print(f"  {t.code:5s} " + " ".join(f"{c.pre_sn_max_abs[i]:12.2e}" for c in results))
    best = min(c.pre_sn_max_abs.max() for c in results)
    print(f"  best variant, worst Z: {best:.2e}")
    print("(a') mass created: max (living + E - M0) / M0 over log age <= 7.0 (physical: <= 0)")
    for i, t in enumerate(tracks):
        print(
            f"  {t.code:5s} " + " ".join(f"{c.below_10myr_max_excess[i]:12.2e}" for c in results)
        )
    strict = next(c for c in results if c.rule == "excl" and c.first_bin == FIRST_BINS[1])
    strict_worst = strict.below_10myr_max_excess.max()
    print(f"  strict lower bound (6.0_to_6.05, excl) worst Z: {strict_worst:.2e}")
    return bool(best < GATE_TOL and strict_worst < GATE_TOL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="directory of extracted BPASS files")
    group.add_argument("--fixture", type=Path, help="committed CSV excerpt")
    parser.add_argument("--write-fixture", type=Path, help="write the CSV excerpt (with --dir)")
    args = parser.parse_args()
    tracks = tracks_from_dir(args.dir) if args.dir else tracks_from_fixture(args.fixture)
    if args.write_fixture:
        write_fixture(tracks, args.write_fixture)
    passed = _report(tracks)
    print("GATE", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
