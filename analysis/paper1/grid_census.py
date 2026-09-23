#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Every number Section 7 asks of the grid, computed from the cells.

Section 7 carries seventeen ``\\confirm{TBD}`` entries, each a statistic over
the finished grid, each with a source comment naming the field it must be read
from. Filling them by hand is seventeen chances to read one cell's value as
the sample's, and the paper has already been wrong that way: it named a
machine nobody had measured and described a two-legged adoption bar the grid
applies three legs of.

So the numbers come from here instead, and this script holds the rules the
hand-fill kept losing:

**Absent is not zero.** A field a cell does not carry is reported as not
recorded. ``meta.get("divergences") or 0`` once turned a missing diagnostic
into a clean bill of health, and ``rhat_max: None`` made every mock fit look
converged. Every quantity below states its own coverage.

**A partial grid cannot answer a question about the whole one.** Nineteen
adopted cells of one configuration say nothing about how six configurations
compare, so an incomplete grid exits non-zero and every figure is labeled with
what it was computed over. ``--allow-partial`` is for watching a run, not for
filling the paper.

**The grid is not uniform in what it records.** Row III predates 837ac7663 and
carries no seed; later rows carry ``seed`` and ``prng_key_seed``. Both cases
are held at once rather than averaged into a sentence true of neither.

CLI:
    python analysis/paper1/grid_census.py [--results-dir DIR] [--allow-partial]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _adoption import is_adopted
from _figure_style import CONFIG_ORDER
from _grid_completeness import load_expected_galaxy_ids

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = ANALYSIS_DIR / "results" / "fits"
SELECTION_20 = ANALYSIS_DIR / "results" / "selected_galaxies_20.json"

#: Field a cell may or may not carry, depending on when it was run.
NOT_RECORDED = "not recorded"

#: A draw counts as sitting at a bound when it lies within this fraction of the
#: prior's width of it. A flat prior puts exactly this share in each band, so
#: the band doubles as the null the observed share is judged against.
EDGE_BAND = 0.05

#: How close a declared lower bound must be to zero to count as the physical
#: "none of this component" limit rather than an imposed floor.
PHYSICAL_ZERO_ATOL = 1e-12

#: Share of draws inside one edge band above which the cell is called pinned.
#: Four times the flat-prior expectation, so ordinary posterior width near a
#: bound does not register.
PIN_THRESHOLD = 0.20

#: Parameters whose lower bound *may* mean something physical: a posterior
#: against tau = 0 is the fit saying "no dust is needed", an inference and not a
#: defect, and agn_lum_ratio = 0 is the same statement about the AGN. Membership
#: here is necessary but not sufficient -- the bound must also actually BE zero.
#: Configuration I declares dust_tau_diff as Uniform(0.5, 3.0), and a floor of
#: 0.5 is a forced minimum screen, so naming the parameter alone would file that
#: wall under "physical" and hide it.
PHYSICAL_ZERO_BOUNDS = frozenset({"dust_tau_v", "dust_tau_bc", "dust_tau_diff", "agn_lum_ratio"})

#: Simplex coordinates. An edge means "all the mass in one age bin", which is a
#: statement about the SFH rather than a wall the model was pushed into, so
#: these are held apart from both other groups rather than silently counted.
SIMPLEX_PARAMS_PREFIX = "sfh_dir_z"

_UNIFORM = re.compile(r"Uniform\(([-\d.eE+]+),\s*([-\d.eE+]+)\)")


def load_cells(results_dir: Path) -> dict[str, dict]:
    """Every cell on disk, keyed ``<galaxy>_<config>``."""
    cells = {}
    for path in sorted(results_dir.glob("*_*.json")):
        try:
            cells[path.stem] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  WARNING: {path.name} unreadable ({exc}); excluded", file=sys.stderr)
    return cells


def band_residuals(results_dir: Path, name: str) -> tuple[np.ndarray, list[str]] | None:
    """Per-band ``(model - observed) / sigma`` for one cell, or None.

    ``obs_sigma`` in the cell is the floored error the likelihood actually saw
    (``fit_one.save_fit_outputs`` is handed ``sigma_floor``), so these
    residuals are in the units the fit was scored in rather than in raw
    catalog errors.
    """
    npz_path = results_dir / f"{name}.npz"
    if not npz_path.is_file():
        return None
    with np.load(npz_path, allow_pickle=True) as npz:
        need = ("obs_fnu", "obs_sigma", "model_photometry_median", "filter_names")
        if not all(k in npz.files for k in need):
            return None
        obs = np.asarray(npz["obs_fnu"], dtype=float)
        sigma = np.asarray(npz["obs_sigma"], dtype=float)
        model = np.asarray(npz["model_photometry_median"], dtype=float)
        names = [str(n) for n in npz["filter_names"]]
    usable = np.isfinite(obs) & np.isfinite(sigma) & np.isfinite(model) & (sigma > 0)
    if not usable.any():
        return None
    return (model[usable] - obs[usable]) / sigma[usable], [
        n for n, keep in zip(names, usable) if keep
    ]


def _config_of(name: str, cell: dict) -> str:
    """The cell's configuration, from the record if it has one, else its filename."""
    return cell.get("config") or name.rsplit("_", 1)[-1]


def prior_boundary_pressure(cells: dict[str, dict], results_dir: Path):
    """Which parameters sit against a prior bound, in how many cells.

    The adoption bar cannot answer this. Zero divergences, split R-hat below
    1.01 and a healthy ESS are all satisfied by a chain that has converged
    cleanly onto a wall, so a cell can clear every leg of the bar while the
    number it reports is set by the edge of the prior rather than by the data.
    Section 7 asks where the posteriors lean on their boundaries precisely
    because the rest of the census cannot see it.

    Bounds are read from the cell's own ``priors`` record, so a prior derived
    per library -- ``met_logzsol`` is held 0.02 dex inside its grid's outermost
    node -- is judged against the bound that cell actually ran with, never
    against a bound copied from another row.

    Returns ``(rows, scanned, skipped)``; a cell with no NPZ is skipped and
    counted rather than treated as unpinned.
    """
    rows, scanned, skipped = [], 0, 0
    for name, cell in cells.items():
        npz_path = results_dir / f"{name}.npz"
        if not npz_path.is_file():
            skipped += 1
            continue
        with np.load(npz_path, allow_pickle=True) as npz:
            available = set(npz.files)
            draws = {
                k: np.asarray(npz[k], dtype=float).ravel()
                for k in (cell.get("priors") or {})
                if k in available
            }
        if not draws:
            skipped += 1
            continue
        scanned += 1
        for param, prior in (cell.get("priors") or {}).items():
            match = _UNIFORM.match(str(prior))
            if match is None or param not in draws:
                continue
            lo, hi = float(match.group(1)), float(match.group(2))
            values = draws[param]
            if hi <= lo or values.size == 0 or not np.isfinite(values).all():
                continue
            band = (hi - lo) * EDGE_BAND
            at_lo = float(np.mean(values < lo + band))
            at_hi = float(np.mean(values > hi - band))
            if max(at_lo, at_hi) < PIN_THRESHOLD:
                continue
            low_end = at_lo >= at_hi
            if param.startswith(SIMPLEX_PARAMS_PREFIX):
                kind = "simplex"
            elif low_end and param in PHYSICAL_ZERO_BOUNDS and abs(lo) <= PHYSICAL_ZERO_ATOL:
                kind = "physical"
            else:
                kind = "artificial"
            rows.append(
                {
                    "cell": name,
                    "config": cell.get("config"),
                    "adopted": bool(is_adopted(cell, _config_of(name, cell)).adopted),
                    "param": param,
                    "end": "lo" if low_end else "hi",
                    "share": max(at_lo, at_hi),
                    "kind": kind,
                }
            )
    return rows, scanned, skipped


def coverage(cells: dict[str, dict], field: str) -> tuple[list, int]:
    """Values present for ``field``, and how many cells lack it.

    Returned rather than defaulted, so a caller cannot average over a field
    half the grid does not carry without seeing that it did.
    """
    present, missing = [], 0
    for cell in cells.values():
        value = cell.get(field)
        if value is None:
            missing += 1
        else:
            present.append(value)
    return present, missing


def _fmt(values, spec=".4g"):
    if not values:
        return NOT_RECORDED
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"{lo:{spec}}"
    return f"{lo:{spec}} to {hi:{spec}}"


def report(cells: dict[str, dict], expected_ids, config_keys, results_dir: Path) -> bool:
    """Print the census. Returns True when the grid is complete."""
    total = len(expected_ids) * len(config_keys)
    have = len(cells)
    by_config = defaultdict(dict)
    for name, cell in cells.items():
        by_config[cell.get("config") or name.rsplit("_", 1)[-1]][name] = cell

    print(f"cells present        : {have} of {total}")
    print(f"configurations seen  : {sorted(by_config)}")
    complete = have == total
    if not complete:
        print(f"\n*** PARTIAL GRID -- {have} of {total} cells. Every figure below is")
        print("*** computed over what is present and is NOT a statement about the")
        print("*** grid. Do not fill Section 7 from a partial run.\n")

    # is_adopted judges per configuration -- the relaxed bar applies to some
    # rows and not others -- so the config is read from the cell, with the
    # filename as the fallback for a cell that does not record one.
    adopted = {k: c for k, c in cells.items() if is_adopted(c, _config_of(k, c)).adopted}
    refused = {k: c for k, c in cells.items() if k not in adopted}
    print(f"\nadopted              : {len(adopted)} of {have}")
    if refused:
        print(f"not adopted          : {sorted(refused)}")

    # --- the adoption bar's three legs, over the adopted set ----------------
    for field, spec in (("divergences", ".0f"), ("rhat_max", ".4f"), ("ess_min", ".1f")):
        values, missing = coverage(adopted, field)
        note = f"  ({missing} adopted cell(s) do not record it)" if missing else ""
        print(f"adopted {field:<13}: {_fmt(values, spec)}{note}")

    # --- retune ladder ------------------------------------------------------
    rungs = defaultdict(int)
    unknown_rung = 0
    for cell in adopted.values():
        rung = cell.get("retune_attempt")
        if rung is None:
            unknown_rung += 1
        else:
            rungs[int(rung)] += 1
    print(
        f"adopted by rung      : {dict(sorted(rungs.items()))}"
        + (f"  ({unknown_rung} without a rung recorded)" if unknown_rung else "")
    )

    # --- why the refusals were refused -------------------------------------
    # "Of the non-adopted cells, how many fail on divergences alone while
    # meeting both other legs" is a TBD in its own right, and the answer is
    # different from "how many had divergences".
    div_only = [
        k
        for k, c in refused.items()
        if c.get("divergences") not in (None, 0)
        and c.get("rhat_max") is not None
        and c["rhat_max"] <= 1.01
        and c.get("ess_min") is not None
        and c["ess_min"] >= 100
    ]
    print(
        f"refused on divergences alone : {len(div_only)}{' -> ' + str(sorted(div_only)) if div_only else ''}"
    )

    # --- cost, and the contention it was measured under --------------------
    walls, wall_missing = coverage(adopted, "wall_time_s")
    print(
        f"\nadopted wall_time_s  : {_fmt(walls, '.0f')}"
        + (f"  (median {statistics.median(walls):.0f})" if walls else "")
        + (f"  ({wall_missing} not recorded)" if wall_missing else "")
    )
    s_per_ess = [
        c["wall_time_s"] / c["ess_min"]
        for c in adopted.values()
        if c.get("wall_time_s") and c.get("ess_min")
    ]
    print(
        f"adopted s per ESS    : {_fmt(s_per_ess, '.2f')}  "
        f"(from {len(s_per_ess)} of {len(adopted)} adopted cells)"
    )

    cpus, loads, concurrent = set(), [], set()
    for cell in cells.values():
        for where in ("load_at_start", "load_at_end"):
            load = cell.get(where) or {}
            if load.get("n_cpus") is not None:
                cpus.add(int(load["n_cpus"]))
            if load.get("load_avg_1m") is not None:
                loads.append(float(load["load_avg_1m"]))
            if load.get("n_concurrent_fits") is not None:
                concurrent.add(int(load["n_concurrent_fits"]))
    print(f"machine n_cpus       : {sorted(cpus) or NOT_RECORDED}")
    print(f"concurrent fits      : {sorted(concurrent) or NOT_RECORDED}")
    if loads and cpus:
        n = max(cpus)
        print(
            f"load average         : {min(loads):.1f} to {max(loads):.1f} on {n} CPUs "
            f"({min(loads) / n:.2f}x to {max(loads) / n:.2f}x oversubscribed)"
        )
        print("  ^ wall and s/ESS above are contended; quote them with this or not at all.")
    print('  ^ no hostname or CPU model is recorded; say "N-core", do not name a machine.')

    # --- goodness of fit, which the adoption bar does not measure ----------
    # The bar asks whether the sampler converged. It cannot see whether the
    # model fits: a chain settles just as cleanly onto a bad posterior as a
    # good one, so a cell can clear every leg of the bar and still miss the
    # photometry badly. Section 7 asks for this number separately and for that
    # reason.
    chi2, trimmed, worst, no_arrays = [], [], defaultdict(list), 0
    for name, cell in adopted.items():
        found = band_residuals(results_dir, name)
        if found is None:
            no_arrays += 1
            continue
        residuals, bands = found
        squares = residuals**2
        chi2.append(float(squares.mean()))
        worst[_config_of(name, cell)].append(bands[int(np.argmax(squares))])
        # The same chi2 with each cell's single largest contributor removed.
        # One bad photometric point and a model that misses everywhere give
        # the same headline number and want opposite responses; only the gap
        # between these two separates them.
        if squares.size > 1:
            trimmed.append(float((squares.sum() - squares.max()) / (squares.size - 1)))
    if chi2:
        over = sum(1 for c in chi2 if c > 2.0)
        print(
            f"\nchi2 per band        : {_fmt(chi2, '.2f')}  (median {statistics.median(chi2):.2f})"
        )
        print(
            f"  cells above 2       : {over} of {len(chi2)}"
            + (f"   ({no_arrays} adopted cell(s) carry no photometry arrays)" if no_arrays else "")
        )
        print(
            "  ^ the adoption bar measures sampler convergence, not fit quality;"
            " these cells all passed it."
        )
        if trimmed:
            print(
                f"  worst band dropped  : {_fmt(trimmed, '.2f')}  "
                f"(median {statistics.median(trimmed):.2f})"
            )
            print(
                "  ^ collapsing toward 1 would mean one bad point per cell;"
                " staying high means the model misses broadly."
            )
        for config, bands in sorted(worst.items()):
            tally = Counter(bands).most_common(3)
            print(f"  {config:<4} worst band  : " + ", ".join(f"{b} x{n}" for b, n in tally))
    elif no_arrays:
        print(f"\nchi2 per band        : {NOT_RECORDED} ({no_arrays} cells carry no arrays)")

    # --- per-configuration facts -------------------------------------------
    print("\nper configuration:")
    for config in sorted(
        by_config, key=lambda c: config_keys.index(c) if c in config_keys else 99
    ):
        group = by_config[config]
        adopted_here = {k: c for k, c in group.items() if k in adopted}
        ess, _ = coverage(adopted_here, "ess_min")
        profiled = {c.get("profile_mass") for c in group.values()}
        print(
            f"  {config:<4} cells {len(group):>3}  adopted {len(adopted_here):>3}  "
            f"ess_min {_fmt(ess, '.1f'):<20} profile_mass {sorted(str(p) for p in profiled)}"
        )
    if not complete:
        print("  ^ 'which configurations mix worst' needs every row; this is not that.")

    # --- prior boundaries, which the adoption bar cannot see ---------------
    rows, scanned, skipped = prior_boundary_pressure(cells, results_dir)
    print(
        f"\nprior-boundary pressure (edge band {EDGE_BAND:.0%} of prior width, "
        f"pinned at {PIN_THRESHOLD:.0%} of draws)"
    )
    print(
        f"  cells scanned        : {scanned}"
        + (f", skipped for want of an NPZ: {skipped}" if skipped else "")
    )
    artificial = [r for r in rows if r["kind"] == "artificial"]
    pinned_cells = {r["cell"] for r in artificial}
    adopted_pinned = {r["cell"] for r in artificial if r["adopted"]}
    print(
        f"  cells against an artificial bound: {len(pinned_cells)} of {scanned}, "
        f"of which adopted: {len(adopted_pinned)}"
    )
    by_param = Counter(r["param"] + " (" + r["end"] + ")" for r in artificial)
    for label, count in by_param.most_common():
        print(f"    {label:<34} {count:>3} cells")
    for kind, note in (
        ("physical", "a bound at zero: the fit saying the component is not needed"),
        ("simplex", "a simplex edge: all the mass in one age bin"),
    ):
        held = {r["cell"] for r in rows if r["kind"] == kind}
        if held:
            print(f"  held apart -- {kind}: {len(held)} cells ({note})")
    if not complete:
        print("  ^ a partial grid undercounts every line above.")

    # --- seed provenance, which the grid is not uniform about --------------
    seeds, seed_missing = coverage(cells, "seed")
    keys, key_missing = coverage(cells, "prng_key_seed")
    print(
        f"\nseed recorded        : {len(seeds)} of {have} cells"
        f"{'  values ' + str(sorted(set(seeds))) if seeds else ''}"
    )
    print(
        f"prng_key_seed        : {len(keys)} of {have} cells"
        f"{'  values ' + str(sorted(set(keys))) if keys else ''}"
    )
    if seed_missing or key_missing:
        print("  ^ cells without these predate 837ac7663. For them the key is")
        print("    PRNGKey(seed + attempt), attempt 1-based: rung 1 -> 43, 2 -> 44, 3 -> 45.")
    return complete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--selection", type=Path, default=SELECTION_20)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit 0 on an incomplete grid; for watching a run, not for filling the paper",
    )
    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"no results directory at {args.results_dir}", file=sys.stderr)
        return 2
    cells = load_cells(args.results_dir)
    if not cells:
        print(f"no cells in {args.results_dir}", file=sys.stderr)
        return 2
    try:
        expected_ids = load_expected_galaxy_ids(args.selection)
    except (OSError, ValueError, KeyError) as exc:
        print(f"cannot read the locked sample at {args.selection}: {exc}", file=sys.stderr)
        return 2

    complete = report(cells, expected_ids, list(CONFIG_ORDER), args.results_dir)
    if not complete and not args.allow_partial:
        print(
            "\nexiting non-zero: the grid is incomplete, so these are not its numbers.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
