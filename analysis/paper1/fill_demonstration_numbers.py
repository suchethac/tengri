# SPDX-License-Identifier: BSD-3-Clause
r"""Compute the \confirm{TBD} numbers of section 7 from the grid's own output.

Every TBD in ``7-demonstration.tex`` carries a ``% source:`` comment naming the
field it comes from. This reads those fields and prints the values, so filling
the section is transcription from a file rather than recall. The rule it exists
to enforce is that no number reaches the paper except from real fit output.

Two things it deliberately refuses to do.

It reads ``results/fits/`` and nothing else by default. A directory of
superseded fits sits beside it (``fits_superseded_oldsuite_20260920``, 101
cells from an earlier suite), and those numbers would look perfectly plausible
in the paper. Pointing this script at them takes an explicit flag, and anything
produced that way is stamped so it cannot be mistaken for a result.

It does not estimate. Several TBDs need the posterior draws rather than the
per-cell diagnostics JSON -- goodness of fit across the grid, per-band
systematic residuals, where posteriors lean on prior boundaries, and everything
sample-level. Those are listed at the end as outstanding rather than
approximated from what is here.

Run::

    python -m paper1.fill_demonstration_numbers
    python -m paper1.fill_demonstration_numbers --from-dir <other>   # not for the paper
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
CANONICAL = RESULTS / "fits"
SUPERSEDED = RESULTS / "fits_superseded_oldsuite_20260920"

#: TBDs that still need something this script does not read. The first needs the
#: declared priors, which means rebuilding each configuration; the second needs
#: the published per-code catalog. Named so the report says what is missing
#: rather than quietly covering fewer numbers.
NEEDS_MORE = [
    "where the posteriors lean on prior boundaries (needs the declared priors, "
    "so a model rebuild per configuration)",
    "where the configurations fall against the published inter-code spread "
    "(needs results/art_sedfitting_z1.csv joined on galaxy ID)",
]


def posterior_numbers(directory: Path, cells: list[dict]) -> dict | None:
    """The section-7 claims that live in the NPZ rather than the JSON.

    Derived quantities (``stellar_mass``, ``sfr_100myr``) are stored on a
    500-draw subsample while the sampled parameters are the full chain, so
    anything joining the two has to say which it used. Everything here is a
    median over the derived draws, which is what the section quotes.
    """
    import numpy as _np

    per_cell = []
    for cell in cells:
        gal, cfg = cell.get("gal_id"), cell.get("config")
        npz_path = directory / f"{gal}_{cfg}.npz"
        if not npz_path.exists():
            continue
        with _np.load(npz_path, allow_pickle=True) as handle:
            need = ("model_photometry_median", "obs_fnu", "obs_sigma", "filter_names")
            if any(k not in handle.files for k in need):
                continue
            model = _np.asarray(handle["model_photometry_median"])
            obs = _np.asarray(handle["obs_fnu"])
            sigma = _np.asarray(handle["obs_sigma"])
            bands = [str(b) for b in handle["filter_names"]]
            resid = (obs - model) / _np.where(sigma > 0, sigma, _np.nan)
            row = {
                "gal": gal,
                "config": cfg,
                "adopted": bool(cell.get("adoption_pass")),
                "chi2": float(_np.nansum(resid**2)),
                "n_bands": len(bands),
                "n_free": cell.get("n_free"),
                "resid": dict(zip(bands, [float(r) for r in resid], strict=True)),
            }
            for key, label in (("stellar_mass", "log_mstar"), ("sfr_100myr", "log_sfr")):
                if key in handle.files:
                    vals = _np.asarray(handle[key])
                    vals = vals[vals > 0]
                    row[label] = float(_np.log10(_np.median(vals))) if vals.size else None
            per_cell.append(row)
    return {"cells": per_cell} if per_cell else None


def load_cells(directory: Path) -> list[dict]:
    cells = []
    for path in sorted(directory.glob("*.json")):
        try:
            cells.append(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path} is not readable JSON: {exc}") from exc
    return cells


def fmt_range(values, unit="", places=1):
    if not values:
        return "n/a"
    lo, hi = min(values), max(values)
    med = statistics.median(values)
    return f"{lo:.{places}f} to {hi:.{places}f} {unit} (median {med:.{places}f})".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dir", type=Path, default=None)
    args = parser.parse_args()

    directory = args.from_dir or CANONICAL
    canonical = args.from_dir is None

    if not directory.exists() or not any(directory.glob("*.json")):
        print(f"no per-cell JSON in {directory}")
        if canonical and SUPERSEDED.exists():
            n = len(list(SUPERSEDED.glob("*.json")))
            print(
                f"\n{SUPERSEDED.name} holds {n} cells from an earlier suite. Those are NOT a\n"
                "substitute: they were produced by a different configuration set and would\n"
                "look entirely plausible in the paper. To inspect them anyway, pass\n"
                f"--from-dir {SUPERSEDED}, and note the output is stamped not-for-the-paper."
            )
        return 1

    cells = load_cells(directory)
    if not canonical:
        print("=" * 72)
        print("NOT FOR THE PAPER -- reading", directory.name)
        print("=" * 72)

    adopted = [c for c in cells if c.get("adoption_pass")]
    print(f"\ncells read: {len(cells)}   adopted: {len(adopted)}   from: {directory}")

    bands = [len(c["filter_names"]) for c in cells if "filter_names" in c]
    warmups = sorted({c.get("n_warmup") for c in cells if c.get("n_warmup") is not None})
    samples = sorted({c.get("n_samples") for c in cells if c.get("n_samples") is not None})

    print("\n--- section 7 TBDs computable from the per-cell JSON ---\n")
    print(
        f"band-count range      : {min(bands)}-{max(bands)}" if bands else "band-count range: n/a"
    )
    print(
        f"n_warmup              : {warmups if len(warmups) > 1 else warmups[0] if warmups else 'n/a'}"
    )
    print(
        f"n_samples             : {samples if len(samples) > 1 else samples[0] if samples else 'n/a'}"
    )
    print(f"n adopted             : {len(adopted)} of {len(cells)}")

    rungs = Counter(c.get("target_accept_rate") for c in adopted)
    attempts = Counter(c.get("retune_attempt") for c in adopted)
    print(
        f"retune distribution   : adopted at target_accept {dict(sorted(rungs.items(), key=lambda kv: (kv[0] is None, kv[0])))}"
    )
    print(
        f"                        adopted on attempt {dict(sorted(attempts.items(), key=lambda kv: (kv[0] is None, kv[0])))}"
    )

    walls = [c["wall_time_s"] for c in adopted if c.get("wall_time_s") is not None]
    print(f"wall per adopted cell : {fmt_range(walls, 's')}")

    per_ess = [
        c["wall_time_s"] / c["ess_min"]
        for c in adopted
        if c.get("wall_time_s") is not None and c.get("ess_min")
    ]
    print(f"s/ESS over adopted    : {fmt_range(per_ess, 's/ESS', places=2)}")

    ess = [c["ess_min"] for c in adopted if c.get("ess_min") is not None]
    print(f"min ESS over adopted  : {min(ess):.1f}" if ess else "min ESS: n/a")

    by_config = defaultdict(list)
    for c in adopted:
        if c.get("ess_min") is not None:
            by_config[c.get("config", "?")].append(c["ess_min"])
    if by_config:
        print("\nmixing by configuration (min ESS, over adopted cells):")
        for cfg, vals in sorted(by_config.items(), key=lambda kv: min(kv[1])):
            print(
                f"   {cfg:<6} worst {min(vals):>8.1f}   median {statistics.median(vals):>8.1f}   n={len(vals)}"
            )

    failed = [c for c in cells if not c.get("adoption_pass")]
    if failed:
        print(
            f"\nnot adopted ({len(failed)}), which the section must account for rather than omit:"
        )
        for c in sorted(failed, key=lambda c: -(c.get("divergences") or 0))[:10]:
            print(
                f"   {c.get('gal_id')!s:>7}/{c.get('config', '?'):<4} "
                f"div={c.get('divergences')} rhat_max={c.get('rhat_max'):.4f} "
                f"ess_min={c.get('ess_min')}"
                if c.get("rhat_max") is not None
                else f"   {c.get('gal_id')}/{c.get('config')}"
            )

    post = posterior_numbers(directory, cells)
    if post:
        rows = post["cells"]
        ad = [r for r in rows if r["adopted"]] or rows
        print(f"\n--- section 7 TBDs from the posterior NPZs ({len(rows)} cells read) ---\n")
        ok = [r for r in ad if r["chi2"] == r["chi2"]]
        chi2_band = [r["chi2"] / r["n_bands"] for r in ok]
        dofs = [r["n_bands"] - (r["n_free"] or 0) for r in ok]
        chi2_dof = [r["chi2"] / max(d, 1) for r, d in zip(ok, dofs, strict=True)]
        print(f"posterior-predictive chi2 per band : {fmt_range(chi2_band, '', places=2)}")
        print(f"posterior-predictive chi2 / dof    : {fmt_range(chi2_dof, '', places=2)}")
        if dofs and min(dofs) < 8:
            print(
                f"   CAUTION: dof = n_bands - n_free runs from {min(dofs)} to {max(dofs)}. "
                "At single-digit dof chi2/dof is a\n"
                "   noisy statistic and its spread across cells is mostly that noise; "
                "quote chi2 per band, or the\n"
                "   residual distribution, rather than leaning on chi2/dof in the text."
            )

        band_resid = defaultdict(list)
        for r in ad:
            for band, value in r["resid"].items():
                if value == value:
                    band_resid[band].append(value)
        print("\nper-band standardized residual, median over cells")
        print("(a band offset the same way in every configuration is the model, not noise):")
        for band, values in sorted(
            band_resid.items(), key=lambda kv: -abs(statistics.median(kv[1]))
        ):
            med = statistics.median(values)
            flag = "  <-- systematic" if abs(med) > 0.5 else ""
            print(f"   {band:<14} median {med:>7.3f}  n={len(values):>3}{flag}")

        by_gal = defaultdict(dict)
        for r in ad:
            if r.get("log_mstar") is not None:
                by_gal[r["gal"]][r["config"]] = r["log_mstar"]
        multi = [v for v in by_gal.values() if len(v) > 1]
        ranges = [max(v.values()) - min(v.values()) for v in multi]
        stdevs = [statistics.stdev(list(v.values())) for v in multi if len(v) > 1]
        if ranges:
            print(
                f"\nconfiguration-to-configuration scatter in log M*, over {len(multi)} galaxies:"
            )
            print(f"   range  (max-min)      : {fmt_range(ranges, 'dex', places=3)}")
            if stdevs:
                print(f"   stdev across configs  : {fmt_range(stdevs, 'dex', places=3)}")
            print(
                "   Compare the STDEV against the published inter-code figure, not the\n"
                "   range: Pacifici+2023's ~0.1 dex in M* and ~0.3 dex in SFR are scatter,\n"
                "   and on that catalog the median range is 2.4x the median stdev. Run\n"
                "   `python -m paper1.published_code_spread` for both, measured."
            )

    print("\n--- still outstanding: these need more than the fits ---")
    for item in NEEDS_MORE:
        print(f"   - {item}")
    if not canonical:
        print("\nNOT FOR THE PAPER -- see the banner above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
