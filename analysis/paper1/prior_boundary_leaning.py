# SPDX-License-Identifier: BSD-3-Clause
r"""Which posteriors lean on a prior boundary, and in how many cells.

Section 7 asks where the posteriors rail against their priors. That question
cannot be answered from the saved draws alone: it needs the declared prior, so
it needs the model rebuilt per configuration. This does that, and reports the
answer per parameter and per configuration.

What "leaning" means here
-------------------------
A prior is a distribution, so some posterior mass sits near its edges even when
the data say nothing. A flat posterior under ``Uniform(lo, hi)`` puts exactly
``EDGE_FRAC`` of its draws in the outermost ``EDGE_FRAC`` of the range at each
end. So the baseline is the prior itself, not zero, and a parameter is called
leaning only when its edge mass exceeds that baseline by ``FACTOR``.

Comparing against zero instead would flag every parameter in the grid, which is
how a diagnostic becomes noise people stop reading.

Two ways this refuses to guess
------------------------------
The free-parameter list is read from each cell's ``rhat_dict``, which is
computed per free parameter. It is NOT inferred from which NPZ arrays vary: a
Fixed parameter's draws differ at the last bit (measured: ``redshift`` at a
relative spread of 2e-16), so "nonzero variance" would have called 9 parameters
free in a cell that had 5.

A configuration whose SSP grid is absent is named and counted as unchecked, not
skipped quietly. Two of the six (``fsps_prsc_c3k_a_chabrier`` for II and
``fsps_mist_miles_chabrier`` for III) are not bundled, so a run on a machine
without them covers four, and must say which four.

Run::

    python -m paper1.prior_boundary_leaning
    python -m paper1.prior_boundary_leaning --from-dir <dir> --factor 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
CANONICAL = RESULTS / "fits"

#: Outermost fraction of a bounded prior's range counted as "the edge".
EDGE_FRAC = 0.05

#: How many times the prior-implied edge mass counts as leaning.
FACTOR = 3.0

#: Priors do not depend on the filter set, so the rebuild uses a tiny one.
PROBE_FILTERS = ["sdss_g", "sdss_r"]


def edge_fractions(draws: np.ndarray, lo: float, hi: float, edge: float) -> tuple[float, float]:
    """Fraction of draws within ``edge`` of each end of ``[lo, hi]``.

    Parameters
    ----------
    draws : ndarray
        Posterior draws for one parameter.
    lo, hi : float
        Declared prior bounds.
    edge : float
        Outermost fraction of the range to count.

    Returns
    -------
    tuple of (float, float)
        Fraction at the lower end and at the upper end.
    """
    span = hi - lo
    if not math.isfinite(span) or span <= 0:
        return (float("nan"), float("nan"))
    width = edge * span
    n = draws.size
    return (float(np.sum(draws <= lo + width)) / n, float(np.sum(draws >= hi - width)) / n)


def redshift_of(cell_json: Path) -> float | None:
    """Read the fit's redshift from its NPZ.

    The per-cell JSON does not record it, and several priors depend on it
    (config II bounds tau by the age at z; config III sets its bin edges from
    it), so defaulting to a constant would silently build the wrong prior.

    Parameters
    ----------
    cell_json : Path
        Path to a cell's ``.json``; the ``.npz`` beside it is read.

    Returns
    -------
    float or None
        The redshift, or None when the NPZ is absent or carries no redshift.
    """
    npz_path = cell_json.with_suffix(".npz")
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=True) as npz:
        if "redshift" not in npz.files:
            return None
        return float(np.asarray(npz["redshift"], float).ravel()[0])


def build_distributions(config_key: str, z: float):
    """Declared priors for one configuration, or the reason there are none.

    Returns
    -------
    tuple of (dict or None, str or None)
        ``({param: distribution}, None)`` on success, or ``(None, reason)``.
        The reason is reported, never swallowed: a configuration that could not
        be checked must be distinguishable from one that was checked and clean.
    """
    try:
        from tengri import Observation, Photometry

        from . import configs as _configs

        builder = getattr(_configs, f"config_{config_key}")
        ssp = _configs.load_ssp_for(config_key)
        obs = Observation(photometry=Photometry.from_names(PROBE_FILTERS))
        model = builder(ssp, obs, z)
        spec = model.spec
        return ({p: spec.get_distribution(p) for p in spec.free_params}, None)
    except FileNotFoundError as exc:
        return (None, f"SSP grid not available: {exc}".split(". Call")[0])
    except Exception as exc:  # reported, not hidden
        return (None, f"{type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-dir", type=Path, default=None)
    ap.add_argument("--edge-frac", type=float, default=EDGE_FRAC)
    ap.add_argument("--factor", type=float, default=FACTOR)
    args = ap.parse_args(argv)

    directory = args.from_dir or CANONICAL
    cells = sorted(directory.glob("*.json"))
    if not cells:
        print(f"no per-cell JSON in {directory}")
        return 1

    by_config: dict[str, list[Path]] = defaultdict(list)
    for path in cells:
        try:
            by_config[json.loads(path.read_text()).get("config", "?")].append(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  unreadable: {path.name}: {exc}")

    threshold = args.edge_frac * args.factor
    print(f"reading {len(cells)} cell(s) from {directory}")
    print(
        f"leaning = more than {threshold:.1%} of draws inside the outermost "
        f"{args.edge_frac:.0%} of a bounded prior\n"
        f"(a flat posterior gives {args.edge_frac:.0%}; the baseline is the prior, not zero)\n"
    )

    leaning: dict[tuple[str, str, str], int] = defaultdict(int)
    checked: dict[str, int] = defaultdict(int)
    unchecked: list[tuple[str, str]] = []
    unbounded: set[tuple[str, str]] = set()

    for config_key in sorted(by_config):
        paths = by_config[config_key]
        z = redshift_of(paths[0])
        if z is None:
            unchecked.append((config_key, "no redshift recorded in the cell's NPZ"))
            continue
        dists, reason = build_distributions(config_key, z)
        if dists is None:
            unchecked.append((config_key, reason))
            continue

        for path in paths:
            cell = json.loads(path.read_text())
            npz_path = path.with_suffix(".npz")
            if not npz_path.exists():
                continue
            with np.load(npz_path, allow_pickle=True) as npz:
                checked[config_key] += 1
                for name in sorted(cell.get("rhat_dict", {})):
                    dist = dists.get(name)
                    if dist is None or name not in npz.files:
                        continue
                    lo, hi = getattr(dist, "bounds", (None, None))
                    if lo is None or not (math.isfinite(lo) and math.isfinite(hi)):
                        unbounded.add((config_key, name))
                        continue
                    f_lo, f_hi = edge_fractions(
                        np.asarray(npz[name], float), lo, hi, args.edge_frac
                    )
                    if f_lo > threshold:
                        leaning[(config_key, name, "lower")] += 1
                    if f_hi > threshold:
                        leaning[(config_key, name, "upper")] += 1

    if checked:
        print("cells checked per configuration:")
        for key in sorted(checked):
            print(f"   {key:<5} {checked[key]}")
    if leaning:
        print("\nparameters leaning on a prior boundary (config, parameter, end, cells):")
        for (cfg, name, end), count in sorted(leaning.items(), key=lambda kv: -kv[1]):
            print(f"   {cfg:<5} {name:<32} {end:<6} {count:>3} of {checked[cfg]}")
    elif checked:
        print("\nno parameter leans on a prior boundary above the threshold.")
    if unbounded:
        print("\nunbounded priors, not checkable this way:")
        for cfg, name in sorted(unbounded):
            print(f"   {cfg:<5} {name}")
    if unchecked:
        print("\nNOT CHECKED -- these configurations contribute nothing above:")
        for cfg, reason in unchecked:
            print(f"   {cfg:<5} {reason}")
        print("   Their cells are absent from the counts, not counted as clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
