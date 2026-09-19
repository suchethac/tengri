# 20 x 6 CANDELS grid — run brief

Target: 120 cells, ~24 h. Everything below is pinned; do not pull after checkout.

## 1. Pin

Check out the branch this file arrives on and pin the commit. One code version
for all 120 cells — that is the whole contract. If a cell has to be re-run
later on a different commit, the grid is not comparable and wants a clean
sweep, not a merge.

```bash
export PYTHONPATH=<worktree>/src        # editable installs point at main's src
export JAX_PLATFORMS=cpu
```

`JAX_PLATFORMS=cpu` is not negotiable on a single-galaxy fit. NUTS is serial per
leapfrog step, and `bench/reports/2026-08-20-cuda-device-matrix.md` measured this
exact CPU/GPU pair at 33x in the CPU's favour for one galaxy; the GPU only wins
between roughly 128 and 512 galaxies batched.

## 2. Stellar libraries

The suite spans five grids. Three are usually already present; **two are new**:

| Configuration | Grid | Note |
|---|---|---|
| I, VI | `fsps_mist_c3k_a_chabrier` | |
| II | `fsps_prsc_c3k_a_chabrier` | **fetch** |
| III | `fsps_mist_miles_chabrier` | **fetch** |
| IV | `fsps_prsc_miles_chabrier` | note: the plain grid, **not** the `_wNE_` variant |
| V | `bpss_stars_c3k_a_chabrier` | |

Configuration IV wants the bare MILES grid and supplies nebular emission from
Cloudy. The `wNE` grid carries nebular emission baked into the templates; using
it here would double-count the lines. Do not substitute a near neighbour for any
of these — a silent library swap is what gave a previous run a configuration
with no nebular emission at all.

Writing to `.part` and moving into place is the right instinct; there is no
checksum manifest.

## 3. Run

```bash
python -m paper1.run_candels_fits --jobs 3 --only-missing
```

`results/fits/` must be **empty** before the first launch. Cells are named
`{galaxy}_{config}.json` and `--only-missing` keys on that filename, which is
identical between this suite and the superseded one, so any leftover file is a
cell silently skipped and quietly wrong.

Recipe and bar, unchanged from the pinned driver:

- 150 warmup / 300 kept draws, 4 chains, seed 42
- ladder `target_accept` 0.85 -> 0.95 -> 0.99, then doubled warmup
- adoption bar: **0 divergences AND max split R-hat < 1.01**
- `RETUNE_ATTEMPTS_BY_CONFIG` is now empty — see the comment in `fit_one.py`.
  The old `{"III": 2}` cap was written about a model that no longer bears that
  numeral.

## 4. Memory

`profile_mass` peaks well above steady state at the reinsertion step, and the
peak is easy to miss because steady state is only ~3.4 GB/cell. Measured on the
previous suite: ~22-29 GB for the lightest model, ~12.5 GB for the heaviest,
~3.5 GB for the middle one. The counter-intuitive part is that the *lightest*
payload peaks highest — a small per-draw payload yields a wide chunk and so a
big working set.

Those numbers were measured on different configurations than these, so treat
them as a warning about shape, not as a table to schedule against. Watch the
first cell of each configuration and size concurrency off the observed peak,
not off steady state. Do not raise the OOM watchdog ceiling to make a cell fit.

## 5. What changed versus the 3 x 3

Everything about the models. The six configurations are a new suite — SFH,
library, attenuation, dust IR and nebular all vary across rows — so divergence
counts and R-hat are **not** comparable cell-by-cell with any earlier run. The
bar itself is untouched.

- Sample is 20 galaxies: 7 blue, 7 red, 3 dusty, and **3 mid-infrared AGN
  candidates (1826, 4056, 24786)**. Earlier selections screened those three out.
- Configuration VI is Configuration I's host plus a QSOgen disc and SKIRTOR
  torus. Every galaxy is fit under every row, including AGN-on for the hosts and
  AGN-off for the candidates — that contrast is the point, not an oversight.
- Metallicity priors are derived per library from its own grid, held 0.02 dex
  inside the outermost node.

## 6. Report back

Per cell: divergences, max split R-hat, min ESS, attempts, adopted, wall. Plus
peak RSS per configuration, and the commit SHA. Flag any cell where the
adoption bar was met only on the doubled-warmup rung.

`.npz` posteriors are gitignored; leave them on disk and say where.
