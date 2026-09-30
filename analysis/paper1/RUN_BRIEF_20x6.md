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
exact CPU/GPU pair at 33x in the CPU's favor for one galaxy; the GPU only wins
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
it here would double-count the lines. Do not substitute a near neighbor for any
of these — a silent library swap is what gave a previous run a configuration
with no nebular emission at all.

Writing to `.part` and moving into place is the right instinct; there is no
checksum manifest.

## 2a. Filters — the two U bands are now fit

`CTIO/MosaicII.U` and `Paranal/VIMOS.U` were added to the filter registry on
2026-09-20 and their curves are committed, so nothing needs to reach SVO at run
time. Both catalog U columns now enter the likelihood.

They had been omitted because the registry held neither curve, and substituting
another telescope's U was rightly declined. But the effect was to silently drop
the two bluest measurements in the catalog: at z ~ 1 those sample the rest-frame
ultraviolet near 1500 A, which is where the unobscured young stars and most of
the attenuation leverage live. Nineteen of the twenty galaxies have both U bands
detected and all twenty have at least one, so this is real information the fits
were discarding, not a marginal addition.

Both are kept rather than one, unlike the Ks pair. They are different bandpasses
from different telescopes (MosaicII peaks at 3644 A spanning 3044-4139 A; VIMOS
at 3851 A spanning 3329-4004 A), so they are two independent measurements.
ISAAC_KS and HAWKI_KS are the opposite case — both resolve to the same VISTA Ks
stand-in curve, so fitting both would enter one response twice with correlated
errors, and the driver takes the first detected one only.

Consequences for what you will see: **16 distinct filters** across 17 catalog
columns, and 11-15 detected bands per galaxy (mean 14.2 over the sample), up
from 13-14 before. A band count below 17 is expected, not a parsing fault.

## 3. Run

Run **a configuration at a time**, not the whole grid in one call:

```bash
python -m paper1.run_candels_fits --configs I --jobs 1               # measure the row
python -m paper1.run_candels_fits --configs I --jobs N --only-missing # then widen
```

Without `--configs` the driver builds all 120 cells galaxy-major under one
global `--jobs`, so the first fill is one galaxy crossed with every row and the
configurations interleave from the start. Concurrency has to be sized per row
(see §4), and that cannot be measured from a mixture.

`aggregate_summary` iterates the full sample regardless, so a per-row run writes
a summary in which every not-yet-run cell reads as failed. Rebuild it once at
the end with `--summary-only`; the intermediate ones are not a scoreboard.

`--only-missing` keys on `adoption_pass`, not on presence, so re-running a row
picks up cells that ran but missed the bar.

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
them as a warning about shape, not as a table to schedule against. Every row's
physics changed, so which row is heaviest is unknown.

Size each row against **N simultaneous reinsertion peaks of that row**, not one
peak plus N-1 steady states. The launch stagger (`stagger_seconds`, 20 s) is
applied only to the initial fill, deliberately — it exists to keep N JIT
compilations off the box at startup. Refills are not staggered, so when a cell
finishes its replacement starts immediately and its compile can land on top of
another cell's reinsertion peak at an arbitrary point hours in. Nothing spaces
those, and that overlap is what killed two cells on the previous grid.

Do not raise the OOM watchdog ceiling to make a cell fit. If a row's peak makes
even two concurrent cells uncomfortable, run that row at `--jobs 1` and widen
the cheap rows instead.

Report the free-parameter count per configuration from the cell JSONs.
`get_config_dimensions` carries measured values for I, IV, V and VI; II and III
report 0 because their libraries were absent from the machine that measured the
others, and 0 is a placeholder rather than a number to quote.

## 4a. The IGM precompute fix does not gate this grid

The writing plan says to land the exact IGM fold before remaking anything that
depends on `WavePrecomp` photometry at z >~ 1, and every configuration here uses
`WavePrecomp` at z ~ 1. That ordering does not bind for this sample, and the
reason is checkable rather than a judgement call.

The node fold differs from the exact fold only where a Lyman break falls
*inside* a bandpass. IGM absorption acts on rest wavelengths blueward of
Ly-alpha, 1216 A. At this sample's maximum redshift, z = 1.098, that lands at
2551 A observed, with the Lyman limit at 1913 A.

Both U bands are now fit (see §2a), so the bluest response in the likelihood is
CTIO MosaicII U, not ACS F435W. Measured from the curves themselves:

| band | nonzero from | 1% of peak from |
|---|---|---|
| `ctio_u` | 3044 A | 3059 A |
| `vimos_u` | 3329 A | 3329 A |
| `hst_f435w` | 3526 A | 3606 A |

That leaves 509 A of margin at the 1% threshold and 494 A at the absolute
floor. Half what it was before the U bands were added, and still unambiguous:
nothing straddles the break, so no band in this fit sees IGM attenuation and
the two folds agree here.

**The rule, not the number.** A band is affected when its blue edge lies
blueward of observed Ly-alpha:

    1216 * (1 + z_max)  >  blue edge of the bluest fitted band

Stated as a threshold redshift that has to be restated whenever the filter set
or the sample changes, and both changed here. Stated as the inequality it
re-derives itself, so the driver now checks it at startup rather than trusting
this paragraph: `candels_io.assert_igm_node_fold_adequate` raises if the break
has moved inside any fitted band, and `run_candels_fits` logs the headroom
before the first cell. Run `python -m paper1.candels_io` to see the table.

For the current band set the binding band is CTIO U, which admits the break at
**z = 1.503** — not 1.4, which was a conservative guess before the curve was
measured. The other edges fall at z = 1.738 (VIMOS U), 1.900 (F435W) and 2.757
(F606W).

The fix remains required before the mock figure (z = 1 with GALEX FUV, where
the break *is* inside the bandpass, which is where the appendix FUV spike comes
from) and before any high-redshift claim. It is not a prerequisite for these
120 cells.


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
