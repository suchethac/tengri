# CANDELS 3x3 grid — run provenance (Linux, main + profile_mass)

This directory holds a re-run of the paper's 3x3 CANDELS NUTS grid executed on
Linux against **current main**, not against the branch that produced the
committed results. Read this before comparing these numbers to the paper.

## What was run

- **Grid**: 3 galaxies (13097 blue, 15336 red, 16049 intermediate) x 3
  configurations (I, II, III) = 9 cells. This is the grid the committed driver
  defines. It is **not** a 20x6 = 120-cell grid; no such code exists on
  `origin/paper1/analysis`.
- **Recipe**: unchanged from the committed code — 600 warmup / 600 kept draws /
  4 chains, seed 42, retune ladder target_accept 0.85 -> 0.95 -> 0.99, with
  config III capped at 2 attempts (`RETUNE_ATTEMPTS_BY_CONFIG`, ruling R60).
- **Adoption bar**: unchanged — `n_divergent == 0 and rhat_max < 1.01`.
- **Platform**: Linux, AMD Ryzen 9 5900X (12C/24T), 62 GB RAM, JAX 0.11.0,
  `JAX_PLATFORMS=cpu` (set by the driver itself). Cells were run as concurrent
  `fit_one.py` processes rather than the driver's sequential loop; argv, env and
  seed match `run_fit_subprocess` exactly.

## Code version — differs from the committed results

- **tengri**: `d715d8677` (current `main`), which includes #2281 (analytic mass
  profiling), #2358/#2367 (bounded, reported reinsertion chunking) and #2374
  (profile_mass engagement rules).
- The committed cell JSONs in git were produced by `183f8d763`
  (`origin/paper1/analysis`), whose merge-base with main is `6f52fb3f7`
  (2026-08-28) — i.e. **before** mass profiling existed.
- Consequence: **these results do not reproduce the committed paper analysis.**

## Analysis-code changes required to run on main

`analysis/paper1/` does not exist on main, so it was copied from
`183f8d763` and then migrated. To see exactly what changed, diff against the
originals in git: `git diff 183f8d763 -- analysis/paper1/configs.py` (and
`fit_one.py`).

1. `configs.py`: `FIXED` was removed from `tengri`'s public API. Migrated to
   `Fixed(DEFAULT)` (main's documented equivalent, see
   `src/tengri/parameters/groups.py`), 13 sites, plus the import line.
   **Verified model-preserving**: configs I/II/III build with unchanged free
   parameter counts (5/8/11) *and* unchanged parameter names.
2. `fit_one.py`: added `profile_mass=True` to `base_kwargs` (one line).

## What profile_mass changed in the results

- Mass is marginalized analytically and reinserted, so NUTS samples one fewer
  dimension per config (III: 10 of 11; II: 7 of 8). The reinserted parameter
  carries its full 2400-draw posterior in the `.npz`, not a point estimate.
- **ESS improved on every completed cell** (config II: 586->672, 435->664,
  328->636 vs macOS).
- **Divergence counts and R-hat are NOT comparable to the committed values.**
  The sampled geometry differs, so a cell may clear or miss the bar on a
  different ladder rung than it did before. The bar itself is unchanged.

## Results: 9/9 cells complete, 7 adopted

| cell | macOS div / adopted | this run div / adopted | attempt |
|------|--------------------|------------------------|---------|
| 13097/I   | 0 / ✓ | 0 / ✓ | 3 |
| 13097/II  | 0 / ✓ | 0 / ✓ | 3 |
| 13097/III | 9 / ✗ | **0 / ✓** | 2 |
| 15336/I   | 0 / ✓ | 0 / ✓ | 2 |
| 15336/II  | 0 / ✓ | 0 / ✓ | 1 |
| 15336/III | 25 / ✗ | 1 / ✗ | 2 (capped) |
| 16049/I   | 0 / ✓ | 0 / ✓ | 1 |
| 16049/II  | 0 / ✓ | 0 / ✓ | 1 |
| 16049/III | 36 / ✗ | 2 / ✗ | 1 (best of 2) |

**7 adopted here vs 6 on macOS.** Every configuration I and II cell is adopted
on both. The difference is configuration III.

### Configuration III: the ladder cap, not the grid edge, is now the constraint

`RETUNE_ATTEMPTS_BY_CONFIG = {"III": 2}` caps III's ladder on ruling R60's
finding (#2089) that its divergences are irreducible ``met_logzsol`` grid-edge
geometry, so raising ``target_accept_rate`` to 0.99 "cannot clear a structural
edge". Under ``profile_mass`` that no longer holds:

- 13097/III reaches **0 divergences and is adopted** — the first configuration
  III cell ever to clear the bar.
- 15336/III: 25 -> 1 divergences. 16049/III: 36 -> 2.
- Mean tree depth 5.0-6.8 with 0% of iterations at the depth cap, so the
  sampler is not depth-limited; the geometry itself got easier.

The natural reading is that ``log_total_mass`` was the direction most badly
conditioned against the metallicity grid edge, and marginalising it analytically
relieves most of the pathology. The two cells that still miss do so by 1 and 2
divergences out of 2400, and the 0.99 rung that cleared 13097/I (4 -> 1 -> 0)
and 13097/II is forbidden to them by the cap. **R60 should be revisited under
profile_mass**: the cap was a sound economy for the old sampler and is now the
binding constraint on adoption.

Note the ladder is not monotonic for III: 16049/III went 2 -> 10 divergences
from 0.85 to 0.95, and the code correctly kept attempt 1.

## Memory: what killed two cells, and the safe concurrency limit

Both configuration I cells were killed mid-reinsertion on the first pass by the
host's OOM watchdog (`~/.local/bin/tengri-oom-watchdog`, kills the largest venv
run when summed venv RSS exceeds 40 GB). 16049/I had already finished sampling
with 0/2400 divergences.

Measured peaks at the reinsertion / derived-quantity step:

| configuration | scratch/draw | chunk width | observed RSS peak |
|---------------|--------------|-------------|-------------------|
| II  | — (no chunking: width 4386 >= 2400 draws) | — | ~3.5 GB |
| I   | 2.99 MB | 334 draws | **22-29 GB** |
| III | 13.46 MB | 74 draws | ~12.5 GB |

Steady state is only ~3.4 GB/cell, which is what makes this trap easy to fall
into — concurrency sized on steady state will die at the peak. Configuration I
peaks highest despite being the smallest model, because its light per-draw
payload yields a wide chunk (334 draws) and therefore a large working set;
configuration III's heavy payload forces a narrow chunk and a lower peak.

**Rule: run configuration I cells one at a time.** Two would be ~59 GB summed.
Both losses were recovered by re-running serially, and both reproduced their
killed runs exactly (identical step sizes 0.1737 and 0.4168, identical
divergence counts), confirming scheduling affects only wall clock.

## Performance: no wall-clock win

Per-cell wall time relative to the committed macOS run was 0.18-1.18x
(median ~0.4x, i.e. mostly *slower* per cell). ``profile_mass`` buys sampling
quality, not latency, at this concurrency:

- ESS improved on every configuration II cell: 586->672, 435->664, 328->636.
- 15336/I's adopted attempt reached ESS 168 vs 51 on macOS.
- 16049/I ran 1.18x faster than macOS; everything else was slower, because
  nine concurrent cells each got ~2.5 of 24 threads and NUTS is serial per
  leapfrog step (measured: cells used only 1.6-5.3 cores each, and the box was
  never saturated — 20 of 24 cores at peak).
- `fit_summary.json`'s `log M*` / `log SFR` columns print `-999`. This is a
  pre-existing placeholder in the committed driver (`run_candels_fits.py`,
  "use -999 as placeholder"), not a regression — the real posteriors are in the
  `.npz` files (`sfh_*_log_total_mass`, `stellar_mass`, `sfr_*`).

## Superseded results kept for reference

`results/fits_superseded_183f8d763/` holds cells run earlier on Linux under the
**committed** code version (`183f8d763`, no mass profiling): 16049/II complete
and adopted, plus two mid-ladder partials. Different code version — do not mix
these with the main-based results.
