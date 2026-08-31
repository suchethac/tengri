# Paper I Demonstration Fits — Reproduction Commands

## Environment Setup (from worktree root)

```bash
cd <path-to-your-tengri-checkout>
export PYTHONPATH=$PWD/src
export JAX_PLATFORMS=cpu
python_exe=/opt/anaconda3/bin/python3
```

The scripts under `analysis/paper1/` default to paths resolved relative to the
checkout; set `TENGRI_CANDELS_CATALOG`, `ART_SEDFITTING_DIR`, or
`TENGRI_PAPER_FIGURES_DIR` to override the CANDELS catalog, the
`art_sedfitting` checkout, or the figures output directory respectively.

## Deliverable 1: Single Fit (fit_one.py)

**Purpose:** Fit a single galaxy with a specified SED model configuration via NUTS MCMC.

**Usage:**
```bash
$python_exe analysis/paper1/fit_one.py --galaxy ID --config {I,II,III} --method mcmc_nuts --out DIR [--seed N] [--n-warmup N] [--n-samples N] [--n-chains N]
```

`--n-warmup`, `--n-samples` and `--n-chains` default to the paper's 600, 600 and 4 (the settings below); the examples and the grid use those defaults. Pass small values to smoke-test the pipeline in a few minutes per configuration rather than half an hour: the fit, the retunes, the per-attempt JSON and the NPZ all run (the best attempt is saved even when no attempt clears the adoption bar of 0 divergences and max split-R̂ < 1.01, with `adoption_pass: false`); expect a 20-draw budget to miss the bar (configuration II cleared it at 20/20/1; I and III needed 200/200/4).

**Examples:**
```bash
# Fit galaxy 13097 with config I
$python_exe analysis/paper1/fit_one.py --galaxy 13097 --config I --method mcmc_nuts --out analysis/paper1/results/fits --seed 42

# Fit galaxy 15336 with config II
$python_exe analysis/paper1/fit_one.py --galaxy 15336 --config II --method mcmc_nuts --out analysis/paper1/results/fits --seed 42

# Fit galaxy 24497 with config III
$python_exe analysis/paper1/fit_one.py --galaxy 24497 --config III --method mcmc_nuts --out analysis/paper1/results/fits --seed 42
```

**Outputs:**
- `results/fits/<ID>_<config>.npz` — Parameter samples (thinned to ≤ 4000 flattened draws per parameter: tengri returns every chain's kept draws concatenated into one 1-D array, so the cap is on the whole record, not per chain), derived quantities (stellar mass, SFR, dust), SFH posterior grid. The derived quantities and the SFH percentiles are computed from draws strided across the whole record, so they span every chain rather than the front of chain 0.
- `results/fits/<ID>_<config>.json` — Diagnostics: divergences, max R̂, min ESS, wall time, plus `attempts` (one entry per attempt, written before each retune begins so a killed retune still leaves attempt 1's evidence) and `retune_history` (the attempts that failed the adoption bar)

## Deliverable 2: 3×3 Grid (run_candels_fits.py)

**Purpose:** Run 3 galaxies × 3 configurations = 9 NUTS fits sequentially.

**Command:**
```bash
cd analysis/paper1
$python_exe run_candels_fits.py
```

**Second pass (only the cells that have not been adopted):**
```bash
cd analysis/paper1
$python_exe run_candels_fits.py --only-missing
```

`--only-missing` skips a cell whose `results/fits/<ID>_<config>.json` already records
`adoption_pass: true`, reusing that JSON for the summary table and `fit_summary.json`, and runs
every other cell. Without the flag the driver runs all nine cells, as before.

**Retune policy (per cell, in `fit_one.py`):** attempt 1 uses a diagonal mass matrix at
`target_accept_rate` 0.85; attempt 2 keeps the same warmup and raises `target_accept_rate` to
0.95; attempt 3 keeps the same warmup again and raises it to 0.99; attempt 4 and each further
attempt keep 0.99 and double the warmup. The step size is therefore tuned twice, at the base
warmup, before the expensive knob is touched — measured on cell 13097/III (600 warmup + 4×600
draws, D = 11), attempt 1 at 0.85 missed the bar on 77/2400 divergences at max R̂ 1.012 after
5741 s, and percent-level divergences are a step-size problem (the standard remedy is a higher
`adapt_delta`). The mass matrix is never switched to dense by a retune — on cell 13097/II
(D = 8), attempt 1 on a diagonal mass matrix gave 3/2400 divergences at max R̂ 1.0014 and the
old dense-mass retune gave 79/2400 at 1.023. The default is 3 attempts.

**Every missed attempt is saved before the next one starts.** Once an attempt returns a
posterior that misses the bar, the best attempt so far is written to `results/fits/<ID>_<config>.npz`
and `.json` (with `adoption_pass: false` and `best_attempt: <n>`) before the retune begins, and
the final write — an adopted attempt, or the best one at the end of the loop — overwrites it. A
per-cell timeout or a crash during a retune therefore leaves the last completed attempt's draws
on disk instead of diagnostics alone.

**Cells that never clear the adoption bar are still saved.** If no attempt reaches 0 divergences
and max R̂ < 1.01, the best attempt is written to the NPZ and the JSON with
`adoption_pass: false` and `best_attempt: <n>`, and `fit_one.py` exits 0. The best attempt has to
mix first — every attempt with max R̂ < 1.02 outranks every attempt at or above it, and only
within those classes does the fewest-divergences rule (then lowest max R̂) decide, because
divergence counts only compare between chains that sampled the same distribution.
Only a cell in which every attempt raised is a failure. The summary table prints the adoption
verdict per cell (`✓`, or `✗ best att <n>`), so a saved-but-not-adopted cell is never read as a
pass, and `fit_summary.json` carries `adoption_pass` and `best_attempt` in each row plus an
`adopted_fits` count in its metadata.

**Behavior:**
- Spawns one subprocess per fit (sequential; never two JAX processes at once)
- Logs subprocess output to `results/fits/<ID>_<config>.log`
- Aggregates diagnostics into `results/fit_summary.json`
- Prints 3×3 summary table to stdout, with the adoption verdict per cell

**Outputs:**
- `results/fits/*.log` — Per-fit subprocess logs
- `results/fits/*.npz` — Per-fit results (parameters, derived quantities, SFH), written for every
  cell that produced a posterior, adopted or not
- `results/fits/*.json` — Per-fit diagnostics, including `adoption_pass` and `best_attempt`
- `results/fit_summary.json` — Aggregated summary table

## Deliverable 3: Backend Sweep (run_backend_sweep.py)

**Purpose:** Test all inference backends on one galaxy (13097) with one configuration (II).

**Command:**
```bash
cd analysis/paper1
$python_exe run_backend_sweep.py [--methods map,laplace] [--out-dir DIR]
```

**Options:**
- `--methods` — Comma-separated subset of the six below, run in the order given. Default: all six. A name outside the list is rejected before any fit starts.
- `--out-dir` — Output directory. Default: `results/backend_sweep`. Use a scratch directory to smoke-test the cheap rows without overwriting the paper's results.

**Methods tested (in order):**
1. `map` — Maximum a posteriori (ADAM optimization, 500 steps, 8 restarts)
2. `laplace` — Laplace approximation (Gaussian from the Hessian at the MAP)
3. `mcmc` — tengri's automatic sampler selector, which resolves to NUTS at this dimensionality (D ≤ 20); the row measures the selector, at the same settings as the explicit NUTS row (600 warmup + 4 × 600 draws)
4. `mcmc_nuts` — NUTS sampler (cold + warm compile), 600 warmup + 4 × 600 draws: the grid cell's own budget, so this row is the paper's NUTS fit for 13097/II rather than a cheaper stand-in
5. `mcmc_hmc` — Standard HMC (cold + warm compile)
6. `mcmc_raytrace` — Ray tracing sampler (cold + warm compile); its runner takes `n_burnin`/`n_steps`, not `n_warmup`/`n_samples`

**Behavior:**
- One process, one method after another (never two JAX processes at once)
- Captures wall time (cold = first run including compile, warm = second run in same process; `map` and `laplace` have no warm run and report `N/A`)
- Captures ESS, s/ESS and max R̂ whenever the posterior carries samples
- Point estimates for `map` and `laplace` come from `posterior.params`; the derived quantities are `sed_model.predict` at those parameters merged with the fixed values
- Sampler rows take the median over draws strided across the whole flattened record
- Every row records `dispatched_to`, the backend the fitter ran (e.g. `NUTS (BlackJAX)`) — the point of the `mcmc` row, which names a selector rather than a sampler
- Prints summary table to stdout

**Outputs:**
- `<out-dir>/<method>.npz` — that method's thinned draws, one array per parameter under the parameter's own name (the schema `fit_one.py` writes, at the same `MAX_SAVED_DRAWS` cap), plus the diagnostics. `map` (and any method whose backend returns no draws; `laplace` returns 2000 draws by default and so is saved like the samplers) contributes its point estimate as length-1 arrays. Loads with `np.load(path, allow_pickle=False)`: a `None` diagnostic (the warm time of a row that has no warm run) is dropped and strings are stored as `np.str_` arrays, both of which the JSON keeps
- `<out-dir>/<method>.json` — Method diagnostics, including the `None` warm times
- `<out-dir>/summary.json` — Aggregated backend comparison

## NUTS Settings (Canonical from Quickstart)

All NUTS fits use:
```python
n_warmup=600
n_samples=600
n_chains=4
n_burnin=0
dense_mass_matrix=False
max_tree_depth=10
```

**Adoption bar:** 0 divergences and max split-R̂ < 1.01

**Retune logic (if fit fails bar):** see "Retune policy" under Deliverable 2 — attempt 2 raises
`target_accept_rate` to 0.95 and attempt 3 to 0.99, both on the base warmup and the same diagonal
mass matrix; attempt 4 onward doubles the warmup at 0.99. The mass matrix is never switched to
dense, 3 attempts by default, and the best attempt so far is saved after every miss.

## Error Floor

All fits apply a 5% systematic flux-error floor in quadrature:
```python
sigma_floor = sqrt(sigma_measurement^2 + (0.05 * fnu)^2)
```

## Photometry (candels_io.py)

- AB zero point: `AB_ZERO_POINT_ERG = 3.631e-20` erg s⁻¹ cm⁻² Hz⁻¹ (3631 Jy).
  The first grid ran with 3.63e-23 (1000× too faint) and every NUTS
  transition diverged (#2089).
- Column map `CANDELS_TO_TENGRI`: ACS F435W/F606W/F775W/F814W/F850LP,
  WFC3 F098M/F105W/F125W/F160W and IRAC 3.6/4.5/5.8/8.0 use their own
  tengri curves. **Stand-ins:** ISAAC Ks and HAWK-I Ks both use `vista_ks`
  (no ISAAC or broadband HAWK-I curve in the registry). **Unmapped:**
  CTIO U and VIMOS U (no CTIO or VIMOS U curve in the registry).
- One Ks band per galaxy: ISAAC first, HAWK-I only when ISAAC is undetected.
- A mapped column missing from the catalog header raises; nothing is dropped
  silently.
- Driver timeout per cell: `DEFAULT_FIT_TIMEOUT_S = 21600` s — three attempts at the base warmup, but attempts 2 and 3 run at target acceptance 0.95 and 0.99, which shrink the step and deepen the trees, so the per-draw cost rises even though the draw count does not; a dead fit still ends in ~10 min (measured 2026-08-30: a
  healthy 600-warmup + 4x600-draw configuration I cell takes ~22 min, configurations II/III
  cost 2-3x per draw, and each retune at a higher target acceptance costs more than the first
  attempt at the same draw count — configuration III's first attempt alone measured 96 min, so
  three attempts on III can approach the cap. A dead fit finishes in ~10
  min rather than hanging, so the larger cap costs nothing in hang detection).

## Linting

All scripts pass ruff checks:
```bash
$venv/bin/ruff check analysis/paper1/fit_one.py analysis/paper1/run_candels_fits.py analysis/paper1/run_backend_sweep.py
```

## Expected Runtime

Measured 2026-08-30 on CPU, not estimated:

- Single fit, configuration I (D=5), 600 warmup + 4×600 draws: ~22 minutes. Configurations II/III (D=8, D=11) cost 2–3× per draw; a retune keeps the warmup and raises the target acceptance (0.95, then 0.99), which deepens the NUTS trees, so a retuned attempt costs more than the first at the same draw count; configuration III's first attempt alone measured 96 minutes. Budget a few hours for a cell that needs all three attempts.
- 3×3 grid (9 fits sequential): several hours; the per-cell cap is `DEFAULT_FIT_TIMEOUT_S = 21600` s (three attempts).
- Backend sweep (6 methods, one galaxy): `map` and `laplace` are seconds to a couple of minutes each; the four sampler rows dominate and each runs cold + warm.

## Figure 8: Gradient Sensitivity (Jacobian & Fisher Matrix)

**Purpose:** Compute and visualize the gradient sensitivity of photometric flux to model parameters, demonstrating end-to-end differentiability.

**Command:**
```bash
cd <path-to-your-tengri-checkout>
PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python analysis/paper1/fig08_gradient_sensitivity.py
```

**Configuration:**
- Model: Configuration II (DPL SFH, two-component Calzetti dust, DL07 dust emission)
- Redshift: z = 1.1
- Filters: 13 bands (HST ACS F435W, F606W, F775W, F814W, F850LP; HST WFC3 F105W, F125W, F160W; VISTA Ks; Spitzer IRAC 3.6, 4.5, 5.8, 8.0 µm)
- Free parameters: 8 (α, β, τ_peak, age, log M*, log Z, τ_bc, τ_diff)
- Noise assumption: 5% fractional flux uncertainty

**Panels:**
- (a) Jacobian heatmap: ∂log₁₀ f_b / ∂θ_k with diverging colormap (RdBu_r) centered on zero
- (b) Fisher information correlation matrix: derived from F = J^T N^{-1} J with forecast 1-sigma marginal uncertainties

**Outputs:**
- `figures/fig08_gradient_sensitivity.pdf` — Publication-ready figure (two panels side-by-side)
- `figures/fig08_gradient_sensitivity.png` — Raster version
- `results/fig08_gradient_sensitivity_data.json` — Jacobian matrix, correlation matrix, forecast sigmas, filter names, parameter names/values, timings, metadata

**Timings (CPU):**
- Forward pass: ~0.18 ms median
- Jacobian computation: ~2.1 ms median (~11.6× forward time)

## Figure 3, panel (a): forward-model speed (re-measured 2026-08-30 on a quiet machine)

Run the benchmark with no other JAX process on the machine (check `ps` for fit or sweep processes first),
stamp the log, parse it, and regenerate the figure:

```bash
cd <path-to-your-tengri-checkout>
log=analysis/paper1/results/fig03_bench_forward_$(date +%F).log
{ echo "START $(date '+%F %T') host=$(hostname) commit=$(git rev-parse --short HEAD) jax=$(.venv/bin/python -c 'import jax;print(jax.__version__)')"
  PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu caffeinate -i .venv/bin/python bench/scripts/benchmark_forward_model.py
  echo "EXIT=$? $(date '+%F %T')"; } > "$log" 2>&1
.venv/bin/python analysis/paper1/parse_forward_benchmark.py --log "$log" --out analysis/paper1/results/fig03_bench_forward_$(date +%F).json
PYTHONPATH=$PWD/src .venv/bin/python analysis/paper1/fig03_precompute.py --bench-json analysis/paper1/results/fig03_bench_forward_$(date +%F).json
```

The archived run is `results/fig03_bench_forward_2026-08-30.{log,json}` (Apple M4 Pro, JAX 0.11.1, float64,
SDSS ugriz at z=0.1, 200 timed calls after 5 warmup calls, commit 996b72ccb, no other fit process running).

## Figure 5: CANDELS Galaxies × Configurations

**Purpose:** Generate a 3 rows (galaxies) × 3 columns (panels) figure showing three CANDELS galaxies 
(IDs 13097, 15336, 24497) across three model configurations (I, II, III), with panels showing:
- (a) Observed photometry and posterior-predictive photometry bands
- (b) Star formation history with 16-84% confidence band
- (c) Joint posterior contours of log M* and log SFR(100 Myr)

**Command:**
```bash
cd <path-to-your-tengri-checkout>
export PYTHONPATH=<checkout-that-produced-the-fits>/src
export JAX_PLATFORMS=cpu
python analysis/paper1/fig05_candels_galaxies.py --results-dir <results-dir>
```

**Outputs:**
- `analysis/paper1/figures/fig05_candels_galaxies.pdf` — Publication-quality PDF
- `analysis/paper1/figures/fig05_candels_galaxies.png` — PNG for preview
- `analysis/paper1/results/fig05_candels_galaxies_data.json` — Sidecar with diagnostics, posterior statistics, and metadata

**Notes:**
- Script handles missing/incomplete fits gracefully, marking cells as "pending" or "failed adoption bar"
- Reads fit results from the fix-2089-candels worktree (default path; override with `--results-dir`)
- Uses the same forward model configuration as the fits for consistency
- Computes derived quantities (M*, SFR) from up to 200 posterior samples per galaxy-config combination
- Colors: Configuration I = #0072B2 (blue), II = #E69F00 (orange), III = #009E73 (green)

## Figure 6: Overlay tengri posteriors on published SED-fitting results

**Purpose:** Visualize tengri posterior constraints overlaid on published code values
from the CANDELS SED-fitting workshop. Shows joint (M*, SFR) posteriors as 68% contours
for three galaxy types, with individual codes plotted as grayscale markers with error bars.

**Prerequisites:** Fit results from `run_candels_fits.py` (executed separately on the grid).

**Commands (from paper1 directory):**
```bash
# 1. Ingest published workshop results
$python_exe ingest_art_sedfitting.py

# 2. Generate figure (script handles incomplete fits gracefully)
PYTHONPATH=$PWD/../../src:$PWD/../.. JAX_PLATFORMS=cpu $python_exe fig06_code_overlay.py \
  --results-dir ../fix-2089-candels/analysis/paper1/results/fits \
  --max-samples 200
```

**Outputs:**
- `figures/fig06_code_overlay.pdf` — Publication-ready figure (single-column width)
- `figures/fig06_code_overlay.png` — High-resolution raster version
- `results/fig06_code_overlay_data.json` — Numerical data sidecar with:
  - Published code central values and 68% confidence intervals for each galaxy
  - tengri posterior percentiles (16, 50, 84) for surviving and formed stellar mass
  - SFR values in log space [Msun/yr]
  - List of pending cells (fits still running)
  - Published inter-code ranges and tengri inter-configuration ranges

**Notes:**
- Surviving mass is computed via `model.predict_properties` with at most 200 posterior samples
- Formed mass (total formed stellar mass) is plotted as a small open marker at each configuration's median
- Figure panels are fixed to x ∈ [9.5, 11.8], y ∈ [-1.0, 3.5] for direct comparison
- Published codes use distinct grayscale levels and marker styles (see legend)
- tengri configurations use distinct colors: I = blue (#0072B2), II = orange (#E69F00), III = green (#009E73)

## Figure 7: Backend Comparison

**Purpose:** Compare all inference backends (MAP, Laplace, NUTS, HMC, Ray Tracing) on one galaxy
(13097) with one configuration (II), showing marginal posterior distributions and timing analysis.

**Prerequisites:** Backend sweep results from `run_backend_sweep.py` (reads JSON and NPZ files).

**Command (from paper1 worktree root):**
```bash
cd <path-to-your-tengri-checkout>
PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python analysis/paper1/fig07_backends.py \
  --sweep-dir <sweep-dir> \
  --out-dir analysis/paper1
```

**Arguments:**
- `--sweep-dir`: Path to backend sweep results directory (default: `analysis/paper1/results/backend_sweep`)
- `--out-dir`: Output directory for figures and sidecar (default: `analysis/paper1`)

**Outputs:**
- `figures/fig07_backends.pdf` — Publication-ready two-column figure
- `figures/fig07_backends.png` — High-resolution raster version
- `results/fig07_backends_data.json` — Complete diagnostics sidecar including per-backend budgets, wall times, ESS metrics, and agreement statistics

**Figure panels:**
- **Left (3 columns):** Marginal posterior distributions for log M*, log SFR/100Myr, and tau_diff
  - MAP: vertical black line
  - Laplace: gray Gaussian approximation (from ESS-derived width)
  - Samplers (NUTS, HMC, Ray Tracing): dashed colored lines at medians (Okabe-Ito palette)
- **Right:** Timing comparison (log-scale wall time bars) with per-sampler efficiency (s/ESS annotations)
  - Rows ordered: MAP, Laplace, Auto (MCMC dispatcher), NUTS, HMC, Ray Tracing
  - Budgets annotated in italic text to the left of each row

**Noted limitations:**
- Sampler budgets from run_backend_sweep.py (lines 99–210): MAP (500 steps + 8 restarts); Laplace (Gaussian);
  MCMC/Auto (600+600x2); NUTS (600+600x2); HMC (200+300x4, L=50); Ray Tracing (400+400x2, step=0.05)
- The sweep script saves JSON diagnostics only; full posterior samples are not persisted to NPZ files
- Agreement summary (sidecar) computed only when all samplers present (requires full sweep, not smoke subset)
