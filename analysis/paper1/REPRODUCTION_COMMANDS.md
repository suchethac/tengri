# Paper I Demonstration Fits — Reproduction Commands

## Environment Setup (from worktree root)

```bash
cd /Users/suchethacooray/Projects/tengri/.claude/worktrees/paper1
export PYTHONPATH=$PWD/src
export JAX_PLATFORMS=cpu
python_exe=/opt/anaconda3/bin/python3
```

## Deliverable 1: Single Fit (fit_one.py)

**Purpose:** Fit a single galaxy with a specified SED model configuration via NUTS MCMC.

**Usage:**
```bash
$python_exe analysis/paper1/fit_one.py --galaxy ID --config {I,II,III} --method mcmc_nuts --out DIR [--seed N]
```

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
- `results/fits/<ID>_<config>.npz` — Parameter samples (thinned to ≤1000/chain), derived quantities (stellar mass, SFR, dust), SFH posterior grid
- `results/fits/<ID>_<config>.json` — Diagnostics: divergences, max R̂, min ESS, wall time, s/ESS

## Deliverable 2: 3×3 Grid (run_candels_fits.py)

**Purpose:** Run 3 galaxies × 3 configurations = 9 NUTS fits sequentially.

**Command:**
```bash
cd analysis/paper1
$python_exe run_candels_fits.py
```

**Behavior:**
- Spawns one subprocess per fit (sequential; never two JAX processes at once)
- Logs subprocess output to `results/fits/<ID>_<config>.log`
- Aggregates diagnostics into `results/fit_summary.json`
- Prints 3×3 summary table to stdout

**Outputs:**
- `results/fits/*.log` — Per-fit subprocess logs
- `results/fits/*.npz` — Per-fit results (parameters, derived quantities, SFH)
- `results/fits/*.json` — Per-fit diagnostics
- `results/fit_summary.json` — Aggregated summary table

## Deliverable 3: Backend Sweep (run_backend_sweep.py)

**Purpose:** Test all inference backends on one galaxy (13097) with one configuration (II).

**Command:**
```bash
cd analysis/paper1
$python_exe run_backend_sweep.py
```

**Methods tested (in order):**
1. `map` — Maximum a posteriori (ADAM optimization)
2. `laplace` — Laplace approximation (diagonal covariance)
3. `mcmc_nuts` — NUTS sampler (cold + warm compile)
4. `mcmc_hmc` — Standard HMC (cold + warm compile)
5. `mcmc_raytrace` — Ray tracing sampler (cold + warm compile)
6. `vi` — NIFTy geoVI (cold + warm compile)
7. `nss` — Nested sampling (experimental; may skip if >15 min or raises)

**Behavior:**
- One subprocess per method
- Captures wall time (cold = first run including compile, warm = second run in same process)
- Captures ESS and s/ESS for samplers
- Captures point estimates (mean/median) for optimization and VI methods
- Prints summary table to stdout

**Outputs:**
- `results/backend_sweep/<method>.npz` — Method-specific results
- `results/backend_sweep/<method>.json` — Method diagnostics
- `results/backend_sweep/summary.json` — Aggregated backend comparison

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

**Retune logic (if fit fails bar):**
1. On first failure: double warmup
2. For D ≥ 8: toggle `dense_mass_matrix`
3. Re-run up to 2 attempts

## Error Floor

All fits apply a 5% systematic flux-error floor in quadrature:
```python
sigma_floor = sqrt(sigma_measurement^2 + (0.05 * fnu)^2)
```

## Linting

All scripts pass ruff checks:
```bash
$venv/bin/ruff check analysis/paper1/fit_one.py analysis/paper1/run_candels_fits.py analysis/paper1/run_backend_sweep.py
```

## Expected Runtime (estimate)

- Single fit (NUTS, D=5–11): 5–15 minutes per fit
- 3×3 grid (9 fits sequential): ~90–150 minutes
- Backend sweep (7 methods, one galaxy): ~30–60 minutes

Total: ~2–4 hours for all three deliverables

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
