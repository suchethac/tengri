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

`--n-warmup`, `--n-samples` and `--n-chains` default to the paper's 600, 600 and 4 (the settings below); the examples and the grid use those defaults. Pass small values to smoke-test the pipeline in a few minutes per configuration rather than half an hour: the fit, the retune and the per-attempt JSON always run, but the NPZ is written only when an attempt clears the adoption bar (0 divergences, max split-R̂ < 1.01), which a 20-draw budget usually does not (configuration II did at 20/20/1; I and III needed 200/200/4).

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
0.95; attempt 3 keeps 0.95 and doubles the warmup (each further attempt doubles it again). The
mass matrix is never switched to dense by a retune — measured on cell 13097/II (600 warmup +
4×600 draws, D = 8), attempt 1 on a diagonal mass matrix gave 3/2400 divergences at max R̂
1.0014 and the old dense-mass retune gave 79/2400 at 1.023 — because divergences at the 0.1%
level with R̂ ≈ 1.00 are a step-size problem, not a mass-matrix one. The default is 3 attempts.

**Cells that never clear the adoption bar are still saved.** If no attempt reaches 0 divergences
and max R̂ < 1.01, the best attempt (fewest divergences, then lowest max R̂) is written to the
NPZ and the JSON with `adoption_pass: false` and `best_attempt: <n>`, and `fit_one.py` exits 0;
only a cell in which every attempt raised is a failure. The summary table prints the adoption
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
3. `mcmc` — tengri's automatic sampler selector, which resolves to NUTS at this dimensionality (D ≤ 20); the row measures the selector, at the same settings as the explicit NUTS row
4. `mcmc_nuts` — NUTS sampler (cold + warm compile)
5. `mcmc_hmc` — Standard HMC (cold + warm compile)
6. `mcmc_raytrace` — Ray tracing sampler (cold + warm compile); its runner takes `n_burnin`/`n_steps`, not `n_warmup`/`n_samples`

**Behavior:**
- One process, one method after another (never two JAX processes at once)
- Captures wall time (cold = first run including compile, warm = second run in same process; `map` and `laplace` have no warm run and report `N/A`)
- Captures ESS, s/ESS and max R̂ whenever the posterior carries samples
- Point estimates for `map` and `laplace` come from `posterior.params`; the derived quantities are `sed_model.predict` at those parameters merged with the fixed values
- Sampler rows take the median over draws strided across the whole flattened record
- Prints summary table to stdout

**Outputs:**
- `<out-dir>/<method>.npz` — Method-specific results
- `<out-dir>/<method>.json` — Method diagnostics
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

**Retune logic (if fit fails bar):**
1. On first failure: double warmup
2. For D ≥ 8: toggle `dense_mass_matrix`
3. Re-run up to 2 attempts

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
- Driver timeout per cell: `DEFAULT_FIT_TIMEOUT_S = 14400` s (measured 2026-08-30: a
  healthy 600-warmup + 4x600-draw configuration I cell takes ~22 min, a retune doubles
  the warmup for ~50 min, and configurations II/III cost 2-3x per draw, reaching
  100-150 min — 7200 s could still kill a healthy retune. A dead fit finishes in ~10
  min rather than hanging, so the larger cap costs nothing in hang detection).

## Linting

All scripts pass ruff checks:
```bash
$venv/bin/ruff check analysis/paper1/fit_one.py analysis/paper1/run_candels_fits.py analysis/paper1/run_backend_sweep.py
```

## Expected Runtime

Measured 2026-08-30 on CPU, not estimated:

- Single fit, configuration I (D=5), 600 warmup + 4×600 draws: ~22 minutes. Configurations II/III (D=8, D=11) cost 2–3× per draw; a retune doubles the warmup, so a retuned cell reaches ~50 minutes for I and 100–150 minutes for II/III.
- 3×3 grid (9 fits sequential): several hours; the per-cell cap is `DEFAULT_FIT_TIMEOUT_S = 14400` s.
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
