# Reproducing Synthesizer's physics with tengri

This folder places Synthesizer (Lovell et al. 2025, OJA, doi:10.33232/001c.145766;
Roper et al. 2026, JOSS, doi:10.21105/joss.09436 — **cite both**) next to tengri
component by component — same parameters, same
units, one figure per physics block — with a focus on the **Unified AGN** model
(§9). tengri and Synthesizer are **independent codes**; this is a peer-to-peer
comparison, exactly like the CIGALE, Prospector, and BAGPIPES notebooks.

Synthesizer builds spectra by extracting from pre-computed HDF5 grids and walking
an `EmissionModel` tree. The left panel of every figure is Synthesizer, driven
through `synthesizer.*` (`Grid`, `parametric.Stars`, `BlackHole`, `UnifiedAGN`);
the right panel is tengri. For the stellar SSP (§1) both read the *same*
templates — Synthesizer's `test_grid` `incident` spectra re-shaped into tengri's
DSPS HDF5 — so the §1 residual is interpolation alone. For the AGN line regions
(§9c/§9d) tengri reads the *same* Synthesizer Cloudy AGN grids through its
`compute_nlr_sed_synthesizer` / `compute_blr_sed_synthesizer` adapters.

## Files

- **`01_synthesizer.py`** — the notebook, jupytext percent format.
- **`_drivers/`** — code-side glue:
  - `units.py` — Synthesizer (`unyt` erg/s/Hz) → tengri (plain erg/s/Hz). Ships
    `verify_unit_conversion(rtol=1e-3)`; the notebook trips at Setup if the
    converter drifts.
  - `synthesizer_driver.py` — thin wrappers around the Synthesizer forward model
    (stellar grids, parametric SFH, attenuation/dust/IGM laws, and the
    `UnifiedAGN` black-hole emission tree), returning everything in tengri's
    convention. Heavy grids are cached at module level.
  - `synthesizer_ssp_to_dsps.py` — one-off repackaging of Synthesizer's stellar
    `incident` grid into a DSPS-shaped HDF5 (run automatically by the notebook's
    "Common stellar grid" cell if the file is absent).
  - `data/` — the repackaged grid lands here and is **git-ignored**.
- **`_figs/`** — generated figures.

## Prerequisites

```bash
pip install cosmos-synthesizer jupytext jupyter
```

Then download Synthesizer's **test** grids (the notebook runs on these — no
production grids needed):

```bash
synthesizer-download --stellar-test-grids --agn-test-grids --dust-grid
```

By default these land in Synthesizer's application-support directory; point the
driver at a custom location with `export SYNTHESIZER_GRID_DIR=/path/to/grids`.

> **Grids.** The test grids are the same physics as the production grids
> (Box: <https://sussex.app.box.com/v/SynthesizerGrids>) at lower resolution —
> the AGN test grids sample each photoionization axis at just two nodes. Because
> both codes read the *same* file, the coarseness never appears as a
> tengri-vs-Synthesizer disagreement. Swapping in a production grid (and
> re-pointing `SYNTHESIZER_GRID_DIR`) is a drop-in higher-resolution re-render.

## Running

```bash
jupytext --to ipynb 01_synthesizer.py
PYTHONPATH=$PWD/../..:$PWD/../../src \
  SYNTHESIZER_GRID_DIR="$HOME/Library/Application Support/Synthesizer/grids" \
  jupyter nbconvert --to notebook --execute --inplace 01_synthesizer.ipynb \
  --ExecutePreprocessor.timeout=900
```

Expected runtime: a few minutes on a CPU. The §8 nebular and §7 panchromatic
panels use tengri's Cue emulator (needs `data/cue_weights.npz`); the first-time
JAX compilation dominates and subsequent runs reuse the persistent cache.

## What the notebook covers

§1 SSPs · §2 delayed-τ SFH · §3 stellar SED · §4 attenuation curves
(Calzetti, power-law) · §5 attenuation applied · §6 dust IR + energy balance
(Draine & Li 2007) · §7 panchromatic · §8 nebular (Synthesizer Cloudy grid vs
tengri Cue) · **§9 AGN — the Unified AGN model** · §12 IGM (Inoue 2014 / Madau 1995).

X-ray and radio are skipped — Synthesizer has no counterpart. The numbering keeps
the gap (§10 X-ray, §11 radio) to line up with the CIGALE master sequence.

### §9 — the Unified AGN model (the focus)

- **§9a disc** — Synthesizer disc vs tengri `kubota_done` (qsosed) and
  `schartmann2005` (broken power law).
- **§9b** disc incident → escaped + transmitted → observed (the disc–line-region
  geometry).
- **§9c NLR** / **§9d BLR** — tengri reads the same Synthesizer Cloudy grids via
  `compute_nlr_sed_synthesizer` / `compute_blr_sed_synthesizer`; line positions
  match (the spectra differ in line representation — narrow Gaussians vs grid bins).
- **§9e torus** — Synthesizer's blackbody torus vs tengri's Nenkova / two-temperature.
- **§9f** — the full `UnifiedAGN` spectrum and the **hard inclination mask**
  (disc + BLR vanish once `inclination + θ_torus > 90°`, the Type-1 → Type-2
  transition).

## Star formation histories — the class-to-type map

The notebook compares one SFH (§2, delayed-τ). The rest of `synthesizer.parametric.SFH`
maps onto tengri's SFH registry as follows; `sfh={'type': '<name>'}` selects a row,
and `tengri.list_sfh_models()` is the live menu.

| Synthesizer | tengri | Convention notes |
|---|---|---|
| `Constant(min_age, max_age)` | `const` | `max_age` → `start_gyr`, `min_age` → `end_gyr`; both are lookback ages. |
| `Gaussian(peak_age, sigma)` | `norm` | Synthesizer `exp(-((t-peak)/σ)²)`, tengri `exp(-(t-peak)²/2w²)`: `w = σ/√2`. |
| `Exponential(tau>0)`, `DecliningExponential` | `declining_exp` | Same closed form. `Exponential(tau<0)` (rising) is tengri's `exp`. |
| `TruncatedExponential` | `declining_exp` | tengri has no separate truncation parameter; bound the SFH age instead. |
| `DelayedExponential(tau)` | `delayed` | Same `t exp(-t/τ)`; tengri evaluates it on cosmic elapsed time. |
| `LogNormal` | `lnorm` | **Different functional forms.** Synthesizer is a log-normal PDF in cosmic time; tengri's `lnorm` is a Gaussian in log₁₀(lookback) with no 1/t Jacobian. |
| `DoublePowerLaw(peak_age, alpha, beta)` | `dpl` | **Different variable.** Synthesizer evaluates `((age/peak)^α + (age/peak)^β)^-1` in **lookback age**, monotonically decreasing for positive exponents; tengri follows BAGPIPES (Carnall+2018) and evaluates `1/(x^α + x^-β)` in **cosmic time**, which peaks. |
| `DenseBasis` | `dense_basis` | `dense_basis_pure` drops the SFR-constraint points; the swap is automatic when a compositor is present. |
| `Continuity` | `continuity` | Same Leja+2019 log-SFR-ratio parametrization; tengri declares the smoothness prior as StudentT(0, 0.3, df=2). |
| `ContinuityFlex` | `continuity_flex` | |
| `Dirichlet` | `dirichlet` | Leja+2017 stick-breaking; tengri's auxiliary variables carry Beta(1,1). |
| `ContinuityPSB` | `psb_suess2022` | Suess+2021/2022: youngest bin, flexible zone, fixed old bins. |
| `CombinedSFH(sfhs, weights)` | `sfh={'type': ['a', 'b']}` | **Different normalization.** Synthesizer weights raw shapes, and the sum is normalized to the `Stars` object's `initial_mass`; each member of tengri's list carries its own `log_total_mass`, so the relative weighting is the mass difference and there is no separate weights vector. |
| `Stochastic(kernel=DampedRandomWalk(σ, τ))` | `sfh={'type': ['<mean>', 'field'], 'psd_sigma': σ, 'psd_tau_myr': 1000·τ_Gyr}` | Same kernel; see below. |

### Stochastic SFH ↔ the `field` compositor

Both codes draw log-SFR fluctuations about a mean SFH from a Gaussian process with
a damped-random-walk covariance (the Iyer et al. 2024, arXiv:2208.05938, GP + PSD
formalism). Synthesizer's kernel is `C(Δt) = σ² exp(-|Δt|/τ)` in dex²; tengri's is
`K(Δt) = (σ ln10)² exp(-|Δt|/τ)` in natural log, which is the same function once
divided by `(ln10)²`. `psd_sigma` therefore *is* Synthesizer's `sigma` in dex and
`psd_tau_myr` its `tau` in Myr — no factor of 2π, and no natural-log rescaling.
Measured on a 100 × 100 Toeplitz matrix at σ = 0.3 dex, τ = 1 Gyr, the two agree to
2.7 × 10⁻¹⁴ relative in the worst entry; a 4000-draw Monte Carlo through
`compute_field_gp` on tengri's 256-node log-age grid gives a variance of
0.089844 dex² against σ² = 0.09 (0.17 %) and an autocorrelation of 0.3593 at a lag
of 1.000047 Gyr against e⁻¹ = 0.36788. Through the public build grammar and
`spec.sample`, 64 field draws give 0.3039 dex. Pinned by
`tests/crossval/test_synthesizer_stochastic_sfh.py`.

Three differences remain, all of them conventions rather than disagreements.

**Log-normal centering.** tengri applies `exp(x - K(0)/2)`, so the ensemble-mean
*linear* SFR equals the mean SFH. Synthesizer forms `10**(log10(base) + fluctuations)`
with no correction, so its ensemble-mean linear SFR sits `exp((σ ln10)²/2) - 1` above
its base SFH — **+26.95 %** at σ = 0.3 dex, and +94.0 % at σ = 0.5 dex. The
absence preserves the mean *log* SFR instead. A normalization carried between the
two codes has to divide this out.

**One realization vs a fitted field.** Synthesizer draws a single realization at
construction from `random_seed` and freezes it. tengri carries the standardized
latent `sfh_field_xi` — an N(0, I) vector, one entry per SFH grid node — as a
parameter, so the same construction serves both mock generation (draw it with
`spec.sample(key)`) and inference (the sampler explores it). The gallery example
`examples/sfh/plot_stochastic_sfh_tau_sweep.py` runs the same τ sweep as Synthesizer's documentation page,
holding one draw fixed across three timescales.

**Kernel menu and grid.** `DampedRandomWalk` is the only kernel on Synthesizer's
`main`. tengri's `field` defaults to the same DRW and additionally offers Matern
(`psd_matern`, which recovers the DRW at ν = 0.5) and the two-component extended
regulator of Tacchella+2020 / Caplar & Tacchella 2019 (`psd_extended_regulator`).
Synthesizer samples on a uniform cosmic-time grid of `n_grid=1000` points from the
Big Bang to the observation epoch; tengri samples on a log-spaced lookback grid of
`n_grid` nodes (256 by default) spanning 1 Myr to ~13.8 Gyr. The process is
stationary in physical time in both, so this is a difference in where the field is
evaluated, not in what it is.

## What the comparison found

The per-section scalars printed by the notebook are the quantitative record; the
figures in `_figs/` are the visual one. The shared SSP grid (§1), the delayed-τ
SFH (§2), the DL07 dust IR (§6, far-IR peak agreeing to ~1 µm), and the IGM (§12)
reproduce Synthesizer to floating point or a fraction of a percent. The nebular
block (§8) is the deliberate exception — Synthesizer's Cloudy grid and tengri's
Cue emulator use different photoionization inputs, so the Hα ratio is reported
rather than forced to agree. In §9, the disc, torus, and inclination treatment
are independent implementations and compared on shape/amplitude; the line regions
share grids and so match in line content.
