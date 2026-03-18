# diffsed Development Handoff

**Last updated:** 2026-03-14 (session 3)
**Repo:** `~/Projects/diffsed/`
**Paper draft:** `~/writing-workspace/projects/differentiable_psd_sed_fitting/`

---

## Executive Summary

**What it is:** A fully differentiable galaxy SED fitting code using JAX. Treats the star formation history as an IFT correlated field (GP with PSD-governed correlation structure) on top of a smooth mean SFH. End-to-end differentiable: PSD params → GP latent field → SFH → DSPS SPS → dust → photometry/spectra.

**Code name:** Currently `diffsed` (working name). Final name TBD — will be renamed before public release.

**Locations:**
- Code: `~/Projects/diffsed/` (this repo)
- Paper draft: `~/writing-workspace/projects/differentiable_psd_sed_fitting/`
- SSP data: `~/Projects/diffsed/data/` (not in git, 64 MB HDF5 files)

**Environment:**
```bash
cd ~/Projects/diffsed
source .venv/bin/activate          # Python 3.12 venv
pytest tests/ -q                    # 302 tests pass
jupyter lab notebooks/              # 6 tutorial notebooks
```

**Key dependencies:** JAX ≥0.4.20, DSPS ≥0.3, NIFTy8.re ≥8.5, BlackJAX ≥1.3, optax ≥0.2, h5py, matplotlib, scipy

**What's done:**
- Complete code with 4 inference methods (MAP, Ray Tracing Sampler, NUTS, geoVI), 302 tests passing
- 6 tutorial notebooks, all executing
- Paper I draft complete (20 pages, compiles), all sections written
- 5 of 8 paper figures generated from analysis scripts
- Hierarchical PSD inference infrastructure implemented

**What the paper needs before submission:**
1. **Improve hierarchical PSD recovery** — current flat-vector approach is biased. Need to use NIFTy's `CorrelatedFieldMaker` for joint PSD hyperparameter learning (see Eberle+2025, Roth+2024 for patterns)
2. **Production figure runs** — current figures use 1-3 mocks per regime; need 50-100 for publishable statistics
3. **Fig 1 schematic** — framework overview diagram (only figure not yet created)
4. **Wire remaining figures into LaTeX** — Figs 2-3 (from notebooks) need `\includegraphics`

**What Paper II needs (real data):** Apply the same framework to SDSS spectra, intermediate-z photometry, JWST data.

---

**Status:** 302 tests pass. All 6 notebooks execute. 4 inference methods. Hierarchical PSD inference implemented. Paper draft complete (20 pages). 5 paper figures generated.

---

## High-Level API

```python
from diffsed import (
    Model, ParamSpec, Uniform, Gaussian, Fixed, Fitter, Posterior,
    HierarchicalFitter, load_ssp_data, load_filter_set, sample_raytrace,
)

# Define model
spec = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0), sfh_beta=Uniform(0.3, 2.0),
    sfh_tau_peak_gyr=Uniform(0.5, 10.0), sfh_peak_sfr=Uniform(0.1, 50.0),
    psd_sigma=Uniform(0.1, 3.0), psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Gaussian(-0.3, 0.2, lo=-2.0, hi=0.2),
    dust_tau_bc=Uniform(0.0, 3.0), dust_tau_diff=0.3, dust_slope=-0.7,
    redshift=0.1, stochastic=True, n_grid=128,
)
model = Model(spec, load_ssp_data("data/ssp.h5"), filters=load_filter_set([...]))
mock = model.mock(spec.sample(key), snr=20.0, key=noise_key)

# Fit: MAP → Ray Tracing or geoVI (equal-priority primaries)
fitter = Fitter(model, mock.flux_obs, mock.noise)
result_map = fitter.run("map", n_steps=2000, learning_rate=0.03)
result_rt  = fitter.run("raytrace", init_from=result_map, n_burnin=100, n_steps=300)
result_vi  = fitter.run("geovi", init_from=result_map, n_iterations=15, n_posterior_samples=80)

# Results
posterior.summary()                        # median ± 68% CI
posterior.effective_sample_size()           # ESS per parameter
posterior.diagnostics_summary()            # formatted table with ESS
posterior.plot_corner(truths=true_params)   # KDE contour triangle plot
model.plot_sfh_posterior(posterior, ...)    # SFH with 16-84% fill

# Hierarchical: shared PSD across N galaxies
hfitter = HierarchicalFitter(model_factory, galaxies)
result = hfitter.run("raytrace", n_burnin=100, n_steps=300)
result.summary()  # posterior on shared (σ_PSD, τ_PSD)
```

---

## Modules

| Module | Purpose | Tests |
|--------|---------|-------|
| `distributions.py` | Uniform, Gaussian, LogUniform, Fixed (JAX-jittable) | 39 |
| `param_spec.py` | ParamSpec: parameter defs, validation, sampling | 28 |
| `model.py` | Model: forward model, mock generation, plotting, photometry precomputation | 24 |
| `fitter.py` | Fitter: MAP, Ray Tracing, NUTS, geoVI, MGVI — all with burn-in | 8 |
| `raytrace_jax.py` | Ray Tracing Sampler (Behroozi 2025, Apache 2.0) — Snell's law MCMC | 6 |
| `posterior.py` | Posterior: summary, resample, corner plot overlay, autocorrelation, ESS, ArviZ | 9 |
| `hierarchical.py` | HierarchicalFitter: shared PSD via CorrelatedFieldMaker (geoVI/MGVI/RT) | TBD |

---

## Inference Methods (equal priority for RT and geoVI)

| Method | How it works | When to use | Smooth (5D) | Stochastic (137D) |
|--------|-------------|-------------|-------------|-------------------|
| **MAP (Adam/AdamW/SGD/custom)** | Gradient descent on information Hamiltonian | Point estimates, initialization | ~4s | ~4s |
| **Ray Tracing** | Snell's law MCMC, n(x)=L(x)^{1/(D-1)} | Exact MCMC, stochastic-gradient resilient | ~1s (300 samp) | ~10s (300 samp) |
| **NUTS** | Hamiltonian MC via BlackJAX | Gold-standard validation (low-D only) | ~10s (500 samp) | Too slow |
| **geoVI** | Geometric VI via NIFTy.re | Non-Gaussian posteriors, moderate D | ~65s (80 samp) | ~65s (80 samp) |
| **MGVI** | Metric Gaussian VI via NIFTy.re | Very large problems (D>10^5), fastest VI | ~30s (60 samp) | ~30s (60 samp) |

**Key:** Ray Tracing has ~250× more gradient-noise variance tolerance than HMC/NUTS. geoVI scales to >10^5 params for hierarchical. Both are primary methods.

**Step sizes:** Default `0.03*sqrt(D)` for D≤10, `0.01` for D>10. For hierarchical (D>100), use `0.005`.

**Burn-in:** Both RT and NUTS support `n_burnin` — samples discarded after warmup/before collection. All Posterior objects contain only post-burn-in samples.

---

## Photometry Precomputation (Zacharegkas+2025)

When redshift is fixed and filters are present, `Model.__init__` precomputes SSP broadband fluxes. At inference: `flux = einsum("i,if,if->f", weights, dust_at_eff, ssp_phot)`. **21.6× gradient speedup.** Auto-activates.

---

## Hierarchical PSD Inference

`HierarchicalFitter` shares PSD hyperparameters across N galaxies. Each galaxy retains its own `ξ_i` + physical params.

**Three methods available:**
- `hfitter.run("geovi")` — **Recommended.** Uses NIFTy's `CorrelatedFieldMaker` to learn PSD hyperparameters (fluctuation amplitude ≈ σ_PSD, spectral slope ≈ τ_PSD) jointly inside the generative model. Follows Edenhofer+2024, Eberle+2025 patterns.
- `hfitter.run("mgvi")` — Same as geoVI but uses MGVI (faster per iteration, for very large N). Best for D > 10^5.
- `hfitter.run("raytrace")` — Flat-vector RT with MAP initialization. Works for small N (~5-20), high acceptance (~99%) but mixing limited for large D.

**CorrelatedFieldMaker approach:**
- PSD shape is part of the generative model, not an external parameter
- Uses 4-8 samples per KL iteration (not 80) — following literature best practices
- Per-galaxy params initialized via individual MAP fits
- Tested: 3 galaxies completes in ~90s with geoVI(CFM)

**Reference IFT applications:**
- Edenhofer+2024 (2308.01295): 661M DOF dust map, MGVI, 12 samples, GPU
- Eberle+2025 (2410.14599): eROSITA imaging, geoVI, 8 samples, joint PSD learning
- Roth+2024 (2406.09144): fast-resolve, major/minor cycle scheme
- Terveer+2026 (2602.19864): air shower, 180K params, multi-stage geoVI

---

## Paper Status

**Paper I:** "Information Field Theory for Galaxy SED Fitting: Reconstructing Bursty Star Formation Histories"
**Draft:** 20 pages, compiles cleanly. All sections written.
**Paper II:** Real data application (SDSS, JWST) — future work.

| Section | Status |
|---------|--------|
| §1 Introduction | Complete — IFT motivation, PSD formalism, Ray Tracing + geoVI |
| §2 Methods | Complete — IFT, SFH field, SPS, dust, 5 inference methods (MAP/RT/NUTS/geoVI/MGVI) |
| §3 Recovery Tests | Complete — Tests 1-7 design, mock program |
| §4 Results | Written — SFH recovery, PSD recovery, hierarchical, speed, PPC |
| §5 Discussion | Complete — comparisons, outshining, multi-tracer, extensibility |
| §6 Conclusion | Complete — 7 bullet points with concrete numbers |
| Appendix | Complete — SPS details, obs model, sampler comparison |

### Paper Figures

| Fig | Script | Status |
|-----|--------|--------|
| 1: Framework schematic | Manual/TikZ | Placeholder |
| 2: PSD → SFH | NB01 | ✅ Copied to paper |
| 3: SED + photometry | NB02 | ✅ Copied to paper |
| 4: SFH recovery | `analysis/fig04_sfh_recovery.py` | ✅ Generated (3 mocks, first pass) |
| 5: PSD recovery | `analysis/fig05_psd_recovery.py` | ✅ Generated |
| 6: Hierarchical PSD | `analysis/fig06_hierarchical_psd.py` | ✅ Generated (biased, needs improvement) |
| 7: Speed benchmarks | `analysis/fig07_speed_benchmarks.py` | ✅ Generated |
| 8: Gradient sensitivity | `analysis/fig08_gradient_sensitivity.py` | ✅ Generated |

---

## Analysis Scripts

```
analysis/
├── common.py                      # Shared utilities: mock gen, fitting, metrics
├── fig04_sfh_recovery.py          # SFH recovery grid (4 regimes × phot/spec)
├── fig05_psd_recovery.py          # PSD (σ,τ) corner plots
├── fig06_hierarchical_psd.py      # N-convergence + population distinction
├── fig07_speed_benchmarks.py      # MAP/RT/NUTS/geoVI timing
├── fig08_gradient_sensitivity.py  # Jacobian heatmap
└── figures/                       # Generated PDFs
```

---

## What Needs Doing Next

### Critical (for paper submission)
- [x] **Hierarchical geoVI with CorrelatedFieldMaker**: DONE — `hfitter.run("geovi")` uses NIFTy CFM natively
- [ ] **Production figure runs**: Re-run fig04 with 50-100 mocks per regime, fig06 with N=50-200 galaxies using geoVI(CFM)
- [ ] **Fig 1 schematic**: Draw framework overview diagram (TikZ or manual)
- [ ] **Wire notebook figures into LaTeX**: Figs 2-3 need `\includegraphics` (PNG files already copied)
- [ ] **Tune hierarchical recovery**: Current CFM geoVI gives fluctuations=0.89, slope=-0.13 for truth σ=1.5, τ=30 Myr. Needs more iterations or better initialization.

### Important
- [ ] **Internal param rename**: `sigma_ps` → `psd_sigma` across ~20 files (low-level API)
- [ ] **Paper compile check**: Recompile after MGVI/CFM updates, verify no broken refs

### Nice to have
- [ ] **GPU benchmarks**: Re-run speed comparison on GPU
- [ ] **S/N dependence study**: Vary photometric S/N from 5-100
- [ ] **2D PSD recovery map**: 16×100 mock grid

---

## Key Design Decisions

1. **ParamSpec as single source of truth** — same object for mock generation and inference
2. **Model wraps ForwardModel** — new param names on top, old internals unchanged
3. **Fitter separate from Model** — JAX pattern (model = physics, fitter = inference)
4. **RT and geoVI equal priority** — RT for exact MCMC, geoVI for high-D/hierarchical
5. **NUTS as gold standard** — validation tool, not primary method
6. **Burn-in for all samplers** — properly discarded before Posterior creation
7. **Autocorrelation/ESS** — FFT-based ACF, Geyer (1992) initial positive sequence
8. **Sigmoid transform** — k=1.0, x0=0 so sampler can reach prior edges
9. **Metallicity offset** — log10(Zsun)=-1.848 in PARAM_MAP for DSPS grid compatibility
10. **Photometry precomputation** — Zacharegkas+2025, 21.6× gradient speedup

---

## SSP Data

Real FSPS SSP templates. All files go in `data/` (gitignored via `*.h5`).

**Download from:** https://halos.as.arizona.edu/suchethacooray/dsps_ssp/

| File | Library | Resolution | Size | Use case |
|------|---------|-----------|------|----------|
| `ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5` | MILES | R~2000 | 64 MB | Default (most notebooks) |
| `ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5` | C3K | R~10000 | 109 MB | High-res spectral analysis (NB10) |
| `fsps_prsc_miles_chabrier.h5` | MILES | R~2000 | 64 MB | Without nebular emission |

```bash
# Download all SSP templates
cd ~/Projects/diffsed/data
curl -O https://halos.as.arizona.edu/suchethacooray/dsps_ssp/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5
curl -O https://halos.as.arizona.edu/suchethacooray/dsps_ssp/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5
```

---

## Dependencies

```
jax>=0.4.20, dsps>=0.3, h5py>=3.0, matplotlib>=3.7, scipy  (core)
nifty8[re]>=8.5  (geoVI)
blackjax>=1.3    (NUTS)
optax>=0.2       (MAP)
```

---

## How to Resume

```bash
cd ~/Projects/diffsed
source .venv/bin/activate
pytest tests/ -q                    # should show 302 passed
python -c "from diffsed import Model, ParamSpec, Uniform, Fitter, Posterior, HierarchicalFitter, sample_raytrace"

# Generate paper figures
python analysis/fig04_sfh_recovery.py --n-mocks 3 --method raytrace
python analysis/fig07_speed_benchmarks.py --n-repeats 2
python analysis/fig08_gradient_sensitivity.py

# Compile paper
cd ~/writing-workspace/projects/differentiable_psd_sed_fitting
latexmk -pdf 0-ms.tex
```
