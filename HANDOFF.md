# diffsed Development Handoff

**Last updated:** 2026-03-14
**Repo:** `~/Projects/diffsed/`
**Status:** 296 tests (181 original + 108 new API + 7 benchmarks). All 6 notebooks execute. MAP + NUTS + geoVI all working. Photometry precomputation gives 21.6x gradient speedup.

## New High-Level API

```python
from diffsed import Model, ParamSpec, Uniform, Gaussian, Fixed, Fitter
from diffsed import load_ssp_data, load_filter_set

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

# Fit: MAP → Ray Tracing → geoVI
fitter = Fitter(model, mock.flux_obs, mock.noise)
result_map = fitter.run("map", n_steps=2000, learning_rate=0.03)
result_rts = fitter.run("raytrace", init_from=result_map, n_steps=500)
posterior = fitter.run("geovi", init_from=result_map, n_iterations=15, n_posterior_samples=80)

# Results
posterior.summary()                     # median ± 68% CI
posterior.plot_corner(truths=true_params)  # triangle plot with derived quantities
model.plot_sfh_posterior(posterior, true_params=true_params)  # SFH with 16-84% fill
```

## New Modules (this session)

| Module | Purpose | Tests |
|--------|---------|-------|
| `distributions.py` | Uniform, Gaussian, LogUniform, Fixed (JAX-jittable) | 39 |
| `param_spec.py` | ParamSpec: parameter defs, validation, sampling | 28 |
| `model.py` | Model: forward model, mock generation, plotting | 24 |
| `fitter.py` | Fitter: MAP (early stopping, optimizer choice), NUTS (target_accept), geoVI (post-optimization sampling), Ray Tracing (Snell's law MCMC) | 8 |
| `raytrace_jax.py` | Ray Tracing Sampler (Behroozi 2025) - Snell's law MCMC | TBD |
| `posterior.py` | Posterior: summary, resample, plot_corner, to_arviz, to_param_spec | 9 |

## Inference Performance

| Method | Smooth (7D) | Stochastic (137D) |
|--------|-------------|-------------------|
| MAP (Adam) | 3.8s | 4.4s |
| NUTS (BlackJAX) | ~23min (500 samples, 26 div) | ~2min (50 samples, 0 div) |
| Ray Tracing (Behroozi 2025) | TBD | TBD |
| geoVI (NIFTy.re) | ~95s (80+ samples) | ~170s (80+ samples) |

## Key Design Decisions (this session)

1. **ParamSpec as single source of truth** — same object for mock generation and inference
2. **Model wraps ForwardModel** — new param names on top, old internals unchanged
3. **Fitter separate from Model** — follows JAX pattern (model = physics, fitter = inference)
4. **geoVI post-optimization sampling** — `draw_linear_residual` draws 80+ additional samples after `optimize_kl` converges
5. **DSPS calc_obs_mag** for cosmologically correct magnitudes
6. **Sigmoid transform fixed** — k=1.0, x0=0 so sampler can reach prior edges

## Done (session 2, 2026-03-14)

### Code
- [x] **Ray Tracing Sampler**: Integrated Behroozi (2025) ray tracing MCMC (arXiv:2510.25824). New inference method in Fitter: `fitter.run("raytrace")`. Propagates rays through parameter space using Snell's law.
- [x] **Photometry precomputation**: Zacharegkas+2025 Eq 6-7 integrated into Model.__init__. 21.6x gradient speedup. Auto-activates when redshift fixed + filters present.
- [x] **Metallicity units fix**: Added log10(Zsun) = -1.848 offset in PARAM_MAP. met_logzsol (solar-relative) now correctly maps to SSP grid's log10(Z) (absolute).
- [x] **KDE corner plots**: Replaced scatter with 68%/95% KDE contours in posterior.plot_corner().
- [x] **charlot_fall_at_wavelengths()**: Fast dust evaluation at filter effective wavelengths.
- [x] **Benchmark tests**: 6 tests in tests/integration/test_precompute_speedup.py.

### Notebooks
- [x] **NB01**: Replaced excursion plot with GP realizations; added log-time SFH plot.
- [x] **NB00**: Updated geoVI to n_iterations=15, n_posterior_samples=80. Re-executed.
- [x] **NB03**: Rewritten with new Model/ParamSpec API + mock_batch benchmarks.

### Session 2 continued
- [x] **Convergence fix**: Reduced free params 7→5 (fix met_logzsol, dust_tau_diff) for well-determined photometric fits
- [x] **Star-forming galaxy**: Active SFH params showing recent SF (tau_peak=3 Gyr, peak_sfr=10)
- [x] **NB05**: Converged NUTS (1/500 div), PPC plots, suppressed geoVI verbosity, KDE corner contours
- [x] **NB02**: Real SVO filter transmission curves in SED+photometry plot
- [x] **NB01**: Removed archetype labels, parameter-only labels
- [x] **Corner plot size**: Capped at 14x14 for readable labels
- [x] **Paper analysis plan**: 14 figures, 9 analysis scripts at docs/paper_analysis_plan.md

## What Needs Doing Next

### Notebook Updates (highest priority)
- [ ] **NB00**: Add spectral fitting comparison — show how posteriors differ between photometry-only vs spectroscopy
- [ ] **NB04**: Add spectral fitting section (currently photometry-only MAP)
- [ ] **NB01**: Use star-forming galaxy params for SFH demonstrations (current mean SFH may fall at low lookback)

### Code Improvements
- [ ] **Internal param rename**: `sigma_ps` → `psd_sigma` etc. (~20 files, 181 tests)
- [ ] **Hierarchical inference**: Population-level PSD parameter recovery (Paper I Test 5-7)

### Paper Analysis (see docs/paper_analysis_plan.md)
- [ ] **Phase 1**: Individual SFH recovery — 100 mocks × 4 PSD regimes × geoVI at z=0.1
- [ ] **Phase 2**: Population-level PSD recovery — hierarchical inference
- [ ] **Phase 3**: Computational benchmarks — speed comparison table
- [ ] **Figure 5**: SFH recovery from photometry (most important paper figure)
- [ ] **Speed benchmarks**: MAP vs Ray Tracing vs NUTS vs geoVI vs Prospector/dynesty

## SSP Data

Real FSPS/MILES SSP templates (15 metallicities × 93 ages × 5994 wavelengths):
- `data/fsps_prsc_miles_chabrier.h5` — without nebular emission (64 MB)
- `data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5` — with nebular emission (64 MB, DEFAULT)
- Source: https://halos.as.arizona.edu/suchethacooray/

## Documentation

- `docs/specs/2026-03-13-paramspec-model-redesign.md` — full design spec
- `docs/design_philosophy.md` — paper-ready architecture description
- `README.md`, `AGENTS.md`, `CLAUDE.md` — existing docs

## Dependencies
```
jax>=0.4.20, dsps>=0.3, h5py>=3.0, matplotlib>=3.7  (core)
nifty8[re]>=8.5  (geoVI — installed)
blackjax>=1.3    (NUTS — installed)
optax>=0.2       (MAP — installed)
```

## How to Resume
```bash
cd ~/Projects/diffsed
source .venv/bin/activate
pytest tests/ -q   # should show 296 passed
python -c "from diffsed import Model, ParamSpec, Uniform, Fitter, Posterior"
```
