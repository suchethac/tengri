# diffsed Development Handoff

**Last updated:** 2026-03-14
**Repo:** `~/Projects/diffsed/`
**Status:** 289 tests (181 original + 108 new API). 28 commits this session. All 6 notebooks execute. MAP + NUTS + geoVI all working.

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

# Fit: MAP → geoVI chaining
fitter = Fitter(model, mock.flux_obs, mock.noise)
result_map = fitter.run("map", n_steps=2000, learning_rate=0.03)
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
| `fitter.py` | Fitter: MAP (early stopping, optimizer choice), NUTS (target_accept), geoVI (post-optimization sampling) | 8 |
| `posterior.py` | Posterior: summary, resample, plot_corner, to_arviz, to_param_spec | 9 |

## Inference Performance

| Method | Smooth (7D) | Stochastic (137D) |
|--------|-------------|-------------------|
| MAP (Adam) | 3.8s | 4.4s |
| NUTS (BlackJAX) | ~23min (500 samples, 26 div) | ~2min (50 samples, 0 div) |
| geoVI (NIFTy.re) | ~95s (80+ samples) | ~170s (80+ samples) |

## Key Design Decisions (this session)

1. **ParamSpec as single source of truth** — same object for mock generation and inference
2. **Model wraps ForwardModel** — new param names on top, old internals unchanged
3. **Fitter separate from Model** — follows JAX pattern (model = physics, fitter = inference)
4. **geoVI post-optimization sampling** — `draw_linear_residual` draws 80+ additional samples after `optimize_kl` converges
5. **DSPS calc_obs_mag** for cosmologically correct magnitudes
6. **Sigmoid transform fixed** — k=1.0, x0=0 so sampler can reach prior edges

## What Needs Doing Next

### Notebook Quality (highest priority)
- [ ] **NB00**: Re-execute with `n_iterations=15, n_posterior_samples=80` for proper posteriors
- [ ] **NB01**: Add log-time SFH plot below `01_sfh_zoom_recent.png`; replace excursion plot with GP realizations (linear + offset per PSD model)
- [ ] **NB03**: Rewrite with new Model/ParamSpec API; add wall time benchmarks for `mock_batch()`
- [ ] **NB05**: Re-execute with updated geoVI (80+ post-optimization samples); add contour corner plots

### Code Improvements
- [ ] **Photometry precomputation** in Model: integrate `precompute_photometry()` into `Model.__init__` when redshift fixed + filters present → 30-50x speedup for fitting
- [ ] **Internal param rename**: update low-level function signatures from `sigma_ps` → `psd_sigma` etc. (~20 files, 181 tests — do as focused task)
- [ ] **Corner plot contours**: use KDE or arviz for 2D contours instead of scatter
- [ ] **More optimizers**: the `optimizer` param in MAP already supports "adam"/"adamw"/"sgd"/custom optax

### Paper Demonstrations
- [ ] HMC/NUTS posterior: speed vs Prospector, posterior quality
- [ ] Fisher forecasting: FIM as function of filter set, S/N, redshift
- [ ] Gradient SEDs / saliency: dflux/dtheta per wavelength

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
pytest tests/ -q   # should show 289 passed
python -c "from diffsed import Model, ParamSpec, Uniform, Fitter, Posterior"
```
