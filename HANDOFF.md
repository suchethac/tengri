# diffsed Development Handoff

**Last updated:** 2026-03-14
**Repo:** `~/Projects/diffsed/`
**Status:** 289 tests. New high-level API (Model, ParamSpec, Fitter, Posterior). DSPS calc_obs_mag for magnitudes. NB00 rewritten with new API. NB01-05 still use old API.

## New High-Level API

```python
from diffsed import Model, ParamSpec, Uniform, Gaussian, Fixed, Fitter
from diffsed import load_ssp_data, load_filter_set

spec = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0), sfh_beta=Uniform(0.3, 2.0),
    sfh_tau_peak_gyr=Uniform(0.5, 10.0), sfh_peak_sfr=Uniform(0.1, 50.0),
    met_logzsol=Gaussian(-0.3, 0.2, lo=-2.0, hi=0.2),
    dust_tau_bc=Uniform(0.0, 3.0), dust_tau_diff=0.3, dust_slope=-0.7,
    redshift=0.1, stochastic=False,
)
model = Model(spec, load_ssp_data("data/ssp.h5"), filters=load_filter_set([...]))
mock = model.mock(spec.sample(key), snr=20.0, key=noise_key)
result = model.fit(mock.flux_obs, mock.noise, method="map")
```

**New modules:** distributions.py, param_spec.py, model.py, fitter.py, posterior.py
**Design spec:** docs/specs/2026-03-13-paramspec-model-redesign.md
**Design doc:** docs/design_philosophy.md

## SSP Data

Real FSPS/MILES SSP templates (15 metallicities × 93 ages × 5994 wavelengths):
- `data/fsps_prsc_miles_chabrier.h5` — without nebular emission (64 MB)
- `data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5` — with nebular emission (64 MB)
- Source: https://halos.as.arizona.edu/suchethacooray/

## Recent Bug Fixes

1. **GP amplitude ~150x too small** — `xi_to_complex()` had wrong variance normalization (E[|ξ̂_k|²] ≈ 1-2 instead of N). Fixed by using `jnp.fft.rfft(xi)` directly in `gp_from_xi()`. Now empirical σ_x matches expected within 10-40% (finite-grid truncation).
2. **Hartley GP same bug** — Fixed `gp_from_xi_hartley()` to use `inverse_hartley(amplitude * hartley(xi))` instead of `inverse_hartley(amplitude * xi)`.
3. **ForwardModel missing Jacobian correction** — `compute_sqrt_power()` was evaluating PSD at FFT frequencies without the log-age→physical frequency Jacobian. Fixed to use `compute_sqrt_power_drw()`.
4. **Tutorial 1 no figures** — Missing `%matplotlib inline`. Also fixed `\o` escape warnings, added 11 pedagogical figures.

## What Exists

### Package (src/diffsed/) — 20 modules, ~3500 lines

| Module | Purpose | Tests |
|--------|---------|-------|
| `models/sfh/psd_models.py` | DRW, Matern, Extended Regulator PSDs | 29 |
| `models/sfh/gp_sfh.py` | IFT correlated field IFFT(sqrt(P)*xi), Jacobian correction | 15 |
| `models/sfh/mean_sfh.py` | Double power law (default), delayed-tau, constant | 10 |
| `models/dust/charlot_fall.py` | Charlot & Fall 2000, smooth sigmoid + hard step | 12 |
| `models/sps/dsps_wrapper.py` | DSPS CSP integral, metallicity interpolation, SSP loading | — |
| `models/sps/precompute.py` | Pre-computed SSP photometry/spectroscopy at fixed z (30-50x speedup) | 10 |
| `models/observation/photometry.py` | Filter convolution | — |
| `models/observation/spectroscopy.py` | Pixel-level spectra + Chebyshev calibration | — |
| `forward_model.py` | Full pipeline: params -> SED/photometry | — |
| `inference/common.py` | Loss function, PriorConfig, parameter transforms | 8 |
| `inference/map_optimizer.py` | MAP via optax Adam | — |
| `inference/nuts.py` | NUTS via BlackJAX with window adaptation | — |
| `inference/geovi.py` | geoVI/MGVI via NIFTy.re optimize_kl | — |
| `diagnostics/fisher.py` | Fisher Information Matrix via autodiff Jacobian | — |
| `diagnostics/saliency.py` | Gradient SEDs: dSED/dtheta per wavelength | — |
| `utils/transforms.py` | Bounded/unbounded sigmoid transforms | 5 |
| `utils/grid.py` | Log-age grid construction | — |
| `utils/cosmology.py` | Flat LCDM luminosity distance, age_at_z | — |
| `utils/devices.py` | JAX platform detection, GPU memory management | — |
| `utils/optimizations.py` | Hartley transform, gradient checkpointing, batched vmap | 9 |

### Docs
- `README.md` — comprehensive with quick start, architecture, roadmap
- `AGENTS.md` — AI agent guide (architecture, parameters, how to extend)
- `CLAUDE.md` — Claude Code build/test instructions
- `notebooks/01_understanding_the_model.ipynb` — executed, 9 cells, 0 errors

### Git log (9 commits on main)
```
0f88029 feat: add Fisher matrix and gradient SED diagnostics
06287ff feat: add inference backends — MAP, NUTS, geoVI
c1b1b2f docs: add executed outputs to Tutorial 1 notebook
605f639 docs: add hardware appendix, AGENTS.md, CLAUDE.md
6baf7c0 feat: add SSP pre-computation for fast inference
bf7aac8 feat: add Tutorial 1 notebook and device management
4d40ba9 feat: add optimizations, gradient tests, Hartley transform
80e511b docs: add comprehensive README
76ae0c5 feat: initial package structure with 72 passing tests
```

## Key Design Decisions

1. **Own implementations** for dust/SFH (not diffsky dependency) — diffsky's models are population-level; ours are individual-galaxy
2. **NIFTy.re for inference only** — use our own GP generation (physically meaningful sigma_PS, tau_PS), NIFTy.re's optimize_kl for geoVI
3. **DSPS as core dependency** — differentiable SPS, SSP template loading
4. **Pre-computation at fixed z** — SSP broadband fluxes computed once, eliminating wavelength integrals from MCMC (Zacharegkas+2025)
5. **Jacobian correction** for DRW PSD on log-age grid: P_u(q) = P_t(q/(t_ref*ln10)) / (t_ref*ln10)
6. **Lognormal correction**: SFR = mean * exp(x - K(0)/2) preserves linear-SFR interpretation

## What Needs Doing

### High Priority (for paper)
- [ ] Tutorial 2: Dust, SPS, full forward model
- [ ] Tutorial 3: Mock galaxy generation (z=0.1, z=2, z=6)
- [ ] Tutorial 4: Fitting — photometry-only, spectroscopy-only, joint
- [ ] Tutorial 5: Inference comparison — Adam vs NUTS vs geoVI
- [ ] Integration tests with real DSPS SSP templates
- [ ] Paper figure pipeline (recovery tests, 2D PSD grid, speed benchmarks)

### Paper Demonstrations (from user's list)
- [ ] **HMC/NUTS posterior**: speed vs Prospector, posterior quality, degeneracy handling
- [ ] **Fisher forecasting**: FIM as function of filter set, S/N, redshift
- [ ] **Optimal filter design**: differentiate FIM w.r.t. filter choices
- [ ] **Gradient MAP + Laplace**: fast MAP + Hessian posteriors for large catalogs
- [ ] **Gradient SEDs / saliency**: dflux/dtheta per wavelength, which features matter

### Lower Priority (post-paper)
- [ ] Nebular emission, dust emission, AGN components
- [ ] YAML config system
- [ ] Sphinx documentation website
- [ ] Population-level hierarchical model
- [ ] Push to GitHub as public repo

## Dependencies
```
jax>=0.4.20, dsps>=0.3, h5py>=3.0, matplotlib>=3.7  (core)
nifty8[re]>=8.5  (optional: geoVI)
blackjax>=1.0    (optional: NUTS)
optax>=0.2       (optional: MAP)
```

## How to Resume
```bash
cd ~/Projects/diffsed
source .venv/bin/activate
pytest tests/ -q   # should show 108 passed
```
Then continue with Tutorial 2 or integration tests.
