# diffsed Development Handoff

**Last updated:** 2026-03-13
**Repo:** `~/Projects/diffsed/`
**Status:** Core package complete. 108 tests passing. Inference backends implemented. Tutorial 1 done. Needs tutorials 2-5, integration tests with real SSP data, and paper figure pipeline.

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
