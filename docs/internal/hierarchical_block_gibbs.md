# Hierarchical Block Gibbs: Design Document

## Parameter Structure

The hierarchical flat array has this layout:

```
[psd_sigma_u, psd_tau_u, gal0_phys..., gal0_xi..., gal1_phys..., gal1_xi..., ...]
 ←─ shared ─→  ←───────── per-galaxy (N_gal repeats) ─────────────────────→
```

Where:
- `psd_sigma_u`, `psd_tau_u`: 2 shared PSD params (unbounded)
- `gal_phys`: ~5-7 physical params per galaxy (dust, metallicity, mass, etc.)
- `gal_xi`: ~64-130 GP field latent variables per galaxy

## Block Schedule

Three blocks per outer iteration, ordered by importance:

### Block 1: Shared PSD Parameters (highest priority)
- **Updated**: `psd_sigma_u`, `psd_tau_u` (2 params)
- **Frozen** (constants): all `gal_phys` + all `gal_xi`
- **Point estimates**: per-galaxy params (for speed — their uncertainty is less relevant for the 2D PSD posterior)
- **Sample mode**: `nonlinear_resample` (geoVI — the PSD response is nonlinear)
- **n_samples**: 6 (more samples for precise population constraint)
- **Why first**: PSD params set the prior for every galaxy's SFH. Getting them right first propagates correct information downstream.

### Block 2: Per-Galaxy Physical Parameters (medium priority)
- **Updated**: all `gal_phys` (N_gal × ~7 params)
- **Frozen** (constants): `psd_sigma_u`, `psd_tau_u`, all `gal_xi`
- **Sample mode**: `nonlinear_resample` (geoVI — age-dust degeneracy)
- **n_samples**: 3
- **Why second**: Physical params interpret each galaxy's data given the current SFH prior. Conditioned on PSD + xi, the per-galaxy problems are independent.

### Block 3: Per-Galaxy SFH Fields (lowest priority)
- **Updated**: all `gal_xi` (N_gal × n_grid params)
- **Frozen** (constants): `psd_sigma_u`, `psd_tau_u`, all `gal_phys`
- **Sample mode**: `linear_resample` (MGVI — GP response is nearly linear)
- **n_samples**: 2 (cheap, high-D)
- **Why last**: SFH fields are the "leaf" of the hierarchy. Nearly Gaussian conditioned on PSD + physical params.

## Implementation Plan

### What exists
- `evi_step_full` in `fitter.py` already supports `constants_mask` and `pe_mask`
- `BlockSchedule.hierarchical()` in `vi_config.py` defines the blocks
- `HierarchicalFitter._run_evi_jit` has a flat-array JIT engine
- `fitter.fit_batch(galaxies)` for batch fitting (default method: `native_geovi`)
- `OptimizationSchedule` class in `vi_config.py` for custom schedules
- `_simple_cg` is a module-level function for catalog engine

### What's needed
1. **Build block masks** from the hierarchical parameter layout:
   - `shared_mask`: True for psd_sigma_u, psd_tau_u
   - `gal_phys_mask`: True for all per-galaxy physical params
   - `gal_xi_mask`: True for all per-galaxy xi

2. **Cycle through blocks** in the while_loop:
   - Use `iteration % 3` to select which block
   - Each block has its own `constants_mask`, `pe_mask`, and `sample_mode`
   - `jax.lax.switch` or `jax.lax.cond` for block dispatch

3. **Variable n_samples per block**: Block 1 uses 6, Block 2 uses 3, Block 3 uses 2.
   Since `n_samples` is a static arg, need separate compiled versions or use the max and mask unused samples.

## Resample+Update Schedule Within Blocks

Each block follows the same resample+update pattern:
- Outer cycle 0: `nonlinear_resample` for blocks 1+2, `linear_resample` for block 3
- Outer cycles 1-4: `nonlinear_update` for blocks 1+2, `linear_sample` for block 3
- Outer cycle 5: resample again (prevent staleness)

## Expected Performance

For N=100 galaxies, D_total = 2 + 100×(7+130) = 13,702:
- Block 1 (2D shared PSD): ~0.01ms per iteration
- Block 2 (700D physical): ~0.5ms per iteration
- Block 3 (13,000D SFH): ~2ms per iteration (MGVI, no curving)
- 25 outer iterations × 3 blocks = 75 total steps: ~190ms

Compile time: ~60s (one-time, cached to XLA disk cache).
