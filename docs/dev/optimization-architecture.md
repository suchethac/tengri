# Forward Model Optimization Architecture

## Context

The forward model computes galaxy SEDs from physical parameters. For inference
(MAP, VI, MCMC), the forward model and its gradient are called thousands to
millions of times. Wall-clock time is dominated by the forward pass, so
optimization matters.

The key insight: **only the stellar CSP einsum is expensive** (O(n_age x n_wave)
= ~658,000 ops). Every other component — nebular, dust IR, AGN, radio, X-ray —
is O(n_wave) = ~7,000 ops. This 94x cost ratio drives the entire optimization
strategy.

## The Four Prediction Modes

```
model.predict_photometry(params, mode="...")

  "exact"          Raw pipeline, no JIT.           Reference path.
  "compositional"  Full-resolution JIT kernel.     Bit-identical to exact.
  "precomputed"    SSP x filter at eff. wavelengths. Fast, ~0.4% error (stellar).
  "hybrid"         Precomputed SSP + exact non-stellar. Fast AND exact non-stellar.
  "auto"           Picks fastest available.
```

### Performance (SDSS ugriz, Apple M-series CPU, post-JIT)

**Stellar only (DPL SFH + two-component dust):**

| Mode           | Latency | Speedup | Error vs exact |
|----------------|---------|---------|----------------|
| hybrid         |  277 us |    34x  | 0.16%          |
| precomputed    |  274 us |    34x  | 0.16%          |
| compositional  |  940 us |    10x  | 0.000000       |
| exact          | 9544 us |     1x  | reference      |

**Stellar + AGN (frac mode):**

| Mode           | Latency | Speedup | Error vs exact | Status  |
|----------------|---------|---------|----------------|---------|
| hybrid         |  515 us |   126x  | 1.04%          | OK      |
| precomputed    |      —  |      —  | —              | BLOCKED |
| compositional  | 1010 us |    64x  | 0.000000       | OK      |
| exact          |64705 us |     1x  | reference      | OK      |

**Stellar + AGN (parametric):**

| Mode           | Latency | Speedup | Error vs exact | Status  |
|----------------|---------|---------|----------------|---------|
| hybrid         |  553 us |   117x  | 1.04%          | OK      |
| precomputed    |      —  |      —  | —              | BLOCKED |
| compositional  | 1006 us |    64x  | 0.000000       | OK      |
| exact          |64846 us |     1x  | reference      | OK      |

Note: precomputed is BLOCKED for AGN because `from_config(agn=...)` injects
`agn_frac=Uniform(0,1)` which forces frac mode. The hybrid path handles all
AGN configurations correctly.

**Panchromatic (21 bands UV-to-submm: GALEX + SDSS + 2MASS + WISE + Herschel + SCUBA2):**

Full physics: DPL SFH + two-component dust + MBB dust emission + parametric AGN.

| Mode           | Latency | Speedup | Max error | Mean error | Status  |
|----------------|---------|---------|-----------|------------|---------|
| hybrid         | 1073 us |    61x  | 2.9%      | 0.8%       | OK      |
| precomputed    |  668 us |    98x  | >1e14%    | >1e13%     | BROKEN  |
| compositional  | 1114 us |    59x  | 0.000000  | 0.000000   | OK      |
| exact          |65339 us |     1x  | reference | reference  | OK      |

Note: precomputed mode is catastrophically broken for panchromatic filter sets
because MBB dust emission evaluated at effective wavelengths produces wildly
incorrect normalization in the far-IR. The hybrid mode handles this correctly
by computing dust IR emission at full wavelength resolution via emission_helpers.

**L_absorbed broadband estimate:** Uses Voronoi frequency bandwidth weighting
to convert the per-band sum into a proper ∫L_ν dν quadrature. Without this
weighting, the estimate is off by orders of magnitude for UV-to-submm filter
sets (the naive sum has wrong units: erg/s/Hz × n_filters, not erg/s).

## Architecture: Four Data/Kernel Layers

```python
@dataclass
class PrecomputedData:
    """Tensors pre-integrated through filters. No kernels."""
    photometry           # SSP x filter (n_met, n_age, n_filt)
    photometry_ztable    # SSP x filter on z-grid (free redshift)
    spectroscopy         # SSP rebinned to wave_obs pixels
    dust_age_weights     # sigmoid weights for two-component dust
    igm_at_effective_wavelengths  # IGM T(lambda_eff) for fixed z

@dataclass
class PrecomputedKernels:
    """Mode 1: JIT kernels evaluating everything at filter effective wavelengths."""
    photometry           # build_fused_photometry
    photometry_ztable    # build_fused_photometry_ztable
    spectrum             # build_fused_spectrum

@dataclass
class CompositionalKernels:
    """Mode 2: Full-resolution JIT kernels."""
    rest_sed             # build_fused_rest_sed (core engine)
    photometry           # params -> photometry (end-to-end JIT)
    spectrum             # params -> spectrum (end-to-end JIT)
    exact_sed            # build_exact_sed (JIT wrapper for exact path)

@dataclass
class HybridKernels:
    """Mode 3: Precomputed SSP stellar + exact non-stellar."""
    photometry           # build_hybrid_photometry
    photometry_ztable    # (planned)
    spectrum             # (planned)
```

Dispatch priority in `mode="auto"`:
**Hybrid -> Precomputed -> Compositional -> Exact**

## Optimization 1: JIT Compilation (Compositional Mode)

All forward model operations are pure JAX functions with no Python side
effects. The `@jax.jit` decorator compiles the entire computation graph into
a single fused XLA kernel. This eliminates:

- Python interpreter overhead between operations
- Intermediate array materializations
- Memory allocation/deallocation per step

The compositional kernel (`build_fused_rest_sed`) captures all physics
functions (dust laws, nebular backend, AGN model, etc.) in a Python closure.
At JIT trace time, Python `if` branches on captured booleans are resolved
statically — XLA only sees the active code path.

**Cost:** First call compiles (~56 seconds for full model). Subsequent calls
dispatch the cached XLA executable in ~microseconds. The persistent XLA cache
at `~/.cache/tengri_jax_cache` survives across Python sessions, so
recompilation only happens when the model configuration changes.

**Result:** 9x speedup over exact path, bit-identical physics.

## Optimization 2: Photometric Precomputation (Precomputed Mode)

Reference: Zacharegkas+2025 (arXiv:2506.19919), Section 3.

The broadband photometry integral is:
```
c_gal(band) = Integral[ T(lambda) * L_gal(lambda) * lambda ] d_lambda
```

where `L_gal = Sum_age[ SFR(age) * SSP(age, Z, lambda) * dust(lambda, age) ]`.

The key observation: `SSP(age, Z, lambda)` and `T(lambda)` don't depend on
the sampled parameters (SFH, dust). Only `SFR(age)` and `dust(lambda)` change
per sample. If we pull the dust curve outside the integral (approximating it
as constant across each filter bandpass), the wavelength integration becomes
precomputable:

```
c_SSP(age, Z, band) = Integral[ T(lambda) * SSP(age, Z, lambda) * lambda ] d_lambda
```

This is computed once at model init. At runtime, photometry reduces to:
```
c_gal(band) = Sum_age[ SFR(age) * c_SSP(age, Z, band) ] * dust(lambda_eff)
```

A ~94-element dot product per filter instead of ~658,000 multiply-adds.

**Taylor correction** (optional): The first-order moment tensor captures the
SSP-dust covariance to first order, reducing the factorization error ~5x:
```
c_gal ~ A(lambda_eff) * Phi + A'(lambda_eff) * Psi
```

where Phi is the SSP photometry and Psi is the SSP first moment.

**Accuracy:** <0.4% for most dust laws, ~7% for SMC in u-band (steep UV rise).

**Speedup:** 30-50x over exact path.

**Limitations:** Non-stellar components (nebular lines, AGN continuum, dust IR
emission) are either approximated at effective wavelengths or BLOCKED entirely:

| Component    | Handling in precomputed mode          | Issue                    |
|-------------|--------------------------------------|--------------------------|
| Dust atten.  | Evaluated at ~5 effective wavelengths | <3% err (36% for SMC)    |
| Dust IR      | MBB reimplemented at eff. wavelengths | Approximate L_absorbed   |
| AGN (param.) | Evaluated at eff. wavelengths         | 10-20% error             |
| AGN (frac)   | BLOCKED                              | Needs full SED integral  |
| Nebular      | BLOCKED                              | Lines are sharp features |
| Shock        | BLOCKED                              | Not reimplemented        |
| Radio        | BLOCKED                              | Not reimplemented        |
| X-ray        | BLOCKED                              | Not reimplemented        |

The `is_fused_compatible()` gate checks which components are active and
falls back to the exact path if any blocked component is enabled.

## Optimization 3: Hybrid Mode (NEW)

The hybrid kernel combines precomputed stellar with exact non-stellar:

```
                     params
                       |
            +----------+----------+
            |                     |
       STELLAR (fast)       NON-STELLAR (exact)
            |                     |
       precomputed SSP       emission_helpers.py
       x filter einsum       at full wavelength
       + dust at eff.        + filter integration
            |                     |
       stellar_phot(b)      non_stellar_phot(b)
            |                     |
            +----------+----------+
                       |
                 total_phot(b)
```

**Why this works:** Non-stellar functions don't need the full stellar SED.
They need `weights` (CSP from SFH), `ssp_wave`, `sfr`, and `L_absorbed`.
All available without computing the expensive stellar einsum at full
wavelength resolution.

**L_absorbed estimation:** Broadband approximation from precomputed fluxes:
```
L_absorbed_stellar ~ Sum_bands(flux_intrinsic - flux_attenuated) * LSUN
```
Nebular L_absorbed is exact (from `attenuate_emission()` at full resolution).

**Filter integration for non-stellar:** The non-stellar SED (computed at
full wavelength resolution) is integrated through each filter via
`compute_flux_density()`. The Python for-loop over filters is unrolled by
JAX at trace time into a single XLA graph.

**IGM handling:**
- Stellar: precomputed at effective wavelengths (fast)
- Non-stellar: applied at full wavelength resolution before filter integration

**Result:** Same ~36x speedup as precomputed mode, but:
- Nebular emission: EXACT (previously blocked)
- AGN (all modes): EXACT (previously 10-20% error or blocked)
- Dust IR: EXACT template shapes (previously MBB approximation)
- Radio, X-ray, shock: EXACT (previously blocked)
- No `is_fused_compatible()` gate needed

## Optimization 4: Spectroscopic Precomputation

For spectroscopic fitting, SSP templates are pre-interpolated to the observed
wavelength grid at fixed redshift:

```python
model.precompute_spectroscopy(wave_obs)
```

This replaces per-call wavelength interpolation with a precomputed lookup,
giving ~20x speedup for spectroscopic inference.

## Optimization 5: Free-Redshift Z-Table

For free-redshift fitting, SSP photometry is precomputed on a redshift grid:

```python
model.precompute_ztable(z_min=0.001, z_max=3.0, n_z=100)
```

At inference time, the table is interpolated to the current z. 100 points
gives <0.01% interpolation error.

## Optimization 6: Mixed Precision

```python
model = Model(spec, ssp, forward_dtype="float32")
```

Halves memory, ~1.5x speedup, <0.01% error vs float64. Single precision is
sufficient for photometric inference.

## Optimization 7: Persistent XLA Cache

```python
# Auto-enabled on import
import jax
jax.config.update("jax_compilation_cache_dir", "~/.cache/tengri_jax_cache")
```

Compiled XLA executables persist across Python sessions. Eliminates the ~56s
first-call compilation cost on subsequent runs with the same model config.

## What the Paper Uses

The paper (Paper II: Stochastic SFH + Hierarchical PSD) uses the following
forward model configuration for all figures:

| Component        | Model                              | Paper reference          |
|------------------|------------------------------------|--------------------------|
| SSP templates    | FSPS/MIST/C3K, Chabrier IMF       | Conroy+2009, Choi+2016   |
| Mean SFH         | Double power law                   | Carnall+2018             |
| Stochastic SFH   | GP field with DRW PSD              | This work                |
| Dust attenuation | Two-component Charlot & Fall       | Charlot & Fall 2000      |
| Dust curve       | Power law (n free)                 | Charlot & Fall 2000      |
| Nebular          | Baked into SSP (fixed logU)        | —                        |
| AGN              | None                               | —                        |
| Dust emission    | None                               | —                        |
| Radio/X-ray      | None                               | —                        |
| IGM              | None (low-z mocks)                 | —                        |

**Dimensionality:** D ~ 136 (128 GP latent + 4 DPL mean SFH + 1 metallicity
+ 3 dust parameters).

**Inference methods used:**
- MAP (Adam optimizer) for initialization
- Ray Tracing (Behroozi 2025) for individual galaxy posteriors
- geoVI (NIFTy, Frank+2021) for hierarchical population inference
- NUTS (BlackJAX) for gold-standard validation (D <= 20 only)

## What the Paper Does NOT Use (but tengri implements)

The paper deliberately uses a minimal forward model (stellar + dust only) to
isolate the SFH recovery question. The following components are implemented
in the code but not exercised in Paper II figures:

| Category         | Implemented models                           | Status   |
|------------------|----------------------------------------------|----------|
| Nebular          | CLOUDY CB19, Cue, MAPPINGS Photo/Shock       | Complete |
| AGN              | K&D 3-zone disc, SKIRTOR torus, BLR/NLR      | Complete |
| Dust emission    | DL07, Dale+2014, Casey 2012, MBB             | Complete |
| Radio            | Bell 2003, Delvecchio+2021, AGN jets          | Complete |
| X-ray            | Grimm+2003 XRB, AGN corona                   | Complete |
| IGM              | Inoue+2014                                    | Complete |
| Dust curves      | Calzetti, K&C, SMC, LMC, Cardelli, TEA, etc. | Complete |
| Non-param SFH    | Continuity (Leja+2019), Dirichlet             | Complete |

## Roadmap Models (Not Yet Implemented)

These are documented as specs in `docs/dev/roadmap/` but have no code:

| Model             | Reference                | Purpose                         |
|-------------------|--------------------------|---------------------------------|
| PAH features      | PAHFIT decomposition     | Mid-IR spectral decomposition   |
| Chemical evolution| Z(t) time-dependent      | Mass-metallicity evolution      |
| BOSA templates    | Boquien & Salim 2021     | sSFR-parameterized dust         |
| MAGPHYS dust      | da Cunha+2013            | Multi-temperature starburst     |
| Astrodust+PAH     | Hensley & Draine 2023    | Next-gen dust grain model       |
| THEMIS dust       | Jones+2017               | European dust model             |
| Patchy IGM        | Stochastic transmission  | Sightline-to-sightline scatter  |
| Shock emission    | MAPPINGS V full grids    | Beyond current simple model     |
| TEA attenuation   | Haskell+2024             | NIHAO-SKIRT empirical curve     |

## Testing the Hybrid Kernel

**Current test configuration:** The hybrid kernel was tested with the same
minimal model the paper uses (DPL SFH + two-component dust, no non-stellar
components). In this configuration, the hybrid kernel produces identical
results to the precomputed kernel (~0.4% error vs exact), since the
non-stellar contribution is zero.

**What needs testing:** Models with non-stellar components enabled:
- Stellar + CLOUDY nebular + dust emission (the common real-data config)
- Stellar + AGN (parametric and frac modes)
- Full panchromatic: stellar + nebular + AGN + dust IR + radio + X-ray

The hybrid kernel should produce results within <1% of exact for stellar
(precomputed approximation) and within floating-point precision for all
non-stellar components.
