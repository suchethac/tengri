# tengri Notebook Suite — Comprehensive Plan

## Architecture

Two labeled tracks, numbered independently:

| Track | Purpose | Audience |
|-------|---------|----------|
| **T** (Tutorial) | Learn the code, understand the method | Astronomers who want to use tengri |
| **A** (Analysis) | Reproduce paper figures, validate claims | Referees, collaborators, the author |

Tutorials are the Sphinx documentation. Analysis notebooks are the paper's computational backbone.

---

## Tutorial Track (7 notebooks)

### T00 — Quickstart: Fit a Galaxy in 60 Seconds

**Goal:** Minimal end-to-end demo. User sees results before understanding internals.

**Sections:**
1. Install + load SSP data + filters (5 lines)
2. **Part A: Parametric fit** (smooth SFH, 7 params)
   - Define ParamSpec with Fixed(psd_sigma=0)
   - Generate mock SDSS photometry (SNR=20)
   - MAP → NUTS (gold standard for D=7)
   - SFH recovery panel + corner plot
3. **Part B: Stochastic fit** (bursty SFH, 137 params)
   - Define ParamSpec with free psd_sigma, psd_tau
   - Generate mock with σ_PS=2.0, τ_PS=20 Myr
   - MAP → Ray Tracing (the recommended path for high-D)
   - SFH recovery panel **with inset zooming into last 200 Myr**
   - Corner plot of physical parameters only (not ξ)
4. What's next: roadmap to other tutorials

**Key figures (5):**
- F1: Mock photometry (parametric)
- F2: SFH recovery + corner (parametric, NUTS)
- F3: True bursty SFH + mock photometry (stochastic)
- F4: SFH recovery with 200 Myr inset (stochastic, Ray Tracing)
- F5: Corner plot of physical params (stochastic)

**Changes from current NB00:**
- Remove geoVI from quickstart (defer to T03)
- Remove sampler cross-comparison (defer to T03)
- Remove derived-quantity violin plots (nice but premature)
- Add 200 Myr inset on every SFH panel
- Tighten to ~250 lines

---

### T01 — The IFT Correlated Field Model

**Goal:** Self-contained introduction to the mathematical framework. This is the theory notebook — substantial, because the PSD→GP→SFH construction is the core intellectual contribution.

**Sections:**
1. **What is Information Field Theory?**
   - IFT = Bayesian inference on continuous fields
   - Signal, data, response, noise — the measurement equation
   - The information Hamiltonian: H = ½χ² + ½ξᵀξ
   - Why standardization matters (isotropic prior geometry)
2. **The Power Spectral Density**
   - DRW PSD: P(ω) = σ²τ / (1 + (τω)²)
   - Physical meaning of σ_PS and τ_PS
   - Table: τ_PS ranges ↔ feedback mechanisms
   - **Figure: PSD + ACF + GP realizations (3-panel, from current NB01)**
3. **Generating GP realizations from a PSD**
   - The FFT recipe: ξ → rfft → multiply √P → irfft → x(t)
   - The log-age grid and why it provides natural resolution scaling
   - **Figure: Same ξ, different PSD → different GP (2×2)**
   - **Figure: Multiple realizations per regime (2×2)**
4. **The mean SFH: double power law**
   - Equation, parameter meanings (α, β, τ, A)
   - Galaxy archetypes
   - **Figure: Archetypes in lookback time with redshift twin axis**
   - **Figure: Vary α, β, τ (1×3)**
5. **The full SFH: mean × exp(x − σ²/2)**
   - Lognormal correction explained
   - **Figure: Step-by-step assembly (mean → GP → full SFH, 1×3)**
   - **Figure: 4 regimes, log scale with inset of last 200 Myr**
   - **Figure: 4 regimes, ensemble of 10 realizations**
6. **The burstiness plane**
   - σ_PS × τ_PS grid showing SFH diversity
   - **Figure: 3×3 grid (log scale)**
7. **The PSD as a physical prior**
   - σ_PS → main-sequence scatter mapping
   - τ_PS → feedback timescale mapping
   - Window functions: which observables probe which timescales
   - **Figure: Observable window functions (Hα, UV, D4000, Balmer break)**
   - This section absorbs the best of current NB08
8. **End-to-end JAX gradients**
   - Verify ∂M*/∂θ is finite and nonzero for all params
   - Why this enables gradient-based inference
   - **Figure: Parameter sensitivity (2×3, vary one param at a time)**
9. **Summary table + next steps**

**Key figures (~12):**
- PSD overview (3-panel)
- Same ξ different PSD (2×2)
- Multiple GP realizations (2×2)
- Galaxy archetypes
- Mean SFH parameter exploration (1×3)
- Step-by-step SFH assembly (1×3)
- 4 regimes log scale with 200 Myr inset
- Ensemble SFHs (2×2)
- Burstiness plane (3×3)
- Observable window functions
- Parameter sensitivity (2×3)

**Changes from current NB01:**
- Remove log/linear duplicates (keep log, add insets)
- Absorb PSD-physics material from NB08 (§7)
- Remove "zoom recent" and "zoom recent logtime" as separate figures — replace with insets
- Keep substantial explanatory text (this is documentation)

---

### T02 — The Forward Model: SFH → SED → Photometry

**Goal:** Step-by-step walkthrough of the differentiable SPS pipeline.

**Sections:**
1. **Simple stellar populations (SSPs)**
   - What an SSP is; spectral features table
   - **Figure: SSP spectra at 5 ages (current NB02 fig01)**
2. **The composite stellar population (CSP)**
   - SFH-weighted integral over SSP ages
   - Weights w_i = SFR(t_i)Δt_i
   - **Figure: SFH → weights → SSP contributions → total CSP (1×3)**
3. **Dust attenuation: Charlot & Fall (2000)**
   - Two-component model, sigmoid transition
   - **Figure: Attenuation curves + sigmoid + before/after dust (2×2)**
4. **Redshift and filter convolution**
   - Observer-frame flux from rest-frame SED
   - Filter transmission curves
   - **Figure: Filters overlaid on redshifted SED**
5. **The Jacobian: what constrains what**
   - ∂f_b/∂θ_k for all bands and parameters
   - **Figure: Signed Jacobian heatmap (bands × params)**
   - Key degeneracies: dust–age, metallicity–dust
6. **Photometry vs. spectroscopy**
   - Same galaxy: broadband fluxes vs. R=100 spectrum
   - **Figure: SED with photometric points overlaid + spectrum**
   - Preview of information gain (developed in T04)
7. **Summary + next steps**

**Key figures (~8):**
- SSP spectra
- CSP assembly (1×3)
- Dust (2×2)
- Filters on redshifted SED
- Jacobian heatmap
- Photometry vs spectrum comparison

**Changes from current NB02:**
- Tighten — remove redundant panels
- Add Jacobian heatmap (from current NB02's gradient section)
- Defer calibration polynomial to future implementation
- Add filter+SED overlay figure

---

### T03 — Inference: Five Methods, One Loss Function

**Goal:** Deep dive into the inference machinery. When to use each method.

**Sections:**
1. **The standardized loss function**
   - H = ½χ² + ½ξᵀξ — same for all methods
   - Bijective transforms: physical params ↔ ξ-space
2. **MAP optimization (Adam)**
   - Fast point estimate, sampler initialization
   - ~1 second for D=137
3. **Ray Tracing Sampler (Behroozi 2025)**
   - Snell's law optics analogy
   - ~250× gradient-noise tolerance vs HMC
   - Step size tuning for high-D
   - **Demo: fit stochastic mock, show trace plots + acceptance**
4. **NUTS (Hoffman & Gelman 2014)**
   - Gold standard for D≲20
   - Impractical for D~137 — demonstrate why
   - **Demo: fit parametric mock, show effective sample size**
5. **geoVI (Frank et al. 2021)**
   - Variational inference on a Riemannian manifold
   - Approximate but scales to D>10⁴
   - When it fails: multimodal posteriors
   - **Demo: fit stochastic mock, compare to Ray Tracing**
6. **MGVI (Knollmüller & Enßlin 2019)**
   - Linearized limit of geoVI
   - Cheaper per iteration, less accurate
   - The hierarchical workhorse
7. **Sampler comparison**
   - Same mock, all methods overlaid
   - **Figure: Corner plot comparison (RT vs geoVI vs NUTS on parametric)**
   - **Figure: SFH recovery comparison with 200 Myr inset**
   - **Figure: Wall-clock timing bar chart**
   - **Table: Method properties (exact/approximate, D range, wall time)**
8. **Practical recommendations**
   - Decision tree: which method for your problem
9. **Summary**

**Key figures (~8):**
- Ray Tracing trace plots + acceptance
- NUTS ESS demonstration
- geoVI convergence (KL iterations)
- Corner comparison (3 methods)
- SFH recovery comparison with inset
- Timing bar chart

**Changes from current NB03:**
- Much shorter (current is 1127 lines)
- Remove redundant setup code
- Focus on practical guidance, not re-derivation
- Keep sampler comparison as the climax

---

### T04 — Fitting Galaxies: Photometry and Spectroscopy

**Goal:** The practical fitting notebook. Mock generation → fit → diagnostics → interpretation. Combined photometry + spectroscopy with progressive information gain.

**Sections:**
1. **Mock generation API**
   - `model.mock()` for single galaxies
   - Controlling PSD regime, redshift, SNR
   - **Figure: Mock photometry + mock spectrum for same galaxy**
2. **Fitting broadband photometry**
   - SDSS ugriz at z=0.1
   - Full stochastic model fit
   - **Figure: SFH posterior with 200 Myr inset — photometry only**
   - **Figure: Posterior predictive check (model SED vs data)**
3. **Fitting a spectrum (R=100)**
   - Same galaxy, now with pixel-level spectral fitting
   - **Figure: SFH posterior with 200 Myr inset — spectroscopy**
   - **Figure: Posterior predictive check (model spectrum vs data)**
4. **Information gain: photometry → spectroscopy**
   - Side-by-side comparison
   - **Figure: Photometry vs spectroscopy posteriors overlaid, with 200 Myr inset**
   - **Figure: Corner plot comparison (phot vs spec)**
   - Key message: spectroscopy constrains recent SFH and σ_PS
5. **PSD parameter constraints**
   - Can you measure burstiness from one galaxy?
   - **Figure: Joint (σ_PS, τ_PS) posterior — phot vs spec**
   - Spectroscopy constrains σ_PS; τ_PS requires populations (→ T05)
6. **Diagnostics checklist**
   - Reduced χ², trace plots, effective sample size, posterior predictive
7. **Summary**

**Key figures (~10):**
- Mock photometry + spectrum
- SFH posterior (phot only) with inset
- SFH posterior (spec) with inset
- Overlaid phot vs spec with inset
- Posterior predictive checks ×2
- Corner comparison
- Joint PSD posterior

---

### T05 — Hierarchical Inference: Population-Level Burstiness

**Goal:** Share PSD hyperparameters across N galaxies. The defining science case.

**Sections:**
1. **Why hierarchical?**
   - Individual τ_PS is poorly constrained
   - Population averaging breaks the degeneracy
2. **The hierarchical model (Eq. 24 from paper)**
   - Shared ϕ = (σ_PS, τ_PS), per-galaxy (ξ_i, θ_i)
   - Total dimensionality: N × (128+9) + 2
3. **Demo: N=100 galaxies, moderate burstiness**
   - Generate population with true (σ_PS, τ_PS)
   - geoVI at D~14,000
   - **Figure: Recovered (σ_PS, τ_PS) posterior vs truth**
4. **√N posterior shrinkage**
   - Vary N from 10 to 500
   - **Figure: Posterior width vs N (log-log)**
5. **Distinguishing two populations**
   - Moderate vs bursty populations
   - **Figure: Well-separated posteriors**
6. **Practical considerations**
   - Memory, wall time, convergence diagnostics
   - When MGVI vs geoVI
7. **Summary**

**Key figures (~6):**
- Hierarchical posterior on (σ, τ)
- √N shrinkage
- Population distinction
- Individual vs hierarchical comparison
- Convergence diagnostics

---

### T06 — Extending tengri: Custom Models and New Physics

**Goal:** How to modify the code. For developers and advanced users.

**Sections:**
1. **Architecture overview**
   - Modular design: PSD ↔ mean SFH ↔ SPS ↔ dust ↔ observation
2. **Custom PSD models**
   - Implementing a broken power law or Matérn PSD
   - Registering with the framework
3. **Alternative dust laws**
   - Replacing Charlot & Fall with Calzetti or Kriek & Conroy
4. **New SSP templates**
   - Loading alternative libraries (BPASS, BC03, ProGeny)
   - DSPS HDF5 format specification
5. **Adding parameters**
   - Time-varying metallicity, nebular emission, AGN
6. **The multiscale gradient scalogram**
   - Survey design tool: which resolution constrains which parameter
   - **Figure: Scalogram for a z=0.1 star-forming galaxy**
7. **Summary**

**Key figures (~5):**
- Architecture diagram
- Custom PSD comparison
- Scalogram

---

## Analysis Track (5 notebooks)

### A01 — Mock Program: Generating the Paper's Test Suite

**Goal:** Generate all mock data used in the paper (§3.1). Reproducible.

**Sections:**
1. **The 4 PSD regimes** (smooth, moderate, bursty, highly bursty)
2. **Three redshifts** (z=0.1, z=2, z=6)
3. **Filter sets** per redshift
4. **100 galaxies per regime per redshift**
5. **Save to disk for A02–A04**

**Paper figure generated:**
- **Fig 2: PSD → SFH (4 regimes)** — the DRW power spectra + corresponding SFHs
- **Fig 3: Recovery test design matrix** (schematic)

---

### A02 — Individual SFH Recovery (Paper §4.2)

**Goal:** Tests 1–3 from the paper. Individual galaxy SFH recovery across PSD regimes.

**Sections:**
1. **SFH recovery from photometry** (Test 1)
   - 4 regimes × SDSS ugriz
   - GP-PSD prior vs continuity prior comparison
2. **SFH recovery from spectroscopy** (Test 2)
   - Same galaxies, R=100 spectra
   - Progressive improvement with insets
3. **Individual PSD parameter recovery** (Test 3)
   - Joint (σ_PS, τ_PS) posteriors

**Paper figures generated:**
- **Fig 5: Individual SFH recovery (4 regimes × 2 data types)** — with 200 Myr insets
- **Fig 6: Joint PSD posterior (individual)**

---

### A03 — Population-Level PSD Recovery (Paper §4.3)

**Goal:** Tests 5–7 from the paper. Hierarchical inference.

**Sections:**
1. **DRW PSD recovery** (Test 5) — N=500, three redshifts
2. **Low-burst vs high-burst distinction** (Test 6)
3. **Minimum population size** (Test 7) — N from 50 to 500

**Paper figures generated:**
- **Fig 7: Population-level PSD recovery + N-scaling + population distinction**

---

### A04 — Computational Benchmarks (Paper §4.4)

**Goal:** Wall-clock performance of all 5 inference methods.

**Sections:**
1. **Smooth config** (D=4–7): MAP, RT, NUTS, geoVI, MGVI
2. **Stochastic config** (D~137): MAP, RT, geoVI, MGVI
3. **Hierarchical scaling** with N

**Paper figures generated:**
- **Fig 8: Wall-clock inference time (5 methods × 2 configs)**
- **Fig 10: Sampler comparison in posterior space (appendix)**

---

### A05 — Gradient Sensitivity and Survey Design (Paper §5.2)

**Goal:** The multiscale gradient scalogram and Fisher information analysis.

**Sections:**
1. **End-to-end Jacobian** (∂f_ν/∂θ_k)
2. **Multiscale gradient scalogram** (Eq. 25)
3. **Fisher information** (Eq. 26)
4. **PSD parameter sensitivity** — which wavelengths constrain σ_PS, τ_PS

**Paper figures generated:**
- **Fig 4: Gradient sensitivity Jacobian**
- **Fig 9: Multiscale gradient scalogram**

---

## Shared Infrastructure

### _plot_style.py

Shared plotting utilities imported by all notebooks:

```python
# Style constants
COLORS = {
    "truth": "#222222",
    "map": "#999999",
    "rt": "#1b9e77",      # Ray Tracing — green
    "nuts": "#d95f02",     # NUTS — orange
    "geovi": "#7570b3",    # geoVI — purple
    "mgvi": "#e7298a",     # MGVI — pink
}

PSD_REGIME_COLORS = {
    "smooth": "#1b9e77",
    "moderate": "#d95f02",
    "bursty": "#7570b3",
    "highly_bursty": "#e7298a",
}

# SFH plotting with inset
def plot_sfh_with_inset(ax, t_gyr, sfr, inset_range_myr=200, **kwargs):
    """Plot SFH on lookback-time axis with inset zooming into recent past."""
    ...

# Standard figure sizes
FIG_SINGLE = (7, 4.5)
FIG_DOUBLE = (14, 5)
FIG_QUAD = (10, 7)
```

Key helper functions:
- `plot_sfh_with_inset()` — the workhorse SFH visualization
- `plot_sfh_posterior_with_inset()` — posterior SFH with credible bands + inset
- `setup_style()` — rcParams for publication quality
- `add_redshift_axis()` — twin x-axis with redshift labels
- `safe_corner()` — corner plot wrapper
- `plot_corner_comparison()` — overlaid corner plots from multiple samplers
- `convergence_table()` — Rhat, ESS summary
- `savefig()` — save with consistent naming

---

## The SFH Inset Design

Every SFH recovery figure uses this layout:

```
┌──────────────────────────────────┐
│                    ┌───────────┐ │
│  Full SFH          │ 0–200 Myr │ │
│  (lookback time)   │ inset     │ │
│                    │ (linear   │ │
│                    │  Myr axis)│ │
│                    └───────────┘ │
│                                  │
│  0    2    4    6    8   10  12  │
│       Lookback time [Gyr]       │
└──────────────────────────────────┘
```

The inset uses:
- Linear Myr axis (0–200 Myr lookback)
- Same y-axis scale as main panel (or auto)
- Shaded credible bands clearly visible
- Truth curve thick black
- Hα timescale (~10 Myr) and UV timescale (~100 Myr) marked

This is where burstiness lives, and where the photometry→spectroscopy information gain is most visible.

---

## Notebook Dependencies

```
T00 (standalone)
T01 → T02 → T03 → T04 → T05
                         ↘
T06 (standalone, references T01–T04)

A01 → A02 → A03
A01 → A04
A05 (standalone)
```

---

## File Naming

```
tutorials/
  T00_quickstart.py
  T01_ift_model.py
  T02_forward_model.py
  T03_inference.py
  T04_fitting.py
  T05_hierarchical.py
  T06_extending.py
  _plot_style.py

analysis/
  A01_mock_program.py
  A02_sfh_recovery.py
  A03_population_psd.py
  A04_benchmarks.py
  A05_gradient_sensitivity.py
  _plot_style.py  (symlink or shared)
```
