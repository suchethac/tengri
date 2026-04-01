# Tengri Notebooks & Documentation Refactor
## Implementation Plan

**Date:** 2026-04-01  
**Author:** Planning session  
**Status:** Ready for implementation

---

## 1. Vision and Story

Tengri is a holistic Bayesian SED fitting framework designed to address the known limitations of existing galaxy SED fitting codes. The unifying narrative running through all notebooks and documentation is:

> *Inferring physical properties of galaxies from observations is hard — not because we lack data, but because our models are too simple, our priors are too rigid, and our inference methods cannot scale to the dimensionality the problem demands. Tengri addresses all of this at once.*

The problems tengri solves, in order of scientific importance:

1. **The parametric SFH problem.** Parametric SFHs (delayed-tau, double power law, etc.) are smooth by construction and cannot represent bursty star formation. This causes systematic bias in derived quantities — particularly stellar mass and SFR — for galaxies at high redshift or in low-mass, bursty regimes. Tengri treats the SFH as a continuous correlated field governed by a power spectral density, enabling the full range of smooth-to-bursty behavior within a single probabilistic model.

2. **The high-dimensional inference problem.** An SFH field over 128 time bins is a 128-dimensional object. Traditional MCMC (emcee, MultiNest) cannot explore this space efficiently. Tengri's end-to-end JAX differentiability enables gradient-based inference (variational inference via geoVI, HMC via NUTS and Ray Tracing) that scales to hundreds of dimensions.

3. **The degeneracy problem.** Age, dust, and metallicity produce similar broadband colors. This is not a bug — it is a fundamental information-theoretic limit of the data. Tengri's Fisher analysis makes these limits explicit and shows what data breaks them.

4. **The population inference problem.** Individual-galaxy fits constrain the SFH amplitude (σ) but not the correlation timescale (τ). These two parameters together define the burstiness of a galaxy population. Tengri's hierarchical inference pools information across N galaxies, recovering both σ and τ with posterior width shrinking as 1/√N.

5. **The completeness problem.** BAGPIPES has no AGN model. Prospector's AGN is two parameters. Neither treats nebular emission with free ionization parameters. Tengri includes a complete, modular physics library — dust attenuation (14 curves) and emission (10 models), AGN disc/torus/lines, nebular emission with three backends including the Cue neural emulator, IGM absorption, radio, X-ray — all differentiable, all fittable jointly.

6. **The scalability problem.** Fitting one galaxy at a time is insufficient for survey-scale science. Tengri's vmap-enabled batch fitting, precomputed photometry (21.6× speedup), and fused JIT kernels enable catalog-scale inference.

Every notebook should reflect this story. The quickstart notebooks show the payoff immediately. The science notebooks demonstrate where existing codes fail. The theory notebooks explain why the approach works. The model galleries show what physics is available.

---

## 2. Current State Analysis

### 2.1 Active notebooks (39 total)

The 39 active notebooks are organized into three legacy directories that no longer reflect the code's maturity:

- `tutorials/` (5 notebooks): API walkthrough and theory exposition
- `demonstrations/` (15 notebooks): scientific use cases  
- `reference/` (19 notebooks): physics reference + model galleries

### 2.2 Problems with current organization

**Duplication.** Each physics domain (dust, AGN, nebular) appears 2–3 times:
- Dust: `03_dust_models` (narrative) + `15_model_gallery_attenuation` + `16_model_gallery_dust_emission` (galleries)
- AGN: `04_agn_and_igm` (basic) + `11_advanced_agn` (detailed) + `17_model_gallery_agn` (gallery)
- Nebular: `05_nebular_emission` + `12_nebular_backends` + `19_model_gallery_nebular`

This means an astronomer looking for dust models finds three notebooks without knowing which to start from.

**Mislabeled and duplicate notebooks.** `demonstrations/05_inference_methods.py` is titled "inference methods" but contains derived quantities and vmap scaling — nearly identical to `demonstrations/06_derived_quantities.py`. One of these should be retired.

**Fragmented science cases.** The bursty SFH argument is split across `03_bursty_sfh_recovery` (low-redshift) and `09_high_redshift_jwst` (JWST). These two notebooks together make the scientific argument; they should be one.

**Hierarchical inference split.** `04_hierarchical_inference` (photometry) and `15_hierarchical_spectroscopy` (spectroscopy) cover the same scientific question. Merging them makes the complete story: photometry constrains σ, spectroscopy constrains τ.

**Theory overlap.** `tutorials/03_the_model.py` and `reference/01_psd_physics.py` both derive the DRW power spectral density. One is sufficient.

**Flat organization.** No clear ordering tells a reader where to start or what to read in sequence.

### 2.3 Sphinx documentation problems

The current docs site (Furo theme, MyST Markdown) has:
- Three deprecated sections (`tutorials/`, `demonstrations/`, `reference/`) that now redirect to newer sections — confusing for users who arrive via search
- No architecture overview showing how the pieces fit together
- No "start here" routing for different user types
- No inference method decision table
- Performance section has stub pages without real benchmark numbers
- The Sphinx gallery (`examples/` → `docs/auto_examples/`) is not currently building due to outdated API in several scripts

---

## 3. New Notebook Structure

### 3.1 Folder organization

Replace the three legacy directories with five purpose-built tracks. Each track has its own `notebook_code/` and `figures/` subdirectory.

```
notebooks/
├── quickstart/          3 notebooks — entry point for new users
├── fitting/             7 notebooks — core workflows and inference
├── theory/              2 notebooks — IFT/PSD depth
├── models/              7 notebooks — physics gallery
└── specialist/          7 notebooks — deep dives
```

Total: 26 notebooks (vs 39 today).

### 3.2 Archive list

Move to `notebooks/archive/` (do not delete):

| Notebook | Why archived |
|---------|-------------|
| `tutorials/02_the_api.py` | Replaced by `quickstart/02_tengri_capabilities.py` |
| `demonstrations/05_inference_methods.py` | Duplicate of `06_derived_quantities.py`, mislabeled |
| `demonstrations/15_hierarchical_spectroscopy.py` | Merged into `fitting/07_hierarchical_psd.py` |
| `reference/01_psd_physics.py` | Merged into `theory/01_sfh_prior.py` |
| `reference/02_data_information_content.py` | Merged into `specialist/03_model_checking.py` |
| `reference/03_dust_models.py` | Unique content absorbed into `models/02_dust_attenuation.py` |
| `reference/04_agn_and_igm.py` | Split: IGM → `models/05_igm.py`; AGN → `models/04_agn.py` |
| `reference/05_nebular_emission.py` | Shock/DIG content absorbed into `models/06_nebular.py` |
| `reference/07_gradient_sensitivity.py` | Jacobian treatment merged into `theory/02_forward_model.py` |
| `reference/08_ray_tracing_sampler.py` | Merged into `fitting/06_advanced_inference.py` |
| `reference/11_advanced_agn.py` | K&D physics and Type 1/2 content merged into `models/04_agn.py` |
| `reference/12_nebular_backends.py` | Q_H physics and decision tree merged into `models/06_nebular.py` |

### 3.3 Notebooks kept unchanged (copy to new location)

| Old path | New path |
|---------|---------|
| `tutorials/01_quickstart.py` | `quickstart/01_quickstart.py` |
| `demonstrations/01_spectroscopic_fitting.py` | `fitting/01_fitting_spectra.py` |
| `demonstrations/10_degeneracies.py` | `fitting/04_degeneracies.py` |
| `demonstrations/11_joint_fitting.py` | `fitting/03_joint_fitting.py` |
| `demonstrations/13_model_comparison.py` | `fitting/05_model_comparison.py` |
| `demonstrations/06_derived_quantities.py` | `specialist/02_derived_quantities.py` |
| `demonstrations/07_extending_tengri.py` | `specialist/04_extending_tengri.py` |
| `demonstrations/08_fitting_real_data.py` | `specialist/01_real_data.py` |
| `demonstrations/14_emission_line_marginalization.py` | `specialist/05_emission_line_marginalization.py` |
| `reference/09_simulation_sfh.py` | `specialist/06_simulation_sfh.py` |
| `reference/10_multiwavelength.py` | `models/07_multiwavelength.py` |
| `reference/16_model_gallery_dust_emission.py` | `models/03_dust_emission.py` |
| `reference/18_model_gallery_sfh.py` | `models/01_sfh_models.py` (minor enrich) |

---

## 4. Notebook-by-Notebook Specifications

### TRACK 1 — QUICKSTART

---

#### `quickstart/01_quickstart.py`
**Source:** `tutorials/01_quickstart.py` — copy unchanged. This notebook is already excellent.

**Contents:** Fit a galaxy spectrum in 10 seconds (smooth D=7); fit a bursty galaxy (D=137); geoVI vs NUTS method comparison; timing benchmarks. No changes needed.

---

#### `quickstart/02_tengri_capabilities.py`
**Source:** Revamp of `tutorials/02_the_api.py`. Discard all JAX internal explanations (vmap, jit, grad internals). Keep only the setup code. Build entirely new content around science-facing figures.

**Purpose:** A "visual abstract" showing what tengri delivers. Six standalone figures, one per capability. An astronomer should be able to scan this notebook and immediately understand what they can extract from their data.

**Section 1 — A Posterior SFH**
Run native_geovi on a spectroscopic mock (standard 7-parameter tsnorm model). Produce a figure showing 50 posterior SFH draws with a 68% credible interval band and the true SFH overlaid. This is the core product of the code — show it first.

**Section 2 — A Corner Plot**
From the same posterior, produce a 7-parameter corner plot. The corner plot should show the banana-shaped age-dust contour clearly — this is a feature, not a bug, and demonstrates the code is sampling the true posterior including all correlations.

**Section 3 — Convergence at a Glance**
Print the convergence diagnostics table (ESS, R-hat, autocorrelation time per parameter). Add brief annotation: what ESS > 400 means for 68% CI reliability. Sourced from `_plot_style.py`'s `convergence_table()` function.

**Section 4 — Scaling to a Catalog**
Show vmap timing: forward-model throughput at N = 1, 10, 100, 1000 galaxies. Produce a timing bar chart with a dashed line showing linear scaling for comparison. The point: tengri's XLA compilation gives sublinear scaling per galaxy.

**Section 5 — What Wavelengths Constrain What**
Compute the Jacobian (∂SED/∂θ) via autodiff and display as a normalized heatmap (parameters on y-axis, wavelength on x-axis). This figure reveals which parameters are constrained by which wavelength ranges — the scientific motivation for multiwavelength fitting. Code pattern from `reference/07_gradient_sensitivity.py`.

**Section 6 — The Gradient is Almost Free**
Time the forward pass versus the gradient computation. Show that the gradient costs approximately 1.5× the forward pass. One-sentence annotation: "This is why HMC and variational inference are practical at D=137."

---

#### `quickstart/03_bursty_sfh_recovery.py`
**Sources:** `demonstrations/03_bursty_sfh_recovery.py` (base) + `demonstrations/09_high_redshift_jwst.py` (appended).

**Purpose:** The scientific argument for tengri over parametric SFH codes. Self-contained. Should be readable as a standalone document.

**Opening narrative cell:** Two sentences framing the problem — "Parametric SFH models cannot represent burstiness. This causes systematic bias in derived quantities, particularly at high redshift where burstiness is most common."

**Section 1 — Four Burstiness Regimes**
The table of PSD parameter combinations defining the four regimes (smooth, mildly bursty, bursty, highly bursty). Taken from D3. Show example SFH realizations from each regime.

**Section 2 — SFH Recovery**
4-row × 2-column figure: rows are the four regimes; columns are photometry-only and spectroscopy fits. Each panel shows the true SFH, the posterior median, and the 68% CI band. This is the equivalent of Figure 4 from the paper.

**Section 3 — The Wrong-Model Trap**
Fit the same bursty mock galaxy with (a) a stochastic SFH model and (b) a parametric double-power-law model. Show that the parametric model achieves similar chi-squared but recovers the wrong posterior on derived quantities (log M★, SFR_100, SFR_10). Quantify the bias: approximately 0.3 dex in M★ for highly bursty galaxies.

**Section 4 — Why It Matters at JWST: The High-Redshift Stakes**
The z=6 JWST case from D9. Setup: JWST NIRCam + NIRSpec filters at z=6. Show (a) the SFH recovery at cosmic dawn where burstiness peaks and (b) the stellar mass posterior comparison showing the parametric code overestimates M★ by 0.3 dex. This is the punchline: the wrong-model bias is worst exactly where JWST is making new discoveries.

**Combined takeaway cell:** Bursty galaxies at high redshift are the norm, not the exception. A model that forces smooth SFHs will systematically bias the stellar mass function and SFR density at z > 4.

---

### TRACK 2 — FITTING

---

#### `fitting/01_fitting_spectra.py`
**Source:** `demonstrations/01_spectroscopic_fitting.py` — copy unchanged. Already comprehensive and well-structured.

---

#### `fitting/02_fitting_photometry.py`
**Source:** `demonstrations/02_photometric_catalogs.py` (base). Add three new efficiency sections.

**Purpose:** Photometric catalog fitting with maximum computational efficiency. The new sections teach the implementing agent how to unlock the three major speedups: photometry precomputation, fused kernels, and vmap batch inference.

**Existing sections (copy from D2):**
The current D2 content covers single galaxy fit, vmap batch timing, and parameter recovery for 10 galaxies. Keep all of this.

**New Section A — Photometry Precomputation (21.6× speedup)**
When the redshift is fixed in ParamSpec, Model automatically precomputes SSP fluxes through the filter set at initialization time. Subsequent calls to `predict_photometry` skip this integration entirely. Show the setup (requires `redshift=Fixed(...)` in ParamSpec), demonstrate with `model.summary()` output confirming precomputation is active, and time the forward pass with and without precomputation. Produce a bar chart comparing the two. Note clearly: "Precomputation activates automatically when redshift is Fixed(). No extra code required."

**New Section B — Fused JIT Kernels**
Fused kernels combine multiple operations (age-weight computation, metallicity interpolation, dust application, and the final einsum) into a single JIT scope, eliminating intermediate array materializations. These activate automatically. Show `model.summary()` confirming "Fused kernels: YES" and include a brief timing comparison showing the speedup (see benchmarks in CLAUDE.md). This section is informational — no API change needed, just show it's working.

**New Section C — Batch Fitting via fit_batch**
Show the full catalog fitting workflow using `fitter.fit_batch(galaxies)`. This is the high-level API for fitting multiple galaxies with independent inference. Distinguish it from vmap: `fit_batch` runs separate inference per galaxy (full posterior per galaxy); vmap over `predict_photometry` is for batch forward modeling only. Produce a parameter recovery figure for 10 galaxies showing truth vs posterior median with 68% CI error bars.

---

#### `fitting/03_joint_fitting.py`
**Source:** `demonstrations/11_joint_fitting.py` — copy unchanged.

---

#### `fitting/04_degeneracies.py`
**Source:** `demonstrations/10_degeneracies.py` — copy unchanged. The Fisher matrix analysis and progressive filter set comparison are already comprehensive.

---

#### `fitting/05_model_comparison.py`
**Source:** `demonstrations/13_model_comparison.py` — copy unchanged.

---

#### `fitting/06_advanced_inference.py`
**Sources:** `demonstrations/12_advanced_inference.py` (base) + `reference/08_ray_tracing_sampler.py` (Ray Tracing section inserted).

**Purpose:** Complete comparison of all inference methods with guidance on when to use each.

**Structure:** Copy all sections from D12. After the NUTS section and before the decision tree, insert R8's content as "Ray Tracing in Depth." This covers: the Snell's law physical analogy, step-size sensitivity (the viability cliff at step_size ~ 0.06 for D~137), the DKD vs KDK integrator choice, and empirical acceptance rate guidance. The combined decision tree at the end should include a Ray Tracing row.

---

#### `fitting/07_hierarchical_psd.py`
**Sources:** `demonstrations/04_hierarchical_inference.py` (base) + `demonstrations/15_hierarchical_spectroscopy.py` (second half).

**Purpose:** Population-level PSD inference — the paper II showcase. Shows why individual fits fail and how hierarchical pooling solves it.

**Part 1 (from D4):**
Individual galaxy posteriors on (σ, τ): σ is well-constrained but τ is poorly constrained by any single galaxy. Set up HierarchicalFitter with N=2, 4, 6, 8, 10 galaxies. Show posterior width vs N on a log-log plot and the 1/√N scaling. Show two-population separation in σ-τ space (bursty dwarfs vs smooth disks) becoming significant at N~5.

**Part 2 (from D15):**
Repeat the analysis with spectroscopic data instead of photometry. Show the comparison: photometry at N=10 achieves similar τ constraint as spectroscopy at N=2. Produce a side-by-side figure comparing the σ-τ posterior from 10 photometric vs 5 spectroscopic galaxies. Key message: spectroscopy is ~5× more informative for the coherence timescale τ.

**Combined summary:** Guidance on when photometry is sufficient (constraining σ with N~10) vs when spectroscopy is needed (constraining τ with N~5).

---

### TRACK 3 — THEORY

---

#### `theory/01_sfh_prior.py`
**Sources:** `tutorials/03_the_model.py` (base) + Green's functions section from `reference/01_psd_physics.py`.

**Purpose:** Deep treatment of the IFT correlated field SFH model. This is the scientific core of the paper.

**Structure:** Copy all sections from T3 verbatim. After the "Burstiness Plane" section (the 3×3 σ-τ grid), insert the Green's functions section from R1.

**Sections from T3 (copy verbatim):**
Power spectral density definition; what σ and τ physically control; building an SFH from scratch; the lognormal correction K(0)/2 and why it preserves the mean; the secular backbone; the 3×3 burstiness plane figure; observable connection.

**New section from R1 (insert after burstiness plane):**
Green's functions connecting PSD parameters to observational tracers. Hα probes SFR on 10 Myr timescales; UV-continuum probes 100 Myr; D4000 and Balmer break probe Gyr-old stellar populations. A figure with three panels showing the sensitivity kernel of each tracer overlaid on the DRW autocorrelation function for different τ values. Physical message: the ratio of Hα/UV diagnoses τ_PS; the Hα/optical ratio diagnoses σ_PS. This connects the abstract PSD parameters to quantities astronomers already measure.

---

#### `theory/02_forward_model.py`
**Sources:** `tutorials/04_the_forward_model.py` (base) + full Jacobian treatment from `reference/07_gradient_sensitivity.py`.

**Purpose:** Step-by-step pipeline from SFH parameters to observable SED, with a focus on what differentiability gives us scientifically.

**Structure:** Copy T4 sections 1–6 verbatim (SSP building blocks, CSP construction, dust attenuation, metallicity effects, SED to photometry, complete pipeline). Then replace T4's brief Jacobian section with R7's full treatment.

**Expanded Jacobian section (from R7):**
Compute the exact Jacobian (∂SED/∂θ) via autodiff. Display as a sensitivity heatmap: parameters on y-axis, wavelength on x-axis, color proportional to |∂F/∂θ| normalized per parameter. Add a multiscale wavelet scalogram showing the decomposition of gradient energy across spectral scales — this separates broadband-sensitive parameters (dust, stellar mass) from narrow-feature-sensitive parameters (metallicity, nebular emission lines). Physical message: the Jacobian explains why spectroscopy constrains metallicity orders-of-magnitude better than photometry — the narrow spectral features carry all the information.

---

### TRACK 4 — MODELS (GALLERY FORMAT)

All models notebooks follow gallery format: figure-heavy, minimal prose. Show the options. The implementing agent should not spend time explaining physics — just produce comprehensive, well-labeled figures. An astronomer should be able to look at any figure and immediately know what parameter was varied.

---

#### `models/01_sfh_models.py`
**Source:** `reference/18_model_gallery_sfh.py` (base, enrich slightly).

This IS a deeper notebook because SFH is the scientific contribution. Keep all of R18's parameter sweeps unchanged. Add two things:

First, a "when to use which model" table at the top: model name, best use case, free parameter count, whether stochastic. This is a one-sentence-per-model orientation for a reader choosing a parameterization.

Second, a brief composition system section (if not already in R18) showing how to combine parametric + burst + GP field into a single model, with an example figure.

---

#### `models/02_dust_attenuation.py`
**Source:** `reference/15_model_gallery_attenuation.py` (base). Copy all parameter sweeps unchanged.

**One addition:** After the two-component model section, add a single figure showing the age-dust degeneracy in r-i color space. This figure is currently in `reference/03_dust_models.py` Section 5. It shows that an old dusty galaxy and a young less-dusty galaxy are photometrically indistinguishable in SDSS photometry — the fundamental reason the age-dust degeneracy is hard to break. One figure, one annotation, no extended discussion.

---

#### `models/03_dust_emission.py`
**Source:** `reference/16_model_gallery_dust_emission.py` — copy unchanged. Already compact and comprehensive.

---

#### `models/04_agn.py`
**Sources:** `reference/17_model_gallery_agn.py` (base gallery structure) + specific physics content from `reference/11_advanced_agn.py`.

**Framing:** AGN is one module in tengri's complete physics library. The notebook shows what is available and how to use it. The emphasis is on completeness — showing that tengri covers the full AGN SED from accretion disc to emission lines to torus — not on making AGN the hero of the story.

**Section 1 — Model Registry Overview**
A table listing all registered AGN models: name, disc type, torus type, free parameter count, typical use case. This is the practical entry point for a user choosing an AGN configuration. Source: AGN_MODEL_COMPARISON.md and R17 Section 1.

**Section 2 — Accretion Disc Models**
Three panels comparing the disc SED from: power-law (simple, 2 params), SS73 multicolor disc (standard, 4 params), and Kubota & Done 3-zone disc (kubota_done, 9 params). Show how each model captures more physics at the cost of free parameters. Source: R17 Section 2.

**Section 3 — BH Spin and Eddington Ratio**
Two side-by-side panels: (left) disc SED vs BH spin a=[0, 0.5, 0.9, 0.998] at fixed mass and Eddington ratio; (right) disc SED vs log(L/L_Edd)=[-2, -1.5, -1, -0.5] at fixed spin. These are the two main knobs for constraining the central engine. Source: R11 Section 2.

**Section 4 — Torus Models**
Two panels: single modified-blackbody torus vs two-temperature torus. Vary optical depth and temperature. Show the 9.7 μm silicate feature. Source: R17 Section 4.

**Section 5 — Emission Lines: NLR and BLR**
Two panels showing the NLR optical spectrum zoom (3500–7000 Å with 11 forbidden/Balmer lines) and the BLR UV spectrum zoom (900–3000 Å with 9 permitted lines). This is what tengri has that BAGPIPES, Prospector, and ProSpect do not have at all. Caption should note: line strengths calibrated to Vanden Berk+2001 quasar composite. Source: R17 Sections 5–6.

**Section 6 — Unified Type 1/2 Model**
Two SEDs on one plot: face-on (Type 1, cos_inc=0.9) and edge-on (Type 2, cos_inc=0.1). Annotate which components are visible in each orientation. Source: R11 Section 5. Brief annotation: the smooth sigmoid masking means the AGN classification is continuous and differentiable — not a discrete Type 1/2 switch.

**Section 7 — AGN + Galaxy Composite**
Full galaxy SED with AGN fractions of 1%, 10%, and 50%. Shows where AGN contribution dominates (MIR torus, UV disc) vs where galaxy dominates (optical continuum, NIR). Source: R4 "Galaxy + AGN SED" section.

---

#### `models/05_igm.py`
**Sources:** IGM sections from `reference/04_agn_and_igm.py` and `reference/19_model_gallery_nebular.py`.

**Purpose:** Compact reference for high-redshift observers. Three figures total.

Figure 1: IGM transmission curves T_IGM(λ_obs) at z = 0.5, 1, 2, 3, 5, 7 overlaid on the same axes. Shows the progressive Lyman forest suppression as redshift increases.

Figure 2: Same transmission curves with SDSS filter curves and JWST NIRCam filter curves overlaid, showing which photometric bands are affected at which redshift.

Figure 3: Patchy reionization damping wing at z=7 for neutral fractions x_HI = 0, 0.3, 0.7, 1.0. Shows the damping wing added to the continuum near Ly-α.

No extended physics discussion. These three figures are the complete reference.

---

#### `models/06_nebular.py`
**Sources:** `reference/19_model_gallery_nebular.py` (base gallery) + `reference/12_nebular_backends.py` (Q_H and Cue architecture) + `reference/05_nebular_emission.py` (shock and DIG figures).

**Purpose:** Comprehensive Cue coverage plus compact gallery treatment of all other nebular content.

**Section 1 — Backend Decision Table**
A single markdown table: BakedIn (0 free params, use for photometric fitting, logU fixed), CloudyGrid (3 free params, use for spectroscopy with ionization variation), Cue (12 free params, use when abundance ratios or non-stellar ionizing sources matter). This should be the first thing a reader sees.

**Section 2 — Q_H: The Link Between Stars and Nebular Emission**
From `reference/12_nebular_backends.py`. A figure showing Q_H (ionizing photon rate) vs stellar age for three metallicities. Shows that Q_H drops sharply after 10 Myr — young stars dominate ionizing photons entirely, making nebular strength a direct probe of current SFR.

**Section 3 — CloudyGrid: Line Ratios vs Ionization Parameter**
From `reference/19_model_gallery_nebular.py`. Key emission line ratios ([OII]/[OIII], [NII]/Hα, [SII]/Hα) vs logU at fixed metallicity. Standard reference for choosing CloudyGrid parameter ranges.

**Section 4 — Cue Neural Emulator: Architecture and Parameters**
This section should be comprehensive. Describe and illustrate all 12 input parameters:
- 7 ionizing spectrum shape parameters: slopes at the He II, O II, He I, and H I ionizing edges (α₁–α₄) plus three inter-segment flux ratios at those breaks
- 5 gas parameters: log U (ionization parameter), log n_H (hydrogen density), [O/H] (oxygen abundance), [N/O] (nitrogen-to-oxygen ratio), [C/O] (carbon-to-oxygen ratio)

The architecture: three hidden layers of 256 units with Swish activation, PCA output basis with 50 components, 16 sub-networks for emission-line groups plus one continuum sub-network. Reference: Li et al. 2025, ApJ 986, 9.

**Section 5 — Cue Parameter Effects**
Three figures showing the effect of Cue's unique parameters:
- [N/O] effect: vary N/O ratio from −0.5 to +1.0 at fixed logU/metallicity; show [NII]/Hα ratio change in the optical spectrum. This is the primary BPT discriminant.
- [C/O] effect: vary C/O ratio; show CIII] 1909 Å in the UV spectrum. Relevant for high-redshift spectroscopy.
- Ionizing spectrum shape: vary the hard-UV slope (α₄ parameter) from −1 to −5; show effect on optical line ratios. Relevant when the ionizing source is non-stellar (AGN nebular, shocks).

**Section 6 — Unique to Tengri: Differentiable AGN Nebular Chain**
A brief section (1-2 figures) showing the unique capability: connect the AGN disc's extreme-UV output directly to Cue's ionizing spectrum parameters, creating a self-consistent AGN + nebular emission model. The disc EUV → Cue ionspec → NLR line prediction chain. No other SED fitting code does this differentiably. Show an example spectrum with AGN + Cue NLR output.

**Section 7 — Cue vs CloudyGrid Comparison**
Side-by-side optical spectra at matched logU and metallicity. Show where the two backends agree (standard HII region conditions) and where they diverge (high-ionization lines, edge cases).

**Section 8 — Shock Diagnostics (BPT)**
One figure: the standard BPT diagram ([OIII]/Hβ vs [NII]/Hα) with the HII region sequence from CloudyGrid, the shock sequence from MAPPINGS V (varying velocity), and the f_shock mixing trajectory. One figure, no extended discussion.

**Section 9 — DIG Contamination**
One figure: [NII]/Hα and [SII]/Hα as a function of DIG fraction (0 to 60%). Caption: "30–60% of Hα in local spiral galaxies is diffuse ionized gas (Haffner+2009; Tacchella+2022), biasing BPT diagnostics."

**Section 10 — Observation Models (compact)**
Brief section from R19: filter convolution, LSF convolution, Chebyshev calibration polynomial. Show figures only, no extended discussion — these are covered in detail in `specialist/07_advanced_spectroscopy.py`.

---

#### `models/07_multiwavelength.py`
**Source:** `reference/10_multiwavelength.py` — copy unchanged. Compact already. Covers radio (FIR-radio correlation, HMXB) and X-ray (AGN corona), building a panchromatic SED.

---

### TRACK 5 — SPECIALIST

---

#### `specialist/01_real_data.py`
**Source:** `demonstrations/08_fitting_real_data.py` — copy unchanged.

---

#### `specialist/02_derived_quantities.py`
**Source:** `demonstrations/06_derived_quantities.py` — copy unchanged. Shows M★, SFR, sSFR from posterior samples, the star-forming main sequence from the prior, and vmap scaling benchmarks.

---

#### `specialist/03_model_checking.py`
**Sources:** `tutorials/05_prior_predictive.py` (Part 1) + `reference/02_data_information_content.py` (Part 2).

**Part 1 — Prior Predictive Checks:** 200-galaxy prior ensemble colored by u−r; good vs pathological prior comparison with SDSS color locus; stochastic SFH prior realizations colored by σ_PS. From T5.

**Part 2 — Information Content:** Progressive data reveal (1 band → 3 bands → 5 bands → spectroscopy); posterior width vs number of photometric bands per parameter; spectroscopy vs photometry information gain quantified (~1–2 orders of magnitude for metallicity and stellar age). From R2.

---

#### `specialist/04_extending_tengri.py`
**Source:** `demonstrations/07_extending_tengri.py` — copy unchanged.

---

#### `specialist/05_emission_line_marginalization.py`
**Source:** `demonstrations/14_emission_line_marginalization.py` — copy unchanged.

---

#### `specialist/06_simulation_sfh.py`
**Source:** `reference/09_simulation_sfh.py` — copy unchanged.

---

#### `specialist/07_advanced_spectroscopy.py`
**Sources:** `reference/06_noise_models.py` (Part 1) + `reference/13_spectroscopic_tools.py` (Part 2) + `reference/14_alpha_enhancement.py` (Part 3).

**Part 1 — Noise Models:** Calibration floor concept; Student-t likelihood for outlier robustness; comparison of residuals with/without noise model.

**Part 2 — Spectroscopic Tools:** Chebyshev calibration polynomial; NIRSpec PRISM/G140M resolution profiles; LSF convolution; velocity broadening (Ca II triplet example); emission line blending at different spectral resolutions; analytic marginalization of calibration coefficients.

**Part 3 — Alpha Enhancement:** 4D SSP grids ([α/Fe] as fourth axis); spectral effect of [α/Fe] (Mg b strengthens, Fe lines weaken); time-evolving [α/Fe] parameterization; Salaris relation ([Fe/H] vs [M/H] convention conversion).

---

## 5. Sphinx Documentation Revamp

### 5.1 Current structure assessment

The current docs have three deprecated redirect stubs (`tutorials/`, `demonstrations/`, `reference/`) that send users to the newer sections. These are now irrelevant and should be removed. The new notebook organization should align directly with the docs structure.

Additionally: the docs `index.md` lacks an architecture overview; the `inference/index.md` lacks a method selection table; the `performance/` section has stub pages; and the Sphinx gallery is not building.

### 5.2 New docs site structure

```
docs/
├── index.md                          UPDATE: add architecture diagram and "start here" table
├── getting_started/
│   ├── index.md                      UPDATE: point to quickstart/ track
│   └── concepts.md                   NEW: one-page "how tengri works"
├── the_model/
│   ├── index.md                      UPDATE: reorganize for new theory/ + models/ structure
│   ├── sfh_prior.md                  UPDATE: link to theory/01_sfh_prior
│   └── physics_reference.md          UPDATE: link to models/ track
├── inference/
│   ├── index.md                      UPDATE: add decision table at top
│   ├── methods.md                    UPDATE: link to fitting/06_advanced_inference
│   └── hierarchical.md               UPDATE: link to fitting/07_hierarchical_psd
├── worked_examples/
│   └── index.md                      UPDATE: reorganize by new track structure
├── observation/                       KEEP — already good
├── performance/
│   ├── benchmarks.md                 UPDATE: add real timing numbers
│   └── optimization.md               UPDATE: add precomputation + fused kernel guidance
├── advanced/                          KEEP
├── api/                               KEEP
├── developer/                         KEEP
└── auto_examples/                     Sphinx gallery output — regenerate after fixing scripts
```

Remove the three deprecated stub files:
- `docs/tutorials/index.md`
- `docs/demonstrations/index.md`
- `docs/reference/index.md`

### 5.3 Key pages to rewrite

**`docs/index.md` — add:**

An architecture flowchart (ASCII or Mermaid) showing:
```
Observations → Fitter → Posterior
                ↑
            Model(spec, ssp)
            ↑           ↑
        ParamSpec      SSP grid
        (physics)      (templates)
```

A "Start here" table routing users to the right notebook:

| I want to... | Go to |
|---|---|
| Fit my first galaxy in 10 seconds | quickstart/01_quickstart |
| Understand why tengri beats parametric SFH codes | quickstart/03_bursty_sfh_recovery |
| Fit a photometric catalog efficiently | fitting/02_fitting_photometry |
| Understand the SFH prior | theory/01_sfh_prior |
| See all available dust/AGN/nebular models | models/ track |
| Choose an inference method | fitting/06_advanced_inference |
| Infer population-level SFH statistics | fitting/07_hierarchical_psd |

**`docs/inference/index.md` — add decision table at top:**

| Method | Dimensionality | Time | Exact? | Recommended for |
|---|---|---|---|---|
| MAP | any | seconds | No | Initialization only |
| Laplace | D ≲ 20 | instant | Gaussian approx | Quick uncertainty estimate |
| Pathfinder | D ≲ 100 | ~5s | Approx | NUTS warm-start |
| geovi (NIFTy) | any | ~12s | Approx | **Default: single galaxy** |
| native_geovi | any | 30s compile + fast | Approx | vmap/catalog fitting |
| NUTS | D ≲ 30 | minutes | Yes | Validation |
| Ray Tracing | D ≲ 200 | hours | Yes | High-D exact posterior |
| NSS | D ≲ 30 | hours | Yes | Bayesian model evidence |
| Hierarchical | any | depends | Approx | Population-level PSD |

**`docs/performance/benchmarks.md` — add real numbers from CLAUDE.md:**
- Forward model: 140 μs (D=7 smooth), 356 μs (D=137 stochastic), MacBook Pro M-series CPU
- Gradient: 56 μs (D=7), 63 μs (D=137)
- Photometry precomputation speedup: 21.6×
- native_geovi compile: 56s (cached), run: 0.3s (D=7), 0.8s (D=137)

**`docs/getting_started/concepts.md` — new page:**
One-page explanation of how tengri works without code. Three paragraphs: (1) the SFH is a latent field, (2) the forward model maps the field to observables, (3) inference inverts this mapping. One figure: the model schematic from Figure 1 of the paper.

---

## 6. Sphinx Gallery Plan

### 6.1 What the gallery is

The Sphinx Gallery produces `docs/auto_examples/` from short standalone scripts in `examples/`. Each script produces one figure and is compiled into a visual gallery page. This is separate from the notebooks — the gallery is for quick visual discovery; the notebooks are for full workflows.

### 6.2 Why the gallery is broken

All 21 existing `examples/plot_*.py` scripts likely fail because of deprecated API names. The following replacements are needed across all scripts (from CLAUDE.md):

| Old name | New name |
|---|---|
| `ForwardModel` | `Model` |
| `fit_catalog` | `fit_batch` |
| `tau_v1`, `tau_v2` | `tau_bc`, `tau_diff` |
| `dust_n` | `dust_slope` |
| `sigma_ps`, `tau_ps` | `psd_sigma`, `psd_tau_yr` |
| `log_z` | `log_z_abs` |
| `geovi_nifty`, `geovi_full` | `nifty_geovi` |
| `mgvi_nifty`, `mgvi_full` | `nifty_mgvi` |
| `charlot_fall` import | `two_component_dust(law_bc="power_law")` |

Fix all 21 existing scripts first, then run the gallery build to confirm they execute cleanly.

### 6.3 Gallery organization (new)

Reorganize with `README.rst` introductions in each subdirectory. Add 8 new scripts (described below). Final gallery structure:

```
examples/
├── quickstart/          2 existing + README.rst
├── sfh/                 5 existing + 2 new + README.rst
├── inference/           3 existing + 1 new + README.rst
├── agn/                 2 existing + 1 new + README.rst
├── dust/                2 existing + 1 new + README.rst
├── nebular/             1 existing + 1 new + README.rst
├── photometry/          2 existing + README.rst
├── spectroscopy/        1 existing + 1 new + README.rst
└── advanced/            4 existing + 1 new + README.rst
```

### 6.4 New gallery scripts (8 scripts)

Each script is 60–80 lines, produces one figure, is fully self-contained, and starts with a docstring that becomes the gallery page title and description.

**`examples/sfh/plot_bursty_recovery.py`**
Title: "SFH Recovery Across Four Burstiness Regimes"
Figure: 4-panel SFH recovery showing truth + 68% CI for smooth, mildly bursty, bursty, highly bursty galaxies. Source: D3 Section 2, first figure only.

**`examples/sfh/plot_wrong_model_trap.py`**
Title: "The Wrong-Model Trap: Parametric Bias in Derived Quantities"
Figure: M★ posterior comparison — stochastic model vs parametric model on the same bursty galaxy. The parametric model is overconfident and offset. Source: D3 Section 3–4.

**`examples/inference/plot_hierarchical_convergence.py`**
Title: "Population PSD Recovery: 1/√N Convergence"
Figure: Posterior width on σ and τ vs number of galaxies N on a log-log scale, with 1/√N line overlaid. Source: D4 Section 3.

**`examples/agn/plot_agn_type12.py`**
Title: "Type 1 vs Type 2 AGN: Geometric Unification"
Figure: Two SEDs (face-on and edge-on) from the unified_nlr_blr model. Source: R11 Section 5.

**`examples/dust/plot_dust_emission_models.py`**
Title: "Dust Emission Models: Overview"
Figure: All 10 dust emission models at similar effective temperature. Source: R16 overview section.

**`examples/nebular/plot_bpt_diagnostics.py`**
Title: "BPT Diagnostics: Star Formation, Shocks, and AGN"
Figure: BPT diagram with HII sequence, shock track (MAPPINGS V), and Kauffmann+2003 demarcation line. Source: R5 shock section.

**`examples/spectroscopy/plot_spectral_features.py`**
Title: "Key Spectral Features as Age and Metallicity Probes"
Figure: D4000, Hδ, and Mg b vs stellar age for three metallicities. New code using model.predict_spectrum().

**`examples/advanced/plot_fisher_degeneracy.py`**
Title: "Age-Dust-Metallicity Degeneracy: Fisher Analysis"
Figure: Parameter uncertainty vs filter set (SDSS, SDSS+NIR, SDSS+NIR+MIR) as bar chart, plus Fisher eigenvalue spectrum. Source: D10 Fisher section.

---

## 7. Implementation Order

### Phase 1 — Archive and restructure (no code changes)
Create new folder structure. Move 12 notebooks to `notebooks/archive/`. Copy 13 notebooks unchanged to new locations.

### Phase 2 — Expansion notebooks (copy base + add sections)
These require inserting new content into existing notebooks but no fundamentally new code:
1. `fitting/02_fitting_photometry.py` — add precomputation + fused + fit_batch sections
2. `fitting/06_advanced_inference.py` — insert Ray Tracing section from R8
3. `fitting/07_hierarchical_psd.py` — merge D4 + D15
4. `models/01_sfh_models.py` — add usage table to R18
5. `models/02_dust_attenuation.py` — add age-dust degeneracy figure from R3
6. `specialist/03_model_checking.py` — merge T5 + R2

### Phase 3 — Merged/rewritten notebooks (significant effort)
1. `quickstart/02_tengri_capabilities.py` — new science figures notebook
2. `quickstart/03_bursty_sfh_recovery.py` — merge D3 + D9
3. `theory/01_sfh_prior.py` — merge T3 + R1
4. `theory/02_forward_model.py` — merge T4 + R7
5. `models/04_agn.py` — revamp from R17 + R11
6. `models/05_igm.py` — compact new notebook
7. `models/06_nebular.py` — comprehensive revamp with full Cue coverage
8. `specialist/07_advanced_spectroscopy.py` — merge R6 + R13 + R14

### Phase 4 — Fix Sphinx gallery (21 existing scripts)
Find and replace all deprecated API names listed in Section 6.2. Run gallery build to verify all scripts execute without error.

### Phase 5 — Write new gallery scripts (8 scripts)
Write the 8 new `examples/*/plot_*.py` scripts. Each is short and standalone.

### Phase 6 — Add README.rst files to gallery sections
One RST file per gallery subdirectory with a 2–3 sentence introduction.

### Phase 7 — Update Sphinx docs
Update `docs/index.md`, `docs/inference/index.md`, `docs/performance/benchmarks.md`, `docs/getting_started/index.md`, `docs/worked_examples/index.md`. Write new `docs/getting_started/concepts.md`. Remove three deprecated stub pages.

---

## 8. Critical Technical Notes for Implementing Agent

The implementing agent must follow these rules from CLAUDE.md to avoid silent bugs:

1. Every notebook must call `jax.config.update("jax_enable_x64", True)` before any JAX operation.
2. Free parameter names use full prefixes: `sfh_dpl_alpha`, not `sfh_alpha`. Always verify with `spec.free_params`.
3. Photometry precomputation triggers only when `redshift=Fixed(...)` in ParamSpec AND photometry is set in Observation.
4. `fit_batch` is the correct method name (not `fit_catalog`).
5. `vmap` batch path requires `method="native_geovi"` (NIFTy fast path does not vmap).
6. PSD timescale: user-facing is `psd_tau_myr` (Myr); internal is `psd_tau_yr` (years). Do not mix.
7. SSP metallicity grid is log10(Z) absolute, not log10(Z/Zsun). The offset LOG10_ZSUN = −1.848 is applied by the param_map.
8. IGM transmission function takes observed-frame wavelengths, not rest-frame.
9. Ray Tracing: for D~137, the viability cliff is at step_size ~ 0.06. Use 0.05 with more leapfrog steps.
10. `jax.random.fold_in(key, hash(string))` overflows uint32. Use `abs(hash(x)) % (2**31)`.
11. DL07/Dale2014 templates require `data/dl07_templates.npz` to exist. If missing, the code uses an analytic fallback with a warning — this fallback is not suitable for science.
12. Import `convergence_table`, `plot_sfh`, `setup_style`, `COLORS` from `notebooks/_plot_style.py` using the sys.path insertion pattern shown in all existing notebooks.
13. The standard SSP file used in most notebooks: `data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`.

---

*End of implementation plan.*
