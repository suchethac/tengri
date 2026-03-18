# diffsed SED Model Priority Roadmap

**Last Updated:** 2026-03-18

## Overview

diffsed's core strengths — stochastic PSD-based SFH, fully differentiable JAX pipeline,
standardized latent space, and five inference backends — are complete. This document
prioritizes the physics and observation models needed for feature parity with
Prospector, Bagpipes, CIGALE, BEAGLE, and Synthesizer.

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| SFH (8 parametric + burst + GP field) | Complete | Best-in-class stochastic model |
| SSP/CSP (DSPS JAX) | Complete | Metallicity interpolation, precomputed tables |
| Dust attenuation (Charlot & Fall) | Partial | Only power-law curve; no flexibility |
| Dust emission | Missing | Cannot fit IR/FIR data |
| Nebular emission | Missing | No emission lines or continuum |
| AGN | Missing | Cannot model active galaxies |
| IGM absorption | Missing | Cannot fit z > 3 UV photometry |
| Metallicity | Basic | Single value, no age evolution |
| Noise model | Basic | Gaussian only, no jitter/outlier |
| Calibration | Missing | No spectrophotometric polynomial |
| Observation types | Complete | Photometry + spectroscopy + combined |
| Filters | Complete | 50+ from SVO |
| Inference | Complete | MAP, RT, NUTS, geoVI, MGVI, EVI |

---

## Phase 1: "Can Fit Real Photometry" (Tier 1 — Essential)

These features are blocking for any real-data application.

### 1.1 Nebular Emission (Lines + Continuum)

**Priority:** Critical
**Why:** Emission lines dominate broadband fluxes at specific redshifts. JWST NIRCam
F444W boosted by 0.5+ mag from H-alpha at z~5-7. Without nebular, stellar masses
biased high by ~0.3 dex. Every SED code comparison (Pacifici+2022) confirms this.

**When it matters:**
- High-z galaxies (z > 1) where strong lines enter photometric bands
- Star-forming galaxies with high sSFR
- JWST photometry (medium bands designed to isolate lines)
- Any spectroscopic fitting

**Architecture:** Modular backend system:
- **BakedIn** (default): wNE SSP files with fixed logU=-3, logZ=0 (current behavior)
- **CLOUDY grid**: Precomputed Byler+2017 grids with free logU, Z_gas (via diffhtwo)
- **Cue** (future): Neural net emulator (Li+2025), 12 parameters, JAX port

**New parameters:** `neb_logU`, `neb_logZ_gas`, `neb_fesc`, `cloudy_grid_path`

**Q_H computation:** On-the-fly JIT-compiled integral below 912 A (differentiable).

**References:** Byler+2017, Li+2025 (Cue), diffhtwo (ArgonneCPAC)

### 1.2 IGM Absorption (Inoue+2014)

**Priority:** Critical for z > 2
**Why:** Lyman-alpha forest and Gunn-Peterson trough attenuate UV flux. Without IGM,
photometric redshifts wrong, UV SFRs biased. Every high-z study uses this.

**When it matters:**
- z > 2: Ly-alpha forest in broadband photometry
- z > 3: Lyman break enters optical (dropout technique)
- z > 6: Complete absorption below Ly-alpha (JWST)
- Photometric redshift estimation

**Implementation:** Pure analytic function T_IGM(lambda_obs, z). Four components:
Lyman series LAF/DLA + Lyman continuum LAF/DLA. ~150 lines of JAX, zero dependencies.

**New parameters:** None (deterministic given redshift)

**References:** Inoue+2014 (MNRAS 442, 1805), eazy-py reference implementation

### 1.3 Dust Attenuation Law Flexibility

**Priority:** High
**Why:** Attenuation curve shape varies between galaxies. SMC-like steep curves at
high-z vs flat Calzetti in starbursts. Affects UV SFR corrections by factors of 2-5.

**When it matters:**
- UV-selected samples (curve slope directly affects SFR)
- Comparing galaxy populations across redshift
- Rest-frame UV spectroscopy
- Energy balance fitting

**Implementation:** Pluggable curve registry with 6 laws:
1. Power law (current default)
2. Calzetti+2000 (starburst standard)
3. Kriek & Conroy 2013 (Calzetti + UV bump + slope mod; Prospector default)
4. SMC/Gordon+2003 (steep UV, no bump; common at high-z)
5. Cardelli/MW (free R_V)
6. Salim+2018 (modified Calzetti + Drude bump; DSPS default)

**Additional features:**
- **f_obscuration** (Lower 2022, Zacharegkas 2025): Fraction of unobscured sightlines
  for clumpy dust geometry. F_att = F_floor + (1-F_floor) * exp(-tau).
- **Per-component laws**: Birth cloud and diffuse ISM can use different curves
  (default: same law for both, backward compatible).

**New parameters:** `dust_law_bc`, `dust_law_diff`, `dust_f_obscuration`,
`dust_bump_strength`, `dust_delta`, `dust_Rv`

**References:** Calzetti+2000, Kriek & Conroy 2013, Gordon+2003, Cardelli+1989,
Salim+2018, Lower+2022, Zacharegkas+2025

### 1.4 Noise Model (Jitter + Outlier)

**Priority:** High
**Why:** Real photometric uncertainties are underestimated. Without jitter, fits are
overconfident. Outlier models prevent bad bands from biasing the whole fit.

**When it matters:**
- Large surveys (COSMOS, CANDELS) with systematic errors
- Heterogeneous filter sets (ground + space)
- Spectroscopy with sky subtraction artifacts
- Any real data

**Implementation:**
- **Jitter:** sigma_eff^2 = sigma_obs^2 + (f_jitter * flux)^2
- **Outlier (Hogg+2010):** Mixture likelihood with broad background component

**New parameters:** `noise_jitter_frac`, `noise_outlier_frac`, `noise_outlier_sigma`

**References:** Hogg+2010

---

## Phase 2: "Can Fit Real Spectroscopy" (Tier 2a — Competitive)

### 2.1 Spectrophotometric Calibration Polynomial

**Priority:** Essential for spectroscopy
**Why:** Observed spectra have wavelength-dependent calibration errors. Without
correction, artifacts bias physical parameters.

**When it matters:**
- Any ground-based spectroscopic fitting
- Joint photometry + spectroscopy
- Grism spectroscopy (HST, Euclid, Roman)

**Implementation:** Chebyshev polynomial of order N (typically 5-10):
C(lambda) = sum_n a_n T_n(lambda_normalized). Coefficients as free parameters
with Gaussian prior. Can be analytically marginalized (linear) for speed.

**New parameters:** `cal_order` (setting), `cal_coeffs` (vector)

**References:** Prospector (Johnson+2021)

### 2.2 Time-Evolving Metallicity

**Priority:** Moderate
**Why:** Real galaxies enrich over time. Constant Z biases mass-weighted ages and
stellar masses for galaxies with extended SFHs.

**When it matters:**
- Massive galaxies with old populations
- Chemical evolution studies
- UV spectroscopy (sensitive to Z of young stars)
- Dwarf galaxies with rapid enrichment

**Implementation:** Linear-in-log ramp:
log Z(t) = log Z_0 + (log Z_final - log Z_0) * t/t_universe.
CSP integral indexes Z per age bin instead of single scalar. Negligible performance impact.

**New parameters:** `met_logzsol_0` (initial Z), `met_logzsol_final` (final Z)

---

## Phase 3: "Full Panchromatic" (Tier 2b — Competitive)

### 3.1 Dust Emission + Energy Balance

**Priority:** Required for IR/FIR
**Why:** Without dust emission, cannot fit lambda > 3 um rest-frame. Excludes
Herschel, ALMA, WISE W3/W4, Spitzer MIPS. Energy balance provides self-consistency.

**When it matters:**
- Dusty star-forming galaxies (DSFGs, SMGs)
- WISE W3/W4, Spitzer 24um, Herschel bands
- Total (UV+IR) SFR measurements
- ALMA-detected high-z galaxies

**Implementation:** Draine & Li 2007 model:
- Parameters: U_min (radiation field), gamma (PDR fraction), q_PAH (PAH mass)
- Pre-tabulated DL07 templates as HDF5 grid, JAX interpolation
- Energy balance: L_dust_emission = L_dust_absorbed (self-consistent normalization)
- Later add: Dale+2014 (1 param), modified blackbody

**New parameters:** `dust_Umin`, `dust_gamma`, `dust_qpah`

**References:** Draine & Li 2007, Dale+2014

### 3.2 AGN Contribution

**Priority:** Moderate (essential for AGN hosts)
**Why:** ~10-20% of galaxies have AGN contributing to UV/optical/MIR SED. Without
AGN component, stellar mass and SFR biased for AGN hosts. JWST revealing AGN at
higher z and lower luminosities than expected.

**When it matters:**
- X-ray detected AGN hosts
- MIR excess galaxies (WISE W1-W2 > 0.8)
- Broad-line AGN (spectroscopy)
- "Little red dots" (JWST high-z compact red objects)

**Implementation:** Two-component model:
1. Accretion disc: power-law f_nu ~ nu^alpha with UV cutoff
2. Torus: heated dust re-emission (CLUMPY templates or simple hot BB)
Additive to stellar SED: L_total = L_stellar + L_AGN.

**New parameters:** `agn_frac`, `agn_tau`, `agn_alpha`

**References:** Nenkova+2008, FSPS AGN templates

---

## Phase 4: "Beyond Parity" (Tier 3 — Frontier)

### 4.1 Nonparametric SFH with Continuity Prior

**Priority:** Useful for comparison
**Why:** The continuity prior (Leja+2019) is the most widely used nonparametric SFH.
Implementing it enables direct comparison with PSD-based stochastic SFH, strengthening
the scientific case for the stochastic approach.

**Implementation:** N time bins with Student-t prior on Delta log SFR between adjacent
bins. ~100 lines of JAX code. Registered via existing SFH registry.

**References:** Leja+2019

### 4.2 Cue Neural Emulator (JAX Port)

**Priority:** Future nebular upgrade
**Why:** Most flexible nebular model. 12 parameters (7 ionizing spectrum shape + 5 gas).
SPS-agnostic (works with any ionizing source). Free [C/O], [N/O] abundance ratios.

**Implementation:** Port TensorFlow architecture (3-layer MLP x 256 units + PCA inverse)
to JAX. Load pre-trained weights. ~50-100 lines. Prospector has a branch but may not
be working — direct port from the cue package is more reliable.

**New parameters:** 12 total (7 ionizing spectrum + log U, log n_H, [O/H], [C/O], [N/O])

**References:** Li+2025 (ApJ 986, 9), github.com/yi-jia-li/cue

### 4.3 Emission Line Specific Fitting

**Priority:** Useful for JWST spectroscopy
**Why:** Instead of fitting lines through broadband, directly fit line fluxes as data.
Joint photometry + line flux fitting common with JWST NIRSpec.

**Implementation:** Line flux prediction from nebular model + separate likelihood term.
Data type: `data_type="lines"` or `data_type="joint_phot_lines"`.

### 4.4 Photometric Redshift Mode

**Priority:** Useful for surveys
**Why:** Free redshift with redshift prior. Makes diffsed usable for photo-z estimation
alongside physical parameter inference.

**Implementation:** Already supports free redshift. Add z-prior options (template-based
or N(z) from photo-z code). Enhance z-table precomputation for speed.

### 4.5 Simulation-Based Inference Interface

**Priority:** Future scalability
**Why:** Following Synthesizer/Synference, enable SBI for ~1000x speedup on large
surveys. diffsed's JAX forward model is perfect for generating SBI training sets.

**Implementation:** Training data generator + neural posterior estimator wrapper
(e.g., sbi package or custom normalizing flow in JAX).

**References:** Cranmer+2020, Synthesizer/Synference

---

## Competitive Comparison

| Feature | diffsed | Prospector | Bagpipes | CIGALE | BEAGLE |
|---------|---------|-----------|----------|--------|--------|
| SFH models | 8+burst+GP | NP+cont. | 7+NP | Flexible | Limited |
| Stochastic SFH | PSD-based | Power-spec | No | No | No |
| Dust atten. laws | 6 (planned) | 7 | 4 | Multiple | CF00 |
| f_obscuration | Yes (planned) | No | No | No | No |
| Dust emission | Phase 3 | DL07 | DL07 | 5 models | Limited |
| Nebular (CLOUDY) | Phase 1 | Byler+17 | CLOUDY25 | Inoue11 | Gutkin+16 |
| Nebular (Cue) | Phase 4 | Branch | No | No | No |
| AGN | Phase 3 | FSPS | No | Fritz/SKIRTOR | BEAGLE-AGN |
| IGM | Phase 1 | Inoue+14 | No | No | Inoue+14 |
| Noise model | Phase 1 | Full | GP | Chi2 | Basic |
| Calibration poly | Phase 2 | Yes | Yes | No | No |
| Evolving Z | Phase 2 | No | Yes | Discrete | Independent |
| Inference | 5 methods | dynesty | MultiNest | Grid | MultiNest |
| Differentiable | Full JAX | No | No | No | No |
| Hierarchical | geoVI | No | No | No | No |

---

## Implementation Principles

1. **Pure JAX** — differentiable, JIT-compilable, GPU-compatible
2. **Optional** — every module disabled by default; enable via ParamSpec
3. **Immutable** — no mutation; return new arrays
4. **Precomputable** — grids loaded at init, interpolated at inference
5. **Registry pattern** — dust laws, nebular backends registered like SFH models
6. **Backward compatible** — existing tests/notebooks pass unchanged

## Data Dependencies

| Module | Data File | Source | Size |
|--------|-----------|--------|------|
| Nebular (CLOUDY) | `cloudy_grid_mist.h5` | FSPS/Byler+2017 | ~50 MB |
| Nebular (Cue) | `cue_weights.npz` | Li+2025 GitHub | ~5 MB |
| Dust emission (DL07) | `dl07_templates.h5` | Draine & Li 2007 | ~100 MB |
| AGN (torus) | `agn_templates.h5` | Nenkova+2008 | ~20 MB |
| IGM | None (analytic) | Inoue+2014 coefficients hardcoded | 0 |
| Dust laws | None (analytic) | Polynomial coefficients hardcoded | 0 |
