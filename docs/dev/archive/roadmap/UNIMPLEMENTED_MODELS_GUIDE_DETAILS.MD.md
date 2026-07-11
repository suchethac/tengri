# tengri — Unimplemented Models: Complete Implementation Guide

This document provides full specifications for models described in the
Paper I appendix but not yet implemented in code, plus additional models
not in the paper that would strengthen the framework. Each section contains
paper references, key equations, parameters, implementation plan,
reference code, and parity tests.

Papers referenced here are available at:
*(private arxiv source library)*

---

## 1. Astrodust+PAH Dust Emission (Hensley & Draine 2023)

**Paper:** Hensley & Draine 2023, ApJ 948, 55  
**arXiv:** 2208.12365  
**Replaces:** Draine & Li 2007 grain composition  
**Status:** Appendix F.3.8 describes it; no code or templates exist.

### Template & Data Downloads

| Resource | URL | Contents |
|----------|-----|----------|
| Astrodust+PAH model output | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3B6E6S | FITS files: extinction, emission, polarized emission, spinning dust, integrated over fiducial size distribution (RV=3.1 MW). 1000 wavelengths, 167 grain sizes. |
| PAH emission spectra | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LPUHIQ | PAH emission for a wide range of radiation fields (3 Myr starburst to M31 bulge old stars). Individual PAH sizes + selected size distributions. Cite Draine+2021 ApJ 917:3. |
| Astrodust dielectric functions | https://datacommons.princeton.edu/discovery/catalog/doi-10-34770-9ypp-dv78 | Dielectric functions from 1 Å to 5 cm (12.4 keV to 2.6×10⁻⁵ eV). Cross sections for spheroids (a_eff = 3.16 Å to 5.01 μm). Multiple axial ratios. CC BY 4.0. Download via Globus. Cite Draine & Hensley 2021a, ApJ 910:47. |
| Python toolkit | https://github.com/brandonshensley/Astrodust | Routines for reading dielectric functions and cross sections. Tutorial notebook: `notebooks/model_file_tutorial.ipynb` shows how to read the FITS files, extract emission spectra per grain size, and integrate over size distributions. |
| Draine dust page | https://www.astro.princeton.edu/~draine/dust/astrodust.html | Central hub linking all data releases. Also hosts DL07 templates for comparison. |

**Critical citation requirements:** When using astrodust data, cite (1) Hensley & Draine 2023 for the model, (2) Draine & Hensley 2021a for the dielectric function, and (3) Hensley & Draine 2021 for the observational constraints.

### Physics

Replaces the silicate-graphite-PAH grain composition of DL07 with a single
composite grain material ("astrodust") plus separate PAH nanoparticles.
Astrodust grains are 1.4:1 oblate spheroids with mass density
ρ = 2.74 g cm⁻³ and porosity P = 0.2. The dielectric function is from
Draine & Hensley (2021a, ApJ 910, 47).

Key physical differences from DL07:
- Single composite grain replaces separate silicate + graphite populations
- Laboratory-measured optical properties replace astronomical silicate/graphite proxies
- Silicate features at 9.7 and 18 μm differ in shape and relative strength
- Far-UV opacity slope differs
- MW reference q_PAH shifts from 4.6% (DL07) to 5.91%

The emission model uses the same Draine & Li (2007) framework for the
radiation field distribution: a fraction (1 − γ) of the dust mass is heated
by a single radiation field of intensity U_min, and a fraction γ is exposed
to a power-law distribution of radiation fields:

```
dM_dust/dU ∝ U^{-α}   for U_min ≤ U ≤ U_max
```

with α = 2 (fixed) and U_max = 10⁶ (fixed), exactly as in DL07. The
difference is entirely in the grain cross-sections and emission spectra
at each U.

### Parameters (same interface as DL07)

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| `dust_qpah` | q_PAH | 0–10% | PAH mass fraction (MW ref = 5.91%) |
| `dust_umin` | U_min | 0.1–25 | Minimum radiation field intensity |
| `dust_gamma_dl` | γ | 0–1 | Fraction from PDR (high-U) regions |

### Input/Output

- **Input:** wavelength array (Å), L_IR (erg/s), parameters above
- **Output:** L_ν(λ) in erg/s/Hz, energy-normalized to L_IR

### Key Equations

Total emitted power per H atom (Eq. 8 of DL07, applied with astrodust cross-sections):

```
j_ν = (1 − γ) j_ν(U_min) + γ ∫_{U_min}^{U_max} j_ν(U) (U^{-2}/C) dU
```

where C = ln(U_max/U_min) for α = 2. The per-grain emission j_ν(U) is
computed from the stochastic heating formalism using the astrodust
dielectric function.

Energy-balance normalization in tengri:

```
L_ν^{dust} = (η L_abs / ∫ j_ν dν) × j_ν
```

### Data Files

The Harvard Dataverse release provides FITS files with HDUs:
- **SIZE_DISTRIBUTION**: 167 effective radius bins
- **EXTINCTION**: C_ext/H as function of λ (1000 wavelengths)
- **EMISSION**: j_ν/n_H as function of λ for a grid of U values
- **POLARIZED_EMISSION**: Stokes Q, U emission (not needed for SED fitting)

For SED fitting, we need the emission templates integrated over the
size distribution for a grid of (q_PAH, U_min, γ) values.

### Implementation Plan

1. Download model output from Harvard Dataverse (`doi:10.7910/DVN/3B6E6S`)
   and PAH emission spectra from `doi:10.7910/DVN/LPUHIQ`.
2. Generate template grid: for each (q_PAH, U_min, γ) triplet, compute the
   total j_ν using the DL07 radiation field distribution formula with
   astrodust emission spectra. The grid should span:
   - q_PAH: [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 5.26, 5.91] %
   - U_min: [0.10, 0.15, 0.20, ..., 25.0] (same as DL07 grid)
   - γ: 0–1 enters analytically (same formula as DL07)
3. Store as `data/astrodust_templates.npz` with shape
   `(n_qpah, n_umin, n_lambda)` — γ enters as an analytic linear
   combination, not a grid axis.
4. Write `astrodust_emission()` in `models/dust/emission.py` following
   the `draine_li2007_emission()` pattern:
   - Load templates lazily (first-call caching)
   - Bilinear interpolation in (q_PAH, U_min) space
   - γ mixing is analytic: `j_ν = (1-γ) j_ν(U_min) + γ j_ν(PDR)`
   - Apply CMB correction (da Cunha+2013) identically to DL07
5. Register as `dust_emission="astrodust"` in ParamSpec.

### Reference Code

- **DustPedia/CIGALE:** CIGALE 2022+ includes an astrodust module
  (`pcigale/sed_modules/dl2014.py` is the DL07 successor; astrodust
  would follow the same pattern with different templates)
- **Draine's IDL code:** https://www.astro.princeton.edu/~draine/dust/
  provides IDL routines for computing emission from the astrodust model
- **DL07 in tengri:** `draine_li2007_emission()` — the new function
  should have identical calling convention

### Parity Tests

1. **Template self-consistency:** At q_PAH = 5.91%, U_min = 1.0, γ = 0,
   the template should match the Dataverse emission spectrum for U = 1 to
   < 1% across 3–1000 μm.
2. **DL07 comparison:** At matched parameters, the total IR luminosity
   should be identical (both are energy-normalized). The spectral shape
   should differ most at 8–25 μm (silicate features) and in the PAH
   band ratios.
3. **CIGALE cross-check:** Fit a DustPedia galaxy (e.g., NGC 628) with
   both DL07 and astrodust; compare χ² and recovered parameters.
4. **CMB correction:** At z = 8, U_min = 0.1, verify that CMB-corrected
   T_eff matches da Cunha+2013 Eq. 18.

---

## 2. THEMIS Dust Emission (Jones et al. 2017)

**Paper:** Jones et al. 2017, A&A 602, A46  
**arXiv:** 1411.6293  
**Code dependency:** DustEM (Fortran), or pre-tabulated grids from CIGALE  
**Status:** Appendix F.3.9 describes it; no code or templates exist.

### Physics

The Heterogeneous dust Evolution Model for Interstellar Solids (THEMIS)
uses two grain populations with physically motivated properties:

1. **Amorphous hydrocarbon a-C(:H) grains**: Optical properties depend on
   the hydrogen content (band gap E_g). In harsh UV fields, a-C(:H) grains
   lose hydrogen (photolysis), becoming more graphitic (lower E_g, higher
   absorption). This UV-processing is parameterized by the small grain mass
   fraction q_hac (analogous to q_PAH but physically distinct).

2. **Amorphous silicate grains (a-Sil)**: Fe-rich amorphous silicates with
   a-C mantles. The mantle thickness depends on the processing history.

The key innovation over DL07: dust properties evolve with radiation field
environment. The size distribution is physically motivated from grain
growth and destruction processes, not empirically calibrated.

### Parameters

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| `dust_qhac` | q_hac | 0–15% | Small hydrocarbon grain fraction |
| `dust_umin` | U_min | 0.1–50 | Minimum radiation field intensity |
| `dust_alpha_themis` | α | 1–4 | Power-law slope of dM/dU |

### Key Equations

The radiation field distribution follows Dale et al. (2001):

```
dM_dust(U) ∝ U^{-α} dU,   U_min ≤ U ≤ U_max
```

The emitted SED is:

```
L_ν = ∫_{U_min}^{U_max} j_ν(U, q_hac) U^{-α} dU / ∫_{U_min}^{U_max} U^{-α} dU
```

where j_ν(U, q_hac) is computed from the THEMIS grain model using DustEM.

### Implementation Plan

1. Obtain pre-tabulated templates from CIGALE's THEMIS module. CIGALE
   ships grids computed with DustEM for a range of (q_hac, U_min, α).
   The CIGALE source is `pcigale/sed_modules/themis.py`.
2. Convert to tengri format: `data/themis_templates.npz` with shape
   `(n_qhac, n_umin, n_alpha, n_lambda)`.
3. Write `themis_emission()` in `models/dust/emission.py`:
   - Trilinear interpolation in (q_hac, U_min, α) space
   - Same energy-balance normalization as all other dust emission models
   - Apply CMB correction (da Cunha+2013) at high z
4. Register as `dust_emission="themis"` in ParamSpec.

### Reference Code

- **CIGALE:** `pcigale/sed_modules/themis.py` has the template loading
  and interpolation. The templates are in
  `pcigale-data/themis/` as FITS tables.
- **DustEM:** Fortran code from https://www.ias.u-psud.fr/DUSTEM/ —
  can generate templates for arbitrary grain models.

### Parity Tests

1. **CIGALE comparison:** For a fixed (q_hac, U_min, α) triplet, the
   tengri template should match CIGALE's output to < 1%.
2. **Energy conservation:** Verify ∫ L_ν dν = η L_abs for all parameter
   combinations.
3. **Limiting cases:** At q_hac → 0 (no small grains), the SED should be
   dominated by the cold large-grain MBB component.

---

## 3. BOSA Dust Emission Templates (Boquien & Salim 2021)

**Paper:** Boquien & Salim 2021, A&A 653, A149  
**arXiv:** 2104.10721  
**Status:** Appendix F.3.10 describes it; no code or templates exist.

### Physics

Empirical templates adding sSFR as a second axis alongside L_TIR. Based on
2584 star-forming galaxies from the GAMA survey with Herschel FIR
photometry. Captures the observation that at fixed IR luminosity, galaxies
with higher sSFR have:
- Warmer effective dust temperatures
- Stronger MIR PAH features relative to FIR continuum
- Broader IR SED peaks

### Parameters

| Parameter | Range | Meaning |
|-----------|-------|---------|
| (none free) | — | L_TIR set by energy balance, sSFR from SFH |

This is a **zero-free-parameter** model: the template is selected entirely
by the predicted L_TIR and sSFR, both derived from other model components.

### Key Equations

The template grid is indexed by:
```
log(L_TIR / L_⊙) ∈ [8.5, 12.5]
log(sSFR / yr⁻¹) ∈ [-12.0, -8.5]
```

At runtime:
```
sSFR = SFR_100Myr / M_stellar
L_TIR = η × L_abs
```

The template SED shape f_ν(λ | log L_TIR, log sSFR) is obtained by
bilinear interpolation, then scaled:

```
L_ν^{dust}(λ) = L_TIR × f_ν(λ) / ∫ f_ν dν
```

### Implementation Plan

1. Obtain template grid from Boquien & Salim data release or from CIGALE
   (which includes these templates as `pcigale/sed_modules/dl2014.py`
   with sSFR-dependent extension).
2. Grid axes: `(n_logLTIR, n_logsSFR, n_lambda)` — 2D interpolation.
3. At runtime: compute sSFR = SFR (averaged over last 100 Myr) / M_stellar
   from the SFH weights. Both quantities are already available in the
   tengri forward model.
4. Write `bosa_emission()` in `models/dust/emission.py`:
   - Bilinear interpolation in (log L_TIR, log sSFR)
   - Energy-balance normalization
   - CMB correction at high z
5. Register as `dust_emission="bosa"` in ParamSpec.

### Parity Tests

1. **Self-consistency:** For an input (L_TIR, sSFR) pair that matches a
   grid node, the output should match the template exactly.
2. **CIGALE comparison:** Fit a GAMA galaxy with both Dale+2014 and BOSA;
   the BOSA model should produce smaller residuals in MIR bands.
3. **Zero-parameter test:** Verify that the model correctly produces no
   additional free parameters in the ParamSpec.

---

## 4. MAGPHYS Dust Emission (da Cunha et al. 2008)

**Paper:** da Cunha, Charlot & Elbaz 2008, MNRAS 388, 1595  
**arXiv:** 0806.1020  
**Status:** Appendix F.3.7 describes equations; no code exists.

### Physics

Four-component model decomposing the IR SED into:
1. **PAH features** — Drude profiles following Smith+2007 (PAHFIT)
2. **Hot MIR continuum** — T ~ 130–250 K modified blackbody, β = 1.5
3. **Warm grains** — T_W ~ 30–60 K, β_W = 1.5 (birth cloud) or 2.0 (ISM)
4. **Cold grains** — T_C ~ 15–25 K, β_C = 2.0

### Parameters

| Parameter | Symbol | Range | Default | Meaning |
|-----------|--------|-------|---------|---------|
| `dust_xi_pah` | ξ_PAH | 0–0.5 | 0.10 | Energy fraction in PAH features |
| `dust_xi_mir` | ξ_MIR | 0–0.3 | 0.05 | Energy fraction in hot MIR continuum |
| `dust_xi_w` | ξ_W | 0–0.5 | 0.25 | Energy fraction in warm grains |
| `dust_T_w` | T_W | 30–60 K | 45 | Warm grain temperature |
| `dust_T_c` | T_C | 15–25 K | 20 | Cold grain temperature |

Constraint: ξ_C = 1 − ξ_PAH − ξ_MIR − ξ_W (not free).

### Key Equations

Each modified blackbody component:

```
L_ν^{(i)} = ξ_i × L_abs × (ν/ν₀)^{β_i} B_ν(T_i) / ∫ (ν/ν₀)^{β_i} B_ν(T_i) dν
```

where ν₀ corresponds to λ₀ = 100 μm and B_ν is the Planck function.

PAH features using Drude profiles from Smith+2007 Table 2:

```
I_ν^{PAH} = ξ_PAH × L_abs × Σ_k [ b_k γ_k² / ((λ/λ_k − λ_k/λ)² + γ_k²) ] / N
```

where the sum runs over ~17 Drude components and N is the normalization
integral. The Drude profile function is (Smith+2007 Eq. 2; PAHFIT IDL):

```
D(λ; λ₀, γ) = γ² / ((λ/λ₀ − λ₀/λ)² + γ²)
```

where γ is the fractional FWHM. The 17 PAH features and their
central wavelengths λ_k and fractional FWHMs γ_k from Smith+2007 Table 2:

| Feature | λ_k (μm) | γ_k | Blend |
|---------|-----------|-----|-------|
| 1 | 5.27 | 0.034 | — |
| 2 | 5.70 | 0.035 | — |
| 3 | 6.22 | 0.030 | 6.2 complex |
| 4 | 6.69 | 0.070 | — |
| 5 | 7.42 | 0.126 | 7.7 complex |
| 6 | 7.60 | 0.044 | 7.7 complex |
| 7 | 7.85 | 0.053 | 7.7 complex |
| 8 | 8.33 | 0.050 | — |
| 9 | 8.61 | 0.039 | 8.6 feature |
| 10 | 10.68 | 0.020 | — |
| 11 | 11.23 | 0.012 | 11.3 complex |
| 12 | 11.33 | 0.032 | 11.3 complex |
| 13 | 11.99 | 0.045 | — |
| 14 | 12.62 | 0.042 | 12.7 feature |
| 15 | 12.69 | 0.013 | 12.7 feature |
| 16 | 13.48 | 0.040 | — |
| 17 | 14.04 | 0.016 | — |

The relative amplitudes b_k (peak intensities) come from the median
SINGS galaxy fits in Smith+2007. They should be hardcoded as defaults
but can optionally be scaled by f_neutral (neutral/ionized PAH ratio)
for the 6.2, 7.7, and 8.6 μm features.

Hot MIR continuum (T_MIR ≈ 130–250 K, fixed in MAGPHYS):

```
L_ν^{MIR} = ξ_MIR × L_abs × ν^{β_MIR} B_ν(T_MIR) / ∫ ν^{β_MIR} B_ν(T_MIR) dν
```

with β_MIR = 1.0 and T_MIR = 250 K (da Cunha+2008 Table 1).

### Birth cloud vs ISM dust

In the two-component Charlot & Fall framework:
- Birth cloud dust: β_W = 1.5 for warm component
- ISM dust: β_C = 2.0 for cold component

The energy absorbed by each component (L_abs,BC and L_abs,ISM) is
already computed by the dust attenuation module. The MAGPHYS emission
model should be called twice with the appropriate L_abs and β values.

### Implementation Plan

1. Implement as a **pure analytic function** — no templates needed.
2. PAH Drude profiles: hardcode wavelengths and fractional FWHMs from
   Smith+2007 Table 2 (above). Relative amplitudes from SINGS median.
3. MBB components: straightforward Planck function × ν^β evaluation.
4. All four components are summed and normalized to L_abs.
5. Write `magphys_emission()` in `models/dust/emission.py`.
6. Register as `dust_emission="magphys"` in ParamSpec.

### Differentiability Notes

All components are smooth functions of the parameters — Planck function,
power laws, and Drude profiles are all analytically differentiable.
The normalization integrals can be precomputed on a fixed wavelength grid
via trapezoidal quadrature (differentiable through JAX).

### Reference Code

- **MAGPHYS:** da Cunha's Fortran code (http://www.iap.fr/magphys/)
  contains the exact component definitions.
- **CIGALE:** `pcigale/sed_modules/dale2014.py` and `casey2012.py` for
  comparison (different models but same energy-balance framework).
- **PAHFIT Python:** https://github.com/PAHFIT/pahfit — the Python implementation
  uses `astropy.modeling` with the same Drude profiles.
- **PAHFIT Classic IDL:** https://github.com/PAHFIT/pahfit_classic/blob/main/pahfit.pro
  — `pahfit_drude` function gives the exact profile:
  `central_inten * frac_fwhm^2 / ((lambda/lam_0 - lam_0/lambda)^2 + frac_fwhm^2)`

### Parity Tests

1. **MAGPHYS comparison:** For the default da Cunha+2008 parameter set
   (Table 1), the total IR SED should match MAGPHYS output to < 5%.
2. **PAH profile test:** Compare individual Drude profiles against
   PAHFIT IDL output at several wavelength grids.
3. **Energy conservation:** Verify Σ ξ_i = 1 and ∫ L_ν dν = L_abs for
   all parameter combinations.
4. **Limiting cases:** At ξ_PAH = 0, ξ_MIR = 0, the SED should be a
   two-temperature MBB.

---

## 5. TEA Dust Attenuation (Haskell et al. 2024)

**Paper:** Haskell et al. 2024, arXiv:2401.11007  
**Status:** Code EXISTS at `models/dust/attenuation.py` as `tea()`. Already registered.

**No action needed.** Remove the roadmap stub.

---

## 6. Chemical Evolution Z(t) — Full Bellstedt+2021 Model

**Paper:** Bellstedt et al. 2020, MNRAS 498, 5581; Bellstedt et al. 2021, MNRAS 503, 3309  
**arXiv:** 2005.11917 (2020); 2102.11514 (2021)  
**Reference code:** ProSpect R package, `Zfunc_massmap_box`  
**Status:** Partially implemented. Needs Bellstedt+2021 yield solver.

### What EXISTS

- `compute_log_z_evolving()` produces Z(t) from SFH (linear ramp)
- `dsps_met_table` CSP mode supports per-age metallicity tables
- The linear ramp mode works

### What's NOT Implemented

The full Bellstedt+2021 self-consistent closed-box/leaky-box model where:
1. The nucleosynthetic yield y is solved self-consistently to match
   the present-day gas metallicity Z_gas,now
2. The mass-loading factor η is a free parameter (leaky-box)
3. Z(t) is derived from the cumulative SFH, not parameterized

### Key Equations

**Closed box** (Bellstedt+2020, following Tinsley 1980):

Gas mass fraction at time t:
```
μ(t) = M_gas(t) / M_total = 1 − M_★,formed(t) / M_total
```

where M_★,formed(t) = ∫₀ᵗ SFR(t′) dt′ is the cumulative stellar mass
formed (before mass loss). M_total is the initial total baryonic mass,
which includes the gas reservoir.

Gas-phase metallicity:
```
Z(t) = −y × ln(μ(t))
```

where y is the nucleosynthetic yield — the mass of metals produced per
unit mass of stars formed (net of recycling). This assumes instantaneous
recycling: metals from short-lived massive stars are returned to the ISM
immediately.

**Yield solver:** The yield y is not a free parameter. Instead, the
present-day gas metallicity Z_gas,now is the free parameter, and y is
solved by requiring:

```
Z(t = t_now) = Z_gas,now
−y ln(μ(t_now)) = Z_gas,now
y = −Z_gas,now / ln(μ(t_now))
```

This requires knowing μ(t_now), i.e., what fraction of the initial gas
remains. In ProSpect, M_total is chosen so that the gas fraction at the
present day is physically reasonable (μ ∈ [0.01, 0.99]).

**Leaky box** (Bellstedt+2021):

Outflows carry gas (and metals) out of the galaxy at a rate η × SFR,
where η is the mass-loading factor:

```
dM_gas/dt = −(1 + η − R) × SFR(t)
```

where R is the return fraction (mass recycled back to ISM by stellar winds
and supernovae, typically R ≈ 0.4 for Chabrier IMF). The effective yield
becomes:

```
y_eff = y / (1 + η − R)
```

and the metallicity evolution is:

```
Z(t) = −y_eff × ln(μ(t))
```

The free parameters are Z_gas,now and η; the yield is solved as before.

### Parameters

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| `met_z_gas_now` | Z_gas,now | [-2.5, 0.5] dex | Present-day gas metallicity (log Z/Z_⊙) |
| `met_eta_outflow` | η | [0, 10] | Mass-loading factor (0 = closed box) |

### Implementation Plan

1. Implement `chemical_evolution_closed_box(sfh_weights, ssp_ages, z_gas_now)`:
   ```python
   # Compute cumulative formed mass (ascending lookback time = descending cosmic time)
   M_formed_cumulative = jnp.cumsum(sfh_weights * dt)  # in M_⊙
   M_total = M_formed_cumulative[-1] / (1 - mu_final)  # solve for total mass
   mu = 1.0 - M_formed_cumulative / M_total
   
   # Yield from boundary condition
   y = -Z_gas_now_linear / jnp.log(jnp.clip(mu[-1], 1e-10, 1.0))
   
   # Z(t) at each age
   Z_t = -y * jnp.log(jnp.clip(mu, 1e-10, 1.0))
   log_Z_t = jnp.log10(jnp.clip(Z_t / Z_sun, 1e-10, None))
   ```

2. Implement `chemical_evolution_leaky_box(sfh_weights, ssp_ages, z_gas_now, eta)`:
   - Replace y with y_eff = y / (1 + η)
   - Gas fraction evolves faster: μ(t) = exp(−(1+η) × M_formed/M_total/y_eff)

3. Wire into ParamSpec as `met_evolution="closed_box"` or `"leaky_box"`.
4. Feed Z(t) array to `dsps_met_table` mode (already supports per-age Z).

### Differentiability Notes

The closed-box model is fully differentiable: cumsum, log, and clip are
all JAX-compatible. The yield solver is a simple division.

The one subtlety is the choice of M_total (or equivalently μ_final).
ProSpect uses μ_final as an implicit parameter set by the gas fraction at
z = 0. For tengri, a pragmatic choice is to set μ_final = 0.1 (10% gas
remaining), which gives y values in the physically reasonable range
y ∈ [0.01, 0.05]. Alternatively, μ_final can be a derived quantity from
a gas consumption model.

### Reference Code

- **ProSpect:** `Zfunc_massmap_box()` in the R package — implements the
  closed-box model with yield solver.
- **Bellstedt+2020 Appendix A:** Explicit derivation of the yield solver.

### Parity Tests

1. **Constant SFR test:** For constant SFR, the analytical solution is
   Z(t) = −y ln(1 − t × SFR / M_total). Verify numerically.
2. **ProSpect comparison:** For the GAMA median SFH from Bellstedt+2020,
   compare Z(t) from tengri against ProSpect output.
3. **Closed → leaky transition:** At η = 0, the leaky-box should reduce
   exactly to the closed-box model.
4. **Present-day metallicity:** Verify that Z(t_now) = Z_gas,now for all
   parameter combinations (boundary condition).

---

## 7. PAH Feature Decomposition (PAHFIT-style forward model)

**Paper:** Smith et al. 2007, ApJ 656, 770; Python implementation: https://pahfit.readthedocs.io/  
**Status:** Not as a standalone module. PAH profiles are needed for MAGPHYS emission (§4).

### Implementation

This is NOT a separate model but a utility used by MAGPHYS emission and
potentially by a future JWST MIRI spectroscopic fitting module. The
Drude profiles from Smith+2007 Table 2 should be implemented as a
reusable function:

```python
def drude_profile(wavelength_um, lambda_0, gamma_fwhm):
    """Smith+2007 / PAHFIT Drude profile.
    
    Parameters
    ----------
    wavelength_um : array, wavelength in microns
    lambda_0 : float, central wavelength in microns
    gamma_fwhm : float, fractional FWHM (dimensionless)
    
    Returns
    -------
    D : array, Drude profile (normalized to peak = 1)
    """
    x = wavelength_um / lambda_0 - lambda_0 / wavelength_um
    return gamma_fwhm**2 / (x**2 + gamma_fwhm**2)
```

Place in `models/dust/drude_profiles.py` with the Smith+2007 feature
table as a module-level constant.

---

## 8. Patchy IGM Transmission

**Papers:** Miralda-Escudé 1998, ApJ 501, 15; Mason et al. 2018, ApJ 856, 2;
Mesinger & Furlanetto 2007, ApJ 669, 663  
**Status:** Not implemented. Appendix J.0.2 describes the model.

### Physics

At z ≳ 6, the IGM is not uniformly ionized. Galaxies reside in ionized
bubbles of comoving radius R_b embedded in a neutral IGM with
volume-averaged neutral fraction x̄_HI. The Gunn-Peterson optical depth
for a fully neutral IGM is enormous:

```
τ_GP ≈ 7.16 × 10⁵ × [(1 + z_s) / 10]^{3/2}
```

The damping wing extends redward of Lyα, producing a characteristic
absorption profile that depends on x̄_HI and R_b.

### Key Equations

Following Miralda-Escudé (1998), the damping wing optical depth at
observed wavelength λ_obs (redward of Lyα at the source redshift z_s) is:

```
τ_DW(λ_obs) = x̄_HI × (τ_GP / π) × ∫_{z_bubble}^{∞} σ_α(ν(1+z)) / σ_0 × (1+z)² / H(z) dz
```

where z_bubble is the redshift corresponding to the edge of the ionized
bubble at comoving distance R_b from the source, and σ_α is the Lyα
cross-section in the damping wing (Lorentzian far-wing):

```
σ_α(Δν) = σ_0 × (Γ_Lyα / 4π) / [(Δν)² + (Γ_Lyα / 4π)²]
≈ σ_0 × (Γ_Lyα / 4π) / (Δν)²   for |Δν| ≫ Γ_Lyα
```

with σ_0 = 5.9 × 10⁻¹⁴ cm², Γ_Lyα = 6.265 × 10⁸ s⁻¹.

For computational convenience, the integral can be evaluated analytically
in the approximation of a uniform neutral IGM outside the bubble. The
result (Miralda-Escudé 1998 Eq. 16) gives τ_DW as a function of
the velocity offset from Lyα:

```
τ_DW(Δv) ≈ x̄_HI × R_α × n̄_H(z_s) × (c / H(z_s)) × I(Δv, R_b)
```

where R_α = πe²f_α/(m_e c) is the Lyα cross-section coefficient,
and I(Δv, R_b) is an integral that depends on the bubble geometry.

### Simplified two-parameter model (for SED fitting)

For broadband photometric fitting, the full integral is overkill. A
simplified model modifies the Inoue+2014 mean transmission:

```
T_patchy(λ, z_s) = T_Inoue(λ, z_s) × exp(−τ_DW(λ, x̄_HI, R_b))
```

where τ_DW is evaluated at the velocity offset:

```
Δv = c × (λ_obs / λ_Lyα − (1 + z_s)) / (1 + z_s)
```

Only photons redward of Lyα at the source are affected by the damping
wing; blueward photons are already absorbed by Gunn-Peterson.

### Parameters

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| `igm_x_HI` | x̄_HI | 0–1 | Volume-averaged neutral fraction |
| `igm_bubble_Mpc` | R_b | 0.1–100 | Ionized bubble radius (comoving Mpc) |

### Implementation Plan

1. Implement damping wing optical depth as a JAX function:
   ```python
   def tau_damping_wing(wavelength_obs, z_source, x_HI, R_bubble_cMpc):
       """Damping wing optical depth from Miralda-Escudé 1998."""
       # Only active redward of Lyα at source
       lambda_lya_obs = 1215.67 * (1 + z_source)
       delta_lambda = wavelength_obs - lambda_lya_obs
       # Convert to frequency offset
       delta_nu = c / wavelength_obs - c / lambda_lya_obs
       # Compute optical depth using Lorentzian far-wing
       ...
   ```

2. Apply multiplicatively to the Inoue+2014 transmission:
   ```python
   T_total = T_inoue * jnp.exp(-tau_damping_wing(...))
   ```

3. Only activate at z > 5.5 (below that, mean IGM is sufficient).
4. Wire as `igm_model="patchy"` with `igm_x_HI` and `igm_bubble_Mpc`
   as free parameters.

### Reference Code

- **Mason+2018:** Bayesian inference framework for x̄_HI from Lyα
  visibility statistics. The damping wing model is in their appendix.
- **21cmFAST:** Semi-numerical reionization simulations that produce
  ionization maps; the `py21cmfast` Python wrapper can generate
  sightline-averaged damping wing profiles.
- **Mesinger & Furlanetto 2007:** Analytic framework for patchy damping
  wings with stochastic bubble distributions.

### Parity Tests

1. **Uniform IGM limit:** At x̄_HI = 1, R_b = 0, the damping wing should
   match the Miralda-Escudé (1998) analytic formula for a fully neutral IGM.
2. **Fully ionized limit:** At x̄_HI = 0, the transmission should be
   exactly T_Inoue (no modification).
3. **Photometric impact:** At z = 7, x̄_HI = 0.5, verify that the Lyα
   break is deeper and redder than the Inoue+2014 prediction, consistent
   with JWST observations.
4. **Mason+2018 comparison:** Reproduce their Fig. 3 (damping wing
   transmission vs. x̄_HI for R_b = 1 pMpc).

---

## 9. ADAF Disc Model (Mahadevan 1997) — Full Rewrite

**Paper:** Mahadevan 1997, ApJ 477, 585; Nemmen et al. 2014, MNRAS 438, 2804  
**Status:** Code EXISTS at `models/agn/adaf.py` but has WRONG EQUATIONS.
Needs full rewrite.

### Physics

For low-luminosity AGN (L/L_Edd < 0.01), the inner accretion disc
transitions to an advection-dominated accretion flow (ADAF). In the ADAF:
- Ions are virial-temperature hot: T_i ≈ 10¹² / r K (r in R_g)
- Electrons are sub-virial: T_e ≈ 10⁹–10¹⁰ K (set by Coulomb coupling)
- Most gravitational energy is advected into the BH (f_adv ~ 1)
- Radiative efficiency η_rad ≪ 0.1 (vs. ~0.1 for thin disc)

### Key Equations (Mahadevan 1997)

**Self-similar solution** (Narayan & Yi 1994, 1995b):

Electron temperature from energy balance (Mahadevan 1997, §5.1):
```
q_ie = q_synch + q_brem + q_IC
```
where q_ie is the Coulomb energy transfer rate from ions to electrons,
and the three cooling terms are synchrotron, bremsstrahlung, and
inverse Compton respectively. The electron temperature T_e is found by
iterating to balance this equation.

Approximate scaling (Mahadevan 1997 Eq. 47):
```
T_e ≈ (6.66 × 10⁹ / f(α)) × [1.36 + 0.076 ln(m_dot / α²)]⁻¹   [K]
```
where f(α) is a weak function of the viscosity parameter.

**Synchrotron emission** (Mahadevan 1997, Eqs. 19–25):

Peak frequency:
```
ν_peak = (3/2) × (eB / 2πm_e c) × θ_e² × x_M
```
where θ_e = kT_e / m_e c², B is the magnetic field strength, and x_M
is the critical frequency parameter. The magnetic field follows from
equipartition: B² / 8π = β_adaf × (total pressure).

Synchrotron luminosity:
```
L_synch ≈ (4/3) σ_T c (B²/8π) n_e V θ_e² × F(x_M)
```
where V is the emitting volume and F is a tabulated function (Mahadevan
1997 Eq. 22; approximated analytically in Eq. 25).

Self-absorption frequency ν_sa (Mahadevan 1997 Eq. 24):
```
ν_sa ∝ ṁ^{2/5} θ_e^{-12/5} r_min^{-6/5} B^{1/5}
```
Below ν_sa, L_ν ∝ ν^{5/2} (Rayleigh-Jeans). Above, L_ν ∝ ν^{1/3} exp(−ν/ν_c).

**Bremsstrahlung** (Mahadevan 1997, Eq. 30):
```
L_brem = 1.4 × 10⁻²⁷ T_e^{1/2} n_e² V g_ff   [erg/s/Hz]
```
where g_ff is the Gaunt factor. For relativistic electrons (kT_e > m_e c²),
use the relativistic correction factor:
```
g_ff ≈ (4/π) [2θ_e / (π)]^{1/2} [1 + 1.78 θ_e^{1.34}]
```
(Svensson 1984). **The current code is missing this Gaunt factor.**

**Inverse Compton** (Mahadevan 1997, Eqs. 33–34):

Compton enhancement of synchrotron:
```
L_IC = η_3 × L_synch × (A − 1) × min(1, τ_es)
```
where A = 1 + 4θ_e + 16θ_e² is the mean amplification factor per
scattering, τ_es = n_e σ_T R is the electron scattering optical depth,
and η_3 accounts for multiple scatterings:
```
η_3 ≈ (3kT_e / m_e c²) for τ_es ≪ 1 (single scattering)
```

The Compton spectral index:
```
α_C = −ln(τ_es) / ln(A)
```
L_ν ∝ ν^{−α_C} for ν_sa < ν < ν_cut.

### Parameters

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| `agn_log_mdot` | log(ṁ) | [−4, −1] | log₁₀(Ṁ/Ṁ_Edd) |
| `agn_log_mbh` | log(M_BH) | [6, 10] | log₁₀(M_BH/M_⊙) |
| `agn_a_spin` | a_★ | [0, 0.998] | BH spin parameter |
| `agn_delta` | δ | [0.01, 0.5] | Fraction of viscous energy → electrons |
| `agn_beta_adaf` | β | [0.5, 0.95] | Gas/total pressure ratio |
| `agn_r_transition` | r_tr | [10, 10⁴] | ADAF → thin disc transition radius (R_g) |

### Implementation Plan

1. **Read Mahadevan 1997 carefully.** The current code errors are:
   - Electron temperature calculation is wrong (§5.1)
   - Synchrotron self-absorption not properly implemented
   - Missing bremsstrahlung Gaunt factor (relativistic correction)
   - Compton spectral index formula error

2. Implement the three emission components as separate functions:
   ```python
   def adaf_synchrotron(nu, Te, ne, B, R_min, R_max): ...
   def adaf_bremsstrahlung(nu, Te, ne, R_min, R_max): ...
   def adaf_inverse_compton(nu, Te, ne, B, tau_es, L_synch): ...
   ```

3. Self-consistent electron temperature solver:
   - Use the approximate formula (Mahadevan 1997 Eq. 47) as initial guess
   - Iterate energy balance: q_ie = q_synch + q_brem + q_IC
   - In JAX: use `jax.lax.while_loop` or fixed-point iteration

4. Transition radius from ADAF to thin disc at r_tr:
   - For R < r_tr: ADAF emission
   - For R > r_tr: standard Shakura-Sunyaev (Kubota-Done disc for R > r_tr)
   - Total SED = ADAF + thin_disc + torus

5. Test against Nemmen+2014 models for Sgr A* and M87.

### Known Bugs in Current Code (from KNOWN_BUGS.md)

1. Electron temperature: uses Narayan & Yi 1995 approximate scaling
   instead of the self-consistent solution from Mahadevan 1997 §5.1
2. Synchrotron: missing self-absorption below ν_sa
3. Bremsstrahlung: uses non-relativistic Gaunt factor; should use
   Svensson (1984) relativistic correction
4. Missing Compton spectral slope computation

### Reference Code

- **Nemmen+2014:** Python implementation (https://github.com/rsnemmen/adaf)
  — reference for Sgr A* and M87 fits
- **Yuan & Narayan 2014 review:** Comprehensive review of hot accretion
  flows with self-consistent equations
- **RELAGN/qsosed:** The thin-disc portion (R > r_tr) can use the existing
  K&D disc implementation

### Parity Tests

1. **Sgr A* SED:** At M_BH = 4 × 10⁶ M_⊙, ṁ = 10⁻⁵, reproduce the
   radio-to-X-ray SED from Nemmen+2014 Fig. 1.
2. **M87 SED:** At M_BH = 6.5 × 10⁹ M_⊙, ṁ = 10⁻³, reproduce the
   submm peak and X-ray slope.
3. **Thin-disc limit:** As r_tr → r_ISCO, the ADAF contribution should
   vanish and the SED should approach the K&D disc model.
4. **Radio slope:** Below ν_sa, verify L_ν ∝ ν^{5/2} (synchrotron
   self-absorption).

---

## 10. MAPPINGS V Full Shock Grids

**Paper:** Allen et al. 2008, ApJS 178, 20; Alarie & Morisset 2019, RMxAA 55, 377  
**3MdB database:** http://3mdb.astro.unam.mx/  
**Status:** Basic `shock_emission()` exists using a simplified model.
Full MAPPINGS V grid interpolation not implemented.

### Physics

Fast radiative shocks (v_s = 100–1000 km/s) in the ISM produce emission
from shock-heated and photoionized gas. MAPPINGS V computes self-consistent
shock + precursor models where:
- **Shock:** Gas is heated, compressed, and cools radiatively
- **Precursor:** Upstream gas is photoionized by the shock's UV radiation

### Grid Axes

| Axis | Symbol | Range | N_pts | Meaning |
|------|--------|-------|-------|---------|
| Shock velocity | v_s | 100–1000 km/s | ~20 | Sets shock temperature |
| Pre-shock density | n_pre | 0.01–1000 cm⁻³ | ~10 | Gas density |
| Magnetic parameter | B/√n | 0.5–10 μG cm^{3/2} | ~5 | Sets compression ratio |
| Gas metallicity | Z_gas | 0.001–2 Z_⊙ | ~7 | Sets cooling function |
| Model component | — | shock/precursor/total | 3 | Which emission |

The grid produces ~140 emission line luminosities + continuum spectrum
for each parameter combination.

### Implementation Plan

1. Download MAPPINGS V grid data from the 3MdBs database
   (http://3mdb.astro.unam.mx/) using the Allen+2008 or Alarie &
   Morisset 2019 model sets.
2. Store as `data/mappings_shock_v.npz` with shape
   `(n_vs, n_npre, n_Bn, n_Z, n_component, n_lines)` for line
   luminosities and `(..., n_lambda)` for continuum.
3. Implement 4D multilinear interpolation (same pattern as SKIRTOR torus
   grid): v_s and Z_gas are interpolated continuously; B/√n, n_pre, and
   component are snapped to nearest grid point.
4. Write `mappings_shock_emission()` in `models/nebular/shock.py`:
   - Input: v_s, n_pre, B/√n, Z_gas, L_Hα_shock
   - Output: emission line luminosities + continuum
   - Scale by shock fraction of total Hα
5. Replace simplified `shock_emission()` with the full grid version.

### Reference Code

- **CIGALE:** Has a simple shock module that uses Allen+2008 grids
- **3MdB Python interface:** http://3mdb.astro.unam.mx/help/python/
- **Current tengri:** `shock_emission()` in `emission_helpers.py`

### Parity Tests

1. **Grid node test:** At exact grid points, verify line ratios match
   the 3MdBs database entries to machine precision.
2. **BPT ratios:** For v_s = 300 km/s, solar metallicity, verify that
   [NII]/Hα and [OIII]/Hβ ratios fall in the shock region of the BPT
   diagram (above the Kewley+2001 maximum starburst line).
3. **Velocity scaling:** Line luminosities should scale roughly as
   v_s³ (kinetic energy flux) for the total model.

---

## 11. ADDITIONAL MODELS NOT IN THE PAPER

The following models are not described in the current Paper I appendix but
would strengthen the framework. These are ordered by scientific impact and
implementation difficulty.

---

### 11.1. Flexible Dust-to-Stellar Mass Ratio (Rémy-Ruyer+2014)

**Paper:** Rémy-Ruyer et al. 2014, A&A 563, A31  
**Motivation:** The dust-to-gas ratio (and hence dust-to-stellar mass ratio)
varies with metallicity. At Z < 0.1 Z_⊙, the dust-to-gas ratio drops
more steeply than linearly, affecting IR SED predictions for low-metallicity
and high-z galaxies.

**Implementation:** Add a metallicity-dependent dust-to-gas ratio that
modifies the energy balance parameter η:
```
η_eff(Z) = η × (Z/Z_⊙)^α_DG
```
where α_DG ≈ 1.0 for Z > 0.1 Z_⊙ and α_DG ≈ 2.0–3.0 below (broken
power law from Rémy-Ruyer+2014 Table 1). This is a single additional
parameter.

**Priority:** Medium. Relevant for JWST high-z galaxy fitting.

---

### 11.2. Redshift-Dependent IMF (van Dokkum+2008; Sneppen+2022)

**Paper:** van Dokkum & Conroy 2010; Sneppen et al. 2022, ApJ 931, 57  
**Motivation:** There is growing evidence that the IMF may evolve with
redshift or with galaxy properties (velocity dispersion, metallicity).
A bottom-heavy IMF in massive ellipticals and a potentially top-heavy
IMF at high-z would affect M/L ratios and SFR calibrations.

**Implementation:** Add an IMF slope parameter that interpolates between
SSP libraries computed with different IMFs:
```
L_SSP(IMF_slope) = (1 − f) × L_SSP(Chabrier) + f × L_SSP(Salpeter)
```
This requires loading two SSP libraries simultaneously. The IMF parameter
could also be a per-galaxy latent variable in hierarchical fits.

**Priority:** Low for Paper I; important for massive galaxy archeology.

---

### 11.3. Non-Equilibrium Dust Temperature Distribution (Galliano+2011)

**Paper:** Galliano et al. 2011, A&A 536, A88  
**Motivation:** Small grains (a < 0.01 μm) undergo stochastic heating:
single UV photon absorption heats the grain to T >> T_eq, followed by
rapid cooling. This produces MIR emission that is not captured by
equilibrium MBB models. The DL07 and Astrodust models handle this
implicitly through their grain-size-dependent heating formalism, but the
MAGPHYS and MBB models do not.

**Implementation:** Add a stochastic heating correction factor to the
MBB/Casey models:
- Compute the equilibrium temperature T_eq(U, a) for each grain size
- Use the Desert et al. (1990) formalism for the temperature probability
  distribution P(T|a, U) of small grains
- Integrate over the size distribution

**Priority:** Low for broadband photometry; high for MIR spectroscopy
(JWST MIRI). The template-based models (DL07, Astrodust, THEMIS) already
handle this.

---

### 11.4. Lyman Continuum Escape Fraction Model

**Paper:** Chisholm et al. 2022, ApJ 931, 37 (LzLCS); Naidu et al. 2020  
**Motivation:** The ionizing photon escape fraction f_esc is a critical
parameter for reionization studies and is currently a simple fixed
parameter in tengri. An empirical model relating f_esc to galaxy
properties (UV slope β, stellar mass, sSFR) would enable physically
motivated priors.

**Implementation:** Add a parametric f_esc model:
```
f_esc = f_esc,0 × 10^{a1 × (β + 2)} × 10^{a2 × log(sSFR/sSFR_0)}
```
calibrated on the Low-z Lyman Continuum Survey (Chisholm+2022).
Free parameters: f_esc,0, a1, a2 (or fix to empirical calibration).

**Priority:** High for JWST reionization science. Connects to the
patchy IGM model (§8).

---

### 11.5. Dust Grain Growth/Destruction Model (Asano+2013)

**Paper:** Asano et al. 2013, MNRAS 432, 637  
**Motivation:** At high redshift (z > 4), dust masses inferred from
FIR observations are often too large to be explained by stellar sources
alone — grain growth in the ISM is required. A model that tracks dust
mass evolution self-consistently with the SFH would provide physically
motivated dust-to-gas ratios.

**Implementation:** Coupled ODEs for dust mass, gas mass, and metallicity:
```
dM_dust/dt = Y_dust × SFR − M_dust/(τ_dest × M_gas) × SFR + M_dust/τ_grow
```
where Y_dust is the dust yield, τ_dest is the destruction timescale
(from SN shocks), and τ_grow is the grain growth timescale (proportional
to Z × n_gas). This is a low-dimensional ODE that can be integrated
alongside the SFH using `jax.lax.scan`.

**Priority:** Medium-high for JWST high-z dust studies.

---

### 11.6. Variable Attenuation Curve via Radiative Transfer (Narayanan+2018)

**Paper:** Narayanan et al. 2018, ApJ 869, 70  
**arXiv:** 1705.05858  
**Motivation:** The `narayanan_z` curve in Table 5 applies a
redshift-dependent effective attenuation law calibrated on SIMBA
cosmological simulations with POWDERDAY dust radiative transfer. A more
physical extension would make the attenuation curve shape depend on
galaxy properties (M_★, sSFR, metallicity, inclination) rather than
just redshift.

**Implementation:** The Narayanan+2018 functional form:
```
k(λ) = (A_λ/A_V)(τ_V, Z, sSFR, ...)
```
parameterized as a neural network or lookup table trained on RT
simulations. This is a drop-in replacement for any attenuation curve.

**Priority:** Medium. The existing 14 attenuation curves cover most use
cases; this would add physics-driven curve selection.

---

## Priority Order for Implementation

| Rank | Model | Effort | Impact | Notes |
|------|-------|--------|--------|-------|
| 1 | Chemical evolution Z(t) | Small | High | Partially done; yield solver is ~50 LOC |
| 2 | ADAF rewrite | Large | High | Code exists but wrong; blocks LLAGN science |
| 3 | Astrodust+PAH | Medium | High | Drop-in DL07 replacement; needs template generation |
| 4 | MAGPHYS dust emission | Medium | Medium | Analytic; no templates needed |
| 5 | Patchy IGM | Small | High | Critical for JWST z>6; ~100 LOC |
| 6 | THEMIS dust | Medium | Medium | Needs templates from DustEM/CIGALE |
| 7 | BOSA templates | Small | Medium | Zero free parameters; attractive for surveys |
| 8 | MAPPINGS V full grids | Large | Medium | Needs grid data; complex interpolation |
| 9 | f_esc model (§11.4) | Small | High | New; not in paper. Connects to patchy IGM |
| 10 | Dust grain growth (§11.5) | Medium | Medium | New; not in paper. High-z dust |
| 11 | PAH decomposition | Small | Low | Utility function for MAGPHYS; ~30 LOC |
| 12 | ~~TEA attenuation~~ | — | — | Already implemented; remove roadmap stub |

---

## Appendix A: Cross-Code Reference — Where Each Model Lives

This appendix maps each unimplemented tengri model to its implementation
(or absence) in major SED fitting codes, with exact source file paths,
template locations, and download URLs.

### A.1. Code Repositories

| Code | Language | Repository | Key Reference |
|------|----------|-----------|---------------|
| **Synthesizer** | Python | https://github.com/flaresimulations/synthesizer | Roper+2025 |
| **BAGPIPES** | Python | https://github.com/ACCarnall/bagpipes | Carnall+2018 |
| **CIGALE** | Python | https://gitlab.lam.fr/cigale/cigale (official); https://github.com/JohannesBuchner/cigale (mirror) | Boquien+2019 |
| **X-CIGALE** | Python | Merged into CIGALE ≥2022 | Yang+2020, 2022 |
| **AGNfitter** | Python | https://github.com/GabrielaCR/AGNfitter | Calistro Rivera+2016; Martínez-Ramírez+2024 |
| **Prospector** | Python | https://github.com/bd-j/prospector | Johnson+2021 |
| **ProSpect** | R | https://github.com/asgr/ProSpect | Robotham+2020 |
| **MAGPHYS** | Fortran | http://www.iap.fr/magphys/ | da Cunha+2008 |
| **S³Fit** | Python | https://github.com/xychcz/S3Fit | Chen+2025 (arXiv:2503.xxxxx) |
| **PAHFIT** | Python/IDL | https://github.com/PAHFIT/pahfit (Python); https://github.com/PAHFIT/pahfit_classic (IDL) | Smith+2007 |
| **Synthesizer** | Python | https://github.com/flaresimulations/synthesizer | Lovell+2025 |
| **BEAGLE** | Fortran | Private; contact Chevallard | Chevallard & Charlot 2016 |
| **Dense Basis** | Python | https://github.com/kartheikiyer/dense_basis | Iyer+2019 |
| **FSPS** | Fortran/Python | https://github.com/cconroy20/fsps ; python-fsps | Conroy+2009, 2010 |
| **DSPS** | Python/JAX | https://github.com/ArgonneCPAC/dsps | Hearin+2023 |

### A.2. Model Implementation Matrix

| Model | tengri | CIGALE | Prospector | BAGPIPES | AGNfitter | ProSpect | MAGPHYS | S³Fit | Synthesizer |
|-------|--------|--------|------------|----------|-----------|----------|---------|-------|-------------|
| **DL07 dust emission** | ✓ | `dl2007.py` | via FSPS | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **DL14 dust emission** | ✓ | `dl2014.py` | via FSPS | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Astrodust+PAH** | ✗ TODO | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **THEMIS dust** | ✗ TODO | `themis.py` ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **BOSA templates** | ✗ TODO | partial (via `dl2014`) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **MAGPHYS dust** | ✗ TODO | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ native | ✗ | ✗ |
| **Dale+2014** | ✓ | `dale2014.py` | ✗ | ✗ | `dale_lib` | `Dale_Helou` ✓ | ✗ | ✗ | ✗ |
| **Casey 2012 MBB** | ✓ | `casey2012.py` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Closed-box Z(t)** | partial | ✗ | ✗ | ✗ | ✗ | `Zfunc_massmap_box` ✓ | ✗ | ✗ | ✗ |
| **ADAF** | buggy | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Patchy IGM** | ✗ TODO | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **MAPPINGS V shocks** | simplified | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **SKIRTOR torus** | ✓ | `skirtor2016.py` ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **K&D disc** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **QSOgen** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Cue emulator** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **CLOUDY nebular** | ✓ | `nebular.py` ✓ | via FSPS ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **α-enhancement** | ✓ | ✗ | ✗ | ✗ | ✗ | via EMILES | ✗ | ✗ | ✗ |

### A.3. Specific Source Files for Each Model

#### Dust Emission Templates

**CIGALE DL07:**
- Source: `pcigale/sed_modules/dl2007.py`
- Templates stored in CIGALE database (sqlite → HDF5 in recent versions)
- q_PAH grid: [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58]
- U_min grid: [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00, 10.0, 12.0, 15.0, 20.0, 25.0]
- γ enters analytically, not as grid axis

**CIGALE THEMIS:**
- Source: `pcigale/sed_modules/themis.py`
- Contributed by DustPedia team (Nersesian & Galliano)
- Templates generated with DustEM Fortran code
- Download CIGALE database: run `pcigale-data` after install to pull templates
- Release notes: CIGALE v0.12.0 first included THEMIS

**CIGALE Dale+2014:**
- Source: `pcigale/sed_modules/dale2014.py`
- 64 templates indexed by α (power-law slope of dM/dU)
- Also serves as the base for DL14 (different template set, same interface)

**ProSpect Dale+Helou:**
- Source: `ProSpect::Dale_Helou` R function
- Built into ProSpectData R package
- Uses the Dale et al. 2014 templates with α parameterization

**MAGPHYS native dust emission:**
- Source: Fortran code at http://www.iap.fr/magphys/
- Four-component model (PAH + hot MIR + warm + cold MBB)
- Template library pre-computed on a grid; not easily extractable
- For tengri: reimplement analytically (no templates needed)

**AGNfitter dust emission:**
- Source: `AGNfitter/models/GALAXY/`
- Uses Dale+2014 templates (`dale_lib`) for host galaxy dust
- AGN torus: separate templates in `AGNfitter/models/TORUS/`
- Radio: double power-law model in AGNfitter-rx (Martínez-Ramírez+2024)

#### Chemical Evolution

**ProSpect closed-box:**
- Source: `ProSpect::Zfunc_massmap_box()` in R
- Implements exactly the Bellstedt+2020 model
- Free parameter: `Zfinal` (present-day gas metallicity, log Z/Z_⊙)
- Yield solved internally via `y = -Z_final / ln(mu_final)`
- Also: `Zfunc_massmap_lin()` for linear ramp (simpler)

**Prospector metallicity:**
- Uses a single metallicity or log-uniform prior
- No self-consistent chemical evolution model
- Metallicity is a free parameter at each age bin in non-parametric mode

#### AGN Models

**CIGALE AGN:**
- Fritz+2006 torus: `pcigale/sed_modules/fritz2006.py`
- SKIRTOR torus: `pcigale/sed_modules/skirtor2016.py` (added in v0.12.1)
- X-ray: `pcigale/sed_modules/xray.py` (Yang+2020, 2022)
- αox bridge: Just+2007 relation, same as tengri

**AGNfitter AGN:**
- Accretion disc: empirical templates + Richards+2006 composite
- Torus: Fritz+2006 or Silva+2004 templates
- AGNfitter-rx (2024): adds CAT3D-Wind, double power-law radio, X-ray

**S³Fit AGN (https://github.com/xychcz/S3Fit):**
- Simultaneous spectrum + photometric SED fitting
- AGN continuum: power-law + Fe II pseudo-continuum
- Emission lines: narrow + broad + outflow components
- Host galaxy: BC03 SSPs with parametric SFH
- Dust: Calzetti attenuation + energy-balance IR emission
- Key innovation: joint spectro-photometric fitting with many components
- Source structure: `s3fit/models/` contains all component models
- Useful reference for implementing simultaneous spectrophotometric
  fitting with complex AGN decomposition

#### Nebular Emission

**CIGALE nebular:**
- Source: `pcigale/sed_modules/nebular.py`
- CLOUDY-based line ratios, metallicity-dependent
- Updated to Villa-Vélez+2021 grids in recent versions
- log U grid: [-4, -3, -2, -1]

**Prospector/FSPS nebular:**
- Uses Byler+2017 CLOUDY grids built into FSPS
- Lines + continuum, parameterized by log U and Z_gas
- `nebemlineinspec` flag controls whether lines are in spectra

**BAGPIPES nebular:**
- Uses pre-computed CLOUDY grids (Byler+2017 or Gutkin+2016)
- Parameterized by log U; metallicity tied to stellar Z

**PAHFIT (for PAH decomposition):**
- Python: https://github.com/PAHFIT/pahfit (astropy.modeling framework)
- IDL: https://github.com/PAHFIT/pahfit_classic
- Drude profile function (IDL): `pahfit_drude(lambda, lam_0, central_inten, frac_fwhm)`
  returns `central_inten * frac_fwhm^2 / ((lambda/lam_0 - lam_0/lambda)^2 + frac_fwhm^2)`
- Python PAHFIT v2 uses `astropy.modeling` with the same Drude profiles
  but parameterizes by integrated power (preserves power when FWHM varies)
- Feature packs: "classic" (Smith+2007 17 features) and "PDR" (JWST-era
  with sub-components of 7.7 μm complex)
- Do NOT reimplement PAHFIT's fitting; extract the physics (Drude profile
  parameters) for use in the MAGPHYS emission model

#### IGM Models

**Prospector IGM:**
- Uses Madau (1995) or Inoue+2014 mean IGM
- No patchy reionization model

**BAGPIPES IGM:**
- Inoue+2014 mean transmission
- No patchy model

**No existing code** implements patchy IGM for broadband SED fitting.
The Miralda-Escudé (1998) damping wing formalism is used in Lyα
analyses (Mason+2018, Mesinger & Furlanetto 2007) but has not been
incorporated into any SED fitting code as a broadband photometric model.
tengri would be the first.

#### ADAF Models

**Reference implementations for comparison:**
- Nemmen+2014 Python: https://github.com/rsnemmen/adaf
  - Implements Mahadevan 1997 + Narayan & Yi 1995 self-similar solution
  - Includes synchrotron, bremsstrahlung, inverse Compton
  - Validated against Sgr A* and M87
  - **Use for parity tests, not as base code** (not JAX-compatible)
- Yuan & Narayan 2014 review (ARA&A 52, 529): comprehensive equations
  for hot accretion flows including ADAF, LHAF, CDAF
- Lecture notes: https://www.astro.ru.nl/~wilberth/agn/adaf.pdf
  — pedagogical derivation of ADAF structure equations
- Véron-Cetty & Véron AGN review: https://ned.ipac.caltech.edu/level5/VCetty/
  — general AGN physics context

---

## Appendix B: Master Template Download Table

All template files needed for the unimplemented models, with exact URLs
and file formats.

| Model | Source | URL | Format | Size (approx) |
|-------|--------|-----|--------|---------------|
| Astrodust emission | Hensley & Draine | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3B6E6S | FITS (HDU: emission vs λ for each U) | ~50 MB |
| Astrodust dielectric | Draine & Hensley | https://datacommons.princeton.edu/discovery/catalog/doi-10-34770-9ypp-dv78 | gzipped text files | ~200 MB |
| PAH emission spectra | Draine+2021 | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LPUHIQ | FITS | ~30 MB |
| Astrodust Python tools | Hensley | https://github.com/brandonshensley/Astrodust | Python notebooks | ~1 MB |
| THEMIS templates | CIGALE/DustPedia | Install CIGALE, run `pcigale-data` | HDF5 in CIGALE DB | ~100 MB |
| DustEM code | IAS Orsay | https://www.ias.u-psud.fr/DUSTEM/ | Fortran + data | ~50 MB |
| BOSA templates | Boquien & Salim | CDS via A&A 653, A149 (2021) | ASCII/FITS | ~10 MB |
| Dale+2014 templates | Dale+2014 | Via CIGALE DB or original release | FITS | ~5 MB |
| Smith+2007 PAH Drude | Smith+2007 / PAHFIT | https://github.com/PAHFIT/pahfit (Python); Table 2 of ApJ 656, 770 | Code constants | — |
| MAPPINGS V grids | 3MdBs | http://3mdb.astro.unam.mx/ | Online query or bulk download | ~500 MB |
| SKIRTOR torus | Stalevski+2016 | Via CIGALE DB or http://dc.g-vo.org/skirtor | FITS | ~1 GB |
| Nemmen ADAF code | Nemmen+2014 | https://github.com/rsnemmen/adaf | Python | ~1 MB |
| ProSpect (for Z(t)) | Robotham+2020 | https://github.com/asgr/ProSpect | R package | ~50 MB |

---

## Appendix C: SEMPER — Semi-EMPirical Model for Extragalactic Radio Emission

**Paper:** Giulietti et al. 2025, A&A (submitted), arXiv:2503.20525  
**Status:** Semi-empirical model for radio luminosity functions, not an SED fitting code.

SEMPER (Semi-EMPirical model for Extragalactic Radio emission) is NOT an
SED fitting code — it is a forward model for predicting radio luminosity
functions and number counts of star-forming galaxies at 1.4 GHz and
150 MHz. It combines:

1. Redshift-dependent galaxy stellar mass functions from COSMOS2020
   (Weaver+2023)
2. The galaxy main sequence (SFR–M★ relation)
3. The mass- and redshift-dependent FIR–radio correlation (FIRRC) from
   Delvecchio+2021 and McCheyne+2022

**Relevance to tengri:** SEMPER's FIRRC parameterization is directly
relevant to tengri's radio emission module (§I.2). Specifically:

- SEMPER uses the Delvecchio+2021 q_IR(M★, z) relation:
  `q_IR = q₀(1+z)^{z_s} − m_s(log M★ − 10)`
  with q₀ = 2.743, m_s = 0.234, z_s = −0.025
- Also tests McCheyne+2022: q₀ = 1.98, m_s = −0.22, z_s = 0.02
- Both calibrations are already implemented in tengri's radio module
  (Table 9, `delvecchio2021` and `mccheyne2022` backends)

**What tengri could adopt from SEMPER:**
- The COSMOS2020-calibrated stellar mass function as a prior for
  hierarchical inference
- The validation that mass-dependent FIRRC reproduces observed radio
  LFs at z = 0–5 — useful for validating tengri's radio predictions
- SEMPER Paper II (forthcoming) will add AGN radio emission, which
  could inform tengri's AGN radio backends

No templates or code to download; SEMPER is a population model, not a
per-galaxy SED fitter.

---

## Appendix D: Additional SED Fitting Codes

### D.1. Lightning (Doore+2023)

**Paper:** Doore et al. 2023, ApJS 266, 39 (arXiv:2304.06753)  
**Repository:** https://github.com/rafaeleufrasio/lightning  
**Language:** IDL (v8.3+)  
**Wavelength range:** X-ray to submillimeter

Lightning provides:
- **Stellar populations:** FSPS (Conroy+2009, 2010) with flexible SFH
  (non-parametric age bins, default N=4)
- **Dust attenuation:** Modified Calzetti (Noll+2009) with variable UV
  slope δ and bump strength E_b; Charlot & Fall 2000 two-component
- **Dust emission:** Draine & Li 2007 templates (q_PAH, U_min, γ) with
  energy balance
- **AGN:** SKIRTOR clumpy torus (Stalevski+2016); optional X-ray AGN
  corona via αox bridge
- **X-ray:** HMXB (∝ SFR) + LMXB (∝ M★) + optional AGN power-law
- **Nebular:** Byler+2017 CLOUDY grids
- **IGM:** Madau 1995
- **Inference:** Adaptive MCMC, affine-invariant MCMC (emcee), MPFIT

**Relevant implementations for tengri comparison:**
- DL07 dust emission with energy balance: compare template
  interpolation approach
- X-ray binary scaling relations: same Grimm+2003 / Gilfanov+2004
  as tengri
- SKIRTOR torus: same templates, different interpolation

**What Lightning lacks that tengri has:** differentiability, variational
inference, correlated-field SFH, Cue emulator, K&D disc, ADAF, chemical
evolution, α-enhancement, THEMIS/Astrodust/MAGPHYS dust.

### D.2. MCSED (Bowman+2020)

**Paper:** Bowman et al. 2020, ApJ 899, 7 (arXiv:2006.13245)  
**Repository:** https://github.com/grzeimann/MCSED  
**Documentation:** https://mcsed.readthedocs.io/en/latest/  
**Language:** Python (emcee MCMC)

MCSED provides:
- **SSPs:** FSPS with PADOVA or MIST isochrones; flexible IMF
- **SFH:** constant, exponential, double power law, burst, polynomial,
  binned (non-parametric with N age bins)
- **Dust attenuation:** Calzetti, Noll+2009 (with variable UV slope δ
  and 2175 Å bump E_b), Kriek & Conroy 2013
- **Dust emission:** DL07 templates + PAH continuum prescription
- **Nebular:** continuum + line emission from ionized gas, parameterized
  by ionization parameter log U
- **IGM:** Madau 1995 correction
- **Metallicity:** fixed or mass-metallicity relation (Ma+2016)
- **NO AGN emission** — MCSED explicitly excludes AGN modeling

**Notable design features:**
- Accepts emission-line fluxes and absorption-line indices as
  constraints alongside photometry
- Targets z ~ 1–3 emission-line galaxies (cosmic noon)
- Uses emcee (500 walkers × 20,000 steps)
- Flexible PAH + dust continuum emission model

**What MCSED lacks that tengri has:** AGN modeling, differentiability,
variational inference, chemical evolution, multiple dust emission models,
spectroscopic fitting with calibration polynomials, hierarchical inference.

### D.3. Dense Basis (Iyer & Gawiser 2017; Iyer+2019)

**Papers:** Iyer & Gawiser 2017, ApJ 838, 127 (arXiv:1702.04371);
Iyer et al. 2019, ApJ 879, 116 (arXiv:1901.02877)  
**Repository:** https://github.com/kartheikiyer/dense_basis  
**Documentation:** https://dense-basis.readthedocs.io/  
**PyPI:** `pip install dense-basis`  
**Language:** Python (FSPS backend; GP-SFH module usable standalone)

Dense Basis is a Bayesian SED fitting code whose central innovation is a
Gaussian Process (GP) nonparametric SFH representation. Rather than
parameterizing the SFH with a functional form (exponential, delayed-τ,
double power law), it describes the SFH through the lookback times t_x
at which a galaxy assembles specified quantiles of its stellar mass
(e.g., t_25, t_50, t_75). A GP interpolates between these quantile
times to produce a smooth, continuous SFH with no fixed functional form.

**SFH parameterization (the key physics):**

The SFH is specified by a tuple: `(M★, SFR, Z, t_x1, t_x2, ..., t_xN, A_V)`
where:
- M★ is the total stellar mass formed
- SFR is the current star formation rate (or averaged over 100 Myr)
- t_x are the lookback times at which quantiles x of M★ are assembled
- The number of quantile parameters N is flexible (typically 3–7),
  adjusted to match the information content of the available photometry

The GP kernel produces SFHs that are smooth by construction, without
requiring explicit regularization priors (cf. the continuity prior in
Prospector's non-parametric bins, or the IFT correlated field in tengri).

**Comparison with tengri's IFT correlated-field SFH:**

| Feature | Dense Basis GP-SFH | tengri IFT SFH |
|---------|-------------------|----------------|
| SFH representation | Mass-assembly quantile times t_x | Fourier-space correlated field x(t) |
| Smoothness | GP kernel (squared exponential) | PSD shape (DRW / Matérn) |
| Burstiness control | Implicit (N_params, GP length scale) | Explicit (σ_PS, τ_PS in PSD) |
| Dimensionality | 3–7 SFH params + physical params | ~128–256 latent dims + PSD params |
| Physical interpretation | When was mass assembled? | What is the variability power spectrum? |
| Hierarchical inference | Not built-in | Native (shared PSD across population) |
| Differentiability | No (FSPS backend, gradient-free) | Yes (JAX, gradient-based) |

The GP-SFH and IFT approaches are complementary. Dense Basis excels at
low-dimensional SFH recovery with minimal assumptions; tengri's IFT
approach enables high-dimensional inference and hierarchical population
constraints on burstiness.

**Forward model components:**
- **SSPs:** FSPS (Conroy+2009, 2010) via python-fsps
- **Nebular:** CLOUDY grids via FSPS (Byler+2017)
- **Dust attenuation:** Calzetti+2000 (default); Kriek & Conroy 2013
  available
- **Dust emission:** Not modeled (UV-to-NIR only; no IR energy balance)
- **AGN:** Not modeled
- **IGM:** Not modeled
- **Metallicity:** Fixed or mass-metallicity relation
- **Inference:** Atlas-based χ² fitting (fast); optional emcee MCMC

**What Dense Basis lacks relative to tengri:** IR dust emission, AGN
models, spectroscopic fitting, energy balance, chemical evolution,
differentiability, variational inference, hierarchical fitting, radio/X-ray.

**What Dense Basis offers that is relevant to tengri:**
1. The GP-SFH module (`gp_sfh`) is usable standalone without FSPS —
   it generates smooth SFHs from quantile times. This could serve as
   an alternative SFH prior or as a validation target for tengri's
   IFT SFH.
2. The `tx_alpha` sampling from IllustrisTNG (0 < z < 6, N_param < 10)
   provides simulation-calibrated priors on mass assembly times.
3. The atlas-based fitting approach (pre-compute a grid of SEDs, then
   find nearest neighbors) is a useful fast-mode complement to tengri's
   iterative inference — potentially as an initializer for MAP/geoVI.

**Parity tests with tengri:**
1. For a common galaxy (e.g., CANDELS GOODS-S, z ~ 1, M★ ~ 10¹⁰ M⊙),
   fit with both Dense Basis GP-SFH and tengri's IFT SFH using the
   same photometry. Compare recovered M★, SFR, and SFH shape.
2. Generate mock SFHs using Dense Basis's GP kernel and tengri's DRW
   PSD. Verify that both produce statistically similar SFH variability
   when matched in burstiness amplitude.

### D.4. Updated Code Table Entry

Add to Appendix A.1:

| Code | Language | Repository | Key Reference |
|------|----------|-----------|---------------|
| **Lightning** | IDL | https://github.com/rafaeleufrasio/lightning | Doore+2023 |
| **MCSED** | Python | https://github.com/grzeimann/MCSED | Bowman+2020 |
| **Dense Basis** | Python | https://github.com/kartheikiyer/dense_basis | Iyer & Gawiser 2017; Iyer+2019 |
| **SEMPER** | (population model) | arXiv:2503.20525 | Giulietti+2025 |

### D.5. Updated Model Matrix Rows

| Model | Lightning | MCSED | Dense Basis |
|-------|-----------|-------|-------------|
| DL07 dust emission | ✓ | ✓ (+ PAH continuum) | ✗ (UV-NIR only) |
| Astrodust+PAH | ✗ | ✗ | ✗ |
| THEMIS | ✗ | ✗ | ✗ |
| MAGPHYS dust | ✗ | ✗ | ✗ |
| Closed-box Z(t) | ✗ | ✗ | ✗ |
| ADAF | ✗ | ✗ | ✗ |
| Patchy IGM | ✗ | ✗ | ✗ |
| SKIRTOR torus | ✓ | ✗ | ✗ |
| X-ray binaries | ✓ (Grimm/Gilfanov) | ✗ | ✗ |
| AGN X-ray | ✓ (αox) | ✗ | ✗ |
| Radio | ✗ | ✗ | ✗ |
| Nebular CLOUDY | ✓ (Byler+2017) | ✓ (simplified) | ✓ (via FSPS) |
| 2175 Å bump | ✓ (Noll+2009) | ✓ (Noll/KC13) | ✗ (Calzetti default) |
| GP/nonparametric SFH | ✗ | ✓ (binned) | ✓ (GP quantile) |
| Correlated-field SFH | ✗ | ✗ | ✗ |
