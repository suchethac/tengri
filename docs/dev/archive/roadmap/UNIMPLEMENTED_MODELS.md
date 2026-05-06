# Unimplemented Models — Implementation Guide

This document provides comprehensive specifications for models described
in the Paper I appendix (999-appendix.tex) but not yet implemented in code.
Each section contains everything a future agent needs: paper references,
equations, parameters, input/output contract, and integration points.

Papers referenced here are available at:
*(private arxiv source library)*

---

## 1. Astrodust+PAH Dust Emission (Hensley & Draine 2023)

**Paper:** Hensley & Draine 2023, ApJ 948, 55
**arxiv:** 2306.03268 (check latex_sources/)
**Replaces:** Draine & Li 2007 grain composition
**Status:** Appendix describes it; no code or templates exist.

### Physics

Replaces the silicate-graphite-PAH grain composition of DL07 with a single
composite grain material ("astrodust") plus separate PAH nanoparticles.
Uses laboratory-measured optical properties rather than the astronomical
silicate/graphite proxies of DL07.

### Parameters (same interface as DL07)

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| `dust_qpah` | q_PAH | 0-10% | PAH mass fraction (shifted to 5.91% vs DL07's 4.6%) |
| `dust_umin` | U_min | 0.1-25 | Minimum radiation field intensity |
| `dust_gamma_dl` | gamma | 0-1 | Fraction from PDR (high-U) regions |

### Input/Output

- **Input:** wavelength array (Angstrom), L_IR (erg/s), parameters above
- **Output:** L_nu(lambda) in erg/s/Hz, energy-normalized to L_IR

### Implementation Plan

1. Obtain pre-tabulated templates from Hensley & Draine (their Table 1 or
   online data release). Templates are a 3D grid: (q_PAH, U_min, gamma) x wavelength.
2. Store as `data/astrodust_templates.npz` (same format as `dl07_templates.npz`)
3. Write `astrodust_emission()` in `models/dust/emission.py` following the
   `draine_li2007_emission()` pattern — load templates lazily, interpolate in
   (q_PAH, U_min, gamma) space.
4. Register as `dust_emission="astrodust"` in ParamSpec.

### Key difference from DL07
The q_PAH calibration shifts: DL07 MW reference is 4.6%, Astrodust MW is 5.91%.
The spectral shape differs most at 8-25 um (silicate features) and in the
far-UV opacity. CMB correction (da Cunha+2013) applies identically.

---

## 2. THEMIS Dust Emission (Jones et al. 2017)

**Paper:** Jones et al. 2017, A&A 602, A46
**arxiv:** 1411.6293 (check latex_sources/)
**Code dependency:** DustEM (Fortran), or pre-tabulated grids
**Status:** Appendix describes it; no code or templates exist.

### Physics

Models amorphous hydrogenated carbon a-C(:H) nanoparticles and amorphous
silicate grains with physically motivated size distributions. Optical
properties depend on UV processing history (hydrogenation state).

The key innovation over DL07: dust properties evolve with environment.
In harsh UV fields, a-C(:H) grains lose hydrogen and become more graphitic,
changing their optical properties. THEMIS captures this with a UV-processing
parameter.

### Parameters

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| `dust_qhac` | q_hac | 0-15% | Small hydrocarbon grain fraction (analogous to q_PAH) |
| `dust_umin` | U_min | 0.1-25 | Minimum radiation field intensity |
| `dust_alpha_themis` | alpha | 1-4 | Power-law slope of dM/dU (like Dale) |

### Implementation Plan

1. Generate templates via DustEM or obtain from CIGALE's THEMIS module
   (CIGALE ships pre-computed THEMIS grids).
2. Store as `data/themis_templates.npz`
3. Write `themis_emission()` in `models/dust/emission.py`
4. Register as `dust_emission="themis"`

### Integration notes
- CIGALE source: `pcigale/sed_modules/themis.py` has the template grid
- Template grid axes: (q_hac, U_min, alpha_themis) x wavelength
- Same energy-balance normalization as all other dust emission models

---

## 3. BOSA Dust Emission Templates (Boquien & Salim 2021)

**Paper:** Boquien & Salim 2021, A&A 653, A149
**arxiv:** 2104.10721
**Status:** Appendix describes it; no code or templates exist.

### Physics

Empirical templates adding sSFR as a second axis alongside L_TIR. Captures
the observation that galaxies at fixed IR luminosity but higher sSFR have
warmer dust and stronger MIR features. Based on 2584 star-forming galaxies.

### Parameters

| Parameter | Range | Meaning |
|-----------|-------|---------|
| (none free) | — | L_TIR set by energy balance, sSFR computed from SFH |

This is a zero-free-parameter model: the template is selected entirely by
the predicted L_TIR and sSFR, both derived from other model components.

### Implementation Plan

1. Obtain template grid from Boquien & Salim data release or CIGALE
2. Grid axes: (log_L_TIR, log_sSFR) x wavelength — 2D interpolation
3. At runtime: compute sSFR = SFR_current / M_stellar from the SFH weights
4. Write `bosa_emission()` in `models/dust/emission.py`
5. Register as `dust_emission="bosa"`

### Key feature
No additional free parameters — the dust SED shape is fully predicted by
the galaxy's SFR and stellar mass. This is attractive for reducing
degeneracies in photometric fitting.

---

## 4. MAGPHYS Dust Emission (da Cunha et al. 2008)

**Paper:** da Cunha, Charlot & Elbaz 2008, MNRAS 388, 1595
**arxiv:** 0806.1020
**Status:** Appendix describes equations; no code exists.

### Physics

Four-component model decomposing the IR SED into:
1. **PAH features** — Drude profiles following Smith+2007
2. **Hot MIR continuum** — T ~ 180 K modified blackbody
3. **Warm grains** — T_W ~ 30-60 K, beta=1.5 (birth cloud) or 2.0 (ISM)
4. **Cold grains** — T_C ~ 15-25 K, beta=2.0

### Parameters

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| `dust_xi_pah` | xi_PAH | 0-0.5 | Energy fraction in PAH features |
| `dust_xi_mir` | xi_MIR | 0-0.3 | Energy fraction in hot MIR continuum |
| `dust_xi_w` | xi_W | 0-0.5 | Energy fraction in warm grains |
| `dust_T_w` | T_W | 30-60 K | Warm grain temperature |
| `dust_T_c` | T_C | 15-25 K | Cold grain temperature |

xi_C = 1 - xi_PAH - xi_MIR - xi_W (constraint).

### Equations

Each component is a modified blackbody:
```
L_nu_component = xi * L_abs * (nu/nu_ref)^beta * B_nu(T) / Integral[(nu/nu_ref)^beta * B_nu(T) dnu]
```

PAH features use Drude profiles from Smith+2007:
```
L_nu_PAH = xi_PAH * L_abs * Sum_k[ A_k * gamma_k^2 / ((nu/nu_k - nu_k/nu)^2 + gamma_k^2) ]
```
with ~17 Drude components for the 3.3, 6.2, 7.7, 8.6, 11.3, 12.7 um features.

### Implementation Plan

1. Implement as analytic function (no templates needed)
2. PAH Drude profiles: use wavelengths and widths from Smith+2007 Table 2
3. Write `magphys_emission()` in `models/dust/emission.py`
4. Register as `dust_emission="magphys"`

### Integration notes
- Birth cloud and ISM use different beta values (1.5 vs 2.0)
- The two-component dust framework already separates BC vs ISM L_absorbed
- Energy balance: Sum of all four components = eta * L_absorbed

---

## 5. TEA Dust Attenuation (Haskell et al. 2024)

**Paper:** Haskell et al. 2024, arXiv:2401.11007
**Status:** Code EXISTS at `models/dust/attenuation.py` as `tea()`. Already registered.

This model IS implemented. The roadmap stub is outdated. The TEA curve
imposes a physically motivated E_bump(delta) correlation calibrated on
NIHAO-SKIRT radiative transfer simulations, reducing free parameters to two.

**No action needed.** Remove the roadmap stub.

---

## 6. Chemical Evolution Z(t)

**Paper:** Bellstedt+2021, Robotham+2020
**Status:** Appendix describes it; code EXISTS as `dsps_met_table` CSP mode.

The closed-box / leaky-box chemical evolution model IS partially implemented:
- `compute_log_z_evolving()` produces Z(t) from SFH
- `dsps_met_table` CSP mode supports per-age metallicity tables
- The linear ramp mode also exists

What's NOT implemented:
- The full Bellstedt+2021 self-consistent model where yield is solved to
  match present-day gas metallicity
- The mass-loading factor eta as a free parameter
- The connection between Z(t) and the SFH integration in a single
  self-consistent forward pass

### Implementation Plan

1. Implement `chemical_evolution_closed_box(sfh_weights, ssp_ages, Z_gas_now)`:
   - Compute cumulative mass: M_formed(t) = cumsum(SFR * dt)
   - Gas fraction: mu(t) = 1 - M_formed(t) / M_total
   - Z(t) = -y * ln(mu(t)), solve y to match Z_gas_now at t=0
2. Implement `chemical_evolution_leaky_box(sfh_weights, ssp_ages, Z_gas_now, eta)`:
   - Replace y with y_eff = y / (1 + eta)
3. Wire into ParamSpec as `met_evolution="closed_box"` or `"leaky_box"`
4. Feed Z(t) array to `dsps_met_table` mode (already supports per-age Z)

---

## 7. PAH Feature Decomposition (PAHFIT)

**Paper:** Smith et al. 2007, ApJ 656, 770 (PAHFIT)
**Status:** Not implemented. This is a spectral decomposition tool, not an SED component.

### Physics

Mid-IR spectra (5-38 um) contain strong PAH emission features at 6.2, 7.7,
8.6, 11.3, 12.7, 17 um, plus silicate absorption at 9.7 and 18 um. PAHFIT
decomposes observed spectra into:
- Thermal dust continuum (sum of modified blackbodies at T = 35-300 K)
- PAH emission (Drude profiles, ~17 components)
- Atomic/ionic emission lines (Gaussians)
- Silicate absorption (tau_9.7, tau_18)

### Use case in tengri

Two possible integrations:
1. **Forward model component:** Add PAH Drude profiles to dust emission models
   (partially covered by MAGPHYS PAH component above)
2. **Spectral fitting module:** A PAHFIT-like decomposition for mid-IR spectra
   (Spitzer IRS, JWST MIRI) — this is a specialized tool beyond the SED fitter

### Implementation Plan (forward model only)

1. Add PAH Drude profiles (Smith+2007 Table 2) to `dust_ir_emission`
2. Parameters: q_PAH (overall PAH strength), f_neutral (neutral/ionized ratio)
3. The DL07 and Astrodust templates already include PAH features — this would
   be for the analytic (MAGPHYS-style) path only

---

## 8. Patchy IGM Transmission

**Paper:** No single reference; concept from reionization models
**Related:** Inoue+2014 (current mean IGM), Mason+2018, Bosman+2022
**Status:** Not implemented.

### Physics

At z > 6, the IGM is not fully ionized. The mean Inoue+2014 transmission
model breaks down because sightline-to-sightline scatter dominates:
some sightlines pass through ionized bubbles (high transmission), others
through neutral patches (strong Gunn-Peterson absorption).

### Parameters

| Parameter | Range | Meaning |
|-----------|-------|---------|
| `igm_x_HI` | 0-1 | Neutral hydrogen fraction |
| `igm_bubble_size` | 1-100 Mpc | Characteristic ionized bubble radius |

Or simpler: a single `igm_damping_wing_tau` parameter that modifies the
Lya damping wing opacity.

### Implementation Plan

1. Implement damping wing model: T(lambda, z, x_HI) = T_Inoue(lambda, z) *
   exp(-tau_DW(lambda, x_HI))
2. The damping wing profile is a Voigt function centered at Lya
3. Or: use the Dijkstra+2014 / Mason+2018 parameterization
4. Wire as `igm_model="patchy"` with `igm_x_HI` as free parameter
5. Only active at z > 5.5 (below that, mean IGM is sufficient)

### Science case
Critical for JWST high-z galaxy fitting where Lya damping wing shape
constrains x_HI and hence the reionization epoch.

---

## 9. ADAF Disc Model (Mahadevan 1997)

**Paper:** Mahadevan 1997, ApJ 477, 585; Nemmen+2014, MNRAS 438, 2804
**Status:** Code EXISTS at `models/agn/adaf.py` but has WRONG EQUATIONS
(documented in KNOWN_BUGS.md). Needs full rewrite.

### Physics

For low-luminosity AGN (L/L_Edd < 0.01), the inner accretion disc
transitions from a thin Shakura-Sunyaev disc to an advection-dominated
accretion flow (ADAF). In an ADAF:
- Ions are much hotter than electrons (two-temperature plasma)
- Most gravitational energy is advected into the black hole
- Radiative efficiency is very low (eta << 0.1)
- The SED is dominated by synchrotron (radio), inverse Compton (X-ray),
  and bremsstrahlung

### Key equations (Mahadevan 1997)

Synchrotron emission from thermal electrons:
```
L_synch = (2/3) * n_e * sigma_T * c * B^2 / (8*pi) * V * x_M * F(x_M)
```
where x_M = nu / nu_c (cyclotron frequency), F is the synchrotron function.

Bremsstrahlung:
```
L_brem = 1.4e-27 * T_e^(1/2) * n_e^2 * V * g_ff
```

Inverse Compton scattering of synchrotron photons:
```
L_IC = eta_3 * L_synch * (T_e / (m_e * c^2))^2 * tau_es
```

### Parameters

| Parameter | Range | Meaning |
|-----------|--------|---------|
| `agn_log_mdot` | -4 to -1 | log10(Mdot / M_Edd) |
| `agn_log_mbh` | 6-10 | log10(M_BH / Msun) |
| `agn_a_spin` | 0-0.998 | BH spin |
| `agn_delta` | 0.01-0.5 | Fraction of viscous energy heating electrons |
| `agn_beta_adaf` | 0.5-0.95 | Ratio of gas to total pressure |

### Implementation Plan

1. Read Mahadevan 1997 equations carefully (the current code has errors)
2. Implement three emission components: synchrotron, bremsstrahlung, IC
3. Transition radius from ADAF to thin disc: R_tr ~ alpha^2 * R_S
4. Outer thin disc: standard Shakura-Sunyaev (kubota_done_disc for R > R_tr)
5. Total SED = ADAF(R < R_tr) + thin_disc(R > R_tr) + torus
6. Test against Nemmen+2014 Sgr A* and M87 models

### Known bugs in current code
- Electron temperature calculation is wrong
- Synchrotron self-absorption not properly implemented
- Missing bremsstrahlung Gaunt factor
- See `docs/known-bugs.md` for full list

---

## 10. MAPPINGS V Full Shock Grids

**Paper:** Allen et al. 2008, ApJS 178, 20; Alarie & Morisset 2019
**Status:** Basic `shock_emission()` exists in `emission_helpers.py` using
a simplified model. Full MAPPINGS V grid interpolation not implemented.

### Physics

Fast radiative shocks (v = 100-1000 km/s) in the ISM produce emission
lines from the shock-heated and photoionized gas. MAPPINGS V computes
self-consistent shock+precursor models on a 5D grid.

### Grid axes

| Axis | Range | N_points | Meaning |
|------|-------|----------|---------|
| v_shock | 100-1000 km/s | ~20 | Shock velocity |
| n_pre | 0.01-1000 cm^-3 | ~10 | Pre-shock density |
| B / sqrt(n) | 0.5-10 uG cm^(3/2) | ~5 | Magnetic parameter |
| Z_gas | 0.001-2 Z_sun | ~7 | Gas metallicity |
| model | shock, precursor, shock+precursor | 3 | Emission component |

### Implementation Plan

1. Obtain MAPPINGS V grid data (available from 3MdB database or
   authors' website)
2. Store as `data/mappings_shock_v.npz`
3. Implement 5D multilinear interpolation (same pattern as SKIRTOR)
4. Write `mappings_shock_emission()` in `models/nebular/shock.py`
5. Returns emission line luminosities + continuum at full wavelength

### Current simplified model
The existing `shock_emission()` uses a parametric approximation based
on Halpha luminosity. The full grid provides physically self-consistent
line ratios and continuum shape as a function of shock parameters.

---

## Priority Order for Implementation

1. **Chemical evolution Z(t)** — Partially done, needs Bellstedt+2021 yield solver
2. **ADAF rewrite** — Code exists but is wrong, blocking low-luminosity AGN science
3. **Astrodust+PAH** — Drop-in replacement for DL07, same interface
4. **MAGPHYS dust** — Analytic, no templates needed
5. **THEMIS dust** — Needs templates from DustEM/CIGALE
6. **BOSA templates** — Zero free parameters, attractive for photometric surveys
7. **Patchy IGM** — Critical for JWST z>6 science
8. **MAPPINGS V full grids** — Needs grid data, complex interpolation
9. **PAH decomposition** — Specialized, low priority unless JWST MIRI fitting
10. ~~TEA attenuation~~ — Already implemented, remove roadmap stub
