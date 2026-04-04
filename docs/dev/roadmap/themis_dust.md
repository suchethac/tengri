# THEMIS Dust Model

## Citation

Jones, A. P., Kohler, M., Ysard, N., Bocchio, M., & Verstraete, L. 2017,
A&A, 602, A46.
"The global dust modelling framework THEMIS (The Heterogeneous dust Evolution
Model for Interstellar Solids)."

## Overview

THEMIS is a comprehensive dust modeling framework built on **laboratory-measured
properties** of physically realistic interstellar dust analog materials. Unlike
traditional astronomical dust models that use "astronomical silicate" and
"graphite" (materials that do not exist in the lab), THEMIS uses amorphous
silicates and hydrogenated amorphous carbons, a-C(:H), whose optical properties
are measured in the laboratory.

The key innovation is that dust properties **evolve** with environment: UV
processing in the diffuse ISM transforms aliphatic carbon into aromatic carbon,
while accretion in dense clouds builds up hydrogen-rich mantles. This provides a
physical basis for the observed variations in dust properties across different
ISM phases.

## Grain Composition

### Two grain families

| Family | Material | Size range | Description |
|--------|----------|------------|-------------|
| Amorphous carbon a-C(:H) | Hydrogenated amorphous carbon | 0.4 nm -- 200 nm | Size-dependent optical properties |
| Amorphous silicate a-Sil | Olivine + pyroxene + Fe inclusions | 8 nm -- 3000 nm | Core-mantle grains |

### Amorphous silicates (a-Sil)

A 1:1 mass mixture of:
- **Amorphous olivine-type** (forsterite composition)
- **Amorphous pyroxene-type** (enstatite composition)

With nano-inclusions:
- Metallic iron: 7% of grain volume
- Iron sulfide (FeS): 3% of grain volume

### Amorphous hydrogenated carbons a-C(:H)

The a-C(:H) family spans a continuous range from:
- **H-poor, aromatic-rich** ($\sim$15 at.% H): UV-processed surfaces in diffuse
  ISM, graphite-like optical properties
- **H-rich, aliphatic-rich** ($\sim$60 at.% H): Dense cloud mantles,
  diamond-like optical properties

Optical properties are computed from the **optEC(s)** model (Jones 2012a,b,c),
which parameterizes the complex refractive index as a function of:
- Band gap $E_g$ (eV): controls aromatic/aliphatic ratio
- Grain radius $a$: size-dependent surface effects

## Aromatic Feature Emergence

A distinctive feature of THEMIS is the **physical explanation** for aromatic
emission features (the "PAH bands"):

1. In the **diffuse ISM**, UV photolysis converts aliphatic C-H bonds to
   aromatic C=C bonds in the outer layers of carbonaceous grains
2. Small a-C nanoparticles ($a \lesssim 20$ nm) become entirely aromatic due to
   their high surface-to-volume ratio
3. These UV-processed small grains produce the 3.3, 6.2, 7.7, 8.6, 11.3, 12.7
   $\mu$m features
4. In **dense clouds**, mantles of accreted H-rich a-C:H suppress aromatic
   features

This replaces the ad hoc PAH population of DL07 with a physically motivated
size-dependent aromaticity.

## Size Distribution

### Diffuse ISM model

| Component | Distribution | Size range | Peak |
|-----------|-------------|------------|------|
| a-C nanoparticles | Power law | 0.4--20 nm | -- |
| Large a-Sil/a-C CM grains | Log-normal | 10--3000 nm | $\sim$140--160 nm |

CM = core-mantle structure: amorphous silicate core + aromatic carbon mantle.

## Optical Properties

Cross-sections are computed as:

$$
C_i(a, \varphi, \xi, \lambda) = \pi a^2 \times Q_i(a, \varphi, \xi, \lambda)
$$

where:
- $a$ = particle radius
- $\varphi$ = material composition (band gap $E_g$)
- $\xi$ = grain structure (core-mantle, aggregate, etc.)
- $Q_i$ = efficiency factor for process $i$ (extinction, absorption, scattering)

### Effective density for core-mantle grains

$$
\langle\rho_{\rm CM}\rangle = \langle\rho_{\rm core}\rangle
\left[1 - (d/a)^3\right] + \rho_{\rm mantle}\left\{1 - \left[1 - (d/a)^3\right]\right\}
$$

where $d$ is the mantle thickness.

### Computational methods

- **Spherical/coated grains**: Mie theory (core-mantle Mie for CM grains)
- **Aggregates**: Discrete Dipole Approximation (DDSCAT)

## Elemental Abundances

The diffuse ISM model requires (in ppm relative to H):

| Element | Required abundance (ppm) |
|---------|-------------------------|
| Carbon | 206--218 (143--144 in nanoparticles) |
| Oxygen | 110 |
| Magnesium | 45 |
| Silicon | 32 |
| Iron | 19 |
| Sulfur | 3 |

## Grain Evolution with Environment

THEMIS provides a framework for dust evolution across ISM phases:

| Environment | Grain structure | Designation | Key process |
|-------------|----------------|-------------|-------------|
| Diffuse ISM | Core-mantle | CM | UV processing of mantles |
| Translucent cloud | Core-mantle-mantle | CMM | a-C:H accretion |
| Dense cloud | Aggregated CMM | AMM | Coagulation |
| Very dense | AMM + ice | AMMI | Ice mantle accretion |

Each evolution step changes:
- FIR opacity (increases with aggregation)
- Dust temperature (decreases with increasing FIR emissivity)
- MIR spectral features (aromatic bands weaken in dense regions)

## Differences from DL07

| Property | DL07 | THEMIS |
|----------|------|--------|
| Large grain material | Silicate + graphite (separate) | a-Sil CM + a-C (core-mantle) |
| Carbon grain material | Astronomical graphite | Amorphous hydrogenated carbon |
| Aromatic features | Separate PAH molecules | Aromatic fraction of a-C nanoparticles |
| Optical properties | Astronomical (empirical) | Laboratory-measured |
| Environmental evolution | None | UV processing, accretion, coagulation |
| Dense cloud behavior | Same as diffuse ISM | Modified (CMM, AMM, AMMI) |
| UV bump origin | Graphite $\pi$-$\pi^*$ | a-C band gap transition |

## CIGALE Integration

THEMIS is the default dust model in the DustPedia project and is available in
CIGALE (Code Investigating GALaxy Emission). CIGALE uses THEMIS through the
DustEM code to compute emission from the THEMIS grain populations heated by the
ISRF.

In CIGALE, the THEMIS dust model is parameterized similarly to DL07:

| Parameter | Description |
|-----------|-------------|
| $q_{\rm hac}$ | Mass fraction of small a-C grains (analogous to $q_{\rm PAH}$) |
| $U_{\min}$ | Minimum radiation field intensity |
| $\alpha$ | Power-law slope of $U$ distribution |

## Implementation Notes for tengri

1. **Template approach**: Pre-compute THEMIS emission spectra on a grid of
   $(q_{\rm hac}, U_{\min}, \alpha)$ using DustEM, similar to the existing DL07
   template grid.

2. **DustEM dependency**: The full THEMIS calculation requires the DustEM code.
   For tengri, pre-tabulated templates are more practical than running DustEM
   at each likelihood evaluation.

3. **Mapping to DL07 parameters**: The THEMIS parameterization maps roughly to
   DL07:
   - $q_{\rm hac} \leftrightarrow q_{\rm PAH}$
   - $U_{\min} \leftrightarrow U_{\min}$
   - $\alpha \leftrightarrow \alpha$

4. **Data files**: THEMIS data are available from the THEMIS website
   (https://www.ias.u-psud.fr/themis/). The optical property tables and grain
   size distributions can be downloaded for template generation.

5. **Environmental dependence**: For a first implementation, use the diffuse ISM
   (CM) model only. The dense cloud variants (CMM, AMM) would be relevant for
   spatially resolved fitting or for galaxies with significant molecular gas
   fractions.

6. **Comparison to existing code**: SKIRT uses THEMIS dust mixes natively
   (ThemisDustMix class). The SKIRT documentation provides a useful reference
   for parameter values and implementation details.
