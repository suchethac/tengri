# Astrodust+PAH Dust Model

## Citation

Hensley, B. S. & Draine, B. T. 2023, ApJ, 948, 55.
"The Astrodust+PAH Model: A Unified Description of the Extinction, Emission,
and Polarization from Dust in the Diffuse Interstellar Medium."

## Overview

The Astrodust+PAH model replaces the traditional two-component
silicate+graphite dust framework (used in DL07 and earlier Draine models) with a
**single composite material** ("astrodust") for all grains larger than
$\sim$0.02 $\mu$m, combined with distinct PAH nanoparticles. The key motivation
is that a single-composition model most naturally explains the **lack of
frequency dependence** in the far-infrared polarization fraction and the
characteristic ratio of optical to FIR polarization.

## Grain Composition

### Astrodust (large grains, $a \gtrsim 0.02\;\mu$m)

A composite material containing:

- Silicates: $\mathrm{Mg}_{1.3}(\mathrm{Fe,Ni})_{0.3}\mathrm{SiO}_{3.6}$
- Iron oxides and carbides
- Hydrocarbons
- SiO$_2$, Al$_2$O$_3$, CaCO$_3$
- Porosity $\mathcal{P} = 0.2$
- Bulk density: $\rho = 2.74$ g cm$^{-3}$

### PAH nanoparticles ($a \lesssim 0.02\;\mu$m)

Standard PAH population as in earlier Draine models, but with updated mass
fractions.

## Grain Geometry

All astrodust grains are modeled as **oblate spheroids** with a fixed axial
ratio of **1.4:1** (minor-to-major axis). This choice is constrained by the
observed FIR polarization fraction.

## Size Distribution

The astrodust size distribution uses a **lognormal + polynomial** form:

$$
\frac{1}{n_H} \frac{dn}{da} = \frac{B_{\rm Ad}}{a} \exp\!\left[
-\frac{1}{2}\left(\frac{\ln(a/a_{0,\rm Ad})}{\sigma_{\rm Ad}}\right)^2
\right] + \text{polynomial terms}
$$

### Fitted parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $B_{\rm Ad}$ | $3.31 \times 10^{-10}$ H$^{-1}$ | Lognormal normalization |
| $a_{0,\rm Ad}$ | 63.8 A | Lognormal peak radius |
| $\sigma_{\rm Ad}$ | 0.353 | Lognormal width |
| $A_0$--$A_5$ | (polynomial coefficients) | Small-grain behavior |
| $V_{\rm Ad}$ | $3.92 \times 10^{-27}$ cm$^3$ H$^{-1}$ | Total volume per H |

## Alignment Function

The degree of grain alignment with the magnetic field is described by:

$$
f_{\rm align}^{\rm Ad}(a) = \frac{f_{\max}}{1 + (a_{\rm align}/a)^{\alpha_{\rm align}}}
$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $f_{\max}$ | 1.00 | Maximum alignment efficiency |
| $a_{\rm align}$ | 0.0749 $\mu$m | Half-maximum alignment radius |
| $\alpha_{\rm align}$ | 1.80 | Steepness of transition |

This yields a mass-weighted alignment fraction of $\sim$70%.

## PAH Mass Fraction ($q_{\rm PAH}$)

The parameter $q_{\rm PAH}$ is defined as the **fraction of total dust mass in
PAH grains containing fewer than 1000 C atoms**.

| Model | $q_{\rm PAH}$ |
|-------|---------------|
| Astrodust+PAH (this work) | 5.91% |
| Draine & Li 2007 | 4.6% |
| Draine et al. 2021 | 3.8% |

The higher $q_{\rm PAH}$ in the Astrodust model compensates for the revised
optical properties of the large-grain component.

## Radiation Field and Emission

Emission is computed for grains heated by the interstellar radiation field (ISRF)
scaled by the parameter $U$:

$$
u_\nu = U \times u_\nu^{\rm MMP83}
$$

where $u_\nu^{\rm MMP83}$ is the Mathis, Mezger & Panagia (1983) ISRF. The
best-fit scaling for the diffuse ISM is $U \approx 1.6$ (higher than DL07 due
to larger submillimeter opacities and revised extinction normalization).

Cross-sections are computed via the **Modified Picket Fence Approximation**
(MPFA) for three orthogonal orientations, yielding random-orientation
extinction, scattering, and polarization cross-sections.

## Key Differences from DL07

| Property | DL07 | Astrodust+PAH |
|----------|------|---------------|
| Large grain composition | Separate silicate + graphite | Single composite astrodust |
| Grain shape | Spheres (Mie theory) | 1.4:1 oblate spheroids |
| $q_{\rm PAH}$ | 4.6% | 5.91% |
| ISRF scaling $U$ | $\sim$1.0 | $\sim$1.6 |
| FIR polarization | Not predicted | Self-consistent |
| Number of grain types | 3 (silicate, graphite, PAH) | 2 (astrodust, PAH) |
| Optical constants | Separate for each mineral | Derived from composite |

## Parameter Space for SED Fitting

When used in SED fitting (analogous to DL07), the key parameters are:

| Parameter | Range | Description |
|-----------|-------|-------------|
| $q_{\rm PAH}$ | 0.5% -- 7.3% | PAH mass fraction |
| $U_{\min}$ | 0.1 -- 50 | Minimum starlight intensity |
| $U_{\max}$ | $10^3$ -- $10^7$ | Maximum starlight intensity |
| $\gamma$ | 0 -- 1 | Fraction of dust mass exposed to $U > U_{\min}$ |

The power-law distribution of starlight intensities follows:

$$
\frac{dM_d}{dU} \propto U^{-\alpha}, \quad U_{\min} \le U \le U_{\max}
$$

with $\alpha = 2$ as in DL07.

## Implementation Notes for tengri

1. **Template grid**: Pre-compute astrodust+PAH emission templates on a grid of
   $(q_{\rm PAH}, U_{\min}, \gamma)$. The template generation follows the same
   structure as the existing DL07 support.

2. **Data files**: Hensley & Draine provide all model data and Python interfaces
   via Harvard Dataverse. The key files are cross-section tables and emission
   spectra.

3. **Mapping to DL07 interface**: The Astrodust+PAH model can use the same
   3-parameter $(q_{\rm PAH}, U_{\min}, \gamma)$ interface as DL07, making it a
   drop-in replacement for the existing `draine_li2007` module.

4. **Polarization**: Unlike DL07, this model self-consistently predicts
   polarization. This could enable joint SED+polarization fitting in future
   tengri extensions.

5. **Computational cost**: The MPFA calculation for oblate spheroids is more
   expensive than Mie theory for spheres. Pre-tabulation is essential.
