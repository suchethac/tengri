# TEA Dust Attenuation Model

## Citation

Faucher, N., et al. 2024, ApJ, 977, 164.
"A dust attenuation model inspired by the NIHAO-SKIRT-Catalog."

(Often referred to as the "TEA model" -- THEMIS Effective Attenuation.)

## Overview

The TEA model is a fully analytic, **single-component** dust attenuation curve
with only **three free parameters**. It was calibrated against the NIHAO-SKIRT
Catalog -- a set of synthetic attenuation curves derived from radiative transfer
simulations (SKIRT) applied to cosmological zoom-in simulations (NIHAO) using
THEMIS dust properties and MAPPINGS-III subgrid models for birth clouds.

Unlike two-component models (e.g., Charlot & Fall 2000), TEA treats **all
starlight equally** -- there is no separate attenuation law for young and old
stellar populations. The model naturally encapsulates the effects of star-dust
geometry via its three parameters.

## Attenuation Curve Formula

$$
A_\lambda = A_V \left(\frac{\lambda}{5500\;\text{\AA}}\right)^p
+ b_{\rm UV} \left\{
\left[1 + \mathrm{erf}\!\left(s_{\rm UV} \cdot f_1(\lambda)\right)\right]
\left[1 + \left(6\, f_1(\lambda)\right)^2\right]^{-3/2}
- \frac{1}{4}\left(\frac{f_2(\lambda)^5}{f_2(\lambda)^4 + 1} + f_2(\lambda)\right)
\right\}
$$

### Helper functions

$$
f_1(\lambda) = \log_{10}(\lambda) - 3.312
$$

$$
f_2(\lambda) = 16 \left(\log_{10}(\lambda) - 3.285\right)
$$

where $\lambda$ is in Angstroms.

### Fixed calibration constants

| Constant | Value | Description |
|----------|-------|-------------|
| $s_{\rm UV}$ | 6.95 | Bump skewness coefficient |
| $\lambda_0$ | 2175 A | UV bump center wavelength |
| $\Delta\lambda$ | 350 A | UV bump width |

## Three Free Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| V-band attenuation | $A_V$ | 0 -- 10 mag | Overall attenuation strength |
| Power-law slope | $p$ | $-5$ -- 0 | Slope of the continuum attenuation |
| UV bump strength | $b_{\rm UV}$ | 0 -- 10 | Amplitude of the 2175 A feature |

### Physical interpretation

- **$A_V$**: Total dust column along the line of sight. Larger values indicate
  more dust.
- **$p$**: Controls the wavelength dependence of the continuum attenuation. More
  negative values give steeper (bluer) attenuation curves; values closer to 0
  give grayer curves (characteristic of more complex/mixed geometries).
- **$b_{\rm UV}$**: Strength of the 2175 A UV bump feature, which is attributed
  to carbonaceous grains/PAHs.

## The $E_b$--$\delta$ Correlation

A key finding from the NIHAO-SKIRT catalog is that **grayer attenuation curves
have weaker UV bumps**:

$$
E_b \propto -\delta \quad (\text{anti-correlated})
$$

where $\delta$ is the power-law deviation from Calzetti and $E_b$ is the bump
strength. This arises because more complex star-dust geometries (which produce
grayer curves) also tend to dilute the bump feature through scattering and
mixing of sightlines.

**Crucially**: In the TEA model, this correlation is **not imposed** -- $p$ and
$b_{\rm UV}$ are independent free parameters. The correlation emerges naturally
when fitting real or simulated galaxies.

### Contrast with Kriek & Conroy (2013)

The Kriek-Conroy model **deterministically ties** the bump strength to the slope:

$$
E_b^{\rm KC} = -1.9 \times \delta_{\rm KC}
$$

This deterministic relation is too rigid for the diversity of attenuation curves
seen in NIHAO-SKIRT simulations.

## Comparison to Related Models

| Model | Components | Free params | UV bump | Slope-bump relation |
|-------|-----------|-------------|---------|---------------------|
| **TEA** | Single | 3 ($A_V$, $p$, $b_{\rm UV}$) | Independent | Not imposed |
| Calzetti+2000 | Single | 1 ($A_V$) | None | N/A |
| Charlot & Fall 2000 | Two (BC+ISM) | 3 ($\hat\tau_V$, $\mu$, $n$) | None | N/A |
| Kriek & Conroy 2013 | Single | 2 ($A_V$, $\delta$) | Tied to $\delta$ | Deterministic |
| Noll+2009 | Single | 4 ($A_V$, $\delta$, $E_b$, $\Delta\lambda$) | Free | None |
| Salim+2018 | Single | 3 ($A_V$, $\delta$, $E_b$) | Free | None |

## Advantages of TEA

1. **Single component**: No need to separate young/old stellar populations,
   avoiding the age-dependent weighting issue.
2. **Three parameters**: Fewer than Noll+2009 (4 params) while being more
   flexible than Calzetti (1 param) or Kriek-Conroy (2 params).
3. **Physically motivated**: Calibrated against radiative transfer simulations
   with realistic galaxy geometries, not just empirical fitting.
4. **Flexible bump**: The UV bump strength is a free parameter, not tied to the
   slope.

## Implementation Notes for tengri

1. **Formula structure**: The TEA formula is fully analytic and smooth, making it
   trivially compatible with JAX autodiff. The `erf` function is available in
   `jax.scipy.special.erf`.

2. **Parameter mapping**: Map to tengri's internal dust parameters:
   - `dust_av` $\to A_V$
   - `dust_slope` $\to p$ (note: tengri's existing `dust_slope` is the
     power-law index $n$ from Charlot & Fall; for TEA this would be a new
     parameter)
   - `dust_bump` $\to b_{\rm UV}$

3. **Integration with existing code**: TEA could be offered as an alternative
   attenuation law alongside the existing Charlot & Fall / Kriek-Conroy options.
   Since it is single-component, the birth-cloud vs ISM split would not be
   needed.

4. **Prior ranges**: The recommended priors from the paper are uniform on the
   ranges listed above. For $p$, a narrower range of $[-2, -0.2]$ covers most
   physically realistic galaxies.

5. **Wavelength units**: The formula assumes $\lambda$ in Angstroms. Ensure
   consistent unit conversion in the implementation.
