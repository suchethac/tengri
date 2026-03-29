# MAGPHYS Dust Emission Model

## Citation

da Cunha, E., Charlot, S., & Elbaz, D. 2008, MNRAS, 388, 1595.
"A simple model to interpret the ultraviolet, optical and infrared emission from galaxies."

## Overview

The MAGPHYS dust emission model is a physically motivated, largely empirical
framework for computing the infrared spectral energy distribution of galaxies.
The central idea is **energy balance**: the total luminosity absorbed by dust in
the UV/optical must equal the total luminosity re-radiated in the infrared. The
model decomposes the dust emission into contributions from **stellar birth
clouds** (BCs) and the **ambient interstellar medium** (ISM), each with distinct
temperature and composition components.

## Two-Component ISM Framework

Stars form inside dense molecular clouds (birth clouds). After a timescale
$t_{\rm esc} \approx 10^7$ yr, they migrate into the diffuse ISM. The total
dust luminosity is:

$$
L_d^{\rm tot} = L_d^{\rm BC} + L_d^{\rm ISM}
$$

where $L_d^{\rm BC}$ is the dust luminosity originating from birth clouds and
$L_d^{\rm ISM}$ is from the ambient ISM.

### Energy balance constraint

$$
L_d^{\rm tot} = \int_0^\infty \left[ L_\lambda^{\rm unatt} - L_\lambda^{\rm att} \right] d\lambda
$$

The total infrared luminosity equals the integrated difference between the
unattenuated and attenuated stellar SED.

## Birth Cloud Dust Components

The dust SED from birth clouds is the sum of **three components**:

$$
L_\lambda^{\rm BC} = L_d^{\rm BC} \left[ \xi_{\rm PAH}^{\rm BC}\, l_\lambda^{\rm PAH}
+ \xi_{\rm MIR}^{\rm BC}\, l_\lambda^{\rm MIR}
+ \xi_W^{\rm BC}\, l_\lambda^W(T_W^{\rm BC}) \right]
$$

where $l_\lambda$ denotes the normalized SED template for each component, and:

| Parameter | Definition | Prior range |
|-----------|-----------|-------------|
| $\xi_{\rm PAH}^{\rm BC}$ | Fractional contribution of PAHs to BC dust luminosity | 0.0 -- 0.25 |
| $\xi_{\rm MIR}^{\rm BC}$ | Fractional contribution of hot MIR continuum to BC dust luminosity | 0.0 -- 0.25 |
| $\xi_W^{\rm BC}$ | Fractional contribution of warm grains in thermal equilibrium | $1 - \xi_{\rm PAH}^{\rm BC} - \xi_{\rm MIR}^{\rm BC}$ |

The constraint $\xi_{\rm PAH}^{\rm BC} + \xi_{\rm MIR}^{\rm BC} + \xi_W^{\rm BC} = 1$ ensures that
the three fractional contributions sum to unity.

### Component 1: PAH emission ($l_\lambda^{\rm PAH}$)

The PAH template is taken from empirical mid-infrared spectra and includes the
well-known emission features at 3.3, 6.2, 7.7, 8.6, 11.3, and 12.7 $\mu$m. da
Cunha et al. adopt the fixed PAH template from Madden et al. (2006), normalized
to unit bolometric luminosity.

### Component 2: Hot mid-infrared continuum ($l_\lambda^{\rm MIR}$)

This represents stochastically heated very small grains (VSGs). It is modeled as
a sum of modified blackbodies over a temperature range:

$$
l_\lambda^{\rm MIR} \propto \sum_T B_\lambda(T)\, \lambda^{-\beta}
$$

with temperatures $T_{\rm MIR}$ in the range **130 -- 250 K** and emissivity
index $\beta = 1$ (for small grains). This component dominates emission at
$\sim$3--30 $\mu$m.

### Component 3: Warm grains in thermal equilibrium ($l_\lambda^W$)

A single-temperature modified blackbody:

$$
l_\lambda^W(T_W^{\rm BC}) = \frac{\lambda^{-\beta}\, B_\lambda(T_W^{\rm BC})}
{\int_0^\infty \lambda^{-\beta}\, B_\lambda(T_W^{\rm BC})\, d\lambda}
$$

| Parameter | Range | Default prior |
|-----------|-------|---------------|
| $T_W^{\rm BC}$ | 30 -- 60 K | Uniform |
| $\beta$ | 1.5 (birth clouds), 2.0 (ISM) | Fixed |

## ISM Dust Components

The ambient ISM dust SED has **four components**:

$$
L_\lambda^{\rm ISM} = L_d^{\rm ISM} \left[ \xi_{\rm PAH}^{\rm ISM}\, l_\lambda^{\rm PAH}
+ \xi_{\rm MIR}^{\rm ISM}\, l_\lambda^{\rm MIR}
+ \xi_W^{\rm ISM}\, l_\lambda^W(T_W^{\rm ISM})
+ \xi_C^{\rm ISM}\, l_\lambda^C(T_C^{\rm ISM}) \right]
$$

The ISM adds a **cold dust** component not present in birth clouds:

| Parameter | Definition | Fixed value |
|-----------|-----------|-------------|
| $\xi_{\rm PAH}^{\rm ISM}$ | PAH fraction in ISM | Fixed to MW cirrus value |
| $\xi_{\rm MIR}^{\rm ISM}$ | Hot MIR fraction in ISM | Fixed to MW cirrus value |
| $\xi_W^{\rm ISM}$ | Warm grain fraction in ISM | Fixed to MW cirrus value |
| $\xi_C^{\rm ISM}$ | Cold grain fraction in ISM | Free (residual) |

The relative proportions of PAHs, hot continuum, and warm dust in the ISM are
**fixed** to reproduce the spectral shape of diffuse cirrus emission in the
Milky Way. Only the cold dust temperature is free.

### Component 4: Cold grains in thermal equilibrium ($l_\lambda^C$)

$$
l_\lambda^C(T_C^{\rm ISM}) = \frac{\lambda^{-\beta}\, B_\lambda(T_C^{\rm ISM})}
{\int_0^\infty \lambda^{-\beta}\, B_\lambda(T_C^{\rm ISM})\, d\lambda}
$$

| Parameter | Range |
|-----------|-------|
| $T_C^{\rm ISM}$ | 15 -- 25 K |
| $\beta$ | 2.0 (fixed) |

## Summary of Temperature Ranges

| Component | Location | Temperature range |
|-----------|----------|-------------------|
| PAH | BC + ISM | Non-equilibrium (template) |
| Hot MIR continuum | BC + ISM | 130 -- 250 K |
| Warm grains | BC | 30 -- 60 K |
| Warm grains | ISM | Fixed (MW cirrus) |
| Cold grains | ISM only | 15 -- 25 K |

## Free Parameters for Dust Emission

| Parameter | Symbol | Range |
|-----------|--------|-------|
| BC PAH fraction | $\xi_{\rm PAH}^{\rm BC}$ | 0 -- 0.25 |
| BC hot MIR fraction | $\xi_{\rm MIR}^{\rm BC}$ | 0 -- 0.25 |
| BC warm temperature | $T_W^{\rm BC}$ | 30 -- 60 K |
| ISM cold temperature | $T_C^{\rm ISM}$ | 15 -- 25 K |
| BC dust fraction | $f_\mu = L_d^{\rm ISM} / L_d^{\rm tot}$ | 0 -- 1 |

The parameter $f_\mu$ controls the fraction of total infrared luminosity
contributed by the ISM (as opposed to birth clouds). Note: some references
define $f_\mu$ as the ISM fraction, others as the BC fraction.

## Modified Blackbody

The general modified blackbody (greybody) spectrum used throughout:

$$
S_\nu \propto \nu^\beta\, B_\nu(T) = \nu^\beta \frac{2 h \nu^3}{c^2}
\frac{1}{e^{h\nu / k_B T} - 1}
$$

In wavelength units:

$$
S_\lambda \propto \lambda^{-(2+\beta)}\, B_\lambda(T)
$$

where $\beta$ is the dust emissivity index.

## Comparison to Related Models

| Model | Dust components | Energy balance | Free dust params |
|-------|----------------|----------------|------------------|
| **da Cunha+2008 (MAGPHYS)** | 4 (PAH + MIR + warm + cold) | Yes | 5 |
| Draine & Li 2007 | Power-law starlight distribution $U$ | No (separate) | 3 ($q_{\rm PAH}$, $U_{\rm min}$, $\gamma$) |
| Dale & Helou 2002 | Single-parameter templates | No | 1 ($\alpha$) |
| Boquien & Salim 2021 (BOSA) | Empirical templates | Optional | 1--2 |

## Implementation Notes for tengri

1. **Energy balance**: The total absorbed UV/optical luminosity from the
   Charlot & Fall (2000) attenuation model must exactly equal the total
   infrared emission. This constraint reduces the effective dimensionality.

2. **Birth cloud vs ISM split**: The attenuation model provides the split
   naturally via $\hat\tau_V$ (birth cloud optical depth) and $\mu$ (ISM-to-BC
   ratio).

3. **Temperature grids**: Pre-compute modified blackbody templates on a grid
   of temperatures at the required wavelength sampling. The MIR continuum
   requires integration over the 130--250 K range.

4. **PAH template**: Load as a fixed template array. No free parameters in the
   PAH spectral shape itself; only the fractional contribution $\xi_{\rm PAH}$
   varies.

5. **Differentiability**: All components are smooth functions of temperature
   and $\xi$ parameters. The modified blackbody is trivially differentiable.
   Use `jax.scipy.special` for any needed special functions.

6. **Parameter mapping**: Map the high-level MAGPHYS parameters to tengri's
   internal dust emission parameters. The $\xi$ parameters can be parameterized
   as a simplex (e.g., via softmax or stick-breaking).
