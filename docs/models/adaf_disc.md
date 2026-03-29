# ADAF (Advection-Dominated Accretion Flow) Spectral Model

## Citations

Mahadevan, R. 1997, ApJ, 477, 585.
"Scaling Laws for Advection-dominated Flows: Applications to Low-Luminosity
Galactic Nuclei."

Nemmen, R. S., Storchi-Bergmann, T., & Eracleous, M. 2014, MNRAS, 438, 2804.
"Spectral models for low-luminosity active galactic nuclei in LINERs: the role
of advection-dominated accretion and jets."

Yuan, F. & Narayan, R. 2014, ARA&A, 52, 529.
"Hot Accretion Flows Around Black Holes." (Review)

## Overview

An advection-dominated accretion flow (ADAF), also called a radiatively
inefficient accretion flow (RIAF), describes accretion onto a black hole at low
accretion rates ($\dot{m} \ll \dot{m}_{\rm crit}$) where the gas cannot cool
efficiently. The gravitational energy is stored as thermal energy in the ions and
advected inward rather than radiated. The resulting SED is fundamentally
different from the standard thin disc (Shakura-Sunyaev), with emission dominated
by synchrotron, bremsstrahlung, and inverse Compton processes in a hot,
two-temperature plasma.

## Two-Temperature Plasma

In an ADAF, viscous dissipation primarily heats the **ions**, which transfer
energy to electrons via Coulomb collisions. Because the coupling is weak at low
densities, the ion and electron temperatures decouple:

$$
T_i \sim 10^{12}\;\text{K} \quad (\text{virial temperature})
$$

$$
T_e \sim 10^{9}\text{--}10^{10}\;\text{K}
$$

The electron temperature $T_e$ is the key parameter for the emitted spectrum.
Mahadevan (1997) showed that $T_e$ is **relatively insensitive** to $M$,
$\dot{M}$, and other model parameters.

### Electron temperature scaling (Mahadevan 1997)

For the self-similar ADAF solution, the equilibrium electron temperature is
approximately:

$$
T_e \approx \frac{3.5 \times 10^9}{\left[1 + 0.5\,(\dot{m}/\dot{m}_c)^{0.5}\right]}\;\text{K}
$$

where $\dot{m}_c$ is a critical accretion rate and $\dot{m} = \dot{M}/\dot{M}_{\rm Edd}$.

## Spectral Components

The ADAF spectrum spans from radio ($\sim 10^8$ Hz) to X-ray ($\sim 10^{20}$
Hz) frequencies, composed of three main emission processes:

### 1. Synchrotron emission (radio to sub-mm)

Thermal electrons gyrating in the magnetic field produce synchrotron radiation.
The spectrum is self-absorbed at low frequencies, producing the characteristic
$\nu^{1/3}$ radio spectral index:

$$
L_\nu^{\rm syn} \propto \nu^{1/3} \quad (\nu < \nu_{\rm peak}^{\rm syn})
$$

Above the self-absorption frequency, the spectrum turns over exponentially.

**Peak synchrotron frequency** (Mahadevan 1997 scaling):

$$
\nu_{\rm peak}^{\rm syn} \propto M^{-1/2}\, \dot{m}^{1/2}\, T_e^2\, B
$$

where $B$ is the magnetic field strength. For typical parameters:

$$
\nu_{\rm peak}^{\rm syn} \sim 10^{11}\text{--}10^{12}\;\text{Hz}
$$

### 2. Bremsstrahlung (soft X-ray)

Free-free emission from the hot electrons produces a thermal spectrum:

$$
L_\nu^{\rm brem} \propto n_e^2\, T_e^{-1/2}\, e^{-h\nu/k_B T_e}
$$

This component dominates at intermediate frequencies ($\sim 10^{14}$--$10^{17}$
Hz for typical parameters).

### 3. Inverse Compton scattering (X-ray)

Synchrotron photons are upscattered by the hot electrons (synchrotron
self-Compton, SSC):

$$
\nu_{\rm IC} \sim 4\, \gamma_e^2\, \nu_{\rm syn}
$$

$$
L_\nu^{\rm IC} \propto L_\nu^{\rm syn} \times \tau_{\rm es}
$$

where $\tau_{\rm es}$ is the electron scattering optical depth and $\gamma_e
\sim k_B T_e / m_e c^2$ is the electron Lorentz factor. Multiple Compton
scatterings produce a sequence of "humps" in the spectrum.

## Scaling Laws (Mahadevan 1997)

The key result is that the ADAF SED can be predicted from analytical scaling
relations without detailed numerical calculations:

### Total luminosity

$$
L_{\rm bol,ADAF} \approx \begin{cases}
0.44\, (\dot{m}/0.01)\, \epsilon\, \dot{M} c^2 & \dot{m} > 7.5 \times 10^{-6} \\
6.3 \times 10^{-5}\, \epsilon\, \dot{M} c^2 & \dot{m} \lesssim 7.5 \times 10^{-6}
\end{cases}
$$

where $\epsilon$ is the radiative efficiency.

### Radio-X-ray correlation

A strong testable prediction: radio and X-ray luminosities are correlated:

$$
L_X \propto L_R^p
$$

with $p \sim 0.6$--0.7 for ADAFs, consistent with the fundamental plane of
black hole activity.

### Critical accretion rate

The ADAF solution exists below a critical accretion rate:

$$
\dot{m}_{\rm crit} \sim \alpha^2 \approx 0.01\text{--}0.1
$$

where $\alpha$ is the Shakura-Sunyaev viscosity parameter. Above this rate, the
flow transitions to a standard thin disc.

## Truncated Disc Geometry

In many LLAGN, the accretion flow has a **truncated thin disc + inner ADAF**
structure:

$$
\text{Thin disc:} \quad R_{\rm tr} < r < R_{\rm out}
$$

$$
\text{ADAF:} \quad R_{\rm ISCO} < r < R_{\rm tr}
$$

The **truncation radius** $R_{\rm tr}$ controls the transition:

| $R_{\rm tr}$ | Regime |
|---------------|--------|
| $R_{\rm tr} \to R_{\rm ISCO}$ | Standard thin disc (high $\dot{m}$) |
| $R_{\rm tr} \sim 10$--$100\, R_g$ | Hybrid: ADAF + truncated disc |
| $R_{\rm tr} \to R_{\rm out}$ | Pure ADAF (very low $\dot{m}$) |

## Model Parameters (Nemmen+2014)

| Parameter | Symbol | Typical range | Description |
|-----------|--------|---------------|-------------|
| Black hole mass | $M$ | $10^6$--$10^{10}\, M_\odot$ | Gravitational scaling |
| Accretion rate | $\dot{m}$ | $10^{-6}$--$10^{-2}$ | In Eddington units |
| Viscosity | $\alpha$ | 0.1--0.3 | Shakura-Sunyaev parameter |
| Magnetic pressure | $\beta$ | 0.5--0.95 | Gas-to-total pressure ratio |
| Electron heating | $\delta$ | 0.01--0.5 | Fraction of dissipation heating electrons |
| Adiabatic index | $\gamma$ | 1.5 | Thermodynamic index |
| Wind index | $s$ | 0--0.4 | Mass outflow: $\dot{M}(R) = \dot{M}_0 (R/R_0)^s$ |
| Outer radius | $R_0$ | $10^3$--$10^5\, R_g$ | Where $\dot{M}$ is measured |

### Mass-loss via outflows

The accretion rate varies with radius due to winds:

$$
\dot{M}(R) = \dot{M}_0 \left(\frac{R}{R_0}\right)^s
$$

where $s$ is the wind index. For $s = 0$, there is no outflow (classical ADAF).
For $s \sim 0.3$, significant mass loss occurs, reducing the density profile:

$$
\rho(R) \propto R^{-3/2 + s}
$$

## Comparison to Related Models

| Model | Geometry | Emission | Application |
|-------|----------|----------|-------------|
| **ADAF/RIAF** | Hot, geometrically thick | Synchrotron + brems + IC | LLAGN ($L < 0.01 L_{\rm Edd}$) |
| Shakura-Sunyaev thin disc | Cold, geometrically thin | Thermal blackbody | Luminous AGN |
| Slim disc | Warm, moderately thick | Modified thermal | Near-Eddington |
| Jet (synchrotron) | Collimated outflow | Non-thermal synchrotron | Radio-loud AGN |

## Implementation Notes for tengri

1. **Semi-analytical approach**: Follow Yuan et al. (2005) / Nemmen et al.
   (2014). Solve for the radial structure (density, temperature, velocity) using
   the self-similar ADAF equations, then compute emission at each radius.

2. **Spectral computation**: For each annulus at radius $R$:
   - Compute $T_e(R)$, $n_e(R)$, $B(R)$ from the dynamical solution
   - Evaluate synchrotron emissivity (including self-absorption)
   - Evaluate bremsstrahlung emissivity
   - Compute Compton scattering iteratively

3. **Differentiability**: The main challenge is making the spectral calculation
   differentiable. The synchrotron self-absorption turnover and Compton
   scattering involve integrals that need careful treatment in JAX.

4. **Parameter reduction**: For SED fitting, fix $\alpha$, $\beta$, $\gamma$ to
   standard values and fit only $M$, $\dot{m}$, $\delta$, and $s$. This gives a
   4-parameter ADAF model.

5. **Hybrid model**: Combine with the existing AGN disc model in
   `src/tengri/models/agn/` for a truncated disc + ADAF geometry. The thin disc
   provides the UV/optical "big blue bump" while the ADAF provides the radio and
   hard X-ray emission.

6. **Reference implementation**: The `riaf-sed` Python package by Nemmen
   (github.com/rsnemmen/riaf-sed) provides a reference implementation. It uses
   OpenMP-parallelized Fortran for the dynamical solution.
