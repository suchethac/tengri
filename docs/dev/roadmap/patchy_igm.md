# Patchy IGM Transmission Model

## Citations

Miralda-Escude, J. 1998, ApJ, 501, 15.
"Reionization of the Intergalactic Medium and the Damping Wing of the
Gunn-Peterson Trough."

Mason, C. A., Treu, T., Dijkstra, M., et al. 2018, ApJ, 856, 2.
"The Universe Is Reionizing at z ~ 7: Bayesian Inference of the IGM Neutral
Fraction Using Lya Emission from Galaxies."

## Overview

At redshifts $z \gtrsim 6$, the intergalactic medium (IGM) transitions from
mostly ionized to significantly neutral as we approach the Epoch of Reionization
(EoR). Neutral hydrogen in the IGM imprints two distinct absorption features on
background source spectra:

1. **Gunn-Peterson trough**: Complete absorption blueward of Ly$\alpha$ due to
   even tiny neutral fractions ($x_{\rm HI} \gtrsim 10^{-4}$).
2. **Damping wing**: Smooth absorption extending **redward** of Ly$\alpha$ due
   to the Lorentzian wing of the Ly$\alpha$ cross-section, requiring substantial
   neutral fractions ($x_{\rm HI} \gtrsim 0.1$).

During reionization, the IGM is **patchy**: ionized bubbles surround galaxies
while neutral regions persist between them. This patchiness introduces
stochastic variations in the IGM transmission.

## Gunn-Peterson Optical Depth (Miralda-Escude 1998)

The Gunn-Peterson optical depth for a uniform IGM at redshift $z_s$ is:

$$
\tau_{\rm GP} = \frac{\pi e^2}{m_e c}\, \frac{f_\alpha\, n_{\rm HI}(z_s)}{H(z_s)\, \nu_\alpha}
$$

Numerically, for a fully neutral IGM:

$$
\tau_{\rm GP} \approx 7.16 \times 10^5 \left(\frac{1 + z_s}{10}\right)^{3/2}
$$

where:
- $f_\alpha = 0.4162$ is the Ly$\alpha$ oscillator strength
- $\nu_\alpha = 2.47 \times 10^{15}$ Hz is the Ly$\alpha$ frequency
- $\Lambda_\alpha = 6.25 \times 10^8$ s$^{-1}$ is the Ly$\alpha$ decay rate
- $n_{\rm HI}$ is the neutral hydrogen number density

Even a neutral fraction of $x_{\rm HI} \sim 10^{-4}$ produces $\tau \gg 1$,
making the Gunn-Peterson trough essentially black.

## Damping Wing Profile (Miralda-Escude 1998)

### Voigt profile wing approximation

Far from line center, the Ly$\alpha$ cross-section is dominated by the
Lorentzian (natural broadening) wing:

$$
\sigma(\nu) = \frac{3 \lambda_\alpha^2}{8\pi} \frac{\Lambda_\alpha^2}
{4\pi^2 (\nu - \nu_\alpha)^2 + (\Lambda_\alpha/2)^2}
$$

For $|\nu - \nu_\alpha| \gg \Lambda_\alpha$, this simplifies to:

$$
\sigma(\nu) \approx \frac{3 \lambda_\alpha^2 \Lambda_\alpha^2}
{32\pi^3 (\nu - \nu_\alpha)^2}
$$

### Optical depth redward of Ly$\alpha$

For a source at redshift $z_s$ observed at frequency $\nu_{\rm obs}$ (redward of
Ly$\alpha$), the damping wing optical depth from a uniform neutral IGM extending
from $z_{\rm start}$ to $z_s$ is:

$$
\tau_{\rm DW}(\nu_{\rm obs}) = \frac{\tau_{\rm GP}}{2\pi}
\frac{\Lambda_\alpha}{\nu_\alpha}
\int_{z_{\rm start}}^{z_s} \frac{(1+z)^{3/2}\, dz}
{\left[\nu_{\rm obs}(1+z_s)/(\nu_\alpha(1+z)) - 1\right]^2}
$$

For a fully neutral IGM ($x_{\rm HI} = 1$) extending to the source redshift,
the damping wing produces significant absorption out to
$\Delta\lambda / \lambda_\alpha \sim 0.01$--0.05 redward of Ly$\alpha$ (i.e.,
several hundred Angstroms in the rest frame).

### Analytic approximation

Miralda-Escude (1998) derived an analytic form for the damping wing far from
resonance:

$$
\tau_\alpha(\lambda_{\rm obs}) \approx \frac{\tau_{\rm GP}\, R_\alpha}{\pi}
\frac{1}{\Delta_\lambda^2}
$$

where $\Delta_\lambda = (\lambda_{\rm obs} - \lambda_\alpha^{\rm obs})/\lambda_\alpha^{\rm obs}$
and $R_\alpha = \Lambda_\alpha / (4\pi\nu_\alpha)$.

## Patchy Reionization (Mason+2018)

### Bubble model

During reionization, ionized bubbles of characteristic size $R_b$ surround
galaxies. The damping wing absorption depends on:

1. **Neutral fraction** $\bar{x}_{\rm HI}$: Volume-averaged neutral hydrogen
   fraction of the IGM.
2. **Bubble size** $R_b$: Characteristic radius of ionized regions. Larger
   bubbles push the neutral IGM further from the source, reducing damping wing
   absorption.

The transmission through the IGM is:

$$
T_{\rm IGM}(\lambda) = e^{-\tau_{\rm DW}(\lambda)}
$$

where $\tau_{\rm DW}$ now depends on the distance to the nearest neutral patch
rather than being computed from a uniform IGM.

### Miralda-Escude (1998) overlap criterion

Damping wings from neighboring neutral patches overlap when the proper path
length through an ionized region is less than $\sim 1\, h^{-1}$ Mpc. This sets
the scale where individual bubble effects become important.

### Mason+2018 framework

Mason et al. (2018) combined:
- Reionization simulations (semi-numerical, 21cmFAST-style) for the bubble
  geometry
- Empirical ISM Ly$\alpha$ radiative transfer models
- Bayesian inference of $\bar{x}_{\rm HI}$

Key result: $\bar{x}_{\rm HI}(z=7) = 0.59^{+0.11}_{-0.15}$ (68% CI).

### Parameterization for SED fitting

For practical SED fitting, the IGM transmission can be parameterized as:

$$
T_{\rm IGM}(\lambda, z, x_{\rm HI}) = \begin{cases}
0 & \lambda_{\rm obs} < \lambda_\alpha (1+z) \quad \text{(GP trough)} \\
\exp\!\left[-\tau_{\rm DW}(\lambda, z, x_{\rm HI})\right] & \lambda_{\rm obs} > \lambda_\alpha (1+z) \quad \text{(damping wing)}
\end{cases}
$$

The key parameters are:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Neutral fraction | $\bar{x}_{\rm HI}$ | 0 -- 1 | Volume-averaged HI fraction |
| Bubble radius | $R_b$ | 0.1 -- 50 Mpc | Characteristic ionized region size |
| Source redshift | $z_s$ | $>$ 6 | Relevant only during EoR |

## Observational Constraints on $\bar{x}_{\rm HI}$

| Redshift | $\bar{x}_{\rm HI}$ | Method | Reference |
|----------|---------------------|--------|-----------|
| $z \sim 5.5$ | $< 0.06$ | GP trough statistics | Fan+2006 |
| $z \sim 6$ | $\sim 0.04$--0.08 | Ly$\alpha$ fraction | various |
| $z \sim 7$ | $0.59^{+0.11}_{-0.15}$ | Ly$\alpha$ EW distribution | Mason+2018 |
| $z \sim 8$ | $\gtrsim 0.6$ | JWST damping wings | Various JWST |

## Comparison to Existing IGM Models

| Model | Physics | Parameters | Application |
|-------|---------|------------|-------------|
| Inoue+2014 (current tengri) | Mean IGM transmission | $z$ only | $z < 6$ (post-reionization) |
| **Miralda-Escude 1998** | Uniform damping wing | $z$, $x_{\rm HI}$ | EoR, uniform IGM |
| **Mason+2018** | Patchy damping wing | $z$, $x_{\rm HI}$, $R_b$ | EoR, realistic topology |
| Mesinger+2015 (21cmFAST) | Full simulation | Many | Research-grade |

## Implementation Notes for tengri

1. **Current IGM model**: tengri uses `igm_transmission(wave_obs, z)` from
   Inoue+2014, which takes **observed-frame** wavelengths and returns the mean
   IGM transmission. This is appropriate for $z \lesssim 6$.

2. **Extension for EoR**: Add a `patchy_igm_transmission(wave_obs, z, x_HI, R_b)`
   function that:
   - Uses Inoue+2014 for $\tau_{\rm eff}$ blueward of Ly$\alpha$
   - Adds the Miralda-Escude damping wing redward of Ly$\alpha$
   - Parameterizes patchiness via effective $x_{\rm HI}$ and $R_b$

3. **Differentiability**: The damping wing integral can be computed analytically
   (Miralda-Escude 1998 provides closed-form solutions). The exponential
   transmission is trivially differentiable.

4. **Stochastic scatter**: For a full treatment, one would draw from a
   distribution of IGM transmissions at each redshift. For SED fitting, using
   the mean transmission with $x_{\rm HI}$ as a free parameter is a reasonable
   approximation.

5. **Wavelength convention**: Remember that tengri's `igm_transmission` takes
   **observed-frame** wavelengths, while bagpipes' `get_Inoue14_trans` takes
   **rest-frame**. Keep this convention consistent for the damping wing.

6. **Parameter regime**: The damping wing is only relevant at $z \gtrsim 6$.
   For lower redshifts, $x_{\rm HI} \to 0$ and the damping wing vanishes. A
   flag or automatic switching based on redshift would be appropriate.
