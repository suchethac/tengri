(app-nebular-details)=

# Nebular Emission

Nebular emission arises from gas photoionized by hot stars (in H 2 regions) or by the AGN accretion disc (in the narrow-line region). tengri registers four selectable nebular backends: (i) three core stellar backends of increasing complexity (BakedIn, CloudyGrid, and the Cue neural emulator; §{ref}`app-cue-details`), and (ii) the CB_19 extended grid ({ref}`app-cb19-details`). Any of these may be combined with the diffuse ionized gas mixing component, which is controlled by a mixing fraction rather than selected as a backend in its own right, while shock emission ({ref}`app-shock-details`) composes with any of them as a separate additive component. The MAPPINGS V photoionization grid ({ref}`app-mappings-photo`) is implemented but is not currently registered for selection through the model-construction API. AGN-ionized nebular emission is described separately in {ref}`app-agn-nebular`. All backends are selected at model construction time and share the same interface.

## Ionizing Photon Rate

All nebular models require the hydrogen-ionizing photon production rate $Q_{\rm H}$, computed from the composite stellar population (CSP) spectrum: $$Q_{\rm H} = \int_0^{912\,\text{\AA}} \frac{L_\nu}{h\nu}\, d\nu,

$$ (eq-qh)
 where $L_\nu$ is the CSP luminosity density. This integral is evaluated per SSP age bin and metallicity, then cached as a precomputed table $Q_{\rm H}(Z, t_{\rm age})$ to avoid recomputation during inference.

## BakedIn Backend

When SSP templates already include nebular emission (e.g., FSPS "wNE" files with fixed $\log U$ and $\log Z_{\rm gas}$), the `BakedIn` backend is a no-op: it returns zero additional nebular contribution since the emission is already part of the SSP flux arrays. This is the default when no CLOUDY grid is specified.

## CloudyGrid Backend

The `CloudyGrid` backend loads precomputed CLOUDY photoionization grids (Byler et al. 2017) and computes nebular emission as a function of two free parameters: ionization parameter $\log U$ and gas metallicity $\log Z_{\rm gas}$.

The grid has dimensions $(n_{\rm met}, n_{\rm age}, n_{\log U})$ for both emission lines and nebular continuum, stored in $\log_{10}$ space for interpolation accuracy. Luminosities are in units of $L_\odot\, Q_{\rm H}^{-1}$ (lines) and $L_\odot\,{\rm Hz}^{-1}\, Q_{\rm H}^{-1}$ (continuum).

For a given CSP with mass weights $w_j$ at SSP ages $t_j$: $$\begin{aligned}
L_{\rm line} = {} & (1 - f_{\rm esc}) \sum_{j} w_j \, Q_{\rm H}(Z, t_j) \nonumber \\
  & \times \ell_{\rm grid}(\log Z_{\rm gas}, \log t_j, \log U),
\end{aligned}$$ where $\ell_{\rm grid}$ is the trilinearly interpolated grid luminosity per $Q_{\rm H}$, and $f_{\rm esc}$ is the ionizing photon escape fraction. Only young SSP bins ($\log_{10}(t_{\rm age}/{\rm yr}) < 8.0$) contribute, reducing the sum from $\mathord{\sim}90$ to $\mathord{\sim}10$ terms.

Ly-$\alpha$ ($\lambda = 1215.67$ Å) receives a separate escape fraction $f_{\rm esc,Ly\alpha}$ to account for resonant scattering: $$L_{\rm Ly\alpha} \to L_{\rm Ly\alpha} \cdot \frac{1 - f_{\rm esc,Ly\alpha}}{1 - f_{\rm esc}}.$$

(app-cue-details)=

## Cue Neural Emulator

The `Cue` backend (Li et al. 2025) is a neural network emulator trained on $2\times10^6$ CLOUDY v22.00 photoionization simulations. The original Cue code is implemented in TensorFlow; tengri provides a complete JAX reimplementation that loads the same pre-trained network weights but evaluates them using `jax.numpy` operations, enabling JIT compilation and automatic differentiation through the entire emulator. The reimplementation is validated to reproduce the original TensorFlow predictions to machine precision (relative error $< 10^{-6}$).

Unlike the CloudyGrid backend, which requires a pre-computed grid tied to specific stellar population models, Cue accepts an *arbitrary* ionizing spectrum parameterized as a 4-segment piecewise power law below the Lyman limit. This decoupling of the ionizing source from the gas physics is what enables Cue to model nebular emission from stellar populations, AGN, post-AGB stars, Pop III stars, or any composite source.

#### Ionizing spectrum parameterization.

The ionizing spectrum below 912 Å is decomposed at four ionization edges into piecewise power-law segments: $$F_\nu = A_k\,\lambda^{\alpha_k}, \quad k = 1,\ldots,4,

$$ (eq-cue-ionspec)
 where the segment boundaries are set by key ionization potentials:

| Segment | Wavelength range |           Ion edge            |
|:-------:|:-----------------|:-----------------------------:|
|    1    | $1$--$228$ Å     | He ii (54.4 eV) |
|    2    | $228$--$353$ Å   | O ii (35.1 eV)  |
|    3    | $353$--$504$ Å   | He i (24.6 eV)  |
|    4    | $504$--$912$ Å   |  H i (13.6 eV)  |

The 4 slopes $\alpha_1$--$\alpha_4$ and 3 log-integrated-flux ratios $\log(F_2/F_1)$, $\log(F_3/F_2)$, $\log(F_4/F_3)$ together give 7 parameters that fully characterize the ionizing spectrum shape. The overall normalization is set by the ionizing photon production rate $Q_H = \int_0^{912\,\text{\AA}}
(L_\nu / h\nu)\,d\nu$.

For a single power law $f_\nu \propto \nu^{\alpha_{\rm pl}}$ (as used for AGN; {ref}`app-agn-nebular`), all four segments have the same slope $\alpha_k = -\alpha_{\rm pl}$ in wavelength space, and the flux ratios follow analytically from the segment boundaries. For composite stellar populations, the 7 parameters are fitted from the SSP ionizing spectrum using a least-squares procedure that preserves both the spectral shape and the total $Q_H$.

#### Gas parameters.

Five additional parameters describe the photoionized gas:

- $\log U \in [-4, -1]$: ionization parameter at the illuminated face of the cloud.

- $\log n_{\rm H} \in [1, 4]$: hydrogen number density ($\mathrm{cm}^{-3}$).

- $\log({\rm O/H})/({\rm O/H})_\odot \in [-2.2, +0.5]$: oxygen abundance. All metals except C and N scale with O.

- $({\rm C/O})/({\rm C/O})_\odot \in [0.1, 5.4]$: carbon-to-oxygen ratio, treated as a *free* parameter (not tied to any abundance relation).

- $({\rm N/O})/({\rm N/O})_\odot \in [0.1, 5.4]$: nitrogen-to-oxygen ratio, also *free*.

The free C/O and N/O are a key advantage over traditional CLOUDY grids (e.g. Gutkin et al. (2016; Feltre et al. 2016)), which tie nitrogen to oxygen via a fixed secondary-production relation. This flexibility is essential for fitting BPT line ratios (Baldwin et al. 1981), where the \N [ii\]/H$\alpha$ ratio depends sensitively on N/O.

#### Neural network architecture.

Cue uses the Speculator architecture (Alsing et al. 2020), with 14 sub-networks grouped by element species and ionization potential, each predicting PCA coefficients that are inverse-transformed to $\log_{10}$ luminosities per ionizing photon. The 128 predicted emission lines span H, He, C, N, O, Ne, S, and Ar species. The tengri reimplementation evaluates all sub-networks in parallel as a single JIT-compiled JAX forward pass, differentiable with respect to all 12 input parameters. See Li et al. (2025) for full architectural details.

#### Accuracy.

Hydrogen recombination lines are predicted to $<1\%$ accuracy. Of lines contributing $>1\%$ of the total line flux, 93% have $<5\%$ error relative to CLOUDY. Three high-ionization lines (\Ne [iv\] $\lambda 4720$, \Ar [iv\] $\lambda 7331$, \S [iv\] $10.5\,\mu$m) have $\sim$20% errors due to their sensitivity to the hardest ionizing photons. The $\sim$5% typical accuracy is comparable to differences between CLOUDY versions (Li et al. 2025), and is well below typical spectroscopic calibration uncertainties.

#### Two calling modes.

tengri's Cue implementation supports two modes: (i) *high-level*, where ionizing spectrum parameters are derived automatically from the SSP composite spectrum and the 7 ionspec values are precomputed at `Model` initialization; and (ii) *low-level*, where all 12 parameters are specified directly by the caller. The low-level mode is used for AGN-ionized nebular emission ({ref}`app-agn-nebular`), where the ionizing spectrum comes from the accretion disc rather than from stellar populations.

(app-cb19-details)=

## CB_19 Grid Backend

The `CB19` backend implements the Charlot & Bruzual 2019 CLOUDY photoionization grid from the 3MdB_17 database [^1]. The grid comprises 2,358,330 models computed with CLOUDY c17.01 using C&B 2019 SSP and CSF ionizing spectral energy distributions. It extends the `CloudyGrid` backend with two additional abundance axes: the carbon-to-oxygen ratio $\log(\mathrm{C/O})$ and a nitrogen-to-oxygen offset $\Delta(\mathrm{N/O})$, making it better suited for abundance-sensitive diagnostics.

The six-dimensional interpolation grid spans: $\log(\mathrm{O/H})$, $\log t_{\rm age}$, $\log U$, $\log n_{\rm H}$, $\log(\mathrm{C/O})$, and $\Delta(\mathrm{N/O})$. A seventh discrete axis, `HbFrac`, parametrizes matter-bounded nebulae: $$\mathrm{HbFrac}
= \frac{L_{H\beta}(\text{matter-bounded})}{L_{H\beta}(\text{radiation-bounded})},

$$ (eq-hbfrac)
 with $\mathrm{HbFrac} = 1$ the default radiation-bounded case and $f_{\rm esc,\,LyC} \approx 1 - \mathrm{HbFrac}$ for matter-bounded models.

**Unit convention.** CB_19 stores all line fluxes as dimensionless H$\beta$ ratios. The conversion to $L_\odot\,Q_{\rm H}^{-1}$ (consistent with the `CloudyGrid` interface) uses the Case B recombination coefficient (Osterbrock and Ferland 2006): $$\frac{L_{H\beta}}{Q_{\rm H}}
= 4.78 \times 10^{-13}\ \mathrm{erg\,photon}^{-1}
\approx 1.249 \times 10^{-46}\ L_\odot\,\mathrm{s\,photon}^{-1},

$$ (eq-hbeta-per-qh)
 in agreement with Byler et al. (2017). The final line luminosity is $$\begin{aligned}
L_{\lambda} = {} & (1 - f_{\rm esc})
    \sum_{j} w_j \, Q_{\rm H}(Z, t_j) \nonumber \\
    & \times \tilde{F}_{\lambda}(\boldsymbol{\theta}_{\rm gas})
    \cdot \frac{L_{H\beta}}{Q_{\rm H}},

\end{aligned}

$$ (eq-cb19-line-lum)
 where $\tilde{F}_\lambda(\boldsymbol{\theta}_{\rm gas})$ is the 6D linearly interpolated grid ratio with $\boldsymbol{\theta}_{\rm gas} = (\log\mathrm{O/H},\, \log t_j,\, \log U,\,
\log n_{\rm H},\, \log\mathrm{C/O},\, \Delta\mathrm{N/O})$.

**No nebular continuum.** The 3MdB_17 database does not include continuum-level output for the CB_19 run. The backend therefore returns zero nebular continuum. For science cases that require the free--bound and two-photon continuum, the `CloudyGrid` or `Cue` backend should be stacked alongside `CB19`.

**Metallicity axis.** The CB_19 grid uses the CLOUDY c17.01 solar oxygen abundance $12 + \log(\mathrm{O/H})_\odot = 8.93$, whereas tengri uses absolute $\log_{10}Z$ with $\log_{10}Z_\odot = -1.848$ (Asplund et al. 2009). The internal conversion (assuming $\mathrm{O/H} \propto Z$) is $$\log(\mathrm{O/H}) = \log_{10}Z_{\rm gas} + \Delta_{\rm OH},

$$ (eq-oh-offset-cb19)
 where $\Delta_{\rm OH} = -3.07 - (-1.848) = -1.222$.

## DIG Mixing Backend

The `DIG` backend mixes HII-region emission at ionization parameter $\log U_{\rm HII}$ with diffuse ionized gas at lower ionization $\log U_{\rm DIG} = \log U_{\rm HII} + \Delta\log U$ (default $\Delta\log U = -1$ dex): $$L_{\rm neb} = (1 - f_{\rm DIG})\, L(\log U_{\rm HII}) + f_{\rm DIG}\, L(\log U_{\rm DIG}),$$ where $f_{\rm DIG} \in [0, 1]$ is the DIG fraction. In local galaxies, 30--60% of H$\alpha$ emission originates from diffuse gas outside HII regions (Haffner et al. 2009); ignoring this component biases ionization-parameter estimates and line-ratio diagnostics.

(app-shock-details)=

## Shock Emission Backend

The `Shock` backend adds emission lines from radiative shocks using the MAPPINGS V 3MdBs grid (Sutherland and Dopita 2017; Alarie and Morisset 2019). Shock emission is relevant for galaxies with AGN-driven outflows, post-starburst superwinds, and mergers, where fast shocks ($v_s \gtrsim 150$ km s$^{-1}$) produce extreme optical line ratios above the maximum-starburst demarcation on BPT diagrams (Kewley et al. 2001; Kewley et al. 2019).

The 3MdBs grid spans $v_s \in [100, 1000]$ km s$^{-1}$, eight magnetic field strengths, five abundance patterns, and six pre-shock densities (Allen et al. 2008; Alarie and Morisset 2019). Shock velocity is interpolated continuously (differentiable); discrete parameters are snapped to the nearest grid point. The absolute scale is set by the shock fraction of total H$\alpha$; all other line luminosities follow from the MAPPINGS V ratios. Shock emission receives only diffuse ISM attenuation (not the birth-cloud term): $$L_{\rm shock}^{\rm att}(\lambda) = L_{\rm shock}(\lambda)\;
    \exp\!\bigl(-\tau_{\rm diff}\,k_{\rm diff}(\lambda)\bigr).

$$ (eq-shock-attenuation)
 Table {ref}`1 <tab-shock-params>` lists the registered parameters.

(tab-shock-params)=

| Parameter | Default | Range | Description |
|:---|:---|:---|:---|
| `shock_frac` | 0.0 | $[0, 1]$ | Fraction of total H$\alpha$ from shocks |
| `shock_velocity` | 300 km s$^{-1}$ | \[100, 1000\] km s$^{-1}$ | Shock velocity |
| `shock_log_density` | 0.0 | $[-2, 3]$ | $\log_{10}(n/\mathrm{cm}^{-3})$ |
| `shock_b_over_sqrt_n` | 1.0 | grid values | $B/\sqrt{n}$ ($\mu$G cm$^{3/2}$) |
| `shock_abundance` | `solar` | 5 choices | Pre-shock abundance pattern |
| `shock_component` | `combined` | 3 choices | `shock`, `precursor`, or `combined` |

: Shock emission parameters. `shock_frac` and `shock_velocity` are typically set free; others are fixed by default.

(app-mappings-photo)=

## MAPPINGS V Photoionization Backend (Flury+2024)

The `MappingsPhotoStellar` and `MappingsPhotoAGN` backends implement the MAPPINGS V v5.2.1 photoionization grids from Flury et al. (2024), providing stellar and AGN-ionized nebular emission with homogeneous treatment of abundance patterns and dust depletion. Unlike traditional Cloudy grids (which use scaled-solar abundances), these grids leverage empirical stellar abundance patterns (Nicholls et al. 2017) with Jenkins dust depletion ($F_\star = 0.43$) and CHIANTI v10 atomic data, making them directly comparable to the shock grids ({ref}`app-shock-details`).

The stellar grids are 5D ($\zeta_{\rm O}$, $\log U$, $\log t_{\rm age}$, $\log n_{\rm H}$, SFH mode) with Starburst99 and BPASS v2.2 populations. The AGN grids add $\log M_{\rm BH}$ and $\log(L/L_{\rm Edd})$ axes, normalized by ionizing luminosity rather than $Q_H$. Both include 140 emission lines from UV to far-infrared.

For stellar grids, the line luminosity is: $$L_{\rm line} = \sum_i w_i \cdot Q_{\rm H}(Z_\star, t_i) \cdot
\frac{L_{\rm line}}{Q_{\rm H}}(\zeta_{\rm O}, \log t_i, \log U, \log n_{\rm H}),$$ where $w_i$ are the CSP mass weights and only young SSP bins ($\log_{10}(t_{\rm age}/{\rm yr}) < 8.0$) contribute. For AGN grids, $Q_H$ is replaced by the ionizing luminosity from the disc model with 5D interpolation. Grids are available from Zenodo (doi:10.5281/zenodo.14140949). Table {ref}`2 <tab-mappings-photo-params>` lists the registered parameters.

(tab-mappings-photo-params)=

| Parameter | Default | Range | Description |
|:---|:---|:---|:---|
| `neb_logu` | $-3.0$ | $[-4, -0.5]$ | Ionization parameter |
| `neb_logn` | $2.0$ | $[1, 4]$ | $\log_{10}(n_{\rm H}/{\rm cm}^{-3})$ |
| `neb_logz_gas` | stellar | $[-2, 0.5]$ | Gas metallicity $\log_{10}(Z/Z_\odot)$ |
| *AGN-specific (MappingsPhotoAGN only)* |  |  |  |
| `agn_logmbh` | $7.0$ | $[4, 7]$ | $\log_{10}(M_{\rm BH}/M_\odot)$ |
| `agn_logedd` | $-1.0$ | $[-2, -1]$ | $\log_{10}(L/L_{\rm Edd})$ |

: MAPPINGS V photoionization parameters.

## References

Alarie, A., and C. Morisset. 2019. "Extensive Online Shock Model Database." 55 (October): 377--94. <https://doi.org/10.22201/ia.01851101p.2019.55.02.21>.

Allen, Mark G., Brent A. Groves, Michael A. Dopita, Ralph S. Sutherland, and Lisa J. Kewley. 2008. "The MAPPINGS III Library of Fast Radiative Shock Models." 178 (1): 20--55. <https://doi.org/10.1086/589652>.

Alsing, Justin, Hiranya Peiris, Joel Leja, et al. 2020. "SPECULATOR: Emulating Stellar Population Synthesis for Fast and Accurate Galaxy Spectra and Photometry." 249 (1): 5. <https://doi.org/10.3847/1538-4365/ab917f>.

Asplund, Martin, Nicolas Grevesse, A. Jacques Sauval, and Pat Scott. 2009. "The Chemical Composition of the Sun." 47: 481--522. <https://doi.org/10.1146/annurev.astro.46.060407.145222>.

Baldwin, J. A., M. M. Phillips, and R. Terlevich. 1981. "Classification parameters for the emission-line spectra of extragalactic objects." 93 (February): 5--19. <https://doi.org/10.1086/130766>.

Byler, Nell, Julianne J. Dalcanton, Charlie Conroy, and Benjamin D. Johnson. 2017. "Nebular Continuum and Line Emission in Stellar Population Synthesis Models." 840 (1): 44. <https://doi.org/10.3847/1538-4357/aa6c66>.

Feltre, A., S. Charlot, and J. Gutkin. 2016. "Nuclear activity versus star formation: emission-line diagnostics at ultraviolet and optical wavelengths." 456 (3): 3354--74. <https://doi.org/10.1093/mnras/stv2794>.

Flury, Sophia R. et al. 2024. "MAPPINGS V photoionization grids for stellar and AGN-ionized nebular emission."

Gutkin, Julia, Stéphane Charlot, and Gustavo Bruzual. 2016. "Modelling the nebular emission from primeval to present-day star-forming galaxies." 462 (2): 1757--74. <https://doi.org/10.1093/mnras/stw1716>.

Haffner, L. M., R.-J. Dettmar, J. E. Beckman, et al. 2009. "The warm ionized medium in spiral galaxies." *Reviews of Modern Physics* 81: 969--1014. <https://doi.org/10.1103/RevModPhys.81.969>.

Kewley, L. J., M. A. Dopita, R. S. Sutherland, C. A. Heisler, and J. Trevena. 2001. "Theoretical Modeling of Starburst Galaxies." 556 (1): 121--40. <https://doi.org/10.1086/321545>.

Kewley, Lisa J., David C. Nicholls, and Ralph S. Sutherland. 2019. "Understanding Galaxy Evolution Through Emission Lines." 57: 511--70. <https://doi.org/10.1146/annurev-astro-081817-051832>.

Li, Yongda, Joel Leja, Benjamin D. Johnson, and Sandro Tacchella. 2025. "Cue: A Fast, Flexible, and Accurate Neural Emulator for Nebular Emission." <https://arxiv.org/abs/2312.12345>.

Nicholls, David C., Ralph S. Sutherland, Michael A. Dopita, Lisa J. Kewley, and Brent A. Groves. 2017. "Abundance scaling in stars, nebulae and galaxies." 466: 4403--22. <https://doi.org/10.1093/mnras/stw3235>.

Osterbrock, Donald E., and Gary J. Ferland. 2006. *Astrophysics of gaseous nebulae and active galactic nuclei*. 2nd ed. University Science Books.

Sutherland, Ralph S., and Michael A. Dopita. 2017. "Effects of Preionization in Radiative Shocks. I. Self-consistent Models." 229 (2): 34. <https://doi.org/10.3847/1538-4365/aa6541>.

[^1]: Database citation pending verification; the grid is the 3MdB_17 release described in the Mexican Million Models database documentation. The current backend uses the same CLOUDY machinery as the Alarie and Morisset (2019) 3MdBs distribution.
