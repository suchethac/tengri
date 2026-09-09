(app-dust-models)=

# Dust Models

This section details tengri's dust attenuation and emission models.

(app-dust-details)=

## Attenuation Framework

tengri provides a configurable dust attenuation framework with three modes of increasing physical fidelity, selectable at configuration time. All modes share the same attenuation curve registry ({ref}`app-dust-curves`) and support the clumpy geometry of Lower et al. (2022).

### Single-Component (Uniform Screen)

The simplest mode applies a single optical depth $\tau_V$ and attenuation curve $k(\lambda)$ identically to all stellar ages: $$T(\lambda) = f_{\rm obs} + (1 - f_{\rm obs}) \cdot \exp\!\bigl[-\tau_V \cdot k(\lambda)\bigr].

$$ (eq-dust-single)
 Because the transmission is independent of stellar age, it factors completely out of the CSP age sum, reducing the dust computation from $\mathcal{O}(n_{\rm age} \times n_\lambda)$ to $\mathcal{O}(n_\lambda)$.

### Two-Component Framework

The default mode follows Charlot and Fall (2000): stars are attenuated by two physically distinct dust components:

- **Birth cloud** (BC): affects young stars ($t_{\rm age} < t_{\rm birth}$), optical depth $\tau_{\rm BC}$.

- **Diffuse ISM**: affects all stars, optical depth $\tau_{\rm ISM}$.

Birth cloud and diffuse ISM can use different attenuation curves $k_{\rm BC}(\lambda)$ and $k_{\rm ISM}(\lambda)$.

Two evaluation modes are available. The *fast* mode (default) uses a hard age threshold at $t_{\rm birth} = 10$ Myr and exploits a two-CSP decomposition: $$\begin{aligned}
 
F(\lambda) = {} & f_{\rm obs}\, F_{\rm total}
  + (1 - f_{\rm obs})\, T_{\rm ISM} \nonumber \\
  & \times \bigl[T_{\rm BC}\, F_{\rm young}
  + F_{\rm old}\bigr],
\end{aligned}

$$ (eq-dust-fast)
 where $T_{\rm BC} = \exp(-\tau_{\rm BC}\cdot k_{\rm BC})$ and $T_{\rm ISM} = \exp(-\tau_{\rm ISM}\cdot k_{\rm ISM})$ are both 1D vectors of shape $(n_\lambda)$, and $F_{\rm young}$/$F_{\rm old}$ are separate CSP age sums. No $(n_{\rm age} \times n_\lambda)$ intermediate arrays are materialized, yielding the same $\mathcal{O}(n_\lambda)$ cost as the single-component mode.

The *exact* mode replaces the hard threshold with a differentiable sigmoid age transition: $$w(t_{\rm age}) = \sigma\!\left(-\frac{\log_{10} t_{\rm age} - \log_{10} t_{\rm birth}}{\Delta_{\rm trans}}\right),

$$ (eq-dust-sigmoid)
 where $\sigma(x) = 1/(1 + e^{-x})$ is the logistic sigmoid, $t_{\rm birth} = 10^7$ yr (default), and $\Delta_{\rm trans} = 0.3$ dex (default) controls the transition width. The sigmoid spans approximately 5--20 Myr, consistent with the typical timescale for molecular cloud dispersal (Chevance et al. 2020). The total optical depth is then $$\tau(\lambda, t_{\rm age}) = w(t_{\rm age}) \cdot \tau_{\rm BC}\cdot k_{\rm BC}(\lambda) + \tau_{\rm ISM}\cdot k_{\rm ISM}(\lambda),

$$ (eq-tau-total)
 where $k(\lambda)$ is normalized at $\lambda = 5500$ Å. This requires the full $(n_{\rm age} \times n_\lambda)$ outer product for the exponential, but preserves smooth differentiability through the age boundary, which may matter for gradient-based inference when the data constrain the transition region.

### Clumpy Geometry

All three modes support the clumpy-screen geometry of Lower et al. (2022), in which a fraction $f_{\rm obs}$ of sightlines are unobscured: $$T(\lambda, t_{\rm age}) = f_{\rm obs} + (1 - f_{\rm obs}) \cdot \exp\!\bigl[-\tau(\lambda, t_{\rm age})\bigr].

$$ (eq-dust-transmission)
 Setting $f_{\rm obs} = 0$ recovers the standard uniform screen geometry.

(app-dust-curves)=

## Attenuation Curve Library

Table {ref}`1 <tab-dust-curves>` lists the attenuation curves registered in tengri. All curves return $k(\lambda)$ normalized such that $k(5500\,\text{\AA}) = 1$.

(tab-dust-curves)=

| Name | Reference | UV Bump | Free Parameters |
|:---|:---|:--:|:---|
| `power_law` | Charlot & Fall 2000 | No | $n$ (slope) |
| `calzetti` | Calzetti+2000 | No | None ($R_V = 4.05$ fixed) |
| `leitherer02` | Leitherer+2002 | No | None (UV extension 970--1800 Å) |
| `kriek_conroy` | Kriek & Conroy 2013 | Yes | $\delta$, $E_b$ |
| `noll09` | Noll+2009 | Yes | $\delta$, $E_b$ |
| `salim_sbl18` | Salim+2018 (modified Calzetti+L02) | Yes | $\delta$, $E_b$ |
| `smc` | Pei 1992 | No | None |
| `lmc` | Pei 1992 | Weak | None |
| `cardelli` | Cardelli+1989 | Yes | $R_V$ |
| `li08` | Li et al. (2008) | Yes | $c_1$, $c_2$, $c_3$, $c_4$ |
| `salim` | Salim+2018 (DSPS default) | Yes | $\delta$, $E_b$ |
| `tea` | Haskell et al. (2024) | Yes | $p$, $b_{\rm UV}$ |
| `narayanan_z` | Narayanan+2018 ($z$-dependent) | Yes | $z$ (redshift) |
| `conroy2010` | Conroy+2010 (MW+power law) | Yes | $f_{\rm MW}$ |

: Attenuation curves available in tengri. Three dust geometry models (`wg00_shell`, `wg00_cloudy`, `wg00_dusty`; Witt and Gordon (2000)) are also available.

In addition to the curves listed above, tengri also registers older-generation grain-composition laws, `vw07_bc` and `vw07_diff` (Wild et al. (2007) birth-cloud and diffuse-ISM power-laws), `prevot_smc` (the original Prevot+1984 SMC parameterization prior to Pei 1992), and the `wd01_smcbar`, `wd01_mwrv31`, `d03_mwrv31`, and `hd23_mwrv31` grain-model laws, which provide curves tied directly to specific dust-grain populations. These are exposed through the same registry interface as the curves in Table {ref}`1 <tab-dust-curves>`.

Several curves incorporate the 2175 Å UV bump via a Drude profile: $$D(\lambda; \lambda_0, \gamma_D) = \frac{(\lambda\, \gamma_D)^2}{(\lambda^2 - \lambda_0^2)^2 + (\lambda\, \gamma_D)^2},

$$ (eq-drude)
 with default $\lambda_0 = 0.2175\,\mu{\rm m}$ and $\gamma_D = 0.035\,\mu{\rm m}$. The Kriek--Conroy family of curves (Kriek and Conroy 2013; Salim et al. 2018) modifies the Calzetti base by adding a power-law slope deviation $\delta$ and UV bump amplitude $E_b$: $$k(\lambda) = k_{\rm base}(\lambda) \cdot \left(\frac{\lambda}{5500\,\text{\AA}}\right)^{\delta} + E_b\cdot D(\lambda).

$$ (eq-kriek-conroy)
 Several variants differ in operator ordering: Noll09 applies $(\text{base} + \text{bump}) \times \text{slope}$, whereas Kriek--Conroy and SBL18 apply $\text{base} \times \text{slope} + \text{bump}$; the SBL18 variant (Salim et al. 2018) uses the far-UV Leitherer02 extension. The TEA curve (Haskell et al. 2024) imposes a physically motivated $E_b(\delta) = 2.5\,\exp(3.5\,\delta)\times 10^s$ correlation calibrated on NIHAO-SKIRT radiative transfer, reducing the free parameters to two.

The Li et al. (2008) analytical curve provides a four-parameter family ($c_1$--$c_4$) that reproduces MW, LMC, SMC, and Calzetti-like curves within a single functional form (see also Markov et al. 2023, 2025): $$\begin{aligned}
 
\frac{A_\lambda}{A_V} = {} &
  \frac{c_1}{(\lambda/0.08)^{c_2} + (0.08/\lambda)^{c_2} + c_3} \nonumber \\
  & + \frac{233\,[1 - c_1/D_0 - c_4/4.60]}{(\lambda/0.046)^2 + (0.046/\lambda)^2 + 90} \nonumber \\
  & + \frac{c_4}{(\lambda/0.2175)^2 + (0.2175/\lambda)^2 - 1.95},
\end{aligned}

$$ (eq-li08)
 where $\lambda$ is in $\mu$m, $D_0 = 6.88^{c_2} + 0.145^{c_2} + c_3$, and the three terms represent the UV/optical continuum, far-UV rise, and 2175 Å bump respectively. This is the fiducial attenuation model in recent high-redshift JWST analyses (Markov et al. 2025).

The remaining curves in Table {ref}`1 <tab-dust-curves>`, power law (Charlot and Fall 2000), Calzetti (Calzetti et al. 2000), Cardelli (Cardelli et al. 1989), SMC/LMC (Pei 1992), and Leitherer02 (Leitherer et al. 2002), follow their published functional forms without modification.

`narayanan_z` is the exception, because Narayanan et al. (2018) publish no functional form for the redshift dependence. They report median attenuation curves at integer $z = 0$ to $6$ from a 25 Mpc MUFASA radiative-transfer run and distribute those curves as a table. tengri fits the Kriek and Conroy (2013) form to each published median over 1250 Å to 1 $\mu$m and interpolates the fitted slope and bump linearly in redshift, holding the end value outside $0 \le z \le 6$ rather than extrapolating. The residual root-mean-square of that fit is 0.010 to 0.024 on curves of order unity, so it reproduces the published medians to a few percent and no better. Redshift is the only free parameter: the law is the median curve at $z$, and a fit that wants a slope or bump of its own uses `kriek_conroy`, which is that model. `scripts/fit_narayanan2018_medians.py` reproduces the fit from the repackaged table in `data/attenuation/`.

(app-dust-emission-details)=

## Dust Emission Models

Dust emission models span parametric (modified blackbody, Casey (2012)) and tabulated (Dale et al. (2014), Draine and Li (2007), Draine and Li (2014), Hensley and Draine (2023), Jones et al. (2017), da Cunha et al. (2008), Boquien and Salim (2021)) approaches. Tabulated models are loaded lazily from pre-computed template grids at first use; the templates are interpolated in the model parameters via JAX-compatible routines and are fully differentiable. All models are normalized by an energy-balance constraint.

### Energy Balance

The absorbed luminosity is computed as $$L_{\rm abs} = \int_0^{\infty} \bigl[1 - T(\lambda)\bigr]\, L_{\nu,\rm intrinsic}(\lambda)\, d\nu,

$$ (eq-energy-balance)
 where $T(\lambda)$ is the SFH-weighted effective dust transmission from Equation {eq}`eq-dust-transmission`. Each emission model is then normalized so that $\int L_{\nu,\rm emit}\, d\nu = \eta \, L_{\rm abs}$, where $\eta$ is the energy-balance relaxation parameter (`dust_eta_balance`, default $\eta = 1$). Setting $\eta = 1$ enforces strict energy conservation; $\eta > 1$ accommodates additional infrared luminosity from sources not in the line of sight (e.g., embedded AGN); $\eta < 1$ accounts for geometric configurations where some absorbed energy is re-emitted out of the observed aperture.

(app-mbb)=

### Modified Blackbody

The optically thin modified blackbody (2--3 parameters: $T_{\rm dust}$, $\beta_{\rm IR}$, and optionally redshift for CMB correction): $$L_\nu^{\rm dust} \propto \left(\frac{\nu}{\nu_{\rm ref}}\right)^{\beta_{\rm IR}} B_\nu(T_{\rm eff}),$$ where $B_\nu(T)$ is the Planck function, $\nu_{\rm ref}$ corresponds to $250\,\mu$m, and $T_{\rm eff}$ is the CMB-corrected dust temperature.

#### CMB temperature correction.

At high redshift the CMB sets a temperature floor on dust grains. Following da Cunha et al. (2013): $$T_{\rm eff} = \bigl(T_{\rm dust}^{\,n} + T_{\rm CMB}(z)^{\,n} - T_{\rm CMB,0}^{\,n}\bigr)^{1/n},

$$ (eq-cmb-temperature)
 where $n \equiv 4 + \beta_{\rm IR}$, $T_{\rm CMB}(z) = 2.725\,(1+z)$ K, and $T_{\rm CMB,0} = 2.725$ K.

#### CMB contrast factor.

The observed dust flux is suppressed relative to the intrinsic emission when observed against the CMB: $$\frac{S_{\rm obs}}{S_{\rm int}} = 1 - \frac{B_\nu(T_{\rm CMB}(z))}{B_\nu(T_{\rm eff})}.

$$ (eq-cmb-contrast)


(app-casey)=

### Casey (2012) MBB + Mid-IR Power Law

The Casey (2012) model adds a mid-IR power-law component to the modified blackbody, capturing the warm dust continuum that a single MBB underestimates at $\lambda \lesssim 50\,\mu$m. Three free parameters: $T_{\rm dust}$, $\beta_{\rm IR}$, and $\alpha_{\rm MIR}$ (mid-IR slope). An internal `optically_thin` flag suppresses the power-law component when set; in that limit the model reduces to a pure modified blackbody with two free parameters.

(app-dale)=

### Dale et al. (2014) Templates

A 1-parameter family controlled by the power-law slope $\alpha$ of the radiation field intensity distribution $dM_{\rm dust}/dU \propto U^{-\alpha}$ (Dale et al. 2014). Low $\alpha$ ($\mathord{\sim}1$--$1.5$) produces warm, peaked SEDs characteristic of luminous IR galaxies; high $\alpha$ ($\mathord{\sim}3$--$4$) produces cooler SEDs. Templates are loaded from a pre-computed grid of 64 spectra and interpolated in $\alpha$; CMB corrections (Equation {eq}`eq-cmb-temperature`) are applied at $z > 0$.

(app-draine-li)=

### Draine & Li (2007)

A 3-parameter model ($U_{\min}$, $\gamma_{\rm PDR}$, $q_{\rm PAH}$) following Draine and Li (2007). Templates are loaded from a pre-computed grid and interpolated in the three parameters. The radiation field intensity $U_{\min}$ sets the diffuse ISM dust temperature, $\gamma_{\rm PDR}$ controls the fraction arising from photo-dissociation regions, and $q_{\rm PAH}$ modulates mid-IR features.

(app-draine-li2014)=

### Draine & Li (2014) Update

The Draine and Li (2014) update adds a fourth parameter controlling the high-$U$ dust mass fraction independently of $\gamma_{\rm PDR}$, with revised emissivities from updated laboratory data. Templates are loaded from a separate pre-computed grid.

(app-magphys)=

### MAGPHYS (da Cunha et al. 2008), planned

*This backend is planned but not yet registered in `DUST_EMISSION_MODELS`; the description below corresponds to the model that will be exposed once the implementation lands.* The four-component model of da Cunha et al. (2008) decomposes the IR SED into PAH features (Drude profiles following Smith et al. (2007)), a hot MIR continuum ($T \sim 180$ K), warm grains ($T_W \sim 30$--$60$ K), and cold grains ($T_C \sim 15$--$25$ K), each as a modified blackbody with $\beta = 1.5$ (birth cloud) or $\beta = 2.0$ (ISM). The energy fractions $\xi_{\rm PAH}$, $\xi_{\rm MIR}$, $\xi_W$ are free parameters; $\xi_C = 1 - \xi_{\rm PAH} - \xi_{\rm MIR} - \xi_W$.

(app-astrodust)=

### Astrodust+PAH (Hensley & Draine 2023)

The Hensley and Draine (2023) model replaces the silicate-graphite-PAH composition of DL07 with a single composite grain material ("astrodust") plus separate PAH nanoparticles, using laboratory-measured optical properties. The interface is identical to DL07 ($q_{\rm PAH}$, $U_{\min}$, $\gamma_{\rm PDR}$); the revised grain composition shifts $q_{\rm PAH}$ to 5.91% (vs. DL07's 4.6%).

(app-themis)=

### THEMIS (Jones et al. 2017)

The Jones et al. (2017) THEMIS framework models amorphous hydrogenated carbon a-C(:H) nanoparticles and amorphous silicate grains with physically motivated size distributions and UV-processing-dependent optical properties. The small-grain fraction $q_{\rm hac}$ (analogous to $q_{\rm PAH}$) and $U_{\min}$ are the primary free parameters; templates are pre-tabulated via DustEM.

(app-bosa)=

### Boquien & Salim (2021) Empirical Templates

The Boquien and Salim (2021) templates add sSFR as a second axis alongside $L_{\rm TIR}$, capturing the observation that galaxies at fixed IR luminosity but higher sSFR have warmer dust and stronger MIR features. Templates are interpolated from a grid of 2584 star-forming galaxies; $L_{\rm TIR}$ is set by the energy balance and sSFR is computed from the SFH.

## References

Boquien, Médéric, and Samir Salim. 2021. "New dust emission templates for star-forming galaxies." 653: A149. <https://doi.org/10.1051/0004-6361/202140734>.

Calzetti, Daniela, Lee Armus, Ralph C. Bohlin, Anne L. Kinney, Jan Koornneef, and Thaisa Storchi-Bergmann. 2000. "The Dust Content and Opacity of Actively Star-forming Galaxies." 533 (2): 682--95. <https://doi.org/10.1086/308692>.

Cardelli, Jason A., Geoffrey C. Clayton, and John S. Mathis. 1989. "The Relationship between Infrared, Optical, and Ultraviolet Extinction." 345 (October): 245. <https://doi.org/10.1086/167900>.

Casey, Caitlin M. 2012. "Far-infrared spectral energy distribution fitting for galaxies near and far." 425 (4): 3094--103. <https://doi.org/10.1111/j.1365-2966.2012.21455.x>.

Charlot, Stéphane, and S. Michael Fall. 2000. "A Simple Model for the Absorption of Starlight by Dust in Galaxies." 539 (2): 718--31. <https://doi.org/10.1086/309250>.

Chevance, Mélanie, J. M. Diederik Kruijssen, Alexander P. S. Hygate, et al. 2020. "The lifecycle of molecular clouds in nearby star-forming disc galaxies." 493 (2): 2872--909. <https://doi.org/10.1093/mnras/stz3525>.

da Cunha, Elisabete, Stéphane Charlot, and David Elbaz. 2008. "A simple model to interpret the ultraviolet, optical and infrared emission from galaxies." 388 (4): 1595--617. <https://doi.org/10.1111/j.1365-2966.2008.13535.x>.

da Cunha, Elisabete, Brent Groves, Fabian Walter, et al. 2013. "On the Effect of the Cosmic Microwave Background in High-redshift (Sub-)millimeter Observations." 766: 13. <https://doi.org/10.1088/0004-637X/766/1/13>.

Dale, Daniel A., George Helou, Georgios E. Magdis, Lee Armus, Tanio Dı́az-Santos, and Yong Shi. 2014. "A Two-parameter Model for the Infrared/Submillimeter/Radio Spectral Energy Distributions of Galaxies and Active Galactic Nuclei." 784: 83. <https://doi.org/10.1088/0004-637X/784/1/83>.

Draine, B. T., and Aigen Li. 2007. "Infrared Emission from Interstellar Dust. IV. The Silicate-Graphite-PAH Model in the Post-Spitzer Era." 657 (2): 810--37. <https://doi.org/10.1086/511055>.

Draine, Bruce T., and Aigen Li. 2014. "Interstellar Dust Models with Updated Optical Properties." 785: 159. <https://doi.org/10.1088/0004-637X/785/2/159>.

Haskell, P., S. Das, D. J. B. Smith, R. K. Cochrane, C. C. Hayward, and D. Anglés-Alcázar. 2024. "Beware the recent past: a bias in spectral energy distribution modelling due to bursty star formation." 530 (1): L7--12. <https://doi.org/10.1093/mnrasl/slae019>.

Hensley, Brandon S., and Bruce T. Draine. 2023. "An Updated Model for Interstellar Grain Alignment and the Astrodust+PAH Grain Model." 948: 55. <https://doi.org/10.3847/1538-4357/acc4c2>.

Jones, A. P., M. Köhler, N. Ysard, M. Bocchio, and L. Verstraete. 2017. "The global dust modelling framework THEMIS." 602: A46. <https://doi.org/10.1051/0004-6361/201630225>.

Kriek, Mariska, and Charlie Conroy. 2013. "The Dust Attenuation Law in Distant Galaxies: Evidence for Variation with Spectral Type." 775 (1): L16. <https://doi.org/10.1088/2041-8205/775/1/L16>.

Leitherer, Claus, I. -Hui Li, Daniela Calzetti, and Timothy M. Heckman. 2002. "Global Far-Ultraviolet (912-1800 Å) Properties of Star-forming Galaxies." 140 (2): 303--29. <https://doi.org/10.1086/342486>.

Li, Aigen, S. L. Liang, D. A. Kann, D. M. Wei, S. Klose, and Y. J. Wang. 2008. "On Dust Extinction of Gamma-Ray Burst Host Galaxies." 685: 1046--54. <https://doi.org/10.1086/591049>.

Lower, Sidney, Desika Narayanan, Joel Leja, Benjamin D. Johnson, Charlie Conroy, and Romeel Davé. 2022. "How Well Can We Measure Galaxy Dust Attenuation Curves? The Impact of the Assumed Star-dust Geometry Model in Spectral Energy Distribution Fitting." 931 (1): 14. <https://doi.org/10.3847/1538-4357/ac6959>.

Markov, V., S. Gallerani, A. Pallottini, et al. 2023. "Dust attenuation at high redshifts: the JWST perspective." 679: A12. <https://doi.org/10.1051/0004-6361/202346983>.

Markov, V., S. Gallerani, A. Pallottini, et al. 2025. "Unveiling the trends between dust attenuation and galaxy properties at $z \sim 2$--$12$ with JWST."

Narayanan, Desika, Romeel Davé, Benjamin D. Johnson, Robert Thompson, Charlie Conroy, and James Geach. 2018. "The IRX-$\beta$ dust attenuation relation in cosmological galaxy formation simulations." 474 (2): 1718--36. <https://doi.org/10.1093/mnras/stx2860>.

Pei, Yichuan C. 1992. "Interstellar Dust from the Milky Way to the Magellanic Clouds." 395 (August): 130. <https://doi.org/10.1086/171637>.

Salim, Samir, Médéric Boquien, and Janice C. Lee. 2018. "Dust Attenuation Curves in the Local Universe: Demographics and New Laws for Star-forming Galaxies and High-redshift Analogs." 859 (1): 11. <https://doi.org/10.3847/1538-4357/aabf3c>.

Smith, J. D. T., B. T. Draine, D. A. Dale, et al. 2007. "The Mid-Infrared Spectrum of Star-forming Galaxies: Global Properties of Polycyclic Aromatic Hydrocarbon Emission." 656: 770--91. <https://doi.org/10.1086/510549>.

Wild, Vivienne, Guinevere Kauffmann, Tim Heckman, et al. 2007. "Bursty stellar populations and obscured active galactic nuclei in galaxy bulges." 381 (2): 543--72. <https://doi.org/10.1111/j.1365-2966.2007.12256.x>.

Witt, Adolf N., and Karl D. Gordon. 2000. "Multiple Scattering in Clumpy Media. II. Galactic Environments." 528 (2): 799--816. <https://doi.org/10.1086/308197>.
