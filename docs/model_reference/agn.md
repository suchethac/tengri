(app-agn-details)=

# AGN Emission

(app-agn-design)=

## Design Philosophy

Existing SED-fitting codes each cover a subset of AGN physics: X-CIGALE (Yang et al. 2020, 2022) provides an $\alpha_{\rm ox}$ disc--corona bridge, SKIRTOR torus, and polar dust but no AGN-ionized emission lines; BEAGLE-AGN (Vidal-Garcı́a et al. 2024) uses CLOUDY grids to predict NLR emission but fixes nitrogen abundance; AGNfitter-rx (Martı́nez-Ramı́rez et al. 2024) provides flexible torus and radio backends but no nebular emission; Synthesizer (Lovell et al. 2025) introduces an emission-tree architecture but is a forward-modeling code without fitting infrastructure (see Table {ref}`4 <tab-agn-comparison>` for a detailed comparison).

tengri's unified AGN framework synthesizes these capabilities into a single, modular, differentiable pipeline. The guiding principle is *causal energy flow with backend-swappable nodes*: the accretion engine ($M_{\rm BH}$, $\dot{m}$) sets the bolometric luminosity, photons propagate outward through successive emission zones, and at each zone the radiation is decomposed into incident, nebular, transmitted, and escaped channels. Every zone is served by an interchangeable backend, and presets wire backends together to reproduce existing codes' configurations. The emission-tree architecture is adapted from Synthesizer (Lovell et al. 2025); disc physics from qsosed/RELAGN (Kubota and Done 2018; Hagen and Done 2023); the $\alpha_{\rm ox}$ bridge and polar dust from X-CIGALE; the CAT3D-Wind torus from AGNfitter-rx; the Feltre NLR grids from BEAGLE-AGN, all reimplemented in pure JAX for differentiability. The novel contributions are: (1) a self-consistent disc$\to$NLR pipeline via the Cue emulator ({ref}`app-agn-nebular`), where NLR line ratios shift with accretion rate; (2) free N/O and C/O abundance ratios enabling BPT-diagram fitting; and (3) full JAX differentiability of $\sim$20 AGN parameters.

(app-unified-agn)=

## Unified AGN Emission Framework

#### Physical picture: energy flow from the accretion engine outward.

The central engine is a supermassive black hole (SMBH) of mass $M_{\rm BH}$ accreting at Eddington ratio $\dot{m} \equiv L_{\rm bol}/L_{\rm Edd}$, where $L_{\rm Edd} = 4\pi G M_{\rm BH} m_p c / \sigma_T
\approx 1.26 \times 10^{38} (M_{\rm BH}/M_\odot)\;\mathrm{erg\,s^{-1}}$. Gravitational energy released in the accretion disc is the primary power source for the entire AGN. Photons produced in the disc propagate outward through successive physical zones, each of which intercepts, reprocesses, or transmits a fraction of the radiation. The observed SED depends on which zones lie along the line of sight, which is set by the inclination angle $i$ relative to the torus opening angle $\theta_{\rm torus}$ (the AGN unification scheme; Antonucci (1993; Urry and Padovani 1995)).

The energy flow proceeds as follows:

1.  **Accretion disc** (UV/optical, $\sim$1000--10000 Å). Gravitational energy is radiated as thermal emission from an optically thick, geometrically thin disc, producing the "big blue bump" that dominates the rest-frame UV. For the physical disc backends (`qsosed`, `relagn`), the disc is further stratified into three radial zones: an outer standard disc (Shakura--Sunyaev), a warm Comptonization region that bridges the UV to soft X-rays, and a hot inner corona ({ref}`app-xray-module`).

2.  **Hot corona** (X-ray, $\sim$0.1--300 keV). A fraction $f_{\rm hard} \approx 0.02$ of the accretion luminosity is dissipated in a hot, optically thin plasma above the inner disc. This corona upscatters seed photons from the disc via inverse Compton scattering, producing a power-law X-ray spectrum $L_\nu \propto E^{1-\Gamma} e^{-E/E_{\rm cut}}$. The corona emission is moderately anisotropic ($L_X \propto 1 + \cos\theta$; Yang et al. (2022)).

3.  **Broad-line region** (BLR; optical/UV emission lines, FWHM $\sim$1000--10000 km s$^{-1}$). Dense gas clouds ($n_{\rm H} \sim 10^{9}$--$10^{11}\,{\rm cm}^{-3}$) orbiting at $r \sim 0.01$--$0.1\,$pc are photoionized by the disc UV/EUV radiation. A covering fraction $f_{\rm cov}^{\rm BLR}
      \sim 0.1$ of the disc luminosity is intercepted. The BLR lies inside the torus and is visible only for Type 1 sightlines ($i < 90^\circ - \theta_{\rm torus}$); for Type 2 orientations it is obscured.

4.  **Polar dust** (FIR, $\sim$30--200 $\mu$m). Along Type 1 sightlines, optically thin dust at polar latitudes (above the torus opening) moderately reddens the disc continuum with SMC-like extinction, and re-emits the absorbed energy as a modified blackbody at $T \sim 100$ K ({ref}`app-polar-dust`).

5.  **Narrow-line region** (NLR; optical forbidden lines, FWHM $\sim$200--1000 km s$^{-1}$). Lower-density gas ($n_{\rm H} \sim 10^{2}$--$10^{4}\,{\rm cm}^{-3}$) on kpc scales is photoionized by the AGN EUV radiation, producing forbidden lines (\O [iii\] $\lambda 5007$, \N [ii\] $\lambda 6583$, \S [ii\] $\lambda\lambda 6716,6731$) and recombination lines (H$\alpha$, H$\beta$). The NLR extends beyond the torus opening and is visible at *all* inclinations. A covering fraction $f_{\rm cov}^{\rm NLR} \sim 0.1$ determines the fraction of ionizing photons intercepted. The line ratios depend on the shape of the ionizing spectrum, the gas metallicity, density, and abundance ratios, this is the physical basis for the BPT diagnostic diagram (Baldwin et al. 1981).

6.  **Dusty torus** (MIR/FIR, $\sim$1--100 $\mu$m). A geometrically thick distribution of dusty clouds surrounding the central engine at $r \sim 0.1$--$10\,$pc absorbs the UV/optical disc and corona radiation and reprocesses it as thermal infrared emission. The torus geometry (opening angle $\theta_{\rm torus}$, radial and angular density profiles) determines the covering fraction and the viewing-angle-dependent IR SED. The torus is responsible for the AGN unification: Type 1 AGN are viewed through the polar opening (disc and BLR visible), while Type 2 AGN are viewed through the torus (disc and BLR obscured).

7.  **X-ray binaries** (XRBs; X-ray, $\sim$0.5--10 keV). High-mass X-ray binaries (HMXBs, scaling with star-formation rate) and low-mass X-ray binaries (LMXBs, scaling with stellar mass) contribute galaxy-origin X-ray emission independent of the AGN.

8.  **Radio** (synchrotron, $\sim$0.1--100 GHz). Synchrotron emission from supernova remnants (scaling with SFR via the FIR--radio correlation) and, for radio-loud AGN, relativistic jets produce radio continuum.

#### Photon-path decomposition.

Following the Synthesizer `UnifiedAGN` formalism (Lovell et al. 2025), we track photons through each emission zone using an explicit incident/nebular/transmitted/escaped decomposition. For each zone (BLR, NLR, torus), the incident radiation from the disc is partitioned into three channels governed by the covering fraction $f_{\rm cov}$: $$L_\nu^{\rm zone} = f_{\rm cov}\bigl[L_\nu^{\rm neb} + L_\nu^{\rm trans}\bigr]
  + (1 - f_{\rm cov})\,L_\nu^{\rm esc},

$$ (eq-agn-zone-decomp)
 where $L_\nu^{\rm neb}$ is the nebular emission (lines $+$ continuum) produced by the photoionized gas, $L_\nu^{\rm trans}$ is the transmitted continuum (the disc spectrum attenuated by passage through the gas), and $L_\nu^{\rm esc}$ is the fraction of the incident radiation that escapes without interacting with the zone. Energy conservation requires that the absorbed luminosity, $$L_{\rm abs} = \int \bigl(L_\nu^{\rm inc} - L_\nu^{\rm trans}
  - L_\nu^{\rm neb}\bigr)\,d\nu,$$ is reprocessed: as thermal IR emission (for the torus and polar dust), or as emission lines and free--free/bound--free continuum (for the BLR and NLR). This decomposition ensures that the covering fractions $f_{\rm cov}^{\rm BLR}$ and $f_{\rm cov}^{\rm NLR}$ simultaneously and self-consistently control the disc attenuation, the nebular line normalization, and the torus reprocessing budget.

#### Total observed AGN SED.

The complete AGN SED at viewing angle $\theta$ is: $$\begin{aligned}
 
L_\nu^{\rm AGN}(\theta) &= M_{\rm disc}(\theta)\,
  \bigl[L_\nu^{\rm disc,trans} + L_\nu^{\rm disc,esc}\bigr]
  \nonumber \\
&\quad + L_\nu^{\rm torus}(\theta)
  + L_\nu^{\rm polar}(\theta) \nonumber \\
&\quad + M_{\rm BLR}(\theta)\, L_\nu^{\rm BLR}
  + L_\nu^{\rm NLR} \nonumber \\
&\quad + L_\nu^{\rm X\text{-}ray}(\theta)
  + L_\nu^{\rm radio},
\end{aligned}

$$ (eq-agn-total)
 where $M_{\rm disc}$ and $M_{\rm BLR}$ are smooth sigmoid masks (Equation {eq}`eq-agn-mask`) that suppress the disc and BLR for Type 2 sightlines. The NLR emission is isotropic (no mask). The total galaxy SED is the sum of the stellar component (with its own dust attenuation, HII nebular emission, and dust IR emission) and $L_\nu^{\rm AGN}$.

#### Self-consistency chains.

Three internal consistency relations wire components together, ensuring that changes to the accretion engine propagate causally through the tree:

1.  **Chain 1: Disc $\to$ Corona** (reimplemented from qsosed/RELAGN). For the physical disc backends, the accretion flow is stratified into three radial zones following Kubota and Done (2018): an outer standard disc (Shakura--Sunyaev), a warm Comptonization region ($kT_e \sim 0.2$ keV, $\tau \sim 10$--20), and a hot inner corona ($kT_e \sim 100$ keV, $\tau \sim 1$). All three zones share a single Novikov--Thorne emissivity profile. The corona dissipates a luminosity $L_{\rm diss} \approx 0.02\,L_{\rm Edd}$, and the hard X-ray photon index is derived self-consistently from the energy balance via the Beloborodov (1999) relation (Equation {eq}`eq-beloborodov`). As $\dot{m}$ increases, the corona truncation radius $R_{\rm hot}$ shrinks, the corona intercepts more seed photons, and $\Gamma_{\rm hot}$ steepens,  matching the observed $\Gamma$--$\dot{m}$ correlation. Our JAX reimplementation follows the qsosed code (Quera-Bofarull) and RELAGN (Hagen and Done 2023) as reference; those codes use numpy/scipy and are not JIT-compatible. For simpler disc backends, the corona is connected via the empirical $\alpha_{\rm ox}$ bridge ({ref}`app-xray-module`).

2.  **Chain 2: Disc $\to$ Ionizing field $\to$ NLR lines** (novel to tengri). The disc EUV output (below 912 Å) determines the shape of the ionizing spectrum incident on the NLR. This spectrum is characterized by the 7-parameter piecewise power-law accepted by the Cue neural emulator (Li et al. 2025), which predicts physically consistent NLR emission-line luminosities as a function of the ionizing shape and five gas parameters ($\log U$, $\log n_{\rm H}$, $\log Z$, $\log {\rm N/O}$, $\log {\rm C/O}$). As $\dot{m}$ varies, the EUV spectrum hardens, the ionizing photon production rate $Q_H$ increases, and the predicted position on the BPT diagram shifts accordingly. No other SED-fitting code implements this self-consistent coupling between the accretion physics and the narrow-line emission ({ref}`app-agn-nebular`).

3.  **Chain 3: Polar dust $\leftrightarrow$ Torus energy budget** (adapted from X-CIGALE). Following Yang et al. (2020), optically thin polar dust along Type 1 sightlines attenuates the disc before the torus intercepts radiation. The absorbed luminosity is reemitted as a gray body with strict energy conservation ({ref}`app-polar-dust`). When the CAT3D-Wind torus model (Hönig and Kishimoto 2017) is selected, the separate polar-dust node is disabled because the wind component already accounts for polar-direction reprocessing.

#### Geometric masking.

Type 1 and Type 2 orientations are distinguished by the inclination angle $i$ relative to the torus half-opening angle $\theta_{\rm torus}$. The disc and BLR are visible only for Type 1 sightlines ($i < 90^\circ - \theta_{\rm torus}$); the NLR extends beyond the torus and is visible at all angles. The transition is implemented as a smooth sigmoid mask $$M(\cos i) = \sigma\!\left(-\frac{i - i_{\rm crit}}{\Delta_i}\right),
\quad i_{\rm crit} = 90^\circ - \theta_{\rm torus},

$$ (eq-agn-mask)
 with transition width $\Delta_i = 2^\circ$ (default), ensuring differentiability through the Type 1/2 boundary.

(app-kd-approximations)=

## K&D Three-Zone Disc Model

The physical disc backend (`qsosed`) implements the Kubota and Done (2018) three-zone accretion disc model in JAX. Three radial zones: (1) outer standard disc (Shakura--Sunyaev), (2) warm Comptonization region ($kT_e \sim 0.2$ keV, $\tau \sim 10$--20), (3) hot inner corona ($kT_e \sim 100$ keV, $\tau \sim 1$). All zones share the Novikov--Thorne emissivity $T(r) = T_{\rm in}\bigl(r/r_{\rm in}\bigr)^{-3/4}\bigl[1 - \sqrt{r_{\rm in}/r}\bigr]^{1/4}$ and radiative efficiency $\eta = 1 - \sqrt{1 - 2/(3\,r_{\rm ISCO})}$.

Corona dissipates $L_{\rm diss} \approx 0.02\,L_{\rm Edd}$; hard X-ray photon index derived self-consistently from $$\Gamma_{\rm hot} = \tfrac{7}{3}\left(\frac{L_{\rm diss}}{L_{\rm seed}}\right)^{\!-0.1}.

$$ (eq-beloborodov-text)


Translating this model into a fully differentiable JAX pipeline requires replacing several non-differentiable operations with smooth approximations. Table {ref}`1 <tab-kd-approximations>` summarizes the key choices and their accuracy relative to the reference qsosed implementation (Kubota and Done 2019).

(tab-kd-approximations)=

| Component | Approximation | Accuracy |
|:---|:---|:---|
| $R_{\rm hot}$ | 40-step bisection on N--T integral | exact ($10^{-12}$) |
| Warm Compton. | Precomputed `nthcomp` templates, interpolated in $(\Gamma_w, kT_e)$ | $\lesssim 2\%$ |
| Seed photons | K&D Eq. 3, 100-pt log grid | exact |
| Reprocessing | Omitted | $\lesssim 3\%$ on $\Gamma_{\rm hot}$ |
| Energy balance | Per-annulus renormalization | exact |
| $R_{\rm out}$ | Laor & Netzer self-gravity | $2$--$4\times$ improved |

: Differentiable approximations in the K&D disc model and their accuracy relative to the qsosed reference.

The precomputed `nthcomp` templates replaced an earlier analytic approximation that had $\lesssim 20\%$ shape error; the template-based approach achieves $\lesssim 2\%$ accuracy across the UV--soft-X-ray range. Full zone-by-zone derivations, error budgets, and comparison figures are documented in the code's online reference (`docs/dev/design/agn_kd_model.md`).

(app-adaf)=

## ADAF Model for Low-Luminosity AGN

At sub-Eddington accretion rates ($\dot{m} \lesssim 0.01$), the inner accretion flow transitions from a geometrically thin, optically thick disc to an advection-dominated accretion flow (ADAF) in which most of the viscously dissipated energy is advected into the black hole rather than radiated (Narayan and Yi 1994; Mahadevan 1997). The tengri ADAF backend implements a three-component spectral model:

1.  **Synchrotron** (radio/mm): self-absorbed ($L_\nu \propto \nu^{2}$, Rayleigh--Jeans) below the self-absorption frequency $\nu_{\rm sa}$, transitioning to optically thin ($L_\nu \propto \nu^{1/3}\,\exp(-\nu/\nu_c)$) above it (Mahadevan 1997, Eqs. 19--25). The peak frequency $\nu_p \propto m^{-1/2}\,\dot{m}^{1/2}\,T_e^2\,r_{\rm min}^{-5/4}$ (Eq. 24) places the peak in the sub-mm band for typical SMBH masses. The magnetic pressure fraction $\beta$ controls the synchrotron emission strength.

2.  **Bremsstrahlung** (X-ray): $L_\nu \propto \exp(-h\nu/kT_e)$, spectrally flat below the exponential cutoff (Mahadevan 1997, Eq. 30). The electron temperature is solved self-consistently from the coupled ion--electron energy balance (Mahadevan 1997, §5.1); the approximate scaling is $T_e \sim 10^{9}$--$10^{10}$ K, where $\delta$ (the fraction of viscous energy directly heating electrons, typically $\delta \sim 0.01$) and $\dot{m}$ set the equilibrium.

3.  **Inverse Compton** (hard X-ray): $L_\nu \propto \nu^{-\alpha_c}$, where $\alpha_c = -\ln\tau_{\rm es}/\ln A$ is the Compton spectral slope, $A = 1 + 4\theta_e + 16\theta_e^2$ is the mean amplification factor per scattering, and $\theta_e = kT_e/m_e c^2$ (Mahadevan 1997, Eqs. 33--34). Comptonization dominates bremsstrahlung in the sub-mm to X-ray band when $\alpha_c < 1$ (high $\dot{m}$).

The outer disc remains as a standard Shakura--Sunyaev disc truncated at a transition radius $r_{\rm tr}$ (in gravitational radii), which is a free parameter. The total ADAF luminosity scales as $L_{\rm ADAF} \sim L_{\rm bol}\,r_{\rm ISCO}/r_{\rm tr}$: larger truncation radii produce weaker ADAF emission and stronger outer disc emission. This model is appropriate for low-luminosity AGN (LINERs, FR I radio galaxies), where the standard thin-disc model over-predicts the UV/optical emission.

(app-kyconv-relagn)=

## Relativistic Outer Disc and Warm Comptonization

For sources where the inner accretion flow is not adequately described by a stationary multi-color blackbody, tengri exposes two optional extensions of the K&D disc that can be enabled at configuration time without altering the rest of the pipeline.

#### The relagn disc backend (`relagn`).

The standard Kubota and Done (2018) parameterization assumes a non-relativistic disc; for moderate-to-high black-hole spins this under-predicts the redward asymmetry of the UV/optical continuum, and the warm-Comptonized intermediate zone is more accurately described by the `nthcomp` kernel than by an analytic power-law (Hagen and Done 2023). The `relagn` backend addresses both points in a single registered model. A pre-computed disc grid is sampled at runtime by triweight trilinear interpolation in $(\log\,M_{\rm BH},\,\log\,\dot{m},\,a_\star)$ over the spin range $a_\star \in [-0.998,\,+0.998]$, with the inclination dependence applied analytically as a $2\cos i$ projection rather than as an explicit grid axis. The warm-Compton component reads a separate pre-computed `pyNTHCOMP` grid of $\sim 1.5\times10^4$ Kompaneets solves indexed by photon index $\Gamma$, electron temperature $kT_e$, and seed-photon temperature $kT_{\rm bb}$, and is reconstructed by log-space trilinear interpolation, which preserves accuracy in the Wien tail far better than linear interpolation in $L_\nu$. When either pre-computed grid is absent on disk, tengri emits a structured warning and falls back to the analytic proxy, so the dependency is opt-in; build scripts (`scripts/build_relagn_disc_grid.py` and `scripts/build_nthcomp_templates.py`) regenerate the grids in of order minutes on a single core.

#### Planned: standalone differentiable Kerr ray-tracing backend.

A separate `kyconv` backend that exposes the relativistic transfer function with an explicit inclination axis $i$, allowing $\partial F_\nu / \partial i$ and $\partial F_\nu / \partial a_\star$ to be propagated as posterior parameters of a fit, is planned for a future release. In the current implementation, KYCONV-style relativistic effects are folded into the `relagn` disc grid above, with inclination treated as a fixed projection rather than a fit parameter.

(app-qsogen)=

## QSOGen Empirical Quasar SED

The `qsogen` backend (Temple et al. 2021) provides an empirical quasar SED model calibrated on SDSS composite spectra, producing the characteristic "V-shaped" spectrum: a blue power-law from the accretion disc in the UV, turning over in the optical, then rising again in the infrared from hot dust emission. The SED is constructed from four additive components in $f_\nu$ space: $$\begin{aligned}
f_\nu = {} & f_\nu^{\rm disc}(\alpha_1, \alpha_2, \lambda_{\rm break})
  + f_\nu^{\rm dust}(T_{\rm bb}, A_{\rm bb}) \nonumber \\
  & + f_\nu^{\rm lines}(\eta_{\rm line})
  + f_\nu^{\rm reddening}(E(B{-}V)),
\end{aligned}$$ where $\alpha_1$ (default $-0.349$) and $\alpha_2$ (default $+0.593$) are the blue/UV and red/optical power-law slopes, $\lambda_{\rm break} = 3880$ Å is the break wavelength, $T_{\rm bb} = 1240$ K is the hot dust temperature, $A_{\rm bb} = 3.96$ is the dust blackbody normalization at 2 $\mu$m, $\eta_{\rm line}$ scales all emission-line equivalent widths, and $E(B{-}V)$ applies SMC-like reddening. This 7-parameter model is computationally inexpensive (a single analytic evaluation) and well-suited for broadband photometric fitting where a physical disc model is unnecessary.

(app-agn-backends)=

## Backend System and Presets

Each node in the emission tree is served by a *backend*, a pure JAX callable implementing the physics for that component. Backends are registered in a global dictionary and selected at model construction time. This design separates the *architecture* (the tree topology, covering fractions, energy conservation) from the *physics* (which specific disc/torus/NLR model is used), allowing users to swap components without changing the pipeline.

Table {ref}`2 <tab-agn-backends>` lists the available backends at each node. Each backend is a standalone function with a standardized interface: it accepts a wavelength grid, physical parameters, and returns $L_\nu$ in $L_\odot\,{\rm Hz}^{-1}$.

(tab-agn-backends)=

| Node | Backend | Params | Source | Notes |
|:---|:---|:--:|:---|:---|
| Disc | `powerlaw` | 3 | --- | $f_\nu \propto \nu^\alpha e^{-h\nu/kT_{\max}}$ |
|  | `broken_pl` | 2 | X-CIGALE | Schartmann+2005, optional $\delta_{\rm AGN}$ tilt |
|  | `multicolor` | 6 | --- | Shakura--Sunyaev with spin |
|  | `qsosed` | 3 | qsosed (Quera-Bofarull) | Simplified K&D 3-zone; geometry hardwired |
|  | `relagn` | 10 | RELAGN (Hagen and Done 2023) | Full K&D with GR corrections |
|  | `adaf` | 6 | --- | ADAF + truncated disc for LLAGN |
|  | `qsogen` | 9 | Temple et al. (2021) | Empirical quasar SED with emission lines |
| Torus | `skirtor_grid` | 7 | Stalevski et al. (2016) | Full RT clumpy torus templates (default) |
|  | `cat3d_wind` | 3 | AGNfitter-rx | Clumpy disc + polar wind; subsumes polar dust |
|  | `simple` | 4 | --- | Single-$T$ toy model; deprecated |
| NLR | `analytic` | 3 | Groves et al. (2004) | Fixed line ratios, fast |
|  | `feltre` | 5 | Feltre et al. (2016) | CLOUDY grids; $\xi_d$; 4 discrete $\alpha_{\rm pl}$ |
|  | `cue` | 7 | Li et al. (2025) | Neural emulator; continuous $\alpha_{\rm pl}$; free N/O, C/O |
| X-ray | `alpha_ox` | 2 | X-CIGALE | Just+2007 $\alpha_{\rm ox}$ bridge |
|  | `alpha_ox_aniso` | 2 | CIGALE v2022 | \+ viewing-angle anisotropy |
|  | `physical` | 0 | qsosed/RELAGN | Self-consistent K&D Zone 3 corona |
| Radio | `bell2003` | 2 | Bell (2003) | Constant $q_{\rm IR}=2.64$ |
|  | `delvecchio2021` | 4 | Delvecchio et al. (2021) | $q(M_\star,z)$; 2 extra free params |
|  | `mccheyne2022` | 4 | McCheyne et al. (2022) | $q(M_\star,z)$; alternative calibration |
|  | \+ free-free | +2 | Murphy et al. (2011) | Thermal bremsstrahlung ($T_e$, $\alpha_{\rm ff}$) |
|  | \+ AGN simple power law | +2 | --- | Radio-loudness $R$ + $\alpha_{\rm jet}$ |
|  | \+ AGN double power law | +4 | AGNfitter-rx | \+ turnover $\nu_t$; spectral curvature |

: Available backends at each AGN emission-tree node. "Params" gives the number of free parameters beyond the shared engine parameters ($M_{\rm BH}$, $\dot{m}$, $a_\star$, $\cos i$).

#### Presets.

Table {ref}`3 <tab-agn-presets>` lists pre-registered configurations that reproduce the model choices of several existing codes. Each preset selects a specific backend at every tree node; users can also override individual backends within a preset to create hybrid configurations.

(tab-agn-presets)=

| Preset | Disc | Torus |
|:---|:---|:---|
| `simple` | power-law | simple (single-temperature) |
| `standard` | multi-color (Shakura-Sunyaev) | two-temperature |
| `multicolor_agn` | multi-color + spin | two-temperature |
| `kubota_done_full` | K&D 3-zone disc | two-temperature |
| `skirtor` | power-law | SKIRTOR clumpy torus |
| `adaf` | ADAF + truncated disc | simple |
| `unified_nlr_blr` | K&D 3-zone + NLR/BLR | two-temperature + polar dust |

: AGN emission presets. Each row selects a backend at every emission-tree node.

#### Additional presets.

The backend architecture supports additional preset configurations that reproduce the model choices of X-CIGALE (Yang et al. 2022), BEAGLE-AGN (Vidal-Garcı́a et al. 2024), AGNfitter-rx (Martı́nez-Ramı́rez et al. 2024), and full self-consistent disc--corona--NLR chains with RELAGN (Hagen and Done 2023). These will be documented in future code releases as the corresponding backends reach production quality.

#### Comparison with existing codes.

Table {ref}`4 <tab-agn-comparison>` compares tengri's unified AGN framework with existing SED-fitting codes.

(tab-agn-comparison)=

| Capability | tengri | X-CIGALE | BEAGLE-AGN | AGNfitter-rx | Synthesizer |
|:---|:--:|:--:|:--:|:--:|:--:|
| Physical disc (K&D 3-zone) | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Self-consistent disc$\to$corona | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| **Disc$\to$NLR line ratios** | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| **Free N/O, C/O in NLR** | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Polar dust + energy conservation | $\checkmark$ | $\checkmark$ | $\times$ | $\sim$ | $\times$ |
| X-ray anisotropy | $\checkmark$ | $\checkmark$ | $\times$ | $\times$ | $\times$ |
| CAT3D-Wind torus (polar wind) | $\checkmark$ | $\times$ | $\times$ | $\checkmark$ | $\times$ |
| Double power-law radio | $\checkmark$ | $\times$ | $\times$ | $\checkmark$ | $\times$ |
| Emission-tree decomposition | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\checkmark$ |
| **Fully differentiable (JAX)** | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\sim$ |
| **Gradient-based inference** | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| HII + NLR simultaneous fitting | $\checkmark$ | $\times$ | $\checkmark$ | $\times$ | $\times$ |
| **Unified preset system** | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Multiple NLR backends | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |

: Comparison of AGN modeling capabilities across SED-fitting codes. $\checkmark$ = built-in, $\sim$ = partial or external, $\times$ = not available. Boldface marks capabilities novel to tengri.

(app-agn-nebular)=

## AGN-Ionized Nebular Emission

Hard ionizing photons from the AGN accretion disc photoionize gas in the narrow-line region (NLR), producing forbidden and recombination lines whose ratios carry information about the ionizing spectrum shape, gas metallicity, and density. Traditional SED-fitting codes either ignore AGN nebular emission entirely (e.g. CIGALE) or use pre-computed CLOUDY grids at a fixed set of power-law slopes and fixed nitrogen abundance (e.g. BEAGLE-AGN; Vidal-Garcı́a et al. (2024)). tengri provides three interchangeable NLR backends with different trade-offs, and introduces a novel self-consistent link between the disc model and the predicted NLR line ratios.

#### Backend 1: Analytic (Groves et al. 2004).

Fixed emission-line ratios from Groves et al. (2004) are applied with Gaussian profiles of width FWHM $= 500$ km s$^{-1}$ (or free), scaled by $f_{\rm cov}^{\rm NLR} L_{\rm acc}$. This is fast ($\sim\!1\,\mu$s) but cannot capture how line ratios change with AGN luminosity or gas conditions.

#### Backend 2: Feltre et al. (2016) CLOUDY grids.

Full photoionization grids from Feltre et al. (2016) are interpolated in five dimensions ($\alpha_{\rm pl}$, $\log U$, $\log n_{\rm H}$, $\log Z/Z_\odot$, $\xi_d$). This gives exact CLOUDY results including the dust-to-metal ratio $\xi_d$, but the EUV slope is restricted to four discrete values ($\alpha_{\rm pl} \in
\{-2.0, -1.7, -1.4, -1.2\}$) and nitrogen tracks oxygen at fixed solar N/O. The Cue backend (Backend 3) is recommended for continuous ionizing spectra and free abundance ratios.

#### Backend 3: Cue emulator (Li et al. 2025).

The Li et al. (2025) neural-network emulator accepts a *continuous* ionizing spectrum parameterization (4 slopes + 3 log luminosity ratios across four EUV segments) together with gas parameters including free N/O and C/O. All operations are differentiable. Accuracy is $\sim\!5\%$ in log line luminosities relative to CLOUDY, but the Cue training set does not include $\xi_d$.

#### Chain 2 pipeline: Disc $\to$ Cue $\to$ NLR.

When a physical disc backend is used with the Cue NLR backend, tengri constructs the self-consistent chain:

1.  The disc EUV output (below 912 Å) is characterized by a power-law slope $\alpha_{\rm pl}$ (with $f_\nu \propto
      \nu^{\alpha_{\rm pl}}$). For a single power law, all four Cue segment slopes equal $-\alpha_{\rm pl}$ in wavelength space ($f_\nu \propto \lambda^{-\alpha_{\rm pl}}$), and the three log-luminosity ratios follow from the segment-integrated fluxes.

2.  The 7 ionizing-spectrum parameters are passed to the Cue emulator together with the NLR gas parameters ($\log U$, $\log n_{\rm H}$, $\log Z/Z_\odot$, $\log \mathrm{N/O}$, $\log \mathrm{C/O}$).

3.  The ionizing photon rate $Q_{H,\rm AGN}$ is computed from $L_{\rm acc}$ and $\alpha_{\rm pl}$.

4.  Line luminosities are scaled by the NLR covering fraction $f_{\rm cov}^{\rm NLR}$.

As the accretion rate $\dot{m}$ varies, the EUV spectrum hardens, $Q_{H,\rm AGN}$ increases, and the predicted BPT-diagram position shifts physically, an effect that fixed-$\alpha_{\rm pl}$ approaches cannot capture.

#### Total nebular emission.

The galaxy's total nebular emission is the independent sum of stellar H 2-region emission and AGN NLR emission: $$L_\nu^{\rm neb} =
  (1 - f_{\rm esc})\,Q_{H,\star}\,\ell_\nu^{\rm Cue}(\boldsymbol{\theta}_{\rm HII})
  \;+\;
  f_{\rm cov}^{\rm NLR}\,Q_{H,\rm AGN}\,\ell_\nu^{\rm Cue}(\boldsymbol{\theta}_{\rm NLR}),

$$ (eq-total-nebular)
 where $\ell_\nu^{\rm Cue}$ denotes the Cue-predicted luminosity per ionizing photon, and $\boldsymbol{\theta}_{\rm HII}$ and $\boldsymbol{\theta}_{\rm NLR}$ are the (independent) gas parameter vectors for H 2 regions and the NLR respectively. No mixing parameter is introduced; the two components are summed directly.

#### Comparison with BEAGLE-AGN.

Table {ref}`5 <tab-nlr-comparison>` summarizes the key differences between tengri's AGN nebular treatment and that of BEAGLE-AGN (Vidal-Garcı́a et al. 2024).

(tab-nlr-comparison)=

| Feature | tengri(Cue backend) | BEAGLE-AGN |
|:---|:--:|:--:|
| Ionizing spectrum | Continuous $\alpha_{\rm pl}$ | 4 discrete values |
| N/O abundance | Free parameter | Fixed (solar) |
| C/O abundance | Free parameter | Fixed (solar) |
| Disc$\to$NLR coupling | Self-consistent (Chain 2) | External $\alpha_{\rm pl}$ |
| Dust-to-metal $\xi_d$ | Not modeled | Free parameter |
| Differentiable | Yes (JAX) | No |
| Gradient-based sampling | Yes | No |
| NLR backend choices | 3 (analytic/Feltre/Cue) | 1 (Feltre) |

: Comparison of AGN nebular emission approaches.

(app-blr)=

## Broad Line Region

The BLR consists of dense gas clouds ($n_{\rm H} \sim 10^{9}$--$10^{11}\,{\rm cm}^{-3}$) orbiting at $r \sim 0.01$--$0.1$ pc, producing broad permitted emission lines with $\mathrm{FWHM} \sim 1000$--$10{,}000$ km s$^{-1}$. The tengri BLR model generates broad Gaussian profiles at the wavelengths of nine major permitted lines (Ly$\alpha$, N 5 $\lambda 1240$, Si 4+O 4\] $\lambda 1400$, C 4 $\lambda 1549$, C 3\] $\lambda 1909$, Mg 2 $\lambda 2800$, H$\gamma$, H$\beta$, H$\alpha$), with relative strengths calibrated on the SDSS composite quasar spectrum (Vanden Berk et al. 2001). The line luminosities are normalized by a covering fraction $f_{\rm cov}^{\rm BLR} \sim 0.1$ and a radiative efficiency $\eta_{\rm BLR} \approx 0.08$ (the fraction of intercepted continuum re-emitted as broad lines): $$L_{\rm BLR} = \eta_{\rm BLR}\, f_{\rm cov}^{\rm BLR}\, L_{\rm disc}.$$

An optional Fe ii pseudo-continuum adds a blend of broad emission features in the UV (2200--3000 Å) and optical (4400--5400 Å) bands, modeled as a sum of Gaussians at the principal Fe ii multiplet wavelengths following Tsuzuki et al. (2006) and Kovačević et al. (2010). The Fe ii strength is parameterized by the Fe ii/H$\beta$ ratio (Boroson and Green 1992).

The BLR lies inside the torus and is visible only for Type 1 sightlines ($i < 90^\circ - \theta_{\rm torus}$), enforced by the smooth sigmoid mask (Equation {eq}`eq-agn-mask`).

(app-polar-dust)=

## Polar Dust

Following Yang et al. (2020), optically thin polar dust along Type 1 sightlines is modeled as SMC extinction (Pei 1992) with color excess $E(B{-}V)$, applied through a smooth sigmoid mask at the Type 1/2 boundary: $$M(\cos i) = \sigma\!\left[s\,(\cos i - \cos i_{\rm crit})\right],
\quad i_{\rm crit} = 90^\circ - \theta_{\rm torus},

$$ (eq-polar-mask)
 with sigmoid sharpness $s = 20$ (default). For Type 2 orientations ($M \approx 0$), polar dust has no effect; for Type 1 ($M \approx 1$), the disc is attenuated by $A_\lambda = E(B{-}V)\,R_V^{\rm SMC}\,k(\lambda)$ with $R_V^{\rm SMC} = 2.93$.

The absorbed luminosity is reemitted as a modified blackbody: $$L_\nu^{\rm polar} \propto \left[1 - \exp\!\left(-\!\left(\frac{\lambda_0}{\lambda}\right)^{\!\beta}\right)\right] B_\nu(T),

$$ (eq-polar-reemit)
 with default parameters $T = 100$ K, $\beta = 1.6$, and $\lambda_0 = 200\,\mu$m. Energy conservation is enforced by normalizing such that $\int L_\nu^{\rm polar}\,d\nu = \int L_\nu^{\rm abs}\,d\nu$.

When the CAT3D-Wind torus model (Hönig and Kishimoto 2017) is selected, polar dust is automatically disabled because the wind component of CAT3D already accounts for polar-direction reprocessing.

## References

Antonucci, Robert. 1993. "Unified models for active galactic nuclei and quasars." 31 (January): 473--521. <https://doi.org/10.1146/annurev.aa.31.090193.002353>.

Baldwin, J. A., M. M. Phillips, and R. Terlevich. 1981. "Classification parameters for the emission-line spectra of extragalactic objects." 93 (February): 5--19. <https://doi.org/10.1086/130766>.

Bell, E. F. 2003. "Estimating Star Formation Rates from Infrared and Radio Luminosities: The Origin of the Radio-Infrared Correlation." 586 (April): 794--813. <https://doi.org/10.1086/367829>.

Beloborodov, Andrei M. 1999. "On the Number of Active Galactic Nuclei at High Accretion Rates." 510: L123--26. <https://doi.org/10.1086/311810>.

Boroson, Todd A., and Richard F. Green. 1992. "The Emission-Line Properties of Low-Redshift Quasi-stellar Objects." 80 (May): 109. <https://doi.org/10.1086/191661>.

Delvecchio, I., E. Daddi, M. T. Sargent, et al. 2021. "The infrared-radio correlation of star-forming galaxies is strongly $M_{\star}$-dependent but nearly redshift-invariant since $z \sim 4$." 647 (March): A123. <https://doi.org/10.1051/0004-6361/202039647>.

Feltre, A., S. Charlot, and J. Gutkin. 2016. "Nuclear activity versus star formation: emission-line diagnostics at ultraviolet and optical wavelengths." 456 (3): 3354--74. <https://doi.org/10.1093/mnras/stv2794>.

Groves, Brent, Michael Dopita, and Ralph Sutherland. 2004. "Dusty, radiation pressure dominated photoionization: The solution to the narrow line region problem." In *The Interplay Among Black Holes, Stars and ISM in Galactic Nuclei*, edited by Thaisa Storchi-Bergmann, Luis C. Ho, and Henrique R. Schmitt, vol. 222. IAU Symposium. <https://doi.org/10.1017/S1743921304002224>.

Hagen, Scott, and Chris Done. 2023. "RELAGN: a relativistic multi-component AGN spectral model." 525: 3455--67. <https://doi.org/10.1093/mnras/stad2499>.

Hönig, Sebastian F., and Makoto Kishimoto. 2017. "Dusty Winds in Active Galactic Nuclei: Reconciling Observations with Models." 838: L20. <https://doi.org/10.3847/2041-8213/aa6838>.

Kovačević, Jelena, Luka Č. Popović, and Milan S. Dimitrijević. 2010. "Analysis of Optical Fe II Emission in a Sample of Active Galactic Nucleus Spectra." 189: 15--36. <https://doi.org/10.1088/0067-0049/189/1/15>.

Kubota, Aya, and Chris Done. 2018. "A physical model of the broad-band continuum of AGN and its implications for the UV/X relation and optical variability." 480 (1): 1247--62. <https://doi.org/10.1093/mnras/sty1890>.

Kubota, Aya, and Chris Done. 2019. "A physical interpretation of the optical-UV variability in changing-look AGN." 489: 524--33. <https://doi.org/10.1093/mnras/stz2140>.

Li, Yongda, Joel Leja, Benjamin D. Johnson, and Sandro Tacchella. 2025. "Cue: A Fast, Flexible, and Accurate Neural Emulator for Nebular Emission." <https://arxiv.org/abs/2312.12345>.

Lovell, Christopher C., William J. Roper, Aswin P. Vijayan, Scott Hagen, and Stephen M. Wilkins. 2025. "Synthesizer: a flexible code for generating synthetic astrophysical observations." *arXiv e-Prints*. <https://arxiv.org/abs/2508.03888>.

Mahadevan, Rohan. 1997. "Scaling Laws for Advection-dominated Flows: Applications to Low-Luminosity Galactic Nuclei." 477: 585--601. <https://doi.org/10.1086/303727>.

Martı́nez-Ramı́rez, Gabriela C., Gabriela Calistro Rivera, Elisabeta Lusso, and Francesco Shankar. 2024. "AGNfitter-rx: modelling AGN and galaxy SEDs from radio to X-rays." 535: 2961--85. <https://doi.org/10.1093/mnras/stae2437>.

McCheyne, I., S. Oliver, M. Sargent, et al. 2022. "The LOFAR Two-metre Sky Survey Deep fields. The mass dependence of the far-infrared radio correlation at 150 MHz using deblended Herschel fluxes." 662 (June): A100. <https://doi.org/10.1051/0004-6361/202141307>.

Murphy, E. J., J. J. Condon, E. Schinnerer, et al. 2011. "Calibrating Extinction-free Star Formation Rate Diagnostics with 33 GHz Free-free Emission in NGC 6946." 737 (2): 67. <https://doi.org/10.1088/0004-637X/737/2/67>.

Narayan, Ramesh, and Insu Yi. 1994. "Advection-dominated Accretion: A Self-similar Solution." 428: L13. <https://doi.org/10.1086/187381>.

Pei, Yichuan C. 1992. "Interstellar Dust from the Milky Way to the Magellanic Clouds." 395 (August): 130. <https://doi.org/10.1086/171637>.

Stalevski, Marko, Claudio Ricci, Yoshihiro Ueda, Paulina Lira, Jacopo Fritz, and Maarten Baes. 2016. "The dust covering factor in active galactic nuclei." 458: 2288--302. <https://doi.org/10.1093/mnras/stw444>.

Temple, Matthew J., Paul C. Hewett, and Manda Banerji. 2021. "QSOgen: a model of the UV-to-submillimetre spectral energy distributions of quasars." 508: 737--54. <https://doi.org/10.1093/mnras/stab2586>.

Tsuzuki, Yuka, Kimiaki Kawara, Yuzuru Yoshii, Shinki Oyabu, Toshihiko Tanabé, and Yoshiki Matsuoka. 2006. "Fe II Emission in 14 Low-Redshift Quasars. I. Observations." 650: 57--79. <https://doi.org/10.1086/506376>.

Urry, C. Megan, and Paolo Padovani. 1995. "Unified Schemes for Radio-Loud Active Galactic Nuclei." 107: 803. <https://doi.org/10.1086/133630>.

Vanden Berk, Daniel E., Gordon T. Richards, Amanda Bauer, et al. 2001. "Composite Quasar Spectra from the Sloan Digital Sky Survey." 122: 549--64. <https://doi.org/10.1086/321167>.

Vidal-Garcı́a, A., A. Plat, E. Curtis-Lake, et al. 2024. "BEAGLE-AGN I: Simultaneous constraints on the properties of gas in star-forming and AGN narrow-line regions." 527: 7217--42. <https://doi.org/10.1093/mnras/stad3578>.

Yang, Guang, Médéric Boquien, W. N. Brandt, et al. 2022. "Fitting AGN/galaxy X-ray-to-radio SEDs with CIGALE and improvement of the code." 927: 192. <https://doi.org/10.3847/1538-4357/ac4971>.

Yang, Guang, Médéric Boquien, Véronique Buat, et al. 2020. "X-CIGALE: fitting AGN/galaxy SEDs from X-ray to infrared." 491: 740--57. <https://doi.org/10.1093/mnras/stz3001>.
