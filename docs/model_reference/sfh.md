(app-sfh-library)=

# Star Formation History Models

(app-sfh-parametric)=

## Parametric Model Library

tengri provides a registry of parametric mean SFH models $\bar{\dot{M}}_{\star}(t)$, each returning the star formation rate in $M_\odot\,{\rm yr}^{-1}$ as a function of lookback time $t$ (in yr). Table {ref}`1 <tab-sfh-models>` summarizes the available models.

(tab-sfh-models)=

| Name | $N_p$ | Key Reference | Functional Form | Notes |
|:---|:--:|:--:|:---|:--:|
| `tsnorm` | 5 | Bellstedt et al. (2020) | $\dot{M}^{\rm pk} K(t)\,\tfrac{1}{2}\mathrm{erfc}[(t{-}t_p)/\sqrt{2}w\alpha_t]$ | Default; truncation suppresses recent SF |
| `snorm` | 4 | Robotham et al. (2020) | $\dot{M}^{\rm pk} K(t)$ | No truncation ($\alpha_t \to 0$) |
| `norm` | 3 | --- | $\dot{M}^{\rm pk} \exp[{-(t{-}t_p)^2/2w^2}]$ | Symmetric ($\gamma = 0$) |
| `lnorm` | 3 | --- | $\dot{M}^{\rm pk} \exp[{-(\log t{-}\log t_p)^2/2\sigma_{\log}^2}]$ | Gaussian in $\log t$ |
| `dpl` | 4 | Carnall et al. (2018) | $\dot{M}^{\rm pk} / [(t/\tau)^\alpha + (t/\tau)^{-\beta}]$ | Rising/declining slopes |
| `dexp` | 3 | --- | $\dot{M}^{\rm pk} (\Delta t/\tau)\,e^{-\Delta t/\tau + 1}$ | Peaks at $t_{\rm start} + \tau$ |
| `exp` | 3 | --- | $\dot{M}^{\rm pk} \exp[{-(t{-}t_{\rm start})/\tau}]$ | Declining exponential |
| `const` | 3 | --- | $10^{\log{\rm SFR}}$ for $t_{\rm start} \le t \le t_{\rm end}$ | Top-hat |
| `table` | 0 | --- | Interpolated from input array | Simulation-derived SFHs |
|  |  |  |  |  |
| *Non-parametric models* |  |  |  |  |
| `dense_basis` | 5 | Iyer and Gawiser (2017; Iyer et al. 2019) | GP on cumulative mass quantiles | **Default**; $M_\star$ direct |
| `dense_basis_pure` | 4 | Iyer and Gawiser (2017) | As above, no $\dot{M}_{\star,\rm inst}$ constraint | Auto for field composition |
| `continuity` | 7 | Leja et al. (2019) | Piecewise-constant, Student-$t$ smoothness | $\log M_\star$ + 6 ratios |
| `dirichlet` | 7 | Leja et al. (2017) | Piecewise-constant, Dirichlet prior | $\log M_\star$ + 6 stick-break |
| `burst` | 2 | Zacharegkas et al. (2025) | Triweight in $\log t_{\rm age}$ (Eq. {eq}`eq-triweight-burst`) | Amplitude set by $f_b$ |

: Parametric mean SFH models available in tengri. $N_p$ denotes the number of free parameters per model. All amplitudes are specified as $\log_{10}(\dot{M}_\star^{\rm peak} / M_\odot\,{\rm yr}^{-1})$.

Beyond the models listed above, tengri's SFH registry also exposes ProSpect-style burst-modulated variants (`snorm_burst`, `tsnorm_burst`), a flexible-prior non-parametric mode (`continuity_flex`), delayed-quenching and periodic SFHs (`delayed_bq`, `periodic`), the Buat et al. (2008) declining template (`buat08`), post-starburst (`psb` / `psb_wild2020`) and constant-then-exponential (`const_exp`) parametrizations, and a GP-modulator entry (`field`) that wraps any mean SFH with a Gaussian process-based correlated field model. All of these share the same registration interface; each is documented in the code under `src/tengri/components/sfh/`.

The `tsnorm`, `snorm`, and `norm` models share a common skewed Gaussian kernel (Robotham et al. 2020; Bellstedt and Robotham 2024): $$K(t) = \exp\!\left[-\frac{Y^2}{2}\right], \;\;
Y = X\,\exp\!\bigl[\gamma\,\mathrm{arcsinh}(X)\bigr], \;\;
X = \frac{t - t_p}{w},

$$ (eq-skewed-kernel)
 where $t_p$ is the peak lookback time, $w$ is the Gaussian width, and $\gamma$ is the skewness parameter ($\gamma = 0$ recovers a symmetric Gaussian).

(app-sfh-burst)=

## Burst Component

The triweight burst kernel (Zacharegkas et al. 2025) is a compact-support function in $\log_{10}(t_{\rm age}/{\rm Myr})$ space, with 2 shape parameters ($\log_{10} t_{\rm peak}$, $\log_{10} t_{\rm max}$): $$\dot{M}_\star(u) = \frac{35}{96}\left(1 - \left(\frac{u - u_0}{3\,\Delta u}\right)^{\!2}\right)^{\!3} \quad \text{for } |u - u_0| < 3\,\Delta u,

$$ (eq-triweight-burst)
 where $u = \log_{10}(t_{\rm age}/{\rm Myr})$, $u_0 = \log_{10}(t_{\rm peak}/{\rm Myr})$, and $\Delta u = \log_{10}(t_{\rm max}/{\rm Myr})$. This is a shape-only function; the burst amplitude is set by the burst mass fraction $f_b$ in the composition step.

(app-sfh-nonparametric)=

## Nonparametric SFH Models

Two nonparametric SFH models are available for compatibility with existing codes and for comparison tests. Both describe the SFH as piecewise-constant in $N = 7$ lookback-time bins (default edges: 0, 30, 100, 300 Myr, 1, 3, 6, 13.7 Gyr, logarithmically spaced), but differ in how the free parameters are defined and what priors they imply.

#### Continuity prior.

Following Leja et al. (2019), the free parameters are the $\log_{10}$ SFR ratios between adjacent bins, $r_j = \log_{10}(\mathrm{SFR}_j / \mathrm{SFR}_{j+1})$ for $j = 1, \ldots, N-1$. A Student-$t$ smoothness prior penalizes sharp inter-bin jumps: $$r_j \sim \mathrm{Student}\text{-}t(\nu = 2,\; \mu = 0,\; \sigma = 0.3\;\mathrm{dex}),$$ where $\nu = 2$ provides heavy tails that permit occasional sharp transitions (e.g., quenching) while penalizing them relative to a Gaussian prior. The absolute SFR scale is set by the total formed stellar mass $M_{\star,\rm formed}$, which serves as an additional free parameter. The oldest bin is the reference ($\log \mathrm{SFR}_N = 0$), and each younger bin accumulates the sum of ratios: $\log \mathrm{SFR}_j = \sum_{k=j}^{N-1} r_k$. This is the default SFH model in Prospector(Johnson 2021).

#### Dirichlet prior.

Following Leja et al. (2017), the mass fractions $\{f_j\}_{j=1}^{N}$ in $N$ time bins are drawn from a symmetric Dirichlet distribution via a stick-breaking construction. The $N - 1$ free parameters are auxiliary variables $z_j \sim \mathrm{Beta}(1, 1) = \mathrm{Uniform}(0, 1)$, mapped to mass fractions by $$\begin{aligned}
f_1 &= z_1, \nonumber \\
f_j &= z_j \prod_{k=1}^{j-1} (1 - z_k), \quad j = 2, \ldots, N-1, \\
f_N &= \prod_{k=1}^{N-1} (1 - z_k). \nonumber
\end{aligned}$$ The SFR in each bin follows from $\mathrm{SFR}_j = f_j \cdot M_{\star,\rm formed} / \Delta t_j$, where $\Delta t_j$ is the bin width. The symmetric Dirichlet$(1, \ldots, 1)$ prior assigns equal probability to all mass-fraction simplices, placing no smoothness constraint between bins.

## References

Bellstedt, Sabine, and Aaron S. G. Robotham. 2024. "ProGeny II: the impact of libraries and model configurations on inferred galaxy properties in SED fitting." *arXiv e-Prints*, arXiv:2410.17698. <https://doi.org/10.48550/arXiv.2410.17698>.

Bellstedt, Sabine, Aaron S. G. Robotham, Simon P. Driver, et al. 2020. "Galaxy And Mass Assembly (GAMA): a forensic SED reconstruction of the cosmic star formation history and metallicity evolution by galaxy type." 498 (4): 5581--603. <https://doi.org/10.1093/mnras/staa2620>.

Buat, V., A. Boselli, G. Gavazzi, and C. Bonfanti. 2008. "Star formation rates of galaxies as a function of wavelength and of the total IR luminosity." 483: 107--19. <https://doi.org/10.1051/0004-6361:20078263>.

Carnall, A. C., R. J. McLure, J. S. Dunlop, and R. Davé. 2018. "Inferring the star formation histories of massive quiescent galaxies with BAGPIPES: evidence for multiple quenching mechanisms." 480 (4): 4379--401. <https://doi.org/10.1093/mnras/sty2169>.

Iyer, Kartheik G., Eric Gawiser, Sandra M. Faber, et al. 2019. "Nonparametric Star Formation History Reconstruction with Gaussian Processes. I. Counting Major Episodes of Star Formation." 879 (2): 116. <https://doi.org/10.3847/1538-4357/ab2052>.

Iyer, Kartheik, and Eric Gawiser. 2017. "Reconstruction of Galaxy Star Formation Histories through SED Fitting:The Dense Basis Approach." 838 (2): 127. <https://doi.org/10.3847/1538-4357/aa63f0>.

Johnson, Benjamin D. 2021. *bd-j/sedpy: sedpy v0.2.0*. Version v0.2.0. Zenodo. <https://doi.org/10.5281/zenodo.4582723>.

Leja, Joel, Adam C. Carnall, Benjamin D. Johnson, Charlie Conroy, and Joshua S. Speagle. 2019. "How to Measure Galaxy Star Formation Histories. II. Nonparametric Models." 876 (1): 3. <https://doi.org/10.3847/1538-4357/ab133c>.

Leja, Joel, Benjamin D. Johnson, Charlie Conroy, Pieter G. van Dokkum, and Nell Byler. 2017. "Deriving Physical Properties from Broadband Photometry with Prospector: Description of the Model and a Demonstration of its Accuracy Using 129 Galaxies in the Local Universe." 837 (2): 170. <https://doi.org/10.3847/1538-4357/aa5ffe>.

Robotham, A. S. G., S. Bellstedt, C. del P. Lagos, et al. 2020. "ProSpect: generating spectral energy distributions with complex star formation and metallicity histories." 495 (1): 905--31. <https://doi.org/10.1093/mnras/staa1116>.

Zacharegkas, Georgios, Andrew Hearin, and Andrew Benson. 2025. "Bayesian Posteriors with Stellar Population Synthesis on GPUs." *The Open Journal of Astrophysics* 8 (December). <https://doi.org/10.33232/001c.151255>.
