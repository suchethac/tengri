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
| `dexp` | 3 | --- | $\dot{M}^{\rm pk} (\Delta t/\tau)\,e^{-\Delta t/\tau + 1}$, $\Delta t = t_{\rm start} - t$ | Forms in $[0, t_{\rm start}]$; peaks at lookback $t_{\rm start} - \tau$ |
| `exp` | 3 | --- | $\dot{M}^{\rm pk} \exp[{-(t_{\rm start}{-}t)/\tau}]$ | Declining exponential in $[0, t_{\rm start}]$; $t_{\rm start}$ is the onset lookback |
| `const` | 3 | --- | $10^{\log{\rm SFR}}$ for $t_{\rm start} \le t \le t_{\rm end}$ | Top-hat |
| `table` | 0 | --- | Interpolated from input array | Simulation-derived SFHs |
|  |  |  |  |  |
| *Non-parametric models* |  |  |  |  |
| `dense_basis` | 5 | Iyer and Gawiser (2017; Iyer et al. 2019) | GP on cumulative mass quantiles | **Default**; $M_\star$ direct |
| `dense_basis_pure` | 4 | Iyer and Gawiser (2017) | As above, no $\dot{M}_{\star,\rm inst}$ constraint | Auto for field composition |
| `continuity` | 7 | Leja et al. (2019) | Piecewise-constant, Student-$t$ smoothness | $\log M_\star$ + 6 ratios |
| `dirichlet` | 7 | Leja et al. (2017) | Piecewise-constant, Dirichlet prior | $\log M_\star$ + 6 stick-break |
| `burst` | 2 | Zacharegkas et al. (2025) | Triweight in $\log t_{\rm age}$ (Eq. {eq}`eq-triweight-burst`) | Amplitude set by $f_b$ |
| `recent` | 2 | Chevallard & Charlot (2016) | $\psi$ for $0\le t\le\Delta t$, smooth history outside (Eq. {eq}`eq-recent-episode`) | Spliced over a smooth member; $\psi = 10^{\texttt{log\_sfr}}$ |
| `recent_ssfr` | 2 | Chevallard & Charlot (2016) | As `recent`, with $\psi$ set by $\psi_{\rm S}=10^{\texttt{log\_ssfr}}$ (Eq. {eq}`eq-recent-ssfr`) | Pole at $\psi_{\rm S}\Delta t=1$, refused at build |

: Parametric mean SFH models available in tengri. $N_p$ denotes the number of free parameters per model. All amplitudes are specified as $\log_{10}(\dot{M}_\star^{\rm peak} / M_\odot\,{\rm yr}^{-1})$.

**Onset, age and peak-time parameters are fractions of the age of the universe.** Every lookback-time parameter that marks a star-formation onset, an age or a peak time has a `*_frac` twin: `sfh_dpl_age_frac`, `sfh_exp_start_frac`, `sfh_const_start_frac`, `sfh_tsnorm_peak_lbt_frac` and so on, 23 in all. The absolute time is $t = f \, t_{\rm univ}(z)$, with $t_{\rm univ}(z)$ the age of the universe at the galaxy's redshift, so a free fraction cannot place star formation before the Big Bang at any redshift. The default and the free prior are the Gyr values divided by the age of the universe at $z = 0$, with values at or above it mapping to exactly 1.

`all_params: FREE` frees the fractions. An absolute Gyr value on the old name (`sfh_dpl_age_gyr=Fixed(5.0)`) is still accepted and stays an absolute time; at a fixed redshift it must not exceed the age of the universe there, and at a free redshift it is reported when it exceeds the age of the universe at the upper end of the range. A free prior on an old name is refused: give the fraction instead, with its bounds divided by the age of the universe at $z = 0$.

Beyond the models listed above, tengri's SFH registry also exposes ProSpect-style burst-modulated variants (`snorm_burst`, `tsnorm_burst`), a flexible-prior non-parametric mode (`continuity_flex`), delayed-quenching and periodic SFHs (`delayed_bq`, `periodic`), the Buat et al. (2008) declining template (`buat08`), post-starburst (`psb` / `psb_wild2020`) and constant-then-exponential (`const_exp`) parametrizations, and a GP-modulator entry (`field`) that wraps any mean SFH with a Gaussian process-based correlated field model. All of these share the same registration interface; each is documented in the code under `src/tengri/components/sfh/`.

The `tsnorm`, `snorm`, and `norm` models share a common skewed Gaussian kernel (Robotham et al. 2020; Bellstedt and Robotham 2024): $$K(t) = \exp\!\left[-\frac{Y^2}{2}\right], \;\;
Y = X\,\exp\!\bigl[\gamma\,\mathrm{arcsinh}(X)\bigr], \;\;
X = \frac{t - t_p}{w},

$$ (eq-skewed-kernel)
 where $t_p$ is the peak lookback time, $w$ is the Gaussian width, and $\gamma$ is the skewness parameter ($\gamma = 0$ recovers a symmetric Gaussian).

(app-sfh-burst)=

## Current Episode Component

(app-sfh-current-episode)=

The current-episode types (`recent`, `recent_ssfr`) add a constant star-formation rate over the most recent $\Delta t$ on top of a smooth history, following the BEAGLE current-episode parameterization (Chevallard & Charlot 2016). With $t$ the lookback time and $\psi$ the episode rate,

$$\dot M_\star(t) = \psi \quad (0\le t\le\Delta t), \qquad \dot M_\star(t) = S(t)\,\frac{M_S}{\int_{\Delta t}^{\infty} S(t')\,dt'} \quad (t>\Delta t),
$$ (eq-recent-episode)

where $S(t)$ is the smooth history and $M_S$ its formed mass, which the splice keeps. The total formed mass is $M_S+\psi\,\Delta t$. The `recent_ssfr` form sets the specific rate $\psi_{\rm S}=\psi/(M_S+\psi\,\Delta t)$ [yr$^{-1}$], which inverts to

$$\psi = \frac{\psi_{\rm S}\,M_S}{1-\psi_{\rm S}\,\Delta t},
$$ (eq-recent-ssfr)

with a pole at $\psi_{\rm S}\Delta t = 1$. The joint constraint is $\psi_{\rm S}\Delta t < 1$ over the prior box: the build refuses a box whose upper bounds reach it (`ParameterError`, naming both knobs and the arithmetic). The registry bound on $\log_{10}\psi_{\rm S}$ alone does not protect the 100 Myr end of the duration prior.

**Mass bookkeeping.** BEAGLE's `mass` includes the episode. Tengri's smooth `log_total_mass` is therefore BEAGLE's $M-\psi\,\Delta t$, and BEAGLE's `mass` equals tengri's derived formed `stellar_mass`. The published `sfr_10myr` and `sfr_100myr` are the window-averaged rates, and `stellar_mass_surviving` excludes the mass that has since been lost.

Build the episode beside a smooth member: `sfh={'type': ['delayed', 'recent'], ...}`. On its own, a current episode has no history to splice over and is refused. Cross-code: BAGPIPES `burst` is set by a mass, CIGALE `sfhdelayedbq` by a rate ratio, and Prospector has no equivalent.

## Burst Component

The triweight burst kernel (Zacharegkas et al. 2025) is a compact-support function in $\log_{10}(t_{\rm age}/{\rm Myr})$ space, with 2 shape parameters ($\log_{10} t_{\rm peak}$, $\log_{10} t_{\rm max}$): $$\dot{M}_\star(u) = \frac{35}{96}\left(1 - \left(\frac{u - u_0}{3\,\Delta u}\right)^{\!2}\right)^{\!3} \quad \text{for } |u - u_0| < 3\,\Delta u,

$$ (eq-triweight-burst)
 where $u = \log_{10}(t_{\rm age}/{\rm Myr})$, $u_0 = \log_{10}(t_{\rm peak}/{\rm Myr})$, and $\Delta u = \log_{10}(t_{\rm max}/{\rm Myr})$. This is a shape-only function; the burst amplitude is set by the burst mass fraction $f_b$ in the composition step.

(app-sfh-nonparametric)=

## Nonparametric SFH Models

Two nonparametric SFH models are available for compatibility with existing codes and for comparison tests. Both describe the SFH as piecewise-constant in $N = 7$ lookback-time bins (default edges built from the source redshift as described below; at $z = 0$: 0, 30, 100 Myr, then 0.33, 1.11, 3.72, 12.41 and 13.79 Gyr), but differ in how the free parameters are defined and what priors they imply.

**The default bin ladder is built from the source redshift.** When no explicit `bin_edges_gyr` is given, `continuity`, `dirichlet`, `bursty_continuity`, and `prospector_beta` all build their ladder with `tengri.make_agebins_from_zred` (the Prospector-beta scheme, Wang et al. 2024): for a build redshift $z \le 3$ the two youngest edges stay fixed at 30 and 100 Myr, and the remaining interior edges are log-spaced up to 90% of $t_{\rm age}(z)$. For $z > 3$ the universe is too young to hold a 100 Myr bin and still resolve the rest of cosmic time, so none of the youngest edges are fixed: the edges are log-spaced against a grid anchored at 13.47 Myr up to 90% of $t_{\rm age}(z)$ (Prospector-beta's own $a_{\rm min} = 7.1295$ in log10 yr), which the youngest edge itself skips past, matching that scheme's own construction exactly. Either way the oldest edge is $t_{\rm age}(z)$ at the build's own redshift, so no default bin can lie beyond cosmic time. For a fixed redshift the ladder is built once, exactly at that value. For a free `redshift` the edges cannot be re-built per posterior draw (`make_agebins_from_zred` is a NumPy, non-traceable function), so they are built once, at $t_{\rm age}$ of the redshift prior's upper bound -- the youngest universe the prior admits -- and a build-time advisory (`NonparametricBinEdgesAtRedshiftCeilingWarning`) names the cosmic time a lower-redshift draw then cannot reach. Pass an explicit `sfh={'bin_edges_gyr': ...}` to use a different ladder; an explicit ladder is never modified.

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
