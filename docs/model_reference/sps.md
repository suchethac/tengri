(app-sps-details)=

# Stellar Population Synthesis

tengri uses DSPS(Hearin et al. 2023) as its differentiable SPS engine. The subsections below describe the SSP template format ({ref}`app-sps-templates`), the five CSP integration modes ({ref}`app-sps-csp`), photometric precomputation ({ref}`app-sps-precompute`), and the surviving mass fraction ({ref}`app-sps-mass-remaining`). Metallicity interpolation and chemical evolution are covered in Appendix {ref}`app-met-details`; $\alpha$-element enhancement in Appendix {ref}`app-alpha-details`.

(app-sps-templates)=

## SSP Template Structure

Each SSP library is stored as a three-dimensional flux tensor $\mathbf{F} \in \mathbb{R}^{n_Z \times n_{\rm age} \times n_\lambda}$ in units of $L_\odot\,\mathrm{Hz}^{-1}\,M_\odot^{-1}$, loaded from an HDF5 file in the DSPS format (Hearin et al. 2023). The three axes are:

- **Metallicity grid** $\{Z_i\}_{i=1}^{n_Z}$ in $\log_{10}(Z/Z_\odot)$, typically 4--22 points depending on the library;

- **Age grid** $\{t_j\}_{j=1}^{n_{\rm age}}$ in $\log_{10}(t_{\rm age}/\mathrm{Gyr})$, logarithmically spaced from ${\sim}1\,\mathrm{Myr}$ to $t_{\rm Hubble}$ (typically 94--188 points);

- **Wavelength grid** $\{\lambda_k\}_{k=1}^{n_\lambda}$ in Å (rest-frame), from ${\sim}100$ Å to ${\sim}2\,\mu$m (typically 1200--5000 points).

Each file also carries a surviving mass fraction array $f_{\rm surv}(Z, t_{\rm age})$ of shape $(n_Z, n_{\rm age})$ and, optionally, a wavelength-integrated bolometric luminosity array for energy-balance dust emission. tengri supports templates from Bruzual and Charlot (2003) (BC03), BPASS (Eldridge et al. 2017), FSPS(Conroy et al. 2009; Conroy and Gunn 2010), and ProGeny (Bellstedt and Robotham 2024). The IMF is encoded entirely in the templates; switching from Chabrier (2003) to Kroupa or Salpeter requires loading a different SSP file with no code changes.

(app-sps-csp)=

## CSP Integration Schemes

The composite stellar population (CSP) flux is the convolution of the SFH with the SSP template library: $$F_\nu(\lambda) = \int_0^{t_{\rm obs}} \mathrm{SFR}(t)\,
S_\nu(\lambda,\,t,\,Z)\,dt,

$$ (eq-csp-integral)
 where $S_\nu$ is the SSP spectrum (in $L_\odot\,\mathrm{Hz}^{-1}\,M_\odot^{-1}$), $t$ is lookback time, and $Z$ is the stellar metallicity interpolated as described in {ref}`app-sps-metallicity`. Both $\mathrm{SFR}(t)$ and $S_\nu(t)$ are only known at the discrete age grid $\{t_j\}_{j=0}^{N-1}$ (spaced logarithmically from ${\sim}1\,\mathrm{Myr}$ to $t_{\rm Hubble}$), so the integral must be approximated numerically. tengri provides five integration modes; the recommended default is described below, followed by four alternatives for specialized use cases.

#### DSPS native (`dsps_native`).

This is the recommended integration mode. It delegates both age integration and metallicity marginalization to the dsps library (Hearin et al. 2023), using a single call to `calc_rest_sed_sfh_table_lognormal_mdf`.

The SFH is integrated on *cosmic* time $t_{\rm cosmic}$ (not lookback time), which avoids the endpoint half-width ambiguity that affects the trapezoidal lookback-time schemes. The metallicity distribution at each age is a lognormal (Gaussian in $\log_{10}Z$) with scatter $\sigma_{\log Z}$ (default 0.2 dex), following the Prospector convention (Johnson 2021): $$P(\log_{10}Z) \propto
  \exp\!\left[-\frac{(\log_{10}Z - \mu_Z)^2}{2\sigma_{\log Z}^2}\right],

$$ (eq-lognormal-mdf)
 where $\mu_Z = \log_{10}Z_{\rm gal}$ is the mean galaxy metallicity. DSPS evaluates this integral with the triweight kernel ({ref}`app-sps-metallicity`; (Hearin et al. 2023, Eq. 10)), which computes the metallicity-weighted SSP flux $\bar{S}_\nu(\lambda, t)$ at each age simultaneously with the age weights. The age weights are normalized (sum $= 1$) and then scaled to absolute mass by $M_{\star} = \int \mathrm{SFR}\,dt$.

The tengri wrapper (`compute_dsps_native_weights`) handles the lookback-to-cosmic time conversion internally and returns results in tengri's ascending-age convention; the outputs drop directly into the existing dust and AGN pipeline.

#### Alternative integration modes.

Four additional modes are available for comparison and specialized use cases. All modes share the same external interface, the choice is a model initialization flag (`csp_integration`).

#### DSPS met-table (`dsps_met_table`).

This mode extends `dsps_native` to support a *time-evolving* metallicity $Z(t)$, replacing the single-$\mu_Z$ lognormal with a per-age metallicity table. It delegates to `calc_rest_sed_sfh_table_met_table` in dsps, which accepts a vector $\{\log_{10}Z(t_j)\}_{j=0}^{N-1}$ of the same length as the SFH table and returns a two-dimensional weight tensor of shape $(n_{\rm met}, n_{\rm age})$: $$w_{mj} = P\!\left(\log_{10}Z_m \mid \log_{10}Z(t_j),\,\sigma_{\log Z}\right)
          \cdot w_j^{\rm age},

$$ (eq-dsps-met-table)
 where $P(\cdot|\mu,\sigma)$ is the lognormal kernel evaluated at SSP metallicity grid point $m$, $w_j^{\rm age}$ is the normalized age weight from the SFH integration, and $\sigma_{\log Z}$ (default 0.2 dex) is the intrinsic scatter applied at each age independently. The metallicity-marginalized SSP flux at each age is then $$\bar{S}_\nu(\lambda, t_j)
  = \sum_m \tilde{w}_{mj}\, S_\nu(\lambda, Z_m, t_j),

$$ (eq-dsps-met-table-flux)
 where $\tilde{w}_{mj} = w_{mj} / \sum_m w_{mj}$ are the column-normalized weights.

The primary use case is chemical evolution: the per-age metallicity array $\{\log_{10}Z(t_j)\}$ is produced by `compute_log_z_evolving`, which implements a closed-box or leaky-box model (Bellstedt et al. 2021) parametrized by an initial and final metallicity. When the metallicity is constant across all ages, `dsps_met_table` reduces exactly to `dsps_native` up to floating-point rounding.

#### Trapezoidal (`trapz`).

The integrand is held constant within each bin at its grid-point value, yielding mass weights $$w_j = \mathrm{SFR}(t_j)\,\Delta t_j,

$$ (eq-csp-trapz)
 where the bin half-widths are $\Delta t_0 = (t_1 - t_0)/2$, $\Delta t_j = (t_{j+1} - t_{j-1})/2$ for $0 < j < N-1$, and $\Delta t_{N-1} = (t_{N-1} - t_{N-2})/2$. This is standard trapezoidal quadrature of the SFH evaluated at the SSP age nodes.

#### Log-spaced trapezoidal (`log_trapz`).

Substituting $u = \log_{10}(t)$ transforms the Jacobian as $dt = t\ln(10)\,du$, giving $$w_j = \mathrm{SFR}(t_j)\,t_j\,\ln(10)\,\Delta(\log_{10} t_j),

$$ (eq-csp-logtrapz)
 where $\Delta(\log_{10} t_j)$ are half-widths in log-age. The extra $t_j$ factor upweights older populations, which occupy wider linear-time intervals per log-age bin. For the exponentially sampled grids used by FSPS and DSPS (where adjacent bins span factors of $2$--$10$ in $t$), this scheme reduces quadrature error relative to `trapz` without any additional cost.

#### Log-linear interpolation (`log_interp`).

Following Johnson (2021) Appendix B, the SSP spectrum $S_\nu(t)$ is assumed to vary linearly in $u = \log_{10}(t)$ between adjacent grid points. In each interval $[t_j, t_{j+1}]$, two piecewise-linear hat functions define the interpolation: $$\begin{aligned}
a_j(t) &= \frac{\log_{10} t_{j+1} - \log_{10} t}%
               {\log_{10} t_{j+1} - \log_{10} t_j}, \\
b_{j+1}(t) &= 1 - a_j(t), 
\end{aligned}

$$ (eq-hat-a)
 so that $S_\nu(t) \approx a_j(t)\,S_\nu(t_j) + b_{j+1}(t)\,S_\nu(t_{j+1})$. Substituting into Equation {eq}`eq-csp-integral` and collecting terms by SSP index, the CSP flux becomes $F_\nu(\lambda) = \sum_j m_j\,S_\nu(\lambda,t_j)$, where the effective mass weight assigned to SSP $j$ is $$\begin{aligned}
 
m_j = {} & \int_{t_{j-1}}^{t_j} \mathrm{SFR}(t)\,b_j(t)\,dt \nonumber \\
    & + \int_{t_j}^{t_{j+1}} \mathrm{SFR}(t)\,a_j(t)\,dt.
\end{aligned}

$$ (eq-csp-loginterp-weights)
 The SFR within each interval is further approximated by linear interpolation between adjacent grid values, $\mathrm{SFR}(t)\approx s_j(1-p)+s_{j+1}p$ with $p=(t-t_j)/(t_{j+1}-t_j)$, so the integrand in each interval is a product of two piecewise-linear functions. Both integrals are evaluated with 5-point Gauss--Legendre quadrature per interval, which is exact for polynomial integrands of degree $\leq 9$. The overall weight computation then reduces to a single matrix--vector product, $$\mathbf{m} = \mathbf{A}\,\mathbf{s},

$$ (eq-csp-matrix)
 where $s_j = \mathrm{SFR}(t_j)$, and the weight matrix $\mathbf{A}\in\mathbb{R}^{N\times N}$ is tridiagonal, non-negative, and precomputed once at model initialization from the SSP age grid alone.

(app-sps-precompute)=

## Photometric Precomputation

The AB mean flux density in filter $b$ is $$f_b = \frac{\int F_\nu(\lambda)\,T_b(\lambda)\,w(\lambda)\,d\lambda}
             {\int T_b(\lambda)\,w(\lambda)\,d\lambda},

$$ (eq-bandpass-flux)
 where $T_b(\lambda)$ is the filter transmission curve and $w(\lambda)$ is the bandpass weight set by the convention. We adopt by default the photon-counting (Bessell) convention, $w(\lambda)=1/\lambda$, in which {eq}`eq-bandpass-flux` is the standard photon-counting AB mean (Hogg et al. 2002, Eq. 5); this is the convention of fsps (Conroy and Wechsler 2009) and dsps (Hearin et al. 2023) and is correct for photon-counting detectors, with $F_\nu$ in $L_\odot\,{\rm Hz^{-1}}$ relative to $AB_0 = 1.13492\times
10^{-13}\,L_\odot\,{\rm Hz^{-1}}$ (Oke and Gunn 1983). tengri also supports the energy convention $w(\lambda)=1/\lambda^2$ (i.e. $\int F_\nu
T\,d\nu/\int T\,d\nu$), which reproduces cigale (Boquien et al. 2019); the two differ by $5$--$40$ mmag for non-flat SEDs. On a typical SSP wavelength grid ($n_\lambda \sim 7000$) this integral dominates the per-call cost of photometric inference.

The key observation is that the SSP spectrum $F_\nu^{\rm SSP}(Z_i, t_j, \lambda)$ is a fixed lookup table that does not depend on the sampled model parameters (SFH weights, dust optical depths, metallicity index). Only the weights $w_{ij}$ and the dust attenuation $A(\lambda)$ change from sample to sample.

Template-based dust emission models (DL07, Dale et al. 2014) share this structure: their grids depend on a small number of free parameters ($q_{\rm PAH}$, $U_{\rm min}$ for DL07; $\alpha$ for Dale) and can also be preintegrated through filters. Energy-normalized templates $\hat{\Phi}_b(q_{\rm PAH}, U_{\rm min})$ are precomputed at model initialization, and at inference the dust IR photometry is simply $L_{\rm abs} \times \hat{\Phi}_b$, no wavelength-level template evaluation is needed. Components without fixed template grids (AGN disc, radio, X-ray, Cue neural emulator) are always evaluated at full wavelength resolution.

#### Precomputed SSP photometry.

Following Zacharegkas et al. (2025), we factor the integral by pulling the dust attenuation $A(\lambda)$ out of the wavelength sum. We precompute the filter-integrated SSP tensor $$\Phi_{ijb}
    = \frac{\int F_\nu^{\rm SSP}(Z_i,\,t_j,\,\lambda)\,T_b(\lambda)\,w(\lambda)\,d\lambda}
           {\int T_b(\lambda)\,w(\lambda)\,d\lambda}

$$ (eq-ssp-phot)
 once on the full wavelength grid via trapezoidal quadrature. This tensor has shape $(n_{\rm met}, n_{\rm age}, n_{\rm filt})$ and is exact to machine precision. The per-call cost drops from $\mathcal{O}(n_\lambda)$ multiplications to $\mathcal{O}(n_{\rm met}\cdot n_{\rm age})$, yielding a ${\sim}20\times$ speedup in gradient evaluation (Zacharegkas et al. 2025).

#### Single-wavelength dust approximation.

Dust attenuation is evaluated at a single transmission-weighted effective wavelength per filter, $$\lambda_{\rm eff,b}
    = \frac{\int \lambda\,T_b\,w(\lambda)\,d\lambda}
           {\int T_b\,w(\lambda)\,d\lambda}\,

$$ (eq-lambda-eff)
 the first moment of the bandpass weight (so that $\Psi_{ijb}=0$ for a flat template, centering the Taylor expansion consistently), giving $$f_b \approx A(\lambda_{\rm eff,b})\sum_{ij}w_{ij}\,\Phi_{ijb}\,

$$ (eq-phot-approx-n1)
 where $w_{ij}$ are the CSP mass weights (Appendix {ref}`app-sfh-library`). This is the approach of Zacharegkas et al. (2025), also used by cigale (Boquien et al. 2019) and Sedition (Behroozi et al. 2019). It is exact when $A(\lambda)$ is constant across the bandpass. In practice the single-wavelength approximation at $\lambda_{\rm eff}$ is already an excellent estimator of the filter-weighted dust average; the dominant residual comes not from the dust evaluation point but from the *factorization* itself, pulling $A$ outside the integral neglects the covariance of $F_\nu^{\rm
SSP}$ and $A$ within the filter: $$\langle F_\nu A \rangle_b
  = \langle F_\nu\rangle_b\, \langle A\rangle_b
  + \operatorname{Cov}_b(F_\nu, A).

$$ (eq-factorization-error)
 For a $\lambda^{-2}$ SSP with Charlot--Fall dust ($\tau_{\rm BC}=1$, $n=-0.7$), this covariance term reaches $1.1\%$ in SDSS $g$ and sets an irreducible floor on any scheme that precomputes $\Phi$ and $A$ separately (Table {ref}`1 <tab-quad-accuracy>`).

#### Taylor correction.

We capture the SSP--dust covariance to first order by expanding $A(\lambda)$ around $\lambda_{\rm eff}$: $$A(\lambda) = A(\lambda_{\rm eff})
  + A'(\lambda_{\rm eff})\,(\lambda - \lambda_{\rm eff})
  + \mathcal{O}\!\bigl((\lambda - \lambda_{\rm eff})^2\bigr).$$ Substituting into Equation {eq}`eq-bandpass-flux` and keeping the linear term gives $$f_b \approx \sum_{ij} w_{ij}\bigl[
    A(\lambda_{\rm eff,b})\,\Phi_{ijb}
    + A'(\lambda_{\rm eff,b})\,\Psi_{ijb}
  \bigr],

$$ (eq-taylor-photometry)
 where the *spectral moment tensor* $$\Psi_{ijb}
    = \frac{\int F_\nu^{\rm SSP}(Z_i,t_j,\lambda)\,
      (\lambda - \lambda_{\rm eff,b})\,T_b\,w(\lambda)\,d\lambda}
      {\int T_b\,w(\lambda)\,d\lambda}

$$ (eq-ssp-moment)
 has the same shape as $\Phi_{ijb}$ and is precomputed once on the full wavelength grid at model initialization. For a spectrally flat SSP, $\Psi_{ijb} = 0$ by definition of $\lambda_{\rm eff}$; for steep spectra $\Psi$ encodes how much flux is shifted blueward or redward of $\lambda_{\rm eff}$ within the bandpass.

At inference the only additional cost is $A'(\lambda_{\rm eff,b})$, the derivative of the attenuation curve at one wavelength per filter. For any dust law (power-law, Calzetti, Cardelli, SMC) this is computed via central finite differences at $\pm 1\,\text{\AA}$, two extra dust-law evaluations per filter, independent of the SSP grid size. For the Charlot--Fall two-component model with birth-cloud and diffuse attenuation: $$A'_{\rm young}(\lambda_{\rm eff})
    = \frac{A(\lambda_{\rm eff}+\delta) - A(\lambda_{\rm eff}-\delta)}{2\delta},
    \qquad \delta = 1\,\text{\AA},$$ where $A = \exp(-\tau_{\rm BC}\,k_{\rm BC})\,\exp(-\tau_{\rm diff}\,k_{\rm diff})$ for young stars and $A = \exp(-\tau_{\rm diff}\,k_{\rm diff})$ for old stars.

#### Accuracy.

Table {ref}`1 <tab-quad-accuracy>` compares the single-wavelength approximation (Equation {eq}`eq-phot-approx-n1`) against the Taylor-corrected version (Equation {eq}`eq-taylor-photometry`) for SDSS $ugriz$ and LSST $grizy$ with a worst-case $\lambda^{-2}$ power-law SSP and Charlot--Fall dust ($\tau_{\rm BC}=1$, $n=-0.7$). The reference is a 10 000-point trapezoidal integral. The Taylor correction reduces the worst-case error from $1.3\%$ to $0.26\%$ for SDSS and from $1.6\%$ to $0.31\%$ for LSST, a ${\sim}5\times$ improvement, by capturing the leading-order SSP--dust covariance that the single-wavelength method misses.

(tab-quad-accuracy)=

|            | Photometry error (%) |                            |
|:-----------|---------------------:|---------------------------:|
| 2-3 Filter |         $A\cdot\Phi$ | $A\cdot\Phi + A'\cdot\Psi$ |
| *SDSS*     |                      |                            |
| $u$        |                 0.67 |                       0.11 |
| $g$        |                 1.33 |                       0.26 |
| $r$        |                 0.51 |                       0.11 |
| $i$        |                 0.38 |                       0.08 |
| $z$        |                 0.43 |                       0.10 |
| Max        |                 1.33 |                       0.26 |
| Mean       |                 0.66 |                       0.13 |
| *LSST*     |                      |                            |
| $g$        |                 1.56 |                       0.31 |
| $r$        |                 0.70 |                       0.15 |
| $i$        |                 0.36 |                       0.08 |
| $z$        |                 0.16 |                       0.04 |
| $y$        |                 0.16 |                       0.04 |
| Max        |                 1.56 |                       0.31 |
| Mean       |                 0.59 |                       0.12 |

: Relative photometry error (%) of the precomputed-photometry approximation. Test: $\lambda^{-2}$ power-law SSP, Charlot--Fall dust ($\tau_{\rm BC}=1$, $n=-0.7$). Reference: 10 000-point trapezoidal integration. "$A\cdot\Phi$" is the single-wavelength method of Zacharegkas et al. (2025); "$A\cdot\Phi + A'\cdot\Psi$" adds the first-order Taylor correction (Equation {eq}`eq-taylor-photometry`).

For realistic galaxy SEDs (smoother than $\lambda^{-2}$) these worst-case values overestimate the actual bias by roughly an order of magnitude. The Taylor correction adds one tensor of the same shape as $\Phi$ to the precomputed state and negligible inference-time cost.

#### Free-redshift precomputation.

When the redshift is a free parameter, we precompute $\Phi_{ijb}$ (and $\Psi_{ijb}$) on a uniform grid of $n_z$ redshifts and interpolate to the current $z$ at each inference step. We use the same triweight kernel described in {ref}`app-sps-metallicity` for this interpolation, with bandwidth $h = 0.5\,\Delta z$. The $C^2$-continuous weights eliminate the derivative discontinuities that piecewise-linear interpolation would introduce at grid nodes, these kinks slow gradient-based optimizers (VI, MAP) by creating a corrugated log-likelihood surface. At $n_z=100$ on $z\in[0.01,3]$ the maximum interpolation error is $<\!0.05\%$ for smooth spectra.

#### Prediction modes.

tengri provides three forward model evaluation paths with different speed/accuracy trade-offs:

1.  **Exact**, evaluates the full SED pipeline at all $n_\lambda$ wavelengths with Python dispatch between components. Machine-precision reference path; typical latency ${\sim}0.1$--$1\,\text{s}$ depending on model complexity.

2.  **Compositional**, fuses the entire forward model (SFH $\to$ CSP $\to$ dust $\to$ nebular $\to$ AGN $\to$ filter integration) into a single `jax.jit`-compiled XLA graph. Bit-identical to exact (same physics, same code paths); $50$--$250\times$ faster by eliminating Python dispatch overhead and enabling XLA fusion of array operations. Default mode.

3.  **Hybrid**, stellar photometry via the precomputed $\Phi_{ijb}$ tensor (Equation {eq}`eq-ssp-phot`), non-stellar components at full wavelength. $200$--$750\times$ faster than exact; ${\lesssim}0.06\%$ error from the Taylor dust approximation. Used when speed is critical (initial exploration, batch inference, real-time visualization).

Table {ref}`2 <tab-mode-benchmarks>` shows representative timings for the full model (stellar + Cue nebular + THEMIS dust emission + Kubota & Done 3-zone AGN + radio + X-ray, $D=10$, SDSS $ugriz$, $z=0.1$, Apple M-series CPU).

(tab-mode-benchmarks)=

| Mode          | Latency |     Speedup |   Max error |
|:--------------|--------:|------------:|------------:|
| Exact         |   1.4 s |   $1\times$ |   reference |
| Compositional |   11 ms | $108\times$ | $10^{-5}\%$ |
| Hybrid        |  4.4 ms | $224\times$ |    $0.01\%$ |

: Forward model latency by prediction mode for the full panchromatic model ($D=10$). "Speedup" is relative to exact. "Error" is the maximum per-band relative error versus exact on filters with non-negligible flux.

IGM absorption (Inoue et al. 2014) is evaluated at full wavelength resolution in all three modes. The hybrid kernel pre-computes IGM transmission on the SSP wavelength grid at model initialization and applies it to non-stellar emission before filter integration, ensuring correct handling of Lyman-series absorption features at all redshifts ($z = 0$--$8$ tested).

(app-sps-subband)=

## Sub-band Quadrature Precomputation

The single-wavelength scheme of {ref}`app-sps-precompute` pulls the attenuation out of the bandpass integral, and Equation {eq}`eq-factorization-error` identifies what that costs: the covariance $\operatorname{Cov}_b(F_\nu, A)$ between the SSP spectrum and the dust screen inside the filter, an irreducible floor for any scheme that precomputes $\Phi$ and $A$ separately. The default precomputation path in tengri does not treat that term as irreducible. It retains the parameter-independent SSP integral but captures the in-band covariance to quadrature order by evaluating the dust screen at a handful of nodes per band rather than at a single effective wavelength.

#### Equal-mass sub-band partition.

Each filter $b$ is split into $K$ sub-bands carrying equal filter mass, i.e. the edges $\lambda_{b,0}<\dots<\lambda_{b,K}$ are the $K$-quantiles of the cumulative bandpass weight, $$\int_{\lambda_{b,k}}^{\lambda_{b,k+1}} T_b(\lambda)\,w(\lambda)\,d\lambda
  = \frac{1}{K}\int T_b(\lambda)\,w(\lambda)\,d\lambda ,
  \qquad k = 0,\dots,K-1 .

$$ (eq-subband-edges)
 For each SSP template $(Z_i, t_j)$ we precompute the filter integral restricted to each sub-band and the template's own flux-weighted centroid there, $$\begin{aligned}
  \Phi_{ijbk}
    &= \frac{\int_{\lambda_{b,k}}^{\lambda_{b,k+1}}
        F_\nu^{\rm SSP}(Z_i, t_j, \lambda)\,T_b\,w(\lambda)\,d\lambda}
        {\int T_b\,w(\lambda)\,d\lambda} , \\[4pt]
  \lambda^{\star}_{ijbk}
    &= \frac{\int_{\lambda_{b,k}}^{\lambda_{b,k+1}}
        \lambda\,F_\nu^{\rm SSP}(Z_i, t_j, \lambda)\,T_b\,w(\lambda)\,d\lambda}
        {\int_{\lambda_{b,k}}^{\lambda_{b,k+1}}
        F_\nu^{\rm SSP}(Z_i, t_j, \lambda)\,T_b\,w(\lambda)\,d\lambda} .
  
\end{aligned}

$$ (eq-subband-phi)
 Both tensors have shape $(n_{\rm met}, n_{\rm age}, n_{\rm filt}, K)$; they are built once at model initialization from the SSP grid alone (the partition is constructed from cumulative integrals interpolated at the edges, so the sub-band integrals sum to the full band integral by construction: $\sum_k \Phi_{ijbk} = \Phi_{ijb}$). The nodes $\lambda^{\star}_{ijbk}$ are observed-frame and depend on the template through its spectral shape.

#### Runtime contraction.

Writing the age- and metallicity-dependent multiplicative screen in the Charlot and Fall (2000) form as $A(\lambda\,|\,j) = A_{\rm diff}(\lambda)\,
A_{\rm bc}(\lambda)^{y_j}$, where $A_{\rm diff}$ is the diffuse-ISM transmission, $A_{\rm bc}$ the birth-cloud transmission, and $y_j \in [0,1]$ the smooth young-star indicator of age bin $j$, the band flux is approximated by evaluating the screen at each node, $$f_b \approx \sum_{ij} w_{ij} \sum_{k=1}^{K}
    \Phi_{ijbk}\;
    A_{\rm diff}\!\bigl(\lambda^{\star}_{ijbk}\bigr)\,
    A_{\rm bc}\!\bigl(\lambda^{\star}_{ijbk}\bigr)^{y_j} ,

$$ (eq-subband-flux)
 with $w_{ij}$ the CSP mass weights (Appendix {ref}`app-sfh-library`). This is exact in the limit that the screen is piecewise constant across sub-bands, and because each node is the template's own first moment, the first moment of the in-band residual vanishes identically, a first-order Taylor term on top of Equation {eq}`eq-subband-flux` would add exactly zero, so the sub-band rule subsumes the moment correction of {ref}`app-sps-precompute` rather than competing with it. At inference the only per-sample work is $2K$ dust-law evaluations per band (the screen at the $K$ nodes for the diffuse and birth-cloud components), independent of the SSP grid size $n_\lambda$; the metallicity axis is contracted into the per-age weights so the contraction is a single sum over age bins and sub-bands.

#### IGM band-averaging.

The same node structure carries the intergalactic-medium absorption. The mean Inoue et al. (2014) transmission $T_{\rm IGM}(\lambda_{\rm obs}, z)$ is a fixed function of observed-frame wavelength and redshift with no free parameters, so it can be folded into the sub-band weights at build time, $$\Phi^{\rm IGM}_{ijbk}
    = \Phi_{ijbk}\;
      T_{\rm IGM}\!\bigl(\lambda^{\star}_{ijbk}(1+z),\; z\bigr),

$$ (eq-igm-fold)
 leaving the runtime contraction {eq}`eq-subband-flux` unchanged in shape and cost. Folding $T_{\rm IGM}$ at the nodes, rather than multiplying the band flux by a bandpass-averaged transmission, matters wherever $T_{\rm IGM}$ varies strongly inside a filter. Band-averaging the transmission alone forms $\langle F_\nu\rangle_b\,\langle T_{\rm IGM}\rangle_b$ where the flux requires $\langle F_\nu\,T_{\rm IGM}\rangle_b$; the two differ by the same covariance identity as Equation {eq}`eq-factorization-error`, $$\langle F_\nu\,T_{\rm IGM}\rangle_b
  = \langle F_\nu\rangle_b\,\langle T_{\rm IGM}\rangle_b
  + \operatorname{Cov}_b\!\bigl(F_\nu,\,T_{\rm IGM}\bigr),

$$ (eq-igm-covariance)
 and that covariance reaches ${\approx}-9.5\%$ in GALEX FUV at $z\approx0.8$, where $T_{\rm IGM}$ runs from ${\sim}1$ to ${\sim}0$ across the band. The fold is applied on the metallicity axis *before* the CSP contraction: the node $\lambda^{\star}_{ijbk}$ depends on the template and hence on $Z_i$, shifting by up to ${\sim}68\%$ of a sub-band width across the SSP grid (and $T_{\rm IGM}$ there by up to $1.3\%$ in GALEX FUV), so evaluating $T_{\rm IGM}$ at a single metallicity-averaged node would miss this dependence. Absorption models that read free parameters (patchy reionization, discrete DLAs; Appendix {ref}`app-igm-details`) are not fixed functions of $(\lambda, z)$ and are kept on the live per-call evaluation.

#### Additive components.

Sub-band quadrature is needed only for the multiplicative screen on the stellar and nebular continuum. Additive emitters, dust infrared re-emission, radio, X-ray, and the AGN components, factor through the filters exactly. Each is a sum of rank-one terms, a scalar amplitude $A_k(\boldsymbol{\theta})$ times a spectral shape $S_k(\lambda)$ fixed by the emitter's shape parameters, $$L_\nu(\lambda) = \sum_k A_k(\boldsymbol{\theta})\, S_k(\lambda),

$$ (eq-rank1-sed)
 so by linearity of the filter integral the band flux is $$f_b = \sum_k A_k(\boldsymbol{\theta})\, R_{kb},
  \qquad
  R_{kb} = \frac{\int S_k(\lambda)\,T_b\,w(\lambda)\,d\lambda}
                {\int T_b\,w(\lambda)\,d\lambda},

$$ (eq-rank1-response)
 where the band-response tensor $R_{kb}$, of shape $(n_{\rm terms}, n_{\rm filt})$, is a build-time constant. No effective-wavelength or quadrature approximation enters: Equation {eq}`eq-rank1-response` is machine-precision exact, and the amplitudes $A_k$ are the only quantities recomputed per sample.

#### Accuracy.

The quadrature converges as $1/K^2$. Table {ref}`3 <tab-subband-convergence>` lists the worst-case per-band error against the exact wavelength-grid path ({ref}`app-sps-precompute`, the reference is always the exact integral, never a lower-order preintegral) over $z\le1$ and dust optical depth $\tau\le2$. The default $K=5$ holds the worst rest-UV band below ${\sim}0.6\%$, where the single-wavelength scheme reads GALEX FUV $+45\%$ high at $z=0.05$, rising to $+215\%$ at $z=1$ as the attenuation curve steepens and the linear extrapolation diverges. At $K=5$ the runtime cost is slightly *below* that of the Taylor form it replaces, because the moment scheme carries a second tensor and a power with a traced exponent.

(tab-subband-convergence)=

| $K$ (sub-bands) | Worst-band error |
|----------------:|-----------------:|
|               1 |             8.7% |
|               3 |             1.4% |
|               5 |             0.6% |
|               8 |             0.3% |

: Convergence of the sub-band quadrature. Worst-case per-band relative photometry error versus the exact wavelength-grid path, over $z\le1$ and dust optical depth $\tau\le2$; the worst band is GALEX FUV, where the attenuation curve is steepest across the bandpass. The default $K=5$ is used throughout tengri; $K=0$ recovers the single-wavelength scheme of {ref}`app-sps-precompute`.

(app-feature-precompute)=

## Emission-Line and Spectroscopic Precomputation

The photometric precomputation of {ref}`app-sps-subband` has line and spectral siblings that share its design: integrate the parameter-independent structure once at build time, and reduce each likelihood call to a small contraction.

#### Emission lines.

For an emulator backend such as Cue (Li et al. 2025), each nebular line luminosity is exactly linear in the hydrogen-ionizing photon rate $Q_{\rm H}$ and independent of the *shape* of the star formation history, $$L_{\rm line} = Q_{\rm H}\,\ell(\boldsymbol{\theta}_{\rm ion}, Z),

$$ (eq-line-per-qh)
 where $\ell$ \[erg s$^{-1}$ per photon s$^{-1}$\] is the per-photon line luminosity, a function of the fixed gas ionization parameters $\boldsymbol{\theta}_{\rm ion}$ (ionization parameter, gas metallicity, escape fraction) and the stellar metallicity $Z$, but not of the SFH, the SFH enters lines only through the scalar $Q_{\rm H}$. The factor $\ell$ is tabulated on a grid over the free ionization axes (by default $16$ points per axis; the metallicity dependence is nonlinear through the shape of the ionizing spectrum, so a dense grid is required). At inference the observed line flux is $$F_{\rm line} = \frac{Q_{\rm H}\,\ell(\boldsymbol{\theta}_{\rm ion}, Z)}
                      {4\pi\,d_L(z)^2},

$$ (eq-line-reconstruct)
 with $Q_{\rm H}$ the ionizing rate published by the stellar component. The table stores a luminosity (distance-independent), so it is valid at any redshift and the cosmological factor is applied at evaluation. Because no downstream quantity needs the full-wavelength SED, the stellar synthesis is pruned entirely from the line likelihood. Backends that carry their lines inside the SSP templates use a per-line window LUT of SSP integrals instead, measured off the spectrum and contracted with the SFH weights and the dust screen at the line-window centers.

#### Spectroscopy.

The spectroscopic LUT precomputes the stellar${}\times{}$dust${}\times{}$IGM stack at the spectrum pixel centers, rest-frame effective wavelengths, and caches it per redshift, the pixel-space analog of the photometric LUT. A pixel is a point rather than a bandpass, so the dust attenuation is evaluated exactly at each pixel wavelength; there is no in-band covariance to correct and no sub-band quadrature is needed. Emission lines, being delta-like, cannot be represented by a per-pixel effective-wavelength LUT and are rasterized onto the pixel grid separately at projection time, while the smooth continuum uses the LUT. The continuum reconstruction is machine-precision with the dust screen off and holds to $\lesssim1\%$ with dust, degrading only in the deep rest-UV at high optical depth.

#### Free redshift and API.

When the redshift is a free parameter, each table, the photometric sub-band tensor with folded IGM {eq}`eq-igm-fold`, the spectral stack, and the line grid, is built on a uniform grid of $n_z$ redshifts (default $100$) and interpolated at the current $z$ with the same $C^2$-continuous triweight kernel used for the fixed-redshift tensor ({ref}`app-sps-precompute`), whose smooth weights keep the log-likelihood surface free of the derivative kinks that piecewise-linear interpolation would introduce. Precomputation is opt-in at model construction through the `approx` argument: `approx=None` selects the exact wavelength-grid reference path, while `WavePrecomp`, `SpectrumPrecomp`, and `FeaturePrecomp` (singly or in combination) build the corresponding tables. At fit time the default policy routes the likelihood through the table appropriate to the data, photometry through the sub-band LUT, emission lines through the line grid, unless the model was built with an explicit `approx`, while `predict` and all post-hoc analysis retain the exact path. Every table is validated against `approx=None`: the exact integration is the sole reference, and a precompute is never checked against a lower-order preintegral.

#### Performance.

Table {ref}`4 <tab-precompute-timing>` lists per-call forward-photometry latencies for the sub-band LUT against the exact path, by emitter family, for the SDSS $ugriz$ filter set at $z=0.1$. The stellar continuum, the dominant cost of the exact path, accelerates by ${\sim}400\times$; a panchromatic model combining all emitter families accelerates by ${\sim}30\times$, the residual cost being the components evaluated at full wavelength resolution. What matters for inference is the gradient: `jax.grad` of the photometry through the LUT runs $9$--$19\times$ faster than through the compiled full-wavelength path (largest for the stellar-dominated models, ${\sim}9.6\times$ for the full panchromatic model), a speedup that carries directly into MCMC and VI wall-clock. On the end-to-end headline case, joint GALEX-to-WISE photometry ($11$ bands) plus five DESI emission-line fluxes, a MAP fit completes in $6.4$ s with a $3.4$ ms warm forward evaluation (versus $33$ ms exact), and the compiled kernel is reused across a catalog without recompilation at ${\approx}1000$ galaxies per core-hour. Against a mock generated with the exact model, the sub-band LUT recovers MAP parameters to $0.002$ per parameter at $z=0.05$ while running $13\times$ faster, so the approximation is recovery-unbiased for low-redshift catalogs; the exact path remains available for rest-UV-critical high-redshift work. Precomputing the line channel on top of the photometry compounds the gain: on a local DESI catalog (DECam $grz$ + WISE, ten optical lines, redshift pinned), a Cue model costs $8.96$ s per galaxy at full wavelength resolution, $0.62$ s with the photometry on the LUT, and $0.046$ s with the line channel also precomputed, a $196\times$ reduction at matched final loss, with the cold compile falling from ${\sim}10$ minutes to ${\sim}5$ seconds.

(tab-precompute-timing)=

| Configuration                      |   Exact |        LUT |     Speedup |
|:-----------------------------------|--------:|-----------:|------------:|
| Stellar only                       | 23.9 ms |  59 $\mu$s | $408\times$ |
| ${}+{}$dust IR (THEMIS)            | 27.3 ms | 158 $\mu$s | $173\times$ |
| ${}+{}$nebular, dust, radio, X-ray | 27.3 ms | 472 $\mu$s |  $58\times$ |
| Panchromatic (all emitters)        | 76.1 ms |    2.45 ms |  $31\times$ |

: Forward-photometry latency of the sub-band precompute LUT versus the exact wavelength-grid path, per call, for a double-power-law SFH ($D=6$) with the SDSS $ugriz$ filter set at $z=0.1$ (median of $200$ timed calls, CPU). "Speedup" is exact${}/{}$LUT. Gradient speedups over the compiled full-wavelength path and the end-to-end catalog figures are given in the text.

(app-sps-mass-remaining)=

## Mass-Remaining Fraction

When the SSP file contains the surviving mass fraction $f_{\rm surv}(Z, t_{\rm age})$, the surviving stellar mass is $$M_{\star,\rm surv} = \sum_j w_j \cdot f_{\rm surv}(Z, t_j),$$ where $w_j$ are the CSP mass weights ({ref}`app-sfh-library`). This accounts for mass loss from stellar winds, supernovae, and remnant formation, and depends on the assumed IMF and isochrone library.

## References

Behroozi, Peter, Matthew Becker, Frank C. van den Bosch, et al. 2019. "Empirically Constraining Galaxy Evolution." 51 (3): 125. <https://doi.org/10.48550/arXiv.1903.04509>.

Bellstedt, Sabine, and Aaron S. G. Robotham. 2024. "ProGeny II: the impact of libraries and model configurations on inferred galaxy properties in SED fitting." *arXiv e-Prints*, arXiv:2410.17698. <https://doi.org/10.48550/arXiv.2410.17698>.

Bellstedt, Sabine, Aaron S. G. Robotham, Simon P. Driver, et al. 2021. "Galaxy and mass assembly (GAMA): the inferred mass-metallicity relation from z = 0 to 3.5 via forensic SED fitting." 503 (3): 3309--25. <https://doi.org/10.1093/mnras/stab550>.

Boquien, M., D. Burgarella, Y. Roehlly, et al. 2019. "CIGALE: a python Code Investigating GALaxy Emission." 622 (February): A103. <https://doi.org/10.1051/0004-6361/201834156>.

Bruzual, G., and S. Charlot. 2003. "Stellar population synthesis at the resolution of 2003." 344 (4): 1000--1028. <https://doi.org/10.1046/j.1365-8711.2003.06897.x>.

Chabrier, Gilles. 2003. "Galactic Stellar and Substellar Initial Mass Function." 115 (809): 763--95. <https://doi.org/10.1086/376392>.

Charlot, Stéphane, and S. Michael Fall. 2000. "A Simple Model for the Absorption of Starlight by Dust in Galaxies." 539 (2): 718--31. <https://doi.org/10.1086/309250>.

Conroy, Charlie, and James E. Gunn. 2010. "The Propagation of Uncertainties in Stellar Population Synthesis Modeling. III. Model Calibration, Comparison, and Evaluation." 712 (2): 833--57. <https://doi.org/10.1088/0004-637X/712/2/833>.

Conroy, Charlie, James E. Gunn, and Martin White. 2009. "The Propagation of Uncertainties in Stellar Population Synthesis Modeling. I. The Relevance of Uncertain Aspects of Stellar Evolution and the Initial Mass Function to the Derived Physical Properties of Galaxies." 699 (1): 486--506. <https://doi.org/10.1088/0004-637X/699/1/486>.

Conroy, Charlie, and Risa H. Wechsler. 2009. "Connecting Galaxies, Halos, and Star Formation Rates Across Cosmic Time." 696 (1): 620--35. <https://doi.org/10.1088/0004-637X/696/1/620>.

Eldridge, J. J., E. R. Stanway, L. Xiao, et al. 2017. "Binary Population and Spectral Synthesis Version 2.1: Construction, Observational Verification, and New Results." 34: e058. <https://doi.org/10.1017/pasa.2017.51>.

Hearin, Andrew P., Jonás Chaves-Montero, Alex Alarcon, Matthew R. Becker, and Andrew Benson. 2023. "DSPS: Differentiable stellar population synthesis." 521 (2): 1741--56. <https://doi.org/10.1093/mnras/stad456>.

Hogg, David W., Ivan K. Baldry, Michael R. Blanton, and Daniel J. Eisenstein. 2002. "The K correction." *arXiv e-Prints*, astro--ph/0210394. <https://doi.org/10.48550/arXiv.astro-ph/0210394>.

Inoue, Akio K., Ikkoh Shimizu, Ikuru Iwata, and Masayuki Tanaka. 2014. "An updated analytic model for attenuation by the intergalactic medium." 442 (2): 1805--20. <https://doi.org/10.1093/mnras/stu936>.

Johnson, Benjamin D. 2021. *bd-j/sedpy: sedpy v0.2.0*. Version v0.2.0. Zenodo. <https://doi.org/10.5281/zenodo.4582723>.

Li, Yongda, Joel Leja, Benjamin D. Johnson, and Sandro Tacchella. 2025. "Cue: A Fast, Flexible, and Accurate Neural Emulator for Nebular Emission." <https://arxiv.org/abs/2312.12345>.

Oke, J. B., and J. E. Gunn. 1983. "Secondary standard stars for absolute spectrophotometry." 266 (March): 713--17. <https://doi.org/10.1086/160817>.

Zacharegkas, Georgios, Andrew Hearin, and Andrew Benson. 2025. "Bayesian Posteriors with Stellar Population Synthesis on GPUs." *The Open Journal of Astrophysics* 8 (December). <https://doi.org/10.33232/001c.151255>.
