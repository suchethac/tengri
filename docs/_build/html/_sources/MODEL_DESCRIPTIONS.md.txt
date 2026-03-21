# tengri: Complete Model Descriptions

**For paper documentation — describes all physics modules, their mathematical
formulations, parameters, and references.**

---

## 1. Star Formation History

### 1.1 Parametric Models (8 types)

All SFH models return SFR(t) in Msun/yr as a function of lookback time.

| Model | Parameters | Reference |
|-------|-----------|-----------|
| `dpl` — Double Power Law | alpha, beta, tau_gyr, log_peak_sfr | Carnall+2018 |
| `tsnorm` — Truncated Skew-Normal | log_peak_sfr, peak_lbt_gyr, width_gyr, skew, trunc | Bellstedt+2020 |
| `snorm` — Skew-Normal | log_peak_sfr, peak_lbt_gyr, width_gyr, skew | Robotham+2020 |
| `norm` — Gaussian | log_peak_sfr, peak_lbt_gyr, width_gyr | — |
| `lnorm` — Log-Normal | log_peak_sfr, peak_lbt_gyr, width_gyr | — |
| `exp` — Exponential Decline | log_peak_sfr, tau_gyr, start_gyr | — |
| `dexp` — Delayed Exponential | log_peak_sfr, tau_gyr, start_gyr | — |
| `const` — Constant | log_sfr, start_gyr, end_gyr | — |

### 1.2 Burst Model

`burst` — Triweight kernel burst (Zacharegkas+2025): log_fburst, log_tpeak_myr, log_tmax_myr

### 1.3 Stochastic SFH (GP Field)

`field` — Gaussian process modulation via damped random walk PSD.

SFR_full(t) = SFR_mean(t) × exp(x(t) - K(0)/2)

where x(t) is a correlated Gaussian field with power spectrum:
P(f) = sigma^2 × tau / (1 + (2*pi*f*tau)^2)

Parameters: psd_sigma (amplitude, dex), psd_tau_myr (correlation timescale)
Latent: xi ~ N(0, I), dimension = n_grid (default 128)

**Composable:** Models can be combined: e.g., `["dpl", "burst", "field"]`

---

## 2. Stellar Population Synthesis

**SSP Library:** DSPS (Hearin+2023) with FSPS-generated templates

**CSP Integral:**
L_nu(lambda) = sum_i [w_i × SSP(lambda, Z, age_i) × dust(lambda, age_i)]

where w_i = SFR(age_i) × Delta_t_i (trapezoidal weights)

**Metallicity:** Linear interpolation in log(Z) between SSP grid points.
Supports single Z or evolving Z(t) = Z_final + (Z_initial - Z_final) × t/t_universe

---

## 3. Dust Attenuation

### 3.1 Two-Component Structure (Charlot & Fall 2000)

tau(lambda, age) = w(age) × tau_v1 × k_bc(lambda) + tau_v2 × k_diff(lambda)

where w(age) = sigmoid(-(log_age - log_t_birth) / width) is a smooth
birth-cloud dissolution transition at t_birth = 10 Myr.

### 3.2 Attenuation Curves k(lambda)

| Law | Parameters | Reference |
|-----|-----------|-----------|
| `power_law` | n_slope (default -0.7) | Charlot & Fall 2000 |
| `calzetti` | none (R_V=4.05 fixed) | Calzetti+2000 |
| `kriek_conroy` | dust_bump_strength, dust_delta | Kriek & Conroy 2013 |
| `smc` | none | Gordon+2003 / Pei 1992 |
| `cardelli` | dust_Rv | Cardelli+1989 |
| `salim` | dust_bump_strength, dust_delta | Salim+2018 |

**Per-component control:** Birth cloud and diffuse ISM can use different curves.

### 3.3 Clumpy Dust Geometry (f_obscuration)

transmission = f_obs + (1 - f_obs) × exp(-tau)

f_obs in [0,1]: fraction of sightlines that are completely unobscured.
f_obs = 0 recovers standard slab attenuation.

Reference: Lower+2022, Zacharegkas+2025

---

## 4. Dust Emission

### 4.1 Energy Balance

L_IR = eta × L_absorbed, where L_absorbed = integral[(L_intrinsic - L_attenuated) d_nu]

eta = dust_eta_balance (default 1.0 = strict energy balance)

### 4.2 Emission Models

| Model | Parameters | Reference |
|-------|-----------|-----------|
| `modified_blackbody` | dust_T, dust_beta_ir | Hildebrand 1983 |
| `dale2014` | dust_alpha_dale | Dale+2014 |
| `draine_li2007` (analytic) | dust_umin, dust_gamma_dl, dust_qpah | Draine & Li 2007 |
| `dl07_tabulated` | dust_umin, dust_gamma_dl, dust_qpah | Draine & Li 2007 (real templates) |

**DL07 Tabulated:** 7 q_PAH × 22 U_min × 1001 wavelength points.
j_nu = (1-gamma) × single_U(q_PAH, U_min) + gamma × powerlaw(q_PAH, U_min)
Bilinear interpolation in (q_PAH, U_min), linear gamma mixing.

---

## 5. Nebular Emission

### 5.1 CLOUDY Grid Backend

Precomputed CLOUDY photoionization grids (Byler+2017) with:
- 166 emission lines + nebular continuum
- Grid: 11 metallicities × 10 ages × 7 log(U) values
- Units: L_sun per ionizing photon Q_H
- Trilinear interpolation in (log Z, log age, log U)
- Q_H computed on-the-fly from SSP spectra (integral below 912 A)

Parameters: neb_logU, neb_logZ_gas, neb_fesc

### 5.2 Cue Neural Emulator (Li+2025)

Neural network emulator trained on CLOUDY with 12 input parameters:
- 7 ionizing spectrum shape (4 piecewise power-law slopes at He II/O II/He I/H I
  edges + 3 inter-segment flux ratios)
- 5 gas properties: log U, log n_H, [O/H], [N/O], [C/O]

Architecture: 3 hidden layers × 256 units, learned Swish activation,
PCA output basis (50 components). 16 sub-networks for line groups + 1 continuum.

**Ionizing spectrum precomputation:** 4-segment piecewise power law fitted
to SSP ionizing spectrum (lambda < 912 A) at each (Z, age). Users can
override as free parameters for non-stellar ionizing sources.

Reference: Li et al. 2025, ApJ 986, 9

### 5.3 BakedIn (Default)

Uses SSP files with pre-included nebular emission (wNE files with fixed
log U = -3, log Z = 0). No free parameters. Zero additional computation.

---

## 6. AGN Emission

### 6.1 Disc Models

| Model | Formula | Parameters |
|-------|---------|-----------|
| Power law | L_nu ~ nu^alpha × exp(-h*nu/kT_max) | agn_alpha, agn_T_max |
| Multi-color disc (Shakura-Sunyaev) | Sum of blackbodies at T(r) = T_in × (r/r_in)^{-3/4} | agn_log_mbh, agn_log_ledd, agn_a_spin |

### 6.2 Torus Models

| Model | Formula | Parameters |
|-------|---------|-----------|
| Single BB | L_nu ~ B_nu(T) × (1-exp(-tau×(9.7um/lambda)^beta)) | agn_T_torus, agn_tau_torus |
| Two-temperature | f_hot × B_nu(T_hot) + (1-f_hot) × B_nu(T_warm) | agn_T_hot, agn_T_warm, agn_frac_hot |

### 6.3 Registered Combined Models

| Name | Disc | Torus | Total params |
|------|------|-------|-------------|
| `simple` | Power law | Single BB | 3 (agn_frac, agn_alpha, agn_T_torus) |
| `standard` | Multi-color SS73 | Two-temperature | 6 |
| `kubota_done` | Kubota & Done 2018 3-zone | Two-temperature | 8+ |

AGN luminosity: L_bol_AGN = agn_frac × L_bol_stellar

References: Shakura & Sunyaev 1973, Kubota & Done 2018, Nenkova+2008

---

## 7. IGM Absorption (Inoue+2014)

T_IGM(lambda_obs, z) = exp(-tau_total)

tau_total = tau_LS_LAF + tau_LS_DLA + tau_LC_LAF + tau_LC_DLA

- Lyman-series absorption (39 lines, j=2..40): piecewise power-law
  with lower bound lambda_j < lambda_obs < lambda_j × (1+z)
- Lyman-continuum absorption: below 911.8 A with z-dependent regimes

Vectorized over all 39 lines simultaneously (no Python loop).
Cross-validated against bagpipes to < 1% agreement.

Reference: Inoue et al. 2014, MNRAS 442, 1805

---

## 8. Metallicity

### 8.1 Single Metallicity (Default)

Parameter: met_logzsol = log10(Z/Z_sun)
Linear interpolation in log(Z) between SSP grid points.

### 8.2 Evolving Metallicity

log Z(t) = Z_final + (Z_initial - Z_final) × t_lookback / t_universe

Parameters: met_logzsol_0 (oldest stars), met_logzsol_final (present-day)
Implemented via jax.vmap over SSP age bins.

---

## 9. Observation Models

### 9.1 Photometry

Broadband filter convolution with 50+ filters (SVO registry).
f_nu = (1+z)/(4*pi*d_L^2) × integral[L_nu × T(lambda) × lambda d_lambda] / integral[T × lambda d_lambda]

### 9.2 Spectroscopy

Pixel-level spectrum prediction with optional velocity dispersion broadening
(FFT convolution in log-wavelength).

### 9.3 Spectrophotometric Calibration

Chebyshev polynomial: C(lambda) = 1 + sum_n a_n × T_n(x)
Applied as multiplicative correction to observed spectrum.

---

## 10. Noise Model

### 10.1 Fractional Calibration Floor

sigma_eff^2 = sigma_obs^2 + (f_cal × flux)^2

### 10.2 Student-t Outlier Robustness

Replaces Gaussian likelihood with Student-t distribution for heavy tails.
Parameter: noise_dof (degrees of freedom; 0 = Gaussian)

---

## 11. Inference Methods

| Method | Type | Parameters | Reference |
|--------|------|-----------|-----------|
| MAP | Optimization | n_steps | — |
| NUTS | HMC (exact MCMC) | n_warmup, n_samples | Hoffman & Gelman 2014 |
| Ray Tracing | Exact MCMC | n_steps, step_size, n_leapfrog | Behroozi+2025 |
| geoVI | Variational (nonlinear) | n_iterations, n_samples | Frank+2021 |
| MGVI | Variational (linear) | n_iterations, n_samples | Knollmuller+2019 |
| EVI | JIT-compiled fast VI | n_iterations, n_samples | — |

---

## 12. Standardized Latent Space

All parameters (physical + GP latent) are reparameterized to:
xi ~ N(0, I)

The Hamiltonian:
H(xi) = 1/2 × chi^2(d, f(xi)) + 1/2 × xi^T × xi

This is exact (not an approximation): the Jacobian |dh/dxi| of each
transform h_k cancels the prior P(theta_k) by construction.

---

## Summary: Total Parameters

| Component | Free params (typical) |
|-----------|----------------------|
| SFH (dpl + field) | 6 + 2 = 8 |
| GP latent (n_grid=128) | 128 |
| Metallicity (evolving) | 2 |
| Dust attenuation (Kriek & Conroy) | 4 (tau_bc, tau_diff, bump, delta) |
| f_obscuration | 1 |
| Dust emission (DL07) | 3 (U_min, gamma, q_PAH) + 1 (eta) |
| Nebular (CLOUDY/Cue) | 3 (logU, Z_gas, f_esc) |
| AGN (simple) | 1 (agn_frac) |
| Redshift | 0-1 |
| Noise | 1-2 |
| **Total** | **~150-160** |
