# Noise Model for Photometric SED Fitting

## 1. Motivation

Photometric SED fitting requires a likelihood function that accurately
describes the statistical relationship between model predictions and
observed data. The standard approach assumes independent Gaussian noise
with known variance:

$$
\ln \mathcal{L}(\boldsymbol{d} \mid \boldsymbol{\theta}) = -\frac{1}{2} \sum_k \frac{(d_k - m_k(\boldsymbol{\theta}))^2}{\sigma_{\mathrm{obs},k}^2}
$$

where $d_k$ is the observed flux in band $k$, $m_k(\boldsymbol{\theta})$
is the model prediction, and $\sigma_{\mathrm{obs},k}$ is the reported
photometric uncertainty.

This formulation breaks down when photometric uncertainties are
underestimated, which is common in practice due to:

1. **Calibration systematics**: zero-point offsets, filter curve
   uncertainties, and photometric calibration errors that scale with flux
   (typically 1-10% per band; Bellstedt et al. 2020; Leja et al. 2019).

2. **Model inadequacy**: stellar population synthesis models carry
   irreducible uncertainties from TP-AGB treatment, horizontal branch
   morphology, and stellar atmosphere approximations (Conroy 2013;
   Conroy & Gunn 2010).

3. **Aperture and deblending errors**: systematic biases from source
   extraction, particularly in crowded fields (Robotham et al. 2018).

When the true noise exceeds the reported uncertainties, the chi-squared
term dominates the posterior, which contracts to an overconfident point
estimate. In the standardized formulation used by tengri, the loss is
$H(\boldsymbol{\xi}) = \frac{1}{2}\chi^2 + \frac{1}{2}\boldsymbol{\xi}^T\boldsymbol{\xi}$,
where $\boldsymbol{\xi} \sim \mathcal{N}(0, I)$ are standardized latent
variables. If $\chi^2 \gg \|\boldsymbol{\xi}\|^2$, the prior term
(which encodes the GP correlated field and all physical parameter priors)
becomes negligible, and the posterior collapses.

## 2. Information Field Theory Framework

We adopt the Information Field Theory (IFT; Ensslin et al. 2009;
Ensslin 2019) perspective where the noise is part of the generative
model:

$$
\boldsymbol{d} = R(\boldsymbol{s}) + \boldsymbol{n}, \qquad \boldsymbol{n} \sim \mathcal{N}(0, N(\boldsymbol{\gamma}))
$$

where $R$ is the instrument response (the SED forward model),
$\boldsymbol{s}$ is the signal (physical galaxy parameters), and
$N(\boldsymbol{\gamma})$ is the noise covariance parameterized by
hyperparameters $\boldsymbol{\gamma}$.

In the standard IFT approach, both $\boldsymbol{s}$ and
$\boldsymbol{\gamma}$ are inferred jointly from the data. This is
implemented through NIFTy's (Edenhofer et al. 2024) variable-covariance
likelihood classes, which naturally include the log-determinant
normalization that prevents the trivial solution of infinite noise.

## 3. Effective Noise Model

### 3.1 Calibration Floor

The effective noise variance per photometric band is:

$$
\sigma_{\mathrm{eff},k}^2 = \sigma_{\mathrm{obs},k}^2 + (f_{\mathrm{cal}} \cdot |m_k(\boldsymbol{\theta})|)^2
$$

where $f_{\mathrm{cal}}$ is a fractional calibration uncertainty parameter.
This parameterization captures the multiplicative nature of calibration
errors: brighter sources have larger absolute calibration uncertainties.
This is the same form used by Prospector (Johnson et al. 2021,
Appendix D), where the calibration uncertainty is implemented as an
"uncorrelated kernel" with weights equal to the model flux vector, and
by Alsing et al. (2022), who decompose the model uncertainty as
$\Sigma_b^2 = (\gamma_b F_b)^2 + \text{emission line terms}$.

The parameter $f_{\mathrm{cal}}$ is either fixed at a known value for a
given survey (as in Bellstedt et al. 2020, who use band-dependent floors
of 3-16%) or inferred from the data with a prior such as
$f_{\mathrm{cal}} \sim \mathrm{Uniform}(0.01, 0.2)$.

### 3.2 Log-Likelihood with Variable Noise

When the noise covariance depends on model parameters, the full
Gaussian log-likelihood is:

$$
\ln \mathcal{L} = -\frac{1}{2} \sum_k \frac{(d_k - m_k)^2}{\sigma_{\mathrm{eff},k}^2} - \sum_k \ln \sigma_{\mathrm{eff},k} - \frac{N_{\mathrm{bands}}}{2} \ln(2\pi)
$$

The log-determinant term $-\sum_k \ln \sigma_{\mathrm{eff},k}$ is
critical: it penalizes noise inflation and prevents the sampler from
trivially reducing chi-squared by inflating uncertainties to infinity.
Without this term, the model would have a flat direction in the
$f_{\mathrm{cal}} \to \infty$ limit. With the log-determinant, there
exists a finite optimal $f_{\mathrm{cal}}$ that balances data fit quality
against noise model complexity.

Equivalently, writing $\tau_k = 1/\sigma_{\mathrm{eff},k}$ (the
precision), the negative log-likelihood energy is:

$$
E_{\mathrm{lh}} = \frac{1}{2} \sum_k (d_k - m_k)^2 \tau_k^2 - \sum_k \ln \tau_k
$$

This is the energy implemented by NIFTy's `VariableCovarianceGaussian`
likelihood (Edenhofer et al. 2024), which expects the forward model to
return a tuple $(m_k, \tau_k)$.

### 3.3 Combined Loss in Standardized Coordinates

In the standardized coordinate system where all parameters are
transformed to $\boldsymbol{\xi} \sim \mathcal{N}(0, I)$, the full
Hamiltonian (negative log-posterior) is:

$$
H(\boldsymbol{\xi}) = \frac{1}{2} \sum_k \left(\frac{d_k - m_k(\boldsymbol{\xi})}{\sigma_{\mathrm{eff},k}(\boldsymbol{\xi})}\right)^2 + \sum_k \ln \sigma_{\mathrm{eff},k}(\boldsymbol{\xi}) + \frac{1}{2} \boldsymbol{\xi}^T \boldsymbol{\xi}
$$

where $f_{\mathrm{cal}}(\boldsymbol{\xi})$ enters through its own prior
transform (e.g., $f_{\mathrm{cal}} = \mathrm{sigmoid}(\xi_{f_{\mathrm{cal}}}) \cdot (f_{\max} - f_{\min}) + f_{\min}$).

## 4. Outlier Model (Student-t Likelihood)

### 4.1 Motivation

Even with a calibration floor, individual photometric measurements can
be catastrophically wrong due to cosmic ray hits, detector artifacts,
incorrect deblending, or unmasked satellite trails. The Gaussian
likelihood assigns exponentially vanishing probability to such outliers,
which can dominate the chi-squared sum and bias the fit.

Two approaches exist in the literature:

1. **Mixture model** (Hogg, Bovy & Lang 2010): the likelihood for each
   data point is a mixture of a "good" Gaussian and a broadened "bad"
   Gaussian:

   $$
   p(d_k \mid \boldsymbol{\theta}) = (1 - f_{\mathrm{out}}) \cdot \mathcal{N}(d_k; m_k, \sigma_{\mathrm{eff},k}^2) + f_{\mathrm{out}} \cdot \mathcal{N}(d_k; m_k, s_{\mathrm{out}}^2 \cdot \sigma_{\mathrm{eff},k}^2)
   $$

   This is used by Prospector (Johnson et al. 2021) with parameters
   $f_{\mathrm{out}} \sim \mathrm{TopHat}(10^{-5}, 0.5)$ and
   $s_{\mathrm{out}} = 50$ (fixed).

2. **Student-t likelihood**: replace the Gaussian with a Student-t
   distribution with $\nu$ degrees of freedom:

   $$
   p(d_k \mid \boldsymbol{\theta}) = \frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi}\,\sigma_{\mathrm{eff},k}} \left(1 + \frac{(d_k - m_k)^2}{\nu\,\sigma_{\mathrm{eff},k}^2}\right)^{-(\nu+1)/2}
   $$

   This is used by ProSpect (Robotham et al. 2020) with adaptive
   degrees of freedom, and by Alsing et al. (2022) with $\nu = 2$.

### 4.2 Our Implementation: VariableCovarianceStudentT

We use NIFTy's `VariableCovarianceStudentT` likelihood (Edenhofer et al.
2024), which combines variable noise with heavy tails. The energy is:

$$
E_{\mathrm{lh}} = \frac{\nu + 1}{2} \sum_k \ln\left(1 + \frac{(d_k - m_k)^2}{\nu\,\sigma_{\mathrm{eff},k}^2}\right) + \sum_k \ln \sigma_{\mathrm{eff},k}
$$

The parameter $\nu$ controls outlier robustness:
- $\nu = 2$: heavy tails (Alsing et al. 2022 recommendation)
- $\nu = 4$: moderate robustness
- $\nu \to \infty$: recovers the Gaussian likelihood

The `VariableCovarianceStudentT` expects the forward model to output
$(m_k, \sigma_{\mathrm{eff},k})$ and includes the log-determinant term
$\sum_k \ln \sigma_{\mathrm{eff},k}$.

For $\nu = 2$, a data point with a $5\sigma$ residual contributes
$\sim 2.3$ to the energy, compared to $12.5$ for a Gaussian. This
natural downweighting of outliers avoids the need for explicit
outlier identification and masking.

### 4.3 Degrees of Freedom

The degrees of freedom $\nu$ can be:

- **Fixed**: $\nu = 2$ (Alsing et al. 2022) or $\nu = 4$ (moderate).
  This is the simplest approach and avoids an additional free parameter.

- **Adaptive** (ProSpect approach): estimated from the residual variance
  at each likelihood evaluation:
  $\hat{\nu} = 2\hat{V}/(\hat{V} - 1)$
  where $\hat{V} = \mathrm{Var}((d_k - m_k)/\sigma_{\mathrm{eff},k})$.
  When residuals are well-described by the noise ($\hat{V} \approx 1$),
  $\hat{\nu} \to \infty$ and the Student-t converges to a Gaussian.

- **Inferred**: $\nu$ as a free parameter with prior
  $\nu \sim \mathrm{Uniform}(1, 30)$.

## 5. Comparison with Other Codes

| Feature | tengri | Prospector | BAGPIPES | ProSpect | Alsing+2022 |
|---------|---------|-----------|----------|----------|-------------|
| **Fractional calibration floor** | $f_{\mathrm{cal}}$ (free or fixed) | Multiplicative jitter kernel | No | Band-dependent, pre-applied | $\gamma_b$ (hierarchical) |
| **Log-determinant** | Yes (VariableCovGaussian) | Yes (when C varies) | Yes (spec scaling only) | No | Yes |
| **Outlier robustness** | Student-t (VariableCovStudentT) | Hogg+2010 mixture | No | Adaptive Student-t | Student-t($\nu=2$) |
| **NIFTy-native** | Yes | No | No | No | No |
| **Noise params inferred** | Yes (part of $\boldsymbol{\xi}$) | Yes (separate $\boldsymbol{\beta}$) | Partial (spec only) | No (pre-applied) | Yes (hierarchical) |
| **Correlated noise** | Not yet (diagonal) | GP kernels | GP kernels | No | No |

## 6. Implementation Details

### 6.1 Parameters

| Parameter | Symbol | Description | Default | Typical Prior |
|-----------|--------|-------------|---------|---------------|
| `noise_frac_cal` | $f_{\mathrm{cal}}$ | Fractional calibration floor | `Fixed(0.0)` = OFF | `Uniform(0.01, 0.2)` |
| `noise_dof` | $\nu$ | Student-t degrees of freedom | `Fixed(2.0)` | `Fixed(2.0)` or `Uniform(1, 30)` |

When `noise_frac_cal=Fixed(0.0)` (default), the noise model is
inactive and the code uses the standard fixed-noise Gaussian likelihood.
This ensures backward compatibility.

### 6.2 NIFTy Likelihood Dispatch

The likelihood class is selected based on active noise parameters:

```
if noise_frac_cal is free:
    if noise_dof is set:
        → jft.VariableCovarianceStudentT(data, dof)
            model output: (predicted, sigma_eff)
    else:
        → jft.VariableCovarianceGaussian(data)
            model output: (predicted, std_inv)
else:
    → jft.Gaussian(data, 1/noise_obs^2)
        model output: predicted
```

### 6.3 Gauss-Newton Metric for Variable Noise

The geoVI algorithm requires the Fisher information metric (or its
Gauss-Newton approximation) for the coordinate transformation. For the
`VariableCovarianceGaussian` energy with $E(f, \tau) = \frac{1}{2}\sum r_k^2 - \sum \ln \tau_k$
where $r_k = (d_k - f_k)\tau_k$, the Hessian of $E$ with respect to
$(f, \tau)$ is:

$$
\frac{\partial^2 E}{\partial f_k^2} = \tau_k^2, \qquad
\frac{\partial^2 E}{\partial \tau_k^2} = (d_k - f_k)^2 + \tau_k^{-2}, \qquad
\frac{\partial^2 E}{\partial f_k \partial \tau_k} = -2(d_k - f_k)\tau_k
$$

The metric-vector product for the latent parameters $\boldsymbol{\xi}$
is then:

$$
M(\boldsymbol{\xi}) \cdot \boldsymbol{v} = J^T H_E \, J \boldsymbol{v} + \boldsymbol{v}
$$

where $J = [\partial f / \partial \boldsymbol{\xi}; \partial \tau / \partial \boldsymbol{\xi}]$
is the Jacobian of the forward model output. The cross-terms between
$f$ and $\tau$ are included, coupling the signal and noise Jacobians
in the metric. This is computed efficiently using one JVP (forward-mode)
and one VJP (reverse-mode) autodiff pass.

## 7. Practical Recommendations

1. **Start with the calibration floor only**: set
   `noise_frac_cal=Uniform(0.01, 0.2)` and leave `noise_dof` at default.
   This handles the most common failure mode (underestimated
   uncertainties for bright sources).

2. **Add Student-t for real data**: set `noise_dof=Fixed(2.0)` to
   activate outlier robustness. This is particularly important for
   survey data with occasional catastrophic photometric errors.

3. **For surveys with known calibration**: fix $f_{\mathrm{cal}}$ at
   the known value per band (e.g., Bellstedt et al. 2020, Table 1:
   3-16% depending on band). This avoids adding a free parameter.

4. **Hierarchical inference**: $f_{\mathrm{cal}}$ can be shared across
   a galaxy population, effectively learning the survey's calibration
   quality from the ensemble. This follows the approach of Alsing et al.
   (2022), who find band-level calibration uncertainties $\gamma_b$ at
   the few-percent level.

## References

- Alsing, J., Peiris, H., Mortlock, D., Leistedt, B., & Leja, J. 2022,
  arXiv:2207.07673 — Hierarchical noise model with multiplicative
  zero-points and Student-t likelihood

- Bellstedt, S., et al. 2020, MNRAS, 498, 5581 — ProSpect SFH fitting
  with band-dependent error floors (Table 1)

- Carnall, A. C., et al. 2018, MNRAS, 480, 4379 — BAGPIPES
  spectroscopic noise model with GP kernels

- Conroy, C. 2013, ARA&A, 51, 393 — Sources of model uncertainty in
  stellar population synthesis

- Conroy, C., & Gunn, J. E. 2010, ApJ, 712, 833 — Propagation of
  uncertainties in SPS models

- Czekala, I., et al. 2015, ApJ, 812, 128 — Starfish: decomposed noise
  covariance with GP model inadequacy kernels

- Edenhofer, G., et al. 2024, arXiv:2402.16683 — NIFTy.re: JAX-based
  re-implementation with variable-covariance likelihoods

- Ensslin, T. A., et al. 2009, PhRvD, 80, 105005 — Information field
  theory for cosmological perturbation reconstruction

- Ensslin, T. A. 2019, Annalen der Physik, 531, 1800127 — Information
  theory for fields

- Frank, P., Leike, R., & Ensslin, T. A. 2021, Entropy, 23, 853 —
  Geometric Variational Inference (geoVI)

- Hogg, D. W., Bovy, J., & Lang, D. 2010, arXiv:1008.4686 — Data
  analysis recipes: fitting a model to data (outlier mixture model)

- Johnson, B. D., et al. 2021, ApJS, 254, 22 — Prospector: stellar
  population inference with flexible noise model (Appendix D)

- Knollmuller, J., & Ensslin, T. A. 2019, arXiv:1901.11033 — Encoding
  prior knowledge in the structure of the likelihood

- Leja, J., et al. 2019, ApJ, 877, 140 — Photometric error floors and
  zero-point adjustments

- Robotham, A. S. G., et al. 2020, MNRAS, 495, 905 — ProSpect: SFH
  fitting with adaptive Student-t likelihood

- Robotham, A. S. G., et al. 2018, MNRAS, 476, 3137 — ProFound:
  source extraction and photometric uncertainty estimation
