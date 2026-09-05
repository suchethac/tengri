(app-observation-details)=

# Observation Models

This appendix details the observation-layer models that project the intrinsic SED into observable space.

(app-photometry)=

## Photometry

Observed flux densities are computed by convolving the redshifted SED through filter transmission curves in the AB system: $$f_\nu^{\rm obs}(b) = \frac{1+z}{4\pi d_L^2} \cdot \frac{\int L_\nu\bigl(\lambda/(1+z)\bigr)\, T_b(\lambda)\, \lambda\, d\lambda}{\int T_b(\lambda)\, \lambda\, d\lambda},

$$ (eq-photometry)
 where $T_b(\lambda)$ is the filter transmission, $d_L$ is the luminosity distance, and the factor $(1+z)$ converts the rest-frame $L_\nu$ to the observed frame. AB magnitudes follow as $m_{\rm AB} = -2.5\,\log_{10}(f_\nu) - 48.6$.

(app-spectroscopy)=

## Spectroscopy

#### Pixel-level model.

The model spectrum at observed pixel wavelengths $\lambda_j$ is $$f(\lambda_j) = \frac{1+z}{4\pi d_L^2} \cdot L_\nu\!\left(\frac{\lambda_j}{1+z}\right).$$

#### Velocity broadening.

Stellar velocity dispersion $\sigma_v$ is applied via FFT convolution in log-wavelength space (equivalent to velocity space, $\Delta v/c = \Delta\ln\lambda$). The kernel is a Gaussian with width $\sigma_{\rm pix} = (\sigma_v/c) / \Delta\ln\lambda$ pixels.

#### Line spread function (LSF).

The effective LSF subtracts the SSP library resolution in quadrature: $$\sigma_{\rm eff}(\lambda) = \sqrt{\sigma_{\rm inst}(\lambda)^2 - \sigma_{\rm lib}^2},$$ where $\sigma_{\rm inst} = c/(2.355\, R(\lambda))$ in km s$^{-1}$. For constant $R$ (e.g., a grating), a single FFT convolution suffices. For wavelength-dependent $R$ (e.g., JWST NIRSpec PRISM, $R \approx 30$--$330$), a piecewise-constant approximation splits the wavelength range into $\mathord{\sim}16$ segments with smooth raised-cosine blending, yielding $\mathord{\sim}1\%$ accuracy.

#### Chebyshev calibration polynomial.

A multiplicative polynomial absorbs flux-calibration uncertainties (Johnson 2021): $$C(\lambda) = \sum_{k=0}^{K} c_k\, T_k(u), \quad
u \equiv \frac{2\lambda - \lambda_{\min} - \lambda_{\max}}{\lambda_{\max} - \lambda_{\min}},

$$ (eq-chebyshev)
 where $T_k$ are Chebyshev polynomials of the first kind, $c_0 = 1$ (fixed), and $c_1, \ldots, c_K$ are free parameters with priors $c_k \sim \mathcal{N}(0, \sigma_c)$. The default is $K = 0$, i.e. calibration is disabled and the predicted spectrum is returned unmodified; $K = 3$ with $\sigma_c = 0.1$ is the recommended configuration when continuum systematics are expected, and both $K$ and $\sigma_c$ are configurable via `Parameters`. The model spectrum is $f_{\rm obs}(\lambda) = C(\lambda) \cdot f_{\rm model}(\lambda)$.

(app-joint-fitting)=

## Joint Fitting

Photometry and spectroscopy can be fit simultaneously by concatenating their data vectors and noise vectors. Each data type uses its own forward model (Equations {eq}`eq-photometry` and {eq}`eq-chebyshev`) but shares the same physical parameters.

## References

Johnson, Benjamin D. 2021. *bd-j/sedpy: sedpy v0.2.0*. Version v0.2.0. Zenodo. <https://doi.org/10.5281/zenodo.4582723>.
