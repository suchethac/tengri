# Changelog

## v0.2.0 (in development)

Comprehensive physics model audit, new models, and cross-validation.

### New Physics Models
- **Dust**: SMC/LMC extinction curves (Pei 1992 Drude profiles), Witt & Gordon (2000) RT-based dust geometries (shell, cloudy, dusty), Casey (2012) MBB + mid-IR power law emission, Narayanan+2018 z-dependent attenuation priors
- **AGN**: Full Kubota & Done (2018) 3-zone accretion disc (outer disc + warm Comptonization + hot corona), Fe II pseudo-continuum in BLR (Tsuzuki+2006), polar dust reddening for Type 1 AGN (SMC law), spin-dependent Novikov-Thorne radiative efficiency
- **Nebular**: MAPPINGS V shock emission lines (Allen+2008), diffuse ionized gas (DIG) mixing, analytic calibration polynomial marginalization (Johnson+2021)
- **Inference**: Laplace approximation, Pathfinder, Elliptical Slice Sampling, Nested Slice Sampling (NSS)

### Bug Fixes
- SMC attenuation curve: replaced broken ad-hoc polynomials with Pei 1992 Drude profile sum
- BLR line strengths recalibrated to Vanden Berk+2001 (H-alpha was 4.7× too weak)
- [OIII] 5007/4959 ratio fixed to 2.98 (Storey & Zeippen 2000)
- AGN disc ring area: 4π²r·dr → 2πr·dr
- Radio/X-ray: fixed dimensional inconsistency in bolometric corrections (Hopkins+2007)
- CueBackend: neb_fesc now applied to nebular continuum (was silently dropped)
- SFR time-averaging: trapezoidal integration instead of biased grid-point counting
- Pipeline: radio/X-ray models now use computed L_ir, SFR, M* (were hardcoded defaults)
- CGM damping wing: disabled by default (experimental, unpublished reference)

### Documentation
- Reference notebooks extended with 14 new demonstration sections
- 13 new output figures for dust, AGN, nebular, and spectroscopic features
- API docs updated with all new modules

### Testing
- 1076 unit tests (up from 808)
- 115+ cross-validation tests against published reference values (CCM89, Calzetti, Inoue+2014, Bardeen+1972, Temple+2021, da Cunha+2013)

## v0.1.0

Initial release — methods paper (Paper I).

- IFT correlated field SFH model with PSD-governed burstiness priors
- Differentiable forward model in JAX (SFH → CSP → dust → photometry/spectroscopy)
- Six inference methods: MAP, Ray Tracing, NUTS, geoVI, MGVI, EVI
- Hierarchical population-level PSD inference
- Two-component dust attenuation (Calzetti, Kriek-Conroy, SMC, Cardelli, Salim, power-law)
- Dust IR emission (DL07, DL14, Dale 2014)
- AGN models (disc, SKIRTOR torus, power-law, QSOgen)
- Nebular emission (baked-in, CLOUDY grids, CUE neural emulator)
- Fused JIT kernels for optimized performance
- Mixed precision support
- 808 tests across unit, integration, and cross-validation tiers
