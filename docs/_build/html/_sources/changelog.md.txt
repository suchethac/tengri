# Changelog

## v0.1.0 (in development)

Initial release — methods paper (Paper I).

- IFT correlated field SFH model with PSD-governed burstiness priors
- Differentiable forward model in JAX (SFH → CSP → dust → photometry/spectroscopy)
- Six inference methods: MAP, Ray Tracing, NUTS, geoVI, MGVI, EVI
- Hierarchical population-level PSD inference
- Two-component dust attenuation (Calzetti, Kriek-Conroy, SMC, Cardelli, Salim, power-law)
- Dust IR emission (DL07, DL14)
- AGN models (disc, SKIRTOR torus, power-law)
- Nebular emission (baked-in, CLOUDY grids, CUE neural emulator)
- Fused JIT kernels for optimized performance
- Mixed precision support
- 808 tests across unit, integration, and cross-validation tiers
