# Changelog

## v0.2.0 (in development)

Comprehensive physics model audit, new models, and cross-validation.

### New Physics Models
- **Dust**: SMC/LMC extinction curves (Pei 1992 Drude profiles), Witt & Gordon (2000) RT-based dust geometries (shell, cloudy, dusty), Casey (2012) MBB + mid-IR power law emission, Narayanan+2018 z-dependent attenuation priors
- **AGN**: Full Kubota & Done (2018) 3-zone accretion disc (outer disc + warm Comptonization + hot corona), Fe II pseudo-continuum in BLR (Tsuzuki+2006), polar dust reddening for Type 1 AGN (SMC law), spin-dependent Novikov-Thorne radiative efficiency
- **Nebular**: MAPPINGS V shock emission lines (Allen+2008), diffuse ionized gas (DIG) mixing, analytic calibration polynomial marginalization (Johnson+2021)
- **Inference**: Laplace approximation, Pathfinder, Elliptical Slice Sampling, Nested Slice Sampling (NSS)

### Bug Fixes
- **CGS unit standardization**: All SED component functions now return `erg/s/Hz` throughout. Previously `disc.py`, `torus.py`, `skirtor.py`, `unified.py`, `qsogen.py`, `radio.py`, `xray.py`, `cloudy_grid.py`, `cue.py`, `mappings_photo.py` returned `Lsun/Hz`. The CSP assembly always output `erg/s/Hz` — the mismatch was self-cancelling for self-normalizing components but physically wrong. `agn_log_lbol` is now documented as the one API boundary in `log10(Lsun)`; all returns are `erg/s/Hz`.
- **Radio constants renamed**: `_L0_SYNCH_LSUN_HZ` → `_L0_SYNCH` (3.0e28 erg/s/Hz), `_C_FF_LSUN_HZ` → `_C_FF` (1/4.6e-28).
- SMC attenuation curve: replaced broken ad-hoc polynomials with Pei 1992 Drude profile sum
- BLR line strengths recalibrated to Vanden Berk+2001 (H-alpha was 4.7× too weak)
- [OIII] 5007/4959 ratio fixed to 2.98 (Storey & Zeippen 2000)
- AGN disc ring area: 4π²r·dr → 2πr·dr
- Radio/X-ray: fixed dimensional inconsistency in bolometric corrections (Hopkins+2007)
- CueBackend: neb_fesc now applied to nebular continuum (was silently dropped)
- SFR time-averaging: trapezoidal integration instead of biased grid-point counting
- Pipeline: radio/X-ray models now use computed L_ir, SFR, M* (were hardcoded defaults)
- CGM damping wing: disabled by default (experimental, unpublished reference)
- Nebular line profile: spurious `* _LSUN_ERG` removed from CLOUDY/Cue/shock Gaussian profiles
- Shock `sigma_nu`: missing 1e-8 Å→cm factor fixed; line widths were ~10⁸× too narrow
- XRB normalization: integrated over 2–10 keV band (200-point grid); was single-point (~2–3× error)
- CSP endpoint weights: trapezoidal half-widths at both endpoints (youngest/oldest bins were 2× too heavy)
- `continuity_sfh`/`dirichlet_sfh`: step-function bin assignment via `searchsorted` (Leja+2019); use `.shape[0]` to avoid `ConcretizationTypeError` under JIT
- Emission line wavelengths: all vacuum throughout (Hα = 6564.61 Å, Hβ = 4862.68 Å, [OIII]5007 = 5008.24 Å)
- QSOgen Balmer continuum optical depth direction corrected: τ ∝ (λ_BE/λ)³
- QSOgen hot dust normalization: `bbnorm` is now ratio f_bb/f_cont at 2μm anchor
- `narayanan_z`: tolerance comparison for float equality on traced values (JIT-safe)
- IGM LAF: clamps `z_obs ≥ 0` before fractional-power laws to prevent NaN
- BPT line ratios: return `NaN` for non-detections (not `log10(1e-30)`)
- nthcomp warm Comptonization: HDF5 template path (`data/nthcomp_templates.h5`, build once via RELAGN)

### Documentation
- Reference notebooks extended with 14 new demonstration sections
- 13 new output figures for dust, AGN, nebular, and spectroscopic features
- API docs updated with all new modules

### Testing
- 2211 tests (up from 1221), including cross-validation against RELAGN nthcomp, bagpipes/FSPS
- Cross-validation against published reference values (CCM89, Calzetti, Inoue+2014, Bardeen+1972, Temple+2021, da Cunha+2013)

## v0.1.0

Initial release — methods paper (Paper I).

- IFT correlated field SFH model with PSD-governed burstiness priors
- Differentiable forward model in JAX (SFH → CSP → dust → photometry/spectroscopy)
- Ten inference methods: MAP, native_geovi, Ray Tracing, NUTS, Laplace, Pathfinder, Elliptical Slice, NSS, geoVI, MGVI
- Hierarchical population-level PSD inference
- Two-component dust attenuation (Calzetti, Kriek-Conroy, SMC, Cardelli, Salim, power-law)
- Dust IR emission (DL07, DL14, Dale 2014)
- AGN models (disc, SKIRTOR torus, power-law, QSOgen)
- Nebular emission (baked-in, CLOUDY grids, CUE neural emulator)
- Fused JIT kernels for optimized performance
- Mixed precision support
- 1221 tests across unit, integration, and cross-validation tiers
