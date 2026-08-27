# Changelog

## v0.2.0 (in development)

Physics-model audit, new models, and cross-validation.

### API Changes
- **Metallicity is the `met` build group; `stellar` is removed (#1720).** Rename in your code: `stellar={'met_mode': 'table'}` → `met={'type': 'table'}`, `stellar={'met_logzsol': …}` → `met={'logzsol': …}`. Breaking with no alias — tengri is pre-1.0, and a build-group key is parsed rather than imported, so supporting both spellings would mean two grammars through `parse_groups`, `to_groups()`, the provenance tags and every wildcard sweep. `stellar=` raises carrying the translation. The old form stacked two anomalies: every other group selects its variant with `type` while `stellar` alone used `met_mode`, and the group was named for the component rather than for what it configured — so `met={'type': 'table'}`, the spelling both conventions imply, was the one form the grammar rejected. `tengri.list_metallicity_modes()` is the live menu; full before/after table in `docs/dev/api_migration_v0.x.md`.
- British → American spelling of public API identifiers (#819). tengri is pre-1.0, so the public API is not yet stable — these are renamed in place without deprecation aliases. Rename in your code: `cue_full_catalogue` / builder key `full_catalogue` → `cue_full_catalog` / `full_catalog`; `rest_frame_colour` → `rest_frame_color`; the four `*MarginalisedLikelihood` classes (and module `inference.likelihoods.marginalised`) → `*MarginalizedLikelihood` / `…marginalized`; `normalised_excess_variance` → `normalized_excess_variance`; `rank_normalise` / `rank_normalised_rhat` → `rank_normalize` / `rank_normalized_rhat`; `finalise` → `finalize`; `SSP_CATALOGUE_URL` → `SSP_CATALOG_URL`. Docs, examples, notebooks, and docstrings were converted to American English throughout (the HDF5 Synthesizer-grid keys `ionisation_parameter` / `log10_specific_ionising_luminosity` keep their upstream British spelling).

### New Physics Models
- **Dust**: SMC/LMC extinction curves (Pei 1992 Drude profiles), Witt & Gordon (2000) RT-based dust geometries (shell, cloudy, dusty), Casey (2012) MBB + mid-IR power law emission, Narayanan+2018 z-dependent attenuation priors
- **AGN**: Full Kubota & Done (2018) 3-zone accretion disc (outer disc + warm Comptonization + hot corona), Fe II pseudo-continuum in BLR (Tsuzuki+2006), polar dust reddening for Type 1 AGN (SMC law), spin-dependent Novikov-Thorne radiative efficiency
- **Nebular**: MAPPINGS V stellar and AGN photoionization grids (Flury et al. 2024, `neb={'type': 'mappings'}` or `'mappings_agn'` — both evaluate at runtime only; precompute adapters are defective pending #2078), MAPPINGS V shock emission lines (Allen+2008), diffuse ionized gas (DIG) mixing, analytic calibration polynomial marginalization (Johnson+2021)
- **Inference**: Laplace approximation, Pathfinder, Elliptical Slice Sampling, Nested Slice Sampling (NSS)

### Bug Fixes
- **`line_lums` published in [Lsun] by three of four nebular backends (2026-08, #1559)**: `state.derived["line_lums"]` declares `[erg/s]`, and `SEDModel.predict_line_fluxes` consumes it as such — it applies only `1 / (4 pi d_L^2)`. The `* L_sun` conversion lived inside `CueBackend` and nowhere else, so **CloudyGrid, CB19 and MappingsPhoto published `[Lsun]` into that key and every line flux they produced was a factor 3.839e33 too faint**, selected only by which backend the user named. Fixed by moving the conversion to the one seam in `NebularSEDComponent`: all four backends now return `[Lsun]` (restoring the documented backend contract), the component multiplies once, and the `#1073` compensating divide in `agn_nlr_cue` is gone with nothing left to cancel. Cue's published values are unchanged. **Affects saved posteriors and figures from line-flux fits using `cloudy`, `cb19` or `mappings` — those line fluxes were wrong and will move by 33.6 dex.** The constant is per-backend (`backend.lsun_erg`): Cue's network was trained on `L_sun = 3.839e33`, the grid backends on IAU 3.828e33, and using one for both is a systematic 0.287%. No per-backend test could see any of this — line ratios, `fesc` monotonicity and upstream-tabulation parity are all invariant under a global scale — so the guard pins `L(Halpha) / Q_H` against Case B recombination instead, which is backend-independent and moves by 33.6 dex under a unit error.
- **CGS unit standardization**: All SED component functions now return `erg/s/Hz` throughout. Previously `disc.py`, `torus.py`, `skirtor.py`, `unified.py`, `qsogen.py`, `radio.py`, `xray.py`, `cloudy_grid.py`, `cue.py`, `mappings_photo.py` returned `Lsun/Hz`. The CSP assembly always output `erg/s/Hz` — the mismatch was self-canceling for self-normalizing components but physically wrong. `agn_log_lbol` is now documented as the one API boundary in `log10(Lsun)`; all returns are `erg/s/Hz`. **Scope clarification (2026-05-17):** applies to continuous SED density (`predict_nebular_sed`, assembled SEDs). The discrete primitives `predict_nebular_line_luminosities` (Lsun) and `predict_nebular_continuum` (Lsun/Hz) use a separate convention to match FSPS/CLOUDY upstream — see the internal audit record for BUG-API-01 scope details.
- **Cue backend Lsun unification (2026-05-17)**: `CueBackend.predict_nebular_line_luminosities` was the only one of four nebular backends returning `[erg/s]` (CloudyGrid, CB19, MappingsPhoto, and the documented `state_to_emission_lines` bridge all use `[Lsun]`). Same method name, 3.83e33× factor difference; any caller switching backends silently got wrong line fluxes. Removed the spurious `* _LSUN_ERG` at the return; updated docstring units for both `predict_nebular_line_luminosities` and `predict_nebular_continuum` (the latter never multiplied — only the docstring lied). Closes the 5 remaining `TestCloudyVsCue` crossval failures.
- **Cue autodiff gradient (2026-05-17)**: `test_gradient_through_cue` was previously xfailed because the NN backward pass produced NaN at the erg/s output magnitude scale (~1e36). With the Lsun unification (above) operating at ~1e3, gradients are finite and match FD — Cue is now end-to-end differentiable. Removed the stale xfail.
- **Air/vacuum probe robustness (2026-05-16)**: `NebularSEDComponent.apply` classified Cue/CloudyGrid line lists from a single Hβ probe (±0.1 Å of 4861.333). Replaced with multi-line Balmer consensus (Hα, Hβ, Hγ) requiring ≥2 air hits with air > vacuum count. No known SSP grid triggered the brittle path, but the new probe survives single-line near-coincidences and float jitter.
- **Crossval suite un-rotted (2026-05-16)**: `tests/crossval/test_dust_crossval.py:200` used `np.clip(x, 0.0)` which NumPy 2.x rejects (added explicit `a_max=None`). `tests/crossval/test_nebular_crossval.py` compared backend `[erg/s/Hz]` output to raw SSP grid `[Lsun/Hz/Msun]` data without conversion — every failing ratio was exactly L_sun. Extracted a `_baked_nebular_lnu` helper. 24 of 29 failures resolved at this layer; the remaining 5 by the Cue unification (above).
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
