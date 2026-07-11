# Synthesizer-parity reference

This document maps synthesizer-project/synthesizer's default SED-model assumptions
to the equivalent tengri config. It is the spec for `tengri.presets.synthesizer_default()`.

**Purpose:** tengri implements a subset of synthesizer's unified AGN + dust + nebular
physics using JAX-compatible analytic models rather than grids. This reference ensures
that identical parameter choices produce equivalent SEDs (parity test via
`tests/regression/synthesizer_parity/`).

> **Citation rule:** every paper named below must have a bibkey in
> `docs/dev/synthesizer_parity_citations.md`. Implementations cite the original paper,
> not synthesizer's source. See `CLAUDE.md` for verification workflow.

---

## Component Default Table

| Layer | Synthesizer default | Tengri config field | Tengri constructor | Bibkey | Notes / pitfall ID |
|-------|---------------------|---------------------|-----------------------|--------|--------------------|
| **SPS backend** | BC03 grid | `SFHConfig.sps_backend` | `"bc03"` | `Bruzual_2003` | |
| **SFH parametric** | delayed-tau (DPL) | `SFHConfig.mean_type` | `["dpl"]` | `Carnall_2017` | Synthesis uses age-dependent Bayesian prior; tengri uses fixed parametric |
| **Metallicity** | uniform metallicity | `met_logzsol` | `Fixed(-0.3)` | n/a | synthesizer uses SSP grid Z; tengri user-facing Z is log10(Z/Zsun) offset by LOG10_ZSUN = -1.848 |
| **Dust geometry** | two-component (CF00) | `DustConfig.model` | `"two_component"` | `Charlot_2000` | Young stars (age < t_birth) use tau_v_bc; old stars use tau_v_diff |
| **Dust attenuation BC** | Calzetti (starburst) | `DustConfig.law_bc` | `"calzetti"` | `Calzetti_2000` | R_V = 4.05; polynomial fit to starburst extinction |
| **Dust attenuation ISM** | Calzetti (same) | `DustConfig.law_diff` | `"calzetti"` or `None` | `Calzetti_2000` | If `None`, defaults to `law_bc` |
| **Dust emission** | DL14 (Draine-Li 2014) | `DustConfig.emission` | `"draine_li2014"` | `Draine_2014` | |
| **Dust emission geometry** | Optically thin (default) | (implicit in `DL14`) | n/a | n/a | Dust temperature T_dust computed from radiation field; U_min=1.0 (solar neighborhood) |
| **IGM absorption** | Inoue+2014 | `MultiwavelengthConfig.igm_model` | `"inoue"` | `Inoue_2014` | IGM transmission applied to observed-frame wavelengths; _avoid_ rest-frame passband (P-20) |
| **Nebular emission** | CLOUDY grids (Byler+17) | `NebularConfig.backend` | `"cloudy"` / `"cue"` | `Byler_2017` / `Li_2024a` | synthesizer uses `NebularGrid` (CLOUDY+Byler); tengri uses Cue NN emulator (equivalent via validation) |
| **Nebular metallicity** | SSP Z; grid-dependent | `neb_logZ_gas` via `Parameters` | `Fixed(-0.3)` | n/a | Synthesizer uses CLOUDY grid axis log(Z); tengri param is log10(Z_gas/Zsun) (grain-depleted; P-6) |
| **Nebular ionization** | log10(U) = -3.0 (HII region) | `neb_logU` | `Fixed(-3.0)` | n/a | synthesizer's NebularEmission default |
| **Nebular Lyα escape** | f_esc,Lyα = 0.0 (fixed) | `neb_fesc_lya` | `Fixed(0.0)` | n/a | No resonant scattering (simplified) |
| **Nebular ionizing escape** | f_esc = 0.0 (bound) | `neb_fesc` | `Fixed(0.0)` | n/a | P-9: must be wired to nebular continuum calculation |
| **AGN disc** | Shakura-Sunyaev (multicolor) | `AGNConfig.disc` | `"multicolor"` | `Shakura_1973` | Multi-color disc (Mitsuda 1984 / Kubota 2018 approximation); standard Eddington-limited |
| **AGN torus** | SKIRTOR (clumpy) | `AGNConfig.torus` | `"skirtor"` | `Stalevski_2016` | 3D radiative transfer; precomputed templates at various inclinations; science-grade |
| **AGN NLR** | Gaussian lines (analytic) | `AGNConfig.nlr` | `"analytic"` | `Groves_2004` | Empirical Gaussian profiles; alternative `"cue"` uses neural emulator |
| **AGN BLR** | Vanden Berk composite template | `AGNConfig.blr` | `True` | MISSING (P-1) | BLR included by default for Type 1 sightlines; hard mask at θ_torus |
| **AGN polar dust** | None (not default) | `AGNConfig.polar_dust` | `False` | `Pei_1992` / `Gordon_2003` | SMC extinction to disc/BLR if enabled; CIGALE ``skirtor2016`` includes this |
| **AGN Eddington** | sub-Eddington (typical AGN) | `agn_eddington_ratio` | `Uniform(0.01, 1.0)` | `Shakura_1973` | P-2: must convert both to same units before division (catch erg/s vs solar luminosity) |
| **Filters** | JWST NIRCam, HST ACS | (externally provided) | User's filter list | n/a | P-13 P-14: shape/unit consistency on addition |
| **Cosmology** | Planck 2018 | (implicit in `utils.cosmology`) | astropy `Planck18` | n/a | Standard assumption; verify via `test_cosmology_variants` (P-19) |
| **Photometry noise** | Gaussian (σ) | `NoiseModel` in fitter | Gaussian | `Hogg_2010` (MISSING) | P-21: all photometry unified to one unit system (mJy, erg/s/cm²) |
| **SED wavelength** | Rest-frame Angstrom | (computed internally) | Angstrom, vacuum | n/a | Consistent with Synthesizer; emission lines are vacuum wavelengths (P-20 corollary) |
| **Emission line list** | Byler+17 subset (synthesizer default ~60 lines) | `LineListConfig.preset` | (TBD; see registry below) | `Moustakas_2023` | [OIII]4959/5007 ratio fixed 1:3 if both present (P-12); avoid independent scaling |

---

## Parameter Prefix Mapping

Tengri free-parameter naming follows the NAMING_CONTRACT (§3.2 in `docs/dev/NAMING_CONTRACT.md`).
Each parameter below can be passed to `Parameters(sfh_dpl_alpha=...)` or set via
`priors={'dust_tau_bc': Uniform(0.0, 5.0)}`.

| Tengri parameter | Synthesizer name | Units / range | Default | Notes |
|------------------|------------------|---------------|---------|-------|
| `sfh_dpl_alpha` | `alpha` (DPL) | Gyr⁻¹ | registry | e-folding timescale inverse; DPL = (t/τ) × exp(-t/τ) |
| `sfh_dpl_beta` | `beta` (DPL) | dimensionless | registry | Early-time power-law index; DPL = t^β × exp(-t/τ) |
| `sfh_dpl_tau` | `tau` (DPL) | Gyr | registry | e-folding timescale |
| `sfh_field_psd_sigma` | (no direct equiv; GP variance) | log Msun/yr | registry | IFT PSD amplitude (stochastic component); only if `"field"` in `mean_type` |
| `sfh_field_psd_tau_myr` | (no direct equiv; GP correlation length) | Myr | registry | Burstiness timescale (correlation time); Tengri exposes in **Myr** at the user-facing API; internal compute is in years (CLAUDE.md gotcha). |
| `met_logzsol` | `metallicity_initial` | log10(Z/Zsun) | `Uniform(-2.0, 0.2)` | Free in default fitter (registry default is Uniform, not Fixed). Uniform-Z assumption unless `SFHConfig.evolving_metallicity=True`. |
| `dust_tau_bc` | `tau_v_young` / `dust_bc` | mag, lo ≥ 0 | `Uniform(0.0, 4.0)` | Birth-cloud optical depth (age < t_birth ~10 Myr); registry default = `Uniform(0, 4)`. |
| `dust_tau_diff` | `tau_v_old` / `dust_ism` | mag, lo ≥ 0 | `Uniform(0.0, 3.0)` | Diffuse ISM optical depth (all ages); registry default = `Uniform(0, 3)`. |
| `dust_slope` | (implicit in power-law) | dimensionless | `Fixed(-0.7)` | Power-law index if `law_bc/law_diff="power_law"`. |
| `agn_log_lbol` | `bolometric_luminosity` (erg/s) | **log10(L/L_sun)**, ~[10, 14] | declared in registry | AGN bolometric luminosity. Synthesizer expresses L_bol in erg/s; tengri uses L_sun. Add `LOG10(L_SUN_ERG) ≈ 33.58` when implementing the model in tengri (test: `test_log_lbol_unit_convention_is_l_sun`). The `unified_nlr_blr(...)` function default (`agn_log_lbol=44.0`) is **unphysical** under the L_sun convention (implies L_bol ≈ 4×10^77 erg/s) — see TODO at `unified.py:1243`. P-2. |
| `agn_log_mbh` | `black_hole_mass` | log10(M/M_sun), ~[6, 10] | registry | BH mass; used to compute Eddington luminosity. |
| `agn_log_ledd` | (derived: L_bol / L_Edd in synthesizer) | log10(eddington ratio) | registry | Stored as log Eddington ratio. P-2: convert L_bol and L_Edd to same units before division. |
| `agn_cos_inc` | `cos(inclination)` | [0, 1], 1 = face-on | registry | Tengri parameterizes by **cos(inclination)** (synthesizer uses `inclination` in degrees). Convert via `cos(deg2rad(i))`. |
| `agn_theta_torus` | `theta_torus` (deg) | [0, 90] degrees | kwarg only (not in `_param_defs.py`) | Half-opening angle of torus (kwarg of `unified_nlr_blr`). |
| `agn_oa_skirtor` | `theta_torus` (SKIRTOR axis) | [10, 80] degrees | registry | SKIRTOR-grid torus opening angle. |
| `agn_torus_frac` | `torus_fraction = theta_torus/90°` (derived in synth) | [0, 1] | registry | **Decoupled** from `theta_torus` in tengri (avoids gradient discontinuity, see `unified.py:47-53`). |
| `agn_blr_cf` | `covering_fraction_blr` | [0, 1] | registry | BLR covering fraction in registry & forward kernel. Unified with `unified_nlr_blr(...)` kwarg as of 2026-05-08. |
| `agn_nlr_cf` | `covering_fraction_nlr` | [0, 1] | registry | NLR equivalent of `agn_blr_cf`; kwarg names unified in `unified_nlr_blr(...)` as of 2026-05-08. |
| `agn_polar_ebv` | (CIGALE skirtor2016 only) | E(B-V), [0, 0.5] mag | registry | Polar dust reddening (SMC law) for Type 1 sightlines; absent from synthesizer default UnifiedAGN. |
| `neb_logU` | `log_ionization_parameter` | log10(U), [-5, 0] | `Fixed(-3.0)` | Ionization parameter; higher U → harder ionizing spectrum ionizes more ions. |
| `neb_logZ_gas` | `metallicity` (gas-phase in CLOUDY) | log10(Z_gas/Zsun) | `Fixed(-0.3)` | Gas-phase metallicity (excludes grain-depleted metals; P-6). Default is Fixed but a `_NEBULAR_PARAMS` comment notes "will be overridden to match met_logzsol if not set". |
| `neb_fesc` | `fesc` (ionizing photon) | fraction, [0, 1] | `Fixed(0.0)` | Ionizing photon escape fraction; P-9 verified wired into Cue's gradient at `cue.py:1172,1519`. |
| `neb_fesc_lya` | `fesc_lya` | fraction, [0, 1] | `Fixed(0.0)` | Lyα resonant scattering escape; 0 = Lyα stays in nebula. |
| `neb_dig_frac` | `dig_fraction` | fraction, [0, 1] | `Fixed(0.0)` | DIG (diffuse ionized gas) fraction; Tacchella et al. 2022 decomposition. |
| `noise_frac_cal` | (photometric calibration) | [0, 0.1] fraction | registry | Fractional systematic error. |
| `noise_dof` | (Student-t robust noise dof) | [2, ∞) | registry | If using outlier-robust likelihood (`Hogg_2010` MISSING from bib). |
| `redshift` | `redshift` | [0, ∞) | n/a (passed in fitter) | Source redshift; IGM transmission requires **observed-frame** wavelengths (P-20). |

> **Note:** rows above were audit-corrected against
> `src/tengri/parameters/_param_defs.py` and `src/tengri/components/agn/unified.py`
> on 2026-05-08. Earlier draft of this table contained hallucinated parameter
> names (`agn_inclination_deg`, `agn_torus_opening_angle_deg`,
> `agn_covering_fraction_blr`, `agn_black_hole_mass`,
> `agn_eddington_ratio`) that do not exist in the codebase. Re-verify by
> grepping `_param_defs.py` whenever this doc is touched.

---

## Conventions Checklist

### Wavelength
- **Tengri**: vacuum Angstrom (Å), rest-frame unless otherwise stated
- **Synthesizer**: vacuum Angstrom, rest-frame
- **Parity**: ✓ Identical
- **Caution**: IGM transmission (Inoue14) must use *observed-frame* wavelengths (P-20)

### Time
- **Tengri**: years internally; user-facing SFH timescales in Gyr (DPL), Myr (PSD)
- **Synthesizer**: Gyr (lookback time) or years (internal grids)
- **Parity**: ✓ Convertible; watch for unit mismatches in parametric SFH definitions
- **Caution**: PSD correlation timescale exposed in Myr (e.g., `sfh_field_psd_tau_myr`), not Gyr

### Metallicity
- **Tengri user-facing**: `met_logzsol` = log10(Z/Zsun) (dimensionless, relative)
- **Tengri SSP grid**: log10(Z) absolute (convert via LOG10_ZSUN = -1.848 offset)
- **Synthesizer**: SSP grid uses absolute log10(Z); user exposure varies by grid backend
- **Parity**: Watch offset; `neb_logZ_gas` must account for grain depletion (P-6)
- **Caution**: CLOUDY grids (Synthesizer's default nebular) use gas-phase Z; differ from SSP Z by ~0.3 dex (dust depletion)

### Bolometric Luminosity
- **Tengri**: L_bol = 10^`agn_log_lbol` L_sun; stored as log10(L/L_sun)
- **Synthesizer**: UnifiedAGN computes L_bol from accretion rate; user specifies via bolometric_luminosity
- **SED output**: All components output **erg/s/Hz** (L_nu; spectral luminosity density)
- **Line luminosity**: Returns L_sun for emission line total (integrated); no unit in name
- **Parity**: Must verify AGN line luminosity doesn't explode (P-1); ratio L_line / L_bol ~ 1e-2 to 1e-3 typical

### Reference Frame
- **IGM absorption**: Inoue14 `igm_transmission(wave_obs, z)` takes **observed-frame** λ, not rest-frame (P-20)
- **Dust attenuation**: Applied to rest-frame SED; wavelengths redshifted *after* attenuation
- **Emission lines**: Vacuum wavelengths only; no air wavelengths in tengri registry (P-12)
- **Caution**: Easy to pass rest-frame λ to IGM and get garbage (z_eff mismatch)

### Doublet Ratios
- **[OIII] 4959/5007** = 1:3 (forbidden transition; should be fixed)
- **[NII] 6548/6584** = 1:3 (forbidden transition; should be fixed)
- **Tengri**: Line registry should pre-combine or document ratio lock (P-12)
- **Caution**: If both lines returned separately as free parameters, physics breaks

### SFR Scaling
- **Tengri**: SFR output in M_sun/yr (from SFH normalization or input parameter)
- **Synthesizer**: SFR in M_sun/yr
- **Caution**: Check units when converting from parametric SFH integral

---

## Open Questions

### AGN BLR Template
- **Question**: Synthesizer uses Vanden Berk et al. 2001 composite SDSS quasar template for BLR.
  Bibkey status: **MISSING** (see `synthesizer_parity_citations.md` item #1).
- **Action**: Tengri currently uses empirical Gaussian profiles (Groves 2004) for NLR/BLR.
  Validate BLR luminosity and line ratios against Vanden Berk template in `tests/regression/synthesizer_parity/`.

### Dust Emission Grid Extrapolation
- **Question**: Synthesizer's DL14 dust emission model uses fixed wavelength grid (0.5–1000 µm typical).
  What happens when user provides filters outside this range (e.g., radio continuum)?
- **Action**: Synthesizer docs recommend restricting wavelengths to grid bounds. Tengri should either
  (a) auto-resample DL14 templates to user wavelengths or (b) raise error if out of bounds.
  See `docs/dev/synthesizer_parity_citations.md` for DL14 grid specifics (not yet fully documented in Draine 2014).

### Eddington Units (P-2)
- **Question**: Synthesizer PR #1068 fixed Eddington luminosity unit mismatch. Verify tengri's
  implementation converts both `agn_bolometric_luminosity` and `eddington_luminosity` to same units
  (erg/s) before computing `agn_eddington_ratio`.
- **Action**: Code review of `components/agn/_phys.py` or equivalent; write unit-aware docstring with
  explicit L_bol (erg/s) and M_BH (M_sun) → L_Edd formula.

### IGM Transmission at Extreme Redshift
- **Question**: Asada et al. 2025 (IGM model "asada") adds circumgalactic medium (CGM) absorption
  at z ≥ 6 (reionization era). Synthesizer integration status unclear.
- **Action**: If `MultiwavelengthConfig.igm_model="asada"` is supported, validate high-z transmission
  (z=7) returns T ≤ 1 and differs from Inoue14 at extreme z (test_A24_transmission_high_z mirror).

### Emission Line Velocity Dispersion
- **Question**: Tengri parameters `eline_sigma_kms` and `eline_delta_v_kms` (broadening + offset).
  Synthesizer's default behavior: are these added in quadrature, linearly, or absent?
- **Action**: Document whether narrow-line core + broad component coexist, and if AGN broad lines
  are included (see `NebularConfig.eline_broad`).

### SFH Parametric Form Equivalence
- **Question**: Synthesizer Prospector bridge uses delayed-tau + random sfh; tengri supports DPL ± GP field.
  Are these forward-equivalent for parity test, or do they target different priors?
- **Action**: The parity test should use pure DPL (no field), so compare like-to-like. Document that
  adding `"field"` in `mean_type` makes the model more flexible but is not in Synthesizer's defaults.

---

## Implementation Note: Building the Preset

`src/tengri/presets/synthesizer.py` should:

1. **Import and assemble config objects**:
   ```python
   def synthesizer_default() -> SEDModelConfig:
       """Synthesizer-parity default SED model.
       
       Returns
       -------
       SEDModelConfig
           Frozen config matching Synthesizer's UnifiedAGN + Calzetti + DL14 + CLOUDY defaults.
       
       Notes
       -----
       This preset is NOT the "full" Synthesizer model — analytic disc/NLR/BLR
       replace grids for JIT/grad compatibility. See docs/dev/synthesizer_parity.md.
       """
       return SEDModelConfig(
           sfh=SFHConfig(mean_type=("dpl",)),
           dust=DustConfig(
               model="two_component",
               law_bc="calzetti",
               law_diff="calzetti",
               emission="draine_li2014",
           ),
           nebular=NebularConfig(backend="cue"),  # or "cloudy" if user provides grid_path
           multiwavelength=MultiwavelengthConfig(
               radio=False, xray=False, shock=False,
               apply_igm=True, igm_model="inoue"
           ),
           agn_model="unified_nlr_blr",
           agn_config=AGNConfig(
               disc="multicolor",
               torus="skirtor",
               nlr="analytic",
               blr=True,
               polar_dust=False,
               fe2=False,
           ),
       )
   ```

2. **Wire default Parameter priors** in a companion function:
   ```python
   def synthesizer_default_priors() -> dict[str, Prior]:
       """Priors for synthesizer_default() config.
       
       Returns priors dict matching Synthesizer's fit assumptions.
       See parameter_prefix_mapping above for all entries.
       """
       return {
           "sfh_dpl_alpha": Uniform(0.1, 3.0),
           "sfh_dpl_beta": Uniform(0.0, 2.0),
           "sfh_dpl_tau": Uniform(0.5, 10.0),
           "met_logzsol": Fixed(-0.3),
           "dust_tau_bc": Uniform(0.0, 3.0),
           "dust_tau_diff": Uniform(0.0, 2.0),
           "agn_log_lbol": Uniform(8.0, 13.0),
           "agn_eddington_ratio": Uniform(0.01, 1.0),
           # ... (all 25+ rows from table above)
       }
   ```

3. **Use existing `core/component.py` Protocol** — no new abstractions.
   The SEDComponent-based pipeline (radio, xray, nebular) should be compatible; if not,
   ensure `forward/sed_model.py` routes config → component instantiation correctly.

4. **Be JIT-safe**: no Python-level branches on traced values. All config choices are
   static (frozen dataclass); they do not depend on parameter values.

5. **Cover 5 canonical galaxies** in parity test (see test structure in `synthesizer-tests-to-mirror.md`):
   - Low-redshift quiescent (z=0.05, M* ~ 1e11, no AGN)
   - Star-forming main sequence (z=0.5, M* ~ 1e10, weak AGN)
   - Starburst (z=2.0, high dust, intense SFR)
   - AGN-dominated (z=2.0, L_bol ~ 1e46 erg/s, high Eddington)
   - High-z quasar (z=6.0, IGM absorption critical)

6. **Regression test suite** in `tests/regression/synthesizer_parity/`:
   - Mirror top 10 tests from `synthesizer-tests-to-mirror.md`
   - Tolerance: np.testing.assert_allclose (rtol=1e-7, atol=0) unless noted
   - Cite original Synthesizer test file in docstring (e.g., `# Mirrors synthesizer/tests/test_unified_agn.py::test_weighted_combination_uses_transmission_fraction_attrs`)
   - Do NOT copy code; paraphrase assertion shape only.

---

## Summary Table: Config Fields Inventory

| Category | Field | Status | Notes |
|----------|-------|--------|-------|
| **SFH** | `SFHConfig.mean_type` | ✓ Exists | `["dpl"]` default |
| | `SFHConfig.sps_backend` | ✗ **NEW** | Must add to `SFHConfig` dataclass |
| | `SFHConfig.evolving_metallicity` | ✓ Exists | Not used in parity (fixed Z default) |
| **Dust** | `DustConfig.model` | ✓ Exists | `"two_component"` default |
| | `DustConfig.law_bc` | ✓ Exists | `"power_law"` default; change to `"calzetti"` |
| | `DustConfig.law_diff` | ✓ Exists | `None` default; must support `"calzetti"` |
| | `DustConfig.emission` | ✓ Exists | `None` default; must support `"draine_li2014"` |
| **Nebular** | `NebularConfig.backend` | ✓ Exists | `"off"` default; set to `"cue"` or `"cloudy"` |
| | `NebularConfig.grid_path` | ✓ Exists (conditional) | Required only if backend=`"cloudy"` |
| **Multi-wavelength** | `MultiwavelengthConfig.apply_igm` | ✓ Exists | `True` default ✓ |
| | `MultiwavelengthConfig.igm_model` | ✓ Exists | `"inoue"` default ✓ |
| | `MultiwavelengthConfig.radio` | ✓ Exists | `False` default ✓ |
| | `MultiwavelengthConfig.xray` | ✓ Exists | `False` default ✓ |
| **AGN** | `agn_model` | ✗ **Needs map** | Must document valid names (e.g., `"unified_nlr_blr"`) and register in `components/agn/unified.py` |
| | `AGNConfig.disc` | ✓ Exists | `"multicolor"` default ✓ |
| | `AGNConfig.torus` | ✓ Exists | `"skirtor"` default ✓ |
| | `AGNConfig.nlr` | ✓ Exists | `"analytic"` default; `"cue"` available |
| | `AGNConfig.blr` | ✓ Exists | `True` default ✓ |
| | `AGNConfig.polar_dust` | ✓ Exists | `False` default ✓ |
| **Parameters** | All 25+ free param defaults | ✓ Exists in `param_defs.py` | Review ranges and fix AGN-related ones (P-2, P-4) |

### NEW Config Keys Required

1. **`SFHConfig.sps_backend`** — Add field `sps_backend: str = "bc03"` to allow Synthesizer-style SPS selection
   - Valid: `"bc03"`, `"fsps"`, `"bpass"`, `"c3k"` (future)
   - Routed to `forward/sps/dsps_wrapper.py` or equivalent SPS loader

### Existing Config Keys Needing Content Updates

1. **`DustConfig.law_bc`** — Change default from `"power_law"` to `"calzetti"` in parity preset
2. **`DustConfig.emission`** — Default is `None`; set to `"draine_li2014"` in parity preset
3. **`agn_model`** — Document valid model names and register `"unified_nlr_blr"` as the parity match

### Parameter Ranges & Defaults to Verify

- **`agn_eddington_ratio`** — Verify computation uses erg/s for both L_bol and L_Edd (P-2)
- **`neb_fesc`** — Verify it actually reduces nebular continuum (not orphaned parameter; P-9)
- **`agn_covering_fraction_blr/nlr`** — Verify escape_frac = 1 - blr - nlr (energy conservation; P-3)
- **Dust attenuation curves** — Verify extrapolation is bounded (not negative/infinite) for λ > 10 µm (P-5)

---

## References

All citations resolved via `docs/dev/synthesizer_parity_citations.md`. See that file for verification workflow.

**Key papers cited in this document:**
- Bruzual & Charlot (2003) — BC03 SPS grid
- Calzetti et al. (2000) — Starburst attenuation law
- Charlot & Fall (2000) — Two-component dust geometry
- Draine & Li (2014) — Dust emission templates
- Inoue et al. (2014) — IGM absorption (Lyman forest + Lyman continuum)
- Asada et al. (2025) — High-z IGM + CGM absorption
- Lovell et al. (2025) — Synthesizer, Open J. Astrophys. 8 (doi:10.33232/001c.145766)
- Roper et al. (2026) — Synthesizer, JOSS 11, 9436 (doi:10.21105/joss.09436)
  - Both Synthesizer papers must be cited together (upstream citation policy).

**Pitfall references:**
- See `docs/dev/synthesizer-pitfall-catalog.md` for detailed diagnosis of P-1 through P-25
- Top 5 highest-risk areas: AGN luminosity scales, dust extrapolation, nebular Z convention, filter edge cases, SFH/SPS integration
