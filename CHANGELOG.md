## [Unreleased]

### Changed

- **ADAF normalization (#2768).** `adaf_spectrum` normalizes with closed-form bremsstrahlung plus a 30-point Gauss-Legendre rule on each of the four segments between the spectrum's own breaks (`0.02 nu_min`, `nu_min`, `nu_p`, `3 k T_e / h`, `100 k T_e / h`) instead of an 8193-node trapezoid. ADAF SEDs move by 2.3e-6 to 2.7e-6 relative, the trapezoid's error against an independent dense reference (3.2e-6 measured there); the new normalization agrees with that reference to 3.9e-12 (declared prior box corners and 300 draws).

- **The conserving line debit takes the ADAF disc's power in closed form (#2768).** The ADAF registers `adaf_disc_power` (`L_acc`, the quantity its spectrum is normalized to), as `multicolor` and `kubota_done` do (#2743), instead of being integrated on the 13001-node ledger grid. The debited fraction moves by up to 1.1e-3 relative (the ledger integral fell short of `L_acc` by 0 to 1.1e-3 across the prior box), and by more with `agn_ebv_disc > 0`, where the ledger power was the reddened disc and the debit is now of the intrinsic power.

- **A traced FeII width below 500 km/s returns NaN (#2768).** The FeII broadening is a band-limited synthesis (one 2^15-point inverse FFT on constants computed once) valid for FWHM >= 500 km/s; it matches the full-lattice convolution to 2.3e-9 relative and leaves the R_Fe normalization unchanged (1e-15). A concrete width below 500 km/s (including the unbroadened template) takes the exact full-lattice path; a *traced* width below it cannot raise inside a trace and the FeII spectrum and power are NaN, rather than silently low-passed. The width is a block keyword (`agn_blr_fwhm_kms`, default 5000), not a fit parameter, so no model build reaches this; it affects direct calls of `compute_blr_sed`, `unified_nlr_blr` and the FeII/BLR blocks under `jax.jit`/`vmap` with a width below 500 km/s.

- **ADAF and FeII compile cost (#2768).** The ADAF's scalar solve is traced once per composition instead of once per disc evaluation, and the FeII template is no longer rebuilt and FFT-ed on a 138 061-node lattice inside the gradient. Measured: the `disc/adaf` wildcard test case 293.7 s to 124.2 s and 18.66 GB to 6.24 GB peak footprint (`/usr/bin/time -l`, cold cache); cold first `jax.grad` of an ADAF + analytic-BLR model 66.5 s to 28.8 s (29.1 s before #2728); gradient FLOPs of `_fe2_pseudo_continuum` 119.9M to 7.0M and of the whole ADAF + BLR model 88.9M to 10.6M.

- The default star-forming radio flux changes: Bell (2003)'s `radio_q_ir = 2.64` calibrates the TOTAL 1.4 GHz luminosity (Eq. 1; about 10 % of it thermal, Sect. 4), and the default `bell2003` block used to put that whole total in the synchrotron term and add the thermal emission on top (the Murphy et al. (2011) free-free term, or the nebular continuum under the nebular auto-rule), counting it twice. With `freefree` unset the synchrotron term is now `(1 - f_th)` of `L_IR / (3.75e12 Hz x 10^q)`, `f_th` the Murphy free-free share at 1.4 GHz (0.1335 at q = 2.64, T_e = 1e4 K), and the thermal term comes from the radio block or from the nebular continuum. Without a nebular backend the 1.4 GHz total equals the calibration; with one it equals it to within that backend's free-free share minus `f_th` (Cue: 1.7 % below the calibration for a constant star formation history, 7.3 % below for a delayed-tau = 1 Gyr galaxy at 5 Gyr; the difference follows the ionizing photon rate per unit L_IR). For q = 2.64, alpha = 0.8, T_e = 1e4 K and no nebular the on/off ratio at 1.4, 5, 30 and 100 GHz falls from 1.134, 1.326, 2.141 and 3.650 to 1.000, 1.192, 2.007 and 3.516; below 1.4 GHz the synchrotron is lower by the thermal share. `freefree: False` is the non-thermal (CIGALE) spelling: the synchrotron carries the whole `q` total, no radio thermal term, as before. `bell2003_split` (AGNFITTER-RX 90/10 construction) is unchanged. The `L_ir` that enters q is documented as the total absorbed dust power `L_absorbed x eta`; the 8-1000 micron band holds 0.944 (DL14) and 0.958 (Casey 2012) of it for the default delayed-tau galaxy. `SEDModel.build` now refuses a `radio_q_ir` support that reaches `q_* = 3.5145 - 0.45 log10(T_e/1e4)` (the synchrotron share would be negative); the declared free prior of `radio_q_ir` narrows from Uniform(1.8, 3.5) to Uniform(1.8, 3.37). `delvecchio2021` and `mccheyne2022` are unchanged. (#2590)

- One exact young/old split serves every attenuator. The stellar component publishes, per SSP age node, the share of its formed mass younger than each boundary age (`age_boundary_younger_fraction`), computed through the same SFH kernel as the node weights, instead of each screen evaluating a step or logistic at the node ages. `two_component` defaults to a hard step at `t_birth_yr=1e7`; `transition_width_dex > 0` opts into the smooth law, and `age_binned` windows use the same machinery. A node's transmission is the mixture of its populations' transmissions (not of their optical depths). Nebular and line screens are weighted by ionizing luminosity, the energy-balance lookup table carries young and old populations and mixes them at runtime, `age_binned` gains the `lyc_` key family, and the refusal of age windows narrower than five node spacings is gone. Golden: `two_component` `L_absorbed` 5.932047709369588e59 -> 5.93036676326139e59 erg/s (-0.0283%), matching an independent dense-parcel step reference to 5.6e-9.
- **`dust_emission={'type': 'schreiber2016'}` now evaluates the tabulated Schreiber et al. (2018) dust library, and the analytic modified-blackbody plus Drude stand-in is removed (#2597).** The old model had band powers 0.000 to 8.4 times those of the library CIGALE ships under that name (3 to 24 micron power 0.00 to 0.4 of the library's, 70 to 1000 micron power 1.4 to 8.0 times it) and peaked at 131, 75 and 52.5 micron where the library peaks at 107.6, 7.6 and 7.6 micron for (T, f_PAH) of (20, 0.05), (35, 0.2) and (50, 0.5). The tabulated templates (dust continuum plus PAH per temperature node) are mixed per kilogram, so `dust_f_pah` is the PAH mass fraction of Schreiber et al. (2018), Sect. 3.2, and the band powers match CIGALE to 2e-3. A missing `data/schreiber2016_templates.h5` raises with the regeneration recipe; there is no analytic fallback. The `schreiber2016_ir` component, `Schreiber2016IRConfig` and the analytic closure and component classes are removed (`tengri.components.dust.Schreiber2016IRSEDComponent` is the tabulated component under the registry name). The `dust_T` default of `schreiber2016` is 35 K, the registry default, in the closure, the component and a built model alike (it was 30, 25 and 35 K); there is no CMB heating correction on this template. The paper reference is A&A 609, A30 (arXiv:1710.10276) in the data file attributes and the bibliography; the A&A 589, A35 entry (`schreiber2016`) is removed. (#2597)

- The AGN torus has one inclination and one opening angle. `agn_cos_inc` is the inclination of every torus; the Fritz et al. (2006) library's viewing elevation is derived from it, psi = 90 deg - i (`cos i = sin psi`), so the torus SED, the disc screen and the polar-dust mask see one sightline, and `agn_fritz_psy` is retired: passing it (`agn_fritz_psy`, `fritz_psy`, `psy` in the torus dict) raises a `ValueError` naming `agn_cos_inc` and the mapping. The sightline is Type 1 when `i` is inside the dust-free polar cone, `i < agn_fritz_oa` for Fritz (the cone half-angle, full opening 180 - 2 x half) and `i < 90 - agn_oa_skirtor` for SKIRTOR, for the disc screen and the polar mask alike: the screen's Type-1 limit for half-angles 20 / 40 / 60 deg was i = 70 / 50 / 30 deg. The polar cone follows the selected torus's own opening angle by default (`agn_polar_oa` now defaults to 0, "follow the torus"; a positive value overrides it), the polar mask, the screen and the generic tori (`agn_theta_torus`) share one sigmoid width of 0.025 in cos i (the mask's was 0.05 with a different midpoint, the generic tori's 2 deg in i), a prior on `agn_polar_oa` that reaches 0, and a negative value, are refused (a fixed 0 means the cone follows the torus; an explicit angle must be > 0) and `spec.summary()` says which torus angle the cone follows, and `agn_fritz_oa` outside the grid [20, 60] is refused at build time instead of clamped to the edge template. A face-on sightline (`agn_cos_inc` = 1) now keys the Fritz library at psi = 89.92 deg rather than its 89.99 deg edge (the arccos slope is capped for float32 gradients) (#2605, #2602)

### Added

- The polar-dust extinction curve is selectable: `agn={'polar_law': 'calzetti'}` (or `agn={'atten': {'type': 'polar_dust', 'polar_law': ...}}`) takes `smc` (Pei 1992, the default), `calzetti`, `gaskell` or `bongiorno` (CIGALE's `extinction_law = 0`: the power law 1.39 lambda_um^-1.2 above 100 nm and the shape of the Weingartner & Draine (2001) SMC-bar dust-mixture extinction below it, matched at 100 nm, the splice CIGALE makes with its own SMC-mixture table; the two shapes, relative to 100 nm, agree within 5.5 per cent above 35 nm, within 15.5 per cent between 10 and 35 nm, and 19 per cent at 1 nm); an unknown name raises naming the four, and `spec.summary()` shows the law (#2602)
- **`register_disc_state` / `DISC_STATE_BLOCKS` (#2768).** A disc block whose scalar solve does not depend on the wavelength registers it (`fn(agn_log_lbol, *, dtype, **params) -> state`) and accepts `disc_state=`; the composable runner solves once per composition and hands the state to every evaluation of the disc (caller grid, anchors, 5100 A, budget grids). The ADAF is the first user (`adaf_scalar_state`, `adaf_spectrum_from_state`, `AdafState`).

- The fold that `WavePrecomp(igm_fold="auto")` resolved to ("exact", "node", or `None` when no fold was built) is reported beside the declared mode in `precompute_engagement_report(model).observed_facts["igm_fold"]` and in `summary_text()`. (#2445).

- `draine_li2007` and `draine_li2014` publish the derived key `dust_umean`, the mean starlight intensity `U_min [(1 - gamma) + gamma R]` of the model (CIGALE's `dust.umean`), with `R` the power-law to single-U luminosity ratio at `U_max = 1e6` (alpha = 2) for DL07 and `1e7` (free alpha) for DL14 (#2599).

### Fixed

- **The ADAF spectrum shape agrees between float32 and float64 over the declared box (#2783).** The float32 synchrotron self-absorption solve `_adaf_x_m` formed its bracket `2.49e-10 4 pi n_e R / (B theta^3 K_2)` as one linear ratio, ~4e39 at the electron-temperature floor where the clipped accretion rate lands (`alpha = 0.5`), which overflows float32 (max 3.4e38). The Newton solve then stuck at its upper bound and returned x_M = 1e6 against the float64 9.07e4, so the shape differed by up to 0.997 relative at 18 of the 72 box corners (`alpha = 0.5`, and `log M_BH = 10` with `beta = 0.1`). The bracket, the peak luminosity and the normalization (quadrature and integral) are now formed as sums of logs. Float64 changes by at most 3.6e-14 relative on the 72 box points, and the float32 shape agrees with float64 to 2.8e-5 relative at worst (tolerance 1e-4 pointwise; see `tests/physics/agn/test_adaf_float32_shape.py`). The Eq. 43 square root and the normalization are also made differentiable at their zero floors: the gradient is finite at the box corners in both precisions, where it was NaN in float64 at two corners and in float32 at one.
- `d/d(agn_cos_inc)` at the face-on endpoint (`agn_cos_inc = 1`) is finite for the generic-torus unified models: the Type-1/2 line and disc weight is the cos i sigmoid (`type1_weight`) in place of a sigmoid of `arccos(cos i)`, whose infinite slope at the pole made the gradient `+inf` (`NaN` where the SED vanished), as `cat3d_wind` showed.

- **Every wavelength is vacuum, converted once at ingestion**: the Lick/Lick-IDS windows
  (`STANDARD_INDICES`: `HdA`, `HdF`, `HgA`, `HgF`, `Hbeta`, `Mgb`, `Fe4383`, `Fe5270`,
  `Fe5335`, `Ca4227`) were published in air but applied to vacuum spectra, putting every
  window 1-2 Å off; they are now stored as the published air values and converted at import.
  The Feltre+2016 table (air [OIII], [OI], [NII], [SII], Hβ beside vacuum Hα) and the
  MAPPINGS V / Flury+2024 air line labels are converted at load, so line placement in a
  filter and line matching agree with the vacuum catalog. One public converter pair,
  `tengri.utils.air_vacuum.vac_to_air` / `air_to_vac` (IAU, Morton 2000 / Ciddor 1996; closed
  form forward, exact fixed-point inverse), replaces the mismatched Morton (1991) and Edlén
  (1953) formulas; `vacuum_to_air` / `air_to_vacuum` remain as aliases. The CB19 builder table
  is all-vacuum. Cue and CB_19 convert at load too (see the next entry).
- **Cue and CB_19 ingest vacuum wavelengths; the post-hoc Balmer vote is gone.** The Cue weights
  file stores Cloudy's air labels (`6562.80`, `5006.84`, ...); `load_cue_weights` now converts
  `nn_line_wavelength` and `sorted_line_wavelength` once with `air_to_vac`, so
  `CueBackend.published_line_wavelengths`, the SED's line centres, `state.derived["line_waves"]`,
  `_published_line_wavelengths_static` and the #2701 `nebular_line_phot_waves_rest` are one vacuum
  array (Hα 6564.61, [N II] 6584 at 6585.27; they had been 1.2-1.8 Å blueward in the raw frame).
  The hosted `cb19_templates.h5` mixes conventions (measured: Hβ, Hα and [O III] 5008 vacuum;
  Hγ 4340.47, [O I] 6300.30, [N II] 6548.05 and 6583.45 air); its four air labels are converted
  at load and a regenerated all-vacuum file passes through unchanged. `nebular_line_waves_to_vacuum`
  (a Balmer-series vote at publication) is removed, since every backend now ingests vacuum and a
  vote on top would be a second conversion. CB_19's [N II] 6584 now sits 0.01 Å from the vacuum
  catalog, not 1.8 Å. `data/cue_weights.npz` itself still stores air
  (`scripts/convert_cue_weights.py` is the raw-data writer).
- `_pdr_luminosity_weight` (the power-law to single-U luminosity ratio of the Draine & Li 2007 Eq. 33 mass distribution, which weights the DL07/DL14 PDR component and `dust_umean`) is one closed form at every α: `R = g((2 − α)L)/g((1 − α)L)` with `g(u) = expm1(u)/u` and `L = ln(U_max/U_min)`, `g` by its series below |u| < 1e-3. It had held the α = 1 and α = 2 limit values across |α − pole| < 1e-3, which left `R` 3–4e-3 off and flat in α there and made it jump by 5–8e-3 at the window edges — a likelihood step with a zero gradient on a free `alpha_dl14` (#2727).
- Lick equivalent widths and magnitude indices follow Trager et al. (1998, ApJS 116, 1, Eqs. 1-3). The index operator converts its per-frequency input (`L_ν` or `F_ν`) to `F_λ ∝ F_ν/λ²` and builds the pseudo-continuum as the straight line through the two sideband means placed at the sideband mid-wavelengths, integrating `1 − F_λ/F_C` over the feature window (the window-LUT path, `predict_spectral_indices(approx=True)`, evaluates the same definition at the window grid points, as the line path of #2677 does, and agrees with the exact path to round-off, also under a strong dust screen). A constant mean-of-sidebands continuum is the continuum at the wrong wavelength for asymmetric sidebands (Fe4383: 12 Å from the feature center): on solar SSP spectra it differs from the Lick definition by up to 1.0 Å (HγA, 10 Gyr), 0.4 Å (Fe4383) and 0.5 Å (HγF). Break indices (`Dn4000`, `D4000`, `F_ν` ratios) and `uv_slope_beta` are unchanged. `measure.spectral_index` documents its flux argument as a per-frequency flux density (pass `F_λ` as `flux_lambda * wave_rest**2`); `SpectralIndexDef(pseudo_continuum="mean")` keeps the constant continuum of `bagpipes.input.spectral_indices.single_index`, measured on the array as given, for comparison with BAGPIPES. An EW or magnitude index must declare exactly two continuum windows under the default definition. (#2690).
- The Synthesizer-grid NLR and BLR line regions (`nlr/blr={'type': 'synthesizer_spectra'}`
  and `{'type': 'synthesizer'}`) read the grid's black-hole-mass and Eddington-ratio axes
  from `agn_log_mbh` and `agn_log_ledd`: all four blocks called the grid backend without
  them, so the axes sat at the backend's hard-coded node (8.0, -0.3) and the two
  parameters were inert (the [O III] 5007 light of the `synthesizer_spectra` NLR moves by
  about 18% between `agn_log_ledd` = -0.3 and -1.0). The blocks now forward the registry
  defaults (`agn_log_mbh` 7.0, `agn_log_ledd` -1.0) when the model leaves them unset, so
  a model that never set them sees the (7.0, -1.0) grid node instead of (8.0, -0.3)
  (#2634).
- `bell2003_split` is now covered by the build-time check that refuses a FIRRC radio block without a dust component, which would otherwise return an all-zero radio SED silently. (#2590)
- **Spectra read the Lyman edge as a step (#2447):** when a photoionized nebular backend
  masks the Lyman continuum, the spectrum projection now treats the SSP cell straddling
  911.76 Å as the same step photometry integrates, in both point-sampled and
  pixel-integrated resampling. A pixel next to the edge previously took a linear ramp
  across the cell and could be off by up to ~20%.
- **Cue's ionizing photon rate includes the partial bin at the Lyman edge:** Cue line
  and continuum fluxes rise by ~1.7% for a constant star formation history, matching the
  stellar `nion` already reported.
- **Behaviour changes.** (1) With `agn_norm="cigale_joint"` and a SKIRTOR torus, a tied corona-bearing disc (`kubota_done`) now emits below 10 A: its corona rides on top of the tied disc, where the library's 10 A edge used to cut everything. At log M_BH = 8, `agn_log_lbol` = 11, `agn_ir_frac` = 0.1 the 2-10 keV luminosity is 1.45e44 erg/s at i = 0 and 1.50e44 erg/s at i = 30 deg, from zero before (2.77e43 erg/s for the untied disc). The tied disc's own soft X-ray tail below 10 A is still cut at the library edge. (2) A Fritz torus now ties its disc to `agn_power` (it stayed at `agn_log_lbol`, 1.3e10 times the torus), and `agn_log_lbol` beside `agn_ir_frac` is refused for Fritz, as it is for SKIRTOR. (3) The conserving line debit takes a slightly larger fraction of the disc (see below). (4) The `L_2500_intrinsic` and `L_4400_intrinsic` anchors of a tied model follow `agn_power`; the tied optical disc of `kubota_done` rises by `1 + x` (x the in-grid corona-to-disc power ratio).

- With a SKIRTOR or Fritz torus and `agn_norm="cigale_joint"` the tie normalizes the thin disc and warm zone only, as CIGALE's `skirtor2016` does for its pure-disc library. A hot corona (`kubota_done`) is isotropic and not reprocessed by the library's disc: it carries `f/(1-f)` times the tied disc's angle-integrated power (`f` the closed-form corona share of the accretion power, 0.0063 / 2.06 / 2.30 for `f/(1-f)` at log M_BH = 6 / 8 / 10; the tied disc's 4 pi power is its face-on library power times 7/18 for SKIRTOR and 1 for the isotropic Fritz disc). Per unit intrinsic disc power the corona is therefore the same at every viewing angle. At fixed `agn_power` both the corona and the anchors still rise with inclination (x1.03 / 1.09 / 2.1 / 2.6 / 3.2 at i = 30 / 40 / 60 / 70 / 80 deg) because the library normalizes each viewing-angle record to its own observed dust luminosity; CIGALE's `intrin_Lnu_2500A_30deg` does the same (1.0291 / 1.0909 / 2.1088 / 2.5884 / 3.1807). On a Type-2 sightline the corona is dimmed by the torus screen's dust extinction only (the screen is far more transparent to X-rays than the library disc is; #2760). Before, the corona was normalized together with the disc and took 0.15 % / 42 % / 47 % of its budget at log M_BH = 6 / 8 / 10. Corona-less discs are unchanged bit for bit. The disc block exposes its split through the new `register_disc_split` protocol (#2750).

- `L_2500_intrinsic` and `L_4400_intrinsic`, the anchors of the X-ray alpha_ox and the radio loudness, follow the disc as normalized in the model: with `agn_ir_frac` they scale with `agn_power`, as `intrin_Lnu_2500A_30deg` does in CIGALE, so X-ray and radio follow fracAGN (they were constant at the default `agn_log_lbol`). A Fritz torus ties the disc to the library the way CIGALE's `fritz2006` does: disc over `agn_power` equals CIGALE's `disk_luminosity` to 6e-6 at psi = 50.1, 70.1 and 89.99 deg (4e-7 at 0.001 deg), and the anchor equals CIGALE's at every psi to 5e-4 (2e-5 after CIGALE's own linear interpolation of its disc at 2500 Å is divided out) once the library's per-record `norm`, now stored with `data/fritz2006_torus_grid.h5` (a new `fritz2006/norm` dataset; the existing arrays are unchanged), is applied. An older grid file without that dataset is refused with the command that adds it (`scripts/build_fritz2006_grid.py --add-norm`) (#2603).

- The conserving line debit takes the disc's bolometric power in closed form (`L_acc` for the `kubota_done` and `multicolor` discs, registered with `register_disc_power`) instead of integrating the disc on an 8501-node grid: the traced gradient no longer evaluates the disc for it (35.4e6 to 2.3e6 FLOPs on a 12-node grid), and the debited fraction rises by the grid integral's shortfall: 7.3e-4 of `L_acc` for `kubota_done` at log M_BH = 8 (dense grid; 7.26e-4 on its budget grid) and 3.3e-5 for `multicolor` (a quadrature artifact of the budget grid; the integral is 1 to 1.2e-6) (#2743).

- The polar Calzetti curve is held at zero beyond 3.1 um, the zero of its long-wavelength polynomial, instead of going negative (k = -0.80 at 30 um), so the absorbed polar power is never negative for any law; and with `torus='none'` the gradient of the AGN component sum with respect to `agn_polar_ebv` at E(B-V) = 0 is 0, as the sum is, instead of the graybody's own derivative (#2744)
- The polar-dust cone share of a Fritz torus follows its opening angle, `1 - cos(half)` of the disc luminosity (CIGALE `fritz2006`); it was the SKIRTOR form at a separate angle, identical for `agn_fritz_oa` = 20 and 60 deg. On the CIGALE-tied path (SKIRTOR and Fritz) the polar graybody now sits inside the unit-integral normalization, so the disc is divided by `1 + l_ext` with the torus (disc / `agn_power` at E(B-V) = 0.1 and 0.5, i = 0, against CIGALE `skirtor2016`: 1.59x and 2.01x before; 1.015x and 1.055x after with the default `smc` law, which is a different curve from CIGALE's; 1.0019x and 1.0006x with `polar_law='bongiorno'`, whose polar share is 0.9993 / 0.9952 / 0.9936 of CIGALE's at oa 10 / 40 / 70, E(B-V) 0.03); `polar_dust_extinction` raises on an unknown law instead of falling back to the SMC curve, and the `bongiorno` law (1.39 lambda_um^-1.2) is available beside `smc`, `calzetti` and `gaskell` (#2602)
- The polar absorbed power is formed as `L_nu (1 - exp(-tau))` rather than clipped with `maximum(., 0)`: the clip tied its two arguments at E(B-V) = 0, the lower edge of the prior, and halved the polar and torus gradients there (#2744)
- The Fritz docstrings list the grid half-angles 20 / 40 / 60 deg (they said 60 / 100 / 140, the full angles of another convention) (#2605)

- **Kubota-Done warm and hot Comptonization no longer rounds its template coordinates to
  float32 (#2739):** the nthcomp interpolation located `gamma`, `kTe` and `kTbb` in float32
  (relative 6e-8), so a 1e-16 difference between `jax.jit` and eager evaluation flipped a
  rounding on a steep template cell and moved the disc SED by 1e-6 (measured 1.5e-6 to
  3.2e-6 at log M_BH 7 to 8). Coordinates, template axes and weights now keep the caller's
  dtype (float64 under x64, float32 in pure-float32 mode); the float32 template values are
  promoted where they are gathered. JIT against eager now agrees to 2e-13, and AD against a
  central difference of `agn_gamma_warm` and `agn_kt_warm` agrees to 1e-5 (was 3e-5).

- **Free-redshift photometry table (`WavePrecomp`) interpolates in z with a monotone local cubic Hermite,
  exact at nodes, with nodes added where each band's Lyman-limit crossing falls (#2749):** the old triweight kernel
  (C² but blurred) had non-monotone interpolation error that did not decrease monotonically with grid refinement
  (11.6 / 1.5 / 0.7 / 1.2 % error at 100 / 200 / 400 / 800 nodes, worst case GALEX FUV). The new shape-preserving
  cubic Hermite (PCHIP) is exact at nodes, keeps the grid monotone, and resolves sharp features like the
  Lyman-limit edge via edge-aware node clusters (refinement factor 8, graded spacing). Measured on GALEX/SDSS/DES/
  NIRCam free-z fits: FUV/NUV/u/des_i worst error 0.59/0.06/0.04/0.08 % at n_z=250; NIRCam F090W/F115W at z 4–12
  reduce from 7.39/5.64 % to 0.58/0.77 %. Gradient vs central difference error drops to ≤0.8 % (was ≤23 %).
  Gradient FLOP count down 22 % (2.94M vs 3.76M).

- The radio wing of the master wavelength grid is sampled at 100 points per decade from at
  most 1e8 A (was 20 per decade from the end of the longest template), so a 10 %-wide radio
  band (1.4 GHz, 3 GHz, 150 MHz) holds about four nodes at every redshift instead of one: with
  a template that reaches into the radio (a CIGALE Dale grid ends at 2.2e9 A) the free-z LUT
  radio band read -1.03 % against the exact path at z = 0.5; it now reads -7.8e-4.

- The composable AGN runner's energy budgets no longer depend on the caller's wavelength
  grid. The CIGALE-joint disc tie (`agn_power x R`) took the torus power and the disc's
  reweighted bolometric as trapezoids over the caller's array, so the SED of a
  `kubota_done` + SKIRTOR model moved by 4e-3 between a grid starting at 10 A and one
  starting at 0.01 A (the corona carries 0.5 % of the disc energy below 10 A). The torus
  power is now the torus's own integral on a fixed budget grid, the disc is evaluated on
  the SKIRTOR library's own axis for `R` and, on a fine fixed grid, for the reweighted
  bolometric, the conserving line debit measures disc and line energy on fixed
  grids (by 3e-2 on the disc between two grids before), and the polar-dust ledger (the
  graybody's normalization, the absorbed power of the bolometric disc, the torus and
  graybody budgets) is measured on fixed grids. `l5100_disc` and the 2500/4400 A anchors
  are evaluated at their wavelengths instead of interpolated from the caller's nodes.
  On a grid covering the library the values agree with the former ones to the
  trapezoid error of that grid (about 1e-3).

- The fixed budget grids of the composable AGN runner cost a fraction of what they did.
  The conserving line debit takes the power of the analytic NLR, BLR and Fe II blocks in
  closed form (their lines are unit-integral Gaussians and the Fe II template conserves
  flux) instead of evaluating them on a 30 001-node grid, and skips the disc energy
  altogether when no line block is selected. The SKIRTOR torus power is the exact integral
  of its log-log interpolant on the library's own nodes, the polar-dust budget is the
  absorbed power the graybody is normalized to (no second graybody evaluation on the budget
  grid), and the face-on and bolometric polar references are chosen with `lax.cond` instead
  of both being evaluated. The Kubota & Done disc is integrated on 8501 nodes (1000 per
  decade over 1e-2 - 1e6 A, 100 per decade outside, 8.9e-7 against a dense reference, the
  error of the former 13 001), and its model-grid EUV carries 75 points per decade (6604
  instead of 7219 nodes; 75 is the smallest tested density with a bolometric error under
  1.3e-4 over log M 7 - 10, spin 0 - 0.9: 1.2e-4, against 4.0e-4 at 40 per decade). A concrete `agn_fe2_strength = 0` skips the template broadening.
  `jit(grad)` with respect to `agn_log_lbol`, `kubota_done` + SKIRTOR, FLOP from the
  compiled HLO: conserving with analytic NLR/BLR 1.5e8 to 3.7e7 on 12 nodes (1.6e8 to 4.3e7
  on 1500), CIGALE-joint with polar dust 1.6e8 to 1.4e8, the delayed + Calzetti + conserving
  model 1.55e8 to 9.7e7. The AGN components move by at most 2e-6; the node sum and the
  photometry of a model carrying a `kubota_done` disc move through the master grid, and the gradient of the node-summed torus component with respect to the
  polar `E(B-V)` at `E(B-V) = 0` now has the sign of the live gradient.

- AGN template and analytic components no longer normalize over the caller's wavelength
  grid. `torus_lnu_from_grid` (skirtor/nenkova AGNfitter 1p/2p/3p, `cat3d_wind`,
  `cat3d_wind_lowfwd`), the CLUMPY closure, `silva04`, `fritz`, `kd18_agnfitter` (+
  `warmindex`), `slone_netzer`, `relagn_disc_from_grid`, `qsogen`, the toy tori, `adaf`
  and the `relagn_agn` torus budget divided by a trapezoid over
  whatever grid they were handed, so L_ν at a fixed wavelength moved by 2e-4 to 2e-2
  between a coarse and a fine grid, by 15% (KD18) to a factor 44 (ADAF) on a grid that
  starts at 10 Å, and by order unity on a 912 Å-3 µm grid. Templates integrate the
  power-law interpolant on their own native grid before resampling; analytic shapes
  integrate on a fixed internal grid spanning their support. On a grid that covers the
  support the values reproduce the old fine-grid ones to ≤ 2e-4 (most to ~2e-6); ADAF
  moves by 4.7% because the old reference grid stopped at 1e8 Å and dropped that share of
  the low-frequency synchrotron.

- `radio_log_nu_cut` sets the synchrotron-aging cutoff of the power-law AGN
  radio jet (#2689): `L_nu = L_5GHz (nu / 5 GHz)^-alpha exp(-nu / 10^cut)`,
  with the default 13.0 (10 THz) as before. The key is read by both AGN radio
  models; on the power-law jet the `all_params: FREE` wildcard frees only
  `radio_loudness` and `radio_alpha_agn`, and the cutoff is freed when named
  (`radio={'agn': {'radio_log_nu_cut': FREE}}`, prior `Uniform(12, 14)`). The cutoff exponent is formed as `nu * 10^-cut`, so the
  value and the gradient are finite in float32 for any `cut` (10^cut overflows
  there from about 38.5). The DPL-only keys `radio_alpha_thin`,
  `radio_alpha_thick` and `radio_log_nu_t` are refused on the power-law jet with
  an error naming the model that reads them; a misspelt key gets the ordinary
  unknown-key error.
  A hand-built parameter dict passed to the radio component or to `tengri.pipeline` for
  the power-law model needs `radio_log_nu_cut` (13.0 is the declared default).

- Casey (2012) dust emission emits from 1 µm rest-frame and longer, normalized
  to L_absorbed on the supplied wavelength grid. The residual grid
  dependence is first order in the width of the cell straddling 1 µm: below
  1e-3 in L_nu(350 µm) on grids with cells of about 100 Å or finer there, and
  about 1e-2 at α = 1 for a 1000 Å cell. The mid-IR power law has no finite blue limit as α → 1, so
  emission below 1 µm is set to zero; without that bound the 250–500 µm band
  flux at α = 1.0 on grids starting at 10 Å, 912 Å and 1 µm is 0.81, 1.29 and
  1.86 times its value on a grid starting at 100 Å (#2708).

- The NIFTy (`vi*`) and native (`native_vi_*`) variational engines score a photometric upper or lower limit (`data_mask` 1 / -1) as -ln Φ((F − m)/σ) or -ln Φ((m − F)/σ) like `map`, the samplers and `vi_fullrank`, with the limit bands entering the geoVI/MGVI metric as detections at their limit value; a free noise model together with a limit raises `ParameterError`. The NIFTy likelihood is built per `Fitter` and only the data-free physics is cached on the model, so a second `Fitter` on one model object fits its own data; the `hmc_is` evidence evaluation takes the data at call time, and the NIFTy free-noise likelihood passes free parameters only to the forward model (#2667, #2668).

- The BAGPIPES reproduction compares tengri and BAGPIPES on matched inputs:
  BAGPIPES is built on a converged wavelength grid (median λ/Δλ asserted),
  band integrals run on each SED's own nodes through `band_average(...,
  integrate="sed")` (the default `"filter"` is unchanged), the SSP file
  carries BAGPIPES's L☉, tengri's metallicity scatter is set to a single
  node, the continuity SFH takes explicit bin edges, and each code's own
  photometry is compared at z = 0 and 0.5.
- `dense_basis` and `dense_basis_pure` place their tx quantiles on, and normalize their mass over, the age of the universe at the galaxy's redshift, so the declared mass forms inside [0, age(z)] at every redshift (Iyer et al. 2019); that age comes from the redshift and the cosmology — `sfh_db_age_universe_gyr` / `sfh_dbp_age_universe_gyr` are not settings, and writing either raises at build time; `predict_sfh`, `predict_sfh_quantities` and `sample_sfh_prior` (which takes a `redshift`) evaluate the age-anchored families at the age of the universe of the model's redshift, through the same rule as the forward model (#2592).

- The model reference weights the band-averaged flux by `w = 1/λ` (photon counting, the default) instead of `λ`, states the AGN radio loudness as `log10(L_5GHz/L_4400)` instead of `L_5GHz/L_2500`, and gains the CIGALE convention differences it had not stated: equivalent-width sign and continuum, the star-forming radio normalization (q_IR and the anchor frequency), the AGN jet cutoff and loudness anchor, the nebular density axes and the emission-line profile (#2627, #2663, #2626).

- NSS `log_evidence_err` is now sqrt(H / n_eff), with H the information in nats and
  n_eff the live count corrected for batch deletion, instead of sqrt(ESS) / n_live,
  which does not track H. log Z is unchanged. On a d = 5 ball prior with R = 100 the
  old value was 0.24 nats against a measured scatter of 0.44 nats; the new one is 0.42 (#2443).

- Line-flux limits are scored as censored likelihoods on every path: an upper
  limit contributes ln Phi((F - m)/sigma) and a lower limit ln Phi((m - F)/sigma)
  (F the limit value, m the model flux, sigma the flux uncertainty), evaluated
  with a log-CDF with no floor, so a strongly violated limit keeps a finite value
  and gradient. `LineFluxData.log_likelihood` honors `is_lower_limit`,
  `LineFluxData.chi2` sums detections only, and the joint/spectroscopy loss scores
  the line-flux term through `censored_neg_log_likelihood` when the data carry
  limit flags (#2665, #2666).

- **The tabulated Schreiber library mixed PAH by power, not by mass (#2597):** the standalone component unit-normalised the continuum and PAH templates separately before mixing, so its `f_pah` was a power fraction (a factor of about 3 in PAH power against the mass fraction CIGALE and Schreiber et al. 2018 define); templates are now mixed per kilogram and renormalised after mixing. (#2597)
- The analytic dust precompute (`modified_blackbody`, `casey2012`, `graybody`) interpolates ln(band flux) with a monotone cubic Hermite (PCHIP) on nodes that span each parameter's declared prior, geometric in `dust_T` and `dust_lambda_0_um`. The nodes are band integrals of the closed-form model on a rest grid of 0.01 µm to 10 m, evaluated in batches, so a band from the far infrared to the radio reads the model's flux; a band whose rest-frame red edge lies beyond 10 m raises `ValueError`, and below 0.01 µm the template is taken as zero. The grid is stored as ln(band flux) taken in float64, so float32 values and gradients are finite. Against the exact closure at random points inside the declared priors, the maximum error at the default nodes is 3.5e-4 (`modified_blackbody`), 2.9e-4 (`graybody`) and 8.0e-4 (`casey2012`, 60-90 µm; 6.7e-4 and 7.0e-4 in 250-500 and 750-950 µm) in the 60-90, 250-500 and 750-950 µm bands at z = 0, and under 1e-3 in 250-500 and 750-950 µm at z = 3; `casey2012` at 8-24 µm is 1.6e-3 at z = 0, and at z = 3 it is 2.0e-3 in 60-90 µm and 7.6e-4 in 8-24 µm. The 1 µm lower bound of `casey2012` is a node of the rest grid. The #2676 reproducer (T 47.3 K, β 1.65, λ₀ 130 µm) gives lookup/exact of 1.0000 in all three bands for all three models, where the old lookup was 4 % low to 10 % high, and 0.9999 / 0.9998 for 15 K dust in 8-24 µm, where it was 0.9-1.0 % high (#2676).

- `fit_batch`'s shared vmap adaptation forwards the spec to the dense-mass
  gate (#2513). It was the one `resolve_dense_mass_gate` caller without
  `spec=`, and with `spec=None` the auto-policy's dense_basis exception
  cannot fire: a dense_basis spec at `n_dim <= 12` was actively granted the
  dense mass matrix the policy exists to refuse (the 22.78 GB adaptation
  spike of #319), on the one seam whose single shared adaptation serves
  every galaxy in the batch. The regression test drives the real gate
  through `fit_batch` with a dense_basis and a DPL arm, so the diagonal
  verdict is pinned as spec-driven.
- `compute_effective_wavelength` returns the pivot wavelength √(∫Tλdλ/∫T/λ dλ) its name and docstring promise; the filter-convention text attributes the photon-counting mean to BAGPIPES as well as DSPS/FSPS/Prospector/Synthesizer and the energy mean to CIGALE's energy-type filters; the facade SED plot derives band wavelengths from the filter curves (#2610).

- `load_ssp_data` gives FSPS MIST + Chabrier grids that carry no `ssp_mass_remaining`
  the metallicity-dependent FSPS table (12 × 107, packaged) when the grid's age and
  metallicity nodes match it; other grids keep the metallicity-independent DSPS fit
  (#2614).

- Spectroscopy-only models under `SpectrumPrecomp` redden the same young stars as the exact screen: the spectrum LUT published its own, 2.3× sharper birth-cloud age indicator, which put the LUT spectrum of a 1–100 Myr population up to 21 % above the exact path at rest 1600 Å; the LUT agrees with the exact path to the documented two-component residual (#2591).

- The Fritz et al. (2006) torus template is read as luminosity per unit wavelength: the shipped grid holds CIGALE's `model.dust` and `model.disk` arrays (W/nm), and the loader converts them with L_ν = L_λ λ²/c before scaling ∫L_ν dν to the requested luminosity, as the SKIRTOR loader does. Read as L_ν, the array tilted every Fritz torus SED by λ⁻²: at the CIGALE default node (r = 60, τ = 1, β = −0.5, γ = 4, Θ = 100°, ψ = 50.1°) the median wavelength of the torus power was 1.8 µm and the 1–3 µm band held 76 % of it, against 5.8 µm and 28 % in the library. The torus power is unchanged. At the four tested nodes the band fractions agree with the library within 12 % per band (bands holding at least 10 % of the power) and the median power wavelength within 10 %; elsewhere on the grid the triweight lookup in parameter space differs from the library's node values by up to 28 % per band and 18 % in the median (β = −0.75, τ = 0.1; #2606). The grid builder labels `dust` as W/nm per unit integral over wavelength and `disk` as W/nm relative to it, in place of erg/s/Hz (#2604).

- The THEMIS `U^−α` dust-emission component for α ≠ 2 is tabulated per q_hAC grain composition (Jones et al. 2017; Draine & Li 2007 Eq. 23), so the power-law axis follows the carbon-grain fraction. Relative to the composition-averaged axis, band power at α = 1 changes by 1.40× (24–1000 µm at q_hAC 0.02, U_min 0.5) and 0.78× (q_hAC 0.40, U_min 30), the 8–24 µm band at α = 3 by 1.94× (q_hAC 0.02, U_min 50, γ 0.9), and the PDR power weight by 0.71–1.28×; α = 2 is unchanged. (#2598)

- Direct calls to the composable AGN runner (`compose_l_nu`) overflowed float32: the reference-L_bol factoring (#1206) lived only in `AGNSEDComponent`, so the runner exponentiated the true `agn_log_lbol` inside the blocks. The factoring lives in `components/agn/_lbol_reference.py` and is called by the runner and by the component's monolithic branch, so the direct call and the `SEDModel` path share it. float64 outputs on the `SEDModel` path are bit-identical (measured). (#2321)

- CI compile caches fit the GitHub Actions quota: pull request runs no longer save
  JAX compile cache entries to the repo's shared cache (reducing PR-scoped bloat from
  6 GB per push), and per-job `TENGRI_JAX_CACHE_MAX_GB` caps bind each job's entries
  so their union fits GitHub's 10 GB repository cache limit (sizes from warm working
  sets: contract 1.55, components 0.6, regression-a 1.0, regression-b1/b2 0.7/1.1,
  regression-c 0.05, regression-d 0.65, physics 0.28, slow inference parts 1/2/3
  and integration 0.2/0.2/0.65/0.5, components-unit-regression-contract-physics 0.12,
  agn-wildcard-liveness 0.15, crossval 0.05, notebooks 0.25 GiB). Contract and
  regression-a timeout budgets now cover a cold cache: 90 and 85 minutes respectively,
  without renaming the required checks (#2549).
- The SKIRTOR disc tied to the torus power (`agn_ir_frac` > 0) carries the library ratio disk(i)/disk(0) once: that ratio is the accretion-disc anisotropy η(i) = cos i (1 + 2 cos i)/3 (∫disk(i)/∫disk(0) over η is 0.9998–1.006 for i ≤ 40°) and, for i > 90° − oa, the torus extinction, so `R` carries no explicit η and the disc is not screened again; the broad lines and FeII keep the torus screen, and at `agn_ir_frac` = 0 the disc keeps it too. With polar dust off, disc power per unit `agn_power` is 0.9946, 0.9945, 0.9943, 0.9920, 0.9892 of CIGALE at i = 0, 30, 50, 70, 90° (with polar dust on, the disc normalization differs: #2602); the 0.55 % offset at i = 0 is the torus template's triweight smoother (∫torus/`agn_power` = 0.9963) times the 136-node library axis (R_library/R_CIGALE = 0.9981). The CIGALE piecewise discs (`disk_type` 0, 1, 2) are zero below 8 nm and from 10⁶ nm up and are normalized by the closed-form integral of the broken power law (log-space, float32-safe), so their level at a wavelength is independent of the wavelength sampling; the 2500 Å and 4400 Å intrinsic-luminosity anchors of the `skirtor` piecewise disc rise by 14 % with its 8 nm cut (α_ox and radio loudness follow), those of the Schartmann disc by 0.2 %. (#2601)

- Upper and lower limits are scored with the Gaussian CDF at their own σ_obs (Boquien et al. 2019, Eq. 15); the calibration floor `noise_frac_cal · |model|` enters detections only, as in the CIGALE implementation (#2619).

- `psb_flex` and `psb_suess2022` now include the link ratio (`ratio_old_0`) between the oldest flexible bin and the youngest fixed bin, matching Suess et al. 2022's one-ratio-per-fixed-bin count (previously `n_fixed − 1` ratios, with that step silently pinned at 0) and enabling independent control of the post-starburst SFH amplitude across the quenching-to-old transition. **Breaking:** `ratio_old_*` indices shift: the new `ratio_old_0` is the link (default 0, reproducing today's SFHs bit-exactly); former `ratio_old_0` through `ratio_old_{n_fixed−2}` (the adjacent-step ratios) are now `ratio_old_1` through `ratio_old_{n_fixed−1}`. A caller that passes only `n_fixed − 1` values through the flat-kwarg path (e.g. `ratio_old_0=0.2, ratio_old_1=-0.3`) now leaves the link (`ratio_old_0`) at its default 0 and reads the adjacent steps one index higher than it meant to — there is no silent re-indexing to the old meaning, so update call sites to `ratio_old_1=0.2, ratio_old_2=-0.3` (plus an explicit `ratio_old_0` if a nonzero link is wanted) (#2612).
- `psb_flex` and `psb_suess2022`'s fixed old bins now span `[tflex_gyr, age_at_z(z)]` instead of a redshift-independent `[tflex_gyr, 13.7 Gyr]`, through the same `age_universe_yr` injection `psb_wild2020`'s burst already receives. Previously the default model at z = 0.5 formed 33% of its stellar mass before the Big Bang (`SFHBeforeBigBangWarning`) and published star formation out to 13.3 Gyr lookback, 4.7 Gyr older than the universe; now no default-configuration psb build warns. At z = 0 the oldest edge shifts from 13.7 to `age_at_z(0)` = 13.7869 Gyr (#2645).

- The Asada et al. (2025) CGM Lyα damping-wing cross-section is σ(ν) = [3λ_α²A/(8π)]·A(ν/ν_α)⁴/[4π²(ν−ν_α)² + A²(ν/ν_α)⁶/4] (Miralda-Escudé 1998); the prefactor 3λ_α²A/(8π) = πe²f/(m_e c) already contains the oscillator strength, which the code multiplied in a second time (0.0046 against 0.01105 cm² Hz). Transmission at z = 8, rest 1220 / 1230 / 1240 Å: 0.0007 / 0.523 / 0.802 (was 0.050 / 0.764 / 0.912) (#2629).

- A `params_override` redshift on a non-catalog precompute model evaluated tables built at the model's own redshift (a 45% loss error on a `WavePrecomp` model moved from z=0.05 to 1.0): the fixed-z stellar LUT, IGM band factors, nebular grid reference, dust-IR band response, energy-balance LUT, radio/X-ray term responses and luminosity distance all stayed at the build redshift. The `Fitter` now evaluates a model built at the override redshift (`SEDModel.with_fixed_redshift`, cached per redshift), so the override is exactly a direct build; `fitter.model` is that rebuilt model. This is also the fix for catalog rows fitted with a per-galaxy `redshift_col` and no `catalog_z_range`. `catalog_z_range` models keep their runtime redshift route.

- The `FeaturePrecomp` nebular grid's photometry followed a build-time reference redshift (a prior draw for a free redshift, the placeholder for a `catalog_z_range` fit) instead of the evaluation redshift, so a dust-free Cue model's band fluxes were off by 8-22 % (20-22 % for `Uniform(0.05, 2)`, 12-15 % for `FREE`, 8-18 % for a catalog fit), and `Fitter`'s `approx="auto"` attaches that grid to every photometry-only fit of such a model. The grid now splits the nebular SED into continuum plus line catalog (both linear in Q_H), places the lines in each band at the evaluation redshift as delta lines, and tabulates the continuum over ln(1+z); a model whose redshift is a build-time constant keeps exact rendered line responses. The reference redshift is now a deterministic convention (the `Fixed` value, else the prior's or `catalog_z_range`'s lower bound). The grid also publishes the split as `nebular_phot_lnu_lines_precomp`, `nebular_phot_lnu_cont_precomp` and `nebular_line_phot_waves_rest`.

- A band wholly blueward of the observed Lyman limit (SDSS g at z = 6) carried rounding noise instead of exact zero in `WavePrecomp` photometry with a live `neb_fesc`: the escape-fraction mask built the stellar bucket as the whole-band integral minus `(1 - fesc)` times the Lyman-continuum integral, two equal ~1e30 numbers whose residue (2e14, a different value in each compiled graph) made a free-redshift build differ by 22 % from the fixed-redshift one in a band 1e-17 of the others. The stellar component now publishes the λ >= 912 Å half of the split from the same cumulative integral (`stellar_phot_lnu_precomp_nolyc` and its per-age twin), and the mask adds `fesc` times the other half to it. The on-disk z-table cache is bumped to version 5. Building a fast nebular model no longer re-integrates the stellar redshift table three extra times.

- The analytic dust precompute (`modified_blackbody`, `casey2012`, `graybody`, `pah_drude`) integrates observed-frame filters at rest wavelengths λ_obs/(1+z); the source redshift reached only the CMB heating term, so at z > 0 the lookup returned the band average at λ_obs instead (#2647).

- The ChEES upstream-limitation test asserts the BlackJAX behaviour per version: below 1.7 the diagonal-mass + length-floor combination raises under `jit`, from 1.7 it traces; CI (BlackJAX 1.7.1) was failing on the old assumption (#2695).
- AGN emission lines (composable NLR, BLR and FeII; GRAHSP's lines and FeII forest; QSOGen's line template) receive the instrument kernel alone in the spectrum projection, like nebular and shock emission: each line is painted at its own width and never passes through the stellar library, so the observed width is √(σ_line² + σ_inst²) and no longer grows with the stellar σ_v (a 500 km/s FWHM narrow line read 290 km/s at σ_v = 200 km/s against the true 218 km/s, +33 %). The AGN component publishes the line-only light `sed_agn_lines_attenuated`, as it enters the SED after the AGN's own screen, and the dust adapters apply the host `agn_screen` to it when the AGN runs first (#2565).
- `age_kernel='dsps'` gives the histogram kernel an SFR table refined 8-fold between SSP nodes; with one row per node the node containing the SFH onset lost its whole weight (delayed-tau, onset 5.0 Gyr: dsps/cic flux +2.32/+2.00/+1.48/+1.23 % in FUV/u/r/H, now -0.05/-0.06/-0.04/-0.03 %) and the flux jumped as the onset crossed a node. `dsps` outputs move; `cic` is unchanged for non-field models. Structure narrower than the node spacing raises `DSPSUnresolvedHistoryWarning` (#2683).

- The window LUT behind `measure_line_fluxes(approx=True)`, `predict_spectral_indices(approx=True)` and the line-flux loss channel applied the dust screen at each window center, where the exact path applies it across the window; a faint line beside a strong one (a small difference of two large window means) therefore disagreed by 13 % for [N II] 6584 next to Hα (and 5e-5 to 3e-4 for the other lines and indices). The LUT now keeps the SSP integrand per grid point and applies the screen there, so it equals the exact measurement to float rounding on every line and break/EW index (#2677).
- The analytic dust-emission precompute (`WavePrecomp` on `modified_blackbody`, `casey2012`,
  `graybody`) built its node axes over the declared free prior (dust_T 20-80 K, dust_beta_ir
  1-2.5, dust_alpha_mir 1-3, dust_lambda_0_um 50-500 um), and the lookup holds the edge value
  with exactly zero gradient beyond the nodes. A widened prior (`dust_T: Uniform(10, 120)`) or a
  `Fixed` value outside the declared range therefore gave a flat likelihood and no gradient over
  the part the nodes did not reach, with nothing raised. The default axes now span the declared
  range extended to what the model can reach, at the declared node density in the interpolation
  coordinate (ln for dust_T and dust_lambda_0_um), so the #2676 accuracy carries over; with
  default priors they are unchanged bit for bit. A supplied axis that does not cover that reach
  raises `ValueError`, a supplied axis of fewer than 4 nodes warns (PCHIP degrades to a parabola
  or a chord), and an unbounded prior (which cannot be spanned) emits one `GridSupportWarning`
  (#2722).

### Changed

- Behaviour change: `agn_log_lbol` is the accretion power radiated into all directions, and the
  line-of-sight luminosity is a derived output. A thin disc radiates proportional to cos i from
  each face and the hot corona is isotropic (Kubota & Done 2018, Sects. 2.1 and 2.2), so with
  `D_nu` the angle-integrated spectrum of the disc and warm zones, `H_nu` that of the corona and
  `L_acc = int (D_nu + H_nu) dnu = 10**agn_log_lbol L_sun`, the `kubota_done`, `multicolor` and
  `relagn` discs return `L_nu(i) = 2 cos i D_nu + H_nu` (the disc alone for the latter two): the
  luminosity density an observer at inclination i assigns assuming isotropy. Before, the
  bolometric normalization carried the same cos i as the rings, so it cancelled it (the 5100 A
  level moved by 0.999, 0.991 and 0.976 at i = 30, 60 and 75 deg instead of 0.866, 0.5 and 0.26
  relative to face-on) and the true power was 1/cos i times `agn_log_lbol`. Now the power at
  cos i = 0.5 equals `L_acc`, the mean over cos i in [0, 1] equals `L_acc`, the zone radii, T(r),
  mdot and the corona power do not depend on i, and the default i = 30 deg disc is 2 cos 30 =
  1.732 times its `D_nu` (the optical and UV of a default AGN is 1.7 times brighter relative to
  the torus and the corona). `L_2500_intrinsic` and `L_4400_intrinsic` are the line-of-sight
  values at 30 deg. The new `log_L_agn_los` (dex re erg/s, composable model) is
  log10 int L_nu(i) dnu of the direct emission before the torus screen and the polar dust: compare
  it, not `agn_log_lbol`, with a catalog L_bol from a bolometric correction; `L_agn_bol` stays
  `L_acc`. `ring_area` is the isotropic-equivalent projected area 4 pi (2 pi r dr) cos i with no
  floor at cos i = 0.01. (#2678)

- Breaking: `Spectroscopy.resample` defaults to `"auto"` (was `"point"`), decided in the model's rest frame at the fixed (or lowest prior) redshift by one function that every spectrum path calls, including `spectrum_from_sfh`. Pixels wider than the model grid now return the pixel mean of the light after the line-spread function (the Gaussian LSF acts on the model grid, then the bin integral; a DESI resolution matrix still acts on the pixels): a sigma = 1 Å line in 2 Å pixels read +14 % at its centre when point-sampled. `Spectroscopy(resample="point")` restores the old values. `SpectrumPrecomp` raises on pixels wider than the model grid instead of warning; pixel-integral gradients in redshift are continuous (#2530).

### Added

- `dust_attenuation={'nebular_screen': 'own'}` gives the nebular continuum and every line flux their own screen, `T_neb(λ) = exp(−tau_neb · k_neb(λ))`, with the law `law_neb` and the new parameter `dust_tau_neb` (Fixed at 1, `Uniform(0, 4)` when freed), on `two_component` and `age_binned`: the total nebular attenuation (CIGALE's `E(B−V)_lines` convention), with no diffuse cascade, no young/old mixture and no obscuration floor; its absorbed energy joins the dust budget like the other screens. `tau_neb` (and, on `age_binned`, `law_neb`) is refused with any other choice, naming `nebular_screen='own'`, and `age_binned` now refuses a `nebular_screen` of `'diffuse'` or `'none'` instead of ignoring it (Part of #2625).
- `neb={'type': 'cue', 'nitrogen': ...}` selects the meaning of `gas_logno`: `'absolute'` (default, unchanged) is Cue's [N/O] input, and a relation name (`'nicholls17'`, the Nicholls+2017 two-regime N/O--O/H fit, `_default_nitrogen.py`) makes it the offset from that relation at the gas metallicity, as `neb_dno` is for the grid backends. Before, Cue's [N/O] stayed solar at every `neb_logZ_gas`, so [N II] 6584 / H-beta at 0.3 Z_sun was 2.2x CloudyGrid's; under `'nicholls17'` it is 0.88x (1.06x at Z_sun). The effective absolute [N/O] is published as the `log_no` property, the #2569 trained-range warning and narrowing bound it in both modes, and `nitrogen` on a non-Cue backend raises (#2693).

- `ingest_catalog(default_relative_error=f)` and `read_catalog(default_relative_error=f)` keep a flux column that has no error column with `error = f * |flux|` (CIGALE's `defaulterror`), and `ingest_catalog(lim_flag=...)` reads CIGALE's error-column encoding of limits: `"none"` drops a band with `err <= 0`, `"noscaling"` and `"full"` take `err < 0` as an upper limit at the flux with sigma `|err|`; both options default to the previous behavior (#2628).
- `generate_mock(model, params, key, noise=sigma_obs)` draws each band from `N(flux_true, |sigma_obs|)` with the supplied per-band observed errors (CIGALE's `mock_flag` draw); `snr` is ignored when `noise` is given and the default is unchanged (#2628).

- `WavePrecomp(igm_fold="exact")` and `"auto"` now fold the IGM exactly on a free redshift: the sub-band ratio is taken at every node of the photometry z-table, where it previously refused. Measured on a z = 6.5-7.5 bare-stellar model against `approx=None`, worst over bands with more than 5 % surviving flux: exact 0.42 % at `n_z=32` (1.6 % at 16), node fold 105 %. What remains is the triweight z-interpolation, so it shrinks with the z spacing. Bands redward of Ly-alpha skip the quadrature (their ratio is exactly one) and the table is content-cached beside the z-table in `~/.cache/tengri_precomp`, so the exact build costs about what the node build does.

- `gordon03_smcbar`: Gordon et al. (2003) SMC Bar empirical extinction curve, tabulated and interpolated, normalized to k(5500 Å) = 1, alongside the existing `smc` (Pei 1992) and `prevot_smc` curves. Registered as a parameterless dust law repackaged from dust_extinction.averages.G03_SMCBar (#2528).
- The per-Q_H nebular grid serves broadband photometry and line fluxes to dusty models when the dust component has a stellar energy-balance LUT (`nebular_from_grid=True`). Dust channels populate the table: observed and rest-frame sub-band nebular photometry, flux-weighted wavelength per sub-band, and dust-absorbed nebular luminosity per unit Q_H on the optical-depth grid (signed). Measured on the paper's configurations, galaxy 79, quiet machine: fit-surface gradient 2.69 ms (configuration I) and 2.35 ms (II) against 25-32 ms; gradient FLOPs 4.0-4.1 M against 67-94 M (#2387).

- Single-component dust now reddens the nebular band flux where the emission is (the band integral of the reddened continuum), matching two-component dust since #1738. `WavePrecomp()` against the exact model at tau_v = 2.3 on paper configuration II: 4.0e-2 before, 2.7e-3 after (#1738).

- The grid applies `neb_fesc` and `neb_fdust` at reconstruction (computed per-galaxy from parameters); the table is built at zero for both and every channel is scaled by `lyc_dust_escape_factor`. Any other free nebular parameter held at the build value (`neb_fesc_lya`, `ionspec_*`, `gas_*`, `neb_eline_sigma_kms`, `neb_log_nH`, `neb_co`, `neb_dno`, `neb_hbfrac`) is refused by enumeration of the namespace.

- With a free redshift (``redshift=Uniform(...)``) or a runtime redshift (``WavePrecomp(catalog_z_range=...)``), the grid serves line fluxes only; band fluxes take the exact nebular path. Measured before: up to 7.35e-2.

- **Known limits** (not changed by this feature): the grid holds the ionizing spectrum shape at the reference star formation history. For a population with no recent star formation and zero birth-cloud optical depth the u bands were off by 3.3e-2 on one prior draw of configuration I; over 32 prior draws as drawn the worst band is 3.9e-3 (I) and 1.1e-2 (II); posterior draws are within 7.2e-4 (I) and 1.3e-3 (II).

- The vmapped catalog MCMC engine now profiles the stellar mass: `profile_mass="auto"` applies to `CatalogFitter`'s native NUTS/HMC path, and the analytically marginalized mass is reinserted per galaxy (via `mass_profile.reinsert_profiled_mass`, against that galaxy's own channels) before summaries are attached — 4.9x on a 6-galaxy photometry catalog. Previously the vectorized engines pinned `profile_mass=False` (#2254); a positional-array `init_from` still stands profiling down, since its width is the un-profiled dimension (#2423).

- `dust_emission={'diffuse_screen': True}` passes the re-emitted IR dust emission once through the diffuse dust screen (single pass; the IR energy absorbed on the way out is removed, not re-emitted); `log_L_ir_emergent` reports the escaping IR luminosity while `L_ir`/`L_absorbed` keep the absorbed budget. Off by default (#2533).

- `lsf_scale`: a free multiplicative scale on the instrument LSF resolution
  (default `Fixed(1.0)`, prior `Uniform(0.8, 1.2)`), applied to both the
  stellar-continuum and emission-line spectroscopy kernels (#2526).
- The spine sync script gains a `--check` mode that diffs the normalized twins against the committed files and the smoke job runs it, so a stale docs/spine twin fails CI instead of shipping (#2134).
- `predict_line_fluxes` and `measure_line_fluxes` now apply the model's configured IGM transmission, profile-averaged over the line's Gaussian width in observed wavelength, matching the same `igm_absorption` dispatch the spectrum/photometry channels already use. Previously the line-flux surfaces read a rest-frame catalog and divided by `4 pi d_L^2` directly, so a line at a redshift where the Lyman forest bites (e.g. Ly-alpha at z >~ 2) came out brighter than the same galaxy's own spectrum. `igm={'type': 'none'}` (or an absent IGM component) leaves line fluxes unchanged; every registered mean-IGM model is supported (#2520).
- A declared `line_flux_scaling` nuisance parameter (default `Fixed(1.0)`, free via `spec.merge_observation_params(line_flux_scaling=LogNormal(mu=0.0, sigma=0.05))`) multiplies every predicted flux of the `Observation.line_fluxes` channel before the likelihood comparison, absorbing an aperture / absolute-flux-calibration mismatch between that channel and the rest of a joint fit. Fixed at 1.0 by default, so an existing fit is unaffected unless a user opts in; `profile_mass=True` refuses when it is free alongside a profiled line-flux channel, since the two amplitudes are otherwise degenerate (#2527).
- `dust_attenuation={'type': 'age_binned', 'screens': [...]}` generalizes the Charlot & Fall (2000) two-component (birth-cloud/diffuse) screen to N independent screens, each its own registered attenuation law and a log-age window `(lo, hi)` in `log10(age/yr)` (`None` = unbounded); windows need not partition the age axis. Optical depth is `tau(t, lambda) = sum_i w_i(t) tau_i k_i(lambda)` with `w_i` a shared-width logistic window weight; nebular continuum and the line catalog are attenuated at the youngest age. `screens = [{'law': law_bc, 'window_log_yr': (None, log10(t_birth))}, {'law': law_diff, 'window_log_yr': (None, None)}]` reproduces `two_component` bit-identically (the N=2 case). Per-screen parameters are indexed (`dust_tau_0`, `dust_tau_1`, ..., `dust_<lawparam>_i`), generated from the screen count the way the nonparametric SFH/metallicity ladders generate indexed declarations; each law-parameter's registry default is that law's own published value, not the shared two_component stem. Not yet supported under `approx=WavePrecomp()`/`SpectrumPrecomp()` (both raise, naming the exact path); a fit's `approx="auto"` policy resolves to the exact path for this type instead of raising (#2528).
- `agb_dust={'type': 'fsps_shell', 'weight': ...}`: a runtime-tunable weight on the Villaume, Conroy & Johnson (2015) circumstellar AGB dust-shell reprocessing FSPS bakes into `fsps_mist_*` SSP grids at its own default (`agb_dust=1.0`). One free parameter `agb_dust_weight` (default `Fixed(1.0)` = the grid as shipped, free `Uniform(0, 3)`), applied as a per-(Z, age, wavelength) multiplicative ratio measured directly from FSPS (MIST + MILES + Chabrier, reused for the `c3k_a` grids and other IMFs; `scripts/generate_agb_dust_shell_ratios.py`, repackaged as `data/agb_dust_shell_ratios_mist.h5`) to the SSP flux cube before the metallicity interpolation and SFH age-weight sum, so it is exact and differentiable. The ratio is stored on a ladder of 13 weights and interpolated linearly, within 2% of direct FSPS between stored weights (3.6% below w = 1/1024, where the ratio is nearly discontinuous). A Fixed weight is baked into the SSP grid at build time (every precompute table stays exact); a free weight is applied live on the exact path, and `WavePrecomp`, `SpectrumPrecomp`, the `FeaturePrecomp` SSP window table and `approx=True` index / line-flux measurement refuse to build, naming the exact path (`approx="auto"` resolves to it automatically). `agb_dust={'type': 'none'}` or omitting the group leaves the grid untouched; only supported on FSPS MIST grids (#2534).
- `age_kernel='cic'` accepts a correlated-field SFH: the draw is the linear interpolation of its own lookback nodes, integrated with the nodes as exact knots; `dsps` samples the same interpolant. **Breaking:** field SFHs default to `age_kernel='cic'`; pass `'dsps'` to keep the old kernel (#2684).

### Changed

- The AGNfitter-rX reproduction notebook source is corrected and extended: the capstone sums every component of upstream's `ymodel` (GALAXY, STARBURST, BBB, TORUS, radio) from upstream's own templates and equations, the stellar-mass and radio `q_IR` conventions are mapped explicitly, torus and dust residuals are attributed to model-grid sampling, and Summary and References are separate cells. The driver gains the matching upstream evaluations (`ymodel_sum`, `bbb_with_xrays`, `galaxy_lnu`, `galaxy_redden_calzetti`, CAT3D union library). The committed render is stale until the notebook is re-rendered.

### Fixed

- The dust-IR band response, the radio and X-ray term responses, and the
  energy-balance LUT of a redshift-reading attenuation law (`narayanan_z`)
  followed a build-time redshift, so under `WavePrecomp(catalog_z_range=...)`,
  where each galaxy evaluates at its own runtime redshift while the spec carries a
  placeholder, catalog fits got 50-300% WISE W3/W4 errors (measured 0.50-0.98 in
  W4 and 0.51-2.98 in W3 at z = 0.5-1.5 for `catalog_z_range=(0.05, 2)`). They
  are now tabulated over the model's redshift range, uniform in ln(1+z), and read
  at the evaluation redshift, so every catalog engine inherits them, and a
  free-redshift model uses the fast path instead of falling back to the exact
  per-call integral.
- A dusty model whose nebular flux the per-Q_H grid serves (#2570) now takes the IGM at each nebular sub-band chunk's node rather than the band-averaged `<T>`: a two-component Cue model with dust emission at z = 7 read F090W 10.1 % off `approx=None` near Ly-alpha, now 0.94 % at worst over z = 6.5-7.3 (#2679).
- Under `WavePrecomp` the IGM now reaches nebular, shock and AGN band fluxes through each component's own spectrum, `∫S·T/∫S` per band, instead of the band-averaged `<T>` alone, which formed `<S><T>`. Near Ly-alpha that was the dominant LUT error at high redshift: a Cue model at z = 7.3 read F115W +10.3 % and sdss_z +5.6 % against `approx=None` under the exact stellar fold; both are now 0.0000 %. The transmission is tabulated at build time over the absorbed end of the rest grid, and only for bands the IGM can reach in the model's redshift range, so a low-redshift model is unchanged bit for bit and pays nothing. Under the exact fold the stellar dust screen on the IGM-folded tensor is also evaluated where the IGM-surviving light sits: at the bare sub-band node a two-component model at z = 7 read sdss_z 8.7 % off, now 0.11 %. A composable AGN beside Cue reads <= 0.28 % (node fold 48 %).
- The exact IGM fold now takes its sub-band ratio over the partition of the tensor it multiplies. A model with a live nebular Lyman-continuum mask (every Cue model: `neb_fesc` is fixed below one by default) splits each band into K + 1 chunks with a forced edge at 912 Å; the ratio was built as K + 1 equal-mass chunks without the edge, so shapes matched and nothing raised. Band fluxes straddling the Lyman limit were off by 12.7 % (GALEX NUV, z = 2) to 44 % (z = 2.5); a Cue model at z = 7.3 read i +11.6 % under the exact fold, now -1.2 %. The filter convention is passed through for the same reason.
- `bins` and `bins_continuity` metallicity histories take their bin count from `met_bin_edges_log_yr` — a one-bin ladder gives the base metallicity at every age (both modes), a three-bin ladder gives three metallicities; a ladder with more bins than the six declared parameters, or a `bin_<i>` / `d_log_z_<i>` beyond the ladder, is refused at build time (#2600).

- `read_catalog`: a negative error marks an upper limit at the signed flux (any flux sign);
  a flux or error below −9990, or an error of zero, masks the band and one `UserWarning`
  per column lists the masked rows; a negative flux with a positive error is a detection
  with its signed value; the `-1` lower-limit flag is the ingest path's (`catalog_ingest`),
  never this reader's (#2586).
- `analysis.diagnostics.spectral.uv_slope_beta` is one least-squares fit of log F_λ against
  log λ over the pixels inside the ten Calzetti et al. (1994) Table 2 windows (window 6 =
  1677–1740 Å) with hard window bounds per Eq. 3 and a centered abscissa, so it is stable in
  float32. The window means in `_window_mean_flux`, `dn4000`, `equivalent_width` and the
  spectral-index and line-flux window LUT (`soft_window_ssp_integral`) are wavelength
  integrals, ∫F dλ / ∫dλ, trapezoid-weighted and edge-inclusive, so they do not depend on
  how the wavelength grid is sampled; the `equivalent_width` pseudo-continuum is the
  integrated mean over both sidebands (Vollmann & Eversberg 2006) (#2588).

- The surviving stellar mass is `M_formed · Σ_age Σ_Z w(age, Z) · m_rem(age, Z)` over the joint weights the spectrum uses — each (age, Z) node at its own remaining-mass fraction, for every metallicity history; `predict_sfh_quantities` reads the component's published `log_mstar_surviving` (#2613).

- The dense-mass step-size stability probe (#1999) now also runs after
  adaptation in the dynamic-HMC backend and in `fit_batch`'s shared window
  adaptation, so those paths refuse a step above the metric's stability limit
  like the single-galaxy NUTS/HMC paths; the fused-scan paths remain the design
  item in #2157 (Refs #2157).

- `neb_logU` is documented as Cue's inner-face ionization parameter at R = 10^19 cm, not the
  Synthesizer ionization parameter, which is three times the Strömgren-radius U_S of Gutkin et al.
  (2016, eq. 7), with the mapping between the two at n_H = 100 cm⁻³; the FSPS CLOUDY grid converter
  describes its metallicity axis as log10(Z / Z_sun) as tabulated by FSPS, not "absolute
  metallicity" (#2632, #2633).

- The emission-line catalog's [O I] 6300 entry is the vacuum wavelength (6302.05 Å, the 6300.304 Å
  air value converted with the IAU standard relation), and the hard-coded line-wavelength tables
  (Richardson NLR template, shock fallback lines, `KEY_LINES`, `DESI_LINES`, plotting
  `SPECTRAL_FEATURES`) take the catalog's vacuum values for every line it lists; the `noll09` and
  `salim_sbl18` docs state the Leitherer/Calzetti switch at 1800 Å, as the code does (#2617).

- A model on an SSP that includes nebular emission (a wNE grid) with a radio block carries one
  thermal free-free term at every wavelength (#2574): the SSP flux already holds the nebular
  continuum up to the SSP grid edge (1 cm for `ssp_prsc_miles_chabrier_wNE`), and the radio
  block's Murphy+2011 term ran on top of it from 1 mm, so the total SED was 1.30x at 100 GHz and
  1.57-1.60x at 30-50 GHz of the one-term SED, and a 3 mm band read 1.26x on both the exact and
  the `WavePrecomp` path. With `freefree` unset and `ssp_data.nebular == "included"`, the factory
  now sets `RadioSEDComponentConfig.freefree_wave_min` to the SSP edge: the radio thermal term is
  zero below it and unchanged from it upward. The rule reads the SSP's nebular stamp, not the
  declared `neb` type, so `neb={'type': 'none'}` on such a grid is covered too. `freefree: True`
  keeps the term over its whole range, `freefree: False` removes it, and models on a bare-stellar
  or unstamped SSP are unchanged. The SSP's Cloudy continuum and the Murphy+2011 calibration
  differ by 22-28%, so the thermal component steps by that much at the SSP edge. Validation
  against pcigale, whose radio module is synchrotron only and whose nebular module owns the
  thermal continuum, set the rule.

- The X-ray corona shape `(E/E_ref)^(1-Γ) × exp(−(E−E_ref)/E_cut)` equals 1 at E_ref = 2 keV with
  the exponential cutoff included, so `L_ν(2 keV)` is the monochromatic luminosity of
  Yang et al. 2020 Eq. 2 for every `E_cut` and Γ. The LMXB photon index default is 1.56
  (Fabbiano 2006; Yang et al. 2020 Sect. 2.2.2), as in pcigale (#2583).

- The X-ray block's HMXB and hot-gas terms scale with the SFR averaged over the last
  100 Myr (`sfr_100myr`), the quantity the Lehmer et al. 2016 relations are calibrated on
  (Yang et al. 2022, Sect. 3.3); the instantaneous SFR stands in only for an SFH that
  publishes no 100 Myr average. The registered properties `log_l_x_xrb` and `log_l_x_agn`
  are the 2-10 keV luminosities of the emitted HMXB + LMXB terms and of the emitted AGN
  corona (absorber, scattered fraction and anisotropy included), published by the X-ray
  component as `log_L_x_xrb_2_10` / `log_L_x_agn_2_10` in log10 space, so they equal the
  band integral of `sed_xray`'s terms in float64 and float32. `log_l_x_agn` is `-inf`
  without an AGN. `compute_log_l_x_xrb`, `compute_log_l_x_agn`, `compute_l_x_xrb` and
  `compute_l_x_agn` (the 2.6e39·SFR and Duras relations, none re-exported at a public
  `__init__`) are removed (#2582).

- The `lopez24` corona is anchored to the 12 um nu L_nu of the AGN model itself. The
  AGN component publishes `log_L_12um` and `log_L_6um` (dex re erg/s; disc + torus + polar
  dust of the composable model, the whole SED of a monolithic one), and
  `L(2-10 keV) = nu L_nu(12 um) / 10^alpha_IRX` is formed in log10 space, so the X-ray
  wing is finite in pure float32. A model with no AGN has a zero corona, and the 0.07 L_bol
  bolometric-correction anchor is removed together with `compute_l_12um_from_lbol`. `xray_agn_corona_lopez24` and
  `xray_total_lopez24*` take `log_l_12um_erg` (dex) in place of `l_12um_erg_hz` (#2581).

- Meiksin (2006) IGM: every Lyman-series optical depth (n = 2–30) is evaluated
  at its absorber redshift z_n = λ_obs/λ_n − 1, so the transmission blueward
  of Lyβ follows the paper's Table 2 (#2585).

- The FeII pseudo-continuum (`agn.feii` `boroson_green`, and the FeII term of the analytic `compute_blr_sed`) treated the PyQSOFit template's F_lambda column as L_nu and then multiplied by c/lambda^2, imprinting a spurious lambda^-2 tilt (fitted log-slope error -2.05; 0.39 and 1.99 slope offsets against the template over 4000-6000 and 2200-3000 A in the regression test). The template shape is now carried as L_lambda, normalized so the 4434-4684 A energy equals `agn_fe2_strength` x L(H-beta) (window edges honored exactly, grid independent), negative template nodes are clipped before resampling and broadening, and the resampling is linear in wavelength (log-log resampling of the sign-changing template was off by up to 1.3 dex). `data/agn_fe2/PROVENANCE.md` recorded SHA256 values that differed from the shipped files by one character each; corrected and now tested.

- The QSOGen Balmer continuum optical depth ran the wrong way: `tau = tau_BE (lambda_BE/lambda)^3` rose toward the blue, the inverse of the photoionization cross-section scaling sigma_bf ~ nu^-3 (Grandi 1982) and of upstream QSOGen's `taube * (nuzero/nu)**3`, which is `(lambda/lambda_BE)^3`. It also disagreed with the component's own 3000 A normalization, so `agn_bcnorm=1` produced a Balmer continuum 1.95x the power law at 3000 A instead of 1x. Default spectra (`agn_bcnorm=0`) are unchanged.

- The BAGPIPES reproduction stores each BC03+MILES node's absolute log10 Z (BAGPIPES's
  metallicity grid is in units of Z☉ = 0.02) and pins the cross-code comparison at one
  absolute Z (`met_logzsol = log10(0.02) − log10(0.0142)`) in every stellar-metallicity
  request, metallicity sweeps included, while gas metallicity is matched solar-scaled
  (`neb_logZ_gas = log10(z)`); its L_λ↔L_ν conversion uses tengri's speed of light; the
  validator's birth-cloud control states the `eta` it corresponds to (#2616).

- The composable AGN precompute LUT's accuracy is now measured and pinned
  against the exact recipe evaluation (#2288). `interp_nd_triweight` is a
  kernel smoother, not an interpolant, so node parity is not a valid invariant
  for this LUT; the honest numbers on the documented standard 21-node
  `agn_grahsp_log_l5100` axis are ~0 relative error at the grid-center node,
  8.7% at the edge node (one-sided kernel), and a 15.9% maximum at interior
  midpoints (the kernel's Jensen bias plateau on a photometry that is
  exponential in the axis coordinate — well under the 50% refusal rule). The
  bound is pinned at test time with a corruption probe on the engaged
  preintegrated grid; no check runs inside `precompute()` itself.

- `dust_emission={'type': 'graybody'}` is the general-opacity greybody `(1 − e^{−(λ0/λ)^β})·B_ν(T)` of Casey (2012) Eq. 1 with no additional `ν^β` emissivity factor (νL_ν peak 72 µm at T = 50 K, β = 1.5, λ0 = 200 µm; CIGALE `mbb` and Synthesizer `Greybody(optically_thin=False)` agree) (#2596).

- The analytic dust precompute (`modified_blackbody`, `casey2012`, `graybody`) integrates the
  thermal continuum on a 0.01 µm–10 mm rest-frame grid, so 70–1000 µm filters read the
  band-averaged closure to 1e-3 instead of zero; the grid ended at 31.6 µm (#2642).

- The `richards2006` disc template read its L_nu [erg/s/Hz] column as nu*F_nu and divided
  by nu, putting its nu*L_nu peak at 3055 Å instead of the quasar big blue bump near 1260 Å
  and its 1 µm / 10 µm levels +0.57 / +1.57 dex too high. The crossval test applied the
  same misreading; a units contract now pins the L_nu reading against Richards et al. 2006
  Table 3 and guards against future regression (#2563).

- Every registered AGN torus block and dust-emission model now declares its
  wavelength support, so `SEDModel.build` samples it on its own grid instead of
  the SSP grid. The AGNfitter-lineage torus blocks (`nenkova_agnfitter*`,
  `skirtor_agnfitter*`), `cat3d_wind_lowfwd`, `fritz`, `nenkova`, the analytic
  tori, `schreiber2018` and `dh02_ce01` had none: their IR peak sat on an
  SSP node (20 or 40 µm; native 19-31 µm), and the two tabulated dust models were cut at
  160 µm and renormalized there (3-1000 µm band mean up to 1.9x, 160-1000 µm
  fraction 2.2-3.6x). AGN disc blocks emit outside the SSP window as well (energy
  outside 91 Å - 160 µm: `kd18_agnfitter*` 28 %, `kubota_done` 21 %, `skirtor` 15 %,
  `multicolor` 2.5 %, `adaf` 99 %, `adaf_lopez2024` 1.2 %, `schartmann2005*` 0.3 %);
  each now declares its template axis or an analytic support range (the coronal/X-ray end and the outer Rayleigh-Jeans tail), at
  200 points per decade shortward of 1000 A and 40 above. The deprecated `powerlaw` disc has no
  low-frequency cut-off and takes the 1 cm end used by the analytic
  dust and torus grids. `nenkova_agnfitter`'s 4096-point axis is declared at
  stride 4. A contract test fails for any newly registered torus, dust-emission
  or disc block that neither declares support nor is explicitly grid-less
  (#2564).

- The nthcomp template kernel behind the `kubota_done` and `kubota_done_full` discs
  now differentiates exactly in `agn_gamma_warm`, `agn_kt_warm` and the seed
  temperature. Its `custom_jvp` had dropped the seed-temperature tangent and taken
  cell-spanning finite differences for the other two, so `jax.grad` disagreed with
  finite differences (`agn_log_mbh` -1190.6 vs -293.9, `agn_log_lbol` 6% off, at
  `log L_bol` 11.5, `log M_BH` 8.5). A class-wide gradient contract now covers every
  registered disc block. The `kubota_done` / `kubota_done_full` hot-flow zone is also
  fixed. `R_hot` had zero derivative wherever unclipped (bisection differentiated through
  its iterations; now an implicit-function-theorem `custom_jvp`, spin included). Its
  closed-form NT dissipation integral omitted the `R dR` area element. The corona radiated
  an invented `min(f_hard L_Edd, L_bol/2)` while `R_hot` was solved from `f_hard L_Edd`;
  K&D 2018 Sec 4.3 ("Ldiss,hot = 0.02 LEdd, which defines rhot") and the QSOSED/RELQSO
  source have no such cap, so `R_hot` and the corona now share `L_hot = f_hard L_Edd`
  (saturating at 99% of the disc's total dissipation when it cannot be supplied). And the
  disc temperature was the Newtonian zero-torque profile, which dissipates 1.46 `L_bol` at
  spin 0; it is now the relativistic Page & Thorne (1974) emissivity K&D 2018 and RELQSO
  use (energy at infinity = `eta(a) Mdot c^2`), for the outer and warm rings, the seed
  photon temperature, `L_seed` and the `R_hot` integral. `multicolor` stays
  Shakura-Sunyaev by definition. **The `kubota_done` / `kubota_done_full` forward SED
  changes** (2500 A x1.4 and 5100 A x1.4 at `log L_bol` 11.5, `log M_BH` 8.5; +30% from
  the optical to the X-rays at `log L_bol` 11 with the default mass). The self-gravity
  outer radius, shared by `multicolor` and the K&D disc, was 1.67x too large: Laor &
  Netzer's `alpha^{2/9}` had been written `(alpha/0.1)^{2/9}` against the 10^9 M_sun
  normalization (qsosed, RELQSO and K&D 2018 use `2150 alpha^{2/9} m9^{-2/9} mdot^{4/9}`).
  With it, `test_kd18_vs_agnfitter` passes at all nine nodes (the (8, 0) node at 0.005 dex).
  The `kubota_done`, `multicolor` and `powerlaw` discs no longer normalize on the caller's
  wavelength grid: the warm-ring blackbody power, the disc bolometric and the EUV-tail budget
  are closed form, the tail excess and the `powerlaw` band sit on fixed internal nodes, so an
  SED value at a wavelength no longer moves (up to 4e-3 at 10-912 A, 3.7e-2 for `multicolor`
  on coarse grids) with the grid it is evaluated on (#2572).

- `double_powerlaw` and `delayed_tau` now evaluate their shapes in cosmic time
  since formation (T = age − t_lookback) and take a required keyword-only `age`;
  both previously treated lookback time as cosmic time and returned mirror-imaged
  histories (#2524).
- The exact (non-precomputed) forward path's Lyman-continuum mask now steps at
  the physical 912 A edge instead of at whichever SSP grid node sits just
  below it: a photometric band whose rest-frame coverage straddles 912 A no
  longer carries an edge-placement bias of up to a few percent when
  `neb_fesc < 1` (#2447).
- `profile_mass` now reaches six backends it had been silently skipping:
  `nss`, `mcmc_raytrace`, `mcmc_ess`, `pathfinder`, `vi_fullrank` and
  `vi_meanfield` were absent from `PROFILE_MASS_BACKENDS`, so
  `resolve_profile_mass_for_method` disabled profiling before they ran. The
  clearest case is `nss`: `build_profiled_loglikelihood_fn` was written for
  nested sampling and says so in its docstring, but the omission made it
  unreachable, so every NSS evidence run scored its live points at the mass
  placeholder instead of the marginal likelihood — exactly what that docstring
  warns about. Each of the six was audited to its call site into the Fitter's
  loss; the set goes 16 to 22 of 29 registered backends, the remaining 7 being
  the NIFTy and native-VI backends that build their objective from the model
  and spec directly. `tests/inference/test_profile_mass_backend_coverage.py`
  now pins a *partition* (registry == allowlist | excluded-with-reason) rather
  than a membership list, so a newly registered backend fails the test until
  someone classifies it; a membership list would have stayed green through all
  six omissions.
- WG00 attenuation now reaches the emission-line catalog (``predict_line_fluxes``,
  ``predict_line_ratios``, line properties, ``predict_emission_lines``); previously
  lines passed through unattenuated under dust type wg00 while the continuum and
  SED-measured lines were attenuated. ``_line_dust_component`` now selects the dust
  component by capability (``hasattr(c, "attenuate_line_catalog")``) instead of name,
  enabling wg00 to publish attenuated lines on the same surfaces as single_component
  and two_component. The mismatch broke joint-fit consistency for dust parameters (#2541).
- The `profile_mass` linearity guard reports which of **three** kinds it
  measured — `proportional`, `affine` or `nonlinear` — where it previously
  answered only proportional-or-not. The distinction is load-bearing:
  `chi2(M)` stays exactly quadratic for an affine prediction
  `M f(theta) + g(theta)`, so such a model is marginalizable once the offset is
  known, while a nonlinear one never is. The extra mass evaluation that
  separates the two is taken only on the refusal branch, so the proportional
  fast path (every SFH that renormalizes to the mass, hence eight of the nine
  shipped recipes) is unchanged.
- The linearity refusal no longer misdiagnoses. It asserted that an order-1
  deviation "indicates a mass-independent additive component such as an AGN
  continuum" on every model; measured on a `dense_basis` photometry fit with no
  AGN block, the deviation is order 1 and the cause is a mass-dependent SFH
  *shape* (`_build_quantile_points` builds its GP knots from
  `sfr_inst*age/M`, making `log_total_mass` a shape parameter rather than an
  amplitude). #2374 improved the wording without covering that case. The
  message now names the measured kind and names an additive component only
  when the model carries one. An audit of every registered SFH found
  `dense_basis` is the only one that leaks the mass into its shape: the other
  20 measurable ones sit at 9e-15 to 1.3e-14, the roundoff floor, via
  `mean_sfh._renormalize_to_mass`. The linearity probe now evaluates at the
  fit's own fixed values (including a `params_override` redshift) and lets
  model-evaluation errors propagate instead of logging them as invalid thetas.
  The same fix now covers the whole guard chain rather than just that one
  evaluation loop: the affine-vs-nonlinear retest's own third-mass evaluation
  (`_classify_nonproportional`) and the `profile_mass="auto"` guard-check
  dispatch (`configure_profile_mass`) no longer fold a model-evaluation error
  into a silent `"nonlinear"` classification or `"auto-disabled"` reason
  either, and the probe's fixed-values resolution takes the raw
  `params_override` argument directly, so a call reached during
  `Fitter.__init__` — before `self._fixed_values`/`self._params_override`
  exist — no longer silently falls back to the spec's own declared value.
- Student-t noise Hamiltonian now includes the dof-dependent normalisation, so a free `noise_dof` is sampled under a correctly normalised density (#2525).
- `profile_mass="auto"` no longer engages when a user-supplied likelihood owns the data (any data type; previously only the line-flux case was refused, so the mass was profiled against the Fitter's placeholder data and the user's likelihood silently replaced in the mass direction), nor when a spectral covariance is used (#2509); `PrecompBiasWarning` now covers joint photometry+spectroscopy fits per channel and states the LUT forward bias when a user likelihood owns the data (#2510).
- **Breaking**: `delayed_bq`, `periodic` and `buat08` now evaluate CIGALE's formulas
  in time since formation (T = age − t_lookback), as `sfhdelayed`/`sfh2exp` and #549's
  `dpl`/`lognormal` do; previously they read CIGALE's forward time as lookback, giving
  the time-reversed history (a "recent" delayed_bq burst formed at the oldest end with
  SFR(now) = 0; periodic bursts rose slowly and cut off at their onset; buat08 SFR → 0
  today). `periodic` no longer stops after 100 bursts (#2515). `buat08` gains
  `sfh_buat08_age_gyr` (default: age of the universe). Every fit using these three SFHs
  changes meaning; re-fit before comparing (#2514, #2515).

- `dirichlet` joins the bin-edge count rule that `continuity`-backed ladders
  already obey: six declared `z_frac_*` require exactly eight `bin_edges_gyr`,
  and `resolve_sfh` now runs `validate_bin_edges_gyr` itself so direct calls
  cannot bypass the rule the build path enforces. `dirichlet()` refuses unknown
  `z_frac_*` keywords naming the accepted range instead of ignoring them
  (#2479, #2503).

- The eline fitted-mode test mocks now attach `Spectroscopy` through
  `observation.spectroscopy` instead of the private `_spectroscopy_config` that #2455
  stopped reading. The loss builder's channel-scale probe drew its reference
  parameter point from `spec.sample()` (free-only per #2296) that lacked Fitter-registered
  eline amplitudes, raising `KeyError` in fitted mode; it now samples the fitter's
  working spec and merges the fixed values (#2502).

- Lazy DSPS imports (deferred to function-local scope via #2276) now hold the x64 preference where the caller left it. DSPS modules run `jax.config.update("jax_enable_x64", True)` at import time, and lazy imports that execute after the user has set `JAX_ENABLE_X64=0` would silently flip x64 back on mid-run, inflating float32 dtypes to float64. Every lazy DSPS import now runs under `hold_x64_preference()`, a shared context manager that snapshots the current `jax.config.jax_enable_x64` flag at entry and restores it on exit, preserving the caller's preference regardless of whether it was set via environment variable or `jax.config.update()` call. All 10 function-local DSPS imports across `utils/cosmology.py`, `components/stellar/component.py`, `components/stellar/sps/dsps_wrapper.py`, and `observation/filters/custom.py` are wrapped (#2504).
- `StudentT` priors with df ≥ 3 realize their quantile pushforward exactly,
  with an **analytic gradient** rather than one obtained by differentiating
  the incomplete beta function: `dz/dp = 1/f_t(z)` and `dF_t/dz = f_t(z)` are
  supplied directly via `jax.custom_jvp`, so the sampler's Jacobian
  `dθ/dξ = σ·φ(ξ)/f_t(z)` is exact everywhere, including exactly at ξ = 0 (the
  declared prior midpoint). A first attempt at this fix autodiffed through
  `jax.scipy.special.betainc` instead and made things worse: it zeroed the
  Jacobian at ξ = 0 (a 5× error, not the interpolant's kinks) and returned NaN
  gradients for ξ in the open neighborhood around 0. The CDF is reparameterized
  as `F_t(z) = ½ + ½·sign(z)·I_y(½, df/2)`, `y = z²/(df+z²)`, removing a
  near-z=0 cancellation the naive `x = df/(df+z²)` form has; the quantile's
  Newton seed is a monotone cubic Hermite guess with exact knot slopes
  `1/f_t(z_k)`, needing one refinement step instead of four. θ and the round
  trip agree with the exact quantile to the measured accuracy floor of
  `jax.scipy.special.betainc` itself (~1e-9, worst case at |ξ| = 4.5); `df ∈
  {1, 2}` keep their closed forms and are bit-identical to their prior
  behavior in both directions (#2576).
- CloudyGrid, CB19 and MAPPINGS nebular backends no longer divide the Lyα luminosity by the escape/dust factor; Lyα is suppressed by `neb_fesc`/`neb_fdust` like every other recombination line, through one shared helper from the P-11 fix in cue.py (#2531, #743).
- `compute_qh`'s docstring now states its actual input unit (L☉/Hz per M☉, converted in log space for float32 safety), matching `compute_qh_log10`; a test pins the contract (#2532).
- `sigma_v_kms` is now applied on the resolution-matrix branch of `project_spectrum`
  (previously silently skipped there, so intrinsic galaxy velocity dispersion had
  zero effect and zero gradient on the DESI spectroscopy path). `observation/banded.py`
  gains `row_sigma_kms` and `deconvolve_library_lsf` to remove the SSP library's LSF
  from a banded resolution matrix before it is applied, so the library and DESI
  resolution contributions are not double-counted (#2506).
- `FiberSpectroscopyObservation.predict` is now jit/grad-safe in redshift: the
  fiber centre stays a traced array instead of being concretized via `float()`,
  which previously raised `ConcretizationTypeError` whenever `predict` was
  wrapped in `jax.jit` or differentiated with respect to redshift.
- A `Fixed` redshift now reaches the emission-line paths the same way it reaches
  photometry. Under `WavePrecomp(catalog_z_range=...)` the build keeps redshift out
  of the compiled kernel (so `model.z_fixed` is `None` by design), and
  `predict_line_fluxes`, `predict_line_ratios`, `measure_line_fluxes`, and the
  `FeaturePrecomp` catalog snap resolved z only from that baked value, raising
  `KeyError: Redshift not in params and not fixed in spec` at build time with
  `FeaturePrecomp` and at likelihood setup with line-flux data. The loss also
  stripped the evaluation's fixed values before calling the line methods, so a
  runtime redshift (`params_override={"redshift": z}` or a catalog row's z)
  reached photometry but not lines, which silently computed line fluxes at the
  model's build-time redshift. One resolver, `SEDModel._evaluation_params`, now
  merges the spec's Fixed values with the evaluation's own, and the line methods
  take it through a new `fixed_values=` argument, so lines, dust, and photometry
  read one redshift.

- Spectral indices (`predict_spectral_indices`, both the FeaturePrecomp
  window-LUT path and the exact path) now read the evaluation's fixed values,
  including a runtime redshift, like the line methods did in #2499. The redshift
  affects age-sensitive indices via the cosmic age truncation of the SFH: it
  reaches the same `age_at_z` cutoff in `compute_joint_weights` (window-LUT
  path) and in the orchestrator's stellar `apply()` (exact path), so an
  evaluation-time override changes which lookback ages are truncated on both
  paths alike. Two sibling gaps in the SAME class of call, found while
  covering this: `_feature_fast_indices`'s one-off exact measurement for a
  slope index (e.g. `uv_slope_beta`, not a single-window functional) dropped
  the evaluation's fixed values even though the window-LUT slots in the same
  call honored them, so a slope index in `index_defs` could silently disagree
  with a break/EW index measured alongside it; and `measure_line_fluxes`'s
  exact (`state=None`) branch rescaled the luminosity distance to the
  overridden redshift but measured the rest-frame SED itself at the model's
  own build-time redshift. Both now thread the same `fixed_values` (#2510).
- Madau (1995) IGM transmission (`igm_transmission_madau`) now includes the
  metal-line blanketing term (eq. 15), 0.0017·(λ_obs/λ_α)^1.68 blueward of
  Lyα(1+z); this adds up to ~1% attenuation in the Lyα–Lyβ forest at z = 2–4 (#2516).

- `skirtor_sed()` and the deprecated alias `skirtor_analytic()` now accept
  `wavelength` as a keyword argument. Previously, calling with all keyword arguments
  raised `IndexError: tuple index out of range`. Both functions now resolve
  `wavelength` from positional or keyword argument and raise `TypeError` if omitted.
  The CIGALE-era cross-validation test now calls `skirtor_analytic()` with the
  current parameter names (`agn_tau_skirtor`, `agn_p_skirtor`, etc.) instead of
  retired CIGALE-style names (`t`, `pl`, `q`, `oa`, `R`, `Mcl`, `i`) (#2464).
- The numeric-guard ledger now skips `jnp.clip` calls whose results are assigned to a
  plain Name and used only as gather indices (element of `Subscript.slice`, inside an
  expression within a subscript slice such as `table[i + 1]`, argument to
  `jnp.take` / `jnp.take_along_axis`, or slice of `.at[...]` access). These index bounds
  with literal floor 0 do not present a subnormal-risk floor on the value path; the ledger
  improves by ratcheting down count on seven files (#2327).
- A wide log-normal SFH was a staircase in `age`: its support boundary moves with
  the age parameter and a hard mask switched each dense-grid node on at full weight,
  so the trapezoid integral jumped at every node crossing (89 steps above four times
  the median step over a 161-point age sweep on FSPS MIST C3K at width 2.14 dex) and
  autodiff could not see the jumps, which NUTS read as divergences. The boundary cell
  now carries a smoothstep partial-cell weight at the grid's own spacing; bit-identical
  wherever the kernel was already small at onset; ported from the paper-1 pin branch
  (f01975f46). See the SFH-support entry below for the other forms this class of
  defect touches (#2476).
- The `met` group accepts `met_bin_edges_log_yr` (a structural key) for the `bins` and
  `bins_continuity` metallicity types, refusing it on ladder-free types. The key is
  threaded through `parse_groups()`, `sed_model`, and `component_factory()` to
  `StellarSEDComponentConfig`. The #2204 cosmic-age reachability check now judges the
  configured ladder when provided and names the key in the error message (#2433).
- The shipped DL07 template grids carry the published axes of Draine & Li
  (2007): the U_min axis is the 22-node ladder (0.1, 0.15, …, 8.0, 12.0, 15.0,
  20.0, 25.0; the files had labelled the last four columns 10, 12, 15, 20, so
  U_min above 8 selected the neighbouring template and 25 was unreachable) and
  the q_PAH axis carries only the seven MW3.1 nodes (no SMC/LMC2 grain models).
  Spectra are unchanged; `dust_umin`'s prior widens to 25.0 (#2535, #2441).
- Two AGN-NLR fallback defaults read their own parameter declarations instead of
  literals: the `gas_logn` fallbacks in `components/nebular/agn_nebular.py` read
  `declared_default(AGN_PARAMS, "agn_nlr_logn")` and the `neb_logU` fallback in
  `MappingsPhotoAGNBackend` (`components/nebular/mappings_photo.py`) reads
  `declared_default(AGN_PARAMS, "agn_nlr_logU")`. Both values equal the former
  literals, so built models predict identically; the same parameter name denotes a
  different physical quantity in the AGN and stellar contexts, which is why each
  site reads its own declaration. The shock-normalization fallback is unchanged (#2297).
- Four fail-open probes for never-assigned attributes are resolved. `SEDModel.hybrid` property (dead accessor for `_hybrid` never assigned) is removed, along with its mirror property on `ForwardModel`; `SEDModel.wave_obs` property's dead `_wave_obs` cache probe is removed; `sed_model.py` dust-emission detection's legacy `dust.config.emission_model` probe (unreachable after component migration) is removed; `profiling/pipeline.py`'s dead `_compositional` probe is removed; `profiling/memory.py`'s `_weights` probe is fixed to use the correct `weights` attribute on `CueBackend` (#1240). The profiling pipeline's fused/exact path selection now reads `has_fixedz_photometry_precompute` alone; the dead `hybrid` probe kept it on the exact path unconditionally (#1240).
- Three reproduction pages re-rendered with their prose reconciled to the
  measurements (#2341 rows; #2419/#2442 page follow-ups): cigale's §7/§8/§11
  and capstone describe one thermal free-free term on the whole grid, §11 now
  comparing the nebular tail against Murphy+2011 component-wise inside one
  `freefree: True` build — against pcigale the 3 cm ratio moves 0.84× → 1.02×
  and 1.4 GHz 0.95× → 1.00×, with the mm decade unchanged; bagpipes' §12b
  names the Lyman-β edge (1025.70 Å, 0.8–11.4% over z = 1–5) and drops the
  single-pixel 610% Lyman-α claim, and its §7 bump row closes at 2.082 vs
  2.079 after the `salim_sbl18` normalization fix (both sides anchored at
  exactly 2175 and 5500 Å, removing a grid-sampling term that inflated the
  residual to 0.6%); synthesizer's §12b
  attributes the Madau+1995 residual to line-wavelength conventions at the
  Lyman-series edges (vacuum 1025.72 Å vs rounded 1026 Å; 14.6% at z=3,
  21.1% at z=5, single node). The parity matrix's M4 bagpipes arm is closed.
- `PLANCK18` includes radiation and massive-neutrino densities; D_L and age(z) match
  astropy's Planck18 to < 1e-4 (previously +0.09 % at z=1, +0.21 % at z=10 in D_L)
  (#2517). Every named cosmology (`PLANCK18`, `PLANCK15`, `WMAP5`) states its own
  published Tcmb0/Neff/m_nu explicitly (`WMAP5`: Tcmb0=2.725 K, Neff=3.04, massless
  neutrinos), matching astropy's Planck18/Planck15/WMAP5 to rtol 1e-6; `CosmoParams`'
  field defaults are radiation-free (Tcmb0=0.0), so a user-built `CosmoParams(Om0=...,
  w0=..., wa=..., h=...)` is unaffected by this fix unless it passes `Tcmb0` explicitly.
  D_L and age(z), and their gradients with respect to z, Om0, and h, are all finite in
  float32. `luminosity_distance_mpc` is exactly 0 at z = 0 (`distance_modulus` keeps
  the 10 pc convention at its own log10), and the age of the universe that the SFH
  age defaults derive from follows the same cosmology: 13.787 Gyr, where it was 13.81.
- Unknown-name errors recognize citation keys and name the registry entry they
  cite (#2429): when a user provides a citation key (e.g., `charlot_fall2000`)
  instead of a registry name (e.g., `power_law`), the error message now
  explains which registry entry it cites -- appended after the usual difflib
  "Did you mean...?" suggestion whenever the key names an entry from another
  group, and standing in for it when the key names an entry valid at that
  very site (where the difflib guess would only distract from the exact
  remedy). The hint can also never name a value no group's grammar would
  accept (an internal backend spelling, an author-name alias, an
  inference-backend name): it is filtered down to the union of every
  routed validator's accepted names first, falling back to plain difflib
  when nothing in that union survives. Applied to the 22 sites validated
  against a registry: dust laws (single- and two-component, foreground, and
  the AGN `atten` block's own law check), dust_emission, SFH, metallicity
  mode, dust_attenuation type, nebular, shock, IGM, radio, X-ray, AGN blocks
  (per sub-block type and the top-level `agn['type']` model selector), the
  generic per-group "unknown key" checker, and the top-level group-key list.
  Citation keys that only cite themselves (e.g. `cue`, `tengri`) are not
  reported as citation keys, since there is no other name to redirect to.

- The photoionized nebular backends floor the SSP age axis at 0.1 Myr, so an
  age-0 anchor template no longer turns every nebular output into NaN
  (#2418): the BC03 STELIB SSP carries `ssp_lg_age_gyr[0] = -inf`, and
  `CloudyGridBackend`, the CB19 backend and `MappingsPhotoStellarBackend`
  built their Q_H table's age axis as `ssp_lg_age_gyr + 9.0`, so
  `_interp_index_weight` (`components/nebular/_shared.py`) saw an infinitely
  wide first interval and returned NaN weights for the anchor and for every
  query inside that interval; the per-age sum then spread the NaN to every
  wavelength and every line, and `SEDModel.build(ssp_data=<BC03>,
  neb={"type": "cloudy"})` returned an all-NaN `sed_nebular` at all 7955
  master-grid nodes. `ssp_log_age_yr_axis` now applies the same 0.1 Myr floor
  that the stellar path applies to the anchor for surviving mass (#1016),
  read from one constant (`ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR`, `utils/ssp_anchor.py`) by
  both paths — bit-identical for every grid whose youngest
  template is already at least 0.1 Myr, which is every other shipped SSP —
  and the shared interpolator treats a non-finite axis node as a zero-width
  interval. The grid-family mismatch suspected in the issue was not the cause:
  the eight shipped Cloudy grids share one (log_U, log_age, log_Z) axis set.

- The accuracy bound of `age_kernel='dsps'` (roughly 1e-3 at the sharpest SFH
  shapes) is now stated on the discovery surface: the registry rows for each age
  kernel and the model configuration guide. A new advisory warns at build time when
  `field=True` silently forces the DSPS kernel over the user's default or
  explicit choice, so the coupling between the field path and the coarse kernel
  is no longer invisible. (#2368)

- WavePrecomp photometry now applies the nebular Lyman-continuum mask
  (`neb_fesc`) that the exact path applies below rest-frame 912 Å (#2439,
  #2427): `predict_via_precomp` summed the stellar photometric LUT with no
  such correction, so any band whose observed passband sampled rest λ < 912 Å
  carried the full, unabsorbed stellar Lyman continuum regardless of
  `neb_fesc`, K-invariant. Measured on the issue's model (Cue nebular,
  default `neb_fesc=0.0`, no dust): z=2 GALEX NUV +915 %, z=3 SDSS u +69 %;
  #2427's Inoue-IGM rows (z=0.8-3.0) are the same defect. The whole-band
  stellar LUT (`stellar_phot_lnu_precomp`) now carries an exact algebraic
  split of the SSP × filter integral at the 912 Å edge
  (`stellar_phot_lnu_precomp_lyc` and its per-age twin, zero-clamped against
  catastrophic cancellation in a fully-Lyman-continuum band), collapsing the
  residual to the pre-existing WavePrecomp floor.

  Fix round (2026-09): the K-node sub-band tensors are now matched to each
  consumer's OWN dense-path rule instead of one flat approximation shared by
  all three. `preintegrate_grid` forces an extra quadrature edge exactly at
  the physical 912 Å boundary whenever a live nebular mask is present, so no
  chunk straddles the break; `NebularSEDComponent` publishes a flat
  `stellar_subband_lyc_factor_precomp` factor (the only rule available
  without a birth-cloud concept — the dust-free mean-IGM branch's own case),
  and `two_component`, when it runs, overwrites the same key with its
  y(age)-graded `1-y(a)(1-fesc)` rule (or the flat rule under
  `lyc_absorb_all=True`) — never both, so there is exactly one factor per
  model. `SpectrumPrecomp` gets the identical fix as an exact per-pixel mask
  (a spectrum pixel is a single wavelength, so there is no partition to make
  approximate). Four conservation invariants are now asserted directly: the
  raw partition sums to the raw whole band; `neb_fesc=1.0` is bit-for-bit
  identical to no nebular component at all; the corrected sub-band sum
  matches the corrected whole band to ~1e-9 relative; and the per-age split
  sums over age to its whole-band twin. The on-disk z-table cache version was
  bumped (a warm cache built one day earlier would otherwise have satisfied
  an unversioned key and silently served a table with no Lyman-continuum
  split), and a missing `ssp_phot_lyc_table` key is now treated as a cache
  miss rather than trusted.

  Also fixed, found via a BASE-vs-HEAD zero-diff probe of this fix round's
  own changes: `lyc_mask_live` (the build-time gate above) tested only
  `isinstance(c, NebularSEDComponent)`, but `neb={'type': 'none'}` and
  `backend="shock"` both still put a `NebularSEDComponent` in the chain
  (`backend="baked_in"` for `'none'`) and both return from `apply` before
  ever reaching the `neb_fesc` masking block, so a model with either one was
  wrongly treated as "live": pure wasted compute for most filters, and for a
  very wide/red far-IR filter whose observed-frame footprint sits nowhere
  near the forced 912(1+z) edge, `subband_quadrature`'s own partition-
  conservation assertion could raise outright (measured: WISE W3 and
  Herschel PACS green/100um on the real `fsps_prsc_miles_chabrier.h5` grid,
  a `neb={'type': 'none'}` model that built and predicted cleanly before
  this fix round). The gate now also checks the nebular backend is one of
  the photoionized ones (`cue`, `cloudy_grid`, `cb19`, `mappings`).

  A real-grid residual remains after this fix (measured on the real
  `fsps_prsc_miles_chabrier.h5` + Cue grid, this round's own SFH: z=2 GALEX
  NUV ~10 %, z=3 SDSS u ~5 %, K-invariant) — this is the EXACT path's own
  SSP-grid-node quantization of the 912 Å edge (the dense mask cuts at
  whichever SSP wavelength node sits just below 912 Å, not at 912 Å itself,
  while this LUT's split is exact at the true physical edge), filed as #2447
  and not fixed this round; the magnitude is SFH- and filter-dependent.

  `tools/check_zero_hiding_clamps.py`'s pinned count moves 91 -> 92: the new
  Lyman-continuum twin's `num / jnp.maximum(denom, ...)` shares its
  denominator with the whole-band tensor's existing site and the same
  count/scale-floor classification — the filter integral of a loaded filter
  cannot vanish by construction.

- `conroy2010` is CCM89 with a scalable 2175 Å bump (`dust_bump_strength`), as in
  Conroy et al. (2010) and FSPS `dust_type=1`; it was a sigmoid Cardelli/power-law
  blend that over-attenuated the NIR by up to 3.3× and had no bump control.
  `dust_slope` is no longer a parameter of this law. The curve carries the FSPS
  continuity term on the near-UV segment (3.3 ≤ x < 5.9 μm⁻¹), so it is continuous
  at x = 3.3 μm⁻¹ for any bump strength, and at bump strength 1 it differs from
  `cardelli` by that term (up to 2×10⁻⁴). In the far-UV it follows FSPS as well:
  the CCM89 cubic is evaluated to x = 12 μm⁻¹ (833 Å) and held constant beyond,
  while `cardelli` holds it constant from x = 10 μm⁻¹ (1000 Å), the limit of the
  range Cardelli et al. (1989) fitted. The two agree at 1000 Å and differ by a
  factor 1.83 at and below 833 Å for R_V = 3.1 (1.92 for R_V = 2, 1.57 for
  R_V = 5) (#2522).
- `reddy15` is continuous at 0.6 µm (red-branch offset −0.0362, as in FSPS
  `dust_type=6`) and constant below 1500 Å (#2523).
- The li08 (c1–c4), noll09 and salim_sbl18 (UV-bump center and width) and tea (scatter) attenuation-law parameters are now declared and reachable through the grammar, and two_component forwards every law parameter on all screens (bc/diff previously dropped them silently) (#2542). **Breaking**: `li08`'s default is now the Li et al. (2008) Milky-Way (R_V=3.1) curve (c1..c4 = 14.4, 6.52, 2.04, 0.0519); the previous default and docstring presets did not correspond to the paper.
- Shock line ratios are normalized over the populated grid cells, so
  `Hb_4861A` is 1.0 again (#2435): `shock_line_ratios` is documented to return
  ratios relative to Hbeta, but `components/nebular/shock.py` zeroed the
  unpopulated MAPPINGS cells and `utils/grid_interp._tensor_contract` then
  contracted with triweight weights that still summed to one over the whole
  axis — so every line came back scaled by the fraction of kernel weight
  landing on populated cells. With 3992 of 38850 cells (10.3%) populated that
  fraction ran from 0.88 down to 0.0037, and was exactly 0 at
  B = 100 uG / log n = -1, where every line was silently zero. `Hb_4861A`
  read 0.7135 instead of 1.0 and Halpha/Hbeta read 2.14, below the Case B
  recombination floor of 2.86. `_tensor_contract` now performs a normalized
  convolution (the mask is contracted with the same weights and divides the
  result) and returns NaN where no populated cell is in range; the unmasked
  path, which is the only one `components/agn/disc.py` uses, is unchanged.
  Fitted SEDs were never affected — `_shock_line_arrays` anchors on Halpha, so
  the common factor canceled — and this does not fix #2066.

- 6 broad skip handlers narrowed to specific exceptions; the skip-handler ratchet is now empty (#1615). The handlers had been hiding failures #2464 and #2465.

- JAX 0.11.2's cache-write path no longer raises on an orphan-atime entry
  (#2416): the #1661 regression test's reproduction arm, which pinned JAX's
  cache-write failure on orphaned -atime files, became vacuous on JAX 0.11.2
  (released 2026-09-17). Upstream tolerated the condition instead of raising
  `FileNotFoundError`, so the test now asserts the write outcome (success and
  entry readable) under both JAX 0.11.1 (pre-fix) and 0.11.2+ (post-fix),
  gated on `packaging.version` comparison. The orphan-atime repair stays
  load-bearing on JAX < 0.11.2 and remains useful for recovery on all versions.

- Every SFH family's star-formation support is bounded to `[0, age(z)]`,
  forming the requested mass on both age kernels and for additive
  composites (pooled total; a uniform rescale, so per-member truncation is
  still approximate). `exp`/`dexp` get a genuine `[0, start]` window
  (`start` now reads as the lookback time of formation, not of an
  unbounded-into-the-past peak). **Breaking**: `sfh_exp_start_gyr`/
  `sfh_dexp_start_gyr` default to `Fixed(5.0)` (floor `0.5`), not
  `Fixed(0.0)` (floor `0.0`) -- under the corrected `[0, start]` window a
  zero onset is a zero-width window with no stars at all (ill-posed;
  #1031's own fixture divided a vanishing mass by a vanishing width, 0/0).
  `psb_wild2020`'s burst is anchored to the cosmic age at z (`age_at_z(z)`,
  following Wild et al. 2020 Eq. 5), not its own `age` parameter (the OLD
  component's independent formation epoch) and not a hardcoded module
  constant; the burst carries no lookback window on its support, filling
  the whole bounded range rather than being cut off at `burstage`. A free
  `redshift`'s
  upper end can still admit an onset draw before the Big Bang; `SEDModel
  .build` now warns (`FreeRedshiftOnsetCeilingWarning`) instead of
  truncating unremarked. `continuity`/`dirichlet`'s default bin ladder is
  built from the source redshift (Prospector-beta scheme) instead of fixed
  at 0-13.7 Gyr; a free redshift warns too
  (`NonparametricBinEdgesAtRedshiftCeilingWarning`). Moving-boundary
  staircases (#2476) are fixed for `psb_suess2022`/`psb_flex`'s bin edges;
  `delayed_bq`'s burst/quench switch, `periodic`'s burst onset, spacing and
  width, `periodic`'s rectangular type, `tsnorm`'s SSP-grid aliasing, and
  `psb_suess2022`/`psb_flex`'s `tlast_gyr` on the `dsps` age kernel (an
  exactly flat direction: both the finite difference and the analytic
  gradient are zero) remain measured staircases on at least one age kernel
  (#2521, #2457, #2476) -- a partial-cell quadrature narrow enough to pass
  every existing CIGALE parity test for these two families was not found
  this round, so their moving-boundary integration stays the CIGALE-matching
  hard edge.
- A catalog fit's per-galaxy `redshift_col` can free a z-capped onset
  parameter (e.g. `sfh_dpl_age_gyr`) whose ceiling was narrowed to the age
  of the universe at the model's single placeholder redshift, not the
  catalog's actual per-galaxy range. `Catalog` now re-narrows that ceiling
  to the age of the universe at the catalog's lowest redshift -- the
  widest single ceiling valid for every galaxy in it, since the
  mass-conserving truncation (#2521) still forms each galaxy's declared
  mass inside its own `[0, age(z_i)]` at fit time regardless of the prior's
  width -- and warns (`FreeRedshiftOnsetCeilingWarning`) naming the
  catalog's z range instead of refusing the fit outright.
- Composable AGN torus no longer collapses at the 1 mm node (#1512): the
  composable disc+skirtor path with cigale_joint normalization computes an
  inclination-attenuation ratio `disk(i)/disk(0)` by resampling the SKIRTOR
  template grid (136 nodes, last at 1e8 Å) onto the model grid. The SKIRTOR
  disk template is zeroed beyond node 130 (8.71e6 Å); when resampling reached
  the boundary, `right=0.0` fill zeroed the inclination ratio, causing the
  reweighted disc (`L_lambda_disc * incl_n`) to collapse to zero despite the
  smooth torus. The ratio `sed_agn[1e7]/sed_agn[prev]` fell from 1.009 (smooth)
  to 3e-6 (catastrophic) for powerlaw disc, and showed a 60% step for qsogen
  disc. Fixed by computing the last finite inclination ratio at the template
  edge (0.793341 for cos_inc=0.866, wavelength-independent inside template)
  and carrying it smoothly beyond the boundary at both the fill value (line 1043
  `incl_n`) and the resampling `right=` parameter (line 1104). Affected
  configurations: any composable disc under cigale_joint with fracAGN set.
  Verified on powerlaw (measured: 1.0093 ratio, <0.5% error) and qsogen
  (measured: 1.0017 ratio, <1.0% error).

- `Spectroscopy` now validates `wave_obs` at construction time, refusing grids
  that are non-finite (NaN/inf), non-positive, or non-increasing, with
  descending grids raising a hint to reverse them alongside the flux and error
  arrays. `calibration_order` is also checked to be non-negative. When
  `wave_obs_segment_sizes` is set (by the DESI loader for multi-camera spectra),
  monotonicity is enforced per camera segment, allowing overlaps at seams
  where adjacent cameras meet. Segment sizes must be positive and sum to the
  wavelength grid length. (#2172)

- Emission line mode validation unified: `Spectroscopy` and `NebularConfig`
  now share a single legal set of modes (`ELINE_MODES`), defined in
  `observation.constants`. The retired `"fixed"` mode is rejected with a hint
  to use `"off"` or one of the analysis modes (`"marginalized"` / `"fitted"`).
  The settings validator now accepts `"fitted"` (previously omitted), and the
  dead probe read of `_spectroscopy_config` in `Fitter._init_emission_lines`
  has been removed. (#2191)

- The default Gaussian LSF path now subtracts the loaded SSP library's own
  per-wavelength resolution instead of a flat scalar, and stops applying
  that subtraction and the galaxy's velocity dispersion to nebular/shock
  emission lines, which keep only the instrument LSF, matching Prospector's
  convention (#2518, #2519). `SSPData` gains `ssp_resolution_kms` from
  FSPS's own per-node tables (MILES σ_v ≈ 92→43 km/s; C3K σ_v ≈ 42.4 km/s;
  `SSP_LIBRARY_RESOLUTIONS["c3k"]` corrected 15.0 → 42.4 km/s), and warns
  instead of silently clamping wherever the instrument is sharper than the
  library. Every path that predicts a spectrum from the model applies the same
  kernel rule through one function.
- Unknown key validation now precedes grid-file resolution for CLOUDY nebular
  configuration (#2328): when `neb={'type': 'cloudy'}` with no 'grid' key is
  supplied and no CLOUDY grid is on disk, a typo in the group was silently
  misreported as "missing grid file", because the throwaway enumeration spec
  that key validation derives its parameter census from hit the grid check
  first. The enumeration spec now defers resource-path resolution (private
  `_defer_resource_paths` flag), so unknown keys are reported first. The real
  Parameters construction is untouched — a valid group without an on-disk grid
  still raises the same grid message.

- `measure_line_fluxes` now refuses (`ValueError`, naming the grid and the
  remedy) when the configured nebular backend contributes no line flux of its
  own (`neb={'type': 'ssp'}` or the default) and the SSP grid's metadata says
  it carries no nebular emission (`nebular='bare'`); it warns instead when the
  grid's status is merely unstamped (`'unknown'`), since that cannot be
  resolved either way. Previously it silently measured stellar
  continuum/absorption in that configuration and returned a value
  indistinguishable from `neb={'type': 'none'}`, with no indication the
  nebular emission it is meant to measure was absent (#2540).
- `predict_line_fluxes` no longer raises `ValueError: attempt to get argmin of
  an empty sequence` when the attached per-Q_H nebular grid tabulates no lines
  (a photometry-only fit's `approx="auto"` grid, or `FeaturePrecomp()` with no
  line targets): it now falls through to the exact catalog path instead,
  matching the model's own `predict_line_fluxes(approx=None)` answer. A grid
  that tabulates some but not the requested line still refuses with the
  existing "no match within tolerance_aa" message (#2561).

- Self-whitening backends (MCLMC and low-rank HMC) now refuse to compose with
  the analytic metric when `precondition=` is supplied (#2196). Two whitenings
  multiply to produce catastrophic degradation (measured as 472 divergences on a
  stochastic-field posterior where either alone gave 0–19). Backends that learn a
  metric from warmup (via `diagonal_preconditioning=True` or
  `blackjax.window_adaptation_low_rank`) now declare `self_whitening=True` and
  raise `ValueError` before sampling when both conditions hold, rather than
  silently degrading. The analytic metric and a backend's own whitening cannot be
  composed; choose one or the other.

- Float32 refusal on Hessian-based inference now names the dtype in both
  Laplace and preconditioning routes, clarifying that non-finiteness is a
  float32 artifact (the SED model's photometry Hessian is all-NaN in float32)
  rather than a diverged MAP initialization or genuine curvature failure. The
  disable-profile-mass-then-raise anti-pattern at the float32 check no longer
  silently mutates a `Fitter` on an exception path, preserving the invariant
  that reuse-after-exception is safe (#2378).

- Bare `import tengri` now succeeds on float64-less backends (jax-mps, MLX) by
  deferring DSPS module imports until first use (#2276, #2271): DSPS modules
  allocate float64 device buffers at module import time, causing a hard failure
  on backends lacking float64 support. Two mechanisms defer these imports: (1)
  all `from dsps.*` statements in `utils/cosmology.py` are now function-local,
  deferred until the function is first called; (2) the age-of-universe constant
  (`_AGE_UNIV_GYR`) used in the SFH registry is computed at module scope via a
  pure-numpy implementation (`age_at_z0_host()`) that mirrors DSPS's 512-node
  trapezoidal integration without importing JAX or DSPS. The registry default
  is now a cached constant, not a deferred function. The first use of a DSPS path
  still fails loudly on float64-less backends (the expected behavior for
  unsupported operations), and no compatibility is altered for code paths that
  do use DSPS.

- Data locator hermeticity (#2329): a nested worktree's test run found untracked
  data (CLOUDY grids, Cue weights) in the main checkout via the locator's
  ancestor-directory walk, so suites passed locally and failed in CI. A new
  `TENGRI_DATA_NO_ANCESTOR_WALK` env var pins `data_dirs()` to `$TENGRI_DATA_DIR`
  plus the repository under test, and `tests/conftest.py` sets it unconditionally
  (the precomp-cache precedent), so pytest always sees what CI sees. The
  measured flip-list was six tests: two hand-rolled parent-walking data probes
  (`requires_cloudy`, `requires_cue`) now route through the canonical locator so
  they skip under the pin exactly where CI skips instead of running into a build
  that cannot see the grid, and the two locator-contract tests (#1209, #1431)
  lift the pin explicitly to keep testing the default walk, whose pinned side is
  owned by `tests/unit/test_data_locator_pin.py`. Outside pytest nothing changes
  unless the env var is set (see `tests/TESTING.md`).

- LUT-bias advisory probes z-table midpoints on a free-redshift fit (#2105):
  the advisory probed one redshift and so could not see the LUT's z-interpolation
  error; it now probes the inter-node midpoints nearest the prior median (or the
  prior bounds when no node lies inside) and names the peak redshift and
  `approx=None`, the exact path on every surface since #2385. (#2105)
- Bare `import tengri` now succeeds on float64-less backends (jax-mps, MLX) by
  deferring DSPS module imports until first use (#2276, #2271): DSPS modules
  allocate float64 device buffers at module import time, causing a hard failure
  on backends lacking float64 support. Two mechanisms defer these imports: (1)
  all `from dsps.*` statements in `utils/cosmology.py` are now function-local,
  deferred until the function is first called; (2) the age-of-universe constant
  used in the SFH registry parameter definitions is now cached on first use via
  `@functools.cache(_age_univ_gyr())` instead of module-scope evaluation. The
  first use of a DSPS path still fails loudly on float64-less backends (the
  expected behavior for unsupported operations), and no compatibility is altered
  for code paths that do use DSPS.

- `salim_sbl18` UV bump normalization (#2397): the UV bump term now normalizes
  by the δ-dependent R_V,mod of Salim, Boquien & Lee (2018) Eq. 4 instead of
  the fixed Calzetti R_V = 4.05, implementing the paper's own correction over
  the pre-v0.12 CIGALE bug described in footnote 7. Both `dust_bump_strength`
  and `dust_delta` default to `Fixed(0.0)`, so this is a no-op unless both are
  explicitly set/freed on `salim_sbl18` specifically; `kriek_conroy` and `noll09`
  are unaffected (neither claims Eq. 4; both correctly use fixed R_V per their
  own papers). Any prior-declared or pinned `dust_bump_strength` on `salim_sbl18`
  beside a nonzero `dust_delta` now maps to a different UV bump amplitude; at
  δ = +0.3 the bump-to-base ratio at 2175 Å shifts from 0.461 to 0.311 (factor
  R_V,Cal / R_V,mod ≈ 0.673), and A(2175) / A_V at (δ = 0.3, B = 3) from 2.317
  to 2.079, matching an independent BAGPIPES evaluation (2.082) to < 0.5%
  precision versus ~10% prior miss.

- Student-t and Gaussian noise energies now share one convention: both return
  the negative log-density up to a parameter-independent constant, with the
  Student-t branch tending to the Gaussian branch as dof → ∞. A fit at fixed
  dof is unchanged; comparing evidence across noise families no longer carries
  an offset of n·½·log(2π) (#2560).

- Nebular backends' parameter reach. `neb_logU` (Cloudy, CB19 and both
  MAPPINGS backends) and `neb_logZ_gas` (Cloudy) are now narrowed to the
  vendored grid's axis at build time and warn when a value cannot be
  narrowed away from it, instead of silently clipping onto a dead edge node
  with an exactly-zero gradient (#2460). `CueBackend`'s low-level path
  (`ssp_weights=None`) now threads `neb_logZ_gas` into `gas_logz` instead of
  silently forcing solar metallicity (#2437). `neb={'type': 'cloudy'}` with
  no explicit `grid` now prefers the packaged grid whose isochrone matches
  the SSP, warning when it falls back to a mismatched sole grid and
  refusing to guess among several mismatched ones; the resolved path is
  logged at INFO (#2426). Cue's reproduction markdown, READMEs, validation scripts and
  `cue.py` cite Cloudy 22.00 (Li et al. 2025) instead of "c17+"; three
  code-cell labels and the rendered `docs/reproduction` copies refresh with the
  next executing re-render (part of #2555).

- The Lyman-continuum energy HII-region dust absorbs now enters the dust IR
  budget (`L_absorbed`), as in CIGALE
  (`dust.luminosity = (lum_ly_young + lum_ly_old) * fdust`,
  `pcigale/sed_modules/nebular.py:191-193`), for the population the nebular
  component reprocesses: the whole stellar SED for `single_component`, `wg00`,
  and `two_component` with `lyc_reprocessed_by='all'`, and only the young/birth-cloud
  population for `two_component` with `lyc_reprocessed_by='young'` (the population its
  own `neb_fesc`/dust screen actually applies to). Previously this energy
  only suppressed nebular emission and vanished from the energy balance. The
  combine (`energy_balance.log10_add_fdust_credit`, a fused `log1p` form) is
  bit-identical to the pre-credit value at `f_dust == 0` and has a FINITE,
  NONZERO gradient there too (`L_LyC / (L_absorbed * ln 10)`), since
  `L_absorbed` is exactly linear in `f_dust`; a first version log-added an
  already `fdust`-multiplied credit term whose own double-where derivative was
  deliberately zero at the boundary, which zeroed the combined gradient as
  well. Also threads the `lyc_in_energy_balance` (FSPS/Prospector-parity) toggle to
  `wg00` (`dust_type=3`), which the grammar already accepted but
  `component_factory.py` silently dropped (#2539). A sibling defect in the same
  budget is fixed alongside it: `two_component`'s own `lyc_in_energy_balance=True`
  screen-absorption integral for `lyc_reprocessed_by='young'` now reads the same
  per-age, fesc-aware population `sed_attenuated` itself attenuates rather
  than a uniform all-ages bookkeeping value, in both the exact and WavePrecomp
  LUT paths, bit-identical at the `lyc_in_energy_balance=False` default;
  `single_component` and `wg00` already integrated the same SED they
  attenuate. The WavePrecomp energy-balance LUT is now EXACT in a live
  `neb_fesc` rather than declining to the exact integral whenever
  `lyc_in_energy_balance=True` met a live photoionized nebular component: the
  stellar absorbed integral is affine in `fesc`
  (`A(fesc) = A_0 + fesc * A_1`, `A_1` the young/birth-cloud-weighted -- or,
  under `lyc_reprocessed_by='all'`, unweighted -- Lyman-continuum-only term), so
  `build_energy_balance_lut` now bakes both the `fesc`-independent `A_0`
  family and the `fesc`-linear `A_1` family, and
  `lut_l_absorbed_stellar_log10` combines them with the runtime `fesc` at
  evaluation time -- an exact, O(1) linear combine (no interpolation, no
  approximation), not a fallback.

  **One Lyman edge (#2447)**: every Lyman-continuum consumer -- the nebular
  LyC mask, the dust energy-balance mask and its WavePrecomp LUT, the
  HII-dust credit, the Q_H integral and the photometric filter integral --
  now steps at the physical hydrogen limit, `LYMAN_LIMIT_AA = 911.76` A,
  through one module, `tengri.components.lyc` (`ionizing_mask`,
  `edge_trapezoid`, `edge_interp`, `lyc_shares`, `log10_lyc_luminosity`).
  Previously the exact path's mask stepped at whichever SSP node sits just
  below 912 A (MIST/C3K brackets the edge at 911.5716/913.3967 A) and each
  consumer placed the edge its own way, biasing a band whose rest-frame
  coverage straddles the edge by up to a few percent at `neb_fesc < 1`
  (-2.0651% MIST/C3K, +11.3813% MILES on GALEX NUV at z=2). In the bracketing
  cell the ionizing side holds the bluer node's value up to the edge and the
  non-ionizing side the redder node's, so the two pieces partition the
  integral exactly and a per-node mask is exact under this model; the
  per-node `lyman_edge_transmission` reweighting is retired. The filter
  integral inserts the observed-frame edge `LYMAN_LIMIT_AA * (1 + z)` as a
  node pair, closing the band residual to round-off. Q_H moves by about
  1e-5 dex, and the ionizing-spectrum cache version is bumped so cached
  tables rebuild. The stellar component publishes per-age ionizing
  luminosities (`log_L_lyc_age`) from the ionizing slice of the SSP grid,
  so `two_component`'s young-only credit integrates once rather than once
  per age, and the credit is formed only when a photoionized nebular
  backend publishes its HII-dust share and `neb_fdust_frac` is not
  `Fixed(0.0)`.

  **Breaking: one `lyc_` key family on `dust_attenuation`**: the structural
  keys deciding what happens to ionizing photons are renamed to read as one
  family. `lyc_absorb_all` (bool) becomes `lyc_reprocessed_by`, `'young'`
  (default) or `'all'` (`False` -> `'young'`, `True` -> `'all'`);
  `eb_include_lyc` becomes `lyc_in_energy_balance` (same bool, default
  `False`). Writing an old key raises naming the new key and the value
  mapping, as do the flat `Parameters` kwargs `dust_lyc_absorb_all` and
  `dust_eb_include_lyc`. `lyman_cutoff` (the attenuation-curve clip) is not
  part of the family and is unchanged.

  **Breaking (#2436)**: `neb_fdust`, the absolute HII-region
  dust-absorption fraction, is retired: declaring it independently of
  `neb_fesc` let `neb_fesc + neb_fdust` exceed 1, an impossible >100% of the
  ionizing-photon budget that only `lyc_dust_escape_factor`'s internal clamp
  caught. `neb_fdust_frac` (default `Fixed(0.0)`, same `Uniform(0, 1)` prior
  range) replaces it: the fraction of the NON-escaping budget
  (`1 - neb_fesc`) HII-region dust absorbs, so the three per-photon shares
  (escape, HII-region dust, photoionization) sum to exactly 1 for any
  `(neb_fesc, neb_fdust_frac)` in `[0, 1]^2` -- the whole prior box is
  physical, with no clamp needed downstream. The one absolute-share helper,
  `lyc_shares(neb_fesc, neb_fdust_frac) -> (f_esc, f_dust, f_gas)`
  (`components/nebular/_recombination_coeffs.py`), is now the single place
  every consumer of the absolute `f_dust`/`f_gas` shares reads through:
  `lyc_dust_escape_factor`'s callers in Cue, CloudyGrid and CB19, and the
  #2539 HII-dust LyC credit above. Convert an old absolute value with
  `neb_fdust_frac = neb_fdust / (1 - neb_fesc)`; writing the retired
  `neb_fdust` anywhere in the `neb` group now raises naming the replacement
  and the conversion formula.

  **Added (#2529): age-selective LyC escape geometry.** A fourth
  `dust_attenuation` structural key joins the `lyc_` family,
  `lyc_escape_geometry`: whether the escaping fraction (`neb_fesc`)
  bypasses the birth-cloud screen through a geometric hole, instead of
  only skipping nebular reprocessing (the pre-#2529 behavior, still the
  default). `'screened'` (default, unchanged): `neb_fesc` never touches
  the dust screen. `'birth_cloud_holes'`: a covering fraction `neb_fesc`
  of the young population's light bypasses the birth-cloud screen but
  still crosses the diffuse ISM (FSPS `frac_obrun`-like). `'clear'`: that
  same fraction sees no dust at all (Synthesizer `fesc`-like). Applies at
  every wavelength, not only below the Lyman limit -- a hole in a birth
  cloud is geometric, not wavelength-selective. One shared formula,
  `tengri.components.lyc.escape_geometry_transmission`, used by both the
  exact path and the `WavePrecomp()` energy-balance LUT and photometry
  LUT (both agree with the exact path to the existing quadrature
  tolerance). Two-component only (a birth-cloud screen distinct from the
  diffuse screen is the one thing a hole needs to be in); refused
  together with `lyc_reprocessed_by='all'` (both would reduce the young
  population's escaping light from the same `neb_fesc`, double-counting
  it).

### Fixed

- SKIRTOR grid caches are keyed on the process float dtype, so a float32
  forward no longer perturbs a later float64 forward of the same model
  (2.5e-9 measured shift) (#2275).

- Test `test_the_threaded_values_actually_reach_the_backend` now owns its CB19 grid instead of relying on whatever the locator finds, ensuring hermetic test isolation (#2318).

- Flat-form `lgmet_scatter` kwarg is now LIVE in predictions (was dead): on
  flat-form builds (e.g. `Parameters(lgmet_scatter=0.3)`), the kwarg now sets
  the registered `met_logzsol_scatter` Fixed value at the parameter registry
  seam, so the stellar component receives the instance-specific scatter width
  instead of always falling back to the global default (issue #2255). Grammar
  builds (with explicit `met_logzsol_scatter` as parameter name) remain
  unchanged; both spellings cannot be passed together (raises if shadowing is
  detected).

### Changed

- `WavePrecomp(igm_fold=...)` now defaults to `"auto"`: the exact IGM fold wherever it can be built (fixed and free redshift), the node fold only where the transmission carries free parameters (patchy reionization, DLAs). The node fold was off `approx=None` by 85-107 % at z = 7 in bands straddling Ly-alpha and ~10 % in GALEX FUV at z = 1.5; the exact fold is ~1e-14 at fixed z and 0.42 % on a free z grid of 32 nodes. WavePrecomp photometry of any band the IGM reaches moves; pass `igm_fold="node"` for the old behavior (#2445).
- `agn_attenuation_ebv` is retired; every AGN attenuation block (`smc_prevot`,
  `qsogen`) reads `agn_ebv`, the precompute-axis name; the retired spelling —
  flat or under `agn={'atten': {...}}` — is refused with a rename hint (#2325).
### Fixed

- Release version now has a single source: `pyproject.toml`. `src/tengri/__init__.py`
  derives `__version__` via `importlib.metadata`, with a fallback for source-tree
  installs: in an uninstalled checkout, the version is read from `pyproject.toml`
  at `tengri._data_setup.source_tree_root()`, the package's one sanctioned anchor
  for reading repository-relative files (preserving environment isolation; #1431, #2103).
  `docs/conf.py` derives `release` from the imported `tengri.__version__`.
  `CITATION.cff` remains a manual copy, but `tools/check_version_single_source.py`
  (wired to the `lint` job) ensures it never drifts from `pyproject.toml`. Removes
  the inert `setuptools-scm` requirement and the false assertion in `publish.yml`
  that full history is needed (#2103).

- `mcmc_nuts_fast` registry row and pmap hint: removed false claim "pmapped chains
  by default", now states "vmapped chains on one device; pmapped when the platform
  exposes at least n_chains devices". The TENGRI_HOST_DEVICES hint is now gated
  to GPU/TPU platforms only; on CPU, vmap is 8% faster than pmap and the hint was
  counterproductive (418.3 s vmap vs 451.5 s pmap, same 5-param broadband fit).
  Removed "20 s" from short_doc: timing varies widely by model (15 s–30+ min),
  not a property of the recipe alone. Error message for `chain_parallel='pmap'`
  is now platform-aware: suggests TENGRI_HOST_DEVICES on GPU/TPU, recommends
  vmap/auto on CPU (#2361).

### Changed

- **X-ray absorption nomenclature**: `tbabs_transmission()` renamed to
  `wabs_transmission()` to accurately reflect the Morrison & McCammon (1983)
  cross-section convention it implements, not Wilms et al. (2000) tbabs. The
  function behavior is unchanged; `tbabs_transmission` remains available as a
  deprecated alias. The soft-band difference between wabs and tbabs (10–30%
  below ~1 keV) is now documented in the docstring. (#901)

### Fixed

- `Posterior.to_param_spec()` now reads the posterior's own SFH type instead of
  defaulting to `dpl`, and preserves all structural settings (dust law, nebular
  backend, etc.) from the model. Previously, posteriors fit with non-`dpl` SFH
  types like `delayed` would raise "Unknown parameter" errors because the method
  only copied `stochastic` and `n_grid`, losing the component type information.
  The fix reconstructs the full nested-dict groups from the model spec, then
  injects the posterior's empirical parameter distributions. (#2180)

- `Posterior.validate()` now dispatches on the MCMC method's valid arguments
  instead of always passing `n_steps` (a MAP-only parameter) to `mcmc_nuts` and
  `mcmc_raytrace`, which do not accept it. After a MAP fit, `validate()` now
  raises a helpful error explaining that validation requires a posterior with
  samples, rather than failing with a cryptic `TypeError` about `mcmc_nuts`. (#2180)

### Added

- `WavePrecomp(igm_fold="auto")`, a third IGM fold that takes the exact
  bandpass integral wherever it can be built and the node fold everywhere
  else. The existing `"exact"` fold carries transmission inside the bandpass
  integral instead of forming <S><T> at the quadrature node, which matters
  where a Lyman break falls inside a band (measured ~10% on GALEX FUV at
  z = 1.5, ~83% at z = 3); but it raises for a free redshift and for a
  transmission carrying free parameters (patchy reionization, DLAs), so it
  could never be proposed as the default while naming it was the only way to
  ask for it. `"auto"` asks for it and falls back without raising. The default
  is unchanged at `"node"`, and an explicit `"exact"` still raises where it
  cannot be served: a mode named by the caller is never silently downgraded.
  The refusal and the fall-back read one predicate, so the two lists cannot
  drift; `tests/contract/test_igm_exact_fold.py` pins that, and that `"auto"`
  also falls back where the exact fold would do nothing at all (no templates,
  no filters) while the node fold still applies one.
- Per-screen dust law shape keys accept `Fixed`/priors and become declared
  `dust_<shape>_<screen>` parameters; a plain number keeps the build-time
  path; the flat `dust_law_overrides` dict refuses a prior with the named
  remedy (#2428). `FREE` frees the key on its declared range, and
  `Fixed(DEFAULT)` pins it at the same registry default the plain-number
  spelling of that default already predicts bit-identically to. The 12 names
  are two-component only (`single_component`/`wg00` never carry them) and
  wildcard-inert (`all_params: FREE` never frees one; name it explicitly).
  Every two-component spec now carries 12 more declared (Fixed-by-default)
  parameters than before, which changes `cache_key()` for such specs (a
  single_component spec's `cache_key()` is untouched). `compile_signature()`
  changes for EVERY model, single_component included: it embeds the
  registry-wide name -> law-kwarg table, which now carries a row per
  per-screen name. Both are one-time cache invalidations on upgrade, not a
  behavior change to any existing prediction. A no-`all_params`-disposition
  `dust_attenuation` build (`DefaultFixedParametersWarning`) now lists 17
  parameters instead of 5 on a two-component spec, the same 12 names added.

- SkyMapper Southern Survey filters: `skymapper_u`, `skymapper_v`,
  `skymapper_g`, `skymapper_r`, `skymapper_i`, `skymapper_z`, with their
  curves tracked under `data/filters/` so they load offline like the rest of
  the registry (425 -> 431 aliases). SkyMapper observes in **six** bands,
  `uvgriz`, not the `ugriz` five that `sdss_*` / `lsst_*` / `ps1_*` establish
  as the shape of an optical survey pack; the extra `v` is a violet band at
  3838 A between `u` and `g`. Note that band letter: the registry already
  holds `johnson_v` and `xmm_v`, both Johnson V near 5500 A, so `*_v` now
  means two different bands depending on the prefix. The alias still takes
  its letter from the SVO identifier, as every other alias does, and
  `tests/components/observation/test_skymapper_filters.py` pins the
  wavelength rather than relying on the name — including the `u`/`v` pair,
  whose nominal pivots are 9.43% apart and whose bandpasses genuinely
  overlap, which is what caps the pack's pivot tolerance at 1%. All six
  curves agree with the published effective wavelengths (Wolf et al. 2018)
  to within 0.21%.

- Each non-stellar emission source now picks its own dust screen: the
  `dust_attenuation` group gains `nebular_screen` (governs the nebular
  continuum, the line catalog, and the fast-nebular fallback grid; default
  `"birth_cloud"`), `shock_screen` (governs the MAPPINGS V shock SED; default
  `"diffuse"`), and `agn_screen` (default `"none"` — the AGN component runs
  after dust and carries its own polar-dust screen). Each accepts
  `"birth_cloud"`, `"diffuse"`, `"none"`, or the synonym `"off"`; flat
  spellings `dust_nebular_screen` / `dust_shock_screen` / `dust_agn_screen`
  mirror `dust_law_bc` / `dust_law_neb`. One validator
  (`tengri.parameters._dust_keys.resolve_screen_choices`) backs both
  surfaces: an unknown value names the three choices;
  `single_component` dust refuses any value other than `none`/`off` or the
  source's own default (a single screen has no birth-cloud/diffuse
  distinction); `wg00`/`off` dust refuse the keys outright, like the other
  two-screen-only keys. Every attenuation site (the nebular continuum, the
  discrete line catalog, and the shock SED) routes through ONE helper,
  `DustSEDComponent._screen_transmission`, so there is exactly one
  implementation of the screen formula. Closes #2234 by replacement: a
  configurable nebular screen (`neb_dust` modes `bc`/`diff`/`neb`/`none`)
  existed in 2026-04 and died with #923/#2230, leaving `_neb_dust_mode` /
  `_neb_dust_law_bc_fn` in `sed_model.py` as write-only remains (now
  deleted); the choice is restored as explicit, validated config rather than
  the old ungated mode string. Also closes #2235 by making the
  `docs/model_reference/nebular.md` prose ("shock receives only diffuse ISM
  attenuation") and the code agree — the code previously attenuated shock
  unconditionally with the birth-cloud form — and by snapping (see Fixed,
  below).

- `agn_screen` is functional: `birth_cloud` / `diffuse` put AGN light
  through the galaxy's dust screens with the absorbed power joining the
  energy balance; refused with `agn={'norm': 'cigale_joint'}`, which reads
  that balance.

- `neb_hbfrac` (CB_19's HbFrac axis, matter- vs radiation-bounded escape
  proxy) now declares `free_prior=Uniform(0.0, 1.0, default=1.0)`, so
  `neb={'type': 'cb19', 'all_params': FREE}` and `neb={'type': 'cb19',
  'hbfrac': FREE}` both free it instead of leaving it silently pinned.
  Dropped from `tools/check_param_free_priors.py`'s `REFUSED` ledger
  (freeable 90 -> 91, pinned 25 -> 24; it left the `inert` ground). The
  shipped `data/cb19_templates.h5` carries no real variation along this
  axis (measured: its two HbFrac nodes are bit-identical), the same
  placeholder gap #2181 already found for `neb_log_nH` / `neb_co` /
  `neb_dno` (#2198 tracks the pending 3MdB erratum) -- `neb_hbfrac` joins
  those three: declared and freeable, refused loudly by
  `check_cb19_free_params` against the current shipped grid rather than
  silently inert (#2213).

- `radio={'sf': {'type': ..., 'freefree': False}}` turns the star-forming
  block's Murphy+2011 thermal free-free term off (synchrotron only — what
  pcigale's `radio` module emits); omitted, it stays on except for
  `bell2003_split`, which carries its own thermal fraction. Reaches
  `RadioSEDComponentConfig.include_freefree`; documented in
  `docs/model_reference/xray_radio.md`.

- `run_nuts`/`run_dynamic_hmc` (and, via the same `_vmap_chains` seam,
  `mcmc_hmc`'s existing `chain_method="parallel"`) accept
  `chain_parallel: {"auto", "vmap", "pmap"}`, default `"auto"`. `"pmap"` maps
  `n_chains` chains one-per-device via `jax.pmap` instead of SIMD-batching
  them onto one device with `jax.vmap`, and raises `ValueError` (naming
  `TENGRI_HOST_DEVICES`) if fewer than `n_chains` devices are visible;
  `"auto"` picks `"pmap"` when enough devices of the platform in use are
  visible and `n_chains > 1`, else falls back to `"vmap"`. Warmup stays
  single-chain either way; per-chain adaptation was measured worse (inflated
  R-hat from per-chain metrics) and is not offered. Measured on `ctl-dpl`
  (D=8, 4 chains) with 4 forced host CPU devices: the sampling phase drops
  19.5s -> 3.5s. The resolved choice is recorded in
  `posterior.diagnostics["chain_parallel"]`. New env hook
  `TENGRI_HOST_DEVICES=<n>` (read by `tengri/__init__.py` before the first
  `import jax`) appends `--xla_force_host_platform_device_count=<n>` to
  `XLA_FLAGS` so CPU users can get extra JAX devices without knowing the XLA
  flag spelling; a no-op if `XLA_FLAGS` already requests a host device count.
  See `docs/dev/inference_methods.md` and the JAX section of `CLAUDE.md`.


- `Fitter`/`ForwardModel.fit(..., profile_mass=...)` analytically marginalizes

- The per-Q_H nebular grid (`enable_fast_nebular` / `approx=FeaturePrecomp()`)
  now serves DIG mixing instead of refusing it: `neb_logU` joins the grid axes
  whenever `neb_dig_frac` could be active (free, or fixed non-zero), even when
  `neb_logU` is itself Fixed, and its range extends (never clips, never
  refuses) to cover the DIG-shifted query point `neb_logU + neb_dig_delta_logU`.
  Reconstruction mixes two lookups against the same table via the new
  `mix_dig_grid_reconstruction` (`dig.py`), sharing the one mixing core
  `mix_dig_emission` / `mix_dig_line_luminosities` already use. Measured
  worst-case relative error over 10 seeds against the exact path: 1.17e-3
  (photometry) / 1.41e-3 (lines) at `neb_dig_frac = 0.3` with the axis extended,
  versus 1.72e-3 / 2.04e-3 at `neb_dig_frac = 0` on the same fixture -- both
  well inside the repository's 3e-2 parity ceiling (#2222).

- ``mix_dig_line_luminosities`` (``tengri.components.nebular.dig``), exported
  from ``tengri.components.nebular``: the line-luminosity counterpart of
  ``mix_dig_emission``, sharing its DIG mixing core. It calls
  ``predict_nebular_line_luminosities`` (instead of ``predict_nebular_sed``),
  keeps the HII call's ``line_waves``, and mixes only the luminosities (#2221).


- `met_logzsol_scatter` (the lognormal MDF width, in dex) now declares a
  `free_prior` of `Uniform(0.02, 0.4)`, so `met={'all_params': FREE}` (delta
  mode) and `met={'logzsol_scatter': FREE}` both work instead of refusing.
  The interval brackets the Milky Way disk MDF widths measured by
  Hayden et al. 2015 (sigma[Fe/H] = 0.17-0.32 dex across the disk) and the
  registry default `Fixed(0.1)`. All ten shipped recipes' free parameter
  sets are unchanged (#2245).

- The star-formation ordering constraint "SF cannot stop before it starts"
  is now enforced for the `dpl_lookback` and `trunc_exp` SFH types:
  `sfh_*_age_gyr` (onset lookback) must exceed `sfh_*_end_gyr` (cessation
  lookback), raising `ValueError` at build time on inversion, exactly like
  the existing `sfh_const_start_gyr`/`sfh_const_end_gyr` pair. Previously
  these two constraints were stated only in prose and an inverted pair
  built without complaint (#2247, phase 1).

- `bench/scripts/benchmark_float32_mps_parity.py` -- a self-contained pure-float32
  parity sweep for the Apple GPU (#1206). Apple's own `jax-metal` last released 0.1.1
  on 2024-10-08 and pins `jax == jaxlib >= 0.4.34`, not viable against tengri's JAX
  0.11; the community `jax-mps` plugin (MLX-backed) is the path that works, but it has
  no float64 at all. This script writes a float64 reference on CPU
  (`--write-reference`) for six progressive model seams (`stellar_dust`, `+dust IR`,
  `+Cue`, `+AGN`, `+radio+xray`, `panchromatic`) and, in its default mode, checks a
  float32 rebuild against it -- max relative forward error, gradient error, and a
  *converged* MAP fit's optimum parameter-vector deviation, PASS/FAIL at
  3e-3 / 1e-2 / 1e-2. The MAP fit itself uses L-BFGS to genuine convergence
  (`init_from` pinned to the shared truth; restarted from a stall rather than given a
  bigger iteration cap, since scipy's line search can abandon a call well short of
  its budget), with a non-zero exit if `--write-reference` cannot converge every seam,
  so an unconverged reference can never be committed. The MAP-loss deviation is
  reported as an informational column, not gated: at a converged optimum the loss is
  stationary, so a parameter agreement of 1e-6 implies a loss agreement of order
  1e-12 from that channel alone, and the ~1e-4 gap actually seen is float32's own
  chi-squared evaluation (cancellation in `data - model` at SNR 30) -- already bounded
  by the forward and gradient columns. Runnable on CPU, CUDA, or a Mac under the
  `jax-mps` venv; refuses to run with `jax_enable_x64=True` and prints the one-line
  environment fix instead. `docs/internal/getting_started/gpu.md` gains an "Apple GPU
  via jax-mps" section replacing the old Metal note, with a shorter mirror in
  `docs/performance/index.md`, `README.md`, and `docs/installation.md`.

- `dust_log_L_ir` (`log10(L_IR/Lsun)`): a total dust IR budget override.
  Declaring it -- `Fixed` or any free prior, via `dust_emission={'log_L_ir':
  ...}` -- replaces the energy-balance IR budget (`log_L_ir =
  log_L_absorbed + log10(dust_eta_balance)`) outright; leaving it undeclared
  keeps strict/relaxed energy balance exactly as before. Declares no
  `free_prior` (an absolute luminosity has no galaxy-independent interval),
  so `dust_emission={'all_params': FREE}` never frees it. `dust_eta_balance`
  is inert once the override is declared, and `SEDModel` now raises
  `ParameterError` at construction if it is free or `Fixed` at a value other
  than 1.0 alongside a declared `dust_log_L_ir`. Radio's FIR-radio-correlation
  amplitude follows the override too (#2187-series).

- `Observation` and its nested data classes, `Parameters` and `SSPData` expose `cache_key()`, each derived from a written policy ledger over every attribute (`tengri._cache_keys`), so a later structural signature can delegate instead of reaching into their fields (#2163).


- `tools/check_gradient_assertions.py`, wired into the `lint` job — a guard on
  the "undecided treated as good" assertion class. A gradient has three states,
  not two, and a predicate written against the *bad* state is satisfied for free
  by the *undecided* one. Both halves have already shipped: #2100 pinned the
  float32 seam gradients `isfinite`, and a gradient of exactly **zero is
  finite**, so the identically-zero `sum(predict_photometry)` gradient passed on
  CPU and GPU alike and the coverage meant to catch it structurally could not.
  #2178 is the mirror — the seam checks pinned the gradient `!= 0.0`, `nan !=
  0.0` is `True`, so a NaN satisfied a non-zero assertion, XPASSed a strict
  xfail as a repaired underflow, and shipped a float32 NaN on the default
  spectroscopy path. The rule is the conjunction: **finite AND non-zero,
  asserted together**, never either alone. Scope is gradients everywhere under
  `tests/`, plus every array under test in `tests/regression/precision/`, where
  a forward that has collapsed to zero fails the float32 question exactly as a
  NaN does. AST-based, with taint tracked through assignment so
  `leaves = [np.asarray(v) for v in tree_leaves(g)]` is still `g` — deliberately
  not a regex over source text, which 5d08a293e removed from this repository on
  purpose (#2108) and which could not tell `x > 0` on a gradient from
  `rel_err < 1e-5` on a residual. A lower bound (`max(abs(g)) > 0`) settles both
  halves on its own, because NaN fails every ordered comparison; a value pinned
  *as* zero or *as* NaN is the subject rather than the accident and is not asked
  for a partner. The narrow escape hatch is
  `# grad-assert: finite-only — <reason>` / `# grad-assert: nonzero-only —
  <reason>`, and a marker carrying no reason is itself a CI failure. The guard
  is verified against history, not intuition:
  `tests/fixtures/assertion_holes/historical.py` transcribes both pre-fix
  assertions verbatim and `tests/contract/test_gradient_assertion_guard.py`
  pins that the guard fires on each (#2100, #2178).


- `sfh_exp_start_gyr` / `sfh_dexp_start_gyr` / `sfh_const_start_gyr` (the
  SF-onset lookback for the `exp`, `dexp` and `const` SFH models) declare a
  `free_prior` and are dropped from `tools/check_param_free_priors.py`'s
  REFUSED ledger. A static ceiling of today's cosmic age is narrowed to
  `age_at_z(z)` at parse time (`parameters/groups.py`'s new
  `_narrow_free_priors_to_z`) whenever the build's redshift is knowable, so
  `'all_params': FREE` genuinely frees these onsets instead of silently
  leaving them pinned. A catalog with a per-galaxy redshift refuses the
  combination outright (the cap is only valid for one redshift). See
  `docs/dev/api_migration_v0.x.md` for the full migration note, including the
  one shipped recipe (`quiescent_z0`) whose free-parameter count changes.


- `bench/scripts/probe_block_metric_structure.py` — scores a candidate
  mass-matrix structure against the analytic metric without running a sampler.
  For a layout it forms the structured inverse mass matrix, whitens with it, and
  reports the condition number that survives alongside the matrix entries stored,
  so diagonal / block / low-rank / dense sit on one frontier. It also reports a
  **per-group verdict** — internal off-diagonal mass and internal-over-external
  coupling for each candidate group — which turns "which groups deserve a dense
  block" into a measurement. Used to answer #2166: block-structured mass matrices
  are dominated by a rank-`k` correction to a diagonal on every fixture and every
  storage budget tested, so the feature was declined rather than built. Numbers
  and the reasoning in `bench/reports/2026-09-06_block_metric_structure.md`.


- `tools/check_float32_scale_seams.py` — enumerates the float32 **scale seams**
  themselves rather than sampling a representative model. A scale seam is a site
  where a large physical constant or unit conversion multiplies a
  parameter-derived quantity; four bugs have now come out of that one shape
  (#1388, #1439, #2100, #2178) and each was fixed where it was found. The check
  parses `src/tengri` (AST, and evaluated constants — not a grep over source
  text, per #2108), and for each seam asks how large the product can get inside
  the range the parameter **declares in the registry**, never a range copied
  from a grid axis (45741f4cd). 52 seams; 46 of them exceed float32's range
  within their own declared prior, in 4 families — `L_sun * 10**agn_log_lbol`
  (38 sites), `L_sun * 10**log_total_mass` (5), `M_sun * 10**agn_log_mbh` (2)
  and the Lehmer LMXB mass term (1). Three families carry a recorded grouping
  that keeps them safe; the fourth is filed as an open defect (#2210), because
  recording a live defect as "handled" is worse than not recording it. Anything
  new is an error, and a registration whose seam is gone is also an error, so
  the inventory cannot rot. Runs in the `smoke` job, beside
  `check_float32_representable_constants.py`, which is where the checks that
  need tengri installed live — `lint` installs only ruff.


- `tests/regression/precision/test_float32_scale_seam_sweep.py` — sweeps each
  enumerated seam family across its parameter's whole declared prior in float32
  and requires the gradient to be finite **and** non-zero at every point (`nan
  != 0.0` is `True`, and zero is finite, so neither half is coverage alone). The
  inventory is read from the tool rather than written down twice: the module
  fails if the enumeration grows a family it does not sweep, which is what makes
  the recorded reason a measurement instead of the only evidence. The sweep
  refuted the first draft's reading of the black-hole-mass family as safe
  (#2210): `M_sun * 10**agn_log_mbh` is 1.99e39 at the *bottom* of its declared
  prior, and the float32 forward of a `kubota_done` disc is `nan` there under
  jaxlib 0.11.1 while finite under 0.11.0.


- **New AGN template-library blocks**, vendored at native grid resolution from AGNfitter-rX: `kd18_agnfitter` / `kd18_agnfitter_warmindex` (two distinct grids — KD18's warm-Comptonization spectral index is not interchangeable with a fixed value, up to 27% off at any single `warmIndex`), `nenkova_agnfitter_2p` / `nenkova_agnfitter_3p`, `skirtor_agnfitter_1p` / `skirtor_agnfitter_2p`, and `cat3d_wind_lowfwd`. Each ships crossval tests against the vendored AGNfitter-rX reference and a `GRID_EXTENT_SOURCES` entry so its declared prior bounds are guarded against the vendored grid's own axis extent.

- **Per-component AGN SED publishing**: `state.derived["sed_agn_disc"]`, `["sed_agn_torus"]`, `["sed_agn_lines"]` (NLR + BLR + Fe II combined), and `["sed_agn_polar"]` alongside the existing combined `["sed_agn"]`, so a caller can inspect or plot the disc/torus/line/polar-dust contributions separately instead of only their sum.

- **AGNfitter-rX informative priors**: `tengri.parameters.agn_priors` (also reachable as `tengri.agn.priors`, a real registered import path — `from tengri.agn.priors import agnfitter_priors` and `import tengri.agn.priors` both work) implements the eight optional composite log-prior penalty terms AGNfitter-rX offers (energy balance, AGN-fraction luminosity-function ties, mid-IR/UV/X-ray consistency), independently validated against upstream's own formulas. Wire into a fit via `Fitter(..., extra_log_prior=your_prior_fn)`; the public `agnfitter_priors(pred, redshift, dlum, ...)` adapter computes every prior's physical inputs from a `model.predict(params)` result for post-fit inspection.

- `tools/check_repro_render_fresh.py`: every reproduction-notebook render `scripts/render_reproduction_notebook.py` produces now carries a `metadata["tengri_render"]` stamp (`source_sha256`, `executed_at`, `tengri_version`) recording the exact source state it was rendered from; the guard recomputes the source hash and fails if a committed render's stamp disagrees with the `.py` sitting beside it today (`--strict` additionally requires every notebook to carry a stamp).

- Three public-API gaps closed, surfaced by the reproduction notebook needing workarounds for each: `tengri.declining_exponential` (the FSPS/bagpipes "tau" SFH shape, registered type `"tau"`) now resolves at the top level like its siblings `exponential`/`delayed_exponential` (it was already reachable via `tengri.sfh.declining_exponential`, just not flat-imported); `tengri.agn.priors.agnfitter_priors`'s Notes and `AGNFITTER_PRIOR_DEFAULTS` are now documented in `docs/api/models.rst` (the function itself already worked, only its docs entry and the defaults constant's were missing); `FilterCurve` is promoted from importable-but-undemoted to `tengri.__all__` and `docs/api/core.rst`, alongside `FilterConvention`.


- `mcmc_smc` — tempered Sequential Monte Carlo via BlackJAX, at
  `tier="experimental"`. A particle population annealed from the exact
  standardized `N(0, I)` prior to the posterior, so it is the first sampler here
  that does not start at the MAP: the MAP is used only to build the
  preconditioning metric. The tempering split is a split of the objective
  `build_loss_fn` already composes (data term + `standardized_neg_log_prior`),
  not a second implementation of it, and a contract test pins the sum against
  `_get_flat_logdensity` numerically. `n_chains` runs **independent** particle
  populations that share no state, so their split R-hat is a between-run test.
  `log_evidence` comes free with the weights and is validated against an
  analytic Gaussian evidence to 0.02 nats. **Not** on the batched catalog path;
  `_MCMC_VMAPPABLE` is unchanged.

  Two things a caller has to know, both of which cost this campaign a
  measurement. `diagnostics["min_ancestor_ess"]`, **not** the autocorrelation
  ESS, is the mixing diagnostic: a resampled particle population is
  exchangeable, and a within-population permutation that changes nothing about
  the sample moves the autocorrelation estimate by 1.4-2.1x. And the divergence
  denominator is `diagnostics["n_inner_transitions"]`, not `total_draws()` —
  SMC makes `n_temperatures * n_mcmc_steps` Metropolis transitions per particle
  and keeps one draw from each, so the usual ratio read 205% on the first row
  measured (the #2087 arithmetic, one sampler further out).

  Two defects were found in this backend before it merged, by cross-checking
  against BlackJAX's own tempered-SMC page, and both were in code that has never
  shipped. `blackjax.smc.base.step` resamples, moves the particles under the
  *old* temperature, then reweights toward the new one, so a ladder exiting at
  `lambda = 1` leaves a **weighted** sample; reading `state.particles` without
  `state.weights` returned draws from a slightly tempered posterior, biased
  -0.0164 in the mean and +0.0142 in the sd of an analytic Gaussian, with the
  same sign on every coordinate. A closing rung pinned at `lambda = 1` consumes
  those weights and rejuvenates under the true posterior; its `delta` is zero,
  so the log-Z increment is exactly `0.000e+00` and the evidence is untouched.
  Separately, an inner step-size controller aimed at fixed-length HMC's 0.651
  acceptance was actively harmful — an SMC inner move is a rejuvenation burst
  where a rejection leaves a duplicate, not a chain step that must decorrelate —
  and `step_size_gain` now defaults to `0.0`, matching the reference page.
  Measured in `bench/reports/2026-08-31_smc_evaluation.md`.


- `tengri.utils.scale.loss_scaled_grad` — `jax.grad` with the cotangent chain
  lifted into float32's normal range (multiply the scalar by `2**100`, divide
  the gradient back; exact for a power of two, so float64 gradients are
  bit-identical). It recovers the pure-float32 photometry gradient, which
  `jax.grad` returns as **exactly zero** because reverse mode has to store
  `d(F_nu)/d(L_nu) = 10**(-58)` at the flux projection and float32's smallest
  subnormal is 1.4e-45. Measured against float64 on three scale seams
  (stellar+dust, dust IR +44.5 dex, AGN +34.6 dex): ~1e-06, where the
  unboosted call is `0.0` on all three. The default was sized by sweeping it
  and not by the arithmetic, which says `2**70` should suffice and is wrong by
  0.7--18% on CPU (the cotangent picks up further O(1e-3) factors downstream
  and lands back among the subnormals, which XLA's CPU backend flushes; the
  same boost measures 1e-06 on CUDA). A *fit* never needed this — a
  likelihood's `1/sigma**2` is the same lift arriving for free, and
  `grad(neg_log_posterior_fn)` tracks float64 to ≤5.3e-04 in pure float32 on
  every measured seam. The underlying seam is unchanged and still needs
  #1388's scaled-SED contract (#1415).


- `DeadFitError`: NUTS, HMC and dynamic HMC keep the per-step divergence
  flags of their own warmup and refuse to sample when the final 10% of
  warmup (at least 10 steps) is 90% or more divergent, before the adaptation
  is cached and before the sampling scan compiles. `warmup_divergence_frac`
  joins `diagnostics` whenever warmup ran in that call, and is absent (not
  `None`) when a cached adaptation is reused, so `Posterior.save()` never
  warns about an entry it cannot write. The warmup log line prints the
  fraction when there is one, and the NUTS completion line prints the
  divergence percentage and the tree-depth summary (#2088). **Behavior
  change:** these methods now raise where they previously returned a frozen
  posterior with a warning, including on the `hmc_is` evidence path that BMA
  runs, so a caller looping over galaxies should catch `DeadFitError` (an
  `InferenceError`, and so a `RuntimeError`) and record that galaxy as a
  failed fit. It is exported from the top level as `tengri.DeadFitError`. A
  warmup shorter than the minimum window (10 steps) carries no verdict and is
  never refused: BlackJAX opens dual averaging well above the stable step
  size, so the opening steps of every warmup diverge whatever the posterior.

- `tools/check_param_restatements.py`: a new CI guard that a `ParamDeclaration` restated as a class-level `Uniform(lo, hi, ..., default=d)` literal on a `SEDModelComponent` subclass matches the canonical declaration for that parameter name in its domain's `_params.py` `PARAMS` tuple, unless allowlisted with a reason. AST-only (no `tengri` import), following `check_param_grid_extent.py`'s precedent. First run found 18 pre-existing mismatches across five legacy AGN disc/torus classes (`CAT3DTorus`, `KD18Disc`, `PowerLawDisc`, `Silva04Torus`, `SKIRTORAgnfitterTorus`), recorded as `docs/dev/known_bugs.md` PARITY-01 and since fixed (see Fixed, below).


### Fixed

- `agn_grahsp_a_bc` parameter description now correctly names the 5100 Å (510 nm)
  reference wavelength, matching the implementation and upstream GRAHSP. The
  previous description incorrectly stated 3000 nm, a wavelength at which the
  Balmer continuum is identically zero (#2158).

- `params_override` now rejects a noise parameter the built likelihood cannot
  read. Noise parameters are only consumed when declared in the spec (free or
  Fixed at nonzero); with the default spec (noise_frac_cal at Fixed(0.0)), the
  likelihood is plain Gaussian and ignores any noise_* override. The override
  was silently accepted, reporting success while having zero effect. It now
  raises with a message that explains the mechanism and suggests declaring it
  in the spec. (#2193).


### Changed

- The cigale reproduction's two §9e torus panels compare AGN dust emission
  instead of the full SED. Both took their ratio over stellar + host dust +
  disc + polar screen + torus through IR filters while their headings named a
  torus library, so they reported a whole-SED difference under an AGN heading.
  Each arm is now the torus plus polar screen summed, which is independent of
  how the two codes partition those components — pcigale subtracts the polar
  re-emission from the torus dust, tengri rescales the torus against a shared
  budget and carries polar separately. `cigale_driver.to_lnu_contribution`
  reads one named pcigale contribution out of `sed.luminosities`, mirroring the
  `sed.components["sed_agn_torus"]` accessor the agnfitter page already uses.
  The SKIRTOR panel now separates its axes: the residual holds flat against
  optical depth (0.899×, 0.898×, 0.900×) and inclination (0.895×, 0.923×) and
  swings 1.8× across the opening angle (0.824× / 0.898× / 1.485×), so one axis
  carries the disagreement. Both blocks also drop the deprecated
  `predict_rest_sed` for `predict`.

- `profile_mass` no longer refuses a fit that measures emission-line fluxes.
  Guard #8 was the OR of three unrelated situations — line amplitudes already
  analytically marginalized (`eline_marginalize`), line amplitudes as free
  parameters (`eline_mode="fitted"`), and a **measured** line-flux data channel
  — and refused all three with one reason that named none of them. Only the
  third is a plain Gaussian channel whose prediction is proportional to the
  mass amplitude, and it is the configuration users actually have (photometry
  plus measured lines). It now engages, and the analytic marginal covers the
  line block as well: `A = sum f^2/sigma^2` and `B = sum d f/sigma^2` are
  additive across independent Gaussian blocks, so the line block concatenates
  onto the photometry block with no new math. Linearity is structural rather
  than incidental — the nebular grid tabulates luminosity per ionizing photon
  and multiplies by `Q_H` afterwards, with `neb_logU` an independent grid axis,
  so `L_line = Q_H * l(met, logU, logZ_gas)` with `Q_H` linear in the stellar
  amplitude and `l` independent of it. Measured on a wNE grid with eight bands
  and three lines, the guard's own worst-of-nine-thetas deviation is 1.705e-13
  with the line block and 1.705e-13 without, the maximum set by a photometry
  band either way. The other two cases still refuse, now each with its own
  reason naming which fired. Two further refusals are new rather than relaxed:
  **censored line fluxes** (`LineFluxData.is_upper_limit` / `is_lower_limit`),
  which enter as `ln Phi((F_lim - F_model)/sigma)` and cannot join a quadratic
  — the pre-existing censored-data guard cannot see them, since it reads
  `fitter.data_mask`, which covers the photometry/spectroscopy vector only —
  and a **user-supplied likelihood**, which the profiler cannot inspect to
  confirm the line block is scored as a plain Gaussian. Closes #2360.

- Reproduction parameter sweeps encode the swept value in a single-hue
  sequential colormap with a colorbar, replacing the categorical color cycle
  that gave a temperature sweep three unrelated hues and repeated a color
  outright past ten cases. Line style continues to carry which code is which,
  so color is free to carry magnitude; sweeps whose cases are model families
  rather than values of one parameter (casey / schreiber / dale) keep the
  categorical colors, since a ramp there would imply an ordering that does not
  exist.
- The three star formation history comparisons drew nothing. `sfh_grid_lbt_yr`
  spans the full cosmic lookback, so `age - lbt` ran negative for every sample
  older than the galaxy — to -8.8 Gyr on a 5 Gyr delayed-tau — and the two arms
  shared no interval. Both arms now sit on their common interval.
- `sweep_fig` and `overlay_ratio_fig` refuse a case whose two arms share no
  region where both are positive, naming both ranges, instead of plotting a
  ratio against interpolation zero-fill. Three residuals turned out to be this
  artefact rather than a disagreement between codes.
- Reproduction comparisons that were not comparing like with like are matched:
  the AGN torus covering factor `agn_torus_frac` is pinned wherever the
  reference emits the reprocessed luminosity over the full sphere (CIGALE,
  Synthesizer, ProSpect and now Prospector/FSPS), since its 0.5 default halves
  the tengri arm and accounted for most of a 0.45-0.24x span; IR template
  comparisons normalize on a common wavelength rather than each arm's own peak;
  and off-grid sweep nodes are moved onto nodes both codes tabulate.
- Reproduction sweep figures frame their y-axis on the data. `sweep_fig` gained
  the `dyn_range` floor `overlay_ratio_fig` already had, so a sweep covering
  three decades is no longer drawn on an axis spanning eighteen.

- Physics reproduction comparisons move from side-by-side panels onto one
  shared axis with a tengri/reference ratio panel and a tolerance band,
  drawn via the unified `overlay_ratio_fig` function in
  `reproduction/_validation.py` so every cross-code comparison follows the
  same visual grammar.

- Reproduction comparisons compare like with like across multiple codes:
  Prospector normalizes both panels on the same stellar mass convention;
  Synthesizer pairs the analytic-emitter sweep at `beta_ir = 0`;
  redshift-dependent attenuation curves sweep across their full range;
  BAGPIPES' attenuation sweep matches its prescription on both sides;
  AGNfitter's cold dust lands on a shared grid node, and its radio section
  states each side's luminosity normalization in place of a ratio between
  two different ones. BAGPIPES'
  Summary reads its numbers from the render. cigale's IR-template figures
  render at 120 dpi with a ratio row per model family; Schreiber 2016's
  ratios span 0.5× to 2× across the dust continuum at T = 25 K while the
  8–1000 µm integral agrees to under half a percent, both sides normalized
  on the same absorbed energy. The reproduction CONTRACT names the overlay
  as the default comparison figure.

- The six reproduction notebooks (`reproduction/{agnfitter,bagpipes,cigale,prospect_r,prospector,synthesizer}/01_*.py`)
  now compare parameter sweeps and model cases in every physics block instead of one fiducial point
  each: SFH families and τ × age grids, every attenuation law the reference code offers with its
  slope / bump / R_V knobs and an A_V ladder, dust-emission library nodes (DL07, DL14, Casey 2012,
  Schreiber 2016, Dale 2014, analytic emitters), nebular logU × Z_gas × f_esc grids, AGN disc and
  torus node grids, IGM redshift sweeps, X-ray corona and radio grids — each new section is one
  ratio-panel figure plus one printed table, and every Summary table is assembled from the render's
  own printed numbers. Every model on every page now carries nebular emission on both sides (tengri:
  Cue at the matched logU / Z_gas / f_esc; the reference code: its own nebular module), except the
  raw-SSP check and AGNfitter-rX's host, which has no nebular term. Shared helpers in
  `reproduction/_validation.py`: `sweep_fig` (overlay + ratio panel), `window_rows` (with a
  peak-relative deviation for curves that reach zero, so SFR(t) tables no longer flag exact matches)
  and `print_window_table` (unit and scale of the x column), and `filter_rows_native` (band
  averages on each spectrum's own grid, since interpolating tengri's emission lines onto a coarse
  reference grid before band-averaging aliases). The bagpipes driver aliases `np.trapz`
  to `np.trapezoid` for NumPy ≥ 2, and ProSpect's `massfunc_dtau` is compared on its recent branch,
  the only part that is a delayed-τ.

- `dust_frac_agn` is now declared with a `free_prior` of `Uniform(0.0, 0.99)`,
  making it wildcard-reachable (`all_params: FREE`) exactly on
  `dale2014_cigale`, the engine variant whose shipped template grid carries the
  QSO template where the parameter is live. On plain `dale2014` (QSO-less grid),
  the parameter remains inert and unreachable by the wildcard, preventing the
  sampler from exploring a flat dimension (#2244). The fix follows the
  engine-scoped declared-exception pattern: `dale2014_cigale` declares `frac_agn`
  at the component class level; plain `dale2014` omits it. Simultaneously, the
  AGN+Dale double-count warning (#721) now fires on `dale2014_cigale` (where
  `frac_agn` is live and can double-count with a composable AGN torus) instead
  of plain `dale2014` (where the parameter is inert). One behavioral break
  rides along: an explicit `frac_agn` key beside plain `dale2014` — previously
  accepted and silently ignored — now raises `ParameterError` naming the
  variant that reads it, via the grammar's standard unknown-key check
  (#2244, #721). The same warning now also covers `energy_balance_split`, whose
  `dust_L_agn_ir` slot adds AGN-heated IR beside an active composable AGN torus
  (#2251).

- The GRAHSP AGN disc normalization `agn_grahsp_l5100` (`LogUniform(1e42, 1e47)`, erg/s) is
  renamed `agn_grahsp_log_l5100` (`Uniform(42.0, 47.0)`, dex), with no alias (#1206). The
  linear parameter *value itself* is `inf` in float32 before any kernel runs; translate with
  `agn_grahsp_log_l5100 = log10(agn_grahsp_l5100)`. The composable `grahsp_sbpl` disc block
  is now float32-exact (previously the last disc in
  `tests/regression/precision/test_agn_disc_float32_inventory.py` that was not); the
  `Float32UnsafeAGNWarning` escape hatch it used is removed as unused.

- The eleven line-luminosity properties (`halpha`, `hbeta`, `lya`, `oii`, `oiii_4959`,
  `oiii_5007`, `nii_6548`, `nii_6584`, `sii_6717`, `sii_6731`, `civ_1549`) and the three
  X-ray luminosities (`l_x_xrb`, `l_x_agn`, `l_x_total`) now return `Lsun`, not `erg/s`
  (#1206). **Breaking, with no alias** — a value of ~1e40-1e45 erg/s is `inf` in float32
  (max 3.4028e38) as a bare number, before any physics runs; `halpha` now Lsun: multiply by
  `3.828e33` for erg/s. The `log_<name>` / `log_l_x_*` companions are unchanged, still dex
  re erg/s (`log_halpha == log10(halpha * L_sun)`); `log_l_x_agn` and `log_l_x_total` are
  fixed alongside the unit change (previously `nan` in float32: they read the linear
  `L_agn_bol` and took its `log10`; they now read the `log_L_agn_bol` companion the AGN
  component already publishes). `IonizingQuantities.q_h` (the `state_to_ionizing_quantities`
  bridge) and `XRayQuantities` are updated to match. `NAMING_CONTRACT.md` §4c documents the
  unit-standard rule (luminosities in Lsun, unbounded rates as `log_*` in dex).

- The linear ionizing photon rate `q_h` is retired, with no alias (#1206). Every physical
  ionizing rate is ~1e53-1e56 photons/s, past float32's ceiling in any linear unit — there
  is no float32-safe form to keep, unlike the line/X-ray luminosities above. `log_q_h`
  (dex re photons/s) is the sole surviving property; `q_h` → `10**log_q_h`. `pred.q_h`,
  `pred.ionizing.q_h` and `predict_properties(names=("q_h",))` now raise `KeyError` naming
  `log_q_h`. `IonizingQuantities` drops the `q_h` field.
- **`profile_mass` now covers spectroscopy and joint photometry+spectroscopy fits, not
  only photometry.** Every channel tengri fits is linear in the total stellar mass, so
  the exact `chi2(M) = chi2_min + A*(M - M*)^2` quadratic (`tengri.inference.mass_profile`)
  holds over the FULL data vector, not just the photometric one: `A` and `B = A*M*` are
  now sums over whichever vector `data_type` assembles (photometry, spectroscopy, or
  their photometry-then-spectrum concatenation for `"joint"`, the order
  `tests/regression/bug/test_bug_1366_joint_data_record.py` pins), reusing
  `loss_functions._build_prediction` (and its JIT-threaded SSP-grid path) so the
  profiled statistics see exactly the vector and noise the unprofiled Gaussian
  likelihood does. The linearity guard's two-mass probe is generalized the same way.
  Calibration marginalization, any emission-line/line-ratio/spectral-index channel,
  Student-t noise, a variable-noise model, and censored data remain refused
  unconditionally (each is either its own marginalized linear block or carries its own
  likelihood plumbing this module does not yet share); `"auto"` still steps aside
  silently for them and an explicit `profile_mass=True` still raises naming the guard.
  Behavioral change for spectroscopy/joint fits that satisfy every other guard: they
  now profile the mass under `profile_mass="auto"` where they previously always sampled
  it.

- **The default inference method is `mcmc_nuts_fast`** (was `vi`): four NUTS
  chains, 150 warmup steps, no separate burn-in, 300 draws, target acceptance
  0.8, on the mass-profiled posterior with the dense metric and, when the CPU
  is exposed as devices (`TENGRI_HOST_DEVICES`), pmapped chains. Measured at
  9.5-17.2 s per galaxy on eight logical cores across twelve `ctl-dpl` /
  `ctl-jwst` seeds with min ESS >= 100 on eleven
  (`bench/reports/2026-09-11_profile_mass_20s.md`). `forward.fit(data)`,
  `Fitter.run()` and `fit_batch` all share it; `fit_batch` runs it one galaxy
  at a time (the vmapped shared-adaptation engine is measured to freeze
  lanes). `method="vi"` is unchanged and still selectable. The new method is
  canonical (`mcmc_nuts_fast`), primary tier, and every setting is
  overridable; its draw budget lives in `defaults.toml`
  `[inference.mcmc_nuts_fast]`.


- **NUTS/HMC/dynamic-HMC `dense_mass_matrix=None` auto-policy is dense at
  D <= 12, not D < 8** (behavioral change, #319 revision). The D < 8 cliff
  generalized a `mean_sfh_type="dense_basis"` finding (22.78 GB warmup peak
  at D=8) to every SFH. Measured on `ctl-dpl` (D=8 photometry, 14 bands, a
  non-`dense_basis` DPL SFH): the dense window adaptation uses 1.1-1.9 GB
  RSS and costs 3.4x fewer gradients per effective sample than diagonal
  (dense 34 g/draw, ESS 83; diagonal 550 g/draw, ESS 113; six-seed sweep).
  `_resolve_dense_mass_matrix` now returns dense for `n_dim <= 12` *unless*
  the spec's SFH is `dense_basis` (diagonal at any D in that case — it is
  `dense_basis`'s per-sample derived-quantity publishing, not dimensionality
  on its own, that drives the historical spike), and diagonal above D = 12
  regardless of SFH. `HMC`/`dynamic HMC`/`CatalogFitter`/`fit_batch` all
  route through the shared `resolve_dense_mass_gate`, so the revision applies
  uniformly; the `DENSE_MASS_MAX_DIM=30` cap and explicit `True`/`False`
  overrides are unchanged. A fit at D=8-12 that pinned a diagonal-metric
  posterior mean or wall-time to a tight tolerance may need updating; pass
  `dense_mass_matrix=False` to keep the previous diagonal behavior exactly.


- **MAP defaults to L-BFGS, not Adam** (behavioral change). `run_map`'s
  `optimizer=` default is now `"lbfgs"` (alias `"lbfgs_scipy"`), and every
  internal MAP seed that does not pass an explicit `optimizer=`
  (`_maybe_map_init`'s NUTS/HMC/VI warm start, `run_laplace`, `run_pathfinder`,
  the vmapped batch-MAP path in `Fitter._fit_batch_vmap_map`) picks it up.
  Measured on a D=8, 14-band mock recovery fixture: the population default
  (Adam, 8 restarts × 800 steps) reached a negative log posterior of 6.33 and
  had not converged (a 300-step single run reached 7.88, a 100-step run 115),
  while a single scipy L-BFGS-B start reached 6.0008 in well under a second;
  on a second fixture the Hessian at the Adam point carried a negative
  eigenvalue, i.e. was not even a local minimum. Every downstream consumer of
  a MAP point — sampler warm starts, the Laplace approximation, preconditioning
  metrics — is better served by a converged optimum than by a fixed
  gradient-step budget that may or may not have reached one. Two
  implementations share the `"lbfgs"` name because scipy is not JAX-traceable
  and so cannot be vmapped: the single-start path (`n_restarts=1`, the
  default) runs scipy's L-BFGS-B; the multi-start and vmapped-batch paths
  (`n_restarts>1`, or `Fitter.fit_batch(method="map")`) run
  `jax.scipy.optimize.minimize(method="BFGS")` instead, which is pure JAX and
  therefore vmappable — and, like scipy, needs no optional dependency
  (`jax.scipy` ships with `jax` itself, unlike `optax`/`jaxopt`). `"adam"`,
  `"adamw"`, `"sgd"`, and pre-built optax optimizers remain fully supported by
  name.

- **`profile_mass` defaults to `"auto"`, not off** (behavioral change). Every

- `dust_eta_balance`'s declared free prior is a linear `Gaussian(1.0, 0.2)`
  truncated at 0 (was `LogNormal(0, 0.2)` on log eta);
  `builders.dust.emission.relaxed_energy_balance(sigma=)` takes the linear
  sigma.


- `import tengri` raises the default matmul precision to `"highest"` at
  import, unconditionally, unless `JAX_DEFAULT_MATMUL_PRECISION` is already
  set or the live config already holds a value; `tengri.utils.devices.setup_jax`
  mirrors it. On Ampere+, XLA otherwise lowers float32 matmuls to TF32
  (measured 4.5% error on Fisher-matrix parameter error bars); the knob only
  affects float32 matmuls, so this is a no-op for a float64 session and for
  CPU (no TF32 path), and measured zero speed cost. Being unconditional also
  covers a float32 arm entered later through a bare
  `with jax.enable_x64(False): ...` while the process default stays x64-on --
  exactly the pattern `test_fisher_float32.py`'s own float32 arm uses. An
  explicit `JAX_DEFAULT_MATMUL_PRECISION` always wins (#2022).


- Cue's default line catalog is now the full ~138-line set instead of the
  128-line CLOUDY/FSPS-matched subset (`cue_full_catalog` defaults to
  `True`); pass `neb={'type': 'cue', 'full_catalog': False}` (or
  `Parameters(cue_full_catalog=False)`) to keep the legacy subset for
  cross-code comparisons. Reverses the 2026-05 `#303` back-compat default.
  `predict_photometry`, `rest_sed` and every other headline line are
  bit-identical either way: only the discrete line catalog and `civ_1549`
  change (#2239).


- The shock SED's default dust screen flips from the unconditional
  birth-cloud form (`tau_bc·k_bc + tau_diff·k_diff`) to diffuse-only
  (`tau_diff·k_diff`): an AGN-outflow shock is not, in general, still
  confined to the compact star-forming birth cloud the young-star screen
  models, and the diffuse-only default now applies to every shock
  normalization (`norm='frac'` and `norm='lhalpha'` alike). This is a
  numeric change: on the `test_shock_attenuation_equivalence.py` fixture
  (τ_bc=2, τ_diff=1, z=0.5, Calzetti, SDSS *gri*) the shock's photometric
  contribution grows by roughly 8×-33× band to band, because dropping the
  `tau_bc·k_bc` term removes the dominant attenuation factor. The
  exact-vs-precomp agreement is unaffected by the flip (both screens agree
  to float precision on that fixture, and the `test_precomp_channel_drift.py`
  gap moves from 6.343e-04 to 6.289e-04 on its own fixture) because both
  paths read the one `sed_shock_attenuated` value regardless of which screen
  is selected. Set `dust_attenuation={'shock_screen': 'birth_cloud'}` to keep
  the old default.

- The dust energy-balance integral (`L_absorbed`, and so `L_ir`) now counts
  the shock SED's absorbed power under its own `shock_screen` choice; before,
  the shock SED was attenuated (once #1434 unified the exact and precomp
  paths) but its absorbed power was never added to the integral — a
  pre-existing gap between what the screens remove and what the IR
  re-emission pool receives. A source whose screen choice is `"none"` is
  unattenuated and so contributes exactly zero to the integral, with no
  separate on/off branch needed.


- The four inference-side hand-written cache keys are now policy-derived too (#2163 E.5): `Fitter._engine_cache_key()` and `_data_fingerprint()` share a pair of complementary ledgers (`tengri.inference._engine_policy.ENGINE_POLICY`/`FINGERPRINT_POLICY`) over every `Fitter` attribute — engine `shape` rows are exactly the fingerprint's `content` rows — instead of two independently hand-maintained field lists that had never been checked against each other or against the live attribute set; the engine key gains a `_user_likelihood` row a custom Likelihood previously had no representation in at all, and `_line_flux_override`'s row is now the strictly more complete `LineFluxData.cache_key()` (per-line upper/lower-limit flags, not merely "any limit mask present"). Every MCMC backend's adaptation cache (`nuts`/`hmc`/`dynamic_hmc`/`chees`/`ghmc`/`mclmc`/`adjusted_mclmc`/`first_order`) now builds its tuning tuple through one `adaptation_method_key()` helper that binds the runner's own signature and drops a written exclusion ledger (`_ADAPT_IRRELEVANT`: `context`, `key`, `init_from`, `n_burnin`, `n_samples`, `n_chains`, `chain_method`, `verbose`), instead of a hand-picked tuple every backend maintained separately. `PreconditionedProblem.cache_key` is now a small policy ledger (`strength` content, everything else excluded — the wrapped closure and the per-galaxy starting position cannot be keyed without either aliasing two different whitening bases together or defeating cross-galaxy adaptation sharing) rather than a hand-picked `("whiten", strength)` tuple.

- `SEDModel.compile_signature()` is derived from a policy ledger over every model attribute (`tengri.forward._signature_policy`) with the nested `cache_key()` of the observation, parameters and SSP grid, memoized on the instance and invalidated by the two structural mutators; four structural attributes the hand-written list never keyed (`lgmet_scatter`, the GP field kernel, `lsf_n_bins`, `igm_patchy`) now are, and an attribute nobody classifies fails a contract test instead of shipping a wrong number (#2163).


- The on-disk WavePrecomp z-table and IGM subband caches are keyed by every field of a frozen request dataclass (`ZTableRequest`, `SubbandRequest`) instead of a hand-written field list, with one version constant per cache (both bumped, so existing tables recompute once) and the cosmology the integrand uses folded in as the #2145 tripwire; the ionizing-spectrum table gains a version constant (#2163).


- `SEDModel.from_config`'s dust parameter docstring stated a MODEL name
  (`"charlot_fall"`) and LAW names (`"calzetti"`, `"kl04"`, …) as though they
  were the same kind of thing. It now states plainly that one law is applied
  explicitly to BOTH attenuation screens (birth cloud + diffuse ISM), and that
  `"charlot_fall"` (the default) is an alias for `"power_law"` on both
  screens — the classic Charlot & Fall (2000) model — not a law-registry name
  in its own right (#2021). `suggest_parameters`'s `dust_law_bc` default is
  aligned from a stale hardcoded `"power_law"` to `None`, and its resolved
  `(dust_law_bc, dust_law_diff)` pair now goes through the same
  `resolve_dust_screen_laws` rule `Parameters()` itself uses, so the printed
  cheatsheet cannot describe a configuration `Parameters()` would refuse
  (#2224).

- **An explicit `neb={'type': 'ssp'}` (or `{'type': 'none'}`) now silences
  `BakedInNebularWarning`; an omitted `neb=` still fires it (R49).** Both
  spellings resolve to the same `BakedInBackend` as an omitted `neb=`
  (`nebular_mode='off'`/`'ssp'`), so the advisory could not previously tell
  "the user said no nebular emission" from "the user never mentioned it" —
  every reproduction build fired the same warning regardless of intent.
  `Parameters` now records whether any of the three neb-group kwargs
  (`nebular`, `nebular_ssp`, `nebular_cue`) was explicitly present (same
  presence-based mechanism as `_user_provided`, at group granularity), and
  `SEDModel` constructs `BakedInBackend(ionizing_source_warning='suppress')`
  when it was. The warning text no longer advises a `warnings.filterwarnings
  (message=...)` filter (a message filter is the anti-pattern that hides
  every other warning matching the same text); it names the explicit
  `neb={'type': 'ssp'}` acknowledgment instead.

- **AGN parameter ownership: disc physics nests under `disc`.** Eleven names
  every reader of which is a disc block now belong to the `agn.disc` sub-block
  rather than the shared `agn` top level: `agn_alpha`, `agn_log_mbh`,
  `agn_log_ledd`, `agn_a_spin`, `agn_f_hard`, `agn_gamma_warm`, `agn_kt_warm`,
  `agn_gamma_hard`, `agn_kt_hot`, `agn_r_warm_ratio` and `agn_ebv_disc`.
  **Breaking for composable builds**: written at the agn top level they now
  raise with the nesting to use —
  `agn={'disc': {'type': ..., 'agn_log_mbh': ...}}`. The gain is that
  `disc={'type': T, 'all_params': FREE}` frees that disc type's own physics,
  which it did not for 13 of 14 registered disc types. `agn_attenuation_ebv`
  and the polar-dust knobs nest under `atten` for the same reason.

- **AGN parameter ownership is complete, and 31 further names moved out of the
  shared group.** Every declared and consumed `agn_*` name now has exactly one
  owner; before, 31 of 94 had no entry at all and silently defaulted to shared,
  which is why a wildcard could not free eight parameters that measurably move
  `predict_photometry`. **Breaking for composable builds**: each of these was
  accepted at the `agn` top level and now raises with the sub-block to nest it
  under.
  - `disc` (13): `agn_T_max`, `agn_adaf_alpha`, `agn_adaf_beta`,
    `agn_adaf_delta`, `agn_astar`, `agn_cigale_disk_delta`,
    `agn_grahsp_cutoff_nm`, `agn_grahsp_l5100`, `agn_grahsp_plbendloc_nm`,
    `agn_grahsp_plbendwidth`, `agn_grahsp_plslope`, `agn_grahsp_uvslope`,
    `agn_log_mdot`
  - `nlr` (7): `agn_nlr_alpha_pl`, `agn_nlr_fwhm_kms`,
    `agn_nlr_line_efficiency`, `agn_nlr_logU`, `agn_nlr_logZ`, `agn_nlr_logn`,
    `agn_nlr_xi_d`
  - `blr` (4): `agn_blr_line_efficiency`, `agn_blr_logU`, `agn_blr_logZ`,
    `agn_blr_logn`
  - `torus` (2): `agn_theta_torus`, `agn_delta`
  - `atten` (1): `agn_ebv`
  - still shared, and each says why in the table (4): `agn_ir_frac`, read by the
    runner's cross-block normalization stage rather than by any one block; and
    `agn_grahsp_a_bc`, `agn_grahsp_tor_temp`, `agn_grahsp_tor_cutoff_um`, which
    no composable block reads at all -- only the monolithic `grahsp` model,
    where every parameter is written flat.

  Migration is the same one line in every case: nest the parameter under the
  sub-block named above, e.g. `agn={'disc': {'type': ..., 'agn_astar': ...}}`.
  Two consequences worth stating separately:
  - `nlr={'type': 'grahsp', 'all_params': FREE}` is now a no-op and warns.
    `agn_grahsp_a_lines` and `agn_grahsp_linewidth_kms` are read by the nlr,
    blr AND feii GRAHSP blocks, so no single sub-block can own them; they are
    shared, and the agn-level wildcard frees them instead. Previously they were
    nlr-owned, which freed them for that one block and left `blr='grahsp'`
    reading two parameters its own wildcard could not reach.
  - `agn={'atten': {'law': 'prevot_smc', 'ebv': ...}}` now frees `agn_ebv` --
    the `qsogen_smc` block's own E(B-V) -- and not `agn_attenuation_ebv`. Each
    name keeps its own prefix-stripped short spelling; write
    `'attenuation_ebv'` for the attenuation-stage screen.


- **`agn['type']` is validated at build time.** It was forwarded to
  `agn_model` unchecked and the first `predict_photometry` raised `Unknown AGN
  model`, so `agn={'type': 'fritz'}` and `agn={'type': 'totally_bogus_xyz'}`
  built identically. A registered *block* name is now refused with the
  composable form that works — `agn={'type': 'composable', 'torus': {'type':
  'fritz', ...}}` — and anything else with the model menu plus close matches.
  **Breaking** only for builds that never worked: the refusal replaces a
  deferred failure, not a working spelling.

- **fracAGN belongs at the agn top level.** Written inside a sub-block, the
  four spellings (`ir_frac`, `agn_ir_frac`, `fracAGN`, `agn_fracAGN`) split
  three ways: the builder ignored the key so `agn_ir_frac` stayed 0.0, the
  #2189 legacy scan saw it and narrowed `agn_torus_frac` out of that block's
  wildcard anyway — dropping a live dimension (measured 8.74 relative
  photometry change for `fritz` with fracAGN inactive) — and the conflict
  guard, which reads provenance, stayed quiet. All four spellings now raise
  inside any sub-block, naming the placement. It governs the runner's
  cross-block normalization stage, not one block's physics.

- **Monolithic AGN models keep their parameters flat.** The nesting guard above
  applies to composable builds only. `agn={'type': 'kd18_agnfitter',
  'agn_log_mbh': ...}` and every other non-composable type accept the
  parameters that type declares at the agn top level, as they must: a
  monolithic build has no sub-block, and one carrying sub-block keys is refused
  outright, so applying the guard there left no working spelling at all
  (measured: all 13 non-composable names raised). Unknown names still raise.

- **`agn_band_frac` is retired in favor of `agn_torus_frac`.** SKIRTORTorus's
  own, single-consumer name for the covering fraction every other torus block
  already called `agn_torus_frac`. The old name raises a one-message redirect
  from either placement — the agn top level (where pre-rename configs wrote it)
  and the `torus` sub-block — branching on the build for the spelling that
  works. `agn_frac_agn` now resolves to `agn_torus_frac`.

- **`agn_polar_temperature` is an alias of `agn_polar_T`**, with the standard
  deprecation warning; the duplicate declaration is gone.

- **`list_agn_models()` lists every buildable AGN model.** It returned
  `['composable']` while eleven deprecated preset names and two self-contained
  ones (`skirtor_stalevski`, `grahsp`) were all accepted by
  `agn={'type': ...}` — buildable and undiscoverable at the same time.

- **The atten sub-block's short key is `attenuation_ebv`.** Each of the two
  `agn_*` E(B-V) names keeps its own prefix-stripped short spelling:
  `attenuation_ebv` for `agn_attenuation_ebv` (the atten block's own) and `ebv`
  for `agn_ebv` (the separate `qsogen_smc` knob). The retired-spelling
  migration message advertises the working one; it previously advertised
  `'ebv'`, which froze the parameter it was meant to free.

- **Recipe free-parameter counts move by one.** `agn_panchromatic` and
  `composable_agn` both gain `agn_ebv_disc`: the runner reddens every disc
  block's continuum with it (#916), so it is live for all 15 disc types, and
  both recipes' disc sub-dicts state no wildcard of their own, so the
  top-level one now reaches it. `composable_agn` loses `agn_torus_frac`, a
  measured-dead dimension under that recipe's active fracAGN (see below).
  Net −1/+1 on `composable_agn`, +1 on `agn_panchromatic`.

- `agn_bcnorm` (qsogen_balmer's Balmer-continuum strength, #2175) belongs to
  the `feii` sub-block, so `feii={'all_params': FREE}` reaches it. Update to
  #2175: `qsogen_balmer` does differ from `boroson_green` — the earlier
  "identical" reading came from evaluating both at `Fixed(DEFAULT)`, and
  `agn_bcnorm`'s default 0.0 is exactly the value at which the Balmer
  continuum is defined to vanish. At each type's own prior median they differ.

- SFH short keys resolve for multi-word type names: bare `tau_gyr`, `age_gyr`
  and `log_total_mass` now work for `declining_exp` as they already did for
  `delayed`.

- The Feltre+2016 NLR dust-to-metal grid axis had two names: `agn_nlr_xi_d`,
  which the `nlr='feltre'` block reads, and `neb_xid`, an orphan declared
  under every composable AGN build and read by nothing. `neb_xid` is retired;
  writing it in any group (`neb`, the `agn` top level, or nested under
  `agn.nlr`) raises a loud legacy-key error naming `agn_nlr_xi_d` and the
  placement that works. The `_AGN_EXTRAS` table, the `_LAZY_DECL_EXTRAS`
  bucket hook, and the registry's adapter loop for them are removed with it,
  so every parameter bucket is now exactly its component's own declarations.
  The axis was also inert: the Feltre backend snapped $\xi_d$ and
  $\alpha_{\rm pl}$ to their nearest tabulated node, and a nearest-neighbor
  lookup is piecewise constant, so `agn_nlr_xi_d` measured a gradient of
  exactly 0.0 at every prior quantile — dead by construction, which is why the
  `nlr='feltre'` wildcard excluded it. All five Feltre axes are now
  interpolated together with the same C²-continuous triweight kernel the
  continuous three already used, so `agn_nlr_xi_d` moves `sed_agn` by a
  relative 0.154 across its prior (against 0.384 for `agn_nlr_logU`), is
  listed in the block's `AGN_BLOCK_CONSUMES` entry, and is freed by
  `agn={'nlr': {'type': 'feltre', 'all_params': FREE}}`. Numbers move: at the
  grid node ($\alpha_{\rm pl}=-1.7$, $\xi_d=0.3$, solar $Z$) the 20 line
  luminosities shift by a median 6.7% (min 1.1%, max 23.2%) against the
  snapped values, because the triweight kernel spreads weight over
  neighboring nodes rather than taking one exactly (#2214).

- The Feltre+2016 NLR ionizing power-law slope had the same disease one
  ruling later: `agn_nlr_alpha_pl`, which `blocks/nlr.py` reads, and
  `agn_alpha_ion`, a duplicate declaration with an identical prior and
  default, partitioned to `agn.nlr` and read by nothing. `agn_alpha_ion` was
  reachable and silently inert — `nlr={'type': 'feltre', 'agn_alpha_ion':
  FREE}` (or the short form `alpha_ion`) parsed, freed the parameter, and
  moved nothing. `agn_alpha_ion` is retired; writing it (or `alpha_ion`) in
  any group raises a loud legacy-key error naming `agn_nlr_alpha_pl` and the
  placement that works (#2214).

- **`'off'` is now accepted as a synonym for `'none'` across every group with
  an off switch, not just three of them.** `dust_attenuation`, `dust_emission`
  and `agn` already normalized `'off'` onto `'none'`; `neb`, `shock`,
  `radio`'s `sf` and `agn` sub-blocks, `xray` and `igm` raised `Unknown type
  'off'` for the identical request spelled the other way. A single shared
  helper, `_normalize_off_switch` (`parameters/groups.py`), is now the one
  place the off-switch vocabulary is defined; every one of the eight groups'
  translators calls it immediately after reading its raw `type` value, before
  any type-menu validation.


### Fixed

- The nebular continuum (Cue, CloudyGrid) no longer stops at its 1 cm table
  edge but continues as optically thin free-free (L_nu ∝ nu^-0.1, anchored at
  the last node) to the grid end, and a Cue model without a radio block now
  declares wavelength nodes to 1 m. The radio block's Murphy+2011 free-free
  term defaults to off when the nebular backend carries a free-free continuum
  (implementation: `interp_continuum_with_freefree_tail` in
  `tengri.components.nebular._shared`, index `NEBULAR_FREEFREE_TAIL_ALPHA_NU =
  -0.1` in `tengri.components.nebular._constants`, wavelength ceiling
  `NEBULAR_CONTINUUM_WAVE_MAX = 1e10` Å, auto-rule
  `nebular_backend_carries_freefree` in `tengri.components.nebular._models`),
  so the default Cue/CloudyGrid + radio model carries exactly one thermal term
  (was 1.64 to 1.71 times the correct total between 1 mm and 1 cm, and zero
  beyond 1 cm). Explicit `radio.sf.freefree: True/False` always overrides;
  CB19 and baked-in SSP are unchanged. Validation against pcigale surfaced
  this. Closes #2346.

- **Docs**: `CalibrationELineMarginalizedLikelihood` states in the class
  docstring that the emission-line block is handled via a plug-in point
  estimate rather than marginalized, that the log-determinant volume term is
  not included, and that this understates the uncertainty the line amplitudes
  contribute. The exact nesting of the two marginalizers is a separate design
  decision. (#2354)

- The `lognormal` SFH's entry in the `mean_sfh` module index described it as a
  Gaussian in log10(age). The function is a lognormal in cosmic time since
  formation, `T = age − t_lookback`, with a 1/T Jacobian and
  `sigma = width × ln(10)` — age and cosmic time since formation run in
  opposite directions, and the one-line summary contradicted the function's
  own docstring.


### Deprecated


- Attenuation-law keyword `n_slope` is renamed `dust_slope` on `power_law` and
  `conroy2010`, so the law keyword equals the registry name (`dust_slope`) and the
  grammar stem (`slope`) for every shape parameter. `n_slope=` still works on the
  public law functions with a DeprecationWarning; the registry callables, the
  law-kwarg resolver and the per-screen override dicts (`dust_law_overrides`,
  `bc_law_overrides`, `neb_law_overrides`) use `dust_slope` only.


- `SEDModel.from_config(dust=...)` / `build_model_from_config(dust=...)`:
  renamed to `dust_attenuation_law=...`. `dust=` still works and forwards to
  `dust_attenuation_law`, but emits a `DeprecationWarning`; passing both with
  disagreeing values raises `ValueError`. `dust=` will be removed in a later
  release (#2021).


### Removed

- `DIGNotOnNebularGridError` and the refusal it backed
  (`nebular_grid_precompute._refuse_active_dig_mixing`). Building the per-Q_H
  nebular grid with an active `neb_dig_frac` no longer raises: the grid now
  reconstructs DIG mixing via two lookups instead (see `### Added`, #2222).


- The `stellar` build group (#1720). Metallicity is now configured through
  `met`, parallel to `sfh`: `stellar={'met_mode': 'table'}` becomes
  `met={'type': 'table'}`, and `stellar={'met_logzsol': …}` becomes
  `met={'logzsol': …}`. **Breaking, with no alias** — a build-group key is
  parsed rather than imported, so accepting both spellings would mean carrying
  two grammars through `parse_groups`, `to_groups()`, the provenance tags and
  every wildcard sweep, which is the duplication the change removes. `stellar=`
  raises carrying the translation, because `difflib` will not suggest `met` for
  `stellar` — they share no prefix. Two anomalies stacked in the old form:
  every other group selects its variant with `type` while `stellar` alone used
  `met_mode`, and the group was named for the component rather than for what it
  configured — so `met={'type': 'table'}`, the spelling both conventions imply,
  was the one form the grammar rejected. `tengri.list_metallicity_modes()` is
  the live menu; the before/after table is in
  `docs/dev/api_migration_v0.x.md`.

- Toy AGN registered models `"simple"` (`simple_agn`) and `"standard"`
  (`standard_agn`). Both were modified-blackbody-based demo models flagged
  with once-per-process warnings; the science path remains the
  Kubota & Done 2018 models (`"multicolor_agn"` = deprecated alias
  `"kubota_done"`, `"kubota_done_full"`) and the SKIRTOR / Silva+04 /
  CAT3D-Wind / RELAGN templates.

- `builders.agn.simple()` and `builders.agn.standard()`. They named the two
  toy models deleted above, so the configs they produced raised at predict
  time; `_TOP_LEVEL_MODELS` was a hand-written list, which is also why
  `richards2006` and `skirtor_stalevski` had no factory at all. The factory
  set is now derived from `monolithic_agn_model_names()` — the same registry
  the build-time check validates against — so it gains those two and
  `builders.agn.available()` is pinned to equal it. Migration:
  `builders.agn.simple()` has no successor; pick a registered model from
  `available()`, or use the composable grammar.

- Public re-exports of `simple_torus` and `two_temperature_torus` from
  `tengri.components.agn`. The functions remain importable from
  `tengri.components.agn.torus` for the production models that still
  call them internally (`multicolor_agn`, `kubota_done_full`, `adaf`
  `relagn`) — see #233 for the planned IR-torus substitution.

- Demo examples `examples/agn/plot_agn_polar_dust_temp_sweep.py`,
  `plot_agn_templates.py`, `plot_polar_dust.py`,
  `plot_torus_comparison.py` and the corresponding
  `docs/auto_examples/agn/` artifacts. They used the deleted toy AGN
  public surface; the SKIRTOR-based examples (`plot_agn_cos_inc_sweep`,
  `plot_agn_oa_sweep`, `plot_skirtor_variants`, etc.) remain.

- `tests/contract/test_torus_deprecation.py` (the warn-once contract
  test for the now-private toy torus functions).

- `tests/components/agn/test_simple_agn.py` and `test_standard_agn.py`.

- The `dust` build group (#2000). Attenuation and IR emission are now peer top-level
  groups: `dust_attenuation={...}` (type, `law` or `law_bc`+`law_diff`, and the
  `tau_*`/`Rv_*`/`delta_*`/`slope_*`/`bump_strength_*` params) and
  `dust_emission={...}` (type, `eta_balance`, params). The nested
  `dust_attenuation={'emission': ...}` form is retired with it. **Breaking, with no
  alias** — `dust=` raises carrying the translation. Energy balance moved to the
  emission group as `dust_eta_balance` (default `Fixed(1.0)`, strict balance:
  `L_IR = eta * L_absorbed`). Before/after table in
  `docs/dev/api_migration_v0.x.md`. **Note:** readers migrating from before #1989
  encounter both changes at once; the renamed group is *also* now subject to the
  explicit-law rule.


### Changed

- **Params dicts are free-only; every entry point refuses a Fixed key
  (#2296; breaking change).**
  ``model.predict(params)`` and every prediction surface now refuse a `params`
  key the spec declared ``Fixed``, raising ``ParameterError`` naming the key,
  the pinned value, and the remedy. ``Parameters.sample(key)`` and
  ``Posterior.params`` / ``.samples`` now carry free parameters only, not Fixed
  ones. Fixed values are accessible through ``spec.get_fixed_values()`` or the
  new ``Posterior.fixed_values`` property (which reflects any
  ``Fitter(params_override=...)`` re-pin actually used by the fit, not just
  the spec's declared value). This closes the silent physics error
  where a Fixed-key override was honored on some specialized paths
  (FeaturePrecomp) and dropped on others (exact), producing stealthily different
  physics. To pin a *different* value for one fit or one galaxy, the
  sanctioned route is still ``Fitter(params_override={...})`` (validated at
  construction to name only Fixed parameters) or ``CatalogFitter``'s per-galaxy
  redshift override — both unaffected by this refusal.

  Neighbors of the same fix, registered here rather than as separate entries:
  - A mirror target (e.g. ``neb_logZ_gas="met_logzsol"``) present in ``params``
    is refused when its value differs from its resolved source; a value equal
    to the source (what ``sample()`` produces) is still accepted.
  - ``Catalog.from_histories`` refuses a Fixed ``met_gas=``/``redshift=`` at
    construction, not lazily inside ``predict()``/``simulate()``.
  - ``Posterior.fixed_values`` excludes any name free on the user's model,
    closing a leak where a ``profile_mass``-pinned mass appeared in both
    ``params`` (its real value) and ``fixed_values`` (a stale placeholder).
  - ``PopulationFitter`` requires its two population-shared PSD names
    (``sfh_field_psd_sigma``, ``sfh_field_psd_tau_myr``) free on
    ``model_factory``'s spec, and refuses construction otherwise, naming the
    remedy (breaking change for any ``model_factory`` that pinned them Fixed;
    ``SEDModel.fit_population``'s own factory is updated to match).
  - Three idioms are refused the same way everywhere they were found:
    ``spec.get_fixed_values()`` spread into a params dict,
    ``{**dict(spec.sample(...)), "<key>": value}`` sweeping a Fixed key, and
    an explicit Fixed key restated at its own pinned value. Repair recipe:
    declare the swept/restated parameter FREE in ``SEDModel.build`` instead,
    and pass only the free (swept) keys.
  - Call sites that restated a pinned value in the dict, or swept a pinned
    parameter through it (gallery examples, spine notebooks, slow-tier
    fixtures), declare the swept parameter free and pass only the free keys;
    a fixture that mocks the model may need the same.

### Fixed

- **FeaturePrecomp docstring now states the line-flux accuracy it was measured to (#2376).** The line LUT reproduces measured line fluxes to 4.8e-5–1.0e-3 relative; against typical 5% line errors that is ≲0.002 σ. This accuracy is now documented in the class docstring where a user chooses the approximation.

- `enable_fast_nebular` now refuses when a CB19 optional parameter
  (`neb_log_nH`, `neb_co`, `neb_dno`, `neb_hbfrac`) is freed. The per-Q_H
  grid bakes these axes at their reference values and cannot respond to the
  sampler's variations, producing a silent mismatch: the likelihood never
  observes the freed dimensions while the posterior reports only the prior.
  Mirror the CB19 flat-axis guard (issue #2181) to refuse at build time,
  naming the offenders and the remedy (pin them or skip fast-nebular). (#2307)

- `BakedInBackend` now checks whether the SSP grid has nebular emission before
  silently returning zero nebular flux. On bare-stellar grids
  (`ssp_data.nebular == "bare"`), it raises `BakedInNebularBareError`
  immediately. On unstamped grids (`ssp_data.nebular == "unknown"`), it emits
  `BakedInNebularGridWarning` naming `tools/stamp_ssp_nebular_attrs.py` for
  disambiguation. The grid-status warning is a `BakedInNebularWarning` subclass
  and honors `suppress` and an explicit `neb` declaration; the bare-grid
  refusal does not fire when nebular emission is off (#2362).

- Both unwired guards are wired and the class is closed (#2326):
  `tools/check_harness_parity.py` (benchmark-fixture provenance) and
  `tools/check_docs_voice.py` (the enforcement `NAMING_CONTRACT.md` names for
  confusable codepoints; warn-only until a docs cleanup clears its 24 standing
  findings) now run in CI, and `tools/check_ci_pr_coverage.py` fails on any
  `tools/check_*.py` invoked by no file under `.github/workflows/` — scanning
  every workflow, since three guards run from `docs.yml`/`notebooks.yml`.
  Rot found on restoration, fixed: a stale `%%`-block entry in
  `check_docs_voice.py` named a checker that does not exist, so reaching it
  raised `NameError` instead of reporting (the code-cell path at its real
  call site is untouched).
- A bare `CB19Backend` (without `ssp_data`, so `_lum_scale ≈ 1.25e-46`) returned silently exact-zero gradients for every grid-axis parameter (`neb_co`, `neb_hbfrac`, …) while forward values changed correctly. Traced to the interpolation coordinate being cast to float32 by `_frac_idx` and the tiny cotangent underflowing to exactly 0.0 in the backward pass. The fix keeps the coordinate in float64 throughout (matching the #1568 data-protection pattern), so the cotangent stays in range. On the normal `SEDModel.build` path (with `ssp_data`), the float64 forward outputs move by ≈2e-8 relative (max 1.9e-8 measured on the AGN-inventory build; the regression pin is rtol=1e-7, five times that): the old float32-rounded coordinate had been rounding the interpolation weights, so this is a small precision gain, not a change of model. Not reachable via `SEDModel.build` (which always supplies `ssp_data`); the hazard was direct-backend use only. (#2306)

- The offline filter remedy is now a command that runs. `load_filter`'s
  network-unavailable error hands the user one instruction, and it was wrong
  three ways at once: it named `tools/download_filters.py` while the script
  lives under `scripts/`, it passed the filter name positionally where the
  script requires `--filter NAME`, and the script carried its own copy of the
  registry behind a "keep in sync" comment that had drifted to 250 of 431
  entries — ALHAMBRA, J-PAS, J-PLUS, SHARDS, HAWK-I and SkyMapper were all
  absent, so for 181 filters it would have refused the name even when invoked
  correctly. `scripts/download_filters.py` now reads
  `src/tengri/observation/data/filters_registry.json` with stdlib `json`,
  which keeps the constraint the copy existed for ("avoid importing tengri so
  the script works in bare envs" — reading a data file is not importing the
  package) while removing 288 lines of duplicated data and the drift class
  with them. Three contract tests in `test_filter_offline_mode.py` now assert
  the recommended path resolves to a real file, that the command carries the
  flag the script requires, and that the script holds no inline
  alias-to-SVO-id pairs, so a reintroduced duplicate goes red. A
  recommendation living in an f-string is executed by nothing, which is why
  none of the three had anything to report it.
- `test_bma_weights_ranking_agreement` now uses a shared mock fixture so both
  models score identical data. Model B is model A with `dust_tau_diff` pinned
  at 1.5 (measured ΔlogZ ≈ 6–7 nats across NSS, HMC+IS, Laplace), designed to
  separate decisively. The ranking guard fails (not skips) on separation loss
  below 2σ, treating fixture regression as a test failure (#2364).

- **`check_render_diagnostics.py` enumeration via git ls-files (#2315, #2050 drift-proofness).** The guard now uses `git ls-files` instead of filesystem globbing to enumerate notebooks, matching CI enumeration and ensuring untracked local renders (e.g., from interrupted notebook restarts) cannot fail a local pre-push run that CI would pass. This prevents users from dismissing the guard as unreliable when a branch touching no notebooks goes red due to stale renders on disk — both local and CI verdicts now depend only on tracked state. Raises (documents sibling behavior) when run in a `git archive` export. Companion tests added.
- Radio preset rows kept their buildable composable `use` and carry the
  not-builder-available marker in `short_doc`, so the menu's `name` column types
  and every production row's `use` is built by a contract test. The marker is
  defined as a module constant; `list_sfh_models` uses it at a second site. The
  agn-block `use` strings carried trailing whitespace that made the generated
  component tables invalid RST; check_component_page.py now parses the generated
  fragment with docutils (#2201).

- ``check_literal_param_defaults.py`` (the CI guard that prevents bare literals
  from standing in for declared parameter defaults) had two blind spots, both
  fixed: it was scoped to ``dust/emission/`` only, and it never saw negative
  or explicit-positive defaults at all -- ``ast.parse`` renders ``-3.0`` as
  ``UnaryOp(USub, Constant)``, which the bare ``ast.Constant`` gate skipped.
  The no-argument run now covers every swept tree (``dust/emission``,
  ``radio``, ``stellar``, ``igm``; the rest join as #2297's rulings land),
  and all closure defaults in radio, stellar, and igm read their values
  through ``declared_default(...)`` or a shared named module constant rather
  than repeating them as bare numerals. Zero-diff probes over radio and
  stellar confirm bit-identical output. One real slip surfaced by the sweep:
  igm's two ``dla_log_n_hi`` fallbacks said 20.0 where the declaration says
  20.3 (``DEFAULT_DLA_LOG_N_HI``, which both sites now read). That changes
  DIRECT calls that omit the argument with ``use_dla=True`` -- measured max
  relative difference up to ~1.0 at z=2.0 in the damped wings, smaller at
  other redshifts -- and changes nothing on the grammar path, which always
  supplied 20.3. Part of #2265.
- (#2363) The slow inference tier runs as three file-partitioned legs, each
  printing a pytest summary and its 30 slowest tests inside its budget; a
  labeled PR whose slow or crossval job is skipped fails ci-ok instead of
  reading as approval.

- The energy-balance-split closure's docstring tagged its luminosity arguments
  `L_absorbed_stellar` and `L_agn_ir` as `[Lsun]`, while the component path
  supplies both in `erg/s` (component_factory.py:346). The docstring is now
  unit-agnostic ("as passed"), with a note that both arguments must share the
  same units, and the `dust_L_agn_ir` parameter declaration now explicitly
  states `units="erg/s"` (#2251).

- (#2386) The coverage jobs carry the per-test `--timeout=600
  --timeout-method=thread` again and their own budgets (1.5× the test
  shard's), so a hung test is named instead of an anonymous budget kill; the
  two `Resolve test paths` steps are one script, `tools/ci_split_paths.py`,
  whose empty-list floor is fixed.

- `finalize_profile_mass` reinserts the marginalized mass through one `jax.jit`
  program cached on the model per engine key (draws, keys, data, noise and
  presence traced), instead of an eager `jax.vmap` over every draw that
  dispatched each forward prediction op-by-op and re-traced per fit: 1.2 s ->
  0.4 s warm and 5.5 s -> 2.0 s cold on 1200 draws of a 14-band model, and no
  `(n_draws, n_pixels)` spike on spectroscopy models. With it, `forward.fit(data)`
  pays what the bench harness pays per gradient on every phase; `run_nuts` now
  reports `n_grad_adapt` / `n_grad_sample` / `n_grad_total` so the comparison is
  made in gradients. `benchmark_laplace_nuts_20s.py` builds its context with
  `profile_mass=False` again (its own marginalization needs the full-D
  context; under the `"auto"` default every `--profile-mass` row died), and
  `test_nuts_split_warmup_keeps_sampling_quality` asks for 400 draws (the
  profiled fixed-key realization sat on its 1.1 R-hat bar at 200)
  (`bench/reports/2026-09-12_library_path_parity.md`).
- DH02_CE01 template shape and float32 safety (#2366): the Dale & Helou (2002) /
  Chary & Elbaz (2001) template library is indexed by log₁₀(L_TIR/L_sun) and
  the whole point of that family is that the SED shape correlates with
  luminosity — warmer, broader templates at higher L_IR. The closure carried a
  hardcoded `dust_log_lir=10.0` default and inherited `factors_l_ir=True` from
  the base class, so ``apply()`` always evaluated it at unit luminosity
  (``log10(1) = 0``), pinning the shape lookup to a single grid node regardless
  of the actual budget; only the amplitude was rescaled afterwards. The template
  shape therefore never tracked the fitted luminosity. The normalization path
  also materialized L_absorbed (~1e43 erg/s, inf in pure float32) as a linear
  value, leaving the SED entirely inf/NaN. Fixed by setting `factors_l_ir=False`
  on the component (matching BosaIRSEDComponent), declaring
  `optional_inputs={'log_L_ir': 'dex'}`, and rewriting the closure as a SINGLE
  code path (no dtype-gated branches) that: (1) uses live `log_L_ir` for the
  grid-axis lookup after converting to L_sun (same precedent as #2272/#2273),
  and (2) applies log-domain rescaling via ``apply_log10_scale()`` when
  `log_L_ir` is provided, never materializing L_absorbed. Mirrors
  bosa_emission exactly — one arithmetic path works in both float32 and float64.
  The model's total power still integrates to the absorbed luminosity exactly;
  the shape now varies appropriately with the fitted L_IR. **Model output
  changes for every dh02_ce01 fit** (#2366).

- The analytic dust-emission closures (``modified_blackbody``, ``graybody``,
  ``casey2012``, ``schreiber2016``, ``energy_balance_split``) read their
  signature defaults from the declared parameter table
  (``declared_default(PARAMS, ...)``) or a shared named constant instead of
  repeating the value as a bare literal; no default value changes (a
  zero-diff probe over every closure at only-defaults confirms bit-identical
  output before/after). ``EnergyBalanceSplitIRSEDComponent.predict`` now
  subscripts ``p["f_cold"]`` and five siblings instead of falling back to a
  stale ``.get(name, literal)`` default, so a hand-built params dict missing
  a key raises ``KeyError`` naming it rather than silently substituting the
  literal. A new stdlib-only guard, ``tools/check_literal_param_defaults.py``,
  scans ``src/tengri/components/dust/emission/`` for a bare numeral standing
  in for a name a ``ParamDeclaration`` or component class attribute already
  owns, and is wired into the same CI job as ``check_param_defaults.py``.
  ``dust_T``/``dust_beta_ir`` disagree between the shared
  ``components/dust/_params.py`` table and every analytic template's own
  default; that disagreement is left as-is and tracked separately (#2261)
  (#2241).

- Unknown dict keys now name the type's accepted parameter short names (#2176):
  when a user writes an unknown key like `sfh={'type': 'delayed', 'zzz': 1.0}`
  with no close difflib match, the error message now includes the type's
  parameter short names (e.g., "Parameter names this type accepts: tau_gyr,
  age_gyr"), so the user sees what they can write instead of only the structural
  keys. For types with many parameters (> 12), the message points to
  `tengri.describe('<type>')` instead of listing them.

- Dust tree literal defaults aligned with declarations (#2265):
  ``tools/check_literal_param_defaults.py``'s scope widens from
  ``dust/emission/`` to the whole ``dust/`` tree and reports zero
  literal-copy sites (previously 44); every default and ``.get`` fallback
  now reads ``declared_default(...)`` or a named module constant. The
  shared table's ``dust_beta_ir`` (and its free-prior default) is
  corrected ``Fixed(1.6)`` -> ``Fixed(1.8)`` to match the three analytic
  classes that already declared 1.8 (``Casey2012IRSEDComponent``,
  ``GraybodyIRSEDComponent``, ``ModifiedBlackbodyIRSEDComponent`` --
  ``Schreiber2016AnalyticIRSEDComponent`` declares no ``dust_beta_ir`` and
  its closure pins beta=1.5 internally); on a fixture with far-IR bands
  this reaches the wildcard-Fixed graybody, modified_blackbody and
  casey2012 builds, up to 3.3% (max relative difference: graybody
  3.319e-2, modified_blackbody 3.210e-2, casey2012 8.921e-3), every other
  grammar-path build measured bit-identical. ``schreiber2018_tabulated(dust_T)``
  30.0 -> 25.0 and ``astrodust_emission(dust_qpah)`` 3.0 -> 2.5 now match
  their declarations (component class; shared table -- astrodust's grid
  has no qpah axis). ``kriek_conroy``'s ``dust_bump_strength=1.0`` and
  ``tea``'s ``dust_delta=-0.2`` keep the laws' own citation-backed values
  as named constants: the #1833 ``live_shape_params`` gate hands a
  wildcard caller the law's own default, not the shared table's
  structural-off 0.0. ``dust_T`` stays ``Fixed(35.0)``, left unchanged
  pending #2261, with per-class constants for the three that disagree;
  ``dust_lgU``'s table/class disagreement is tracked separately (#2261).

- AGN attenuation did-you-mean routes law-form names to the law form (#2201):
  when a user writes an invalid AGN atten type like `agn={'atten':
  {'type': 'prevot'}}`, difflib may suggest `smc_prevot` (a law-mapped type).
  Before the fix, the error message suggested using `type='smc_prevot'`, which
  itself would be refused with "no longer supported. Use the law form instead"
  (two hops to the same fix). The message now suggests the law form directly:
  `law='prevot_smc'` (one hop).

- `vmap_chunked`'s jittability probe caught only `ConcretizationTypeError`,
  believing it the base of the `Tracer*ConversionError` family. On jax
  0.11.1 that belief is false: `TracerArrayConversionError` (raised by
  `np.asarray` on a tracer) and `TracerIntegerConversionError` (raised by
  `operator.index` on a tracer) are siblings of `ConcretizationTypeError`
  under `JAXTypeError`, not subclasses, so a mapped function that inspects
  its input with `np.asarray` raised through the handler instead of
  falling back to the eager per-draw loop. The handler now catches the
  whole family explicitly (#2264).

- The nebular component's DIG mixing no longer evaluates the DIG branch when
  the spec pins ``neb_dig_frac`` at the declared ``Fixed(0.0)`` default. A
  build-time predicate ``_dig_may_be_active(spec)`` resolves to a frozen
  ``dig_active`` config field, threaded to all seven mixing call sites (exact
  path: cue + cloudy/cb19 continuum/lines; grid path: photometry + restband
  reconstructions in ``apply``, plus ``predict_line_fluxes``'s own
  line-luminosity reconstruction call), so a default model evaluates the
  nebular backend once per channel instead of two, both the exact and the
  fast-grid path. When ``dig_active=False``, the mixing core skips the second
  evaluation entirely, returning the HII result unconditionally, not a
  zero-weighted one. Measured gradient FLOPs of ``predict_photometry`` on the
  #2195 fixture: the declared default is 147,434,528 against 159,926,608
  forced active (159,926,608 / 147,434,528 = 1.085x), well short of a flat
  50% -- the removed DIG evaluation is a small share of a
  photometry gradient once dust attenuation and emission are in the graph, so
  the saving scales with how much of the graph the nebular backend is, not a
  fixed fraction. Because ``dig_active`` is resolved once from the spec at
  build time, a call-time override of a spec-pinned ``neb_dig_frac`` (e.g.
  passing a nonzero value through ``params`` at predict time) is not honored
  on this path; declare the fraction ``FREE`` or ``Fixed`` at the intended
  nonzero value instead (#2296) (#2262).

- Metallicity-history bins are now refused at build time when unreachable at the model's
  redshift (issue #2204): the fixed z=0 lookback ladder (_DEFAULT_MET_BIN_EDGES_LOG_YR
  spanning 1 Myr–13.8 Gyr) becomes unreachable at high redshift where cosmic age is
  younger than a bin's lower edge (start in lookback time). When `met={'type': 'bins'}`
  or `'bins_continuity'`, `SEDModel.build` now checks that each bin's lower edge fits
  within `age_at_z(z_floor)`, where z_floor is the lowest redshift the prior admits
  (the fixed value for Fixed, the minimum for a free Uniform prior). A bin is unreachable
  only when its lower edge exceeds cosmic age at z_floor — reachability is judged by each
  bin's start at the lowest admitted redshift. Raises `ParameterError` naming the
  unreachable bins [start, end] and cosmic age. The refusal message points users to the
  actual remedies: use a lower redshift where all bins are reachable, or use a different
  metallicity mode. The bin ladder is not yet configurable through `SEDModel.build()`
  (see issue #2433 for future support). Bins older than the universe silently become
  identically inert (zero gradient, flat direction in the sampler) until checked; the
  new guard makes them fail loudly at build time with guidance. The docstring claim in
  `metallicity_history.py` that the bins mode pairs with the continuity SFH model
  (different bin-edge sets) is now corrected.

- `neb_hbfrac` was silently inert at any value: declared as a CB_19
  parameter, but `CB19Backend.__init__`'s `hbfrac` constructor argument was
  never threaded from `params`, and `load_cb19_grid` collapsed the HbFrac
  axis to a single slice at load time regardless. `load_cb19_grid` now
  retains both HbFrac nodes; `predict_nebular_line_luminosities` /
  `predict_nebular_sed` interpolate `neb_hbfrac` at runtime via the same
  `map_coordinates` scheme as `neb_log_nH` / `neb_co` / `neb_dno` (linear
  over the grid's two nodes -- the only interpolant they support); and
  `_BACKEND_OPTIONAL_PARAMS` threads it from `params` exactly like those
  three siblings. `cb19_precompute.precompute` (the `WavePrecomp` adapter for
  `neb={'type': 'cb19'}`) keeps HbFrac a discrete, load-time-style choice for
  that surface, selecting the nearest retained node itself immediately after
  loading (#2213).

- BOSA's dust-emission template **shape** was silently pinned at the
  unit-luminosity template regardless of the fitted luminosity. BOSA
  (Boquien & Salim 2021) interpolates its template library on a
  `(log L_TIR, log sSFR)` grid, so which row gets selected is itself a
  function of the absorbed luminosity -- not just the overall normalization.
  `BosaIRSEDComponent` never overrode `EmissionComponent.factors_l_ir`
  (default `True`, unlike `energy_balance_split`, which does), so the
  generic `apply()`-level speed shortcut always evaluated `predict()` at
  `L_ir = 1` and rescaled the result afterwards -- correct total power,
  wrong shape, always the same shape, across any luminosity range.
  `factors_l_ir` is now `False` for BOSA, so `predict()` sees the real
  budget and the grid lookup selects the luminosity-appropriate row. This
  also required threading a float32-safe `log_L_ir` [dex] input through the
  component and its closure (mirroring `energy_balance_split`), since the
  real linear `L_ir` (~1e43 erg/s) overflows to `inf` in pure float32 while
  its log does not. A second, compounding defect made the first fix alone
  insufficient at real galaxy scales: the packaged grid's axis is
  `log10(L_TIR / Lsun)` (Boquien & Salim 2021), while `L_ir` arrives in
  erg/s (the tengri-wide SED contract) with no conversion applied, so any
  astrophysically realistic `L_ir` (~1e42-1e45 erg/s) numerically saturated
  the grid's ceiling node regardless of the real budget. The axis lookup
  (only -- normalization stays in erg/s) now subtracts `LOG10_L_SUN`
  (`tengri.utils.sed_quantities`, the `dust_log_L_ir` precedent) from the
  erg/s log budget, so the template shape now tracks the fitted L_TIR across
  the grid's full Lsun-relative span at real galaxy luminosities, not only
  in the abstract (#2272).

- Importing a submodule through an aliased package spelling
  (``from tengri.sps.dsps_wrapper import ...``) re-executed the module file:
  two module objects for one file in one process, each with its own
  module-level state (e.g. the SSP content-hash cache), the second execution
  overwriting the canonical package attribute. A meta-path finder now binds
  the existing canonical module object under the aliased name with no
  re-execution, in both import orders, for all nine component aliases;
  aliases stay lazy (no eager submodule imports at ``import tengri``)
  (#2256).


- Four AGN sites integrated over the descending frequency grid by reversing
  both trapezoid operands (``polar_dust.py``'s anisotropic polar luminosity,
  ``adaf.py``'s float32 and float64 normalization integrals, ``unified.py``'s
  disc L_bol). On Apple GPU via jax-mps under default MLX compile a reversed
  array beside a broadcast scalar is silently zeroed past element 0
  (jax-mps#232), and ``jnp.trapezoid`` multiplies by 0.5 internally, so the
  torus lost its far-IR graybody entirely (measured x0.067 at 100 um in the
  Herschel 250 band). Each site now integrates over the descending ``nu``
  directly and negates the scalar result -- float64 moves only by summation
  order (measured <= 2.2e-16 per site), MPS forward probes and the recorded
  gradient probe match CPU exactly, and a source scan forbids reversed
  trapezoid operands anywhere in ``src/tengri`` (#2295). The Apple-GPU
  recipe's ``MLX_DISABLE_COMPILE=1`` rule stays until jax-mps#232 closes:
  VJPs elsewhere still emit ``lax.rev``.

- The nebular component's four DIG-mixing call sites (cue continuum, cloudy/cb19
  continuum, cue lines, cloudy/cb19 lines) now call the one implementation in
  ``dig.py`` -- ``mix_dig_emission`` for the continuum, ``mix_dig_line_luminosities``
  for lines -- instead of each carrying its own copy of the
  ``(1 - f) * HII + f * DIG`` mixing arithmetic. Backend kwargs (e.g. Cue's
  resolved ionizing population) now reach both the HII and DIG evaluations
  from one frozen dict, and the inline snapshot-before-mutation copy that let
  #2195 ship behind twenty green unit tests of the unused ``mix_dig_emission``
  is gone. Output is bit-identical to the prior inline arithmetic: measured
  max absolute difference 0.0 across ``predict_photometry``, ``rest_sed()``
  and line luminosities, both the cue and cb19 backends, at ``neb_dig_frac``
  in ``{0.0, 0.3, 0.9, 1.0}`` (#2221).


- ``neb={'type': 'cb19', 'grid': <path>}`` now reaches the cb19 backend as
  ``nebular_cb19_grid_path``, the way the ``cloudy`` and ``mappings`` ``neb``
  types' own ``grid`` keys already did, and the path now round-trips through
  ``spec.to_groups()`` instead of silently reverting to the packaged default
  on re-parse. Before, the cb19 branch of the grammar never read the key and
  the path vanished without an error, so the only route to a non-default
  grid was the module default. ``grid`` is refused by name on every ``neb``
  type that never reads it (``cue``, ``ssp``, ``none``) instead of being
  silently accepted and dropped; the three cb19 refusal messages now name
  the key (#2220).


- `marginalize_emission_lines` no longer crashes float32 geoVI on CUDA. Its
  `(n_lines, n_lines)` normal-equation GEMM (`g.T @ g`, degenerate at the
  handful of emission lines this is ever called with) hit "GEMM is not
  supported by cublasLt and legacy cublas fallback is removed" under JAX
  0.11 whenever the operands arrived float64-valued and were traced under
  x64 disabled. Replaced with an explicit broadcast-multiply-sum, which
  never lowers to a GEMM; float64 CPU output is bit-identical to the matmul
  it replaced (rtol 1e-12) (#2023).


- A headline line property (`civ_1549`, from `KEY_LINES`) now warns instead of
  returning a silent NaN when the currently selected nebular catalog carries
  no entry within tolerance of its target wavelength; the warning names the
  property, the backend, the nearest catalog line and its offset in
  Angstrom, and the remedy (`neb={'type': 'cue', 'full_catalog': True}` when
  the backend is cue and on the legacy subset). Generalizes across every
  line-catalog backend (#2239).


- The #2239 warning seam's static catalog accessor
  (`_published_line_wavelengths_static`) now applies tengri's vacuum-wavelength
  contract (`nebular_line_waves_to_vacuum`, hoisted into
  `components/nebular/_shared.py` and shared with
  `NebularSEDComponent.apply`) before comparing against a `KEY_LINES` target,
  instead of comparing the backend's raw, sometimes-air catalog directly; the
  mismatch reached up to 2.70 Angstrom against the 5 Angstrom match tolerance
  (measured on cue's upstream, air-frame `.npy`), close enough to risk a false
  warning or a missed one for lines not already covered by the #2239
  regression test. `predict_photometry`, `rest_sed` and every already-tested
  headline line are unaffected (#2239).


- `log_L_ir` conflated the re-emitted IR budget with the ABSORBED
  stellar+nebular energy for three readers (`pred.l_dust_absorbed`, the
  legacy `predict_sed_quantities` bridge, and the AGN CIGALE fracAGN torus
  coupling), which was only silently correct at `dust_eta_balance == 1`.
  Every dust-attenuation publisher now also publishes a `log_L_absorbed` /
  `L_absorbed` companion pair invariant under `dust_eta_balance`, and the
  three readers are repointed to it -- a relaxed `dust_eta_balance` no
  longer leaks into the absorbed-energy reading (#2187-series).

- `SEDModel.enable_fast_nebular` now snaps each requested target wavelength
  within 0.5 Å of a true backend catalog line (read from
  `state.derived["line_waves"]` via one reference forward pass) to that
  line's exact wavelength, before building the per-Q_H grid. Previously the
  fast grid tabulated exactly the caller's (possibly imprecise, e.g. a
  rounded literature value or an air/vacuum slip) request, while the exact
  path evaluated the dust screen at the backend's true, nearest-matched line
  wavelength — two different points on the attenuation curve, up to ~0.1 Å
  apart. `tests/regression/bug/test_bug_2223_line_screen_kwargs.py::test_fallback_is_actually_exercised_by_fast_nebular`
  tightens from `rtol=2e-4` to `rtol=1e-10` now that both paths agree on the
  identical wavelength.


- The ``n_slope`` deprecated alias for ``dust_slope`` now survives registration in
  ``DUST_LAWS``. Swapped decorator order on ``power_law`` and ``conroy2010`` so
  ``@renamed_kwarg`` wraps the function before ``@register_dust_law`` stores it
  in the registry; the registry callable and ``list_laws()`` result now accept
  the alias with a DeprecationWarning instead of raising TypeError. Per-dict
  strictness is unchanged: ``select_law_kwargs`` and ``reject_unread_law_kwargs``
  still reject ``n_slope`` (only the callable wrapper accepts it) (#2257).


- The audit that came with `tools/check_gradient_assertions.py`: **279 test
  sites** across 140 files asserted half the finite-AND-non-zero rule and now
  assert both. 248 were the #2100 shape (finite, never non-zero) and 31 the
  #2178 shape (non-zero, never finite). No assertion was weakened to make the
  guard pass. 19 of the 279 carry the documented escape hatch
  (`# grad-assert: finite-only — <reason>`): they evaluate at a point where the
  derivative is zero for a reason. Some construct a degenerate input on purpose
  — a zeroed window, an empty band, zero ionizing flux, an exact `log10_add`
  cancellation, the Hessian-vector product of a linear scaling, a kernel
  evaluated outside its band. Others sit on a genuine stationary point or an
  inert direction: a prior's log-density differentiated at its own mode, a
  Student-t NLL differentiated at `sigma`'s own maximum-likelihood point, a GP
  field's PSD *correlation time* at an identically zero field (the timescale
  only colors the field, so with no field there is nothing to color). Either way
  zero is the correct answer there and only the finite half is a claim — and
  where the surrounding test's real claim was that a gradient *flows*, that
  claim is now stated a step away from the zero, where it can actually fail.


- Two further tests turned out to be measuring nothing, both found by the
  non-zero half of the rule and neither a gradient defect in `src/`:
  `test_forward_model_end_to_end_jit` took its model from a fixture that fixes
  *every* parameter, so `params` was `{}` — `all(... for g in grads.values())`
  is vacuously True over an empty dict, and the test promised "finite
  gradients" in its own docstring while taking none. It now builds a model with
  two free dust optical depths and asserts the precondition that a free
  parameter exists. `test_stochastic_gradients_finite` hand-rolled a loss that
  attached only `psd_xi`, but `StellarSEDComponent` reads `sfh_field_xi` —
  the exact trap `inference/loss_functions.py` attaches both keys to avoid, and
  says so in a comment. The latent field never reached the model: `psd_xi`'s own
  gradient summed to exactly 0.0 and `sfh_field_psd_sigma`'s was bit-identical
  for a zero field and a random one. A finite-only check cannot see that, because
  an identically zero array is finite. (Counts are what the guard reports when run over the upstream
  tree at the merge base: 279 across 140 files at `6cc1a8b25`, against 277
  across 139 at the previous merge base `850be10bc` — a delta of exactly the
  two sites `main` added since, both in `test_float32_scale_seam_sweep.py`,
  where the swept *forward* was pinned finite but never non-zero. An earlier
  revision of this entry said 276/137, which was not one of those
  measurements.)
  Two of the repaired sites are the historical bugs themselves:
  `test_inference_grad_float32.py` (still finite-only on `main`, which is how
  #2100 stayed invisible) and the `!= 0.0` seam checks in
  `test_float32_fitting_path_seams.py`.

  The count is **disjoint from #2171's sweep**: re-measured against `main`
  *after* that landed, this guard still reports the same 272 sites it reported
  before, because #2171 repaired a different defect (an assertion wrapped in a
  guard derived from its own subject, which declines to run) while this one
  repairs a predicate that runs and admits the undecided state. Complementary,
  not duplicative.


- `test_met_table_grad_wrt_lgmet` was **vacuous**, and the guard found it. It
  differentiated the total CSP mass with respect to `lgmet_table` and asserted
  only `isfinite`. The metallicity table chooses which SSP template each age bin
  draws from; it does not move mass between bins, so the total is *exactly*
  invariant and the gradient is identically zero — as is the finite-difference
  reference it was compared against, so the check compared 0 with 0. Measured:
  `total_mass` is `7942282347.242821693420` at `lgmet`, at `lgmet+0.5`, at
  `lgmet+2.0` and at `lgmet-2.0`, the same digits to the last one. The
  conservation is now the claim, stated positively, and a second assertion
  differentiates the *metallicity weights*, which the table does steer
  (measured `max|grad| = 25.7`, all 20 entries non-zero), so the test measures a
  gradient rather than a conservation law twice.


- `TestCmbContrastFactorBounds::test_gradient_safety_float64` was **vacuous**,
  and the guard found it. It differentiated `cmb_contrast_factor` at
  `T_eff = 25 K, z = 10` and asserted only `isfinite`. The z = 10 CMB floor is
  `2.725 x 11 = 29.98 K`, so at 25 K the factor is clamped to exactly zero at
  all 601 wavelengths and the gradient is `-0.0` — finite, and measuring
  nothing. Measured 2026-09-06: `sum = 0.0, grad = -0.0` there, against
  `grad = 2.6` at `T_eff = 50 K` on the same grid. The sub-CMB point is now
  pinned *as* zero (which is the correct physics) and a live point above the
  floor is pinned finite AND non-zero, so the test measures a gradient again.


- The flat `Parameters(...)` form refuses a dust shape parameter or a
  `dust_law_overrides` entry that the resolved attenuation law never reads, and
  an override screen other than `bc`/`diff`/`neb`, through the same validator
  the `SEDModel.build` grammar uses; before, `Parameters(dust_law_bc="calzetti",
  dust_Rv=Fixed(4.0))` built and `dust_Rv` silently never reached the model.
  Registry defaults on omission are unchanged. Inside `dust_attenuation={...}`
  the full registry spellings (`dust_tau_bc`, `dust_law_bc`) are normalized to
  the grammar stems before any check runs, so `tau_bc` plus `dust_tau_diff` no
  longer trips a false completeness error and two spellings of one key raise;
  conversely a lone `dust_tau_diff` in a two-component group is now refused
  exactly like a lone `tau_diff` (the full spelling used to bypass the check),
  so pin or free `tau_bc` explicitly next to `**narayanan_tau_prior(z)`.
  `with_params()` and `merge_observation_params()` carry a fresh provenance map,
  so a shape parameter merged into a flat spec is live. The twelve per-screen
  grammar keys derive from one constant (`tengri.parameters._dust_keys`), and
  `check_dust_law_kwargs.py` checks law keyword spelling at every call site in
  `src/`, `tests/`, `bench/`, `examples/` and `analysis/`. Test, bench and analysis call sites that pinned `dust_slope` beside a law that never reads it drop the dead kwarg (`power_law` keeps its registry default of -0.7), and one engine-cache test that varied `dust_Rv` now does so under `cardelli`.


- LogNormal, StudentT and Laplace derive their truncation flag from the distribution's natural support instead of from CDF values that underflow beyond ~8 sigma, so a far finite bound is no longer silently ignored in latent space; Gaussian shares the same rule via Distribution._is_truncated (#2233).


- The no-state emission-line dust screen (`SEDModel._attenuate_line_catalog`,
  used when `dust_model` is `off`/`wg00` and by the #950
  `enable_fast_nebular()` grid path) built its own law kwargs from exactly
  `dust_slope` and `dust_bump_strength` via `emission_helpers.attenuate_emission`,
  so `dust_delta` (kriek_conroy, salim, noll09, salim_sbl18, tea), `dust_Rv`
  (cardelli, conroy2010) and `redshift` (narayanan_z) reached the CONTINUUM
  screen but not the LINE screen, and per-screen overrides
  (`slope_bc`/`slope_diff`) and `dust_f_obscuration` reached neither the
  Lyman clip nor the covering fraction on the line side at all. It now
  dispatches to the dust component's own `attenuate_line_catalog`
  (`DustSEDComponent` / `DustAttenuationSEDComponent`), the same method the
  live forward pass calls for its continuum, so there is exactly one
  implementation of the two-component line screen. `attenuate_emission` is
  removed; it had no public callers left (#2223).


- A flat `Parameters(...)` spec that freed or pinned a dust attenuation shape
  parameter (`dust_slope`, `dust_delta`, `dust_Rv`, `dust_bump_strength`) had
  the forward model never read it: `SEDModel._requested_law_shape_params`
  decides which shape parameters are "live" from `spec._group_provenance`,
  the richer map `parse_groups` attaches after construction, and a flat spec
  never gets one, so every name resolved to `"registry_default"` and the
  attenuation law silently evaluated its own published default no matter
  what the flat spec declared — the parameter still appeared in
  `free_params` and sampled a posterior that was exactly its prior.
  `Parameters.__init__` now records a `_flat_provenance` map (distinct from
  `_group_provenance`, so `parse_groups`, `translate.py`'s
  `legacy_flat_spec` gate, and the flat-form `summary()` are all unaffected)
  for every parameter the constructor call actually named, by presence in
  the call rather than by comparing against a default — an explicit value
  equal to a law's own published default (e.g. `dust_bump_strength=Fixed(1.0)`,
  KC13's own value) is still a request. Measured: `dust_bump_strength`
  0.0 -> 3.3 on `single_component` `kriek_conroy`, `galex_nuv` relative
  change 0.00% -> 17.04%, matching the equivalent `parse_groups` build to
  `rtol=1e-10` (#2231).


- `SEDModel.compile_signature()` did not key on which dust attenuation shape
  parameters (`dust_slope`, `dust_delta`, `dust_Rv`, `dust_bump_strength`) a
  build resolved "live", so two structurally-identical models that disagreed
  only on liveness collided on one compiled, closure-captured prediction
  kernel — whichever was built (and called) first silently decided the
  live/not-live branch for both. Before the #2231 fix above this axis was
  unreachable from a flat `Parameters(...)` spec (its shape parameters were
  always not-live), so the collision could not fire from that surface; that
  fix is exactly what exposes it, since a flat spec can now resolve a shape
  parameter live. `compile_signature()` now includes the sorted set of live
  shape-parameter names (`dust_live_shape_params_sig`), same rationale as the
  existing `dust_law_overrides_sig` / `dust_lyman_cutoff_sig` color-leak
  entries. Measured end to end: a not-live `kriek_conroy` build followed by a
  live one with `dust_bump_strength` overridden to 3.3 via the same
  `params` dict previously reported identical `galex_nuv` photometry
  (0.00% difference, the not-live kernel silently reused); with the fix the
  two differ by 12.06% (#2231).


- The `optimization_barrier` that PR #2194 put on `_mass_scale_lnu`'s primal
  costs neither memory nor time, measured rather than assumed. On the
  `spec/lut` seam at a realistic `(n_age, n_wave) = (93, 4096)`, four
  interleaved before/after repetitions on an otherwise idle box (1-minute load
  average stamped per run, 0.46 to 3.6): XLA's own compiled-memory analysis is
  **byte-identical** in both arms — `temp` 1.558 MB (forward) and 4.701 MB
  (forward+gradient) in float32, 3.115 MB and 9.396 MB in float64, with
  `output`, `argument` and `alias` zero throughout — and peak process RSS is
  2481.5 MB without the barrier against 2495.2 MB with it, a 0.55 % difference
  dominated by the SSP load and the model build rather than the kernel. Forward
  wall time is 1.040 ms against 1.034 ms; forward+gradient is 3.083 ms against
  3.095 ms, a 0.4 % difference inside a per-arm spread of 22 %. The premise that
  the barrier forces an extra `(n_age, n_wave)` materialization does **not**
  hold: the einsum already produces that array as its own output and the barrier
  sits on a scalar multiply of it, which XLA does in place. No approach is
  switched, and the docstring's note that folding `L_sun` into the einsum
  operand "does not survive" the SSP-as-`Parameter` path is left standing
  un-relitigated — a standalone reproducer at the seam's shape does not
  reproduce the defect at all, so it cannot adjudicate that note either way
  (#2178, #2194).


- **Symptom 2 of #2178 was the same defect, not a second one.** The float64
  spectroscopy forward was reported non-finite on six `spec/*/auto_*` seams (CI
  run 33958554553), and `_skip_if_lut_forward_is_broken` (#2143) was left in
  place until that could be answered. Reproduced under jaxlib 0.11.1: the
  float64 arm builds, fits and differentiates cleanly, and the `ValueError`
  from `_check_channel_scales` — carrying that run's own
  `max |data| = 1.618e-27` and `2.751e-29`, to the digit — comes from the
  **float32** arm the same module-scoped fixture builds next. Six errors is two
  seams times three tests. One defect at one threshold, attributed to the wrong
  arm. With the forward grouping stated in the graph the guard fires on **no**
  seam, so it is deleted rather than widened, and
  `tests/regression/precision/test_float32_fitting_path_seams.py` runs
  41 passed / 0 skipped / 6 xfailed on jaxlib 0.11.1 (#2178, #2143).


- `narayanan_prior(z)` and `narayanan_tau_prior(z, log_mstar)` centered
  unbounded `Gaussian` priors on `dust_bump_strength` and `dust_tau_diff`,
  both of which declare a `lo >= 0` `bound_check` — so the docstrings' own
  Examples raised `ValueError: ... bounds (-inf, inf) violate physical
  constraint: must be >= 0` at `Parameters` construction. Both Gaussians now
  truncate at zero (`lo=0.0`); `dust_delta`, which has no such constraint
  and whose fitted means straddle zero, stays unbounded. That fix exposed a
  second, independent defect in `Gaussian`: `_truncated` was derived from
  `self._cdf_lo > 0.0`, and for a bound more than ~8 sigma from the mean
  `erf` underflows to exactly 0.0, so `_truncated` read `False` and
  `unstandardize`/`sample` silently fell back to the untruncated affine map
  — inert for a `lo=0.0` bound 6.6–12 sigma away from these two priors'
  means. `_truncated` now reads the bound directly
  (`self._lo > -inf or self._hi < inf`); behavior-preserving for every other
  caller (every other bounded `Gaussian` in the tree already has at least one
  bound within a few sigma, where the CDF does not underflow, so
  `_truncated` already read `True` before this fix; every unbounded
  `Gaussian` is untouched). Finally, `dust_bump_strength`'s declared
  `free_prior` widened from `Uniform(0.0, 2.0)` to `Uniform(0.0, 4.0)`, since
  the old ceiling
  could not reach the Narayanan et al. (2018) MUFASA-fitted bump multipliers
  (up to 3.634 at z=4) that `narayanan_prior` itself now centers on (#2226).


- `mcmc_hmc_lowrank` ran its warmup fused into chain 0's sampling scan, which
  had two consequences. The #1999 post-adaptation stability probe had nowhere to
  run, leaving the one dense-capable metric path reachable above the D=30 cap
  with no step-size remediation; and chain 0 sampled inside the warmup program
  while chains 1..n-1 ran the separate `_hmc_chain_scan`, so a multi-chain fit
  ran two structurally different compiled programs over one adaptation — the
  shape that made NUTS irreproducible under a pinned key before its own split.
  The fused scan is replaced by `_hmc_low_rank_warmup_only` plus the shared
  chain scan; the probe and the dead-warmup refusal (#2088) are wired in, and
  `dense_mass_step_backoffs` / `warmup_divergence_frac` join the diagnostics.
  Measured on a D=74 posterior, the probe declines on all 12 rows and returns a
  bit-identical adapted step size, so this is insurance rather than repair
  (`bench/reports/2026-09-06_low_rank_metric_d74.md`, Finding 6).


- The dense mass-matrix cap is one seam, and crossing it is no longer silent.
  `use_dense = <policy> and n_dim <= 30` existed at **six** sites with four
  behaviors: `mcmc_nuts` logged the downgrade at INFO and only when
  `verbose=True`, `mcmc_hmc` applied it silently, `mcmc_dynamic_hmc` applied it
  silently from a signature that *defaults* to `dense_mass_matrix=True`,
  `CatalogFitter` applied the auto-policy without the cap at all — under a
  comment claiming it used "the same policy the single-galaxy samplers use" —
  and `fit_batch`, which shares one adaptation across a whole batch, applied it
  silently too. So an explicit `dense_mass_matrix=True` on a wide problem got a
  diagonal metric, or an O(D^2) allocation, depending only on which entry point
  the caller used, and in most cases with no way to find out. All six now route
  through `resolve_dense_mass_gate`, which honors the request where it can and
  raises a `UserWarning` carrying `n_dim` and `max_dim` where it cannot. The
  warning fires regardless of `verbose`: losing the sampler's most consequential
  setting is not a verbosity question. Nothing about which metric is *chosen*
  changes — every existing fit gets the same mass matrix it got before.


- Flat `Parameters(dust_model="single_component", dust_law_diff=...)` silently
  discarded `dust_law_diff` and built `power_law` on the one attenuation
  screen; a disagreeing `(dust_law_bc, dust_law_diff)` pair silently kept
  `dust_law_bc` and dropped the other, so the model built was not the one
  requested and nothing said so. Both shapes now raise `ValueError` naming
  `dust_law_bc` as the single-screen spelling; the working shapes are
  unaffected -- `dust_law_bc` alone still inherits into `dust_law_diff`, and
  an already-equal pair (what the grammar path writes for
  `single_component`) still builds. `two_component`/`wg00`/`off` inheritance
  (#1989) is unchanged in both directions (#2224).


- `SEDModel.from_config(dust=...)` named only the birth-cloud screen
  (`spec_kwargs["dust_law_bc"] = dust`); the diffuse-ISM screen's law was
  filled in only because the model happens to stay `dust_model="two_component"`
  and the low-level inheritance of #1989 backfilled `dust_law_diff` from
  `dust_law_bc` -- an accident of a default `from_config` never set on
  purpose, not an explicit choice. `from_config` now resolves both screens
  explicitly through the same resolver #2224 introduced
  (`resolve_dust_screen_laws`), so the diffuse screen's law is always stated,
  not inherited (#2021).


- **`download_ssp` and `download_template` no longer print an absolute path.**
  `SSP file already exists at {filepath}; skipping download.` and its three
  siblings interpolated the full path, and a rendered notebook captures those
  lines verbatim — so running a reproduction notebook from anywhere but the
  repository root wrote the machine's own directory into
  `docs/reproduction/*.ipynb`, four leaks across three renders, and a fresh one
  on every re-render. Every such message now goes through one helper,
  `_display_path`, which shows the path relative to the working directory when
  the file is under it and the file's name alone otherwise. Never absolute.


- **The composable AGN precompute's default wavelength grid is derived from
  the blocks it is about to evaluate, instead of a hard-coded
  `np.logspace(2.0, 6.0, 1500)` that truncated them.** Under `polar_dust` +
  `agn_norm='cigale_joint'` + `torus='skirtor'` the polar dust's
  absorbed-power reference is built on the SKIRTOR templates' native
  10 Å – 1e8 Å axis, and that grid covered neither end: measured with
  `disc='skirtor'` at i=30 and `agn_ir_frac=0.3`, `int(polar)/int(torus)` came
  out `0.286324800` on it against `0.264046724` on a covering grid, **+8.44%**.
  The build-time grid guard could not see it — that grid is chosen inside
  `blocks/composable_precompute.precompute` and never becomes the model's rest
  wavelength. The default now unions the legacy 100 Å – 1e6 Å span with
  whatever native support the recipe's own blocks declare, read through the
  same function the guard uses, and samples it at the legacy points-per-decade
  with a floor of 1500 points (10 Å – 1e8 Å in 2625 points for that recipe;
  the legacy grid unchanged where there is no tie). It reproduces the covering
  grid's share to 1e-6. A `wave_rest` the caller passes is checked by the
  guard itself and raises the same `ConfigError`.


- **`agn_radius_ratio` now reaches the `cigale_joint` disc tie, which used the
  SKIRTOR grid's R = 20 node whatever the model asked for.** The torus block
  always honored the value; `compose_l_nu` forwarded `agn_tau_skirtor`,
  `agn_p_skirtor`, `agn_q_skirtor`, `agn_oa_skirtor` and `agn_cos_inc` to
  `skirtor_disc_dust_ratio` and not the radius ratio. Measured at i=80 with
  `agn_ir_frac=0.3` and `agn_polar_ebv=0.3`, `int(polar)/int(torus)` was
  `2.605153276` at `agn_radius_ratio` 10, 20 **and** 30 — bit-identical. It is
  now `2.582960860` / `2.605153276` / `2.345936285`. Since the stored
  inclination normalization went into `R_faceon`, the missing axis also picked
  the wrong `norm(0)/norm(i)`: that factor is R-dependent (2.896205 /
  3.172626 / 3.338009 at i=80 for R = 10 / 20 / 30, a 15% spread), so a fit
  that pinned or freed `agn_radius_ratio` away from 20 got a polar reference
  off by up to that much at edge-on sightlines. The fiducial is R = 20, so no
  shipped number moves.


- **`grad` of an AGN prediction is finite at `agn_polar_ebv = 0`, the registry
  default.** The AGN dust-budget split shares one budget between the torus and
  the polar graybody as `share = polar / (torus + polar)` and then rescales the
  graybody by `budget * share / polar`. At zero polar reddening the screen
  absorbs nothing, so both quotients are `0/0`; both denominators were floored
  at `1e-300`, which gives the right forward value and a **NaN cotangent**,
  because division's VJP carries `-num/den**2` and `1e-300` squares to zero.
  Measured, `d(polar + torus)/d(agn_polar_ebv)` at `agn_polar_ebv = 0`: `nan`
  before, `6.743538e+33` (torus `'none'`) and `4.134017e+33` (torus
  `'skirtor'`) after. Away from the degenerate point nothing moves
  (`2.192167e+33` at `E(B-V) = 0.1`, both forms). Any gradient-based fit that
  started the polar screen at its default took a NaN on step one. Each
  denominator is now selected with the live predicate before the divide, and
  each degenerate answer is stated: share 0 (the torus keeps the whole budget)
  and a rescale factor of 1.0 (the zero re-emission stays zero).

  The same rewrite fixes a smaller one beside it: when the raw SKIRTOR
  disk/dust grid is absent, the face-on disc's unit-area shape was normalized
  by `jnp.maximum(integral, 1e-30)`, so a disc fainter than that floor came
  back scaled by `integral / 1e-30` rather than to unit area — measured
  9.9e-06 instead of 1.0.


- **A model wavelength grid that truncates the polar dust's absorbed-power
  reference is now refused at build time instead of shifting the torus/polar
  split silently.** Under `agn_norm='cigale_joint'` with `torus='skirtor'` and
  the `polar_dust` attenuation block, that reference is built on the SKIRTOR
  templates' native axis (10 Å – 1e8 Å) — the grid CIGALE's `skirtor2016`
  integrates its `l_ext` proxy over — and the caller's disc array is
  resampled onto it with zero fill, so wherever the model's grid does not
  reach, the disc is zeroed and the unit-area shape is renormalized over a
  truncated spectrum. Measured (`disc='skirtor'`, i=30, `agn_ir_frac=0.3`),
  `int(polar)/int(torus)`:

  | model grid | `polar/torus` | vs covering |
  |---|---|---|
  | 8 Å – 1e8 Å, n=3000 | 0.264046724 | 1.000000 |
  | 0.0413 Å – 3e11 Å, n=4000 | 0.264046724 | 1.000000 |
  | 1 Å – 1e9 Å, n=6000 | 0.264046724 | 1.000000 |
  | 8 Å – 1e8 Å, n=6000 (resolution control) | 0.264046724 | 1.000000 |
  | 80 Å – 1e7 Å, n=3000 | 0.284060926 | **1.075798** |
  | 500 Å – 1e8 Å, n=3000 | 0.290982429 | **1.102011** |
  | 8 Å – 1e6 Å, n=3000 | 0.263936198 | 0.999581 |
  | 100 Å – 1e6 Å, n=1500 | 0.286324800 | **1.084372** |

  Four covering grids agree bit-for-bit at two resolutions, so the effect is
  extent and not quadrature. The 80 Å – 1e7 Å row is why the requirement is
  the template axis rather than the disc block's own breakpoints: it spans the
  CIGALE piecewise disc's declared 8–1e6 nm limits in full and is still 7.6%
  off, because `piecewise_powerlaw_disk` extrapolates its end segments (those
  limits hold only 86.99% of the `skirtor` shape's integral) and because the
  zero-fill happens on the template axis. The bounds are read off the array
  the resampling targets, so a regenerated grid moves the requirement with it.
  How much a truncation costs depends on the shape being zero-filled — the
  same grids move `disc='schartmann2005'` by 0.1% / −3.0% / 0.1% / 0.1% —
  which is why the guard refuses the truncation rather than bounding the
  error.

  A `torus='skirtor'` build covers this by construction — the torus
  contributes its template axis to the master-grid union, measured to take a
  91 Å – 1e8 Å SSP grid to 10 Å – 1e8 Å — so the refusal is a ratchet on that
  union rather than something callers will meet, and its message says so.


- **`agn={'norm': 'independent'}` and `agn={'norm': 'conserving'}` beside an
  active `agn_ir_frac` now raise `ConfigError` at build time instead of
  silently producing an AGN whose disc/torus ratio reports the stellar mass.**
  fracAGN is the CIGALE `skirtor2016` coupling: it derives the AGN power from
  the dust-absorbed stellar luminosity, `L_absorbed * f/(1 - f)`, and every
  `agn_norm` policy routes the torus through that derived power — measured,
  the torus integral is identical to six digits under all three policies
  (`7.626100e+42` at `log M* = 10`). Only `'cigale_joint'` routes the *disc*
  through it as well, via the SKIRTOR template ratio `R`. The other two put
  the disc back on `10**agn_log_lbol`: verbatim under `'independent'`,
  debited by `(1 - agn_torus_frac)` under `'conserving'`. Measured (composable
  `disc='schartmann2005'` + `torus='skirtor'`,
  `dust_emission='dale2014_cigale'`), sweeping only the stellar mass and
  reading `int(sed_agn_disc)/int(sed_agn_torus)`:

  | `log M*` | `independent` | `conserving` | `cigale_joint` |
  |---|---|---|---|
  | 0.0 | 5.004902e+10 | 5.004902e+10 | 2.838156 |
  | 7.0 | 5.004902e+03 | 5.003901e+03 | 2.838156 |
  | 10.0 | 5.004902e+00 | 4.004135e+00 | 2.838156 |
  | 12.0 | 5.004902e-02 | 0.000000e+00 | 2.838156 |

  Twelve orders of magnitude in both refused columns — the ratio scales as
  `1/M*` — and constant in the legal one. `'conserving'` is the worse of the
  two: at `log M* = 12` the derived `agn_torus_frac` clips to 1 and the disc
  is debited to exactly zero. Without fracAGN both policies are coherent
  (`'independent'` holds `2.001533` at every mass), so the pathology is the
  *pair*, and only the pair is refused. Nothing raised or warned before, and
  the four sub-block SEDs still summed to `sed_agn` exactly, so the accounting
  looked intact.

  An explicit `agn_ir_frac=Fixed(0.0)` states "no coupling" and stays legal —
  it is one of the two remedies the refusal names, the other being
  `agn={'norm': 'cigale_joint'}`. The #2069/R55 `agn_log_lbol` refusal
  previously offered "set `agn_norm='independent'`" as a way out; that advice
  named this configuration, so it has been corrected to say
  `agn_ir_frac=0.0` (which then lets `'independent'` put the disc on
  `agn_log_lbol`) and to state that switching policy while keeping fracAGN
  active is refused separately.


- **The CIGALE-lineage SKIRTOR grid's stored inclination normalization is now
  read and applied, so every `agn_norm='cigale_joint'` + `torus='skirtor'`
  render becomes inclination-dependent where it was flat.**
  `data/skirtor_templates_v3.h5` stores each SKIRTOR model divided by its own
  dust integral, with the physical scale factored out into `spectra/norm` (one
  number per parameter cell, no wavelength axis). The loader never read it, so
  the face-on disc integral and the observer-inclination dust integral were
  divided across two different luminosity scales: `R_faceon` came out
  **4.4246 at every inclination** (measured at i = 0, 30, 50, 60, 80, 90),
  and the polar dust's share of the AGN dust budget came out
  inclination-**flat** at 0.200035. `R_faceon` now carries `norm(0)/norm(i)`
  — 1.0 face-on to 3.621 edge-on at the fiducial (tau=7, p=q=1, oa=40, R=20)
  — which is where CIGALE's `skirtor2016` applies it
  (`AGN1.disk *= AGN1.norm / self.SKIRTOR2016.norm`, one line before the
  face-on disc becomes the polar dust's absorbed-power reference). It reaches
  exactly that one quantity: the observed disc divides the factor straight back
  out, so `R` and the disc SED are untouched.

  Measured at the SKIRTOR fiducial with `agn_ir_frac=0.3`, the polar share
  against a live CIGALE `skirtor2016`, before → after:

  | i | tengri before | tengri after | CIGALE | after/CIGALE |
  |---|---|---|---|---|
  | 0 | 0.200035 | 0.200035 | 0.204988 | 0.9758 |
  | 30 | 0.200035 | 0.204670 | 0.209708 | 0.9760 |
  | 60 | 0.200035 | 0.344936 | 0.352217 | 0.9793 |
  | 80 | 0.200035 | 0.442378 | 0.450589 | 0.9818 |

  Blast radius: **every** `agn_norm='cigale_joint'` build carrying the
  composable `torus={'type': 'skirtor'}` (with the `disc` blocks `skirtor`,
  `schartmann2005`, `schartmann2005_skirtor_atten` or `adaf_lopez2024`) and a
  non-zero `agn_ir_frac` — the CIGALE reproduction, the `polar_dust` recipe,
  and `examples/agn/plot_polar_dust_ebv_type12_sweep.py`. At the i=30 fiducial
  the polar component rises 2.91%; at i=80 it rises 2.21x. Face-on is
  unchanged, and `agn_ir_frac=0` is unchanged (no R-tie there).

  **The AGNfitter-rX SKIRTOR reductions are unchanged.** `skirtor_agnfitter`,
  `skirtor_agnfitter_1p` and `skirtor_agnfitter_2p` read their own
  inclination-averaged, dust-only libraries
  (`data/skirtor_mean{3,1,2}p_torus_grid.h5`), which carry no such
  normalization and no disc component; so are the monolithic `skirtor` and
  `skirtor_stalevski` models and the `schartmann2005_skirtor_atten` disc
  transmission, all pinned bit-identical across this change.
  `tests/components/agn/test_skirtor_lineage_separation.py` holds the
  separation.


- The composable-AGN polar-dust attenuation block no longer disagrees with
  CIGALE's `skirtor2016` module on how the torus and polar re-emission share
  the AGN dust budget, which reference luminosity the polar covering factor
  multiplies, and what the polar screen reddens. Under
  `agn_norm='cigale_joint'`/`'conserving'`, the AGN dust budget now **includes**
  the polar re-emission (torus + polar = the budget, so the total is invariant
  in `agn_polar_ebv` to floating point -- measured `1.0` to `1e-16` relative
  across `agn_polar_ebv` in `{0, 0.03, 0.1, 0.3}`, against 1.00/1.41/1.92/2.33
  before); under `'independent'` the polar term stays additive on its own
  scale, as the policy's contract requires (measured 1.00/1.29/1.65/1.96 --
  unchanged). `polar_cone_covering_factor(opening_angle_deg, reference=...)`
  replaces the single-reference `polar_cone_covering_fraction`: CIGALE's
  `g(oa) = 7/18 - sin^2(oa)/6 - (2/9)sin^3(oa)` (referenced to
  `int L(theta=0) dlambda`, its face-on flux-table convention) applies when the
  disc is CIGALE's inclination-specific `disk` template -- that is,
  `agn_norm='cigale_joint'` with the SKIRTOR torus **and** a non-zero
  `agn_ir_frac`, which is the *traced* condition under which the runner ties
  the disc to `agn_power x R`. The reference is selected by that same
  predicate, so it cannot disagree with the frame the disc in the SED actually
  carries: at `agn_ir_frac = 0` (the registry default, and what the shipped
  gallery example and any composable SKIRTOR + polar-dust build without an
  explicit `ir_frac` carries) there is no `R`-tie, the disc is the
  bolometric-frame one debited by `(1 - agn_torus_frac)`, and `f_cone` applies
  to that array. Deciding the reference by a *static* branch on
  `agn_norm`/`torus` alone instead applied `g` to a rebuilt face-on array in
  both regimes, which left `sed_agn_polar` **1.57x** high at the default
  `agn_ir_frac = 0` (3.731924e+44 against 2.370195e+44 erg/s at the fiducial
  below) and bit-identical across a 2.84x change in the disc it reprocesses.
  `f_cone(oa) = 1 - (3/7)sin^2 oa - (4/7)sin^3 oa` (hemisphere-integrated
  bolometric) applies under `'independent'`/`'conserving'`. The two factors are
  exactly proportional (`f_cone/g = 18/7`), so picking the wrong one silently
  moves the polar re-emission by 2.571x; the function refuses an unrecognized
  `reference` rather than defaulting. Reproducing CIGALE's face-on integral
  needed the disc's *un-inclination-weighted* shape rescaled by
  `skirtor_disc_dust_ratio`'s `R_faceon = int_disk0/int_dust` output (already
  derived there for exactly this, previously discarded) -- rescaling the
  Stage-4 R-tied (inclination-weighted, `R`-scaled) disc by the inclination
  ratio alone reuses the wrong proportionality constant and was off by ~20%.
  That face-on reference is normalized, and its absorbed power integrated, on
  the **SKIRTOR templates' own wavelength grid** -- the grid `R_faceon` was
  derived on, and the one CIGALE integrates over throughout
  (`trapezoid(AGN1.disk * (1 - ext_fac), x=AGN1.wl)`). Unit-normalizing the
  same shape on the caller's grid instead paired a native-grid ratio with a
  caller-grid integral and made the polar share depend on the model's
  wavelength extent, which is not a physical parameter: `sed_agn_polar /
  sed_agn_torus` read 0.2559089362 on an 8 Å - 1e8 Å grid and 0.2538381129 on
  a 0.0413 Å - 3e11 Å one for the `skirtor` disc block (0.8% apart, and 11.0%
  apart against a 500 Å - 1e8 Å grid), where it is now bit-identical
  (0.2565718334) across every grid that spans the disc's own support.
  `skirtor_disc_dust_ratio` returns a named `SkirtorDiscTie` rather than a
  bare 3-tuple, so the face-on shape and the grid it is normalized on travel
  together and cannot drift apart. **Caveat**: a model whose wavelength grid
  starts redward of its disc block's own support still carries no disc light
  there, so the resampled shape genuinely differs and the split moves with it
  (+10.2% on a 500 Å grid for the `skirtor` disc block, with the disc's own
  share of the budget moving alongside it) -- span the disc's support.
  Measured at the SKIRTOR fiducial (`t=7, pl=1, q=1, oa=40, i=30, disk_type=1,
  fracAGN=0.3, law=0, EBV=0.03, T=100, beta=1.6`) against a live `pcigale`
  `skirtor2016` run, normalized to the same AGN dust budget: torus 1.05x
  (100 um) / 1.08x (1 mm), polar 0.95x at both -- both within 5%, the residual
  torus-side gap traced (via a polar-dust-disabled control) to the pre-existing
  SKIRTOR torus *template* interpolation, unaffected by this fix. Finally, the
  polar screen now reddens the disc only: the torus IR is removed from both
  the screen input and the absorbed-luminosity integrand (CIGALE reddens only
  `disk`, never its torus thermal emission).

  **Behavior change at Type-2 sightlines.** The polar re-emission is now
  ISOTROPIC: the absorbed-luminosity integrand is the *unmasked* disc, so
  `sed_agn_polar` is present at full strength at every inclination. The cone
  dust intercepts a fixed share of the disc's light regardless of where the
  observer stands and re-radiates it as an optically-thin FIR graybody, which
  no viewing angle can hide; what is Type-1-only is the *reddening of the disc
  we see*, and at Type-2 inclinations that sightline does not pass through the
  near cone and the disc arrives already screened by the equatorial torus.
  Previously the Stage-4.5 Type-1/2 mask multiplied the re-emission integrand
  too, which switched the polar component nearly off edge-on. Measured at the
  SKIRTOR fiducial with `agn_polar_ebv=0.3`, `int(sed_agn_polar) dnu` at
  i = 80 deg: **1.767569e+45 against 1.791134e+43 erg/s (98.7x) under
  `agn_norm='independent'`** and **6.035863e+44 against 8.913741e+42 erg/s
  (67.7x) under `'conserving'`**; at i = 30 deg it moves by 0.4% and 0.3%.
  Under `'cigale_joint'` with a non-zero `agn_ir_frac` the polar reference is
  the face-on `disk`, which never carried the mask, so that path is unchanged
  (8.421947e+44 erg/s either way). CIGALE `skirtor2016` does the same, checked
  against a live run rather than inferred: `self.SKIRTOR2016.disk *= ext_fac`
  is gated on `i <= 90 - oa` while `l_ext` and
  `self.SKIRTOR2016.dust += blackbody` are unconditional, and its
  `int(polar_dust)` per unit dust budget at `oa=40` runs 0.204988 (i=0),
  0.209708 (i=30), 0.253217 (i=50), 0.352217 (i=60), 0.450589 (i=80), 0.483635
  (i=90) -- non-zero and largest edge-on. Renders that move:
  `examples/agn/plot_polar_dust_ebv_type12_sweep.py` and its committed
  `docs/auto_examples/agn/` outputs.


- Rule 4 of the composable-AGN recipe validator no longer names
  `nlr={'type': 'analytic'}` as disc-anchored. `nlr_analytic_block` is
  illuminated by the intrinsic bolometric `10**agn_log_lbol` and its body opens
  with `del l5100_disc`, so warning that it "scales by the disc's 5100 A
  luminosity (zero)" when no disc is selected was a false advisory -- R48's own
  rule, that a block which does not normalize off the disc must not be listed.
  Measured with every other slot off, marginal `sum|sed_agn|` over 500 A - 1 mm
  at `agn_log_lbol=12`: `nlr='analytic'` gives **2.022386e+31 both with
  `disc='none'` and with a full `multicolor` disc** -- bit-identical -- while
  every entry that stays in the table goes to exactly zero without a disc
  (`nlr='grahsp'` 0 -> 1.642265e+31, `blr='analytic'` 0 -> 4.686986e+30,
  `blr='grahsp'` 0 -> 7.246841e+31, `feii='grahsp'` 0 -> 5.363636e+30,
  `torus='grahsp'` 0 -> 2.493180e+34). Rule 4's negative control was probing
  with `analytic`, which the fix makes un-fireable, so it would have passed
  vacuously; it now probes `nlr='grahsp'`, measured disc-anchored.


- The #1970 refusal (Dale+2014's embedded star-forming radio synchrotron
  double-counted against an SF radio block) now measures the selected template
  instead of testing its registry name. Keying on
  `spec.dust_emission == 'dale2014'` was neither sufficient nor necessary: a
  tail-free grid registered under that name --
  `register_dale2014_tabulated(cigale_grid, name='dale2014')` -- was refused
  although it carries no radio, and the tail-bearing grid filed under the
  tail-free name `dale2014_cigale` was accepted. The guard now reads the grid
  via the new `dust_emission_radio_tail_aa`, which requires two measured
  conditions: the emitting span reaches *strictly* past 1e8 Å (1 cm, 30 GHz,
  blueward of the whole 1.34-10 GHz double-count window) **and** its red end
  is **non-thermal**. Reach alone would have newly refused `astrodust`, whose
  spinning-dust component emits to 3.0e8 Å and double-counts nothing.
  Non-thermal is stated as a threshold on the red-end spectral index in
  FREQUENCY, `alpha = dlnL_nu/dlnnu < 1` measured over the reddest decade of
  the emitting span, because that is where the two families actually separate:
  radio continua are flat or falling toward higher frequency (optically-thin
  synchrotron `alpha ~ -0.8`, optically-thin free-free `alpha ~ -0.1`,
  flat-spectrum `alpha = 0`) while thermal dust on its Rayleigh-Jeans side
  rises as `nu^(2+beta)`, i.e. `alpha >= 3`, and spinning dust below its
  ~30 GHz peak rises too. Measured red-end `alpha`: `dale2014` **-0.665** (a
  textbook SF synchrotron index) against +3.111 (`bosa`), +3.326
  (`astrodust`), +4.810 (`schreiber2016`), +5.510 (`dale2014_cigale`) -- the
  two families are 3.8 apart, with the threshold between them and 1.7-2.1 of
  margin on each side, so only `dale2014` qualifies. The weaker rule this
  replaces -- "`L_nu` rising toward longer wavelength", i.e. `alpha < 0` --
  let the whole flat-and-inverted radio family through: measured on synthetic
  grids, an `alpha = 0` flat-spectrum tail at 2.2e9 Å and an `alpha = 0.99`
  one were both accepted. Both thresholds are now pinned at their boundaries
  with synthetic grids (`alpha` = -0.8/-0.1/0/0.99 refused, 1.0/3.6 accepted;
  edge 9.9e7 and exactly 1.0e8 Å accepted, 1.0000001e8 Å refused) -- `bosa`
  sits exactly at 1.0e8 Å but its verdict is double-caused, so it pinned
  neither. The span is the union over
  every template row, not one row's: `dale2014_cigale` stops emitting at
  7.727e7 Å over its 64 alpha rows (the strip edge its component documents)
  while its `alpha=2.0` row alone stops at 6.026e7 Å, and a build-time refusal
  has to hold for every alpha a fit can reach. A model whose red end cannot be
  measured -- a closed-form law, or an uninstalled grid -- is not refused.
  The reader no longer assumes a wavelength unit the file does not declare: it
  reads a declared `unit`/`units` attribute on the wavelength dataset or a
  file-level `wavelength_unit`, and only falls back to this repository's key
  convention (`wavelength_aa` and the bare `wavelength` in Å,
  `wavelength_um` in micron) when the file declares none, reporting which it
  used in the refusal message. The bare key `wavelength` was being scaled by
  1e4 as if it were micron, while every grid here that uses it stores Å --
  `dl07_templates{,_v2}.h5` and `dl14_templates.h5` at 1e4-1e8 Å,
  `skirtor_templates_v{2,3}.h5` at 10-1e8 Å, three of them saying so in an
  attribute. It was masked only because none of those files carries a row
  dataset under a key the reader recognizes; measured on a synthetic grid, a
  3600-2.2459e9 Å axis under that key read **2.2459e13 Å** and would have
  been falsely refused whenever SF radio was active. `_red_end_from_grid_file`
  now returns a named `GridRedEnd` carrying the key and unit alongside the
  edge and index.


- A user-provided `agn_log_lbol` is no longer accepted and discarded when the
  CIGALE fracAGN coupling owns the AGN power. #2069 already refused a **free**
  `agn_log_lbol` under `agn_norm='cigale_joint'` with a SKIRTOR torus and an
  active `agn_ir_frac`, by measuring that the SED is identical at the two prior
  bounds; a `Fixed` value the user spelled out went the other way -- silently
  computed over. Measured across the declared `Uniform(8, 14)` prior (5%/95%
  quantiles 8.3 and 13.7, a 5.4-dex range) with `agn_ir_frac=0.3`: that one
  configuration moves `sed_agn` by 6.4e-15 relative, floating-point roundoff,
  while an active `nlr` or `blr` block, a `fritz` torus, no torus,
  `norm='independent'`, and `agn_ir_frac=0.0` each move it by 2.5e5. So the
  refusal is the same *measurement* widened to the user-provided case rather
  than a second, static guard listing those five carve-outs -- it cannot go
  stale as blocks are added. The registry default stays exempt (every
  `'all_params': Fixed(DEFAULT)` AGN build carries one), and the flat-kwarg
  `Parameters(...)` escape hatch is untouched, since it records no provenance.
  The refusal's advice also no longer reads "Fix agn_log_lbol (any value; it
  cancels)", which this same guard now refuses -- advice a guard refuses is the
  #1364 defect.


- A Dale+2014 template grid must now **declare** the convention its rows are
  stored in, and `load_dale2014_lnu_grid` refuses a grid it cannot type instead
  of guessing. The stored unit decides whether the L_lambda -> L_nu Jacobian is
  applied, and the decision was an exact-string comparison against one magic
  value, so every other `spectra_unit` -- absent, prose, a typo, a future
  spelling -- silently meant "convert". A grid already in L_nu but labeled any
  other way was therefore multiplied by `lambda^2/c` a second time, with no
  error and no tolerance at which that is a small mistake: on the shipped
  CIGALE-sourced grid the two typings of the same rows differ by a factor
  spanning 1.07e-2 to 9.60e4 across 2.0e4-6.0e7 A once each is unit-normalized
  in L_nu (the grid's own lambda-Jacobian). The two accepted declarations are
  now named constants, `DALE2014_UNIT_L_NU` and `DALE2014_UNIT_L_LAMBDA`;
  `scripts/regenerate_dale2014_from_cigale.py` writes the machine-readable one
  and keeps its descriptive text in a separate `spectra_unit_note` attribute,
  and `data/dale2014_templates_cigale.h5` was regenerated to carry it (all four
  datasets bitwise identical, max|delta| = 0 -- only the attributes moved).
  A contract test also pins that the three public routes onto one Dale grid
  (the `dale2014_cigale` registry entry, `create_dale2014_from_grid`, and
  `register_dale2014_tabulated`, which had no test reference anywhere) deliver
  bit-identical templates -- its array-level half now compares the literal
  path against the grid the registry entry *resolves*, rather than against a
  second call on the same path, which asserted only determinism.
  The refusal reaches `.npz` grids too (that branch used to hard-code
  "convert"), and its advice is now conditional on the container: for HDF5 it
  names the file attribute and the two regeneration scripts, and for `.npz` it
  names the array entry to add, because both scripts write HDF5 and could
  never produce that file -- advice a user cannot follow is the #1364 defect.


- The composable AGN `validate_block_recipe` "no disc, active downstream"
  advisory (Rule 3) named every active `nlr`/`blr`/`feii`/`torus` block when
  `agn_disc_block='none'`, but only a block that actually reads
  `l5100_disc` for its normalization goes to zero there. Measured: a
  torus-only build (`agn_torus_block='cat3d_wind'`, `agn_disc_block='none'`)
  emits `sed_agn_torus` summing to 3.849e34 under both `agn_norm='independent'`
  and `agn_norm='cigale_joint'` — never zero — so the advisory was false for
  it (and for every production torus except `'grahsp'`, the one torus block
  whose body reads `l5100_disc`). The advisory now names only the
  `(category, block)` pairs in `_DOWNSTREAM_NEEDS_L5100` (R48). `agn_norm` is
  threaded from `Parameters` into `validate_block_recipe` so the check can
  become policy-aware if a future norm policy changes anchoring; measured at
  this HEAD, none does.


- `_mass_scale_lnu`'s forward product went `nan` in float32 on the
  `SpectrumPrecomp` path under jaxlib 0.11.1, where jaxlib 0.11.0 was finite —
  with **byte-identical optimized HLO**, so the graph did not change and the
  emitted kernel did. `total_mass * L_sun` is ~3.8e43 (`inf` in float32), and a
  backend that emits its own kernel for the fused `multiply -> multiply ->
  reduce` may hoist the two scalar broadcasts into that single factor. Ages
  beyond the galaxy's age carry an exactly-zero SFH weight, so `inf * 0` is
  `nan` and the reduction over age is `nan` at every pixel. PR #2100 had
  already pinned the *reverse* pass's grouping for the same overflow; this is
  the same hazard reached from the forward. The grouping is now stated in the
  graph with `optimization_barrier`, on both spellings of the product — the
  function body and the `custom_jvp`'s `primal_out` — because fixing only one
  leaves the differentiated forward `nan` while the undifferentiated one is
  finite. Float64 is bit-identical, verified as equality rather than tolerance
  across all sixteen seams, which matters because the barrier changes emitted
  HLO for every fit. Note the assertion hole that hid this: the seam checks
  asserted gradients were non-zero, and `nan != 0.0` is `True` — the mirror of
  #2100's hole, where `isfinite` admitted zero. This closes the float32
  symptom, and the `spec/*/auto_*` symptom with it — see the next entry
  (#2178, #2100).


- `compute_l_dust_absorbed` (`tengri.utils.sed_quantities`) integrated the
  whole wavelength grid, while `bolometric_absorbed_log10`
  (`tengri.forward.energy_balance`, the pipeline's own dust normalization)
  masks `lambda < 912` Å (Lyman-continuum photons ionize hydrogen rather than
  heat dust). The two now build their integrand through one shared helper
  (`absorbed_integrand`, mask constant `LYMAN_CUTOFF_AA`), so LyC energy is no
  longer counted as dust-absorbed by the utility path either. `agnfitter_priors`'s
  `energy_balance` prior compared the unmasked total against the masked
  `L_ir`, so any nonzero LyC fraction made the absorbed side exceed the
  emitted side and returned `AGNFITTER_HARD_REJECT` for every
  Calzetti-attenuated star-forming galaxy, independent of
  `tau_v`/`dust_T`/`dust_eta_balance`. `compute_l_dust_absorbed` gains an
  `include_lyc=False` keyword; pass `True` for the pre-fix unmasked total
  (#922).

- `dust_eta_balance` (`L_IR = eta * L_absorbed`) was read only on the
  two-component dust-attenuation path; the single-component screen
  (`components/dust/component.py`) and the WG00 screen
  (`components/dust/wg00_model.py`) published `L_ir = L_absorbed`
  unconditionally, so freeing `dust_eta_balance` on either path had no effect
  on the SED at all — a declared free parameter whose posterior always
  equaled its prior. Both now apply the same log-space treatment as the
  two-component path (`L_ir = eta * L_absorbed`; `eta <= 0` re-emits nothing).
  Default `eta = 1.0` changes no existing SED (`log10(1.0) == 0`).

- WG00-attenuated models (`dust_attenuation={'type': 'wg00', ...}`) silently
  dropped the configured `dust_emission` component with no warning:
  `component_factory.py` excluded `wg00` from the dispatch that attaches dust
  IR re-emission, an exclusion carried into the unified single-dispatch
  conditional when attenuator selection converged onto the `_REGISTRY` seam
  (b5ffa65e1); WG00 screen attenuation itself originates in #560/#665. WG00
  now receives its configured `dust_emission` component exactly like the
  other two attenuation types. **Far-IR photometry of a WG00-attenuated model
  that configures `dust_emission=` changes**: the dust IR bump that was
  previously silently absent now appears.

- **`agn={'feii': {'type': 'qsogen_balmer'}}` was selectable but inert** (#2175): its Balmer-continuum normalization, `agn_bcnorm`, fell into the block function's `**params` catch-all instead of a named keyword argument, so it was silently discarded — photometry was bit-identical to `boroson_green` regardless of the value. `agn_bcnorm` is now a named keyword with a `blocks/_consumes.py` entry; an explicit value now moves photometry (23.6x at `agn_bcnorm` 0 to 2 in the fix's own measurement). The default (`0.0`, matching upstream `qsogen`) is unchanged.

- **`'all_params': FREE` on a group that has nothing left to free now warns or raises instead of silently doing nothing** (#2187). `met`, `sfh`, `dust_emission`, and the AGN sub-blocks (`agn.disc`, `.torus`, `.nlr`, `.blr`, `.feii`, `.atten`) each had at least one `type` for which the wildcard covered zero declared parameters, so the config looked like it declared free parameters and fit none. `_check_wildcard_freed_something` now adjudicates every wildcard's outcome uniformly across all groups: covering zero parameters warns `WildcardNoOpWarning`; covering one or more but freeing none of them raises `ParameterError` (that case was never intended); freeing some but not all warns `WildcardPartialFreeWarning` naming what stayed pinned (#1474).

- **AGN X-ray double-counting guard.** `disc={'type': 'kd18_agnfitter'|'kd18_agnfitter_warmindex', ...}` paired with any of the five corona X-ray variants (`simple`, `yang20`, `lopez24`, `xray_aird`, `agn_xray_corona`) now raises a `ConfigError` naming both — the KD18 disc template already carries its own hot-corona X-ray emission, so stacking a second corona model double-counts the same physical component. No host-XRB-only X-ray variant exists to combine safely with KD18 discs today.

- **Radio: `bell2003_split` no longer double-counts free-free emission.** The split thermal/non-thermal radio SED already includes the free-free (thermal) term inside its own calibration; a component-level `include_freefree=True` alongside `bell2003_split` now raises `ConfigError` instead of silently adding a second free-free contribution.

- **Radio free-free calibration corrected.** `sfr_from_lir`'s free-free constant was an uncited `1.73e10 Lsun` figure matching neither Kennicutt (1998), Murphy et al. (2011), nor Bell (2003); replaced with the cited Murphy et al. (2011) calibration. **Behavior change:** the free-free contribution to `bell2003_split` radio SEDs is now ~2.6x higher, moving the thermal fraction from 4.9% to 11.8% at the fiducial configuration (closer to Condon 1992's ~10%).

- `SKIRTORTorus`'s twelve class-level free parameters (`agn_band_frac`, `agn_polar_ebv`, `agn_log_lbol`, and nine others) are now derived from `declared_prior(PARAMS, name)` instead of restated literals; two had drifted (`agn_polar_ebv` default `0.1` vs the canonical `0.03`; `agn_log_lbol` default `11.0` vs the canonical `10.0`), silently pinning every `SKIRTORTorus`-based fit that did not override them to the wrong starting point.

- `multicolor_disc`'s pure-float32 bolometric renormalization returned
  `l_nu_intrinsic * scale`, and transposing that product makes JAX form
  `sum(g * l_nu_intrinsic)`. With the raw disc SED (~1e28) and the cotangent
  the AGN reference offset hands back (~10^34.6) that inner product is ~1e64
  — `inf` in float32 — while its partner `d scale/d arr` ~1e-64 flushes to
  zero, and `inf * 0` is NaN. So `d(sum rest_sed)/d(agn_log_lbol)` was **NaN
  in pure float32** while the forward pass and `jacfwd` were both exact. The
  renormalization now returns the L1-normalized SED against a correspondingly
  inflated scale — algebraically the same number, both factors in range — and
  the gradient matches float64 to 1.000002 across the whole declared
  `agn_log_lbol` prior. Float64 is untouched: the change is inside the
  `wavelength.dtype == jnp.float32` branch. `kubota_done` is a *different*
  defect at the same call site (wrong by -0.034x with an O(1) cotangent, and
  cured by `agn_f_hard=0`, so it is the hot-corona zone) and stays open
  (#1439, #1388).


- The construction-time dead-fit guard (`DeadFitWarning`) and
  `convergence_check` compared the divergence count, which is summed over
  every chain, with the per-chain draw count, so the "every transition
  diverged" branch never fired for a multi-chain run and the percentage
  read 400% on four chains. `total_draws()` owns that arithmetic now, the
  backends' completion lines print the total, and single-chain paths record
  `n_chains` (#2087).

- The frozen-parameter half of the same guard scanned every column of
  `samples`, which carries `Fixed` parameters as constant arrays by design,
  so any model with a pinned parameter warned "dead fit" and named the
  pinned parameters. `Posterior.free_names` reads the free names off the
  model's spec and the check restricts itself to them (#2087).

- `convergence_check` scanned every column of `samples` for its FROZEN check
  too, so the same fit was reported `converged=False` naming 41 pinned
  parameters; it now reads the free names, and no longer skips `psd_xi` (a
  frozen stochastic-SFH field latent is as dead as a frozen named parameter).
  `Posterior.save()` writes the free names into the file and `Posterior.load()`
  restores them, so a reload without `model=` no longer re-creates the false
  positive; files written before this load unchanged (#2087).

- **Naming both `agn_torus_frac` and an active fracAGN raises `ConfigError`**
  (#2189). With fracAGN active, `AGNSEDComponent.apply` overrides whatever
  `agn_torus_frac` was supplied with a value derived from the dust-absorbed
  stellar luminosity, so the parameter is inert: measured 0.0 relative change
  in photometry across its full range, against 8.5x-30.6x with fracAGN
  inactive. A sub-block wildcard silently narrows it out instead of raising.
  Every spelling and placement the grammar honors is seen, including
  `ir_frac`/`fracAGN` written inside a sub-block.

- `slone_netzer`'s `agn_log_ledd` is freed by the disc wildcard again. It had
  been recorded as inert, from a gradient measured at the shared declared
  default -1.0 — outside the block's own grid axis `[-4, -1.9586]`, where the
  clip makes the gradient exactly zero by construction (#1586). Inside the
  axis it runs 5.9e-2 to 8.8e-1.

- **Polar dust reemission on the composable `agn={'type': 'composable', ...}` path is applied exactly once, only under `atten='polar_dust'`.** Previously that path computed a polar-dust contribution up to three times: the runner's Stage-1.5 line-of-sight reddening (for every `atten` type whenever `agn_polar_ebv > 0`), the `skirtor_torus_block`'s own bundled Casey (2012) polar term (`torus='skirtor'`), and the standalone `atten='polar_dust'` reemission block — double-applying it when `torus='skirtor'` and `atten='polar_dust'` were combined, and leaking an energy-non-conserving line-of-sight reddening with no reemission credit under every *other* `atten` choice whenever `agn_polar_ebv` was nonzero (measured: up to 76% of the AGN SED removed at `agn_polar_ebv=0.3` under `atten='none'`). **Behavior change**: on the composable path, an AGN model with a nonzero `agn_polar_ebv` and `atten` set to anything other than `'polar_dust'` no longer applies polar dust at all — set `atten='polar_dust'` to reddened + reemit it. `skirtor_torus_block` no longer declares or reads any `agn_polar_*` parameter, and the runner's Stage-1.5 reddening is removed. The standalone `SKIRTORTorus` `SEDModelComponent` (used directly, outside the composable builder) is unaffected — it still bundles its own polar-dust term.

- Dust attenuation laws are explicit and required (#1989). A dust attenuation group
  spells its law as either `law` (one law, both screens) or, on `two_component` only,
  both `law_bc` and `law_diff` together — never one half of the pair, and never
  neither. **Breaking**, in three shapes:
  - no law raises. It previously defaulted to `power_law`; `'law': 'power_law'`
    reproduces the old fit exactly, but check whether that default was intended.
  - a lone `law_bc` raises. The old form applied it to both screens, so `'law': X` is
    behavior-preserving; use the pair only when the screens genuinely differ.
  - `single_component` with `law_bc`/`law_diff` raises — a single screen takes `law`,
    and its depth is `tau_v`, not `tau_bc`/`tau_diff`.
  The low-level `Parameters(dust_law_bc=…)` kwargs path is unchanged and still
  inherits `dust_law_diff` from `dust_law_bc`.

- **Example gallery curated and refocused**: Pruned 283 → 121 gallery
  scripts across 17 sections; removed inference/fit-comparison examples (they
  belong in notebooks), dissolved `inference`, `workflows`, `multiwavelength`,
  and `contrib` sections, collapsed duplicates, and routed all kept examples
  through the public API. Every example now fits in a single render
  timebox. Added the composable shock-group sweep example (shock parameters ×
  physics code choice × SSP grid). New top-level export: `igm_transmission_meiksin06`;
  physical constants `C_AA` and `LOG10_ZSUN` are now exposed via `tengri.units`.

- **Fits default to the precompute LUT** (behavioral change). `Fitter`,
  `forward.fit(...)` and `Galaxy.fit(...)` gained `approx="auto"` (the default):
  a fit now auto-routes through the fast precompute lookup table chosen by data
  type — `WavePrecomp` for photometry, `SpectrumPrecomp` for spectroscopy/joint,
  plus `FeaturePrecomp` when emission lines are fit — while a model already
  built with an explicit `approx=` is respected untouched. Model **construction
  and prediction stay exact** (`SEDModel.build` still defaults to `approx=None`);
  only the fit is accelerated (~10–25× per step; posterior shift ≪ noise —
  validated to <0.07 σ at SNR 20 and <0.006% band deviation at z=4 with IGM).
  Opt out with `forward.fit(..., approx=None)` (exact wave-grid); pass an
  explicit config to override. The fit clones the model via the new
  `SEDModel.with_approx` / `ForwardModel.with_approx`, so the caller's model is
  never mutated and the returned `Posterior` references the fast clone.
  Additionally, `run(..., prewarm=True)` (default) JIT-compiles the
  loss/gradient plus `predict_photometry` / `predict_properties` before the fit
  loop — warming the persistent cache and post-fit posterior-predictive /
  derived-quantity exploration; pass `prewarm=False` for the prior lazy-compile
  behavior.

- British → American spelling of public API identifiers, renamed in place
  without deprecation aliases (tengri is pre-1.0 — the public API is not yet
  stable, so renames ship directly; #819). Update imports and call sites:
  - `cue_full_catalogue` kwarg and the `neb={'type': 'cue', …}` builder
    short-key `full_catalogue` → `cue_full_catalog` / `full_catalog`.
  - `rest_frame_colour()` → `rest_frame_color()`
    (`tengri.analysis.diagnostics`).
  - `CalibrationMarginalisedLikelihood`, `ELineMarginalisedLikelihood`,
    `CloudyELineMarginalisedLikelihood`,
    `CalibrationELineMarginalisedLikelihood` → `…MarginalizedLikelihood`;
    module `tengri.inference.likelihoods.marginalised` → `…marginalized`.
  - `normalised_excess_variance()` → `normalized_excess_variance()`
    (`tengri.components.agn.grahsp.variability`).
  - `rank_normalise()` / `rank_normalised_rhat()` → `rank_normalize()` /
    `rank_normalized_rhat()` (`tengri.analysis.diagnostics.autocorrelation`).
  - `finalise()` → `finalize()` (`tengri.inference.backends.nested.utils`).
  - `SSP_CATALOGUE_URL` → `SSP_CATALOG_URL` (`tengri.data`).

  Prose across docs, gallery examples, notebooks, and docstrings was
  likewise converted to American English. The HDF5 grid dataset keys
  `ionisation_parameter` and `log10_specific_ionising_luminosity` retain
  their upstream British spelling — they index a third-party Synthesizer
  data file, so the Python strings must match the keys on disk.

- Default AGN model for `AGNSEDComponentConfig` changed from `"simple"`
  to `"multicolor_agn"` (the Kubota & Done 2018 outer-zone disc + 2-T
  torus). Existing fits that explicitly set `agn_model="simple"` will
  fail with a clear `ValueError`; update to one of the production
  models listed in the AGN module docstring.

- Promoted to the top-level `tengri` public surface (added to
  `__all__`): `FIXED`, `FREE`, `fit_batch`, `SEDResult`,
  `PriorPredictive`, `data_path`. No behaviour change — they were
  importable but not advertised. `load_filter_set` was considered but
  stays demoted per existing design — import from
  `tengri.observation.load_filter_set`.

- Experimental notebook `multimodel_bma_candels` now builds its plotted
  posterior SEDs via the exact public `lnu_to_fnu(1, d_L, z)` conversion
  instead of an empirical `predict_photometry`-anchored scale factor
  (the old anchor was ~10% off because it equated a filter-integrated
  flux with a point-interpolated `L_ν`). The eager-warm tracer-leak
  workaround is dropped — `jax.jit(jax.vmap(...))` over `predict_obs_sed`
  / `predict_sfh_quantities` now runs cold. Possible now that
  `predict_spectrum(wave_obs=...)` is fixed (#707, #712). No change to
  fits, evidences, or BMA weights — plotting/prediction only (#730).

- `multimodel_bma_candels` fits the four configs per galaxy concurrently
  with a `ThreadPoolExecutor` (XLA releases the GIL during compute, so
  this is a ~2–3× wall-clock win for bit-identical results) and uses
  `n_live=250` (≈2× faster than 500, negligible `log Z` shift). Added a
  note documenting that compilation is *not* the bottleneck (~0.3 s,
  cached) and why `fit_batch_map_vmap` (MAP-only, single shared model)
  cannot vectorise nested sampling across these structurally-different
  configs.

- Editorial pass on `multimodel_bma_candels` for the public docs: prose
  rewritten in plain scientific style, one publication-quality figure per
  galaxy (the separate compact/presentation variants are merged), the
  $M_\star$-SFR panel zoomed out so the broader BMA contour is not
  clipped, the on-figure weight annotation removed, and XLA/PjRt C++ logs
  suppressed via `TF_CPP_MIN_LOG_LEVEL`.

- Made `multimodel_bma_candels` reproducible and swapped its non-parametric
  SFHs. The per-fit PRNG seed was derived from Python's built-in `hash`,
  which is salted per process (`PYTHONHASHSEED`), so every run drew a
  different nested-sampling realisation and the figures changed run to
  run; it now uses a `hashlib`-based `stable_seed`, so the notebook
  reproduces exactly. Configs A and B now use the continuity (Leja+2019)
  and Dirichlet (Leja+2017) priors instead of Dense Basis, whose quantile
  parameters are strongly degenerate (near-singular Hessian) and left the
  evidence — and therefore the BMA weights — unstable from seed to seed.
  (Laplace/MAP evidence was evaluated as a faster, deterministic
  alternative but disagreed with converged nested sampling for the same
  degeneracy reason, so the calibrated nested-sampling `log Z` is kept.)

- Gave every `multimodel_bma_candels` config baked-in nebular emission, so
  the averaging is no longer confounded by an on/off nebular switch. C and
  D moved off the bare-stellar BC03/BPASS grids (which have no baked-nebular
  variant — FSPS does not implement those isochrones) onto the wNE Padova
  and BaSTI grids (logU=−2, the FSPS default; downloaded from the `dsps_ssp`
  catalogue). All four configs now use a distinct isochrone (MIST, PARSEC,
  Padova, BaSTI) with nebular baked into the SSP LUT — so the full render
  stays fast (~0.7 ms/eval, vs ~2 ms for the Cue emulator, which timed out
  the render at 7 galaxies).

- **The 18 restatement drifts `check_param_restatements.py` found across five legacy AGN disc/torus classes are fixed at the source.** `CAT3DTorus`, `KD18Disc`, `PowerLawDisc`, `Silva04Torus`, and `SKIRTORAgnfitterTorus` now derive their class-level free-parameter literals from `declared_prior(PARAMS, name)` instead of restating them, the same pattern `SKIRTORTorus` already used (above). **Behavior change:** every one of the five classes' `agn_log_lbol` default moves `11.0 -> 10.0`; `agn_torus_frac` bounds/default are corrected on `CAT3DTorus`, `Silva04Torus`, `SKIRTORAgnfitterTorus`; `KD18Disc` corrects `agn_log_mbh`, `agn_log_ledd`, `agn_a_spin`, `agn_cos_inc`, `agn_f_hard`, `agn_gamma_warm`, `agn_kt_warm`, `agn_gamma_hard`, `agn_kt_hot`, `agn_r_warm_ratio`, and `agn_lum_ratio` bounds/defaults; `PowerLawDisc` corrects `agn_alpha` and `agn_lum_ratio` bounds/defaults. Any existing caller that constructed one of these classes and relied on its unset defaults or declared support (e.g. a `Fixed(DEFAULT)` sample, or a sampler exploring the class's own stated prior range) samples differently now.

- `scripts/build_slone_netzer_grid.py`'s HDF5 attrs key is restored to `g.attrs["edd_labelling"]`. An earlier `--fix` pass of `check_british_spelling.py` had renamed it to `edd_labeling`, but the shipped `data/slone_netzer_disc_grid.h5` was not regenerated and still carries the old key on disk, desynchronizing the generator script from its own committed output. No consumer reads this key today, so this had no runtime effect; `check_british_spelling.py` gains a scoped allowlist entry for the on-disk key spelling.

- `agn={'type': 'none'}` built a model that raised `Unknown AGN model 'none'`
  on the first `predict_photometry`. `'none'` is the grammar's universal off
  switch — `neb`, `shock`, `radio`, `xray`, `igm` and both dust groups all take
  it — but the AGN translator forwarded it to `agn_model` as though it named a
  model. It now normalizes onto the same off sentinel an omitted `agn` group
  carries, so the build has no AGN component and its photometry is bit-identical
  to the omitted-`agn` build's. Writing a sub-block beside the off switch
  (`agn={'type': 'none', 'disc': {...}}`) is refused rather than silently
  dropped (#2186).

- `tools/check_param_grid_extent.py` had no `GRID_EXTENT_SOURCES` entries for
  any of the five Feltre+2016 NLR grid axes (`agn_nlr_xi_d`,
  `agn_nlr_alpha_pl`, `agn_nlr_logU`, `agn_nlr_logZ`, `agn_nlr_logn`), so a
  declared-bound/grid-extent drift there would go uncaught. Adding the
  annotations surfaced one: `agn_nlr_logZ`'s declared upper bound, `-1.155`,
  disagreed with the vendored grid's top node, `log10(0.07) =
  -1.154901959985743`, by 9.8e-5 — five orders of magnitude above the guard's
  1e-9 tolerance. Transcribed exactly now; the guard covers 29 cases (#2214).

- The spectroscopy projector multiplied the IGM transmission into the rest-frame
  SED before convolving with the galaxy's own velocity dispersion, so `sigma_v_kms`
  smeared the IGM's sharp Lyman-limit/Lyman-alpha-forest edge — a line-of-sight
  feature imprinted after the light leaves the galaxy, which the galaxy's own
  kinematics cannot broaden. `sigma_v_kms` now acts on the stellar piece before
  the IGM transmission, on every spectrum-prediction path, including
  `analysis.simulate.spectrum_from_sfh` (#2589).

- `agn={'type': 'off'}` raised `agn['type']='off' is not an AGN model` —
  both dust groups already accept `'off'` as a synonym of `'none'`
  (`dust_attenuation`, `dust_emission`), but `agn`'s own validator took
  `'none'` only. `'off'` now normalizes onto `'none'` before the validator
  runs, so the two spellings parse and predict identically (#2214).

## [0.1.0] - 2026-05-22

First public preview release.

### Added

- **The 23 7DT bands, bundled.** `7dt_g`, `7dt_r`, `7dt_i` and the 20
  medium bands `7dt_m400` … `7dt_m875` load by name like any built-in, with no
  network and no cache. They ship inside the package rather than being fetched,
  because they are *total system response* — detector QE and optics folded in,
  which is what 7DT photometry is measured through — and a filter-glass-only
  curve would be a different quantity. New
  `tengri.observation.filters.bundled` resolves them, after both user routes
  and before the SVO registry, so a user curve still shadows them. They appear
  in `tengri.list_filters(survey="7dt")`. Provenance, digests, and the
  regeneration command are in `tengri/data/filters_7dt/PROVENANCE.md`;
  `tools/build_7dt_filter_curves.py` rebuilds them from the delivery.
- **`wave_unit=` on `register_filter` / `register_filter_from_file`.** Accepts
  `"AA"` (default), `"nm"`, `"um"` and converts at the boundary. Stating the
  unit skips the range heuristic, and is the *only* protection against micron
  input, which no range rule can detect: an optical curve in microns lands at
  0.5-0.7 Å, indistinguishable from a real NuSTAR band.

### Fixed

- **Custom filter files advertised `.csv` but could not parse one.**
  `_load_filter_from_directory` listed `.dat`, `.txt`, `.csv` as accepted
  while `_load_filter_file` called bare `np.loadtxt`, which dies on a
  comma-separated file with a header (`could not convert string 'lam,trans' to
  float64`). Curve files are now sniffed for delimiter and header row by parse
  attempt rather than by extension, since a `.csv` of whitespace and a `.dat`
  of commas both occur. Affected both `register_filter_from_file` and the
  `$TENGRI_FILTER_DIR` route.
- **The nanometer guard went silent on the most common nanometer grid.**
  `_warn_implausible_wavelength_range` tested `wave_max < 1000.0`, so a curve
  zero-padded to exactly 300-1000 nm — `wave_max == 1000.0` — did not warn.
  The first user to bring their own curves hit it on all 23 files at once. The
  bound is now the blue edge of GALEX FUV (1340 Å), the bluest bandpass tengri
  ships, and the comparison is inclusive: the rule is "wholly inside the gap
  where the ISM is opaque", which is a physical statement, not a round number.
  This also catches nanometer sets running past 1000 nm. **Fails open, so it
  produced confident nonsense rather than an error.**

- **`tengri.PopulationSEDModel` — hierarchical SubModel.**
  Bundles one `SEDModel` template + a list of per-galaxy data dicts +
  the names of parameters tied across the population (default: the
  two PSD hyperparameters `sfh_field_psd_sigma`,
  `sfh_field_psd_tau_myr`) + their priors. Held by `ForwardModel`
  via the new `ForwardModel.build(population=pop, observation=obs)`
  kwarg slot, so the outer-shell construction signature stays uniform
  across SubModel variants (`SEDModel`, `SpatialModel`,
  `SpatialSEDModel`, `PopulationSEDModel`,
  `PopulationSpatialSEDModel` *(far future)*). The hierarchical
  inference path itself is tracked in
  [issue #211](https://github.com/suchethac/tengri/issues/211); until
  it lands, users continue to drive the fit via
  `tengri.PopulationFitter` directly (legacy entry point preserved
  for backward compatibility).
- **Multi-population galaxy decompositions (ADR-0012 accepted).**
  `ForwardModel.build(populations=[...])` now accepts N > 1
  populations for AGN + bulge + disc and similar galaxy
  decompositions. Parameter names use the namespace
  `"<population_name>.<prefix>_<param>"` (e.g.
  `"disc.sfh_dpl_alpha"`); bare names like `redshift` flow to every
  population. Each population's prediction is summed in linear flux
  at the observation layer via `JointObservation.predict_summed`.
  Cross-population state reads are supported by namespaced keys in
  `state.derived` (e.g. `"agn.L_bolometric"`). The prefix CI guard
  strips the namespace before applying the prefix discipline. See
  `docs/adr/0012-forward-model-population.md`.
- **`tengri.ForwardModel`** — the outer-shell forward-model class.
  Wraps populations + observation and exposes a single
  `.predict(params)` method. See
  `docs/dev/archive/forward-model-architecture.md`.
- **`tengri.Population`** — one (SED, spatial) pair held by
  `ForwardModel`. Spatial submodel is reserved (`None`) in this
  slice.
- **`tengri.protocols.SubModel`** — runtime-checkable Protocol for
  one mode of `ForwardModel` (SED, spatial, joint). Two-method
  contract (`run`, `declared_parameters`).
- **`tengri.protocols.SpatialComponent`** — mirror of `SEDComponent`
  on the spatial side; runtime-checkable Protocol with the same
  `declared_parameters`/`precompute`/`apply` shape.
- **`SpatialModelComponent`** astronomer-facing base class
  (`tengri.components.spatial_model_component`). Mirror of
  `SEDModelComponent`. Auto-discovers class-level `Distribution`
  attrs as free parameters, supports `reads`/`publishes` dicts, and
  provides a default `apply()` that handles param slicing, grid
  lookup, and writes the resulting profile to
  `state.derived["spatial_profile_2d"]`.
- **`tengri.components.spatial.{Sersic, Exponential, FlatSlab}`** —
  three concrete spatial-profile blocks. `Sersic` implements the
  full Ciotti & Bertin (1999) expansion for `b_n`; `Exponential` is
  the standalone n=1 case; `FlatSlab` is the explicit form of the
  uniform-aperture model that classical SED codes use implicitly.
- `DerivedBundle` gained two canonical fields, `spatial_profile_2d`
  and `spatial_grid_xy_kpc`, with matching entries in the
  orchestrator's `_CANONICAL_UNITS` table.
- **`tengri.forward.spatial_model.SpatialModel`** — SubModel composer
  over a list of `SpatialComponent`s. Mirror of `SEDModel` at the
  sub-model layer (no physics of its own; aggregates declared
  parameters; threads `ForwardState` through components).
- **`tengri.forward.spatial_model.SpatialSEDModel`** — joint
  composer holding one SED SubModel + one `SpatialModel`. Runs
  SED → Spatial per architecture spec §4.3 so spatial components can
  optionally read SED-derived state keys. The scientific main path
  for combined spatial+SED fits once observation adapters land.

### Internal

- **`SEDModel` directly satisfies `tengri.protocols.SubModel`.** The
  `run(state, params)` and `declared_parameters()` methods are now on
  `SEDModel` itself, alongside a `name = "sed"` class attribute. The
  transitional `_LegacySEDSubModel` adapter introduced in the
  forward-model tracer-bullet is deleted; `ForwardModel.build` and
  `ForwardModel.predict` consume `SEDModel` instances directly. No
  user-visible change to the public API — additive on `SEDModel`,
  internal cleanup on `ForwardModel`.
- **`ForwardModel.predict` now projects through `Observation.predict`.**
  Previously the outer shell reached into `SEDModel.predict_photometry`
  for the photometric channel; it now follows the architectural seam —
  per-population SED `SubModel.run(state, params) → ForwardState`, then
  `Observation.predict(state, params) → dict`. Fixed parameter values
  are merged into the params dict before projection so callers can pass
  only their free-parameter overrides. No user-visible API change; the
  prediction dict still matches the legacy path numerically.
- Scaffolded per-component `_params.py` skeletons (PR1/5 of the
  parameter-registry consolidation) and extended
  `tengri.core.component.ParamDeclaration` with optional `bound_check`
  and `bound_error` fields. No user-visible change; priors still live
  in `tengri.parameters._param_defs`.
- Moved radio priors into `tengri.components.radio._params.PARAMS`
  (PR2/5). `tengri.parameters._param_defs._RADIO_PARAMS` is now a
  derived view via module-level `__getattr__`, and
  `RadioSEDComponent.declared_parameters()` returns the same tuple —
  drift between the two paths is now structurally impossible. No
  user-visible change.
- Moved AGN and X-ray priors into their component `_params.PARAMS`
  tuples (PR3/5). `_AGN_PARAMS` and `_XRAY_PARAMS` are now derived
  views; the AGN bucket additionally merges a small `_AGN_EXTRAS`
  registry holding the `neb_xid` orphan (consumed by the Feltre NLR
  backend alongside `agn_alpha_ion`). `AGNSEDComponent.declared_parameters()`
  now returns the full ~45-entry tuple — previously a 17-entry subset
  that drifted from the registry. No user-visible change.
- Moved nebular priors into `tengri.components.nebular._params.PARAMS`
  (PR3b/5). `_NEBULAR_PARAMS` is now a derived view.
  `NebularSEDComponent.declared_parameters()` was left unchanged
  because it performs backend dispatch (cloudy_grid / cue / shock /
  baked_in) with intentionally-different `Uniform` priors for the
  SEDComponent path. Unifying the two prior sets is deferred to a
  dedicated nebular PR. No user-visible change.
- Moved IGM declared params and dust-emission priors into
  `tengri.components.{igm,dust}._params.PARAMS` (PR3c/5).
  `IGMSEDComponent.declared_parameters()` now returns the canonical
  IGM tuple; `_DUST_EMISSION_PARAMS` is a derived view via the
  module-level `__getattr__`. The conditional `_IGM_PATCHY_PARAMS`,
  `_DLA_PARAMS`, `_DUST_EXTRA_PARAMS`, and `_SINGLE_COMPONENT_DUST_PARAMS`
  buckets remain in `_param_defs.py` pending PR4's structural
  consolidation. No user-visible change.
- Split `_NON_SFH_PARAMS` and migrated four more conditional buckets
  (PR4/5). Dust attenuation entries (`dust_tau_bc`, `dust_tau_diff`,
  `dust_slope` plus the existing `dust_f_obscuration`,
  `dust_bump_strength`, `dust_delta`, `dust_Rv`) now live in
  `tengri.components.dust._params.ATTENUATION_PARAMS`; the single-screen
  alternative `dust_tau_v` is in `SINGLE_COMPONENT_PARAMS`. The
  patchy-IGM (`igm_x_HI`, `igm_bubble_mpc`) and DLA absorber
  (`dla_log_n_hi`, `dla_z`, `dla_temp`, `dla_b_turb`) buckets moved to
  `tengri.components.igm._params.{PATCHY_PARAMS, DLA_PARAMS}`. The
  legacy bucket names (`_DUST_EXTRA_PARAMS`, `_SINGLE_COMPONENT_DUST_PARAMS`,
  `_IGM_PATCHY_PARAMS`, `_DLA_PARAMS`) remain available as derived
  views via the lazy `__getattr__`. `_NON_SFH_PARAMS` shrinks to its
  genuinely-shared residue: `met_logzsol`, `redshift`, `noise_*`,
  `sigma_v_kms`. No user-visible change.
- Migrated the remaining nebular sub-buckets and stellar α/Fe priors
  (PR5/5). `_CB19_PARAMS`, `_ELINE_PARAMS`, `_ELINE_BROAD_PARAMS`,
  `_CUE_IONSPEC_PARAMS`, `_CUE_GAS_EXTRA_PARAMS`, `_SHOCK_PARAMS` now
  live in `tengri.components.nebular._params`; `_ALPHA_FE_PARAMS` and
  `_EVOLVING_ALPHA_PARAMS` in `tengri.components.stellar._params`.
  Dead `_EVOLVING_MET_PARAMS` and `_CHEM_EVOL_PARAMS` (defined but
  never consumed — superseded by the live `met_registry`) deleted.
  `tengri.parameters._param_defs` shrinks from 1189 lines (PR0
  baseline) to 467 (−61%). No user-visible change.

### Added (5 metallicity modes wired in the orchestrator, 2026-05-06)

`Parameters(met_mode="two_step" | "psb_two_step" | "bins" |
"bins_continuity" | "table")` now have working consumers in
`StellarSEDComponent.apply()`. Previously these modes were
declarable through the registry (and through the auto-infer added
in commit `b1ff2c2`) but raised `NotImplementedError` at chain
runtime. All five dispatch to existing pure-JAX primitives in
`components/stellar/sfh/metallicity_history.py`:

- **`two_step`** — sigmoid-smoothed step at `met_step_age_gyr`
  between `met_logzsol_old` and `met_logzsol_young`.
- **`psb_two_step`** — step tied to the PSB SFH burst onset
  (`sfh_psb_burstage_gyr`); pre-burst → `met_logzsol_old`,
  burst-and-younger → `met_logzsol_burst`.
- **`bins`** — piecewise-constant Z per age bin, parameterised by
  `met_bin_<i>` for `i=0..N-1` (default N=6).
- **`bins_continuity`** — cumulative delta-log-Z steps from the
  oldest bin: `met_logzsol_base` + `met_d_log_z_<i>` for
  `i=0..N-2`.
- **`table`** — user-provided Z(t) on the component config:
  `StellarSEDComponentConfig.met_table_log_age_yr` and
  `met_table_log_z_abs` (constructor-time settings, not JAX
  params).

All five flow through DSPS's `calc_rest_sed_sfh_table_met_table`
(per-age metallicity table) — same code path `ramp` / `chem_evol`
already use. New `StellarSEDComponentConfig` fields: `met_n_bins`,
`met_bin_edges_log_yr` (defaults to log-spaced 1 Myr → 13.7 Gyr,
7 edges = 6 bins to match `MET_REGISTRY`), `met_table_log_age_yr`,
`met_table_log_z_abs`.

Limiting-case tests verify each mode reduces to `delta`-mode SED
when configured to produce constant Z (commit `38eea6e`).

### Removed (Phase II-3 closure: `SEDModel._compute_sed_components`, 2026-05-06)

- The legacy `SEDModel._compute_sed_components` wrapper is **deleted**.
  Every production code path now goes through `predict_via_orchestrator`.
  The five parity-check tests that referenced it have been migrated
  to read `state.sed_intrinsic` and `state.derived["sed_dust_attenuated"]` /
  `["lnu_age"]` from the orchestrator's `PipelineState`. The underlying
  `_sed_pipeline.compute_sed_components` function survives as an
  internal utility (no callers in the source tree); a separate
  cleanup PR can prune it. (commit `72f4b64`)
- **Single-component dust mode now wired in the orchestrator.**
  `Parameters(dust_model="single_component", dust_tau_v=...)` now
  routes to `DustAttenuationSEDComponent` via
  `build_components(dust_model=...)` instead of silently falling back
  to two-component (which would `KeyError` on the missing
  `dust_tau_bc`). The component was updated for the Phase II-3
  contract: it overwrites `state.sed_intrinsic` with the
  post-attenuation SED (matching `DustSEDComponent`) and publishes
  `L_absorbed` + `sed_dust_attenuated`. Without this update,
  `predict_rest_sed` returned the pre-dust SED for single-component
  models — a silent ~1.8× over-prediction. (commit `72f4b64`)

### Fixed (Radio/X-ray panchromatic regression, 2026-05-06)

PR 5a's `predict_rest_sed` migration silently regressed
multiwavelength tests by switching from `self._rest_wavelength`
(panchromatic-extended for radio/X-ray) to `state.wave`
(= `ssp.ssp_wave`, typically 91–100000 Å). Radio/X-ray adapters
write SED on whatever `state.wave` is initialised with, so their
contributions were being clipped to the SSP range.

`predict_via_orchestrator` now initialises `state.wave =
self._rest_wavelength`. `StellarSEDComponent.apply()` projects
`sed_intrinsic` and `lnu_age` onto `state.wave` via linear
interpolation with zero-mask outside the SSP range — pure Python-
level shape branch with zero JIT overhead in the no-extension
case. 24/24 `test_panchromatic_integration` tests now green
(was 21/24). (commit `72f4b64`)

### Changed (Phase II-3 closure: orchestrator path is the single truth, 2026-05-06)

All five `predict_*` methods and the lazy `Prediction` wrapper now
route through `predict_via_orchestrator`. `_compute_sed_components`
has zero production callers; it is preserved only as an internal
parity-check helper for 5 test files. Sub-PR commit map:

- `predict_rest_sed` (default-`wave` and custom-`wave` paths) →
  `predict_via_orchestrator` (`77704a8`, `5f1d862`).
- `predict_line_fluxes` → `state.derived["line_lums"]` (`b7dff1b`).
- `predict_sed_quantities` → `predict_sed_quantities_via_orchestrator`
  (`63f036f`).
- `Prediction._ensure_sfh` and `_ensure_sed` → orchestrator state
  (`0c72ac5`, `6d5ee9f`); single cached `_state` keeps SFH-only and
  SED-consuming properties on the same DSPS-canonical numerics.

The orchestrator integrates the SFH on `spec.n_grid` (default 64)
unconditionally; legacy `SEDModel` used `n_grid=256` for non-stochastic
configs. `Prediction._ensure_sfh` interpolates `sfr_history` onto
`model.age_yr` so age-mask consumers (`sfr_100myr`, `sfr_10myr`)
remain valid.

**Breaking-ish (semantic shift, not API)**: `luminosity_weighted_age_gyr`
now reflects the energy-conserving DSPS-canonical CSP integration.
For models built with `csp_integration='trapz'` (legacy default) the
value drifts by ~12% from previous versions; with `csp_integration='dsps_native'`
the drift is sub-0.1%. The new value is energy-conserving by construction
(its per-age cube IS `sed_intrinsic`); the legacy trapz reconstruction
was incoherent in that sense. Other published quantities (`l_bol`,
`l_tir`, `irx`, `dn4000`, etc.) shift by < 0.13% under either mode.

### Fixed (orchestrator-fidelity gaps surfaced by the migration)

- `met_alpha_fe` was silently dropped by `StellarSEDComponent.apply()`
  on 3D SSP grids. Fixed via the Salaris+05 `effective_metallicity`
  shift (legacy parity); 4D α-grid SSPs raise `NotImplementedError`
  with a clear pointer to the legacy `interpolate_met_alpha` path.
- `dust_emission=None` was silently coerced to `"modified_blackbody"`
  with `dust_eta_balance=1.0`, fabricating thermal IR re-emission for
  users who didn't configure it. Now passes `None` end-to-end;
  `DustSEDComponent.apply()` skips the emission template when
  `emission_model is None`.
- `NebularSEDComponent` was passing `ssp_weights=jnp.ones(n_age)`
  (CloudyGrid) and only `gas_logqion` (Cue), so Cue used its default
  ionising-spectrum shape. `StellarSEDComponent` now publishes
  `age_weights` (Msun/bin); nebular calls run in high-level
  (SSP-derived) mode.
- `neb_logZ_gas` was passed in Z/Zsun while Cue/CloudyGrid expect
  absolute log10(Z). `NebularSEDComponent` now applies the
  `LOG10_ZSUN` offset (legacy `param_map` parity). Was the dominant
  ~240× drift on Cue line luminosities before the fix.
- `state_to_sed_quantities` `l_tir` switched to legacy semantics
  (`compute_l_tir(sed, wave)` integration over 8–1000 μm) for parity
  (`fd07dee`).

### Added (`Parameters.met_mode` auto-inference, 2026-05-06)

`Parameters` now infers `met_mode` from the metallicity-related
prior keys present in kwargs — `met_logzsol_0` + `met_logzsol_final`
implies `"ramp"`, any `chem_*` key implies `"chem_evol"`, etc.
Driven by a registry-driven discriminator table
(`_MET_MODE_DISCRIMINATORS` in
`src/tengri/components/stellar/sfh/met_registry.py`). Explicit
`met_mode=...` still wins; explicit + key-mismatch raises with a
helpful hint.

### Added (precompute coverage + collaborator-handoff polish, 2026-05-06)

- **Precompute coverage now spans every emitter** (commit `531bc4c`).
  CB19 and MAPPINGS-V nebular backends expose
  `preintegrate_for_photometry` with the duck-typed CLOUDY-shape
  surface; pah_drude, casey2012 and modified_blackbody dust analytics
  are wired into the hybrid kernel; AGN disc/torus, AGN-nebular,
  MAPPINGS shock, radio and X-ray all run through the fast path.
  Benchmark in `bench/reports/2026-05-06_forward_model_speedup.md`
  shows 31–424× speedup over the exact path with sub-1 % typical error.
  AGN BLR/NLR config knobs (`agn_blr_enabled`,
  `agn_nlr_gaussian_enabled`, `agn_nlr_backend`) added with validation
  in `AGNConfig.__post_init__`.
- **`get_internal_params(strict_unknown_params=True)` is now the
  default** (commit `531bc4c`). Unknown parameter names raise
  `ValueError` instead of silently warning, so typos surface
  immediately during model construction. Pass
  `strict_unknown_params=False` to restore the legacy warn-only
  behaviour. Test contract pinned in
  `tests/unit/test_cue_param_translation.py`
  (`test_unknown_param_raises_by_default` +
  `test_unknown_param_warns_when_strict_false`).
- **`tengri._display` sink + `TENGRI_QUIET` env var** (commit
  `531bc4c`). 25 user-facing helper outputs (`doctor()`, citations,
  help, search, parameter summary) now route through
  `tengri._display._display`, which respects `TENGRI_QUIET=1` and is
  monkey-patchable. Pinned by `tests/unit/test_display_quiet.py`.
- **Toy AGN torus models warn on use** (commit `531bc4c`).
  `simple_torus` and `two_temperature_torus` emit a once-per-process
  `UserWarning` directing users to the SKIRTOR-based components for
  science fits. They remain reachable for pedagogy.

### Fixed (mstar consistency + Cue test fixture, 2026-05-06)

- **`predict_sfh_quantities` ↔ `predict_derived` 4.1 % stellar-mass
  drift** (commit `4288e4a`). The trapz path in
  `predict_sfh_quantities` was rectangle-rule weighting where the
  orchestrator path uses the DSPS canonical trapezoidal cosmic-time
  integral. `predict_sfh_quantities` now routes through
  `predict_via_orchestrator` and reads `state.derived["age_weights"]`,
  giving bit-identical stellar mass between the two methods. Pinned
  by `test_mstar_consistent_between_methods` and
  `test_mstar_paths_agree_within_tolerance`.
- **`Parameters.is_fixed()` and `Parameters.fixed_value()` restored**
  (commit `531bc4c`). The `is_fixed`/`get_fixed_values` refactor
  removed two single-name accessors that were still called from six
  precompute modules (`stellar/sps`, `feltre`, `mappings_photo`,
  `mappings_shock`, `dust_emission`, `skirtor`). Both methods are
  back as documented public API on top of `_distributions`.
- **CueBackend test fixture missing `ssp_data`** (commit `4288e4a`).
  `state_with_cue` fixture in `test_state_quantities_bridges` now
  passes `ssp_data` at construction, fixing 3 `RuntimeError`
  collection errors.

### Documented (deferred-feature limitations, 2026-05-06)

- **ADAF disc precompute deferral** (`components/agn/disc.py`).
  Replaced the bare `TODO` with a `# Known limitation:` block
  explaining that ADAF precompute waits on the full ADAF rewrite vs
  Mahadevan 1997 (currently flagged in `project_adaf_rewrite.md`),
  with a four-step resolution path.
- **Hybrid kernel mirrored per-component blocks**
  (`forward/_kernels/hybrid.py`). Replaced two `TODO(refactor)`
  comments with a `# Known limitation:` block describing why the
  table-driven dispatch from `core/nonstell.py:build_nonstell_fn()`
  cannot yet absorb the photometry-shortcut paths
  (`_has_preint_dust_ir`, `_has_preint_neb`) and what an interface
  redesign would need to do.

### Changed (Phase 6 second wave — top-level API trim, 2026-05)

`tengri.__all__` shrinks from 73 to ~55 entries. 25 result/observation/
fitter/config classes that were previously top-level are now canonical
under sub-namespaces — old names still resolve via a `__getattr__`
deprecation shim (PEP 562) and emit a one-shot `DeprecationWarning`
pointing at the new path. Will be removed in v1.0.

| Old | New canonical path |
|---|---|
| `tengri.FitResult`, `Provenance`, `MockData` | `tengri.results.*` |
| `tengri.Posterior`, `CatalogPosterior`, `PopulationPosterior` | `tengri.results.*` |
| `tengri.generate_mock`, `posteriors_to_dataframe` | `tengri.results.*` |
| `tengri.Fitter`, `CatalogFitter`, `PopulationFitter`, `VIConfig` | `tengri.inference.*` |
| `tengri.AGNConfig`, `DustConfig`, `NebularConfig`, `SEDModelConfig`, `SFHConfig` | `tengri.config.*` |
| `tengri.Photometry`, `Spectroscopy`, `NoiseModel`, `Observation` | `tengri.observation.*` |
| `tengri.LineList`, `LineFluxData`, `SpectralIndexDef`, `SpectralIndexData` | `tengri.observation.*` |

Pinned by `tests/unit/test_public_api_surface.py` and
`tests/unit/test_public_surface.py`. See
`docs/dev/api_migration_v0.x.md` Phase 6 second wave.

### Changed (Phase 3 — `_sfh` suffix removal, 2026-05)

Inside `tengri.components.sfh`, the redundant `_sfh` suffix on 19
functions was dropped in favour of canonical short names. Old names are
now `deprecated_alias`-wrapped and emit a one-shot `DeprecationWarning`
on call. The `SFH_REGISTRY` references the canonical short names
internally, so registry-driven fits emit no warnings.

Affected names: `constant_sfh`, `exponential_sfh`,
`delayed_exponential_sfh`, `gaussian_sfh`, `lognormal_sfh`,
`powerlaw_sfh`, `skewnormal_sfh`, `truncated_skewnormal_sfh`,
`snorm_burst_sfh`, `snorm_trunc_burst_sfh`, `spline_sfh`,
`dense_basis_sfh`, `dense_basis_pure_sfh`, `dirichlet_sfh`,
`continuity_sfh`, `continuity_flex_sfh`, `psb_continuity_sfh`,
`declining_exponential_sfh`, `constant_then_exponential_sfh`. Pinned by
`tests/unit/test_sfh_deprecations.py`.

### Fixed (gallery examples, 2026-05)

`examples/{agn,dust}/*.py` and `docs/auto_examples/{agn,dust}/*.py`
were importing names removed in the Phase B alias-deletion commits
(`2059c72`, `73aa6a2`) — `get_dust_law`, `get_agn_model`,
`blr_emission`, `nlr_emission`. Updated to canonical names
(`resolve_dust_law`, `resolve_agn_model`, `compute_blr_sed`,
`compute_nlr_sed`). 8 files affected.

### Changed (Phase II-2.6 closure path A — CSP integral canonicalised end-to-end)

The CSP integral migration to the DSPS-canonical joint formulation
(tracked in `docs/dev/20260504-csp-integral-canonicalization.md`) is
now **fully closed across all metallicity branches**. Every
orchestrator-vs-legacy equivalence test in
`tests/integration/test_orchestrator_vs_legacy.py` passes; **no
xfails remain**. Phase B's monolith-deletion gate is unblocked.

What changed numerically:

- **No-α delta-Z path** (default for most users): now goes through
  ``calc_rest_sed_sfh_table_lognormal_mdf`` with grid resolution
  `n_grid=64` (was 256 in legacy non-stochastic) and DSPS canonical
  trapezoidal-in-cosmic-time SFH integration. Previously used a
  rectangle-rule lookback weighting with bilinear metallicity interp
  + JIT-kernel two-step einsum.
- **α-aware path** (when ``met_alpha_fe`` is free or 4D α-grid loaded):
  now does α-only bilinear interp first (giving a 3D `(n_met, n_age,
  n_wave)` cube at the requested α), then the same lognormal MDF
  triweight kernel. Previously did 4D bilinear in (Z, α) directly,
  giving a narrower metallicity distribution.
- **chem_evol path** (gas-regulator metallicity): now passes the
  per-age `log_z_per_age` through ``calc_rest_sed_sfh_table_met_table``.
  Previously collapsed Z(t) to a SED-luminosity-weighted scalar +
  bilinear interp, giving a ~50% UV-side approximation error.

User-visible impact: existing spectra computed with API versions <
this commit will differ from the new path by:

- ≤0.5% on integrated luminosities (L_bol, broad-band magnitudes).
- ~1-3% per-wavelength on metallicity-sensitive line features (CaT,
  Mgb, NaD).
- ~5% per-wavelength for piecewise-constant SFHs (continuity,
  dirichlet) where the legacy rectangle rule amplified the SFH-
  integration mismatch.
- For chem_evol specifically, ~50% per-wavelength UV-side change —
  the new path is the canonical formulation; the old scalar-collapse
  was a known approximation.

Strict xfails flipped to passing tests:
- ``test_orchestrator_rest_sed_bit_exact_to_legacy``
- ``test_orchestrator_tau_close_to_legacy``
- ``test_alpha_zero_matches_no_alpha``
- ``test_chem_evol_orchestrator_rest_sed_close_to_legacy``

### Added (Phase II-2.5c — three bounded-fraction SFH variants pinned)

- ``StellarSEDComponent._SUPPORTED_SFH`` further extends to ``psb``
  (post-starburst, Wild+ 2020), ``delayed_bq`` (delayed
  burst-quench), and ``dense_basis_pure``. Each is pinned by an
  explicit-priors equivalence test in
  ``tests/integration/test_orchestrator_vs_legacy.py`` that supplies
  the registry's default priors directly (the generic
  ``±multiplicative-bound`` generator can't satisfy the ``[0,1]``
  fraction constraints these variants impose).
- ``psb`` matches at ``rtol=1e-1`` (the burst component's sharp DPL
  rise/fall picks up the same log-vs-linear interpolation residual
  as ``const_exp``); the other two match at ``rtol=5e-2``.
- Total orchestrator-supported parametric SFH variants: 16
  (tsnorm, dpl, continuity, dirichlet, dense_basis, lnorm, snorm,
  snorm_burst, tsnorm_burst, norm, const, const_exp,
  continuity_flex, psb, delayed_bq, dense_basis_pure).
  Variants still blocked on the SFH-side CSP-canonicalisation:
  ``exp``, ``dexp``, ``tau``.

### Added (Phase II-2.5b — three more smooth-SFH variants pinned)

- ``StellarSEDComponent._SUPPORTED_SFH`` extends to ``const``,
  ``const_exp`` (constant-then-exponential), and ``continuity_flex``.
  Each is pinned by an equivalence test in
  ``tests/integration/test_orchestrator_vs_legacy.py``.
- Variants whose SFH has a sharp discontinuity (``exp``, ``dexp``,
  ``tau``) diverge by ≥ 13% per-wavelength because the legacy
  log-space SFR interpolation and the orchestrator's linear-space
  interpolation resolve the cutoff differently. They are blocked
  on the SFH-side CSP-canonicalisation work and remain
  unsupported by the orchestrator.
- Variants with bounded-fraction priors (``psb``, ``delayed_bq``,
  ``dense_basis_pure``) need variant-specific test fixtures (the
  generic fixture's ±multiplicative-bound generator can't satisfy
  ``[0,1]`` constraints) and are deferred.
- Total orchestrator-supported parametric SFH variants: 13
  (tsnorm, dpl, continuity, dirichlet, dense_basis, lnorm, snorm,
  snorm_burst, tsnorm_burst, norm, const, const_exp,
  continuity_flex).

### Added (Phase II-2.5 — orchestrator now supports 5 more SFH variants)

- ``StellarSEDComponent._SUPPORTED_SFH`` allowlist expanded with
  ``lnorm``, ``snorm``, ``snorm_burst``, ``tsnorm_burst``, and
  ``norm``. Each has a new equivalence test in
  ``tests/integration/test_orchestrator_vs_legacy.py`` pinning
  orchestrator output to legacy ``predict_rest_sed.sed`` at
  ``rtol=5e-2`` (the same physical-range tolerance as the existing
  ``tsnorm`` test).
- Brings the total of orchestrator-supported parametric SFH
  variants to 10: ``tsnorm``, ``dpl``, ``continuity``, ``dirichlet``,
  ``dense_basis``, plus the 5 added here. The remaining registered
  modes (e.g. ``const_exp``, ``psb``, ``dense_basis_pure``) still
  raise ``NotImplementedError`` from the orchestrator until each is
  similarly pinned.

### Added (Phase II-2 stellar migration finishing — II-2.3 → II-2.6)

- ``StellarSEDComponent`` (``tengri.components.stellar.component``) now
  supports the full Phase II-2 scope through the orchestrator path:
  - **SFH modes**: ``tsnorm``, ``dpl``, ``continuity``, ``dirichlet``,
    ``dense_basis`` + ``field=True`` PSD-governed GP modulation. SFH
    evaluation is registry-driven via ``SFH_REGISTRY[mode].internal_param_map``,
    matching the legacy ``SEDModel._compute_sfr`` translation.
  - **Metallicity modes**: ``delta``, ``ramp``, ``chem_evol``.
- New ``Galaxy.predict(params, backend='legacy' | 'component')``
  unified entry point. ``backend='legacy'`` returns the lazy
  ``Prediction`` (default, unchanged behaviour). ``backend='component'``
  returns the orchestrator's ``PipelineState`` with all cross-component
  derived quantities published. Default remains ``'legacy'`` until
  Phase B v1.0 cutover.
- ``tests/integration/test_orchestrator_vs_legacy.py`` pins the new
  configurations against legacy at:
  - ``tsnorm``: rtol = 9.7e-4
  - ``continuity``: rtol = 6.7e-3
  - ``dense_basis``: rtol = 7.3e-3
  - ``dirichlet``: rtol = 3.0e-2 (piecewise-constant SFH amplifies
    the SFH-integration mismatch tracked in
    ``docs/dev/20260504-csp-integral-canonicalization.md``)
  - ``field=True``: rtol = 3.3e-3 (closed by aligning SFH log-age grid
    with ``make_log_age_grid`` — fixes a 13% divergence flagged before
    the alignment commit)
  - ``chem_evol`` orchestrator-vs-legacy is xfail-strict — the legacy
    ``trapz`` CSP path collapses chem_evol's per-age Z(t) to a scalar
    via SED-luminosity-weighted average; the orchestrator threads
    the full per-age table through ``calc_rest_sed_sfh_table_met_table``.
    The two are different formulations.

### Added (compute_dsps_age_weights helper for future SFH-side migration)

- New ``tengri.components.stellar.sps.dsps_wrapper.compute_dsps_age_weights(
  sfr_on_ssp_ages, ssp_ages_yr, ssp_lg_age_gyr, t_obs_gyr)``
  helper computes the SFH→age weight tensor (Hearin+ 2021 Eq. 9)
  in absolute mass units, without the metallicity dispatch. Mirrors
  the negative-cosmic-time safety from ``compute_dsps_native_weights``
  (T_TABLE_MIN ramp + zero SFR for invalid bins).
- Building block for the next-step SFH-side canonicalisation.
  **Not yet wired into the legacy CSP path** — doing that cleanly
  requires also migrating the JIT-fused Tier 2 kernels
  (``_compositional.rest_sed`` / ``_compositional.photometry`` /
  hybrid-spec) and regenerating golden-value snapshots, which is a
  multi-PR effort. Available now for any caller that wants
  explicit DSPS-canonical age weighting.

### Fixed (compute_dsps_native_weights NaN robustness for too-old SSPs)

- ``compute_dsps_native_weights`` now safely handles SSP grids
  whose ages exceed ``t_obs`` (the universe age at the observation
  redshift). Previously, the implied negative-cosmic-time bins
  caused DSPS to NaN at ``cumulative_mstar_formed → log10(0)``
  because ``t_table[0] < dsps.constants.T_TABLE_MIN = 0.01 Gyr``.
- New behaviour: invalid SSP bins (``t_obs - ssp_age ≤ 0``) get a
  strictly-monotonic linear ramp at ``T_TABLE_MIN`` with SFR set to
  zero, contributing no mass to the CSP integral. Valid bins are
  also floored at ``T_TABLE_MIN`` so very-high-z observations don't
  underflow.
- This was an architectural prerequisite for the SFH-side
  canonicalisation (replacing the legacy lookback-rectangle SFH
  integration with DSPS's trapezoidal-in-cosmic-time scheme), but
  the SFH migration itself remains deferred: aligning it requires
  also migrating the α-fallback's SFH integration (currently
  shares ``weights = sfr_on_ssp * _csp_age_dt`` with the no-α
  branch). The 0.2% strict-gating residual stays as
  ``xfail(strict=True)`` for now.

### Fixed (α-aware fallback path canonicalised on DSPS lognormal MDF)

- ``interp_met_alpha_dispatch`` (the α-aware fallback when no real
  4D α-grid is loaded) now applies the DSPS-canonical lognormal-MDF
  triweight kernel on ``effective_metallicity(log_z, alpha_fe)``,
  matching the no-α-enhancement path of the previous
  Z-canonicalisation commit.
- Result: ``alpha_fe = 0`` now reduces **bit-exactly**
  (``rtol = 1e-12``) to the non-α path. Restored
  ``test_alpha_zero_matches_no_alpha`` to its original strict
  tolerance.
- Net test suite delta after this commit: **+5 tests** (4963 →
  4968 passing). 0 new regressions; the 3 pre-existing failures
  (``test_dl07_hybrid_error_below_2pct``,
  ``test_dl07_worst_case_hybrid_error_below_2pct``,
  ``test_auto_high_d_routes_to_vi``) are unaffected.

### Added (Phase E — observation sub-namespaces)

- New additive sub-namespaces under ``tengri.observation``:
  - ``tengri.observation.containers`` (8 names) — user-facing data
    classes: ``Photometry``, ``Spectroscopy``, ``LineFluxData``,
    ``LineList``, ``NoiseModel``, ``Observation``,
    ``SpectralIndexData``, ``SpectralIndexDef``.
  - ``tengri.observation.physics`` (20 names) — transformation
    primitives: ``apply_calibration``, ``apply_lsf``,
    ``build_eline_design_matrix``, ``marginalize_calibration``,
    ``marginalize_emission_lines``, etc.
  - ``tengri.observation.constants`` (9 names) — catalogs and status
    flags: ``DETECTED``, ``UPPER_LIMIT``, ``LOWER_LIMIT``,
    ``DEFAULT_LINE_NAMES``, ``DEFAULT_LINE_WAVELENGTHS``,
    ``CLOUDY_LINE_NAMES``, ``CLOUDY_LINE_WAVELENGTHS``,
    ``STANDARD_INDICES``, ``SSP_LIBRARY_RESOLUTIONS``.
- Sub-namespace bindings are ``is``-identical to the flat-surface
  bindings (verified by ``test_observation_subnamespaces.py``).
- ``LineList`` is now exported from ``tengri.observation`` directly
  (was previously only reachable via top-level ``tengri.LineList``).
- The flat ``tengri.observation.X`` surface is unchanged and emits
  no DeprecationWarning. Phase B will collapse the flat re-exports
  into deprecation shims pointing at the sub-namespaces after
  Paper I submission.

### Changed (Phase A — public-API housekeeping)

- ``tengri.pipeline.__all__`` now re-exports
  ``CloudyELineMarginalisedLikelihood``, ``ELineFittedLikelihood``,
  and ``CalibrationELineMarginalisedLikelihood`` (28 → 31 names) —
  the marginalised-likelihood cohort is fully discoverable from the
  pipeline namespace.
- ``build_loglikelihood_unbounded_fn`` (in
  ``inference/loss_functions.py``) now composes directly around
  ``_build_data_neg_log_likelihood_fn``, matching the shape of
  ``build_loss_fn`` and ``build_loglikelihood_fn``. All three
  wrappers share the same data-term core and cannot drift in sign,
  formula, or branch coverage.

### Changed (CSP integral canonicalised on DSPS — Hearin+ 2021 Eq. 11)

- Legacy ``compute_sed_components`` (in ``forward/pipeline.py``) no
  longer applies bilinear ``interp_metallicity`` (which assumed
  σ_MDF = 0 — a delta MDF). It now applies the DSPS-canonical
  lognormal-MDF triweight kernel (``calc_lgmet_weights_from_lognormal_mdf``)
  for the no-α-enhancement case, then marginalises via
  ``einsum("m,maw->aw", lgmet_w, ssp_flux)``.
- This brings legacy and orchestrator paths into close agreement
  (~0.2% per-wavelength residual; was 158%) and matches what DSPS
  endorses as canonical (Hearin+ 2021, Eq. 11). The ~0.2% residual
  comes from legacy SFH integration (rectangle rule on lookback time)
  vs DSPS canonical (trapezoidal in cosmic time) — closing this gap
  is the next milestone.
- The α-enhancement path (4D bilinear in (Z, α)) is unchanged for
  now. Migrating it to the joint einsum is the next milestone after
  closing the SFH-side residual.
- See ``docs/dev/20260504-csp-integral-canonicalization.md`` for
  rationale and implementation notes.
- **User-visible spectral changes**: galaxies with old/metal-rich
  populations may show ~3-4% per-wavelength differences from prior
  versions on metallicity-sensitive features (CaT, Mgb). Total
  integrated luminosities (L_bol, broad-band magnitudes) change by
  ≤0.5%. To recover the prior delta-MDF behaviour, set
  ``lgmet_scatter`` close to 0.05 in the params dict (this
  effectively concentrates the triweight kernel on a single SSP
  metallicity bin given Δlog Z ≈ 0.4 dex grid spacing).
- Test impact: 1 unit test (``test_alpha_zero_matches_no_alpha``)
  had its tolerance widened from rtol=1e-12 to rtol=5e-2,
  documenting the α-aware vs non-α path divergence (closed by the
  next milestone). Two pre-existing failures
  (``test_dl07_hybrid_error_below_2pct``, ``test_auto_high_d_routes_to_vi``)
  are unaffected by this change.

### Fixed (Phase II-2.9 — fixed-param injection in orchestrator path)

- ``SEDModel.predict_via_orchestrator(params)`` now injects fixed
  values from ``self.spec`` for any parameter absent from ``params``,
  matching the legacy ``predict_rest_sed`` /
  ``predict_*`` convention via ``spec.get_fixed_values()``.
- Before: caller had to pass the full param set (free + fixed),
  causing ``KeyError`` on ``met_logzsol``/``redshift``/etc. when
  using realistic free-only param dicts (the divergence cited in
  Phase II-2.8's blocker xfail).
- After: free-only params dicts work end-to-end through the
  orchestrator. Explicit values still override the spec's fixed
  defaults via standard dict-merge semantics (``{spec_fixed,
  **user_params}``).
- Two new tests in ``tests/integration/test_orchestrator_vs_legacy.py``
  pin the injection (``test_orchestrator_injects_fixed_values_from_spec``)
  and the override-precedence (``test_orchestrator_explicit_param_overrides_spec_fixed``).
- Closes one of the two divergence sources blocking the gating
  xfail. Remaining: DSPS joint-vs-separable mass factorization
  (legacy and orchestrator now agree on integrated L_bol to 0.4%,
  but per-wavelength SED shape still differs because of the
  factorization mismatch).

### Added (Phase II-2.8 — orchestrator vs legacy gating tests)

- ``tests/integration/test_orchestrator_vs_legacy.py`` pins the
  orchestrator-path-vs-legacy-path comparison that gates monolith
  deletion. Three sanity tests pass (shape, finiteness, physical
  M* agreement within 0.1 dex). One ``xfail(strict=True)`` test
  documents the **gating criterion** for monolith deletion: strict
  ``rtol=1e-6`` bit-exact equality between
  ``predict_rest_sed(params).sed`` and
  ``predict_via_orchestrator(params).sed_intrinsic``.
- Closing that ``xfail`` is the blocking work item before any
  ``sed_model.py`` / ``pipeline.py`` legacy branch can be deleted.
  Documented divergence sources: (1) DSPS joint vs separable mass
  factorization (legacy still on the separable path), (2) interface
  mismatch — orchestrator expects fixed params in the dict; legacy
  reads them from ``spec``.

### Added (Phase II-2.7 — filter pre-integration utility)

- ``tengri.forward.preintegrate_ssp_filter_grid(ssp_data, filter_waves,
  filter_trans, redshift=0.0)`` — JAX-compatible utility that convolves
  every SSP template with every filter at construction time, returning
  an ``(n_met, n_age, n_filters)`` array. For SDSS ugriz on the
  PRSC-MILES grid this is **~1200× smaller** than the underlying
  ``(n_met, n_age, n_wave=5994)`` SSP cube (63.8 MB → 53 KB).
- This is the architectural **ingredient** for fast photometry-only
  orchestrator workflows. Wiring it into a photometry-mode
  ``StellarSEDComponent`` variant — and parallel photometry-mode
  Dust / Nebular / AGN / Radio / X-ray adapters operating on the
  ``(n_filters,)`` axis instead of ``(n_wave,)`` — is the follow-up
  architectural pass (tracked but not blocking Phase II-2 closure).
- New tests in ``tests/unit/test_filter_preintegrate.py`` pin the
  shape, finiteness, redshift-shift behaviour, and ≥1000× compression
  invariant.

### Added (Phase II-2.6 — Quantities bridges from PipelineState)

- ``tengri.forward.state_to_sfh_quantities(state)`` — converts an
  orchestrator ``PipelineState`` into the legacy ``SFHQuantities``
  NamedTuple (all 7 fields populated; mass-weighted age and metallicity
  computed from the published SFH grid).
- ``tengri.forward.state_to_sed_quantities(state)`` — converts to
  ``SEDQuantities``; **all 15 fields populated** (commit b83981e
  added ``luminosity_weighted_age_gyr`` and
  ``luminosity_weighted_metallicity`` using ``state.derived["L_age"]``
  + linear-interpolated metallicity history onto the SSP age axis).
  Pre-dust UV uses the per-age cube reconstructed before dust
  overwrites ``sed_intrinsic`` (fuv_intrinsic / fuv ~3 with
  tau_bc=0.5 — physical).

These are the first half of full ``Prediction`` parity (which gates
monolith deletion). Existing code reading
``predict_sfh_quantities(...).stellar_mass`` keeps working when the
prediction is sourced from ``run_components``.

### Added (Phase II-2.6 — emission-lines bridge: full Prediction parity)

- ``NebularSEDComponent.apply`` now also calls
  ``backend.predict_nebular_line_luminosities`` (when supported —
  Cue, CloudyGrid) and publishes the discrete line catalogue as
  ``state.derived["line_waves"]`` / ``state.derived["line_lums"]``.
- ``tengri.forward.state_to_emission_lines(state)`` extracts the 11
  standard survey-diagnostic lines via the legacy nearest-wavelength
  matcher. Returns all-NaN when the active backend didn't publish a
  catalogue (BakedIn, shock).
- ``SEDModel.predict_emission_lines_via_orchestrator(params)``
  exposes the bridge as a method.
- Together with SFH / SED / Radio / XRay / Ionizing bridges, this
  completes the **JAX-pytree mirror of the legacy Prediction object**.
  Smoke fixture (Cue + PRSC-MILES) returns Balmer decrement
  Hα/Hβ ≈ 2.8 (near case-B), [OIII]5007/Hβ ≈ 2.9, [NII]6584/Hα ≈ 0.23.

### Tests (Phase II-2.6 — bridge regression suite)

- ``tests/integration/test_state_quantities_bridges.py`` — **13 tests
  pinning all six bridge types** (SFH, SED, Radio, XRay, Ionizing,
  Lines). Includes physical-range assertions (Balmer decrement
  Hα/Hβ ≈ 2.85 for the Cue chain), JIT-compatibility (rtol=1e-12),
  and the explicit "no-AGN → l_x_agn=0" / "no-Cue/Cloudy → all-NaN
  Lines" boundary cases.

### Added (Phase II-2.6 — Radio/XRay/Ionizing Quantities bridges)

- ``RadioQuantities``, ``XRayQuantities``, ``IonizingQuantities``
  NamedTuples + ``state_to_radio_quantities``,
  ``state_to_xray_quantities``, ``state_to_ionizing_quantities``
  bridges.
- Radio: ``l_1p4ghz`` (interpolated from L_radio at 21 cm),
  ``l_thermal`` (free-free), ``l_nonthermal``, ``q_ir``.
- X-ray: ``l_x_xrb`` (Lehmer+10/16), ``l_x_agn`` (Duras+20; returns
  0 when no AGN component is in the chain rather than NaN),
  ``l_x_total``.
- Ionizing: ``q_h`` = ``state.derived["nion"]``,
  ``xi_ion`` = q_h / νLν(1500 Å).
- Re-exported from ``tengri.forward``.

### Added (Phase II-2.6 — predict_*_quantities_via_orchestrator)

- ``SEDModel.predict_sfh_quantities_via_orchestrator(params)`` and
  ``SEDModel.predict_sed_quantities_via_orchestrator(params)``: drop-in
  replacements for the legacy ``predict_sfh_quantities`` /
  ``predict_sed_quantities`` that route through the orchestrator and
  return the same legacy ``SFHQuantities`` / ``SEDQuantities``
  NamedTuple shapes via the state-to-quantities bridge. JIT-compatible.
- ``SEDModel.predict_radio_quantities_via_orchestrator(params)``,
  ``predict_xray_quantities_via_orchestrator(params)``,
  ``predict_ionizing_quantities_via_orchestrator(params)``: same
  pattern for the new ``RadioQuantities``/``XRayQuantities``/
  ``IonizingQuantities`` types. SEDModel now exposes 5
  ``predict_*_via_orchestrator`` entry points (Lines is the remaining
  hold-out, gated on backend-specific extraction).

### Added (Phase II-2.6 — SEDModel + Galaxy bridges to orchestrator)

- ``SEDModel.predict_via_orchestrator(params)`` — public bridge from
  the legacy ``SEDModel`` configuration surface to the Phase II
  ``SEDComponent`` orchestrator. Builds the chain via
  ``_build_component_chain()`` (maps ``spec.mean_sfh_type``,
  ``self._met_mode``, the dust/AGN/nebular/radio/xray/IGM flags, and
  the pre-constructed nebular backend instance to
  ``tengri.forward.build_components`` kwargs) and threads ``params``
  through ``run_components``. Returns a ``PipelineState`` directly;
  the legacy ``predict_photometry`` / ``predict_spectrum`` paths are
  unchanged. Wrapping the state in a legacy ``Prediction`` shape is
  a follow-up.
- ``Galaxy.predict_via_components(params)`` — convenience wrapper that
  lazy-builds the underlying ``SEDModel`` and delegates. Lets users
  with ``Galaxy.from_arrays(...)`` reach the orchestrator with one
  call.

### Fixed (Phase II-4.1 — NebularSEDComponent prefix covers all backends)

- ``NebularSEDComponent.parameter_prefix`` is now a tuple
  ``("neb_", "shock_", "ionspec_", "gas_")`` instead of a single
  ``"neb_"``. Previously the orchestrator's prefix-slicer dropped
  every ``shock_*`` key when MAPPINGS V was active, raising
  ``KeyError`` inside ``apply``. Cue + CloudyGrid were also silently
  missing their ``ionspec_*`` / ``gas_*`` extras unless those keys
  were threaded via a different prefix. After the fix, Cue and
  MAPPINGS both run end-to-end (Cue ~0.7% of stellar SED; MAPPINGS
  ~16%).

### Fixed (Phase II-2 — CSP integration via DSPS joint weights)

- `StellarSEDComponent.apply` now produces physically correct
  bolometric luminosities. The previous implementation used the
  separable factorisation (lgmet_w ⊗ age_w) of DSPS's joint weights,
  which gave the right marginals but the wrong per-bin product when
  convolved with ``ssp_flux``, over-scaling L_bol by ~10³×.
  After the fix: ``L_bol/M* ≈ 2.7 Lsun/Msun`` for a 10¹⁰ Msun population
  with a 2 Gyr-peaked tsnorm SFH; ``L_ir/L_bol ≈ 0.42`` for
  ``tau_bc=1, tau_diff=0.3``; energy balance is exact.
- `compute_dsps_native_weights` and `compute_dsps_met_table_weights`
  in `components/stellar/sps/dsps_wrapper.py` apply the same fix —
  use the joint ``result.weights`` (n_met, n_age) directly instead of
  the separable approximation. Public API unchanged. Legacy callers
  (`forward/pipeline.py`, `forward/sed_model.py`,
  `forward/_kernels/compositional.py`) inherit the fix; their
  bolometric output now matches DSPS's authoritative ``rest_sed``.
  ⚠️ Crossval baselines that compared the legacy path against
  bagpipes/FSPS may need re-recording because the absolute
  bolometric was previously wrong.

### Performance (Phase II — orchestrator JIT compile-time benchmark, refreshed 2026-05-04)

- New benchmark file ``bench/results/orchestrator_jit_benchmark.json``
  records cold (cache-hit) and warm (in-process) JIT-compile times
  for the orchestrator path on PRSC-MILES SSP, CPU. Sample numbers:

  ===================  ============  ============
  chain                cold (ms)     warm-min (ms)
  ===================  ============  ============
  stellar_only         ~520          ~0.8
  stellar_dust         ~460          ~1.3
  stellar_dust_igm     ~520          ~1.8
  full_chain (7 comp)  ~465          ~1.9
  ===================  ============  ============

  The ~500 ms cache-hit cold floor is intrinsic to XLA compiling the
  8 MB SSP-grid einsum + DSPS internals. Warm runs are < 2 ms — fast
  enough that compilation is < 25% of a typical 1000-step inference.

### Performance (Phase II — orchestrator JIT compile-time benchmark)

- Cold compile of the full 7-component chain
  (Stellar+Nebular+AGN+Dust+Radio+XRay+IGM) at z=0 on PRSC-MILES SSP:
  **~885 ms**. Warm runs: **~2 ms**. The plan's 2× ceiling (vs
  ~64 ms per-fusion baseline at `bench/results/jit_compile_benchmark.json`)
  is exceeded for cold compile; the baseline was a single tier-2
  photometry fusion, not the full chain. Warm-run latency is in line
  with legacy. Cold-compile optimisation is deferred follow-up.

### Added (Phase II-2.6 + II-4.1 — public-API orchestrator + full nebular params)

- **`tengri.forward.build_components(...)`** — public-API factory for
  the orchestrator chain. Assembles an ordered list of
  :class:`SEDComponent` adapters from a flat keyword-argument call.
  Re-exported alongside ``run_components`` and the new
  ``chain_summary`` helper from ``tengri.forward``. Users opt into the
  component-orchestrator path via:

  .. code-block:: python

     from tengri.forward import build_components, run_components
     components = build_components(
         ssp_data=ssp,
         sfh_model="tsnorm", metallicity_model="ramp",
         dust_law_bc="calzetti", dust_emission_model="dale2014",
         agn_model="standard",
         use_radio=True, use_xray=True, use_igm=True,
     )
     state = run_components(components, PipelineState(wave=ssp.ssp_wave), params)

  ``SEDModel`` is untouched — the legacy tier-dispatch path keeps
  working; the orchestrator is reachable in parallel. Full
  integration into ``SEDModel.predict()`` (with ``use_component_orchestrator``
  flag and benchmark parity) is follow-up work.
- **`NebularSEDComponent` now declares full parameter sets per
  backend**:

  - ``cue`` declares **14** parameters (4 standard nebular + 7
    ionspec_* shape + 3 gas_* extras: ``gas_logn``, ``gas_logno``,
    ``gas_logco``); ``apply()`` forwards every declared param to
    ``predict_nebular_sed`` so users get the full 12-param Cue
    surface.
  - ``cloudy_grid`` declares 4 standard nebular params.
  - **NEW** ``shock`` backend (MAPPINGS V): declares 4 distinct
    params (``shock_velocity``, ``shock_log_density``,
    ``shock_b_over_sqrt_n``, ``shock_log_lhalpha``). The
    string-valued ``shock_abundance`` and ``shock_component``
    are configured on the pre-constructed ``ShockBackend`` instance,
    not free params.
  - ``baked_in`` declares 0 params (unchanged).
- ``apply()`` reads ``state.derived["nion"]`` → ``gas_logqion`` for
  Cue (when not overridden via ``params``), and
  ``state.derived["shock_log_lhalpha"]`` → fallback to
  ``params["shock_log_lhalpha"]`` for MAPPINGS.

### Added (Phase II-3 / II-4 / II-5 — Dust + Nebular + AGN components, full-chain composability)

- **`DustSEDComponent`** at `src/tengri/components/dust/two_component.py`
  — full Charlot-Fall two-component attenuation + IR re-emission with
  energy balance. Reads stellar's published `lnu_age` cube and applies
  an age-dependent (n_age, n_wave) transmission, then computes
  `L_ir = abs(trapezoid(absorbed_lnu, dν))` and dispatches to any
  registered IR-emission template via
  `resolve_emission_model(self.config.emission_model)`. Publishes
  `state.derived["L_ir"]` for radio/X-ray to consume. Lives alongside
  the existing single-screen `DustAttenuationSEDComponent`.
- **`NebularSEDComponent` extended** to dispatch on backend: BakedIn
  (no-op marker, unchanged), CloudyGrid (HDF5 grid interpolation),
  Cue (NN emulator). Cue/CloudyGrid require a pre-constructed
  backend instance via the new ``backend`` constructor field; they
  declare ``neb_logU``/``neb_logZ_gas``/``neb_fesc``/``neb_fesc_lya``.
  Adds nebular SED to ``state.sed_intrinsic`` so the downstream dust
  attenuation transmits it.
- **`AGNSEDComponent`** at `src/tengri/components/agn/component.py`
  — wraps `resolve_agn_model()` so any registered model (simple,
  standard, kubota_done_full, adaf, unified_nlr_blr, …) is plug-and-
  play via `config.model`. Publishes
  `state.derived["L_agn_bol"] = 10**agn_log_lbol × L_SUN` for
  X-ray (and Radio's AGN-loudness branch). Respects the CLAUDE.md
  gotcha: ``agn_torus_frac`` is an independent free parameter, never
  derived from inclination.
- **Full-chain composability verified**: the orchestrator pipeline
  ``Stellar + Nebular(BakedIn) + AGN + Dust + Radio + XRay + IGM``
  composes correctly across **2 AGN models × 4 dust laws × 2 IR
  emission templates = 16 combinations**, with bit-exact JIT match
  to eager (rtol=1e-12). Locked in via 19 parametrised tests at
  `tests/integration/test_orchestrator_jit.py`.
- **Architectural payoff**: every physics block is now swappable
  by changing a single config string. Adding a new dust law /
  emission template / AGN model means registering it in the
  existing `DUST_LAWS` / `EMISSION_MODELS` / `AGN_MODELS`
  registries; the orchestrator chain doesn't change.

### Added (Phase II-2.3 / II-2.4 / II-2.5 — StellarSEDComponent feature parity)

- **GP-field SFH branch** (II-2.3): `config.field=True` is now supported.
  When set, the mean SFH is multiplicatively modulated by a
  PSD-governed Gaussian process: `SFR_total(t) = SFR_mean(t) ×
  exp(x(t) - K(0)/2)`, where `x(t)` is a DRW realisation generated
  from `sfh_field_xi` (n_grid Gaussian draws) via
  `compute_field_gp(...)` and `K(0)/2 = sigma²/4` is the lognormal
  bias correction. User-facing `sfh_field_psd_tau_myr` is converted
  to `psd_tau_yr` internally (×1e6) per CLAUDE.md convention.
- **Ramp metallicity** (II-2.4): `config.metallicity_model="ramp"`
  now produces a per-age linear interpolation between
  `met_logzsol_0` (oldest stars) and `met_logzsol_final`
  (present-day) via `compute_log_z_evolving`. The component
  switches CSP backend from
  `compute_dsps_native_weights` (single scalar lgmet) to
  `compute_dsps_met_table_weights` (per-age lgmet table) for the
  ramp path. Mass-remaining interpolation uses the present-day
  metallicity. `chem_evol`, `two_step`, `bins`, `tabulated`
  remain `NotImplementedError` for follow-up.
- **DPL SFH** (II-2.5 partial): `config.sfh_model="dpl"` is now
  supported alongside `"tsnorm"`. Both compose with any
  `metallicity_model` and `field` setting. Non-parametric forms
  (`continuity`, `dirichlet`, `dense_basis`) remain
  `NotImplementedError` — they need vector-valued parameter
  declarations and per-bin handling that deserves a dedicated PR.
- All combinations smoke-tested at z=0 on the PRSC-MILES SSP
  (`data/ssp_prsc_miles_chabrier_*.h5`), producing finite
  derived-key values; JIT-compiled paths match eager at rtol=1e-12.

### Added (Phase II-2.2-followup — PipelineState as JAX pytree)

- `tengri.core.PipelineState`, `SEDComponentState`, and
  `SEDComponentConfig` are now registered as JAX pytrees via
  `jax.tree_util.register_dataclass`. Threading these dataclasses
  through `jax.jit` / `jax.grad` / `jax.vmap` previously errored with
  *"Error interpreting argument as an abstract array"*; they now flow
  natively. Verified end-to-end: `Stellar + Radio + XRay + IGM` chained
  through `run_components`, JIT-compiled, produces bit-exact (rtol=1e-12)
  match to the eager path; gradients are finite with the expected signs.
  This unblocks **II-2.6** (orchestrator JIT integration) at the
  machinery level — what remains for II-2.6 is wiring into
  `Galaxy.predict()` / `SEDModel.predict()` as a public-facing path.

### Added (Phase II-2.2 — StellarSEDComponent first-slice apply())

- `StellarSEDComponent.apply()` (Phase II-2.2 slice) now implements the
  full physics path for `sfh_model="tsnorm"` + `metallicity_model="delta"`
  + `sps_backend="dsps"` + `field=False`. Other configurations raise
  `NotImplementedError` until later sub-PRs (II-2.3 → II-2.5) land.
- The component publishes the 11-key stable contract documented in
  `docs/dev/phase_ii_2_stellar_migration.md`: `log_mstar`,
  `log_mstar_formed`, `sfr`, `sfr_10myr`, `sfr_100myr`, `L_age`,
  `lnu_age`, `nion`, `sfh_grid_lbt_yr`, `sfr_history`,
  `log_metallicity_history`.
- `nion` (ionising photon rate, λ < 911.76 Å) integrated via the
  JAX-friendly `where`-mask pattern, mirroring the eager-numpy legacy
  at `components/nebular/ionizing_spectrum.py:299`.
- Architectural decision: `ssp_data` is held on the component instance
  as a constructor field (parallel to `config`), and `precompute()`
  remains a no-op marker — consistent with `RadioSEDComponent`,
  `IGMSEDComponent`, `XRaySEDComponent`. Documented at the top of
  `components/stellar/component.py`.
- Smoke-tested on `data/ssp_prsc_miles_chabrier_*.h5` at z=0:
  produces physically sensible quantities (1.22e10 Msun for ~10 Msun/yr
  peak SFH, 6.9e52 photons/s ionising rate). Bit-exact rtol=1e-8
  equivalence vs legacy `SEDModel.predict` is deferred to a follow-up;
  both paths share the same DSPS edge case where
  `t_obs < ssp_lg_age_gyr.max()` produces NaN (upstream DSPS issue).

### Changed (Phase II-2.1 — stellar package consolidation)

- `tengri.components.sfh` and `tengri.components.sps` have been folded
  into a unified `tengri.components.stellar` package
  (`stellar/sfh/`, `stellar/sps/`). The old dotted paths remain
  importable as deprecation shims at
  `src/tengri/components/{sfh,sps}/__init__.py` — they fire one
  `DeprecationWarning` on first import and forward all attribute and
  submodule access to the new locations via `sys.modules` aliasing.
  No call-site updates are required for downstream code; the shims
  will be removed in tengri v1.0.
- The top-level convenience aliases `tengri.sfh`, `tengri.sps` (and the
  new `tengri.stellar`) now resolve to the canonical locations
  without firing a deprecation warning.
- All `src/tengri/` internal imports were updated to the canonical
  paths; the only remaining users of the old paths are tests and
  external notebooks/scripts, which exercise the shim and validate
  back-compat.

### Changed (Phase II-2.3 — full legacy χ² migration + edge-case assertions)

Three further follow-ups, picked for typical astronomy use cases:

- **New combined adapter** `CalibrationELineMarginalisedLikelihood`
  (in `likelihoods.marginalised`). Covers the most common galaxy
  spectroscopy configuration — Prospector-style joint Chebyshev
  calibration + emission-line amplitude marginalisation. Sequential
  composition: marginalise lines (returning MAP amplitudes) → augment
  the prediction with `G @ a_hat` → run cal-marg on the line-augmented
  model. Supports both flat and Cloudy line priors via the
  `eline_prior_type` flag, both for spectroscopy and joint data.
  `cal_marg + eline_fitted` (mixed marg/fit) raises
  `NotImplementedError` rather than silently degrading.
- **Joint + variable-noise** now auto-builds a `CompositeLikelihood`
  of two `StudentTLikelihood` instances (one per channel sharing
  `f_cal_param="noise_frac_cal"`). Spectroscopy-only + variable-noise
  builds a single `StudentTLikelihood(channel="spec_fnu")`. Was a
  legacy fall-through before.
- **Edge-case bail-outs replaced with assertions**: `n_phot is None`
  and `_wave_obs is None` no longer cause silent legacy fall-through.
  These cases indicate misconfiguration (joint data without
  `model.observation.n_data_phot`, or calibration marginalisation
  without `_wave_obs`) and now raise loudly, rather than producing
  the wrong likelihood with no warning.

After this, `loss_functions.py` legacy switch reduces to two cases:
non-photometry censored data (mask spans concatenated array, not
addressable via single-channel adapters) and a defensive default
diagonal Gaussian fallback that should be unreachable. File:
644 → 548 lines.

### Changed (Phase II-2.2 — eline cohort migrated to adapters + drift fixes)

- **New adapters**: `CloudyELineMarginalisedLikelihood` and
  `ELineFittedLikelihood` in
  `tengri.inference.likelihoods.marginalised`. Both use the existing
  `design_matrix_builder` closure pattern from
  `ELineMarginalisedLikelihood`. `_maybe_build_default_likelihood`
  now wires both, and the corresponding pure-eline branches in the
  `loss_functions.py` legacy χ² switch are gone. The combined
  `cal_marg + eline_{marg,fitted}` case still falls through to
  legacy (sequential composition the auto-build cohort does not yet
  express); auto-build bails to `None` in that case so the legacy
  combined branch fires correctly.
- **Bug fix**: `_maybe_build_default_likelihood` now bails to `None`
  for `data_mask + non-photometry` data. Previously fell through
  to subsequent checks and returned a plain
  `SpectroscopyLikelihood` / `Composite` that silently ignored the
  mask, treating upper-limit pixels as detected zero-flux. Legacy
  fall-through correctly applies censoring across the concatenated
  data, so the bail-out routes spec/joint+mask through it.
- **Renamed**: `tengri.observation.noise.censored_log_likelihood`
  → `censored_neg_log_likelihood`. The function name suggested a
  log-likelihood (positive when fit good) but it returns energy
  (= negative log-likelihood). Every caller already treated it as
  energy; the rename brings the name in line with the convention.
  All 50+ call sites updated.
- File `loss_functions.py`: 704 → 644 lines (further dedup from
  removing pure-eline legacy branches).

### Changed (Phase II-2 — unified loss-function core)

- `tengri.inference.loss_functions` now has a single
  `_build_data_neg_log_likelihood_fn` core. `build_loss_fn`,
  `build_loglikelihood_fn` and `build_loglikelihood_unbounded_fn`
  are thin wrappers over it, so the data term cannot drift in sign
  or branch coverage between the three builders. File shrank
  ~960 → ~700 lines. Two shared helpers,
  `_unstandardize_parameters` and `_build_prediction`, replace the
  inline copies of the unstandardize-and-resolve-mirrors block and
  the `predict_photometry` / `predict_spectrum` dispatch. Public
  API and behaviour unchanged.
- **Bug fix (drift)**: `build_loglikelihood_fn` previously had no
  censored-data branch — NSS evidence and Elliptical Slice Sampling
  on photometry with non-detections silently treated masked bands
  as zero-flux detections. Now eliminated by construction since the
  censored case lives once in the auto-built `CensoredLikelihood`
  shared by all three builders.

### Added (Phase II-1 — auto-build wires every legacy feature through Protocol path)

- **`Fitter._maybe_build_default_likelihood` now handles every case**:
  - censored data → `CensoredLikelihood`
  - Student-t / variable noise → `StudentTLikelihood` (with
    `f_cal_param="noise_frac_cal"` reading from the params dict)
  - calibration marginalisation → `CalibrationMarginalisedLikelihood`
  - flat-prior e-line marginalisation → `ELineMarginalisedLikelihood`
    with a per-call `design_matrix_builder` closure (line wavelengths
    shift with redshift)
  - line fluxes → composed `GaussianLikelihood(channel="line_fluxes")`
  - spectral indices → composed `GaussianLikelihood(channel="indices")`
- The only fall-backs to legacy dispatch that remain:
  - `eline_prior_type="cloudy"` (uses a different math primitive)
  - `_eline_fitted=True` (line amplitudes are explicit free params,
    not marginalised)
- **`StudentTLikelihood` and `CensoredLikelihood`** extended with
  `f_cal_param: str | None` keyword. When set, `log_prob` reads
  `f_cal` from `params[f_cal_param]` at evaluation time — required
  because `noise_frac_cal` is a free parameter the inference engine
  fits.
- **`ELineMarginalisedLikelihood`** extended with optional
  `design_matrix_builder: Callable[[params], ndarray]`. When set,
  the design matrix is rebuilt every `log_prob` call using the
  current params dict (typically wraps
  `_build_eline_G_eff` so line wavelengths shift with redshift).
  Constructor enforces exactly one of `design_matrix` (static) or
  `design_matrix_builder` (dynamic).
- **`build_loss_fn` and `build_loglikelihood_fn`** in `loss_functions.py`
  now populate `prediction["line_fluxes"]` and `prediction["indices"]`
  in the user-likelihood short-circuit when `has_line_fluxes` /
  `has_indices` are configured. Lets composed `GaussianLikelihood`
  constraints score against `model.predict_line_fluxes` /
  `model.predict_spectral_indices` outputs.
- 12 tests in `test_fitter_auto_likelihood.py` validate every
  auto-build branch (10 build the right adapter + 2 confirm the
  remaining fall-backs are intentional). 272/272 pass on the full
  inference + likelihood surface.

### Added (Phase II-1 — full Likelihood adapter cohort, channel-parameterised)

- **Channel-parameterised Likelihood classes**: one class per *math
  type*, parameterised by which prediction-dict key to read.
  Replaces the per-channel-class proliferation that was building up.
  - `GaussianLikelihood(channel, obs, err, sigma_floor=0)` —
    workhorse. `channel="phot_fnu"` for photometry,
    `channel="spec_fnu"` for spec, `channel="line_fluxes"` for line
    flux constraints, `channel="indices"` for spectral index
    constraints, `channel="imaging_fnu_pixel"` for future imaging,
    etc. **The user adds a new observation channel by passing a new
    string** — no new class needed.
  - `StudentTLikelihood(channel, obs, err, dof, f_cal=0)` — heavy-
    tailed alternative for outlier tolerance (wraps
    `variable_noise_hamiltonian`, sign-flipped).
  - `CensoredLikelihood(channel, obs, err, mask, f_cal, dof)` —
    upper / lower limits via the normal CDF (wraps
    `censored_log_likelihood`).
  - `MultivariateGaussianLikelihood(channel, obs, cov_inv)` —
    correlated noise via a pre-inverted covariance matrix
    (replaces the legacy ``spec_cov_inv`` branch).
- **Marginalised Likelihoods** (analytic nuisance integration):
  - `CalibrationMarginalisedLikelihood(fnu_obs, fnu_err, wavelength,
    n_poly, prior_sigma)` — wraps `marginalize_calibration`
    (Chebyshev polynomial integrated out, Prospector approach).
  - `ELineMarginalisedLikelihood(fnu_obs, fnu_err, design_matrix,
    prior_variance)` — wraps `marginalize_emission_lines` (linear
    line amplitudes integrated out).
- **`PhotometryLikelihood` / `SpectroscopyLikelihood` are now thin
  subclasses of `GaussianLikelihood`** — pinned to ``"phot_fnu"`` /
  ``"spec_fnu"`` channels, preserve the legacy ``fnu_obs``/``fnu_err``
  constructor names and attribute aliases. Backward-compatible.
- **Fitter auto-build path** now constructs
  `MultivariateGaussianLikelihood` for spec_cov data (one less
  legacy fall-back). The other 6 cases (cal-marg, eline-marg,
  Student-t, censored, line-fluxes, indices) now have **adapters
  ready to go** — auto-build wiring is documented in the helper
  with explicit migration notes for each case.
- 17 tests at `tests/unit/test_likelihood_full_cohort.py`:
  channel-parameterised behaviour, bit-for-bit equivalence vs
  legacy primitives, outlier robustness (Student-t > Gaussian on
  outliers), upper-limit semantics, MVN ↔ diagonal recovery, all
  adapters compose into a single `CompositeLikelihood`.

### Added (Phase II-1 — auto-build Protocol likelihood from data on Fitter)

- **`Fitter(model, data, noise)` now auto-builds the matching
  `Likelihood` Protocol object** (`PhotometryLikelihood`,
  `SpectroscopyLikelihood`, or `CompositeLikelihood` for joint
  data) when no legacy-only features (calibration marginalisation,
  e-line marginalisation, Student-t / variable noise, spec
  covariance, censored data, line fluxes, spectral indices) are
  configured. Routes simple cases through the new path so
  `diag_gaussian_log_prob` is the single source of truth. Legacy
  dispatch still handles cases that need the extras.
- **Joint data auto-split**: `data_type=="joint"` splits
  `data`/`noise` at `observation.n_data_phot` into
  `PhotometryLikelihood` + `SpectroscopyLikelihood` and wraps in a
  `CompositeLikelihood`.
- **Opt-out via `auto_protocol_likelihood=False`** for users who
  want to force the legacy χ² path even on simple cases.
- 10 tests at `tests/unit/test_fitter_auto_likelihood.py` cover
  the three auto-build cases (phot / spec / joint) and 7 fall-back
  cases (cal-marg / eline-marg / censored / spec-cov / line-fluxes
  / indices / joint-without-n_phot).
- Numerical equivalence to the legacy path is enforced by the
  existing 240-test inference suite, which still passes after
  auto-build is enabled (1 unrelated pre-existing failure).
- Future-extension design notes captured (no code change):
  - `docs/dev/spatial_model_extension.md` — joint imaging +
    photometry + fiber spectroscopy via `SpatialProfileSEDComponent`
    + per-instrument `ObservationModel`s + `CompositeLikelihood`.
  - Memory: `project_spatial_extension.md` and
    `project_transient_extension.md` (time-variable / transient
    fitting via `LightCurveObservationModel`).

### Added (Phase II-1 — `SpectroscopyLikelihood` + `CompositeLikelihood`)

- **`SpectroscopyLikelihood`** at
  `tengri.inference.spectroscopy_likelihood`. Parallel to
  `PhotometryLikelihood` but reads ``prediction["spec_fnu"]``. Same
  shared `diag_gaussian_log_prob` primitive — second concrete adapter
  graduates the `Likelihood` Protocol from "scaffold" to "real seam"
  per the two-adapter rule.
- **`CompositeLikelihood(*likelihoods)`** at
  `tengri.inference.composite_likelihood` — composition primitive for
  joint likelihoods. Sums log-probabilities across constituents,
  unions their declared nuisance parameters (raises on duplicates),
  ignores prediction keys that no constituent reads. Pattern mirrors
  `run_components` for SED forward modules.
- All three new classes re-exported through `tengri.pipeline`.
  Two-adapter rule satisfied: `Photometry` + `Spectroscopy` exercise
  distinct prediction keys; `Composite` validates the composition
  channel.
- 11 tests at `tests/unit/test_likelihood_adapters.py` cover
  contract conformance, numerical equivalence vs the helper, sigma
  floor, composite commutativity, duplicate-param detection,
  declared-param union.
- 1 additional test at `tests/unit/test_user_likelihood_override.py`
  validates `CompositeLikelihood` flows through the legacy `Fitter`
  override path identically to a single-channel wrapper.

### Added (Phase II-1 — user-composable `likelihood=` on `Fitter`)

- **`Fitter(model, data, noise, likelihood=Custom)`** — option β API.
  When the user passes a `Likelihood` Protocol object, the entire
  built-in χ² dispatch in `build_loss_fn` / `build_loglikelihood_fn` is
  short-circuited and replaced by `likelihood.log_prob(prediction,
  params)`. Standard prior penalty (½ ξᵀξ) still added in
  `build_loss_fn`. Calibration / e-line marginalisation / Student-t /
  spec_cov are NOT applied automatically — the user owns their data
  term entirely. Three Fitter paths now exist:
  - **Path 1**: `Fitter(model, data, noise)` — classic, unchanged.
  - **Path 2**: `Fitter(model, data, noise, likelihood=Custom)` —
    classic forward model + custom likelihood. Most common
    power-user case.
  - **Path 3** (deferred): `Fitter(components=, observation=,
    likelihood=, parameters=)` — fully composable. Tracked.
- **`PhotometryLikelihood`** at `tengri.inference.photometry_likelihood`
  (renamed from `BroadbandLikelihood`). Same diagonal-Gaussian χ² applies
  to broadband and narrowband photometry — bandwidth was an artificial
  restriction. Re-exported as `tengri.pipeline.PhotometryLikelihood`.
- **`tengri.inference.likelihoods.gaussian`** — extracted
  `diag_gaussian_chi2` and `diag_gaussian_log_prob` as the **single
  source of truth** for the Gaussian χ² that previously appeared in 16
  inlined call-sites in `loss_functions.py`. Both legacy and new path
  delegate here.
- 4 tests at `tests/unit/test_user_likelihood_override.py` validate
  paths 1–2: legacy unchanged, Gaussian wrapper matches built-in,
  custom L1 likelihood produces a different scalar than Gaussian.
- 16 inline `jnp.sum(((d-μ)/σ)**2)` patterns in `loss_functions.py`
  now route through `diag_gaussian_chi2`. Numerically identical;
  240 existing inference tests still pass.

### Added (Phase II-1 — `sample_params_dict` end-to-end helper)

- **`tengri.pipeline.sample_params_dict(components, key, overrides=None)`** —
  closes the user-facing loop ``components → declarations → params dict
  → run_components``. Splits the PRNG key once per declared parameter,
  draws from each prior, and threads ``overrides`` (including bare-name
  allowlist entries like ``redshift``) into the output. Silently drops
  override keys that no component owns and aren't in the allowlist.
  7 tests at `tests/unit/test_sample_params_dict.py` cover sampling
  shape, override pinning, bare-redshift threading, deterministic
  draws, end-to-end `run_components` chaining, slice-round-trip.
- Re-exported as `tengri.pipeline.sample_params_dict`.

### Added (Phase II-1 — public-API entry point + Phase II-2 stellar skeleton)

- **`tengri.pipeline`** namespace — single canonical import path for the
  Phase II-1 component pipeline. Re-exports the `SEDComponent` Protocol,
  `PipelineState`, all six adapters (`Radio`, `IGM`, `XRay`,
  `DustAttenuation`, `DustEmission`, `Nebular`), and the orchestrator
  helpers (`run_components`, `merge_declared_parameters`,
  `slice_params_for_component`). Added to `tengri.__all__` and
  `ALLOWED_TOP_LEVEL` surface guard. End-to-end smoke test at
  `tests/integration/test_pipeline_public_api.py` exercises the full
  six-adapter chain through the public namespace.
- **`StellarSEDComponent` skeleton** at
  `src/tengri/components/stellar/component.py` — Phase II-2 contract
  surface. `declared_parameters()` resolves the per-`sfh_model` /
  `metallicity_model` parameter set from `SFH_REGISTRY` / `MET_REGISTRY`;
  `apply()` raises `NotImplementedError` until the SSP-grid migration
  lands. Anchors the documented design decisions so downstream adapters
  (dust two-component, Cue/CloudyGrid nebular) can be designed against
  a stable contract.
- **Tuple `parameter_prefix` support** — orchestrator's
  `slice_params_for_component` and `merge_declared_parameters` now
  accept `tuple[str, ...]` for `parameter_prefix` (single-`str` still
  works, fully backward compatible). Required by `StellarSEDComponent`
  which owns three prefixes: `("sfh_", "met_", "chem_")`. Validates
  empty-prefix and empty-tuple as before.
- **Phase II-2 design questions resolved** in
  `docs/dev/phase_ii_2_stellar_migration.md` §"Open questions —
  resolved 2026-05-03": surviving stellar mass for `log_mstar`, eager
  `lnu_age` publishing, tuple parameter_prefix for stellar, `rtol=1e-8`
  feature-parity target.

### Added (Phase II-1 sixth adapter — closed-loop dust energy balance)

- **`tengri.components.dust.emission_component.DustEmissionSEDComponent`** —
  sixth Phase II-1 adapter and the **first cross-component closed
  loop**. Wraps :func:`modified_blackbody`; reads ``state.derived["L_ir"]``
  (published by :class:`DustAttenuationSEDComponent` via the energy-
  balance integral) and re-emits it as a modified blackbody added to
  ``state.sed_intrinsic``. Two free parameters (``dust_T``,
  ``dust_beta_ir``); ``dust_tau_v`` stays owned by the attenuator
  (no name collision because the merger validates per-name uniqueness,
  not per-prefix). Casey/Dale/DL07/DL14/Astrodust/BOSA/THEMIS adapters
  land in Phase II-3 once their precompute paths migrate.
- **`DustAttenuationSEDComponent` now publishes ``state.derived["L_ir"]``** —
  the integrated absorbed luminosity ``∫(L_ν_intrinsic − L_ν_attenuated) dν``.
  Closes the energy-balance handshake with the new IR adapter
  (and feeds Radio's existing FIR-radio correlation read of ``L_ir``,
  which previously fell back to 0).
- **`tests/integration/test_dust_emission_pipeline.py`** — 10 tests:
  4 parametrized numerical-equivalence cases against direct
  :func:`modified_blackbody`, no-op-when-no-attenuator, attenuation/
  emission handshake, ``tau_v=0`` zeroing of L_ir, two-adapter chain
  end-to-end, an ∫L_dust dν ≈ L_ir energy-conservation check (~1%),
  and a ``merge_declared_parameters`` test for the disjoint-names
  rule when both dust adapters share the ``dust_`` prefix.
- **Contract test matrix expanded to 6 adapters** — 5 contract tests ×
  6 adapters + 4 standalone = **34 contract tests passing**. Cumulative
  Phase II-1 surface: **90 tests passing**.

### Added (Phase II-1 fifth adapter — `NebularSEDComponent` BakedIn)

- **`tengri.components.nebular.component.NebularSEDComponent`** — fifth
  Phase II-1 adapter and the **first zero-parameter** adapter. Wraps
  :class:`BakedInBackend` (the case where nebular emission is folded
  into the SSP grid at fixed ``logU`` and escape fraction). Declares
  no free parameters, does not transform the SED — just publishes
  ``state.derived["nebular_backend"] = "baked_in"`` so downstream
  observation models can decide whether to add separate emission
  lines. Cue / CloudyGrid / shock variants will land in Phase II-3
  once :class:`StellarSEDComponent` publishes the ionising-photon
  production rate they need.
- **`tests/integration/test_nebular_pipeline.py`** — 6 tests covering
  zero-parameter declaration, no-op SED behaviour, marker publication,
  three-adapter chain integration, ``merge_declared_parameters``
  handling of empty contributions, and a clear ``NotImplementedError``
  for unsupported backends.
- **Contract test relaxation** — `test_declared_parameters_obey_prefix_rule`
  no longer asserts ``len(decls) > 0``; zero-parameter adapters are
  now an explicit valid contract case (the prefix-rule loop simply
  iterates over an empty list).
- **Contract test matrix expanded to 5 adapters** — 5 contract tests ×
  5 adapters + 4 standalone = **29 contract tests passing**. Cumulative
  Phase II-1 surface: **75 tests passing**.

### Added (Phase II-2 design doc + cross-component contract standardisation)

- **`docs/dev/phase_ii_2_stellar_migration.md`** — design document
  scoping the migration of stellar physics (sfh + sps + chemistry)
  onto the `SEDComponent` Protocol. Specifies what `StellarSEDComponent`
  declares (parameters per SFH/metallicity model), what it reads
  (nothing — head of pipeline), what it publishes to `state.derived`
  (`log_mstar`, `sfr`, `sfr_10myr`, `sfr_100myr`, `L_age`, `lnu_age`,
  `nion`, `sfh_grid_lbt_yr`, `sfr_history`, `log_metallicity_history`),
  the 6-PR migration sequence, 7 verification gates, 5 risks with
  mitigations, and 4 open questions for design review. Total scope:
  ~5500 lines of physics moved, no rewrites.
- **Standardised `state.derived["log_mstar"]`** as the canonical stellar-
  mass key across all adapters. `XRaySEDComponent` previously read
  `state.derived["stellar_mass"]` (linear M_⊙); now reads `log_mstar`
  (log10 M_⊙) and exponentiates internally via `M_* = 10**log_mstar`,
  matching `RadioSEDComponent` and the contract specified in the
  Phase II-2 design doc. Test fixtures updated; numerical equivalence
  preserved (still 1e-10 rtol against direct `xray_total` calls).

### Added (Phase II-1 — `merge_declared_parameters` orchestrator helper)

- **`tengri.forward.orchestrator.merge_declared_parameters(components)`** —
  flattens per-component `declared_parameters()` lists into a single
  ``{name: prior}`` dict suitable for spreading into
  :class:`tengri.Parameters` once Phase II-6 lands. Validates the prefix
  rule (every name starts with the owning component's
  ``parameter_prefix`` or is in :data:`BARE_NAME_ALLOWLIST`) and rejects
  collisions when two components claim the same parameter name. This
  closes the Parameter-side of the seam: each adapter declares its
  own parameters, the orchestrator merges them into a single prior dict,
  and the existing :class:`tengri.Parameters` factory will eventually
  consume that dict directly.
- **`tests/unit/test_merge_declared_parameters.py`** — 9 tests covering
  the happy path (single-component round-trip, four-adapter disjoint
  merge, prior pass-through, ordering preservation, bare-name-allowlist
  acceptance) and every contract violation the helper rejects (wrong
  prefix, duplicate name, non-`ParamDeclaration` entry, empty input).

### Added (Phase II-1 first-cohort adapters — fourth adapter, transforming)

- **`tengri.components.dust.component.DustAttenuationSEDComponent`** —
  fourth Phase II-1 adapter and the **first one that transforms**
  (rather than adds to) the SED. Reads ``state.sed_intrinsic`` and
  writes ``state.sed_attenuated = sed_intrinsic * exp(-tau_v * k(λ))``.
  Wraps a single attenuation law from the registry (Calzetti+2000 by
  default; ``cardelli`` / ``smc`` / ``lmc`` / ``prevot_smc`` etc.
  available via the ``law=`` config knob). Declares one free parameter
  ``dust_tau_v``. Publishes ``state.derived["dust_attenuation_factor"]``.
  - Intentionally a single-component screen — two-component
    (Charlot & Fall 2000, birth-cloud + diffuse ISM) attenuation
    will be a separate adapter once the stellar component publishes
    per-age luminosities.
- **`tests/integration/test_dust_attenuation_pipeline.py`** — 5
  parametrized numerical-equivalence tests, no-op-when-no-upstream
  test, tau_v=0 identity test, SMC-law sanity check, plus a
  **four-adapter end-to-end chain** (Radio + Dust + X-ray + IGM)
  exercising additive emitters composed with a transforming attenuator.
- **Contract test matrix expanded to 4 adapters** — 5 contract tests ×
  4 adapters + 4 standalone = **24 contract tests passing**.

### Added (Phase II-1 first-cohort adapters — third adapter)

- **`tengri.components.xray.component.XRaySEDComponent`** — third
  Phase II-1 adapter joining the existing Radio + IGM pair (already
  landed by upstream). Wraps :func:`xray_total` (XRBs + AGN corona)
  with no physics changes. Reads ``sfr``/``stellar_mass``/``L_agn_bol``
  from the published-derived fallback pattern; declares 5 free
  parameters (``xray_gamma_hmxb``, ``xray_gamma_lmxb``,
  ``xray_gamma_agn``, ``xray_E_cut``, ``xray_alpha_ox``). Publishes
  ``state.derived["L_xray"]`` for downstream readers.
- **`tests/integration/test_xray_pipeline.py`** — numerical-equivalence
  tests for the orchestrator's X-ray path (5 parameterised cases),
  fallback-when-no-AGN test, immutability test, and a three-adapter
  end-to-end chain (Radio + X-ray + IGM) that exercises the full
  composition.
- **`tests/unit/test_component_protocol.py`** — `ADAPTERS` list extended
  to 3 entries; the parameterised contract tests (Protocol shape,
  prefix rule, declared-parameters validity, immutability) now run
  for X-ray automatically. 5 contract tests × 3 adapters + 4 standalone
  = **19 contract tests passing**.

### Added (Phase II-1 scaffold — `tengri.core` protocols)

- **`tengri.core.SEDComponent`** — Protocol every physics block (stellar,
  dust, nebular, AGN, IGM, radio, X-ray) will implement in Phase II-2+.
  Specifies `name`, `parameter_prefix`, `config`, `declared_parameters()`,
  `precompute(ssp_data, wave_grid)`, and `apply(state, params)`.
- **`tengri.core.PipelineState`** — Immutable frozen dataclass threaded
  through a chain of components. Fields: `wave`, `sed_intrinsic`,
  `sed_attenuated`, `sed_observed`, `lines`, `derived`. Provides
  `state.with_(...)` for ergonomic immutable updates.
- **`tengri.core.SEDComponentConfig`** / **`tengri.core.SEDComponentState`**
  — Frozen-dataclass base classes for component-specific configuration
  and precomputed-tensor caches.
- **`tengri.core.ObservationModel`** — Protocol for the data-side of the
  forward model (`predict(state, params)` → dict of channel-keyed
  predicted observables).
- **`tengri.core.Likelihood`** — Protocol for `log_prob(prediction,
  params)` → scalar. Decouples inference from forward model.
- **`tests/unit/test_core_protocols.py`** — 6 contract tests using
  minimal duck-typed implementations. Validates `isinstance(..., Protocol)`
  checks, immutability of `PipelineState.with_(...)`, and an end-to-end
  toy chain (component → observation → likelihood).

This is a scaffold: nothing in `tengri` consumes these classes yet.
Phase II-2 onwards (deferred until after Paper I) will migrate one
physics module at a time onto this contract, slimming
`forward/sed_model.py` from 2957 L toward ~250 L.

### Changed (Phase 6 — top-level surface slim-down)

- **`tengri.__all__` shrank from 80 → 62 entries.** Implementation
  detail helpers were demoted out of the advertised top-level surface
  but remain importable for back-compat. The recommended import paths
  are now:
  - Branding (`LOGO`, `LOGO_BANNER`, `print_logo`) — internal only.
  - Citation helpers (`Bibliography`, `Citation`, `cite`, `cite_all`,
    `cites`, `collect_citations`, `paper_citation`, `citations_bibtex`,
    `citations_report`, `print_bibtex`, `print_citations`,
    `print_paper_citation`) — use `from tengri import citations`
    instead.
  - Noise kernel helpers (`exp_squared_kernel`, `gp_noise_covariance`,
    `matern32_kernel`) — use `tengri.observation.noise.*` instead.
  - Single-purpose loaders (`load_filter_set`, `load_ssp_data`) — use
    `tengri.observation.load_filter_set` and `tengri.sps.load_ssp_data`
    instead.

  The surface-guard test (`tests/unit/test_public_api_surface.py`)
  partitions names into `ALLOWED_TOP_LEVEL` (advertised) and
  `DEMOTED_BUT_IMPORTABLE` (importable but not advertised). A future
  phase will add `DeprecationWarning` shims to the demoted set.
- **`tengri.citations` exposed as a subpackage namespace** for the
  recommended import path of citation helpers.

### Deprecated (will be removed in v1.0)

- **Phase 2 — Verb-rule enforcement (NAMING_CONTRACT §4).** Registry-lookup
  functions are renamed `get_*` → `resolve_*`; pure compute functions are
  renamed `*_emission` / `*_sed` → `compute_*_sed`. Old names continue to
  work but emit `DeprecationWarning`:
  - `get_dust_law` → `resolve_dust_law`
  - `get_agn_model` → `resolve_agn_model`
  - `get_emission_model` → `resolve_emission_model`
  - `blr_emission` → `compute_blr_sed`
  - `nlr_emission` → `compute_nlr_sed`
  - `nlr_emission_richardson2014` → `compute_nlr_sed_richardson2014`
  - `shock_emission_sed` → `compute_shock_sed`
  - `qsogen_sed` → `compute_qsogen_sed`
  - `pah_template` → `compute_pah_template`
  - `radio_components` → `compute_radio_components`
- **Phase 3 — Drop redundant `_sfh` suffix inside `tengri.components.sfh`.**
  Old names are kept as deprecated aliases. Registry string keys
  (e.g. `SFH_REGISTRY["exponential_sfh"]`) are unchanged so YAML configs
  and notebooks keep working:
  - `constant_sfh` → `constant`
  - `exponential_sfh` → `exponential`
  - `delayed_exponential_sfh` → `delayed_exponential`
  - `gaussian_sfh` → `gaussian`
  - `lognormal_sfh` → `lognormal`
  - `powerlaw_sfh` → `powerlaw`
  - `skewnormal_sfh` → `skewnormal`
  - `truncated_skewnormal_sfh` → `truncated_skewnormal`
  - `snorm_burst_sfh` → `snorm_burst`
  - `snorm_trunc_burst_sfh` → `snorm_trunc_burst`
  - `spline_sfh` → `spline`
  - `dense_basis_sfh` → `dense_basis`
  - `dense_basis_pure_sfh` → `dense_basis_pure`
  - `dirichlet_sfh` → `dirichlet`
  - `continuity_sfh` → `continuity`
  - `continuity_flex_sfh` → `continuity_flex`
  - `psb_continuity_sfh` → `psb_continuity`

### Added

- **Phase 4 — AGN and dust sub-namespaces** for clearer physics grouping.
  Pure re-export modules; existing import paths still work:
  - `tengri.components.agn.disc_api` (powerlaw / multicolor / K&D / ADAF /
    qsogen disc models, plus `compute_l2500`, `beloborodov_gamma_hot`).
  - `tengri.components.agn.torus_api` (simple / two-temperature / Nenkova
    torus, SKIRTOR / CAT3D-wind / Silva04 templates).
  - `tengri.components.agn.lines` (`compute_nlr_sed`, `compute_blr_sed`).
  - `tengri.components.agn.compose` (`unified_agn`, `unified_nlr_blr`,
    `adaf_agn`, `kubota_done_full_agn`).
  - `tengri.components.dust.attenuation_models` (all attenuation laws +
    composite models + `resolve_dust_law`).
  - `tengri.components.dust.emission_models` (all IR emission models +
    grid loaders + `energy_balance_split` helpers).
  - `tengri.components.dust.pah` (Drude profile, PAH decomposition,
    Smith+2007 features).
- **Phase 5 — Free-parameter prefix CI guard.** New `tools/check_param_prefixes.py`
  walks every preset (`starforming`, `quiescent`, `high_z`, `photoz`,
  `jwst_spec`, `agn_host`) and asserts every free-parameter name matches
  the NAMING_CONTRACT §3.2 prefix regex. Audited the codebase: no
  violations found (the internal `psd_xi` / `psd_sigma` / `psd_tau_myr`
  identifiers are already aliased to compliant `sfh_field_*` names by
  the parameters translation layer in `parameters/translate.py`).
  New `tests/unit/test_param_prefix_guard.py` adds 8 parameterised
  preset-compliance tests.
- **Public API hierarchy (Phase 1)**: three new top-level namespaces grouping
  pre-existing helpers under physics-meaningful names. Pure re-exports — no
  behavioural change.
  - `tengri.cosmology` re-exports `PLANCK18`, `luminosity_distance`,
    `lookback_time`, `age_at_z`, `comoving_volume_element`, etc. from
    `tengri.utils.cosmology`.
  - `tengri.units` re-exports F_nu/L_nu conversions (`fnu_to_jy`,
    `flambda_to_fnu`, `lnu_to_fnu`, ...) and AB-magnitude helpers
    (`ab_mag_to_fnu`, `lnu_to_absolute_ab_mag`, `distance_modulus_from_dl`,
    ...) from `tengri.utils.{conversions,magnitudes}`.
  - `tengri.plot` re-exports `plot_sed_fit`, `plot_sfh`, `safe_corner`,
    `setup_style`, `COLORS`, `SPECTRAL_FEATURES` from
    `tengri.analysis.plotting`.
- `tengri._deprecated` — internal helpers (`deprecated_alias`,
  `deprecated_attribute`) used to keep old import paths working with a
  single `DeprecationWarning` while the API is reorganised. Will be reused
  by Phases 2–6.
- `tests/unit/test_public_api_surface.py` — guards `tengri.__all__` against
  accidental top-level pollution. New top-level symbols must be added to
  `ALLOWED_TOP_LEVEL` in the same commit.
- `docs/dev/api_migration_v0.x.md` — running migration table tracking every
  public-API rename/move and its scheduled drop version.

- Galaxy facade class with `from_arrays` and `from_observation` constructors for ergonomic observation handling.
- `tengri.doctor` environment health check utility; run `python -m tengri doctor` to verify dependencies and configuration.
- Citations subsystem: `Citation` dataclass, registry with 16 seed entries, `cite()` and `cite_all()` helper functions for academic attribution.
- Presets module with factory functions: `starforming()`, `quiescent()`, `high_z()` for common model configurations.
- `FitResult` and `Provenance` wrapper classes with optional HDF5 save/load for reproducible inference workflows.
- Preprocessing module with zero-point registry, systematic-error-floor helper, and upper-limit utilities for photometry.
- I/O module with readers for SDSS, DESI, and generic FITS spectra; adapter for `specutils.Spectrum1D` integration.
- `tengri` CLI with `doctor` and `cite` subcommands.
- LICENSE file (BSD-3-Clause).
- CONTRIBUTING.md with contributor guidelines.
- Docstring standard reference in `docs/dev/spdx-headers.md`.

### Changed

- Declared license updated from MIT to BSD-3-Clause in `pyproject.toml` and `CITATION.cff`.

### Fixed

- (None in this release.)

---

## Notes for Pre-1.0 Users

Tengri is pre-1.0 software. The public API, configuration format, and file layout may change without semantic versioning guarantees until a stable 1.0 release is declared. We appreciate early feedback and encourage users to report breaking changes or feature requests via GitHub Issues.
