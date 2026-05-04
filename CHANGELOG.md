# Changelog

All notable changes to tengri are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Phase II-2.6 — Quantities bridges from PipelineState)

- ``tengri.forward.state_to_sfh_quantities(state)`` — converts an
  orchestrator ``PipelineState`` into the legacy ``SFHQuantities``
  NamedTuple (all 7 fields populated; mass-weighted age and metallicity
  computed from the published SFH grid).
- ``tengri.forward.state_to_sed_quantities(state)`` — converts to
  ``SEDQuantities``; populates ``l_bol``, ``l_tir``,
  ``l_dust_absorbed`` directly from the state. UV-slope, Dn4000,
  Balmer-break, M_UV, and luminosity-weighted quantities return ``NaN``
  pending the next bridge step.

These are the first half of full ``Prediction`` parity (which gates
monolith deletion). Existing code reading
``predict_sfh_quantities(...).stellar_mass`` keeps working when the
prediction is sourced from ``run_components``.

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

- New benchmark file ``analysis/orchestrator_jit_benchmark.json``
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
  ~64 ms per-fusion baseline at `analysis/jit_compile_benchmark.json`)
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
