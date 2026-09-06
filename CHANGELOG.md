# Changelog

All notable changes to tengri are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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

### Added

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
  and the Lehmer LMXB mass term (1). Each family carries a recorded grouping;
  anything new is an error, and a registration whose seam is gone is also an
  error, so the inventory cannot rot. Runs in the `smoke` job, beside
  `check_float32_representable_constants.py`, which is where the checks that
  need tengri installed live — `lint` installs only ruff.

- `tests/regression/precision/test_float32_scale_seam_sweep.py` — sweeps each
  enumerated seam family across its parameter's whole declared prior in float32
  and requires the gradient to be finite **and** non-zero at every point (`nan
  != 0.0` is `True`, and zero is finite, so neither half is coverage alone). The
  inventory is read from the tool rather than written down twice: the module
  fails if the enumeration grows a family it does not sweep, which is what makes
  the recorded reason a measurement instead of the only evidence.

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

### Removed

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
- Public re-exports of `simple_torus` and `two_temperature_torus` from
  `tengri.components.agn`. The functions remain importable from
  `tengri.components.agn.torus` for the production models that still
  call them internally (`multicolor_agn`, `kubota_done_full`, `adaf`,
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
