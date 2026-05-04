# Changelog

All notable changes to tengri are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
