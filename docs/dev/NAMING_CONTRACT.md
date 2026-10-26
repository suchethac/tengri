# Tengri Naming Contract

> **Authority:** This is the single source of truth for naming conventions.
> All new code, renames, and refactors MUST comply. Referenced by CLAUDE.md.
>
> **Last updated:** 2026-09-11
> **Current version:** v0.1-dev
> **Derived from:** `docs/dev/sessions/2026-04-03-api-naming-design.md`, `docs/dev/REFACTOR.md`

---

## 0. Versioning & Deprecation Policy

- **Current release:** pre-v1.0 (development).
- **Deprecation horizon:** Deprecated aliases emit `DeprecationWarning` throughout v0.x. All deprecated aliases will be **removed in v1.0**.
- **Changelog:** See `CHANGELOG.md` at repository root.

---

## 1. Class Suffix Convention

| Suffix | Role | Examples |
|--------|------|----------|
| (none) | Data container / physics object | `Photometry`, `Spectroscopy`, `Observation` |
| `Model` | Physical forward model or statistical model | `SEDModel`, `NoiseModel` |
| `Config` | Frozen structural choice (not fitted) | `DustConfig`, `NebularConfig`, `AGNConfig`, `VIConfig` |
| `Backend` | Swappable computation engine | `CloudyGridBackend`, `CueBackend` |
| `Fitter` / `-er` | Active orchestrator | `Fitter`, `PopulationFitter` |
| `Posterior` | Inference result (samples + diagnostics) | `Posterior`, `PopulationPosterior` |
| `List` | Registry / catalog | `LineList` |
| `Error` | Exception classes (see §8) | `TengriError`, `ParameterError` |

**Forbidden suffixes:**
- `Spec` — too vague; use `Parameters` or a domain-specific name.
- `Result` — use `Posterior`.
- `Config` on **data containers** — `Config` is reserved for frozen structural choices only. Data containers (e.g., `Photometry`, `Spectroscopy`) carry no suffix.

---

## 2. Canonical Class Names

These are the **canonical** names. The old names were removed; the alias tables below record what each one became.

| Canonical name | Removed alias(es) | Defining module |
|---------------|---------------------|-----------------|
| `SEDModel` | `Model` (removed) | `core/model.py` |
| `Parameters` | `ParamSpec` (removed) | `core/parameters.py` |
| `Spectroscopy` | `SpectroscopyConfig` (removed) | `models/observation/spectroscopy.py` |
| `NoiseModel` | `NoiseConfig` (removed) | `models/observation/noise_model.py` |
| `LineList` | `LineCatalog` (removed) | `models/observation/line_list.py` |
| `PopulationFitter` | `HierarchicalFitter` (removed) | `inference/hierarchical.py` |
| `PopulationPosterior` | `HierarchicalResult` (removed) | `inference/hierarchical.py` |
| `SEDModelConfig` | `ModelConfig` (removed) | `config/settings.py` |

**File path stability rule:** Module filenames must match the canonical class name in `snake_case`. If a class is renamed, the file is renamed in the same commit. The table above is the ground truth; if it conflicts with the filesystem, the filesystem is wrong.

**Rule:** New code must use canonical names. Old names must never appear in new code, tests, docs, or examples.

---

## 3. Parameter Namespace

### Two layers only

| Layer | Purpose | Example |
|-------|---------|---------|
| **User-facing (prefixed)** | What users write in `Parameters` / `from_config()` | `sfh_dpl_alpha`, `dust_tau_bc`, `met_logzsol` |
| **Internal** | What physics functions receive after translation | `alpha`, `tau_bc`, `log_z_abs` |

### Prefix table

| Domain | Prefix | Examples |
|--------|--------|---------|
| Star formation | `sfh_{type}_` | `sfh_dpl_alpha`, `sfh_tsnorm_log_total_mass` |
| GP field | `sfh_field_` | `sfh_field_psd_sigma`, `sfh_field_psd_tau_myr`, `sfh_field_xi` |
| Metallicity | `met_` | `met_logzsol`, `met_alpha_fe` |
| Chemistry | `chem_` | *(reserved; populate when chemical evolution module is added)* |
| Dust | `dust_` | `dust_tau_bc`, `dust_tau_diff`, `dust_slope` |
| Nebular | `neb_` | `neb_logU`, `neb_logZ_gas`, `neb_fesc` |
| AGN | `agn_` | `agn_log_lbol`, `agn_tau_skirtor`, `agn_log_mbh` |
| Emission lines | `eline_` | `eline_sigma_kms`, `eline_broad` |
| Noise | `noise_` | `noise_f_cal`, `noise_dof` |
| Radio | `radio_` | `radio_q_ir` |
| X-ray | `xray_` | `xray_gamma_agn` |
| Shock | `shock_` | `shock_frac` |
| Redshift | `redshift` | `redshift` (standalone; no trailing underscore) |

**CI-enforceable regex:** `^(sfh_|met_|dust_|neb_|agn_|eline_|noise_|radio_|xray_|shock_|chem_|redshift)`

This regex validates that every user-facing parameter name begins with a recognized domain prefix. It does **not** enforce deeper structure beyond the prefix; per-domain structural validation is the responsibility of `Parameters.validate()`.

### What is forbidden

- **Short names in public API** (e.g., bare `psd_sigma` instead of `sfh_field_psd_sigma`). Short names exist ONLY inside `resolve_short_names()` for `from_config()` convenience. One exception: `logzsol` → `met_logzsol`.
- **`_u` suffix params** (e.g., `psd_sigma_u`) leaking outside optimizer internals in `hierarchical.py`.
- **Internal names shadowing public names** (e.g., `log_total_mass` meaning different things at different layers).

### Unit contract

All unit translations are implemented in `core/parameters.py :: translate_to_internal()`. The internal variable names listed below are the canonical internal names used downstream of that function.

| User-facing param | User unit | Internal variable | Internal unit | Conversion |
|-------------------|-----------|-------------------|---------------|------------|
| `sfh_field_psd_tau_myr` | Myr | `psd_tau_yr` | yr | ×10⁶ |
| `sfh_*_tau_peak_gyr` | Gyr | `tau_sfh` | yr | ×10⁹ |
| `met_logzsol` | log(Z/Z☉) | `log_z_abs` | log(Z) absolute | + `LOG10_ZSUN` |
| `dust_tau_bc` | optical depth | `tau_bc` | optical depth | (identity) |
| Wavelength (all) | Å | — | Å | (identity) |
| SFR (all) | M☉/yr | — | M☉/yr | (identity) |

---

## 4. Function Verb Convention

| Verb | Meaning | Examples |
|------|---------|---------|
| `compute_*` | Pure function, returns new array | `compute_blr_sed()`, `compute_csp_sed()` |
| `load_*` | Reads from disk | `load_ssp_data()`, `load_cloudy_grid()` |
| `build_*` | Factory returning a callable/kernel | `build_kernel()`, `build_dl07_photometry_lookup()` |
| `apply_*` | Transform a spectrum in-pipeline | `apply_lsf()`, `apply_lyman_cutoff()` |
| `resolve_*` | Registry lookup by string key | `resolve_dust_law()`, `resolve_sfh()` |
| bare name | SFH shape functions registered as **registry keys** in `SFH_REGISTRY` ONLY (see §6) | `tsnorm`, `dpl` |

### Renames applied

| Former (removed) | Now | File |
|---------|-----------|------|
| `blr_emission()` | `compute_blr_sed()` | `models/agn/blr.py` |
| `nlr_emission()` | `compute_nlr_sed()` | `models/agn/nlr.py` |
| `shock_emission_sed()` | `compute_shock_sed()` | `models/nebular/shock.py` |
| `qsogen_sed()` | `compute_qsogen_sed()` | `models/agn/qsogen.py` |
| `get_dust_law()` | `resolve_dust_law()` | `models/dust/attenuation.py` |
| `get_agn_model()` | `resolve_agn_model()` | `models/agn/unified.py` |
| `get_emission_model()` | `resolve_emission_model()` | `models/dust/emission.py` |

---

## 4b. Public Method Verbs & API Contracts

### 4b.1 Method verbs on public surfaces

The public surface uses exactly these verbs. Anything else is a contract violation:

| Verb | Meaning | Module | Examples |
|------|---------|--------|----------|
| `predict_*` | Forward model → observable or derived property | core | `predict_photometry()`, `predict_spectrum()` |
| `measure_*` | Extraction requiring no `SEDModel` — array in, number out | `measure` | `measure_index_jax()`, `measure_line_flux_jax()` |
| `mock*` | Synthesize data | `results` | `generate_mock()` |
| `fit*` | Inference entry point | `inference` | `Fitter.run()`, `PopulationFitter.run()` |
| `plot_*` | Visualization | `plot` | `plot_sed_fit()`, `plot_sfh()` |

**Rationale:** These verbs signal intent immediately. A user reading `model.predict_*` knows the call is deterministic and returns observables; `measure_*` signals extraction that needs no model; `fit*` signals inference.

**"Model-free" means no `SEDModel` — it does NOT mean "observed data."** This
distinction has already cost one wrong turn, so it is written down here. `measure`
is defined by what it does *not* require — a model and a forward pass — never by
where its input came from. Every operator is array-in / number-out, and the same
one measures a spectrum tengri just predicted, one exported to disk, and one the
user reduced themselves; `measure.from_prediction()` measures a `Prediction`
directly. Reading "model-free" as "consumes observed data" is what motivated a
proposed `from_observed` entry point during API Phase 3 — a split by input
provenance that nothing needed and that was correctly never built. Do not
reintroduce it. ("Model-free" also does not mean *reduction* tools: continuum
placement, sky subtraction and bad-pixel repair are a pipeline's job. See the
`tengri.measure` module docstring, which is the long form of this paragraph.)

**The verb can be carried by the module rather than the prefix.** The public
façade spells them `measure.spectral_index()`, `measure.line_flux()` and
`measure.photometry()` — no `measure_` prefix, because `tengri.measure` already
supplies it. The `measure_*`-prefixed engines they dispatch to live in
`observation/`. Note also that measuring an equivalent width is a *mode* of
`measure.spectral_index()` (break / EW / slope), not a function of its own; the
array-returning diagnostic is `compute_equivalent_widths()` under the `compute_*`
verb.

### 4b.2 The property-catalog contract

Every derived physical quantity (stellar mass, SFR, ionization parameter, etc.) is declared by exactly ONE component via a `Property(units, group, doc, fn)` declaration in the protocol. Violations are architecture-level bugs.

**Registry integrity:**
- Flat snake_case keys, Bagpipes-compatible where the concepts overlap (e.g., `stellar_mass`, `sfr_100myr`)
- Name collisions are a **BUILD-TIME error** caught at `SEDModel.build()`
- Accessing an absent property raises `KeyError` listing the available names (never silently returns None/NaN/0 — this is the `silent-failure` class)
- Metadata (units, group, doc, provenance, component name) is queryable via `tengri.list_properties()`, `describe_property()`

**Topology contract — same keys everywhere:**
- `Prediction` topology → scalar or band array
- `Posterior` topology → `(n_samples,)` or `(n_samples, n_bands)`
- `PopulationPosterior` topology → `(n_galaxies, n_samples,)` or `(n_galaxies, n_samples, n_bands)`

Only the **axes change**; the **key name is identical** across topologies.

### 4b.3 SED vs spectrum (frame & units)

Naming rule for panchromatic quantities:

- **SED** = panchromatic, model-grid array, **frame prefix required**: `rest_sed`, `obs_sed`. The frame prefix means "raw array"; every docstring must state frame and units.
- **spectrum** = instrument-grid, LSF-convolved, calibrated observable. No frame prefix (the instrument frame is implied). Called via `pred.spectrum()`.

**Every observable on `Prediction` is a uniform callable with a default.** No `_at` / `_for` / `_on` coinages, no bare properties that a user then has to resample by hand:

```python
pred.rest_sed()             # L_nu, rest-frame axis            [erg/s/Hz]
pred.rest_sed(wave)         # resampled onto YOUR rest-frame grid   [Angstrom]
pred.obs_sed()              # L_nu, observed-frame axis + IGM  [erg/s/Hz]  <- still L_nu!
pred.obs_sed(wave_obs)      # resampled onto YOUR observed-frame grid
pred.photometry(filters=None, approx=False) # F_nu  [erg/s/cm2/Hz]
pred.magnitudes(filters=None, approx=False) # AB mag
pred.spectrum(wave_obs=None, approx=False)  # F_nu  [erg/s/cm2/Hz]
```

### 4b.3b Units: the distance is applied at PROJECTION, not on the SED

This is the single easiest thing to get wrong, and the docstring got it wrong for a long time.

| surface | quantity | units |
|---|---|---|
| `pred.rest_sed()` | L_ν, rest-frame axis | erg/s/Hz |
| `pred.obs_sed()` | **L_ν** — observed-frame *axis* + IGM | **erg/s/Hz** |
| `pred.photometry()` / `magnitudes()` / `spectrum()` | F_ν | erg/s/cm²/Hz (AB mag for `magnitudes`) |

**`obs_sed` is NOT a flux.** "Observed" names the *frame*, not a flux conversion. It does **not** apply `(1+z)/(4π d_L²)`; that factor lives in the projection layer (`observation/redshift_kernel.py`). Measured, not assumed: at z = 3, `obs_sed` differs from `rest_sed` at exactly the 172 grid points below rest-frame Lyman-α — IGM absorption, and nothing else.

Integrating `obs_sed()` as if it were a flux is wrong by ~57 orders of magnitude. If you want a flux, use `photometry()` or `spectrum()`.

**The wavelength argument is in the accessor's own frame.** `rest_sed(wave)` takes rest-frame Angstrom; `obs_sed(wave_obs)` takes **observed**-frame Angstrom. (The deprecated `model.predict_obs_sed(params, wave=...)` took a *rest*-frame grid and redshifted it — an observed-frame result with a rest-frame argument. That asymmetry was a footgun and is deliberately **not** reproduced.)

**The SED arrays do not carry their axis.** The axis is a separate property:

| array | its axis |
|---|---|
| `pred.rest_sed()` | `pred.wave_rest` |
| `pred.obs_sed()`  | `pred.wave_obs` |

Never reconstruct the observed axis by hand as `wave * (1 + params["redshift"])` — see §4b.6.

**Resampling is `jnp.interp` onto the requested grid**, bit-exact with the wavelength argument of the deprecated `predict_rest_sed`. Migrating a call site changes no number.

### 4b.3a Misuse must fail loudly, never silently

`pred.rest_sed` **without the parentheses** is a method object, not an array. Left alone, `np.asarray(bound_method)` yields a `dtype=object` array that plots and arithmetic will happily turn into garbage. It therefore raises `TypeError` with the fix spelled out (`_SEDCallable` in `forward/prediction.py`).

This is the general rule, not a one-off: **a public accessor that can be misused must raise, not fail open.** This package has shipped enough silent NaN-and-carry-on bugs (the `silent-failure` label) to have earned the paranoia.

### 4b.4 Exact vs approximate

Analysis and post-fit prediction is **EXACT by default** using the full wave grid.

The approximate path is a build-time optimization (`approx=WavePrecomp(...)`, `approx=SpectrumPrecomp(...)`, `approx=FeaturePrecomp(...)`) that LUTs the observable and routes the lean `predict_*` surfaces through precomputed weights. It is **opted into explicitly**. A speed knob must **never silently change the physics**.

**One mechanism, one word, at both ends.** The keyword is `approx=` at build time *and* at call time:

```python
model = SEDModel.build(..., approx=WavePrecomp(...))   # configure it
pred.photometry(approx=True)                            # use it
```

It was spelled `fast=` at call time until 2026-08. That was the same mechanism under a second name, and it named the *benefit* while hiding the *cost* — which is what let `pred.photometry(fast=True)`'s own docstring ship a `rtol=1e-12` agreement claim that measurement puts at `8.5e-4` (z = 0.05) to `1.4e-2` (sdss_g, z = 3). `fast=` still works, emits `DeprecationWarning`, and is removed in v1.0. Passing both with contradictory values raises `TypeError` (§4b.3a).

**Contract:**
- Default `model.predict_photometry(params)` uses exact wave-grid integration
- Approximate path `model = SEDModel.build(..., approx=WavePrecomp(...))` precomputes filters and returns results via LUT
- Accuracy difference is bounded and **measured, not asserted**: see the `approx` class docstring and `bench/reports/2026-08-17_wave_precomp_accuracy.md`
- User can never accidentally get the approximate answer from a `Prediction` / `Posterior` accessor; it must be asked for by name at both build and call

### 4b.5 The two prediction surfaces

There are exactly two, and nothing else is public:

| surface | what it is | when |
|---|---|---|
| `model.predict(params)` → `Prediction` | rich, cached; ONE forward pass, everything hangs off it | exploration, plotting, post-fit |
| `model.predict_photometry / predict_spectrum / predict_line_fluxes / predict_properties(params)` | lean shortcuts, JIT/vmap-safe | the inference hot path (what the likelihood calls) |

**`model.predict()` takes `params` and nothing else.** It has no `wave=` argument. (Resampling belongs to the accessor: `pred.rest_sed(wave)`.) Three separate agents have "migrated" a call site to `model.predict(p, wave=...)`; it raises `TypeError`, and `py_compile` does not catch it.

`predict_properties(params, names=...)` is the **single** JIT/vmap surface for derived quantities.

**`predict_state` is not one of them** (#1736). It is the internal bridge to the orchestrator — called by `predict_observables_jit` and `ForwardModel.predict_observables` — and its return type and keys may change without a deprecation cycle. Its docstring called itself "the public bridge" until #1736; two reviewers auditing the published notebooks read the two sources and reached opposite conclusions about whether to teach it to beginners, and the quickstart nearly shipped a paragraph doing so. Saying nothing here is what allowed that, so this says it.

**Per-component SEDs have a public surface — use it:**

```python
comp = model.predict(params).sed.components
comp["sed_intrinsic"]   # stellar, pre-dust [erg/s/Hz]
comp["sed_dust_ir"]     # and sed_nebular, sed_agn, sed_attenuated, sed_total, ...
```

`pred.sed.components` reuses the prediction's cached forward state (no second forward pass) and returns a plain dict of named arrays. This matters because the pull toward `predict_state` is real: the component-decomposition figures are among the best pages in the docs, and before this was written down there was no obvious compliant way to draw them. There is: the line above. Note also that `predict_state` returns a `ForwardState`, whose `derived` dict §4b.6 documents as internal — so the shortcut leads straight into an object this contract tells readers not to touch.

### 4b.5b Params dicts carry free parameters only

A `params` dict carries free parameters only. A key the spec declared `Fixed` is refused (#2296) at every entry point, preventing silent physics errors. Pinned values are read from `spec.get_fixed_values()` (for display or diagnostics) or `posterior.fixed_values` (after a fit). Rebuilding the model with the parameter `FREE` or with a different `Fixed` value is the remedy.

### 4b.6 Two names that are not what they look like

These have each caused real, shipped bugs. Read them before touching prediction code.

**1. `params.get("redshift", 0.0)` is forbidden.** A `Fixed` redshift is legitimately **absent** from the user's `params` dict (the spec holds it). Reading it back with a `0.0` default puts the galaxy at 10 pc and inflates the flux by ~1e17 — silently, with no NaN and no exception. This exact mistake shipped three times (#1097, #1124, #1127). **Always resolve through the spec**: `model._get_redshift(params)`, which lets an explicit value win, falls back to the fixed value, and *raises* if the model has neither.

Do **not** reach for `_get_dl_cm` as "the obvious helper": it short-circuits to a precomputed distance and silently discards an explicit override.

**2. `state.derived` is NOT `Posterior.derived`.**

| expression | what it is | status |
|---|---|---|
| `state.derived["sed_agn"]`, `state.derived["L_absorbed"]`, … | `ForwardState.derived` — the internal cross-component pipeline dict (ADR-0009) | **not deprecated.** Leave it alone. |
| `posterior.derived["stellar_mass"]` | the old `Posterior` accessor | **deprecated** → `posterior.properties[...]` |

An audit that greps for `.derived` and "migrates" every hit will rewrite the physics accessors in the reproduction notebooks. Grep for `posterior.derived` / `\.derived\[` on a `Posterior`, never bare `.derived`.

---

## 4c. Unit standards for published properties and free parameters

Delivered by #1206 (§A–D): the breaking unit change that moved eleven line-luminosity
properties, three X-ray luminosities, and the GRAHSP `l5100` free parameter off erg/s and
retired the linear `q_h`, because a value of ~1e40–1e56 in those units is `inf` in float32
(max 3.4028e38) *as a bare number*, before any physics runs — no amount of intermediate
rearrangement (`apply_log10_scale`, peak-factoring, …) can make an intrinsically
out-of-range final value representable. The rule these fixes converged on:

- **Luminosities are published in `Lsun`** (solar luminosities, ~3.828e33 erg/s), not erg/s.
  A luminosity that is representable in `Lsun` for every physical source in tengri's domain
  (stellar, nebular, AGN) but not in erg/s is the common case, not an edge case — see
  `l_bol`, `l_tir`, `l_dust_absorbed`, and (since #1206) `halpha` and its ten siblings,
  `l_x_xrb`, `l_x_agn`, `l_x_total`.
- **Rates and other quantities with no float32-safe linear form are published as `log_*`,
  in dex, with no linear sibling.** `q_h` (ionizing photon rate, ~1e53–1e56 photons/s) is
  the model: every physical value overflows float32 in *any* unit choice on the linear
  side, so `log_q_h` is the only form and the linear property is retired with no alias
  rather than converted. A property in this category never gets a same-named `Lsun` (or
  other-unit) companion "for completeness" — that would reintroduce exactly the
  unrepresentable value the `log_*` form exists to avoid.
- **`erg/s/Hz` is reserved for spectral density** (`L_nu`, the SED itself) — never for an
  integrated luminosity. The Jacobian to the frequency domain (`L_nu ~ L_lambda * lambda^2
  / c`) is what keeps SED arrays representable even for luminous AGN whose `L_lambda` is
  not; publishing an integrated quantity in `erg/s`/`erg/s/Angstrom` instead reintroduces
  the same overflow the Jacobian was avoiding.
- **A property carried in `Lsun` (or any non-erg/s unit) keeps a `log_<name>` companion in
  dex re erg/s when the erg/s form is still physically meaningful** (it usually is, for
  luminosities): `log_halpha == log10(halpha * L_sun)`, not `log10(halpha)`. The companion
  exists for callers that want the exponent directly (float32-safe by construction) without
  reconstructing it from the `Lsun` value and a conversion constant.
- **A free parameter with the same overflow shape follows the same rule**: `agn_grahsp_l5100`
  (`LogUniform(1e42, 1e47)`, erg/s) became `agn_grahsp_log_l5100` (`Uniform(42.0, 47.0)`,
  dex) rather than a `Lsun`-valued `LogUniform`, because the *parameter value itself* — not
  merely a downstream derived quantity — is `inf` in float32 before it reaches any physics
  kernel; a linear prior over a range spanning float32's ceiling is unfittable in float32 by
  construction, regardless of the unit it is denominated in.

- **An AGN component publishes an integrated luminosity as `log_L_agn_<piece>` in dex re erg/s
  AND as `L_agn_<piece>_lsun` in solar luminosities** (#2842): `log_L_agn_disc` and
  `L_agn_disc_lsun`, likewise `torus`, `polar_dust`, `lines`, `disc_unemitted`, `torus_unemitted`,
  `lines_unemitted`, `absorbed`, `bol`, `los`, `direct_los`, `adaf_accretion`
  and the `_los` siblings of the anisotropic emitters. This is the first bullet of this section
  applied, not an exception to the second: an AGN power of 1e8 to 1e14 Lsun is representable in
  float32, so the `Lsun` value exists and is published; the second bullet (no same-named
  companion) is for quantities such as `q_h` whose every physical value overflows in any unit.
  The sibling carries the unit in its name (`_lsun`) because the bare `L_agn_<piece>` was the
  retired erg/s spelling: reusing it with a different unit would silently change what a stale
  caller reads. A component's value is formed in log space (a peak-factored integral plus the
  peak's `log10`, or a reference evaluation plus a log offset through `utils/scale.py`), never
  as an erg/s number that is then logged, and the `Lsun` sibling is formed from the log as
  `pow10(log - log10(L_sun))`, so the erg/s value is never written (an absent component, `-inf`
  dex, is exactly 0 Lsun). The one erg/s key kept is `L_agn_bol`, a float64 convenience beside
  `log_L_agn_bol` that is `inf` in float32; the radio component reads `log_L_agn_bol` (and
  `log_L_ir`) in float32 and uses the linear key only in float64, and the X-ray component reads
  the log. `erg/s/Hz` keys (`L_2500_30deg`,
  `L_6um`, `L_12um`) are spectral densities and stay.
- **An anisotropic emitter publishes the angle-integrated power under the plain name and the
  line-of-sight value under `_los`** (#2678, #2851): `log_L_agn_disc` is
  `(1/2) int L(theta) sin(theta) dtheta = int_0^1 L(mu) dmu`, `mu = cos i`, and does not depend
  on the inclination; `log_L_agn_disc_los` is the power of the SED emitted along `cos_inc`.
  Kubota & Done 2018 and SKIRTOR (disc, torus, polar dust, unemitted disc) follow it. For
  SKIRTOR the luminosity parameter is the accretion power: the sphere mean of everything the
  radiative transfer carries (the escaping disc over the template's whole native range of
  10 A to 1e8 A, the torus dust and the polar dust) is `10**agn_log_lbol L_sun` times
  `agn_torus_frac`, and the library's own flux ratio between inclinations is kept: the stored
  spectra are divided by their record's dust integral (`spectra/norm`), which is multiplied
  back, and ONE scale factor per parameter point normalizes the sphere mean. The torus dust
  and the line-of-sight disc power then follow the radiative transfer. The polar dust power is
  `int_0^1 w(mu) A(mu) dmu`, with `A(mu)` the power the polar dust absorbs from the disc light
  emitted along `mu` and `w` the Type-1 weight of the SED (the logistic in `cos i` of width
  0.025 about `sin(oa)`), whatever the sightline; it is re-emitted isotropically, so
  `log_L_agn_polar_dust` and `log_L_agn_polar_dust_los` are equal and do not depend on
  `cos_inc`. The escaping disc is published net of that power. The analytic disc the
  component emits is zero below 80 A (the CIGALE `skirtor2016` piecewise disc starts at 8 nm),
  so the template disc outside 80 A to 1e7 A is not in the SED; it is published as
  `log_L_agn_disc_unemitted`, and the four sphere powers
  `disc + torus + polar_dust + disc_unemitted` add up to the luminosity parameter to 1e-12.
  Every published power of an emitted piece equals the integral of that piece of the SED
  (#2745); the unemitted disc is the one published power that is not in the SED.

#### Standalone discs

The standalone power-law disc (`PowerLawDisc`) follows the same ledger, closing on
`agn_lum_ratio * 10**agn_log_lbol L_sun` (the disc's share of the accretion power, as SKIRTOR
closes on `agn_torus_frac`). It is isotropic, so `log_L_agn_disc` (with `L_agn_disc_lsun`) is
its sphere mean and there is no `_los` key. Its SED is zero outside 10 A to 1e8 A, the band it
is defined on (`nu L_nu` does not converge at long wavelength), so nothing of the model lies
outside the emitted SED and there is no `_unemitted` piece. `log_L_agn_disc` is the
power-law-between-nodes integral on the normalization's own nodes, the quadrature the
normalization sets to the budget, so it equals the budget to rounding (C2, 4e-14), and the
dense-grid integral of the emitted SED matches it to 3.4e-7 (C3; the floor is the
normalization's interpolant against the smooth cut-off). `KD18Disc` is ruled separately
(Mdot inferred from the disc's own dissipation integral). Checked over the declared prior box
by `tests/contract/test_agn_disc_ledger_closes.py`.

### Template tori (#2851)

A torus-only template component has no disc of its own. It publishes three powers, each a
`log_L_agn_*` [dex re erg/s] with its `L_agn_*_lsun` [Lsun] sibling:

- `log_L_agn_torus`: the angle-integrated torus power, `int_0^1 P(mu) dmu`, which is
  `agn_torus_frac * 10**agn_log_lbol L_sun`;
- `log_L_agn_torus_los`: the frequency integral of the torus SED emitted along the sightline;
- `log_L_agn_disc_unemitted`: the disc share `(1 - agn_torus_frac) 10**agn_log_lbol L_sun`
  that the component does not emit, so `torus + disc_unemitted` is the accretion power to
  1e-12.

Whether the sightline value differs from the angle-integrated one depends on the library.
A library whose records share one radiative-transfer scale across inclination is emitted as
two factors: the records normalized to unit power and interpolated as for a per-sightline
library (so the spectral shape at every point is that one; measured over 180 random off-node
Fritz points it equals it to 1.5e-14), times a scalar, the records' own powers over their
model's mean, interpolated with the same node-exact PCHIP and divided by its sphere mean
(Gauss-Legendre on the panels between the inclination nodes, in the variable the library is
interpolated in). The angle-integrated power is `agn_torus_frac L_bol` and every sightline
keeps the library's flux ratio: Fritz et al. 2006 (the scalar is the `norm` record),
`SKIRTOR_mean_1p` and the CLUMPY reductions `NK0_mean_1p`, `2p` and `3p`. A library
whose scale across inclination is not established stays normalized per sightline, so its
sightline and angle-integrated powers are equal: `SKIRTOR_mean_2p` and `3p` (their inclination
ratios do not follow the SKIRTOR v3 records averaged the same way), `CAT3D_mean_3p` (no source
states the scale of its records across inclination) and Silva et al. 2004 (no inclination
axis). The standalone `cat3d_wind`, `skirtor_agnfitter` and `silva04` components publish the
three keys. The restored libraries are also reached through the composable torus blocks, whose
runner publishes them in its own ledger (below).

### Composable runner (#3076)

The composable runner (`components/agn/blocks/runner.py`, ledger in `blocks/_ledger.py`)
publishes the same ledger for the pieces of the SED it emits. For each piece `<p>` in `disc`
(with its corona), `torus`, `lines` (NLR + BLR + FeII, the `sed_agn_lines` piece) and
`polar_dust`, each a `log_L_agn_*` [dex re erg/s] with its `L_agn_*_lsun` [Lsun] sibling:

- `log_L_agn_<p>`: the angle-integrated power, `int_0^1 P_p(mu) dmu`, with `P_p(mu)` the
  frequency integral of the piece as the runner emits it along `mu = cos i` (C1). A torus
  block whose library keeps its flux ratio between sightlines registers its `log10(a/a_bar)`
  (`register_torus_anisotropy`: `fritz`, `nenkova_agnfitter`, `_2p`, `_3p`,
  `skirtor_agnfitter_1p`), and its angle-integrated power is the sightline power minus that,
  exact to the library interpolant;
- `log_L_agn_<p>_los`: the frequency integral of the piece as emitted along `agn_cos_inc`, the
  integral of that piece of the SED (C3);
- `log_L_agn_disc_unemitted`, `log_L_agn_lines_unemitted`, `log_L_agn_torus_unemitted`: the
  angle-integrated power the piece carries before any obscuration but does not emit. For the
  disc, the light the torus screen hides, the polar dust absorbs, an attenuation block removes
  and `agn_ebv_disc` reddens away; the torus does not yet re-emit it. The disc and lines
  values are the sphere mean of the node-by-node difference between the spectrum before
  obscuration and the emitted one, so emitted plus unemitted is the intrinsic power on the same
  quadrature and the difference does not cancel in float32 (`-inf` when nothing is hidden).
  The intrinsic disc is the disc block's own spectrum (un-reddened, before every screen) times
  the runner's debit; under an active `cigale_joint` tie, the face-on library power
  `agn_power x R_faceon` times the library disc's hemisphere mean, plus the tied corona. A
  recipe without a disc block carries its debited share of the accretion power here, as a
  torus-only template does. The torus before an attenuation block (other than `polar_dust`,
  which does not screen it) is the torus block's own emission;
- `log_L_agn_power`: CIGALE's `agn_power`, the torus block's power along the sightline on the
  runner's fixed budget grid, before the polar split. It is what the torus and the polar dust
  are normalized on (`f L_bol` under a debit; `L_absorbed fracAGN / (1 - fracAGN)` under an
  active tie, where it is the ledger's anchor), and `-inf` without a torus.

`log_L_agn_los` is the emitted line-of-sight sum, the integral of `sed_agn` along `agn_cos_inc`.
Every power is scaled by `agn_lum_ratio` with the SED.

**Computed on request.** The ledger re-evaluates the runner at its sightlines on its own grid,
so it is not part of the forward pass: `model.predict`, `predict_state` and
`predict_photometry` cost what they cost without it (measured: the same steady `predict` time
and bit-identical photometry FLOPs). It is a property surface: `pred.properties["log_L_agn_disc"]`,
`pred.log_L_agn_disc`, or `model.predict_properties(params, names=(...))` (the jit/vmap
surface). The keys are in `model.available_properties` for a model with the composable AGN
and absent from `state.derived`. A request evaluates the whole ledger once and serves every
key from it (`Property.bundle`); the AGN publishes `agn_torus_frac_effective` so the request
re-evaluates the chain at the torus fraction the forward pass used. At the runner level,
`composable_agn_l_nu(..., return_components=True, return_ledger=True)` adds the ledger to the
components.

**Quadrature.** The frequency integrals are trapezoids in `ln nu` on a fixed grid
(`_ledger.ledger_wave`): `_publication`'s range at 2460 log nodes per decade, 900 A to 3e5 A at
1.2e-4 in ln lambda (which resolves the lines), and a node pair at every spectral edge of the
selected blocks (`register_spectral_edges`, `blocks/spectral_edges.py`): the band cuts
written in a block (the multicolor EUV tail at 30 A, the power-law and CIGALE disc bands, the
X-ray floors), the first and last node of a template the block interpolates (zero outside it),
and the Morrison & McCammon (1983) absorption edges below 10 A of the SKIRTOR and Fritz disc
screens. With a node on each side of every discontinuity each integral converges at second
order (measured: the reference's change quarters with each doubling of its density). The
sphere means are Gauss-Legendre of 4 points per panel in `cos i`
(`_torus_inclination.inclination_quadrature_in`) on 8 uniform panels, the torus library's
inclination nodes (where its interpolated sightline factors break) and panels laid about each
Type-1/2 sigmoid, at 0, 1, 2, 4 and 8 widths on either side of its center.

**Accuracy** (`tests/contract/test_agn_composable_ledger_closes.py`, one tolerance for every
block): every `_los` key and `log_L_agn_los` equal the integral of their piece of the SED on an
independent 8x denser grid to 1e-6 (largest measured 2.4e-7, the multicolor disc behind
SKIRTOR); the disc's sphere mean and unemitted share match 1025 uniform sightlines
(Richardson) to 1e-6 behind every library torus (largest 3.6e-7; 1e-4 with the uniform
sightline rule this replaced). `agn_ledger_closed` (dimensionless, 0.0 or 1.0, so it traces) is
1.0 when the normalization debits every piece from the accretion power, so that the
angle-integrated pieces plus the unemitted ones are `agn_lum_ratio * 10**agn_log_lbol L_sun`
(C2): `agn_norm='conserving'`, and `'cigale_joint'` without a line block and without an
active `agn_ir_frac` tie. The ledger closes to 1e-6 on the blocks' own power, the dense
integral of each block alone; for every block but two that is the accretion power to 1e-6 (8e-7
largest). The two are the `richards2006` disc, whose template is normalized on the log-log
integral of its nodes but resampled linearly, so it carries 1.22e-4 more than
`10**agn_log_lbol L_sun` (#3106), and the toy `nenkova` torus, 2.09e-4 above `f L_bol`. Flag
0.0:

- the self-contained quasar templates: the `grahsp`, `grahsp_mn12` and `qsogen` tori, the
  `grahsp_sbpl` and `grahsp_netzer` discs (their parameter is `l5100`; the sub-912 A share of
  the GRAHSP big blue bump is `log_L_agn_bol_ionizing`), the `qsogen` disc and the
  `schartmann2005_skirtor_atten` disc (it removes light inside the block);
- `agn_norm='independent'` (every piece on its own scale);
- `'cigale_joint'` with lines (added on top of the disc, as CIGALE adds nebular emission) or with
  an active `agn_ir_frac` tie (CIGALE's `agn_power` normalization, unchanged for parity; the
  anchor is `log_L_agn_power`);
- the `polar_dust` block beside a torus whose library keeps its anisotropy (the five registered
  above): the runner splits the dust budget between the torus and the polar dust on the torus's
  sightline power rather than its angle-integrated power, so the pieces miss the accretion power
  by the anisotropy (measured +2.5% for `nenkova_agnfitter` at `cos i = 0.8`). Splitting on the
  angle-integrated power changes the SED and is not part of this ledger.

The first two kinds also emit a `RecipeWarning` at build naming the reason
(`validate_block_recipe` rule 10). No piece is rescaled to force a closure. In pure float32 the
ledger agrees with float64 to 1e-4 dex (3.6e-5 measured) wherever the float32 SED is finite;
an anisotropic template torus is not (#3101).

`log_L_agn_direct_los` (with `L_agn_direct_los_lsun`) is the line-of-sight isotropic-equivalent
power of the direct emission, the disc block's own spectrum at the viewing angle before the
torus screen, the polar dust, the tie and the energy debits, not scaled by `agn_lum_ratio`. It
was published as `log_L_agn_los` until #3076: a breaking rename with no alias (§0), since the
plain name now carries the emitted line-of-sight sum. It stays in the forward state. The X-ray
and radio components read `log_L_agn_bol` (the accretion power), never either line-of-sight
key.

Every case above is a **breaking change with no alias** (§0): the pre-v1.0 policy applies,
and these are the wording template (`CHANGELOG.md` #1206 entries) for the next one. There are
two exceptions, each keeping a `DeprecationWarning` alias through v0.x because the retired
name travels as a plain mapping key that cannot be intercepted at a call signature:
#2842's standalone-GRAHSP parameter `agn_grahsp_l5100`, and #3085's `agn_incl_skirtor`, which
reaches the SKIRTOR-AGNfitter kernels and blocks inside the composable runner's `params`
mapping (see "One inclination" below).

**One inclination (#2605, #3085).** Every torus reads the AGN's one inclination, `agn_cos_inc`
(`cos i`, `i` from the polar axis, 1 = face-on), as the disc screen and the X-ray corona do.
A library tabulated in another angle derives it: Fritz et al. 2006 at the elevation
`psi = 90 deg - i`, and the AGNfitter-rX SKIRTOR reductions (`skirtor_agnfitter`, `_1p`,
`_2p`) at `i = arccos(agn_cos_inc)` on their 0-90 deg axis. No torus declares an angle of
its own. The retired `agn_incl_skirtor` [deg] is refused by the grammar (in any group, as
`incl_skirtor` or `agn_incl_skirtor`) and by `model.predict`, each naming the `agn_cos_inc`
translation; the kernel and block functions accept it through v0.x as a `DeprecationWarning`
alias for `agn_cos_inc = cos(radians(agn_incl_skirtor))` and refuse a call that passes both
with different sightlines.

---

## 5. Inference Method Names

### Canonical method strings (13 total)

```
Approximate (fast):
  "map"                  — point estimate
  "laplace"              — Gaussian at MAP via Hessian
  "pathfinder"           — L-BFGS path (Zhang+2022)
  "vi"                   — geoVI via NIFTy optimize_kl (DEFAULT)
  "vi_linear"            — MGVI via NIFTy optimize_kl
  "vi_nifty_fast"        — geoVI fast path (~35% faster, no logging)
  "vi_nifty_fast_linear" — MGVI fast path (~35% faster, no logging)
  "vi_native"            — Native JAX geoVI (experimental, multi-seed)
  "vi_native_linear"     — Native JAX MGVI (experimental, multi-seed)

Exact (slow):
  "mcmc"             — auto-select (NUTS if D<=20, else Ray Tracing)
  "mcmc_raytrace"    — Ray Tracing HMC (Behroozi 2025)
  "mcmc_nuts"        — No-U-Turn Sampler
  "mcmc_ess"         — Elliptical Slice Sampling

Model selection:
  "nss"              — log Z via Nested Slice Sampling

Auto:
  "auto"             — D<=20: mcmc_nuts, else: vi
```

**Dimensionality D:** For the purpose of `"auto"` selection and `"mcmc"` auto-dispatch, D is the number of **free (sampled) parameters**, excluding deterministic transforms and derived quantities. For IFT/correlated-field models, D counts the latent excitation field DOF, not the pixel-space dimension.

### Removed aliases

| Old name (removed) | Maps to | Note |
|----------|---------|------|
| `vi_nifty` | `vi` | |
| `vi_nifty_linear` | `vi_linear` | |
| `geovi`, `fast_geovi`, `nifty_geovi`, `geovi_nuts` | `vi` | |
| `mgvi`, `fast_mgvi`, `nifty_mgvi`, `evi` | `vi_linear` | |
| `native_geovi` | `vi_native` | |
| `native_mgvi`, `native_evi` | `vi_native_linear` | |
| `raytrace` | `mcmc_raytrace` | |
| `nuts` | `mcmc_nuts` | |
| `elliptical_slice` | `mcmc_ess` | |
| `evidence` | `nss` | |
| `cat3d_wind_analytic` | `cat3d_wind_sed` | "_analytic" was a misnomer — grid interpolation, not closed-form |
| `silva04_analytic` | `silva04_sed` | as above |
| `skirtor_analytic` | `skirtor_sed` | as above |
| `fritz_analytic` | `fritz_sed` | as above |

Internal-only (not public API; renamed without alias):
`skirtor_agnfitter_analytic` → `skirtor_agnfitter_sed`,
`slone_netzer_analytic` → `slone_netzer_sed`.

---

## 6. SFH Function Names

| Canonical (Python function name) | Registry key (short alias) | Description |
|----------------------------------|---------------------------|-------------|
| `truncated_skewnormal` | `tsnorm` | Truncated skew-normal |
| `skewnormal` | `snorm` | Skew-normal |
| `lognormal` | `lnorm` | Log-normal |
| `gaussian` | `norm` | Normal (Gaussian) |
| `double_powerlaw` | `dpl` | Double power-law |

**Clarification:** The canonical names are the actual Python function names. The short aliases are string keys in `SFH_REGISTRY`, retained for backward compatibility and config-file brevity. `resolve_sfh("tsnorm")` and `resolve_sfh("truncated_skewnormal_sfh")` both resolve to the same function. New code and documentation should use the canonical names; the registry resolves both.

---

## 7. Module Boundary Rules

```
core/       -> never imports inference/
inference/  -> never imports models/ directly (only through core/ public API)
models/     -> only imports utils/
utils/      -> only imports stdlib + third-party (jax, numpy, etc.)
io/         -> may import core/, models/, utils/; never imports inference/
plotting/   -> may import core/, utils/; never imports inference/ or models/
```

**What `core/` exposes to `inference/`:** The `core/` package provides the public interface to `models/` via `SEDModel`, `Parameters`, and `build_*` factories. `inference/` must access all physics functionality exclusively through these `core/` entry points. Direct `from models.dust import ...` in `inference/` is forbidden.

No exceptions. Enforce via import-linter or CI grep.

---

## 8. Exception Hierarchy

```
TengriError (base)
├── ParameterError      — invalid parameter names, values, or conflicts
├── ConfigError         — invalid Config construction or missing fields
├── BackendError        — backend initialization or computation failure
├── InferenceError      — sampler/optimizer failures (convergence, NaN, etc.)
└── IOError             — file I/O, missing data files, format mismatch
```

All Tengri exceptions inherit from `TengriError`. New code must raise domain-specific exceptions, not bare `ValueError` or `RuntimeError`. The base class lives in `core/exceptions.py`.

---

## 9. JAX & Array Conventions

### JAX internals

- `_tree_flatten` / `_tree_unflatten` methods on pytree-registered classes: no prefix beyond the leading underscore.
- Custom differentiation rules: use `@jax.custom_jvp` / `@jax.custom_vjp` decorators. Name the custom rule function `_{original_name}_jvp` or `_{original_name}_fwd` / `_{original_name}_bwd`.
- Donated-buffer arguments: suffix `_donate` on the wrapper if it sets `donate_argnums`.

### Shape annotations

All functions operating on arrays must document shapes in the docstring using the following dimension names:

| Symbol | Meaning |
|--------|---------|
| `N` | Number of galaxies / objects in batch |
| `W` | Number of wavelength bins |
| `T` | Number of time steps (SFH grid) |
| `B` | Number of photometric bands |
| `D` | Number of free parameters |
| `S` | Number of posterior samples |

Example docstring:
```python
def compute_sed(params, ssp_grid):
    """
    Parameters
    ----------
    params : array, shape (N, D)
    ssp_grid : array, shape (T, W)

    Returns
    -------
    sed : array, shape (N, W)
    """
```

Adopt `jaxtyping` annotations where practical for static checking.

---

## 10. Spelling Convention (American English)

**Rule:** All identifiers (classes, functions, parameters, kwargs, constants,
registry keys, module filenames) and all prose (docstrings, comments, docs,
notebooks, example scripts) use **American English** spelling. British spellings
must never appear in new code.

| British (forbidden) | American (canonical) |
|---------------------|----------------------|
| `colour`, `behaviour`, `favour`, `neighbour` | `color`, `behavior`, `favor`, `neighbor` |
| `normalise`/`normalisation`, `marginalise`, `optimise`, `initialise`, `finalise` | `normalize`/`normalization`, `marginalize`, `optimize`, `initialize`, `finalize` |
| `catalogue`/`catalogued`, `analogue` | `catalog`/`cataloged`, `analog` |
| `centre`, `metre`, `fibre` | `center`, `meter`, `fiber` |
| `modelling`/`modelled`, `labelled` | `modeling`/`modeled`, `labeled` |
| `grey`, `analyse` (verb) | `gray`, `analyze` |

**Exception — external data contracts.** Strings that index a third-party file
or upstream API keep that source's spelling, because the literal must match the
bytes on disk. The known cases are the Synthesizer grid HDF5 dataset keys
`ionisation_parameter` and `log10_specific_ionising_luminosity`
(`components/nebular/agn_nebular.py`). Document any such exception inline.

**Note on American-invariant words:** do not "correct" words that are already
American or invariant — `noise`, `raise`, `exercise`, `precise`, `surprise`,
`revise`, `analyses` (plural noun of *analysis*), the `-wise` suffix
(`piecewise`, `otherwise`), and the matplotlib `Greys`/`Greens` colormap tokens.

### Renames applied (#819, no deprecation alias)

These were **hard-renamed** (no alias) under the pre-v1.0 policy of §0 — old
British spellings are gone, not deprecated. Per §2's file-path stability rule the
`marginalised.py` module was renamed in the same change.

| Old (removed) | Canonical | Module |
|-----|-----------|--------|
| `cue_full_catalogue` / builder key `full_catalogue` | `cue_full_catalog` / `full_catalog` | `parameters/`, `forward/`, `components/nebular/` |
| `rest_frame_colour()` | `rest_frame_color()` | `analysis/diagnostics/spectral.py` |
| `CalibrationMarginalisedLikelihood` (+ `ELine`, `CloudyELine`, `CalibrationELine`) | `…MarginalizedLikelihood` | `inference/likelihoods/marginalized.py` |
| `normalised_excess_variance()` | `normalized_excess_variance()` | `components/agn/grahsp/variability.py` |
| `rank_normalise()` / `rank_normalised_rhat()` | `rank_normalize()` / `rank_normalized_rhat()` | `analysis/diagnostics/autocorrelation.py` |
| `finalise()` | `finalize()` | `inference/backends/nested/utils.py` |
| `SSP_CATALOGUE_URL` | `SSP_CATALOG_URL` | `data/__init__.py` |

See `CHANGELOG.md` for the full record.

### Enforcement

`tools/check_british_spelling.py` is a CI guard (in the `lint` job) that fails
the build on any British spelling in `src/`, `tests/`, hand-written docs,
`examples/`, and active notebooks. Run `python tools/check_british_spelling.py`
locally, or `--fix` to rewrite case-preserving. Data-contract exceptions live in
`ALLOWED_TOKENS`; the rename-ledger docs and this guard's own test fixtures are
in `EXCLUDE_FILES`.

---

## 11. Prose Notation

**Governing rule: Unicode in prose, ASCII in code.** Bare ASCII stays inside code spans and literal parameter names — `tau_diff` is the actual kwarg and must never be "corrected" to `τ_diff` when it appears as an identifier. This rule explicitly forbids blanket find-and-replace across all contexts.

### Canonical notation table

| Concept | Prose | Code / literal |
|---|---|---|
| Citation | `Calzetti+2000`; multiples `Calzetti+2000; Draine+2007` | full ref in `Reference:` line |
| micron | `μm` (U+03BC GREEK SMALL LETTER MU only) | `um` in identifiers |
| Angstrom | `Å` | `AA` |
| Solar mass | `M☉` | `Msun` |
| Solar metallicity | `Z☉` | `Zsun` |
| Optical depth | `τ_diff`, `τ_bc`, `τ_V` | `tau_diff`, `tau_bc`, `tau_V` |
| Redshift, exact | `z = 3` (spaced) | — |
| Redshift, approximate | `z ≈ 3` (spaced); never `z~3` | — |
| Balmer alpha | `Hα` | `Halpha` |
| Gelman-Rubin | `R̂` | `rhat` |
| Multiplier | `1.46×` | — |
| Signal-to-noise | `S/N` | `snr` |
| Numeric range | en-dash: `16–84%` | — |
| Chi-square | `χ²` | `chi2` |
| Library name | `tengri` lowercase; `Tengri` only sentence-initial | `tengri` |
| Reference codes | each project's own styling: `BAGPIPES`, `Prospector`, `CIGALE`, `AGNfitter`, `Synthesizer`, `FSPS`, `DSPS` | — |

### Enforcement

Enforcement targets confusable codepoints — U+00B5 MICRO SIGN (which is why U+03BC is canonical), ligatures, homoglyphs, and non-breaking spaces — with intentional typography (superscripts, subscripts, ellipsis) explicitly permitted. The CI guard `tools/check_docs_voice.py` enforces this contract.

### Motivation

Measured across the 325-file user-facing corpus:
- 472 citations split three ways (inconsistent separators and spacing)
- 167 redshift forms split six ways (spacing, `~` vs `≈`, bare values)
- 126 micron uses split across two visually identical codepoints: `µm` (U+00B5 MICRO SIGN, 53 uses) vs `μm` (U+03BC GREEK SMALL LETTER MU, 56 uses) plus bare `um` and full `micron`
