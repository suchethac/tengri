# Model configuration reference

<!-- COUPLING NOTE: tools/check_doc_grammar_keys.py parses this file.
     Structure is strict: section headings "### Domain: `key`", marker "**Structural keys:**",
     and bullets "- `'key'` —". Changes to markdown format require updating the guard's parser. -->

This is the definitive guide to the nested-dict grammar for building a `SEDModel`. The guide covers universal grammar semantics, per-domain configuration, round-trip serialization, and common patterns.

## I. Universal grammar semantics

The nested-dict grammar used by `SEDModel.build` accepts one dict per physics group, each declaring what physics variant to use, which parameters are free, and per-parameter overrides.

### Three kinds of keys

Every group dict contains three kinds of keys:

**1. Structural keys**: select the model variant or configure the group's behavior. The key `'type'` is universal; some groups have additional structural keys (e.g., `'law'` for dust, `'norm'` for AGN). Structural keys are non-parameter settings.

```python
# 'type' selects the SFH model, 'age_kernel' configures integration
sfh={'type': 'dpl', 'age_kernel': 'cic'}
```

**2. The wildcard keys `'all_params'` and `'other_params'`**: exact synonyms that set the free/fixed status for every parameter in the group not explicitly overridden. Each accepts `FREE` (defer to the registry's default prior) or `Fixed(DEFAULT)` (pin at the registry default value). Never a concrete `Fixed(v)`, since one literal value cannot apply across every parameter in the group, and never an arbitrary `Distribution`. Giving both spellings in the same dict raises. The retired `'*'` synonym raises `ValueError` naming both accepted spellings.

Pick the spelling that reads best for the shape of the dict. Use `'all_params'` when the wildcard is the group's only directive. Use `'other_params'` written **last**, after explicit per-parameter entries, where it reads as "the others." The `to_groups()` method and the builder factories always emit following this convention:

```python
# Sole directive: 'all_params' reads best
dust_attenuation={'type': 'two_component', 'law': 'calzetti', 'all_params': Fixed(DEFAULT)}

# Mixed: explicit overrides first, 'other_params' last
dust_attenuation={'type': 'two_component', 'law': 'calzetti', 'tau_bc': 0.5, 'other_params': Fixed(DEFAULT)}
# NOT (legal, but not the taught style): {..., 'all_params': Fixed(DEFAULT), 'tau_bc': 0.5}
```

**3. Parameter keys**: bare parameter names or full prefixed names that override the wildcard or default. Short names are auto-prefixed by the group context.

```python
# All three mean the same: set tau_bc to 0.5
dust_attenuation={'type': 'two_component', 'tau_bc': 0.5}
dust_attenuation={'type': 'two_component', 'dust_tau_bc': 0.5}  # full prefixed name
dust_attenuation={'type': 'two_component', 'tau_bc': 0.5}       # short name (preferred)
```

### Free vs fixed: one-minute table

Every parameter is either **free** (sampled during inference) or **fixed** (pinned at a single value), and either uses the **registry default** or a **value/prior you provide**:

|            | registry default | your choice |
|------------|------------------|-------------|
| **free**   | `FREE` (sampled with the registry's default prior) | any `Distribution`, e.g. `Uniform(1, 3)` |
| **fixed**  | `Fixed(DEFAULT)` (pinned at the registry's default value) | `Fixed(0.3)` (pinned at your value) |

**`FREE`** means "free to vary using the registry's default prior." **`Fixed(DEFAULT)`** means "pinned at the registry's default value." The `'all_params'` wildcard applies one disposition to all parameters in the group. After explicit per-parameter entries, spell the remainder `'other_params'`:

```python
sfh={'type': 'dpl', 'all_params': FREE}                               # everything free
sfh={'type': 'dpl', 'beta': Uniform(1, 3), 'other_params': Fixed(DEFAULT)}  # beta free, rest pinned
dust_attenuation={'type': 'two_component', 'law': 'calzetti', 'tau_bc': 0.4, 'other_params': Fixed(DEFAULT)}  # tau_bc overridden, rest at defaults
```

### Prefixing and name resolution

Inside each group, parameter names are auto-prefixed with the group's canonical prefix:

- `sfh` group: `'alpha'` → `sfh_dpl_alpha` (with the type inserted)
- `dust_attenuation` group: `'tau_bc'` → `dust_tau_bc`
- `neb` group: `'logz'` → `neb_logz_cue` (varies by type)

Use the short form (unprefixed) whenever possible for readability. The full prefixed name works too, but the parser will resolve it back to the short form in the round-trip `model.spec.to_groups()`.

### Nesting and composition

Optional physics blocks (dust_attenuation, neb, agn, etc.) support nesting via sub-dicts. Each sub-block has its own `'type'`, `'all_params'`, and parameters:

```python
agn = {
    'type': 'composable',
    'disc': {'type': 'analytic_disk'},
    'torus': {'type': 'skirtor', 'all_params': FREE},
    'nlr': {'type': 'cue', 'logz': -0.3},
}
```

Sub-blocks inherit the same grammar rules as top-level groups.

### Activation and optional physics

Optional physics blocks (those NOT in the core set) are **OFF by default**:

- **Core groups with defaults:** `sfh`, `met`, `dust_attenuation`, `dust_emission`, `redshift`
- **Optional groups:** `neb`, `shock`, `agn`, `igm`, `radio`, `xray`, `foreground`

To activate an optional group, provide its dict with a `'type'`:

```python
# IGM is OFF (not provided)
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={...}, redshift=Fixed(0.1))

# IGM is ON (type given)
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={...}, igm={'type': 'inoue'}, redshift=Fixed(0.1))

# Explicit OFF (same as omitting it)
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={...}, igm={'type': 'none'}, redshift=Fixed(0.1))
```

An optional group with no `'type'` but with other keys raises `ParameterError` with the required syntax.

### Wildcard rules and no-op detection

`'all_params': FREE` on a group whose parameters default to `Fixed(DEFAULT)` is valid and cascades. However, when a wildcard cannot actually free anything, `'all_params': FREE` **raises `ParameterError`** instead of silently building a model with that physics pinned (#2187). Two shapes both raise:

**The group declares no parameters at all** under the selected configuration (e.g. `radio` with every sub-model disabled):

```python
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    radio={'sf': {'type': 'none'}, 'agn': {'type': 'none'}, 'all_params': FREE},
)
# Raises ParameterError:
# "'all_params'/'other_params': FREE in group 'radio' covers no parameters --
#  this group declares none to free under the selected configuration.
#  FREE resolves each parameter's registry default; with nothing declared
#  here there is nothing for it to resolve, so the fit would silently not
#  vary anything in this group.
#  Remove the wildcard, or pass explicit priors for the parameters you meant
#  to vary (e.g. radio={'param_name': Uniform(lo, hi)} for whichever
#  parameter your chosen configuration actually declares)."
```

**The group declares parameters, but every one of them is `Fixed`-only** (no declared `free_prior`), so the wildcard would free zero of them:

```python
model = SEDModel.build(ssp_data=ssp, observation=obs, met={'type': 'table', 'all_params': FREE})
# Raises ParameterError:
# "'all_params: FREE' freed 0 of 1 parameters in group 'met'. These have no
#  declared prior, only Fixed defaults:
#    met_alpha_fe
#  FREE resolves to each parameter's registry default, and these default to
#  Fixed; so the wildcard would leave every one of them pinned and the fit
#  would silently not vary this physics.
#  Pass explicit priors instead, e.g. met={'alpha_fe': Uniform(lo, hi)}."

# The remedy depends on the parameter -- it is not always "add a prior".
# met_alpha_fe is declared but not yet part of the released model, and no sweep
# has freed it end to end, so its effect on the SED is unconfirmed. For this
# parameter, drop the wildcard rather than free it: an explicit prior on an axis
# nobody has confirmed does anything is the inert-parameter failure this guard
# exists to catch.
model = SEDModel.build(ssp_data=ssp, observation=obs, met={'type': 'table'})

# CORRECT (general case): an explicit prior on a parameter you genuinely
# mean to vary -- gas-phase metallicity, live in every photoionized
# nebular backend:
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    neb={'type': 'cue', 'logZ_gas': Uniform(-1.0, 0.3)},
)
```

A partial free (where some parameters have a declared range and some do not) warns with `WildcardPartialFreeWarning` rather than raising, naming which ones stay pinned. See the docstring of `_check_wildcard_freed_something` for the full four-outcome table.

The same refusal applies to an **explicitly named per-parameter `FREE`**, not just the wildcard. Naming one specific parameter as `FREE` must free it or refuse, never silently leave it pinned. `met_alpha_fe` is the worked refusal example: the declared but not yet released alpha-enhancement axis has no `free_prior` (a wildcard cannot know whether the loaded SSP grid even carries an alpha-enhanced axis), so naming it `FREE` raises rather than quietly building a model with it pinned:

```python
model = SEDModel.build(
    ssp_data=ssp, observation=obs, sfh={...},
    met={'type': 'table', 'alpha_fe': FREE}, redshift=Fixed(0.1),
)
# Raises ParameterError:
# "'alpha_fe': FREE cannot be honored -- 'met_alpha_fe' has no declared
#  free prior (its registry default is Fixed(0.0)). Pass an explicit
#  prior instead, e.g. alpha_fe: Uniform(lo, hi)."

# CORRECT: an explicit prior for a parameter you genuinely mean to vary
model = SEDModel.build(
    ssp_data=ssp, observation=obs, sfh={...},
    met={'type': 'table', 'alpha_fe': Uniform(-0.2, 0.4)}, redshift=Fixed(0.1),
)
```

`redshift` is the worked example of the opposite outcome: a parameter **with** a declared default free prior. It declares `free_prior=Uniform(0.0, 20.0)`, an interval chosen to span and exceed every shipped recipe's redshift prior (photoz `Uniform(0.01, 6.0)`, high_z `Uniform(3.5, 10.0)`, stochastic/JWST `Uniform(0.01, 12.0)`), so `redshift=FREE` frees it over that range rather than raising:

```python
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={...}, redshift=FREE)
# Builds. model.spec.get_distribution('redshift') == Uniform(0.0, 20.0)

# Narrow it with an explicit prior for a survey-specific photo-z fit
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={...}, redshift=Uniform(0.01, 6.0))
```

### Error handling and suggestions

Unknown structural keys trigger a `ParameterError` with the list of valid keys and suggestions via `difflib`:

```python
dust_attenuation={'type': 'two_component', 'law': 'calzetti', 'tau_bc_': 0.5}
# Raises: "Unknown key 'tau_bc_' in dust_attenuation. Did you mean: tau_bc?"
```

---

## II. Per-domain configuration

Each physics block follows the universal grammar. This section lists the structural keys and one minimal working example for each.

```{include} _generated/parameter_tables.rst
```

Generated per-domain parameter references are shown above. For per-type parameter defaults, units, and descriptions, see [Components Reference](components.md).


### Star-formation history: `sfh`

**Structural keys:**
- `'type'`: SFH model (`'dpl'`, `'delayed_tau'`, `'lognorm'`, `'field'`, etc.). Menu: `tengri.list_sfh_models()`.
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'age_kernel'`: Integration method: `'cic'` (default, cloud-in-cell) or `'dsps'` (histogram). See the model grammar design guide for performance details.
- `'bin_edges_gyr'`: Non-parametric bin edges (Gyr). Only for `type='histogram'` or similar.
- `'field_centering'`: Field draw centering ('none' or 'mean'). Only for `type='field'`.

**Minimal example:**
```python
sfh={'type': 'dpl', 'all_params': FREE, 'beta': Uniform(1, 3), 'age_kernel': 'cic'}
```

**Gotchas:**
- `'age_kernel': 'dsps'` is **not** a performance knob. It is 13% slower than the default. Use `'cic'` unless you need DSPS cross-code parity. The 'cic' kernel preserves mass-proportionality to roundoff; 'dsps' costs it, typically well below 1e-5 but reaching roughly 1e-3 at the sharpest SFH shapes (#2368).
- A field SFH accepts both kernels (#2684): the draw is the linear interpolation of its own lookback nodes, integrated by `'cic'` with the nodes as knots and by `'dsps'` on a table refined 8-fold between SSP nodes. Without an explicit `age_kernel` a field SFH uses `'cic'`, the accurate kernel for field and rough histories; `'dsps'` differs there by up to 16 % in the FUV and 9 % in r-band flux.
- Default `age_kernel` is `'cic'` for every SFH type.


### Metallicity: `met`

**Structural keys:**
- `'type'`: Metallicity model (`'table'` for per-age SSP indexing, `'ramp'` for a linear Z(t) history, etc.). Menu: `tengri.list_metallicity_modes()`.
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'met_bin_edges_log_yr'`: Lookback-time bin edges [log₁₀ yr] for metallicity-history modes (`'bins'` or `'bins_continuity'`). Default spans 1 Myr to 13.8 Gyr. Accepts an array of strictly increasing edge values (at least 2 edges). Mirroring `sfh={'bin_edges_gyr': [...]}` for non-parametric star formation histories.

**Minimal example:**
```python
met={'type': 'table'}  # all_params defaults to Fixed(DEFAULT)
met={'type': 'ramp', 'logzsol_0': Fixed(-0.3), 'logzsol_1': Free}  # two-knot ramp
met={'type': 'bins', 'all_params': Fixed(DEFAULT), 'met_bin_edges_log_yr': [6.0, 8.0, 9.5]}  # custom ladder
```

**Gotchas:**
- Metallicity is **separate from star formation**. `met=` selects SSP templates; `neb={'logZ_gas': ...}` drives nebular emission independently.
- Default `met_logzsol = 0.0` (solar). The SSP grid uses absolute `log10(Z)` internally, with a Zsun offset (Asplund 2009).
- A tabulated (per-SSP-age) `met=` beside a non-tabulated `sfh` warns but does not raise.


### AGB circumstellar dust shell: `agb_dust`

**Structural keys:**
- `'type'`: AGB dust-shell model: `'fsps_shell'` (the only supported type). Omitting the group, or `agb_dust={'type': 'none'}`, leaves the SSP grid untouched.
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).

**Minimal example:**
```python
agb_dust={'type': 'fsps_shell'}  # weight fixed at 1.0 (bit-identical to omitting the group)
agb_dust={'type': 'fsps_shell', 'weight': Uniform(0, 3)}  # free
agb_dust={'type': 'fsps_shell', 'weight': Fixed(2.0)}  # fixed at a non-default strength
```

**Gotchas:**
- `'weight'` is the short-form override for `agb_dust_weight`, the dimensionless scale on FSPS's Villaume, Conroy & Johnson (2015) circumstellar AGB dust-shell reprocessing. Default `Fixed(1.0)` (the grid as shipped, FSPS's own `agb_dust` default). Free prior `Uniform(0, 3)`.
- MIST-isochrone SSP libraries only (`fsps_mist_*`): FSPS's `add_agb_dust_model` routine is MIST-only. Any other grid raises at build time, naming the supported grids.
- The ratio was computed with FSPS 0.4.7 using MIST isochrones, the MILES spectral library and a Chabrier IMF, and is reused for the `c3k_a` grids and the Kroupa and Salpeter IMFs. The IMF dependence was measured only at solar metallicity (ages 0.3, 1, 3 Gyr; weights 0 and 3): at most 0.004 (Kroupa) and 0.019 (Salpeter) in the ratio. The spectral-library dependence could not be measured.
- The ratio is stored on a ladder of weights and interpolated linearly between them: between stored weights it matches direct FSPS to better than 2% over 2-30 um (0.4-1.6% per interval, measured at solar metallicity for 0.3, 1 and 3 Gyr). Below w = 1/1024 the error reaches 4% (3.6% over 2-30 um), because the ratio is nearly discontinuous at w = 0 (FSPS switches the shell off); a weight near 0 carries a few-percent error. A free weight reaches the exact spectrum, the photometry and `compute_log_nion`; the nebular Q_H tables built once at model construction read the unweighted cube (measured effect below 1e-4 dex up to 10 Myr, at most 0.107 dex in single 45-100 Myr cells where Q_H is negligible).
- The gradient with respect to the weight is the slope of the bracketing segment, so it has a kink at every stored weight (1/1024, 1/128, 1/8, 7/32, 5/16, 7/16, 9/16, 13/16, 1, 5/4, 13/8, 2, 3). For a 1 Gyr population the 10 um ratio peaks close to w = 1, so its slope changes sign around it.
- The ratio spans 0.055 (far-infrared, w = 0) to 236 (old, metal-poor populations at 1200-1500 A) and is not clamped. It is exactly 1 outside 284 A to 3.4e7 A; the lower edge comes from a few 45-100 Myr low-metallicity populations at the 1e-4 threshold, not from shell absorption.
- A **fixed** weight is baked into the SSP tensor at `SEDModel.build` time, so the exact path and every precompute table (`WavePrecomp`, `SpectrumPrecomp`, `FeaturePrecomp`) stay bit-exact.
- A **free** weight cannot be baked (its value is only known per sample). Exact photometry, spectra and the ionizing rate apply it live. Everything built once from the SSP cube refuses, naming the exact path: `WavePrecomp`, `SpectrumPrecomp`, the `FeaturePrecomp` window table for baked-in lines, and `approx=True` on `predict_spectral_indices` / `measure_line_fluxes`. The `FeaturePrecomp` Cue grid reads the live ionizing rate and works. `approx='auto'` resolves to the exact path for single, catalog and population fits.


### Dust attenuation: `dust_attenuation`

**Structural keys:**
- `'type'`: Architecture: `'single_component'` (one dust screen) or `'two_component'` (birth-cloud + diffuse). Default varies by law.
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'law'`: Attenuation law: `'calzetti'`, `'ccm89'`, `'mw_rv31'`, `'kext00'`, etc. Menu: `tengri.list_dust_laws()`.
  - On `'single_component'`: one law for the entire dust.
  - On `'two_component'`: `'law'` applies to both screens. Override per-screen with `'law_bc'` and `'law_diff'`.
  - On `'wg00'` (Willis & Graves 2000 screen): use structural keys like `'dust_curve'`, `'geometry'`, `'structure'` instead of a law name.
- `'law_bc'`: Birth-cloud attenuation law (two-component only). Required with `'law_diff'` when not using shared `'law'`.
- `'law_diff'`: Diffuse dust attenuation law (two-component only). Required with `'law_bc'` when not using shared `'law'`.
- `'law_neb'`: Nebular dust law (the curve for the nebular channel; which screen that channel passes through is `'nebular_screen'`).
- `'nebular_screen'`: Which dust screen the nebular continuum, the line catalog and the fast-nebular fallback pass through: `'birth_cloud'` (default: the young-star screen, birth cloud + diffuse), `'diffuse'` (the old-star screen) or `'none'` (`'off'` is a synonym). Two-component only; `'single_component'` accepts only `'none'`.
- `'shock_screen'`: Which screen the shock SED passes through; same values; default `'diffuse'` (AGN-outflow shocks sit outside the birth clouds).
- `'agn_screen'`: Screen choice for AGN continuum attenuation: `'birth_cloud'`, `'diffuse'`, or `'none'` (default). Two-component only. Incompatible with `agn_norm='cigale_joint'` (both read the dust budget); use `agn_norm='independent'` or `'conserving'` instead.
- `'dust_curve'`: WG00 dust curve selector (only for `type='wg00'`).
- `'geometry'`: WG00 geometry ('slab', 'sphere', etc.) (only for `type='wg00'`).
- `'structure'`: WG00 structure ('clumpy', 'homogeneous', etc.) (only for `type='wg00'`).
- `'slope_bc'`, `'slope_diff'`, `'slope_neb'`: Per-screen law-parameter overrides (two-component only). Accepted only when *that screen's* law reads a slope.
- `'bump_strength_bc'`, `'bump_strength_diff'`, `'bump_strength_neb'`: Per-screen bump-strength overrides (two-component only). Accepted only when that screen's law reads a bump strength.
- `'Rv_bc'`, `'Rv_diff'`, `'Rv_neb'`: Per-screen RV overrides (two-component only). Accepted only when that screen's law reads R_V.
- `'delta_bc'`, `'delta_diff'`, `'delta_neb'`: Per-screen delta overrides (two-component only). Accepted only when that screen's law reads a slope modification.
- `'lyman_cutoff'`: Zero attenuation below 912 Å (Lyman limit). Two-component only.
- `'lyc_absorb_all'`: Absorb all ionizing photons (FSPS/CIGALE style) vs young-only (default). Two-component only.
- `'eb_include_lyc'`: Include ionizing luminosity in the dust energy-balance integral (FSPS/Prospector parity). Default false.
  (See also `'diffuse_screen'` under `dust_emission` below: an analogous opt-in single-pass toggle, applied to the *escaping* re-emitted IR through this group's diffuse screen rather than to the absorbed budget.)
- `'screens'`: `type='age_binned'` only (#2528). A list of `{'law': <law name>, 'window_log_yr': (lo, hi)}` dicts, one per screen; `lo`/`hi` are `log10(age/yr)` edges, either or both `None` for unbounded. N independent screens generalize the birth-cloud/diffuse pair to any N; windows need not partition the age axis. Optical depths add over every screen whose window contains a star's age: nested windows (each `(None, hi_i)` with increasing `hi_i`, the last unbounded) cascade as the birth cloud and diffuse medium do in `'two_component'`, and `tau_i` is the depth screen `i` adds; windows that tile the age axis give each age a single screen, and `tau_i` is the total depth of that age bin. Per-screen parameters are indexed from the screen count: `'tau_0'`, `'tau_1'`, ... (full name `dust_tau_i`), plus one `'<lawparam>_i'` for every shape parameter that screen's own law declares (e.g. `'slope_0'`, `'Rv_2'`), defaulting to that law's own published value. `screens = [{'law': law_bc, 'window_log_yr': (None, log10(t_birth))}, {'law': law_diff, 'window_log_yr': (None, None)}]` reproduces `'two_component'` bit-identically. Not supported under `approx=WavePrecomp()`/`SpectrumPrecomp()` (raises at construction; a fit's `approx="auto"` resolves to the exact path instead). Nebular continuum and the line catalog are attenuated only by screens whose window is unbounded below (the `t -> 0` limit, the same convention `'two_component'` applies to lines); a finite lower edge must sit at least five transition widths above the loaded SSP grid's youngest node or the build raises `ConfigError`, naming the screen, the edge, the grid's youngest node, and the two fixes.

Each of the 12 per-screen keys above (`'slope_bc'`, `'bump_strength_bc'`,
`'Rv_bc'`, `'delta_bc'`, and their `'_diff'`/`'_neb'` siblings) takes
**either** a plain number (as before) **or** `Fixed(...)`/a prior
distribution (#2428). A plain number is a build-time config override, baked
into the compiled model exactly as it always was. `Fixed(...)`/`Uniform(...)`/
etc. instead declares a real, free-able parameter named
`dust_<stem>_<screen>` (e.g. `dust_slope_bc`, `dust_Rv_neb`). It appears in
`spec.free_params` when given a prior, and a `Fixed(v)` per-screen
declaration predicts bit-identically to the plain number `v`. These names are
explicit-only: an `all_params: FREE` wildcard never frees them (name one
explicitly to fit it), and the flat `Parameters(dust_law_overrides={...})`
surface still accepts only plain numbers in that dict. Passing a prior there
raises `ParameterError` naming the `dust_<stem>_<screen>` spelling as the
remedy. Naming only one half of a `_bc`/`_diff` pair (e.g. `slope_bc` without
`slope_diff`) raises. Give both explicitly, or use a group-level wildcard
(`'all_params': FREE`/`Fixed(DEFAULT)`) to free or pin them together, following
the same rule the plain-number spelling of these keys already enforces.

**Minimal example:**
```python
# Single component
dust_attenuation={'type': 'single_component', 'law': 'calzetti', 'tau': 0.5, 'other_params': Fixed(DEFAULT)}

# Two component (one law)
dust_attenuation={'type': 'two_component', 'law': 'calzetti', 'tau_bc': 0.4, 'tau_diff': 0.2, 'other_params': Fixed(DEFAULT)}

# Two component (per-screen laws)
dust_attenuation={'type': 'two_component', 'law_bc': 'ccm89', 'law_diff': 'calzetti', 'tau_bc': 0.4, 'tau_diff': 0.2}

# WG00 screen with structural selectors
dust_attenuation={'type': 'wg00', 'dust_curve': 'mw_rv31', 'geometry': 'slab', 'structure': 'clumpy'}

# Age-binned, nested windows (#2528): a cascade; young stars see all three screens,
# and each tau_i is the depth that screen adds
dust_attenuation={
    'type': 'age_binned',
    'screens': [
        {'law': 'calzetti', 'window_log_yr': (None, 7.0)},
        {'law': 'power_law', 'window_log_yr': (None, 8.5)},
        {'law': 'cardelli', 'window_log_yr': (None, None)},
    ],
    'tau_0': 0.5, 'tau_1': 0.3, 'tau_2': 0.2, 'other_params': Fixed(DEFAULT),
}

# Age-binned, tiling windows: each age sees one screen; each tau_i is that bin's total depth
dust_attenuation={
    'type': 'age_binned',
    'screens': [
        {'law': 'calzetti', 'window_log_yr': (None, 7.0)},
        {'law': 'power_law', 'window_log_yr': (7.0, 8.5)},
        {'law': 'cardelli', 'window_log_yr': (8.5, None)},
    ],
    'tau_0': 1.0, 'tau_1': 0.5, 'tau_2': 0.2, 'other_params': Fixed(DEFAULT),
}
```

**Gotchas:**
- Dust attenuation and dust emission are **two separate peer groups**, not nested. The retired `dust={'attenuation': {...}, 'emission': {...}}` form raises.
- Each emission source passes through the screen its selector names (`'nebular_screen'`, `'shock_screen'`, `'agn_screen'`); the absorbed nebular and shock power joins the dust energy balance under those screens.
- Two-component law-pairing rule: if you name one of `'law_bc'`/`'law_diff'`, you must name both (or use a shared `'law'` for both).
- Parameters like `'slope'`, `'bump_strength'`, `'Rv'`, `'delta'` are set per-screen on two-component (`'slope_bc'`, `'slope_diff'`, etc.). On single-component, just `'slope'`: the per-screen spellings raise there, because a single screen has no second screen to name and the value would never reach a curve.
- **A shape key must be one the selected law reads.** Each law declares exactly the parameters it uses, so `'slope'` under `'noll09'` raises and names `'delta'`, the parameter that law does read; `'Rv'` under `'calzetti'` raises, because that curve is fitted at R_V = 4.05; `'slope'` under `'vw07_bc'` / `'vw07_diff'` raises, because those are the Charlot & Fall birth-cloud and diffuse slopes, not knobs (use `'power_law'` for a free slope). The same rule applies per screen: with `'law_bc': 'power_law', 'law_diff': 'noll09'`, `'slope_bc'` is accepted and `'slope_diff'` is not. A lone `'slope_bc'` is then complete, because there is no partner to give.
- The `'neb'` channel (`'law_neb'`, `'slope_neb'`, etc.) reddens **only the nebular birth-cloud continuum**, not the young stars. Used when nebular emission is routed through a different dust screen. `'law_neb'` defaults to `'law_bc'`, and a `'*_neb'` override is checked against whichever of the two is in force.


### Dust emission: `dust_emission`

**Structural keys:**
- `'type'`: Emission model: `'dale2014'`, `'draine2016'`, etc. Menu: `tengri.list_dust_emission_models()`.
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'spinning_dust'`: Include small spinning dust grains (default: auto from type).
- `'f_cnm'`: Cold neutral medium fraction (parametrization-dependent).
- `'eta_balance'`: Energy-balance coupling: `Fixed(1.0)` (default, strict balance where `L_IR = eta * L_absorbed`). `FREE` selects the default free prior, `Gaussian(1.0, 0.2)` truncated at 0. Pass an explicit `Uniform(...)` or any other prior to override it. The `all_params` wildcard reaches it on **every** emission engine, since it scales the budget every template normalizes to, so it remains live regardless of the selected engine (#2286).
- `'log_L_ir'`: Total dust IR budget override, `log10(L_IR/L_sun)`. Declaring it (with `Fixed(...)` or any prior) **replaces** the energy-balance budget outright. Leaving it undeclared keeps energy balance. Because it renders `eta_balance` inert, declaring both (with `eta_balance` free or fixed not equal to 1) raises at build. Radio's FIRRC amplitudes follow this budget, so it is not a dust-only knob. Never reached by the `all_params` wildcard. An explicit `FREE` on it is refused; declare a real prior instead.
- `'diffuse_screen'`: Opt-in single-pass attenuation of the re-emitted IR dust emission by the diffuse dust screen's transmission `T(λ)` (the `dust_diff_transmission` derived key, published by every `dust_attenuation` type). The escaping SED is `sed_dust_ir * T`: the IR energy absorbed on the way out is REMOVED, not re-emitted or iterated back into the budget (unlike FSPS, which iterates the IR to convergence; CIGALE never attenuates its re-emitted IR at all — this is a deliberate single pass). `log_L_ir_emergent` reports the escaping (post-screen) IR luminosity; `L_ir`/`L_absorbed` (and radio's FIRRC amplitudes, which read `L_ir`) keep the pre-screen absorbed budget unchanged. For single-screen attenuators (`dust_attenuation={'type': 'single_component', ...}`) and for `type='wg00'`, "diffuse" means the *single* screen — there is no separate birth-cloud/diffuse split to choose between. Default false (off, bit-identical to a build without this key). Requires an active, non-`'none'` `dust_attenuation` and a real `dust_emission` type; raises a clear `ValueError` at parse time otherwise.

**Minimal example:**
```python
dust_emission={'type': 'dale2014', 'eta_balance': Fixed(1.0), 'other_params': Fixed(DEFAULT)}
```

**Gotchas:**
- Energy balance: `eta_balance` defaults to `Fixed(1.0)`, which enforces `L_IR = L_absorbed`. `FREE` resolves to `Gaussian(1.0, 0.2)` truncated at 0. Setting it free or to a constant ≠ 1 decouples IR and absorption.
- Missing dust_emission (or `{'type': 'none'}`) is valid and common for UV-only work.
- `'diffuse_screen'` under `dust_attenuation={'type': 'wg00', ...}`: the vendored WG00 attenuation tables (`data/wg00_attenuation_grid.h5`) are tabulated only 1000–30001 Å (0.1–3 μm). Past that domain `T = 1` by construction, so under `wg00` the switch has **no effect in the far-IR** where dust re-emission actually peaks — this is a limitation of the vendored table's wavelength coverage, not a defect in the diffuse-screen feature itself.


### Nebular emission: `neb`

**Structural keys:**
- `'type'`: Backend: `'cue'` (Cue, default), `'cloudy'` (CLOUDY, slower, higher fidelity), `'cb19'` (Charlot & Bruzual 2019), `'mappings'` or `'mappings_agn'` (MAPPINGS V stellar and AGN; **both backends are registered as experimental; both refuse loudly pending data rehabilitation** (#2082): stellar grid is 51.2% NaN, AGN backend lacks protocol surface), or `'none'` (off). Menu: `tengri.list_nebular_backends()`.
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'full_catalog'`: `cue` only: bool, default `True` (#2239). Publishes the full ~138-line Cue-trained catalog. Set to `False` to narrow to the legacy 128-line CLOUDY/FSPS-matched subset, kept for cross-code comparisons. No-op on other backends.
- `'nitrogen'`: `cue` only: `'absolute'` (default) or a relation name (`'nicholls17'`) (#2693). Selects the meaning of `gas_logno`; see the nebular notes below. Raises on other backends.
- `'grid'`: Path to the backend's own HDF5 grid file. Accepted only for `'cloudy'`, `'cb19'`, `'mappings'`, and `'mappings_agn'`; `None` (the default) resolves each backend's own packaged grid (#2220).
- `'model'`: MAPPINGS V stellar model (`'mappings'` type only): `'sb99'` (Starburst99) or `'bpass'` (BPASS v2.2).
- `'density'`: MAPPINGS V density structure (`'mappings'`/`'mappings_agn'`): `'cpr'` (isobaric, recommended) or `'cdn'` (isochoric).
- `'ionizing_source_warning'`: MAPPINGS V ionizing-source warning control (`'mappings'`/`'mappings_agn'`): `'raise'`, `'warn'`, or `'suppress'`.

**Minimal example:**
```python
neb={'type': 'cue', 'logZ_gas': -0.3, 'other_params': Fixed(DEFAULT)}
neb={'type': 'cloudy', 'grid': {'logz': [-2, -1, 0], 'logU': [-3, -2, -1]}}
```

**Gotchas:**
- Nebular metallicity (`'neb_logZ_gas'` or short `'logZ_gas'` in the `neb` dict) is **independent** from stellar metallicity (`'met='`).
- Default `neb_logZ_gas = -0.3` (solar). It is **not automatically inherited** from the stellar metallicity, even if tabulated.
- Nebular emission is **additive** to stellar continuum; it composites with dust and shock when both are present.
- Cue's `gas_logno` is its absolute [N/O] input by default; `neb={'type': 'cue', 'nitrogen': 'nicholls17'}` makes it the offset from the Nicholls+2017 N/O--O/H relation at `neb_logZ_gas`, the convention of `neb_dno` in the grid backends (which embody their own, unrecorded, relation; the two agree at the N/[O II] level, Cue/grid 0.78-0.99 over `neb_logZ_gas` -1 to 0). The [N/O] Cue is actually fed is the `log_no` property in both modes (#2693).


### Shock emission: `shock`

**Structural keys:**
- `'type'`: Shock backend: `'mappings'` (MAPPINGS V, default) or `'none'` (off).
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'norm'`: Normalization: `'frac'` (scales the galaxy Hα), `'lhalpha'` (absolute Hα luminosity, decoupled from SFR), or `'component'` (explicit component label).
- `'abundance'`: Abundance mode: `'solar'`, `'lmc'`, etc.
- `'component'`: Component label for compartmentalization (advanced).

**Minimal example:**
```python
shock={'type': 'mappings', 'norm': 'frac', 'frac': Uniform(0, 1)}
shock={'type': 'mappings', 'norm': 'lhalpha', 'lhalpha': 10**42}  # erg/s
```

**Gotchas:**
- `'all_params': FREE` on `shock` **raises** if no free-parameter models are configured. Use explicit priors instead.
- Shock and nebular emission compose (both can be on). The shock norms apply independently.
- Default `shock_abundance = 'solar'`. No abundance parameter by default.


### IGM absorption: `igm`

**Structural keys:**
- `'type'`: IGM model: `'inoue'` (Inoue+2014, default), `'madau'` (Madau+1995), `'meiksin06'` (Meiksin 2006), `'none'` (off).
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'patchy'`: Picket-fence vs smooth IGM: bool, model-dependent default.
- `'dla'`: Damped Lyman alpha: omit for no DLA, or provide `{'type': ...}` for DLA models (e.g., `{'type': 'dla_lookback'}` to evolve DLA properties with redshift).

**Minimal example:**
```python
igm={'type': 'inoue'}  # Smooth IGM
igm={'type': 'inoue', 'patchy': True}  # Picket-fence
igm={'type': 'inoue', 'dla': {'type': 'dla_lookback'}}  # With evolving DLA
```

**Gotchas:**
- IGM is applied **only to observed-frame wavelengths** (after redshifting). It affects photometry and spectroscopy, not rest-frame predictions.
- IGM models are applied to **observer-frame** wavelengths, so redshift must be specified for IGM to have an effect.
- `'patchy'` has minimal effect on Inoue (mostly Rayleigh scattering); larger on Madau.


### Radio emission: `radio`

**Structural keys:**
- `'type'`: Radio model: `'sfonly'` (star-formation only, default), `'agn'` (AGN only), `'sf_agn'` (both), `'none'` (off).
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'sf'`: Star-formation radio sub-block: `{'type': ...}` to customize.
- `'agn'`: AGN radio sub-block: `{'type': ...}` to customize.

**Minimal example:**
```python
radio={'type': 'sfonly'}
radio={'type': 'sf_agn', 'sf': {'type': 'condon'}, 'agn': {'type': 'nandra'}}
```

**Gotchas:**
- Radio parameters are **fixed by default**. `'all_params': FREE` raises unless a model with free params is configured.
- Use explicit priors on individual parameters, e.g., `radio={'q10': Uniform(-0.5, 0.5)}`.
- Radio is **composable**: both SF and AGN can emit at once.


### X-ray emission: `xray`

**Structural keys:**
- `'type'`: X-ray model: `'yang22'` (Yang+2022), `'lehmer'` (Lehmer+2022), `'none'` (off).
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).

**Minimal example:**
```python
xray={'type': 'yang22', 'all_params': Fixed(DEFAULT)}
xray={'type': 'lehmer', 'log_nH': 21.0}  # Hydrogen column density, log10(cm^-2)
```

**Gotchas:**
- X-ray emission scales with star-formation rate and optionally AGN accretion rate.
- Default calibrations assume specific normalization conventions; consult paper for details.
- X-ray can be free or fixed like any component.


### AGN: `agn`

**Structural keys:**
- `'type'`: AGN mode: `'composable'` (six independent emitters), `'legacy'` (single monolithic AGN), or `'none'` (off).
- `'all_params'`: Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'norm'`: Across-component normalization: `'cigale_joint'` (default, CIGALE-style energy conservation across disc/torus/polar) or `'independent'` (each component on its own scale).
- `'disc'`: AGN accretion disk sub-block (with `'type'`, `'all_params'`, parameters).
- `'torus'`: Infrared-obscured torus sub-block (with `'type'`, `'all_params'`, parameters).
- `'nlr'`: Narrow-line region sub-block (with `'type'`, `'all_params'`, parameters).
- `'blr'`: Broad-line region sub-block (with `'type'`, `'all_params'`, parameters).
- `'feii'`: Iron emission sub-block (with `'type'`, `'all_params'`, parameters).
- `'atten'`: AGN-specific attenuation sub-block (with `'type'`, `'all_params'`, parameters, and `'law'` for dust law selection).
- `'lines'`: Deprecated: expands to `'nlr'` + `'blr'`.

**Minimal example:**
```python
agn={
    'type': 'composable',
    'disc': {'type': 'analytic_disk', 'all_params': Fixed(DEFAULT)},
    'torus': {'type': 'skirtor', 'all_params': FREE},
    'nlr': {'type': 'cue', 'all_params': Fixed(DEFAULT)},
    'norm': 'cigale_joint',
}

# Or a monolithic model (one self-contained block, no sub-blocks):
agn={'type': 'skirtor_stalevski', 'all_params': Fixed(DEFAULT)}
```

**Gotchas:**
- On `'composable'`, all six sub-blocks are **optional**. Omitting one deactivates it (e.g., no `'disc'` means no direct accretion disk emission).
- Each sub-block follows the same grammar (type, all_params, parameters) as a top-level group.
- `'norm': 'cigale_joint'` ties disc/torus/polar normalization to a single AGN-power reference; `'norm': 'independent'` lets each float freely. 
- AGN sub-block names (`'disc'`, `'torus'`, etc.) are **single**, not plural. `'discs'` raises.


### Foreground extinction: `foreground`

**Structural keys:**
- `'ebmv_mw'`: Milky Way E(B-V) reddening (mag). Typically 0.01–0.2.
- `'law'`: Dust law: `'mw_rv31'` (Fitzpatrick 1999, default), `'ccm89'` (CCM89), etc.
- `'rv'`: Dust RV parameter override (law-dependent).

**Minimal example:**
```python
foreground={'ebmv_mw': 0.05, 'law': 'mw_rv31'}
```

**Gotchas:**
- Foreground reddening is applied **in addition to** any dust attenuation in the model.
- It is a **source reddening**, not observer-frame extinction.
- No free parameters by default (use Fixed() if you need to pin values precisely).

### Redshift: `redshift`

**Structural keys:** None (redshift is a scalar or Distribution, not a dict).

The redshift can be specified as:
- `Fixed(z)`: Known redshift (e.g., `Fixed(0.05)`).
- `Uniform(z_min, z_max)`: Photo-z prior (e.g., `Uniform(0.0, 2.0)`).
- Any `Distribution` (e.g., `Normal(...)`).
- A bare scalar: `redshift=0.05` auto-converts to `Fixed(0.05)`.

**Minimal examples:**
```python
redshift=Fixed(0.05)                  # Known redshift
redshift=Uniform(0.0, 1.0)            # Photo-z fit
redshift=0.05                         # Auto-converts to Fixed(0.05)
```

**Gotchas:**
- **Redshift is REQUIRED**. Omitting it raises `ParameterError` listing the three allowed forms.
- A free redshift (Uniform or Distribution) is **expensive** with precompute (`approx=WavePrecomp(...)`). See the [performance guide](performance/compilation.md) for details.
- IGM models apply **only** to observed-frame wavelengths, so redshift **must** be known for IGM to work.

---

## III. Round-trip: configuration serialization

A model's resolved configuration can be inspected and edited via serialization.

### Inspect the model config

```python
model = SEDModel.build(ssp_data=ssp, observation=obs, **config)

# Get the resolved config as a nested dict
config = model.spec.to_groups()
# config is a plain dict ready for re-parse or export

# Print a formatted summary with provenance
model.spec.summary()  # Tags show [user], [default], [all_params FREE], etc.

# Access the raw spec (lower-level; rarely needed)
model.spec  # the Parameters object
```

### Export to YAML / JSON

Round-trip serialization (planned for #75) will support:

```python
# Export to YAML
model.to_yaml("config.yaml")
model.to_json("config.json")

# Load from file
model = SEDModel.from_yaml("config.yaml", ssp_data=ssp, observation=obs)
model = SEDModel.from_dict(config_dict, ssp_data=ssp, observation=obs)
```

For now, use:

```python
import yaml
config = model.spec.to_groups()
with open("config.yaml", "w") as f:
    yaml.dump(config, f)
```

### Round-trip semantics

`to_groups()` never expands a wildcard into individual per-parameter priors. A mixed group (explicit overrides plus a wildcard) round-trips with the overrides first and the wildcard collapsed to `'other_params'` last. This follows the same convention the grammar teaches for hand-written dicts:

```python
# Input
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={'type': 'dpl', 'all_params': FREE, 'beta': Uniform(1, 3)}, met={'type': 'table'}, redshift=Fixed(0.1))

# Output of model.spec.to_groups()
{
    'sfh': {
        'type': 'dpl',
        'beta': Uniform(1.0, 3.0),  # user override, kept explicit
        'other_params': FREE,       # everything else in the group, collapsed
    },
    # ... other groups ...
}
```

A sole-directive wildcard (no explicit per-parameter overrides) round-trips unchanged:

```python
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={'type': 'dpl', 'all_params': FREE}, met={'type': 'table'}, redshift=Fixed(0.1))
model.spec.to_groups()['sfh']
# {'type': 'dpl', 'all_params': FREE}
```

Unknown keys in a saved config file are detected on re-parse and raise with suggestions.

---

## IV. Common patterns

### Recipe + tweak

Start with a recipe, then modify:

```python
from tengri import recipes, Uniform

config = recipes.star_forming_photometry()
config['dust_attenuation']['tau_bc'] = 0.6  # override default
config['neb']['logZ_gas'] = Uniform(-0.5, 0.0)  # photo-Z on gas metallicity

model = SEDModel.build(ssp_data=ssp, observation=obs, **config)
```

### Variant swap

Exchange one sub-component:

```python
config = recipes.agn_panchromatic()
config['agn']['disc'] = {'type': 'bbflat'}  # swap the disk model
model = SEDModel.build(ssp_data=ssp, observation=obs, **config)
```

### Wildcard + explicit override

Free everything except a few things:

```python
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh={'type': 'dpl', 'beta': Fixed(2.0), 'other_params': FREE},  # beta pinned, the rest free
    dust_attenuation={'type': 'two_component', 'tau_bc': Uniform(0, 1), 'other_params': Fixed(DEFAULT)},  # only tau_bc free
    redshift=Fixed(0.1),
    ...
)
```

### Composable AGN with mixed freedom

```python
agn = {
    'type': 'composable',
    'disc': {'type': 'analytic_disk', 'all_params': Fixed(DEFAULT)},
    'torus': {'type': 'skirtor', 'all_params': FREE},
    'nlr': {'type': 'cue', 'all_params': Fixed(DEFAULT)},  # fixed at registry defaults
    'blr': {'type': 'cue'},  # uses defaults
    # feii and atten omitted (OFF)
}
model = SEDModel.build(ssp_data=ssp, observation=obs, agn=agn)
```

---

## V. Error messages tour

The grammar is designed to fail **loudly** with actionable guidance.

### Missing required setting

```text
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={'type': 'dpl'})
# ParameterError: redshift is required. Specify one of:
#   - redshift=Fixed(z) for a known redshift
#   - redshift=Uniform(lo, hi) for a photo-z fit
#   - redshift=<any Distribution> for other priors
```

### Missing structural key (when required)

```text
model = SEDModel.build(ssp_data=ssp, observation=obs, dust_attenuation={'type': 'two_component', 'all_params': Fixed(DEFAULT)}, redshift=Fixed(0.1))
# ParameterError: dust_attenuation type 'two_component' requires 'law' or ('law_bc' and 'law_diff').
# See tengri.list_dust_laws() for options.
```

### Mismatched law pair

```text
dust_attenuation={'type': 'two_component', 'law_bc': 'calzetti'}
# ParameterError: On dust_attenuation type 'two_component', 'law_bc' requires 'law_diff'.
# Specify both, or use 'law' for both screens.
```

### Unknown key with suggestion

```text
sfh={'type': 'dpl', 'beta_': Uniform(1, 3)}
# ParameterError: Unknown key 'beta_' in sfh group.
# Did you mean: beta?
```

### Retired spelling

```text
stellar={'type': 'chabrier'}
# ValueError: The 'stellar' kwarg is retired. Use 'met=' instead to select metallicity mode.
# met={'type': 'table'} for the same behavior.
```

### Retired wildcard

```text
sfh={'type': 'dpl', '*': FREE}
# ValueError: The wildcard key '*' has been retired; the wildcard is spelled
# 'all_params' (or its synonym 'other_params'). Write {'all_params': FREE}
# instead of {'*': FREE}.
```

### No-op wildcard

```python
model = SEDModel.build(ssp_data=ssp, observation=obs, radio={'type': 'sfonly', 'all_params': FREE}, redshift=Fixed(0.1))
# ParameterError: Cannot set 'all_params': FREE on radio — it has no free parameters.
# Pass explicit priors instead: radio={'type': 'sfonly', 'q10': Uniform(...)}
```

### IGM without redshift

The model will build, but IGM has no effect:

```python
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={'type': 'dpl'}, igm={'type': 'inoue'}, redshift=Uniform(0, 1))
# ✓ Builds. IGM model is loaded but applied only when redshift is known (during prediction).
```

---

## See also


- [Model grammar philosophy](model_grammar_design.md): design decisions behind the grammar
- [Per-component parameter reference](components.md): every free/default parameter by component
- [Configuration philosophy](model_grammar_design.md): why the grammar is structured this way
