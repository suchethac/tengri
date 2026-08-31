# Model configuration reference

<!-- COUPLING NOTE: tools/check_doc_grammar_keys.py parses this file.
     Structure is strict: section headings "### Domain: `key`", marker "**Structural keys:**",
     and bullets "- `'key'` —". Changes to markdown format require updating the guard's parser. -->

This is the definitive guide to the nested-dict grammar for building a `SEDModel`. It covers universal grammar semantics, per-domain configuration, round-trip serialization, and common patterns.

## I. Universal grammar semantics

The nested-dict grammar for model configuration (when using SEDModel.build) accepts one dict per physics group, each declaring what physics variant to use, which parameters are free, and per-parameter overrides.

### Three kinds of keys

Every group dict contains three kinds of keys:

**1. Structural keys** — select the model variant or configure the group's behavior. The key `'type'` is universal; some groups have additional structural keys (e.g., `'law'` for dust, `'norm'` for AGN). Structural keys are non-parameter settings.

```python
# 'type' selects the SFH model, 'age_kernel' configures integration
sfh={'type': 'dpl', 'age_kernel': 'cic'}
```

**2. The wildcard keys `'all_params'` and `'other_params'`** — exact synonyms that set the free/fixed status for every parameter in the group not explicitly overridden. Each accepts `FREE` (defer to the registry's default prior) or `Fixed(DEFAULT)` (pin at the registry default value) — never a concrete `Fixed(v)` (one literal value cannot apply across every parameter in the group) and never an arbitrary `Distribution`. Giving both spellings in the same dict raises. The retired `'*'` synonym still raises `ValueError` naming both accepted spellings.

Pick the spelling that reads best for the shape of the dict: `'all_params'` when the wildcard is the group's only directive, `'other_params'` written **last**, after explicit per-parameter entries, where it reads as "the others." `to_groups()` and the builder factories always emit by this convention:

```python
# Sole directive: 'all_params' reads best
dust_attenuation={'type': 'two_component', 'law': 'calzetti', 'all_params': Fixed(DEFAULT)}

# Mixed: explicit overrides first, 'other_params' last
dust_attenuation={'type': 'two_component', 'law': 'calzetti', 'tau_bc': 0.5, 'other_params': Fixed(DEFAULT)}
# NOT (legal, but not the taught style): {..., 'all_params': Fixed(DEFAULT), 'tau_bc': 0.5}
```

**3. Parameter keys** — bare parameter names or full prefixed names that override the wildcard or default. Short names are auto-prefixed by the group context.

```python
# All three mean the same: set tau_bc to 0.5
dust_attenuation={'type': 'two_component', 'tau_bc': 0.5}
dust_attenuation={'type': 'two_component', 'dust_tau_bc': 0.5}  # full prefixed name
dust_attenuation={'type': 'two_component', 'tau_bc': 0.5}       # short name (preferred)
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

`'all_params': FREE` on a group whose parameters default to `Fixed(DEFAULT)` is valid and cascades. However, on a group where **all** parameters are inherently fixed (e.g., `radio` without sub-blocks configured to use a free model, or `shock` with only fixed components), `'all_params': FREE` **raises an error** with an example of how to set explicit priors instead:

```python
# WRONG: radio is free-param free
model = SEDModel.build(ssp_data=ssp, observation=obs, radio={'type': 'sfonly', 'all_params': FREE})
# Raises: "Cannot set all_params=FREE on radio — it has no free parameters."
# "  Pass explicit priors instead: radio={'type': 'sfonly', 'q10': Uniform(...)}"

# CORRECT: explicit prior on the one available parameter
model = SEDModel.build(ssp_data=ssp, observation=obs, radio={'type': 'sfonly', 'q10': Uniform(-0.5, 0.5)})
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
- `'type'` — SFH model (`'dpl'`, `'delayed_tau'`, `'lognorm'`, `'field'`, etc.). Menu: `tengri.list_sfh_models()`.
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'age_kernel'` — Integration method: `'cic'` (default, cloud-in-cell) or `'dsps'` (histogram). See the model grammar design guide for performance details.
- `'bin_edges_gyr'` — Non-parametric bin edges (Gyr). Only for `type='histogram'` or similar.
- `'field_centering'` — Field draw centering ('none' or 'mean'). Only for `type='field'`.

**Minimal example:**
```python
sfh={'type': 'dpl', 'all_params': FREE, 'beta': Uniform(1, 3), 'age_kernel': 'cic'}
```

**Gotchas:**
- `'age_kernel': 'dsps'` is **not** a performance knob — it's 13% slower. Use `'cic'` (default) unless you need DSPS cross-code parity.
- A field SFH requires `'age_kernel': 'dsps'` and rejects `'age_kernel': 'cic'`.
- Default `age_kernel` auto-selects: `'cic'` for parametric SFH, `'dsps'` for field.


### Metallicity: `met`

**Structural keys:**
- `'type'` — Metallicity model (`'table'` for per-age SSP indexing, `'ramp'` for a linear Z(t) history, etc.). Menu: `tengri.list_metallicity_modes()`.
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).

**Minimal example:**
```python
met={'type': 'table'}  # all_params defaults to Fixed(DEFAULT)
met={'type': 'ramp', 'logzsol_0': Fixed(-0.3), 'logzsol_1': Free}  # two-knot ramp
```

**Gotchas:**
- Metallicity is **separate from star formation**. `met=` selects SSP templates; `neb={'logZ_gas': ...}` drives nebular emission independently.
- Default `met_logzsol = 0.0` (solar). The SSP grid uses absolute `log10(Z)` internally, with a Zsun offset (Asplund 2009).
- A tabulated (per-SSP-age) `met=` beside a non-tabulated `sfh` warns but does not raise.


### Dust attenuation: `dust_attenuation`

**Structural keys:**
- `'type'` — Architecture: `'single_component'` (one dust screen) or `'two_component'` (birth-cloud + diffuse). Default varies by law.
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'law'` — Attenuation law: `'calzetti'`, `'ccm89'`, `'mw_rv31'`, `'kext00'`, etc. Menu: `tengri.list_dust_laws()`.
  - On `'single_component'`: one law for the entire dust.
  - On `'two_component'`: `'law'` applies to both screens. Override per-screen with `'law_bc'` and `'law_diff'`.
  - On `'wg00'` (Willis & Graves 2000 screen): use structural keys like `'dust_curve'`, `'geometry'`, `'structure'` instead of a law name.
- `'law_bc'` — Birth-cloud attenuation law (two-component only). Required with `'law_diff'` when not using shared `'law'`.
- `'law_diff'` — Diffuse dust attenuation law (two-component only). Required with `'law_bc'` when not using shared `'law'`.
- `'law_neb'` — Nebular dust law (reddens only birth-cloud continuum).
- `'dust_curve'` — WG00 dust curve selector (only for `type='wg00'`).
- `'geometry'` — WG00 geometry ('slab', 'sphere', etc.) (only for `type='wg00'`).
- `'structure'` — WG00 structure ('clumpy', 'homogeneous', etc.) (only for `type='wg00'`).
- `'slope_bc'`, `'slope_diff'`, `'slope_neb'` — Per-screen law-parameter overrides (two-component only).
- `'bump_strength_bc'`, `'bump_strength_diff'`, `'bump_strength_neb'` — Per-screen bump-strength overrides (two-component only).
- `'Rv_bc'`, `'Rv_diff'`, `'Rv_neb'` — Per-screen RV overrides (two-component only).
- `'delta_bc'`, `'delta_diff'`, `'delta_neb'` — Per-screen delta overrides (two-component only).
- `'lyman_cutoff'` — Zero attenuation below 912 Å (Lyman limit). Two-component only.
- `'lyc_absorb_all'` — Absorb all ionizing photons (FSPS/CIGALE style) vs young-only (default). Two-component only.
- `'eb_include_lyc'` — Include ionizing luminosity in the dust energy-balance integral (FSPS/Prospector parity). Default false.

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
```

**Gotchas:**
- Dust attenuation and dust emission are **two separate peer groups**, not nested. The retired `dust={'attenuation': {...}, 'emission': {...}}` form raises.
- Two-component law-pairing rule: if you name one of `'law_bc'`/`'law_diff'`, you must name both (or use a shared `'law'` for both).
- Parameters like `'slope'`, `'bump_strength'`, `'Rv'`, `'delta'` are set per-screen on two-component (`'slope_bc'`, `'slope_diff'`, etc.). On single-component, just `'slope'`.
- The `'neb'` channel (`'law_neb'`, `'slope_neb'`, etc.) reddens **only the nebular birth-cloud continuum**, not the young stars. Used when nebular emission is routed through a different dust screen.


### Dust emission: `dust_emission`

**Structural keys:**
- `'type'` — Emission model: `'dale2014'`, `'draine2016'`, etc. Menu: `tengri.list_dust_emission_models()`.
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'spinning_dust'` — Include small spinning dust grains (default: auto from type).
- `'f_cnm'` — Cold neutral medium fraction (parametrization-dependent).
- `'eta_balance'` — Energy-balance coupling: `Fixed(1.0)` (default, strict balance `L_IR = eta * L_absorbed`), or `Uniform(...)` to leave it free.

**Minimal example:**
```python
dust_emission={'type': 'dale2014', 'eta_balance': Fixed(1.0), 'other_params': Fixed(DEFAULT)}
```

**Gotchas:**
- Energy balance: `eta_balance` defaults to `Fixed(1.0)`, which enforces `L_IR = L_absorbed`. Setting it free or to a constant ≠ 1 decouples IR and absorption.
- Missing dust_emission (or `{'type': 'none'}`) is valid and common for UV-only work.


### Nebular emission: `neb`

**Structural keys:**
- `'type'` — Backend: `'cue'` (Cue, default), `'cloudy'` (CLOUDY, slower, higher fidelity), `'cb19'` (Charlot & Bruzual 2019), `'mappings'` or `'mappings_agn'` (MAPPINGS V stellar and AGN; **both backends are registered as experimental; both refuse loudly pending data rehabilitation** (#2082): stellar grid is 51.2% NaN, AGN backend lacks protocol surface), or `'none'` (off). Menu: `tengri.list_nebular_backends()`.
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'full_catalog'` — Line catalog scope: bool, default backend-dependent.
- `'grid'` — For CLOUDY: grid specification (dict with keys like `'logz'`, `'logU'`, etc.).

**Minimal example:**
```python
neb={'type': 'cue', 'logZ_gas': -0.3, 'other_params': Fixed(DEFAULT)}
neb={'type': 'cloudy', 'grid': {'logz': [-2, -1, 0], 'logU': [-3, -2, -1]}}
```

**Gotchas:**
- Nebular metallicity (`'neb_logZ_gas'` or short `'logZ_gas'` in the `neb` dict) is **independent** from stellar metallicity (`'met='`).
- Default `neb_logZ_gas = -0.3` (solar). It is **not automatically inherited** from the stellar metallicity, even if tabulated.
- Nebular emission is **additive** to stellar continuum; it composites with dust and shock when both are present.


### Shock emission: `shock`

**Structural keys:**
- `'type'` — Shock backend: `'mappings'` (MAPPINGS V, default) or `'none'` (off).
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'norm'` — Normalization: `'frac'` (scales the galaxy Hα), `'lhalpha'` (absolute Hα luminosity, decoupled from SFR), or `'component'` (explicit component label).
- `'abundance'` — Abundance mode: `'solar'`, `'lmc'`, etc.
- `'component'` — Component label for compartmentalization (advanced).

**Minimal example:**
```python
shock={'type': 'mappings', 'norm': 'frac', 'frac': Uniform(0, 1)}
shock={'type': 'mappings', 'norm': 'lhalpha', 'lhalpha': 10**42}  # erg/s
```

**Gotchas:**
- `'all_params': FREE` on `shock` **raises** if no free-parameter models are configured — use explicit priors instead.
- Shock and nebular emission compose (both can be on). The shock norms apply independently.
- Default `shock_abundance = 'solar'`. No abundance parameter by default.


### IGM absorption: `igm`

**Structural keys:**
- `'type'` — IGM model: `'inoue'` (Inoue+2014, default), `'madau'` (Madau+1995), `'meiksin06'` (Meiksin 2006), `'none'` (off).
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'patchy'` — Picket-fence vs smooth IGM: bool, model-dependent default.
- `'dla'` — Damped Lyman alpha: omit for no DLA, or provide `{'type': ...}` for DLA models (e.g., `{'type': 'dla_lookback'}` to evolve DLA properties with redshift).

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
- `'type'` — Radio model: `'sfonly'` (star-formation only, default), `'agn'` (AGN only), `'sf_agn'` (both), `'none'` (off).
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'sf'` — Star-formation radio sub-block: `{'type': ...}` to customize.
- `'agn'` — AGN radio sub-block: `{'type': ...}` to customize.

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
- `'type'` — X-ray model: `'yang22'` (Yang+2022), `'lehmer'` (Lehmer+2022), `'none'` (off).
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).

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
- `'type'` — AGN mode: `'composable'` (six independent emitters), `'legacy'` (single monolithic AGN), or `'none'` (off).
- `'all_params'` — Wildcard: sets every parameter in the group to `FREE` or `Fixed(DEFAULT)`. Exact synonym: `'other_params'` (reads best written last, after explicit per-param entries). Not `'*'` (retired).
- `'norm'` — Across-component normalization: `'cigale_joint'` (default, CIGALE-style energy conservation across disc/torus/polar) or `'independent'` (each component on its own scale).
- `'disc'` — AGN accretion disk sub-block (with `'type'`, `'all_params'`, parameters).
- `'torus'` — Infrared-obscured torus sub-block (with `'type'`, `'all_params'`, parameters).
- `'nlr'` — Narrow-line region sub-block (with `'type'`, `'all_params'`, parameters).
- `'blr'` — Broad-line region sub-block (with `'type'`, `'all_params'`, parameters).
- `'feii'` — Iron emission sub-block (with `'type'`, `'all_params'`, parameters).
- `'atten'` — AGN-specific attenuation sub-block (with `'type'`, `'all_params'`, parameters, and `'law'` for dust law selection).
- `'lines'` — Deprecated: expands to `'nlr'` + `'blr'`.

**Minimal example:**
```python
agn={
    'type': 'composable',
    'disc': {'type': 'analytic_disk', 'all_params': Fixed(DEFAULT)},
    'torus': {'type': 'skirtor', 'all_params': FREE},
    'nlr': {'type': 'cue', 'all_params': Fixed(DEFAULT)},
    'norm': 'cigale_joint',
}

# Or the legacy form (one-block AGN):
agn={'type': 'legacy', 'all_params': Fixed(DEFAULT)}
```

**Gotchas:**
- On `'composable'`, all six sub-blocks are **optional**. Omitting one deactivates it (e.g., no `'disc'` means no direct accretion disk emission).
- Each sub-block follows the same grammar (type, all_params, parameters) as a top-level group.
- `'norm': 'cigale_joint'` ties disc/torus/polar normalization to a single AGN-power reference; `'norm': 'independent'` lets each float freely. 
- AGN sub-block names (`'disc'`, `'torus'`, etc.) are **single**, not plural. `'discs'` raises.


### Foreground extinction: `foreground`

**Structural keys:**
- `'ebmv_mw'` — Milky Way E(B-V) reddening (mag). Typically 0.01–0.2.
- `'law'` — Dust law: `'mw_rv31'` (Fitzpatrick 1999, default), `'ccm89'` (CCM89), etc.
- `'rv'` — Dust RV parameter override (law-dependent).

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
- `Fixed(z)` — Known redshift (e.g., `Fixed(0.05)`).
- `Uniform(z_min, z_max)` — Photo-z prior (e.g., `Uniform(0.0, 2.0)`).
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

`to_groups()` never expands a wildcard into individual per-parameter priors. A mixed group (explicit overrides plus a wildcard) round-trips with the overrides first and the wildcard collapsed to `'other_params'` last — the same convention the grammar teaches for hand-written dicts:

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


- [Model grammar philosophy](model_grammar_design.md) — design decisions behind the grammar
- [Per-component parameter reference](components.md) — every free/default parameter by component
- [Configuration philosophy](model_grammar_design.md) — why the grammar is structured this way
