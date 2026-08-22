# Model grammar philosophy

The nested-dict grammar for building an SEDModel is built on three core principles: **composition**, **explicitness**, and **orthogonality**. This page explains the reasoning behind the design.

## Design principles

### 1. Composition — one dict per physics block

The grammar organizes the model as a flat collection of independent physics blocks, each with the same structure:

```python
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh={'type': '...', 'all_params': ..., 'param': ...},
    dust_attenuation={'type': '...', 'all_params': ..., 'param': ...},
    neb={'type': '...', 'all_params': ..., 'param': ...},
    redshift=Fixed(0.1),
)
```

**Why?** Each block is a unit of configuration. The grammar scales: adding a new physics component (e.g., a new AGN model) requires only registering one class and showing up in one dict. No grammar edits, no menu consolidation.

**Nesting for composition:** Sub-blocks (like AGN's six emitters, or the DLA inside IGM) nest the same structure one level deeper:

```python
agn={'type': 'composable', 'torus': {'type': 'skirtor', 'all_params': FREE}}
igm={'type': 'inoue', 'dla': {'type': 'dla_lookback'}}
```

The six AGN sub-blocks are **mutually independent**. Omitting `'disc'` deactivates it; including it with a type activates it. This beats a flat design (`agn_disc_on=True/False`) because it co-locates the disc's type and its parameters under one `'disc'` dict.

### 2. Explicitness — type activates, no silent defaults

**Activation rule:** An optional physics block is **OFF by default**. To turn it ON, provide its dict with a `'type'`:

```python
# IGM is OFF
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={'type': 'dpl'}, redshift=Fixed(0.1))

# IGM is ON
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={'type': 'dpl'}, igm={'type': 'inoue'}, redshift=Fixed(0.1))
```

No `apply_igm=True` flag. No magic ENV variable. A reader sees `igm={'type': ...}` and knows: IGM is on. A reader sees no `igm=` and knows: IGM is off.

**Error on ambiguity:** A group dict with keys but no `'type'` raises instead of silently using a default. This catches typos:

```text
neb={'logz_gas': -0.3}  # Forgot 'type'
# ParameterError: neb dict provided but 'type' is missing.
# Add a type: neb={'type': 'cue', ...} or omit the dict entirely.
```

**Wildcard never silent:** `'all_params': FREE` on a group with zero free parameters raises instead of silently no-opping. This catches configuration errors:

```text
radio={'type': 'sfonly', 'all_params': FREE}
# ParameterError: 'all_params': FREE has no effect on radio (no free parameters).
# Use explicit priors instead: radio={'q10': Uniform(...)}
```

### 3. Orthogonality — separate knobs for structure and physics

The grammar distinguishes **structural keys** (which variant, how it integrates) from **parameter keys** (what values):

```python
# Structural: 'type', 'law', 'norm', 'age_kernel', etc.
# Physics: parameter names that resolve to class attributes
sfh={'type': 'dpl', 'age_kernel': 'cic', 'beta': Uniform(...), 'alpha': ...}
```

- Structural keys configure **how** the component works (e.g., `'law'` selects a dust law, `'norm'` selects AGN normalization policy).
- Parameters configure **what values** the component takes.

This split keeps configuration readable. A user scanning a dict immediately sees structure (the `type:` and configuration choices) before diving into parameter details.

Dust illustrates this cleanly: `'type'` picks single vs two-component, `'law'` picks the law, `'tau_bc'` sets the opacity. A user can reason about each independently.

## Why these choices resolve past gotchas

### Dust: two peer groups, not nested

Dust has two **logically separate** purposes: extinction (damping starlight) and emission (adding infrared). They have different physics, different parameters, different wavelength coverage. The grammar reflects this:

```python
dust_attenuation={'type': 'two_component', 'law': 'calzetti', ...}
dust_emission={'type': 'dale2014', ...}
```

**Not:**

```python
dust={'attenuation': {...}, 'emission': {...}}  # ← retired, raises
```

Why? Nesting dust under one key invited conflating them. A user might ask "what is my dust configuration?" and see one large `dust=` dict, losing the independence. Keeping them parallel — `dust_attenuation=` and `dust_emission=` — makes it clear: these are separate choices. You can have dust attenuation without emission, or vice versa.

### Redshift: required, three spellings

Redshift is not optional. Omitting it raises:

```text
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={'type': 'dpl'})
# ParameterError: redshift is required. Specify one of:
#   - redshift=Fixed(z) for a known redshift
#   - redshift=Uniform(lo, hi) for a photo-z fit
#   - redshift=<any Distribution>
```

**Why require it?** IGM models, precompute tables, and photo-z fits all hinge on redshift. Making it explicit prevents silent wrong answers (like a galaxy silently placed at z=0 when z was meant to be free).

**Three spellings, one structure:** Users work in three modes:
- **Spectroscopy with known z:** `Fixed(z)`
- **Photometry with photo-z:** `Uniform(lo, hi)` or a full distribution
- **Advanced priors:** Any Distribution (e.g., a mixture, or a prior from a previous fit)

All three are `Distribution` instances under the hood. The grammar accepts them uniformly.

### All_params: the only wildcard

The grammar has **one** wildcard: `'all_params'`. The retired `'*'` raises with a clear message:

```text
sfh={'*': FREE}
# TypeError: The '*' wildcard is retired. Use 'all_params' instead.
# sfh={'all_params': FREE, ...}
```

**Why one?** Fewer names = fewer mental models. A reader sees `'all_params'` once and knows what it is. One name, one mental model: the retired `'*'` spelling bought nothing but a second way to write the same thing.

### No nested wildcard shortcuts

The grammar does **not** support shorthands like `'all_params': {'sfh': FREE, 'neb': FIXED}`. Every group is its own dict:

```python
# NOT supported:
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={'type': 'dpl', 'all_params': FREE}, neb={'type': 'cue', 'all_params': FIXED})

# DO this:
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={'all_params': FREE, ...}, neb={'all_params': FIXED, ...}, redshift=Fixed(0.1))
```

**Why?** Nesting wildcards invites ambiguity: does `'all_params': FREE` apply to all params in all groups? Just the explicit ones? Keeping wildcards local — inside each group — makes scope crystal clear.

### Law explicitness on dust attenuation

Dust attenuation requires you to name the law. No "default law" mode:

```text
dust_attenuation={'type': 'two_component', 'all_params': FIXED}
# ParameterError: dust_attenuation type 'two_component' requires 'law' or ('law_bc' and 'law_diff').
# See tengri.list_dust_laws() for options.
```

**Why?** "The law" is not optional. A reader sees `'law': 'calzetti'` and knows exactly what attenuation curve is applied. A silent default (like assuming Calzetti) hides a choice that should be visible in the config.

On a two-component attenuation, you must provide **either** a single law for both screens or name both `law_bc` and `law_diff`. You cannot give one without the other — that's a configuration error, not a valid partial spec.

- **No silent no-ops.** If `'all_params': FREE` has no effect, it raises.
- **No structural precedence.** A parameter named in the dict **always** overrides the wildcard, which overrides the default. No special cases.
- **No per-module configuration files.** The grammar is Python-in, dict-out. Serialization to YAML/JSON is a planned feature (#75) but not the canonical representation.
- **No "apply" flags.** Physics is activated by presence: omit the dict and it is OFF; provide it (with or without `'type'`) and it is ON. For optional groups without an explicit `'type'`, a documented default variant is used (e.g., igm defaults to 'inoue14', dust_attenuation defaults to 'two_component' if present). No `apply_neb=True` aside from the `neb=` dict.

These omissions are **intentional**. They keep the grammar simple and predictable.

## The price of explicitness

Explicitness costs verbosity. A recipe like `star_forming_photometry()` hides the boilerplate:

```python
# With the recipe
model = SEDModel.build(ssp_data=ssp, observation=obs, **recipes.star_forming_photometry())

# Without (hypothetically)
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh={'type': 'dpl', 'all_params': FIXED, 'alpha': ..., 'tau_gyr': ..., ...},
    met={'type': 'table'},
    dust_attenuation={'type': 'two_component', 'law': 'calzetti', 'all_params': FIXED, ...},
    dust_emission={'type': 'dale2014', 'all_params': FIXED, ...},
    neb={'type': 'cue', 'all_params': FIXED, ...},
    ...
)
```

Recipes are **the answer** to this verbosity. A recipe is a pre-tuned dict (from `tengri.recipes.*`) that captures a common case. A user starts with a recipe, tweaks as needed. The recipe hides boilerplate; the grammar stays explicit.

This is the right tradeoff: common cases are short (one recipe call), uncommon cases are verbose but clear.

## Summary

The grammar is built to make configuration:
1. **Composable** — one dict per block, no coupling.
2. **Explicit** — types activate, no silent defaults, wildcards detect no-ops.
3. **Orthogonal** — structure and physics are separate concerns.
4. **Readable** — a user can scan a model dict and see exactly what it does.
5. **Extensible** — adding a new component or variant requires no grammar edits.

See the [configuration reference](model_configuration.md) for the detailed syntax and the configuration reference for usage details.
