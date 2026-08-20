# Extending tengri

How to plug in custom SFH parametrizations, dust laws, nebular models, and AGN
templates.

:::{note}
A complete worked example with runnable code is in [`05_adding_a_model.py`](https://github.com/suchethac/tengri/blob/main/notebooks/05_adding_a_model.py).
:::

## Architecture overview

tengri's forward model is a pipeline of independent physics modules:

```
Parameters --> SFH --> SPS --> Dust --> Nebular --> AGN --> IGM --> Observation
```

To extend tengri, write a pure JAX function that follows the same interface
and register it with the model.

## Adding a custom component

`SEDModelComponent` is the root base class, but **you rarely subclass it
directly**. Each physics block ships its own base class that already declares
that block's inputs/outputs contract, and subclassing the right one is what
makes your model reachable from the build grammar.

This matters because `SEDModel.build()` does not accept a `type` string merely
because a class registered. Each validator in `parameters/groups.py`
(`_valid_dust_emission_types`, `_valid_nebular_types`, …) derives its accepted
set from a live registry, and most of them apply a *structural* test. For dust
emission the test is "does this component publish `sed_dust_ir`?" — not a name
lookup. A component that registers but fails the structural test is rejected
with `Unknown dust emission type '...'`, which reads like a typo but is a
contract mismatch.

### Example: Custom modified blackbody dust emission

```python
import jax.numpy as jnp
from tengri import Uniform
from tengri.components.dust.emission._component_base import EmissionComponent

class CustomBlackbody(EmissionComponent):
    # Registry key used in
    # SEDModel.build(dust={'emission': {'type': 'custom_blackbody'}})
    name = "custom_blackbody"

    # No `inputs`, `outputs` or `parameter_prefix` here: EmissionComponent
    # already declares optional_inputs={"L_ir": "erg/s"},
    # outputs={"sed_dust_ir": "erg/s/Hz"} and parameter_prefix="dust_".
    # Re-declaring them with names of your own is the most common way to
    # write a component that registers but is never reachable.

    # Class-level priors auto-discovered by __init_subclass__. These reuse
    # the canonical dust-emission parameter names from
    # components/dust/_params.py — see "Parameter names" below.
    T = Uniform(20.0, 80.0, "temperature", units="K")
    beta_ir = Uniform(1.0, 3.0, "emissivity index", units="")

    def load(self, wave):
        """Optional: load pre-computed library. Return None if not needed."""
        return None

    def predict(self, p, sed_in, wave, *, L_ir):
        """Physics: transform SED and publish derived quantities.

        Parameters
        ----------
        p : dict
            Parameter dict with keys "T", "beta_ir" (prefix stripped).
        sed_in : array
            Rest-frame L_ν from upstream [erg/s/Hz]; zeros if first component.
        wave : array
            Rest-frame wavelength grid [Angstrom].
        L_ir : float
            Energy-balance luminosity from the upstream attenuation
            component [erg/s]. The keyword name is the block's contract.

        Returns
        -------
        sed_out : array
            New SED to pass downstream [erg/s/Hz].
        published : dict
            Must contain "sed_dust_ir" — the key the block declares.
        """
        shape = wave ** (-p["beta_ir"]) * planck_nu(wave, p["T"])
        sed = L_ir * shape / trapz_freq(shape, wave)
        return sed_in + sed, {"sed_dust_ir": sed}
```

**Key points:**

- `name` is the key used in
  `SEDModel.build(dust={'emission': {'type': 'custom_blackbody'}})`. Note the
  nesting: `dust={'type': ...}` selects the dust *model*
  (`single_component` / `two_component` / `wg00`), not the emission model.
- Subclass the **block's** base class, not `SEDModelComponent`, and do not
  re-declare `inputs`/`outputs` — that is what the validator tests.
- Class-level `Uniform`, `LogNormal`, etc. priors flow to inference, provided
  the parameter names already exist (see below).
- `predict(p, sed_in, wave, **inputs)` is the physics — pure JAX, fully
  differentiable. Guard `expm1`-style expressions: an unclipped
  `nu**3 / expm1(x)` overflows at optical wavelengths and returns a `nan`
  gradient. See `components/dust/emission/analytic/_closures.py`.
- `__init_subclass__` auto-registers the component with `_REGISTRY` (no extra
  boilerplate).

**Parameter names.** There are two registries with two different lifetimes:

| Registry | Populated by | Sees runtime-defined classes? |
|---|---|---|
| `_REGISTRY` (dispatch) | `__init_subclass__`, at class-definition time | yes |
| parameter map | static scan of `components/*/_params.py` in the *installed package* (ADR-0008) | no |

So a component defined at runtime (in a notebook or a user script) is
dispatchable but **cannot introduce a new parameter name** — an unknown name is
silently dropped from the build, and `tengri.Parameters(...)` later rejects it
with `Unknown parameter`. To add a genuinely new parameter, add its
`ParamDeclaration` to the block's `_params.py` in a checkout and reinstall.
(`tengri.register_component` looks like the seam for this, but it is vestigial:
`_get_registered_components()` has no live callers — the static scan replaced
it.)

**Place your component in:**
- Custom dust laws: `src/tengri/components/dust/laws/`
- Custom dust emission: `src/tengri/components/dust/emission/`
- Custom nebular models: `src/tengri/components/nebular/`
- Custom AGN templates: `src/tengri/components/agn/`
- Or anywhere else — as long as it's imported before `SEDModel.build()` runs.

**Verify it works:**

```python
from tengri import SEDModel, load_ssp, Observation, Photometry, Fixed, Uniform
from tengri.components.sed_model_component import _REGISTRY
from tengri.parameters.groups import _valid_dust_emission_types

# Registration and acceptance are different questions — check both.
assert "custom_blackbody" in _REGISTRY                      # dispatch
assert "custom_blackbody" in _valid_dust_emission_types()   # validator

model = SEDModel.build(
    ssp_data=load_ssp("ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0"),
    observation=Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g"])),
    dust={
        'type': 'single_component',
        'law': 'calzetti',
        'emission': {'type': 'custom_blackbody',   # ← your component
                     'T': Uniform(20.0, 80.0),
                     'beta_ir': Uniform(1.0, 3.0)},
    },
    redshift=Fixed(0.1),
)
print(f"Free params: {model.spec.free_params}")
# → ['dust_T', 'dust_beta_ir']
```

## Adding a custom dust law

A dust attenuation law is **not** an `SEDModelComponent` — it is a plain
function registered with the `@register_dust_law` decorator. It returns the
attenuation curve `k(λ)` [dimensionless]; the framework applies it, handles the
τ_V scaling, and runs the energy balance. This is a much smaller thing to write
than a component:

```python
from tengri.components.dust.attenuation import register_dust_law

@register_dust_law("my_powerlaw")
def my_powerlaw_dust(wavelength, n_slope=-0.7, **kwargs):
    """Power-law attenuation curve, normalized at 5500 Å."""
    return (wavelength / 5500.0) ** n_slope
```

Import it before `SEDModel.build()`, then select it as a **law**, not a type:

```python
model = SEDModel.build(
    ...,
    dust={'type': 'single_component',   # the dust *model*
          'law': 'my_powerlaw',      # ← your law
          'tau_v': Fixed(0.4)},
)
```

`dust={'type': ...}` accepts only `single_component`, `two_component` or
`wg00` — passing a law name there is rejected. The accepted law names come
from `_valid_dust_laws()`, a direct view of `DUST_LAWS`, which the decorator
populates at import time. The full protocol (accepted signature, expected
return) is documented in `components/dust/_protocol.py`.

## Adding a custom nebular model

Like dust laws, a nebular backend is registered with a decorator rather than
written as an `SEDModelComponent` subclass:

```python
from tengri.components.nebular._models import register_nebular_model

@register_nebular_model("my_nebular", short_doc="...")
def my_nebular_backend(...):
    ...
```

`neb={'type': 'my_nebular'}` is then accepted, because `_valid_nebular_types()`
is a direct view of `NEBULAR_MODELS`, which the decorator populates. Read
`components/nebular/_protocol.py` for the callable's expected signature, and
`components/nebular/baked_in.py` or `cloudy_grid.py` for worked examples.

## Adding a custom AGN template

AGN sub-blocks (disc, torus, BLR, NLR) are registered with
`@register_agn_block` from `components/agn/blocks/_protocol.py`, which populates
`AGN_BLOCKS`. The validator (`_agn_block_types`) derives the accepted names from
that registry, so registration is what makes a block selectable. See the shipped
blocks under `components/agn/blocks/` for the shape.

## Finding the seam for any block

The reliable method, rather than guessing at a shape: open the validator that
would reject your `type` string — `parameters/groups.py::_valid_*_types` — and
read what set it builds its answer from. Every one derives from a live registry
(ADR-0005 / ADR-0008), so that function *is* the specification of what your
extension has to do to be accepted.

| What you want to add | Seam | Lives in |
|---|---|---|
| Dust IR emission model | subclass `EmissionComponent` | `components/dust/emission/` |
| Dust attenuation curve | `@register_dust_law(name)` | `components/dust/attenuation.py` |
| Nebular backend | `@register_nebular_model(name)` | `components/nebular/_models.py` |
| AGN sub-block | `@register_agn_block(...)` | `components/agn/blocks/_protocol.py` |
| SFH | `SFH_REGISTRY` entry | `components/stellar/sfh/registry.py` |
| IGM / radio / X-ray | `register_igm_model` / `register_radio_model` / `register_xray_model` | `components/<block>/_models.py` |

## General guidelines

**For all `SEDModelComponent` subclasses:**

- **Pure JAX**: `predict()` must be fully differentiable and JIT-compatible.
  No Python control flow depending on array values, no numpy, no side effects.
- **Immutable arrays**: use `.at[].set()` instead of in-place modification.
- **Units**: follow tengri conventions:
  - Wavelength: Angstrom
  - Time: years
  - SFR: M_sun/yr
  - Luminosity: erg/s/Hz (SED), erg/s (total)
- **Parameter prefixes**: inherit the block's `parameter_prefix` (e.g. `"dust_"`
  from `EmissionComponent`) rather than inventing a unique one. A unique prefix
  does not avoid collisions so much as guarantee the parameter is absent from
  the param map — see "Parameter names" above.
- **Numerical guards**: `predict()` must be gradient-safe, not just finite.
  Clip exponent arguments and prefer formulations whose denominators cannot
  overflow; an unclipped `nu**3 / expm1(x)` evaluates to a harmless `0` at
  optical wavelengths but differentiates to `nan`, and one `nan` poisons the
  whole gradient. `components/dust/emission/analytic/_closures.py` documents
  the standard treatment.
- **Documentation**: include a docstring on your component class and the `predict()` method.
- **Testing**: add tests in `tests/components/` or `tests/unit/`. Verify against other codes
  where possible. Run `pytest tests/ -q` before committing.

**References:**

- **Canonical reference:** `docs/dev/sed-model-components.md` — full how-to with three worked examples
- **Architecture:** `docs/adr/0011-sed-model-component-base.md` — design rationale
- **Notebook example:** `notebooks/05_adding_a_model.py` — complete working example
- **Canonical small component:** `src/tengri/components/dust/wg00_model.py` (closed-form attenuation)
- **Canonical library component:** `src/tengri/components/agn/skirtor_model.py` (template library)
