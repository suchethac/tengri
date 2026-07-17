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

The canonical way to extend tengri is by subclassing `SEDModelComponent` — a base
class that handles registration, prior management, and integration with `SEDModel.build()`.

Any component that has free parameters, a wavelength-dependent transform, and
(optionally) a pre-computed library — custom SFH parametrizations, dust laws,
nebular models, AGN templates — follows the same one-file pattern.

### Example: Custom modified blackbody dust emission

```python
import jax.numpy as jnp
from tengri.components.sed_model_component import SEDModelComponent
from tengri import Uniform

class CustomBlackbody(SEDModelComponent):
    name = "custom_blackbody"         # registry key for SEDModel.build()
    parameter_prefix = "custom_bb_"   # auto-prefixes free parameters

    # Class-level priors auto-discovered by __init_subclass__
    T = Uniform(20.0, 80.0, "temperature", units="K")
    beta = Uniform(1.0, 3.0, "emissivity index", units="")

    # Declare inputs from upstream and outputs published downstream
    inputs = {"L_absorbed": "erg/s"}
    outputs = {"L_ir": "erg/s"}

    def load(self, wave):
        """Optional: load pre-computed library. Return None if not needed."""
        return None

    def predict(self, p, sed_in, wave, *, L_absorbed):
        """Physics: transform SED and publish derived quantities.

        Parameters
        ----------
        p : dict
            Parameter dict with keys like "T", "beta" (prefix stripped).
        sed_in : array
            Rest-frame L_ν from upstream [erg/s/Hz]; zeros if first component.
        wave : array
            Rest-frame wavelength grid [Angstrom].
        **inputs : dict
            Keyword args from upstream (auto-supplied).

        Returns
        -------
        sed_out : array
            New SED to pass downstream [erg/s/Hz].
        published : dict
            Derived quantities matching the `outputs` keys.
        """
        # Implement the physics: modified blackbody emission
        # scaled by absorbed luminosity
        ...
        return sed_out, {"L_ir": jnp.sum(sed_out)}
```

**Key points:**

- `name` is the key you use in `SEDModel.build(dust={'type': 'custom_blackbody'})`.
- Class-level `Uniform`, `LogNormal`, etc. priors auto-flow to inference.
- `inputs`/`outputs` dicts declare what your component consumes and publishes.
- `predict(p, sed_in, wave, **inputs)` is the physics — pure JAX, fully differentiable.
- `__init_subclass__` auto-registers the component with `_REGISTRY` (no extra boilerplate).

**Place your component in:**
- Custom dust laws: `src/tengri/components/dust/`
- Custom nebular models: `src/tengri/components/nebular/`
- Custom AGN templates: `src/tengri/components/agn/`
- Or anywhere else — as long as it's imported before `SEDModel.build()` runs.

**Verify it works:**

```python
from tengri import SEDModel, load_ssp_data, Observation, Photometry, Fixed

# After your component is defined and imported:
model = SEDModel.build(
    ssp_data=load_ssp_data("data/fsps_prsc_miles_chabrier.h5"),
    observation=Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g"])),
    dust={'type': 'custom_blackbody'},  # ← your component is available
    redshift=Fixed(0.1),
)
print(f"Free params: {model.spec.free_params}")
```

## Adding a custom dust law

Dust attenuation laws follow the `SEDModelComponent` pattern. A dust law component
computes attenuation (τ_λ) and optionally emits IR:

```python
import jax.numpy as jnp
from tengri.components.sed_model_component import SEDModelComponent
from tengri import Uniform

class MyDustLaw(SEDModelComponent):
    name = "my_dust_law"
    parameter_prefix = "dust_"

    tau_v = Uniform(0.0, 5.0, "V-band optical depth", units="")
    delta = Uniform(0.0, 2.0, "power-law index", units="")

    inputs = {}  # dust laws don't consume upstream quantities
    outputs = {}

    def predict(self, p, sed_in, wave, **inputs):
        """Compute attenuation and apply to SED."""
        # Power-law attenuation: τ_λ = τ_V * (λ / 0.55 μm)^(-delta)
        wave_micron = wave / 1e4
        tau_lambda = p["tau_v"] * (wave_micron / 0.55) ** (-p["delta"])

        # Attenuate the SED
        sed_out = sed_in * jnp.exp(-tau_lambda)
        return sed_out, {}
```

Register by importing it before `SEDModel.build()`, then use:
`SEDModel.build(..., dust={'type': 'my_dust_law'})`

## Adding a custom nebular model

Nebular emission components follow `SEDModelComponent`. Write a component that
takes the ionizing photon rate (Q_H) from stellar/AGN and gas-phase metallicity,
returning nebular line and continuum emission:

```python
import jax.numpy as jnp
from tengri.components.sed_model_component import SEDModelComponent
from tengri import Uniform

class MyNebularBackend(SEDModelComponent):
    name = "my_nebular"
    parameter_prefix = "neb_"

    logU = Uniform(-4.0, -2.0, "ionization parameter", units="")

    inputs = {"Q_H": "photons/s", "log_met_gas": ""}
    outputs = {"L_nebular": "erg/s"}

    def load(self, wave):
        # Load nebular template grids or emulator weights
        return None

    def predict(self, p, sed_in, wave, *, Q_H, log_met_gas):
        """Return nebular spectrum."""
        # Look up or compute emission given Q_H, metallicity, and logU
        nebular_sed = self._nebular_fn(wave, Q_H, log_met_gas, p["logU"])
        sed_out = sed_in + nebular_sed
        return sed_out, {"L_nebular": jnp.sum(nebular_sed)}
```

Register and use: `SEDModel.build(..., neb={'type': 'my_nebular'})`

## Adding a custom AGN template

AGN components follow `SEDModelComponent`. Load a template library and interpolate:

```python
import jax.numpy as jnp
from tengri.components.sed_model_component import SEDModelComponent
from tengri import Uniform

class MyAGNTemplate(SEDModelComponent):
    name = "my_agn_template"
    parameter_prefix = "agn_"

    log_lbol = Uniform(42.0, 48.0, "bolometric luminosity (log)", units="erg/s")

    inputs = {}
    outputs = {"L_agn": "erg/s"}

    def load(self, wave):
        # Load SED template at build time (not at predict time)
        # Return a dict or object with normalized template(s)
        agn_template = jnp.array([...])  # wavelength-dependent template
        return agn_template

    def predict(self, p, sed_in, wave, **inputs):
        """Scale and add AGN template to SED."""
        # Interpolate template onto model wavelength grid
        template = jnp.interp(wave, self.data_wave, self.data)

        # Scale by luminosity: AGN_sed = template * 10^(log_lbol)
        scale = 10.0 ** p["log_lbol"]
        agn_sed = template * scale

        sed_out = sed_in + agn_sed
        return sed_out, {"L_agn": jnp.sum(agn_sed)}
```

Register and use: `SEDModel.build(..., agn={'type': 'my_agn_template'})`

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
- **Parameter prefixes**: use a short, unique `parameter_prefix` to avoid collisions
  (e.g., `"custom_bb_"` for CustomBlackbody).
- **Documentation**: include a docstring on your component class and the `predict()` method.
- **Testing**: add tests in `tests/components/` or `tests/unit/`. Verify against other codes
  where possible. Run `pytest tests/ -q` before committing.

**References:**

- **Canonical reference:** `docs/dev/sed-model-components.md` — full how-to with three worked examples
- **Architecture:** `docs/adr/0011-sed-model-component-base.md` — design rationale
- **Notebook example:** `notebooks/05_adding_a_model.py` — complete working example
- **Canonical small component:** `src/tengri/components/dust/wg00_model.py` (closed-form attenuation)
- **Canonical library component:** `src/tengri/components/agn/skirtor_model.py` (template library)
