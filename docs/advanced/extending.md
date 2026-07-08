# Extending tengri

How to plug in custom SFH parametrizations, dust laws, nebular models, and AGN
templates.

:::{note}
Worked examples with runnable code are in [`12_extending_tengri.py`](https://github.com/suchethac/tengri/blob/main/notebooks/12_extending_tengri.py).
:::

## Architecture overview

tengri's forward model is a pipeline of independent physics modules:

```
Parameters --> SFH --> SPS --> Dust --> Nebular --> AGN --> IGM --> Observation
```

To extend tengri, write a pure JAX function that follows the same interface
and register it with the model.

## Adding a custom SFH parametrization

SFH modules live in `src/tengri/models/sfh/`. A custom SFH must return SFR as
a function of lookback time:

```python
import jax.numpy as jnp

def my_custom_sfh(t_lookback, params):
    """Compute SFR(t) for a custom parametrization.

    Parameters
    ----------
    t_lookback : array
        Lookback time grid in years.
    params : dict
        Must include your custom parameters.

    Returns
    -------
    sfr : array
        Star formation rate in Msun/yr, same shape as t_lookback.
    """
    # Example: exponentially declining with a burst
    tau = params["tau_sfh"]
    burst_amp = params["burst_amplitude"]
    burst_time = params["burst_time"]
    burst_width = params["burst_width"]

    sfr_smooth = jnp.exp(-t_lookback / tau)
    burst = burst_amp * jnp.exp(-0.5 * ((t_lookback - burst_time) / burst_width) ** 2)
    return sfr_smooth + burst
```

**Requirements:**
- Pure JAX (no numpy, no Python control flow that depends on array values).
- Returns an array of the same shape as `t_lookback`.
- All parameters accessed from the `params` dict.

Register your custom parameters in `Parameters`:

```python
from tengri import Parameters, Uniform

spec = Parameters(
    sfh="custom",
    custom_sfh_fn=my_custom_sfh,
    custom_params={
        "tau_sfh": Uniform(1e8, 1e10),
        "burst_amplitude": Uniform(0.0, 100.0),
        "burst_time": Uniform(1e8, 1e10),
        "burst_width": Uniform(1e7, 1e9),
    },
)
```

## Adding a custom dust law

Dust attenuation modules live in `src/tengri/models/dust/`. A custom dust law
takes wavelength and parameters, returning the optical depth as a function of
wavelength:

```python
def my_dust_law(wave_aa, params):
    """Custom attenuation curve.

    Parameters
    ----------
    wave_aa : array
        Wavelength in Angstrom.
    params : dict
        Must include dust parameters.

    Returns
    -------
    tau_lambda : array
        Optical depth at each wavelength.
    """
    tau_v = params["tau_v"]
    delta = params["dust_delta"]
    # Example: power-law modification of Calzetti
    wave_micron = wave_aa / 1e4
    return tau_v * (wave_micron / 0.55) ** delta
```

The existing two-component model (`attenuation.py`) supports `law_bc="power_law"`
and several built-in curves. If your law fits this interface, you can pass it
directly.

## Adding a custom nebular model

Nebular emission modules live in `src/tengri/models/nebular/`. Three backends
are available: `BakedIn` (simple scaling), `CLOUDY` (grid interpolation), and
`Cue` (neural emulator). To add a new backend:

1. Write a function that takes ionizing photon rate Q(H) and gas-phase
   metallicity, returning an emission spectrum.
2. Register it as a nebular backend in `Parameters`.

The key interface requirement is that the function must be JIT-compatible
and differentiable with respect to its parameters.

## Registering new AGN templates

AGN models live in `src/tengri/models/agn/`. To add a custom AGN template:

```python
# Load your template (e.g., from a file)
import jax.numpy as jnp

agn_wave = ...  # wavelength grid in Angstrom
agn_flux = ...  # flux template

# The AGN module interpolates onto the model wavelength grid
# and scales by a luminosity parameter
```

AGN templates are additive --- they are summed with the stellar + nebular
continuum before dust attenuation (for Type 1) or after (for Type 2/torus).

## General guidelines

- **Pure JAX**: all custom functions must be compatible with `jax.jit` and
  `jax.grad`. No Python side effects, no numpy, no conditionals on traced values.
- **Immutable arrays**: use `.at[].set()` instead of in-place modification.
- **Units**: time in years, wavelength in Angstrom, SFR in Msun/yr.
- **Testing**: add tests in `tests/unit/` for any new module. Run `pytest tests/ -q`
  and `ruff check src/` before committing.

See [`12_extending_tengri.py`](https://github.com/suchethac/tengri/blob/main/notebooks/12_extending_tengri.py) for a full extending walk-through with fitting and validation.
