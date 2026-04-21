# Extending tengri

How to add custom physics modules to tengri. The modular architecture makes it
straightforward to implement new SFH parametrizations, dust laws, nebular models,
and inference backends.

Target audience: astronomers with Python experience who are new to this codebase.

---

## How new components plug into the forward model

Understanding this before writing code saves a lot of confusion.

### The single wiring point: `emission_helpers.py`

`src/tengri/forward/emission_helpers.py` is the **single source of truth** for all
emission physics. Both the non-fused pipeline (`forward/pipeline.py`) and the fused
JIT kernel (`forward/_kernels/`) call these same pure functions. If you add a new
component, add a function here — both execution paths inherit it automatically.

The existing helpers are:
```
nebular_emission(backend, weights, ssp_wave, ...)   → erg/s/Hz
attenuate_emission(sed, wave_aa, tau_v, ...)         → erg/s/Hz (attenuated) + L_absorbed
agn_emission(wave_aa, agn_log_lbol, agn_config, ...) → erg/s/Hz
dust_ir_emission(wave_aa, L_absorbed, u_min, ...)    → erg/s/Hz
radio_emission(wave_aa, sfr, agn_log_lbol, ...)      → erg/s/Hz
igm_absorption(wave_obs, z, sed)                     → erg/s/Hz (absorbed)
```

To add a new component:
1. Write your pure JAX function in the appropriate `components/` sub-module
2. Add a thin helper `my_component_emission(...)` in `emission_helpers.py` that calls it
3. Wire the helper call into `src/tengri/forward/sed_model.py` where the SED is assembled
4. Both code paths (fused + non-fused) pick it up with no further changes

### The precompute protocol: when to use it

**When it matters:** Any component that involves convolving a SED template with filter
curves at inference time should implement the precompute protocol. The forward model
can cache these filter-convolved templates at build time, reducing the MCMC inner loop
to a cheap 1D/2D table lookup instead of a full wavelength integral.

**Rule of thumb:**
| Component type | Precompute useful? | Why |
|---|---|---|
| SFH parametrization | No | Time-domain, no filter convolution |
| Dust attenuation | No | Fast analytic law, no templates |
| **Dust emission templates** | **Yes** | Grid over q_PAH, U_min; expensive |
| **AGN templates (K&D, SKIRTOR)** | **Yes** | Large template grids |
| **Nebular grids (CLOUDY)** | **Yes** | Grid over logZ, logU |
| Nebular analytic (CUE) | Partial | CUE has its own fast path |
| IGM, radio, X-ray | No | Cheap analytic formulae |
| New inference backend | N/A | Not a physics component |

**How to implement it:**

Create `src/tengri/components/<your_component>/<name>_precompute.py` and define:

```python
# The parameter axes of your precomputed grid
AXIS_PARAMS: tuple[str, ...] = ("my_param_a", "my_param_b")

def precompute(filter_waves, filter_trans, redshift, parameters, **kwargs):
    """Build preintegrated grid at model-build time (runs once, outside JIT).

    Steps:
    1. Load / build your SED templates
    2. Build axis grids from AXIS_PARAMS
    3. Call tengri.forward.precompute.grid.preintegrate_grid(...)
    4. Auto-collapse fixed axes via slice_fixed_axes(...)
    5. Return a PreintegratedGrid (or your own dataclass)
    """
    ...

def build_lookup(preint, **kwargs):
    """Return a JIT-compiled callable: (scale, *free_params) → filter photometry.

    Called once after precompute(). The returned function runs inside @jax.jit,
    so it must only use JAX operations. 'scale' is typically L_absorbed or L_bol.
    """
    ...
```

**Auto-collapse:** If the user fixes a parameter (e.g., `dust_qpah = Fixed(0.047)`),
`slice_fixed_axes` automatically collapses that axis to a scalar at precompute time.
Your grid becomes cheaper for free — you don't need to handle this manually.

**Registration:** Register your precompute module in
`src/tengri/forward/precompute/registry.py` so `SEDModel` discovers it:
```python
PRECOMPUTE_REGISTRY["my_component"] = my_component_precompute_module
```

### JIT compilation: what runs where

```
model = SEDModel(...)            ← Python: builds precomputed tables, JIT-compiles kernel
model.predict(params)            ← XLA: runs inside @jax.jit
  → build_lookup(preint)(...)    ← XLA: cheap table lookup
  → attenuate_emission(...)      ← XLA: analytic, fast
  → gp_from_xi(...)              ← XLA: IFFT + scaling
```

Everything inside `predict()` runs as a single XLA graph. This means:
- No Python objects inside the JIT boundary — only JAX arrays and scalars
- No host-side I/O inside helpers (load templates in `precompute()`, not in the helper)
- No `isinstance` or string dispatch inside helpers — wire conditionals at build time

---

## Recipe 1: New SFH parametrization

Add a star formation history model (e.g., "rising-then-quenching", "stochastic
burstiness"). SFH functions are pure JAX, JIT-compatible functions that return
SFR(t) given an age grid and parameters.

### Checklist

- [ ] 1. Write the function in `src/tengri/components/sfh/`

  Create a pure JAX function with signature:
  ```python
  def my_sfh(log_ages_yr: jnp.ndarray, **params) -> jnp.ndarray:
      """Compute SFR(t) for custom parametrization.
      
      Parameters
      ----------
      log_ages_yr : array, shape (T,)
          Log10(age in years)
      **params : dict
          Custom parameter dict (e.g., tau_sfh, alpha, burst_amp, etc.)
      
      Returns
      -------
      sfr : array, shape (T,)
          Star formation rate in M☉/yr
      """
  ```
  Must use `@jax.jit` compatible operations (no Python-side branching on
  traced values; use `jnp.where` instead).

- [ ] 2. Register in `src/tengri/components/sfh/registry.py`

  Add to `SFH_REGISTRY` dict:
  ```python
  SFH_REGISTRY["my_sfh"] = my_sfh
  ```

- [ ] 3. Add parameters to `src/tengri/parameters/_param_defs.py`

  Define parameter definitions with prefix `sfh_my_sfh_*`:
  ```python
  "sfh_my_sfh_tau": default_value,
  "sfh_my_sfh_alpha": default_value,
  ```

- [ ] 4. Test in `tests/unit/test_sfh.py`

  ```python
  def test_my_sfh():
      ages = jnp.logspace(5, 10, 100)
      sfr = my_sfh(ages, tau_sfh=1e9, alpha=2.0)
      assert sfr.shape == ages.shape
      assert jnp.all(sfr >= 0)  # SFR non-negative
  ```

---

## Recipe 2: New dust attenuation law

Add a dust extinction curve (e.g., "modified Calzetti", "SMC dust", custom
power-law slope). Dust laws are pure JAX functions mapping wavelength to
optical depth.

### Checklist

- [ ] 1. Implement the attenuation function in `src/tengri/components/dust/`

  Signature (from `_protocol.py`):
  ```python
  def my_dust_law(wave_aa: jnp.ndarray, tau_v: float, **kwargs) -> jnp.ndarray:
      """Custom dust attenuation curve.
      
      Parameters
      ----------
      wave_aa : array, shape (W,)
          Wavelength in Angstrom
      tau_v : float
          Optical depth at V-band (555 nm)
      **kwargs : dict
          Additional dust parameters
      
      Returns
      -------
      tau_lambda : array, shape (W,)
          Optical depth at each wavelength
      """
  ```

- [ ] 2. Wire into `src/tengri/forward/emission_helpers.py`

  Add a call to your law inside `attenuate_emission()`, or add a new helper
  function if you're adding a wholly new attenuation component. Then wire the
  helper call in `src/tengri/forward/sed_model.py`.

  For a new **attenuation law** (replaces the existing law): map your name in
  `sed_model.py`'s dust law dispatch dict:
  ```python
  dust_law_map = {
      "my_custom_law": my_dust_law,
      ...
  }
  ```

- [ ] 3. Precompute (for new dust **emission** templates only, not attenuation)

  If you're adding a new IR emission template (not just an attenuation curve):
  create `src/tengri/components/dust/my_emission_precompute.py` implementing
  the precompute protocol (see "Precompute protocol" above). Register it in
  `src/tengri/forward/precompute/registry.py`.

  Attenuation laws are analytic and do not need precompute.

- [ ] 4. Add parameters to `src/tengri/parameters/_param_defs.py`

  Dust parameters use prefix `dust_*`:
  ```python
  "dust_slope": default_slope,
  "dust_ebv": default_ebv,
  ```

- [ ] 5. Test in `tests/unit/test_dust.py`

  ```python
  def test_my_dust_law():
      wave = jnp.logspace(3, 4.5, 200)  # UV to NIR
      tau_lam = my_dust_law(wave, tau_v=1.0)
      assert tau_lam.shape == wave.shape
      assert jnp.all(tau_lam >= 0)
  ```

---

## Recipe 3: New nebular/emission backend

Add a nebular emission model (e.g., "new CLOUDY grid", "empirical scaling",
"different photoionization code"). Nebular backends implement the
`NebularBackend` protocol.

### Checklist

- [ ] 1. Read the protocol in `src/tengri/components/nebular/_protocol.py`

  Required methods:
  - `line_luminosities(Q_H, logZ_gas) -> dict of line fluxes`
  - `continuum_sed(Q_H, logZ_gas, wave_aa) -> erg/s/Hz`

- [ ] 2. Implement the backend class

  Create `src/tengri/components/nebular/my_backend.py`:
  ```python
  class MyNebularBackend:
      def __init__(self, grid_data=None):
          """Initialize backend (e.g., load grids from disk)."""
          self.grid = grid_data
      
      def line_luminosities(self, Q_H: float, logZ_gas: float) -> dict:
          """Return dict of {line_name: L_line} in erg/s."""
          # Interpolate in (Q_H, logZ_gas) space
          ...
      
      def continuum_sed(self, Q_H: float, logZ_gas: float, 
                       wave_aa: jnp.ndarray) -> jnp.ndarray:
          """Return continuum SED in erg/s/Hz."""
          ...
  ```

- [ ] 3. Wire into `src/tengri/forward/emission_helpers.py`

  The `nebular_emission()` helper dispatches to backends via the `backend`
  argument. As long as your class satisfies the `NebularBackend` protocol,
  the existing wiring handles it — no changes to `emission_helpers.py` needed.

- [ ] 4. Register in `src/tengri/components/nebular/_shared.py`

  Add to the backend dispatch dict:
  ```python
  NEBULAR_BACKENDS["my_backend"] = MyNebularBackend
  ```

- [ ] 5. Precompute (for grid-based backends)

  If your backend interpolates a precomputed grid (like CLOUDY), implement
  the precompute protocol in `my_backend_precompute.py` so filter convolutions
  are cached at build time, not re-run on every likelihood call.
  Register it in `src/tengri/forward/precompute/registry.py`.

  For analytic backends (no templates), skip precompute.

- [ ] 6. Test with integration test in `tests/integration/`

  ```python
  def test_my_nebular_backend():
      model = SEDModel(..., nebular_model="my_backend")
      params = model.parameters.sample(jax.random.PRNGKey(0))
      sed = model.predict_sed(params)
      assert jnp.all(jnp.isfinite(sed))
  ```

---

## Recipe 4: New inference backend

Add an inference method (e.g., "expectation propagation", "variational
renormalization group", "differentiable nested sampling"). Backends implement
the `InferenceBackend` protocol.

### Checklist

- [ ] 1. Read the protocol in `src/tengri/inference/backends/_protocol.py`

  Required method:
  ```python
  def run(loss_fn, init_params, config) -> BackendResult:
      """Run inference.
      
      Parameters
      ----------
      loss_fn : callable
          Loss function (neg log posterior) as f(params) -> scalar
      init_params : dict
          Initial parameter values
      config : BackendConfig
          Configuration dict (sampler-specific)
      
      Returns
      -------
      result : BackendResult
          Samples, diagnostics, (optionally) chains
      """
  ```

- [ ] 2. Implement the backend in `src/tengri/inference/backends/my_method.py`

  ```python
  class MyBackend:
      def __init__(self, config):
          self.config = config
      
      def run(self, loss_fn, init_params, **kwargs):
          """Execute inference and return BackendResult."""
          # Your algorithm here
          samples = ...  # shape (n_samples, n_params)
          diagnostics = {}  # optional: ESS, R-hat, etc.
          return BackendResult(samples=samples, diagnostics=diagnostics)
  ```

- [ ] 3. Register in `src/tengri/inference/backends/__init__.py`

  Add to the dispatch table:
  ```python
  BACKENDS = {
      "my_method": MyBackend,
      ...
  }
  ```

- [ ] 4. Test with integration test in `tests/integration/`

  Backend is then accessible via:
  ```python
  fitter = Fitter(model, parameters)
  result = fitter.run("my_method", loss_fn, config=MyConfig(...))
  ```

---

## Common gotchas

### JAX tracing and control flow
Never use Python `if` on a JAX traced value inside `@jax.jit`. Use `jnp.where`:

```python
# WRONG:
if sfr[t] > 0:
    something()

# CORRECT:
result = jnp.where(sfr[t] > 0, something(), something_else())
```

### Array mutation
Use immutable array updates, never in-place mutation:

```python
# WRONG:
arr[i] = value

# CORRECT:
arr = arr.at[i].set(value)
```

### SED units
All SED components must return luminosity in **erg/s/Hz** (rest-frame). The
forward model handles redshift, distance, and filter convolution:

```python
# In your component function:
L_component = ...  # must be erg/s/Hz
# Do NOT include (1+z), distance, or filter effects
```

### Emission line wavelengths
All emission line wavelengths are in **vacuum Angstrom**, not air. E.g.,
H-alpha = 6564.61 Å (vacuum). Do NOT use air wavelengths.

### Physical constants
Import physical constants from `src/tengri/utils/physics_constants.py`,
never define local literals:

```python
# WRONG:
L_SUN = 3.839e33  # erg/s

# CORRECT:
from tengri.utils.physics_constants import L_SUN
```

Exception: Cue training uses L_SUN_CUE = 3.839e33 (intentional convention).

---

## References

- `docs/dev/NAMING_CONTRACT.md` — Class and function naming conventions
- `docs/dev/design_philosophy.md` — Architecture philosophy
- `src/tengri/components/` — Existing physics implementations
- `src/tengri/inference/backends/` — Existing inference methods
