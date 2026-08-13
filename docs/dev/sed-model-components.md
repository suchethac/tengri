# Writing SED model components

This is the canonical reference for adding a new model — a stellar continuum,
a dust attenuation law, a nebular emission backend, an AGN torus library, an
IR template, anything that contributes to a galaxy SED. Read this once and
you should be able to write a new model in one file.

The audience is an astronomer who knows the physics they want to add. The
framework details below show what you don't have to think about.

---

## The contract

Every model is a Python class inheriting from `SEDModelComponent`. The class
declares its free parameters, what it reads from upstream, what it publishes
downstream, and how it computes its contribution to L_ν:

```python
class ModifiedBlackbody(SEDModelComponent):
    name = "dust_ir"
    parameter_prefix = "dust_"

    # ─── Free parameters (defaults — overridable per fit)
    T    = Uniform(20.0, 80.0, "dust temperature",      units="K")
    beta = Uniform( 1.0,  3.0, "dust emissivity index", units="")

    # ─── What this model reads from upstream (0.0 fallback when absent)
    optional_inputs = {"L_ir": "erg/s"}

    # ─── What this model publishes for downstream
    outputs = {"L_ir_reemitted": "erg/s"}

    # ─── (Optional) load static data once
    def load(self, wave):
        return None     # closed-form models leave this default

    # ─── The physics
    def predict(self, p, sed_in, wave, *, L_ir):
        addition = modified_blackbody_lnu(wave, L_ir, p["T"], p["beta"])
        return sed_in + addition, {"L_ir_reemitted": trapz_freq(addition, wave)}
```

That is the whole contract. The signature `predict(p, sed_in, wave, **inputs) →
(sed_out, published)` is the same for every model — closed-form, atlas-based,
or neural-net emulator.

Cross-component reads come in two strengths: `inputs` (required — the
pipeline validator rejects a chain that cannot supply them) and
`optional_inputs` (supplied when an upstream component publishes them,
`0.0` otherwise, so the model degrades to a graceful no-op). Dust
re-emission declares `L_ir` optional for exactly this reason: a model with
no attenuator still builds and simply re-radiates nothing.

| Argument        | What you get                                                   |
|-----------------|----------------------------------------------------------------|
| `p`             | Your free parameters, prefix-stripped (`p["T"]`, not `p["dust_T"]`). Globals on the bare-name allowlist (e.g. `p["redshift"]`) pass through un-stripped |
| `sed_in`        | The rest-frame L_ν built up by upstream components (`erg/s/Hz`) |
| `wave`          | Rest-frame wavelength grid in Å                                |
| `**inputs`      | One keyword arg per entry in `inputs`/`optional_inputs`, supplied from upstream (optional ones default to `0.0`) |

| Return          | What it means                                                  |
|-----------------|----------------------------------------------------------------|
| `sed_out`       | The new rest-frame L_ν (emission models: `sed_in + addition`; transformations: `sed_in * factor`; pure pass-through: `sed_in`) |
| `published`     | Dict of quantities to publish into `state.derived` — must match `outputs` keys exactly |

---

## Three flavors of model, one signature

### Closed-form (most common)

No data to load. `predict()` is a formula.

```python
class Calzetti(SEDModelComponent):
    name = "dust_atten"
    parameter_prefix = "dust_"

    tau_v = Uniform(0.0, 4.0, "V-band optical depth", units="")
    delta = Uniform(-0.5, 0.5, "UV slope deviation",  units="")

    outputs = {"L_ir": "erg/s"}   # absorbed luminosity, for energy balance

    def predict(self, p, sed_in, wave):
        atten   = calzetti_atten(wave, p["tau_v"], p["delta"])
        sed_out = sed_in * atten
        L_ir = trapz_freq(sed_in - sed_out, wave)
        return sed_out, {"L_ir": L_ir}
```

### Atlas / template library

Pre-computed grid of spectra, interpolated at the parameter point. `load()`
reads the atlas; `self.data` holds it inside `predict()`.

```python
class SKIRTORTorus(SEDModelComponent):
    name = "agn_torus"
    parameter_prefix = "agn_"

    log_lbol      = Uniform( 8.0, 14.0, "log L_bol",          units="dex (L_sun)")
    theta_view    = Uniform( 0.0, 90.0, "viewing angle",       units="deg")
    optical_depth = Uniform( 3.0, 11.0, "9.7 µm optical depth", units="")

    outputs = {"L_agn_torus": "erg/s"}

    def load(self, wave):
        return load_skirtor("data/skirtor_v3.h5", wave)

    def predict(self, p, sed_in, wave):
        sed = skirtor_interp(self.data, p["log_lbol"],
                             p["theta_view"], p["optical_depth"])
        return sed_in + sed, {"L_agn_torus": trapz_freq(sed, wave)}
```

### Neural-network emulator

Same shape — `load()` returns trained weights, `predict()` does a forward
pass.

```python
class CueNebular(SEDModelComponent):
    name = "nebular"
    parameter_prefix = "neb_"

    logU     = Uniform(-4.0, -2.0, "ionization parameter",       units="dex")
    logZ_gas = Uniform(-2.0,  0.5, "gas metallicity (Z/Zsun)",    units="dex")
    fesc     = Fixed(0.0,          "Lyman continuum escape frac", units="")

    inputs = {"ssp_ages_yr": "yr", "age_weights": ""}   # required — stellar always publishes these
    outputs = {"line_waves": "Å", "line_lums": "erg/s"}

    def load(self, wave):
        return load_cue_nn_weights(wave)

    def predict(self, p, sed_in, wave, *, ssp_ages_yr, age_weights):
        continuum     = cue_continuum(self.data, p, ssp_ages_yr, age_weights, wave)
        line_w, line_L = cue_lines(    self.data, p, ssp_ages_yr, age_weights)
        return sed_in + continuum, {"line_waves": line_w, "line_lums": line_L}
```

### A made-up 5-axis dust IR atlas

To show that more axes don't change the shape:

```python
class BOSADust(SEDModelComponent):
    name = "dust_ir"
    parameter_prefix = "dust_"

    logUmin = Uniform(-1.0, 1.5, "log U_min",                 units="dex")
    qpah    = Uniform( 0.5, 4.5, "PAH mass fraction",         units="%")
    logSFR  = Uniform(-2.0, 3.0, "log SFR coupling",          units="dex")
    alpha   = Uniform( 1.0, 3.0, "intensity slope",           units="")
    fhot    = Uniform( 0.0, 0.3, "hot dust fraction",         units="")

    optional_inputs = {"L_ir": "erg/s"}

    def load(self, wave):
        return load_bosa("data/bosa.h5", wave)

    def predict(self, p, sed_in, wave, *, L_ir):
        sed = interpolate_5d(
            self.data.grid,
            (p["logUmin"], p["qpah"], p["logSFR"], p["alpha"], p["fhot"]),
            self.data.axes,
        ) * L_ir
        return sed_in + sed, {}
```

---

## What the framework does for you

You wrote one `predict()` function. The framework wires it into the rest of
the system at compile time:

### Free parameters

Class-level `Uniform(...)` / `Fixed(...)` / `Gaussian(...)` attributes are
auto-discovered and become this component's `declared_parameters()`. Units
appear in the posterior summary. The astronomer using your model can
override per-fit:

```python
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    dust={
        'type': 'two_component', 'law_bc': 'calzetti', 'all_params': FIXED,
        'emission': {
            'type':    'modified_blackbody',
            'T':       Fixed(35.0),          # pin one
            'beta_ir': Uniform(1.5, 2.5),    # narrow another
        },
    },
)
```

Two spellings here are load-bearing, and this example had both wrong.
`modified_blackbody` is a **dust-emission** model, not a dust *type* —
`dust={'type': 'modified_blackbody'}` raises `Unknown dust type`, because the
only dust types are `single_component`, `two_component` and `wg00`. And the
emission parameter is `beta_ir`, not `beta`. Emission parameters belong inside
`dust={'emission': {...}}`; written at the dust level they used to be accepted
and silently discarded, and are now refused with a message naming the nesting.

The class-level defaults never mutate — they're the prior baseline.

### Registration with the builder

`__init_subclass__` registers `(cls.name, cls)` in a module-level table.
`SEDModel.build(dust={'type': 'bosa', ...})` looks up `bosa` and instantiates
`BOSADust`. No factory edits, no central registry to maintain — define the
class anywhere and it's discoverable.

### Cross-component inputs

For each entry in `inputs` and `optional_inputs`, the framework looks up the
key in `state.derived` and passes it as a keyword argument to `predict()`.
Units are checked against the upstream `outputs` declaration at compile
time — a typo or unit drift fails at construction, not at runtime. Required
`inputs` must be publishable by some upstream component; `optional_inputs`
silently default to `0.0` when nothing upstream provides them.

### Cross-component outputs

The dict returned by `predict()`'s second slot is merged into `state.derived`
with the units declared in `outputs`. Downstream components that list this
key in their `inputs`/`optional_inputs` see it the next step in the chain.

### Inference

The class-level priors flow through `declared_parameters()` →
`Parameters` → the sampler. Whatever you choose — MAP, NUTS, VI, NSS —
sees them as standard free parameters. The posterior `summary()` lists
them, `Posterior.derived` resolves them.

### Wavelength precomputation (`WavePrecomp`)

Two paths, one `predict()`:

| Path                  | What `wave` is              | Cost                          |
|-----------------------|------------------------------|-------------------------------|
| Exact                 | Full rest-frame grid (n_wave) | predict() runs on every step  |
| `WavePrecomp`         | Per-filter effective wavelengths (n_filter) | predict() runs on n_filter ≪ n_wave points |

The framework calls `predict()` with `wave = filter_eff_waves` under
`approx=WavePrecomp()`, lifts the result into the per-component
`*_phot_lnu_precomp` LUT, and `observation.predict_via_precomp` sums these
LUTs + applies cosmology.

This is the [Zacharegkas+2025](https://arxiv.org/abs/2506.19919) effective-wavelength
approximation. Documented accuracy: 0.5% on photometry magnitudes for a
12-parameter SPS fit; ~0.03 mag for LSST bands. The approximation is that
the parameter-dependent factor is smooth across each filter band — true
for every dust/AGN/nebular model that's smooth in wavelength inside a
filter (essentially all of them).

Higher-accuracy Taylor refinement (first-order):

JAX gives `∂predict/∂wave` for free via `jax.grad`. Under
`approx=WavePrecomp(order=1)` the framework evaluates predict at λ_eff
**and** its first wavelength derivative at λ_eff. The downstream
calculation absorbs the per-filter moment Ψ = ∫ T(λ) λ L_SSP(λ) dλ that
the stellar component already publishes:

$$ c_{\rm band} \;\approx\; F(\lambda_{\rm eff})\,\Phi \;+\; F'(\lambda_{\rm eff})\,\Psi $$

You don't write the derivative — JAX differentiates your `predict()`. The
component opts in via a class flag (`taylor_order = 1`) if it benefits
from the extra accuracy; default is zeroth-order.

---

## The whole contract on one page

```python
class MyModel(SEDModelComponent):
    name             = "..."         # identifier — must be unique in registry
    parameter_prefix = "..._"        # prefix for free parameters

    # Class-level priors (auto-discovered as free parameters)
    my_param = Uniform(lo, hi, "description", units="...")

    # Cross-component contract
    inputs          = {"key": "units"}   # required reads from upstream
    optional_inputs = {"key": "units"}   # reads with a 0.0 fallback
    outputs         = {"key": "units"}   # what I publish for downstream

    # Optional: load static data once at compile time
    def load(self, wave):
        return ...                   # available as self.data inside predict()

    # Required: the physics
    def predict(self, p, sed_in, wave, **inputs):
        return new_sed, {"key": value, ...}

    # Optional: opt in to Taylor refinement under WavePrecomp
    taylor_order = 0                 # 0 = zeroth-order, 1 = +derivative
```

---

## What this is not

* **Not for the stellar component.** Stellar emission has a richer state
  machine (SFH + SSP + age weights + cross-component publishes for nine
  derived quantities). It stays on the bare `SEDComponent` protocol.

* **Not for IGM.** IGM transforms the observer-frame SED, not the
  rest-frame one. The signature here is rest-frame; IGM stays on the bare
  protocol.

* **Not the only registered pattern.** The ADR-0019 migration is complete:
  every built-in domain (dust attenuation, dust IR, nebular, radio, X-ray,
  IGM, AGN) now dispatches through the single `_REGISTRY` seam — see
  [`model-construction.md`](model-construction.md) for the per-domain
  table. Some of those entries are bare-Protocol adapters registered
  manually (rich state, e.g. the two-component dust screen); new models
  should still be authored as `SEDModelComponent` subclasses unless they
  genuinely don't fit the `predict()` shape.

---

## Where this code lives

| Thing                              | Path                                                |
|------------------------------------|-----------------------------------------------------|
| The base class                     | `src/tengri/components/sed_model_component.py`      |
| The bare protocol (unchanged)      | `src/tengri/protocols/component.py`                 |
| The registry that builders consult | `src/tengri/components/sed_model_component.py` (module-level dict populated by `__init_subclass__`) |
| New model definitions              | `src/tengri/components/<domain>/<name>.py`          |
| The walked-through example         | `notebooks/05_adding_a_model.py`                    |
| The ADR documenting the decision   | `docs/adr/0011-sed-model-component-base.md`         |

---

## References

* Zacharegkas et al. 2025 — *Differentiable SPS for differentiable cosmology*
  ([arXiv:2506.19919](https://arxiv.org/abs/2506.19919)). The
  effective-wavelength photometry approximation is in §3 + Appendix A.
* Existing components on the bare protocol — `RadioSEDComponent` at
  `src/tengri/components/radio/component.py` is the canonical reference.
