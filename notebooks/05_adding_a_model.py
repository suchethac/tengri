# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # # Adding your own SED model
#
# Many published recipes already ship with tengri (see
# `04_building_models.py` for the menu). This notebook adds a **new**
# one — a dust attenuation law, a dust IR atlas, an AGN torus library,
# anything that follows the *load → predict* shape.
#
# The contract is one class, one file. **Each physics block has its own
# base class that already declares that block's inputs/outputs
# contract**, and subclassing the right one is what makes your model
# reachable from the build grammar. For dust IR emission that base is
# `EmissionComponent`. The file then reads as physics:
#
# 1. Subclass the base class for your physics block.
# 2. Declare free parameters as class attributes (with units).
# 3. (Optional) `load(wave)` to read a pre-computed library off disk into
#    `self.data`.
# 4. `predict(p, sed_in, wave, **inputs)` — the physics.
#
# The model auto-registers; `SEDModel.build(dust_emission={'type':
# 'my_model'})` finds it; class-level priors flow through to
# inference; WavePrecomp picks it up automatically.
#
# > **Subclass the right base.** `build()` accepts a `type` only if that
# > component *publishes* `sed_dust_ir` — a structural check, not a name
# > lookup — and `EmissionComponent` declares it for you. Subclass bare
# > `SEDModelComponent` with your own output names and the class
# > registers happily while `build()` rejects the type with
# > `Unknown dust emission type '...'`.

# %% [markdown]
# ## Setup

# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs


os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import tengri
from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
)
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Distribution

# The base class for the dust IR emission block. It declares the
# cross-component contract every emission model shares — consumes `L_ir`,
# publishes `sed_dust_ir` — so your subclass only writes physics.
from tengri.components.dust.emission._component_base import EmissionComponent

# %% [markdown]
# ## A worked example — your own modified blackbody
#
# Let's take the simplest dust IR emission model, a modified blackbody,
# from physics straight to a working tengri model. The formula is
#
# $$ L_\nu(\lambda, T, \beta) = N(T, \beta, L_{\rm IR})\,\nu^\beta\,B_\nu(T) $$
#
# where $L_{\rm IR}$ is the energy-balance luminosity handed to us by the
# upstream dust attenuation component, and $N$ is the normalization that
# makes the frequency integral equal $L_{\rm IR}$.
#
# The framework passes $L_{\rm IR}$ into `predict` as the keyword
# `L_ir` — that name is the contract `EmissionComponent` declares, not a
# name we choose.

# %%
# Tiny helper for the frequency integral
_c = 2.99792458e18  # Å / s


def _trapz_freq(L_nu, wave_aa):
    nu = _c / wave_aa
    return jnp.trapezoid(L_nu[::-1], nu[::-1])


def _planck_nu(wave_aa, T):
    """B_nu in arbitrary units — normalization cancels later.

    Written the way the shipped closures write it
    (``components/dust/emission/analytic/_closures.py``), because the
    textbook spelling ``nu**3 / expm1(x)`` is not gradient-safe. Two
    guards, both load-bearing:

    * **Clip x.** At optical wavelengths ``x = h*nu/(kB*T)`` reaches
      ~1e4 for cold dust, so ``expm1(x)`` overflows to ``inf``. The
      forward value is a harmless 0, but its derivative is ``nan`` — and
      one ``nan`` anywhere poisons the whole gradient.
    * **Spell 1/expm1(x) as exp(-x) / -expm1(-x).** Same number, but the
      denominator now lives in (0, 1] and cannot overflow at any x,
      while ``exp(-x)`` underflows to exactly 0.0 — the true Wien limit.

    The floor matters at the other end: the occupation number goes as
    1/x, so x = 0 is a division by zero (the Rayleigh-Jeans limit would
    come back ``inf`` rather than large-but-finite).
    """
    h = 6.62607015e-27  # erg·s
    kB = 1.380649e-16  # erg/K
    nu = _c / wave_aa
    x = jnp.clip(h * nu / (kB * T), 1e-10, 500.0)
    return nu**3 * jnp.exp(-x) / -jnp.expm1(-x)


# %% [markdown]
# Now the model class. **One base class, two declarations, one method.**
#
# Note what is *not* here: no `inputs`, no `outputs`, no
# `parameter_prefix`. `EmissionComponent` already declares all three for
# the whole dust-emission block (`optional_inputs = {"L_ir": ...}`,
# `outputs = {"sed_dust_ir": ...}`, `parameter_prefix = "dust_"`).
# Re-declaring them with names of your own is the single most common way
# to write a component that registers but is never reachable.


# %%
class MyModifiedBlackbody(EmissionComponent):
    """Optically-thin modified blackbody dust IR emission."""

    name = "my_modified_blackbody"

    # ─── Free parameters (defaults — overridable per fit)
    #
    # These reuse the canonical dust-emission parameter names declared in
    # `components/dust/_params.py`. That is a requirement, not a
    # convention: the parameter map is built by a *static scan of the
    # installed package* (ADR-0008), so a class defined here in a
    # notebook cannot introduce a brand-new parameter name — see
    # "Adding a genuinely new parameter" below.
    T = Uniform(20.0, 80.0, "dust temperature", units="K")
    beta_ir = Uniform(1.0, 3.0, "dust emissivity index", units="")

    # `load()` is optional — closed-form models like this one leave it
    # as the default (no atlas to load).

    def predict(self, p, sed_in, wave, *, L_ir):
        # Modified-blackbody shape, un-normalized
        shape = wave ** (-p["beta_ir"]) * _planck_nu(wave, p["T"])
        # Normalize so the frequency integral equals L_ir
        sed = L_ir * shape / _trapz_freq(shape, wave)
        # Publish under the contract name the block expects
        return sed_in + sed, {"sed_dust_ir": sed}


# %% [markdown]
# That's the whole model. Now check that it registered — and, separately,
# that the builder will actually *accept* it.
#
# These are two different questions. Registration is automatic for any
# `SEDModelComponent` subclass, so the first check passes even for a
# component `build()` will reject. The second check is the one that
# matters.

# %%
from tengri.components.sed_model_component import _REGISTRY
from tengri.parameters.groups import _valid_dust_emission_types

print("Registered (dispatch):", "my_modified_blackbody" in _REGISTRY)
print("Accepted  (validator):", "my_modified_blackbody" in _valid_dust_emission_types())
print("Publishes            :", [o.name for o in MyModifiedBlackbody._outputs_tuple])
print("Free parameters:")
for d in MyModifiedBlackbody().declared_parameters():
    print(f"  {d.name:20s}  {type(d.prior).__name__:10s}  units={d.units!r}")


# %% [markdown]
# ### The trap, made explicit
#
# Here is the same model written against `SEDModelComponent` with
# hand-rolled input/output names — the shape that looks right and isn't.
# It registers. It is still unusable.

# %%
class _BrokenMBB(SEDModelComponent):
    """Registers fine. `build()` will never accept it."""

    name = "my_mbb_broken"
    parameter_prefix = "dust_"

    T = Uniform(20.0, 80.0, "dust temperature", units="K")

    inputs = {"L_absorbed": "erg/s"}  # noqa: RUF012 — invented name
    outputs = {"L_ir": "erg/s"}  # noqa: RUF012 — invented name

    def predict(self, p, sed_in, wave, *, L_absorbed):
        return sed_in, {"L_ir": jnp.asarray(0.0)}


print("Registered (dispatch):", "my_mbb_broken" in _REGISTRY)
print("Accepted  (validator):", "my_mbb_broken" in _valid_dust_emission_types())
print("Publishes            :", [o.name for o in _BrokenMBB._outputs_tuple])
print()
print("The validator looks for 'sed_dust_ir' among a component's outputs.")
print("This one publishes 'L_ir', so build() reports:")
print("  ValueError: Unknown dust emission type 'my_mbb_broken'.")
print("  Did you mean: modified_blackbody?")
print()
print("...which reads like a typo, but is a contract mismatch.")

# %% [markdown]
# ## Using your model in a fit
#
# `SEDModel.build()` consults the registry, so the `'type'` string we
# declared above is reachable from the standard nested-dict grammar.
# Parameter overrides in the `emission` sub-block use the *unprefixed*
# names (`T`, `beta_ir`) — the `dust_` prefix is added for you:

# %%
# `load_ssp` resolves the grid wherever it lives, so this demo does not depend
# on the working directory. The gate here used to test a working-directory-
# relative path with os.path.exists, which *silently* took the else-branch
# whenever the notebook ran from anywhere but the repository root — publishing
# "skipping the build demo" in place of the content it exists to show.
SSP_NAME = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0"

try:
    ssp = tengri.load_ssp(SSP_NAME)
except FileNotFoundError as exc:
    ssp = None
    _why = exc
if ssp is not None:
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"]),
    )

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "tau_v": Fixed(0.4),
        }, dust_emission={"type": "my_modified_blackbody", "T": Fixed(35.0), "beta_ir": Fixed(1.8)},
        redshift=Fixed(0.05),
    )
    model.spec.summary()
else:
    print(f"Skipping the build demo: {_why}")
    print("Fetch a grid with tengri.download_ssp(), or set $TENGRI_DATA_DIR.")

# %% [markdown]
# ## What the framework gives you for free
#
# Everything below happened automatically — no code added on top of the
# 20 lines of `MyModifiedBlackbody`:
#
# - **Priors flowed to inference.** Pass `Uniform(...)` instead of
#   `Fixed(...)` in the `emission` sub-block and `model.spec.free_params`
#   lists `dust_T` and `dust_beta_ir`. The chosen sampler — MAP, NUTS,
#   VI, NSS — picks them up as standard free parameters. The posterior
#   summary lists them with units intact.
#
# - **Cross-component contract enforced.** `EmissionComponent` declares
#   `optional_inputs = {"L_ir": "erg/s"}` and
#   `outputs = {"sed_dust_ir": "erg/s/Hz"}` on your behalf, so the
#   pipeline check at construction time confirms an upstream component
#   publishes `L_ir` with matching units, and that what you publish is
#   what the downstream energy-balance and observable machinery reads.
#
# - **WavePrecomp compatibility.** Switch `approx=WavePrecomp()` on the
#   `SEDModel.build()` call and the framework calls your `predict` at
#   per-filter effective wavelengths instead of the full grid. Same
#   function, ~5–10× faster broadband-photometry runs at the
#   documented ~0.5 % tolerance (Zacharegkas+2025).
#
# - **Optional Taylor refinement.** Setting `taylor_order = 1` on the
#   class would have JAX auto-differentiate `predict` w.r.t. `wave` and
#   use the moment $\Psi$ that stellar publishes to get a first-order
#   accuracy bump — still no extra code from you.
#
# - **Reachable by name.** `SEDModel.build(dust_emission={'type':
#   'my_modified_blackbody'})` finds the class — `__init_subclass__`
#   registered `(name, cls)` automatically.

# %% [markdown]
# ## Adding a genuinely new parameter
#
# `MyModifiedBlackbody` above reuses `dust_T` and `dust_beta_ir`, which
# already exist. That was deliberate. A component defined in a notebook
# **cannot introduce a new parameter name**, and it is worth knowing why
# before you go looking for the bug.
#
# There are two separate registries with two different lifetimes:
#
# | Registry | Populated by | Sees notebook classes? |
# |---|---|---|
# | `_REGISTRY` (dispatch) | `__init_subclass__`, at class-definition time | **yes** |
# | parameter map | static scan of `components/*/_params.py` in the *installed package* (ADR-0008) | **no** |
#
# So a notebook class is dispatchable but cannot add parameters. Declare
# a name the param map has never heard of and it is silently dropped —
# `tengri.Parameters(dust_my_new_param=...)` then raises
# `Unknown parameter`.
#
# To add a real parameter, add its `ParamDeclaration` to the block's
# `_params.py` in a checkout and reinstall. (`tengri.register_component`
# looks like the seam for this, but it is vestigial — nothing consults
# it any more; the static scan replaced it.)

# %% [markdown]
# ## Other blocks, other seams
#
# The `SEDModelComponent` shape shown here is the seam for whole physics
# *blocks*. Several menus inside a block instead take a **decorator that
# registers a function**, which is a smaller thing to write. Check the
# seam your block actually uses before writing a class:
#
# | What you want to add | Seam | Lives in |
# |---|---|---|
# | Dust IR emission model | subclass `EmissionComponent` | `components/dust/emission/` |
# | Dust attenuation curve | `@register_dust_law(name=...)` | `components/dust/laws/_registry.py` |
# | Nebular backend | `@register_nebular_model(name=...)` | `components/nebular/_models.py` |
# | SFH | `SFH_REGISTRY` entry | `components/stellar/sfh/registry.py` |
# | IGM / radio / X-ray | `register_igm_model` / `register_radio_model` / `register_xray_model` | `components/<block>/_models.py` |
#
# Every block's accepted `type` names come from a live registry
# (ADR-0005 / ADR-0008). For a worked example, read a shipped component
# next to yours — they live in `src/tengri/components/<block>/`.

# %% [markdown]
# ## Further reading
#
# - **The how-to**: `docs/dev/sed-model-components.md` — full contract,
#   conventions, what's optional.
# - **The architecture**: `docs/dev/archive/forward-model-architecture.md` — how
#   your component fits into the wider pipeline, the cross-component
#   contract, WavePrecomp, the build resolver.
# - **The decision**: `docs/adr/0011-sed-model-component-base.md` — why
#   the base class exists and what alternatives were considered.
# - **The base class**: `src/tengri/components/sed_model_component.py`.
# - **Canonical components**: `src/tengri/components/dust/wg00_model.py`
#   (closed-form attenuation), `src/tengri/components/dust/draine2021_pah_ir.py`
#   (template library), `src/tengri/components/agn/skirtor_model.py` (library).
