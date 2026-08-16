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
# # Adding your own SED model
#
# A galaxy SED is built up from physics components: stellar emission, dust
# attenuation, nebular lines, dust IR emission, an AGN, IGM, radio,
# X-ray. Many published recipes already ship with tengri (see
# `04_building_models.py` for the menu). This notebook shows what it
# looks like to add a **new** one — a new dust attenuation law, a new
# dust IR atlas, a new AGN torus library, anything that follows the
# *load → predict* shape.
#
# The contract is one class, one file. The base class
# `SEDModelComponent` does the bookkeeping so the file reads as physics:
#
# 1. Declare free parameters as class attributes (with units).
# 2. Declare what the model reads from upstream and what it publishes.
# 3. (Optional) `load(wave)` to read a pre-computed library off disk into
#    `self.data`.
# 4. `predict(p, sed_in, wave, **inputs)` — the physics.
#
# That's it. The model auto-registers; `SEDModel.build(dust={'type':
# 'my_model'})` finds it; class-level priors flow through to inference;
# WavePrecomp picks it up automatically.

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

# %% [markdown]
# ## A worked example — your own modified blackbody
#
# Let's take the simplest dust IR emission model, a modified blackbody,
# from physics straight to a working tengri model. The formula is
#
# $$ L_\nu(\lambda, T, \beta) = N(T, \beta, L_{\rm absorbed})\,\nu^\beta\,B_\nu(T) $$
#
# where $L_{\rm absorbed}$ comes from the upstream dust attenuation
# component and $N$ is the normalization that makes the frequency
# integral equal $L_{\rm absorbed}$.

# %%
# Tiny helper for the frequency integral
_c = 2.99792458e18  # Å / s


def _trapz_freq(L_lambda, wave_aa):
    nu = _c / wave_aa
    return jnp.trapezoid(L_lambda[::-1], nu[::-1])


def _planck_nu(wave_aa, T):
    """B_nu in arbitrary units — normalization cancels later."""
    h = 6.62607015e-27  # erg·s
    kB = 1.380649e-16  # erg/K
    nu = _c / wave_aa
    x = h * nu / (kB * T)
    return nu**3 / jnp.expm1(x)


# %% [markdown]
# Now the model class. **One file, four declarations, two methods.**


# %%
class MyModifiedBlackbody(SEDModelComponent):
    """Optically-thin modified blackbody dust IR emission."""

    name = "my_modified_blackbody"
    parameter_prefix = "dust_"

    # ─── Free parameters (defaults — overridable per fit)
    T = Uniform(20.0, 80.0, "dust temperature", units="K")
    beta = Uniform(1.0, 3.0, "dust emissivity index", units="")

    # ─── What this model reads from upstream
    inputs = {"L_absorbed": "erg/s"}  # noqa: RUF012

    # ─── What this model publishes for downstream
    outputs = {"L_ir": "erg/s"}  # noqa: RUF012

    # `load()` is optional — closed-form models like this one leave it
    # as the default (no atlas to load).

    def predict(self, p, sed_in, wave, *, L_absorbed):
        # Modified-blackbody shape, un-normalized
        shape = wave ** (-p["beta"]) * _planck_nu(wave, p["T"])
        # Normalize so the frequency integral equals L_absorbed
        norm = L_absorbed / _trapz_freq(shape, wave)
        addition = norm * shape
        L_ir = _trapz_freq(addition, wave)
        return sed_in + addition, {"L_ir": L_ir}


# %% [markdown]
# That's the whole model. Now check that it registered.

# %%
from tengri.components.sed_model_component import _REGISTRY

print("Registered:", "my_modified_blackbody" in _REGISTRY)
print("Free parameters:")
for d in MyModifiedBlackbody().declared_parameters():
    print(f"  {d.name:20s}  {type(d.prior).__name__:10s}  units={d.units!r}")

# %% [markdown]
# ## Using your model in a fit
#
# `SEDModel.build()` consults the registry, so the `'type'` string we
# declared above is reachable from the standard nested-dict grammar:

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
        dust={
            "type": "single_component",
            "law_bc": "calzetti",
            "tau_v": Fixed(0.4),
            "emission": {"type": "my_modified_blackbody", "T": Fixed(35.0), "beta": Fixed(1.8)},
        },
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
# - **Priors flowed to inference.** `model.spec.free_params` lists
#   `dust_T` and `dust_beta` (if you set them `Uniform`). The chosen
#   sampler — MAP, NUTS, VI, NSS — picks them up as standard free
#   parameters. The posterior summary lists them with units intact.
#
# - **Cross-component contract enforced.** Because `inputs = {"L_absorbed":
#   "erg/s"}` is declared, `validate_pipeline` checks at construction
#   time that an upstream component publishes `L_absorbed` with matching
#   units. Wire it wrong and you fail at build time, not silently at
#   runtime.
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
# - **Reachable by name.** `SEDModel.build(dust={'emission': {'type':
#   'my_modified_blackbody'}})` finds the class — `__init_subclass__`
#   registered `(name, cls)` automatically.

# %% [markdown]
# ## Three other shapes the same contract covers
#
# ### Closed-form (no `load`)
#
# Calzetti dust attenuation:
#
# ```python
# class Calzetti(SEDModelComponent):
#     name = "calzetti"
#     parameter_prefix = "dust_"
#
#     tau_v = Uniform(0.0, 4.0, "V-band optical depth", units="")
#     delta = Uniform(-0.5, 0.5, "UV slope deviation",  units="")
#
#     inputs  = {}
#     outputs = {"L_absorbed": "erg/s"}
#
#     def predict(self, p, sed_in, wave):
#         atten   = calzetti_atten(wave, p["tau_v"], p["delta"])
#         sed_out = sed_in * atten
#         L_absorbed = _trapz_freq(sed_in - sed_out, wave)
#         return sed_out, {"L_absorbed": L_absorbed}
# ```
#
# ### Atlas library (load an HDF5 grid, interpolate)
#
# SKIRTOR AGN torus:
#
# ```python
# class SKIRTORTorus(SEDModelComponent):
#     name = "skirtor"
#     parameter_prefix = "agn_"
#
#     log_lbol      = Uniform( 8.0, 14.0, "log L_bol",            units="dex (L_sun)")
#     theta_view    = Uniform( 0.0, 90.0, "viewing angle",        units="deg")
#     optical_depth = Uniform( 3.0, 11.0, "9.7 µm optical depth", units="")
#
#     inputs  = {}
#     outputs = {"L_agn_torus": "erg/s"}
#
#     def load(self, wave):
#         return load_skirtor_atlas(wave)        # → self.data
#
#     def predict(self, p, sed_in, wave):
#         sed = skirtor_interp(self.data, p["log_lbol"],
#                              p["theta_view"], p["optical_depth"])
#         return sed_in + sed, {"L_agn_torus": _trapz_freq(sed, wave)}
# ```
#
# ### NN emulator (load trained weights, forward pass)
#
# Cue nebular emulator (Li+2024):
#
# ```python
# class CueNebular(SEDModelComponent):
#     name = "cue_emulator"
#     parameter_prefix = "neb_"
#
#     logU     = Uniform(-4.0, -2.0, "ionization parameter",       units="dex")
#     logZ_gas = Uniform(-2.0,  0.5, "gas metallicity (Z/Zsun)",   units="dex")
#     fesc     = Fixed(0.0,          "Lyman continuum escape frac", units="")
#
#     inputs  = {"ssp_ages_yr": "yr", "age_weights": ""}
#     outputs = {"line_waves": "Å", "line_lums": "erg/s"}
#
#     def load(self, wave):
#         return load_cue_nn_weights(wave)
#
#     def predict(self, p, sed_in, wave, *, ssp_ages_yr, age_weights):
#         continuum     = cue_continuum(self.data, p, ssp_ages_yr, age_weights, wave)
#         line_w, line_L = cue_lines(    self.data, p, ssp_ages_yr, age_weights)
#         return sed_in + continuum, {"line_waves": line_w, "line_lums": line_L}
# ```
#
# All three are the same shape, just different bodies of `predict`. See
# the canonical components in `src/tengri/components/<domain>/<name>_model.py`.

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
