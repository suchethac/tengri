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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Joint spec-phot with a physical aperture
#
# Surveys like SDSS measure two things about the same galaxy: **total-flux
# photometry** (the whole galaxy as seen through broad filters) and
# **fiber spectroscopy** (only the part of the galaxy that fell inside a
# few-arcsec fiber). The two see different fractions of the galaxy's
# total light. How do you reconcile them in a joint SED fit?
#
# The classical answer is to scale the spectrum by a single number until
# it matches the photometry — equivalent to assuming the galaxy is a
# **flat slab** of uniform surface brightness across the aperture. Real
# galaxies are not flat slabs: they have Sérsic-shaped surface-brightness
# profiles, and the fiber captures only the bright nuclear region while
# the photometry integrates the extended outer envelope. The flat-slab
# assumption biases the recovered stellar mass and SFR by tens of
# percent for typical galaxies and substantially more for compact
# spheroidals.
#
# This notebook walks through the corrected aperture handling that
# Tengri's spatial sub-model makes explicit:
#
# 1. Specify a **`SpatialModel`** with a Sérsic profile.
# 2. Compose an **`Observation`** with both broadband photometry and a
#    **`FiberSpectroscopyObservation`** that knows the fiber radius.
# 3. The `ForwardModel.predict` chain runs `SED → Spatial`, then the
#    observation integrates the spatial profile inside the fiber mask
#    and scales the spectrum by that fraction. The photometry is
#    untouched (it's already total flux).
#
# The architecture is in `docs/dev/archive/forward-model-architecture.md` §3.3.
# The aperture-fraction math is in `tengri.observation.fiber_aperture`.

# %%
from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import jax.numpy as jnp

from tengri.components.spatial.flat_slab import FlatSlab
from tengri.components.spatial.sersic import Sersic
from tengri.forward.spatial_model import SpatialModel, default_grid_kpc
from tengri.observation.fiber_aperture import aperture_fraction, arcsec_to_kpc

# %% [markdown]
# ## How big is a 2-arcsec fiber on this galaxy?
#
# Convert a fiber radius in arcsec to a physical scale in kpc at the
# galaxy's redshift via the angular diameter distance.

# %%
# An SDSS fiber is 3-arcsec diameter, so radius = 1.5". Take z=0.05 as a
# typical low-redshift survey target.
fiber_radius_arcsec = 1.5
z = 0.05
fiber_radius_kpc = float(arcsec_to_kpc(fiber_radius_arcsec, z))
print(f"At z={z}, a {fiber_radius_arcsec}-arcsec fiber radius is {fiber_radius_kpc:.2f} kpc.")

# %% [markdown]
# ## Setup: a typical Milky-Way-mass galaxy
#
# Build a Sérsic profile (the physical model) and a flat-slab profile
# (the classical-codes model) on the same kpc grid. Both are then
# evaluated at the same parameters except the profile shape.

# %%
grid = default_grid_kpc(n=128, extent_kpc=15.0)
spatial_sersic = SpatialModel(components=[Sersic()], grid_kpc=grid)
spatial_flat = SpatialModel(components=[FlatSlab()], grid_kpc=grid)

# Reference galaxy: Sérsic with effective radius 3 kpc, n=1 (exponential
# disk). The flat-slab "galaxy" gets a radius set to match the Sérsic's
# half-light radius — the most charitable interpretation of the
# flat-slab approximation.
re_kpc = 3.0
params_sersic = {
    "spatial_re_kpc": jnp.float64(re_kpc),
    "spatial_n": jnp.float64(1.0),
    "spatial_axis_ratio": jnp.float64(1.0),
    "spatial_pa_deg": jnp.float64(0.0),
}
params_flat = {"spatial_radius_kpc": jnp.float64(re_kpc)}

# %% [markdown]
# ## Run the spatial sub-models
#
# Each `SpatialModel.run` threads state through its components and
# populates `state.derived["spatial_profile_2d"]`. The grid lives in
# `state.derived["spatial_grid_xy_kpc"]`.

# %%
from tengri.protocols.component import ForwardState

state = ForwardState(wave=jnp.zeros(1))
profile_sersic = spatial_sersic.run(state, params_sersic).derived["spatial_profile_2d"]
profile_flat = spatial_flat.run(state, params_flat).derived["spatial_profile_2d"]

print(f"Sersic profile peak: {float(profile_sersic.max()):.3e}")
print(f"FlatSlab profile peak: {float(profile_flat.max()):.3e}  (uniform within radius)")

# %% [markdown]
# ## The aperture fraction
#
# This is the headline number. What fraction of the total galaxy light
# falls inside the fiber?

# %%
frac_sersic = float(aperture_fraction(profile_sersic, grid, fiber_radius_kpc))
frac_flat = float(aperture_fraction(profile_flat, grid, fiber_radius_kpc))

print(f"Sérsic galaxy:   {frac_sersic * 100:.1f}% of total light inside the fiber")
print(f"FlatSlab galaxy: {frac_flat * 100:.1f}% of total light inside the fiber")
print()
print(f"Ratio (flat-slab / Sérsic): {frac_flat / frac_sersic:.2f}")

# %% [markdown]
# **What this means for SED fitting.** When the same flux normalization
# is used for both spectrum and photometry, a flat-slab assumption
# implies the fiber captures `frac_flat` of the total. The physical
# Sérsic profile says it actually captures `frac_sersic`. If you fit
# both jointly with the wrong aperture model, you push the inferred
# total-flux normalization up or down to compensate — biasing
# **stellar mass**, **star-formation rate**, and **dust attenuation**.
#
# Sérsic n=1 (this notebook): the bias is modest. Try `n=4`
# (de Vaucouleurs spheroidal) below and the ratio gets much worse.

# %%
params_dv = {**params_sersic, "spatial_n": jnp.float64(4.0)}
profile_dv = spatial_sersic.run(state, params_dv).derived["spatial_profile_2d"]
frac_dv = float(aperture_fraction(profile_dv, grid, fiber_radius_kpc))
print(
    f"de Vaucouleurs (n=4): {frac_dv * 100:.1f}% inside fiber; "
    f"flat-slab is off by {frac_flat / frac_dv:.2f}x"
)

# %% [markdown]
# ## Building the joint observation
#
# With the spatial sub-model wired in, `FiberSpectroscopyObservation`
# automatically scales the spectrum by the aperture fraction, leaving
# the photometry untouched (total flux).
#
# ```python
# from tengri.observation import Observation
# from tengri.observation.fiber_spectroscopy import FiberSpectroscopyObservation
# from tengri.observation.joint_observation import JointObservation
#
# base = Observation(
#     photometry=Photometry(filters=..., flux=..., error=...),
#     spectroscopy=Spectroscopy(wave=..., flux=..., error=...),
# )
# fiber_spec = FiberSpectroscopyObservation(
#     observation=base,
#     fiber_radius_arcsec=1.5,
# )
# obs = JointObservation(base, fiber_spec)
#
# forward = ForwardModel.build(
#     sed=SEDModel.build(...),
#     spatial=SpatialModel(components=[Sersic()]),
#     observation=obs,
# )
# fit = forward.predict(params)
# # fit["phot_fnu"] — total flux (untouched)
# # fit["spec_fnu"] — scaled by aperture_fraction(Sérsic, fiber)
# ```
#
# Compare this against a flat-slab fit (`SpatialModel(components=[FlatSlab()])`):
# the two ForwardModels share the same SED chain and observation; only
# the spatial sub-model differs. Run both, compare recovered stellar
# masses, and read off the bias the flat-slab approximation would
# introduce on this dataset.
#
# The same machinery generalizes to imaging (resolved 2-D maps via a
# future `ImagingObservation`) and to IFU spectroscopy — the spatial
# sub-model is the common substrate.

# %% [markdown]
# ## Further reading
#
# - Architecture spec: `docs/dev/archive/forward-model-architecture.md` §2 (the
#   motivating story), §3.3 (B-path keys reserved for color gradients).
# - Spatial profiles: `tengri.components.spatial.{Sersic, Exponential, FlatSlab}`.
# - Aperture math: `tengri.observation.fiber_aperture`.
# - Composer: `tengri.observation.joint_observation.JointObservation`.
