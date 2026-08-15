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
# SDSS measures **total-flux photometry** (whole galaxy through broad filters)
# and **fiber spectroscopy** (only the nuclear region inside a ~2-arcsec fiber).
# The two see different fractions of total light. How to reconcile them?
#
# The classical approach scales the spectrum by a single number to match photometry —
# equivalent to assuming a **flat slab** of uniform surface brightness. Real galaxies
# have Sérsic profiles; the fiber captures the bright nuclear region while photometry
# integrates the extended envelope, so the two disagree on what fraction of the light
# the fiber saw. The aperture fractions printed below set the size of that disagreement.
#
# Tengri's spatial sub-model makes the correction explicit:
#
# 1. Specify a **`SpatialModel`** with a Sérsic profile.
# 2. Compose an **`Observation`** with both photometry and a
#    **`FiberSpectroscopyObservation`** that knows the fiber radius.
# 3. The `ForwardModel.predict` chain runs `SED → Spatial`, then integrates
#    the spatial profile inside the fiber mask and scales the spectrum by that
#    fraction. Photometry is untouched (already total flux).
#
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
# **For SED fitting:** When the same flux normalization scales both spectrum and
# photometry, the assumed aperture fraction changes the inferred total flux. The
# flat-slab assumption says the fiber captures `frac_flat` of total light. The
# physical Sérsic profile says it captures `frac_sersic`. If you fit jointly with
# the wrong aperture, the minimizer pushes the total-flux amplitude up or down to
# compensate — biasing stellar mass, SFR, and dust attenuation. Concentration sets
# how far apart the two fractions sit: compare the n=1 exponential disk above with
# the n=4 de Vaucouleurs profile printed below.

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
# Compare against a flat-slab fit (`SpatialModel(components=[FlatSlab()])`):
# the two ForwardModels share the same SED chain and observation; only the spatial
# sub-model differs. Run both and read the bias the flat-slab assumption introduces
# on this dataset. The same machinery generalizes to imaging and IFU spectroscopy.

# %% [markdown]
# ## Reference
#
# - Spatial profiles: `tengri.components.spatial.{Sersic, Exponential, FlatSlab}`.
# - Aperture math: `tengri.observation.fiber_aperture`.
# - Composer: `tengri.observation.joint_observation.JointObservation`.
