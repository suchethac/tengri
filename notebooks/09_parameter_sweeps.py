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
# # Parameter sweeps
#
# Vary one knob, plot a fan of SEDs. It's the cheapest way to develop
# physical intuition for a forward model — and because tengri's model is
# pure JAX, sweeps that would take a Python loop in a non-differentiable
# code can be batched through `vmap` and run in a single compiled call.
#
# This notebook covers four kinds of sweep:
#
# 1. A 1-D sweep with the gallery's `sweep_parameter` helper (the canonical
#    idiom — same one used by the 15+ scripts in `examples/`).
# 2. A prior fan via `sample_sfh_prior` (sweep parameter values *drawn*
#    from a registered prior, not picked by hand).
# 3. An `Instrument`-driven photometric setup (no hand-rolled filter lists).
# 4. A 2-D photometric grid via `predict_photometry_batch` and `jax.vmap`.

# %%
import os
import sys

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Locate _plot_style and the data directory regardless of cwd.
import importlib.util

_repo_data_root = None
_spec_tengri = importlib.util.find_spec("tengri")
if _spec_tengri is not None and _spec_tengri.origin:
    _walk = os.path.dirname(os.path.abspath(_spec_tengri.origin))
    for _ in range(12):
        _candidate = os.path.join(_walk, "notebooks", "_plot_style.py")
        if os.path.isfile(_candidate):
            sys.path.insert(0, os.path.dirname(_candidate))
            _repo_data_root = os.path.dirname(os.path.dirname(os.path.abspath(_candidate)))
            break
        _parent = os.path.dirname(_walk)
        if _parent == _walk:
            break
        _walk = _parent
if _repo_data_root and os.path.isdir(os.path.join(_repo_data_root, "data")):
    os.chdir(_repo_data_root)

import jax
import jax.numpy as jnp
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

import tengri
from tengri import (
    Fixed,
    Instrument,
    Parameters,
    SEDModel,
    Uniform,
    load_ssp_data,
)
from tengri.analysis.plotting import SWEEP_CMAPS, setup_style, sweep_parameter
from tengri.components.stellar.sfh import sample_sfh_prior

setup_style()
print(f"tengri {tengri.__version__}")

# %%
ssp = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

# %% [markdown]
# ## 1-D sweep with `sweep_parameter`
#
# Build a fixed star-forming galaxy and vary one knob. `sweep_parameter`
# loops in Python, but each forward call hits tengri's persistent JIT
# cache, so the per-iteration cost after the first is the cost of
# `predict_rest_sed` itself.

# %%
spec_sf = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),
    sfh_tsnorm_width_gyr=Fixed(1.5),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.5),
    dust_tau_diff=Fixed(0.3),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)
model_sf = SEDModel(spec_sf, ssp)

fig, ax = sweep_parameter(
    model_sf,
    "dust_tau_diff",
    [0.0, 0.3, 0.7, 1.5, 3.0],
    cmap=SWEEP_CMAPS["dust"],
    label_fmt=r"$\tau_{{\rm diff}}$ = {:.1f}",
    wave_range=(1000, 10000),
)
ax.set_title("Diffuse-ISM optical depth: redder continuum, weaker 4000 Å break")
fig.tight_layout()
plt.show()

# %% [markdown]
# Same idiom for any other knob — see `examples/dust/`, `examples/agn/`,
# `examples/spectroscopy/` for ~15 worked sweeps across the model.

# %% [markdown]
# ## A prior fan via `sample_sfh_prior`
#
# Sometimes the question isn't "what does this knob do?" but "what range
# of behaviour does my prior actually allow?". `sample_sfh_prior` draws
# from a registered SFH family's default prior block and returns the SFR
# curves directly; one line, no `Parameters` boilerplate.

# %%
age_grid_yr, curves = sample_sfh_prior(
    "dpl",                       # double-power-law SFH (Carnall+2018)
    jax.random.PRNGKey(0),
    n=24,
)

fig, ax = plt.subplots(figsize=(9, 4))
age_gyr = np.asarray(age_grid_yr) / 1e9
viridis = plt.get_cmap("viridis")
for i, c in enumerate(np.asarray(curves)):
    ax.plot(age_gyr, c, color=viridis(0.05 + 0.8 * i / len(curves)), lw=1.0, alpha=0.85)
ax.set_xlim(0, 13.8)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR(t) [M$_\odot$/yr]")
ax.set_title(r"24 prior draws from $\tt{dpl}$ — what the registry default actually allows")
fig.tight_layout()
plt.show()

# %% [markdown]
# Override one prior to narrow the fan:

# %%
_, narrow = sample_sfh_prior(
    "dpl",
    jax.random.PRNGKey(0),
    n=24,
    sfh_dpl_alpha=Uniform(0.5, 1.5),         # default is Uniform(0.1, 5.0)
    sfh_dpl_tau_gyr=Uniform(2.0, 4.0),       # default is Uniform(0.1, 12.0)
)

fig, ax = plt.subplots(figsize=(9, 4))
for i, c in enumerate(np.asarray(narrow)):
    ax.plot(age_gyr, c, color=viridis(0.05 + 0.8 * i / len(narrow)), lw=1.0, alpha=0.85)
ax.set_xlim(0, 13.8)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR(t) [M$_\odot$/yr]")
ax.set_title(r"Same family, tightened priors on $\alpha$ and $\tau$")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Photometric setup with `Instrument`
#
# Photometric sweeps need a filter set. The new `Instrument` registry
# bundles the canonical sets so you don't reach for `Photometry.from_names([...])`
# every time. Compare a few:

# %%
for inst_factory in (Instrument.SDSS, Instrument.JWST_NIRCam, Instrument.WISE):
    inst = inst_factory()
    print(f"  {inst.name:14s}  {inst.photometry.n_filters} bands  {inst.description}")

# %% [markdown]
# ## A 2-D photometric grid via `predict_photometry_batch`
#
# For sweeps over more than one axis the right tool is `model.predict_photometry_batch`,
# which is a `jax.vmap` of the per-galaxy predictor. We sweep `met_logzsol`
# against `dust_tau_diff` for an SDSS-band model and look at how the
# *colour* `g - r` moves across the grid.

# %%
spec_grid = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),
    sfh_tsnorm_width_gyr=Fixed(1.5),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Fixed(0.5),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.05),
)
inst_sdss = Instrument.SDSS()
model_grid = SEDModel(spec_grid, ssp, observation=inst_sdss.observation())

n_z, n_d = 16, 16
met_axis = jnp.linspace(-1.5, 0.0, n_z)
dust_axis = jnp.linspace(0.0, 1.5, n_d)
mm, dd = jnp.meshgrid(met_axis, dust_axis, indexing="ij")

# Build the batch dict: shape (n_z * n_d,) on the swept keys, broadcast
# the fixed values to match.
base = spec_grid.sample(jax.random.PRNGKey(0))
n_total = n_z * n_d
params_batch = {}
for k, v in base.items():
    v = jnp.asarray(v)
    if k == "met_logzsol":
        params_batch[k] = mm.reshape(-1)
    elif k == "dust_tau_diff":
        params_batch[k] = dd.reshape(-1)
    else:
        params_batch[k] = jnp.broadcast_to(v, (n_total, *v.shape))

flux = model_grid.predict_photometry_batch(params_batch)        # (256, 5) for SDSS
flux = flux.reshape(n_z, n_d, -1)                                # (n_z, n_d, n_filters)

i_g = inst_sdss.filter_names.index("sdss_g")
i_r = inst_sdss.filter_names.index("sdss_r")
g_minus_r = -2.5 * (jnp.log10(flux[..., i_g]) - jnp.log10(flux[..., i_r]))

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(
    np.asarray(g_minus_r).T,
    origin="lower",
    aspect="auto",
    extent=(float(met_axis[0]), float(met_axis[-1]), float(dust_axis[0]), float(dust_axis[-1])),
    cmap="viridis",
)
cbar = fig.colorbar(im, ax=ax, label=r"$g - r$ [mag]")
ax.set_xlabel(r"$\log_{10}(Z / Z_\odot)$")
ax.set_ylabel(r"$\tau_{\rm diff}$")
ax.set_title("SDSS $g-r$ across a 16 × 16 (metallicity, dust) grid")
fig.tight_layout()
plt.show()

# %% [markdown]
# 256 forward evaluations, one compiled `vmap` call. Both axes contribute
# to redder colours; the partial degeneracy between metallicity and dust
# is the canonical reason photometric SED fitting needs spectroscopy
# (see [`06_fitting_spectroscopy`](06_fitting_spectroscopy.py) and
# [`07_joint_photo_spec`](07_joint_photo_spec.py)).

# %% [markdown]
# ## What you can do with these
#
# - Build intuition for a new prior before you let a sampler near it.
# - Stress-test the forward model: extreme values that *should* break
#   things are cheap to try in a sweep.
# - Sanity-check that the registered defaults span the regime you care
#   about. If your prior fan looks wrong, the fit will be wrong.
# - Pick informative bands: a 2-D photometric grid plus a colour cut tells
#   you which filter pairs actually constrain a parameter you care about.
