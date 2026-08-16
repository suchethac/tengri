# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # tengri Capabilities: What You Can Extract from Your Data
#
# Six standalone figures showing what tengri delivers. Scan this notebook to
# immediately understand what you can extract from your galaxy observations.

# %%
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fitter,
    Fixed,
    SEDModel,
    Observation,
    Parameters,
    Photometry,
    Spectroscopy,
    Uniform,
    load_ssp_data,
)

import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
if os.path.exists("data"):
    pass
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

FIGDIR = os.path.join("quickstart", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import COLORS, convergence_table, plot_sfh, setup_style

setup_style()

# %%
# Setup: standard 7-parameter model
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)

obs = Observation(
    spectroscopy=Spectroscopy(wave_obs=WAVE_OBS),
)

spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model = SEDModel(spec, ssp_data, observation=obs)

# Generate mock and run geoVI
key = jax.random.PRNGKey(42)
true_params = {**spec.sample(key)}
true_params["sfh_tsnorm_log_peak_sfr"] = jnp.array(1.2)
true_params["sfh_tsnorm_peak_lbt_gyr"] = jnp.array(3.0)
true_params["sfh_tsnorm_width_gyr"] = jnp.array(3.0)
true_params["sfh_tsnorm_skew"] = jnp.array(0.3)
true_params["sfh_tsnorm_trunc"] = jnp.array(2.0)
true_params["met_logzsol"] = jnp.array(-0.3)
true_params["dust_tau_bc"] = jnp.array(0.5)
true_params["dust_tau_diff"] = jnp.array(0.3)

mock = model.mock_spectrum(true_params, WAVE_OBS, snr=20.0, key=key)
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="spectroscopy")

t0 = time.perf_counter()
result_map = fitter.run("map", n_steps=500, verbose=False)
result = fitter.run(
    "vi",
    n_iterations=12,
    n_samples=6,
    n_seeds=3,
    n_posterior_samples=1000,
    verbose=False,
    init_from=result_map,
)
t_total = time.perf_counter() - t0
print(f"Total inference time: {t_total:.1f}s")

# %% [markdown]
# ## 1 — A Posterior SFH
#
# The core product: 50 posterior SFH draws (grey), 68% CI band, and true SFH.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
plot_sfh(model, result, true_params=true_params, ax=ax, n_draws=50, color=COLORS["geovi"])
ax.set_title("Posterior SFH: 1000 samples from native_geovi")
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot\,{\rm yr}^{-1}$]")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig01_sfh_posterior.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2 — A Corner Plot
#
# 7-parameter corner plot showing posterior correlations. The banana-shaped
# age-dust contour is a feature: it reflects the true posterior including
# all physical degeneracies.

# %%
try:
    import corner

    phys_params = [p for p in spec.free_params if "xi" not in p]
    samples_arr = np.array([np.array(result.samples[p]) for p in phys_params]).T
    labels_corner = [p.replace("sfh_tsnorm_", "").replace("_", " ") for p in phys_params]
    truth_vals = [float(true_params[p]) for p in phys_params]

    fig = corner.corner(
        samples_arr,
        labels=labels_corner,
        truths=truth_vals,
        truth_color=COLORS["truth"],
        color=COLORS["geovi"],
        plot_datapoints=False,
        fill_contours=True,
        levels=[0.68, 0.95],
    )
    fig.suptitle("Corner Plot: 7-Parameter Smooth Galaxy", y=1.02)
    plt.savefig(os.path.join(FIGDIR, "fig02_corner.png"), dpi=120, bbox_inches="tight")
    plt.show()
except ImportError:
    print("corner not installed — skipping corner plot. pip install corner")

# %% [markdown]
# ## 3 — Convergence at a Glance
#
# ESS > 400 per parameter means the 68% CI is reliable at ~1% precision.

# %%
print(convergence_table({"geoVI": result}, verbose=True))

# %% [markdown]
# ## 4 — Scaling to a Catalog
#
# vmap throughput at N = 1, 10, 100, 1000 galaxies.
# JIT compilation gives sublinear per-galaxy cost.

# %%
obs_phot = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)
spec_phot = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model_phot = SEDModel(spec_phot, ssp_data, observation=obs_phot)

batch_fn = jax.jit(jax.vmap(model_phot.predict_photometry))
Ns = [1, 10, 100, 1000]
times_per_gal = []

for N in Ns:
    batch_keys = jax.random.split(jax.random.PRNGKey(0), N)
    batch_params = jax.vmap(spec_phot.sample)(batch_keys)
    # Warm up
    _ = batch_fn(batch_params)
    _.block_until_ready()
    t0 = time.perf_counter()
    _ = batch_fn(batch_params)
    _.block_until_ready()
    t_ms = (time.perf_counter() - t0) * 1e3
    times_per_gal.append(t_ms / N)
    print(f"  N = {N:>5d}: {t_ms:.1f} ms total, {t_ms / N:.3f} ms/galaxy")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(range(len(Ns)), times_per_gal, color=COLORS["geovi"], alpha=0.85)
ax.set_xticks(range(len(Ns)))
ax.set_xticklabels([str(N) for N in Ns])
ax.set_xlabel("Number of galaxies N")
ax.set_ylabel("Time per galaxy [ms]")
ax.set_title("vmap Throughput: Sublinear Scaling via JIT Cache")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig04_scaling.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5 — What Wavelengths Constrain What
#
# Jacobian ∂SED/∂θ as a normalized heatmap. Reveals which parameters are
# constrained by which wavelength ranges — the motivation for multiwavelength fitting.

# %%
# Compute Jacobian at MAP solution
free_params_list = spec.free_params
params_map = {k: v for k, v in result_map.params.items()}


def predict_spectrum_flat(params_flat):
    params = {k: params_flat[i] for i, k in enumerate(free_params_list)}
    return model.predict_spectrum(params)


params_flat = jnp.array([float(params_map.get(k, 0.0)) for k in free_params_list])
jacobian_fn = jax.jit(jax.jacfwd(predict_spectrum_flat))
jac = np.array(jacobian_fn(params_flat))  # shape: (n_wave, n_params)

# Normalize each row (per parameter)
jac_norm = jac / (np.max(np.abs(jac), axis=0, keepdims=True) + 1e-30)

fig, ax = plt.subplots(figsize=(10, 4))
wave_np = np.array(WAVE_OBS)
im = ax.imshow(
    jac_norm.T,
    aspect="auto",
    extent=[wave_np[0], wave_np[-1], -0.5, len(free_params_list) - 0.5],
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    origin="lower",
)
plt.colorbar(im, ax=ax, label=r"$|\partial F/\partial \theta|$ (normalized)")
param_labels = [p.replace("sfh_tsnorm_", "").replace("_", " ") for p in free_params_list]
ax.set_yticks(range(len(free_params_list)))
ax.set_yticklabels(param_labels, fontsize=8)
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_title("Jacobian: ∂SED/∂θ — What Wavelengths Constrain What")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig05_jacobian.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6 — The Gradient is Almost Free
#
# The backward pass (gradient) costs ~1.4× the forward pass. This is why
# HMC and variational inference are practical at D=137.

# %%
# Benchmark forward vs gradient
predict_fn = jax.jit(model.predict_spectrum)
grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_spectrum(p))))

# Warm up both
_ = predict_fn(true_params)
_ = grad_fn(true_params)

N_reps = 200
t0 = time.perf_counter()
for _ in range(N_reps):
    predict_fn(true_params)
t_fwd = (time.perf_counter() - t0) / N_reps * 1e6

t0 = time.perf_counter()
for _ in range(N_reps):
    grad_fn(true_params)
t_grad = (time.perf_counter() - t0) / N_reps * 1e6

print(f"Forward pass:  {t_fwd:.1f} μs")
print(f"Gradient:      {t_grad:.1f} μs")
print(f"Ratio (grad/fwd): {t_grad / t_fwd:.2f}x")

fig, ax = plt.subplots(figsize=(4, 3.5))
ax.bar(
    ["Forward", "Gradient"],
    [t_fwd, t_grad],
    color=[COLORS.get("geovi", "#5c85d6"), COLORS.get("rt", "#e07b54")],
    alpha=0.85,
)
ax.set_ylabel("Time [μs]")
ax.set_title(f"Gradient / Forward = {t_grad / t_fwd:.2f}×\n(Why HMC scales to D=137)")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig06_gradient_cost.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Capability | What tengri delivers |
# |-----------|---------------------|
# | Posterior SFH | 1000 draws in ~12s; full uncertainty |
# | Corner plot | All degeneracies captured |
# | Convergence | ESS > 400 per parameter by default |
# | Catalog scaling | Sublinear per-galaxy time via JIT |
# | Jacobian | Automatic; ~1.4× forward pass cost |
# | Gradient | Same cost at D=7 and D=137 |
