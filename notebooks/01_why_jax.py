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
# # Why JAX — differentiable inference, in the language of SED fitting
#
# Traditional SED fitting (`emcee` + Prospector / Bagpipes / CIGALE) is
# gradient-free: at every step the sampler queries the likelihood and
# guesses the next move. In 10–30 dimensions, with 10⁴ likelihood calls
# per chain step, a single galaxy takes hours.
#
# Tengri builds the same physics — stellar populations, dust, nebular
# emission, AGN, IGM — entirely from JAX primitives. The model is
# differentiable. The likelihood and its gradient come together, at no
# extra cost. Gradient-based samplers (NUTS, HMC, VI) then *use* that
# gradient to climb the posterior efficiently.
#
# The figures below make this concrete: an astronomer-readable map of
# the posterior gradient, and how that translates into wall-clock time.

# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

# Keep the rendered tutorial clean: silence framework notices that do not
# change the science shown here (baked-in nebular, the WavePrecomp blue-band
# approximation, and recipe/parameter-provenance notices). Genuine
# deprecations in user-facing
# calls are fixed in the code, not hidden.
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*WavePrecomp.*")
warnings.filterwarnings("ignore", message=".*was marked FIXED.*")
warnings.filterwarnings("ignore", message=".*Composable AGN.*")
warnings.filterwarnings("ignore", message=".*before the Big Bang.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from _setup import FIG_DIR
import tengri
from tengri import (
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    generate_mock,
    plot,
    recipes,
)

plot.setup_style()

# %% [markdown]
# ## A minimal star-forming galaxy
#
# `recipes.mock_recovery_minimal()` is the cheapest stable model — a
# truncated-skew-normal SFH, single dust optical depth, no nebular
# physics. Five free parameters; tractable in seconds.

# %%
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)

obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1"])
)
model = SEDModel.build(ssp_data=ssp, observation=obs, **recipes.mock_recovery_minimal())

truth = model.spec.sample(jax.random.PRNGKey(0))
mock = generate_mock(model, truth, key=jax.random.PRNGKey(1), snr=20.0)
flux_obs, noise = mock["flux_obs"], mock["noise"]

# %% [markdown]
# ## Figure 1 — the posterior gradient
#
# Left: log-posterior surface, with 1σ / 2σ / 3σ contours. Right: `jax.grad`
# as a vector field. Gradient-based samplers follow these arrows.

# %%
log_sfr_grid = np.linspace(-1.5, 2.0, 30)
tau_grid = np.linspace(0.0, 1.5, 30)
LSFR, TAU = np.meshgrid(log_sfr_grid, tau_grid, indexing="ij")

base = dict(truth)
free_keys = ("sfh_tsnorm_log_total_mass", "dust_tau_bc")


def neg_log_post(log_sfr, tau):
    p = dict(base)
    p[free_keys[0]] = log_sfr
    p[free_keys[1]] = tau
    flux_pred = model.predict_observables(p).phot_fnu
    chi2 = jnp.sum(((flux_pred - flux_obs) / noise) ** 2)
    return 0.5 * chi2  # uniform priors → χ²/2 = -ln posterior up to const


# Sequentialize: vmap-of-vmap of the orchestrator state pytree
# explodes memory (n_age × n_wave × n_grid). Plain JIT'd scalar
# calls reuse one set of buffers and stay well under a GB.
neg_log_post_jit = jax.jit(neg_log_post)
grad_jit = jax.jit(jax.grad(neg_log_post, argnums=(0, 1)))

nll_grid = np.zeros_like(LSFR)
g_log_sfr = np.zeros_like(LSFR)
g_tau = np.zeros_like(LSFR)
for i in range(LSFR.shape[0]):
    for j in range(LSFR.shape[1]):
        nll_grid[i, j] = float(neg_log_post_jit(LSFR[i, j], TAU[i, j]))
        gx, gy = grad_jit(LSFR[i, j], TAU[i, j])
        g_log_sfr[i, j] = float(gx)
        g_tau[i, j] = float(gy)

log_post = -(nll_grid - nll_grid.min())  # peak at zero
step_x, step_y = -g_log_sfr, -g_tau  # ascend the posterior
mag = np.hypot(step_x, step_y)
step_x = step_x / (mag + 1e-12)
step_y = step_y / (mag + 1e-12)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
levels = -np.array([0.5 * s * s for s in (1.0, 2.0, 3.0)])[::-1]
cs = axes[0].contourf(LSFR, TAU, log_post, levels=20, cmap="magma")
axes[0].contour(LSFR, TAU, log_post, levels=levels, colors="white", linewidths=0.8)
axes[0].scatter(
    [truth[free_keys[0]]],
    [truth[free_keys[1]]],
    marker="*",
    s=140,
    c="white",
    edgecolor="k",
    zorder=5,
    label="truth",
)
axes[0].set_xlabel(r"$\log_{10}(\mathrm{peak\ SFR})\ [M_\odot/\mathrm{yr}]$")
axes[0].set_ylabel(r"birth-cloud $\tau_V$")
axes[0].set_title("log posterior")
axes[0].legend(loc="lower right", frameon=False)
fig.colorbar(cs, ax=axes[0], shrink=0.85, label=r"$\ln \mathcal{P}$")

stride = 4
axes[1].contour(LSFR, TAU, log_post, levels=levels, colors="0.5", linewidths=0.8)
axes[1].quiver(
    LSFR[::stride, ::stride],
    TAU[::stride, ::stride],
    step_x[::stride, ::stride],
    step_y[::stride, ::stride],
    mag[::stride, ::stride],
    cmap="viridis",
    pivot="middle",
    scale=35,
    width=0.003,
    alpha=0.9,
)
axes[1].scatter(
    [truth[free_keys[0]]],
    [truth[free_keys[1]]],
    marker="*",
    s=140,
    c="white",
    edgecolor="k",
    zorder=5,
)
axes[1].set_xlabel(r"$\log_{10}(\mathrm{peak\ SFR})\ [M_\odot/\mathrm{yr}]$")
axes[1].set_ylabel(r"birth-cloud $\tau_V$")
axes[1].set_title(r"$-\nabla\, \chi^2 / 2$  (NUTS/HMC follows these arrows)")
fig.savefig(FIG_DIR / "01_gradient_map.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Three lines of code carry that whole picture:


# %%
def loss(p):
    return 0.5 * jnp.sum(((model.predict_observables(p).phot_fnu - flux_obs) / noise) ** 2)


grad_at_truth = jax.grad(loss)(truth)
{k: float(v) for k, v in grad_at_truth.items() if k in free_keys}

# %% [markdown]
# ## Figure 2 — forward-model throughput
#
# Single forward call vs. `vmap` over 10 000 draws. Batched cost is far below
# 10 000× single-call: XLA broadcasts the compiled graph across the batch.

# %%
# Warm the JIT cache.
_ = model.predict_observables(truth).phot_fnu.block_until_ready()

t0 = perf_counter()
for _ in range(50):
    _ = model.predict_observables(truth).phot_fnu.block_until_ready()
t_single = (perf_counter() - t0) / 50

tengri.clear_shared_caches()  # free graphs from the gradient figure
n_batch = 200
keys = jax.random.split(jax.random.PRNGKey(5), n_batch)
batch_params = jax.vmap(model.spec.sample)(keys)
forward = jax.jit(jax.vmap(lambda p: model.predict_observables(p).phot_fnu))

_ = forward(batch_params).block_until_ready()  # cold-compile
t0 = perf_counter()
_ = forward(batch_params).block_until_ready()
t_batch = perf_counter() - t0

t0 = perf_counter()
posterior = ForwardModel.build(sed=model).fit(
    flux_obs, noise,
    method="mcmc_nuts",
    key=jax.random.PRNGKey(2),
    n_warmup=300,
    n_samples=300,
)
t_nuts = perf_counter() - t0

bars = {
    "single forward\n(JIT warm)": t_single,
    f"vmap of {n_batch}\nforwards": t_batch,
    "single NUTS\nposterior (5-D)": t_nuts,
    "emcee, 7-D\ngalaxy (lit.)": 3600.0,
}
fig2, ax = plt.subplots(figsize=(6.8, 4.0))
colors = ["#3b7dd8", "#3b7dd8", "#c3372a", "0.6"]
ax.barh(list(bars), list(bars.values()), color=colors, edgecolor="k", linewidth=0.6)
ax.set_xscale("log")
ax.set_xlabel("wall-clock time [s]")
for i, v in enumerate(bars.values()):
    if v < 1.0:
        label = f"{v * 1e3:.1f} ms"
    elif v < 120:
        label = f"{v:.1f} s"
    else:
        label = f"{v / 60:.0f} min"
    ax.text(v * 1.4, i, label, va="center", fontsize=10)
fig2.tight_layout()
fig2.savefig(FIG_DIR / "01_wallclock.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# `compile="per_component"` is the default and compiles each component
# independently: fast cold start, friendly to notebook edits. `compile="fused"`
# compiles the whole pipeline as one graph — slower first call, fastest steady
# state, which is what a population fit wants.

# %%
model_fused = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    compile="fused",
    **recipes.mock_recovery_minimal(),
)

# %% [markdown]
# `import tengri` enables an on-disk JIT cache at `~/.cache/tengri_jax_cache`,
# so a restarted kernel or a fresh Slurm worker does not recompile unchanged
# components.
